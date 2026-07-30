"""GUI for collecting output-arm episodes.

The GUI talks only to the output arm on one CAN interface and records the
same NPZ format as collect_output_arm.py.  It intentionally does not read
control-command frames or master-arm data.
"""

from __future__ import annotations

import pathlib
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

import cv2

from camera import CameraCapture
from collect_output_arm import (
    DEFAULT_CAN,
    DEFAULT_HIGH_DEVICE,
    DEFAULT_WRIST_DEVICE,
    EpisodeBuffer,
    connect,
    next_episode_index,
    read_output_qpos,
)


class CollectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Piper 数据采集工具")
        self.root.geometry("1000x780")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.piper = None
        self.cameras = None
        self.capture_thread: threading.Thread | None = None
        self.capture_stop: threading.Event | None = None
        self.data_lock = threading.Lock()
        self.latest_images = {}
        self.buffer: EpisodeBuffer | None = None
        self.messages: queue.Queue = queue.Queue()
        self.recording = False
        self.episode_index = 0

        self.can_var = tk.StringVar(value=DEFAULT_CAN)
        self.high_var = tk.StringVar(value="/dev/video8")
        self.wrist_var = tk.StringVar(value="/dev/video16")
        self.fps_var = tk.StringVar(value="30")
        self.out_var = tk.StringVar(value="episodes_output_arm")
        self.task_var = tk.StringVar(value="pick_cube")
        self.instruction_var = tk.StringVar(value="pick up the cube")
        self.status_var = tk.StringVar(value="未连接")
        self.progress_var = tk.StringVar(value="未采集")

        self._build_ui()
        self.refresh_files()
        self.root.after(100, self._poll_messages)

    def _build_ui(self):
        tkfont.nametofont("TkDefaultFont").configure(size=13)
        tkfont.nametofont("TkTextFont").configure(size=13)
        tkfont.nametofont("TkHeadingFont").configure(size=15, weight="bold")
        style = ttk.Style()
        style.configure("TButton", font=("Sans", 13), padding=(12, 7))
        style.configure("TLabel", font=("Sans", 13))
        style.configure("TLabelframe.Label", font=("Sans", 14, "bold"))

        config = ttk.LabelFrame(self.root, text="设备与任务配置", padding=10)
        config.pack(fill="x", padx=12, pady=10)

        rows = [
            ("CAN 接口", self.can_var),
            ("第三视角 RGB", self.high_var),
            ("腕部第一视角 RGB", self.wrist_var),
            ("采样频率", self.fps_var),
            ("输出目录", self.out_var),
            ("任务名称", self.task_var),
            ("任务指令", self.instruction_var),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(config, text=label, width=18).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(config, textvariable=var, width=70, font=("Sans", 13)).grid(row=row, column=1, sticky="ew", pady=6)
        config.columnconfigure(1, weight=1)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=2)
        self.connect_button = ttk.Button(controls, text="连接设备", command=self.toggle_connection)
        self.connect_button.pack(side="left", padx=3)
        self.start_button = ttk.Button(controls, text="开始采集", command=self.start_episode, state="disabled")
        self.start_button.pack(side="left", padx=3)
        self.stop_button = ttk.Button(controls, text="停止采集", command=self.stop_episode, state="disabled")
        self.stop_button.pack(side="left", padx=3)
        ttk.Button(controls, text="刷新文件", command=self.refresh_files).pack(side="left", padx=3)
        ttk.Button(controls, text="回放选中 episode", command=self.replay_selected).pack(side="left", padx=3)

        status = ttk.LabelFrame(self.root, text="运行状态", padding=10)
        status.pack(fill="x", padx=12, pady=10)
        ttk.Label(status, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(status, textvariable=self.progress_var).pack(anchor="w", pady=4)

        files = ttk.LabelFrame(self.root, text="已保存数据", padding=8)
        files.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.listbox = tk.Listbox(files, height=10)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(files, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=scrollbar.set)

    @property
    def out_dir(self) -> pathlib.Path:
        path = pathlib.Path(self.out_var.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def toggle_connection(self):
        if self.piper is not None:
            self.disconnect()
            return
        try:
            fps = int(self.fps_var.get())
            self.status_var.set("正在连接机械臂和相机...")
            self.root.update_idletasks()
            self.piper = connect(self.can_var.get().strip())
            self.cameras = CameraCapture(
                cam_ids={"cam_high": self.high_var.get().strip(), "cam_wrist": self.wrist_var.get().strip()},
                fps=fps,
            )
            self.cameras.open()
            checks = self.cameras.verify()
            bad = [key for key, info in checks.items() if not info["ok"]]
            if bad:
                raise RuntimeError(f"摄像头测试失败: {bad}")
            self.episode_index = next_episode_index(self.out_dir)
            self.capture_stop = threading.Event()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            self.status_var.set(f"已连接: {self.can_var.get()} | 下一个 episode: {self.episode_index:04d}")
            self.connect_button.configure(text="断开设备")
            self.start_button.configure(state="normal")
        except Exception as exc:
            self.status_var.set(f"连接失败: {exc}")
            self._cleanup_devices()
            messagebox.showerror("连接失败", str(exc))

    def start_episode(self):
        if self.piper is None or self.cameras is None or self.recording:
            return
        self.buffer = EpisodeBuffer()
        self.recording = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set(f"正在采集 episode {self.episode_index:04d}")
        self.progress_var.set("帧数: 0")

    def _capture_loop(self):
        assert self.capture_stop is not None
        fps = max(1, int(self.fps_var.get()))
        dt = 1.0 / fps
        try:
            while not self.capture_stop.is_set():
                t0 = time.time()
                qpos = read_output_qpos(self.piper)
                images, image_ts = self.cameras.read()
                with self.data_lock:
                    self.latest_images = {key: value.copy() for key, value in images.items()}
                    if self.recording and self.buffer is not None:
                        self.buffer.add(qpos, images, image_ts)
                        count = len(self.buffer)
                    else:
                        count = 0
                if self.recording:
                    self.messages.put(("progress", count, qpos.copy()))
                delay = dt - (time.time() - t0)
                if delay > 0:
                    time.sleep(delay)
        except Exception as exc:
            self.messages.put(("error", str(exc)))

    def stop_episode(self):
        if not self.recording:
            return
        with self.data_lock:
            self.recording = False
        self.stop_button.configure(state="disabled")
        self.status_var.set("正在停止并整理当前 episode...")
        self.root.after(100, self._finish_stop)

    def _finish_stop(self):
        self.start_button.configure(state="normal" if self.piper is not None else "disabled")
        if self.buffer is None or len(self.buffer) == 0:
            self.status_var.set("当前 episode 为空，未保存")
            return
        self._ask_label_and_save()

    def _ask_label_and_save(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("标注当前 episode")
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(dialog, text=f"episode {self.episode_index:04d}，共 {len(self.buffer)} 帧").pack(padx=20, pady=15)
        buttons = ttk.Frame(dialog)
        buttons.pack(pady=(0, 15))

        def finish(choice):
            dialog.destroy()
            if choice == "discard":
                self.buffer = None
                self.status_var.set("已丢弃当前 episode")
                return
            path = self.out_dir / f"ep_{self.episode_index:04d}.npz"
            instruction = self.instruction_var.get().strip() or self.task_var.get().replace("_", " ")
            self.buffer.save(path, self.task_var.get().strip(), instruction, choice == "success")
            self.episode_index += 1
            self.buffer = None
            self.status_var.set(f"已保存 {path}")
            self.refresh_files()

        ttk.Button(buttons, text="保存为成功", command=lambda: finish("success")).pack(side="left", padx=5)
        ttk.Button(buttons, text="保存为失败", command=lambda: finish("failure")).pack(side="left", padx=5)
        ttk.Button(buttons, text="丢弃", command=lambda: finish("discard")).pack(side="left", padx=5)

    def _poll_messages(self):
        with self.data_lock:
            preview = {key: value.copy() for key, value in self.latest_images.items()}
        if preview:
            self._show_preview(preview)
        try:
            while True:
                kind, *payload = self.messages.get_nowait()
                if kind == "progress":
                    count, qpos = payload
                    self.progress_var.set(f"帧数: {count} | J1={qpos[0]:.3f} rad | gripper={qpos[6] * 1000:.1f} mm")
                elif kind == "error":
                    self.status_var.set(f"采集错误: {payload[0]}")
                    if self.recording:
                        self.stop_episode()
                    messagebox.showerror("采集错误", payload[0])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    @staticmethod
    def _show_preview(images):
        for key, frame in images.items():
            # CameraCapture returns RGB CHW; enlarge for comfortable viewing.
            image = frame.transpose(1, 2, 0)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_NEAREST)
            # OpenCV's built-in Hershey fonts do not support Chinese glyphs;
            # keep preview-window text in ASCII to avoid garbled characters.
            title = "Third-person - cam_high" if key == "cam_high" else "First-person - cam_wrist"
            cv2.putText(image, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(title, image)
        cv2.waitKey(1)

    def refresh_files(self):
        self.listbox.delete(0, tk.END)
        if not self.out_dir.exists():
            return
        for path in sorted(self.out_dir.glob("ep_*.npz")):
            self.listbox.insert(tk.END, str(path))

    def replay_selected(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("回放", "请先选择一个 episode 文件")
            return
        path = self.listbox.get(selection[0])
        viewer = pathlib.Path(__file__).with_name("view_episode.py")
        subprocess.Popen([sys.executable, str(viewer), path])

    def _cleanup_devices(self):
        if self.capture_stop is not None:
            self.capture_stop.set()
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        self.capture_thread = None
        self.capture_stop = None
        if self.cameras is not None:
            self.cameras.close()
        self.cameras = None
        if self.piper is not None:
            self.piper.DisconnectPort()
        self.piper = None
        cv2.destroyAllWindows()

    def disconnect(self):
        if self.recording:
            messagebox.showwarning("无法断开", "请先停止当前采集")
            return
        self._cleanup_devices()
        self.connect_button.configure(text="连接设备")
        self.start_button.configure(state="disabled")
        self.status_var.set("未连接")

    def close(self):
        if self.recording:
            if not messagebox.askyesno("退出", "当前 episode 尚未保存，确定退出吗？"):
                return
            with self.data_lock:
                self.recording = False
        self._cleanup_devices()
        self.root.destroy()


def main():
    root = tk.Tk()
    CollectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
