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
    IMAGE_HW,
    connect,
    next_episode_index,
    read_output_state,
    reset_output_arm,
)


class CollectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Piper Data Collection Tool")
        self.root.geometry("1000x780")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.piper = None
        self.cameras = None
        self.capture_thread: threading.Thread | None = None
        self.capture_stop: threading.Event | None = None
        self.reset_thread: threading.Thread | None = None
        self.data_lock = threading.Lock()
        self.latest_images = {}
        self.buffer: EpisodeBuffer | None = None
        self.messages: queue.Queue = queue.Queue()
        self.recording = False
        self.episode_index = 0
        self.capture_fps = 20

        self.can_var = tk.StringVar(value=DEFAULT_CAN)
        self.high_var = tk.StringVar(value="/dev/video8")
        self.wrist_var = tk.StringVar(value="/dev/video16")
        self.fps_var = tk.StringVar(value="20")
        self.out_var = tk.StringVar(value="episodes_piper_v21")
        self.task_var = tk.StringVar(value="pick_cube")
        self.instruction_var = tk.StringVar(value="pick up the cube")
        self.status_var = tk.StringVar(value="Disconnected")
        self.progress_var = tk.StringVar(value="No active episode")

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

        config = ttk.LabelFrame(self.root, text="Device and Task Configuration", padding=10)
        config.pack(fill="x", padx=12, pady=10)

        rows = [
            ("CAN interface", self.can_var),
            ("Third-person RGB", self.high_var),
            ("Wrist RGB", self.wrist_var),
            ("Capture FPS", self.fps_var),
            ("Output directory", self.out_var),
            ("Task name", self.task_var),
            ("Instruction", self.instruction_var),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(config, text=label, width=18).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(config, textvariable=var, width=70, font=("Sans", 13)).grid(row=row, column=1, sticky="ew", pady=6)
        config.columnconfigure(1, weight=1)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=2)
        self.connect_button = ttk.Button(controls, text="Connect devices", command=self.toggle_connection)
        self.connect_button.pack(side="left", padx=3)
        self.start_button = ttk.Button(controls, text="Start episode", command=self.start_episode, state="disabled")
        self.start_button.pack(side="left", padx=3)
        self.stop_button = ttk.Button(controls, text="Stop episode", command=self.stop_episode, state="disabled")
        self.stop_button.pack(side="left", padx=3)
        self.reset_button = ttk.Button(controls, text="Reset to Home", command=self.reset_arm, state="disabled")
        self.reset_button.pack(side="left", padx=3)
        ttk.Button(controls, text="Refresh files", command=self.refresh_files).pack(side="left", padx=3)
        ttk.Button(controls, text="Replay selected", command=self.replay_selected).pack(side="left", padx=3)

        status = ttk.LabelFrame(self.root, text="Status", padding=10)
        status.pack(fill="x", padx=12, pady=10)
        ttk.Label(status, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(status, textvariable=self.progress_var).pack(anchor="w", pady=4)

        files = ttk.LabelFrame(self.root, text="Saved episodes", padding=8)
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
            if fps <= 0:
                raise ValueError("Capture FPS must be positive")
            self.capture_fps = fps
            self.status_var.set("Connecting to robot and cameras...")
            self.root.update_idletasks()
            self.piper = connect(self.can_var.get().strip())
            self.cameras = CameraCapture(
                cam_ids={"cam_high": self.high_var.get().strip(), "cam_wrist": self.wrist_var.get().strip()},
                fps=fps,
                image_hw=IMAGE_HW,
                parallel_reads=True,
            )
            self.cameras.open()
            checks = self.cameras.verify()
            bad = [key for key, info in checks.items() if not info["ok"]]
            if bad:
                raise RuntimeError(f"Camera verification failed: {bad}")
            self.episode_index = next_episode_index(self.out_dir)
            self.capture_stop = threading.Event()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            self.status_var.set(f"Connected: {self.can_var.get()} | Next episode: {self.episode_index:04d}")
            self.connect_button.configure(text="Disconnect devices")
            self.start_button.configure(state="normal")
            self.reset_button.configure(state="normal")
        except Exception as exc:
            self.status_var.set(f"Connection failed: {exc}")
            self._cleanup_devices()
            messagebox.showerror("Connection failed", str(exc))

    def start_episode(self):
        if self.piper is None or self.cameras is None or self.recording:
            return
        self.buffer = EpisodeBuffer(fps=self.capture_fps)
        self.recording = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.reset_button.configure(state="disabled")
        self.status_var.set(f"Recording episode {self.episode_index:04d}")
        self.progress_var.set("Frames: 0")

    def _capture_loop(self):
        assert self.capture_stop is not None
        fps = self.capture_fps
        dt = 1.0 / fps
        try:
            while not self.capture_stop.is_set():
                t0 = time.time()
                state, qpos = read_output_state(self.piper)
                state_timestamp = time.time()
                images, image_ts = self.cameras.read()
                with self.data_lock:
                    self.latest_images = {key: value.copy() for key, value in images.items()}
                    if self.recording and self.buffer is not None:
                        self.buffer.add(
                            state,
                            images,
                            image_ts,
                            qpos=qpos,
                            state_timestamp=state_timestamp,
                        )
                        count = len(self.buffer)
                    else:
                        count = 0
                if self.recording:
                    self.messages.put(("progress", count, state.copy()))
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
        self.status_var.set("Stopping and preparing the current episode...")
        self.root.after(100, self._finish_stop)

    def _finish_stop(self):
        self.start_button.configure(state="normal" if self.piper is not None else "disabled")
        self.reset_button.configure(state="normal" if self.piper is not None else "disabled")
        if self.buffer is None or len(self.buffer) == 0:
            self.status_var.set("The episode is empty and was not saved")
            return
        self._ask_label_and_save()

    def _ask_label_and_save(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Label current episode")
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(dialog, text=f"Episode {self.episode_index:04d} | {len(self.buffer)} frames").pack(padx=20, pady=15)
        buttons = ttk.Frame(dialog)
        buttons.pack(pady=(0, 15))

        def finish(choice):
            dialog.destroy()
            if choice == "discard":
                self.buffer = None
                self.status_var.set("Current episode discarded")
                return
            path = self.out_dir / f"ep_{self.episode_index:04d}.npz"
            instruction = self.instruction_var.get().strip() or self.task_var.get().replace("_", " ")
            self.buffer.save(path, self.task_var.get().strip(), instruction, choice == "success")
            self.episode_index += 1
            self.buffer = None
            self.status_var.set(f"Saved: {path}")
            self.refresh_files()

        ttk.Button(buttons, text="Save as success", command=lambda: finish("success")).pack(side="left", padx=5)
        ttk.Button(buttons, text="Save as failure", command=lambda: finish("failure")).pack(side="left", padx=5)
        ttk.Button(buttons, text="Discard", command=lambda: finish("discard")).pack(side="left", padx=5)

    def reset_arm(self):
        if self.piper is None or self.recording or self.reset_thread is not None:
            return
        confirmed = messagebox.askyesno(
            "Reset to Home",
            "Move the output arm smoothly to the all-zero joint pose and close the gripper?",
        )
        if not confirmed:
            return
        self.start_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.status_var.set("Resetting output arm to home pose...")
        self.reset_thread = threading.Thread(target=self._reset_worker, daemon=True)
        self.reset_thread.start()

    def _reset_worker(self):
        try:
            reset_output_arm(self.piper)
            self.messages.put(("reset_done",))
        except Exception as exc:
            self.messages.put(("reset_error", str(exc)))

    def _finish_reset(self, success: bool, error: str | None = None):
        self.reset_thread = None
        if success:
            self.status_var.set("Output arm reset completed")
        else:
            self.status_var.set(f"Reset failed: {error}")
            messagebox.showerror("Reset failed", error or "Unknown reset error")
        if self.piper is not None and not self.recording:
            self.start_button.configure(state="normal")
            self.reset_button.configure(state="normal")

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
                    self.progress_var.set(f"Frames: {count} | EEF xyz={qpos[:3].round(3)} | gripper fraction={qpos[9]:.3f}")
                elif kind == "error":
                    self.status_var.set(f"Capture error: {payload[0]}")
                    if self.recording:
                        self.stop_episode()
                    messagebox.showerror("Capture error", payload[0])
                elif kind == "reset_done":
                    self._finish_reset(True)
                elif kind == "reset_error":
                    self._finish_reset(False, payload[0])
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
            messagebox.showinfo("Replay", "Select an episode first")
            return
        path = self.listbox.get(selection[0])
        viewer = pathlib.Path(__file__).with_name("view_episode.py")
        subprocess.Popen([sys.executable, str(viewer), path])

    def _cleanup_devices(self):
        if self.capture_stop is not None:
            self.capture_stop.set()
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.reset_thread is not None and self.reset_thread.is_alive():
            self.reset_thread.join(timeout=2.0)
        self.reset_thread = None
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
            messagebox.showwarning("Cannot disconnect", "Stop the current episode first")
            return
        if self.reset_thread is not None:
            messagebox.showwarning("Cannot disconnect", "Wait for the reset to finish first")
            return
        self._cleanup_devices()
        self.connect_button.configure(text="Connect devices")
        self.start_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.status_var.set("Disconnected")

    def close(self):
        if self.recording:
            if not messagebox.askyesno("Exit", "The current episode is unsaved. Exit anyway?"):
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
