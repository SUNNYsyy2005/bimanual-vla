"""GUI for collecting output-arm episodes.

The GUI talks only to the output arm on one CAN interface and records the
same NPZ format as :mod:`collect_output_arm`.  It intentionally does not read
control-command frames or master-arm data.

The live view is deliberately kept in the Tk window: it shows both RGB
cameras, the measured joint pose, EEF state, and whether the arm is inside the
configured home-pose tolerance.  An episode cannot be started until the
latest robot feedback passes that check.
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
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - OpenCV fallback is for minimal installs.
    Image = None
    ImageTk = None

from collection_session import CollectionConfig, CollectionSession, SessionState
from collect_output_arm import (
    DEFAULT_CAN,
    DEFAULT_CAMERA_FPS,
    DEFAULT_HIGH_DEVICE,
    DEFAULT_WRIST_DEVICE,
    reset_output_arm,
)


JOINT_NAMES = tuple(f"J{i}" for i in range(1, 7)) + ("Gripper",)
PREVIEW_SIZE = (480, 360)


def check_initial_pose(
    qpos: np.ndarray,
    start_qpos: np.ndarray,
    joint_tolerance_rad: float,
    gripper_tolerance_m: float,
) -> tuple[bool, str, np.ndarray]:
    """Check measured qpos against the configured home pose.

    The first six values are Piper joint angles in radians and the last value
    is the gripper position in metres.  The returned error vector uses the
    same units.  This is a pure helper so it can be tested without hardware or
    a Tk display.
    """
    measured = np.asarray(qpos, dtype=np.float64).reshape(-1)
    target = np.asarray(start_qpos, dtype=np.float64).reshape(-1)
    if measured.shape != (7,) or target.shape != (7,):
        return False, "invalid qpos shape (expected 7 values)", np.full(7, np.nan)
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(target)):
        return False, "robot feedback contains NaN/Inf", np.full(7, np.nan)
    if joint_tolerance_rad <= 0 or gripper_tolerance_m <= 0:
        return False, "pose tolerances must be positive", np.full(7, np.nan)

    errors = measured - target
    bad_joints = np.flatnonzero(np.abs(errors[:6]) > joint_tolerance_rad)
    bad_gripper = abs(float(errors[6])) > gripper_tolerance_m
    if len(bad_joints) == 0 and not bad_gripper:
        return True, "OK: robot is at the home pose", errors.astype(np.float32)

    details = [
        f"J{i + 1}={np.degrees(errors[i]):+.1f}°"
        for i in bad_joints
    ]
    if bad_gripper:
        details.append(f"Gripper={errors[6] * 1000:+.1f} mm")
    return False, "Pose diff: " + ", ".join(details), errors.astype(np.float32)


class CollectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Piper · π0.5 Data Collection")
        self.root.geometry("1660x1000")
        self.root.minsize(1280, 800)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.session: CollectionSession | None = None
        self.piper = None
        self.cameras = None
        self.capture_thread: threading.Thread | None = None
        self.capture_stop: threading.Event | None = None
        self.reset_thread: threading.Thread | None = None
        self.data_lock = threading.Lock()
        self.latest_images: dict[str, np.ndarray] = {}
        self.latest_qpos: np.ndarray | None = None
        self.latest_state: np.ndarray | None = None
        self.latest_pose_errors = np.zeros(7, dtype=np.float32)
        self.latest_pose_ok: bool | None = None
        self.latest_pose_reason = "Waiting for robot feedback"
        self.messages: queue.Queue = queue.Queue()
        self.recording = False
        self.episode_index = 0
        self.capture_fps = 20
        self.camera_fps = DEFAULT_CAMERA_FPS

        # The existing reset command uses the all-zero Piper joint pose.  Keep
        # the same reference for the pre-episode safety check.
        self.start_qpos = np.zeros(7, dtype=np.float32)
        self.joint_tolerance_rad = float(np.deg2rad(5.0))
        self.gripper_tolerance_m = 0.005

        # PhotoImage references must be retained for Tk to keep them alive.
        self.preview_photos: dict[str, object] = {}
        self.preview_labels: dict[str, tk.Label] = {}
        self.joint_rows: dict[str, str] = {}

        self.can_var = tk.StringVar(value=DEFAULT_CAN)
        self.high_var = tk.StringVar(value=DEFAULT_HIGH_DEVICE)
        self.wrist_var = tk.StringVar(value=DEFAULT_WRIST_DEVICE)
        self.fps_var = tk.StringVar(value="20")
        self.camera_fps_var = tk.StringVar(value=str(DEFAULT_CAMERA_FPS))
        self.joint_tol_var = tk.StringVar(value="5.0")
        self.gripper_tol_var = tk.StringVar(value="5.0")
        self.out_var = tk.StringVar(value="episodes_piper_v21")
        self.task_var = tk.StringVar(value="pick_cube")
        self.instruction_var = tk.StringVar(value="pick up the cube")
        self.status_var = tk.StringVar(value="Disconnected")
        self.progress_var = tk.StringVar(value="No episode started")
        self.pose_check_var = tk.StringVar(value="Waiting for robot feedback")
        self.eef_var = tk.StringVar(value="EEF: --")
        self.live_var = tk.StringVar(value="Live telemetry: --")
        self.joint_vars = {name: tk.StringVar(value="--") for name in JOINT_NAMES}

        self._build_ui()
        self.refresh_files()
        self.root.after(100, self._poll_messages)

    def _build_ui(self):
        # Use the native ttk theme but make the important controls readable on
        # a 1080p field monitor.
        tkfont.nametofont("TkDefaultFont").configure(size=12)
        tkfont.nametofont("TkTextFont").configure(size=12)
        tkfont.nametofont("TkHeadingFont").configure(size=14, weight="bold")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=("Sans", 12), padding=(9, 5))
        style.configure("TLabel", font=("Sans", 12))
        style.configure("TLabelframe.Label", font=("Sans", 13, "bold"))
        style.configure("Treeview", rowheight=29, font=("Sans", 11))
        style.configure("Treeview.Heading", font=("Sans", 11, "bold"))

        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=10)
        left = ttk.Frame(main, padding=(2, 0, 8, 0), width=620)
        right = ttk.Frame(main, padding=(8, 0, 2, 0))
        main.add(left, weight=5)
        main.add(right, weight=8)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(4, weight=2)
        left.rowconfigure(5, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        config = ttk.LabelFrame(left, text="Devices and Task", padding=10)
        config.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        config.columnconfigure(1, weight=1)
        config.columnconfigure(3, weight=1)
        rows = [
            (("CAN Interface", self.can_var), ("Output Directory", self.out_var)),
            (("Third-Person Camera", self.high_var), ("Wrist Camera", self.wrist_var)),
            (("Capture Rate (Hz)", self.fps_var), ("Camera Rate (Hz)", self.camera_fps_var)),
            (("Task Name", self.task_var), ("Instruction", self.instruction_var)),
        ]
        for row, pair in enumerate(rows):
            for group, (label, var) in enumerate(pair):
                label_col = group * 2
                ttk.Label(config, text=label).grid(
                    row=row, column=label_col, sticky="w", pady=3, padx=(0 if group == 0 else 12, 0)
                )
                ttk.Entry(config, textvariable=var, width=22).grid(
                    row=row, column=label_col + 1, sticky="ew", pady=3, padx=(7, 0)
                )

        pose_config = ttk.LabelFrame(left, text="Pose Status", padding=10)
        pose_config.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        pose_config.columnconfigure(1, weight=1)
        pose_config.columnconfigure(4, weight=1)
        ttk.Label(pose_config, text="Joint Tolerance").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(pose_config, textvariable=self.joint_tol_var, width=9).grid(
            row=0, column=1, sticky="w", padx=(8, 0), pady=3
        )
        ttk.Label(pose_config, text="deg (J1-J6)").grid(row=0, column=2, sticky="w", padx=(5, 18))
        ttk.Label(pose_config, text="Gripper Tolerance").grid(row=0, column=3, sticky="w", pady=3)
        ttk.Entry(pose_config, textvariable=self.gripper_tol_var, width=9).grid(
            row=0, column=4, sticky="w", padx=(8, 0), pady=3
        )
        ttk.Label(pose_config, text="mm").grid(row=0, column=5, sticky="w", padx=(5, 0))
        ttk.Label(
            pose_config,
            text="Reference: all six joints at zero with the gripper closed.",
            foreground="#65717d",
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))

        controls = ttk.LabelFrame(left, text="Collection Controls", padding=8)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.connect_button = ttk.Button(controls, text="Connect", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        self.start_button = ttk.Button(controls, text="Start Episode", command=self.start_episode, state="disabled")
        self.start_button.grid(row=0, column=1, padx=3, pady=3, sticky="ew")
        self.stop_button = ttk.Button(controls, text="Stop Episode", command=self.stop_episode, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=3, pady=3, sticky="ew")
        self.reset_button = ttk.Button(controls, text="Reset Home", command=self.reset_arm, state="disabled")
        self.reset_button.grid(row=1, column=0, padx=3, pady=3, sticky="ew")
        ttk.Button(controls, text="Refresh", command=self.refresh_files).grid(
            row=1, column=1, padx=3, pady=3, sticky="ew"
        )
        ttk.Button(controls, text="Replay Selected", command=self.replay_selected).grid(
            row=1, column=2, padx=3, pady=3, sticky="ew"
        )
        for col in range(3):
            controls.columnconfigure(col, weight=1)

        status = ttk.LabelFrame(left, text="Status", padding=10)
        status.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(
            status, textvariable=self.status_var, wraplength=560, justify="left"
        ).pack(anchor="w", fill="x")
        ttk.Label(
            status, textvariable=self.progress_var, foreground="#52606d",
            wraplength=560, justify="left",
        ).pack(anchor="w", fill="x", pady=(5, 0))
        self.pose_status_label = tk.Label(
            status,
            textvariable=self.pose_check_var,
            anchor="w",
            justify="left",
            wraplength=560,
            font=("Sans", 12, "bold"),
            fg="#65717d",
        )
        self.pose_status_label.pack(fill="x", pady=(7, 0))
        ttk.Label(
            status, textvariable=self.live_var, foreground="#52606d",
            wraplength=560, justify="left",
        ).pack(anchor="w", fill="x", pady=(5, 0))

        joints = ttk.LabelFrame(left, text="Live Robot Pose", padding=8)
        joints.grid(row=4, column=0, sticky="nsew", pady=(0, 8))
        joints.columnconfigure(0, weight=1)
        joints.rowconfigure(0, weight=1)
        self.joint_table = ttk.Treeview(
            joints,
            columns=("joint", "position", "error", "limit"),
            show="headings",
            height=7,
        )
        headings = {"joint": "Joint", "position": "Position", "error": "Home Error", "limit": "Status"}
        widths = {"joint": 90, "position": 125, "error": 125, "limit": 90}
        for column in headings:
            self.joint_table.heading(column, text=headings[column])
            self.joint_table.column(column, width=widths[column], anchor="center")
        for name in JOINT_NAMES:
            self.joint_rows[name] = self.joint_table.insert(
                "", "end", values=(name, "--", "--", "Waiting")
            )
        self.joint_table.tag_configure("ok", foreground="#137333")
        self.joint_table.tag_configure("bad", foreground="#b3261e")
        self.joint_table.tag_configure("waiting", foreground="#65717d")
        self.joint_table.grid(row=0, column=0, sticky="nsew")
        ttk.Label(joints, textvariable=self.eef_var, foreground="#34495e").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )

        files = ttk.LabelFrame(left, text="Saved Episodes", padding=8)
        files.grid(row=5, column=0, sticky="nsew")
        files.columnconfigure(0, weight=1)
        files.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(files, height=4, font=("Sans", 11))
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(files, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)

        preview = ttk.LabelFrame(right, text="Live Camera Views", padding=8)
        preview.grid(row=0, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        preview.rowconfigure(0, weight=1)
        for col, (key, title) in enumerate(
            (("cam_high", "Third-Person Camera"), ("cam_wrist", "Wrist Camera"))
        ):
            card = tk.Frame(preview, bg="#18232f", bd=0, highlightthickness=0)
            card.grid(row=0, column=col, sticky="nsew", padx=5)
            preview.columnconfigure(col, weight=1)
            tk.Label(
                card,
                text=title,
                bg="#18232f",
                fg="#f5f7fa",
                font=("Sans", 13, "bold"),
            ).pack(fill="x", padx=8, pady=(8, 5))
            label = tk.Label(
                card,
                text="Waiting for camera frames...",
                bg="#0f1720",
                fg="#aab7c4",
                width=1,
                height=1,
            )
            label.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.preview_labels[key] = label

        telemetry = ttk.LabelFrame(right, text="Live End-Effector State", padding=10)
        telemetry.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(telemetry, textvariable=self.eef_var, font=("Sans", 12, "bold")).pack(anchor="w")
        ttk.Label(
            telemetry,
            text="The home-pose check uses measured joint feedback. Camera panels show the latest captured frames.",
            foreground="#65717d",
            wraplength=820,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(5, 0))

    @property
    def out_dir(self) -> pathlib.Path:
        path = pathlib.Path(self.out_var.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_pose_check_config(self):
        joint_tol_deg = float(self.joint_tol_var.get())
        gripper_tol_mm = float(self.gripper_tol_var.get())
        if not np.isfinite(joint_tol_deg) or joint_tol_deg <= 0:
            raise ValueError("Joint tolerance must be positive.")
        if not np.isfinite(gripper_tol_mm) or gripper_tol_mm <= 0:
            raise ValueError("Gripper tolerance must be positive.")
        self.joint_tolerance_rad = float(np.deg2rad(joint_tol_deg))
        self.gripper_tolerance_m = float(gripper_tol_mm / 1000.0)

    def _check_initial_pose(self, qpos: np.ndarray):
        return check_initial_pose(
            qpos,
            self.start_qpos,
            self.joint_tolerance_rad,
            self.gripper_tolerance_m,
        )

    def _update_start_button(self):
        if self.piper is None or self.cameras is None or self.recording or self.reset_thread is not None:
            state = "disabled"
        else:
            state = "normal"
        self.start_button.configure(state=state)

    def toggle_connection(self):
        if self.piper is not None:
            self.disconnect()
            return
        try:
            self._load_pose_check_config()
            fps = int(self.fps_var.get())
            camera_fps = int(self.camera_fps_var.get())
            if fps <= 0:
                raise ValueError("Capture rate must be positive.")
            if camera_fps <= 0:
                raise ValueError("Camera source rate must be positive.")
            if fps > camera_fps:
                raise ValueError("Capture rate cannot exceed the camera source rate.")
            self.capture_fps = fps
            self.camera_fps = camera_fps
            self.status_var.set("Connecting to the robot and cameras...")
            self.pose_check_var.set("Waiting for robot feedback...")
            self.root.update_idletasks()
            self.session = CollectionSession(
                CollectionConfig(
                    can_name=self.can_var.get().strip(),
                    cam_high_device=self.high_var.get().strip(),
                    cam_wrist_device=self.wrist_var.get().strip(),
                    capture_fps=fps,
                    camera_fps=camera_fps,
                    output_dir=self.out_dir,
                )
            )
            checks = self.session.connect()
            self.piper = self.session.piper
            self.cameras = self.session.cameras
            self.episode_index = self.session.episode_index
            with self.data_lock:
                self.latest_images = {}
                self.latest_qpos = None
                self.latest_state = None
                self.latest_pose_ok = None
                self.latest_pose_reason = "Waiting for robot feedback"
            self.capture_stop = threading.Event()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            camera_status = ", ".join(
                f"{key}={info['fps']:.0f}FPS" for key, info in checks.items()
            )
            self.status_var.set(
                f"Connected to {self.can_var.get()} | Capture: {fps} Hz | {camera_status} | "
                f"Next episode: {self.episode_index:04d}"
            )
            self.connect_button.configure(text="Disconnect")
            self.reset_button.configure(state="normal")
            self._update_start_button()
        except Exception as exc:
            self.status_var.set(f"Connection failed: {exc}")
            self._cleanup_devices()
            messagebox.showerror("Connection Failed", str(exc))

    def start_episode(self):
        if self.session is None or self.recording:
            return
        try:
            self._load_pose_check_config()
        except ValueError as exc:
            messagebox.showwarning("Invalid Home-Pose Settings", str(exc))
            return
        with self.data_lock:
            qpos = None if self.latest_qpos is None else self.latest_qpos.copy()
            state = None if self.latest_state is None else self.latest_state.copy()
        if qpos is None:
            messagebox.showwarning(
                "Cannot Start Collection",
                "No robot feedback has been received yet. Wait a few seconds and try again.",
            )
            return
        pose_ok, reason, errors = self._check_initial_pose(qpos)
        with self.data_lock:
            self.latest_pose_ok = pose_ok
            self.latest_pose_reason = reason
            self.latest_pose_errors = errors
        self._update_telemetry(qpos, state, pose_ok, reason, errors)

        task_name = self.task_var.get().strip()
        instruction = self.instruction_var.get().strip() or task_name.replace("_", " ")
        try:
            with self.data_lock:
                label = self.session.start_episode(task_name, instruction)
                self.recording = True
        except Exception as exc:
            messagebox.showerror("Cannot Start Episode", str(exc))
            return
        self.task_var.set(label.task_name)
        self.instruction_var.set(label.instruction)
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.reset_button.configure(state="disabled")
        self.status_var.set(
            f"Recording episode {self.episode_index:04d} | Instruction: {label.instruction}"
        )
        self.progress_var.set("Frames: 0")

    def _capture_loop(self):
        assert self.capture_stop is not None
        fps = self.capture_fps
        dt = 1.0 / fps
        try:
            while not self.capture_stop.is_set():
                t0 = time.time()
                with self.data_lock:
                    if self.session is None:
                        return
                    sample = self.session.capture_once()
                    state = np.asarray(sample.state).copy()
                    qpos = np.asarray(sample.joint_qpos).copy()
                    pose_ok, pose_reason, pose_errors = self._check_initial_pose(qpos)
                    self.latest_images = {
                        key: np.asarray(value).copy() for key, value in sample.images.items()
                    }
                    self.latest_state = state
                    self.latest_qpos = qpos
                    self.latest_pose_ok = pose_ok
                    self.latest_pose_reason = pose_reason
                    self.latest_pose_errors = pose_errors.copy()
                    is_recording = self.recording
                    count = self.session.frame_count if is_recording else 0
                if is_recording:
                    self.messages.put(("progress", count))
                delay = dt - (time.time() - t0)
                if delay > 0:
                    time.sleep(delay)
        except Exception as exc:
            self.messages.put(("error", str(exc)))

    def stop_episode(self):
        if not self.recording:
            return
        with self.data_lock:
            self.session.stop_episode()
            self.recording = False
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping the episode and preparing it for review...")
        self.root.after(100, self._finish_stop)

    def _finish_stop(self):
        self._update_start_button()
        self.reset_button.configure(state="normal" if self.piper is not None else "disabled")
        if self.session is None or self.session.frame_count == 0:
            if self.session is not None and self.session.state is SessionState.REVIEW:
                self.session.discard_episode()
            self.status_var.set("The episode is empty and was not saved.")
            return
        self._ask_label_and_save()

    def _ask_label_and_save(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Review Episode")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        ttk.Label(
            dialog,
            text=f"Episode {self.episode_index:04d} | {self.session.frame_count} frames",
        ).pack(padx=20, pady=15)
        buttons = ttk.Frame(dialog)
        buttons.pack(pady=(0, 15))

        def finish(choice):
            if choice == "discard":
                self.session.discard_episode()
                dialog.destroy()
                self.status_var.set("Episode discarded.")
                self._update_start_button()
                return
            instruction = self.instruction_var.get().strip() or self.task_var.get().replace("_", " ")
            try:
                path, stats = self.session.save_episode(
                    success=(choice == "success"),
                    task_name=self.task_var.get().strip(),
                    instruction=instruction,
                )
            except Exception as exc:
                messagebox.showerror("Episode Validation Failed", str(exc), parent=dialog)
                return
            self.episode_index = self.session.episode_index
            dialog.destroy()
            self.status_var.set(
                f"Saved and validated: {path} | FPS={stats.actual_fps:.2f}"
            )
            self.refresh_files()
            self._update_start_button()

        ttk.Button(buttons, text="Save as Success", command=lambda: finish("success")).pack(
            side="left", padx=5
        )
        ttk.Button(buttons, text="Save as Failure", command=lambda: finish("failure")).pack(
            side="left", padx=5
        )
        ttk.Button(buttons, text="Discard", command=lambda: finish("discard")).pack(
            side="left", padx=5
        )

    def reset_arm(self):
        if self.session is None or self.recording or self.reset_thread is not None:
            return
        confirmed = messagebox.askyesno(
            "Reset to Home",
            "Move the robot smoothly to the all-zero joint pose and close the gripper?",
        )
        if not confirmed:
            return
        self.start_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.status_var.set("Resetting the robot to Home...")
        self.pose_check_var.set("Reset in progress; starting an episode is temporarily disabled.")
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
            self.status_var.set("Home reset command completed. Waiting for the pose check to pass.")
        else:
            self.status_var.set(f"Reset failed: {error}")
            messagebox.showerror("Reset Failed", error or "Unknown error")
        if self.piper is not None and not self.recording:
            self.reset_button.configure(state="normal")
        self._update_start_button()

    def _update_telemetry(
        self,
        qpos: np.ndarray | None,
        state: np.ndarray | None,
        pose_ok: bool | None,
        pose_reason: str,
        pose_errors: np.ndarray | None,
    ):
        if qpos is None or np.asarray(qpos).shape != (7,):
            for name in JOINT_NAMES:
                self.joint_table.item(
                    self.joint_rows[name],
                    values=(name, "--", "--", "Waiting"),
                    tags=("waiting",),
                )
            self.eef_var.set("EEF: --")
            self.live_var.set("Live telemetry: waiting for robot feedback")
        else:
            qpos = np.asarray(qpos, dtype=np.float64)
            errors = np.zeros(7, dtype=np.float64) if pose_errors is None else np.asarray(pose_errors)
            for index, name in enumerate(JOINT_NAMES):
                if index < 6:
                    position = f"{np.degrees(qpos[index]):+.2f}°"
                    error = f"{np.degrees(errors[index]):+.2f}°"
                    ok = abs(errors[index]) <= self.joint_tolerance_rad
                else:
                    position = f"{qpos[index] * 1000:+.2f} mm"
                    error = f"{errors[index] * 1000:+.2f} mm"
                    ok = abs(errors[index]) <= self.gripper_tolerance_m
                tag = "ok" if ok else "bad"
                self.joint_table.item(
                    self.joint_rows[name],
                    values=(name, position, error, "OK" if ok else "CHECK"),
                    tags=(tag,),
                )
            if state is not None and np.asarray(state).shape == (10,):
                state = np.asarray(state)
                xyz = ", ".join(f"{value:+.3f}" for value in state[:3])
                rot6d = ", ".join(f"{value:+.2f}" for value in state[3:9])
                self.eef_var.set(f"EEF xyz (m): [{xyz}]   gripper fraction: {state[9]:.3f}")
                self.live_var.set(f"EEF rotation-6D: [{rot6d}]")
            else:
                self.eef_var.set("EEF: --")
                self.live_var.set("Live telemetry: state unavailable")

        self.pose_check_var.set(f"Pose: {pose_reason}")
        if pose_ok is True:
            self.pose_status_label.configure(fg="#137333")
        elif pose_ok is False:
            self.pose_status_label.configure(fg="#b3261e")
        else:
            self.pose_status_label.configure(fg="#65717d")

    def _poll_messages(self):
        with self.data_lock:
            preview = {key: value.copy() for key, value in self.latest_images.items()}
            qpos = None if self.latest_qpos is None else self.latest_qpos.copy()
            state = None if self.latest_state is None else self.latest_state.copy()
            pose_ok = self.latest_pose_ok
            pose_reason = self.latest_pose_reason
            pose_errors = self.latest_pose_errors.copy()
            is_recording = self.recording
        if preview:
            self._show_preview(preview)
        self._update_telemetry(qpos, state, pose_ok, pose_reason, pose_errors)
        if self.piper is not None and not is_recording and self.reset_thread is None:
            self._update_start_button()

        try:
            while True:
                kind, *payload = self.messages.get_nowait()
                if kind == "progress":
                    self.progress_var.set(f"Recorded frames: {payload[0]}")
                elif kind == "error":
                    error = payload[0]
                    with self.data_lock:
                        if self.session is not None and self.session.state is SessionState.RECORDING:
                            self.session.stop_episode()
                            self.session.discard_episode()
                        self.recording = False
                    self._cleanup_devices()
                    self.connect_button.configure(text="Connect")
                    self.start_button.configure(state="disabled")
                    self.stop_button.configure(state="disabled")
                    self.reset_button.configure(state="disabled")
                    self.progress_var.set("Episode discarded because device feedback failed.")
                    self.status_var.set(f"Collection aborted: {error}")
                    messagebox.showerror(
                        "Collection Aborted",
                        f"{error}\n\nThe current episode was discarded. "
                        "Reconnect the devices before collecting again.",
                    )
                elif kind == "reset_done":
                    self._finish_reset(True)
                elif kind == "reset_error":
                    self._finish_reset(False, payload[0])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _show_preview(self, images: dict[str, np.ndarray]):
        if Image is not None and ImageTk is not None:
            for key, frame in images.items():
                label = self.preview_labels.get(key)
                if label is None:
                    continue
                rgb = np.asarray(frame)
                if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4):
                    rgb = rgb.transpose(1, 2, 0)
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    continue
                rgb = np.ascontiguousarray(rgb.astype(np.uint8, copy=False))
                image = Image.fromarray(rgb, mode="RGB")
                resampling = getattr(Image, "Resampling", Image).BILINEAR
                image = image.resize(PREVIEW_SIZE, resampling)
                photo = ImageTk.PhotoImage(image=image)
                label.configure(image=photo, text="")
                self.preview_photos[key] = photo
            return

        # Minimal-install fallback.  The normal project environment includes
        # Pillow, so this path is only used when Tk cannot embed PhotoImage.
        for key, frame in images.items():
            image = np.asarray(frame).transpose(1, 2, 0)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, PREVIEW_SIZE, interpolation=cv2.INTER_NEAREST)
            title = "Third-person - cam_high" if key == "cam_high" else "Wrist - cam_wrist"
            cv2.putText(image, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
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
            messagebox.showinfo("Replay Episode", "Select an episode first.")
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
            self.cameras = None
        if self.session is not None:
            self.session.disconnect(discard_review=True)
        self.session = None
        self.piper = None
        with self.data_lock:
            self.latest_images = {}
            self.latest_qpos = None
            self.latest_state = None
            self.latest_pose_ok = None
            self.latest_pose_reason = "Waiting for robot feedback"
            self.latest_pose_errors = np.zeros(7, dtype=np.float32)
        self.preview_photos.clear()
        for key, label in self.preview_labels.items():
            label.configure(image="", text="Waiting for camera frames...")
        cv2.destroyAllWindows()

    def disconnect(self):
        if self.recording:
            messagebox.showwarning("Cannot Disconnect", "Stop the current episode first.")
            return
        if self.reset_thread is not None:
            messagebox.showwarning("Cannot Disconnect", "Wait for the reset operation to finish.")
            return
        self._cleanup_devices()
        self.connect_button.configure(text="Connect")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.status_var.set("Disconnected")
        self.pose_check_var.set("Waiting for robot feedback")

    def close(self):
        if self.recording:
            if not messagebox.askyesno(
                "Exit",
                "The current episode has not been saved. Exit anyway?",
            ):
                return
            with self.data_lock:
                if self.session is not None and self.session.state is SessionState.RECORDING:
                    self.session.stop_episode()
                self.recording = False
        self._cleanup_devices()
        self.root.destroy()


def main():
    root = tk.Tk()
    CollectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
