"""GUI for collecting single-arm or bimanual output-arm episodes.

The GUI records the same explicit NPZ contracts as :mod:`collect_output_arm`:
single-arm 7D/10D with two cameras, or bimanual 14D/20D in fixed left+right
order with three cameras.  It reads measured output-arm feedback only; for
same-step master-arm joint commands use :mod:`teleop_single` or :mod:`teleop`.

The live view stays in the Tk window and shows every active RGB camera, all
measured joints, schema-specific state, and the initial-pose check.  Pose
mismatches are visible warnings rather than collection blockers; missing or
stale hardware feedback is still rejected before and during recording.
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
    DEFAULT_LEFT_CAN,
    DEFAULT_LEFT_WRIST_DEVICE,
    DEFAULT_RIGHT_CAN,
    DEFAULT_RIGHT_WRIST_DEVICE,
    DEFAULT_WRIST_DEVICE,
    reset_robot_arms,
)
from piper_data_contract import BIMANUAL, DELIVERY_SCHEMA, JOINT_SCHEMA, SINGLE_ARM, EpisodeContract


JOINT_NAMES = tuple(f"J{i}" for i in range(1, 7)) + ("Gripper",)
SINGLE_PREVIEW_SIZE = (480, 360)
BIMANUAL_PREVIEW_SIZE = (360, 270)


def check_initial_pose(
    qpos: np.ndarray,
    start_qpos: np.ndarray,
    joint_tolerance_rad: float,
    gripper_tolerance_m: float,
) -> tuple[bool, str, np.ndarray]:
    """Check measured qpos against the configured home pose.

    Each arm contributes six Piper joint angles in radians followed by one
    gripper position in metres.  Bimanual vectors use fixed ``left + right``
    order.  The returned error vector uses the same units.  This is a pure
    helper so it can be tested without hardware or a Tk display.
    """
    measured = np.asarray(qpos, dtype=np.float64).reshape(-1)
    target = np.asarray(start_qpos, dtype=np.float64).reshape(-1)
    if measured.shape != target.shape or measured.size not in {7, 14}:
        size = target.size if target.size in {7, 14} else measured.size
        return (
            False,
            "invalid qpos shape (expected matching 7 or 14 values)",
            np.full(size if size in {7, 14} else 7, np.nan),
        )
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(target)):
        return False, "robot feedback contains NaN/Inf", np.full(measured.shape, np.nan)
    if joint_tolerance_rad <= 0 or gripper_tolerance_m <= 0:
        return False, "pose tolerances must be positive", np.full(measured.shape, np.nan)

    errors = measured - target
    details: list[str] = []
    arm_sides = ("",) if measured.size == 7 else ("L-", "R-")
    for arm_index, side_prefix in enumerate(arm_sides):
        start = arm_index * 7
        bad_joints = np.flatnonzero(
            np.abs(errors[start : start + 6]) > joint_tolerance_rad
        )
        details.extend(
            f"{side_prefix}J{index + 1}={np.degrees(errors[start + index]):+.1f}°"
            for index in bad_joints
        )
        if abs(float(errors[start + 6])) > gripper_tolerance_m:
            details.append(
                f"{side_prefix}Gripper={errors[start + 6] * 1000:+.1f} mm"
            )
    if not details:
        arm_text = "both robots are" if measured.size == 14 else "robot is"
        return True, f"OK: {arm_text} at the home pose", errors.astype(np.float32)
    return False, "outside tolerance: " + ", ".join(details), errors.astype(np.float32)


class CollectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Piper · π0.5 Data Collection")
        self.root.geometry("1450x980")
        self.root.minsize(1100, 780)
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
        self.latest_pose_reason = "等待机械臂反馈"
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
        self.preview_cards: dict[str, tk.Frame] = {}
        self.preview_title_labels: dict[str, tk.Label] = {}
        self.preview_key_to_slot: dict[str, str] = {}
        self.joint_rows: dict[str, str] = {}

        self.arm_mode_var = tk.StringVar(value=SINGLE_ARM)
        self.schema_var = tk.StringVar(value=DELIVERY_SCHEMA)
        self.arm_side_var = tk.StringVar(value="right")
        self.can_var = tk.StringVar(value=DEFAULT_CAN)
        self.left_can_var = tk.StringVar(value=DEFAULT_LEFT_CAN)
        self.right_can_var = tk.StringVar(value=DEFAULT_RIGHT_CAN)
        self.high_var = tk.StringVar(value=DEFAULT_HIGH_DEVICE)
        self.wrist_var = tk.StringVar(value=DEFAULT_WRIST_DEVICE)
        self.left_wrist_var = tk.StringVar(value=DEFAULT_LEFT_WRIST_DEVICE)
        self.right_wrist_var = tk.StringVar(value=DEFAULT_RIGHT_WRIST_DEVICE)
        self.fps_var = tk.StringVar(value="20")
        self.camera_fps_var = tk.StringVar(value=str(DEFAULT_CAMERA_FPS))
        self.joint_tol_var = tk.StringVar(value="5.0")
        self.gripper_tol_var = tk.StringVar(value="5.0")
        self.out_var = tk.StringVar(value="episodes_piper_v21")
        self.task_var = tk.StringVar(value="pick_cube")
        self.instruction_var = tk.StringVar(value="pick up the cube")
        self.status_var = tk.StringVar(value="未连接")
        self.progress_var = tk.StringVar(value="尚未开始 episode")
        self.pose_check_var = tk.StringVar(value="等待机械臂反馈")
        self.eef_var = tk.StringVar(value="EEF: --")
        self.live_var = tk.StringVar(value="Live telemetry: --")
        self._build_ui()
        self._configure_mode_ui()
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
        style.configure("TButton", font=("Sans", 12), padding=(11, 6))
        style.configure("TLabel", font=("Sans", 12))
        style.configure("TLabelframe.Label", font=("Sans", 13, "bold"))
        style.configure("Treeview", rowheight=29, font=("Sans", 11))
        style.configure("Treeview.Heading", font=("Sans", 11, "bold"))

        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=10)
        left = ttk.Frame(main, padding=(2, 0, 8, 0))
        right = ttk.Frame(main, padding=(8, 0, 2, 0))
        main.add(left, weight=1)
        main.add(right, weight=2)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(4, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        config = ttk.LabelFrame(left, text="设备与任务", padding=10)
        config.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        config.columnconfigure(1, weight=1)
        selectors = [
            ("机械臂模式", self.arm_mode_var, (("单臂", SINGLE_ARM), ("双臂", BIMANUAL))),
            ("训练 schema", self.schema_var, (("关节 joint", JOINT_SCHEMA), ("末端 delivery", DELIVERY_SCHEMA))),
            ("单臂侧", self.arm_side_var, (("right", "right"), ("left", "left"))),
        ]
        self.mode_selectors: list[ttk.Combobox] = []
        self.mode_display_vars: list[tk.StringVar] = []
        for row, (label, variable, values) in enumerate(selectors):
            ttk.Label(config, text=label, width=14).grid(row=row, column=0, sticky="w", pady=3)
            display_to_value = dict(values)
            value_to_display = {value: display for display, value in values}
            display_var = tk.StringVar(value=value_to_display[variable.get()])
            selector = ttk.Combobox(
                config,
                textvariable=display_var,
                values=tuple(display_to_value),
                state="readonly",
                width=34,
            )
            selector.grid(row=row, column=1, sticky="ew", pady=3, padx=(8, 0))

            def changed(_event=None, *, source=display_var, target=variable, mapping=display_to_value):
                target.set(mapping[source.get()])
                self._configure_mode_ui()

            selector.bind("<<ComboboxSelected>>", changed)
            self.mode_selectors.append(selector)
            self.mode_display_vars.append(display_var)

        rows = [
            ("单臂 CAN", self.can_var, "single"),
            ("左臂 CAN", self.left_can_var, "bimanual"),
            ("右臂 CAN", self.right_can_var, "bimanual"),
            ("顶部相机", self.high_var, "all"),
            ("单臂腕部相机", self.wrist_var, "single"),
            ("左腕相机", self.left_wrist_var, "bimanual"),
            ("右腕相机", self.right_wrist_var, "bimanual"),
            ("采集频率 Hz", self.fps_var),
            ("相机源频率 Hz", self.camera_fps_var),
            ("输出目录", self.out_var),
            ("任务名", self.task_var),
            ("指令", self.instruction_var),
        ]
        self.device_rows: dict[str, list[tk.Widget]] = {"single": [], "bimanual": []}
        self.connection_entries: list[ttk.Entry] = []
        for offset, item in enumerate(rows, start=len(selectors)):
            label, var, *mode = item
            label_widget = ttk.Label(config, text=label, width=14)
            entry = ttk.Entry(config, textvariable=var, width=36)
            label_widget.grid(row=offset, column=0, sticky="w", pady=3)
            entry.grid(
                row=offset, column=1, sticky="ew", pady=3, padx=(8, 0)
            )
            if mode and mode[0] in self.device_rows:
                self.device_rows[mode[0]].extend((label_widget, entry))
            if label not in {"任务名", "指令"}:
                self.connection_entries.append(entry)
        self.device_entries = [
            widget
            for widgets in self.device_rows.values()
            for widget in widgets
            if isinstance(widget, ttk.Entry)
        ]

        pose_config = ttk.LabelFrame(left, text="初始位姿安全检查", padding=10)
        pose_config.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        pose_config.columnconfigure(1, weight=1)
        ttk.Label(pose_config, text="关节误差阈值").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(pose_config, textvariable=self.joint_tol_var, width=9).grid(
            row=0, column=1, sticky="w", padx=(8, 0), pady=3
        )
        ttk.Label(pose_config, text="°（J1-J6）").grid(row=0, column=2, sticky="w", padx=(5, 0))
        ttk.Label(pose_config, text="夹爪误差阈值").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(pose_config, textvariable=self.gripper_tol_var, width=9).grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=3
        )
        ttk.Label(pose_config, text="mm").grid(row=1, column=2, sticky="w", padx=(5, 0))
        ttk.Label(
            pose_config,
            text="参考位姿：Reset to Home 使用的全零关节位姿",
            foreground="#65717d",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        controls = ttk.LabelFrame(left, text="采集控制", padding=8)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.connect_button = ttk.Button(controls, text="连接设备", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        self.start_button = ttk.Button(controls, text="开始 episode", command=self.start_episode, state="disabled")
        self.start_button.grid(row=0, column=1, padx=3, pady=3, sticky="ew")
        self.stop_button = ttk.Button(controls, text="停止 episode", command=self.stop_episode, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=3, pady=3, sticky="ew")
        self.reset_button = ttk.Button(controls, text="Reset to Home", command=self.reset_arm, state="disabled")
        self.reset_button.grid(row=1, column=0, padx=3, pady=3, sticky="ew")
        ttk.Button(controls, text="刷新文件", command=self.refresh_files).grid(
            row=1, column=1, padx=3, pady=3, sticky="ew"
        )
        ttk.Button(controls, text="回放选中 episode", command=self.replay_selected).grid(
            row=1, column=2, padx=3, pady=3, sticky="ew"
        )
        for col in range(3):
            controls.columnconfigure(col, weight=1)

        status = ttk.LabelFrame(left, text="状态", padding=10)
        status.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(status, textvariable=self.status_var, wraplength=520).pack(anchor="w")
        ttk.Label(status, textvariable=self.progress_var, foreground="#52606d").pack(anchor="w", pady=(5, 0))
        self.pose_status_label = tk.Label(
            status,
            textvariable=self.pose_check_var,
            anchor="w",
            justify="left",
            wraplength=520,
            font=("Sans", 12, "bold"),
            fg="#65717d",
        )
        self.pose_status_label.pack(fill="x", pady=(7, 0))
        ttk.Label(status, textvariable=self.live_var, foreground="#52606d").pack(anchor="w", pady=(5, 0))

        joints = ttk.LabelFrame(left, text="实时机械臂位姿", padding=8)
        joints.grid(row=4, column=0, sticky="nsew", pady=(0, 8))
        joints.columnconfigure(0, weight=1)
        joints.rowconfigure(0, weight=1)
        self.joint_table = ttk.Treeview(
            joints,
            columns=("joint", "position", "error", "limit"),
            show="headings",
            height=8,
        )
        headings = {"joint": "关节", "position": "当前值", "error": "相对 Home", "limit": "状态"}
        widths = {"joint": 95, "position": 130, "error": 130, "limit": 100}
        for column in headings:
            self.joint_table.heading(column, text=headings[column])
            self.joint_table.column(column, width=widths[column], anchor="center")
        self.joint_table.tag_configure("ok", foreground="#137333")
        self.joint_table.tag_configure("bad", foreground="#b3261e")
        self.joint_table.tag_configure("waiting", foreground="#65717d")
        self.joint_table.grid(row=0, column=0, sticky="nsew")
        ttk.Label(joints, textvariable=self.eef_var, foreground="#34495e").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )

        files = ttk.LabelFrame(left, text="已保存 episodes", padding=8)
        files.grid(row=5, column=0, sticky="nsew")
        files.columnconfigure(0, weight=1)
        files.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(files, height=7, font=("Sans", 11))
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(files, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)

        preview = ttk.LabelFrame(right, text="实时相机画面", padding=8)
        preview.grid(row=0, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        preview.columnconfigure(2, weight=1)
        preview.rowconfigure(0, weight=1)
        for col, (slot, title) in enumerate(
            (
                ("high", "顶部全局摄像头"),
                ("primary_wrist", "单臂夹爪上方摄像头"),
                ("right_wrist", "右臂夹爪上方摄像头"),
            )
        ):
            card = tk.Frame(preview, bg="#18232f", bd=0, highlightthickness=0)
            card.grid(row=0, column=col, sticky="nsew", padx=5)
            preview.columnconfigure(col, weight=1)
            title_label = tk.Label(
                card,
                text=title,
                bg="#18232f",
                fg="#f5f7fa",
                font=("Sans", 13, "bold"),
            )
            title_label.pack(fill="x", padx=8, pady=(8, 5))
            label = tk.Label(
                card,
                text="等待相机画面...",
                bg="#0f1720",
                fg="#aab7c4",
                width=48,
                height=20,
            )
            label.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.preview_cards[slot] = card
            self.preview_title_labels[slot] = title_label
            self.preview_labels[slot] = label

        telemetry = ttk.LabelFrame(right, text="Schema 实时状态", padding=10)
        telemetry.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(telemetry, textvariable=self.eef_var, font=("Sans", 12, "bold")).pack(anchor="w")
        ttk.Label(
            telemetry,
            text="单双臂位姿检查均以输出臂关节反馈为准；双臂 state/action 固定 left + right。",
            foreground="#65717d",
        ).pack(anchor="w", pady=(5, 0))

    @property
    def out_dir(self) -> pathlib.Path:
        path = pathlib.Path(self.out_var.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def arm_mode(self) -> str:
        return self.arm_mode_var.get()

    @property
    def schema(self) -> str:
        return self.schema_var.get()

    @property
    def arm_side(self) -> str:
        return "both" if self.arm_mode == BIMANUAL else self.arm_side_var.get()

    @property
    def contract(self) -> EpisodeContract:
        return EpisodeContract(
            schema=self.schema,
            arm_mode=self.arm_mode,
            arm_side=self.arm_side,
        )

    def _joint_display_names(self) -> tuple[str, ...]:
        if self.arm_mode == BIMANUAL:
            return tuple(f"L-{name}" for name in JOINT_NAMES) + tuple(
                f"R-{name}" for name in JOINT_NAMES
            )
        return JOINT_NAMES

    def _configure_mode_ui(self):
        if not hasattr(self, "joint_table"):
            return
        bimanual = self.arm_mode == BIMANUAL
        for widget in self.device_rows.get("single", []):
            if bimanual:
                widget.grid_remove()
            else:
                widget.grid()
        for widget in self.device_rows.get("bimanual", []):
            if bimanual:
                widget.grid()
            else:
                widget.grid_remove()

        self.start_qpos = np.zeros(14 if bimanual else 7, dtype=np.float32)
        self.latest_pose_errors = np.zeros_like(self.start_qpos)
        self.joint_rows.clear()
        for item in self.joint_table.get_children():
            self.joint_table.delete(item)
        for name in self._joint_display_names():
            self.joint_rows[name] = self.joint_table.insert(
                "", "end", values=(name, "--", "--", "等待")
            )

        if bimanual:
            self.preview_cards["right_wrist"].grid()
            self.preview_key_to_slot = {
                "cam_high": "high",
                "cam_left_wrist": "primary_wrist",
                "cam_right_wrist": "right_wrist",
            }
        else:
            self.preview_cards["right_wrist"].grid_remove()
            self.preview_key_to_slot = {
                "cam_high": "high",
                self.contract.camera_keys[1]: "primary_wrist",
            }
        self.preview_title_labels["primary_wrist"].configure(
            text="左臂夹爪上方摄像头" if bimanual else f"{self.arm_side} 臂夹爪上方摄像头"
        )
        for label in self.preview_labels.values():
            label.configure(image="", text="等待相机画面...")
        self.preview_photos.clear()
        if self.session is None:
            self.latest_qpos = None
            self.latest_state = None
            self.latest_pose_ok = None
            self.latest_pose_reason = "等待机械臂反馈"
            self.pose_check_var.set("等待机械臂反馈")
        self.eef_var.set("State: --")
        self.live_var.set(
            f"{self.arm_mode} / {self.schema} · "
            f"state={self.contract.state_dim}D action={self.contract.action_dim}D"
        )

    def _set_connection_config_enabled(self, enabled: bool) -> None:
        selector_state = "readonly" if enabled else "disabled"
        entry_state = "normal" if enabled else "disabled"
        for selector in self.mode_selectors:
            selector.configure(state=selector_state)
        for entry in self.connection_entries:
            entry.configure(state=entry_state)

    def _load_pose_check_config(self):
        joint_tol_deg = float(self.joint_tol_var.get())
        gripper_tol_mm = float(self.gripper_tol_var.get())
        if not np.isfinite(joint_tol_deg) or joint_tol_deg <= 0:
            raise ValueError("关节误差阈值必须是正数")
        if not np.isfinite(gripper_tol_mm) or gripper_tol_mm <= 0:
            raise ValueError("夹爪误差阈值必须是正数")
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
            # Keep the home-pose check as an operator warning only. Requiring
            # an all-zero pose here prevented valid teleoperation episodes
            # from starting; freshness checks still fail closed on missing or
            # stale Piper feedback.
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
                raise ValueError("采集频率必须是正数")
            if camera_fps <= 0:
                raise ValueError("相机源频率必须是正数")
            if fps > camera_fps:
                raise ValueError("采集频率不能高于相机源频率")
            self.capture_fps = fps
            self.camera_fps = camera_fps
            self.status_var.set("正在连接机械臂和相机...")
            self.pose_check_var.set("等待机械臂反馈，开始初始位姿检测...")
            self.root.update_idletasks()
            self.session = CollectionSession(
                CollectionConfig(
                    can_name=self.can_var.get().strip(),
                    cam_high_device=self.high_var.get().strip(),
                    cam_wrist_device=self.wrist_var.get().strip(),
                    capture_fps=fps,
                    camera_fps=camera_fps,
                    output_dir=self.out_dir,
                    schema=self.schema,
                    arm_mode=self.arm_mode,
                    arm_side=self.arm_side,
                    left_can_name=self.left_can_var.get().strip(),
                    right_can_name=self.right_can_var.get().strip(),
                    cam_left_wrist_device=self.left_wrist_var.get().strip(),
                    cam_right_wrist_device=self.right_wrist_var.get().strip(),
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
                self.latest_pose_reason = "等待机械臂反馈"
            self.capture_stop = threading.Event()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            camera_status = ", ".join(
                f"{key}={info['fps']:.0f}FPS" for key, info in checks.items()
            )
            can_status = (
                self.can_var.get().strip()
                if self.arm_mode == SINGLE_ARM
                else f"left={self.left_can_var.get().strip()}, right={self.right_can_var.get().strip()}"
            )
            self.status_var.set(
                f"已连接 {self.arm_mode}/{self.schema} ({can_status}) | "
                f"state={self.contract.state_dim}D action={self.contract.action_dim}D | "
                f"采集 {fps}Hz | {camera_status} | 下一个 episode: {self.episode_index:04d}"
            )
            self._set_connection_config_enabled(False)
            self.connect_button.configure(text="断开设备")
            self.reset_button.configure(state="normal")
            self._update_start_button()
        except Exception as exc:
            self.status_var.set(f"连接失败: {exc}")
            self._cleanup_devices()
            self._set_connection_config_enabled(True)
            messagebox.showerror("连接失败", str(exc))

    def start_episode(self):
        if self.session is None or self.recording:
            return
        try:
            self._load_pose_check_config()
        except ValueError as exc:
            messagebox.showwarning("初始位姿检查配置错误", str(exc))
            return
        with self.data_lock:
            qpos = None if self.latest_qpos is None else self.latest_qpos.copy()
            state = None if self.latest_state is None else self.latest_state.copy()
        if qpos is None:
            messagebox.showwarning("无法开始采集", "尚未收到机械臂状态反馈，请等待几秒后重试。")
            return
        pose_ok, reason, errors = self._check_initial_pose(qpos)
        with self.data_lock:
            self.latest_pose_ok = pose_ok
            self.latest_pose_reason = reason
            self.latest_pose_errors = errors
        # Pose mismatch is intentionally non-blocking. The red telemetry
        # warning remains visible so the operator can decide whether the
        # episode starts from the intended task pose. A missing/stale Piper
        # stream is rejected separately by the hardware reader.
        self._update_telemetry(qpos, state, pose_ok, reason, errors)

        task_name = self.task_var.get().strip()
        instruction = self.instruction_var.get().strip() or task_name.replace("_", " ")
        try:
            with self.data_lock:
                label = self.session.start_episode(task_name, instruction)
                self.recording = True
        except Exception as exc:
            messagebox.showerror("无法开始 episode", str(exc))
            return
        self.task_var.set(label.task_name)
        self.instruction_var.set(label.instruction)
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.reset_button.configure(state="disabled")
        self.status_var.set(
            f"正在录制 episode {self.episode_index:04d} | Instruction: {label.instruction}"
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
        self.status_var.set("正在停止并准备保存当前 episode...")
        self.root.after(100, self._finish_stop)

    def _finish_stop(self):
        self._update_start_button()
        self.reset_button.configure(state="normal" if self.piper is not None else "disabled")
        if self.session is None or self.session.frame_count == 0:
            if self.session is not None and self.session.state is SessionState.REVIEW:
                self.session.discard_episode()
            self.status_var.set("episode 为空，未保存")
            return
        self._ask_label_and_save()

    def _ask_label_and_save(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("标记当前 episode")
        dialog.transient(self.root)
        dialog.grab_set()
        ttk.Label(
            dialog,
            text=f"Episode {self.episode_index:04d} | {self.session.frame_count} 帧",
        ).pack(padx=20, pady=15)
        buttons = ttk.Frame(dialog)
        buttons.pack(pady=(0, 15))

        def finish(choice):
            if choice == "discard":
                self.session.discard_episode()
                dialog.destroy()
                self.status_var.set("已丢弃当前 episode")
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
                messagebox.showerror("Episode 校验失败", str(exc), parent=dialog)
                return
            self.episode_index = self.session.episode_index
            dialog.destroy()
            self.status_var.set(
                f"已保存并校验: {path} | FPS={stats.actual_fps:.2f}"
            )
            self.refresh_files()
            self._update_start_button()

        ttk.Button(buttons, text="保存为成功", command=lambda: finish("success")).pack(
            side="left", padx=5
        )
        ttk.Button(buttons, text="保存为失败", command=lambda: finish("failure")).pack(
            side="left", padx=5
        )
        ttk.Button(buttons, text="丢弃", command=lambda: finish("discard")).pack(
            side="left", padx=5
        )

    def reset_arm(self):
        if self.session is None or self.recording or self.reset_thread is not None:
            return
        confirmed = messagebox.askyesno(
            "Reset to Home",
            "将当前连接的机械臂平滑移动到全零关节参考位姿并关闭夹爪，是否继续？",
        )
        if not confirmed:
            return
        self.start_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.status_var.set("正在将机械臂复位到 Home...")
        self.pose_check_var.set("复位中：暂时禁止开始 episode")
        self.reset_thread = threading.Thread(target=self._reset_worker, daemon=True)
        self.reset_thread.start()

    def _reset_worker(self):
        try:
            reset_robot_arms(self.piper, arm_mode=self.arm_mode)
            self.messages.put(("reset_done",))
        except Exception as exc:
            self.messages.put(("reset_error", str(exc)))

    def _finish_reset(self, success: bool, error: str | None = None):
        self.reset_thread = None
        if success:
            self.status_var.set("Home 复位指令已完成，等待位姿检测通过")
        else:
            self.status_var.set(f"复位失败: {error}")
            messagebox.showerror("复位失败", error or "未知错误")
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
        expected_joint_dim = self.contract.joint_dim
        display_names = self._joint_display_names()
        qpos_array = None if qpos is None else np.asarray(qpos, dtype=np.float64)
        if qpos_array is None or qpos_array.shape != (expected_joint_dim,):
            for name in display_names:
                row = self.joint_rows.get(name)
                if row is not None:
                    self.joint_table.item(
                        row,
                        values=(name, "--", "--", "等待"),
                        tags=("waiting",),
                    )
            self.eef_var.set("State: --")
            self.live_var.set(
                f"Live telemetry: 等待 {expected_joint_dim}D 机械臂关节反馈"
            )
        else:
            errors = (
                qpos_array - self.start_qpos
                if pose_errors is None
                else np.asarray(pose_errors, dtype=np.float64)
            )
            if errors.shape != qpos_array.shape:
                errors = qpos_array - self.start_qpos
            for index, name in enumerate(display_names):
                local_index = index % 7
                if local_index < 6:
                    position = f"{np.degrees(qpos_array[index]):+.2f}°"
                    error = f"{np.degrees(errors[index]):+.2f}°"
                    ok = abs(errors[index]) <= self.joint_tolerance_rad
                else:
                    position = f"{qpos_array[index] * 1000:+.2f} mm"
                    error = f"{errors[index] * 1000:+.2f} mm"
                    ok = abs(errors[index]) <= self.gripper_tolerance_m
                self.joint_table.item(
                    self.joint_rows[name],
                    values=(name, position, error, "正常" if ok else "超差"),
                    tags=(("ok" if ok else "bad"),),
                )

            state_array = None if state is None else np.asarray(state, dtype=np.float64)
            if state_array is None or state_array.shape != (self.contract.state_dim,):
                actual = "missing" if state_array is None else f"{state_array.size}D"
                self.eef_var.set("State: --")
                self.live_var.set(
                    f"Live telemetry: state {actual}, expected {self.contract.state_dim}D"
                )
            elif self.schema == JOINT_SCHEMA:
                order = "left + right" if self.arm_mode == BIMANUAL else self.arm_side
                self.eef_var.set(
                    f"Joint state: {self.contract.state_dim}D measured qpos ({order})"
                )
                self.live_var.set(
                    f"π0.5 joint action: {self.contract.action_dim}D absolute joint target"
                )
            else:
                arm_summaries: list[str] = []
                rotation_summaries: list[str] = []
                for arm_index, side in enumerate(self.contract.arm_sides):
                    arm_state = state_array[arm_index * 10 : (arm_index + 1) * 10]
                    xyz = ", ".join(f"{value:+.3f}" for value in arm_state[:3])
                    rot6d = ", ".join(f"{value:+.2f}" for value in arm_state[3:9])
                    label = side.upper()[0]
                    arm_summaries.append(
                        f"{label} xyz(m)=[{xyz}] grip={arm_state[9]:.3f}"
                    )
                    rotation_summaries.append(f"{label} rot6D=[{rot6d}]")
                self.eef_var.set("   |   ".join(arm_summaries))
                self.live_var.set("   |   ".join(rotation_summaries))

        self.pose_check_var.set(f"初始位姿: {pose_reason}")
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
                    self.progress_var.set(f"已录制帧数: {payload[0]}")
                elif kind == "error":
                    self.status_var.set(f"采集错误: {payload[0]}")
                    if self.recording:
                        self.stop_episode()
                    messagebox.showerror("采集错误", payload[0])
                elif kind == "reset_done":
                    self._finish_reset(True)
                elif kind == "reset_error":
                    self._finish_reset(False, payload[0])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _show_preview(self, images: dict[str, np.ndarray]):
        preview_size = BIMANUAL_PREVIEW_SIZE if self.arm_mode == BIMANUAL else SINGLE_PREVIEW_SIZE
        if Image is not None and ImageTk is not None:
            for key, frame in images.items():
                slot = self.preview_key_to_slot.get(key)
                label = None if slot is None else self.preview_labels.get(slot)
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
                image = image.resize(preview_size, resampling)
                photo = ImageTk.PhotoImage(image=image)
                label.configure(image=photo, text="")
                self.preview_photos[slot] = photo
            return

        # Minimal-install fallback.  The normal project environment includes
        # Pillow, so this path is only used when Tk cannot embed PhotoImage.
        for key, frame in images.items():
            rgb = np.asarray(frame)
            if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4):
                rgb = rgb.transpose(1, 2, 0)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                continue
            image = cv2.cvtColor(rgb.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, preview_size, interpolation=cv2.INTER_NEAREST)
            title = key.replace("_", " ")
            cv2.putText(
                image,
                title,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
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
            messagebox.showinfo("回放", "请先选择一个 episode")
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
            self.latest_pose_reason = "等待机械臂反馈"
            self.latest_pose_errors = np.zeros(self.contract.joint_dim, dtype=np.float32)
        self.preview_photos.clear()
        for key, label in self.preview_labels.items():
            label.configure(image="", text="等待相机画面...")
        cv2.destroyAllWindows()

    def disconnect(self):
        if self.recording:
            messagebox.showwarning("无法断开", "请先停止当前 episode")
            return
        if self.reset_thread is not None:
            messagebox.showwarning("无法断开", "请等待复位完成")
            return
        self._cleanup_devices()
        self.connect_button.configure(text="连接设备")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self._set_connection_config_enabled(True)
        self._configure_mode_ui()
        self.status_var.set("未连接")
        self.pose_check_var.set("等待机械臂反馈")

    def close(self):
        if self.recording:
            if not messagebox.askyesno("退出", "当前 episode 尚未保存，确定退出吗？"):
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
