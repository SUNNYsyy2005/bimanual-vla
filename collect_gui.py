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

import os
import pathlib
import queue
import re
import shutil
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
from camera import select_video_device
from collect_output_arm import (
    DEFAULT_CAN,
    DEFAULT_CAMERA_FPS,
    DEFAULT_HIGH_DEVICE,
    DEFAULT_LEFT_CAN,
    DEFAULT_LEFT_WRIST_DEVICE,
    DEFAULT_RIGHT_CAN,
    DEFAULT_RIGHT_WRIST_DEVICE,
    DEFAULT_WRIST_DEVICE,
    next_episode_index,
    reset_robot_arms,
)
from piper_data_contract import BIMANUAL, DELIVERY_SCHEMA, JOINT_SCHEMA, SINGLE_ARM, EpisodeContract
from upload_dataset_4090 import DEFAULT_SERVER, safe_dataset_name


JOINT_NAMES = tuple(f"J{i}" for i in range(1, 7)) + ("Gripper",)
SINGLE_PREVIEW_SIZE = (480, 360)
BIMANUAL_PREVIEW_SIZE = (360, 270)
EPISODE_FILE_RE = re.compile(r"ep_\d+\.npz")


def move_episodes_to_trash(
    output_dir: str | pathlib.Path,
    selected_paths: list[str | pathlib.Path],
    *,
    timestamp: str | None = None,
) -> list[pathlib.Path]:
    """Move selected raw episodes into a recoverable dataset-local trash folder."""
    output_root = pathlib.Path(output_dir).expanduser().resolve()
    if not selected_paths:
        raise ValueError("no episodes were selected")
    sources: list[pathlib.Path] = []
    for value in selected_paths:
        source = pathlib.Path(value).expanduser().resolve()
        if source.parent != output_root or not EPISODE_FILE_RE.fullmatch(source.name):
            raise ValueError(f"unsafe episode path outside {output_root}: {source}")
        if not source.is_file():
            raise FileNotFoundError(source)
        sources.append(source)

    suffix = timestamp or f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
    trash_dir = output_root / ".trash" / suffix
    counter = 1
    while trash_dir.exists():
        trash_dir = output_root / ".trash" / f"{suffix}-{counter}"
        counter += 1
    trash_dir.mkdir(parents=True)

    moved: list[pathlib.Path] = []
    for source in sources:
        destination = trash_dir / source.name
        shutil.move(str(source), str(destination))
        moved.append(destination)
    return moved


def build_dataset_tool_command(
    *,
    python_executable: str,
    script_path: str | pathlib.Path,
    source_dir: str | pathlib.Path,
    dataset_name: str,
    fps: int,
    action: str,
    server: str = DEFAULT_SERVER,
    workers: int = 4,
    install_mode: str = "merge",
    allow_incomplete_gripper_coverage: bool = False,
    rebuild: bool = False,
) -> list[str]:
    """Build a token-free uploader command for the GUI background worker."""
    name = safe_dataset_name(dataset_name.strip())
    if fps <= 0 or workers <= 0:
        raise ValueError("FPS and upload workers must be positive")
    if action not in {"prepare", "upload"}:
        raise ValueError(f"unsupported dataset action: {action}")
    if install_mode not in {"install", "merge", "overwrite"}:
        raise ValueError(f"unsupported upload mode: {install_mode}")
    source = pathlib.Path(source_dir).expanduser().resolve()
    command = [
        python_executable,
        str(pathlib.Path(script_path).resolve()),
        str(source),
        "--name",
        name,
        "--fps",
        str(fps),
    ]
    if allow_incomplete_gripper_coverage:
        command.append("--allow-incomplete-gripper-coverage")
    if rebuild:
        command.append("--rebuild")
    if action == "prepare":
        command.append("--prepare-only")
        return command
    if not server.strip():
        raise ValueError("server URL must not be empty")
    command.extend(("--server", server.strip(), "--workers", str(workers)))
    if install_mode == "merge":
        command.append("--merge")
    elif install_mode == "overwrite":
        command.append("--overwrite")
    return command


def ask_english_yes_no(parent: tk.Misc, title: str, message: str) -> bool:
    """Show a Tk confirmation dialog with explicit English button labels.

    ``messagebox.askyesno`` delegates button labels to the desktop locale. On
    this workstation that produces Chinese labels with a font that cannot
    render them, so the native buttons appear garbled even though the message
    itself is English.
    """
    result = {"confirmed": False}
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()
    body = ttk.Frame(dialog, padding=16)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text=message, justify="left", wraplength=520).pack(
        anchor="w",
        fill="x",
    )
    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(16, 0))
    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)

    def finish(confirmed: bool) -> None:
        result["confirmed"] = confirmed
        dialog.destroy()

    ttk.Button(buttons, text="Yes", command=lambda: finish(True)).grid(
        row=0,
        column=0,
        sticky="ew",
        padx=(0, 4),
    )
    ttk.Button(buttons, text="No", command=lambda: finish(False)).grid(
        row=0,
        column=1,
        sticky="ew",
        padx=(4, 0),
    )
    dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
    dialog.wait_window()
    return bool(result["confirmed"])


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
            f"{side_prefix}J{index + 1}={np.degrees(errors[start + index]):+.1f} deg"
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
        self.root.title("Piper - pi0.5 Data Collection")
        self.root.geometry("1450x980")
        self.root.minsize(1100, 780)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.session: CollectionSession | None = None
        self.piper = None
        self.cameras = None
        self.capture_thread: threading.Thread | None = None
        self.capture_stop: threading.Event | None = None
        self.reset_thread: threading.Thread | None = None
        self.dataset_task_thread: threading.Thread | None = None
        self.dataset_task_process: subprocess.Popen | None = None
        self.dataset_task_name: str | None = None
        self.prepared_lerobot_path: str | None = None
        self.dataset_tools_window: tk.Toplevel | None = None
        self.dataset_log_widget: tk.Text | None = None
        self.dataset_action_buttons: list[ttk.Button] = []
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
        self.dataset_source_var = tk.StringVar(
            value=str(pathlib.Path(self.out_var.get()).expanduser().resolve())
        )
        self.dataset_name_var = tk.StringVar(value="episodes_piper_v21")
        self.dataset_server_var = tk.StringVar(
            value=os.environ.get("BIMANUAL_VLA_SERVER", DEFAULT_SERVER)
        )
        self.dataset_token_var = tk.StringVar(
            value=os.environ.get("BIMANUAL_VLA_SERVER_TOKEN", "")
        )
        self.dataset_workers_var = tk.StringVar(value="4")
        self.dataset_install_mode_var = tk.StringVar(value="merge")
        self.dataset_allow_gripper_var = tk.BooleanVar(value=False)
        self.dataset_rebuild_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Disconnected")
        self.progress_var = tk.StringVar(value="No episode started")
        self.pose_check_var = tk.StringVar(value="Waiting for robot feedback")
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

        config = ttk.LabelFrame(left, text="Devices and Task", padding=10)
        config.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        config.columnconfigure(1, weight=1)
        selectors = [
            ("Arm mode", self.arm_mode_var, (("Single arm", SINGLE_ARM), ("Bimanual", BIMANUAL))),
            (
                "Training schema",
                self.schema_var,
                (("Joint", JOINT_SCHEMA), ("End-effector delivery", DELIVERY_SCHEMA)),
            ),
            ("Single-arm side", self.arm_side_var, (("right", "right"), ("left", "left"))),
        ]
        self.mode_selectors: list[ttk.Combobox] = []
        self.mode_display_vars: list[tk.StringVar] = []
        for row, (label, variable, values) in enumerate(selectors):
            ttk.Label(config, text=label, width=22).grid(row=row, column=0, sticky="w", pady=3)
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
            ("Single arm CAN", self.can_var, "single"),
            ("Left-arm CAN", self.left_can_var, "bimanual"),
            ("Right-arm CAN", self.right_can_var, "bimanual"),
            ("Overhead camera", self.high_var, "all"),
            ("Single-arm wrist camera", self.wrist_var, "single"),
            ("Left wrist camera", self.left_wrist_var, "bimanual"),
            ("Right wrist camera", self.right_wrist_var, "bimanual"),
            ("Collection rate (Hz)", self.fps_var),
            ("Camera source rate (Hz)", self.camera_fps_var),
            ("Output directory", self.out_var),
            ("Task name", self.task_var),
            ("Instruction", self.instruction_var),
        ]
        self.device_rows: dict[str, list[tk.Widget]] = {"single": [], "bimanual": []}
        self.connection_entries: list[ttk.Entry] = []
        for offset, item in enumerate(rows, start=len(selectors)):
            label, var, *mode = item
            label_widget = ttk.Label(config, text=label, width=22)
            entry = ttk.Entry(config, textvariable=var, width=36)
            label_widget.grid(row=offset, column=0, sticky="w", pady=3)
            entry.grid(
                row=offset, column=1, sticky="ew", pady=3, padx=(8, 0)
            )
            if mode and mode[0] in self.device_rows:
                self.device_rows[mode[0]].extend((label_widget, entry))
            if label not in {"Task name", "Instruction"}:
                self.connection_entries.append(entry)
        self.device_entries = [
            widget
            for widgets in self.device_rows.values()
            for widget in widgets
            if isinstance(widget, ttk.Entry)
        ]

        pose_config = ttk.LabelFrame(left, text="Initial pose check", padding=10)
        pose_config.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        pose_config.columnconfigure(1, weight=1)
        ttk.Label(pose_config, text="Joint tolerance").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Entry(pose_config, textvariable=self.joint_tol_var, width=9).grid(
            row=0, column=1, sticky="w", padx=(8, 0), pady=3
        )
        ttk.Label(pose_config, text="deg (J1-J6)").grid(
            row=0, column=2, sticky="w", padx=(5, 0)
        )
        ttk.Label(pose_config, text="Gripper tolerance").grid(
            row=1, column=0, sticky="w", pady=3
        )
        ttk.Entry(pose_config, textvariable=self.gripper_tol_var, width=9).grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=3
        )
        ttk.Label(pose_config, text="mm").grid(
            row=1, column=2, sticky="w", padx=(5, 0)
        )
        ttk.Label(
            pose_config,
            text="Reference pose: all-zero joint pose used by Reset to Home",
            foreground="#65717d",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        controls = ttk.LabelFrame(left, text="Collection controls", padding=8)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.connect_button = ttk.Button(
            controls, text="Connect devices", command=self.toggle_connection
        )
        self.connect_button.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        self.start_button = ttk.Button(
            controls, text="Start episode", command=self.start_episode, state="disabled"
        )
        self.start_button.grid(row=0, column=1, padx=3, pady=3, sticky="ew")
        self.stop_button = ttk.Button(
            controls, text="Stop episode", command=self.stop_episode, state="disabled"
        )
        self.stop_button.grid(row=0, column=2, padx=3, pady=3, sticky="ew")
        self.reset_button = ttk.Button(controls, text="Reset to Home", command=self.reset_arm, state="disabled")
        self.reset_button.grid(row=1, column=0, padx=3, pady=3, sticky="ew")
        ttk.Button(controls, text="Refresh files", command=self.refresh_files).grid(
            row=1, column=1, padx=3, pady=3, sticky="ew"
        )
        ttk.Button(controls, text="Replay selected episode", command=self.replay_selected).grid(
            row=1, column=2, padx=3, pady=3, sticky="ew"
        )
        self.swap_camera_button = ttk.Button(
            controls,
            text="Swap Camera Roles",
            command=self.swap_camera_roles,
        )
        self.swap_camera_button.grid(row=2, column=0, columnspan=3, padx=3, pady=3, sticky="ew")
        for col in range(3):
            controls.columnconfigure(col, weight=1)

        status = ttk.LabelFrame(left, text="Status", padding=10)
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

        joints = ttk.LabelFrame(left, text="Live robot pose", padding=8)
        joints.grid(row=4, column=0, sticky="nsew", pady=(0, 8))
        joints.columnconfigure(0, weight=1)
        joints.rowconfigure(0, weight=1)
        self.joint_table = ttk.Treeview(
            joints,
            columns=("joint", "position", "error", "limit"),
            show="headings",
            height=8,
        )
        headings = {"joint": "Joint", "position": "Current", "error": "Relative to Home", "limit": "Status"}
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

        files = ttk.LabelFrame(left, text="Saved episodes", padding=8)
        files.grid(row=5, column=0, sticky="nsew")
        files.columnconfigure(0, weight=1)
        files.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(
            files,
            height=7,
            font=("Sans", 11),
            selectmode=tk.EXTENDED,
        )
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(files, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)
        file_actions = ttk.Frame(files)
        file_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        file_actions.columnconfigure(0, weight=1)
        file_actions.columnconfigure(1, weight=1)
        self.delete_episode_button = ttk.Button(
            file_actions,
            text="Delete selected data",
            command=self.delete_selected_episodes,
        )
        self.delete_episode_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.dataset_tools_button = ttk.Button(
            file_actions,
            text="Convert / upload dataset",
            command=self.open_dataset_tools,
        )
        self.dataset_tools_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        preview = ttk.LabelFrame(right, text="Live camera views", padding=8)
        preview.grid(row=0, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        preview.columnconfigure(2, weight=1)
        preview.rowconfigure(0, weight=1)
        for col, (slot, title) in enumerate(
            (
                ("high", "Overhead camera"),
                ("primary_wrist", "Single-arm wrist camera"),
                ("right_wrist", "Right wrist camera"),
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
                text="Waiting for camera...",
                bg="#0f1720",
                fg="#aab7c4",
                width=48,
                height=20,
            )
            label.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.preview_cards[slot] = card
            self.preview_title_labels[slot] = title_label
            self.preview_labels[slot] = label

        telemetry = ttk.LabelFrame(right, text="Live schema telemetry", padding=10)
        telemetry.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(telemetry, textvariable=self.eef_var, font=("Sans", 12, "bold")).pack(anchor="w")
        ttk.Label(
            telemetry,
            text="Pose checks use measured output-arm joint feedback. Bimanual state/action order is fixed to left + right.",
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

    def _camera_role_title(self, key: str) -> str:
        if key == "cam_high":
            return "Overhead camera"
        if key == "cam_left_wrist":
            return "Left wrist camera"
        if key == "cam_right_wrist":
            return "Right wrist camera"
        return f"{self.arm_side.capitalize()} wrist camera"

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
                "", "end", values=(name, "--", "--", "Waiting")
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
        if hasattr(self, "swap_camera_button"):
            self.swap_camera_button.configure(
                state="disabled" if bimanual or self.session is not None else "normal"
            )
        for camera_key, slot in self.preview_key_to_slot.items():
            self.preview_title_labels[slot].configure(text=self._camera_role_title(camera_key))
        for label in self.preview_labels.values():
            label.configure(image="", text="Waiting for camera...")
        self.preview_photos.clear()
        if self.session is None:
            self.latest_qpos = None
            self.latest_state = None
            self.latest_pose_ok = None
            self.latest_pose_reason = "Waiting for robot feedback"
            self.pose_check_var.set("Waiting for robot feedback")
        self.eef_var.set("State: --")
        self.live_var.set(
            f"{self.arm_mode} / {self.schema} - "
            f"state={self.contract.state_dim}D action={self.contract.action_dim}D"
        )

    def _set_connection_config_enabled(self, enabled: bool) -> None:
        selector_state = "readonly" if enabled else "disabled"
        entry_state = "normal" if enabled else "disabled"
        for selector in self.mode_selectors:
            selector.configure(state=selector_state)
        for entry in self.connection_entries:
            entry.configure(state=entry_state)
        if hasattr(self, "swap_camera_button"):
            self.swap_camera_button.configure(
                state=("normal" if enabled and self.arm_mode == SINGLE_ARM else "disabled")
            )

    def swap_camera_roles(self) -> None:
        """Swap the configured overhead and wrist camera for single-arm mode."""
        if self.arm_mode != SINGLE_ARM:
            messagebox.showinfo(
                "Swap Camera Roles",
                "For bimanual mode, configure the left and right wrist cameras separately.",
            )
            return
        if self.session is not None or self.piper is not None or self.cameras is not None:
            messagebox.showwarning(
                "Disconnect first",
                "Disconnect devices before swapping camera roles. Do not change camera semantics during an episode.",
            )
            return
        wrist_key = "cam_left_wrist" if self.arm_side == "left" else "cam_right_wrist"
        try:
            overhead = select_video_device("cam_high", self.high_var.get().strip())
            wrist = select_video_device(wrist_key, self.wrist_var.get().strip())
        except Exception as exc:
            messagebox.showerror("Cannot resolve cameras", str(exc))
            return
        self.high_var.set(str(wrist))
        self.wrist_var.set(str(overhead))
        self.status_var.set(
            f"Camera roles swapped: overhead={self.high_var.get()} | wrist={self.wrist_var.get()}"
        )

    def _load_pose_check_config(self):
        joint_tol_deg = float(self.joint_tol_var.get())
        gripper_tol_mm = float(self.gripper_tol_var.get())
        if not np.isfinite(joint_tol_deg) or joint_tol_deg <= 0:
            raise ValueError("Joint tolerance must be positive")
        if not np.isfinite(gripper_tol_mm) or gripper_tol_mm <= 0:
            raise ValueError("Gripper tolerance must be positive")
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
        if (
            self.piper is None
            or self.cameras is None
            or self.recording
            or self.reset_thread is not None
            or self._dataset_task_running()
        ):
            state = "disabled"
        else:
            # Keep the home-pose check as an operator warning only. Requiring
            # an all-zero pose here prevented valid teleoperation episodes
            # from starting; freshness checks still fail closed on missing or
            # stale Piper feedback.
            state = "normal"
        self.start_button.configure(state=state)

    def _dataset_task_running(self) -> bool:
        thread = getattr(self, "dataset_task_thread", None)
        return thread is not None and thread.is_alive()

    def _update_dataset_action_buttons(self) -> None:
        busy = self.recording or self._dataset_task_running()
        state = "disabled" if busy else "normal"
        if hasattr(self, "delete_episode_button"):
            self.delete_episode_button.configure(state=state)
        for button in getattr(self, "dataset_action_buttons", []):
            button.configure(state=state)

    def toggle_connection(self):
        if self.piper is not None:
            self.disconnect()
            return
        try:
            self._load_pose_check_config()
            fps = int(self.fps_var.get())
            camera_fps = int(self.camera_fps_var.get())
            if fps <= 0:
                raise ValueError("Collection rate must be positive")
            if camera_fps <= 0:
                raise ValueError("Camera source rate must be positive")
            if fps > camera_fps:
                raise ValueError("Collection rate cannot exceed camera source rate")
            self.capture_fps = fps
            self.camera_fps = camera_fps
            self.status_var.set("Connecting to robot and cameras...")
            self.pose_check_var.set("Waiting for robot feedback; starting initial pose check...")
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
                self.latest_pose_reason = "Waiting for robot feedback"
            self.capture_stop = threading.Event()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            camera_status_parts = []
            for key, info in checks.items():
                video_device = str(info.get("video_device") or info.get("configured_device") or "?")
                selected_device = str(info.get("selected_device") or video_device)
                if key == "cam_high":
                    self.high_var.set(selected_device)
                elif self.arm_mode == SINGLE_ARM:
                    self.wrist_var.set(selected_device)
                elif key == "cam_left_wrist":
                    self.left_wrist_var.set(selected_device)
                elif key == "cam_right_wrist":
                    self.right_wrist_var.set(selected_device)
                camera_status_parts.append(f"{key}={video_device} ({info['fps']:.0f}FPS)")
                slot = self.preview_key_to_slot.get(key)
                if slot is not None:
                    self.preview_title_labels[slot].configure(
                        text=f"{self._camera_role_title(key)}\n{video_device}"
                    )
            camera_status = ", ".join(camera_status_parts)
            can_status = (
                self.can_var.get().strip()
                if self.arm_mode == SINGLE_ARM
                else f"left={self.left_can_var.get().strip()}, right={self.right_can_var.get().strip()}"
            )
            self.status_var.set(
                f"Connected {self.arm_mode}/{self.schema} ({can_status}) | "
                f"state={self.contract.state_dim}D action={self.contract.action_dim}D | "
                f"Collection {fps}Hz | {camera_status} | Next episode: {self.episode_index:04d}"
            )
            self._set_connection_config_enabled(False)
            self.connect_button.configure(text="Disconnect devices")
            self.reset_button.configure(state="normal")
            self._update_start_button()
        except Exception as exc:
            self.status_var.set(f"Connection failed: {exc}")
            self._cleanup_devices()
            self._set_connection_config_enabled(True)
            messagebox.showerror("Connection failed", str(exc))

    def start_episode(self):
        if self.session is None or self.recording:
            return
        try:
            self._load_pose_check_config()
        except ValueError as exc:
            messagebox.showwarning("Initial pose configuration error", str(exc))
            return
        with self.data_lock:
            qpos = None if self.latest_qpos is None else self.latest_qpos.copy()
            state = None if self.latest_state is None else self.latest_state.copy()
        if qpos is None:
            messagebox.showwarning(
                "Cannot start collection",
                "No robot state feedback received yet; wait a few seconds and try again.",
            )
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
            messagebox.showerror("Cannot start episode", str(exc))
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
        self._update_dataset_action_buttons()

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
        self.status_var.set("Stopping and preparing to save the current episode...")
        self.root.after(100, self._finish_stop)

    def _finish_stop(self):
        self._update_start_button()
        self._update_dataset_action_buttons()
        self.reset_button.configure(state="normal" if self.piper is not None else "disabled")
        if self.session is None or self.session.frame_count == 0:
            if self.session is not None and self.session.state is SessionState.REVIEW:
                self.session.discard_episode()
            self.status_var.set("Episode is empty; nothing was saved")
            return
        self._ask_label_and_save()

    def _ask_label_and_save(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Label current episode")
        dialog.transient(self.root)
        dialog.grab_set()
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
                self.status_var.set("Current episode discarded")
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
                messagebox.showerror("Episode validation failed", str(exc), parent=dialog)
                return
            self.episode_index = self.session.episode_index
            dialog.destroy()
            self.status_var.set(
                f"Saved and validated: {path} | FPS={stats.actual_fps:.2f}"
            )
            self.refresh_files()
            self._update_start_button()

        ttk.Button(buttons, text="Save as success", command=lambda: finish("success")).pack(
            side="left", padx=5
        )
        ttk.Button(buttons, text="Save as failure", command=lambda: finish("failure")).pack(
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
            "Smoothly move the connected robot to the all-zero joint reference "
            "pose and close the gripper. Continue?",
        )
        if not confirmed:
            return
        self.start_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.status_var.set("Resetting robot to Home...")
        self.pose_check_var.set("Resetting: starting an episode is temporarily disabled")
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
            self.status_var.set("Home reset command completed; waiting for the pose check")
        else:
            self.status_var.set(f"Reset failed: {error}")
            messagebox.showerror("Reset failed", error or "Unknown error")
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
                        values=(name, "--", "--", "Waiting"),
                        tags=("waiting",),
                    )
            self.eef_var.set("State: --")
            self.live_var.set(
                f"Live telemetry: waiting for {expected_joint_dim}D robot joint feedback"
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
                    position = f"{np.degrees(qpos_array[index]):+.2f} deg"
                    error = f"{np.degrees(errors[index]):+.2f} deg"
                    ok = abs(errors[index]) <= self.joint_tolerance_rad
                else:
                    position = f"{qpos_array[index] * 1000:+.2f} mm"
                    error = f"{errors[index] * 1000:+.2f} mm"
                    ok = abs(errors[index]) <= self.gripper_tolerance_m
                self.joint_table.item(
                    self.joint_rows[name],
                    values=(name, position, error, "OK" if ok else "Out of tolerance"),
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
                    f"pi0.5 joint action: {self.contract.action_dim}D absolute joint target"
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

        self.pose_check_var.set(f"Initial pose: {pose_reason}")
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
                    self.status_var.set(f"Collection error: {payload[0]}")
                    if self.recording:
                        self.stop_episode()
                    messagebox.showerror("Collection error", payload[0])
                elif kind == "reset_done":
                    self._finish_reset(True)
                elif kind == "reset_error":
                    self._finish_reset(False, payload[0])
                elif kind == "dataset_log":
                    line = payload[0]
                    self._append_dataset_log(line)
                    if line.startswith("PREPARED_LEROBOT_PATH="):
                        prepared = line.split("=", 1)[1]
                        self.prepared_lerobot_path = prepared
                        self.status_var.set(f"Prepared LeRobot dataset: {prepared}")
                elif kind == "dataset_done":
                    self._finish_dataset_task(
                        payload[0],
                        int(payload[1]),
                        payload[2],
                    )
        except queue.Empty:
            pass
        self._update_dataset_action_buttons()
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

    def delete_selected_episodes(self):
        if self.recording:
            messagebox.showwarning("Cannot delete data", "Stop the current episode first")
            return
        if self._dataset_task_running():
            messagebox.showwarning(
                "Cannot delete data",
                "Wait for the current conversion or upload task to finish",
            )
            return
        selections = self.listbox.curselection()
        if not selections:
            messagebox.showinfo("Delete data", "Select one or more episodes first")
            return
        paths = [pathlib.Path(self.listbox.get(index)) for index in selections]
        preview = "\n".join(path.name for path in paths[:12])
        if len(paths) > 12:
            preview += f"\n... and {len(paths) - 12} more"
        confirmed = ask_english_yes_no(
            self.root,
            "Delete selected data",
            f"Move {len(paths)} selected episode(s) to the recoverable .trash folder?\n\n{preview}",
        )
        if not confirmed:
            return
        try:
            moved = move_episodes_to_trash(self.out_dir, paths)
        except Exception as exc:
            messagebox.showerror("Delete failed", str(exc))
            return
        self.refresh_files()
        self.episode_index = next_episode_index(self.out_dir)
        if self.session is not None:
            self.session.episode_index = self.episode_index
        trash_dir = moved[0].parent
        self.status_var.set(
            f"Moved {len(moved)} episode(s) to {trash_dir}; next episode: {self.episode_index:04d}"
        )

    def open_dataset_tools(self):
        if self.dataset_tools_window is not None:
            try:
                if self.dataset_tools_window.winfo_exists():
                    self.dataset_tools_window.deiconify()
                    self.dataset_tools_window.lift()
                    return
            except tk.TclError:
                pass
        if not self.dataset_name_var.get().strip():
            self.dataset_name_var.set(self.out_dir.name)

        dialog = tk.Toplevel(self.root)
        dialog.title("Dataset conversion and upload")
        dialog.geometry("820x680")
        dialog.minsize(680, 560)
        self.dataset_tools_window = dialog
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(2, weight=1)

        form = ttk.LabelFrame(dialog, text="NPZ to LeRobot / server upload", padding=12)
        form.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        form.columnconfigure(1, weight=1)
        self.dataset_source_var.set(str(self.out_dir.resolve()))
        rows = (
            ("NPZ source", self.dataset_source_var, "readonly"),
            ("Dataset name", self.dataset_name_var, "normal"),
            ("Server URL", self.dataset_server_var, "normal"),
            ("Server token", self.dataset_token_var, "token"),
            ("Upload workers", self.dataset_workers_var, "normal"),
        )
        for row, (label, variable, mode) in enumerate(rows):
            ttk.Label(form, text=label, width=18).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(
                form,
                textvariable=variable,
                show="*" if mode == "token" else "",
                state="readonly" if mode == "readonly" else "normal",
            )
            entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)

        ttk.Label(form, text="Install mode", width=18).grid(row=5, column=0, sticky="w", pady=4)
        ttk.Combobox(
            form,
            textvariable=self.dataset_install_mode_var,
            values=("merge", "install", "overwrite"),
            state="readonly",
        ).grid(row=5, column=1, sticky="ew", padx=(8, 0), pady=4)
        options = ttk.Frame(form)
        options.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            options,
            text="Allow incomplete gripper coverage",
            variable=self.dataset_allow_gripper_var,
        ).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(
            options,
            text="Rebuild conversion/archive cache",
            variable=self.dataset_rebuild_var,
        ).pack(side="left")

        actions = ttk.Frame(dialog)
        actions.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        prepare_button = ttk.Button(
            actions,
            text="Convert NPZ to LeRobot",
            command=lambda: self._start_dataset_task("prepare"),
        )
        prepare_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        upload_button = ttk.Button(
            actions,
            text="Convert if needed and upload",
            command=lambda: self._start_dataset_task("upload"),
        )
        upload_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.dataset_action_buttons = [prepare_button, upload_button]

        log_frame = ttk.LabelFrame(dialog, text="Progress", padding=8)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log = tk.Text(log_frame, wrap="word", font=("Monospace", 10), state="disabled")
        log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        log.configure(yscrollcommand=log_scroll.set)
        self.dataset_log_widget = log
        self._append_dataset_log(
            "Conversion uses successful ep_*.npz files and stores a reusable LeRobot cache.\n"
        )
        self._update_dataset_action_buttons()

        def close_dialog():
            self.dataset_log_widget = None
            self.dataset_action_buttons = []
            self.dataset_tools_window = None
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    def _append_dataset_log(self, text: str) -> None:
        widget = self.dataset_log_widget
        if widget is None:
            return
        try:
            if not widget.winfo_exists():
                return
            widget.configure(state="normal")
            widget.insert(tk.END, text if text.endswith("\n") else text + "\n")
            widget.see(tk.END)
            widget.configure(state="disabled")
        except tk.TclError:
            self.dataset_log_widget = None

    def _start_dataset_task(self, action: str) -> None:
        if self.recording:
            messagebox.showwarning("Dataset task", "Stop the current episode first")
            return
        if self._dataset_task_running():
            messagebox.showinfo("Dataset task", "A conversion or upload task is already running")
            return
        try:
            source = self.out_dir.resolve()
            if not any(source.glob("ep_*.npz")):
                raise ValueError(f"no ep_*.npz files found in {source}")
            fps = int(self.fps_var.get())
            workers = int(self.dataset_workers_var.get())
            command = build_dataset_tool_command(
                python_executable=sys.executable,
                script_path=pathlib.Path(__file__).with_name("upload_dataset_4090.py"),
                source_dir=source,
                dataset_name=self.dataset_name_var.get(),
                fps=fps,
                action=action,
                server=self.dataset_server_var.get(),
                workers=workers,
                install_mode=self.dataset_install_mode_var.get(),
                allow_incomplete_gripper_coverage=self.dataset_allow_gripper_var.get(),
                rebuild=self.dataset_rebuild_var.get(),
            )
            environment = os.environ.copy()
            if action == "upload":
                token = self.dataset_token_var.get().strip()
                if len(token) < 20:
                    raise ValueError("enter the server token (at least 20 characters)")
                environment["BIMANUAL_VLA_SERVER_TOKEN"] = token
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dataset task configuration", str(exc))
            return

        label = "LeRobot conversion" if action == "prepare" else "dataset upload"
        if action == "prepare":
            self.prepared_lerobot_path = None
        self.dataset_task_name = label
        self.status_var.set(f"Running {label}...")
        self._append_dataset_log(f"\n[{time.strftime('%H:%M:%S')}] Starting {label}")

        def worker():
            error: str | None = None
            return_code = -1
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=environment,
                )
                self.dataset_task_process = process
                assert process.stdout is not None
                for line in process.stdout:
                    self.messages.put(("dataset_log", line.rstrip("\n")))
                return_code = process.wait()
            except Exception as exc:
                error = str(exc)
            finally:
                self.messages.put(("dataset_done", label, return_code, error))

        self.dataset_task_thread = threading.Thread(target=worker, daemon=True)
        self.dataset_task_thread.start()
        self._update_start_button()
        self._update_dataset_action_buttons()

    def _finish_dataset_task(self, label: str, return_code: int, error: str | None) -> None:
        self.dataset_task_process = None
        self.dataset_task_thread = None
        self.dataset_task_name = None
        if error is not None:
            message = f"{label} failed: {error}"
        elif return_code != 0:
            message = f"{label} failed with exit code {return_code}; see progress log"
        elif label == "LeRobot conversion" and self.prepared_lerobot_path:
            message = f"{label} completed: {self.prepared_lerobot_path}"
        else:
            message = f"{label} completed successfully"
        self.status_var.set(message)
        self._append_dataset_log(f"[{time.strftime('%H:%M:%S')}] {message}")
        if error is not None or return_code != 0:
            messagebox.showerror("Dataset task failed", message)
        else:
            messagebox.showinfo("Dataset task complete", message)
        self._update_start_button()
        self._update_dataset_action_buttons()

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
            self.latest_pose_errors = np.zeros(self.contract.joint_dim, dtype=np.float32)
        self.preview_photos.clear()
        for key, label in self.preview_labels.items():
            label.configure(image="", text="Waiting for camera...")
        cv2.destroyAllWindows()

    def disconnect(self):
        if self.recording:
            messagebox.showwarning("Cannot disconnect", "Stop the current episode first")
            return
        if self.reset_thread is not None:
            messagebox.showwarning("Cannot disconnect", "Wait for the reset to complete")
            return
        self._cleanup_devices()
        self.connect_button.configure(text="Connect devices")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self._set_connection_config_enabled(True)
        self._configure_mode_ui()
        self.status_var.set("Disconnected")
        self.pose_check_var.set("Waiting for robot feedback")

    def close(self):
        if self._dataset_task_running():
            if not messagebox.askyesno(
                "Exit",
                "A dataset conversion/upload task is still running. Stop it and exit?",
            ):
                return
            process = self.dataset_task_process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            if self.dataset_task_thread is not None:
                self.dataset_task_thread.join(timeout=3.0)
        if self.recording:
            if not messagebox.askyesno("Exit", "The current episode has not been saved. Exit anyway?"):
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
