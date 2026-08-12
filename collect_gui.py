"""GUI for collecting single-arm or bimanual output-arm episodes.

The GUI records the same explicit NPZ contracts as :mod:`collect_output_arm`:
single-arm 7D/10D with two cameras, or bimanual 14D/20D in fixed left+right
order with three cameras.  It reads measured output-arm feedback only; for
same-step master-arm joint commands use :mod:`teleop_single` or :mod:`teleop`.

The live view stays in the Tk window and shows every active RGB camera plus one
compact seven-value state row per arm. Missing or stale hardware feedback is
still rejected before and during recording.
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
from dataclasses import replace
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

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
    CAMERA_SOURCE_HW,
    next_episode_index,
)
from piper_data_contract import BIMANUAL, DELIVERY_SCHEMA, JOINT_SCHEMA, SINGLE_ARM, EpisodeContract
from piper_action_conventions import rotation6d_to_matrix
from upload_dataset_4090 import DEFAULT_SERVER, safe_dataset_name


# ``CameraCapture`` keeps the model/data contract at 256x256 by letterboxing
# the requested camera stream into a square. Recover each source stream's
# aspect ratio before resizing so the live view is not stretched.
CAMERA_PREVIEW_ASPECT = CAMERA_SOURCE_HW[1] / CAMERA_SOURCE_HW[0]
PREVIEW_SLOTS = (
    ("high", "Overhead camera"),
    ("primary_wrist", "Left wrist camera"),
    ("right_wrist", "Right wrist camera"),
)
PREVIEW_DEFAULT_HW = (320, 560)
ADD_DATASET_OPTION = "Add new dataset..."
EPISODE_FILE_RE = re.compile(r"ep_\d+\.npz")
CAN_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
CAN_BITRATE = 1_000_000
CAN_ACTIVATE_SCRIPT = pathlib.Path(
    "/home/user/dual_ARM_project/piper_sdk/piper_sdk/can_activate.sh"
)
BIMANUAL_CAN_USB_PORTS = ("3-1.4:1.0", "3-1.3:1.0")


def validate_can_name(can_name: str) -> str:
    """Validate a SocketCAN interface name before passing it to system tools."""
    name = can_name.strip()
    if not CAN_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid CAN interface name: {can_name!r}")
    return name


def parse_can_bus_info(output: str) -> str:
    """Extract one USB bus-info value from ``ethtool -i`` output."""
    match = re.search(r"^bus-info:\s*(\S+)\s*$", output, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("ethtool output does not contain bus-info")
    return match.group(1)


def parse_can_link_status(output: str, can_name: str) -> dict[str, object]:
    """Parse operational state and bitrate from ``ip -details link`` output."""
    name = validate_can_name(can_name)
    header = re.search(
        rf"^\d+:\s+{re.escape(name)}:\s+<([^>]*)>",
        output,
        flags=re.MULTILINE,
    )
    bitrate = re.search(r"\bbitrate\s+(\d+)\b", output)
    flags = set() if header is None else set(header.group(1).split(","))
    return {
        "name": name,
        "up": "UP" in flags,
        "bitrate": None if bitrate is None else int(bitrate.group(1)),
    }


def build_can_activation_command(
    can_name: str,
    bus_info: str,
    *,
    helper_path: str | pathlib.Path = CAN_ACTIVATE_SCRIPT,
) -> list[str]:
    """Build a password-free command for activating one physical CAN adapter."""
    name = validate_can_name(can_name)
    address = bus_info.strip()
    if not address or any(character.isspace() for character in address):
        raise ValueError(f"invalid CAN USB bus-info: {bus_info!r}")
    return [
        "sudo",
        "-S",
        "-p",
        "",
        "bash",
        str(pathlib.Path(helper_path)),
        name,
        str(CAN_BITRATE),
        address,
    ]


def activate_can_interfaces(
    can_names: list[str] | tuple[str, ...],
    password: str,
    *,
    expected_bus_info: dict[str, str] | None = None,
    helper_path: str | pathlib.Path = CAN_ACTIVATE_SCRIPT,
    runner=subprocess.run,
) -> dict[str, dict[str, object]]:
    """Activate and verify one or two SocketCAN interfaces at Piper bitrate."""
    names = tuple(validate_can_name(value) for value in can_names)
    if not names:
        raise ValueError("at least one CAN interface is required")
    if len(set(names)) != len(names):
        raise ValueError("CAN interface names must be distinct")
    helper = pathlib.Path(helper_path)
    if not helper.is_file():
        raise FileNotFoundError(f"CAN activation helper not found: {helper}")

    list_result = runner(
        ["ip", "-brief", "link", "show", "type", "can"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if list_result.returncode != 0:
        detail = (list_result.stderr or list_result.stdout).strip()
        raise RuntimeError(f"cannot list CAN interfaces: {detail or 'ip link failed'}")
    current_names = tuple(
        validate_can_name(line.split()[0])
        for line in list_result.stdout.splitlines()
        if line.strip()
    )
    if not current_names:
        raise RuntimeError("no SocketCAN interfaces were detected")

    current_bus_info: dict[str, str] = {}
    for name in current_names:
        result = runner(
            ["ethtool", "-i", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"cannot inspect {name}: {detail or 'ethtool failed'}")
        current_bus_info[name] = parse_can_bus_info(result.stdout)

    desired_bus_info = (
        {name: current_bus_info[name] for name in names}
        if expected_bus_info is None
        else {name: expected_bus_info[name].strip() for name in names}
    )
    if any(not address for address in desired_bus_info.values()):
        raise ValueError("expected CAN USB bus-info must not be empty")
    if len(set(desired_bus_info.values())) != len(desired_bus_info):
        raise ValueError("expected CAN USB bus-info values must be distinct")

    current_name_by_bus = {address: name for name, address in current_bus_info.items()}
    missing = [
        f"{name} expected at {address}"
        for name, address in desired_bus_info.items()
        if address not in current_name_by_bus
    ]
    if missing:
        raise RuntimeError("missing CAN adapter: " + "; ".join(missing))

    desired_buses = set(desired_bus_info.values())
    for desired_name in names:
        occupying_bus = current_bus_info.get(desired_name)
        if occupying_bus is not None and occupying_bus not in desired_buses:
            raise RuntimeError(
                f"cannot use {desired_name}: it is occupied by an unexpected adapter at {occupying_bus}"
            )

    # Kernel CAN names are enumeration-dependent. Move every misnamed adapter
    # to a temporary name first so a complete can0/can1 swap has no collision.
    temporary_by_bus: dict[str, str] = {}
    used_names = set(current_names) | set(names)
    for index, (desired_name, address) in enumerate(desired_bus_info.items()):
        current_name = current_name_by_bus[address]
        if current_name == desired_name:
            continue
        temporary_name = f"pcan_tmp{index}"
        suffix = 0
        while temporary_name in used_names:
            suffix += 1
            temporary_name = f"pcan_t{index}_{suffix}"
        used_names.add(temporary_name)
        for command in (
            ["sudo", "-S", "-p", "", "ip", "link", "set", current_name, "down"],
            [
                "sudo",
                "-S",
                "-p",
                "",
                "ip",
                "link",
                "set",
                current_name,
                "name",
                temporary_name,
            ],
        ):
            result = runner(
                command,
                input=password + "\n",
                check=False,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            if result.returncode != 0:
                detail = "\n".join(
                    part.strip() for part in (result.stdout, result.stderr) if part.strip()
                )
                raise RuntimeError(
                    f"failed to normalize {current_name} for {desired_name}: "
                    f"{detail or 'ip link failed'}"
                )
        temporary_by_bus[address] = temporary_name

    for name in names:
        command = build_can_activation_command(
            name,
            desired_bus_info[name],
            helper_path=helper,
        )
        result = runner(
            command,
            input=password + "\n",
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        if result.returncode != 0:
            detail = "\n".join(
                part.strip() for part in (result.stdout, result.stderr) if part.strip()
            )
            raise RuntimeError(f"failed to activate {name}: {detail or 'activation helper failed'}")

    statuses: dict[str, dict[str, object]] = {}
    for name in names:
        result = runner(
            ["ip", "-details", "link", "show", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"cannot verify {name}: {detail or 'ip link failed'}")
        status = parse_can_link_status(result.stdout, name)
        inspect_result = runner(
            ["ethtool", "-i", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if inspect_result.returncode != 0:
            raise RuntimeError(f"cannot verify USB mapping for {name}")
        status["bus_info"] = parse_can_bus_info(inspect_result.stdout)
        if not status["up"] or status["bitrate"] != CAN_BITRATE:
            raise RuntimeError(
                f"{name} activation verification failed: "
                f"up={status['up']} bitrate={status['bitrate']}"
            )
        if status["bus_info"] != desired_bus_info[name]:
            raise RuntimeError(
                f"{name} USB mapping verification failed: "
                f"expected {desired_bus_info[name]}, got {status['bus_info']}"
            )
        statuses[name] = status
    return statuses


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


def summarize_dataset_directory(directory: str | pathlib.Path) -> dict[str, int]:
    """Return lightweight episode/frame counts for the active dataset folder."""
    root = pathlib.Path(directory).expanduser()
    summary = {"episodes": 0, "frames": 0, "success": 0, "failure": 0, "invalid": 0}
    if not root.is_dir():
        return summary
    for path in sorted(root.glob("ep_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                state = np.asarray(data["state"])
                if state.ndim != 2:
                    raise ValueError("state is not a matrix")
                summary["episodes"] += 1
                summary["frames"] += max(0, int(state.shape[0]) - 1)
                success = bool(np.asarray(data["success"]).reshape(()))
                summary["success" if success else "failure"] += 1
        except (OSError, KeyError, ValueError, TypeError):
            summary["invalid"] += 1
    return summary


def discover_dataset_names(root: str | pathlib.Path) -> tuple[str, ...]:
    """List existing raw dataset directories available below a dataset root."""
    path = pathlib.Path(root).expanduser()
    names: list[str] = []
    if path.is_dir() and any(path.glob("ep_*.npz")):
        names.append(path.name)
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                try:
                    names.append(safe_dataset_name(child.name))
                except ValueError:
                    continue
    return tuple(dict.fromkeys(names))


def episode_list_values(
    path: str | pathlib.Path,
    dataset_name: str,
) -> tuple[str, str]:
    """Return concise table values while keeping the episode path out of the UI."""
    episode_path = pathlib.Path(path)
    match = re.search(r"(\d+)$", episode_path.stem)
    episode = f"Episode {int(match.group(1)):04d}" if match else episode_path.stem
    return dataset_name, episode


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


def prepare_preview_frame(
    frame: np.ndarray,
    *,
    target_aspect: float = CAMERA_PREVIEW_ASPECT,
) -> np.ndarray | None:
    """Convert a camera frame to HWC RGB and remove capture letterboxing.

    The collector stores every observation as a square image for the model
    contract. ``CameraCapture`` preserves the camera content inside that
    square and pads the remaining area with black pixels. Cropping the square
    back to the configured source aspect ratio gives the operator an
    undistorted live view without changing the recorded data.
    """
    rgb = np.asarray(frame)
    if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4) and rgb.shape[-1] not in (1, 3, 4):
        rgb = rgb.transpose(1, 2, 0)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    if not np.isfinite(target_aspect) or target_aspect <= 0:
        raise ValueError("target_aspect must be positive and finite")

    height, width = rgb.shape[:2]
    current_aspect = width / height
    if current_aspect > target_aspect:
        # Wider than the source stream: trim the side padding.
        cropped_width = max(1, round(height * target_aspect))
        left = (width - cropped_width) // 2
        rgb = rgb[:, left : left + cropped_width]
    elif current_aspect < target_aspect:
        # Taller than the source stream: trim the top/bottom padding.
        cropped_height = max(1, round(width / target_aspect))
        top = (height - cropped_height) // 2
        rgb = rgb[top : top + cropped_height, :]
    return np.ascontiguousarray(rgb.astype(np.uint8, copy=False))


def letterbox_preview_frame(
    frame: np.ndarray,
    *,
    target_hw: tuple[int, int] = PREVIEW_DEFAULT_HW,
    source_aspect: float = CAMERA_PREVIEW_ASPECT,
) -> np.ndarray | None:
    """Render a camera frame into a fixed rectangle without distortion."""
    target_h, target_w = (int(value) for value in target_hw)
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_hw values must be positive")
    rgb = prepare_preview_frame(frame, target_aspect=source_aspect)
    if rgb is None:
        return None
    height, width = rgb.shape[:2]
    scale = min(target_w / width, target_h / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    tile = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - resized_height) // 2
    x0 = (target_w - resized_width) // 2
    tile[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return tile


def format_arm_state_rows(
    state: np.ndarray | None,
    *,
    schema: str,
    arm_mode: str,
    arm_side: str,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Return one seven-value display row per arm from the collected state."""
    if state is None:
        return ()
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    sides = ("left", "right") if arm_mode == BIMANUAL else (arm_side,)
    block_size = 10 if schema == DELIVERY_SCHEMA else 7
    if values.shape != (block_size * len(sides),) or not np.isfinite(values).all():
        return ()
    rows = []
    for index, side in enumerate(sides):
        block = values[index * block_size : (index + 1) * block_size]
        if schema == DELIVERY_SCHEMA:
            rotation = rotation6d_to_matrix(block[3:9])
            rpy = Rotation.from_matrix(rotation).as_euler("xyz")
            display = tuple(float(value) for value in (*block[:3], *rpy, block[9]))
        else:
            display = tuple(float(value) for value in block)
        rows.append((side, display))
    return tuple(rows)


class CollectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Piper - pi0.5 Data Collection")
        self.root.geometry("1800x1200")
        self.root.minsize(1350, 950)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.session: CollectionSession | None = None
        self.piper = None
        self.cameras = None
        self.capture_thread: threading.Thread | None = None
        self.capture_stop: threading.Event | None = None
        self.can_activation_thread: threading.Thread | None = None
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
        self.messages: queue.Queue = queue.Queue()
        self.recording = False
        self._space_pressed = False
        self.episode_index = 0
        self.capture_fps = 20
        self.camera_fps = DEFAULT_CAMERA_FPS

        # PhotoImage references must be retained for Tk to keep them alive.
        self.preview_photos: dict[str, object] = {}
        self.preview_labels: dict[str, tk.Canvas] = {}
        self.preview_cards: dict[str, tk.Frame] = {}
        self.preview_title_labels: dict[str, tk.Label] = {}
        self.preview_image_items: dict[str, int] = {}
        self.preview_text_items: dict[str, int] = {}
        self.episode_paths: dict[str, pathlib.Path] = {}
        self.preview_key_to_slot: dict[str, str] = {}
        self.arm_mode_var = tk.StringVar(value=BIMANUAL)
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
        self.out_var = tk.StringVar(value="episodes_piper_v21")
        self.task_var = tk.StringVar(value="pick_cube")
        self.instruction_var = tk.StringVar(value="pick up the cube")
        self.dataset_source_var = tk.StringVar(
            value=str(pathlib.Path(self.out_var.get()).expanduser().resolve())
        )
        self.dataset_name_var = tk.StringVar(value="episodes_piper_v21")
        self._dataset_before_add = self.dataset_name_var.get()
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
        self.dataset_stats_var = tk.StringVar(value="Dataset: no episodes")
        self.device_summary_var = tk.StringVar(value="")
        self.arm_state_header_var = tk.StringVar(value="Arm data")
        self.arm_state_vars = {
            "left": tk.StringVar(value="Waiting for data"),
            "right": tk.StringVar(value="Waiting for data"),
        }
        self.swap_wrist_cameras_var = tk.BooleanVar(value=False)
        self.device_settings_window: tk.Toplevel | None = None
        self._build_ui()
        self._configure_mode_ui()
        self.refresh_files()
        self._install_space_shortcut()
        self.root.after(100, self._poll_messages)

    def _build_ui(self):
        colors = {
            "window": "#eef1f5",
            "surface": "#ffffff",
            "surface_alt": "#f7f8fa",
            "border": "#d9dde5",
            "text": "#202124",
            "secondary": "#68707d",
            "accent": "#1a73e8",
            "camera": "#16181c",
        }
        self.root.configure(bg=colors["window"])
        tkfont.nametofont("TkDefaultFont").configure(family="DejaVu Sans", size=10)
        tkfont.nametofont("TkTextFont").configure(family="DejaVu Sans", size=10)
        tkfont.nametofont("TkHeadingFont").configure(family="DejaVu Sans", size=12, weight="bold")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=colors["window"], foreground=colors["text"])
        style.configure("TFrame", background=colors["window"])
        style.configure("TLabel", background=colors["window"], foreground=colors["text"])
        style.configure("TButton", font=("DejaVu Sans", 10), padding=(13, 8), relief="flat")
        style.configure("Accent.TButton", foreground="#ffffff", background=colors["accent"])
        style.map(
            "Accent.TButton",
            background=[("active", "#1769d2"), ("disabled", "#a8c7ea")],
            foreground=[("disabled", "#f3f3f3")],
        )
        style.configure(
            "Card.TLabelframe",
            background=colors["surface"],
            bordercolor=colors["border"],
            relief="flat",
            borderwidth=1,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=colors["window"],
            foreground=colors["text"],
            font=("DejaVu Sans", 11, "bold"),
        )
        style.configure("Card.TFrame", background=colors["surface"])
        style.configure("Card.TLabel", background=colors["surface"], foreground=colors["text"])
        style.configure("Secondary.Card.TLabel", background=colors["surface"], foreground=colors["secondary"])
        style.configure("TCombobox", padding=(6, 5))
        style.configure("TEntry", padding=(6, 5))
        style.configure(
            "Episodes.Treeview",
            background=colors["surface"],
            fieldbackground=colors["surface"],
            foreground=colors["text"],
            rowheight=30,
            borderwidth=0,
            relief="flat",
            font=("DejaVu Sans", 9),
        )
        style.configure(
            "Episodes.Treeview.Heading",
            background=colors["surface_alt"],
            foreground=colors["secondary"],
            relief="flat",
            padding=(8, 7),
            font=("DejaVu Sans", 9, "bold"),
        )
        style.map(
            "Episodes.Treeview",
            background=[("selected", colors["accent"])],
            foreground=[("selected", "#ffffff")],
        )

        def make_card(parent: tk.Misc, title: str) -> tuple[tk.Frame, ttk.Frame]:
            card = tk.Frame(
                parent,
                bg=colors["surface"],
                highlightthickness=1,
                highlightbackground=colors["border"],
                bd=0,
            )
            tk.Label(
                card,
                text=title,
                bg=colors["surface"],
                fg=colors["text"],
                font=("DejaVu Sans", 11, "bold"),
                anchor="w",
            ).pack(fill="x", padx=14, pady=(12, 7))
            body = ttk.Frame(card, style="Card.TFrame", padding=(14, 0, 14, 12))
            body.pack(fill="both", expand=True)
            return card, body

        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=0, minsize=570)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        left = ttk.Frame(main, padding=(0, 0, 8, 0))
        right = ttk.Frame(main, padding=(8, 0, 0, 0))
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        task_card, task = make_card(left, "Task and dataset")
        task_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        task.columnconfigure(1, weight=1)
        selectors = [
            ("Arm mode", self.arm_mode_var, (("Single arm", SINGLE_ARM), ("Bimanual", BIMANUAL))),
            (
                "Collection format",
                self.schema_var,
                (("Joint", JOINT_SCHEMA), ("End-effector", DELIVERY_SCHEMA)),
            ),
            ("Single-arm side", self.arm_side_var, (("right", "right"), ("left", "left"))),
        ]
        self.mode_selectors = []
        self.mode_display_vars = []
        for row, (label, variable, values) in enumerate(selectors):
            ttk.Label(task, text=label, width=18, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=4)
            display_to_value = dict(values)
            value_to_display = {value: display for display, value in values}
            display_var = tk.StringVar(value=value_to_display[variable.get()])
            selector = ttk.Combobox(
                task,
                textvariable=display_var,
                values=tuple(display_to_value),
                state="readonly",
            )
            selector.grid(row=row, column=1, sticky="ew", pady=3, padx=(8, 0))

            def changed(_event=None, *, source=display_var, target=variable, mapping=display_to_value):
                target.set(mapping[source.get()])
                self._configure_mode_ui()

            selector.bind("<<ComboboxSelected>>", changed)
            self.mode_selectors.append(selector)
            self.mode_display_vars.append(display_var)

        ttk.Label(task, text="Dataset", width=18).grid(row=3, column=0, sticky="w", pady=3)
        self.dataset_name_entry = ttk.Combobox(
            task,
            textvariable=self.dataset_name_var,
            values=(),
            state="readonly",
        )
        self.dataset_name_entry.grid(row=3, column=1, sticky="ew", pady=3, padx=(8, 0))
        self.dataset_name_entry.bind("<<ComboboxSelected>>", self._dataset_selection_changed)
        ttk.Label(task, text="Task name", width=18, style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(task, textvariable=self.task_var).grid(row=4, column=1, sticky="ew", pady=3, padx=(8, 0))
        ttk.Label(task, text="Instruction", width=18, style="Card.TLabel").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(task, textvariable=self.instruction_var).grid(row=5, column=1, sticky="ew", pady=3, padx=(8, 0))
        device_bar = ttk.Frame(task, style="Card.TFrame")
        device_bar.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        device_bar.columnconfigure(0, weight=1)
        ttk.Label(
            device_bar,
            textvariable=self.device_summary_var,
            style="Secondary.Card.TLabel",
            justify="left",
            wraplength=360,
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.device_settings_button = ttk.Button(
            device_bar,
            text="Device settings...",
            command=self.open_device_settings,
        )
        self.device_settings_button.grid(row=0, column=1, sticky="e")
        self.connection_entries = []
        self.device_rows = {"single": [], "bimanual": []}
        self.device_entries = []

        controls_card, controls = make_card(left, "Collection controls")
        controls_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for column in range(3):
            controls.columnconfigure(column, weight=1)
        self.activate_can_button = ttk.Button(controls, text="Activate CAN", command=self.activate_can, style="Accent.TButton")
        self.activate_can_button.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        self.connect_button = ttk.Button(controls, text="Connect devices", command=self.toggle_connection, style="Accent.TButton")
        self.connect_button.grid(row=0, column=1, padx=3, pady=3, sticky="ew")
        self.start_button = ttk.Button(controls, text="Start episode", command=self.start_episode, state="disabled")
        self.start_button.grid(row=0, column=2, padx=3, pady=3, sticky="ew")
        self.stop_button = ttk.Button(controls, text="Stop episode", command=self.stop_episode, state="disabled")
        self.stop_button.grid(row=1, column=0, padx=3, pady=3, sticky="ew")
        self.swap_camera_button = ttk.Checkbutton(
            controls,
            text="Swap wrist cameras",
            variable=self.swap_wrist_cameras_var,
            command=self.swap_camera_roles,
        )
        self.swap_camera_button.grid(row=1, column=1, padx=3, pady=3, sticky="ew")
        ttk.Button(controls, text="Refresh files", command=self.refresh_files).grid(
            row=1, column=2, padx=3, pady=3, sticky="ew"
        )
        ttk.Button(controls, text="Replay episode", command=self.replay_selected).grid(
            row=2, column=0, padx=3, pady=3, sticky="ew"
        )
        self.edit_dataset_button = ttk.Button(controls, text="Add dataset...", command=self.edit_dataset_name)
        self.edit_dataset_button.grid(row=2, column=1, padx=3, pady=3, sticky="ew")
        self.exit_button = ttk.Button(controls, text="Exit", command=self.close)
        self.exit_button.grid(row=2, column=2, padx=3, pady=3, sticky="ew")
        ttk.Label(controls, text="Space starts or stops an episode", foreground=colors["secondary"]).grid(
            row=3, column=0, columnspan=3, sticky="w", padx=3, pady=(5, 0)
        )

        arm_data_card, arm_data = make_card(left, "Robot and collection status")
        arm_data_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        arm_data.columnconfigure(1, weight=1)
        status_line = ttk.Frame(arm_data, style="Card.TFrame")
        status_line.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 7))
        status_line.columnconfigure(0, weight=1)
        ttk.Label(
            status_line,
            textvariable=self.status_var,
            style="Card.TLabel",
            font=("DejaVu Sans", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            status_line,
            textvariable=self.progress_var,
            style="Secondary.Card.TLabel",
        ).grid(row=0, column=1, sticky="e")
        ttk.Separator(arm_data).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.arm_state_dimension_label = ttk.Label(
            arm_data,
            textvariable=self.arm_state_header_var,
            style="Secondary.Card.TLabel",
            font=("DejaVu Sans", 9),
        )
        self.arm_state_dimension_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.arm_state_labels = {}
        for row, side in enumerate(("left", "right"), start=3):
            side_label = ttk.Label(arm_data, text=side.capitalize(), width=7, style="Card.TLabel", font=("DejaVu Sans", 10, "bold"))
            side_label.grid(row=row, column=0, sticky="nw", pady=4)
            value_label = ttk.Label(
                arm_data,
                textvariable=self.arm_state_vars[side],
                justify="left",
                wraplength=410,
            )
            value_label.grid(row=row, column=1, sticky="ew", pady=4)
            self.arm_state_labels[side] = (side_label, value_label)

        files_card, files = make_card(left, "Saved episodes")
        files_card.grid(row=3, column=0, sticky="nsew")
        files.columnconfigure(0, weight=1)
        files.rowconfigure(0, weight=1)
        self.listbox = ttk.Treeview(
            files,
            columns=("dataset", "episode"),
            show="headings",
            selectmode="extended",
            style="Episodes.Treeview",
        )
        headings = {"dataset": "Dataset", "episode": "Episode"}
        widths = {"dataset": 300, "episode": 170}
        for column, heading in headings.items():
            self.listbox.heading(column, text=heading)
            self.listbox.column(column, width=widths[column], minwidth=90, anchor="w", stretch=True)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(files, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.bind("<Double-1>", lambda _event: self.replay_selected())
        ttk.Label(
            files,
            textvariable=self.dataset_stats_var,
            foreground="#52606d",
            justify="left",
            wraplength=480,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        file_actions = ttk.Frame(files)
        file_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        file_actions.columnconfigure(0, weight=1)
        file_actions.columnconfigure(1, weight=1)
        self.delete_episode_button = ttk.Button(
            file_actions,
            text="Delete selected",
            command=self.delete_selected_episodes,
        )
        self.delete_episode_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.dataset_tools_button = ttk.Button(
            file_actions,
            text="Convert / upload",
            command=self.open_dataset_tools,
        )
        self.dataset_tools_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        preview_card, preview = make_card(right, "Live cameras")
        preview_card.grid(row=0, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1, uniform="wrist")
        preview.columnconfigure(1, weight=1, uniform="wrist")
        # A 2:1 row ratio gives the full-width overhead view and the two
        # half-width wrist views approximately the same 16:9 content area.
        preview.rowconfigure(0, weight=2, minsize=360)
        preview.rowconfigure(1, weight=1, minsize=240)
        placements = {
            "high": (0, 0, 2),
            "primary_wrist": (1, 0, 1),
            "right_wrist": (1, 1, 1),
        }
        for slot, title in PREVIEW_SLOTS:
            row, column, span = placements[slot]
            card = tk.Frame(preview, bg=colors["camera"], bd=0, highlightthickness=1, highlightbackground="#303034")
            card.grid(row=row, column=column, columnspan=span, sticky="nsew", padx=5, pady=5)
            title_label = tk.Label(
                card,
                text=title,
                bg=colors["camera"],
                fg="#f5f7fa",
                font=("DejaVu Sans", 10, "bold"),
            )
            title_label.pack(fill="x", padx=7, pady=(6, 4))
            label = tk.Canvas(
                card,
                bg="#0b0b0d",
                highlightthickness=0,
                bd=0,
                width=560,
                height=315,
            )
            label.pack(fill="both", expand=True, padx=6, pady=(0, 6))
            image_item = label.create_image(0, 0, anchor="nw")
            text_item = label.create_text(
                280,
                158,
                text="Waiting for camera...",
                fill="#8e8e93",
                font=("DejaVu Sans", 10),
            )
            label.bind(
                "<Configure>",
                lambda event, item=text_item: event.widget.coords(
                    item,
                    event.width // 2,
                    event.height // 2,
                ),
            )
            self.preview_cards[slot] = card
            self.preview_title_labels[slot] = title_label
            self.preview_labels[slot] = label
            self.preview_image_items[slot] = image_item
            self.preview_text_items[slot] = text_item


    @property
    def dataset_root_dir(self) -> pathlib.Path:
        path = pathlib.Path(self.out_var.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    @property
    def out_dir(self) -> pathlib.Path:
        """Return the active named dataset directory, creating it if needed.

        Keep the historical default layout compatible: when the root directory
        already has the same name as the active dataset, use that directory
        directly. Renaming the dataset creates a named child directory below
        the configured root.
        """
        root = self.dataset_root_dir
        name = safe_dataset_name(self.dataset_name_var.get().strip())
        path = root if root.name == name else root / name
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

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

    def _camera_role_title(self, key: str) -> str:
        if key == "cam_high":
            return "Overhead camera"
        if key == "cam_left_wrist":
            return "Left wrist camera"
        if key == "cam_right_wrist":
            return "Right wrist camera"
        return f"{self.arm_side.capitalize()} wrist camera"

    def _update_device_summary(self) -> None:
        if self.arm_mode == BIMANUAL:
            can_text = f"CAN {self.left_can_var.get()} / {self.right_can_var.get()}"
            camera_text = "3 cameras"
        else:
            can_text = f"CAN {self.can_var.get()}"
            camera_text = "2 cameras"
        self.device_summary_var.set(
            f"{can_text}   |   {camera_text}\n"
            f"{self.fps_var.get()} Hz collection   |   {self.camera_fps_var.get()} Hz camera"
        )

    def _set_preview_message(self, slot: str, message: str) -> None:
        canvas = self.preview_labels[slot]
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.itemconfigure(self.preview_image_items[slot], image="")
        canvas.coords(self.preview_text_items[slot], width // 2, height // 2)
        canvas.itemconfigure(self.preview_text_items[slot], text=message, state="normal")

    def _configure_mode_ui(self):
        if not hasattr(self, "preview_labels"):
            return
        bimanual = self.arm_mode == BIMANUAL
        if bimanual:
            self.preview_key_to_slot = {
                "cam_high": "high",
                "cam_left_wrist": "primary_wrist",
                "cam_right_wrist": "right_wrist",
            }
        else:
            self.preview_key_to_slot = {
                "cam_high": "high",
                self.contract.camera_keys[1]: "primary_wrist",
            }
        for camera_key, slot in self.preview_key_to_slot.items():
            self.preview_title_labels[slot].configure(text=self._camera_role_title(camera_key))
        active_slots = set(self.preview_key_to_slot.values())
        for slot in self.preview_labels:
            self._set_preview_message(
                slot,
                "Waiting for camera..." if slot in active_slots else "Inactive",
            )
        self.preview_photos.clear()
        self.arm_state_header_var.set(
            "Arm data: x y z rx ry rz gripper"
            if self.schema == DELIVERY_SCHEMA
            else "Arm data: j1 j2 j3 j4 j5 j6 gripper"
        )
        for side, widgets in self.arm_state_labels.items():
            visible = bimanual or side == self.arm_side
            for widget in widgets:
                widget.grid() if visible else widget.grid_remove()
            self.arm_state_vars[side].set("Waiting for data")
        if len(self.mode_selectors) >= 3:
            side_state = "disabled" if bimanual or self.session is not None else "readonly"
            self.mode_selectors[2].configure(state=side_state)
        if self.session is None:
            self.latest_qpos = None
            self.latest_state = None
        self.swap_camera_button.configure(
            state="normal" if bimanual and not self.recording else "disabled"
        )
        self._update_device_summary()
        self._refresh_dataset_choices()
        if hasattr(self, "listbox"):
            self.refresh_files()

    def _set_connection_config_enabled(self, enabled: bool) -> None:
        selector_state = "readonly" if enabled else "disabled"
        for selector in self.mode_selectors:
            selector.configure(state=selector_state)
        self.device_settings_button.configure(state="normal" if enabled else "disabled")
        dataset_state = "disabled" if self.recording else "readonly"
        self.dataset_name_entry.configure(state=dataset_state)
        self.edit_dataset_button.configure(state=dataset_state)
        self.swap_camera_button.configure(
            state="normal"
            if self.arm_mode == BIMANUAL and not self.recording
            else "disabled"
        )

    def _refresh_dataset_choices(self) -> None:
        if hasattr(self, "dataset_name_entry"):
            names = discover_dataset_names(self.dataset_root_dir)
            self.dataset_name_entry.configure(values=(*names, ADD_DATASET_OPTION))

    def _apply_dataset_name(self, value: str, *, parent: tk.Misc | None = None) -> bool:
        if self.recording or (self.session is not None and self.session.state is SessionState.REVIEW):
            messagebox.showwarning(
                "Dataset locked",
                "Save or discard the current episode before changing datasets.",
                parent=parent,
            )
            return False
        try:
            name = safe_dataset_name(value.strip())
            root = self.dataset_root_dir
            target = root if root.name == name else root / name
            target.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Invalid dataset name", str(exc), parent=parent)
            return False
        self.dataset_name_var.set(name)
        self._dataset_before_add = name
        self.dataset_source_var.set(str(target.resolve()))
        self.episode_index = next_episode_index(target)
        if self.session is not None:
            self.session.config = replace(self.session.config, output_dir=target)
            self.session.episode_index = self.episode_index
        self._refresh_dataset_choices()
        self.refresh_files()
        self.status_var.set(f"Ready: dataset {name}, next episode {self.episode_index:04d}")
        return True

    def _dataset_selection_changed(self, _event=None) -> None:
        value = self.dataset_name_var.get()
        if value == ADD_DATASET_OPTION:
            current = getattr(self, "_dataset_before_add", "")
            self.dataset_name_var.set(current)
            self.edit_dataset_name()
            return
        self._dataset_before_add = value
        self._apply_dataset_name(value, parent=self.root)

    def open_device_settings(self) -> None:
        if self.device_settings_window is not None and self.device_settings_window.winfo_exists():
            self.device_settings_window.lift()
            self.device_settings_window.focus_force()
            return
        if self.session is not None:
            messagebox.showinfo("Device settings", "Disconnect devices before changing hardware settings.")
            return
        dialog = tk.Toplevel(self.root)
        self.device_settings_window = dialog
        dialog.title("Device settings")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)

        rows = []
        if self.arm_mode == BIMANUAL:
            rows.extend(
                (
                    ("Left-arm CAN", self.left_can_var),
                    ("Right-arm CAN", self.right_can_var),
                    ("Left wrist camera", self.left_wrist_var),
                    ("Right wrist camera", self.right_wrist_var),
                )
            )
        else:
            rows.extend(
                (
                    ("Robot CAN", self.can_var),
                    ("Wrist camera", self.wrist_var),
                )
            )
        rows.extend(
            (
                ("Overhead camera", self.high_var),
                ("Collection rate (Hz)", self.fps_var),
                ("Camera rate (Hz)", self.camera_fps_var),
                ("Dataset root", self.out_var),
            )
        )
        pending = {label: tk.StringVar(value=variable.get()) for label, variable in rows}
        for row, (label, _variable) in enumerate(rows):
            ttk.Label(body, text=label, width=20).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(body, textvariable=pending[label], width=58).grid(
                row=row,
                column=1,
                sticky="ew",
                padx=(8, 0),
                pady=4,
            )

        buttons = ttk.Frame(body)
        buttons.grid(row=len(rows), column=0, columnspan=2, sticky="ew", pady=(12, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        def close_dialog() -> None:
            self.device_settings_window = None
            dialog.destroy()

        def apply_settings() -> None:
            try:
                fps = int(pending["Collection rate (Hz)"].get())
                camera_fps = int(pending["Camera rate (Hz)"].get())
                if fps <= 0 or camera_fps <= 0 or fps > camera_fps:
                    raise ValueError("rates must be positive and collection rate cannot exceed camera rate")
                can_values = (
                    (pending["Left-arm CAN"].get(), pending["Right-arm CAN"].get())
                    if self.arm_mode == BIMANUAL
                    else (pending["Robot CAN"].get(),)
                )
                names = tuple(validate_can_name(value) for value in can_values)
                if len(set(names)) != len(names):
                    raise ValueError("left and right CAN names must be distinct")
                root = pathlib.Path(pending["Dataset root"].get()).expanduser()
                root.mkdir(parents=True, exist_ok=True)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Invalid device settings", str(exc), parent=dialog)
                return
            for label, variable in rows:
                variable.set(pending[label].get().strip())
            self.capture_fps = fps
            self.camera_fps = camera_fps
            self._update_device_summary()
            self._refresh_dataset_choices()
            self._apply_dataset_name(self.dataset_name_var.get(), parent=dialog)
            self.status_var.set("Device settings updated")
            close_dialog()

        ttk.Button(buttons, text="Cancel", command=close_dialog).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(buttons, text="Apply", command=apply_settings).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        dialog.bind("<Escape>", lambda _event: close_dialog())
        dialog.bind("<Return>", lambda _event: apply_settings())
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    def edit_dataset_name(self) -> None:
        """Choose an existing dataset or create a new named directory."""
        if self.recording or (self.session is not None and self.session.state is SessionState.REVIEW):
            messagebox.showwarning("Dataset locked", "Finish the current episode first.")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Add new dataset")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=f"Create a folder under {self.dataset_root_dir}").pack(anchor="w")
        value = tk.StringVar(value="")
        entry = ttk.Entry(
            body,
            textvariable=value,
            width=42,
        )
        entry.pack(fill="x", pady=(10, 0))
        entry.focus_set()

        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(12, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        def cancel() -> None:
            dialog.destroy()

        def confirm() -> None:
            if self._apply_dataset_name(value.get(), parent=dialog):
                dialog.destroy()

        ttk.Button(buttons, text="Cancel", command=cancel).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(buttons, text="Create and use", command=confirm, style="Accent.TButton").grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        dialog.bind("<Return>", lambda _event: confirm())
        dialog.bind("<Escape>", lambda _event: cancel())
        dialog.protocol("WM_DELETE_WINDOW", cancel)

    def swap_camera_roles(self) -> None:
        """Toggle left/right wrist-camera assignments, reconnecting if needed."""
        if self.arm_mode != BIMANUAL:
            self.swap_wrist_cameras_var.set(False)
            return
        if self.recording or (self.session is not None and self.session.state is SessionState.REVIEW):
            self.swap_wrist_cameras_var.set(not self.swap_wrist_cameras_var.get())
            messagebox.showwarning("Camera roles locked", "Finish the current episode first.")
            return
        left = self.left_wrist_var.get()
        right = self.right_wrist_var.get()
        self.left_wrist_var.set(right)
        self.right_wrist_var.set(left)
        self.status_var.set("Wrist cameras swapped" if self.swap_wrist_cameras_var.get() else "Wrist cameras restored")
        if self.session is not None:
            self.disconnect()
            self.root.after(100, self.toggle_connection)

    def _update_start_button(self):
        if (
            self.piper is None
            or self.cameras is None
            or self.recording
            or self._dataset_task_running()
        ):
            state = "disabled"
        else:
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

    def _can_activation_running(self) -> bool:
        thread = self.can_activation_thread
        return thread is not None and thread.is_alive()

    def _configured_can_names(self) -> tuple[str, ...]:
        if self.arm_mode == BIMANUAL:
            return (
                validate_can_name(self.left_can_var.get()),
                validate_can_name(self.right_can_var.get()),
            )
        return (validate_can_name(self.can_var.get()),)

    def activate_can(self) -> None:
        if self.piper is not None:
            messagebox.showwarning(
                "Cannot activate CAN",
                "Disconnect robot devices before reconfiguring CAN interfaces.",
            )
            return
        if self._can_activation_running():
            return
        try:
            can_names = self._configured_can_names()
            if len(set(can_names)) != len(can_names):
                raise ValueError("Left and right CAN interface names must be distinct")
        except ValueError as exc:
            messagebox.showerror("Invalid CAN configuration", str(exc))
            return
        password = simpledialog.askstring(
            "Activate CAN",
            f"Administrator password for activating {', '.join(can_names)}:",
            parent=self.root,
            show="*",
        )
        if password is None:
            return
        if not password:
            messagebox.showerror("Activate CAN", "Administrator password must not be empty")
            return

        self.status_var.set(f"Activating CAN interfaces: {', '.join(can_names)}...")
        self.activate_can_button.configure(state="disabled")
        self.connect_button.configure(state="disabled")
        self._set_connection_config_enabled(False)

        def worker(secret: str) -> None:
            try:
                expected_bus_info = None
                if len(can_names) == 2:
                    expected_bus_info = dict(zip(can_names, BIMANUAL_CAN_USB_PORTS))
                statuses = activate_can_interfaces(
                    can_names,
                    secret,
                    expected_bus_info=expected_bus_info,
                )
                self.messages.put(("can_activation_done", statuses))
            except Exception as exc:
                self.messages.put(("can_activation_error", str(exc)))
            finally:
                # Drop the only long-lived worker reference to the password.
                secret = ""

        self.can_activation_thread = threading.Thread(
            target=worker,
            args=(password,),
            name="can-activation",
            daemon=True,
        )
        password = ""
        self.can_activation_thread.start()

    def _finish_can_activation(
        self,
        success: bool,
        payload: dict[str, dict[str, object]] | str,
    ) -> None:
        self.can_activation_thread = None
        self._set_connection_config_enabled(True)
        self.connect_button.configure(state="normal")
        self.activate_can_button.configure(state="normal")
        if success:
            assert isinstance(payload, dict)
            summary = ", ".join(f"{name}=UP" for name in payload)
            detail = ", ".join(
                f"{name}=UP/{status['bitrate']} ({status['bus_info']})"
                for name, status in payload.items()
            )
            self.status_var.set(f"CAN ready: {summary}")
            messagebox.showinfo("CAN activated", detail)
            return
        message = str(payload)
        self.status_var.set(f"CAN activation failed: {message}")
        messagebox.showerror("CAN activation failed", message)

    def toggle_connection(self):
        if self._can_activation_running():
            messagebox.showwarning("CAN activation", "Wait for CAN activation to finish")
            return
        if self.piper is not None:
            self.disconnect()
            return
        try:
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
            self.status_var.set("Connecting devices...")
            self.activate_can_button.configure(state="disabled")
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
            self.capture_stop = threading.Event()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
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
                slot = self.preview_key_to_slot.get(key)
                if slot is not None:
                    self.preview_title_labels[slot].configure(
                        text=f"{self._camera_role_title(key)}\n{video_device}"
                    )
            self.status_var.set(f"Ready: next episode {self.episode_index:04d}")
            self._set_connection_config_enabled(False)
            self.connect_button.configure(text="Disconnect devices")
            self.activate_can_button.configure(state="disabled")
            self._update_start_button()
        except Exception as exc:
            self.status_var.set(f"Connection failed: {exc}")
            self._cleanup_devices()
            self._set_connection_config_enabled(True)
            self.activate_can_button.configure(state="normal")
            messagebox.showerror("Connection failed", str(exc))

    def _handle_space_key(self, event: tk.Event) -> str | None:
        """Use Space as start/stop even when a selector or button owns focus."""
        if self._space_pressed:
            return "break"
        self._space_pressed = True
        if event.widget.winfo_toplevel()._w != self.root._w:
            return None
        widget_class = str(event.widget.winfo_class())
        if widget_class in {
            "Entry",
            "TEntry",
            "Text",
            "Spinbox",
            "TSpinbox",
        }:
            # Preserve spaces while the operator is typing task/instruction.
            return None
        if self.recording:
            self.stop_episode()
        elif self.session is not None and self.session.state is SessionState.READY:
            self.start_episode()
        elif self.session is None:
            self.status_var.set("Press Space after connecting devices to start an episode")
        else:
            self.status_var.set(
                f"Cannot start an episode while session is {self.session.state.value}"
            )
        return "break"

    def _handle_space_release(self, _event: tk.Event) -> None:
        self._space_pressed = False

    def _install_space_shortcut(self) -> None:
        """Run the collection shortcut before focused widget class bindings."""
        shortcut_tag = "CollectorSpaceShortcut"
        self.root.bind_class(shortcut_tag, "<KeyPress-space>", self._handle_space_key)
        self.root.bind_class(shortcut_tag, "<KeyPress-KP_Space>", self._handle_space_key)
        self.root.bind_class(shortcut_tag, "<KeyRelease-space>", self._handle_space_release)
        self.root.bind_class(shortcut_tag, "<KeyRelease-KP_Space>", self._handle_space_release)

        def add_tag(widget: tk.Misc) -> None:
            tags = widget.bindtags()
            if shortcut_tag not in tags:
                widget.bindtags((shortcut_tag, *tags))
            for child in widget.winfo_children():
                add_tag(child)

        add_tag(self.root)

    def start_episode(self):
        if self.session is None or self.recording:
            return
        with self.data_lock:
            qpos = None if self.latest_qpos is None else self.latest_qpos.copy()
        if qpos is None:
            messagebox.showwarning(
                "Cannot start collection",
                "No robot state feedback received yet; wait a few seconds and try again.",
            )
            return
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
        self.dataset_name_entry.configure(state="disabled")
        self.edit_dataset_button.configure(state="disabled")
        self.swap_camera_button.configure(state="disabled")
        self.status_var.set(f"Recording episode {self.episode_index:04d}")
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
                    self.latest_images = {
                        key: np.asarray(value).copy() for key, value in sample.images.items()
                    }
                    self.latest_state = state
                    self.latest_qpos = qpos
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
        self.dataset_name_entry.configure(state="readonly")
        self.edit_dataset_button.configure(state="normal")
        self.swap_camera_button.configure(state="normal")
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
            self.status_var.set(f"Saved {path.name} ({stats.actual_fps:.1f} FPS)")
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

    def _update_telemetry(self, state: np.ndarray | None) -> None:
        rows = format_arm_state_rows(
            state,
            schema=self.schema,
            arm_mode=self.arm_mode,
            arm_side=self.arm_side,
        )
        visible_sides = set()
        for side, values in rows:
            visible_sides.add(side)
            self.arm_state_vars[side].set("  ".join(f"{value:+.4f}" for value in values))
        for side in self.arm_state_vars:
            if side not in visible_sides:
                self.arm_state_vars[side].set("Waiting for data")

    def _poll_messages(self):
        with self.data_lock:
            preview = {key: value.copy() for key, value in self.latest_images.items()}
            state = None if self.latest_state is None else self.latest_state.copy()
            is_recording = self.recording
        if preview:
            self._show_preview(preview)
        self._update_telemetry(state)
        if self.piper is not None and not is_recording:
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
                elif kind == "can_activation_done":
                    self._finish_can_activation(True, payload[0])
                elif kind == "can_activation_error":
                    self._finish_can_activation(False, payload[0])
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
        if Image is not None and ImageTk is not None:
            for key, frame in images.items():
                slot = self.preview_key_to_slot.get(key)
                label = None if slot is None else self.preview_labels.get(slot)
                if label is None:
                    continue
                source_aspect = CAMERA_PREVIEW_ASPECT
                if self.cameras is not None:
                    source_aspects = getattr(self.cameras, "source_aspects", {})
                    source_aspect = source_aspects.get(key, source_aspect)
                target_w = max(320, label.winfo_width())
                target_h = max(180, label.winfo_height())
                rgb = letterbox_preview_frame(
                    frame,
                    target_hw=(target_h, target_w),
                    source_aspect=source_aspect,
                )
                if rgb is None:
                    continue
                image = Image.fromarray(rgb, mode="RGB")
                photo = ImageTk.PhotoImage(image=image)
                label.coords(self.preview_image_items[slot], 0, 0)
                label.itemconfigure(self.preview_image_items[slot], image=photo)
                label.itemconfigure(self.preview_text_items[slot], state="hidden")
                self.preview_photos[slot] = photo
            return

        # Minimal-install fallback.  The normal project environment includes
        # Pillow, so this path is only used when Tk cannot embed PhotoImage.
        for key, frame in images.items():
            slot = self.preview_key_to_slot.get(key)
            if slot is None:
                continue
            source_aspect = CAMERA_PREVIEW_ASPECT
            if self.cameras is not None:
                source_aspects = getattr(self.cameras, "source_aspects", {})
                source_aspect = source_aspects.get(key, source_aspect)
            label = self.preview_labels.get(slot)
            target_w = max(320, 640 if label is None else label.winfo_width())
            target_h = max(180, 360 if label is None else label.winfo_height())
            rgb = letterbox_preview_frame(
                frame,
                target_hw=(target_h, target_w),
                source_aspect=source_aspect,
            )
            if rgb is None:
                continue
            image = cv2.cvtColor(rgb.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
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
        selections = self.listbox.selection()
        if not selections:
            messagebox.showinfo("Delete data", "Select one or more episodes first")
            return
        paths = [self.episode_paths[item_id] for item_id in selections]
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
            ("Dataset name", self.dataset_name_var, "readonly"),
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
            "Upload archive parts are transport chunks; their count is not the episode count.\n"
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
        source_episode_count = sum(
            1
            for path in (*source.glob("ep_*.npz"), *source.glob("episode_*.npz"))
            if path.is_file()
        )
        if action == "upload":
            install_mode = self.dataset_install_mode_var.get()
            mode_hint = {
                "merge": "merge 会保留服务器已有数据并追加本次 episode，服务器总数可能增加",
                "overwrite": "overwrite 会替换服务器上同名数据集",
                "install": "install 只允许安装尚不存在的同名数据集",
            }.get(install_mode, install_mode)
            self._append_dataset_log(
                f"Source NPZ episodes: {source_episode_count}; upload mode: {install_mode}\n"
                f"{mode_hint}。上传 archive parts 是传输分片，不是 episode 数。"
            )
        else:
            self._append_dataset_log(f"Source NPZ episodes: {source_episode_count}")

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
        for item_id in self.listbox.get_children():
            self.listbox.delete(item_id)
        self.episode_paths.clear()
        directory = self.out_dir
        if not directory.exists():
            self.dataset_stats_var.set(
                f"{self.dataset_name_var.get().strip()}  |  No episodes"
            )
            return
        dataset_name = self.dataset_name_var.get().strip()
        for path in sorted(directory.glob("ep_*.npz")):
            values = episode_list_values(path, dataset_name)
            item_id = self.listbox.insert("", "end", values=values)
            self.episode_paths[item_id] = path.resolve()
        summary = summarize_dataset_directory(directory)
        invalid = f" | invalid {summary['invalid']}" if summary["invalid"] else ""
        self.dataset_stats_var.set(
            f"{dataset_name}  |  {summary['episodes']} episodes  |  "
            f"{summary['frames']} frames  |  {summary['success']} success  |  "
            f"{summary['failure']} failure{invalid}"
        )

    def replay_selected(self):
        selection = self.listbox.selection()
        if not selection:
            messagebox.showinfo("Replay", "Select an episode first")
            return
        path = self.episode_paths[selection[0]]
        viewer = pathlib.Path(__file__).with_name("view_episode.py")
        subprocess.Popen([sys.executable, str(viewer), str(path)])

    def _cleanup_devices(self):
        if self.capture_stop is not None:
            self.capture_stop.set()
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
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
        self.preview_photos.clear()
        for slot in self.preview_labels:
            self._set_preview_message(slot, "Waiting for camera...")
        # Some headless/minimal OpenCV builds do not include GUI backends.
        # Cleanup must still complete so the Exit button can destroy Tk.
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def disconnect(self):
        if self._can_activation_running():
            messagebox.showwarning("Cannot disconnect", "Wait for CAN activation to finish")
            return
        if self.recording:
            messagebox.showwarning("Cannot disconnect", "Stop the current episode first")
            return
        self._cleanup_devices()
        self.connect_button.configure(text="Connect devices")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.activate_can_button.configure(state="normal")
        self._set_connection_config_enabled(True)
        self._configure_mode_ui()
        self.status_var.set("Disconnected")

    def close(self):
        if self._can_activation_running():
            messagebox.showwarning(
                "CAN activation",
                "Wait for CAN activation to finish before exiting.",
            )
            return
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
