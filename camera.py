"""RGB camera capture for π0.5 inference and output-arm data collection.

Camera keys match AlohaInputs convention (what the policy server expects):
  cam_high        – head / front camera     (device ID UNKNOWN)
  cam_left_wrist  – left wrist camera       (device path/index)
  cam_right_wrist – right wrist camera      (device path/index)

Images returned as (C, H, W) uint8 RGB  ← AlohaInputs expected format.

Verify device IDs before connecting:
  ls -la /dev/video*
  v4l2-ctl --list-devices
"""

from pathlib import Path
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# Default inference size. The delivery collector overrides this to 256x256.
IMG_H, IMG_W = 224, 224

# UNKNOWN: set correct device IDs after running `v4l2-ctl --list-devices`
DEFAULT_CAM_IDS = {
    "cam_high":        0,   # head / front camera
    "cam_left_wrist":  2,   # left wrist camera
    "cam_right_wrist": 4,   # right wrist camera
}

STALE_THRESHOLD_S = 0.5   # flag image as stale if older than this

# Camera roles used by the current collection rig.  Device numbers and USB
# paths can change after reconnecting a hub, but these model names and serial
# backed udev properties remain stable.
CAMERA_MODEL_HINTS = {
    # Current physical installation: D435i is the overhead view and D405 is
    # mounted at the single/right wrist. The GUI can swap these roles when the
    # cameras are moved temporarily.
    "cam_high": ("realsense_tm__depth_camera_435i", "depth_camera_435i"),
    "cam_wrist": ("realsense_tm__depth_camera_405", "depth_camera_405"),
    "cam_right_wrist": ("realsense_tm__depth_camera_405", "depth_camera_405"),
    "cam_left_wrist": ("asus_fhd_webcam",),
}
COLOR_FORMAT_SCORES = {
    "MJPG": 40,
    "YUYV": 35,
    "RGB3": 35,
    "BGR3": 35,
    "UYVY": 10,
}


def _command_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _stable_video_selector(device: Path) -> str:
    """Prefer a stable udev symlink for a concrete video node."""
    resolved = device.resolve(strict=False)
    for directory in (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path")):
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.iterdir()):
            try:
                if candidate.resolve(strict=True) == resolved:
                    return str(candidate)
            except OSError:
                continue
    return str(device)


def discover_video_device(camera_key: str, *, device_root: Path = Path("/dev")) -> str:
    """Discover the RGB V4L2 node for a known camera role.

    RealSense devices expose depth, infrared, metadata and RGB nodes under one
    USB device.  Model matching alone is therefore insufficient: candidates
    are also ranked by their advertised colour pixel formats.
    """
    hints = CAMERA_MODEL_HINTS.get(camera_key)
    if not hints:
        raise RuntimeError(
            f"Cannot auto-discover unknown camera role {camera_key!r}; enter an explicit /dev/videoN path."
        )

    candidates: list[tuple[int, int, Path, str]] = []
    for device in device_root.glob("video[0-9]*"):
        match = re.fullmatch(r"video(\d+)", device.name)
        if match is None:
            continue
        properties = _command_output(
            ["udevadm", "info", "--query=property", f"--name={device}"]
        ).lower()
        if not any(hint in properties for hint in hints):
            continue
        formats = _command_output(["v4l2-ctl", "-d", str(device), "--list-formats"])
        format_score = max(
            (score for pixel_format, score in COLOR_FORMAT_SCORES.items() if pixel_format in formats),
            default=0,
        )
        if format_score <= 0:
            continue
        candidates.append((format_score, -int(match.group(1)), device, formats))

    if not candidates:
        raise RuntimeError(
            f"Cannot auto-discover an RGB device for {camera_key}. "
            "Check that the expected camera is connected and visible in 'v4l2-ctl --list-devices'."
        )
    _, _, selected, _ = max(candidates)
    return _stable_video_selector(selected)


def select_video_device(camera_key: str, configured_device: object) -> object:
    """Keep a valid configured selector, otherwise auto-discover by role."""
    if isinstance(configured_device, int):
        numeric_device = Path(f"/dev/video{int(configured_device)}")
        return configured_device if numeric_device.exists() else discover_video_device(camera_key)
    if isinstance(configured_device, str) and configured_device.isdigit():
        numeric_device = Path(f"/dev/video{int(configured_device)}")
        return int(configured_device) if numeric_device.exists() else discover_video_device(camera_key)

    configured_text = str(configured_device).strip()
    if configured_text.lower() != "auto":
        candidate = Path(configured_text).expanduser()
        if candidate.exists():
            return str(candidate)
    return discover_video_device(camera_key)


def resolve_video_device(device: object) -> str:
    """Return the concrete ``/dev/videoN`` path behind a camera selector.

    Collection uses stable ``/dev/v4l/by-path`` symlinks so USB enumeration
    changes do not swap camera roles.  Operators still need to see which
    numeric video node was selected for the current connection.
    """
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        candidate = Path(f"/dev/video{int(device)}")
    else:
        candidate = Path(str(device)).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        resolved = candidate.resolve(strict=False)
    if re.fullmatch(r"video\d+", resolved.name):
        return str(resolved)
    if re.fullmatch(r"video\d+", candidate.name):
        return str(candidate)
    return str(resolved)


class CameraCapture:
    """Open and read from 3 cameras.

    cam_ids: dict mapping π0.5 camera key → /dev/videoN index.
    """

    def __init__(
        self,
        cam_ids: dict = None,
        fps: int = 30,
        image_hw: tuple[int, int] = (IMG_H, IMG_W),
        capture_hw: tuple[int, int] | None = None,
        parallel_reads: bool = False,
    ):
        self._ids = cam_ids or dict(DEFAULT_CAM_IDS)
        self._configured_ids = dict(self._ids)
        self._fps = fps
        self._image_hw = tuple(image_hw)
        self._capture_hw = tuple(capture_hw or image_hw)
        self._parallel_reads = parallel_reads
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._executor: ThreadPoolExecutor | None = None

    def open(self):
        try:
            for key, configured_id in self._configured_ids.items():
                dev_id = select_video_device(key, configured_id)
                self._ids[key] = dev_id
                cap = cv2.VideoCapture(dev_id)
                if not cap.isOpened():
                    raise RuntimeError(
                        f"Cannot open camera {key} at {dev_id} "
                        f"(configured selector: {configured_id})"
                    )
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_hw[1])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_hw[0])
                cap.set(cv2.CAP_PROP_FPS, self._fps)
                # disable internal buffering to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._caps[key] = cap
        except Exception:
            self.close()
            raise
        if self._parallel_reads and len(self._caps) > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=len(self._caps),
                thread_name_prefix="camera-read",
            )

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    @staticmethod
    def _read_frame(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None, float]:
        ret, frame = cap.read()
        return ret, frame, time.time()

    def read(self) -> tuple[dict, dict]:
        """Return (images, timestamps).

        images: {key: np.ndarray (3, H, W) uint8 RGB}
        timestamps: {key: float, unix seconds}
        """
        images, timestamps = {}, {}
        if self._executor is None:
            results = {
                key: self._read_frame(cap)
                for key, cap in self._caps.items()
            }
        else:
            futures = {
                key: self._executor.submit(self._read_frame, cap)
                for key, cap in self._caps.items()
            }
            results = {key: future.result() for key, future in futures.items()}

        for key, (ret, frame, timestamp) in results.items():
            timestamps[key] = timestamp
            if not ret:
                raise RuntimeError(f"Camera {key} read failed")
            # OpenCV returns BGR HWC -> RGB HWC. Preserve aspect ratio and pad
            # with black before returning RGB CHW for the existing callers.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_h, target_w = self._image_hw
            src_h, src_w = rgb.shape[:2]
            scale = min(target_w / src_w, target_h / src_h)
            new_w = max(1, round(src_w * scale))
            new_h = max(1, round(src_h * scale))
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y0 = (target_h - new_h) // 2
            x0 = (target_w - new_w) // 2
            padded[y0:y0 + new_h, x0:x0 + new_w] = resized
            rgb = padded
            images[key] = rgb.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)
        return images, timestamps

    def check_stale(self, timestamps: dict) -> list[str]:
        """Return list of camera keys whose frames are too old."""
        now = time.time()
        return [k for k, t in timestamps.items() if now - t > STALE_THRESHOLD_S]

    def verify(self) -> dict:
        """Read one frame from each camera and return latency info (for setup check)."""
        results = {}
        for key, cap in self._caps.items():
            t0 = time.time()
            ret, frame = cap.read()
            latency_ms = (time.time() - t0) * 1000
            results[key] = {
                "ok": ret,
                "shape": frame.shape if ret else None,
                "latency_ms": round(latency_ms, 1),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "configured_device": str(self._configured_ids[key]),
                "selected_device": str(self._ids[key]),
                "video_device": resolve_video_device(self._ids[key]),
            }
        return results
