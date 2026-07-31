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

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

# Default inference size. The delivery collector overrides this to 256x256.
IMG_H, IMG_W = 224, 224

# UNKNOWN: set correct device IDs after running `v4l2-ctl --list-devices`
DEFAULT_CAM_IDS = {
    "cam_high":        0,   # head / front camera
    "cam_left_wrist":  2,   # left wrist camera
    "cam_right_wrist": 4,   # right wrist camera
}

STALE_THRESHOLD_S = 0.5   # flag image as stale if older than this


class CameraCapture:
    """Open and read from 3 cameras.

    cam_ids: dict mapping π0.5 camera key → /dev/videoN index.
    """

    def __init__(
        self,
        cam_ids: dict = None,
        fps: int = 30,
        image_hw: tuple[int, int] = (IMG_H, IMG_W),
        parallel_reads: bool = False,
    ):
        self._ids = cam_ids or dict(DEFAULT_CAM_IDS)
        self._fps = fps
        self._image_hw = tuple(image_hw)
        self._parallel_reads = parallel_reads
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._executor: ThreadPoolExecutor | None = None

    def open(self):
        for key, dev_id in self._ids.items():
            cap = cv2.VideoCapture(dev_id)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {key} at {dev_id}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._image_hw[1])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._image_hw[0])
            cap.set(cv2.CAP_PROP_FPS, self._fps)
            # disable internal buffering to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._caps[key] = cap
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

        images: {key: np.ndarray (H, W, 3) uint8 RGB}
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
                "ok":        ret,
                "shape":     frame.shape if ret else None,   # raw HWC from OpenCV
                "latency_ms": round(latency_ms, 1),
            }
        return results
