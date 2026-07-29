"""3-camera capture for π0.5 inference.

Camera keys match AlohaInputs convention (what the policy server expects):
  cam_high        – head / front camera     (device ID UNKNOWN)
  cam_left_wrist  – left wrist camera       (device ID UNKNOWN)
  cam_right_wrist – right wrist camera      (device ID UNKNOWN)

Images returned as (C, H, W) uint8 RGB  ← AlohaInputs expected format.

Verify device IDs before connecting:
  ls -la /dev/video*
  v4l2-ctl --list-devices
"""

import time
import numpy as np
import cv2

# π0.5 expects 224×224 RGB images
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

    def __init__(self, cam_ids: dict = None, fps: int = 30):
        self._ids = cam_ids or dict(DEFAULT_CAM_IDS)
        self._fps = fps
        self._caps: dict[str, cv2.VideoCapture] = {}

    def open(self):
        for key, dev_id in self._ids.items():
            cap = cv2.VideoCapture(dev_id)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {key} at /dev/video{dev_id}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMG_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
            cap.set(cv2.CAP_PROP_FPS, self._fps)
            # disable internal buffering to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._caps[key] = cap

    def close(self):
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    def read(self) -> tuple[dict, dict]:
        """Return (images, timestamps).

        images: {key: np.ndarray (H, W, 3) uint8 RGB}
        timestamps: {key: float, unix seconds}
        """
        images, timestamps = {}, {}
        for key, cap in self._caps.items():
            ret, frame = cap.read()
            timestamps[key] = time.time()
            if not ret:
                raise RuntimeError(f"Camera {key} read failed")
            # OpenCV returns BGR HWC → convert to RGB CHW (AlohaInputs expects [C,H,W])
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (IMG_H, IMG_W):
                rgb = cv2.resize(rgb, (IMG_W, IMG_H))
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
