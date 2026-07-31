"""Authoritative Piper collection schema shared by every UI and exporter."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_FPS = 20
IMAGE_HW = (256, 256)
GRIPPER_MAX_M = 0.07

STATE_NAMES = (
    "eef_x_base_m",
    "eef_y_base_m",
    "eef_z_base_m",
    "rotation6d_col0_x",
    "rotation6d_col0_y",
    "rotation6d_col0_z",
    "rotation6d_col1_x",
    "rotation6d_col1_y",
    "rotation6d_col1_z",
    "gripper_closed_fraction",
)

ACTION_NAMES = (
    "delta_x_base_m",
    "delta_y_base_m",
    "delta_z_base_m",
    "delta_rx_base_rad",
    "delta_ry_base_rad",
    "delta_rz_base_rad",
    "gripper_target_closed_fraction",
)

REQUIRED_EPISODE_FIELDS = frozenset(
    {
        "state",
        "actions",
        "timestamps",
        "image",
        "wrist_image",
        "instruction",
        "success",
        "image_timestamps_cam_high",
        "image_timestamps_cam_wrist",
    }
)

LEROBOT_FEATURES = {
    "image": {
        "dtype": "image",
        "shape": (*IMAGE_HW, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": (*IMAGE_HW, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (len(STATE_NAMES),),
        "names": list(STATE_NAMES),
    },
    "actions": {
        "dtype": "float32",
        "shape": (len(ACTION_NAMES),),
        "names": list(ACTION_NAMES),
    },
}


def gripper_closed_fraction(gripper_opening_m: float) -> float:
    """Convert physical opening to the training convention 0=open, 1=closed."""
    return float(np.clip(1.0 - gripper_opening_m / GRIPPER_MAX_M, 0.0, 1.0))


def build_delivery_state(
    xyz_base_m: np.ndarray,
    rotation_base_eef: np.ndarray,
    gripper_opening_m: float,
) -> np.ndarray:
    """Build the fixed 10D EEF state from base-frame pose and gripper opening."""
    xyz = np.asarray(xyz_base_m, dtype=np.float64)
    rotation = np.asarray(rotation_base_eef, dtype=np.float64)
    if xyz.shape != (3,):
        raise ValueError(f"xyz_base_m must have shape (3,), got {xyz.shape}")
    if rotation.shape != (3, 3):
        raise ValueError(
            f"rotation_base_eef must have shape (3,3), got {rotation.shape}"
        )
    rotation6d = rotation[:, :2].T.reshape(-1)
    state = np.concatenate(
        (xyz, rotation6d, [gripper_closed_fraction(float(gripper_opening_m))])
    )
    if not np.isfinite(state).all():
        raise ValueError("delivery state contains NaN or Inf")
    return state.astype(np.float32)


def rotation_matrix_from_state(state: np.ndarray) -> np.ndarray:
    """Recover an orthonormal rotation matrix from the contract rotation 6D."""
    state = np.asarray(state)
    if state.shape != (len(STATE_NAMES),):
        raise ValueError(f"state must have shape ({len(STATE_NAMES)},), got {state.shape}")
    column_0 = np.asarray(state[3:6], dtype=np.float64)
    column_1 = np.asarray(state[6:9], dtype=np.float64)
    norm_0 = float(np.linalg.norm(column_0))
    if norm_0 < 1e-12:
        raise ValueError("rotation6d first column has zero norm")
    column_0 /= norm_0
    column_1 -= column_0 * float(np.dot(column_0, column_1))
    norm_1 = float(np.linalg.norm(column_1))
    if norm_1 < 1e-12:
        raise ValueError("rotation6d second column is degenerate")
    column_1 /= norm_1
    return np.column_stack((column_0, column_1, np.cross(column_0, column_1)))


def build_actions(states: np.ndarray) -> np.ndarray:
    """Build base-frame delta actions and the required terminal no-op."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1:] != (len(STATE_NAMES),):
        raise ValueError(
            f"states must have shape (T,{len(STATE_NAMES)}), got {states.shape}"
        )
    actions = np.zeros((len(states), len(ACTION_NAMES)), dtype=np.float32)
    for index in range(max(0, len(states) - 1)):
        rotation = rotation_matrix_from_state(states[index])
        next_rotation = rotation_matrix_from_state(states[index + 1])
        # Left multiplication expresses the increment in the robot base frame.
        delta_rotvec = Rotation.from_matrix(next_rotation @ rotation.T).as_rotvec()
        actions[index, :3] = states[index + 1, :3] - states[index, :3]
        actions[index, 3:6] = delta_rotvec.astype(np.float32)
        actions[index, 6] = states[index + 1, 9]
    if len(states):
        actions[-1, 6] = states[-1, 9]
    return actions


def _as_contract_image(image: np.ndarray, name: str) -> np.ndarray:
    frame = np.asarray(image)
    if frame.shape == (3, *IMAGE_HW):
        frame = frame.transpose(1, 2, 0)
    if frame.shape != (*IMAGE_HW, 3):
        raise ValueError(
            f"{name} must have shape (3,{IMAGE_HW[0]},{IMAGE_HW[1]}) or "
            f"({IMAGE_HW[0]},{IMAGE_HW[1]},3), got {frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {frame.dtype}")
    return frame.copy()


class EpisodeBuffer:
    """Accumulate real samples and serialize the immutable delivery contract."""

    def __init__(self, fps: int = DEFAULT_FPS):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = int(fps)
        self.start()

    def start(self) -> None:
        self.states: list[np.ndarray] = []
        self.joint_qpos: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.images: dict[str, list[np.ndarray]] = {
            "cam_high": [],
            "cam_wrist": [],
        }
        self.image_timestamps: dict[str, list[float]] = {
            "cam_high": [],
            "cam_wrist": [],
        }

    def add(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        image_ts: dict[str, float],
        qpos: np.ndarray | None = None,
        state_timestamp: float | None = None,
    ) -> None:
        state_array = np.asarray(state)
        if state_array.shape != (len(STATE_NAMES),):
            raise ValueError(
                f"state must have shape ({len(STATE_NAMES)},), got {state_array.shape}"
            )
        if not np.isfinite(state_array).all():
            raise ValueError("state contains NaN or Inf")
        missing_cameras = {"cam_high", "cam_wrist"}.difference(images)
        if missing_cameras:
            raise ValueError(f"missing camera frames: {sorted(missing_cameras)}")
        missing_timestamps = {"cam_high", "cam_wrist"}.difference(image_ts)
        if missing_timestamps:
            raise ValueError(f"missing camera timestamps: {sorted(missing_timestamps)}")

        timestamp = time.time() if state_timestamp is None else float(state_timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("state timestamp must be finite")
        if self.timestamps and timestamp <= self.timestamps[-1]:
            raise ValueError("state timestamps must be strictly increasing")

        qpos_array = None
        if qpos is not None:
            qpos_array = np.asarray(qpos, dtype=np.float32)
            if qpos_array.shape != (7,):
                raise ValueError(f"joint_qpos must have shape (7,), got {qpos_array.shape}")
            if not np.isfinite(qpos_array).all():
                raise ValueError("joint_qpos contains NaN or Inf")

        frames = {
            key: _as_contract_image(images[key], key)
            for key in ("cam_high", "cam_wrist")
        }
        camera_timestamps = {
            key: float(image_ts[key])
            for key in ("cam_high", "cam_wrist")
        }
        if not all(np.isfinite(value) for value in camera_timestamps.values()):
            raise ValueError("camera timestamps must be finite")

        self.states.append(state_array.astype(np.float32, copy=True))
        if qpos_array is not None:
            self.joint_qpos.append(qpos_array.copy())
        self.timestamps.append(timestamp)
        for key in ("cam_high", "cam_wrist"):
            self.images[key].append(frames[key])
            self.image_timestamps[key].append(camera_timestamps[key])

    def __len__(self) -> int:
        return len(self.states)

    def build_payload(
        self,
        task_name: str,
        instruction: str,
        success: bool,
    ) -> dict[str, np.ndarray]:
        if not self.states:
            raise ValueError("cannot save an empty episode")
        task_name = task_name.strip()
        instruction = instruction.strip()
        if not task_name:
            raise ValueError("task_name must not be empty")
        if not instruction:
            raise ValueError("instruction must not be empty")
        if self.joint_qpos and len(self.joint_qpos) != len(self.states):
            raise ValueError("joint_qpos must be present for every frame or omitted entirely")

        states_real = np.asarray(self.states, dtype=np.float32)
        states = np.concatenate((states_real, states_real[-1:]), axis=0)
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        timestamps = np.concatenate((timestamps, [timestamps[-1] + 1.0 / self.fps]))
        payload: dict[str, np.ndarray] = {
            "state": states,
            "actions": build_actions(states),
            "timestamps": timestamps,
            "image": np.concatenate(
                (np.asarray(self.images["cam_high"], dtype=np.uint8), self.images["cam_high"][-1:]),
                axis=0,
            ),
            "wrist_image": np.concatenate(
                (np.asarray(self.images["cam_wrist"], dtype=np.uint8), self.images["cam_wrist"][-1:]),
                axis=0,
            ),
            "task": np.asarray(task_name),
            "instruction": np.asarray(instruction),
            "success": np.asarray(bool(success), dtype=np.bool_),
        }
        for key in ("cam_high", "cam_wrist"):
            image_timestamps = np.asarray(self.image_timestamps[key], dtype=np.float64)
            payload[f"image_timestamps_{key}"] = np.concatenate(
                (image_timestamps, image_timestamps[-1:])
            )
        if self.joint_qpos:
            qpos = np.asarray(self.joint_qpos, dtype=np.float32)
            payload["joint_qpos"] = np.concatenate((qpos, qpos[-1:]), axis=0)
        return payload

    def save(
        self,
        path: str | Path,
        task_name: str,
        instruction: str,
        success: bool,
    ) -> Path:
        path = Path(path)
        payload = self.build_payload(task_name, instruction, success)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.npz")
        try:
            np.savez_compressed(temporary, **payload)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"Saved {len(self)} real steps -> {path}")
        return path
