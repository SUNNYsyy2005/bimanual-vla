"""Authoritative Piper collection contracts for single-arm and bimanual data.

The raw NPZ contract deliberately records the robot embodiment, observation
schema, action semantics/source, dimensions, and camera keys.  Legacy single-
arm delivery episodes remain readable and are still emitted by default when
``EpisodeBuffer()`` is constructed without arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_FPS = 20
IMAGE_HW = (256, 256)
GRIPPER_MAX_M = 0.07
CONTRACT_VERSION = 2

DELIVERY_SCHEMA = "delivery"
JOINT_SCHEMA = "joint"
SINGLE_ARM = "single"
BIMANUAL = "bimanual"

DELIVERY_ACTION_SEMANTICS = "eef_delta_base_xyz_left_rotvec_gripper_target"
JOINT_ACTION_SEMANTICS = "absolute_joint_position"
LEGACY_NEXT_JOINT_ACTION_SEMANTICS = "absolute_next_joint_position"

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

JOINT_NAMES = (
    "joint_1_rad",
    "joint_2_rad",
    "joint_3_rad",
    "joint_4_rad",
    "joint_5_rad",
    "joint_6_rad",
    "gripper_opening_m",
)

# Backwards-compatible required field set for the original single-arm delivery
# files.  New code should use EpisodeContract.required_npz_fields instead.
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


def _prefixed(names: tuple[str, ...], side: str) -> tuple[str, ...]:
    return tuple(f"{side}_{name}" for name in names)


@dataclass(frozen=True)
class EpisodeContract:
    """Machine-readable description of one homogeneous Piper episode."""

    schema: str = DELIVERY_SCHEMA
    arm_mode: str = SINGLE_ARM
    arm_side: str = "right"
    camera_keys: tuple[str, ...] = ()
    action_source: str = ""
    action_alignment: str = ""
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        schema = str(self.schema).strip().lower()
        arm_mode = str(self.arm_mode).strip().lower()
        arm_side = str(self.arm_side).strip().lower()
        if schema not in {DELIVERY_SCHEMA, JOINT_SCHEMA}:
            raise ValueError(f"schema must be delivery or joint, got {self.schema!r}")
        if arm_mode not in {SINGLE_ARM, BIMANUAL}:
            raise ValueError(f"arm_mode must be single or bimanual, got {self.arm_mode!r}")
        if arm_mode == SINGLE_ARM and arm_side not in {"left", "right"}:
            raise ValueError(f"single-arm arm_side must be left or right, got {self.arm_side!r}")
        if arm_mode == BIMANUAL:
            arm_side = "both"
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "arm_mode", arm_mode)
        object.__setattr__(self, "arm_side", arm_side)

        camera_keys = tuple(str(key).strip() for key in self.camera_keys)
        if not camera_keys:
            if arm_mode == BIMANUAL:
                camera_keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
            elif schema == DELIVERY_SCHEMA:
                # Preserve the original delivery collector's public key.
                camera_keys = ("cam_high", "cam_wrist")
            else:
                camera_keys = ("cam_high", f"cam_{arm_side}_wrist")
        if not camera_keys or len(set(camera_keys)) != len(camera_keys):
            raise ValueError("camera_keys must be non-empty and unique")
        if camera_keys[0] != "cam_high" or "cam_high" not in camera_keys:
            raise ValueError("camera_keys must contain cam_high as the first camera")
        expected_wrist = 2 if arm_mode == BIMANUAL else 1
        wrist_keys = [key for key in camera_keys if "wrist" in key]
        if len(wrist_keys) != expected_wrist:
            raise ValueError(
                f"{arm_mode} contract requires {expected_wrist} wrist camera(s), got {camera_keys}"
            )
        if arm_mode == BIMANUAL and set(wrist_keys) != {"cam_left_wrist", "cam_right_wrist"}:
            raise ValueError("bimanual camera keys must include cam_left_wrist and cam_right_wrist")
        object.__setattr__(self, "camera_keys", camera_keys)

        action_source = str(self.action_source).strip()
        if not action_source:
            action_source = (
                "next_measured_qpos" if schema == JOINT_SCHEMA else "next_measured_eef"
            )
        action_alignment = str(self.action_alignment).strip()
        if not action_alignment:
            action_alignment = (
                "next_observation"
                if action_source.startswith("next_measured")
                else "same_step_command"
            )
        if action_alignment not in {"next_observation", "same_step_command"}:
            raise ValueError(
                "action_alignment must be next_observation or same_step_command"
            )
        object.__setattr__(self, "action_source", action_source)
        object.__setattr__(self, "action_alignment", action_alignment)

    @property
    def arm_sides(self) -> tuple[str, ...]:
        return (self.arm_side,) if self.arm_mode == SINGLE_ARM else ("left", "right")

    @property
    def arm_count(self) -> int:
        return len(self.arm_sides)

    @property
    def state_dim(self) -> int:
        per_arm = len(STATE_NAMES) if self.schema == DELIVERY_SCHEMA else len(JOINT_NAMES)
        return per_arm * self.arm_count

    @property
    def action_dim(self) -> int:
        per_arm = len(ACTION_NAMES) if self.schema == DELIVERY_SCHEMA else len(JOINT_NAMES)
        return per_arm * self.arm_count

    @property
    def joint_dim(self) -> int:
        return len(JOINT_NAMES) * self.arm_count

    @property
    def state_names(self) -> tuple[str, ...]:
        base = STATE_NAMES if self.schema == DELIVERY_SCHEMA else JOINT_NAMES
        if self.arm_mode == SINGLE_ARM:
            return _prefixed(base, self.arm_side)
        return _prefixed(base, "left") + _prefixed(base, "right")

    @property
    def action_names(self) -> tuple[str, ...]:
        base = ACTION_NAMES if self.schema == DELIVERY_SCHEMA else JOINT_NAMES
        if self.arm_mode == SINGLE_ARM:
            return _prefixed(base, self.arm_side)
        return _prefixed(base, "left") + _prefixed(base, "right")

    @property
    def action_semantics(self) -> str:
        return (
            DELIVERY_ACTION_SEMANTICS
            if self.schema == DELIVERY_SCHEMA
            else JOINT_ACTION_SEMANTICS
        )

    @property
    def action_offset(self) -> int:
        return 1 if self.action_alignment == "next_observation" else 0

    @property
    def robot_type(self) -> str:
        if self.arm_mode == BIMANUAL:
            return "piper_bimanual"
        return f"piper_single_arm_{self.arm_side}"

    @property
    def gripper_state_indices(self) -> tuple[int, ...]:
        per_arm = len(STATE_NAMES) if self.schema == DELIVERY_SCHEMA else len(JOINT_NAMES)
        local_index = 9 if self.schema == DELIVERY_SCHEMA else 6
        return tuple(index * per_arm + local_index for index in range(self.arm_count))

    @property
    def gripper_action_indices(self) -> tuple[int, ...]:
        return tuple(index * 7 + 6 for index in range(self.arm_count))

    def image_field(self, camera_key: str) -> str:
        if self.schema == DELIVERY_SCHEMA and self.arm_mode == SINGLE_ARM:
            return "image" if camera_key == "cam_high" else "wrist_image"
        return f"images_{camera_key}"

    def timestamp_field(self, camera_key: str) -> str:
        return f"image_timestamps_{camera_key}"

    @property
    def required_npz_fields(self) -> frozenset[str]:
        fields = {"state", "actions", "timestamps", "instruction", "success"}
        for key in self.camera_keys:
            fields.add(self.image_field(key))
            fields.add(self.timestamp_field(key))
        return frozenset(fields)

    def metadata_payload(self) -> dict[str, np.ndarray]:
        return {
            "contract_version": np.asarray(self.version, dtype=np.int64),
            "schema": np.asarray(self.schema),
            "arm_mode": np.asarray(self.arm_mode),
            "arm_side": np.asarray(self.arm_side),
            "robot_type": np.asarray(self.robot_type),
            "state_dim": np.asarray(self.state_dim, dtype=np.int64),
            "action_dim": np.asarray(self.action_dim, dtype=np.int64),
            "camera_keys": np.asarray(self.camera_keys),
            "action_semantics": np.asarray(self.action_semantics),
            "action_source": np.asarray(self.action_source),
            "action_alignment": np.asarray(self.action_alignment),
            "action_offset": np.asarray(self.action_offset, dtype=np.int64),
            "terminal_padding": np.asarray(True, dtype=np.bool_),
        }


def _scalar_text(data: Mapping[str, Any], key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.shape != ():
        return default
    return str(value.item()).strip()


def _camera_keys_from_npz(data: Mapping[str, Any]) -> tuple[str, ...]:
    if "camera_keys" in data:
        values = np.asarray(data["camera_keys"])
        if values.ndim == 1:
            return tuple(str(item).strip() for item in values.tolist())
    keys: list[str] = []
    if "image" in data or "images_cam_high" in data or "observation.images.cam_high" in data:
        keys.append("cam_high")
    for key in ("cam_wrist", "cam_left_wrist", "cam_right_wrist"):
        candidates = (
            f"images_{key}",
            f"observation.images.{key}",
            "wrist_image" if key == "cam_wrist" else "",
        )
        if any(candidate and candidate in data for candidate in candidates):
            keys.append(key)
    return tuple(keys)


def infer_episode_contract(data: Mapping[str, Any]) -> EpisodeContract:
    """Infer a contract from metadata, falling back to legacy array shapes."""
    state_key = next(
        (key for key in ("state", "observation.state", "qpos", "joint_qpos") if key in data),
        None,
    )
    if state_key is None:
        raise ValueError("missing state/observation.state/qpos field")
    state = np.asarray(data[state_key])
    state_dim = int(state.shape[-1]) if state.ndim >= 1 else 0
    schema = _scalar_text(data, "schema")
    if not schema:
        if state_key in {"qpos", "joint_qpos"}:
            schema = JOINT_SCHEMA
        elif state_dim in {10, 20}:
            schema = DELIVERY_SCHEMA
        elif state_dim in {7, 14}:
            schema = JOINT_SCHEMA
        else:
            raise ValueError(f"cannot infer schema from state dimension {state_dim}")
    per_arm = 10 if schema == DELIVERY_SCHEMA else 7
    arm_mode = _scalar_text(data, "arm_mode")
    if not arm_mode:
        if state_dim == per_arm:
            arm_mode = SINGLE_ARM
        elif state_dim == 2 * per_arm:
            arm_mode = BIMANUAL
        else:
            raise ValueError(
                f"state dimension {state_dim} is incompatible with schema {schema!r}"
            )
    camera_keys = _camera_keys_from_npz(data)
    arm_side = _scalar_text(data, "arm_side")
    if arm_mode == BIMANUAL:
        arm_side = "both"
    elif not arm_side:
        arm_side = "left" if "cam_left_wrist" in camera_keys else "right"
    action_source = _scalar_text(data, "action_source")
    action_alignment = _scalar_text(data, "action_alignment")
    if not action_alignment:
        offset = int(np.asarray(data["action_offset"]).item()) if "action_offset" in data else None
        semantics = _scalar_text(data, "action_semantics")
        if offset == 1 or semantics == LEGACY_NEXT_JOINT_ACTION_SEMANTICS:
            action_alignment = "next_observation"
    return EpisodeContract(
        schema=schema,
        arm_mode=arm_mode,
        arm_side=arm_side,
        camera_keys=camera_keys,
        action_source=action_source,
        action_alignment=action_alignment,
    )


def gripper_closed_fraction(gripper_opening_m: float) -> float:
    """Convert physical opening to the training convention 0=open, 1=closed."""
    return float(np.clip(1.0 - gripper_opening_m / GRIPPER_MAX_M, 0.0, 1.0))


def build_delivery_state(
    xyz_base_m: np.ndarray,
    rotation_base_eef: np.ndarray,
    gripper_opening_m: float,
) -> np.ndarray:
    """Build one arm's fixed 10D EEF state in the robot base frame."""
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
    """Recover an orthonormal rotation matrix from one 10D delivery state."""
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


def build_delivery_actions(states: np.ndarray, arm_count: int = 1) -> np.ndarray:
    """Build per-arm base-frame EEF deltas and terminal hold actions."""
    states = np.asarray(states, dtype=np.float32)
    expected_dim = len(STATE_NAMES) * int(arm_count)
    if states.ndim != 2 or states.shape[1:] != (expected_dim,):
        raise ValueError(f"states must have shape (T,{expected_dim}), got {states.shape}")
    actions = np.zeros((len(states), len(ACTION_NAMES) * arm_count), dtype=np.float32)
    for arm_index in range(arm_count):
        state_start = arm_index * len(STATE_NAMES)
        action_start = arm_index * len(ACTION_NAMES)
        arm_states = states[:, state_start : state_start + len(STATE_NAMES)]
        for index in range(max(0, len(states) - 1)):
            rotation = rotation_matrix_from_state(arm_states[index])
            next_rotation = rotation_matrix_from_state(arm_states[index + 1])
            delta_rotvec = Rotation.from_matrix(next_rotation @ rotation.T).as_rotvec()
            actions[index, action_start : action_start + 3] = (
                arm_states[index + 1, :3] - arm_states[index, :3]
            )
            actions[index, action_start + 3 : action_start + 6] = delta_rotvec.astype(np.float32)
            actions[index, action_start + 6] = arm_states[index + 1, 9]
        if len(states):
            actions[-1, action_start + 6] = arm_states[-1, 9]
    return actions


def build_actions(states: np.ndarray) -> np.ndarray:
    """Backwards-compatible single-arm delivery action builder."""
    return build_delivery_actions(states, arm_count=1)


def derive_joint_actions(states: np.ndarray, action_offset: int = 1) -> np.ndarray:
    """Derive future absolute joint targets with final-state padding."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] not in {7, 14}:
        raise ValueError(f"joint states must have shape (T,7) or (T,14), got {states.shape}")
    if not len(states):
        raise ValueError("joint states are empty")
    if action_offset < 0:
        raise ValueError("action_offset must be >= 0")
    indices = np.minimum(np.arange(len(states)) + int(action_offset), len(states) - 1)
    return states[indices].copy()


def terminal_hold_action(contract: EpisodeContract, state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (contract.state_dim,):
        raise ValueError(f"state must have shape ({contract.state_dim},), got {state.shape}")
    if contract.schema == JOINT_SCHEMA:
        return state.copy()
    action = np.zeros(contract.action_dim, dtype=np.float32)
    for arm_index, state_gripper_index in enumerate(contract.gripper_state_indices):
        action[arm_index * 7 + 6] = state[state_gripper_index]
    return action


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
    """Accumulate samples and serialize an explicit single/bimanual contract."""

    def __init__(
        self,
        fps: int = DEFAULT_FPS,
        *,
        schema: str = DELIVERY_SCHEMA,
        arm_mode: str = SINGLE_ARM,
        arm_side: str = "right",
        camera_keys: tuple[str, ...] | list[str] | None = None,
        action_source: str = "",
        action_alignment: str = "",
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = int(fps)
        self.contract = EpisodeContract(
            schema=schema,
            arm_mode=arm_mode,
            arm_side=arm_side,
            camera_keys=tuple(camera_keys or ()),
            action_source=action_source,
            action_alignment=action_alignment,
        )
        self.start()

    def start(self) -> None:
        self.states: list[np.ndarray] = []
        self.commanded_actions: list[np.ndarray] = []
        self._action_presence: list[bool] = []
        self.joint_qpos: list[np.ndarray] = []
        self._qpos_presence: list[bool] = []
        self.timestamps: list[float] = []
        self.images: dict[str, list[np.ndarray]] = {
            key: [] for key in self.contract.camera_keys
        }
        self.image_timestamps: dict[str, list[float]] = {
            key: [] for key in self.contract.camera_keys
        }

    def add(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        image_ts: dict[str, float],
        qpos: np.ndarray | None = None,
        action: np.ndarray | None = None,
        state_timestamp: float | None = None,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.contract.state_dim,):
            raise ValueError(
                f"state must have shape ({self.contract.state_dim},), got {state_array.shape}"
            )
        if not np.isfinite(state_array).all():
            raise ValueError("state contains NaN or Inf")
        missing_cameras = set(self.contract.camera_keys).difference(images)
        if missing_cameras:
            raise ValueError(f"missing camera frames: {sorted(missing_cameras)}")
        missing_timestamps = set(self.contract.camera_keys).difference(image_ts)
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
            if qpos_array.shape != (self.contract.joint_dim,):
                raise ValueError(
                    f"joint_qpos must have shape ({self.contract.joint_dim},), got {qpos_array.shape}"
                )
            if not np.isfinite(qpos_array).all():
                raise ValueError("joint_qpos contains NaN or Inf")

        action_array = None
        if action is not None:
            action_array = np.asarray(action, dtype=np.float32)
            if action_array.shape != (self.contract.action_dim,):
                raise ValueError(
                    f"action must have shape ({self.contract.action_dim},), got {action_array.shape}"
                )
            if not np.isfinite(action_array).all():
                raise ValueError("action contains NaN or Inf")

        frames = {
            key: _as_contract_image(images[key], key)
            for key in self.contract.camera_keys
        }
        camera_timestamps = {
            key: float(image_ts[key]) for key in self.contract.camera_keys
        }
        if not all(np.isfinite(value) for value in camera_timestamps.values()):
            raise ValueError("camera timestamps must be finite")

        self.states.append(state_array.copy())
        self._qpos_presence.append(qpos_array is not None)
        if qpos_array is not None:
            self.joint_qpos.append(qpos_array.copy())
        self._action_presence.append(action_array is not None)
        if action_array is not None:
            self.commanded_actions.append(action_array.copy())
        self.timestamps.append(timestamp)
        for key in self.contract.camera_keys:
            self.images[key].append(frames[key])
            self.image_timestamps[key].append(camera_timestamps[key])

    def __len__(self) -> int:
        return len(self.states)

    def _build_actions(self, states: np.ndarray) -> np.ndarray:
        if any(self._action_presence) and not all(self._action_presence):
            raise ValueError("action must be present for every frame or omitted entirely")
        if all(self._action_presence) and self._action_presence:
            real_actions = np.asarray(self.commanded_actions, dtype=np.float32)
            return np.concatenate(
                (real_actions, terminal_hold_action(self.contract, states[-1])[None]),
                axis=0,
            )
        if self.contract.action_alignment != "next_observation":
            raise ValueError(
                "same_step_command contract requires an explicit action for every frame"
            )
        if self.contract.schema == DELIVERY_SCHEMA:
            return build_delivery_actions(states, self.contract.arm_count)
        return derive_joint_actions(states, action_offset=1)

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
        if any(self._qpos_presence) and not all(self._qpos_presence):
            raise ValueError("joint_qpos must be present for every frame or omitted entirely")

        states_real = np.asarray(self.states, dtype=np.float32)
        states = np.concatenate((states_real, states_real[-1:]), axis=0)
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        timestamps = np.concatenate((timestamps, [timestamps[-1] + 1.0 / self.fps]))
        payload: dict[str, np.ndarray] = {
            "state": states,
            "actions": self._build_actions(states),
            "timestamps": timestamps,
            "task": np.asarray(task_name),
            "instruction": np.asarray(instruction),
            "success": np.asarray(bool(success), dtype=np.bool_),
            **self.contract.metadata_payload(),
        }
        for key in self.contract.camera_keys:
            frames = np.asarray(self.images[key], dtype=np.uint8)
            payload[self.contract.image_field(key)] = np.concatenate(
                (frames, frames[-1:]), axis=0
            )
            image_timestamps = np.asarray(self.image_timestamps[key], dtype=np.float64)
            payload[self.contract.timestamp_field(key)] = np.concatenate(
                (image_timestamps, image_timestamps[-1:])
            )
        if all(self._qpos_presence) and self._qpos_presence:
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
