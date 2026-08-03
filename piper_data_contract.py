"""Authoritative Piper raw-data contracts for collection and validation.

Contract v3 keeps the historical schema names ``joint`` and ``delivery`` while
changing only new data semantics:

* joint: 7D measured state and 7D absolute command, with opening fraction;
* delivery: 10D measured absolute EEF state and 10D absolute EEF target, with
  a separately declared 7D model action representation.

Legacy v2 delivery episodes (including 8_3_64eps) remain explicitly
identifiable as 10D state / 7D one-step EEF delta / closed fraction.  They are
never silently interpreted as v3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from piper_action_conventions import (
    DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS,
    DELIVERY_MODEL_ACTION_SEMANTICS,
    DELIVERY_RAW_ACTION_SEMANTICS,
    DELIVERY_STEP_ACTION_SEMANTICS,
    JOINT_ACTION_SEMANTICS,
    matrix_to_rotation6d,
    rotation6d_to_matrix,
)


DEFAULT_FPS = 20
DEFAULT_ACTION_HORIZON = 50
IMAGE_HW = (256, 256)
GRIPPER_MAX_M = 0.07
CONTRACT_VERSION = 3
LEGACY_CONTRACT_VERSION = 2

DELIVERY_SCHEMA = "delivery"
JOINT_SCHEMA = "joint"
SINGLE_ARM = "single"
BIMANUAL = "bimanual"

DELIVERY_ACTION_SEMANTICS = DELIVERY_RAW_ACTION_SEMANTICS
LEGACY_DELIVERY_ACTION_SEMANTICS = DELIVERY_STEP_ACTION_SEMANTICS
DELIVERY_MEASURED_ACTION_SOURCE = "next_measured_eef_fallback"
DELIVERY_COMMANDED_GRIPPER_ACTION_SOURCE = (
    "next_measured_eef_fallback_with_same_step_commanded_gripper"
)
DELIVERY_MIXED_ACTION_SOURCE = (
    "next_measured_eef_pose_with_same_step_master_gripper_feedback"
)
MASTER_GRIPPER_FEEDBACK_ACTION_SOURCE = "master_gripper_feedback"
JOINT_MEASURED_ACTION_SOURCE = "next_measured_joint_fallback"
JOINT_MAPPED_ACTION_SOURCE = "master_joint_mapped_absolute_target"
DELIVERY_MAPPED_ACTION_SOURCE = "master_eef_mapped_absolute_target"
LEGACY_JOINT_ACTION_SEMANTICS = "absolute_joint_position"
LEGACY_NEXT_JOINT_ACTION_SEMANTICS = "absolute_next_joint_position"

GRIPPER_OPENING_SEMANTICS = "absolute_opening_fraction_0_closed_1_open"
LEGACY_GRIPPER_CLOSED_SEMANTICS = "absolute_closed_fraction_0_open_1_closed"
LEGACY_GRIPPER_OPENING_METRES_SEMANTICS = "absolute_opening_metres"
JOINT_ROTATION_SEMANTICS = "joint_positions_rad_first_6"
DELIVERY_ROTATION_SEMANTICS = (
    "state_and_raw_action_rotation6d_first_two_columns_model_action_left_rotvec"
)
DEFAULT_COORDINATE_FRAME = "slave_base"

DELIVERY_STATE_NAMES = (
    "eef_x_base_m",
    "eef_y_base_m",
    "eef_z_base_m",
    "rotation6d_col0_x",
    "rotation6d_col0_y",
    "rotation6d_col0_z",
    "rotation6d_col1_x",
    "rotation6d_col1_y",
    "rotation6d_col1_z",
    "gripper_opening_fraction",
)
DELIVERY_ACTION_NAMES = (
    "target_eef_x_base_m",
    "target_eef_y_base_m",
    "target_eef_z_base_m",
    "target_rotation6d_col0_x",
    "target_rotation6d_col0_y",
    "target_rotation6d_col0_z",
    "target_rotation6d_col1_x",
    "target_rotation6d_col1_y",
    "target_rotation6d_col1_z",
    "target_gripper_opening_fraction",
)
DELIVERY_MODEL_ACTION_NAMES = (
    "delta_x_from_current_m",
    "delta_y_from_current_m",
    "delta_z_from_current_m",
    "delta_rx_from_current_rad",
    "delta_ry_from_current_rad",
    "delta_rz_from_current_rad",
    "target_gripper_opening_fraction",
)
LEGACY_DELIVERY_STATE_NAMES = DELIVERY_STATE_NAMES[:-1] + ("gripper_closed_fraction",)
LEGACY_DELIVERY_ACTION_NAMES = (
    "delta_x_base_m",
    "delta_y_base_m",
    "delta_z_base_m",
    "delta_rx_base_rad",
    "delta_ry_base_rad",
    "delta_rz_base_rad",
    "gripper_target_closed_fraction",
)
LEGACY_JOINT_NAMES = (
    "joint_1_rad",
    "joint_2_rad",
    "joint_3_rad",
    "joint_4_rad",
    "joint_5_rad",
    "joint_6_rad",
    "gripper_opening_m",
)
JOINT_NAMES = (
    "joint_1_rad",
    "joint_2_rad",
    "joint_3_rad",
    "joint_4_rad",
    "joint_5_rad",
    "joint_6_rad",
    "gripper_opening_fraction",
)

# Public aliases retained for existing imports; they describe v3 delivery.
STATE_NAMES = DELIVERY_STATE_NAMES
ACTION_NAMES = DELIVERY_ACTION_NAMES
MODEL_ACTION_NAMES = DELIVERY_MODEL_ACTION_NAMES

# Original v2 required set is retained for legacy readers.
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
    "image": {"dtype": "image", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (10,), "names": list(DELIVERY_STATE_NAMES)},
    "actions": {"dtype": "float32", "shape": (10,), "names": list(DELIVERY_ACTION_NAMES)},
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
    action_offset: int | None = None
    fps: int = DEFAULT_FPS
    action_horizon: int = DEFAULT_ACTION_HORIZON
    coordinate_frame: str = DEFAULT_COORDINATE_FRAME
    source_frame: str = ""
    version: int = CONTRACT_VERSION
    legacy_delivery_v2: bool = False
    legacy_joint_v2: bool = False

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
        if self.legacy_delivery_v2 and schema != DELIVERY_SCHEMA:
            raise ValueError("legacy_delivery_v2 is only valid for delivery schema")
        if self.legacy_joint_v2 and schema != JOINT_SCHEMA:
            raise ValueError("legacy_joint_v2 is only valid for joint schema")
        if int(self.fps) <= 0 or int(self.action_horizon) <= 0:
            raise ValueError("fps and action_horizon must be positive")
        if not str(self.coordinate_frame).strip():
            raise ValueError("coordinate_frame must not be empty")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "arm_mode", arm_mode)
        object.__setattr__(self, "arm_side", arm_side)
        object.__setattr__(self, "fps", int(self.fps))
        object.__setattr__(self, "action_horizon", int(self.action_horizon))
        object.__setattr__(self, "coordinate_frame", str(self.coordinate_frame).strip())

        camera_keys = tuple(str(key).strip() for key in self.camera_keys)
        if not camera_keys:
            if arm_mode == BIMANUAL:
                camera_keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
            elif schema == DELIVERY_SCHEMA and self.legacy_delivery_v2:
                camera_keys = ("cam_high", "cam_wrist")
            else:
                camera_keys = ("cam_high", f"cam_{arm_side}_wrist")
        if len(set(camera_keys)) != len(camera_keys) or not camera_keys:
            raise ValueError("camera_keys must be non-empty and unique")
        if camera_keys[0] != "cam_high":
            raise ValueError("camera_keys must contain cam_high as the first camera")
        wrist_keys = tuple(key for key in camera_keys if "wrist" in key)
        expected_wrist = 2 if arm_mode == BIMANUAL else 1
        if len(wrist_keys) != expected_wrist:
            raise ValueError(f"{arm_mode} contract requires {expected_wrist} wrist camera(s)")
        if arm_mode == BIMANUAL and set(wrist_keys) != {"cam_left_wrist", "cam_right_wrist"}:
            raise ValueError("bimanual camera keys must include left and right wrist cameras")
        object.__setattr__(self, "camera_keys", camera_keys)

        source = str(self.action_source).strip()
        if not source:
            if self.legacy_delivery_v2:
                source = "next_measured_eef"
            elif schema == DELIVERY_SCHEMA:
                source = DELIVERY_MEASURED_ACTION_SOURCE
            else:
                source = JOINT_MEASURED_ACTION_SOURCE
        alignment = str(self.action_alignment).strip()
        if not alignment:
            alignment = "next_observation" if "next_measured" in source else "same_step_command"
        allowed_alignments = {
            "same_step_command",
            "next_observation",
            "next_observation_pose_same_step_gripper",
        }
        if alignment not in allowed_alignments:
            raise ValueError(f"unsupported action_alignment {alignment!r}")
        expected_offset = 0 if alignment == "same_step_command" else 1
        offset = expected_offset if self.action_offset is None else int(self.action_offset)
        if offset != expected_offset:
            raise ValueError(
                f"action_offset={offset} disagrees with action_alignment={alignment!r}; expected {expected_offset}"
            )
        object.__setattr__(self, "action_source", source)
        object.__setattr__(self, "action_alignment", alignment)
        object.__setattr__(self, "action_offset", offset)
        object.__setattr__(self, "source_frame", str(self.source_frame).strip() or self.coordinate_frame)

    @property
    def arm_sides(self) -> tuple[str, ...]:
        return (self.arm_side,) if self.arm_mode == SINGLE_ARM else ("left", "right")

    @property
    def arm_count(self) -> int:
        return len(self.arm_sides)

    @property
    def state_dim(self) -> int:
        return (10 if self.schema == DELIVERY_SCHEMA else 7) * self.arm_count

    @property
    def raw_action_dim(self) -> int:
        if self.schema == DELIVERY_SCHEMA:
            return (7 if self.legacy_delivery_v2 else 10) * self.arm_count
        return 7 * self.arm_count

    @property
    def action_dim(self) -> int:
        """Backward-compatible alias for raw_action_dim."""
        return self.raw_action_dim

    @property
    def model_action_dim(self) -> int:
        return 7 * self.arm_count

    @property
    def joint_dim(self) -> int:
        return 7 * self.arm_count

    @property
    def state_names(self) -> tuple[str, ...]:
        if self.schema == JOINT_SCHEMA:
            base = LEGACY_JOINT_NAMES if self.legacy_joint_v2 else JOINT_NAMES
        elif self.legacy_delivery_v2:
            base = LEGACY_DELIVERY_STATE_NAMES
        else:
            base = DELIVERY_STATE_NAMES
        if self.arm_mode == SINGLE_ARM:
            return _prefixed(base, self.arm_side)
        return _prefixed(base, "left") + _prefixed(base, "right")

    @property
    def action_names(self) -> tuple[str, ...]:
        if self.schema == JOINT_SCHEMA:
            base = LEGACY_JOINT_NAMES if self.legacy_joint_v2 else JOINT_NAMES
        elif self.legacy_delivery_v2:
            base = LEGACY_DELIVERY_ACTION_NAMES
        else:
            base = DELIVERY_ACTION_NAMES
        if self.arm_mode == SINGLE_ARM:
            return _prefixed(base, self.arm_side)
        return _prefixed(base, "left") + _prefixed(base, "right")

    @property
    def model_action_names(self) -> tuple[str, ...]:
        if self.schema == JOINT_SCHEMA:
            base = LEGACY_JOINT_NAMES if self.legacy_joint_v2 else JOINT_NAMES
        elif self.legacy_delivery_v2:
            base = LEGACY_DELIVERY_ACTION_NAMES
        else:
            base = DELIVERY_MODEL_ACTION_NAMES
        if self.arm_mode == SINGLE_ARM:
            return _prefixed(base, self.arm_side)
        return _prefixed(base, "left") + _prefixed(base, "right")

    @property
    def action_semantics(self) -> str:
        if self.schema == JOINT_SCHEMA:
            return LEGACY_JOINT_ACTION_SEMANTICS if self.legacy_joint_v2 else JOINT_ACTION_SEMANTICS
        return LEGACY_DELIVERY_ACTION_SEMANTICS if self.legacy_delivery_v2 else DELIVERY_RAW_ACTION_SEMANTICS

    @property
    def model_action_semantics(self) -> str:
        if self.schema == JOINT_SCHEMA:
            return self.action_semantics
        if self.legacy_delivery_v2:
            return DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS
        return DELIVERY_MODEL_ACTION_SEMANTICS

    @property
    def gripper_semantics(self) -> str:
        if self.legacy_delivery_v2:
            return LEGACY_GRIPPER_CLOSED_SEMANTICS
        if self.legacy_joint_v2:
            return LEGACY_GRIPPER_OPENING_METRES_SEMANTICS
        return GRIPPER_OPENING_SEMANTICS

    @property
    def rotation_semantics(self) -> str:
        return DELIVERY_ROTATION_SEMANTICS if self.schema == DELIVERY_SCHEMA else JOINT_ROTATION_SEMANTICS

    @property
    def robot_type(self) -> str:
        return "piper_bimanual" if self.arm_mode == BIMANUAL else f"piper_single_arm_{self.arm_side}"

    @property
    def gripper_state_indices(self) -> tuple[int, ...]:
        per_arm = 10 if self.schema == DELIVERY_SCHEMA else 7
        local = 9 if self.schema == DELIVERY_SCHEMA else 6
        return tuple(arm * per_arm + local for arm in range(self.arm_count))

    @property
    def gripper_action_indices(self) -> tuple[int, ...]:
        if self.schema == DELIVERY_SCHEMA and not self.legacy_delivery_v2:
            return tuple(arm * 10 + 9 for arm in range(self.arm_count))
        return tuple(arm * 7 + 6 for arm in range(self.arm_count))

    def image_field(self, camera_key: str) -> str:
        if self.legacy_delivery_v2 and self.arm_mode == SINGLE_ARM:
            return "image" if camera_key == "cam_high" else "wrist_image"
        return f"images_{camera_key}"

    def timestamp_field(self, camera_key: str) -> str:
        return f"image_timestamps_{camera_key}"

    @property
    def required_npz_fields(self) -> frozenset[str]:
        fields = {
            "state",
            "actions",
            "state_timestamp",
            "action_timestamp",
            "instruction",
            "success",
        }
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
            "raw_action_dim": np.asarray(self.raw_action_dim, dtype=np.int64),
            "action_dim": np.asarray(self.raw_action_dim, dtype=np.int64),
            "model_action_dim": np.asarray(self.model_action_dim, dtype=np.int64),
            "state_names": np.asarray(self.state_names),
            "action_names": np.asarray(self.action_names),
            "model_action_names": np.asarray(self.model_action_names),
            "camera_keys": np.asarray(self.camera_keys),
            "action_semantics": np.asarray(self.action_semantics),
            "model_action_semantics": np.asarray(self.model_action_semantics),
            "action_source": np.asarray(self.action_source),
            "action_alignment": np.asarray(self.action_alignment),
            "action_offset": np.asarray(self.action_offset, dtype=np.int64),
            "gripper_semantics": np.asarray(self.gripper_semantics),
            "rotation_semantics": np.asarray(self.rotation_semantics),
            "coordinate_frame": np.asarray(self.coordinate_frame),
            "source_frame": np.asarray(self.source_frame),
            "fps": np.asarray(self.fps, dtype=np.int64),
            "action_horizon": np.asarray(self.action_horizon, dtype=np.int64),
            "terminal_padding": np.asarray(True, dtype=np.bool_),
            "legacy_delivery_v2": np.asarray(self.legacy_delivery_v2, dtype=np.bool_),
            "legacy_joint_v2": np.asarray(self.legacy_joint_v2, dtype=np.bool_),
        }


def _scalar_text(data: Mapping[str, Any], key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = np.asarray(data[key])
    return str(value.item()).strip() if value.shape == () else default


def _scalar_int(data: Mapping[str, Any], key: str, default: int) -> int:
    if key not in data:
        return default
    value = np.asarray(data[key])
    return int(value.item()) if value.shape == () else default


def _camera_keys_from_npz(data: Mapping[str, Any]) -> tuple[str, ...]:
    if "camera_keys" in data:
        values = np.asarray(data["camera_keys"])
        if values.ndim == 1:
            return tuple(str(item).strip() for item in values.tolist())
    keys: list[str] = []
    if any(key in data for key in ("image", "images_cam_high", "observation.images.cam_high")):
        keys.append("cam_high")
    for key in ("cam_wrist", "cam_left_wrist", "cam_right_wrist"):
        candidates = (f"images_{key}", f"observation.images.{key}", "wrist_image" if key == "cam_wrist" else "")
        if any(candidate and candidate in data for candidate in candidates):
            keys.append(key)
    return tuple(keys)


def infer_episode_contract(data: Mapping[str, Any]) -> EpisodeContract:
    """Infer v3 or the explicit legacy-v2 path from metadata and shapes."""
    state_key = next((key for key in ("state", "observation.state", "qpos", "joint_qpos") if key in data), None)
    if state_key is None:
        raise ValueError("missing state/observation.state/qpos field")
    state = np.asarray(data[state_key])
    state_dim = int(state.shape[-1]) if state.ndim >= 1 else 0
    action_key = next((key for key in ("actions", "action") if key in data), None)
    action_dim = int(np.asarray(data[action_key]).shape[-1]) if action_key else 0
    schema = _scalar_text(data, "schema")
    if not schema:
        if state_key in {"qpos", "joint_qpos"} or state_dim in {7, 14}:
            schema = JOINT_SCHEMA
        elif state_dim in {10, 20}:
            schema = DELIVERY_SCHEMA
        else:
            raise ValueError(f"cannot infer schema from state dimension {state_dim}")
    per_arm_state = 10 if schema == DELIVERY_SCHEMA else 7
    arm_mode = _scalar_text(data, "arm_mode")
    if not arm_mode:
        if state_dim == per_arm_state:
            arm_mode = SINGLE_ARM
        elif state_dim == 2 * per_arm_state:
            arm_mode = BIMANUAL
        else:
            raise ValueError(f"state dimension {state_dim} is incompatible with schema {schema!r}")
    arm_count = 1 if arm_mode == SINGLE_ARM else 2
    semantics = _scalar_text(data, "action_semantics")
    version = _scalar_int(data, "contract_version", LEGACY_CONTRACT_VERSION if schema == DELIVERY_SCHEMA and action_dim == 7 * arm_count else 1)
    legacy_flag = (
        bool(np.asarray(data["legacy_delivery_v2"]).item())
        if "legacy_delivery_v2" in data
        else False
    )
    legacy_delivery_v2 = schema == DELIVERY_SCHEMA and (
        action_dim == 7 * arm_count
        or semantics == LEGACY_DELIVERY_ACTION_SEMANTICS
        or legacy_flag
    )
    legacy_joint_flag = (
        bool(np.asarray(data["legacy_joint_v2"]).item())
        if "legacy_joint_v2" in data
        else False
    )
    gripper_semantics = _scalar_text(data, "gripper_semantics")
    legacy_joint_v2 = schema == JOINT_SCHEMA and (
        legacy_joint_flag
        or version < CONTRACT_VERSION
        or semantics in {LEGACY_JOINT_ACTION_SEMANTICS, LEGACY_NEXT_JOINT_ACTION_SEMANTICS}
        or gripper_semantics == LEGACY_GRIPPER_OPENING_METRES_SEMANTICS
    )
    # Guard against incorrectly labelled v3 data instead of silently accepting it.
    if schema == DELIVERY_SCHEMA and action_dim not in {0, 7 * arm_count, 10 * arm_count}:
        raise ValueError(f"delivery action dimension must be {7 * arm_count} (legacy v2) or {10 * arm_count} (v3), got {action_dim}")
    if schema == DELIVERY_SCHEMA and version >= CONTRACT_VERSION and legacy_delivery_v2:
        raise ValueError("contract_version>=3 cannot use the legacy 7D delivery action layout")

    camera_keys = _camera_keys_from_npz(data)
    arm_side = _scalar_text(data, "arm_side")
    if arm_mode == BIMANUAL:
        arm_side = "both"
    elif not arm_side:
        arm_side = "left" if "cam_left_wrist" in camera_keys else "right"
    source = _scalar_text(data, "action_source")
    alignment = _scalar_text(data, "action_alignment")
    offset = _scalar_int(data, "action_offset", -1)
    if not alignment:
        if offset == 1 or semantics in {LEGACY_NEXT_JOINT_ACTION_SEMANTICS, LEGACY_DELIVERY_ACTION_SEMANTICS}:
            alignment = "next_observation"
    return EpisodeContract(
        schema=schema,
        arm_mode=arm_mode,
        arm_side=arm_side,
        camera_keys=camera_keys,
        action_source=source,
        action_alignment=alignment,
        action_offset=None if offset < 0 else offset,
        fps=_scalar_int(data, "fps", DEFAULT_FPS),
        action_horizon=_scalar_int(data, "action_horizon", DEFAULT_ACTION_HORIZON),
        coordinate_frame=_scalar_text(data, "coordinate_frame", DEFAULT_COORDINATE_FRAME),
        source_frame=_scalar_text(data, "source_frame"),
        version=version,
        legacy_delivery_v2=legacy_delivery_v2,
        legacy_joint_v2=legacy_joint_v2,
    )


def gripper_opening_fraction(gripper_opening_m: float) -> float:
    """Convert physical opening to v3 convention: 0 closed, 1 open."""
    return float(np.clip(float(gripper_opening_m) / GRIPPER_MAX_M, 0.0, 1.0))


def gripper_opening_m(opening_fraction: float) -> float:
    return float(np.clip(float(opening_fraction), 0.0, 1.0) * GRIPPER_MAX_M)


def gripper_closed_fraction(gripper_opening_m: float) -> float:
    """Legacy v2 conversion retained for readers/tests, not new collection."""
    return float(1.0 - gripper_opening_fraction(gripper_opening_m))


def build_delivery_state(
    xyz_base_m: np.ndarray,
    rotation_base_eef: np.ndarray,
    gripper_opening_m_value: float,
) -> np.ndarray:
    """Build one v3 10D EEF state from physical gripper opening in metres."""
    xyz = np.asarray(xyz_base_m, dtype=np.float64)
    rotation = np.asarray(rotation_base_eef, dtype=np.float64)
    if xyz.shape != (3,):
        raise ValueError(f"xyz_base_m must have shape (3,), got {xyz.shape}")
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation_base_eef must have shape (3,3), got {rotation.shape}")
    state = np.concatenate((xyz, matrix_to_rotation6d(rotation), [gripper_opening_fraction(gripper_opening_m_value)]))
    if not np.isfinite(state).all():
        raise ValueError("delivery state contains NaN or Inf")
    return state.astype(np.float32)


def build_delivery_state_from_opening_fraction(
    xyz_base_m: np.ndarray,
    rotation_base_eef: np.ndarray,
    opening_fraction: float,
) -> np.ndarray:
    return build_delivery_state(xyz_base_m, rotation_base_eef, gripper_opening_m(opening_fraction))


def rotation_matrix_from_state(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state)
    if values.shape != (10,):
        raise ValueError(f"state must have shape (10,), got {values.shape}")
    return rotation6d_to_matrix(values[3:9])


def build_delivery_actions(states: np.ndarray, arm_count: int = 1) -> np.ndarray:
    """Build v3 next-measured fallback as 10D absolute EEF targets."""
    values = np.asarray(states, dtype=np.float32)
    expected = 10 * int(arm_count)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"states must have shape (T,{expected}), got {values.shape}")
    if len(values) == 0:
        raise ValueError("states are empty")
    indices = np.minimum(np.arange(len(values)) + 1, len(values) - 1)
    return values[indices].copy()


def build_delivery_actions_with_gripper_targets(
    states: np.ndarray,
    gripper_targets: np.ndarray,
    arm_count: int = 1,
) -> np.ndarray:
    """Build next-measured EEF pose targets with same-step gripper feedback.

    Pose dimensions come from the next measured slave observation.  Gripper
    opening fractions are overlaid from the same-step master feedback, so no
    master/slave EEF calibration is required.
    """
    actions = build_delivery_actions(states, arm_count=arm_count)
    targets = np.asarray(gripper_targets, dtype=np.float32)
    expected = (len(actions), int(arm_count))
    if targets.shape != expected:
        raise ValueError(f"gripper_targets must have shape {expected}, got {targets.shape}")
    if not np.isfinite(targets).all():
        raise ValueError("gripper_targets contain NaN or Inf")
    if np.min(targets) < 0.0 or np.max(targets) > 1.0:
        raise ValueError("gripper_targets must use opening fraction in [0,1]")
    for arm in range(int(arm_count)):
        actions[:, arm * 10 + 9] = targets[:, arm]
    return actions


def next_observation_timestamps(state_timestamps: np.ndarray, fps: int = DEFAULT_FPS) -> np.ndarray:
    """Return action timestamps aligned to the next measured observation."""
    values = np.asarray(state_timestamps, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("state_timestamps must be a finite non-empty 1D array")
    if np.any(np.diff(values) <= 0):
        raise ValueError("state_timestamps must be strictly increasing")
    shifted = values[np.minimum(np.arange(len(values)) + 1, len(values) - 1)].copy()
    shifted[-1] = values[-1] + 1.0 / int(fps)
    return shifted


def build_legacy_delivery_step_actions(states: np.ndarray, arm_count: int = 1) -> np.ndarray:
    """Reproduce the legacy v2 7D one-step delta layout exactly."""
    values = np.asarray(states, dtype=np.float32)
    expected = 10 * int(arm_count)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"states must have shape (T,{expected}), got {values.shape}")
    actions = np.zeros((len(values), 7 * int(arm_count)), dtype=np.float32)
    for arm in range(int(arm_count)):
        ss, aa = arm * 10, arm * 7
        arm_states = values[:, ss : ss + 10]
        for index in range(max(0, len(values) - 1)):
            current_rotation = rotation_matrix_from_state(arm_states[index])
            next_rotation = rotation_matrix_from_state(arm_states[index + 1])
            actions[index, aa : aa + 3] = arm_states[index + 1, :3] - arm_states[index, :3]
            actions[index, aa + 3 : aa + 6] = Rotation.from_matrix(next_rotation @ current_rotation.T).as_rotvec()
            actions[index, aa + 6] = arm_states[index + 1, 9]
        if len(values):
            actions[-1, aa + 6] = arm_states[-1, 9]
    return actions


def build_actions(states: np.ndarray) -> np.ndarray:
    return build_delivery_actions(states, arm_count=1)


def derive_joint_actions(states: np.ndarray, action_offset: int = 1) -> np.ndarray:
    values = np.asarray(states, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] not in {7, 14}:
        raise ValueError(f"joint states must have shape (T,7) or (T,14), got {values.shape}")
    if len(values) == 0:
        raise ValueError("joint states are empty")
    if action_offset < 0:
        raise ValueError("action_offset must be >= 0")
    indices = np.minimum(np.arange(len(values)) + int(action_offset), len(values) - 1)
    return values[indices].copy()


def terminal_hold_action(contract: EpisodeContract, state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.float32)
    if values.shape != (contract.state_dim,):
        raise ValueError(f"state must have shape ({contract.state_dim},), got {values.shape}")
    if contract.schema == DELIVERY_SCHEMA and contract.legacy_delivery_v2:
        action = np.zeros(contract.raw_action_dim, dtype=np.float32)
        for arm, index in enumerate(contract.gripper_state_indices):
            action[arm * 7 + 6] = values[index]
        return action
    return values.copy()


def _as_contract_image(image: np.ndarray, name: str) -> np.ndarray:
    frame = np.asarray(image)
    if frame.shape == (3, *IMAGE_HW):
        frame = frame.transpose(1, 2, 0)
    if frame.shape != (*IMAGE_HW, 3):
        raise ValueError(f"{name} must have RGB shape {(*IMAGE_HW, 3)}, got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {frame.dtype}")
    return frame.copy()


class EpisodeBuffer:
    """Accumulate v3 samples and serialize explicit state/action timestamps."""

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
        action_offset: int | None = None,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        coordinate_frame: str = DEFAULT_COORDINATE_FRAME,
        source_frame: str = "",
    ):
        self.fps = int(fps)
        self.contract = EpisodeContract(
            schema=schema,
            arm_mode=arm_mode,
            arm_side=arm_side,
            camera_keys=tuple(camera_keys or ()),
            action_source=action_source,
            action_alignment=action_alignment,
            action_offset=action_offset,
            fps=fps,
            action_horizon=action_horizon,
            coordinate_frame=coordinate_frame,
            source_frame=source_frame,
            version=CONTRACT_VERSION,
            legacy_delivery_v2=False,
            legacy_joint_v2=False,
        )
        self.start()

    def start(self) -> None:
        self.states: list[np.ndarray] = []
        self.commanded_actions: list[np.ndarray] = []
        self._action_presence: list[bool] = []
        self.joint_qpos: list[np.ndarray] = []
        self._qpos_presence: list[bool] = []
        self.state_timestamps: list[float] = []
        self.action_timestamps: list[float] = []
        self.gripper_command_targets: list[np.ndarray] = []
        self.gripper_command_timestamps: list[np.ndarray] = []
        self.images = {key: [] for key in self.contract.camera_keys}
        self.image_timestamps = {key: [] for key in self.contract.camera_keys}

    def add(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        image_ts: dict[str, float],
        qpos: np.ndarray | None = None,
        action: np.ndarray | None = None,
        gripper_targets: np.ndarray | None = None,
        state_timestamp: float | None = None,
        action_timestamp: float | None = None,
        gripper_command_timestamps: np.ndarray | None = None,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.contract.state_dim,) or not np.isfinite(state_array).all():
            raise ValueError(f"state must be finite shape ({self.contract.state_dim},), got {state_array.shape}")
        ts_state = time.time() if state_timestamp is None else float(state_timestamp)
        ts_action = ts_state if action_timestamp is None else float(action_timestamp)
        if not np.isfinite(ts_state) or not np.isfinite(ts_action):
            raise ValueError("state/action timestamps must be finite")
        if self.state_timestamps and ts_state <= self.state_timestamps[-1]:
            raise ValueError("state timestamps must be strictly increasing")
        if self.action_timestamps and action is not None and ts_action <= self.action_timestamps[-1]:
            raise ValueError("action timestamps must be strictly increasing")

        action_array = None if action is None else np.asarray(action, dtype=np.float32)
        if action_array is not None and (action_array.shape != (self.contract.raw_action_dim,) or not np.isfinite(action_array).all()):
            raise ValueError(f"action must be finite shape ({self.contract.raw_action_dim},), got {action_array.shape}")
        qpos_array = None if qpos is None else np.asarray(qpos, dtype=np.float32)
        if qpos_array is not None and (qpos_array.shape != (self.contract.joint_dim,) or not np.isfinite(qpos_array).all()):
            raise ValueError(f"joint_qpos must be finite shape ({self.contract.joint_dim},), got {qpos_array.shape}")

        if gripper_targets is None:
            gripper_array = np.full(self.contract.arm_count, np.nan, dtype=np.float32)
        else:
            if self.contract.schema != DELIVERY_SCHEMA:
                raise ValueError("gripper_targets are only valid for delivery episodes")
            gripper_array = np.asarray(gripper_targets, dtype=np.float32)
            if gripper_array.shape != (self.contract.arm_count,):
                raise ValueError(f"gripper_targets must have shape ({self.contract.arm_count},)")
            if np.any(np.isinf(gripper_array)):
                raise ValueError("gripper_targets contain Inf")
            finite = gripper_array[np.isfinite(gripper_array)]
            if finite.size and (np.min(finite) < 0 or np.max(finite) > 1):
                raise ValueError("gripper_targets must use opening fraction in [0,1]")
        if gripper_command_timestamps is None:
            gripper_ts = np.full(self.contract.arm_count, np.nan, dtype=np.float64)
        else:
            gripper_ts = np.asarray(gripper_command_timestamps, dtype=np.float64)
            if gripper_ts.shape != (self.contract.arm_count,):
                raise ValueError(f"gripper command timestamps must have shape ({self.contract.arm_count},)")
            if np.any(np.isinf(gripper_ts)):
                raise ValueError("gripper command timestamps contain Inf")

        missing_frames = set(self.contract.camera_keys).difference(images)
        missing_ts = set(self.contract.camera_keys).difference(image_ts)
        if missing_frames or missing_ts:
            raise ValueError(f"missing cameras={sorted(missing_frames)} timestamps={sorted(missing_ts)}")
        frames = {key: _as_contract_image(images[key], key) for key in self.contract.camera_keys}
        camera_ts = {key: float(image_ts[key]) for key in self.contract.camera_keys}
        if not all(np.isfinite(value) for value in camera_ts.values()):
            raise ValueError("camera timestamps must be finite")

        self.states.append(state_array.copy())
        self._action_presence.append(action_array is not None)
        if action_array is not None:
            self.commanded_actions.append(action_array.copy())
        self._qpos_presence.append(qpos_array is not None)
        if qpos_array is not None:
            self.joint_qpos.append(qpos_array.copy())
        self.state_timestamps.append(ts_state)
        self.action_timestamps.append(ts_action)
        self.gripper_command_targets.append(gripper_array.copy())
        self.gripper_command_timestamps.append(gripper_ts.copy())
        for key in self.contract.camera_keys:
            self.images[key].append(frames[key])
            self.image_timestamps[key].append(camera_ts[key])

    def __len__(self) -> int:
        return len(self.states)

    def _build_actions(self, padded_states: np.ndarray) -> np.ndarray:
        if any(self._action_presence) and not all(self._action_presence):
            raise ValueError("action must be present for every frame or omitted entirely")
        if all(self._action_presence) and self._action_presence:
            real = np.asarray(self.commanded_actions, dtype=np.float32)
            return np.concatenate((real, terminal_hold_action(self.contract, padded_states[-1])[None]), axis=0)
        if self.contract.action_alignment == "same_step_command":
            raise ValueError("same_step_command contract requires an explicit action for every frame")
        actions = build_delivery_actions(padded_states, self.contract.arm_count) if self.contract.schema == DELIVERY_SCHEMA else derive_joint_actions(padded_states, 1)
        if self.contract.schema == DELIVERY_SCHEMA and self.gripper_command_targets:
            targets = np.asarray(self.gripper_command_targets, dtype=np.float32)
            for frame, arm_targets in enumerate(targets):
                for arm, target in enumerate(arm_targets):
                    if np.isfinite(target):
                        actions[frame, arm * 10 + 9] = target
            for arm, target in enumerate(targets[-1]):
                if np.isfinite(target):
                    actions[-1, arm * 10 + 9] = target
        return actions

    def _padded_timestamps(self, values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return np.concatenate((array, [array[-1] + 1.0 / self.fps]))

    def _build_action_timestamp(self, state_timestamp: np.ndarray) -> np.ndarray:
        if all(self._action_presence) and self._action_presence:
            real = np.asarray(self.action_timestamps, dtype=np.float64)
            terminal = max(real[-1], state_timestamp[-1]) + 1.0 / self.fps
            return np.concatenate((real, [terminal]))
        shifted = state_timestamp[np.minimum(np.arange(len(state_timestamp)) + 1, len(state_timestamp) - 1)].copy()
        shifted[-1] = max(shifted[-2], state_timestamp[-1]) + 1.0 / self.fps
        return shifted

    def build_payload(self, task_name: str, instruction: str, success: bool) -> dict[str, np.ndarray]:
        if not self.states:
            raise ValueError("cannot save an empty episode")
        task_name, instruction = task_name.strip(), instruction.strip()
        if not task_name or not instruction:
            raise ValueError("task_name and instruction must not be empty")
        if any(self._qpos_presence) and not all(self._qpos_presence):
            raise ValueError("joint_qpos must be present for every frame or omitted entirely")
        states_real = np.asarray(self.states, dtype=np.float32)
        states = np.concatenate((states_real, states_real[-1:]), axis=0)
        state_timestamp = self._padded_timestamps(self.state_timestamps)
        action_timestamp = self._build_action_timestamp(state_timestamp)
        payload: dict[str, np.ndarray] = {
            "state": states,
            "actions": self._build_actions(states),
            "timestamps": state_timestamp.copy(),
            "state_timestamp": state_timestamp,
            "action_timestamp": action_timestamp,
            "task": np.asarray(task_name),
            "instruction": np.asarray(instruction),
            "success": np.asarray(bool(success), dtype=np.bool_),
            **self.contract.metadata_payload(),
        }
        if self.contract.schema == DELIVERY_SCHEMA:
            targets = np.asarray(self.gripper_command_targets, dtype=np.float32)
            target_ts = np.asarray(self.gripper_command_timestamps, dtype=np.float64)
            present = np.isfinite(targets)
            payload["gripper_command_target"] = np.concatenate((targets, targets[-1:]), axis=0)
            payload["gripper_command_timestamp"] = np.concatenate((target_ts, target_ts[-1:]), axis=0)
            payload["gripper_command_present"] = np.concatenate((present, present[-1:]), axis=0)
            payload["pose_action_source"] = np.asarray(DELIVERY_MEASURED_ACTION_SOURCE)
            payload["pose_action_alignment"] = np.asarray("next_observation")
            payload["gripper_action_source"] = np.asarray("same_step_command_with_next_measured_fallback" if bool(present.any()) else DELIVERY_MEASURED_ACTION_SOURCE)
            payload["gripper_action_alignment"] = np.asarray("same_step_command_with_next_observation_fallback" if bool(present.any()) else "next_observation")
        for key in self.contract.camera_keys:
            frames = np.asarray(self.images[key], dtype=np.uint8)
            payload[self.contract.image_field(key)] = np.concatenate((frames, frames[-1:]), axis=0)
            payload[self.contract.timestamp_field(key)] = self._padded_timestamps(self.image_timestamps[key])
        if all(self._qpos_presence) and self._qpos_presence:
            qpos = np.asarray(self.joint_qpos, dtype=np.float32)
            payload["joint_qpos"] = np.concatenate((qpos, qpos[-1:]), axis=0)
        return payload

    def save(self, path: str | Path, task_name: str, instruction: str, success: bool) -> Path:
        path = Path(path)
        payload = self.build_payload(task_name, instruction, success)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.npz")
        try:
            np.savez_compressed(temporary, **payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Saved {len(self)} real steps -> {path}")
        return path
