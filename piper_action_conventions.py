"""Piper action conventions shared by collection and policy-facing code.

The raw data contract has two deliberately separate representations:

* ``joint`` stores 7D absolute joint/gripper targets.
* ``delivery`` stores 10D absolute EEF targets.  π0.5 consumes a 7D
  current-anchored ``xyz + rotvec + gripper`` representation, which is
  produced by the functions in this module at the model boundary.

The legacy v2 delivery layout (7D one-step EEF deltas and closed-fraction
 gripper) is kept here as an explicit conversion path; it is never inferred as
 the v3 raw action convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


DELIVERY_STEP_ACTION_CONVENTION = "step"
DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION = "chunk_origin"
DELIVERY_ACTION_CONVENTIONS = frozenset(
    {DELIVERY_STEP_ACTION_CONVENTION, DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION}
)

# 8_3_64eps and the original Piper delivery collector use these exact semantics.
DELIVERY_STEP_ACTION_SEMANTICS = "eef_delta_base_xyz_left_rotvec_gripper_target"
DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS = (
    "eef_delta_chunk_origin_base_xyz_left_rotvec_gripper_target"
)

# v3 raw and model-side semantics.
DELIVERY_RAW_ACTION_SEMANTICS = "absolute_eef_target"
NEW_DELIVERY_RAW_ACTION_SEMANTICS = DELIVERY_RAW_ACTION_SEMANTICS
LEGACY_DELIVERY_STEP_ACTION_SEMANTICS = DELIVERY_STEP_ACTION_SEMANTICS
DELIVERY_MODEL_ACTION_SEMANTICS = (
    "eef_delta_chunk_origin_base_xyz_left_rotvec_gripper_opening_target"
)
JOINT_ACTION_SEMANTICS = "absolute_joint_position_opening_fraction"
NEW_GRIPPER_SEMANTICS = "absolute_opening_fraction_0_closed_1_open"
LEGACY_GRIPPER_SEMANTICS = "absolute_closed_fraction_0_open_1_closed"


def _check_arm_count(arm_count: int) -> int:
    arm_count = int(arm_count)
    if arm_count not in {1, 2}:
        raise ValueError(f"arm_count must be 1 or 2, got {arm_count!r}")
    return arm_count


def rotation6d_to_matrix(rotation6d: np.ndarray) -> np.ndarray:
    """Convert the first two rotation-matrix columns to a proper matrix."""
    values = np.asarray(rotation6d, dtype=np.float64)
    if values.shape[-1] != 6:
        raise ValueError(f"rotation6d must end in 6 values, got {values.shape}")
    col0 = values[..., :3]
    col1 = values[..., 3:]
    norm0 = np.linalg.norm(col0, axis=-1, keepdims=True)
    if np.any(norm0 < 1e-12):
        raise ValueError("rotation6d first column has zero norm")
    col0 = col0 / norm0
    col1 = col1 - col0 * np.sum(col0 * col1, axis=-1, keepdims=True)
    norm1 = np.linalg.norm(col1, axis=-1, keepdims=True)
    if np.any(norm1 < 1e-12):
        raise ValueError("rotation6d second column is degenerate")
    col1 = col1 / norm1
    col2 = np.cross(col0, col1)
    return np.stack((col0, col1, col2), axis=-1)


def matrix_to_rotation6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Store a rotation matrix as its first two columns in row-major 6D form."""
    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3,3), got {matrix.shape}")
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def _validate_eef_state(state: np.ndarray, arm_count: int) -> np.ndarray:
    arm_count = _check_arm_count(arm_count)
    values = np.asarray(state, dtype=np.float32)
    expected = 10 * arm_count
    if values.shape != (expected,):
        raise ValueError(f"current EEF state must have shape ({expected},), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("current EEF state contains NaN or Inf")
    return values


def _validate_eef_targets(targets: np.ndarray, arm_count: int) -> tuple[np.ndarray, bool]:
    arm_count = _check_arm_count(arm_count)
    values = np.asarray(targets, dtype=np.float32)
    expected = 10 * arm_count
    was_one = values.ndim == 1
    if was_one:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"EEF targets must have shape (T,{expected}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("EEF targets contain NaN or Inf")
    return values, was_one


def absolute_eef_targets_to_model_actions(
    current_state: np.ndarray,
    absolute_targets: np.ndarray,
    *,
    arm_count: int = 1,
) -> np.ndarray:
    """Convert absolute 10D targets to current-anchored 7D model actions.

    Every target row is anchored to the *same* ``current_state``.  Rotation is
    ``R_target @ R_current.T`` and is returned as a rotvec.  The gripper is an
    absolute opening fraction and is therefore copied, not differenced.
    """
    current = _validate_eef_state(current_state, arm_count)
    targets, was_one = _validate_eef_targets(absolute_targets, arm_count)
    output = np.empty((len(targets), 7 * arm_count), dtype=np.float32)
    for arm in range(arm_count):
        ss = arm * 10
        aa = arm * 7
        current_rotation = rotation6d_to_matrix(current[ss + 3 : ss + 9])
        target_rotations = rotation6d_to_matrix(targets[:, ss + 3 : ss + 9])
        output[:, aa : aa + 3] = targets[:, ss : ss + 3] - current[ss : ss + 3]
        output[:, aa + 3 : aa + 6] = np.asarray(
            [Rotation.from_matrix(target @ current_rotation.T).as_rotvec() for target in target_rotations],
            dtype=np.float32,
        )
        output[:, aa + 6] = targets[:, ss + 9]
    return output[0] if was_one else output


def absolute_eef_targets_to_chunk_origin(
    state: np.ndarray,
    targets: np.ndarray,
    arm_count: int = 1,
) -> np.ndarray:
    """Stable public alias for v3 absolute-target -> model-action conversion."""
    return absolute_eef_targets_to_model_actions(state, targets, arm_count=arm_count)


def model_actions_to_absolute_eef_targets(
    current_state: np.ndarray,
    model_actions: np.ndarray,
    *,
    arm_count: int = 1,
) -> np.ndarray:
    """Decode current-anchored 7D model actions to absolute 10D EEF targets."""
    current = _validate_eef_state(current_state, arm_count)
    values = np.asarray(model_actions, dtype=np.float32)
    expected = 7 * _check_arm_count(arm_count)
    was_one = values.ndim == 1
    if was_one:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"model actions must have shape (T,{expected}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("model actions contain NaN or Inf")
    output = np.empty((len(values), 10 * arm_count), dtype=np.float32)
    for arm in range(arm_count):
        ss = arm * 10
        aa = arm * 7
        current_rotation = rotation6d_to_matrix(current[ss + 3 : ss + 9])
        output[:, ss : ss + 3] = current[ss : ss + 3] + values[:, aa : aa + 3]
        output[:, ss + 3 : ss + 9] = np.asarray(
            [matrix_to_rotation6d(Rotation.from_rotvec(row).as_matrix() @ current_rotation) for row in values[:, aa + 3 : aa + 6]],
            dtype=np.float32,
        )
        output[:, ss + 9] = np.clip(values[:, aa + 6], 0.0, 1.0)
    return output[0] if was_one else output


def chunk_origin_deltas_to_absolute_eef_targets(
    state: np.ndarray,
    actions: np.ndarray,
    arm_count: int = 1,
) -> np.ndarray:
    """Stable public alias for v3 model-action -> absolute-target conversion."""
    return model_actions_to_absolute_eef_targets(state, actions, arm_count=arm_count)


def step_deltas_to_chunk_origin(actions: np.ndarray, *, arm_count: int) -> np.ndarray:
    """Convert legacy per-step delivery deltas to chunk-origin deltas.

    This function is intentionally only for legacy v2 data.  Gripper values
    are closed-fraction absolute targets in that layout and are copied.
    """
    arm_count = _check_arm_count(arm_count)
    values = np.asarray(actions)
    expected_dim = 7 * arm_count
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(f"actions must have shape (T,{expected_dim}), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("actions contain NaN or Inf")
    dtype = np.result_type(values.dtype, np.float32)
    values = values.astype(dtype, copy=False)
    output = values.copy()
    for arm_index in range(arm_count):
        offset = arm_index * 7
        output[:, offset : offset + 3] = np.cumsum(values[:, offset : offset + 3], axis=0)
        cumulative_rotation = np.eye(3, dtype=np.float64)
        for step, rotvec in enumerate(values[:, offset + 3 : offset + 6]):
            cumulative_rotation = Rotation.from_rotvec(rotvec).as_matrix() @ cumulative_rotation
            output[step, offset + 3 : offset + 6] = Rotation.from_matrix(cumulative_rotation).as_rotvec()
    return output


@dataclass(frozen=True)
class EefCommandMapping:
    """Explicit master-EFF to slave-base mapping used by delivery teleop.

    The calibration is intentionally required for delivery collection.  An
    identity mapping is not silently assumed because that can produce valid-
    looking but physically wrong labels.
    """

    rotation_matrix: np.ndarray
    translation_m: np.ndarray
    position_scale: float = 1.0
    gripper_fraction_scale: float = 1.0
    gripper_fraction_offset: float = 0.0
    source_frame: str = "master_base"
    target_frame: str = "slave_base"

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_matrix, dtype=np.float64)
        translation = np.asarray(self.translation_m, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("EEF calibration rotation_matrix must be (3,3)")
        if translation.shape != (3,):
            raise ValueError("EEF calibration translation_m must be (3,)")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("EEF calibration contains NaN or Inf")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) <= 0:
            raise ValueError("EEF calibration rotation_matrix must be a proper rotation")
        for name, value in (("position_scale", self.position_scale), ("gripper_fraction_scale", self.gripper_fraction_scale), ("gripper_fraction_offset", self.gripper_fraction_offset)):
            if not np.isfinite(float(value)):
                raise ValueError(f"EEF calibration {name} must be finite")
        if not str(self.source_frame).strip() or not str(self.target_frame).strip():
            raise ValueError("EEF calibration frames must not be empty")
        object.__setattr__(self, "rotation_matrix", rotation)
        object.__setattr__(self, "translation_m", translation)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EefCommandMapping":
        if not isinstance(value, Mapping):
            raise ValueError("EEF calibration must be a JSON object")
        return cls(
            rotation_matrix=np.asarray(value.get("rotation_matrix", value.get("rotation")), dtype=np.float64),
            translation_m=np.asarray(value.get("translation_m", value.get("translation")), dtype=np.float64),
            position_scale=float(value.get("position_scale", 1.0)),
            gripper_fraction_scale=float(value.get("gripper_fraction_scale", 1.0)),
            gripper_fraction_offset=float(value.get("gripper_fraction_offset", 0.0)),
            source_frame=str(value.get("source_frame", "master_base")),
            target_frame=str(value.get("target_frame", "slave_base")),
        )

    def map_target(self, master_target: np.ndarray) -> np.ndarray:
        target = _validate_eef_state(master_target, 1)
        xyz = self.translation_m + self.position_scale * (self.rotation_matrix @ target[:3])
        master_rotation = rotation6d_to_matrix(target[3:9])
        slave_rotation = self.rotation_matrix @ master_rotation
        gripper = np.clip(
            self.gripper_fraction_scale * target[9] + self.gripper_fraction_offset, 0.0, 1.0
        )
        return np.concatenate((xyz, matrix_to_rotation6d(slave_rotation), [gripper])).astype(np.float32)


def load_eef_command_mappings(path: str | Path, *, arm_sides: tuple[str, ...]) -> dict[str, EefCommandMapping]:
    """Load explicit delivery calibration, failing closed on missing/invalid data."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"delivery collection requires an EEF calibration file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise ValueError("EEF calibration file must contain a JSON object")
    if "arms" in document:
        document = document["arms"]
    result: dict[str, EefCommandMapping] = {}
    if len(arm_sides) == 1 and ("rotation_matrix" in document or "rotation" in document):
        result[arm_sides[0]] = EefCommandMapping.from_mapping(document)
    else:
        for side in arm_sides:
            if side not in document:
                raise ValueError(f"EEF calibration missing arm {side!r}")
            result[side] = EefCommandMapping.from_mapping(document[side])
    return result
