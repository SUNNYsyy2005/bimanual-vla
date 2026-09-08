"""Reusable, read-only analysis helpers for one recorded episode.

The dashboard and the desktop collection GUI both need the same small set of
operations: turn state/action arrays into per-joint velocities, identify likely
idle frames, and expose a compact end-effector trajectory.  This module keeps
those operations independent of Flask, Tk, parquet, or any particular UI.

All functions accept numpy-like arrays and return plain dictionaries/arrays so
callers can choose their own serialization and rendering strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from bimanual_vla.data.action_conventions import matrix_to_rotation6d, rotation6d_to_matrix
from bimanual_vla.data.arm_geometry import (
    ARM_BASE_AXIS_CONVENTION,
    arm_base_axis_signs,
    arm_base_rotation_metadata,
    arm_base_origins,
    normalize_arm_base_offset,
    normalize_arm_base_rotations,
    normalize_robot_type,
    REAL_PIPER_RIGHT_BASE_ROTATION,
)


DEFAULT_JOINT_NAMES = tuple(
    value
    for side in ("left", "right")
    for value in ([f"{side}_joint_{index}" for index in range(1, 7)] + [f"{side}_gripper"])
)

FRANKA_JOINT_NAMES = tuple(
    value
    for side in ("left", "right")
    for value in ([f"{side}_joint_{index}" for index in range(1, 8)] + [f"{side}_gripper"])
)


def _joint_names_for_robot(robot_type: str | None, side: str, width: int) -> tuple[str, ...] | None:
    family = normalize_robot_type(robot_type)
    arm_count = 2 if width in {14, 16} else 1
    per_arm = width // arm_count if arm_count else 0
    if per_arm not in {7, 8}:
        return None
    joint_count = per_arm - 1
    prefix = "fl" if family == "aloha-agilex" and side == "left" else "fr" if family == "aloha-agilex" else side
    return tuple(f"{prefix}_joint_{index}" for index in range(1, joint_count + 1)) + (f"{prefix}_gripper",)


@dataclass(frozen=True)
class EpisodeAnalysis:
    """Numpy representation used internally by the public helpers."""

    timestamps: np.ndarray
    state: np.ndarray
    action: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    speed_norm: np.ndarray
    idle: np.ndarray
    velocity_reversal: np.ndarray
    jitter: np.ndarray
    abrupt_change: np.ndarray
    anomaly_score: np.ndarray
    jerk_norm: np.ndarray
    joint_names: tuple[str, ...]
    eef: dict[str, dict[str, np.ndarray]]
    eef_method: str
    fps: float
    arm_base_offset: tuple[float, float, float] | None
    arm_base_rotations: dict[str, np.ndarray] | None
    robot_type: str | None

    @property
    def frame_count(self) -> int:
        return int(len(self.timestamps))


def _as_matrix(value: Any, *, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D frame matrix, got shape={array.shape}")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"expected width {width}, got shape={array.shape}")
    return array


def infer_joint_names(
    names: Sequence[Any] | None,
    width: int,
    *,
    arm_side: str = "right",
    robot_type: str | None = None,
) -> tuple[str, ...]:
    """Return stable labels for a state/action vector."""

    if names is not None:
        values = tuple(str(item) for item in names)
        if len(values) == width:
            return values
    if width in {14, 16} and robot_type is not None:
        generated = tuple(
            name
            for side in ("left", "right")
            for name in (_joint_names_for_robot(robot_type, side, width // 2) or ())
        )
        if len(generated) == width:
            return generated
    if width == 14:
        return DEFAULT_JOINT_NAMES
    if width == 16:
        return FRANKA_JOINT_NAMES
    if width == 7:
        side = arm_side if arm_side in {"left", "right"} else "right"
        return tuple(f"{side}_joint_{index}" for index in range(1, 7)) + (f"{side}_gripper",)
    if width == 8:
        side = arm_side if arm_side in {"left", "right"} else "right"
        return tuple(f"{side}_joint_{index}" for index in range(1, 8)) + (f"{side}_gripper",)
    return tuple(f"dim_{index + 1}" for index in range(width))


def _safe_time_axis(timestamps: Any, count: int, fps: float | int | None = None) -> np.ndarray:
    if timestamps is None:
        step = 1.0 / max(1.0, float(fps or 20.0))
        return np.arange(count, dtype=np.float64) * step
    axis = np.asarray(timestamps, dtype=np.float64).reshape(-1)[:count]
    if len(axis) != count:
        step = 1.0 / max(1.0, float(fps or 20.0))
        axis = np.arange(count, dtype=np.float64) * step
    if count > 1:
        deltas = np.diff(axis)
        finite = deltas[np.isfinite(deltas) & (deltas > 0)]
        fallback = float(np.median(finite)) if len(finite) else 1.0 / max(1.0, float(fps or 20.0))
        repaired = axis.copy()
        for index in range(1, count):
            delta = repaired[index] - repaired[index - 1]
            if not np.isfinite(delta) or delta <= 0:
                repaired[index] = repaired[index - 1] + fallback
        axis = repaired
    return axis


def compute_joint_velocities(values: Any, timestamps: Any = None, *, fps: float | int | None = None) -> np.ndarray:
    """Compute per-joint velocity using a repaired timestamp axis.

    Missing values remain NaN.  The first/last samples use one-sided
    differences; a single-frame episode has zero velocity.
    """

    matrix = _as_matrix(values)
    count = len(matrix)
    if count == 0:
        return np.empty_like(matrix)
    axis = _safe_time_axis(timestamps, count, fps)
    result = np.full_like(matrix, np.nan, dtype=np.float64)
    if count == 1:
        result[0] = 0.0
        return result
    for index in range(count):
        if index == 0:
            delta = axis[1] - axis[0]
            result[index] = (matrix[1] - matrix[0]) / delta
        elif index == count - 1:
            delta = axis[-1] - axis[-2]
            result[index] = (matrix[-1] - matrix[-2]) / delta
        else:
            delta = axis[index + 1] - axis[index - 1]
            result[index] = (matrix[index + 1] - matrix[index - 1]) / delta
    invalid = ~np.isfinite(matrix)
    result[invalid] = np.nan
    return result


def compute_joint_accelerations(values: Any, timestamps: Any = None, *, fps: float | int | None = None) -> np.ndarray:
    """Compute per-joint acceleration from a frame matrix."""

    matrix = _as_matrix(values)
    velocities = compute_joint_velocities(matrix, timestamps, fps=fps)
    return compute_joint_velocities(velocities, timestamps, fps=fps)


def _robust_limit(values: np.ndarray, minimum: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float(minimum)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    # A robust limit catches isolated spikes without treating a uniformly fast
    # episode as anomalous.  The explicit floor keeps normal low-noise data
    # from producing false positives due to floating point quantization.
    return max(float(minimum), median + 6.0 * max(mad, 1e-9))


def detect_motion_anomalies(
    state: Any,
    timestamps: Any = None,
    *,
    fps: float | int | None = None,
    velocity_reversal_threshold: float = 0.05,
    acceleration_threshold: float = 2.0,
    jerk_threshold: float = 12.0,
    jitter_window: int = 5,
    jitter_min_reversals: int = 2,
) -> dict[str, np.ndarray]:
    """Detect reversals, oscillatory jitter, and abrupt motion changes.

    Returned masks are per-joint for ``velocity_reversal`` and ``jitter`` and
    per-frame for ``abrupt_change``/``anomaly``.  Thresholds are intentionally
    conservative for joint-radian data; callers can tune them for other units.
    Robust acceleration/jerk limits make the detector useful for both a slow
    demonstration and a faster real-robot recording.
    """

    matrix = _as_matrix(state)
    count, width = matrix.shape
    velocities = compute_joint_velocities(matrix, timestamps, fps=fps)
    accelerations = compute_joint_accelerations(matrix, timestamps, fps=fps)
    jerk = compute_joint_velocities(accelerations, timestamps, fps=fps)
    velocity_reversal = np.zeros((count, width), dtype=bool)
    if count > 2:
        # Use adjacent one-sided sample velocities for reversal detection.  A
        # centered derivative intentionally smooths a sign change at a single
        # sample, while the one-sided values preserve the physical direction
        # change that operators need to inspect.
        axis = _safe_time_axis(timestamps, count, fps)
        sample_velocity = np.diff(matrix, axis=0) / np.diff(axis)[:, None]
        previous = sample_velocity[:-1]
        current = sample_velocity[1:]
        strong = (np.abs(previous) >= float(velocity_reversal_threshold)) & (
            np.abs(current) >= float(velocity_reversal_threshold)
        )
        velocity_reversal[2:] = strong & ((previous * current) < 0.0)

    motion_dims = min(6, width)
    acceleration_norm = np.linalg.norm(np.nan_to_num(accelerations[:, :motion_dims]), axis=1) if motion_dims else np.zeros(count)
    jerk_norm = np.linalg.norm(np.nan_to_num(jerk[:, :motion_dims]), axis=1) if motion_dims else np.zeros(count)
    acceleration_limit = _robust_limit(acceleration_norm, acceleration_threshold)
    jerk_limit = _robust_limit(jerk_norm, jerk_threshold)
    abrupt_change = (acceleration_norm >= acceleration_limit) | (jerk_norm >= jerk_limit)

    window = max(3, int(jitter_window))
    min_reversals = max(2, int(jitter_min_reversals))
    jitter = np.zeros((count, width), dtype=bool)
    if count:
        for index in range(count):
            start = max(1, index - window + 1)
            jitter[index] = np.sum(velocity_reversal[start : index + 1], axis=0) >= min_reversals
    reversal_mask = np.any(velocity_reversal, axis=1)
    anomaly = reversal_mask | np.any(jitter, axis=1) | abrupt_change
    score = np.maximum(
        acceleration_norm / max(acceleration_limit, 1e-9),
        jerk_norm / max(jerk_limit, 1e-9),
    )
    score = np.maximum(score, np.minimum(1.0, np.sum(velocity_reversal, axis=1) / 2.0))
    return {
        "accelerations": accelerations,
        "acceleration_norm": acceleration_norm,
        "jerk_norm": jerk_norm,
        "velocity_reversal": velocity_reversal,
        "jitter": jitter,
        "abrupt_change": abrupt_change,
        "anomaly": anomaly,
        "anomaly_score": score,
        "acceleration_limit": np.asarray(acceleration_limit, dtype=np.float64),
        "jerk_limit": np.asarray(jerk_limit, dtype=np.float64),
    }


def detect_idle_frames(
    state: Any,
    *,
    action: Any = None,
    timestamps: Any = None,
    fps: float | int | None = None,
    velocity_threshold: float = 0.035,
    action_delta_threshold: float = 0.012,
    min_run: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mark frames whose joint motion and command changes are both small.

    Returns ``(idle_mask, velocity_norm, action_delta_norm)``.  A short run is
    deliberately not marked idle; this prevents a single slow sample in the
    middle of a motion from becoming a crop boundary.  If all frames are slow,
    the complete episode is considered idle and callers should avoid cropping
    it to an empty range.
    """

    measured = _as_matrix(state)
    count = len(measured)
    velocities = compute_joint_velocities(measured, timestamps, fps=fps)
    motion_dims = min(6, measured.shape[1])
    speed_norm = np.linalg.norm(np.nan_to_num(velocities[:, :motion_dims]), axis=1)
    if action is None:
        command_delta = np.zeros(count, dtype=np.float64)
    else:
        desired = _as_matrix(action)
        width = min(measured.shape[1], desired.shape[1])
        command_delta = np.zeros(count, dtype=np.float64)
        if count > 1 and width:
            deltas = np.diff(desired[:, :width], axis=0)
            command_delta[1:] = np.linalg.norm(np.nan_to_num(deltas), axis=1)
    candidates = (speed_norm <= float(velocity_threshold)) & (
        command_delta <= float(action_delta_threshold)
    )
    candidates &= np.isfinite(measured).all(axis=1)
    if action is not None:
        candidates &= np.isfinite(desired[:, : min(measured.shape[1], desired.shape[1])]).all(axis=1)
    idle = np.zeros(count, dtype=bool)
    run_start: int | None = None
    run_length = max(1, int(min_run))
    for index, value in enumerate(np.r_[candidates, False]):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            if index - run_start >= run_length:
                idle[run_start:index] = True
            run_start = None
    if count and bool(np.all(candidates)):
        idle[:] = True
    return idle, speed_norm, command_delta


def idle_runs(mask: Any) -> list[dict[str, int]]:
    """Convert a boolean idle mask to inclusive frame ranges."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    runs: list[dict[str, int]] = []
    start: int | None = None
    for index, value in enumerate(np.r_[values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append({"start_frame": start, "end_frame": index - 1, "length": index - start})
            start = None
    return runs


def suggested_crop(mask: Any) -> dict[str, Any]:
    """Suggest removing only leading/trailing idle runs."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not len(values) or bool(np.all(values)):
        end = max(0, len(values) - 1)
        return {"start_frame": 0, "end_frame": end, "removed_leading": 0, "removed_trailing": 0, "all_idle": bool(len(values))}
    start = 0
    while start < len(values) and values[start]:
        start += 1
    end = len(values) - 1
    while end >= 0 and values[end]:
        end -= 1
    return {
        "start_frame": start,
        "end_frame": max(start, end),
        "removed_leading": start,
        "removed_trailing": len(values) - 1 - end,
        "all_idle": False,
    }


def _fallback_fk(joints: np.ndarray) -> np.ndarray:
    """Small deterministic Piper-like FK fallback for dashboard inspection.

    The runtime uses ``piper_sdk`` when available.  A lightweight fallback keeps
    the visualizer useful on machines where the SDK is not installed; its
    output is intentionally labelled ``approx_fk`` by the caller.
    """

    lengths = np.asarray([0.11, 0.10, 0.09, 0.07, 0.05, 0.035], dtype=np.float64)
    angles = np.nan_to_num(np.asarray(joints[:6], dtype=np.float64))
    cumulative = np.cumsum(angles)
    x = float(np.sum(lengths * np.cos(cumulative)))
    y = float(np.sum(lengths * np.sin(cumulative)))
    z = float(0.16 + 0.025 * np.sin(angles[1]) + 0.02 * np.sin(angles[2] + angles[3]))
    return np.asarray([x, y, z], dtype=np.float64)


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


_ROBOTWIN_ROOT_ROTATION = _rotation_z(np.pi / 2.0)


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_from_rpy(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    return _rotation_z(float(yaw)) @ _rotation_y(float(pitch)) @ _rotation_x(float(roll))


def _axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis_x, axis_y, axis_z = vector / norm
    cosine, sine = np.cos(angle), np.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.asarray(
        [
            [cosine + axis_x * axis_x * one_minus_cosine, axis_x * axis_y * one_minus_cosine - axis_z * sine, axis_x * axis_z * one_minus_cosine + axis_y * sine],
            [axis_y * axis_x * one_minus_cosine + axis_z * sine, cosine + axis_y * axis_y * one_minus_cosine, axis_y * axis_z * one_minus_cosine - axis_x * sine],
            [axis_z * axis_x * one_minus_cosine - axis_y * sine, axis_z * axis_y * one_minus_cosine + axis_x * sine, cosine + axis_z * axis_z * one_minus_cosine],
        ],
        dtype=np.float64,
    )


def _rigid_transform(translation: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


@dataclass(frozen=True)
class _UrdfJoint:
    origin: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


_FRANKA_CHAIN = (
    _UrdfJoint((0.0, 0.0, 0.333), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.0, 0.0, 0.0), (-np.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.0, -0.316, 0.0), (np.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.0825, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _UrdfJoint((-0.0825, 0.384, 0.0), (-np.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.0, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.088, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
)

_PIPER_CHAIN = (
    _UrdfJoint((0.0, 0.0, 0.123), (0.0, 0.0, -1.5708), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.0, 0.0, 0.0), (1.5708, 0.0, -1.5708), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.28358, 0.028726, 0.0), (0.0, 0.0, 0.10095), (0.0, 0.0, 1.0)),
    _UrdfJoint((-0.24221, 0.068514, 0.0), (-1.5708, 0.0, 1.3826), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.0, 0.0, 0.0), (1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.0, 0.091, 0.0014165), (-1.5708, -np.pi, 0.0), (0.0, 0.0, 1.0)),
)

_ARX_CHAIN = (
    _UrdfJoint((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    _UrdfJoint((-0.264, 0.0, 0.0), (np.pi, 0.0, 0.0), (0.0, 1.0, 0.0)),
    _UrdfJoint((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    _UrdfJoint((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.02895, 0.0, 0.0865), (-np.pi, 0.0, 0.0), (1.0, 0.0, 0.0)),
)

_ALOHA_CHAIN = (
    _UrdfJoint((0.0, 0.0, 0.058), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.025013, 0.00060169, 0.042), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    _UrdfJoint((-0.26396, 0.0044548, 0.0), (-3.1416, 0.0, -0.015928), (0.0, 1.0, 0.0)),
    _UrdfJoint((0.246, -0.00025, -0.06), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    _UrdfJoint((0.06775, 0.0015, -0.0855), (0.0, 0.0, -0.015928), (0.0, 0.0, 1.0)),
    _UrdfJoint((0.03095, 0.0, 0.0855), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)


def _chain_fk(joints: np.ndarray, chain: Sequence[_UrdfJoint], *, fixed_transform: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    transform = np.eye(4, dtype=np.float64)
    for angle, joint in zip(np.asarray(joints, dtype=np.float64), chain):
        transform = transform @ _rigid_transform(joint.origin, _rotation_from_rpy(joint.rpy))
        transform = transform @ _rigid_transform((0.0, 0.0, 0.0), _axis_angle(joint.axis, float(angle)))
    if fixed_transform is not None:
        transform = transform @ fixed_transform
    return transform[:3, 3], transform[:3, :3]


def _franka_fk(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Panda hand pose from seven joint positions using the Panda URDF chain."""

    fixed_hand = _rigid_transform((0.0, 0.0, 0.107), _rotation_z(-np.pi / 4.0))
    return _chain_fk(joints[:7], _FRANKA_CHAIN, fixed_transform=fixed_hand)


def _rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-float(rotation[2, 0]), -1.0, 1.0))
    cosine = np.cos(pitch)
    if abs(cosine) > 1e-8:
        roll = np.arctan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = np.arctan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = np.arctan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _default_arm_geometry(
    robot_type: str | None,
) -> tuple[tuple[float, float, float] | None, dict[str, np.ndarray]] | None:
    family = normalize_robot_type(robot_type)
    if family in {"piper", "arx-x5", "franka-panda"}:
        return None, {
            "left": _ROBOTWIN_ROOT_ROTATION.copy(),
            "right": _ROBOTWIN_ROOT_ROTATION.copy(),
        }
    if family != "aloha-agilex":
        return None
    left_origin = np.asarray((0.2305, 0.297, 0.782), dtype=np.float64)
    right_origin = np.asarray((0.2315, -0.3063, 0.781), dtype=np.float64)
    left_rotation = _ROBOTWIN_ROOT_ROTATION @ _rotation_from_rpy((0.0, 0.0, 0.02))
    right_rotation = _ROBOTWIN_ROOT_ROTATION @ _rotation_from_rpy((0.0, 0.0, 0.01))
    return (
        tuple(float(value) for value in _ROBOTWIN_ROOT_ROTATION @ (right_origin - left_origin)),
        {"left": left_rotation, "right": right_rotation},
    )


def _resolve_arm_geometry(
    robot_type: str | None,
    arm_base_offset: Any,
    arm_base_rotations: Any,
    *,
    bimanual: bool,
    dataset_origin: Any = None,
) -> tuple[tuple[float, float, float] | None, dict[str, np.ndarray] | None]:
    offset = normalize_arm_base_offset(arm_base_offset)
    rotations = normalize_arm_base_rotations(arm_base_rotations)
    is_simulation = str(dataset_origin or "").strip().lower() in {
        "simulation",
        "sim",
        "synthetic",
    }
    if is_simulation or normalize_robot_type(robot_type) == "aloha-agilex":
        default_geometry = _default_arm_geometry(robot_type)
        if default_geometry is not None:
            default_offset, default_rotations = default_geometry
            if bimanual and offset is None:
                offset = default_offset
            merged_rotations = dict(default_rotations)
            merged_rotations.update(rotations or {})
            rotations = merged_rotations
    elif (
        str(dataset_origin or "").strip().lower() in {"real", "hardware", "physical"}
        and normalize_robot_type(robot_type) == "piper"
        and bimanual
    ):
        merged_rotations = {
            "left": np.eye(3, dtype=np.float64),
            "right": REAL_PIPER_RIGHT_BASE_ROTATION.copy(),
        }
        merged_rotations.update(rotations or {})
        rotations = merged_rotations
    return offset, rotations


def _transform_eef_orientation(values: np.ndarray, base_rotation: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] not in {3, 6}:
        return values
    transformed = values.copy()
    for index, row in enumerate(values):
        if not np.isfinite(row).all():
            continue
        if values.shape[1] == 3:
            local_rotation = _rotation_from_rpy(row)
            transformed[index] = _rotation_to_rpy(base_rotation @ local_rotation)
        else:
            try:
                local_rotation = rotation6d_to_matrix(row)
            except ValueError:
                continue
            transformed[index] = matrix_to_rotation6d(base_rotation @ local_rotation)
    return transformed


def _apply_arm_base_offset(
    eef: dict[str, dict[str, np.ndarray]],
    arm_base_offset: Any,
    arm_base_rotations: Any = None,
    *,
    robot_type: str | None = None,
    dataset_origin: Any = None,
) -> tuple[dict[str, dict[str, np.ndarray]], tuple[float, float, float] | None, dict[str, np.ndarray] | None]:
    offset, rotations = _resolve_arm_geometry(
        robot_type,
        arm_base_offset,
        arm_base_rotations,
        bimanual={"left", "right"}.issubset(eef),
        dataset_origin=dataset_origin,
    )
    bimanual = {"left", "right"}.issubset(eef)
    origins = (
        {"left": (0.0, 0.0, 0.0), "right": offset or (0.0, 0.0, 0.0)}
        if bimanual
        else {side: (0.0, 0.0, 0.0) for side in eef}
    )
    for side, origin in origins.items():
        base_rotation = np.asarray(
            (rotations or {}).get(side, np.eye(3, dtype=np.float64)),
            dtype=np.float64,
        )
        positions = np.asarray(eef[side].get("position", []), dtype=np.float64)
        if positions.ndim == 2 and positions.shape[1] == 3:
            eef[side]["position"] = positions @ base_rotation.T + np.asarray(origin, dtype=np.float64)
        orientations = np.asarray(eef[side].get("orientation", []), dtype=np.float64)
        eef[side]["orientation"] = _transform_eef_orientation(orientations, base_rotation)
    return eef, offset, rotations


def compute_eef_trajectory(
    state: Any,
    *,
    names: Sequence[Any] | None = None,
    arm_side: str = "right",
    arm_base_offset: Any = None,
    arm_base_rotations: Any = None,
    robot_type: str | None = None,
    dataset_origin: Any = None,
) -> tuple[dict[str, dict[str, np.ndarray]], str]:
    """Return per-arm XYZ trajectories and the method used.

    Cartesian 10D/20D states are read directly.  Franka Panda 8D/16D joint
    states use the Panda URDF chain.  Piper, Aloha AgileX, and ARX-X5 7D/14D
    joint states use their RoboTwin URDF chains; Piper keeps the SDK position
    result when the SDK is available.
    """

    matrix = _as_matrix(state)
    width = matrix.shape[1]
    family = normalize_robot_type(robot_type)
    labels = tuple(str(item) for item in names) if names is not None else ()
    result: dict[str, dict[str, np.ndarray]] = {}
    if width in {10, 20} and any("eef_x" in label for label in labels):
        arms = 1 if width == 10 else 2
        sides = (arm_side if arm_side in {"left", "right"} else "right",) if arms == 1 else ("left", "right")
        for arm_index, side in enumerate(sides):
            block = matrix[:, arm_index * 10 : arm_index * 10 + 10]
            result[side] = {
                "position": block[:, :3],
                "orientation": block[:, 3:9],
            }
        transformed, _offset, _rotations = _apply_arm_base_offset(
            result,
            arm_base_offset,
            arm_base_rotations,
            robot_type=robot_type,
            dataset_origin=dataset_origin,
        )
        return transformed, "recorded_eef"

    if family not in {None, "franka-panda", "piper", "aloha-agilex", "arx-x5"}:
        return {}, "unavailable"
    if family == "franka-panda" or (family is None and width in {8, 16}):
        if width not in {8, 16}:
            return {}, "unavailable"
        arms = 1 if width == 8 else 2
        sides = (arm_side if arm_side in {"left", "right"} else "right",) if arms == 1 else ("left", "right")
        for arm_index, side in enumerate(sides):
            block = matrix[:, arm_index * 8 : arm_index * 8 + 7]
            positions = np.full((len(block), 3), np.nan, dtype=np.float64)
            orientations = np.full((len(block), 3), np.nan, dtype=np.float64)
            for index, joints in enumerate(block):
                if not np.all(np.isfinite(joints)):
                    continue
                positions[index], rotation = _franka_fk(joints)
                orientations[index] = _rotation_to_rpy(rotation)
            result[side] = {"position": positions, "orientation": orientations}
        transformed, _offset, _rotations = _apply_arm_base_offset(
            result,
            arm_base_offset,
            arm_base_rotations,
            robot_type=robot_type,
            dataset_origin=dataset_origin,
        )
        return transformed, "franka_panda_fk"

    if family == "franka-panda":
        return {}, "unavailable"
    if width not in {7, 14}:
        return {}, "unavailable"
    arms = 1 if width == 7 else 2
    sides = (arm_side if arm_side in {"left", "right"} else "right",) if arms == 1 else ("left", "right")
    chain = {
        "piper": _PIPER_CHAIN,
        "aloha-agilex": _ALOHA_CHAIN,
        "arx-x5": _ARX_CHAIN,
        None: _PIPER_CHAIN,
    }[family]
    sdk_fk = None
    if family in {None, "piper"}:
        try:
            from piper_sdk import C_PiperForwardKinematics

            sdk_fk = C_PiperForwardKinematics()
        except Exception:
            sdk_fk = None
    for arm_index, side in enumerate(sides):
        block = matrix[:, arm_index * 7 : arm_index * 7 + 6]
        positions = np.full((len(block), 3), np.nan, dtype=np.float64)
        orientations = np.full((len(block), 3), np.nan, dtype=np.float64)
        for index, joints in enumerate(block):
            if not np.all(np.isfinite(joints)):
                continue
            position, rotation = _chain_fk(joints, chain)
            try:
                if sdk_fk is not None:
                    position = np.asarray(sdk_fk.CalFK(joints.tolist())[-1], dtype=np.float64)[:3] / 1000.0
            except Exception:
                pass
            positions[index] = position
            orientations[index] = _rotation_to_rpy(rotation)
        result[side] = {"position": positions, "orientation": orientations}
    transformed, _offset, _rotations = _apply_arm_base_offset(
        result,
        arm_base_offset,
        arm_base_rotations,
        robot_type=robot_type,
        dataset_origin=dataset_origin,
    )
    if family == "piper" and sdk_fk is not None:
        return transformed, "piper_sdk_fk"
    return transformed, f"{family or 'piper'}_urdf_fk"


def analyze_episode(
    state: Any,
    action: Any = None,
    timestamps: Any = None,
    *,
    state_names: Sequence[Any] | None = None,
    action_names: Sequence[Any] | None = None,
    fps: float | int | None = None,
    arm_side: str = "right",
    arm_base_offset: Any = None,
    arm_base_rotations: Any = None,
    robot_type: str | None = None,
    dataset_origin: Any = None,
    velocity_threshold: float = 0.035,
    action_delta_threshold: float = 0.012,
    min_idle_run: int = 2,
) -> EpisodeAnalysis:
    """Build the shared analysis representation for one episode."""

    measured = _as_matrix(state)
    desired = measured.copy() if action is None else _as_matrix(action)
    count = min(len(measured), len(desired))
    measured, desired = measured[:count], desired[:count]
    time_axis = _safe_time_axis(timestamps, count, fps)
    velocities = compute_joint_velocities(measured, time_axis, fps=fps)
    anomalies = detect_motion_anomalies(measured, time_axis, fps=fps)
    idle, speed_norm, _command_delta = detect_idle_frames(
        measured,
        action=desired,
        timestamps=time_axis,
        fps=fps,
        velocity_threshold=velocity_threshold,
        action_delta_threshold=action_delta_threshold,
        min_run=min_idle_run,
    )
    effective_offset, effective_rotations = _resolve_arm_geometry(
        robot_type,
        arm_base_offset,
        arm_base_rotations,
        bimanual=measured.shape[1] in {14, 16, 20},
        dataset_origin=dataset_origin,
    )
    eef, eef_method = compute_eef_trajectory(
        measured,
        names=state_names,
        arm_side=arm_side,
        arm_base_offset=effective_offset,
        arm_base_rotations=effective_rotations,
        robot_type=robot_type,
        dataset_origin=dataset_origin,
    )
    inferred_fps = float(fps or (1.0 / np.median(np.diff(time_axis)) if count > 1 else 20.0))
    return EpisodeAnalysis(
        timestamps=time_axis,
        state=measured,
        action=desired,
        velocities=velocities,
        accelerations=anomalies["accelerations"],
        speed_norm=speed_norm,
        idle=idle,
        velocity_reversal=anomalies["velocity_reversal"],
        jitter=anomalies["jitter"],
        abrupt_change=anomalies["abrupt_change"],
        anomaly_score=anomalies["anomaly_score"],
        jerk_norm=anomalies["jerk_norm"],
        joint_names=infer_joint_names(
            state_names,
            measured.shape[1],
            arm_side=arm_side,
            robot_type=robot_type,
        ),
        eef=eef,
        eef_method=eef_method,
        fps=inferred_fps,
        arm_base_offset=effective_offset,
        arm_base_rotations=effective_rotations,
        robot_type=normalize_robot_type(robot_type),
    )


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_array(value: Any) -> list[Any]:
    array = np.asarray(value)
    if array.ndim == 0:
        return [_json_number(array.item())]
    return [_json_array(item) if np.asarray(item).ndim else _json_number(np.asarray(item).item()) for item in array]


def frame_payload(analysis: EpisodeAnalysis, frame_index: int) -> dict[str, Any]:
    """Serialize one frame for a GUI table."""

    if analysis.frame_count <= 0:
        raise IndexError("episode has no frames")
    index = max(0, min(int(frame_index), analysis.frame_count - 1))
    joints = []
    for joint_index, name in enumerate(analysis.joint_names):
        joints.append(
            {
                "index": joint_index,
                "name": name,
                "angle": _json_number(analysis.state[index, joint_index]),
                "velocity": _json_number(analysis.velocities[index, joint_index]),
                "acceleration": _json_number(analysis.accelerations[index, joint_index]),
                "target": _json_number(analysis.action[index, joint_index]) if joint_index < analysis.action.shape[1] else None,
                "velocity_reversal": bool(analysis.velocity_reversal[index, joint_index]),
                "jitter": bool(analysis.jitter[index, joint_index]),
            }
        )
    eef: dict[str, dict[str, list[float | None]]] = {}
    for side, item in analysis.eef.items():
        side_payload: dict[str, list[float | None]] = {}
        for key in ("position", "orientation"):
            values = np.asarray(item.get(key, []))
            row = values[index].reshape(-1) if values.ndim >= 2 and len(values) > index else np.asarray([])
            side_payload[key] = [_json_number(value) for value in row]
        eef[side] = side_payload
    return {
        "frame_index": index,
        "timestamp": _json_number(analysis.timestamps[index]),
        "relative_time_s": _json_number(analysis.timestamps[index] - analysis.timestamps[0]),
        "idle": bool(analysis.idle[index]),
        "speed_norm": _json_number(analysis.speed_norm[index]),
        "acceleration_norm": _json_number(np.linalg.norm(np.nan_to_num(analysis.accelerations[index, : min(6, analysis.state.shape[1])]))),
        "jerk_norm": _json_number(analysis.jerk_norm[index]),
        "anomaly": bool(analysis.abrupt_change[index] or np.any(analysis.velocity_reversal[index]) or np.any(analysis.jitter[index])),
        "abrupt_change": bool(analysis.abrupt_change[index]),
        "velocity_reversal": bool(np.any(analysis.velocity_reversal[index])),
        "jitter": bool(np.any(analysis.jitter[index])),
        "anomaly_score": _json_number(analysis.anomaly_score[index]),
        "anomaly_types": [
            label
            for label, present in (
                ("jitter", bool(np.any(analysis.jitter[index]))),
                ("velocity_reversal", bool(np.any(analysis.velocity_reversal[index]))),
                ("abrupt_change", bool(analysis.abrupt_change[index])),
            )
            if present
        ],
        "joints": joints,
        "eef": eef,
        "arm_base_offset": list(analysis.arm_base_offset) if analysis.arm_base_offset is not None else None,
        "arm_base_rotations": arm_base_rotation_metadata(analysis.arm_base_rotations),
        "robot_type": analysis.robot_type,
        "arm_axis_convention": (
            ARM_BASE_AXIS_CONVENTION
            if analysis.arm_base_offset is not None or {"left", "right"}.issubset(analysis.eef)
            else None
        ),
        "arm_axis_signs": arm_base_axis_signs(
            analysis.arm_base_offset,
            bimanual={"left", "right"}.issubset(analysis.eef),
        ),
        "arm_origins": arm_base_origins(analysis.arm_base_offset),
    }


def analysis_payload(analysis: EpisodeAnalysis, *, max_points: int = 1200) -> dict[str, Any]:
    """Serialize a compact timeline suitable for a browser chart."""

    count = analysis.frame_count
    if count <= 0:
        return {
            "frame_count": 0,
            "idle_runs": [],
            "suggested_crop": suggested_crop(analysis.idle),
            "frames": [],
            "arm_base_offset": list(analysis.arm_base_offset) if analysis.arm_base_offset is not None else None,
            "arm_base_rotations": arm_base_rotation_metadata(analysis.arm_base_rotations),
            "robot_type": analysis.robot_type,
            "arm_axis_convention": (
                ARM_BASE_AXIS_CONVENTION
                if analysis.arm_base_offset is not None or {"left", "right"}.issubset(analysis.eef)
                else None
            ),
            "arm_axis_signs": arm_base_axis_signs(
                analysis.arm_base_offset,
                bimanual={"left", "right"}.issubset(analysis.eef),
            ),
            "arm_origins": arm_base_origins(analysis.arm_base_offset),
        }
    stride = max(1, int(np.ceil(count / max(1, int(max_points)))))
    indexes = np.arange(0, count, stride, dtype=np.int64)
    if indexes[-1] != count - 1:
        indexes = np.r_[indexes, count - 1]
    trajectory = {
        side: {
            "position": _json_array(np.asarray(item.get("position", []))[indexes]),
            "orientation": _json_array(np.asarray(item.get("orientation", []))[indexes]),
        }
        for side, item in analysis.eef.items()
    }
    return {
        "frame_count": count,
        "fps": analysis.fps,
        "duration_s": float(analysis.timestamps[-1] - analysis.timestamps[0]) if count > 1 else 0.0,
        "joint_names": list(analysis.joint_names),
        "eef_method": analysis.eef_method,
        "arm_base_offset": list(analysis.arm_base_offset) if analysis.arm_base_offset is not None else None,
        "arm_base_rotations": arm_base_rotation_metadata(analysis.arm_base_rotations),
        "robot_type": analysis.robot_type,
        "arm_axis_convention": (
            ARM_BASE_AXIS_CONVENTION
            if analysis.arm_base_offset is not None or {"left", "right"}.issubset(analysis.eef)
            else None
        ),
        "arm_axis_signs": arm_base_axis_signs(
            analysis.arm_base_offset,
            bimanual={"left", "right"}.issubset(analysis.eef),
        ),
        "arm_origins": arm_base_origins(analysis.arm_base_offset),
        "idle_frames": int(np.count_nonzero(analysis.idle)),
        "idle_fraction": float(np.mean(analysis.idle)),
        "anomaly_frames": int(np.count_nonzero(analysis.abrupt_change | np.any(analysis.velocity_reversal, axis=1) | np.any(analysis.jitter, axis=1))),
        "jitter_frames": int(np.count_nonzero(np.any(analysis.jitter, axis=1))),
        "velocity_reversal_frames": int(np.count_nonzero(np.any(analysis.velocity_reversal, axis=1))),
        "abrupt_change_frames": int(np.count_nonzero(analysis.abrupt_change)),
        "idle_runs": idle_runs(analysis.idle),
        "suggested_crop": suggested_crop(analysis.idle),
        "sample_stride": stride,
        "sampled_frames": indexes.tolist(),
        "timeline": {
            "relative_time_s": _json_array(analysis.timestamps[indexes] - analysis.timestamps[0]),
            "speed_norm": _json_array(analysis.speed_norm[indexes]),
            "acceleration_norm": _json_array(
                np.linalg.norm(np.nan_to_num(analysis.accelerations[indexes, : min(6, analysis.state.shape[1])]), axis=1)
            ),
            "jerk_norm": _json_array(analysis.jerk_norm[indexes]),
            "idle": analysis.idle[indexes].astype(bool).tolist(),
            "anomaly": (analysis.abrupt_change[indexes] | np.any(analysis.velocity_reversal[indexes], axis=1) | np.any(analysis.jitter[indexes], axis=1)).tolist(),
            "jitter": np.any(analysis.jitter[indexes], axis=1).tolist(),
            "velocity_reversal": np.any(analysis.velocity_reversal[indexes], axis=1).tolist(),
            "abrupt_change": analysis.abrupt_change[indexes].tolist(),
        },
        "eef": trajectory,
    }


__all__ = [
    "EpisodeAnalysis",
    "analysis_payload",
    "analyze_episode",
    "compute_eef_trajectory",
    "compute_joint_velocities",
    "compute_joint_accelerations",
    "detect_motion_anomalies",
    "detect_idle_frames",
    "frame_payload",
    "idle_runs",
    "infer_joint_names",
    "suggested_crop",
]
