#!/usr/bin/env python3
"""Run the official OpenPI client attached to one or two Piper arms.

The client is fail-closed and follows validated single-arm/bimanual ``delivery``
or ``joint`` policy metadata. By default it only sends real observations
and prints predictions. Robot motion requires both a time-limited Dashboard
``execute`` authorization and the local ``--allow-execution`` flag. Robot
control and camera acquisition run continuously at 20 Hz while a single
asynchronous policy request is launched every 250 ms (4 Hz) when the previous
request has completed. A 50-row OpenPI chunk must contain at least 16 rows.
Every decoded row is timestamped from the observation's monotonic capture time;
each control tick selects the closest future target for its estimated actuator
execution time, then blends a new pose trajectory into the still-active plan
over 2--4 steps (default 3). The gripper is filtered separately and is never
pose-blended. If a plan runs out under a valid double gate, the last safe target
is held until a valid replacement arrives. Every command passes schema-specific
freshness, range, delta, and Piper-status checks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import logging
import math
import os
import re
import socket
import time
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from camera import CameraCapture
from collect_output_arm import require_can_interface_up
from piper_action_conventions import (
    DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS,
    DELIVERY_MODEL_ACTION_SEMANTICS,
    DELIVERY_STEP_ACTION_SEMANTICS,
    JOINT_ACTION_SEMANTICS as WIRE_JOINT_ACTION_SEMANTICS,
    LEGACY_GRIPPER_SEMANTICS,
    NEW_GRIPPER_SEMANTICS,
    chunk_origin_deltas_to_absolute_eef_targets,
    matrix_to_rotation6d,
    step_deltas_to_chunk_origin,
)

from piper_data_contract import (
    CONTRACT_VERSION,
    GRIPPER_MAX_M,
    IMAGE_HW,
    LEGACY_GRIPPER_OPENING_METRES_SEMANTICS,
    STATE_NAMES,
    build_delivery_state,
)


RAD_FACTOR = 57295.7795  # Piper unit: 0.001 degree -> rad
GRIPPER_FACTOR = 1_000_000.0  # Piper unit: 0.001 mm -> metre
JOINT_LIMITS_RAD = np.array(
    [
        (-2.6179, 2.6179),
        (0.0000, 3.1400),
        (-2.9670, 0.0000),
        (-1.7450, 1.7450),
        (-1.2200, 1.2200),
        (-2.0944, 2.0944),
    ],
    dtype=np.float64,
)
JOINT_ACTION_SEMANTICS = frozenset(
    {"absolute_joint_position", "absolute_next_joint_position", WIRE_JOINT_ACTION_SEMANTICS}
)
DEFAULT_POLICY_HOST = "192.168.101.9"
DEFAULT_POLICY_PORT = 8000
DEFAULT_ACTION_HZ = 20.0
DEFAULT_INFERENCE_HZ = 4.0
DEFAULT_CAMERA_FPS = 20
DEFAULT_OPENPI_CHUNK_STEPS = 50
DEFAULT_MIN_ACTION_CHUNK_STEPS = 16
DEFAULT_BLEND_STEPS = 3
DEFAULT_ACTUATOR_DELAY_S = 0.0
DEFAULT_GRIPPER_LOWPASS_ALPHA = 0.5
DEFAULT_GRIPPER_HYSTERESIS = 0.05
DEFAULT_GRIPPER_CONFIRM_STEPS = 2
DEFAULT_FEEDBACK_MAX_AGE_S = 0.5
GRIPPER_OPENING_FRACTION = NEW_GRIPPER_SEMANTICS
GRIPPER_CLOSED_FRACTION = LEGACY_GRIPPER_SEMANTICS
GRIPPER_OPENING_METRES = LEGACY_GRIPPER_OPENING_METRES_SEMANTICS
DEFAULT_CAN = "can0"
DEFAULT_LEFT_CAN = "can1"
DEFAULT_RIGHT_CAN = "can3"
DEFAULT_HIGH_DEVICE = "auto"
DEFAULT_WRIST_DEVICE = "auto"
DEFAULT_LEFT_WRIST_DEVICE = "auto"
DEFAULT_RIGHT_WRIST_DEVICE = "auto"
CAMERA_SOURCE_HW = (240, 424)
# 8_3_64eps full-set envelope: 18,034 frames sampled at 20 Hz. These defaults
# include a small margin over observed maxima. They remain CLI-tightenable, and
# every blended target still passes the same per-step checks before execution.
SAFETY_PROFILE = "8_3_64eps_18034_frames_20hz"
DEFAULT_MAX_TRANSLATION_STEP_M = 0.05  # observed max 0.04830 m
DEFAULT_MAX_ROTATION_STEP_RAD = 0.18  # observed max 0.15766 rad
DEFAULT_MAX_GRIPPER_STEP = 0.30  # observed max 0.261 opening fraction
DEFAULT_WORKSPACE_X_M = (-0.05, 0.30)  # observed [-0.03815, 0.27987]
DEFAULT_WORKSPACE_Y_M = (0.01, 0.50)  # observed [0.02183, 0.47802]
DEFAULT_WORKSPACE_Z_M = (0.14, 0.52)  # observed [0.14706, 0.50322]
DEFAULT_GRIPPER_RANGE_TOLERANCE = 0.02
PIPER_FEEDBACK_MAX_AGE_S = 0.5
IK_JOINT_LIMIT_MARGIN_RAD = 0.002
# Piper feedback can sit a few degrees beyond the SDK's nominal zero-angle
# limits while the controller reports a healthy, non-limit status.  This is a
# feedback/calibration tolerance only: local IK may start from that measured
# pose, but it may not move farther out past it.
IK_FEEDBACK_LIMIT_TOLERANCE_RAD = 0.06
DEFAULT_IK_MAX_JOINT_STEP_RAD = 0.08
DEFAULT_IK_POSITION_TOLERANCE_M = 0.0015
DEFAULT_IK_ROTATION_TOLERANCE_RAD = 0.02
DEFAULT_IK_MAX_NFEV = 100


class ExecutionBlocked(RuntimeError):
    """The action was rejected before a robot command was sent."""


class PiperFeedbackStaleError(ExecutionBlocked):
    """Piper SDK getters contain missing or cached CAN feedback."""


@dataclass(frozen=True)
class PolicyProtocol:
    """Validated observation/action contract advertised by one policy server."""

    schema: str
    state_dim: int
    action_dim: int
    arm_side: str
    action_semantics: str
    camera_keys: tuple[str, ...]
    arm_mode: str = "single"
    # Dataset/action sampling frequency. Older servers may omit it.
    action_hz: float | None = None
    # Old 8_3_64eps delivery checkpoints use closed fraction; v3 and joint
    # policies use opening fraction. The policy metadata selects the branch.
    gripper_semantics: str = GRIPPER_OPENING_FRACTION
    state_gripper_semantics: str = GRIPPER_OPENING_FRACTION
    metadata_gripper_semantics_explicit: bool = False
    contract_version: int | None = None
    action_horizon: int = DEFAULT_OPENPI_CHUNK_STEPS


def connect_piper(can_name: str) -> Any:
    """Connect for feedback; this alone does not enable or command the arm."""
    from piper_sdk import C_PiperInterface_V2

    require_can_interface_up(can_name)
    piper = C_PiperInterface_V2(can_name, judge_flag=False, can_auto_init=False)
    piper.CreateCanBus(
        can_name=can_name,
        bustype="socketcan",
        expected_bitrate=1_000_000,
        judge_flag=False,
    )
    piper.ConnectPort(can_init=True, piper_init=True)
    time.sleep(0.5)
    return piper


def _qpos_from_feedback(joints_message: Any, gripper_message: Any) -> np.ndarray:
    joints = joints_message.joint_state
    gripper = gripper_message.gripper_state
    values = np.array(
        [
            joints.joint_1,
            joints.joint_2,
            joints.joint_3,
            joints.joint_4,
            joints.joint_5,
            joints.joint_6,
        ],
        dtype=np.float32,
    ) / RAD_FACTOR
    return np.append(values, float(gripper.grippers_angle) / GRIPPER_FACTOR).astype(np.float32)


def _require_fresh_feedback(
    messages: dict[str, Any],
    *,
    max_age_s: float | None = PIPER_FEEDBACK_MAX_AGE_S,
) -> None:
    if max_age_s is None:
        return
    now = time.time()
    failures = []
    for name, message in messages.items():
        timestamp = float(getattr(message, "time_stamp", 0.0) or 0.0)
        hz = float(getattr(message, "Hz", 0.0) or 0.0)
        age_s = now - timestamp if timestamp > 0 else float("inf")
        if timestamp <= 0 or age_s > max_age_s or age_s < -1.0:
            failures.append(f"{name}: age={age_s:.3f}s Hz={hz:.1f}")
    if failures:
        raise PiperFeedbackStaleError(
            "Piper CAN feedback is missing or stale: " + "; ".join(failures)
        )


def read_output_qpos(
    piper: Any,
    *,
    max_feedback_age_s: float | None = PIPER_FEEDBACK_MAX_AGE_S,
) -> np.ndarray:
    """Read measured joints/gripper in physical units (radians/metres)."""
    joints_message = piper.GetArmJointMsgs()
    gripper_message = piper.GetArmGripperMsgs()
    _require_fresh_feedback(
        {"joint": joints_message, "gripper": gripper_message},
        max_age_s=max_feedback_age_s,
    )
    return _qpos_from_feedback(joints_message, gripper_message)


def rotation_from_state(state: np.ndarray) -> np.ndarray:
    """Recover an orthonormal rotation matrix from the delivery rotation6d."""
    c0 = np.asarray(state[3:6], dtype=np.float64)
    c1 = np.asarray(state[6:9], dtype=np.float64)
    norm0 = float(np.linalg.norm(c0))
    if norm0 < 1e-6:
        raise ExecutionBlocked("invalid current rotation6d first column")
    c0 /= norm0
    c1 -= c0 * float(np.dot(c0, c1))
    norm1 = float(np.linalg.norm(c1))
    if norm1 < 1e-6:
        raise ExecutionBlocked("invalid current rotation6d second column")
    c1 /= norm1
    return np.column_stack((c0, c1, np.cross(c0, c1)))


class PiperContinuousIK:
    """Numerical Piper IK constrained to the branch near current feedback."""

    def __init__(self, fk: Any | None = None) -> None:
        if fk is None:
            from piper_sdk import C_PiperForwardKinematics

            fk = C_PiperForwardKinematics()
        self._fk = fk

    def pose(self, joints_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(joints_rad, dtype=np.float64)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ExecutionBlocked(f"IK joints must be finite 6D, got {joints.shape}")
        pose = np.asarray(self._fk.CalFK(joints.tolist())[-1], dtype=np.float64)
        xyz_m = pose[:3] / 1000.0
        rotation = Rotation.from_euler("xyz", pose[3:6], degrees=True).as_matrix()
        return xyz_m, rotation

    def solve(
        self,
        current_joints_rad: np.ndarray,
        target_xyz_m: np.ndarray,
        target_rpy_deg: np.ndarray,
        *,
        max_joint_step_rad: float,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        max_nfev: int,
    ) -> np.ndarray:
        current = np.asarray(current_joints_rad, dtype=np.float64)
        target_xyz = np.asarray(target_xyz_m, dtype=np.float64)
        target_rpy = np.asarray(target_rpy_deg, dtype=np.float64)
        if current.shape != (6,) or not np.all(np.isfinite(current)):
            raise ExecutionBlocked("current IK joints are not finite 6D")
        if target_xyz.shape != (3,) or not np.all(np.isfinite(target_xyz)):
            raise ExecutionBlocked("IK target xyz is not finite 3D")
        if target_rpy.shape != (3,) or not np.all(np.isfinite(target_rpy)):
            raise ExecutionBlocked("IK target rpy is not finite 3D")

        hard_lower = JOINT_LIMITS_RAD[:, 0] + IK_JOINT_LIMIT_MARGIN_RAD
        hard_upper = JOINT_LIMITS_RAD[:, 1] - IK_JOINT_LIMIT_MARGIN_RAD
        if np.any(current < hard_lower - IK_FEEDBACK_LIMIT_TOLERANCE_RAD) or np.any(
            current > hard_upper + IK_FEEDBACK_LIMIT_TOLERANCE_RAD
        ):
            raise ExecutionBlocked("current joints are too far outside IK limits")
        # Keep the measured pose itself in the numerical interval when a joint
        # is just beyond a nominal zero limit.  The interval only extends from
        # that measured value back toward the nominal range, so IK cannot drive
        # an already-outside joint farther outward.
        feedback_lower = np.minimum(hard_lower, current)
        feedback_upper = np.maximum(hard_upper, current)
        lower = np.maximum(feedback_lower, current - max_joint_step_rad)
        upper = np.minimum(feedback_upper, current + max_joint_step_rad)
        if np.any(lower >= upper):
            raise ExecutionBlocked("continuous IK has no valid local joint interval")
        initial = np.clip(current, lower + 1e-8, upper - 1e-8)
        target_rotation = Rotation.from_euler("xyz", target_rpy, degrees=True).as_matrix()

        def residual(candidate: np.ndarray) -> np.ndarray:
            xyz, rotation = self.pose(candidate)
            rotation_error = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
            return np.concatenate(
                (
                    (xyz - target_xyz) / position_tolerance_m,
                    rotation_error / rotation_tolerance_rad,
                )
            )

        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        solved = np.asarray(result.x, dtype=np.float64)
        solved_xyz, solved_rotation = self.pose(solved)
        position_error = float(np.linalg.norm(solved_xyz - target_xyz))
        rotation_error = float(
            np.linalg.norm(Rotation.from_matrix(target_rotation @ solved_rotation.T).as_rotvec())
        )
        joint_step = float(np.max(np.abs(solved - current)))
        if np.any(solved < hard_lower - IK_FEEDBACK_LIMIT_TOLERANCE_RAD) or np.any(
            solved > hard_upper + IK_FEEDBACK_LIMIT_TOLERANCE_RAD
        ):
            raise ExecutionBlocked("continuous IK target is outside tolerated joint limits")
        if position_error > position_tolerance_m or rotation_error > rotation_tolerance_rad:
            raise ExecutionBlocked(
                "continuous IK could not reach a nearby solution: "
                f"position_error={position_error:.5f}m, "
                f"rotation_error={rotation_error:.5f}rad, "
                f"max_joint_step={joint_step:.5f}rad"
            )
        if joint_step > max_joint_step_rad + 1e-6:
            raise ExecutionBlocked(
                f"continuous IK joint step {joint_step:.5f}rad exceeds "
                f"{max_joint_step_rad:.5f}rad"
            )
        return solved




def read_output_state(
    piper: Any,
    *,
    max_feedback_age_s: float | None = PIPER_FEEDBACK_MAX_AGE_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical feedback as v3 opening-fraction delivery state and qpos.

    ``policy_observation_state`` applies the policy-advertised gripper
    convention before inference and action decoding.
    """
    joints_message = piper.GetArmJointMsgs()
    gripper_message = piper.GetArmGripperMsgs()
    pose_message = piper.GetArmEndPoseMsgs()
    _require_fresh_feedback(
        {"joint": joints_message, "gripper": gripper_message, "end_pose": pose_message},
        max_age_s=max_feedback_age_s,
    )
    qpos = _qpos_from_feedback(joints_message, gripper_message)
    pose = pose_message.end_pose
    xyz_m = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64) / 1_000_000.0
    rpy_rad = np.deg2rad(
        np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64) / 1000.0
    )
    rotation = Rotation.from_euler("xyz", rpy_rad).as_matrix()
    return build_delivery_state(xyz_m, rotation, float(qpos[6])), qpos


def _canonical_gripper_semantics(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    aliases = {
        GRIPPER_OPENING_FRACTION: GRIPPER_OPENING_FRACTION,
        "absolute_opening_fraction": GRIPPER_OPENING_FRACTION,
        "opening_fraction": GRIPPER_OPENING_FRACTION,
        GRIPPER_CLOSED_FRACTION: GRIPPER_CLOSED_FRACTION,
        "absolute_closed_fraction": GRIPPER_CLOSED_FRACTION,
        "closed_fraction": GRIPPER_CLOSED_FRACTION,
        GRIPPER_OPENING_METRES: GRIPPER_OPENING_METRES,
        "absolute_opening_meters": GRIPPER_OPENING_METRES,
        "opening_m": GRIPPER_OPENING_METRES,
    }
    return aliases.get(normalized)


def policy_observation_state(
    raw_delivery_state: np.ndarray,
    qpos_m: np.ndarray,
    protocol: PolicyProtocol,
) -> np.ndarray:
    """Adapt physical feedback to the state convention advertised by policy."""
    if protocol.schema == "delivery":
        state = np.asarray(raw_delivery_state, dtype=np.float32).copy()
        for arm_index in range(2 if protocol.arm_mode == "bimanual" else 1):
            index = arm_index * 10 + 9
            if protocol.state_gripper_semantics == GRIPPER_CLOSED_FRACTION:
                state[index] = 1.0 - state[index]
        return state
    state = np.asarray(qpos_m, dtype=np.float32).copy()
    for arm_index in range(2 if protocol.arm_mode == "bimanual" else 1):
        index = arm_index * 7 + 6
        if protocol.state_gripper_semantics == GRIPPER_OPENING_FRACTION:
            state[index] = state[index] / GRIPPER_MAX_M
    return state


def arm_status_dict(piper: Any) -> dict[str, Any]:
    feedback = piper.GetArmStatus().arm_status
    return {
        "ctrl_mode": int(feedback.ctrl_mode),
        "arm_status": int(feedback.arm_status),
        "mode_feed": int(feedback.mode_feed),
        "motion_status": int(feedback.motion_status),
        "err_code": int(feedback.err_code),
    }


def _metadata_contract_version(metadata: dict[str, Any], errors: list[str]) -> int | None:
    raw = metadata.get("contract_version")
    if raw is None:
        return None
    try:
        version = int(raw)
    except (TypeError, ValueError):
        errors.append(f"contract_version={raw!r} must be an integer")
        return None
    if version <= 0:
        errors.append(f"contract_version={raw!r} must be positive")
        return None
    return version


def _joint_gripper_semantics_from_metadata(
    metadata: dict[str, Any],
    action_semantics: Any,
    errors: list[str],
) -> tuple[str | None, bool]:
    raw = (
        metadata.get("wire_gripper_semantics")
        or metadata.get("model_gripper_semantics")
        or metadata.get("gripper_semantics")
    )
    explicit = raw is not None
    semantics = _canonical_gripper_semantics(raw)
    if explicit and semantics is None:
        errors.append(f"unsupported gripper_semantics={raw!r}")
        return None, True
    if semantics is not None:
        return semantics, True
    if action_semantics == WIRE_JOINT_ACTION_SEMANTICS:
        return GRIPPER_OPENING_FRACTION, False

    names: list[str] = []
    for key in ("wire_action_names", "model_action_names", "action_names"):
        value = metadata.get(key)
        if isinstance(value, (list, tuple)):
            names.extend(str(item).lower() for item in value)
    if any("gripper_opening_fraction" in name for name in names):
        return GRIPPER_OPENING_FRACTION, False
    if any("gripper_opening_m" in name or "gripper_opening_metre" in name for name in names):
        return GRIPPER_OPENING_METRES, False

    raw_version = metadata.get("contract_version")
    try:
        version = int(raw_version) if raw_version is not None else None
    except (TypeError, ValueError):
        version = None
    if version is not None:
        return (
            GRIPPER_OPENING_FRACTION
            if version >= CONTRACT_VERSION
            else GRIPPER_OPENING_METRES
        ), False

    errors.append(
        "legacy joint checkpoint omits gripper_semantics and has no decisive "
        "contract_version/action_names; refusing to guess metres versus fraction"
    )
    return None, False


def _joint_state_gripper_semantics_from_metadata(
    metadata: dict[str, Any],
    action_semantics: Any,
    errors: list[str],
) -> str | None:
    raw = metadata.get("state_gripper_semantics") or metadata.get(
        "raw_gripper_semantics"
    )
    semantics = _canonical_gripper_semantics(raw)
    if raw is not None:
        if semantics is None:
            errors.append(f"unsupported state_gripper_semantics={raw!r}")
        return semantics
    names = metadata.get("state_names")
    if isinstance(names, (list, tuple)):
        lowered = [str(item).lower() for item in names]
        if any("gripper_opening_fraction" in name for name in lowered):
            return GRIPPER_OPENING_FRACTION
        if any("gripper_opening_m" in name or "gripper_opening_metre" in name for name in lowered):
            return GRIPPER_OPENING_METRES
    if not metadata.get("wire_gripper_semantics") and not metadata.get(
        "model_gripper_semantics"
    ):
        action_names = metadata.get("action_names")
        if isinstance(action_names, (list, tuple)):
            lowered = [str(item).lower() for item in action_names]
            if any("gripper_opening_fraction" in name for name in lowered):
                return GRIPPER_OPENING_FRACTION
            if any(
                "gripper_opening_m" in name or "gripper_opening_metre" in name
                for name in lowered
            ):
                return GRIPPER_OPENING_METRES
    try:
        version = int(metadata.get("contract_version"))
    except (TypeError, ValueError):
        version = None
    if version is not None:
        return GRIPPER_OPENING_FRACTION if version >= CONTRACT_VERSION else GRIPPER_OPENING_METRES
    if action_semantics == WIRE_JOINT_ACTION_SEMANTICS:
        return GRIPPER_OPENING_FRACTION
    errors.append(
        "joint state omits state_gripper_semantics and has no decisive "
        "contract_version/state_names; refusing to guess metres versus fraction"
    )
    return None


def validate_policy_metadata(
    metadata: dict[str, Any],
    arm_side: str,
    arm_mode: str = "single",
) -> PolicyProtocol:
    """Validate server dimensions/cameras before any robot command is possible."""
    advertised_mode = str(metadata.get("arm_mode") or "single")
    expected_side = "both" if arm_mode == "bimanual" else arm_side
    expected_action_dim = 14 if arm_mode == "bimanual" else 7
    advertised_action_dim = (
        metadata.get("wire_action_dim")
        or metadata.get("model_action_dim")
        or metadata.get("action_dim")
    )
    common_expected = {
        "transport": "openpi_websocket_v1",
        "arm_mode": arm_mode,
        "action_dim": expected_action_dim,
        "arm_side": expected_side,
    }
    comparable = dict(metadata, arm_mode=advertised_mode, action_dim=advertised_action_dim)
    errors = [
        f"{key}={comparable.get(key)!r}, expected {value!r}"
        for key, value in common_expected.items()
        if comparable.get(key) != value
    ]

    schema = metadata.get("schema")
    advertised_action_semantics = (
        metadata.get("wire_action_semantics")
        or metadata.get("model_action_semantics")
        or metadata.get("action_semantics")
    )
    if schema == "delivery":
        expected_state_dim = 20 if arm_mode == "bimanual" else 10
        expected_semantics = {
            DELIVERY_STEP_ACTION_SEMANTICS,
            DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS,
            DELIVERY_MODEL_ACTION_SEMANTICS,
        }
    elif schema == "joint":
        expected_state_dim = 14 if arm_mode == "bimanual" else 7
        expected_semantics = JOINT_ACTION_SEMANTICS
    else:
        expected_state_dim = None
        expected_semantics = set()
        errors.append(f"schema={schema!r}, expected 'delivery' or 'joint'")

    if arm_mode == "bimanual":
        expected_camera_key_sets = (
            {"cam_high", "cam_left_wrist", "cam_right_wrist"},
        )
    elif schema == "delivery":
        # Legacy delivery datasets expose the generic ``cam_wrist`` alias,
        # while canonical v3 datasets preserve the physical arm side in the
        # wire key. Both identify the same locally captured wrist stream.
        expected_camera_key_sets = (
            {"cam_high", "cam_wrist"},
            {"cam_high", f"cam_{arm_side}_wrist"},
        )
    else:
        expected_camera_key_sets = (
            {"cam_high", f"cam_{arm_side}_wrist"},
        )

    if expected_state_dim is not None and metadata.get("state_dim") != expected_state_dim:
        errors.append(f"state_dim={metadata.get('state_dim')!r}, expected {expected_state_dim!r}")
    action_semantics = advertised_action_semantics
    if expected_semantics and advertised_action_semantics not in expected_semantics:
        errors.append(
            f"wire/action_semantics={advertised_action_semantics!r}, expected one of "
            f"{sorted(expected_semantics)!r}"
        )
    if schema == "delivery" and advertised_action_semantics == DELIVERY_STEP_ACTION_SEMANTICS:
        expected_gripper_semantics = GRIPPER_CLOSED_FRACTION
        expected_delivery_convention = "step"
        action_semantics = DELIVERY_STEP_ACTION_SEMANTICS
        raw_gripper_semantics = (
            metadata.get("wire_gripper_semantics")
            or metadata.get("model_gripper_semantics")
            or metadata.get("gripper_semantics")
        )
        gripper_semantics = _canonical_gripper_semantics(raw_gripper_semantics)
    elif schema == "delivery":
        expected_gripper_semantics = (
            GRIPPER_OPENING_FRACTION
            if advertised_action_semantics == DELIVERY_MODEL_ACTION_SEMANTICS
            else None
        )
        expected_delivery_convention = "chunk_origin"
        action_semantics = DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS
        raw_gripper_semantics = (
            metadata.get("wire_gripper_semantics")
            or metadata.get("model_gripper_semantics")
            or metadata.get("gripper_semantics")
        )
        gripper_semantics = _canonical_gripper_semantics(raw_gripper_semantics)
    else:
        expected_gripper_semantics = GRIPPER_OPENING_FRACTION
        expected_delivery_convention = None
        raw_gripper_semantics = (
            metadata.get("wire_gripper_semantics")
            or metadata.get("model_gripper_semantics")
            or metadata.get("gripper_semantics")
        )
        gripper_semantics, _ = _joint_gripper_semantics_from_metadata(
            metadata, advertised_action_semantics, errors
        )
    if raw_gripper_semantics is not None and gripper_semantics is None:
        errors.append(
            f"gripper_semantics={raw_gripper_semantics!r}, expected opening_fraction "
            "or closed_fraction"
        )
    elif (
        schema == "delivery"
        and expected_gripper_semantics is not None
        and gripper_semantics is not None
        and gripper_semantics != expected_gripper_semantics
    ):
        errors.append(
            f"gripper_semantics={gripper_semantics!r} conflicts with "
            f"action_semantics={action_semantics!r}; expected {expected_gripper_semantics!r}"
        )
    if gripper_semantics is None and schema == "delivery":
        # Backward compatibility is still explicit: the advertised delivery
        # action semantics is the marker that selects 8_3_64eps (step/closed)
        # versus v3 (chunk-origin/opening). New servers should also publish
        # gripper_semantics for easier operator inspection.
        if expected_gripper_semantics is not None:
            gripper_semantics = expected_gripper_semantics
        else:
            try:
                delivery_version = int(metadata.get("contract_version"))
            except (TypeError, ValueError):
                delivery_version = None
            if metadata.get("legacy_delivery_v2") is True or (
                delivery_version is not None and delivery_version < CONTRACT_VERSION
            ):
                gripper_semantics = GRIPPER_CLOSED_FRACTION
            elif delivery_version is not None and delivery_version >= CONTRACT_VERSION:
                gripper_semantics = GRIPPER_OPENING_FRACTION
            else:
                errors.append(
                    "chunk-origin delivery metadata omits gripper_semantics and contract version; "
                    "refusing to guess legacy closed versus v3 opening fraction"
                )
    raw_state_gripper_semantics = metadata.get("state_gripper_semantics") or metadata.get(
        "raw_gripper_semantics"
    )
    state_gripper_semantics = _canonical_gripper_semantics(raw_state_gripper_semantics)
    if raw_state_gripper_semantics is not None and state_gripper_semantics is None:
        errors.append(f"unsupported state_gripper_semantics={raw_state_gripper_semantics!r}")
    if state_gripper_semantics is None:
        if schema == "delivery":
            state_gripper_semantics = gripper_semantics
        else:
            state_gripper_semantics = _joint_state_gripper_semantics_from_metadata(
                metadata, advertised_action_semantics, errors
            )
    advertised_delivery_convention = metadata.get("delivery_action_convention") or metadata.get(
        "model_action_convention"
    )
    if (
        expected_delivery_convention is not None
        and advertised_delivery_convention is not None
        and advertised_delivery_convention != expected_delivery_convention
    ):
        errors.append(
            f"delivery_action_convention={advertised_delivery_convention!r} conflicts with "
            f"action_semantics={action_semantics!r}; expected {expected_delivery_convention!r}"
        )
    camera_keys = metadata.get("camera_keys")
    raw_action_hz = metadata.get("action_hz")
    contract_version = _metadata_contract_version(metadata, errors)
    raw_action_horizon = metadata.get("action_horizon", DEFAULT_OPENPI_CHUNK_STEPS)
    try:
        action_horizon = int(raw_action_horizon)
    except (TypeError, ValueError):
        errors.append(f"action_horizon={raw_action_horizon!r} must be a positive integer")
        action_horizon = DEFAULT_OPENPI_CHUNK_STEPS
    else:
        if action_horizon <= 0:
            errors.append(f"action_horizon={raw_action_horizon!r} must be a positive integer")
    action_hz: float | None = None
    if raw_action_hz is not None:
        try:
            action_hz = float(raw_action_hz)
        except (TypeError, ValueError):
            errors.append(f"action_hz={raw_action_hz!r} must be a positive number")
        else:
            if not math.isfinite(action_hz) or action_hz <= 0:
                errors.append(f"action_hz={raw_action_hz!r} must be a positive number")
    camera_key_set = set(camera_keys) if isinstance(camera_keys, (list, tuple)) else set()
    if (
        not isinstance(camera_keys, (list, tuple))
        or len(camera_keys) != len(camera_key_set)
        or camera_key_set not in expected_camera_key_sets
    ):
        expected_camera_keys = [sorted(keys) for keys in expected_camera_key_sets]
        errors.append(f"camera_keys={camera_keys!r}, expected one of {expected_camera_keys!r}")
    if errors:
        raise RuntimeError("incompatible policy metadata: " + "; ".join(errors))

    return PolicyProtocol(
        schema=str(schema),
        arm_mode=advertised_mode,
        state_dim=int(metadata["state_dim"]),
        action_dim=int(advertised_action_dim),
        arm_side=str(metadata["arm_side"]),
        action_semantics=str(action_semantics),
        camera_keys=tuple(str(key) for key in camera_keys),
        action_hz=action_hz,
        gripper_semantics=gripper_semantics,
        state_gripper_semantics=state_gripper_semantics,
        metadata_gripper_semantics_explicit=raw_gripper_semantics is not None,
        contract_version=contract_version,
        action_horizon=action_horizon,
    )


def resolve_action_chunk_steps(
    *,
    action_hz: float,
    command_hz: float,
    override: int | None = None,
) -> int:
    """Return which future action row matches one robot command interval.

    For a 20 Hz policy and a 4 Hz compatibility command loop, row 5 (index 4)
    is the target for the end of the next 250 ms command interval. The current
    asynchronous runtime instead commands at 20 Hz and uses latency-prefix
    skipping. New delivery
    policies express every row relative to the same current observation.
    """
    if override is not None:
        if int(override) != override or int(override) <= 0:
            raise ValueError(f"action chunk steps must be a positive integer, got {override!r}")
        return int(override)
    if not math.isfinite(float(action_hz)) or float(action_hz) <= 0:
        raise ValueError(f"action_hz must be positive, got {action_hz!r}")
    if not math.isfinite(float(command_hz)) or float(command_hz) <= 0:
        raise ValueError(f"command_hz must be positive, got {command_hz!r}")
    return max(1, int(math.floor(float(action_hz) / float(command_hz) + 0.5)))


def aggregate_action_chunk(
    actions: np.ndarray,
    protocol: PolicyProtocol,
    steps: int,
) -> tuple[np.ndarray, int]:
    """Convert a model-rate action chunk into one robot command.

    Joint actions and new chunk-origin delivery actions already contain future
    targets aligned to the current observation, so row ``steps - 1`` is selected
    directly. Legacy step-delta delivery checkpoints are still supported by
    composing their consumed prefix. Gripper is absolute in every convention.
    """
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != protocol.action_dim or values.shape[0] <= 0:
        raise ExecutionBlocked(
            f"action chunk must have shape (T,{protocol.action_dim}), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ExecutionBlocked("action chunk contains non-finite values")
    if int(steps) <= 0:
        raise ExecutionBlocked(f"action chunk steps must be positive, got {steps!r}")
    used_steps = min(int(steps), values.shape[0])
    prefix = values[:used_steps]
    if protocol.schema == "joint":
        return prefix[-1].copy(), used_steps
    if protocol.schema != "delivery":
        raise ExecutionBlocked(f"unsupported action schema: {protocol.schema}")
    if protocol.action_semantics == DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS:
        return prefix[-1].copy(), used_steps
    if protocol.action_semantics != DELIVERY_STEP_ACTION_SEMANTICS:
        raise ExecutionBlocked(
            f"unsupported delivery action semantics: {protocol.action_semantics!r}"
        )

    # Compatibility path for markerless/explicit legacy checkpoints whose
    # predicted rows are chained frame-to-frame deltas.
    command = np.empty(protocol.action_dim, dtype=np.float64)
    arm_count = 2 if protocol.arm_mode == "bimanual" else 1
    for arm_index in range(arm_count):
        offset = arm_index * 7
        command[offset : offset + 3] = prefix[:, offset : offset + 3].sum(axis=0)
        total_rotation = np.eye(3, dtype=np.float64)
        for rotvec in prefix[:, offset + 3 : offset + 6]:
            total_rotation = Rotation.from_rotvec(rotvec).as_matrix() @ total_rotation
        command[offset + 3 : offset + 6] = Rotation.from_matrix(total_rotation).as_rotvec()
        command[offset + 6] = prefix[-1, offset + 6]
    return command, used_steps


def connect_policy(host: str, port: int, arm_side: str, arm_mode: str = "single") -> tuple[Any, PolicyProtocol]:
    """Create the official OpenPI client and validate the server handshake."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    logging.info("Connecting to official OpenPI policy at ws://%s:%d ...", host, port)
    policy = WebsocketClientPolicy(host=host, port=port)
    try:
        metadata = policy.get_server_metadata()
        if not isinstance(metadata, dict):
            raise RuntimeError(f"invalid policy metadata: {type(metadata).__name__}")
        protocol = validate_policy_metadata(metadata, arm_side, arm_mode)
    except Exception:
        close_policy(policy)
        raise
    logging.info("Policy connected: %s", metadata)
    return policy, protocol


def close_policy(policy: Any | None) -> None:
    if policy is None:
        return
    connection = getattr(policy, "_ws", None)
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass


def first_action(result: dict[str, Any], action_dim: int = 7) -> np.ndarray:
    """Return the first raw model action for backward-compatible callers."""
    actions = np.asarray(result.get("actions"), dtype=np.float64)
    if actions.ndim == 1:
        action = actions
    elif actions.ndim == 2 and len(actions):
        action = actions[0]
    else:
        raise ExecutionBlocked(f"invalid action chunk shape {actions.shape}")
    if action.shape != (action_dim,) or not np.all(np.isfinite(action)):
        raise ExecutionBlocked(f"first action must be finite {action_dim}D, got {action.shape}")
    return action


@dataclass(frozen=True)
class TimedTarget:
    """One decoded absolute target on the policy's monotonic action timeline."""

    queue_index: int
    wire_action: np.ndarray
    absolute_target: np.ndarray
    target_monotonic: float
    generation: int = 0
    source_index: int | None = None
    blended: bool = False
    blend_step: int | None = None
    hold: bool = False


# Compatibility name retained for existing callers and telemetry consumers.
DecodedQueuedAction = TimedTarget


def _finite_action_chunk(actions: Any, action_dim: int) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != action_dim or values.shape[0] <= 0:
        raise ExecutionBlocked(f"action chunk must have shape (T,{action_dim}), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ExecutionBlocked("action chunk contains non-finite values")
    return values


def _opening_fraction(
    value: float,
    *,
    semantics: str,
    tolerance: float,
) -> float:
    raw = float(value)
    if raw < -tolerance or raw > 1.0 + tolerance:
        raise ExecutionBlocked(
            f"gripper target {raw:.5f} exceeds [0,1] tolerance {tolerance:.5f}"
        )
    normalized = float(np.clip(raw, 0.0, 1.0))
    if semantics == GRIPPER_OPENING_FRACTION:
        return normalized
    if semantics == GRIPPER_CLOSED_FRACTION:
        return 1.0 - normalized
    if semantics == GRIPPER_OPENING_METRES:
        metre_tolerance = tolerance * GRIPPER_MAX_M
        if raw < -metre_tolerance or raw > GRIPPER_MAX_M + metre_tolerance:
            raise ExecutionBlocked(
                f"gripper target {raw:.5f}m exceeds [0,{GRIPPER_MAX_M:.5f}]m "
                f"tolerance {metre_tolerance:.5f}m"
            )
        return float(np.clip(raw, 0.0, GRIPPER_MAX_M) / GRIPPER_MAX_M)
    raise ExecutionBlocked(f"unsupported gripper semantics: {semantics!r}")


def decode_action_queue(
    actions: Any,
    protocol: PolicyProtocol,
    raw_delivery_state: np.ndarray,
    qpos_m: np.ndarray,
    *,
    steps: int | None = None,
    generation: int = 0,
    observation_capture_monotonic: float | None = None,
    action_hz: float | None = None,
    gripper_range_tolerance: float = DEFAULT_GRIPPER_RANGE_TOLERANCE,
) -> tuple[np.ndarray, list[DecodedQueuedAction]]:
    """Decode the first action rows against one immutable inference anchor.

    v3 delivery rows are independently decoded by the shared root-contract
    helper. Legacy 8_3_64eps rows are first composed from one-step deltas to
    chunk-origin deltas, then passed through the same absolute-target helper.
    """
    values = _finite_action_chunk(actions, protocol.action_dim)
    capture_monotonic = (
        time.monotonic()
        if observation_capture_monotonic is None
        else float(observation_capture_monotonic)
    )
    target_hz = float(action_hz or protocol.action_hz or DEFAULT_ACTION_HZ)
    if not math.isfinite(capture_monotonic) or capture_monotonic < 0:
        raise ExecutionBlocked(
            f"observation capture monotonic time is invalid: {capture_monotonic!r}"
        )
    if not math.isfinite(target_hz) or target_hz <= 0:
        raise ExecutionBlocked(f"action_hz must be positive, got {target_hz!r}")
    if steps is None:
        used_steps = len(values)
    else:
        if int(steps) <= 0:
            raise ExecutionBlocked(f"action chunk steps must be positive, got {steps!r}")
        used_steps = min(int(steps), len(values))
    wire = values[:used_steps].copy()
    arm_count = 2 if protocol.arm_mode == "bimanual" else 1
    for arm_index in range(arm_count):
        gripper_index = arm_index * 7 + 6
        for row in wire:
            _opening_fraction(
                row[gripper_index],
                semantics=protocol.gripper_semantics,
                tolerance=gripper_range_tolerance,
            )

    anchor = policy_observation_state(raw_delivery_state, qpos_m, protocol).astype(np.float64)
    if anchor.shape != (protocol.state_dim,) or not np.all(np.isfinite(anchor)):
        raise ExecutionBlocked(
            f"inference anchor must be finite {protocol.state_dim}D, got {anchor.shape}"
        )
    try:
        if protocol.schema == "delivery":
            if protocol.action_semantics == DELIVERY_STEP_ACTION_SEMANTICS:
                model_actions = step_deltas_to_chunk_origin(wire, arm_count=arm_count)
            elif protocol.action_semantics == DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS:
                model_actions = wire
            else:
                raise ExecutionBlocked(
                    f"unsupported delivery action semantics: {protocol.action_semantics!r}"
                )
            absolute = chunk_origin_deltas_to_absolute_eef_targets(
                anchor, model_actions, arm_count=arm_count
            )
        elif protocol.schema == "joint":
            absolute = wire
        else:
            raise ExecutionBlocked(f"unsupported execution schema: {protocol.schema}")
    except (ValueError, FloatingPointError) as exc:
        raise ExecutionBlocked(f"cannot decode action chunk: {exc}") from exc

    absolute = np.asarray(absolute, dtype=np.float64)
    if absolute.ndim == 1:
        absolute = absolute[None, :]
    expected_absolute_dim = 10 * arm_count if protocol.schema == "delivery" else 7 * arm_count
    if absolute.shape != (used_steps, expected_absolute_dim) or not np.all(np.isfinite(absolute)):
        raise ExecutionBlocked(
            f"decoded absolute targets must have shape ({used_steps},{expected_absolute_dim}), "
            f"got {absolute.shape}"
        )
    decoded = [
        DecodedQueuedAction(
            index,
            wire[index].copy(),
            absolute[index].copy(),
            capture_monotonic + (index + 1) / target_hz,
            generation=generation,
            source_index=index,
        )
        for index in range(used_steps)
    ]
    return anchor, decoded


@dataclass(frozen=True)
class InferenceLaunch:
    generation: int
    captured_at: float
    captured_monotonic: float
    launched_at: float
    launched_monotonic: float
    raw_delivery_state: np.ndarray
    qpos_m: np.ndarray
    image_timestamps: dict[str, float]


@dataclass(frozen=True)
class InferenceCompletion:
    launch: InferenceLaunch
    result: dict[str, Any] | None
    arrived_at: float
    arrived_monotonic: float
    error: BaseException | None = None


@dataclass(frozen=True)
class InferenceWorkerResult:
    """Inference result plus camera timestamps captured off the control thread."""

    result: dict[str, Any]
    image_timestamps: dict[str, float]


class AsyncPolicyInference:
    """Single-in-flight inference worker; polling never blocks the control loop."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper-policy")
        self._future: Future | None = None
        self._launch: InferenceLaunch | None = None

    @property
    def in_flight(self) -> bool:
        return self._future is not None

    def launch(
        self,
        policy: Any,
        observation: dict[str, Any],
        launch: InferenceLaunch,
    ) -> bool:
        return self.launch_callable(lambda: policy.infer(observation), launch)

    def launch_callable(
        self,
        task: Callable[[], Any],
        launch: InferenceLaunch,
    ) -> bool:
        """Submit capture/inference work without waiting on the control thread."""
        if self._future is not None:
            return False
        self._launch = launch
        self._future = self._executor.submit(task)
        return True

    def poll(self) -> InferenceCompletion | None:
        future = self._future
        launch = self._launch
        if future is None or launch is None or not future.done():
            return None
        arrived_at = time.time()
        arrived_monotonic = time.monotonic()
        self._future = None
        self._launch = None
        try:
            result = future.result()
        except BaseException as exc:  # surfaced on the 20 Hz control thread
            return InferenceCompletion(
                launch, None, arrived_at, arrived_monotonic, error=exc
            )
        if isinstance(result, InferenceWorkerResult):
            launch = replace(
                launch,
                image_timestamps={
                    key: float(value)
                    for key, value in result.image_timestamps.items()
                },
            )
            result = result.result
        return InferenceCompletion(launch, result, arrived_at, arrived_monotonic, error=None)

    def shutdown(self) -> None:
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class PeriodicSchedule:
    """Drift-resistant periodic launch schedule used by deterministic tests/run."""

    frequency_hz: float
    next_at: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz

    def due(self, now: float) -> bool:
        if now + 1e-9 < self.next_at:
            return False
        elapsed = max(0.0, now - self.next_at)
        periods = int(math.floor(elapsed / self.period_s)) + 1
        self.next_at += periods * self.period_s
        return True


def _blend_absolute_target(
    old_target: np.ndarray,
    new_target: np.ndarray,
    protocol: PolicyProtocol,
    alpha: float,
) -> np.ndarray:
    """Interpolate absolute targets; delivery rotations follow SO(3)."""
    old = np.asarray(old_target, dtype=np.float64)
    new = np.asarray(new_target, dtype=np.float64)
    if old.shape != new.shape or not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ExecutionBlocked("blend targets must have matching finite shapes")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ExecutionBlocked(f"blend alpha must be in [0,1], got {alpha!r}")
    if protocol.schema == "joint":
        arm_count = 2 if protocol.arm_mode == "bimanual" else 1
        output = np.empty_like(old)
        for arm_index in range(arm_count):
            offset = arm_index * 7
            output[offset : offset + 6] = old[offset : offset + 6] + alpha * (
                new[offset : offset + 6] - old[offset : offset + 6]
            )
            # Gripper is never interpolated with pose/joints. During the blend
            # it stays on the old trajectory; the execution filter handles the
            # later opening transition with confirmation and a step limit.
            output[offset + 6] = old[offset + 6]
        return output
    if protocol.schema != "delivery":
        raise ExecutionBlocked(f"unsupported blend schema: {protocol.schema}")
    arm_count = 2 if protocol.arm_mode == "bimanual" else 1
    output = np.empty_like(old)
    for arm_index in range(arm_count):
        offset = arm_index * 10
        old_arm = old[offset : offset + 10]
        new_arm = new[offset : offset + 10]
        output[offset : offset + 3] = old_arm[:3] + alpha * (
            new_arm[:3] - old_arm[:3]
        )
        old_rotation = rotation_from_state(old_arm)
        new_rotation = rotation_from_state(new_arm)
        relative_rotvec = Rotation.from_matrix(
            new_rotation @ old_rotation.T
        ).as_rotvec()
        blended_rotation = (
            Rotation.from_rotvec(alpha * relative_rotvec).as_matrix() @ old_rotation
        )
        output[offset + 3 : offset + 9] = matrix_to_rotation6d(blended_rotation)
        output[offset + 9] = old_arm[9]
    return output


def blend_absolute_trajectories(
    old_actions: list[DecodedQueuedAction],
    new_actions: list[DecodedQueuedAction],
    protocol: PolicyProtocol,
    *,
    blend_steps: int,
) -> list[DecodedQueuedAction]:
    """Build a complete candidate queue, then callers atomically swap it in."""
    if not old_actions:
        return list(new_actions)
    if blend_steps not in {2, 3, 4}:
        raise ExecutionBlocked(f"blend_steps must be 2, 3, or 4, got {blend_steps!r}")
    if len(new_actions) < blend_steps:
        raise ExecutionBlocked(
            f"new trajectory has {len(new_actions)} rows, fewer than {blend_steps} blend rows"
        )
    blended: list[DecodedQueuedAction] = []
    for index in range(blend_steps):
        old_action = old_actions[min(index, len(old_actions) - 1)]
        new_action = new_actions[index]
        alpha = (index + 1) / blend_steps
        blended.append(
            DecodedQueuedAction(
                queue_index=new_action.queue_index,
                wire_action=new_action.wire_action.copy(),
                absolute_target=_blend_absolute_target(
                    old_action.absolute_target,
                    new_action.absolute_target,
                    protocol,
                    alpha,
                ),
                target_monotonic=new_action.target_monotonic,
                generation=new_action.generation,
                source_index=new_action.source_index,
                blended=True,
                blend_step=index + 1,
            )
        )
    blended.extend(new_actions[blend_steps:])
    return blended


def _check_delivery_absolute_target(
    current_state: np.ndarray,
    current_gripper_m: float,
    absolute_target: np.ndarray,
    *,
    gripper_semantics: str,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
    max_gripper_step: float,
    gripper_range_tolerance: float,
    workspace_x: tuple[float, float],
    workspace_y: tuple[float, float],
    workspace_z: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Validate an already-decoded absolute EEF target against fresh feedback."""
    current_state = np.asarray(current_state, dtype=np.float64)
    target = np.asarray(absolute_target, dtype=np.float64)
    if current_state.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(current_state)):
        raise ExecutionBlocked("current delivery state is not finite 10D")
    if target.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(target)):
        raise ExecutionBlocked("decoded absolute delivery target is not finite 10D")

    target_xyz = target[:3]
    current_rotation = rotation_from_state(current_state)
    target_rotation = rotation_from_state(target)
    translation_step = float(np.linalg.norm(target_xyz - current_state[:3]))
    rotation_step = float(
        Rotation.from_matrix(target_rotation @ current_rotation.T).magnitude()
    )
    if translation_step > max_translation_step_m + 1e-9:
        raise ExecutionBlocked(
            f"translation step {translation_step:.5f}m exceeds {max_translation_step_m:.5f}m"
        )
    if rotation_step > max_rotation_step_rad + 1e-9:
        raise ExecutionBlocked(
            f"rotation step {rotation_step:.5f}rad exceeds {max_rotation_step_rad:.5f}rad"
        )
    for axis, value, bounds in zip("xyz", target_xyz, (workspace_x, workspace_y, workspace_z)):
        if not bounds[0] <= float(value) <= bounds[1]:
            raise ExecutionBlocked(
                f"target {axis}={value:.5f}m outside workspace "
                f"[{bounds[0]:.5f}, {bounds[1]:.5f}]"
            )

    opening_fraction = _opening_fraction(
        target[9], semantics=gripper_semantics, tolerance=gripper_range_tolerance
    )
    current_opening_fraction = float(current_gripper_m) / GRIPPER_MAX_M
    gripper_step = abs(opening_fraction - current_opening_fraction)
    if gripper_step > max_gripper_step + 1e-9:
        raise ExecutionBlocked(
            f"gripper step {gripper_step:.5f} exceeds {max_gripper_step:.5f}"
        )
    target_gripper_m = opening_fraction * GRIPPER_MAX_M
    target_rpy_deg = Rotation.from_matrix(target_rotation).as_euler("xyz", degrees=True)
    return target_xyz.copy(), target_rpy_deg, target_gripper_m, opening_fraction


def build_checked_target(
    state: np.ndarray,
    action: np.ndarray,
    *,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
    max_gripper_step: float,
    gripper_range_tolerance: float,
    workspace_x: tuple[float, float],
    workspace_y: tuple[float, float],
    workspace_z: tuple[float, float],
    gripper_semantics: str = GRIPPER_CLOSED_FRACTION,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Backward-compatible single-action decode/check helper."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if state.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(state)):
        raise ExecutionBlocked("current delivery state is not finite 10D")
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise ExecutionBlocked("delivery action is not finite 7D")
    _opening_fraction(
        action[6], semantics=gripper_semantics, tolerance=gripper_range_tolerance
    )
    try:
        absolute = chunk_origin_deltas_to_absolute_eef_targets(state, action, arm_count=1)
    except ValueError as exc:
        raise ExecutionBlocked(f"cannot decode delivery action: {exc}") from exc
    current_opening = _opening_fraction(
        state[9], semantics=gripper_semantics, tolerance=gripper_range_tolerance
    )
    checked = _check_delivery_absolute_target(
        state,
        current_opening * GRIPPER_MAX_M,
        absolute,
        gripper_semantics=gripper_semantics,
        max_translation_step_m=max_translation_step_m,
        max_rotation_step_rad=max_rotation_step_rad,
        max_gripper_step=max_gripper_step,
        gripper_range_tolerance=gripper_range_tolerance,
        workspace_x=workspace_x,
        workspace_y=workspace_y,
        workspace_z=workspace_z,
    )
    return checked[0], checked[1], checked[2]


def build_checked_joint_target(
    qpos: np.ndarray,
    action: np.ndarray,
    *,
    max_joint_step_rad: float,
    max_gripper_step: float | None = None,
    max_gripper_step_m: float | None = None,
    gripper_range_tolerance: float = DEFAULT_GRIPPER_RANGE_TOLERANCE,
    gripper_semantics: str = GRIPPER_OPENING_FRACTION,
) -> tuple[np.ndarray, float]:
    """Validate absolute joints + opening fraction and return Piper units in SI."""
    qpos = np.asarray(qpos, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
        raise ExecutionBlocked("current joint state is not finite 7D")
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise ExecutionBlocked("joint action is not finite 7D")

    for index, (value, bounds) in enumerate(zip(action[:6], JOINT_LIMITS_RAD), start=1):
        if not bounds[0] <= float(value) <= bounds[1]:
            raise ExecutionBlocked(
                f"joint{index} target {value:.5f}rad outside "
                f"[{bounds[0]:.5f}, {bounds[1]:.5f}]"
            )
    opening_fraction = _opening_fraction(
        action[6],
        semantics=gripper_semantics,
        tolerance=gripper_range_tolerance,
    )
    gripper_m = opening_fraction * GRIPPER_MAX_M

    joint_deltas = np.abs(action[:6] - qpos[:6])
    worst_joint = int(np.argmax(joint_deltas))
    if float(joint_deltas[worst_joint]) > max_joint_step_rad + 1e-9:
        raise ExecutionBlocked(
            f"joint{worst_joint + 1} step {joint_deltas[worst_joint]:.5f}rad exceeds "
            f"{max_joint_step_rad:.5f}rad"
        )
    if max_gripper_step is None:
        max_gripper_step = (
            float(max_gripper_step_m) / GRIPPER_MAX_M
            if max_gripper_step_m is not None
            else 0.25
        )
    current_opening_fraction = float(qpos[6]) / GRIPPER_MAX_M
    gripper_delta = abs(opening_fraction - current_opening_fraction)
    if gripper_delta > max_gripper_step + 1e-9:
        raise ExecutionBlocked(
            f"gripper step {gripper_delta:.5f} exceeds {max_gripper_step:.5f}"
        )
    return action[:6].copy(), gripper_m


class ExecutionController:
    def __init__(self, piper: Any | dict[str, Any], args: argparse.Namespace):
        self.args = args
        self.arm_mode = getattr(args, "arm_mode", "single")
        self.arm_side = getattr(args, "arm_side", "right")
        self.pipers = piper if isinstance(piper, dict) else {self.arm_side: piper}
        self.piper = next(iter(self.pipers.values()))
        self.robot_enabled: set[str] = set()
        allow_execution = bool(getattr(args, "allow_execution", False))
        self.state = "client_disabled" if not allow_execution else "shadow"
        self.blocked_reason = (
            "local --allow-execution is absent" if not allow_execution else "dashboard is shadow"
        )
        self.last_command_at: float | None = None
        self.control_revision: int | None = None
        self.robot_status: dict[str, Any] | None = None
        self.inference_hz = float(getattr(args, "hz", DEFAULT_INFERENCE_HZ))
        self.policy_action_hz = float(getattr(args, "action_hz", None) or DEFAULT_ACTION_HZ)
        self.control_hz = float(getattr(args, "control_hz", DEFAULT_ACTION_HZ))
        self.min_action_chunk_steps = int(
            getattr(args, "min_action_chunk_steps", DEFAULT_MIN_ACTION_CHUNK_STEPS)
        )
        self.action_chunk_steps = self.min_action_chunk_steps  # legacy telemetry alias
        self.blend_steps = int(getattr(args, "blend_steps", DEFAULT_BLEND_STEPS))
        self.latency_skip_compensation_steps = int(
            getattr(args, "latency_skip_compensation_steps", 0)
        )
        self.estimated_actuator_delay_s = float(
            getattr(args, "actuator_delay_s", DEFAULT_ACTUATOR_DELAY_S)
        )
        self.gripper_lowpass_alpha = float(
            getattr(args, "gripper_lowpass_alpha", DEFAULT_GRIPPER_LOWPASS_ALPHA)
        )
        self.gripper_hysteresis = float(
            getattr(args, "gripper_hysteresis", DEFAULT_GRIPPER_HYSTERESIS)
        )
        self.gripper_confirm_steps = int(
            getattr(args, "gripper_confirm_steps", DEFAULT_GRIPPER_CONFIRM_STEPS)
        )
        self.expected_action_horizon = DEFAULT_OPENPI_CHUNK_STEPS
        self.pending_actions: list[DecodedQueuedAction] = []
        self.last_safe_target: DecodedQueuedAction | None = None
        self.hold_active = False
        self.hold_count = 0
        self.hold_started_at: float | None = None
        self.current_timed_target: dict[str, Any] | None = None
        self._filtered_gripper_opening: dict[str, float] = {}
        self._gripper_extreme_candidate: dict[str, str | None] = {}
        self._gripper_extreme_count: dict[str, int] = {}
        self._gripper_extreme_latch: dict[str, str | None] = {}
        self.active_generation = 0
        self._next_inference_generation = 1
        self.waiting_fresh_after_enable = False

        self.last_action_chunk_steps = 0
        self.last_composed_action: list[float] | None = None
        self.last_composed_action_at: float | None = None
        self.queue_anchor_state: list[float] | None = None
        self.queue_anchor_qpos_m: list[float] | None = None
        self.queue_anchor_at: float | None = None
        self.queue_loaded_at: float | None = None
        self.queue_image_timestamps: dict[str, float] = {}
        self.queue_control: dict[str, Any] | None = None
        self.authorization_deadline_monotonic: float | None = None
        self.queued_action_index: int | None = None
        self.last_queued_action_index: int | None = None
        self.last_wire_action: list[float] | None = None
        self.last_decoded_absolute_target: dict[str, Any] | None = None
        self.last_feedback_at: float | None = None
        self.dropped_action_count = 0
        self.last_queue_drop_reason = ""

        self.inference_launch_at: float | None = None
        self.inference_capture_at: float | None = None
        self.inference_capture_monotonic: float | None = None
        self.inference_arrival_at: float | None = None
        self.inference_arrival_monotonic: float | None = None
        self.inference_latency_s: float | None = None
        self.inference_skip_steps = 0
        self.inference_blend_steps = 0
        self.inference_generation: int | None = None
        self.inference_old_remaining = 0
        self.inference_launch_count = 0
        self.inference_launch_deferred_count = 0
        self.rejected_result: dict[str, Any] | None = None
        self.rejected_result_count = 0
        self.queue_underrun = False
        self.queue_underrun_count = 0
        self.queue_underrun_at: float | None = None
        self.control_tick_count = 0
        self.control_overrun_count = 0
        self.arm_hold_targets: dict[str, np.ndarray] = {}
        self.arm_hold_started_at: dict[str, float] = {}
        self.ik_solver: PiperContinuousIK | None = None

    def configure_protocol(self, protocol: PolicyProtocol) -> None:
        action_hz = getattr(self.args, "action_hz", None) or protocol.action_hz or DEFAULT_ACTION_HZ
        self.policy_action_hz = float(action_hz)
        self.control_hz = float(
            getattr(self.args, "control_hz", DEFAULT_ACTION_HZ)
        )
        self.inference_hz = float(getattr(self.args, "hz", DEFAULT_INFERENCE_HZ))
        self.min_action_chunk_steps = int(
            getattr(self.args, "min_action_chunk_steps", DEFAULT_MIN_ACTION_CHUNK_STEPS)
        )
        legacy_override = getattr(self.args, "action_chunk_steps", None)
        if legacy_override is not None:
            self.min_action_chunk_steps = int(legacy_override)
        self.action_chunk_steps = self.min_action_chunk_steps
        self.blend_steps = int(getattr(self.args, "blend_steps", DEFAULT_BLEND_STEPS))
        self.latency_skip_compensation_steps = int(
            getattr(self.args, "latency_skip_compensation_steps", 0)
        )
        self.estimated_actuator_delay_s = float(
            getattr(self.args, "actuator_delay_s", DEFAULT_ACTUATOR_DELAY_S)
        )
        self.gripper_lowpass_alpha = float(
            getattr(
                self.args,
                "gripper_lowpass_alpha",
                DEFAULT_GRIPPER_LOWPASS_ALPHA,
            )
        )
        self.gripper_hysteresis = float(
            getattr(self.args, "gripper_hysteresis", DEFAULT_GRIPPER_HYSTERESIS)
        )
        self.gripper_confirm_steps = int(
            getattr(self.args, "gripper_confirm_steps", DEFAULT_GRIPPER_CONFIRM_STEPS)
        )
        self.expected_action_horizon = int(protocol.action_horizon)
        self.pending_actions.clear()
        self.waiting_fresh_after_enable = False
        self.queue_underrun = False
        logging.info(
            "Async action timing: control=%.3g Hz inference=%.3g Hz expected_chunk=%d "
            "minimum_chunk=%d blend=%d actuator_delay=%.4fs latency_compensation=%d",
            self.control_hz,
            self.inference_hz,
            self.expected_action_horizon,
            self.min_action_chunk_steps,
            self.blend_steps,
            self.estimated_actuator_delay_s,
            self.latency_skip_compensation_steps,
        )
        if not math.isclose(self.policy_action_hz, self.control_hz):
            logging.warning(
                "Policy action rate %.3g Hz differs from control rate %.3g Hz; "
                "control remains independently scheduled while timed targets use policy action Hz",
                self.policy_action_hz,
                self.control_hz,
            )

    @property
    def pending_action_count(self) -> int:
        return len(self.pending_actions)

    def allocate_inference_generation(self) -> int:
        generation = self._next_inference_generation
        self._next_inference_generation += 1
        return generation

    def record_inference_launch(self, launch: InferenceLaunch) -> None:
        self.inference_generation = launch.generation
        self.inference_capture_at = launch.captured_at
        self.inference_capture_monotonic = launch.captured_monotonic
        self.inference_launch_at = launch.launched_at
        self.inference_launch_count += 1

    def record_launch_deferred(self) -> None:
        self.inference_launch_deferred_count += 1

    def record_control_tick(self, *, overrun: bool = False) -> None:
        self.control_tick_count += 1
        if overrun:
            self.control_overrun_count += 1

    def metadata(self) -> dict[str, Any]:
        return {
            "allow_execution": bool(getattr(self.args, "allow_execution", False)),
            "execution_state": self.state,
            "blocked_reason": self.blocked_reason,
            "last_command_at": self.last_command_at,
            "control_revision": self.control_revision,
            "robot_arm_status": self.robot_status,
            "policy_action_hz": self.policy_action_hz,
            "command_hz": self.control_hz,
            "control_hz": self.control_hz,
            "inference_hz": self.inference_hz,
            "expected_action_horizon": self.expected_action_horizon,
            "min_action_chunk_steps": self.min_action_chunk_steps,
            "action_chunk_steps": self.action_chunk_steps,
            "last_action_chunk_steps": self.last_action_chunk_steps,
            "last_composed_action": self.last_composed_action,
            "last_composed_action_at": self.last_composed_action_at,
            "queue_anchor_state": self.queue_anchor_state,
            "queue_anchor_qpos_m": self.queue_anchor_qpos_m,
            "queue_anchor_at": self.queue_anchor_at,
            "queue_loaded_at": self.queue_loaded_at,
            "queued_action_count": len(self.pending_actions),
            "queued_action_index": self.queued_action_index,
            "last_queued_action_index": self.last_queued_action_index,
            "last_wire_action": self.last_wire_action,
            "last_decoded_absolute_target": self.last_decoded_absolute_target,
            "last_feedback_at": self.last_feedback_at,
            "dropped_action_count": self.dropped_action_count,
            "last_queue_drop_reason": self.last_queue_drop_reason,
            "inference_launch_at": self.inference_launch_at,
            "inference_capture_at": self.inference_capture_at,
            "inference_capture_monotonic": self.inference_capture_monotonic,
            "inference_arrival_at": self.inference_arrival_at,
            "inference_arrival_monotonic": self.inference_arrival_monotonic,
            "inference_latency_s": self.inference_latency_s,
            "inference_skip_steps": self.inference_skip_steps,
            "inference_blend_steps": self.inference_blend_steps,
            "inference_generation": self.inference_generation,
            "action_generation": self.active_generation,
            "inference_old_remaining": self.inference_old_remaining,
            "old_remaining": self.inference_old_remaining,
            "inference_launch_count": self.inference_launch_count,
            "inference_launch_deferred_count": self.inference_launch_deferred_count,
            "queue_underrun": self.queue_underrun,
            "queue_underrun_count": self.queue_underrun_count,
            "queue_underrun_at": self.queue_underrun_at,
            "hold_active": self.hold_active,
            "hold_count": self.hold_count,
            "hold_started_at": self.hold_started_at,
            "last_safe_target": self._timed_target_telemetry(self.last_safe_target),
            "timed_target": self.current_timed_target,
            "rejected_result": self.rejected_result,
            "rejected_result_count": self.rejected_result_count,
            "control_tick_count": self.control_tick_count,
            "control_overrun_count": self.control_overrun_count,
            "estimated_actuator_delay_s": self.estimated_actuator_delay_s,
            "gripper_filter": {
                "lowpass_alpha": self.gripper_lowpass_alpha,
                "hysteresis": self.gripper_hysteresis,
                "confirm_steps": self.gripper_confirm_steps,
                "opening_fraction": dict(self._filtered_gripper_opening),
                "extreme_latch": dict(self._gripper_extreme_latch),
            },
            "safety_profile": SAFETY_PROFILE,
            "delivery_command_mode": "continuous_ik_joint",
            "continuous_ik": {
                "max_joint_step_rad": float(getattr(self.args, "ik_max_joint_step_rad", DEFAULT_IK_MAX_JOINT_STEP_RAD)),
                "position_tolerance_m": float(getattr(self.args, "ik_position_tolerance_m", DEFAULT_IK_POSITION_TOLERANCE_M)),
                "rotation_tolerance_rad": float(getattr(self.args, "ik_rotation_tolerance_rad", DEFAULT_IK_ROTATION_TOLERANCE_RAD)),
                "max_nfev": int(getattr(self.args, "ik_max_nfev", DEFAULT_IK_MAX_NFEV)),
            },
            "delivery_safety_limits": {
                "max_translation_step_m": float(
                    getattr(self.args, "max_translation_step_m", DEFAULT_MAX_TRANSLATION_STEP_M)
                ),
                "max_rotation_step_rad": float(
                    getattr(self.args, "max_rotation_step_rad", DEFAULT_MAX_ROTATION_STEP_RAD)
                ),
                "max_gripper_step": float(
                    getattr(self.args, "max_gripper_step", DEFAULT_MAX_GRIPPER_STEP)
                ),
                "workspace_x_m": list(getattr(self.args, "workspace_x", DEFAULT_WORKSPACE_X_M)),
                "workspace_y_m": list(getattr(self.args, "workspace_y", DEFAULT_WORKSPACE_Y_M)),
                "workspace_z_m": list(getattr(self.args, "workspace_z", DEFAULT_WORKSPACE_Z_M)),
                "blend_targets_rechecked_each_control_step": True,
            },
        }

    def _block(self, state: str, reason: str) -> bool:
        self.state = state
        self.blocked_reason = reason[:500]
        # Safety failures used to be visible only in Dashboard telemetry. Emit
        # a rate-limited local message as well so an operator can distinguish
        # "chunk accepted" from "robot command actually sent" at the terminal.
        reason_class = re.sub(r"[-+]?\d+(?:\.\d+)?", "#", self.blocked_reason)
        log_key = (self.state, reason_class)
        now = time.monotonic()
        last_key = getattr(self, "_last_block_log_key", None)
        last_at = float(getattr(self, "_last_block_log_at", 0.0))
        if log_key != last_key or now - last_at >= 2.0:
            logging.warning(
                "Robot command not sent: state=%s reason=%s",
                self.state,
                self.blocked_reason,
            )
            self._last_block_log_key = log_key
            self._last_block_log_at = now
        return False

    def _reject_result(self, generation: int, reason: str, arrived_at: float) -> bool:
        self.rejected_result_count += 1
        self.rejected_result = {
            "generation": int(generation),
            "arrived_at": float(arrived_at),
            "reason": str(reason)[:500],
        }
        logging.warning("Rejected inference generation %d: %s", generation, reason)
        return False

    def _enable_robot(self, side: str, piper: Any, hold_qpos: np.ndarray) -> None:
        """Enable Piper and immediately hold its measured joint pose."""
        hold_qpos = np.asarray(hold_qpos, dtype=np.float64)
        if hold_qpos.shape != (7,) or not np.all(np.isfinite(hold_qpos)):
            raise ExecutionBlocked(f"{side} Piper hold qpos is not finite 7D")
        lower = JOINT_LIMITS_RAD[:, 0]
        upper = JOINT_LIMITS_RAD[:, 1]
        excess = np.maximum(lower - hold_qpos[:6], hold_qpos[:6] - upper)
        if float(np.max(excess)) > IK_FEEDBACK_LIMIT_TOLERANCE_RAD:
            raise ExecutionBlocked(
                f"{side} Piper measured joints are too far outside limits to hold safely"
            )
        # Hold the measured pose exactly.  Clipping a calibrated zero-offset
        # feedback value to the SDK's nominal limits would itself create an
        # unsolicited enable-time motion.
        hold_joints = hold_qpos[:6].copy()
        hold_gripper_m = float(np.clip(hold_qpos[6], 0.0, GRIPPER_MAX_M))
        deadline = time.monotonic() + self.args.enable_timeout_s
        while time.monotonic() < deadline:
            if piper.EnablePiper():
                raw_joints = np.rint(hold_joints * RAD_FACTOR).astype(np.int64)
                piper.ModeCtrl(0x01, 0x01, self.args.speed_pct, 0x00)
                piper.JointCtrl(*map(int, raw_joints))
                piper.GripperCtrl(
                    round(hold_gripper_m * GRIPPER_FACTOR),
                    self.args.gripper_effort,
                    0x01,
                    0,
                )
                self.arm_hold_targets[side] = hold_joints.copy()
                self.arm_hold_started_at[side] = time.monotonic()
                self.robot_enabled.add(side)
                return
            time.sleep(0.02)
        raise ExecutionBlocked(f"{side} Piper enable timed out after {self.args.enable_timeout_s:.1f}s")

    def _candidate_execution_control(
        self,
        control: Any,
        *,
        arrived_monotonic: float,
    ) -> tuple[int, float]:
        if not isinstance(control, dict):
            raise ExecutionBlocked("policy response has no execution_control")
        if control.get("mode") != "execute":
            reason = "dashboard authorization expired" if control.get("expired") else "dashboard is shadow"
            raise PermissionError(reason)
        if not control.get("task_id") or not control.get("session_id"):
            raise ExecutionBlocked("execution authorization has no task/session identity")
        try:
            revision = int(control.get("revision", 0))
            remaining_s = float(control["expires_at"]) - float(control["server_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutionBlocked("execution authorization has no valid expiry") from exc
        if remaining_s <= 0:
            raise PermissionError("execution authorization expired")
        return revision, arrived_monotonic + remaining_s

    def _target_telemetry(
        self, protocol: PolicyProtocol, absolute_target: np.ndarray
    ) -> dict[str, Any]:
        sides = ("left", "right") if protocol.arm_mode == "bimanual" else (protocol.arm_side,)
        targets: dict[str, Any] = {}
        for index, side in enumerate(sides):
            if protocol.schema == "delivery":
                target = absolute_target[index * 10 : (index + 1) * 10]
                rotation = rotation_from_state(target)
                opening = _opening_fraction(
                    target[9],
                    semantics=protocol.gripper_semantics,
                    tolerance=float(
                        getattr(
                            self.args,
                            "gripper_range_tolerance",
                            DEFAULT_GRIPPER_RANGE_TOLERANCE,
                        )
                    ),
                )
                targets[side] = {
                    "xyz_m": target[:3].tolist(),
                    "rotation6d": target[3:9].tolist(),
                    "rpy_deg": Rotation.from_matrix(rotation).as_euler("xyz", degrees=True).tolist(),
                    "wire_gripper_target": float(target[9]),
                    "gripper_opening_fraction": opening,
                    "gripper_opening_m": opening * GRIPPER_MAX_M,
                }
            else:
                target = absolute_target[index * 7 : (index + 1) * 7]
                opening = _opening_fraction(
                    target[6],
                    semantics=protocol.gripper_semantics,
                    tolerance=float(
                        getattr(
                            self.args,
                            "gripper_range_tolerance",
                            DEFAULT_GRIPPER_RANGE_TOLERANCE,
                        )
                    ),
                )
                targets[side] = {
                    "joints_rad": target[:6].tolist(),
                    "wire_gripper_target": float(target[6]),
                    "gripper_opening_fraction": opening,
                    "gripper_opening_m": opening * GRIPPER_MAX_M,
                }
        return targets

    def _timed_target_telemetry(
        self,
        target: DecodedQueuedAction | None,
        *,
        now_monotonic: float | None = None,
    ) -> dict[str, Any] | None:
        if target is None:
            return None
        now_monotonic = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        return {
            "target_monotonic": float(target.target_monotonic),
            "target_age_s": now_monotonic - float(target.target_monotonic),
            "source_generation": int(target.generation),
            "source_index": target.source_index,
            "queue_index": int(target.queue_index),
            "blended": bool(target.blended),
            "blend_step": target.blend_step,
            "hold": bool(target.hold),
        }

    def _estimated_execution_time(self, now_monotonic: float) -> float:
        fixed_compensation_s = (
            self.latency_skip_compensation_steps / self.policy_action_hz
            if self.policy_action_hz > 0
            else 0.0
        )
        return (
            float(now_monotonic)
            + self.estimated_actuator_delay_s
            + fixed_compensation_s
        )

    @staticmethod
    def _first_future_target_index(
        targets: list[DecodedQueuedAction], execution_time: float
    ) -> int | None:
        for index, target in enumerate(targets):
            if target.target_monotonic + 1e-9 >= execution_time:
                return index
        return None

    @staticmethod
    def _gripper_slot(protocol: PolicyProtocol, arm_index: int) -> int:
        return arm_index * (10 if protocol.schema == "delivery" else 7) + (
            9 if protocol.schema == "delivery" else 6
        )

    @staticmethod
    def _opening_to_wire(opening: float, semantics: str) -> float:
        opening = float(np.clip(opening, 0.0, 1.0))
        if semantics == GRIPPER_OPENING_FRACTION:
            return opening
        if semantics == GRIPPER_CLOSED_FRACTION:
            return 1.0 - opening
        if semantics == GRIPPER_OPENING_METRES:
            return opening * GRIPPER_MAX_M
        raise ExecutionBlocked(f"unsupported gripper semantics: {semantics!r}")

    def _confirmed_gripper_desired(
        self,
        side: str,
        desired: float,
        previous: float,
    ) -> float:
        """Apply open/closed hysteresis and consecutive-command confirmation."""
        hysteresis = self.gripper_hysteresis
        latch = self._gripper_extreme_latch.get(side)
        extreme: str | None = None
        if latch == "closed" and desired <= min(1.0, 2.0 * hysteresis):
            extreme = "closed"
        elif latch == "open" and desired >= max(0.0, 1.0 - 2.0 * hysteresis):
            extreme = "open"
        elif desired <= hysteresis:
            extreme = "closed"
        elif desired >= 1.0 - hysteresis:
            extreme = "open"

        if extreme is None:
            self._gripper_extreme_candidate[side] = None
            self._gripper_extreme_count[side] = 0
            self._gripper_extreme_latch[side] = None
            return desired
        if latch == extreme:
            return 0.0 if extreme == "closed" else 1.0

        if self._gripper_extreme_candidate.get(side) == extreme:
            count = self._gripper_extreme_count.get(side, 0) + 1
        else:
            self._gripper_extreme_candidate[side] = extreme
            count = 1
        self._gripper_extreme_count[side] = count
        if count < self.gripper_confirm_steps:
            return previous
        self._gripper_extreme_latch[side] = extreme
        self._gripper_extreme_candidate[side] = None
        self._gripper_extreme_count[side] = 0
        return 0.0 if extreme == "closed" else 1.0

    def _filter_gripper_target(
        self,
        queued: DecodedQueuedAction,
        qpos_m: np.ndarray,
        protocol: PolicyProtocol,
    ) -> DecodedQueuedAction:
        """Filter gripper opening independently from joint/EEF trajectory blend."""
        if queued.hold:
            return queued
        output = queued.absolute_target.copy()
        sides = ("left", "right") if protocol.arm_mode == "bimanual" else (protocol.arm_side,)
        tolerance = float(
            getattr(
                self.args,
                "gripper_range_tolerance",
                DEFAULT_GRIPPER_RANGE_TOLERANCE,
            )
        )
        for arm_index, side in enumerate(sides):
            slot = self._gripper_slot(protocol, arm_index)
            desired = _opening_fraction(
                output[slot],
                semantics=protocol.gripper_semantics,
                tolerance=tolerance,
            )
            qpos_slice = np.asarray(qpos_m, dtype=np.float64)[
                arm_index * 7 : (arm_index + 1) * 7
            ]
            current = float(np.clip(qpos_slice[6] / GRIPPER_MAX_M, 0.0, 1.0))
            previous = self._filtered_gripper_opening.get(side, current)
            confirmed = self._confirmed_gripper_desired(side, desired, previous)
            filtered = previous + self.gripper_lowpass_alpha * (confirmed - previous)
            if protocol.schema == "delivery":
                max_step_value = getattr(
                    self.args, "max_gripper_step", DEFAULT_MAX_GRIPPER_STEP
                )
            else:
                max_step_value = getattr(self.args, "max_joint_gripper_step", None)
                if max_step_value is None:
                    legacy_m = getattr(self.args, "max_joint_gripper_step_m", None)
                    max_step_value = (
                        float(legacy_m) / GRIPPER_MAX_M
                        if legacy_m is not None
                        else 0.25
                    )
            max_step = float(max_step_value)
            filtered = float(
                np.clip(filtered, current - max_step, current + max_step)
            )
            filtered = float(np.clip(filtered, 0.0, 1.0))
            self._filtered_gripper_opening[side] = filtered
            output[slot] = self._opening_to_wire(filtered, protocol.gripper_semantics)
        return replace(queued, absolute_target=output)

    def accept_inference_result(
        self,
        result: Any,
        launch: InferenceLaunch,
        protocol: PolicyProtocol,
        *,
        arrived_at: float | None = None,
        arrived_monotonic: float | None = None,
        min_steps_override: int | None = None,
        skip_steps_override: int | None = None,
        blend_steps_override: int | None = None,
    ) -> bool:
        """Validate a completed result and atomically replace the active queue."""
        arrived_at = time.time() if arrived_at is None else float(arrived_at)
        arrived_monotonic = (
            time.monotonic() if arrived_monotonic is None else float(arrived_monotonic)
        )
        latency_s = arrived_monotonic - float(launch.captured_monotonic)
        self.inference_generation = launch.generation
        self.inference_capture_at = launch.captured_at
        self.inference_capture_monotonic = launch.captured_monotonic
        self.inference_launch_at = launch.launched_at
        self.inference_arrival_at = arrived_at
        self.inference_arrival_monotonic = arrived_monotonic
        self.inference_latency_s = latency_s
        self.inference_old_remaining = len(self.pending_actions)
        self.inference_skip_steps = 0
        self.inference_blend_steps = 0
        if not math.isfinite(latency_s) or latency_s < 0:
            return self._reject_result(
                launch.generation, f"invalid capture-to-arrival latency {latency_s!r}", arrived_at
            )
        if not isinstance(result, dict) or "actions" not in result:
            return self._reject_result(launch.generation, "result has no actions", arrived_at)

        control = result.get("execution_control")
        try:
            revision, authorization_deadline = self._candidate_execution_control(
                control, arrived_monotonic=arrived_monotonic
            )
        except PermissionError as exc:
            self.discard_pending_actions(str(exc))
            state = "shadow" if "shadow" in str(exc) else "blocked"
            return self._block(state, str(exc))
        except ExecutionBlocked as exc:
            return self._reject_result(launch.generation, str(exc), arrived_at)

        max_action_age_s = float(getattr(self.args, "max_action_age_s", 2.0))
        stale_images = {
            key: launch.captured_at - float(timestamp)
            for key, timestamp in launch.image_timestamps.items()
            if launch.captured_at - float(timestamp) > max_action_age_s
        }
        if stale_images:
            return self._reject_result(
                launch.generation, f"launch used stale camera frames: {stale_images}", arrived_at
            )

        try:
            values = _finite_action_chunk(result.get("actions"), protocol.action_dim)
        except ExecutionBlocked as exc:
            return self._reject_result(launch.generation, str(exc), arrived_at)
        minimum = self.min_action_chunk_steps if min_steps_override is None else int(min_steps_override)
        if len(values) < minimum:
            return self._reject_result(
                launch.generation,
                f"action chunk has {len(values)} rows; client requires at least {minimum}",
                arrived_at,
            )
        try:
            anchor, decoded = decode_action_queue(
                values,
                protocol,
                launch.raw_delivery_state,
                launch.qpos_m,
                steps=None,
                generation=launch.generation,
                observation_capture_monotonic=launch.captured_monotonic,
                action_hz=self.policy_action_hz,
                gripper_range_tolerance=float(
                    getattr(
                        self.args,
                        "gripper_range_tolerance",
                        DEFAULT_GRIPPER_RANGE_TOLERANCE,
                    )
                ),
            )
        except ExecutionBlocked as exc:
            return self._reject_result(launch.generation, str(exc), arrived_at)
        if skip_steps_override is not None:
            skip_steps = max(0, int(skip_steps_override))
            if skip_steps >= len(decoded):
                self.inference_skip_steps = skip_steps
                return self._reject_result(
                    launch.generation,
                    f"result fully stale: skip={skip_steps}, chunk={len(decoded)}",
                    arrived_at,
                )
        else:
            execution_time = self._estimated_execution_time(arrived_monotonic)
            future_index = self._first_future_target_index(decoded, execution_time)
            if future_index is None:
                self.inference_skip_steps = len(decoded)
                return self._reject_result(
                    launch.generation,
                    f"result fully stale at execution_time={execution_time:.6f}; "
                    f"last_target={decoded[-1].target_monotonic:.6f}",
                    arrived_at,
                )
            skip_steps = future_index
        self.inference_skip_steps = skip_steps
        fresh_actions = decoded[skip_steps:]
        old_actions = list(self.pending_actions)
        if not old_actions and self.last_safe_target is not None and self.hold_active:
            old_actions = [self.last_safe_target]
        blend_steps = self.blend_steps if blend_steps_override is None else int(blend_steps_override)
        if old_actions and blend_steps:
            if len(fresh_actions) < blend_steps:
                return self._reject_result(
                    launch.generation,
                    f"only {len(fresh_actions)} fresh rows remain after skip={skip_steps}; "
                    f"need {blend_steps} blend rows",
                    arrived_at,
                )
            try:
                candidate = blend_absolute_trajectories(
                    old_actions,
                    fresh_actions,
                    protocol,
                    blend_steps=blend_steps,
                )
            except ExecutionBlocked as exc:
                return self._reject_result(launch.generation, str(exc), arrived_at)
            self.inference_blend_steps = blend_steps
        else:
            candidate = list(fresh_actions)
        if not candidate:
            return self._reject_result(launch.generation, "result has no executable rows", arrived_at)

        # All decoding/blending/authorization checks finished. The control thread
        # performs one atomic list replacement; inference never mutates this queue.
        self.pending_actions = candidate
        self.active_generation = launch.generation
        self.control_revision = revision
        self.authorization_deadline_monotonic = authorization_deadline
        self.queue_control = control
        self.queue_anchor_state = anchor.tolist()
        self.queue_anchor_qpos_m = np.asarray(launch.qpos_m, dtype=np.float64).tolist()
        self.queue_anchor_at = launch.captured_at
        self.queue_loaded_at = arrived_at
        self.queue_image_timestamps = dict(launch.image_timestamps)
        self.queued_action_index = candidate[0].queue_index
        self.last_action_chunk_steps = len(values)
        self.last_composed_action = candidate[0].wire_action.tolist()
        self.last_composed_action_at = arrived_at
        self.last_decoded_absolute_target = self._target_telemetry(
            protocol, candidate[0].absolute_target
        )
        self.queue_underrun = False
        self.hold_active = False
        self.hold_started_at = None
        self.waiting_fresh_after_enable = False
        self.rejected_result = None
        if bool(getattr(self.args, "allow_execution", False)):
            self.state = "ready"
            self.blocked_reason = ""
        else:
            self.state = "client_disabled"
            self.blocked_reason = "local --allow-execution is absent"
        return True

    def reject_inference_completion(self, completion: InferenceCompletion) -> bool:
        reason = f"inference failed: {completion.error}"
        self.inference_generation = completion.launch.generation
        self.inference_capture_at = completion.launch.captured_at
        self.inference_capture_monotonic = completion.launch.captured_monotonic
        self.inference_launch_at = completion.launch.launched_at
        self.inference_arrival_at = completion.arrived_at
        self.inference_arrival_monotonic = completion.arrived_monotonic
        self.inference_latency_s = (
            completion.arrived_monotonic - completion.launch.captured_monotonic
        )
        self.inference_old_remaining = len(self.pending_actions)
        return self._reject_result(
            completion.launch.generation, reason, completion.arrived_at
        )

    def queue_result(
        self,
        result: dict[str, Any],
        raw_delivery_state: np.ndarray,
        qpos_m: np.ndarray,
        protocol: PolicyProtocol,
        image_timestamps: dict[str, float],
        infer_elapsed_s: float,
    ) -> int:
        """Compatibility helper for tests/callers; production uses async completion."""
        now = time.time()
        now_monotonic = time.monotonic()
        launch = InferenceLaunch(
            generation=self.allocate_inference_generation(),
            captured_at=now - float(infer_elapsed_s),
            captured_monotonic=now_monotonic - float(infer_elapsed_s),
            launched_at=now - float(infer_elapsed_s),
            launched_monotonic=now_monotonic - float(infer_elapsed_s),
            raw_delivery_state=np.asarray(raw_delivery_state).copy(),
            qpos_m=np.asarray(qpos_m).copy(),
            image_timestamps={key: float(value) for key, value in image_timestamps.items()},
        )
        accepted = self.accept_inference_result(
            result,
            launch,
            protocol,
            arrived_at=now,
            min_steps_override=1,
            skip_steps_override=0,
            blend_steps_override=0,
        )
        return len(self.pending_actions) if accepted else 0

    def discard_pending_actions(self, reason: str) -> None:
        if self.pending_actions:
            self.dropped_action_count += len(self.pending_actions)
            self.pending_actions.clear()
        self.queued_action_index = None
        self.last_queue_drop_reason = reason[:500]

    def _mark_queue_underrun(self, *, holding: bool) -> bool:
        if not self.queue_underrun:
            self.queue_underrun = True
            self.queue_underrun_count += 1
            self.queue_underrun_at = time.time()
        if holding:
            if not self.hold_active:
                self.hold_started_at = time.time()
            self.hold_active = True
            self.state = "holding"
            self.blocked_reason = (
                "action queue underrun: holding last safe absolute target until a valid plan arrives"
            )
            return True
        return self._block(
            "blocked",
            "action queue underrun: no last safe target is available for hold",
        )

    def execute_next(
        self,
        raw_delivery_state: np.ndarray,
        qpos_m: np.ndarray,
        protocol: PolicyProtocol,
        *,
        feedback_captured_at: float | None = None,
    ) -> bool:
        """Execute one time-selected target or hold the last safe absolute target."""
        if not bool(getattr(self.args, "allow_execution", False)):
            return self._block("client_disabled", "local --allow-execution is absent")
        if self.state in {"shadow", "client_disabled"}:
            return False
        if self.state == "blocked":
            return False

        now_monotonic = time.monotonic()
        if (
            self.authorization_deadline_monotonic is None
            or now_monotonic >= self.authorization_deadline_monotonic
        ):
            self.discard_pending_actions("execution authorization expired")
            return self._block("blocked", "execution authorization expired")

        feedback_at = time.time() if feedback_captured_at is None else float(feedback_captured_at)
        self.last_feedback_at = feedback_at
        feedback_age = time.time() - feedback_at
        max_feedback_age_s = float(
            getattr(self.args, "max_feedback_age_s", DEFAULT_FEEDBACK_MAX_AGE_S)
        )
        if feedback_age < -1.0 or feedback_age > max_feedback_age_s:
            return self._block(
                "blocked",
                f"Piper feedback age {feedback_age:.3f}s exceeds {max_feedback_age_s:.3f}s",
            )

        sides = ("left", "right") if protocol.arm_mode == "bimanual" else (protocol.arm_side,)
        if set(self.pipers) != set(sides):
            return self._block(
                "blocked",
                f"connected Piper sides {sorted(self.pipers)} do not match policy sides {list(sides)}",
            )

        execution_time = self._estimated_execution_time(now_monotonic)
        if self.pending_actions:
            future_index = self._first_future_target_index(
                self.pending_actions, execution_time
            )
            if future_index is None:
                self.dropped_action_count += len(self.pending_actions)
                self.last_queue_drop_reason = (
                    "active timed plan exhausted before the estimated actuator execution time"
                )
                self.pending_actions.clear()
                self.queued_action_index = None
            elif future_index:
                self.dropped_action_count += future_index
                self.last_queue_drop_reason = (
                    f"dropped {future_index} targets older than execution_time={execution_time:.6f}"
                )
                del self.pending_actions[:future_index]

        holding = False
        if self.pending_actions:
            queued = self.pending_actions[0]
        else:
            if self.waiting_fresh_after_enable:
                self.state = "armed"
                self.blocked_reason = "Piper enabled; waiting for a fresh inference result"
                return False
            if self.last_safe_target is None:
                return self._mark_queue_underrun(holding=False)
            self._mark_queue_underrun(holding=True)
            queued = replace(self.last_safe_target, hold=True)
            holding = True

        filter_snapshot = (
            dict(self._filtered_gripper_opening),
            dict(self._gripper_extreme_candidate),
            dict(self._gripper_extreme_count),
            dict(self._gripper_extreme_latch),
        )
        try:
            statuses = {side: arm_status_dict(self.pipers[side]) for side in sides}
            self.robot_status = statuses if protocol.arm_mode == "bimanual" else statuses[sides[0]]
            bad_status = {
                side: status
                for side, status in statuses.items()
                if status["arm_status"] != 0 or status["err_code"] != 0
            }
            if bad_status:
                raise ExecutionBlocked(f"Piper status is not normal: {bad_status}")

            waiting_for_hold = []
            for index, side in enumerate(sides):
                hold_target = self.arm_hold_targets.get(side)
                if hold_target is None:
                    continue
                qpos_slice = np.asarray(qpos_m, dtype=np.float64)[index * 7 : (index + 1) * 7]
                hold_age = time.monotonic() - self.arm_hold_started_at[side]
                hold_error = float(np.max(np.abs(qpos_slice[:6] - hold_target)))
                if (
                    hold_age < float(getattr(self.args, "arm_settle_s", 0.0))
                    or hold_error > float(getattr(self.args, "arm_hold_tolerance_rad", 0.02))
                ):
                    waiting_for_hold.append(
                        f"{side}: age={hold_age:.2f}s joint_error={hold_error:.4f}rad"
                    )
                else:
                    self.arm_hold_targets.pop(side, None)
                    self.arm_hold_started_at.pop(side, None)
            if waiting_for_hold:
                return self._block(
                    "armed",
                    "waiting for enable hold to settle: " + "; ".join(waiting_for_hold),
                )

            queued = self._filter_gripper_target(queued, qpos_m, protocol)
            prepared: dict[str, tuple[Any, ...]] = {}
            for index, side in enumerate(sides):
                qpos_slice = np.asarray(qpos_m)[index * 7 : (index + 1) * 7]
                if protocol.schema == "delivery":
                    current_state = np.asarray(raw_delivery_state)[index * 10 : (index + 1) * 10]
                    target = queued.absolute_target[index * 10 : (index + 1) * 10]
                    checked = _check_delivery_absolute_target(
                        current_state,
                        float(qpos_slice[6]),
                        target,
                        gripper_semantics=protocol.gripper_semantics,
                        max_translation_step_m=float(
                            getattr(
                                self.args,
                                "max_translation_step_m",
                                DEFAULT_MAX_TRANSLATION_STEP_M,
                            )
                        ),
                        max_rotation_step_rad=float(
                            getattr(
                                self.args,
                                "max_rotation_step_rad",
                                DEFAULT_MAX_ROTATION_STEP_RAD,
                            )
                        ),
                        max_gripper_step=float(
                            getattr(self.args, "max_gripper_step", DEFAULT_MAX_GRIPPER_STEP)
                        ),
                        gripper_range_tolerance=float(
                            getattr(
                                self.args,
                                "gripper_range_tolerance",
                                DEFAULT_GRIPPER_RANGE_TOLERANCE,
                            )
                        ),
                        workspace_x=tuple(
                            getattr(self.args, "workspace_x", DEFAULT_WORKSPACE_X_M)
                        ),
                        workspace_y=tuple(
                            getattr(self.args, "workspace_y", DEFAULT_WORKSPACE_Y_M)
                        ),
                        workspace_z=tuple(
                            getattr(self.args, "workspace_z", DEFAULT_WORKSPACE_Z_M)
                        ),
                    )
                    target_xyz, target_rpy_deg, target_gripper_m, _ = checked
                    if self.ik_solver is None:
                        self.ik_solver = PiperContinuousIK()
                    target_joints = self.ik_solver.solve(
                        qpos_slice[:6],
                        target_xyz,
                        target_rpy_deg,
                        max_joint_step_rad=float(getattr(self.args, "ik_max_joint_step_rad", DEFAULT_IK_MAX_JOINT_STEP_RAD)),
                        position_tolerance_m=float(getattr(self.args, "ik_position_tolerance_m", DEFAULT_IK_POSITION_TOLERANCE_M)),
                        rotation_tolerance_rad=float(getattr(self.args, "ik_rotation_tolerance_rad", DEFAULT_IK_ROTATION_TOLERANCE_RAD)),
                        max_nfev=int(getattr(self.args, "ik_max_nfev", DEFAULT_IK_MAX_NFEV)),
                    )
                    prepared[side] = (target_joints, target_gripper_m)
                elif protocol.schema == "joint":
                    target = queued.absolute_target[index * 7 : (index + 1) * 7]
                    prepared[side] = build_checked_joint_target(
                        qpos_slice,
                        target,
                        max_joint_step_rad=float(getattr(self.args, "max_joint_step_rad", 0.3)),
                        max_gripper_step=getattr(self.args, "max_joint_gripper_step", None),
                        max_gripper_step_m=getattr(self.args, "max_joint_gripper_step_m", None),
                        gripper_range_tolerance=float(
                            getattr(
                                self.args,
                                "gripper_range_tolerance",
                                DEFAULT_GRIPPER_RANGE_TOLERANCE,
                            )
                        ),
                        gripper_semantics=protocol.gripper_semantics,
                    )
                else:
                    raise ExecutionBlocked(f"unsupported execution schema: {protocol.schema}")

            missing_enabled = [side for side in sides if side not in self.robot_enabled]
            if missing_enabled:
                for side in missing_enabled:
                    side_index = sides.index(side)
                    hold_qpos = np.asarray(qpos_m, dtype=np.float64)[side_index * 7 : (side_index + 1) * 7]
                    self._enable_robot(side, self.pipers[side], hold_qpos)
                (
                    self._filtered_gripper_opening,
                    self._gripper_extreme_candidate,
                    self._gripper_extreme_count,
                    self._gripper_extreme_latch,
                ) = filter_snapshot
                self.discard_pending_actions(
                    "Piper enabled; discarded pre-enable trajectory and waiting for fresh inference"
                )
                self.waiting_fresh_after_enable = True
                self.state = "armed"
                self.blocked_reason = "Piper enabled; waiting for a fresh inference result"
                return False

            # Blend and hold never bypass safety: both arms are fully prevalidated
            # against the same fresh feedback before either arm receives a command.
            for side in sides:
                piper = self.pipers[side]
                target_joints, target_gripper_m = prepared[side]
                raw_joints = np.rint(target_joints * RAD_FACTOR).astype(np.int64)
                piper.ModeCtrl(
                    0x01, 0x01, int(getattr(self.args, "speed_pct", 10)), 0x00
                )
                piper.JointCtrl(*map(int, raw_joints))
                raw_gripper = round(target_gripper_m * GRIPPER_FACTOR)
                piper.GripperCtrl(
                    int(raw_gripper), int(getattr(self.args, "gripper_effort", 1000)), 0x01, 0
                )
        except ExecutionBlocked as exc:
            (
                self._filtered_gripper_opening,
                self._gripper_extreme_candidate,
                self._gripper_extreme_count,
                self._gripper_extreme_latch,
            ) = filter_snapshot
            self.discard_pending_actions(str(exc))
            return self._block("blocked", str(exc))
        except Exception as exc:
            (
                self._filtered_gripper_opening,
                self._gripper_extreme_candidate,
                self._gripper_extreme_count,
                self._gripper_extreme_latch,
            ) = filter_snapshot
            logging.exception("robot command failed")
            self.discard_pending_actions(f"robot command failed: {exc}")
            return self._block("blocked", f"robot command failed: {exc}")

        if not holding:
            self.pending_actions.pop(0)
            self.last_safe_target = replace(queued, hold=False)
        self.last_queued_action_index = queued.queue_index
        self.queued_action_index = self.pending_actions[0].queue_index if self.pending_actions else None
        self.last_wire_action = queued.wire_action.tolist()
        self.last_decoded_absolute_target = self._target_telemetry(
            protocol, queued.absolute_target
        )
        self.current_timed_target = self._timed_target_telemetry(
            replace(queued, hold=holding), now_monotonic=now_monotonic
        )
        self.last_command_at = time.time()
        if holding:
            self.hold_count += 1
            self.state = "holding"
        else:
            self.state = "executing"
            self.blocked_reason = ""
        return True

    def process(
        self,
        result: dict[str, Any],
        delivery_state: np.ndarray,
        qpos: np.ndarray,
        protocol: PolicyProtocol,
        image_timestamps: dict[str, float],
        infer_elapsed_s: float,
    ) -> bool:
        self.queue_result(
            result, delivery_state, qpos, protocol, image_timestamps, infer_elapsed_s
        )
        return self.execute_next(
            delivery_state, qpos, protocol, feedback_captured_at=time.time()
        )


def build_observation(
    *,
    delivery_state: np.ndarray,
    qpos: np.ndarray,
    protocol: PolicyProtocol,
    images: dict[str, np.ndarray],
    image_timestamps: dict[str, float],
    instruction: str,
    source_name: str,
    args: argparse.Namespace,
    execution: ExecutionController,
    captured_at: float | None = None,
    captured_monotonic: float | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    captured_at = time.time() if captured_at is None else float(captured_at)
    captured_monotonic = (
        time.monotonic()
        if captured_monotonic is None
        else float(captured_monotonic)
    )
    state = policy_observation_state(delivery_state, qpos, protocol)
    if state.shape != (protocol.state_dim,) or not np.all(np.isfinite(state)):
        raise RuntimeError(
            f"{protocol.arm_mode} {protocol.schema} observation state must be finite "
            f"{protocol.state_dim}D, got {state.shape}"
        )
    if protocol.arm_mode == "bimanual":
        observation_images = {
            key: np.asarray(images[key], dtype=np.uint8) for key in protocol.camera_keys
        }
        can_names = {"left": args.left_can, "right": args.right_can}
        camera_devices = {
            "cam_high": str(args.cam_high_device),
            "cam_left_wrist": str(args.cam_left_wrist_device),
            "cam_right_wrist": str(args.cam_right_wrist_device),
        }
    else:
        wrist_key = next(key for key in protocol.camera_keys if "wrist" in key)
        observation_images = {
            "cam_high": np.asarray(images["cam_high"], dtype=np.uint8),
            wrist_key: np.asarray(images["cam_wrist"], dtype=np.uint8),
        }
        can_names = {protocol.arm_side: args.can}
        camera_devices = {
            "cam_high": str(args.cam_high_device),
            wrist_key: str(args.cam_wrist_device),
        }
    return {
        "state": state,
        "images": observation_images,
        "prompt": instruction,
        "client_metadata": {
            "captured_at": captured_at,
            "captured_monotonic": captured_monotonic,
            "source_name": source_name,
            "arm_mode": protocol.arm_mode,
            "arm_side": protocol.arm_side,
            "can_names": can_names,
            "camera_devices": camera_devices,
            "image_captured_at": {
                key: float(image_timestamps["cam_wrist"] if protocol.arm_mode == "single" and "wrist" in key else image_timestamps[key])
                for key in protocol.camera_keys
            },
            # Preserve old single-arm telemetry fields.
            "can_name": next(iter(can_names.values())) if protocol.arm_mode == "single" else "",
            "cam_high_device": str(args.cam_high_device),
            "cam_wrist_device": str(args.cam_wrist_device) if protocol.arm_mode == "single" else "",
            "policy_schema": protocol.schema,
            "policy_action_semantics": protocol.action_semantics,
            "policy_gripper_semantics": protocol.gripper_semantics,
            "policy_state_gripper_semantics": protocol.state_gripper_semantics,
            "policy_contract_version": protocol.contract_version,
            "policy_gripper_semantics_explicit": protocol.metadata_gripper_semantics_explicit,
            **(
                execution.metadata()
                if execution_metadata is None
                else execution_metadata
            ),
        },
    }


def print_result(
    count: int,
    state: np.ndarray,
    qpos: np.ndarray,
    protocol: PolicyProtocol,
    result: dict[str, Any],
    elapsed_s: float,
    execution: ExecutionController,
    command_sent: bool,
) -> None:
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    first = actions[0] if actions.ndim > 1 and len(actions) else actions
    try:
        command_action, used_steps = aggregate_action_chunk(
            actions, protocol, execution.action_chunk_steps
        )
    except ExecutionBlocked as exc:
        command_action, used_steps = np.asarray([], dtype=np.float64), 0
        logging.warning("Cannot summarize command action: %s", exc)
    control = result.get("execution_control", {})
    if protocol.schema == "delivery":
        state_summary = " ".join(
            f"{side}_eef={np.array2string(state[i * 10:i * 10 + 3], precision=4)}"
            for i, side in enumerate(("left", "right") if protocol.arm_mode == "bimanual" else (protocol.arm_side,))
        )
    else:
        state_summary = " ".join(
            f"{side}_joints={np.array2string(qpos[i * 7:i * 7 + 6], precision=4)}"
            for i, side in enumerate(("left", "right") if protocol.arm_mode == "bimanual" else (protocol.arm_side,))
        )
    print(
        f"infer={count} mode={protocol.arm_mode} schema={protocol.schema} elapsed={elapsed_s * 1000:.1f}ms "
        f"{state_summary} actions={actions.shape}\n"
        f"  first_action={np.array2string(first, precision=5, suppress_small=True)}\n"
        f"  command_action[{used_steps} steps]={np.array2string(command_action, precision=5, suppress_small=True)}\n"
        f"  queue_last={execution.last_queued_action_index} queue_next={execution.queued_action_index} "
        f"remaining={execution.pending_action_count} decoded={execution.last_decoded_absolute_target}\n"
        f"  server_mode={control.get('mode', 'missing')} "
        f"local_allow={getattr(execution.args, 'allow_execution', False)} "
        f"client_state={execution.state} command_sent={command_sent} "
        f"reason={execution.blocked_reason or '-'}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    for key in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        if args.host not in entries:
            entries.append(args.host)
        os.environ[key] = ",".join(entries)

    source_name = args.source_name or socket.gethostname()
    if args.arm_mode == "bimanual":
        logging.info(
            "Connecting Piper feedback: left=%s right=%s ...",
            args.left_can,
            args.right_can,
        )
        pipers = {"left": connect_piper(args.left_can), "right": connect_piper(args.right_can)}
        camera_ids = {
            "cam_high": args.cam_high_device,
            "cam_left_wrist": args.cam_left_wrist_device,
            "cam_right_wrist": args.cam_right_wrist_device,
        }
    else:
        logging.info("Connecting Piper feedback on %s ...", args.can)
        pipers = {args.arm_side: connect_piper(args.can)}
        camera_ids = {"cam_high": args.cam_high_device, "cam_wrist": args.cam_wrist_device}

    execution = ExecutionController(pipers, args)
    cameras = CameraCapture(
        cam_ids=camera_ids,
        fps=args.camera_fps,
        image_hw=IMAGE_HW,
        capture_hw=CAMERA_SOURCE_HW,
        parallel_reads=True,
    )
    worker = AsyncPolicyInference()
    policy = None
    protocol = None
    count = 0
    command_count = 0
    last_reconnect_attempt = 0.0
    control_period = 1.0 / float(getattr(args, "control_hz", DEFAULT_ACTION_HZ))
    try:
        cameras.open()
        camera_checks = cameras.verify()
        for key, info in camera_checks.items():
            selected_device = str(
                info.get("selected_device")
                or info.get("configured_device")
                or camera_ids[key]
            )
            video_device = str(info.get("video_device") or selected_device)
            if key == "cam_high":
                args.cam_high_device = selected_device
            elif key == "cam_wrist":
                args.cam_wrist_device = selected_device
            elif key == "cam_left_wrist":
                args.cam_left_wrist_device = selected_device
            elif key == "cam_right_wrist":
                args.cam_right_wrist_device = selected_device
            logging.info(
                "Camera %s: %s selected=%s video=%s shape=%s latency=%sms",
                key,
                "OK" if info["ok"] else "FAIL",
                selected_device,
                video_device,
                info["shape"],
                info["latency_ms"],
            )
        logging.warning(
            "%s %s client: control=%.3g Hz inference=%.3g Hz expected_chunk=%d "
            "minimum_chunk=%d. Robot commands still require Dashboard EXECUTE.",
            "EXECUTION-CAPABLE" if args.allow_execution else "SHADOW-ONLY",
            args.arm_mode,
            args.control_hz,
            args.hz,
            DEFAULT_OPENPI_CHUNK_STEPS,
            args.min_action_chunk_steps,
        )

        next_control_at = time.monotonic()
        launch_schedule = PeriodicSchedule(args.hz, next_at=next_control_at)
        while True:
            tick_started = time.monotonic()
            execution.record_control_tick(overrun=tick_started > next_control_at + control_period)
            command_sent = False
            completion: InferenceCompletion | None = None
            try:
                if (
                    policy is None
                    and not worker.in_flight
                    and (protocol is None or not execution.pending_action_count)
                ):
                    if tick_started - last_reconnect_attempt >= args.reconnect_delay:
                        last_reconnect_attempt = tick_started
                        try:
                            policy, protocol = connect_policy(
                                args.host, args.port, args.arm_side, args.arm_mode
                            )
                            execution.configure_protocol(protocol)
                            control_period = 1.0 / execution.control_hz
                            launch_schedule = PeriodicSchedule(
                                execution.inference_hz, next_at=tick_started
                            )
                        except Exception as exc:
                            execution._block("blocked", f"policy connection unavailable: {exc}")
                            logging.warning("Policy connection unavailable: %s", exc)

                if protocol is not None:
                    sides = (
                        ("left", "right") if args.arm_mode == "bimanual" else (args.arm_side,)
                    )
                    states = {
                        side: read_output_state(
                            pipers[side], max_feedback_age_s=args.max_feedback_age_s
                        )
                        for side in sides
                    }
                    delivery_state = np.concatenate([states[side][0] for side in sides]).astype(
                        np.float32
                    )
                    qpos = np.concatenate([states[side][1] for side in sides]).astype(np.float32)
                    observation_captured_at = time.time()
                    observation_captured_monotonic = time.monotonic()

                    completion = worker.poll()
                    if completion is not None:
                        if completion.error is not None:
                            execution.reject_inference_completion(completion)
                            if policy is not None:
                                close_policy(policy)
                            policy = None
                            logging.warning(
                                "Inference generation %d failed; old queue remains active: %s",
                                completion.launch.generation,
                                completion.error,
                            )
                        else:
                            accepted = execution.accept_inference_result(
                                completion.result,
                                completion.launch,
                                protocol,
                                arrived_at=completion.arrived_at,
                                arrived_monotonic=completion.arrived_monotonic,
                            )
                            logging.info(
                                "Inference generation=%d arrival latency=%.3fs skip=%d "
                                "blend=%d old_remaining=%d accepted=%s queue=%d rejected=%s",
                                completion.launch.generation,
                                execution.inference_latency_s or 0.0,
                                execution.inference_skip_steps,
                                execution.inference_blend_steps,
                                execution.inference_old_remaining,
                                accepted,
                                execution.pending_action_count,
                                execution.rejected_result,
                            )
                            count += 1
                            if args.once and accepted:
                                return

                    # This is the only robot command path and runs every control tick.
                    command_sent = execution.execute_next(
                        delivery_state,
                        qpos,
                        protocol,
                        feedback_captured_at=observation_captured_at,
                    )
                    if command_sent:
                        command_count += 1
                        if args.max_commands is not None and command_count >= args.max_commands:
                            logging.warning(
                                "Reached --max-commands=%d; stopping after the checked command.",
                                args.max_commands,
                            )
                            return

                    if launch_schedule.due(tick_started):
                        if worker.in_flight:
                            execution.record_launch_deferred()
                        elif policy is not None:
                            # The control thread snapshots the 20 Hz robot state and
                            # immediately returns to scheduling. Camera I/O, observation
                            # formatting, transport, and policy inference all run on the
                            # single worker thread.
                            captured_at = observation_captured_at
                            captured_monotonic = observation_captured_monotonic
                            delivery_anchor = delivery_state.copy()
                            qpos_anchor = qpos.copy()
                            execution_metadata = execution.metadata()
                            generation = execution.allocate_inference_generation()
                            launch = InferenceLaunch(
                                generation=generation,
                                captured_at=captured_at,
                                captured_monotonic=captured_monotonic,
                                launched_at=time.time(),
                                launched_monotonic=time.monotonic(),
                                raw_delivery_state=delivery_anchor,
                                qpos_m=qpos_anchor,
                                image_timestamps={},
                            )

                            def capture_and_infer(
                                *,
                                policy_ref=policy,
                                protocol_ref=protocol,
                                delivery_ref=delivery_anchor,
                                qpos_ref=qpos_anchor,
                                metadata_ref=execution_metadata,
                                launch_ref=launch,
                            ) -> InferenceWorkerResult:
                                images, image_timestamps = cameras.read()
                                observation = build_observation(
                                    delivery_state=delivery_ref,
                                    qpos=qpos_ref,
                                    protocol=protocol_ref,
                                    images=images,
                                    image_timestamps=image_timestamps,
                                    instruction=args.instruction,
                                    source_name=source_name,
                                    args=args,
                                    execution=execution,
                                    captured_at=launch_ref.captured_at,
                                    captured_monotonic=launch_ref.captured_monotonic,
                                    execution_metadata=metadata_ref,
                                )
                                return InferenceWorkerResult(
                                    result=policy_ref.infer(observation),
                                    image_timestamps=image_timestamps,
                                )

                            if worker.launch_callable(capture_and_infer, launch):
                                execution.record_inference_launch(launch)
                            else:  # defensive; the control thread owns launch()
                                execution.record_launch_deferred()
                        elif not execution.pending_action_count:
                            # Reconnect only after the active queue reaches hold/
                            # underrun; it cannot interrupt a live trajectory.
                            logging.info("Retrying policy connection after queue underrun")

                # Launch retries happen on the next tick; no synchronous infer call.
            except ExecutionBlocked as exc:
                execution._block("blocked", str(exc))
                logging.warning("20 Hz feedback/safety check blocked: %s", exc)
            except Exception as exc:
                execution._block("blocked", f"control tick failed: {exc}")
                logging.exception("20 Hz control tick failed")

            next_control_at += control_period
            now = time.monotonic()
            if now > next_control_at:
                missed = int(math.floor((now - next_control_at) / control_period)) + 1
                next_control_at += missed * control_period
            sleep_s = next_control_at - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        logging.info("Stopped; no further robot commands will be published.")
    finally:
        worker.shutdown()
        close_policy(policy)
        cameras.close()
        for piper in pipers.values():
            piper.DisconnectPort()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("BIMANUAL_VLA_POLICY_HOST", DEFAULT_POLICY_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BIMANUAL_VLA_POLICY_PORT", DEFAULT_POLICY_PORT)))
    parser.add_argument("--arm-mode", choices=("single", "bimanual"), default="single")
    parser.add_argument("--can", default=DEFAULT_CAN, help="single-arm CAN interface")
    parser.add_argument("--left-can", default=DEFAULT_LEFT_CAN)
    parser.add_argument("--right-can", default=DEFAULT_RIGHT_CAN)
    parser.add_argument("--arm-side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--cam-high-device", default=DEFAULT_HIGH_DEVICE)
    parser.add_argument("--cam-wrist-device", default=DEFAULT_WRIST_DEVICE, help="single-arm wrist camera")
    parser.add_argument("--cam-left-wrist-device", default=DEFAULT_LEFT_WRIST_DEVICE)
    parser.add_argument("--cam-right-wrist-device", default=DEFAULT_RIGHT_WRIST_DEVICE)
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="camera acquisition rate (default 20 Hz; independent of 4 Hz inference launches)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=DEFAULT_INFERENCE_HZ,
        help=(
            "asynchronous policy launch frequency (default 4 Hz / every 250 ms); "
            "robot control remains independently configured by --control-hz"
        ),
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=DEFAULT_ACTION_HZ,
        help="continuous robot control frequency (default 20 Hz)",
    )
    parser.add_argument(
        "--action-hz",
        type=float,
        default=None,
        help=f"override model/dataset action rate; default uses policy metadata or {DEFAULT_ACTION_HZ:g} Hz",
    )
    parser.add_argument(
        "--action-chunk-steps",
        type=int,
        default=None,
        help="deprecated alias for --min-action-chunk-steps",
    )
    parser.add_argument(
        "--min-action-chunk-steps",
        type=int,
        default=DEFAULT_MIN_ACTION_CHUNK_STEPS,
        help="reject OpenPI results shorter than this many rows (default 16; expected chunk 50)",
    )
    parser.add_argument(
        "--blend-steps",
        type=int,
        choices=(2, 3, 4),
        default=DEFAULT_BLEND_STEPS,
        help="pose/joint old/new blend length (default 3; gripper is not interpolated)",
    )
    parser.add_argument(
        "--actuator-delay-s",
        type=float,
        default=DEFAULT_ACTUATOR_DELAY_S,
        help=(
            "estimated command-to-actuation delay used for monotonic future-target "
            "selection (default 0)"
        ),
    )
    parser.add_argument(
        "--latency-skip-compensation-steps",
        type=int,
        default=0,
        help=(
            "deprecated fixed compensation expressed on the timed action horizon; "
            "prefer --actuator-delay-s"
        ),
    )
    parser.add_argument(
        "--gripper-lowpass-alpha",
        type=float,
        default=DEFAULT_GRIPPER_LOWPASS_ALPHA,
        help="independent gripper opening low-pass alpha in (0,1]",
    )
    parser.add_argument(
        "--gripper-hysteresis",
        type=float,
        default=DEFAULT_GRIPPER_HYSTERESIS,
        help="opening-fraction hysteresis around fully closed/open endpoints",
    )
    parser.add_argument(
        "--gripper-confirm-steps",
        type=int,
        default=DEFAULT_GRIPPER_CONFIRM_STEPS,
        help="consecutive endpoint requests required before open/closed transition",
    )
    parser.add_argument("--instruction", default="pick up the cube")
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="run one successful inference and exit")
    parser.add_argument(
        "--allow-execution",
        action="store_true",
        help="enable the client-side safety gate; Dashboard EXECUTE is still required",
    )
    parser.add_argument("--max-action-age-s", type=float, default=2.0)
    parser.add_argument(
        "--max-feedback-age-s",
        type=float,
        default=DEFAULT_FEEDBACK_MAX_AGE_S,
        help="maximum age of Piper CAN feedback before each queued command",
    )
    parser.add_argument(
        "--max-translation-step-m",
        type=float,
        default=DEFAULT_MAX_TRANSLATION_STEP_M,
        help=f"delivery translation step limit; {SAFETY_PROFILE} default",
    )
    parser.add_argument(
        "--max-rotation-step-rad",
        type=float,
        default=DEFAULT_MAX_ROTATION_STEP_RAD,
        help=f"delivery rotation step limit; {SAFETY_PROFILE} default",
    )
    parser.add_argument(
        "--max-gripper-step",
        type=float,
        default=DEFAULT_MAX_GRIPPER_STEP,
        help=f"delivery gripper fraction step limit; {SAFETY_PROFILE} default",
    )
    parser.add_argument(
        "--gripper-range-tolerance",
        type=float,
        default=DEFAULT_GRIPPER_RANGE_TOLERANCE,
        help="accept and clip small delivery gripper overshoot outside [0,1]",
    )
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=0.3,
        help="maximum joint-schema absolute target delta per joint",
    )
    parser.add_argument(
        "--max-joint-gripper-step",
        type=float,
        default=None,
        help="maximum joint-schema gripper opening-fraction change per command",
    )
    parser.add_argument(
        "--max-joint-gripper-step-m",
        type=float,
        default=None,
        help="deprecated metre-based joint gripper step override",
    )
    parser.add_argument(
        "--ik-max-joint-step-rad", type=float, default=DEFAULT_IK_MAX_JOINT_STEP_RAD,
        help="maximum per-joint change for delivery actions solved by continuous local IK",
    )
    parser.add_argument(
        "--ik-position-tolerance-m", type=float, default=DEFAULT_IK_POSITION_TOLERANCE_M,
        help="maximum accepted local IK position error",
    )
    parser.add_argument(
        "--ik-rotation-tolerance-rad", type=float, default=DEFAULT_IK_ROTATION_TOLERANCE_RAD,
        help="maximum accepted local IK rotation error",
    )
    parser.add_argument(
        "--ik-max-nfev", type=int, default=DEFAULT_IK_MAX_NFEV,
        help="maximum numerical IK function evaluations per command",
    )
    parser.add_argument(
        "--max-commands", type=int, default=None,
        help="stop after this many checked robot commands",
    )
    parser.add_argument(
        "--arm-settle-s", type=float, default=0.75,
        help="minimum joint-hold settling time after enabling Piper",
    )
    parser.add_argument(
        "--arm-hold-tolerance-rad", type=float, default=0.02,
        help="maximum joint error before leaving the post-enable hold",
    )
    parser.add_argument(
        "--workspace-x", type=float, nargs=2, default=DEFAULT_WORKSPACE_X_M, metavar=("MIN", "MAX"),
        help=f"delivery EEF x bounds; {SAFETY_PROFILE} envelope with margin",
    )
    parser.add_argument(
        "--workspace-y", type=float, nargs=2, default=DEFAULT_WORKSPACE_Y_M, metavar=("MIN", "MAX"),
        help=f"delivery EEF y bounds; {SAFETY_PROFILE} envelope with margin",
    )
    parser.add_argument(
        "--workspace-z", type=float, nargs=2, default=DEFAULT_WORKSPACE_Z_M, metavar=("MIN", "MAX"),
        help=f"delivery EEF z bounds; {SAFETY_PROFILE} envelope with margin",
    )
    parser.add_argument("--speed-pct", type=int, default=10)
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--enable-timeout-s", type=float, default=3.0)
    args = parser.parse_args()
    if args.max_joint_gripper_step is not None and args.max_joint_gripper_step_m is not None:
        parser.error(
            "use only one of --max-joint-gripper-step or --max-joint-gripper-step-m"
        )
    if args.max_joint_gripper_step is None:
        args.max_joint_gripper_step = (
            args.max_joint_gripper_step_m / GRIPPER_MAX_M
            if args.max_joint_gripper_step_m is not None
            else 0.25
        )
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    if args.action_chunk_steps is not None:
        args.min_action_chunk_steps = args.action_chunk_steps
    positive = (
        args.hz,
        args.control_hz,
        args.camera_fps,
        args.action_hz if args.action_hz is not None else 1.0,
        args.max_action_age_s,
        args.max_feedback_age_s,
        args.max_translation_step_m,
        args.max_rotation_step_rad,
        args.max_gripper_step,
        args.gripper_range_tolerance,
        args.max_joint_step_rad,
        args.max_joint_gripper_step,
        args.ik_max_joint_step_rad,
        args.ik_position_tolerance_m,
        args.ik_rotation_tolerance_rad,
        args.gripper_lowpass_alpha,
        args.gripper_hysteresis,
        args.enable_timeout_s,
    )
    if any(value <= 0 for value in positive) or args.reconnect_delay < 0:
        parser.error("frequencies, freshness/safety limits, and timeout must be positive")
    if args.action_chunk_steps is not None and args.action_chunk_steps <= 0:
        parser.error("action-chunk-steps must be positive")
    if args.ik_max_nfev <= 0:
        parser.error("ik-max-nfev must be positive")
    if args.max_commands is not None and args.max_commands <= 0:
        parser.error("max-commands must be positive")
    if args.min_action_chunk_steps <= 0:
        parser.error("min-action-chunk-steps must be positive")
    if args.latency_skip_compensation_steps < 0:
        parser.error("latency-skip-compensation-steps must be non-negative")
    if args.actuator_delay_s < 0:
        parser.error("actuator-delay-s must be non-negative")
    if args.gripper_lowpass_alpha > 1:
        parser.error("gripper-lowpass-alpha must be in (0,1]")
    if not 0 < args.gripper_hysteresis < 0.5:
        parser.error("gripper-hysteresis must be in (0,0.5)")
    if args.gripper_confirm_steps < 1:
        parser.error("gripper-confirm-steps must be positive")
    if not 1 <= args.speed_pct <= 100:
        parser.error("speed-pct must be in [1,100]")
    if not 0 <= args.gripper_effort <= 5000:
        parser.error("gripper-effort must be in [0,5000]")
    for name in ("workspace_x", "workspace_y", "workspace_z"):
        bounds = getattr(args, name)
        if bounds[0] >= bounds[1]:
            parser.error(f"{name.replace('_', '-')} MIN must be less than MAX")
    if args.arm_mode == "bimanual":
        if args.arm_side not in {"both", "right"}:
            parser.error("bimanual mode requires --arm-side both")
        args.arm_side = "both"
        if args.left_can == args.right_can:
            parser.error("--left-can and --right-can must differ in bimanual mode")
        if len({args.cam_high_device, args.cam_left_wrist_device, args.cam_right_wrist_device}) != 3:
            parser.error("bimanual camera devices must be distinct")
    elif args.arm_side not in {"left", "right"}:
        parser.error("single mode requires --arm-side left or right")
    if not args.instruction.strip():
        parser.error("instruction must not be empty")
    args.instruction = args.instruction.strip()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
