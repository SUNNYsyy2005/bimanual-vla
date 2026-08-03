#!/usr/bin/env python3
"""Run the official OpenPI client attached to one or two Piper arms.

The client is fail-closed and follows validated single-arm/bimanual ``delivery``
or ``joint`` policy metadata. By default it only sends real observations
and prints predictions. Robot motion requires both a time-limited Dashboard
``execute`` authorization and the local ``--allow-execution`` flag. Model-rate
actions are combined to match the lower synchronous robot command rate, and
every command passes schema-specific freshness, range, delta, and Piper-status
checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
import os
import socket
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from camera import CameraCapture
from piper_data_contract import (
    ACTION_NAMES,
    GRIPPER_MAX_M,
    IMAGE_HW,
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
    {"absolute_joint_position", "absolute_next_joint_position"}
)
DEFAULT_POLICY_HOST = "192.168.101.9"
DEFAULT_POLICY_PORT = 8000
DEFAULT_ACTION_HZ = 20.0
DEFAULT_CAN = "can0"
DEFAULT_LEFT_CAN = "can0"
DEFAULT_RIGHT_CAN = "can1"
DEFAULT_HIGH_DEVICE = "/dev/video8"
DEFAULT_WRIST_DEVICE = "/dev/video16"
DEFAULT_LEFT_WRIST_DEVICE = "/dev/video14"
DEFAULT_RIGHT_WRIST_DEVICE = "/dev/video16"
CAMERA_SOURCE_HW = (240, 424)
# Real pick-cube delivery data support, expanded by at least one 15 mm command.
# Override these CLI bounds for a different calibrated robot/table setup.
DEFAULT_WORKSPACE_X_M = (-0.04, 0.30)
DEFAULT_WORKSPACE_Y_M = (0.02, 0.52)
DEFAULT_WORKSPACE_Z_M = (0.12, 0.50)
DEFAULT_GRIPPER_RANGE_TOLERANCE = 0.02


class ExecutionBlocked(RuntimeError):
    """The action was rejected before a robot command was sent."""


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


def connect_piper(can_name: str) -> Any:
    """Connect for feedback; this alone does not enable or command the arm."""
    from piper_sdk import C_PiperInterface_V2

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


def read_output_qpos(piper: Any) -> np.ndarray:
    """Read measured joints and gripper for diagnostics only."""
    joints = piper.GetArmJointMsgs().joint_state
    gripper = piper.GetArmGripperMsgs().gripper_state
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


def read_output_state(piper: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the same 10D delivery state used by collect_output_arm.py."""
    qpos = read_output_qpos(piper)
    pose = piper.GetArmEndPoseMsgs().end_pose
    xyz_m = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64) / 1_000_000.0
    rpy_rad = np.deg2rad(
        np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64) / 1000.0
    )
    rotation = Rotation.from_euler("xyz", rpy_rad).as_matrix()
    return build_delivery_state(xyz_m, rotation, float(qpos[6])), qpos


def arm_status_dict(piper: Any) -> dict[str, Any]:
    feedback = piper.GetArmStatus().arm_status
    return {
        "ctrl_mode": int(feedback.ctrl_mode),
        "arm_status": int(feedback.arm_status),
        "mode_feed": int(feedback.mode_feed),
        "motion_status": int(feedback.motion_status),
        "err_code": int(feedback.err_code),
    }


def validate_policy_metadata(
    metadata: dict[str, Any],
    arm_side: str,
    arm_mode: str = "single",
) -> PolicyProtocol:
    """Validate server dimensions/cameras before any robot command is possible."""
    advertised_mode = str(metadata.get("arm_mode") or "single")
    expected_side = "both" if arm_mode == "bimanual" else arm_side
    expected_action_dim = 14 if arm_mode == "bimanual" else 7
    common_expected = {
        "transport": "openpi_websocket_v1",
        "arm_mode": arm_mode,
        "action_dim": expected_action_dim,
        "arm_side": expected_side,
    }
    comparable = dict(metadata, arm_mode=advertised_mode)
    errors = [
        f"{key}={comparable.get(key)!r}, expected {value!r}"
        for key, value in common_expected.items()
        if comparable.get(key) != value
    ]

    schema = metadata.get("schema")
    if schema == "delivery":
        expected_state_dim = 20 if arm_mode == "bimanual" else 10
        expected_semantics = {"eef_delta_base_xyz_left_rotvec_gripper_target"}
    elif schema == "joint":
        expected_state_dim = 14 if arm_mode == "bimanual" else 7
        expected_semantics = JOINT_ACTION_SEMANTICS
    else:
        expected_state_dim = None
        expected_semantics = set()
        errors.append(f"schema={schema!r}, expected 'delivery' or 'joint'")

    if arm_mode == "bimanual":
        expected_camera_keys = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    elif schema == "delivery":
        expected_camera_keys = {"cam_high", "cam_wrist"}
    else:
        expected_camera_keys = {"cam_high", f"cam_{arm_side}_wrist"}

    if expected_state_dim is not None and metadata.get("state_dim") != expected_state_dim:
        errors.append(f"state_dim={metadata.get('state_dim')!r}, expected {expected_state_dim!r}")
    action_semantics = metadata.get("action_semantics")
    if expected_semantics and action_semantics not in expected_semantics:
        errors.append(
            f"action_semantics={action_semantics!r}, expected one of {sorted(expected_semantics)!r}"
        )
    camera_keys = metadata.get("camera_keys")
    raw_action_hz = metadata.get("action_hz")
    action_hz: float | None = None
    if raw_action_hz is not None:
        try:
            action_hz = float(raw_action_hz)
        except (TypeError, ValueError):
            errors.append(f"action_hz={raw_action_hz!r} must be a positive number")
        else:
            if not math.isfinite(action_hz) or action_hz <= 0:
                errors.append(f"action_hz={raw_action_hz!r} must be a positive number")
    if (
        not isinstance(camera_keys, (list, tuple))
        or len(camera_keys) != len(expected_camera_keys)
        or set(camera_keys) != expected_camera_keys
    ):
        errors.append(f"camera_keys={camera_keys!r}, expected {sorted(expected_camera_keys)!r}")
    if errors:
        raise RuntimeError("incompatible policy metadata: " + "; ".join(errors))

    return PolicyProtocol(
        schema=str(schema),
        arm_mode=advertised_mode,
        state_dim=int(metadata["state_dim"]),
        action_dim=int(metadata["action_dim"]),
        arm_side=str(metadata["arm_side"]),
        action_semantics=str(action_semantics),
        camera_keys=tuple(str(key) for key in camera_keys),
        action_hz=action_hz,
    )


def resolve_action_chunk_steps(
    *,
    action_hz: float,
    command_hz: float,
    override: int | None = None,
) -> int:
    """Return how many model-rate actions one robot command should consume.

    Delivery actions are frame-to-frame deltas. If the dataset was recorded at
    20 Hz but the synchronous client sends commands at 5 Hz, four consecutive
    model actions represent one command interval.
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

    For delivery actions, xyz deltas are summed and left-multiplied rotation
    deltas are composed in order. The gripper target is the final target in the
    consumed prefix. For joint actions, each row is an absolute target, so the
    final target is selected without arithmetic on joint coordinates.
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
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate one delivery action and return target xyz, xyz Euler, gripper m."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if state.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(state)):
        raise ExecutionBlocked("current delivery state is not finite 10D")
    translation_norm = float(np.linalg.norm(action[:3]))
    rotation_norm = float(np.linalg.norm(action[3:6]))
    if translation_norm > max_translation_step_m:
        raise ExecutionBlocked(
            f"translation step {translation_norm:.5f}m exceeds {max_translation_step_m:.5f}m"
        )
    if rotation_norm > max_rotation_step_rad:
        raise ExecutionBlocked(
            f"rotation step {rotation_norm:.5f}rad exceeds {max_rotation_step_rad:.5f}rad"
        )
    raw_gripper = float(action[6])
    if raw_gripper < -gripper_range_tolerance or raw_gripper > 1.0 + gripper_range_tolerance:
        raise ExecutionBlocked(
            f"gripper target {raw_gripper:.5f} exceeds [0,1] tolerance "
            f"{gripper_range_tolerance:.5f}"
        )
    gripper = float(np.clip(raw_gripper, 0.0, 1.0))
    gripper_step = abs(gripper - float(state[9]))
    if gripper_step > max_gripper_step:
        raise ExecutionBlocked(
            f"gripper step {gripper_step:.5f} exceeds {max_gripper_step:.5f}"
        )

    target_xyz = state[:3] + action[:3]
    for axis, value, bounds in zip("xyz", target_xyz, (workspace_x, workspace_y, workspace_z)):
        if not bounds[0] <= float(value) <= bounds[1]:
            raise ExecutionBlocked(
                f"target {axis}={value:.5f}m outside workspace [{bounds[0]:.5f}, {bounds[1]:.5f}]"
            )
    current_rotation = rotation_from_state(state)
    # Collection uses R_next @ R_current.T, so reconstruct with left multiplication.
    target_rotation = Rotation.from_rotvec(action[3:6]).as_matrix() @ current_rotation
    target_rpy_deg = Rotation.from_matrix(target_rotation).as_euler("xyz", degrees=True)
    target_gripper_m = (1.0 - gripper) * GRIPPER_MAX_M
    return target_xyz, target_rpy_deg, target_gripper_m


def build_checked_joint_target(
    qpos: np.ndarray,
    action: np.ndarray,
    *,
    max_joint_step_rad: float,
    max_gripper_step_m: float,
) -> tuple[np.ndarray, float]:
    """Validate one absolute joint action and return joints in rad plus gripper metres."""
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
    gripper_m = float(action[6])
    if not 0.0 <= gripper_m <= GRIPPER_MAX_M:
        raise ExecutionBlocked(
            f"gripper target {gripper_m:.5f}m outside [0.00000, {GRIPPER_MAX_M:.5f}]"
        )

    joint_deltas = np.abs(action[:6] - qpos[:6])
    worst_joint = int(np.argmax(joint_deltas))
    if float(joint_deltas[worst_joint]) > max_joint_step_rad:
        raise ExecutionBlocked(
            f"joint{worst_joint + 1} step {joint_deltas[worst_joint]:.5f}rad exceeds "
            f"{max_joint_step_rad:.5f}rad"
        )
    gripper_delta = abs(gripper_m - float(qpos[6]))
    if gripper_delta > max_gripper_step_m:
        raise ExecutionBlocked(
            f"gripper step {gripper_delta:.5f}m exceeds {max_gripper_step_m:.5f}m"
        )
    return action[:6].copy(), gripper_m


class ExecutionController:
    def __init__(self, piper: Any | dict[str, Any], args: argparse.Namespace):
        self.args = args
        self.arm_mode = getattr(args, "arm_mode", "single")
        self.arm_side = getattr(args, "arm_side", "right")
        self.pipers = piper if isinstance(piper, dict) else {self.arm_side: piper}
        self.piper = next(iter(self.pipers.values()))  # Backward-compatible test/access alias.
        self.robot_enabled: set[str] = set()
        self.state = "client_disabled" if not args.allow_execution else "shadow"
        self.blocked_reason = "local --allow-execution is absent" if not args.allow_execution else "dashboard is shadow"
        self.last_command_at: float | None = None
        self.control_revision: int | None = None
        self.robot_status: dict[str, Any] | None = None
        self.policy_action_hz: float = float(
            getattr(args, "action_hz", None) or DEFAULT_ACTION_HZ
        )
        self.action_chunk_steps: int = max(
            1, int(getattr(args, "action_chunk_steps", None) or 1)
        )
        self.last_action_chunk_steps: int = 0
        self.last_composed_action: list[float] | None = None
        self.last_composed_action_at: float | None = None

    def configure_protocol(self, protocol: PolicyProtocol) -> None:
        """Resolve model-rate to command-rate conversion after handshake."""
        override = getattr(self.args, "action_chunk_steps", None)
        action_hz = getattr(self.args, "action_hz", None) or protocol.action_hz or DEFAULT_ACTION_HZ
        self.policy_action_hz = float(action_hz)
        self.action_chunk_steps = resolve_action_chunk_steps(
            action_hz=self.policy_action_hz,
            command_hz=float(getattr(self.args, "hz", 5.0)),
            override=override,
        )
        source = "CLI" if getattr(self.args, "action_hz", None) else (
            "policy metadata" if protocol.action_hz else f"fallback {DEFAULT_ACTION_HZ:g} Hz"
        )
        self.last_action_chunk_steps = 0
        self.last_composed_action = None
        self.last_composed_action_at = None
        logging.info(
            "Action timing: model=%.3g Hz (%s), command=%.3g Hz, consume=%d model steps/command",
            self.policy_action_hz,
            source,
            float(getattr(self.args, "hz", 5.0)),
            self.action_chunk_steps,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "allow_execution": bool(self.args.allow_execution),
            "execution_state": self.state,
            "blocked_reason": self.blocked_reason,
            "last_command_at": self.last_command_at,
            "control_revision": self.control_revision,
            "robot_arm_status": self.robot_status,
            "policy_action_hz": self.policy_action_hz,
            "command_hz": float(getattr(self.args, "hz", 5.0)),
            "action_chunk_steps": self.action_chunk_steps,
            "last_action_chunk_steps": self.last_action_chunk_steps,
            "last_composed_action": self.last_composed_action,
            "last_composed_action_at": self.last_composed_action_at,
        }

    def _block(self, state: str, reason: str) -> bool:
        self.state = state
        self.blocked_reason = reason[:500]
        return False

    def _enable_robot(self, side: str, piper: Any) -> None:
        deadline = time.monotonic() + self.args.enable_timeout_s
        while time.monotonic() < deadline:
            if piper.EnablePiper():
                self.robot_enabled.add(side)
                return
            time.sleep(0.02)
        raise ExecutionBlocked(f"{side} Piper enable timed out after {self.args.enable_timeout_s:.1f}s")

    def process(
        self,
        result: dict[str, Any],
        delivery_state: np.ndarray,
        qpos: np.ndarray,
        protocol: PolicyProtocol,
        image_timestamps: dict[str, float],
        infer_elapsed_s: float,
    ) -> bool:
        """Validate every arm first, then publish one synchronized checked command."""
        try:
            action, used_steps = aggregate_action_chunk(
                result.get("actions"), protocol, self.action_chunk_steps
            )
        except ExecutionBlocked as exc:
            self.last_action_chunk_steps = 0
            self.last_composed_action = None
            self.last_composed_action_at = time.time()
            return self._block("blocked", str(exc))
        self.last_action_chunk_steps = used_steps
        self.last_composed_action = action.tolist()
        self.last_composed_action_at = time.time()

        if not self.args.allow_execution:
            return self._block("client_disabled", "local --allow-execution is absent")
        control = result.get("execution_control")
        if not isinstance(control, dict):
            return self._block("blocked", "policy response has no execution_control")
        try:
            self.control_revision = int(control.get("revision", 0))
        except (TypeError, ValueError):
            self.control_revision = None
        if control.get("mode") != "execute":
            reason = "dashboard authorization expired" if control.get("expired") else "dashboard is shadow"
            return self._block("shadow", reason)
        if not control.get("task_id") or not control.get("session_id"):
            return self._block("blocked", "execution authorization has no task/session identity")
        try:
            remaining_s = float(control["expires_at"]) - float(control["server_time"])
        except (KeyError, TypeError, ValueError):
            return self._block("blocked", "execution authorization has no valid expiry")
        if remaining_s <= 0:
            return self._block("blocked", "execution authorization expired")
        if infer_elapsed_s > self.args.max_action_age_s:
            return self._block(
                "blocked",
                f"policy response took {infer_elapsed_s:.3f}s, limit {self.args.max_action_age_s:.3f}s",
            )
        now = time.time()
        stale = {
            key: now - float(timestamp)
            for key, timestamp in image_timestamps.items()
            if now - float(timestamp) > self.args.max_action_age_s
        }
        if stale:
            return self._block("blocked", f"stale camera frames: {stale}")

        sides = ("left", "right") if protocol.arm_mode == "bimanual" else (protocol.arm_side,)
        if set(self.pipers) != set(sides):
            return self._block(
                "blocked",
                f"connected Piper sides {sorted(self.pipers)} do not match policy sides {list(sides)}",
            )
        try:
            statuses = {side: arm_status_dict(self.pipers[side]) for side in sides}
            self.robot_status = statuses if protocol.arm_mode == "bimanual" else statuses[sides[0]]
            bad_status = {
                side: status for side, status in statuses.items()
                if status["arm_status"] != 0 or status["err_code"] != 0
            }
            if bad_status:
                raise ExecutionBlocked(f"Piper status is not normal: {bad_status}")

            prepared: dict[str, tuple[Any, ...]] = {}
            for index, side in enumerate(sides):
                action_slice = action[index * 7 : (index + 1) * 7]
                if protocol.schema == "delivery":
                    state_slice = np.asarray(delivery_state)[index * 10 : (index + 1) * 10]
                    prepared[side] = build_checked_target(
                        state_slice,
                        action_slice,
                        max_translation_step_m=self.args.max_translation_step_m,
                        max_rotation_step_rad=self.args.max_rotation_step_rad,
                        max_gripper_step=self.args.max_gripper_step,
                        gripper_range_tolerance=self.args.gripper_range_tolerance,
                        workspace_x=tuple(self.args.workspace_x),
                        workspace_y=tuple(self.args.workspace_y),
                        workspace_z=tuple(self.args.workspace_z),
                    )
                elif protocol.schema == "joint":
                    qpos_slice = np.asarray(qpos)[index * 7 : (index + 1) * 7]
                    prepared[side] = build_checked_joint_target(
                        qpos_slice,
                        action_slice,
                        max_joint_step_rad=self.args.max_joint_step_rad,
                        max_gripper_step_m=self.args.max_joint_gripper_step_m,
                    )
                else:
                    raise ExecutionBlocked(f"unsupported execution schema: {protocol.schema}")

            missing_enabled = [side for side in sides if side not in self.robot_enabled]
            if missing_enabled:
                for side in missing_enabled:
                    self._enable_robot(side, self.pipers[side])
                self.state = "armed"
                self.blocked_reason = "Piper enabled; waiting for the next fresh policy response"
                return False

            for side in sides:
                piper = self.pipers[side]
                if protocol.schema == "delivery":
                    target_xyz, target_rpy_deg, target_gripper_m = prepared[side]
                    raw_xyz = np.rint(target_xyz * 1_000_000.0).astype(np.int64)
                    raw_rpy = np.rint(target_rpy_deg * 1000.0).astype(np.int64)
                    piper.MotionCtrl_2(0x01, 0x00, self.args.speed_pct, 0x00)
                    piper.EndPoseCtrl(*map(int, np.concatenate((raw_xyz, raw_rpy))))
                else:
                    target_joints, target_gripper_m = prepared[side]
                    raw_joints = np.rint(target_joints * RAD_FACTOR).astype(np.int64)
                    piper.ModeCtrl(0x01, 0x01, self.args.speed_pct, 0x00)
                    piper.JointCtrl(*map(int, raw_joints))
                raw_gripper = round(target_gripper_m * GRIPPER_FACTOR)
                piper.GripperCtrl(int(raw_gripper), self.args.gripper_effort, 0x01, 0)
        except ExecutionBlocked as exc:
            return self._block("blocked", str(exc))
        except Exception as exc:
            logging.exception("robot command failed")
            return self._block("blocked", f"robot command failed: {exc}")
        self.last_command_at = time.time()
        self.state = "executing"
        self.blocked_reason = ""
        return True


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
) -> dict[str, Any]:
    captured_at = time.time()
    state = np.asarray(delivery_state if protocol.schema == "delivery" else qpos, dtype=np.float32)
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
            **execution.metadata(),
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
        f"  server_mode={control.get('mode', 'missing')} local_allow={execution.args.allow_execution} "
        f"client_state={execution.state} command_sent={command_sent} reason={execution.blocked_reason or '-'}",
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
        logging.info("Connecting Piper feedback: left=%s right=%s ...", args.left_can, args.right_can)
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
    policy = None
    protocol = None
    count = 0
    interval = 1.0 / args.hz
    try:
        cameras.open()
        for key, info in cameras.verify().items():
            logging.info("Camera %s: %s shape=%s latency=%sms", key, "OK" if info["ok"] else "FAIL", info["shape"], info["latency_ms"])
        logging.warning(
            "%s %s client: %s:%d at %.3g Hz. Robot commands still require Dashboard EXECUTE.",
            "EXECUTION-CAPABLE" if args.allow_execution else "SHADOW-ONLY",
            args.arm_mode,
            args.host,
            args.port,
            args.hz,
        )
        while True:
            started = time.monotonic()
            try:
                if policy is None:
                    policy, protocol = connect_policy(args.host, args.port, args.arm_side, args.arm_mode)
                    execution.configure_protocol(protocol)
                if protocol is None:
                    raise RuntimeError("policy protocol is unavailable")
                states = {side: read_output_state(piper) for side, piper in pipers.items()}
                sides = ("left", "right") if args.arm_mode == "bimanual" else (args.arm_side,)
                delivery_state = np.concatenate([states[side][0] for side in sides]).astype(np.float32)
                qpos = np.concatenate([states[side][1] for side in sides]).astype(np.float32)
                images, image_timestamps = cameras.read()
                observation = build_observation(
                    delivery_state=delivery_state,
                    qpos=qpos,
                    protocol=protocol,
                    images=images,
                    image_timestamps=image_timestamps,
                    instruction=args.instruction,
                    source_name=source_name,
                    args=args,
                    execution=execution,
                )
                infer_started = time.monotonic()
                result = policy.infer(observation)
                infer_elapsed = time.monotonic() - infer_started
                if not isinstance(result, dict) or "actions" not in result:
                    raise RuntimeError(f"invalid policy response: {result!r}")
                command_sent = execution.process(
                    result, delivery_state, qpos, protocol, image_timestamps, infer_elapsed
                )
                count += 1
                print_result(count, delivery_state, qpos, protocol, result, infer_elapsed, execution, command_sent)
                if args.once:
                    return
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                close_policy(policy)
                policy = None
                protocol = None
                execution._block("blocked", f"policy disconnected: {exc}")
                logging.warning("Policy inference/connection failed; no command published: %s", exc)
                if args.once:
                    raise
                time.sleep(args.reconnect_delay)
            sleep_s = interval - (time.monotonic() - started)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        logging.info("Stopped; no further robot commands will be published.")
    finally:
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
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--hz", type=float, default=5.0, help="policy inference and robot command frequency")
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
        help="model-rate actions consumed per robot command; default rounds action-hz / hz (legacy=1)",
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
    parser.add_argument("--max-translation-step-m", type=float, default=0.015)
    parser.add_argument("--max-rotation-step-rad", type=float, default=0.15)
    parser.add_argument(
        "--max-gripper-step",
        type=float,
        default=0.25,
        help="maximum delivery-schema gripper closed-fraction change per command",
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
        "--max-joint-gripper-step-m",
        type=float,
        default=0.02,
        help="maximum joint-schema gripper opening change per command",
    )
    parser.add_argument(
        "--workspace-x", type=float, nargs=2, default=DEFAULT_WORKSPACE_X_M, metavar=("MIN", "MAX"),
        help="delivery EEF x bounds in base frame metres; default matches the real pick-cube capture",
    )
    parser.add_argument(
        "--workspace-y", type=float, nargs=2, default=DEFAULT_WORKSPACE_Y_M, metavar=("MIN", "MAX"),
        help="delivery EEF y bounds in base frame metres; default matches the real pick-cube capture",
    )
    parser.add_argument(
        "--workspace-z", type=float, nargs=2, default=DEFAULT_WORKSPACE_Z_M, metavar=("MIN", "MAX"),
        help="delivery EEF z bounds in base frame metres; default matches the real pick-cube capture",
    )
    parser.add_argument("--speed-pct", type=int, default=10)
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--enable-timeout-s", type=float, default=3.0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    positive = (
        args.hz,
        args.camera_fps,
        args.action_hz if args.action_hz is not None else 1.0,
        args.max_action_age_s,
        args.max_translation_step_m,
        args.max_rotation_step_rad,
        args.max_gripper_step,
        args.gripper_range_tolerance,
        args.max_joint_step_rad,
        args.max_joint_gripper_step_m,
        args.enable_timeout_s,
    )
    if any(value <= 0 for value in positive) or args.reconnect_delay < 0:
        parser.error("frequencies, freshness/safety limits, and timeout must be positive")
    if args.action_chunk_steps is not None and args.action_chunk_steps <= 0:
        parser.error("action-chunk-steps must be positive")
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
