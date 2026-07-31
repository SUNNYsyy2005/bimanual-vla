#!/usr/bin/env python3
"""Run the official OpenPI client from the computer attached to one Piper arm.

The client is fail-closed. By default it only sends real observations and prints
predictions. Robot motion requires both a time-limited Dashboard ``execute``
authorization and the local ``--allow-execution`` flag. Only the first action
of each returned chunk is considered, and every command passes local freshness,
workspace, delta, gripper, and Piper-status checks.
"""

from __future__ import annotations

import argparse
import logging
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
DEFAULT_POLICY_HOST = "192.168.101.9"
DEFAULT_POLICY_PORT = 8000
DEFAULT_CAN = "can0"
DEFAULT_HIGH_DEVICE = "/dev/video8"
DEFAULT_WRIST_DEVICE = "/dev/video16"
CAMERA_SOURCE_HW = (240, 424)


class ExecutionBlocked(RuntimeError):
    """The action was rejected before a robot command was sent."""


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


def validate_policy_metadata(metadata: dict[str, Any], arm_side: str) -> None:
    """Fail early if the selected server cannot consume delivery observations."""
    expected = {
        "transport": "openpi_websocket_v1",
        "schema": "delivery",
        "state_dim": 10,
        "action_dim": 7,
        "arm_side": arm_side,
        "action_semantics": "eef_delta_base_xyz_left_rotvec_gripper_target",
    }
    errors = [
        f"{key}={metadata.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    camera_keys = metadata.get("camera_keys")
    if camera_keys is not None and set(camera_keys) != {"cam_high", "cam_wrist"}:
        errors.append(f"camera_keys={camera_keys!r}, expected ['cam_high', 'cam_wrist']")
    if errors:
        raise RuntimeError("incompatible policy metadata: " + "; ".join(errors))


def connect_policy(host: str, port: int, arm_side: str):
    """Create the official OpenPI client and validate the server handshake."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    logging.info("Connecting to official OpenPI policy at ws://%s:%d ...", host, port)
    policy = WebsocketClientPolicy(host=host, port=port)
    try:
        metadata = policy.get_server_metadata()
        if not isinstance(metadata, dict):
            raise RuntimeError(f"invalid policy metadata: {type(metadata).__name__}")
        validate_policy_metadata(metadata, arm_side)
    except Exception:
        close_policy(policy)
        raise
    logging.info("Policy connected: %s", metadata)
    return policy


def close_policy(policy: Any | None) -> None:
    if policy is None:
        return
    connection = getattr(policy, "_ws", None)
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass


def first_action(result: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(result.get("actions"), dtype=np.float64)
    if actions.ndim == 1:
        action = actions
    elif actions.ndim == 2 and len(actions):
        action = actions[0]
    else:
        raise ExecutionBlocked(f"invalid action chunk shape {actions.shape}")
    if action.shape != (len(ACTION_NAMES),) or not np.all(np.isfinite(action)):
        raise ExecutionBlocked(f"first action must be finite 7D, got {action.shape}")
    return action


def build_checked_target(
    state: np.ndarray,
    action: np.ndarray,
    *,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
    max_gripper_step: float,
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
    if not 0.0 <= float(action[6]) <= 1.0:
        raise ExecutionBlocked(f"gripper target {action[6]:.5f} is outside [0,1]")
    if abs(float(action[6] - state[9])) > max_gripper_step:
        raise ExecutionBlocked(
            f"gripper step {abs(float(action[6] - state[9])):.5f} exceeds {max_gripper_step:.5f}"
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
    target_gripper_m = (1.0 - float(action[6])) * GRIPPER_MAX_M
    return target_xyz, target_rpy_deg, target_gripper_m


class ExecutionController:
    def __init__(self, piper: Any, args: argparse.Namespace):
        self.piper = piper
        self.args = args
        self.robot_enabled = False
        self.state = "client_disabled" if not args.allow_execution else "shadow"
        self.blocked_reason = "local --allow-execution is absent" if not args.allow_execution else "dashboard is shadow"
        self.last_command_at: float | None = None
        self.control_revision: int | None = None
        self.robot_status: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "allow_execution": bool(self.args.allow_execution),
            "execution_state": self.state,
            "blocked_reason": self.blocked_reason,
            "last_command_at": self.last_command_at,
            "control_revision": self.control_revision,
            "robot_arm_status": self.robot_status,
        }

    def _block(self, state: str, reason: str) -> bool:
        self.state = state
        self.blocked_reason = reason[:500]
        return False

    def _enable_robot(self) -> None:
        deadline = time.monotonic() + self.args.enable_timeout_s
        while time.monotonic() < deadline:
            if self.piper.EnablePiper():
                self.robot_enabled = True
                return
            time.sleep(0.02)
        raise ExecutionBlocked(f"Piper enable timed out after {self.args.enable_timeout_s:.1f}s")

    def process(
        self,
        result: dict[str, Any],
        state: np.ndarray,
        image_timestamps: dict[str, float],
        infer_elapsed_s: float,
    ) -> bool:
        """Return True only after sending one checked target to Piper."""
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
        try:
            self.robot_status = arm_status_dict(self.piper)
            if self.robot_status["arm_status"] != 0 or self.robot_status["err_code"] != 0:
                raise ExecutionBlocked(f"Piper status is not normal: {self.robot_status}")
            target_xyz, target_rpy_deg, target_gripper_m = build_checked_target(
                state,
                first_action(result),
                max_translation_step_m=self.args.max_translation_step_m,
                max_rotation_step_rad=self.args.max_rotation_step_rad,
                max_gripper_step=self.args.max_gripper_step,
                workspace_x=tuple(self.args.workspace_x),
                workspace_y=tuple(self.args.workspace_y),
                workspace_z=tuple(self.args.workspace_z),
            )
            if not self.robot_enabled:
                self._enable_robot()
                self.state = "armed"
                self.blocked_reason = "Piper enabled; waiting for the next fresh policy response"
                return False
            raw_xyz = np.rint(target_xyz * 1_000_000.0).astype(np.int64)
            raw_rpy = np.rint(target_rpy_deg * 1000.0).astype(np.int64)
            raw_gripper = round(target_gripper_m * GRIPPER_FACTOR)
            self.piper.MotionCtrl_2(0x01, 0x00, self.args.speed_pct, 0x00)
            self.piper.EndPoseCtrl(*map(int, np.concatenate((raw_xyz, raw_rpy))))
            self.piper.GripperCtrl(int(raw_gripper), self.args.gripper_effort, 0x01, 0)
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
    state: np.ndarray,
    images: dict[str, np.ndarray],
    image_timestamps: dict[str, float],
    instruction: str,
    source_name: str,
    args: argparse.Namespace,
    execution: ExecutionController,
) -> dict[str, Any]:
    captured_at = time.time()
    return {
        "state": np.asarray(state, dtype=np.float32),
        "images": {
            "cam_high": np.asarray(images["cam_high"], dtype=np.uint8),
            "cam_wrist": np.asarray(images["cam_wrist"], dtype=np.uint8),
        },
        "prompt": instruction,
        "client_metadata": {
            "captured_at": captured_at,
            "source_name": source_name,
            "can_name": args.can,
            "cam_high_device": str(args.cam_high_device),
            "cam_wrist_device": str(args.cam_wrist_device),
            "cam_high_captured_at": float(image_timestamps["cam_high"]),
            "cam_wrist_captured_at": float(image_timestamps["cam_wrist"]),
            "arm_side": args.arm_side,
            **execution.metadata(),
        },
    }


def print_result(
    count: int,
    state: np.ndarray,
    qpos: np.ndarray,
    result: dict[str, Any],
    elapsed_s: float,
    execution: ExecutionController,
    command_sent: bool,
) -> None:
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    first = actions[0] if actions.ndim > 1 and len(actions) else actions
    control = result.get("execution_control", {})
    print(
        f"infer={count} elapsed={elapsed_s * 1000:.1f}ms "
        f"eef={np.array2string(state[:3], precision=4)} "
        f"gripper={qpos[6] * 1000:.1f}mm actions={actions.shape}\n"
        f"  first_action={np.array2string(first, precision=5, suppress_small=True)}\n"
        f"  server_mode={control.get('mode', 'missing')} local_allow={execution.args.allow_execution} "
        f"client_state={execution.state} command_sent={command_sent} reason={execution.blocked_reason or '-'}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    # Prevent LAN WebSocket traffic from being routed through an HTTP proxy.
    for key in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        if args.host not in entries:
            entries.append(args.host)
        os.environ[key] = ",".join(entries)

    source_name = args.source_name or socket.gethostname()
    logging.info("Connecting Piper feedback on %s ...", args.can)
    piper = connect_piper(args.can)
    execution = ExecutionController(piper, args)
    cameras = CameraCapture(
        cam_ids={"cam_high": args.cam_high_device, "cam_wrist": args.cam_wrist_device},
        fps=args.camera_fps,
        image_hw=IMAGE_HW,
        capture_hw=CAMERA_SOURCE_HW,
        parallel_reads=True,
    )
    policy = None
    count = 0
    interval = 1.0 / args.hz
    try:
        cameras.open()
        for key, info in cameras.verify().items():
            logging.info(
                "Camera %s: %s shape=%s latency=%sms",
                key,
                "OK" if info["ok"] else "FAIL",
                info["shape"],
                info["latency_ms"],
            )
        logging.warning(
            "%s client: %s:%d at %.3g Hz. Robot commands still require Dashboard EXECUTE.",
            "EXECUTION-CAPABLE" if args.allow_execution else "SHADOW-ONLY",
            args.host,
            args.port,
            args.hz,
        )
        while True:
            started = time.monotonic()
            try:
                if policy is None:
                    policy = connect_policy(args.host, args.port, args.arm_side)
                state, qpos = read_output_state(piper)
                images, image_timestamps = cameras.read()
                observation = build_observation(
                    state=state,
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
                command_sent = execution.process(result, state, image_timestamps, infer_elapsed)
                count += 1
                print_result(count, state, qpos, result, infer_elapsed, execution, command_sent)
                if args.once:
                    return
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                close_policy(policy)
                policy = None
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
        piper.DisconnectPort()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("BIMANUAL_VLA_POLICY_HOST", DEFAULT_POLICY_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BIMANUAL_VLA_POLICY_PORT", DEFAULT_POLICY_PORT)))
    parser.add_argument("--can", default=DEFAULT_CAN)
    parser.add_argument("--arm-side", choices=("left", "right"), default="right")
    parser.add_argument("--cam-high-device", default=DEFAULT_HIGH_DEVICE)
    parser.add_argument("--cam-wrist-device", default=DEFAULT_WRIST_DEVICE)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--hz", type=float, default=5.0, help="inference and maximum command frequency")
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
    parser.add_argument("--max-gripper-step", type=float, default=0.25)
    parser.add_argument("--workspace-x", type=float, nargs=2, default=(0.05, 0.60), metavar=("MIN", "MAX"))
    parser.add_argument("--workspace-y", type=float, nargs=2, default=(-0.45, 0.45), metavar=("MIN", "MAX"))
    parser.add_argument("--workspace-z", type=float, nargs=2, default=(0.02, 0.60), metavar=("MIN", "MAX"))
    parser.add_argument("--speed-pct", type=int, default=10)
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--enable-timeout-s", type=float, default=3.0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    positive = (
        args.hz,
        args.camera_fps,
        args.max_action_age_s,
        args.max_translation_step_m,
        args.max_rotation_step_rad,
        args.max_gripper_step,
        args.enable_timeout_s,
    )
    if any(value <= 0 for value in positive) or args.reconnect_delay < 0:
        parser.error("frequencies, freshness/safety limits, and timeout must be positive")
    if not 1 <= args.speed_pct <= 100:
        parser.error("speed-pct must be in [1,100]")
    if not 0 <= args.gripper_effort <= 5000:
        parser.error("gripper-effort must be in [0,5000]")
    for name in ("workspace_x", "workspace_y", "workspace_z"):
        bounds = getattr(args, name)
        if bounds[0] >= bounds[1]:
            parser.error(f"{name.replace('_', '-')} MIN must be less than MAX")
    if not args.instruction.strip():
        parser.error("instruction must not be empty")
    args.instruction = args.instruction.strip()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
