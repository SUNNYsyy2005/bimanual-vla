#!/usr/bin/env python3
"""Offline planner and passive CAN preflight for the Piper home pose.

Initialization pose:
  J1 = 90 degrees, J2-J6 = 0 degrees, gripper fully closed.

Real execution is intentionally disabled.  Hardware tests showed that the SDK's
EmergencyStop(0x02) and ResetPiper() use the same CAN frame and can rebuild the
joint reference.  The reported joint angles therefore cannot currently be used
as a trustworthy absolute pose for an automatic reset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import struct
import time
from typing import Any

import numpy as np

from robot_observation_bridge import (
    GRIPPER_FACTOR,
    JOINT_LIMITS_RAD,
    RAD_FACTOR,
    _qpos_from_feedback,
    _require_fresh_feedback,
    arm_status_dict,
    connect_piper,
)
from collect_output_arm import require_can_interface_up


HOME_JOINTS_DEG = np.array([90.0, 0.0, 0.0, 0.0, 0.0, 0.0])
HOME_JOINTS_RAD = np.deg2rad(HOME_JOINTS_DEG)
HOME_GRIPPER_M = 0.0
DEFAULT_CONTROL_HZ = 100.0
DEFAULT_MIN_DURATION_S = 15.0
DEFAULT_MAX_JOINT_SPEED_DEG_S = 0.5
DEFAULT_SPEED_PCT = 1
DEFAULT_TRACKING_ERROR_RAD = math.radians(5.0)
DEFAULT_FINAL_TOLERANCE_DEG = 3.0
DEFAULT_FINAL_GRIPPER_TOLERANCE_M = 0.003
DEFAULT_ENABLE_POLL_HZ = 100.0
DEFAULT_HANDSHAKE_HZ = 100.0
DEFAULT_HANDSHAKE_DURATION_S = 1.0
DEFAULT_MAX_HANDSHAKE_DRIFT_DEG = 1.0
DEFAULT_PRELOAD_DURATION_S = 0.5
DEFAULT_CAN_FEEDBACK_TIMEOUT_S = 2.0
REAL_EXECUTION_DISABLED_REASON = (
    "real execution is safety-locked: Piper EmergencyStop(0x02) is identical "
    "to ResetPiper() in the installed SDK, and hardware logs show a joint-reference "
    "jump. Do not command motion until the physical joint pose has been verified "
    "independently and AgileX provides the approved recovery/enable sequence"
)
PIPER_JOINT_FEEDBACK_CAN_IDS = frozenset(
    base + offset
    for base in (0x2A5, 0x2A6, 0x2A7)
    for offset in (0x00, 0x10, 0x20)
)


def home_qpos() -> np.ndarray:
    return np.concatenate((HOME_JOINTS_RAD, [HOME_GRIPPER_M])).astype(np.float64)


def wait_for_piper_can_feedback(can_name: str, *, timeout_s: float) -> set[int]:
    """Passively require all three Piper joint feedback frames before SDK init."""
    require_can_interface_up(can_name)
    received: set[int] = set()
    deadline = time.monotonic() + timeout_s
    can_socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        can_socket.bind((can_name,))
        while time.monotonic() < deadline:
            can_socket.settimeout(max(0.001, deadline - time.monotonic()))
            try:
                frame = can_socket.recv(16)
            except socket.timeout:
                break
            if len(frame) < 4:
                continue
            can_id = struct.unpack_from("=I", frame)[0] & 0x1FFFFFFF
            if can_id in PIPER_JOINT_FEEDBACK_CAN_IDS:
                received.add(can_id)
                # Feedback IDs may use the normal 0x2Ax range or one of the
                # SDK-configurable +0x10/+0x20 offsets.  Require one complete trio.
                group_base = can_id & 0xFF0
                if {group_base + 5, group_base + 6, group_base + 7} <= received:
                    return received
    except OSError as exc:
        raise RuntimeError(f"cannot passively read {can_name}: {exc}") from exc
    finally:
        can_socket.close()

    raise RuntimeError(
        f"no complete Piper joint feedback trio was received on {can_name!r} "
        f"within {timeout_s:.1f}s (seen IDs: {[hex(value) for value in sorted(received)]}). "
        "The USB-CAN interface may be UP while the robot-side CAN controller is "
        "offline. Check arm controller power, the physical emergency-stop state, "
        "CAN-H/CAN-L wiring and termination. Do not execute motion until "
        "'candump can0' shows live Piper 0x2A1-0x2A8 feedback."
    )


def smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def plan_home_trajectory(
    start_qpos: np.ndarray,
    *,
    control_hz: float = DEFAULT_CONTROL_HZ,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    max_joint_speed_rad_s: float = math.radians(DEFAULT_MAX_JOINT_SPEED_DEG_S),
) -> tuple[np.ndarray, float]:
    """Return a smooth trajectory whose peak joint speed is bounded."""
    start = np.asarray(start_qpos, dtype=np.float64)
    target = home_qpos()
    if start.shape != (7,) or not np.all(np.isfinite(start)):
        raise ValueError(f"start_qpos must be finite 7D, got {start.shape}")
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if not math.isfinite(min_duration_s) or min_duration_s <= 0:
        raise ValueError("min_duration_s must be positive")
    if not math.isfinite(max_joint_speed_rad_s) or max_joint_speed_rad_s <= 0:
        raise ValueError("max_joint_speed_rad_s must be positive")

    max_delta = float(np.max(np.abs(target[:6] - start[:6])))
    # Cubic smoothstep has a peak normalized slope of 1.5.
    speed_limited_duration = 1.5 * max_delta / max_joint_speed_rad_s
    duration_s = max(float(min_duration_s), speed_limited_duration)
    steps = max(1, int(math.ceil(duration_s * control_hz)))
    duration_s = steps / control_hz
    alpha = smoothstep(np.arange(1, steps + 1, dtype=np.float64) / steps)
    trajectory = start[None, :] + alpha[:, None] * (target - start)[None, :]
    trajectory[-1] = target
    return trajectory, duration_s


def read_qpos(piper: Any, *, max_feedback_age_s: float) -> np.ndarray:
    joints_message = piper.GetArmJointMsgs()
    gripper_message = piper.GetArmGripperMsgs()
    _require_fresh_feedback(
        {"joint": joints_message, "gripper": gripper_message},
        max_age_s=max_feedback_age_s,
    )
    return _qpos_from_feedback(joints_message, gripper_message)


def _raw_qpos_command(qpos: np.ndarray) -> tuple[np.ndarray, int]:
    target = np.asarray(qpos, dtype=np.float64)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise ValueError("command target must be finite 7D")
    raw_joints = np.rint(target[:6] * RAD_FACTOR).astype(np.int64)
    raw_gripper = int(round(float(np.clip(target[6], 0.0, 0.07)) * GRIPPER_FACTOR))
    return raw_joints, raw_gripper


def send_joint_qpos(
    piper: Any,
    qpos: np.ndarray,
    *,
    gripper_effort: int,
) -> None:
    """Publish a joint target without changing the current arm mode."""
    raw_joints, raw_gripper = _raw_qpos_command(qpos)
    piper.JointCtrl(*map(int, raw_joints))
    piper.GripperCtrl(raw_gripper, int(gripper_effort), 0x01, 0)


def send_movej_qpos(
    piper: Any,
    qpos: np.ndarray,
    *,
    speed_pct: int,
    gripper_effort: int,
) -> None:
    """Send one command in the order used by the Piper SDK V2 MOVE J demo."""
    piper.MotionCtrl_2(0x01, 0x01, int(speed_pct), 0x00)
    send_joint_qpos(piper, qpos, gripper_effort=gripper_effort)


def require_normal_status(piper: Any, *, require_control_mode: bool) -> dict[str, Any]:
    status = arm_status_dict(piper)
    if status["arm_status"] == 1:
        raise RuntimeError(
            "Piper SDK emergency stop is latched. This script intentionally does "
            "not send EmergencyStop(0x02). Support the arm, clear the stop through "
            "the vendor controller/approved procedure, verify all motors are "
            f"disabled, then rerun. Status: {status}"
        )
    if status["arm_status"] != 0 or status["err_code"] != 0:
        raise RuntimeError(f"Piper status is not normal: {status}")
    if require_control_mode and (
        status["ctrl_mode"] != 1 or status["mode_feed"] != 1
    ):
        raise RuntimeError(f"Piper did not enter CAN MOVE J mode: {status}")
    return status


def motor_enable_status(piper: Any) -> list[bool]:
    values = [bool(value) for value in piper.GetArmEnableStatus()]
    if len(values) != 6:
        raise RuntimeError(f"expected six Piper motor enable states, got {values}")
    return values


def preload_movej_while_disabled(
    piper: Any,
    anchor_qpos: np.ndarray,
    *,
    speed_pct: int,
    gripper_effort: int,
    duration_s: float,
    preload_hz: float,
    max_drift_rad: float,
    max_feedback_age_s: float,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Preload the measured target and enter MOVE J while all motors are disabled."""
    anchor = np.asarray(anchor_qpos, dtype=np.float64)
    if anchor.shape != (7,) or not np.all(np.isfinite(anchor)):
        raise ValueError("preload anchor must be finite 7D")
    initial_enable_status = motor_enable_status(piper)
    if any(initial_enable_status):
        raise RuntimeError(
            "safe MOVE J preload requires all six motors to be disabled first; "
            f"enable_status={initial_enable_status}"
        )

    # Put all six joint CAN targets into the controller before selecting MOVE J.
    # This closes the stale-target window observed when mode was selected first.
    send_joint_qpos(piper, anchor, gripper_effort=gripper_effort)
    steps = max(1, int(math.ceil(duration_s * preload_hz)))
    period_s = 1.0 / preload_hz
    next_at = time.monotonic()
    for index in range(1, steps + 1):
        if any(motor_enable_status(piper)):
            emergency_stop(piper)
            raise RuntimeError("a motor enabled unexpectedly during disabled preload")
        feedback = read_qpos(piper, max_feedback_age_s=max_feedback_age_s)
        drift = float(np.max(np.abs(feedback[:6] - anchor[:6])))
        if drift > max_drift_rad:
            emergency_stop(piper)
            raise RuntimeError(
                f"disabled preload drift {math.degrees(drift):.2f}deg exceeds "
                f"{math.degrees(max_drift_rad):.2f}deg; SDK emergency stop latched"
            )
        send_movej_qpos(
            piper,
            anchor,
            speed_pct=speed_pct,
            gripper_effort=gripper_effort,
        )
        if logger is not None:
            logger.record(
                "disabled_preload_step",
                step=index,
                total_steps=steps,
                anchor_qpos=anchor,
                feedback_qpos=feedback,
                enable_status=[False] * 6,
                max_drift_rad=drift,
            )
        next_at += period_s
        sleep_s = next_at - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)

    if any(motor_enable_status(piper)):
        emergency_stop(piper)
        raise RuntimeError("a motor enabled unexpectedly before preload verification")
    # Some Piper firmware ignores 0x151 while every motor is disabled.  That is
    # acceptable here: all joint targets are already preloaded, and the enable
    # stage reasserts MOVE J before and after every 0x471 request.
    return require_normal_status(piper, require_control_mode=False)


def enable_all_motors_while_holding(
    piper: Any,
    anchor_qpos: np.ndarray,
    *,
    speed_pct: int,
    gripper_effort: int,
    timeout_s: float,
    max_drift_rad: float,
    max_feedback_age_s: float,
    poll_hz: float = DEFAULT_ENABLE_POLL_HZ,
    logger: Any | None = None,
) -> tuple[dict[str, Any], list[bool], np.ndarray]:
    """Enable all motors while continuously streaming the preloaded hold target."""
    anchor = np.asarray(anchor_qpos, dtype=np.float64)
    deadline = time.monotonic() + timeout_s
    last_status: dict[str, Any] | None = None
    last_enable_status: list[bool] = []
    while time.monotonic() < deadline:
        # Piper may fall back to standby while processing 0x471 enable.  The
        # current target is already preloaded, so reassert MOVE J around every
        # enable request and verify the final mode in the stationary handshake.
        last_status = require_normal_status(piper, require_control_mode=False)
        feedback = read_qpos(piper, max_feedback_age_s=max_feedback_age_s)
        drift = float(np.max(np.abs(feedback[:6] - anchor[:6])))
        if drift > max_drift_rad:
            emergency_stop(piper)
            raise RuntimeError(
                f"enable-stage drift {math.degrees(drift):.2f}deg exceeds "
                f"{math.degrees(max_drift_rad):.2f}deg; SDK emergency stop latched"
            )
        # Target and mode are refreshed immediately before and after 0x471.
        send_movej_qpos(
            piper,
            anchor,
            speed_pct=speed_pct,
            gripper_effort=gripper_effort,
        )
        sdk_confirmed = bool(piper.EnablePiper())
        send_movej_qpos(
            piper,
            anchor,
            speed_pct=speed_pct,
            gripper_effort=gripper_effort,
        )
        last_enable_status = motor_enable_status(piper)
        if logger is not None:
            logger.record(
                "enable_hold_step",
                sdk_confirmed=sdk_confirmed,
                enable_status=last_enable_status,
                anchor_qpos=anchor,
                feedback_qpos=feedback,
                max_drift_rad=drift,
                status=last_status,
            )
        if sdk_confirmed and all(last_enable_status):
            return last_status, last_enable_status, feedback
        time.sleep(1.0 / poll_hz)
    raise RuntimeError(
        f"Piper six-motor enable verification timed out after {timeout_s:.1f}s: "
        f"enable_status={last_enable_status}, arm_status={last_status}"
    )


def emergency_stop(piper: Any) -> None:
    """Latch the SDK fast emergency stop; never issue the resume command here."""
    piper.EmergencyStop(0x01)


def stationary_movej_handshake(
    piper: Any,
    anchor_qpos: np.ndarray,
    *,
    speed_pct: int,
    gripper_effort: int,
    duration_s: float,
    handshake_hz: float,
    max_drift_rad: float,
    max_feedback_age_s: float,
    logger: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Anchor MOVE J to feedback and reject any mode-switch motion immediately."""
    anchor = np.asarray(anchor_qpos, dtype=np.float64)
    if anchor.shape != (7,) or not np.all(np.isfinite(anchor)):
        raise ValueError("handshake anchor must be finite 7D")

    steps = max(1, int(math.ceil(duration_s * handshake_hz)))
    period_s = 1.0 / handshake_hz
    next_at = time.monotonic()
    latest_feedback = anchor.copy()
    for index in range(1, steps + 1):
        # Read and validate before publishing another hold target.  If motion is
        # already present, the only outgoing command from this iteration is E-stop.
        latest_feedback = read_qpos(
            piper, max_feedback_age_s=max_feedback_age_s
        )
        drift = float(np.max(np.abs(latest_feedback[:6] - anchor[:6])))
        if drift > max_drift_rad:
            emergency_stop(piper)
            if logger is not None:
                logger.record(
                    "handshake_emergency_stop",
                    step=index,
                    anchor_qpos=anchor,
                    feedback_qpos=latest_feedback,
                    max_drift_rad=drift,
                )
            raise RuntimeError(
                f"MOVE J handshake drift {math.degrees(drift):.2f}deg exceeds "
                f"{math.degrees(max_drift_rad):.2f}deg; SDK emergency stop latched"
            )

        send_movej_qpos(
            piper,
            anchor,
            speed_pct=speed_pct,
            gripper_effort=gripper_effort,
        )
        if logger is not None:
            logger.record(
                "handshake_step",
                step=index,
                total_steps=steps,
                anchor_qpos=anchor,
                feedback_qpos=latest_feedback,
                max_drift_rad=drift,
            )
        next_at += period_s
        sleep_s = next_at - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)

    latest_feedback = read_qpos(
        piper, max_feedback_age_s=max_feedback_age_s
    )
    final_drift = float(np.max(np.abs(latest_feedback[:6] - anchor[:6])))
    if final_drift > max_drift_rad:
        emergency_stop(piper)
        raise RuntimeError(
            f"MOVE J handshake final drift {math.degrees(final_drift):.2f}deg "
            f"exceeds {math.degrees(max_drift_rad):.2f}deg; "
            "SDK emergency stop latched"
        )
    status = require_normal_status(piper, require_control_mode=True)
    return latest_feedback, status


def competing_controller_processes() -> list[str]:
    names = (
        "robot_observation_bridge.py",
        "teleop.py",
        "teleop_single.py",
        "collect_output_arm.py",
        "collect_gui.py",
    )
    matches: list[str] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and any(name in command for name in names):
            matches.append(f"pid={entry.name} {command.strip()}")
    return matches


class ResetLogger:
    def __init__(self, root: str | Path, args: argparse.Namespace):
        session_id = time.strftime("home_reset_%Y%m%d_%H%M%S", time.localtime())
        self.session_dir = Path(root).expanduser() / session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.session_dir / "events.jsonl"
        self.manifest_path = self.session_dir / "manifest.json"
        self._file = self.events_path.open("a", encoding="utf-8", buffering=1)
        self.manifest_path.write_text(
            json.dumps(
                {
                    "format": "bimanual-vla-home-reset-v1",
                    "started_at": time.time(),
                    "target_joints_deg": HOME_JOINTS_DEG.tolist(),
                    "target_gripper_m": HOME_GRIPPER_M,
                    "args": vars(args),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def record(self, event_type: str, **payload: Any) -> None:
        def safe(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return safe(value.tolist())
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {str(key): safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [safe(item) for item in value]
            return value

        self._file.write(
            json.dumps(
                {"event_type": event_type, "timestamp": time.time(), **safe(payload)},
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

    def close(self, reason: str) -> None:
        self.record("reset_finished", reason=reason)
        self._file.close()


def run(args: argparse.Namespace) -> None:
    # This guard must remain before every process, CAN, and SDK operation.
    raise RuntimeError(REAL_EXECUTION_DISABLED_REASON)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--execute", action="store_true", help="publish real robot commands")
    parser.add_argument("--hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_MIN_DURATION_S)
    parser.add_argument(
        "--max-joint-speed-deg-s",
        type=float,
        default=DEFAULT_MAX_JOINT_SPEED_DEG_S,
    )
    parser.add_argument("--speed-pct", type=int, default=DEFAULT_SPEED_PCT)
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument(
        "--can-feedback-timeout-s",
        type=float,
        default=DEFAULT_CAN_FEEDBACK_TIMEOUT_S,
    )
    parser.add_argument("--enable-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--preload-duration-s",
        type=float,
        default=DEFAULT_PRELOAD_DURATION_S,
    )
    parser.add_argument("--handshake-hz", type=float, default=DEFAULT_HANDSHAKE_HZ)
    parser.add_argument(
        "--handshake-duration-s",
        type=float,
        default=DEFAULT_HANDSHAKE_DURATION_S,
    )
    parser.add_argument(
        "--max-handshake-drift-deg",
        type=float,
        default=DEFAULT_MAX_HANDSHAKE_DRIFT_DEG,
    )
    parser.add_argument("--max-feedback-age-s", type=float, default=0.5)
    parser.add_argument(
        "--max-tracking-error-rad",
        type=float,
        default=DEFAULT_TRACKING_ERROR_RAD,
    )
    parser.add_argument("--settle-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--final-tolerance-deg",
        type=float,
        default=DEFAULT_FINAL_TOLERANCE_DEG,
    )
    parser.add_argument(
        "--final-gripper-tolerance-m",
        type=float,
        default=DEFAULT_FINAL_GRIPPER_TOLERANCE_M,
    )
    parser.add_argument("--monitoring-dir", default="monitoring_data")
    parser.add_argument(
        "--confirm-sdk-enable-sequence",
        action="store_true",
        help=(
            "required with --execute; confirms that the operator has read the "
            "six-motor enable and stationary-handshake safety procedure"
        ),
    )
    # Parse the retired flag only so an old command receives the explicit safety
    # lock message instead of an ambiguous "unrecognized argument" error.
    parser.add_argument(
        "--resume-sdk-estop",
        dest="_retired_resume_sdk_estop",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.execute:
        parser.error(REAL_EXECUTION_DISABLED_REASON)

    positive = (
        args.hz,
        args.duration_s,
        args.max_joint_speed_deg_s,
        args.can_feedback_timeout_s,
        args.enable_timeout_s,
        args.preload_duration_s,
        args.handshake_hz,
        args.handshake_duration_s,
        args.max_handshake_drift_deg,
        args.max_feedback_age_s,
        args.max_tracking_error_rad,
        args.settle_timeout_s,
        args.final_tolerance_deg,
        args.final_gripper_tolerance_m,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        parser.error("frequencies, durations, speeds, and tolerances must be positive")
    if not 1 <= args.speed_pct <= 100:
        parser.error("speed-pct must be in [1,100]")
    if not 0 <= args.gripper_effort <= 5000:
        parser.error("gripper-effort must be in [0,5000]")
    if np.any(HOME_JOINTS_RAD < JOINT_LIMITS_RAD[:, 0]) or np.any(
        HOME_JOINTS_RAD > JOINT_LIMITS_RAD[:, 1]
    ):
        parser.error("configured home pose is outside Piper joint limits")
    run(args)


if __name__ == "__main__":
    main()
