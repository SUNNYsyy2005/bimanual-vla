#!/usr/bin/env python3
"""Canonical robot-side RTC (real-time control) client.

This executable owns the physical control path: Piper CAN feedback and command
I/O, camera capture, OpenPI WebSocket inference, timestamped action chunks, an
20 Hz reference consumption, an optional independent Ruckig actuator clock,
and fail-closed safety gates.
The ``bin/bimanual-vla legacy-bridge`` command is a compatibility alias that
forwards to this module and cannot select a different implementation.

The client is fail-closed and follows validated single-arm/bimanual ``delivery``
or ``joint`` policy metadata. By default it only sends real observations
and prints predictions. Robot motion requires the local ``--allow-execution`` flag. Robot
control and camera acquisition run continuously at 20 Hz while a single
asynchronous policy request is launched every 250 ms (4 Hz) when the previous
request has completed. A 50-row OpenPI chunk must contain at least 16 rows.
Every decoded row is timestamped from the observation's monotonic capture time;
each control tick selects the closest future target for its estimated actuator
execution time, then blends a new pose trajectory into the still-active plan
over 2--4 steps (default 3). The gripper is filtered separately and is never
pose-blended. If a plan runs out, the last safe target
is held until a valid replacement arrives. Every command passes schema-specific
freshness, range, delta, and Piper-status checks.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import socket
import sys
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.collection.camera import CameraCapture, CameraFrameSet, CameraPreview
from bimanual_vla.deployment.recording import DeploymentRunRecorder
from bimanual_vla.deployment.servo import ChunkPlanDeferred, ChunkPlanRejected, ServoConfig, ServoError, ServoLoop, RuckigJointServo
from bimanual_vla.deployment.pose_trajectory import PoseTrajectory, PoseTrajectoryConfig
from bimanual_vla.deployment.native_eef import (
    EEFReference, NativeEEFLoop, NativeEEFError, NATIVE_STARTUP_MARGIN_RAD,
    NATIVE_STARTUP_MIN_MARGIN_RAD,
)
from bimanual_vla.deployment.execution_defaults import (
    DEFAULT_ACTION_HZ, DEFAULT_SERVO_HZ, DEFAULT_INFERENCE_SPEED_PCT, DEFAULT_FEEDBACK_MAX_AGE_S,
    DEFAULT_TRAJECTORY_MAX_SPEED_RAD_S, DEFAULT_TRAJECTORY_MAX_ACCELERATION_RAD_S2,
    DEFAULT_TRAJECTORY_MAX_JERK_RAD_S3, DEFAULT_TRAJECTORY_MAX_TRACKING_ERROR_RAD,
    DEFAULT_GRIPPER_SPEED_M_S,
)
from bimanual_vla.deployment.hardware_safety import (
    query_limits, effective_limits, effective_speed, recover_return_faults,
    install_can_send_guard, require_can_send_healthy,
    NOMINAL_JOINT_LIMITS_RAD,
)
from bimanual_vla.deployment.process_control import graceful_stop_signals
from bimanual_vla.deployment.ik_workers import IKUnreachableError
from bimanual_vla.collection.output import require_can_interface_up
from bimanual_vla.collection.can_identity import resolve_can_selector, resolve_can_pair
from bimanual_vla.device_guard import DeviceCheckRejected, RuntimeDeviceGuard
from bimanual_vla.data.action_conventions import (
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

from bimanual_vla.data.contract import (
    CONTRACT_VERSION,
    GRIPPER_MAX_M,
    IMAGE_HW,
    LEGACY_GRIPPER_OPENING_METRES_SEMANTICS,
    STATE_NAMES,
    build_delivery_state,
)
from bimanual_vla.deployment.gripper_limits import GRIPPER_MAX_M as GRIPPER_EXECUTION_MAX_M
from bimanual_vla.deployment.trajectory import (
    JerkLimitedJointTrajectory,
    TrajectoryTrackingError,
    rate_limit_grippers,
    smootherstep,
    suppress_local_joint_spikes,
)


RAD_FACTOR = 57295.7795  # Piper unit: 0.001 degree -> rad
GRIPPER_FACTOR = 1_000_000.0  # Piper unit: 0.001 mm -> metre
JOINT_LIMITS_RAD = NOMINAL_JOINT_LIMITS_RAD.copy()
JOINT_ACTION_SEMANTICS = frozenset(
    {"absolute_joint_position", "absolute_next_joint_position", WIRE_JOINT_ACTION_SEMANTICS}
)
DEFAULT_POLICY_HOST = "192.168.101.9"
DEFAULT_POLICY_PORT = 8000
DEFAULT_INFERENCE_HZ = 4.0
DEFAULT_INFERENCE_TRIGGER_STEP = 10
DEFAULT_CAMERA_FPS = 30
DEFAULT_OPENPI_CHUNK_STEPS = 50
DEFAULT_MIN_ACTION_CHUNK_STEPS = 16
DEFAULT_BLEND_STEPS = 3
DEFAULT_SYNC_WAIT_STEPS = 8
DEFAULT_RTC_EXECUTION_HORIZON = 12
DEFAULT_RTC_MAX_GUIDANCE_WEIGHT = 5.0
INFERENCE_RATE_HISTORY_SIZE = 32
DEFAULT_ACTUATOR_DELAY_S = 0.0
DEFAULT_GRIPPER_LOWPASS_ALPHA = 0.5
DEFAULT_GRIPPER_HYSTERESIS = 0.05
DEFAULT_GRIPPER_CONFIRM_STEPS = 2
DEFAULT_MAX_IMAGE_STATE_SKEW_S = 0.100  # Allow 100 ms to avoid frequent frame drops.
DEFAULT_TRACKING_LAG_THRESHOLD_RAD = 0.10
DEFAULT_TRACKING_LAG_CONFIRM_CYCLES = 3
DEFAULT_ARM_HOLD_TOLERANCE_RAD = 0.05
DEFAULT_JOINT_LIMIT_TOLERANCE_RAD = 0.10
DEFAULT_TRAJECTORY_SMOOTHING_CUTOFF_HZ = 3.0
DEFAULT_TRAJECTORY_TRACKING_TIME_CONSTANT_S = 0.25
DEFAULT_TRAJECTORY_COMMAND_LOOKAHEAD_RAD = 0.02
DEFAULT_GRIPPER_OPEN_LOOKAHEAD_STEPS = 30
DEFAULT_RETURN_HZ = 20.0
DEFAULT_RETURN_TIMEOUT_S = 30.0
DEFAULT_EXTERNAL_CONTROL_HZ_THRESHOLD = 1.0
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
DEFAULT_WORKSPACE_Y_LEFT_M = (0.01, 0.50)
DEFAULT_WORKSPACE_Y_RIGHT_M = (-0.50, -0.01)
DEFAULT_WORKSPACE_Z_M = (0.14, 0.52)  # observed [0.14706, 0.50322]
DEFAULT_GRIPPER_RANGE_TOLERANCE = 0.02
PIPER_FEEDBACK_MAX_AGE_S = DEFAULT_FEEDBACK_MAX_AGE_S
IK_JOINT_LIMIT_MARGIN_RAD = 0.002
# Piper feedback can sit a few degrees beyond the SDK's nominal zero-angle
# limits while the controller reports a healthy, non-limit status.  This is a
# feedback/calibration tolerance only: local IK may start from that measured
# pose, but it may not move farther out past it.
IK_FEEDBACK_LIMIT_TOLERANCE_RAD = 0.06
# One 20 Hz command must be a small servo step.  The numerical solve may look
# farther along the same joint branch, then the returned command is rate
# limited to this value so a Cartesian target cannot produce a wrist jump.
DEFAULT_IK_MAX_JOINT_STEP_RAD = 0.02
DEFAULT_IK_SEARCH_JOINT_RADIUS_RAD = 0.30
# 30 Hz blue-container demonstrations: per-arm maximum joint increment
# train P99=.0479, P99.9=.0713, max=.0973 rad (continuous source frames).
# This bounds branch search around the previous reference, not command speed.
DEFAULT_IK_REFERENCE_JOINT_RADIUS_RAD = 0.12
# A light preference for the previous branch: stronger regularization adds
# pose residual and can defer motion until an unregularized fallback takes over.
DEFAULT_IK_JOINT_REGULARIZATION_WEIGHT = 0.25
IK_JOINT_REGULARIZATION_SCALE_RAD = 0.08
DEFAULT_IK_POSITION_TOLERANCE_M = 0.003
DEFAULT_IK_ROTATION_TOLERANCE_RAD = 0.02
# A numerical IK step can miss the nominal orientation gate by a few
# hundredths of a radian near a joint limit.  This is a bounded execution
# fallback, not a new target planner: the candidate must still be finite,
# remain on the same local joint branch, and improve task-space error.
# A local numerical solve can miss the exact pose at a joint bound even when
# the resulting Piper target is visually/physically close.  Keep this bounded
# and local: it does not alter driver limits or the search radius.
DEFAULT_IK_FALLBACK_POSITION_TOLERANCE_M = 0.010
DEFAULT_IK_FALLBACK_ROTATION_TOLERANCE_RAD = 0.12
DEFAULT_IK_MAX_NFEV = 100
DEFAULT_MONITORING_DIR = "monitoring_data"
PIPER_CTRL_MODE_CAN = 0x01
PIPER_MOVE_MODE_J = 0x01
PIPER_ARM_STATUS_NORMAL = 0x00
PIPER_ARM_STATUS_JOINT_BRAKE_NOT_RELEASED = 0x06
PIPER_ENABLE_CONFIRM_CYCLES = 3
PIPER_ENABLE_RETRY_S = 0.02


class ExecutionBlocked(RuntimeError):
    """The action was rejected before a robot command was sent."""


class IKGeometryBlocked(ExecutionBlocked, IKUnreachableError):
    """A valid local IK target is unreachable within the existing tolerances."""


class PiperFeedbackStaleError(ExecutionBlocked):
    """Piper SDK getters contain missing or cached CAN feedback."""


class RuntimeHardwareFault(ExecutionBlocked):
    """A live device failed; this session must not reuse its action queue."""


class NonBlockingConsoleHandler(logging.Handler):
    """Move console formatting and pipe writes off the robot control thread."""

    def __init__(self, *, queue_size: int = 1024) -> None:
        super().__init__()
        self.records: queue.Queue[logging.LogRecord | None] = queue.Queue(maxsize=queue_size)
        self.sink = logging.StreamHandler()
        self.sink.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.dropped_count = 0
        self._stopping = threading.Event()
        self.writer = threading.Thread(
            target=self._write_loop, name="console-log-writer", daemon=True
        )
        self.writer.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.put_nowait(record)
        except queue.Full:
            self.dropped_count += 1

    def _write_loop(self) -> None:
        while True:
            try:
                record = self.records.get(timeout=0.1)
            except queue.Empty:
                if self._stopping.is_set():
                    return
                continue
            try:
                if record is None:
                    return
                self.sink.handle(record)
            finally:
                self.records.task_done()

    def close(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        try:
            self.records.put_nowait(None)
        except queue.Full:
            # Shutdown must not wait for a blocked GUI stdout/stderr consumer.
            pass
        self.writer.join(timeout=0.5)
        super().close()


class MonitoringRecorder:
    """Append complete RTC monitoring events to a local JSONL session.

    The recorder is deliberately independent of the control path: a recording
    failure is logged and never blocks or changes a robot command. Numpy values
    are converted to finite JSON values so the resulting file can be consumed
    directly by standard analysis tools.
    """

    def __init__(
        self,
        root: str | Path,
        args: argparse.Namespace,
        *,
        enabled: bool = True,
        periodic_rate_hz: float = 1.0,
        level: str = "compact",
    ):
        self.enabled = bool(enabled)
        self.periodic_rate_hz = float(periodic_rate_hz)
        if not math.isfinite(self.periodic_rate_hz) or self.periodic_rate_hz < 0:
            raise ValueError("periodic_rate_hz must be finite and non-negative")
        self.level = str(level).strip().lower()
        if self.level not in {"compact", "full"}:
            raise ValueError("monitoring level must be compact or full")
        self._last_periodic_at = 0.0
        if not self.enabled:
            self.session_dir = None
            self.events_path = None
            self.manifest_path = None
            self._file = None
            self._queue = None
            self._accepting = False
            self._writer_thread = None
            self.event_count = 0
            self.dropped_event_count = 0
            self._closed = True
            return
        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        session_id = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "_" + uuid.uuid4().hex[:8]
        self.session_dir = root_path / session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.session_dir / "events.jsonl"
        self.manifest_path = self.session_dir / "manifest.json"
        self._file = self.events_path.open("a", encoding="utf-8", buffering=64 * 1024)
        self._flush_counter = 0
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=4096)
        self._accepting = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="monitoring-writer",
            daemon=True,
        )
        self._writer_thread.start()
        self.event_count = 0
        self.dropped_event_count = 0
        self._closed = False
        manifest = {
            "format": "bimanual-vla-monitoring-v1",
            "session_id": session_id,
            "started_at": time.time(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "command": Path(sys.argv[0]).name or "bimanual_vla.deployment.client",
            "args": self._json_safe(vars(args)),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        self.record("session_started", monitoring_dir=str(root_path))
        logging.info("Monitoring recorder: %s", self.events_path)

    def _writer_loop(self) -> None:
        while True:
            row = self._queue.get()
            try:
                if row is None:
                    return
                # JSON normalization is deliberately performed here, not in
                # ``record()``.  Converting nested numpy arrays/dicts can be
                # surprisingly expensive and used to run on the 20 Hz
                # control thread.
                self._file.write(
                    json.dumps(
                        self._json_safe(row),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                self._flush_counter += 1
                if self._flush_counter >= 16:
                    self._file.flush()
                    self._flush_counter = 0
            except Exception:
                logging.exception("Monitoring writer failed")
            finally:
                self._queue.task_done()

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return cls._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return cls._json_safe(value.item())
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

    def record(self, event_type: str, *, periodic: bool = False, **payload: Any) -> None:
        if self._closed or not self._accepting:
            return
        if periodic:
            if not self.periodic_due():
                return
        row = {
            "event_index": self.event_count,
            "event_type": str(event_type),
            "recorded_at": time.time(),
            "recorded_monotonic": time.monotonic(),
            # Keep the control-thread operation to a shallow dict assembly;
            # recursive JSON conversion happens in the writer thread.
            **payload,
        }
        self.event_count += 1
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped_event_count += 1
            if self.dropped_event_count == 1 or self.dropped_event_count % 100 == 0:
                logging.warning(
                    "Monitoring queue full; dropped events=%d",
                    self.dropped_event_count,
                )
        except Exception:
            # Monitoring must not become a new reason to stop or alter control.
            logging.exception("Monitoring recorder failed for event=%s", event_type)

    def periodic_due(self) -> bool:
        """Claim the next periodic slot without constructing a payload.

        Callers can use this before evaluating expensive arguments.  Python
        evaluates function arguments before entering ``record()``, so passing
        ``execution.metadata(...)`` to a throttled ``record(periodic=True)``
        still used to build that metadata on every 20 Hz control tick.
        """
        if self._closed or not self._accepting or self.periodic_rate_hz <= 0:
            return False
        now_monotonic = time.monotonic()
        period = 1.0 / self.periodic_rate_hz
        if (
            self._last_periodic_at > 0
            and now_monotonic - self._last_periodic_at < period
        ):
            return False
        self._last_periodic_at = now_monotonic
        return True

    def close(self, *, reason: str = "stopped") -> None:
        if self._closed:
            return
        try:
            self.record(
                "session_finished",
                reason=reason,
                event_count=self.event_count,
                dropped_event_count=self.dropped_event_count,
            )
            self._accepting = False
            self._queue.put(None, timeout=5.0)
            self._writer_thread.join(timeout=5.0)
        finally:
            self._closed = True
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                logging.exception("Failed to close monitoring recorder")


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
    image_resize_mode: str = "pad"
    image_size: int = 224
    image_transport: str = "raw"
    image_jpeg_quality: int = 90
    action_horizon: int = DEFAULT_OPENPI_CHUNK_STEPS
    # RTC capability advertised by the server. Runtime values remain
    # client-owned and are not copied from server defaults.
    rtc_supported: bool = False
    rtc_max_execution_horizon: int = DEFAULT_RTC_EXECUTION_HORIZON
    rtc_max_guidance_weight: float = DEFAULT_RTC_MAX_GUIDANCE_WEIGHT
    rtc_supported_schedules: tuple[str, ...] = ("zeros", "ones", "linear", "exp")
    rtc_mode: str = ""
    rtc_client_planner_prefix: bool = False

    @property
    def rtc_enabled(self) -> bool:
        """Backward-compatible alias for the server RTC capability flag."""
        return self.rtc_supported


@dataclass(frozen=True)
class ClientRuntimeRates:
    """Resolved client timing knobs with one source of truth."""

    inference_hz: float
    control_hz: float
    action_hz: float
    min_action_chunk_steps: int
    blend_steps: int


def resolve_client_runtime_rates(
    args: argparse.Namespace,
    protocol: PolicyProtocol | None = None,
) -> ClientRuntimeRates:
    action_hz = float(
        getattr(args, "action_hz", None)
        or (protocol.action_hz if protocol is not None else None)
        or DEFAULT_ACTION_HZ
    )
    control_hz = float(getattr(args, "control_hz", DEFAULT_ACTION_HZ))
    inference_hz = float(getattr(args, "hz", DEFAULT_INFERENCE_HZ))
    minimum = int(
        getattr(args, "min_action_chunk_steps", DEFAULT_MIN_ACTION_CHUNK_STEPS)
    )
    legacy_override = getattr(args, "action_chunk_steps", None)
    if legacy_override is not None:
        minimum = int(legacy_override)
    return ClientRuntimeRates(
        inference_hz=inference_hz,
        control_hz=control_hz,
        action_hz=action_hz,
        min_action_chunk_steps=minimum,
        blend_steps=int(getattr(args, "blend_steps", DEFAULT_BLEND_STEPS)),
    )


def connect_piper(can_name: str) -> Any:
    """Connect for feedback; this alone does not enable or command the arm."""
    from piper_sdk import C_PiperInterface_V2

    can_name = resolve_can_selector(can_name)
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
    try:
        install_can_send_guard(piper, error_type=RuntimeHardwareFault)
        query_limits(piper)
    except Exception:
        piper.DisconnectPort()
        raise
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


@dataclass(frozen=True)
class PiperIKSolveResult:
    """Full numerical IK solution and the bounded command sent this control tick."""

    solution_joints_rad: np.ndarray
    command_joints_rad: np.ndarray
    rate_limited: bool | None
    solution_position_error_m: float | None = None
    solution_rotation_error_rad: float | None = None
    solution_max_joint_step_rad: float | None = None
    solution_reference_step_rad: float | None = None
    command_position_error_m: float | None = None
    command_rotation_error_rad: float | None = None
    optimizer_success: bool | None = None
    optimizer_status: int | None = None
    optimizer_nfev: int | None = None
    tolerance_relaxed: bool = False
    accepted_position_tolerance_m: float | None = None
    accepted_rotation_tolerance_rad: float | None = None
    solve_elapsed_s: float | None = None
    selection_method: str = "regularized"
    minimum_motion_elapsed_s: float = 0.0
    minimum_motion_timed_out: bool = False


class PiperContinuousIK:
    """Numerical Piper IK constrained to the branch near current feedback."""

    def __init__(self, fk: Any | None = None) -> None:
        self._kinematics = None
        if fk is None:
            from piper_sdk import C_PiperForwardKinematics
            from bimanual_vla.deployment.piper_kinematics import PiperKinematics

            fk = C_PiperForwardKinematics()
            self._kinematics = PiperKinematics(fk)
        self._fk = fk

    def pose(self, joints_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(joints_rad, dtype=np.float64)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ExecutionBlocked(f"IK joints must be finite 6D, got {joints.shape}")
        if self._kinematics is not None:
            return self._kinematics.evaluate(joints)
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
        initial_joints_rad: np.ndarray | None = None,
        max_joint_step_rad: float,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        max_nfev: int,
        search_joint_radius_rad: float | None = None,
        reference_joint_radius_rad: float = DEFAULT_IK_REFERENCE_JOINT_RADIUS_RAD,
        joint_regularization_weight: float = DEFAULT_IK_JOINT_REGULARIZATION_WEIGHT,
        joint_limits: np.ndarray | None = None,
        minimum_motion: bool = True,
    ) -> np.ndarray:
        return self.solve_with_diagnostics(
            current_joints_rad,
            target_xyz_m,
            target_rpy_deg,
            initial_joints_rad=initial_joints_rad,
            max_joint_step_rad=max_joint_step_rad,
            position_tolerance_m=position_tolerance_m,
            rotation_tolerance_rad=rotation_tolerance_rad,
            max_nfev=max_nfev,
            search_joint_radius_rad=search_joint_radius_rad,
            reference_joint_radius_rad=reference_joint_radius_rad,
            joint_regularization_weight=joint_regularization_weight,
            joint_limits=joint_limits,
            minimum_motion=minimum_motion,
        ).command_joints_rad.copy()

    def solve_with_diagnostics(
        self,
        current_joints_rad: np.ndarray,
        target_xyz_m: np.ndarray,
        target_rpy_deg: np.ndarray,
        *,
        initial_joints_rad: np.ndarray | None = None,
        max_joint_step_rad: float,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        max_nfev: int,
        search_joint_radius_rad: float | None = None,
        reference_joint_radius_rad: float = DEFAULT_IK_REFERENCE_JOINT_RADIUS_RAD,
        joint_regularization_weight: float = DEFAULT_IK_JOINT_REGULARIZATION_WEIGHT,
        joint_limits: np.ndarray | None = None,
        minimum_motion: bool = True,
    ) -> PiperIKSolveResult:
        solve_started = time.perf_counter()
        current = np.asarray(current_joints_rad, dtype=np.float64)
        target_xyz = np.asarray(target_xyz_m, dtype=np.float64)
        target_rpy = np.asarray(target_rpy_deg, dtype=np.float64)
        if current.shape != (6,) or not np.all(np.isfinite(current)):
            raise ExecutionBlocked("current IK joints are not finite 6D")
        if target_xyz.shape != (3,) or not np.all(np.isfinite(target_xyz)):
            raise ExecutionBlocked("IK target xyz is not finite 3D")
        if target_rpy.shape != (3,) or not np.all(np.isfinite(target_rpy)):
            raise ExecutionBlocked("IK target rpy is not finite 3D")
        if not math.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0:
            raise ExecutionBlocked("IK maximum command joint step must be positive")
        search_radius = (
            max_joint_step_rad
            if search_joint_radius_rad is None
            else float(search_joint_radius_rad)
        )
        if not math.isfinite(search_radius) or search_radius < max_joint_step_rad:
            raise ExecutionBlocked(
                "IK search joint radius must be finite and at least the command step"
            )
        regularization_weight = float(joint_regularization_weight)
        if not math.isfinite(reference_joint_radius_rad) or reference_joint_radius_rad <= 0:
            raise ExecutionBlocked("IK reference joint radius must be finite and positive")
        if not math.isfinite(regularization_weight) or regularization_weight < 0:
            raise ExecutionBlocked("IK joint regularization weight must be non-negative")

        limits = JOINT_LIMITS_RAD if joint_limits is None else np.asarray(joint_limits, dtype=np.float64)
        if limits.shape != (6, 2) or not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
            raise ExecutionBlocked("IK joint limits must be finite 6x2 intervals")
        hard_lower = limits[:, 0] + IK_JOINT_LIMIT_MARGIN_RAD
        hard_upper = limits[:, 1] - IK_JOINT_LIMIT_MARGIN_RAD
        if np.any(current < hard_lower - IK_FEEDBACK_LIMIT_TOLERANCE_RAD) or np.any(
            current > hard_upper + IK_FEEDBACK_LIMIT_TOLERANCE_RAD
        ):
            raise ExecutionBlocked("current joints are too far outside IK limits")
        # Feedback tolerance permits reading a small encoder zero offset; it
        # does not permit generating out-of-driver-range targets. Preserve the
        # measured pose inside the optional margin only within driver bounds.
        # Solve this constrained problem directly instead of clipping its result.
        feedback_lower = np.maximum(limits[:, 0], np.minimum(hard_lower, current))
        feedback_upper = np.minimum(limits[:, 1], np.maximum(hard_upper, current))
        lower = np.maximum(feedback_lower, current - search_radius)
        upper = np.minimum(feedback_upper, current + search_radius)
        initial_source = current if initial_joints_rad is None else np.asarray(
            initial_joints_rad, dtype=np.float64
        )
        if initial_source.shape != (6,) or not np.all(np.isfinite(initial_source)):
            raise ExecutionBlocked("initial IK joints are not finite 6D")
        # A warm start alone does not preserve a branch: fallback optimizers
        # can choose opposite edges of the measured +/- radius on successive
        # ticks. Constrain both neighborhoods, and prefer the last accepted
        # reference, not the lagging encoder pose. Never clip a solved pose.
        continuity_anchor = initial_source
        if initial_joints_rad is not None:
            lower = np.maximum(lower, continuity_anchor - reference_joint_radius_rad)
            upper = np.minimum(upper, continuity_anchor + reference_joint_radius_rad)
        if np.any(lower >= upper):
            raise IKGeometryBlocked("continuous IK has no overlapping feedback/reference interval")
        initial = np.clip(initial_source, lower + 1e-8, upper - 1e-8)
        target_rotation = Rotation.from_euler("xyz", target_rpy, degrees=True).as_matrix()

        # Adjacent 30 Hz EEF targets normally stay on one local IK branch.
        # Try a bounded damped-Jacobian update first; the full scipy solve is
        # retained as a fallback for larger changes or near-singular poses.
        task_scales = np.asarray(
            [position_tolerance_m] * 3 + [rotation_tolerance_rad] * 3,
            dtype=np.float64,
        )

        cached_q = cached_error = cached_jacobian = None

        def analytic_task(candidate):
            nonlocal cached_q, cached_error, cached_jacobian
            if cached_q is None or not np.array_equal(candidate, cached_q):
                from bimanual_vla.deployment.piper_kinematics import rotation_error_jacobian
                xyz, rotation, geometric = self._kinematics.evaluate(candidate, jacobian=True)
                rotation_error = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
                cached_error = np.concatenate((xyz - target_xyz, rotation_error))
                cached_jacobian = np.concatenate((geometric[:3],
                    rotation_error_jacobian(rotation_error, geometric[3:])), axis=0)
                cached_q = candidate.copy()
            return cached_error, cached_jacobian

        def task_error(candidate: np.ndarray) -> np.ndarray:
            if self._kinematics is not None:
                return analytic_task(candidate)[0]
            xyz, rotation = self.pose(candidate)
            rotation_error = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
            return np.concatenate((xyz - target_xyz, rotation_error))

        def fast_local_solve(seed: np.ndarray) -> np.ndarray:
            """Solve the common small local update without scipy iteration."""
            candidate = np.clip(np.asarray(seed, dtype=np.float64), lower, upper)
            for _ in range(4):
                error = task_error(candidate)
                if np.max(np.abs(error / task_scales)) <= 0.25:
                    break
                jacobian = (analytic_task(candidate)[1] if self._kinematics is not None
                            else np.empty((6, 6), dtype=np.float64))
                for column in range(0 if self._kinematics is not None else 6):
                    step = 1e-4
                    plus = min(float(candidate[column] + step), float(upper[column]))
                    minus = max(float(candidate[column] - step), float(lower[column]))
                    if plus <= minus:
                        jacobian[:, column] = 0.0
                        continue
                    q_plus = candidate.copy()
                    q_minus = candidate.copy()
                    q_plus[column] = plus
                    q_minus[column] = minus
                    jacobian[:, column] = (
                        task_error(q_plus) - task_error(q_minus)
                    ) / (plus - minus)
                scaled_jacobian = jacobian / task_scales[:, None]
                scaled_error = error / task_scales
                regularizer = regularization_weight / (IK_JOINT_REGULARIZATION_SCALE_RAD ** 2)
                system = scaled_jacobian.T @ scaled_jacobian
                system += (regularizer + 1e-4) * np.eye(6)
                rhs = (
                    -scaled_jacobian.T @ scaled_error
                    + regularizer * (continuity_anchor - candidate)
                )
                try:
                    delta = np.linalg.solve(system, rhs)
                except np.linalg.LinAlgError:
                    return candidate
                if not np.all(np.isfinite(delta)):
                    return candidate
                candidate = np.clip(candidate + delta, lower, upper)
            return candidate

        def residual(candidate: np.ndarray, *, joint_weight=regularization_weight) -> np.ndarray:
            error = task_error(candidate) / task_scales
            if joint_weight == 0:
                return error
            joint_regularization = (
                math.sqrt(joint_weight)
                * (candidate - continuity_anchor)
                / IK_JOINT_REGULARIZATION_SCALE_RAD
            )
            return np.concatenate((error, joint_regularization))

        def residual_jacobian(candidate, *, joint_weight):
            jacobian = analytic_task(candidate)[1] / task_scales[:, None]
            if joint_weight == 0:
                return jacobian
            return np.concatenate((jacobian, np.eye(6) * (
                math.sqrt(joint_weight) / IK_JOINT_REGULARIZATION_SCALE_RAD)), axis=0)

        def optimize(seed: np.ndarray, *, joint_weight: float):
            """Run one bounded local solve from a seed on this joint branch."""
            # Numerical optimization is only needed for delivery control; keep
            # joint-only deployments from importing scipy.optimize at startup.
            from scipy.optimize import least_squares
            return least_squares(
                lambda q: residual(q, joint_weight=joint_weight),
                np.clip(np.asarray(seed, dtype=np.float64), lower + 1e-8, upper - 1e-8),
                bounds=(lower, upper),
                jac=(lambda q: residual_jacobian(q, joint_weight=joint_weight))
                    if self._kinematics is not None else "2-point",
                max_nfev=max_nfev,
                # Jacobian scaling helps when a wrist joint is close to a
                # bound and the translational/rotational residuals have very
                # different numerical curvature.
                x_scale="jac",
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )

        def pose_errors(joints):
            xyz, rotation = self.pose(joints)
            return (float(np.linalg.norm(xyz - target_xyz)), float(np.linalg.norm(
                Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec())))

        minimum = None
        if minimum_motion and self._kinematics is not None:
            from bimanual_vla.deployment.ik_continuity import minimum_motion_candidate
            minimum = minimum_motion_candidate(analytic_task, continuity_anchor, lower, upper,
                position_tolerance_m, rotation_tolerance_rad,
                max_iterations=min(max_nfev, 24))
        minimum_accepted = minimum is not None and minimum.joints is not None
        fast_candidate = minimum.joints if minimum_accepted else fast_local_solve(initial)
        fast_position_error, fast_rotation_error = pose_errors(fast_candidate)
        fast_accepted = (
            fast_position_error <= position_tolerance_m
            and fast_rotation_error <= rotation_tolerance_rad
        )
        if fast_accepted:
            class _FastResult:
                success = True
                status = 1
                nfev = 0

                def __init__(self, x):
                    self.x = x

            result = _FastResult(fast_candidate)
            solved = fast_candidate
            optimizer_nfev = 0
        else:
            # The fast candidate is still a bounded, same-branch point.  Use
            # it as the scipy seed when it did not quite meet tolerance; this
            # reduces iterations for the difficult tail without changing the
            # fallback's limits or acceptance gates.
            result = optimize(fast_candidate, joint_weight=regularization_weight)
            solved = np.asarray(result.x, dtype=np.float64)

        position_error, rotation_error = pose_errors(solved)
        tolerance_relaxed = False
        accepted_position_tolerance = position_tolerance_m
        accepted_rotation_tolerance = rotation_tolerance_rad
        if position_error > position_tolerance_m or rotation_error > rotation_tolerance_rad:
            # Proximity is a soft preference; it must not make a reachable
            # target fail the hard pose tolerances.  A bounded least-squares
            # solve can stop at a wrist bound from one seed even though the
            # same local branch has a valid solution from another nearby seed.
            # Retry only the local problem (never enlarge limits/radius) from
            # a small, deterministic set of existing branch states.
            seeds = [solved, current, initial]
            if initial_joints_rad is not None:
                seeds.append(np.asarray(initial_joints_rad, dtype=np.float64))
            def candidate_record(candidate, candidate_result):
                pos, rot = pose_errors(candidate)
                return (pos, rot, float(np.linalg.norm(candidate-continuity_anchor)),
                        candidate, candidate_result)

            # Do not throw away the regularized solution when an unregularized
            # retry has a marginally smaller residual but a large joint swing.
            candidates = [candidate_record(solved, result)]
            total_nfev = int(getattr(result, "nfev", 0))
            for seed in seeds:
                candidate_result = optimize(seed, joint_weight=0.0)
                candidate = np.asarray(candidate_result.x, dtype=np.float64)
                candidate_position_error, candidate_rotation_error = pose_errors(candidate)
                candidates.append((candidate_position_error, candidate_rotation_error,
                                   float(np.linalg.norm(candidate - continuity_anchor)), candidate,
                                   candidate_result))
                total_nfev += int(getattr(candidate_result, "nfev", 0))
            feasible = [item for item in candidates
                        if item[0] <= position_tolerance_m and item[1] <= rotation_tolerance_rad]
            if feasible:
                # Preserve continuity among feasible solutions while making
                # pose tolerances the hard criterion.
                _, _, _, solved, result = min(feasible, key=lambda item: item[2])
                position_error, rotation_error = pose_errors(solved)
            else:
                relaxed = [item for item in candidates
                           if item[0] <= max(position_tolerance_m, DEFAULT_IK_FALLBACK_POSITION_TOLERANCE_M)
                           and item[1] <= max(rotation_tolerance_rad, DEFAULT_IK_FALLBACK_ROTATION_TOLERANCE_RAD)]
                if initial_joints_rad is not None and relaxed:
                    _, _, _, solved, result = min(relaxed, key=lambda item: item[2])
                else:
                    _, _, _, solved, result = min(
                        candidates, key=lambda item: (item[0] / position_tolerance_m
                                                      + item[1] / rotation_tolerance_rad, item[2]))
                position_error, rotation_error = pose_errors(solved)
            # nfev is diagnostic only; report all bounded retries so a caller
            # can distinguish a genuine local-limit failure from early stop.
            optimizer_nfev = total_nfev
        else:
            optimizer_nfev = int(getattr(result, "nfev", 0))
        solved_joint_step = float(np.max(np.abs(solved - current)))
        if np.any(solved < limits[:, 0]) or np.any(solved > limits[:, 1]):
            raise IKGeometryBlocked("continuous IK target is outside driver joint limits")
        if position_error > position_tolerance_m or rotation_error > rotation_tolerance_rad:
            # Do not stop for a small numerical miss at a joint limit.  Accept
            # the best local candidate only under a deliberately wider,
            # bounded tolerance and only when it improves the measured pose.
            # This keeps replay/inference on the same online control path while
            # avoiding a one-tick abort for errors such as 0.02048 rad vs the
            # nominal 0.02000 rad gate.
            fallback_position_tolerance = max(
                position_tolerance_m, DEFAULT_IK_FALLBACK_POSITION_TOLERANCE_M
            )
            fallback_rotation_tolerance = max(
                rotation_tolerance_rad, DEFAULT_IK_FALLBACK_ROTATION_TOLERANCE_RAD
            )
            if (
                position_error <= fallback_position_tolerance
                and rotation_error <= fallback_rotation_tolerance
            ):
                tolerance_relaxed = True
                accepted_position_tolerance = fallback_position_tolerance
                accepted_rotation_tolerance = fallback_rotation_tolerance
            else:
                raise IKGeometryBlocked(
                    "continuous IK could not reach a nearby solution: "
                    f"position_error={position_error:.5f}m, "
                    f"rotation_error={rotation_error:.5f}rad, "
                    f"search_joint_step={solved_joint_step:.5f}rad"
                )
        if solved_joint_step > search_radius + 1e-6:
            raise ExecutionBlocked(
                f"continuous IK search step {solved_joint_step:.5f}rad exceeds "
                f"{search_radius:.5f}rad"
            )
        if solved_joint_step <= max_joint_step_rad + 1e-9:
            return PiperIKSolveResult(
                solution_joints_rad=solved.copy(),
                command_joints_rad=solved.copy(),
                rate_limited=False,
                solution_position_error_m=position_error,
                solution_rotation_error_rad=rotation_error,
                solution_max_joint_step_rad=solved_joint_step,
                solution_reference_step_rad=float(np.max(np.abs(solved-continuity_anchor))),
                command_position_error_m=position_error,
                command_rotation_error_rad=rotation_error,
                optimizer_success=bool(result.success),
                optimizer_status=int(result.status),
                optimizer_nfev=optimizer_nfev,
                tolerance_relaxed=tolerance_relaxed,
                accepted_position_tolerance_m=accepted_position_tolerance,
                accepted_rotation_tolerance_rad=accepted_rotation_tolerance,
                solve_elapsed_s=time.perf_counter() - solve_started,
                selection_method="minimum_motion" if minimum_accepted else "regularized",
                minimum_motion_elapsed_s=minimum.elapsed_s if minimum is not None else 0.,
                minimum_motion_timed_out=minimum.timed_out if minimum is not None else False,
            )

        # Follow the regularized solution direction with a bounded 20 Hz servo
        # step. Requiring the whole future EEF target to be reached in one tick
        # caused either 0.08 rad wrist jumps or a complete queue stop.
        scale = max_joint_step_rad / solved_joint_step
        command = current + scale * (solved - current)
        if np.any(command < limits[:, 0]) or np.any(command > limits[:, 1]):
            raise IKGeometryBlocked("rate-limited IK step remains outside driver joint limits")
        command_xyz, command_rotation = self.pose(command)
        current_xyz, current_rotation = self.pose(current)
        current_task_error = float(
            np.linalg.norm((current_xyz - target_xyz) / position_tolerance_m) ** 2
            + np.linalg.norm(
                Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()
                / rotation_tolerance_rad
            )
            ** 2
        )
        command_task_error = float(
            np.linalg.norm((command_xyz - target_xyz) / position_tolerance_m) ** 2
            + np.linalg.norm(
                Rotation.from_matrix(target_rotation @ command_rotation.T).as_rotvec()
                / rotation_tolerance_rad
            )
            ** 2
        )
        if command_task_error >= current_task_error - 1e-9:
            raise ExecutionBlocked(
                "continuous IK rate-limited step does not make progress toward target"
            )
        command_position_error = float(np.linalg.norm(command_xyz - target_xyz))
        command_rotation_error = float(
            np.linalg.norm(
                Rotation.from_matrix(target_rotation @ command_rotation.T).as_rotvec()
            )
        )
        return PiperIKSolveResult(
            solution_joints_rad=solved.copy(),
            command_joints_rad=command.copy(),
            rate_limited=True,
            solution_position_error_m=position_error,
            solution_rotation_error_rad=rotation_error,
            solution_max_joint_step_rad=solved_joint_step,
            solution_reference_step_rad=float(np.max(np.abs(solved-continuity_anchor))),
            command_position_error_m=command_position_error,
            command_rotation_error_rad=command_rotation_error,
            optimizer_success=bool(result.success),
            optimizer_status=int(result.status),
            optimizer_nfev=optimizer_nfev,
            tolerance_relaxed=tolerance_relaxed,
            accepted_position_tolerance_m=accepted_position_tolerance,
            accepted_rotation_tolerance_rad=accepted_rotation_tolerance,
            solve_elapsed_s=time.perf_counter() - solve_started,
            selection_method="minimum_motion" if minimum_accepted else "regularized",
            minimum_motion_elapsed_s=minimum.elapsed_s if minimum is not None else 0.,
            minimum_motion_timed_out=minimum.timed_out if minimum is not None else False,
        )




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


@dataclass(frozen=True)
class PiperFeedbackSnapshot:
    """One timestamp-consistent feedback sample shared by the control path."""

    qpos_m: np.ndarray
    delivery_state: np.ndarray
    statuses: Mapping[str, dict[str, Any]]
    driver_statuses: Mapping[str, dict[str, Any]]
    captured_at: float

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos_m)
        delivery = np.asarray(self.delivery_state)
        if qpos.ndim != 1 or not np.all(np.isfinite(qpos)):
            raise ValueError("feedback qpos must be a finite 1D array")
        if delivery.ndim != 1 or not np.all(np.isfinite(delivery)):
            raise ValueError("feedback delivery state must be a finite 1D array")
        if not math.isfinite(float(self.captured_at)):
            raise ValueError("feedback timestamp must be finite")


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
    message = piper.GetArmStatus()
    feedback = message.arm_status
    has_timestamp = hasattr(message, "time_stamp")
    timestamp = float(getattr(message, "time_stamp", 0.0) or 0.0)
    hz = float(getattr(message, "Hz", 0.0) or 0.0)
    feedback_age_s = time.time() - timestamp if timestamp > 0 else None
    feedback_fresh = (
        None
        if not has_timestamp
        else bool(
            timestamp > 0
            and feedback_age_s is not None
            and -1.0 <= feedback_age_s <= PIPER_FEEDBACK_MAX_AGE_S
        )
    )
    return {
        "ctrl_mode": int(feedback.ctrl_mode),
        "arm_status": int(feedback.arm_status),
        "mode_feed": int(feedback.mode_feed),
        "motion_status": int(feedback.motion_status),
        "err_code": int(feedback.err_code),
        "feedback_timestamp": timestamp if has_timestamp else None,
        "feedback_age_s": feedback_age_s,
        "feedback_hz": hz if has_timestamp else None,
        "feedback_fresh": feedback_fresh,
    }


def driver_enable_status_dict(piper: Any) -> dict[str, Any]:
    getter = getattr(piper, "GetArmLowSpdInfoMsgs", None)
    if not callable(getter):
        return {
            "available": False,
            "faulted": None,
            "healthy": None,
            "ready": None,
            "feedback_timestamp": None,
            "feedback_age_s": None,
            "feedback_hz": None,
            "enabled": None,
            "received": None,
            "faults": None,
        }
    try:
        message = getter()
        timestamp = float(getattr(message, "time_stamp", 0.0) or 0.0)
        hz = float(getattr(message, "Hz", 0.0) or 0.0)
        feedback_age_s = time.time() - timestamp if timestamp > 0 else None
        fresh = bool(
            timestamp > 0
            and feedback_age_s is not None
            and -1.0 <= feedback_age_s <= PIPER_FEEDBACK_MAX_AGE_S
        )
        enabled: list[bool] = []
        received: list[bool] = []
        faults: list[list[str]] = []
        fault_fields = (
            "voltage_too_low",
            "motor_overheating",
            "driver_overcurrent",
            "driver_overheating",
            "collision_status",
            "driver_error_status",
            "stall_status",
        )
        for motor_index in range(1, 7):
            motor = getattr(message, f"motor_{motor_index}")
            foc = motor.foc_status
            enabled.append(bool(foc.driver_enable_status))
            received.append(bool(getattr(motor, "can_id", 0)))
            faults.append(
                [name for name in fault_fields if bool(getattr(foc, name, False))]
            )
        faulted = bool(any(faults))
        healthy = bool(fresh and all(received) and not faulted)
        ready = bool(healthy and all(enabled))
        return {
            "available": True,
            "faulted": faulted,
            "healthy": healthy,
            "ready": ready,
            "feedback_timestamp": timestamp,
            "feedback_age_s": feedback_age_s,
            "feedback_hz": hz,
            "enabled": enabled,
            "received": received,
            "faults": faults,
        }
    except Exception as exc:
        return {
            "available": True,
            "faulted": True,
            "healthy": False,
            "ready": False,
            "feedback_timestamp": None,
            "feedback_age_s": None,
            "feedback_hz": None,
            "enabled": None,
            "received": None,
            "faults": [[f"feedback_error:{type(exc).__name__}:{exc}"]],
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
    output_mode: str = "auto",
) -> PolicyProtocol:
    """Validate server dimensions/cameras before any robot command is possible.

    ``output_mode`` is an operator-facing contract lock.  ``auto`` keeps the
    historical behavior and follows the server handshake; ``joint`` and
    ``delivery`` require the server to advertise the corresponding schema.
    An explicit mode never reinterprets an untrusted 7D vector locally: a
    mismatch is rejected before observations or robot commands are accepted.
    """
    output_mode = str(output_mode or "auto").strip().lower()
    if output_mode not in {"auto", "joint", "delivery"}:
        raise ValueError(
            f"output_mode must be one of 'auto', 'joint', or 'delivery', got {output_mode!r}"
        )
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
    if output_mode != "auto" and schema != output_mode:
        errors.append(
            f"server schema={schema!r} conflicts with --output-mode={output_mode!r}; "
            "refusing to reinterpret policy outputs"
        )
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
    capabilities = metadata.get("rtc_capabilities")
    raw_image_size = metadata.get("image_size", 224)
    try:
        image_size = int(raw_image_size)
        if image_size <= 0 or image_size != raw_image_size:
            raise ValueError
    except (TypeError, ValueError):
        image_size = 224
        errors.append(f"image_size={raw_image_size!r} must be a positive integer")
    image_transport = str(metadata.get("image_transport", "raw")).strip().lower()
    if image_transport not in {"raw", "jpeg_chw_v1"}:
        errors.append(f"image_transport={image_transport!r} must be raw or jpeg_chw_v1")
        image_transport = "raw"
    raw_jpeg_quality = metadata.get("image_jpeg_quality", 90)
    try:
        image_jpeg_quality = int(raw_jpeg_quality)
        if image_jpeg_quality != raw_jpeg_quality or not 1 <= image_jpeg_quality <= 100:
            raise ValueError
    except (TypeError, ValueError):
        image_jpeg_quality = 90
        if image_transport == "jpeg_chw_v1":
            errors.append("image_jpeg_quality must be an integer in [1, 100]")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    rtc_supported = bool(metadata.get("rtc_supported", metadata.get("rtc_enabled", False)))
    raw_rtc_execution_horizon = capabilities.get(
        "max_execution_horizon",
        metadata.get("rtc_execution_horizon", DEFAULT_RTC_EXECUTION_HORIZON),
    )
    try:
        rtc_max_execution_horizon = int(raw_rtc_execution_horizon)
    except (TypeError, ValueError):
        rtc_max_execution_horizon = DEFAULT_RTC_EXECUTION_HORIZON
        if rtc_supported:
            errors.append(
                "rtc max_execution_horizon must be a positive integer, got "
                f"{raw_rtc_execution_horizon!r}"
            )
    raw_rtc_max_guidance_weight = capabilities.get(
        "max_guidance_weight",
        metadata.get("rtc_max_guidance_weight", DEFAULT_RTC_MAX_GUIDANCE_WEIGHT),
    )
    try:
        rtc_max_guidance_weight = float(raw_rtc_max_guidance_weight)
    except (TypeError, ValueError):
        rtc_max_guidance_weight = DEFAULT_RTC_MAX_GUIDANCE_WEIGHT
        if rtc_supported:
            errors.append(
                "rtc max_guidance_weight must be a positive number, got "
                f"{raw_rtc_max_guidance_weight!r}"
            )
    raw_schedules = capabilities.get("prefix_attention_schedules", ("zeros", "ones", "linear", "exp"))
    rtc_supported_schedules = tuple(
        schedule for schedule in (str(item).strip().lower() for item in raw_schedules)
        if schedule in {"zeros", "ones", "linear", "exp"}
    ) if isinstance(raw_schedules, (list, tuple)) else ("zeros", "ones", "linear", "exp")
    if rtc_supported:
        if rtc_max_execution_horizon <= 0:
            errors.append("rtc max_execution_horizon must be positive")
        if not math.isfinite(rtc_max_guidance_weight) or rtc_max_guidance_weight <= 0:
            errors.append("rtc max_guidance_weight must be positive")
        if not rtc_supported_schedules:
            errors.append("rtc capabilities must advertise at least one prefix schedule")
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
        image_resize_mode=str(metadata.get("image_resize_mode", "pad")),
        image_size=image_size,
        image_transport=image_transport,
        image_jpeg_quality=image_jpeg_quality,
        rtc_supported=rtc_supported,
        rtc_max_execution_horizon=rtc_max_execution_horizon,
        rtc_max_guidance_weight=rtc_max_guidance_weight,
        rtc_supported_schedules=rtc_supported_schedules,
        rtc_mode=str(metadata.get("rtc_mode", "")),
        rtc_client_planner_prefix=bool(capabilities.get("client_planner_prefix", False)),
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


def connect_policy(
    host: str,
    port: int,
    arm_side: str,
    arm_mode: str = "single",
    output_mode: str = "auto",
) -> tuple[Any, PolicyProtocol]:
    """Create the official OpenPI client and validate the server handshake."""
    from bimanual_vla.deployment.transport import TimedWebsocketClientPolicy

    logging.info("Connecting to official OpenPI policy at ws://%s:%d ...", host, port)
    policy = TimedWebsocketClientPolicy(host=host, port=port)
    try:
        _tune_policy_socket(policy)
        metadata = policy.get_server_metadata()
        if not isinstance(metadata, dict):
            raise RuntimeError(f"invalid policy metadata: {type(metadata).__name__}")
        protocol = validate_policy_metadata(metadata, arm_side, arm_mode, output_mode)
    except Exception:
        close_policy(policy)
        raise
    logging.info("Policy connected: %s", metadata)
    return policy, protocol


def _tune_policy_socket(policy: Any) -> None:
    """Best-effort low-latency tuning for the official sync WebSocket client."""
    connection = getattr(policy, "_ws", None)
    if connection is None:
        return
    candidates = [
        getattr(connection, name, None)
        for name in ("socket", "sock", "_socket")
    ]
    for candidate in candidates:
        if not isinstance(candidate, socket.socket):
            continue
        try:
            candidate.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            logging.debug("Unable to tune policy WebSocket socket", exc_info=True)
        return


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
    if not math.isfinite(raw):
        raise ExecutionBlocked("gripper target is non-finite")
    # Opening fraction is a bounded actuator request, not a joint angle.
    # Saturate endpoints independently so finite gripper overshoot cannot
    # discard an otherwise valid arm trajectory.
    normalized = float(np.clip(raw, 0.0, 1.0))
    if semantics == GRIPPER_OPENING_FRACTION:
        return normalized
    if semantics == GRIPPER_CLOSED_FRACTION:
        return 1.0 - normalized
    if semantics == GRIPPER_OPENING_METRES:
        metre_tolerance = tolerance * GRIPPER_MAX_M
        if raw < -metre_tolerance or raw > GRIPPER_EXECUTION_MAX_M + metre_tolerance:
            raise ExecutionBlocked(
                f"gripper target {raw:.5f}m exceeds [0,{GRIPPER_EXECUTION_MAX_M:.5f}]m "
                f"tolerance {metre_tolerance:.5f}m"
            )
        return float(np.clip(raw, 0.0, GRIPPER_EXECUTION_MAX_M) / GRIPPER_MAX_M)
    raise ExecutionBlocked(f"unsupported gripper semantics: {semantics!r}")


def _gripper_value_from_opening_fraction(
    opening_fraction: float,
    *,
    semantics: str,
) -> float:
    """Encode a normalized opening fraction in the policy's wire semantics."""
    if semantics == GRIPPER_OPENING_METRES:
        return float(np.clip(opening_fraction * GRIPPER_MAX_M, 0.0, GRIPPER_EXECUTION_MAX_M))
    opening = float(np.clip(opening_fraction, 0.0, 1.0))
    if semantics == GRIPPER_OPENING_FRACTION:
        return opening
    if semantics == GRIPPER_CLOSED_FRACTION:
        return 1.0 - opening
    if semantics == GRIPPER_OPENING_METRES:
        return opening * GRIPPER_MAX_M
    raise ExecutionBlocked(f"unsupported gripper semantics: {semantics!r}")


def apply_gripper_open_lookahead(
    actions: list[DecodedQueuedAction],
    protocol: PolicyProtocol,
    *,
    lookahead_steps: int,
    tolerance: float = DEFAULT_GRIPPER_RANGE_TOLERANCE,
) -> list[DecodedQueuedAction]:
    """Advance opening requests across an accepted action chunk.

    The queue stores decoded absolute targets, so this works for both joint
    and delivery policies without changing the model's delta convention.
    Closing is never anticipated: a future open request can only hold or
    increase the current opening.
    """
    if lookahead_steps <= 0 or not actions:
        return actions
    arm_count = 2 if protocol.arm_mode == "bimanual" else 1
    gripper_stride = 10 if protocol.schema == "delivery" else 7
    output = list(actions)
    for arm_index in range(arm_count):
        target_index = arm_index * gripper_stride + (9 if protocol.schema == "delivery" else 6)
        openings = np.asarray(
            [
                _opening_fraction(
                    float(item.absolute_target[target_index]),
                    semantics=protocol.gripper_semantics,
                    tolerance=tolerance,
                )
                for item in actions
            ],
            dtype=np.float64,
        )
        # Compute each forward-window maximum in one reverse pass instead of
        # rescanning up to ``lookahead_steps`` rows for every action.
        anticipated_values = np.empty_like(openings)
        window: deque[int] = deque()
        window_size = int(lookahead_steps)
        for index in range(len(actions) - 1, -1, -1):
            upper = index + window_size
            while window and window[0] > upper:
                window.popleft()
            while window and openings[window[-1]] <= openings[index]:
                window.pop()
            window.append(index)
            anticipated_values[index] = openings[window[0]]
        for index, item in enumerate(actions):
            anticipated = float(anticipated_values[index])
            if anticipated <= openings[index] + 1e-9:
                continue
            absolute = item.absolute_target.copy()
            absolute[target_index] = _gripper_value_from_opening_fraction(
                anticipated,
                semantics=protocol.gripper_semantics,
            )
            output[index] = replace(item, absolute_target=absolute)
    return output


def check_external_control_streams(
    piper: Any,
    *,
    threshold_hz: float = DEFAULT_EXTERNAL_CONTROL_HZ_THRESHOLD,
) -> None:
    """Reject a Piper already driven by another high-rate control stream.

    The SDK exposes the active JointCtrl/GripperCtrl rates on real hardware.
    Test doubles and older SDKs may omit these getters; those are treated as
    unknown rather than making the deployment unusable.
    """
    for name in ("GetArmJointCtrl", "GetArmGripperCtrl"):
        getter = getattr(piper, name, None)
        if not callable(getter):
            continue
        try:
            control = getter()
            hz = float(getattr(control, "Hz", 0.0) or 0.0)
        except Exception:
            continue
        if math.isfinite(hz) and hz > float(threshold_hz):
            raise ExecutionBlocked(
                f"external {name} stream detected at {hz:.2f}Hz; "
                "refusing concurrent Piper control"
            )


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
            opening = _opening_fraction(
                row[gripper_index],
                semantics=protocol.gripper_semantics,
                tolerance=gripper_range_tolerance,
            )
            row[gripper_index] = _gripper_value_from_opening_fraction(
                opening, semantics=protocol.gripper_semantics,
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


def _freeze_snapshot_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        # CameraFrameSet arrays are already immutable.  Reusing them avoids a
        # second full-frame copy when the control thread freezes an observation
        # for the inference worker.  Mutable arrays (robot feedback and test
        # inputs) still get an owned read-only copy.
        if not value.flags.writeable:
            return value
        frozen = np.array(value, copy=True)
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_snapshot_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_snapshot_value(item) for item in value)
    return value


def _thaw_snapshot_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {str(key): _thaw_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_snapshot_value(item) for item in value]
    return value


@dataclass(frozen=True)
class ObservationSnapshot:
    """Immutable robot/image/RTC state consumed by one inference request."""

    generation: int
    state: np.ndarray
    raw_delivery_state: np.ndarray
    qpos_m: np.ndarray
    captured_at: float
    captured_monotonic: float
    images: Mapping[str, np.ndarray]
    image_timestamps: Mapping[str, float]
    image_monotonic_timestamps: Mapping[str, float]
    image_captured_monotonic: float
    rtc_metadata: Mapping[str, Any]
    execution_metadata: Mapping[str, Any]
    executed_plan_command_count: int

    @property
    def image_state_skew_s(self) -> float:
        return abs(float(self.image_captured_monotonic) - float(self.captured_monotonic))


def make_observation_snapshot(
    *,
    generation: int,
    raw_delivery_state: np.ndarray,
    qpos_m: np.ndarray,
    protocol: PolicyProtocol,
    captured_at: float,
    captured_monotonic: float,
    frame_set: CameraFrameSet,
    rtc_metadata: Mapping[str, Any],
    execution_metadata: Mapping[str, Any],
    executed_plan_command_count: int,
    max_image_state_skew_s: float = DEFAULT_MAX_IMAGE_STATE_SKEW_S,
) -> ObservationSnapshot:
    """Validate and freeze one time-consistent inference observation."""
    state = policy_observation_state(raw_delivery_state, qpos_m, protocol)
    if state.shape != (protocol.state_dim,) or not np.all(np.isfinite(state)):
        raise RuntimeError(
            f"{protocol.arm_mode} {protocol.schema} observation state must be finite "
            f"{protocol.state_dim}D, got {state.shape}"
        )
    expected_camera_keys = (
        {"cam_high", "cam_wrist"}
        if protocol.arm_mode == "single"
        else set(protocol.camera_keys)
    )
    policy_images = frame_set.policy_images or frame_set.images
    if set(policy_images) != expected_camera_keys:
        raise RuntimeError(
            f"camera snapshot keys must be {sorted(expected_camera_keys)}, "
            f"got {sorted(policy_images)}"
        )
    skew_s = abs(float(frame_set.captured_monotonic) - float(captured_monotonic))
    if not math.isfinite(skew_s) or skew_s > float(max_image_state_skew_s):
        raise RuntimeError(
            f"nearest camera/state skew {skew_s * 1000.0:.1f}ms exceeds "
            f"{float(max_image_state_skew_s) * 1000.0:.1f}ms"
        )
    metadata = dict(execution_metadata)
    metadata["inference_generation"] = int(generation)
    return ObservationSnapshot(
        generation=int(generation),
        state=_freeze_snapshot_value(np.asarray(state, dtype=np.float32)),
        raw_delivery_state=_freeze_snapshot_value(
            np.asarray(raw_delivery_state, dtype=np.float32)
        ),
        qpos_m=_freeze_snapshot_value(np.asarray(qpos_m, dtype=np.float32)),
        captured_at=float(captured_at),
        captured_monotonic=float(captured_monotonic),
        images=_freeze_snapshot_value(policy_images),
        image_timestamps=_freeze_snapshot_value(frame_set.timestamps),
        image_monotonic_timestamps=_freeze_snapshot_value(
            frame_set.monotonic_timestamps
        ),
        image_captured_monotonic=float(frame_set.captured_monotonic),
        rtc_metadata=_freeze_snapshot_value(dict(rtc_metadata)),
        execution_metadata=_freeze_snapshot_value(metadata),
        executed_plan_command_count=int(executed_plan_command_count),
    )


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
    # Number of non-hold policy targets that had actually reached the robot at
    # observation capture.  The completed inference may only skip a delayed
    # prefix when the old plan really progressed while inference was running.
    executed_plan_command_count: int = 0
    observation_snapshot: ObservationSnapshot | None = None


@dataclass(frozen=True)
class InferenceCompletion:
    launch: InferenceLaunch
    result: dict[str, Any] | None
    arrived_at: float
    arrived_monotonic: float
    error: BaseException | None = None


@dataclass(frozen=True)
class InferenceWorkerResult:
    """Inference result plus camera/transport timing captured off the control thread."""

    result: dict[str, Any]
    image_timestamps: dict[str, float]
    client_timing: dict[str, Any] | None = None


class AsyncPolicyInference:
    """Single-in-flight inference with event wakeup and control-thread admission."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper-policy")
        self._future: Future | None = None
        self._launch: InferenceLaunch | None = None
        self._ready = threading.Event()

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
        self._ready.clear()
        self._launch = launch
        self._future = self._executor.submit(task)
        # The callback only signals. Queue replacement and safety state remain
        # owned by the control thread, including immediately-completed tasks.
        self._future.add_done_callback(lambda _future: self._ready.set())
        return True

    def wait_until(
        self, deadline: float, on_completion: Callable[[InferenceCompletion], bool | None]
    ) -> bool:
        """Admit completed inference during the gap between actuator ticks.

        A callback may request an early reference-refresh tick by returning
        True. Otherwise keep waiting to the original absolute deadline. This
        worker never issues commands or mutates the controller itself.
        """
        while (remaining := deadline - time.monotonic()) > 0:
            if not self._ready.wait(timeout=remaining):
                return False
            completion = self.poll()
            if completion is not None:
                if on_completion(completion):
                    return True
        return False

    def poll(self) -> InferenceCompletion | None:
        future = self._future
        launch = self._launch
        if future is None or launch is None or not self._ready.is_set():
            return None
        arrived_at = time.time()
        arrived_monotonic = time.monotonic()
        self._future = None
        self._launch = None
        self._ready.clear()
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
            payload = dict(result.result)
            if isinstance(result.client_timing, dict):
                payload["_client_transport_timing"] = dict(result.client_timing)
                received = result.client_timing.get("response_received_monotonic")
                if isinstance(received, (float, int)) and math.isfinite(received):
                    payload["_client_transport_timing"]["response_to_admission_ms"] = max(
                        0.0, (arrived_monotonic - received) * 1000.0
                    )
            result = payload
        return InferenceCompletion(launch, result, arrived_at, arrived_monotonic, error=None)

    def shutdown(self) -> None:
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class PeriodicSchedule:
    """Periodic launch clock with at most one coalesced pending request."""

    frequency_hz: float
    next_at: float
    pending_latest: bool = False

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

    def wakeup_deadline(self, control_deadline: float, now: float, *, in_flight: bool) -> float:
        """Wake for a future request deadline without consuming a slot or busy retrying."""
        if not in_flight and now < self.next_at < control_deadline:
            return self.next_at
        return control_deadline

    def mark_launched(self, now: float) -> None:
        """Consume a coalesced request and rate-limit from its actual launch."""
        self.pending_latest = False
        self.next_at = now + self.period_s


class ManualInferenceTrigger:
    """Polls a small file the GUI writes to for a manually requested launch.

    Used only when ``--no-auto-next-inference`` is set: the control loop
    launches the next synchronous-mode request exactly once per distinct
    value written to this file, instead of automatically as soon as the
    truncated chunk drains. Missing/unreadable files are treated as "no
    request yet" rather than an error, since the GUI creates the file before
    the subprocess starts but a transient race is harmless here.
    """

    def __init__(self, path: str | None) -> None:
        self._path = Path(path) if path else None
        # The GUI writes a sentinel value to this file before the subprocess
        # even starts, so it exists with content the instant this is
        # constructed. Treat whatever is already there as the baseline --
        # only a value written *after* startup (a real click) should fire.
        self._last_seen: str | None = self._read()

    def _read(self) -> str | None:
        if self._path is None:
            return None
        try:
            return self._path.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def consume(self) -> bool:
        """Return True at most once per new value written to the file."""
        value = self._read()
        if not value or value == self._last_seen:
            return False
        self._last_seen = value
        return True


def estimate_event_rate_hz(timestamps: Any) -> float | None:
    """Estimate the observed rate from a short monotonic timestamp history."""
    values = [float(value) for value in timestamps]
    if len(values) < 2:
        return None
    elapsed = values[-1] - values[0]
    if not math.isfinite(elapsed) or elapsed <= 0:
        return None
    return (len(values) - 1) / elapsed


def estimate_single_inflight_ceiling_hz(latency_s: Any) -> float | None:
    """Return the throughput ceiling imposed by one serialized request.

    ``AsyncPolicyInference`` intentionally allows only one request in flight.
    Therefore a capture-to-result latency of ``L`` seconds cannot sustain more
    than approximately ``1 / L`` completed requests per second, even when the
    periodic launch schedule is configured higher.  This is a diagnostic upper
    bound, not a replacement for the observed launch/result rates.
    """
    try:
        value = float(latency_s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return 1.0 / value


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
    blend_profile: str = "linear",
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
    if blend_profile not in {"linear", "smootherstep"}:
        raise ExecutionBlocked(
            f"blend_profile must be linear or smootherstep, got {blend_profile!r}"
        )
    blended: list[DecodedQueuedAction] = []
    for index in range(blend_steps):
        old_action = old_actions[min(index, len(old_actions) - 1)]
        new_action = new_actions[index]
        progress = (index + 1) / blend_steps
        alpha = (
            smootherstep(progress)
            if blend_profile == "smootherstep"
            else progress
        )
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
    workspace_x: tuple[float, float] | None,
    workspace_y: tuple[float, float] | None,
    workspace_z: tuple[float, float] | None,
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
        if bounds is None:
            continue
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


def project_delivery_target_step(
    current_state: np.ndarray,
    target: np.ndarray,
    *,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
) -> tuple[np.ndarray, bool]:
    """Project a distant EEF target onto one bounded Cartesian control step.

    Replanning can replace a 50-step chunk with a target several centimetres
    away from the measured pose.  Rejecting that whole chunk makes a reachable
    arm appear to have an IK failure.  Move only the translation/rotation
    portion toward the target; gripper and target semantics remain unchanged.
    """
    state = np.asarray(current_state, dtype=np.float64)
    output = np.asarray(target, dtype=np.float64).copy()
    if state.shape != (10,) or output.shape != (10,):
        raise ExecutionBlocked("delivery target projection requires 10D state/target")
    changed = False
    translation = output[:3] - state[:3]
    distance = float(np.linalg.norm(translation))
    if math.isfinite(max_translation_step_m) and distance > max_translation_step_m:
        output[:3] = state[:3] + translation * (max_translation_step_m / distance)
        changed = True
    current_rotation = rotation_from_state(state)
    target_rotation = rotation_from_state(output)
    relative = Rotation.from_matrix(target_rotation @ current_rotation.T)
    angle = float(relative.magnitude())
    if math.isfinite(max_rotation_step_rad) and angle > max_rotation_step_rad:
        step_rotation = Rotation.from_rotvec(relative.as_rotvec() * (max_rotation_step_rad / angle))
        output[3:9] = matrix_to_rotation6d(step_rotation.as_matrix() @ current_rotation)
        changed = True
    return output, changed


def delivery_workspace_y(args: argparse.Namespace, side: str) -> tuple[float, float] | None:
    """Return an explicitly configured EEF Y envelope, if any."""
    override = getattr(args, f"workspace_y_{side}", None)
    if override is not None:
        return tuple(float(value) for value in override)
    generic = getattr(args, "workspace_y", None)
    return None if generic is None else tuple(float(value) for value in generic)


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
    joint_limit_tolerance_rad: float = 0.0,
    gripper_range_tolerance: float = DEFAULT_GRIPPER_RANGE_TOLERANCE,
    gripper_semantics: str = GRIPPER_OPENING_FRACTION,
    joint_limits: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Validate absolute joints + opening fraction and return Piper units in SI."""
    qpos = np.asarray(qpos, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
        raise ExecutionBlocked("current joint state is not finite 7D")
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise ExecutionBlocked("joint action is not finite 7D")

    if joint_limit_tolerance_rad < 0:
        raise ValueError("joint_limit_tolerance_rad must be non-negative")
    checked_joints = action[:6].copy()
    limits = JOINT_LIMITS_RAD if joint_limits is None else joint_limits
    for index, (value, bounds) in enumerate(zip(checked_joints, limits), start=1):
        value = float(value)
        if value < bounds[0] - joint_limit_tolerance_rad or value > bounds[1] + joint_limit_tolerance_rad:
            raise ExecutionBlocked(
                f"joint{index} target {value:.5f}rad outside "
                f"[{bounds[0]:.5f}, {bounds[1]:.5f}]"
            )
        # Project bounded model prediction overshoot onto the legal target
        # range before planning. This tolerance never expands command bounds.
        checked_joints[index - 1] = np.clip(value, bounds[0], bounds[1])
    opening_fraction = _opening_fraction(
        action[6],
        semantics=gripper_semantics,
        tolerance=gripper_range_tolerance,
    )
    gripper_m = opening_fraction * GRIPPER_MAX_M

    joint_deltas = np.abs(checked_joints - qpos[:6])
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
    return checked_joints, gripper_m


class ExecutionController:
    def __init__(self, piper: Any | dict[str, Any], args: argparse.Namespace):
        self.args = args
        self.arm_mode = getattr(args, "arm_mode", "single")
        self.arm_side = getattr(args, "arm_side", "right")
        self.pipers = piper if isinstance(piper, dict) else {self.arm_side: piper}
        self.piper = next(iter(self.pipers.values()))
        self.robot_enabled: set[str] = set()
        allow_execution = bool(getattr(args, "allow_execution", False))
        self.state = "client_disabled" if not allow_execution else "ready"
        self.blocked_reason = "local --allow-execution is absent" if not allow_execution else ""
        self.last_command_at: float | None = None
        self.robot_status: dict[str, Any] | None = None
        runtime_rates = resolve_client_runtime_rates(args)
        self.inference_hz = runtime_rates.inference_hz
        self.policy_action_hz = runtime_rates.action_hz
        self.control_hz = runtime_rates.control_hz
        self.min_action_chunk_steps = runtime_rates.min_action_chunk_steps
        self.action_chunk_steps = self.min_action_chunk_steps  # legacy telemetry alias
        self.blend_steps = runtime_rates.blend_steps
        self.blend_profile = str(getattr(args, "blend_profile", "linear"))
        self.async_inference = bool(getattr(args, "async_inference", True))
        self.inference_trigger_mode = str(getattr(args, "inference_trigger_mode", "periodic"))
        self.inference_trigger_step = int(getattr(args, "inference_trigger_step", DEFAULT_INFERENCE_TRIGGER_STEP))
        self._chunk_step_launched_generation = 0
        self.sync_wait_steps = max(
            1, int(getattr(args, "sync_wait_steps", DEFAULT_SYNC_WAIT_STEPS))
        )
        self.auto_next_inference = bool(getattr(args, "auto_next_inference", True))
        self.trajectory_shaping = bool(getattr(args, "trajectory_shaping", False))
        self.trajectory_lowpass = bool(getattr(args, "trajectory_lowpass", True))
        self.trajectory_tracking = bool(getattr(args, "trajectory_tracking", True))
        self.trajectory_speed_limit = bool(getattr(args, "trajectory_speed_limit", True))
        self.trajectory_acceleration_limit = bool(getattr(args, "trajectory_acceleration_limit", True))
        self.trajectory_jerk_limit = bool(getattr(args, "trajectory_jerk_limit", True))
        self.trajectory_lookahead = bool(getattr(args, "trajectory_lookahead", True))
        self.trajectory_max_speed_rad_s = float(
            getattr(args, "trajectory_max_speed_rad_s", DEFAULT_TRAJECTORY_MAX_SPEED_RAD_S)
        )
        self.trajectory_max_acceleration_rad_s2 = float(
            getattr(
                args,
                "trajectory_max_acceleration_rad_s2",
                DEFAULT_TRAJECTORY_MAX_ACCELERATION_RAD_S2,
            )
        )
        self.trajectory_max_jerk_rad_s3 = float(
            getattr(args, "trajectory_max_jerk_rad_s3", DEFAULT_TRAJECTORY_MAX_JERK_RAD_S3)
        )
        self.trajectory_smoothing_cutoff_hz = float(
            getattr(
                args,
                "trajectory_smoothing_cutoff_hz",
                DEFAULT_TRAJECTORY_SMOOTHING_CUTOFF_HZ,
            )
        )
        self.trajectory_tracking_time_constant_s = float(
            getattr(
                args,
                "trajectory_tracking_time_constant_s",
                DEFAULT_TRAJECTORY_TRACKING_TIME_CONSTANT_S,
            )
        )
        self.trajectory_command_lookahead_rad = float(
            getattr(
                args,
                "trajectory_command_lookahead_rad",
                DEFAULT_TRAJECTORY_COMMAND_LOOKAHEAD_RAD,
            )
        )
        self.trajectory_max_tracking_error_rad = float(
            getattr(
                args,
                "trajectory_max_tracking_error_rad",
                DEFAULT_TRAJECTORY_MAX_TRACKING_ERROR_RAD,
            )
        )
        self.gripper_open_lookahead_steps = int(
            getattr(args, "gripper_open_lookahead_steps", 0)
        )
        self.joint_trajectory: JerkLimitedJointTrajectory | None = None
        # Direct construction without new arguments retains legacy integration
        # compatibility. CLI/GUI explicitly select the Ruckig backend.
        self.servo_backend = str(getattr(args, "servo_backend", None) or (
            "legacy" if self.trajectory_shaping else "off"
        ))
        self.servo_loop: ServoLoop | None = None
        self.eef_control_mode = str(getattr(args, "eef_control_mode", "local_ik_joint"))
        if self.eef_control_mode not in {"local_ik_joint", "piper_eef"}:
            raise ValueError("unknown EEF control mode")
        self.native_loop: NativeEEFLoop | None = None
        self._native_epoch = 0
        self._pose_trajectories: dict[str, PoseTrajectory] = {}
        self._pose_reference_at: dict[str, float] = {}
        self.planner_prefix_enabled = False
        # If local planner splicing fails, the remote server must not reuse an
        # unexecuted model chunk from the old observation time frame.
        self.planner_prefix_fallback_reason: str | None = None
        self.planner_prefix_fallback_generation: int | None = None
        self._servo_stopped_safely = True
        self.servo_sample_callback: Callable[[dict[str, Any]], None] | None = None
        self.last_command_monotonic: float | None = None
        self.external_control_checked = False
        self.latency_skip_compensation_steps = int(
            getattr(args, "latency_skip_compensation_steps", 0)
        )
        self.estimated_actuator_delay_s = float(
            getattr(args, "actuator_delay_s", DEFAULT_ACTUATOR_DELAY_S)
        )
        self.gripper_lowpass_alpha = float(
            getattr(args, "gripper_lowpass_alpha", DEFAULT_GRIPPER_LOWPASS_ALPHA)
        )
        self.gripper_lowpass = bool(getattr(args, "gripper_lowpass", True))
        self.gripper_endpoint_filter = bool(getattr(args, "gripper_endpoint_filter", True))
        self.ik_rate_limit = bool(getattr(args, "ik_rate_limit", True))
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
        # Wall-clock duration of the hold that just ended, captured right
        # before ``accept_inference_result`` clears ``hold_started_at``. This
        # is the objective, measurable proof of a synchronous-mode pause: it
        # should closely track ``inference_latency_s`` for the accepted
        # generation, since the next launch fires the instant the queue goes
        # empty in synchronous mode.
        self.last_hold_duration_s: float | None = None
        self.current_timed_target: dict[str, Any] | None = None
        # Keep the decoded target object so telemetry can recompute its signed
        # age against the current snapshot time instead of freezing the value at
        # the last command tick.
        self.current_timed_target_action: DecodedQueuedAction | None = None
        self._filtered_gripper_opening: dict[str, float] = {}
        self._gripper_extreme_candidate: dict[str, str | None] = {}
        self._gripper_extreme_count: dict[str, int] = {}
        self._gripper_extreme_latch: dict[str, str | None] = {}
        self.active_generation = 0
        self.active_chunk_capture_monotonic: float | None = None
        self._next_inference_generation = 1
        self.waiting_fresh_after_enable = False
        self.fresh_inference_required_after_monotonic: float | None = None
        self.enable_hold_settled_at: float | None = None
        self.enable_staged_generation: int | None = None

        self.last_action_chunk_steps = 0
        self.last_composed_action: list[float] | None = None
        self.last_composed_action_at: float | None = None
        self.queue_anchor_state: list[float] | None = None
        self.queue_anchor_qpos_m: list[float] | None = None
        self.queue_anchor_at: float | None = None
        self.queue_loaded_at: float | None = None
        self.queue_image_timestamps: dict[str, float] = {}
        self.queued_action_index: int | None = None
        self.last_queued_action_index: int | None = None
        self.last_wire_action: list[float] | None = None
        self.last_decoded_absolute_target: dict[str, Any] | None = None
        self.last_feedback_at: float | None = None
        self.dropped_action_count = 0
        self.unsafe_drop_count = 0
        self.expired_drop_count = 0
        self.other_drop_count = 0
        self.last_queue_drop_reason = ""
        self.last_queue_drop_kind: str | None = None
        self.unsafe_active = False
        self.ik_recovery_target = None
        self.ik_recovery_started = None
        self.ik_recovery_barrier = None
        self.ik_recovery_fault = None
        self.ik_recovery_prefix_reset = False
        self.ik_recovery_braking = False
        self._recovery_settle_feedback = None
        self._recovery_stable_since = None
        self._recovery_tracking_at = None
        self._recovery_tracking_cycles = 0
        self.last_ik_recovery = None
        self.eef_goal_action = None
        self.eef_goal_held_grippers = None
        self.eef_goal_errors = None
        self.eef_goal_step_count = 0
        self.eef_goal_releasing = False
        self.eef_goal_pose_reset_pending = False
        self.eef_goal_pose_metrics = {}
        self.last_ik_requests = {}
        self._eef_committed_grippers = {}
        self.eef_goal_position_tolerance_m = float(getattr(args, "eef_goal_position_tolerance_m", .005))
        self.eef_goal_rotation_tolerance_rad = float(getattr(args, "eef_goal_rotation_tolerance_rad", .05))
        self.eef_goal_timeout_s = float(getattr(args, "eef_goal_timeout_s", 2.))
        for value in (self.eef_goal_position_tolerance_m, self.eef_goal_rotation_tolerance_rad,
                      self.eef_goal_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("EEF arrival tolerances and timeout must be finite and positive")
        self.executed_plan_command_count = 0
        self.inference_progress_steps = 0
        self.inference_timeline_resynced = False
        self.timeline_resync_active = False

        self.inference_launch_at: float | None = None
        self.inference_capture_at: float | None = None
        self.inference_capture_monotonic: float | None = None
        self.inference_arrival_at: float | None = None
        self.inference_arrival_monotonic: float | None = None
        self.inference_latency_s: float | None = None
        self.inference_skip_steps = 0
        self.inference_elapsed_prefix_steps = 0
        self.inference_blend_steps = 0
        self.inference_generation: int | None = None
        self.inference_old_remaining = 0
        self.inference_launch_count = 0
        self.inference_launch_deferred_count = 0
        self._inference_launch_times: deque[float] = deque(maxlen=INFERENCE_RATE_HISTORY_SIZE)
        self._inference_completion_times: deque[float] = deque(maxlen=INFERENCE_RATE_HISTORY_SIZE)
        self.last_client_transport_timing: dict[str, Any] = {}
        self.last_client_timing_source: str | None = None
        self.last_client_one_way_clock: str | None = None
        self.last_client_one_way_clock_sync_required: bool | None = None
        self.last_transport_generation: int | None = None
        self.last_transport_first_command_generation: int | None = None
        self.rejected_result: dict[str, Any] | None = None
        self.rejected_result_count = 0
        self.queue_underrun = False
        self.queue_underrun_count = 0
        self.queue_underrun_at: float | None = None
        self.control_tick_count = 0
        self.control_overrun_count = 0
        self.command_sequence = 0
        self.last_actuator_command: dict[str, Any] | None = None
        self.last_commanded_joint_positions: np.ndarray | None = None
        self.last_command_feedback: dict[str, Any] | None = None
        self._pending_feedback_command: dict[str, Any] | None = None
        self.tracking_lag_threshold_rad = float(
            getattr(
                args,
                "tracking_lag_threshold_rad",
                DEFAULT_TRACKING_LAG_THRESHOLD_RAD,
            )
        )
        self.tracking_lag_confirm_cycles = int(
            getattr(
                args,
                "tracking_lag_confirm_cycles",
                DEFAULT_TRACKING_LAG_CONFIRM_CYCLES,
            )
        )
        self.tracking_lag_consecutive_cycles = 0
        self.tracking_reference_lag_since = None
        self.tracking_reference_lag_cycles = 0
        self.tracking_reference_lag_duration_s = float(getattr(args, "tracking_reference_lag_duration_s", .5))
        if not math.isfinite(self.tracking_reference_lag_duration_s) or self.tracking_reference_lag_duration_s <= 0:
            raise ValueError("tracking reference lag duration must be finite and positive")
        self.tracking_lag_position_threshold_m = float(getattr(args, "tracking_lag_position_threshold_m", .05))
        self.tracking_lag_rotation_threshold_rad = float(getattr(args, "tracking_lag_rotation_threshold_rad", .18))
        for value in (self.tracking_lag_position_threshold_m, self.tracking_lag_rotation_threshold_rad):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("tracking lag EEF thresholds must be finite and positive")
        self.tracking_lag_last_feedback_at = None
        self.tracking_lag_settle_cycles = 0
        self.tracking_lag_settle_joints = None
        self.tracking_lag_settle_feedback_at = None
        self.tracking_lag_active = False
        self.tracking_lag_started_at: float | None = None
        self.tracking_lag_required_after_monotonic: float | None = None
        self.tracking_lag_peak_error_rad = 0.0
        self.tracking_lag_trigger_count = 0
        self.tracking_lag_trigger_generation: int | None = None
        self.tracking_lag_recovered_generation: int | None = None
        self.arm_hold_targets: dict[str, np.ndarray] = {}
        self.arm_hold_gripper_targets: dict[str, float] = {}
        self.arm_hold_started_at: dict[str, float] = {}
        self.arm_hold_stable_since: dict[str, float] = {}
        self.robot_driver_enable_status: dict[str, dict[str, Any]] = {}
        self.ik_solver: PiperContinuousIK | None = None
        self.ik_workers = None
        self._ik_warm_starts: dict[str, np.ndarray] = {}
        self._local_ik_delivery = False
        self.feedback_diagnostics_stride = 5
        self.rtc_enabled = False
        self.rtc_inference_delay_steps = int(getattr(args, "rtc_inference_delay_steps", 8))
        self.rtc_execution_horizon = DEFAULT_RTC_EXECUTION_HORIZON
        self.rtc_max_guidance_weight = DEFAULT_RTC_MAX_GUIDANCE_WEIGHT
        self.rtc_prefix_attention_schedule = "linear"
        self.rtc_config_revision = 0
        self._rtc_config_sent_revision: int | None = None

    def configure_protocol(self, protocol: PolicyProtocol) -> None:
        if not self.close_servo():
            raise RuntimeHardwareFault("previous actuator owner did not settle; cannot reconfigure")
        if self.eef_control_mode == "piper_eef" and protocol.schema != "delivery":
            raise ValueError("piper_eef requires an EEF/delivery policy; select local_ik_joint for joint policies")
        runtime_rates = resolve_client_runtime_rates(self.args, protocol)
        self._local_ik_delivery = (protocol.schema == "delivery" and self.eef_control_mode == "local_ik_joint")
        self.policy_action_hz = runtime_rates.action_hz
        self.control_hz = runtime_rates.control_hz
        self.inference_hz = runtime_rates.inference_hz
        self.min_action_chunk_steps = runtime_rates.min_action_chunk_steps
        self.action_chunk_steps = self.min_action_chunk_steps
        requested_blend_steps = runtime_rates.blend_steps
        self.rtc_enabled = bool(
            getattr(self.args, "rtc_enabled", True) and protocol.rtc_enabled
        )
        self.planner_prefix_enabled = bool(self.rtc_enabled and protocol.schema == "joint"
                                           and protocol.rtc_mode == "trained"
                                           and self.servo_backend == "ruckig")
        self.planner_prefix_fallback_reason = None
        self.planner_prefix_fallback_generation = None
        if self.planner_prefix_enabled and not protocol.rtc_client_planner_prefix:
            raise RuntimeError("trained RTC reference planning requires a policy server with "
                               "client_planner_prefix support; update/restart the policy server")
        self.rtc_execution_horizon = int(
            getattr(self.args, "rtc_execution_horizon", DEFAULT_RTC_EXECUTION_HORIZON)
        )
        if self.rtc_enabled and not 1 <= self.rtc_execution_horizon <= protocol.action_horizon:
            raise ValueError("RTC execution_horizon must fit the model action chunk")
        if self.rtc_enabled and not 0 <= self.rtc_inference_delay_steps <= self.rtc_execution_horizon:
            raise ValueError("RTC requires 0 <= inference_delay_steps <= execution_horizon")
        self.rtc_max_guidance_weight = float(
            min(
                float(getattr(self.args, "rtc_max_guidance_weight", DEFAULT_RTC_MAX_GUIDANCE_WEIGHT)),
                float(protocol.rtc_max_guidance_weight),
            )
        )
        self.rtc_prefix_attention_schedule = str(
            getattr(self.args, "rtc_prefix_attention_schedule", "linear")
        )
        if self.rtc_enabled and self.rtc_prefix_attention_schedule not in protocol.rtc_supported_schedules:
            raise RuntimeError(
                "client RTC prefix schedule is not supported by the Policy: "
                f"{self.rtc_prefix_attention_schedule!r} not in {protocol.rtc_supported_schedules!r}"
            )
        # A new Policy/WebSocket session must receive the full client-owned
        # RTC configuration once. Subsequent requests carry only progress.
        self.rtc_config_revision += 1
        self._rtc_config_sent_revision = None

        self.blend_steps = 0 if self.rtc_enabled else requested_blend_steps
        # Minimal programmatic/test callers predate the CLI option and retain
        # the historical linear profile.  The CLI always supplies the new
        # smootherstep default explicitly.
        self.blend_profile = str(
            getattr(self.args, "blend_profile", "linear")
        )
        self.async_inference = bool(getattr(self.args, "async_inference", True))
        self.inference_trigger_mode = str(getattr(self.args, "inference_trigger_mode", "periodic"))
        self.inference_trigger_step = int(getattr(self.args, "inference_trigger_step", DEFAULT_INFERENCE_TRIGGER_STEP))
        if self.inference_trigger_mode not in {"periodic", "chunk_step"}:
            raise ValueError("inference trigger mode must be periodic or chunk_step")
        if self.async_inference and self.inference_trigger_mode == "chunk_step" and not 1 <= self.inference_trigger_step < int(protocol.action_horizon):
            raise ValueError(
                f"inference trigger step must be in [1, {int(protocol.action_horizon) - 1}]"
            )
        self._chunk_step_launched_generation = 0
        self.sync_wait_steps = max(
            1, int(getattr(self.args, "sync_wait_steps", DEFAULT_SYNC_WAIT_STEPS))
        )
        self.auto_next_inference = bool(getattr(self.args, "auto_next_inference", True))
        # Unit-test callers that construct a minimal Namespace retain the
        # historical direct-command behavior.  The CLI explicitly enables
        # shaping by default.
        self.trajectory_shaping = bool(getattr(self.args, "trajectory_shaping", False))
        self.gripper_open_lookahead_steps = int(
            getattr(
                self.args,
                "gripper_open_lookahead_steps",
                0,
            )
        )
        if protocol.schema == "delivery" and (self.servo_backend == "ruckig" or self.eef_control_mode == "piper_eef"):
            if self.gripper_open_lookahead_steps:
                logging.warning("EEF arrival gating disables gripper opening lookahead (%s requested steps)",
                                self.gripper_open_lookahead_steps)
            self.gripper_open_lookahead_steps = 0
        self.joint_trajectory = None
        self._ik_warm_starts.clear()
        self._pose_trajectories.clear()
        self._pose_reference_at.clear()
        self.last_command_monotonic = None
        self.external_control_checked = False
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
        self.tracking_lag_threshold_rad = float(
            getattr(
                self.args,
                "tracking_lag_threshold_rad",
                DEFAULT_TRACKING_LAG_THRESHOLD_RAD,
            )
        )
        self.tracking_lag_confirm_cycles = int(
            getattr(
                self.args,
                "tracking_lag_confirm_cycles",
                DEFAULT_TRACKING_LAG_CONFIRM_CYCLES,
            )
        )
        if (
            not math.isfinite(self.tracking_lag_threshold_rad)
            or self.tracking_lag_threshold_rad <= 0
        ):
            raise ValueError("tracking_lag_threshold_rad must be positive")
        if self.tracking_lag_confirm_cycles < 1:
            raise ValueError("tracking_lag_confirm_cycles must be positive")
        self.expected_action_horizon = int(protocol.action_horizon)
        self.pending_actions.clear()
        self.tracking_lag_consecutive_cycles = 0
        self.tracking_reference_lag_since = None
        self.tracking_reference_lag_cycles = 0
        self.tracking_lag_active = False
        self.tracking_lag_started_at = None
        self.tracking_lag_required_after_monotonic = None
        self.tracking_lag_peak_error_rad = 0.0
        self.tracking_lag_trigger_generation = None
        self.tracking_lag_last_feedback_at = None
        if protocol.schema == "delivery" and self.rtc_enabled and getattr(self.args, "pose_trajectory", False):
            logging.info("RTC continuous tracking: point-to-point EEF smoothing applies only during IK recovery")
        if self.arm_hold_targets:
            # A policy reconnect must not cancel a physical measured-pose hold.
            # Keep the enable barrier and require a fresh result on the new link.
            self.enable_staged_generation = None
            self.waiting_fresh_after_enable = True
        else:
            self.enable_staged_generation = None
            self.waiting_fresh_after_enable = False
            self.fresh_inference_required_after_monotonic = None
            self.enable_hold_settled_at = None
        self.timeline_resync_active = False
        self.queue_underrun = False
        logging.info(
            "Action timing: control=%.3g Hz periodic/retry=%.3g Hz trigger=%s step=%d expected_chunk=%d "
            "minimum_chunk=%d blend=%d actuator_delay=%.4fs latency_compensation=%d",
            self.control_hz,
            self.inference_hz,
            self.inference_trigger_mode,
            self.inference_trigger_step,
            self.expected_action_horizon,
            self.min_action_chunk_steps,
            self.blend_steps,
            self.estimated_actuator_delay_s,
            self.latency_skip_compensation_steps,
        )
        servo = self._servo_config()
        logging.info(
            "Piper execution: firmware speed=%d%% backend=%s "
            "requested trajectory speed=%.3f rad/s effective trajectory speed=%.3f rad/s "
            "acceleration=%.3f rad/s^2 jerk=%.3f rad/s^3",
            int(getattr(self.args, "speed_pct", DEFAULT_INFERENCE_SPEED_PCT)), self.servo_backend,
            self.trajectory_max_speed_rad_s, servo.max_velocity,
            servo.max_acceleration, servo.max_jerk,
        )
        if self._local_ik_delivery:
            logging.info("Local IK: minimum_motion=%s position_tolerance=%.4fm rotation_tolerance=%.4frad "
                         "servo_synchronization=%s",
                bool(getattr(self.args, "ik_minimum_motion", True)),
                float(getattr(self.args, "ik_position_tolerance_m", DEFAULT_IK_POSITION_TOLERANCE_M)),
                float(getattr(self.args, "ik_rotation_tolerance_rad", DEFAULT_IK_ROTATION_TOLERANCE_RAD)),
                "per_arm" if servo.independent_arms else "all_joints")
        if not math.isclose(
            self.policy_action_hz,
            self.control_hz,
            rel_tol=1e-3,
            abs_tol=1e-3,
        ):
            raise ValueError(
                "control_hz must match policy_action_hz (policy reference rate); configure servo-hz for "
                f"actuator resampling: control={self.control_hz:.6g}Hz "
                f"policy={self.policy_action_hz:.6g}Hz"
            )

    def run_configuration(self) -> dict[str, Any]:
        """Return stable effective settings for the deployment run record."""
        return {
            "eef_control_mode": self.eef_control_mode,
            "tracking_lag_guard": {
                "joint_threshold_rad": self.tracking_lag_threshold_rad,
                "position_threshold_m": self.tracking_lag_position_threshold_m,
                "rotation_threshold_rad": self.tracking_lag_rotation_threshold_rad,
                "confirm_cycles": self.tracking_lag_confirm_cycles,
                "reference_duration_s": self.tracking_reference_lag_duration_s,
            },
            "ik_execution": {
                "backend": ("piper_firmware" if self.eef_control_mode == "piper_eef" else
                            "parallel_processes" if self.ik_workers is not None else "in_process"),
                "worker_pids": ({s: p.pid for s, p in self.ik_workers.processes.items()}
                                if self.ik_workers is not None else {}),
                "solver": "firmware" if self.eef_control_mode == "piper_eef" else "analytic_jacobian",
                "minimum_motion": bool(self._local_ik_delivery and getattr(self.args, "ik_minimum_motion", True)),
                "position_tolerance_m": float(getattr(self.args, "ik_position_tolerance_m", DEFAULT_IK_POSITION_TOLERANCE_M)),
                "rotation_tolerance_rad": float(getattr(self.args, "ik_rotation_tolerance_rad", DEFAULT_IK_ROTATION_TOLERANCE_RAD)),
                "minimum_motion_budget_s": .008,
            },
            "servo": {
                "backend": self.servo_backend,
                "scope": "startup_return_only" if self.eef_control_mode == "piper_eef" else "policy_execution",
                "firmware_speed_pct": int(getattr(self.args, "speed_pct", DEFAULT_INFERENCE_SPEED_PCT)),
                "requested_max_velocity_rad_s": self.trajectory_max_speed_rad_s,
                **vars(self._servo_config()),
            },
            "pose_trajectory": {
                "enabled": bool(getattr(self.args, "pose_trajectory", False)),
                "rtc_streaming_bypassed": bool(self.rtc_enabled and getattr(self.args, "pose_trajectory", False)),
                "kind": "online_cartesian_reference",
                "rtc_prefix_modified": False,
                "config": {name: float(getattr(self.args, "pose_" + name, default))
                           for name, default in vars(PoseTrajectoryConfig()).items()},
            },
            "effective_joint_limits_rad": {side: effective_limits(piper, JOINT_LIMITS_RAD).tolist()
                                           for side, piper in self.pipers.items()},
            "inference_completion": "event_wakeup",
            "immediate_reference_refresh": self.servo_backend == "ruckig",
            "processing_latency_target_ms": 150.0,
            "control_hz": float(self.control_hz),
            "inference_hz": float(self.inference_hz),
            "policy_action_hz": float(self.policy_action_hz),
            "async_inference": bool(self.async_inference),
            "inference_trigger_mode": self.inference_trigger_mode,
            "inference_trigger_step": int(self.inference_trigger_step),
            "sync_wait_steps": int(self.sync_wait_steps),
            "auto_next_inference": bool(self.auto_next_inference),
            "blend_steps": int(self.blend_steps),
            "latency_skip_compensation_steps": int(self.latency_skip_compensation_steps),
            "rtc": {
                "enabled": bool(self.rtc_enabled),
                "prefix_source": ("client_planner" if self.planner_prefix_enabled else "server_previous_chunk"),
                "planner_prefix_enabled": bool(self.planner_prefix_enabled),
                "planner_prefix_fallback_reason": self.planner_prefix_fallback_reason,
                "planner_prefix_fallback_generation": self.planner_prefix_fallback_generation,
                "inference_delay_steps": int(self.rtc_inference_delay_steps),
                "execution_horizon": int(self.rtc_execution_horizon),
                "max_guidance_weight": float(self.rtc_max_guidance_weight),
                "prefix_attention_schedule": self.rtc_prefix_attention_schedule,
            },
            "trajectory_spike_suppression": {
                "enabled": bool(getattr(self.args, "trajectory_spike_suppression", False)),
                "horizon_steps": int(getattr(
                    self.args, "rtc_execution_horizon", DEFAULT_RTC_EXECUTION_HORIZON
                )),
                "action_hz": float(self.policy_action_hz),
                "algorithm": "local_joint_acceleration_spike_suppression",
                "accel_floor_rad_s2": 1.2,
                "outlier_sigma": 1.5,
                "max_correction_rad": 0.03,
                "passes": 2,
            },
            "trajectory_shaper": {
                "enabled": self.trajectory_shaping and self.servo_backend == "legacy",
                "lowpass_enabled": self.trajectory_lowpass,
                "tracking_enabled": self.trajectory_tracking,
                "speed_limit_enabled": self.trajectory_speed_limit,
                "acceleration_limit_enabled": self.trajectory_acceleration_limit,
                "jerk_limit_enabled": self.trajectory_jerk_limit,
                "lookahead_enabled": self.trajectory_lookahead,
                "max_speed_rad_s": self.trajectory_max_speed_rad_s,
                "max_acceleration_rad_s2": self.trajectory_max_acceleration_rad_s2,
                "max_jerk_rad_s3": self.trajectory_max_jerk_rad_s3,
                "smoothing_cutoff_hz": self.trajectory_smoothing_cutoff_hz,
                "tracking_time_constant_s": self.trajectory_tracking_time_constant_s,
                "command_lookahead_rad": self.trajectory_command_lookahead_rad,
                "max_tracking_error_rad": self.trajectory_max_tracking_error_rad,
            },
        }

    @property
    def pending_action_count(self) -> int:
        return len(self.pending_actions)

    def _trajectory_limits(self, qpos_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build per-joint limits while leaving gripper dimensions unconstrained."""
        qpos = np.asarray(qpos_m, dtype=np.float64)
        lower = np.full(qpos.shape, -np.inf, dtype=np.float64)
        upper = np.full(qpos.shape, np.inf, dtype=np.float64)
        arm_count = len(qpos) // 7
        if len(qpos) != arm_count * 7 or arm_count not in {1, 2}:
            raise ExecutionBlocked(f"trajectory qpos must be 7D or 14D, got {qpos.shape}")
        sides = ("left", "right") if arm_count == 2 else (self.arm_side,)
        for arm_index, side in enumerate(sides):
            offset = arm_index * 7
            limits = effective_limits(self.pipers[side], JOINT_LIMITS_RAD)
            lower[offset : offset + 6] = limits[:, 0]
            upper[offset : offset + 6] = limits[:, 1]
        return lower, upper

    def _shape_prepared_targets(
        self,
        qpos_m: np.ndarray,
        prepared: dict[str, tuple[np.ndarray, float]],
        command_pipeline: dict[str, dict[str, Any]],
        sides: tuple[str, ...],
        *,
        now_monotonic: float,
    ) -> list[int]:
        """Shape all arms as one vector so a bimanual boundary stays coordinated."""
        if not self.trajectory_shaping or self.servo_backend != "legacy":
            return []
        qpos = np.asarray(qpos_m, dtype=np.float32)
        lower, upper = self._trajectory_limits(qpos)
        if self.joint_trajectory is None or self.joint_trajectory.lower.shape != qpos.shape:
            self.joint_trajectory = JerkLimitedJointTrajectory(
                qpos,
                lower,
                upper,
                max_speed_rad_s=self.trajectory_max_speed_rad_s,
                max_acceleration_rad_s2=self.trajectory_max_acceleration_rad_s2,
                max_jerk_rad_s3=self.trajectory_max_jerk_rad_s3,
                smoothing_cutoff_hz=self.trajectory_smoothing_cutoff_hz,
                tracking_time_constant_s=self.trajectory_tracking_time_constant_s,
                command_lookahead_rad=self.trajectory_command_lookahead_rad,
                max_tracking_error_rad=self.trajectory_max_tracking_error_rad,
                lowpass_enabled=self.trajectory_lowpass,
                tracking_enabled=self.trajectory_tracking,
                speed_limit_enabled=self.trajectory_speed_limit,
                acceleration_limit_enabled=self.trajectory_acceleration_limit,
                jerk_limit_enabled=self.trajectory_jerk_limit,
                lookahead_enabled=self.trajectory_lookahead,
            )
        proposed = qpos.copy()
        for index, side in enumerate(sides):
            joints, _ = prepared[side]
            proposed[index * 7 : index * 7 + 6] = np.asarray(joints, dtype=np.float32)
        dt = (
            1.0 / max(self.control_hz, 1.0)
            if self.last_command_monotonic is None
            else float(
                np.clip(now_monotonic - self.last_command_monotonic, 0.5 / max(self.control_hz, 1.0), 1.5 / max(self.control_hz, 1.0))
            )
        )
        try:
            shaped, held = self.joint_trajectory.update(qpos, proposed, dt)
        except TrajectoryTrackingError as exc:
            raise ExecutionBlocked(str(exc)) from exc
        for index, side in enumerate(sides):
            _, gripper_m = prepared[side]
            target_joints = shaped[index * 7 : index * 7 + 6].copy()
            prepared[side] = (target_joints, gripper_m)
            command_pipeline[side]["trajectory_shaped"] = True
            command_pipeline[side]["trajectory_held_joint_indices"] = [
                int(item - index * 7)
                for item in held
                if index * 7 <= item < index * 7 + 6
            ]
            command_pipeline[side]["commanded_joints_rad"] = target_joints.tolist()
        return held

    def _servo_config(self) -> ServoConfig:
        gripper_speed = DEFAULT_GRIPPER_SPEED_M_S
        if self.planner_prefix_enabled:
            step = getattr(self.args, "max_joint_gripper_step", None)
            if step is None:
                metres = getattr(self.args, "max_joint_gripper_step_m", None)
                step = float(metres) / GRIPPER_MAX_M if metres is not None else 0.25
            # A limit formerly applied per 20 Hz target is now a continuous
            # rate bound on the materialized gripper plan.
            gripper_speed = min(gripper_speed, float(step) * GRIPPER_MAX_M * self.policy_action_hz)
        return ServoConfig(
            hz=float(getattr(self.args, "servo_hz", DEFAULT_SERVO_HZ)),
            max_velocity=effective_speed(self.pipers, self.trajectory_max_speed_rad_s,
                speed_pct=int(getattr(self.args, 'speed_pct', DEFAULT_INFERENCE_SPEED_PCT))),
            max_acceleration=self.trajectory_max_acceleration_rad_s2,
            max_jerk=self.trajectory_max_jerk_rad_s3,
            max_tracking_error=self.trajectory_max_tracking_error_rad,
            target_timeout_s=float(getattr(self.args, "servo_target_timeout_s", 0.15)),
            follow_waypoint_velocity=bool(getattr(self.args, "servo_waypoint_velocity", False)),
            independent_arms=(self._local_ik_delivery
                              and bool(getattr(self.args, "ik_independent_arms", True))),
            max_gripper_velocity=gripper_speed,
        )

    def close_servo(self) -> bool:
        self._pose_trajectories.clear()
        self._pose_reference_at.clear()
        if self.native_loop is not None:
            self._servo_stopped_safely = self.native_loop.close()
            self.native_loop = None
        if self.servo_loop is not None:
            self._servo_stopped_safely = self.servo_loop.close()
            self.servo_loop = None
        return self._servo_stopped_safely

    def configure_ik_workers(self, protocol):
        enabled = (protocol.schema == "delivery"
                   and self.eef_control_mode == "local_ik_joint"
                   and bool(getattr(self.args, "allow_execution", False))
                   and bool(getattr(self.args, "ik_worker_processes", True))
                   and self.ik_solver is None)
        if not enabled:
            self.close_ik_workers()
        elif self.ik_workers is None:
            from bimanual_vla.deployment.ik_workers import PiperIKWorkers
            sides = ("left", "right") if self.arm_mode == "bimanual" else (self.arm_side,)
            self.ik_workers = PiperIKWorkers(sides)
            logging.info("EEF IK workers ready (spawn, no CAN): %s",
                         {s: p.pid for s, p in self.ik_workers.processes.items()})

    def close_ik_workers(self):
        if self.ik_workers is not None:
            self.ik_workers.close()
            self.ik_workers = None

    def native_inference_ready(self):
        if self.native_loop is None:
            return True
        metrics = self.native_loop.metrics()
        return (not metrics["fault"] and not metrics["stopping"]
                and metrics["epoch"] == self._native_epoch
                and (not metrics["holding"] or metrics["settled"]))

    def _native_feedback(self):
        require_can_send_healthy(self.pipers, error_type=RuntimeHardwareFault)
        frames = {}
        for side, piper in self.pipers.items():
            state, qpos = read_output_state(piper, max_feedback_age_s=self.args.max_feedback_age_s)
            status = arm_status_dict(piper)
            if driver_enable_status_dict(piper)["ready"] is not True:
                raise NativeEEFError(f"{side} native driver enable not confirmed")
            limits = effective_limits(piper, JOINT_LIMITS_RAD)
            if (np.any(qpos[:6] < limits[:, 0]-IK_FEEDBACK_LIMIT_TOLERANCE_RAD)
                    or np.any(qpos[:6] > limits[:, 1]+IK_FEEDBACK_LIMIT_TOLERANCE_RAD)):
                raise NativeEEFError(f"{side} native measured joints exceed driver limits")
            stamp = min(float(piper.GetArmJointMsgs().time_stamp), float(piper.GetArmEndPoseMsgs().time_stamp),
                        float(piper.GetArmGripperMsgs().time_stamp), float(status["feedback_timestamp"] or 0))
            frames[side] = dict(reference=EEFReference(state[:3], rotation_from_state(state.copy()), float(qpos[6])),
                                qpos=qpos, status=status, stamp=stamp,
                                joint_limits_rad=limits.tolist())
        return frames

    def _start_native(self):
        if self.servo_loop is not None or not self._servo_stopped_safely:
            raise RuntimeHardwareFault("joint owner must settle and close before native EEF handoff")
        self.close_ik_workers()
        for side, piper in self.pipers.items():
            if side not in self.robot_enabled:
                self._enable_robot(side, piper, read_output_qpos(piper, max_feedback_age_s=self.args.max_feedback_age_s))
        self.arm_hold_targets.clear()
        self.arm_hold_gripper_targets.clear()
        self.arm_hold_started_at.clear()
        self.arm_hold_stable_since.clear()
        self.native_loop = NativeEEFLoop(self.pipers, self._native_feedback,
            hz=self.control_hz, speed_pct=int(getattr(self.args, "speed_pct", DEFAULT_INFERENCE_SPEED_PCT)),
            gripper_effort=int(getattr(self.args, "gripper_effort", 1000)),
            target_timeout_s=max(float(getattr(self.args, "servo_target_timeout_s", .15)), 2/self.control_hz),
            max_feedback_age_s=self.args.max_feedback_age_s,
            check_target=lambda side, value: self._check_eef_xyz_workspace(value.xyz, side),
            check_transport=lambda: require_can_send_healthy(self.pipers, error_type=RuntimeHardwareFault))
        self._servo_stopped_safely = False
        self.native_loop.start()
        self._native_epoch = self.native_loop.metrics()["epoch"]
        self.discard_pending_actions("native MOVE_P handoff; waiting for fresh stable observation", kind="other")
        self.waiting_fresh_after_enable = True
        self.fresh_inference_required_after_monotonic = None
        self.ik_recovery_prefix_reset = True
        logging.info("Native EEF active: acknowledged CAN/MOVE_P; firmware IK; no JointCtrl writer")

    def prepare_native_return_handoff(self):
        if self.eef_control_mode != "piper_eef":
            return
        if self.native_loop is not None or not self._servo_stopped_safely:
            raise RuntimeHardwareFault("native writer did not close safely; joint return prohibited")
        logging.info("Native EEF settled and closed; acknowledging CAN/MOVE_J measured hold before return")
        for side, piper in self.pipers.items():
            self._enable_robot(side, piper, read_output_qpos(piper, max_feedback_age_s=self.args.max_feedback_age_s))

    def _execute_native(self, raw_state, qpos, protocol, feedback_at, now):
        if not getattr(self.args, "allow_execution", False):
            return self._block("client_disabled", "local --allow-execution is absent")
        if protocol.schema != "delivery":
            return self._block("blocked", "native EEF requires delivery policy")
        try:
            if not self.external_control_checked and getattr(self.args, "reject_external_control_streams", False):
                for piper in self.pipers.values():
                    check_external_control_streams(piper, threshold_hz=float(getattr(
                        self.args, "external_control_hz_threshold", DEFAULT_EXTERNAL_CONTROL_HZ_THRESHOLD)))
                self.external_control_checked = True
            if self.native_loop is None:
                self._start_native()
                return False
            metrics = self.native_loop.metrics()
            if metrics["fault"]:
                raise NativeEEFError(metrics["fault"])
            if metrics["epoch"] != self._native_epoch:
                self._native_epoch = metrics["epoch"]
                self.discard_pending_actions("native hold invalidated RTC suffix", kind="other")
                self.ik_recovery_prefix_reset = True
                self.tracking_lag_active = True
                self.tracking_lag_required_after_monotonic = None
                self._pose_trajectories.clear()
                self._pose_reference_at.clear()
            if metrics["holding"] and not metrics["settled"]:
                self.state, self.blocked_reason = "holding", "native EEF waiting for measured settling"
                return False
            if self.tracking_lag_active:
                if self.tracking_lag_required_after_monotonic is None:
                    self.tracking_lag_required_after_monotonic = time.monotonic()
                self.state, self.blocked_reason = "holding", "native hold settled; waiting for post-settle inference"
                return False
            if self.waiting_fresh_after_enable:
                if self.fresh_inference_required_after_monotonic is None:
                    self.fresh_inference_required_after_monotonic = time.monotonic()
                self.state = "armed"
                return False
            future = self._first_future_target_index(self.pending_actions, self._estimated_execution_time(now))
            dropped = len(self.pending_actions) if future is None else future
            if dropped:
                self._record_queue_drop(dropped, "native timed targets expired", kind="expired")
                del self.pending_actions[:dropped]
            if not self.pending_actions:
                self.native_loop.hold("queue_underrun")
                self._mark_queue_underrun(holding=self.last_safe_target is not None)
                return False
            original = self.pending_actions[0]
            queued = self._filter_gripper_target(original, qpos, protocol)
            targets, pipeline = {}, {}
            for index, side in enumerate(self.pipers):
                current = np.asarray(raw_state[index*10:index*10+10], dtype=float)
                target = queued.absolute_target[index*10:index*10+10].copy()
                self._check_eef_xyz_workspace(target[:3], side)
                target, projected = project_delivery_target_step(current, target,
                    max_translation_step_m=float(getattr(self.args, "max_translation_step_m", DEFAULT_MAX_TRANSLATION_STEP_M)),
                    max_rotation_step_rad=float(getattr(self.args, "max_rotation_step_rad", DEFAULT_MAX_ROTATION_STEP_RAD)))
                xyz, rpy, opening, _ = _check_delivery_absolute_target(current, float(qpos[index*7+6]), target,
                    gripper_semantics=protocol.gripper_semantics,
                    max_translation_step_m=float("inf"), max_rotation_step_rad=float("inf"),
                    max_gripper_step=float(getattr(self.args, "max_gripper_step", DEFAULT_MAX_GRIPPER_STEP)),
                    gripper_range_tolerance=float(getattr(self.args, "gripper_range_tolerance", DEFAULT_GRIPPER_RANGE_TOLERANCE)),
                    workspace_x=getattr(self.args, "workspace_x", None), workspace_y=delivery_workspace_y(self.args, side),
                    workspace_z=getattr(self.args, "workspace_z", None))
                pose_metrics = None
                if not self.rtc_enabled:
                    xyz, rpy, pose_metrics = self._shape_delivery_pose(side, current, xyz, rpy)
                targets[side] = EEFReference(xyz, Rotation.from_euler("xyz", rpy, degrees=True).as_matrix(), opening)
                pipeline[side] = dict(control_path="piper_native_eef", policy_eef_target=original.absolute_target[index*10:index*10+10].tolist(),
                    commanded_joints_rad=None, full_ik_solution_joints_rad=None, pose_trajectory=pose_metrics,
                    pose_trajectory_bypass_reason="rtc_continuous_tracking" if self.rtc_enabled and getattr(self.args, "pose_trajectory", False) else None,
                    projected_for_native_step=projected, piper_endposectrl_units=list(targets[side].wire()))
            sample = self.native_loop.submit(targets)
            for side in targets:
                pipeline[side]["commanded_gripper_m"] = sample["grippers"][side]
            self.pending_actions.pop(0)
            self.last_safe_target = queued
            self.last_queued_action_index = queued.queue_index
            self.last_wire_action = queued.wire_action.tolist()
            self.last_decoded_absolute_target = self._target_telemetry(protocol, queued.absolute_target)
            if not self.pending_actions:
                self.timeline_resync_active = False
            self.queued_action_index = self.pending_actions[0].queue_index if self.pending_actions else None
            self.command_sequence += 1
            self.executed_plan_command_count += 1
            self.state, self.blocked_reason = "executing", ""
            self.queue_underrun = self.hold_active = self.unsafe_active = False
            self.last_command_at, self.last_command_monotonic = time.time(), time.monotonic()
            self.current_timed_target_action = queued
            self.current_timed_target = self._timed_target_telemetry(queued, now_monotonic=now, now_wall=self.last_command_at)
            self.last_actuator_command = dict(trace_version=1, command_sequence=self.command_sequence,
                generation=queued.generation, queue_index=queued.queue_index, source_index=queued.source_index,
                hold=False, command_at=self.last_command_at, command_monotonic=self.last_command_monotonic,
                sides=pipeline, native_eef=sample)
            self._pending_feedback_command = self.last_actuator_command
            return True
        except NativeEEFError as exc:
            raise RuntimeHardwareFault(f"native EEF: {exc}") from exc
        except ExecutionBlocked as exc:
            if isinstance(exc, (RuntimeHardwareFault, PiperFeedbackStaleError)):
                self._block("blocked", str(exc))
                raise
            self.discard_pending_actions(str(exc), kind="other")
            return self._block("ready", str(exc))

    def _shape_delivery_pose(self, side, current_state, target_xyz, target_rpy_deg, *, allow_reanchor=False):
        """Shape only execution references; delivery RTC retains its wire schema."""
        if not getattr(self.args, "pose_trajectory", False):
            return target_xyz, target_rpy_deg, None
        if self.servo_backend != "ruckig" and self.eef_control_mode != "piper_eef":
            raise ExecutionBlocked("pose trajectory requires the guarded Ruckig writer")
        now = time.monotonic()
        try:
            config = PoseTrajectoryConfig(**{
                name: float(getattr(self.args, "pose_" + name, default))
                for name, default in vars(PoseTrajectoryConfig()).items()
            })
            measured_rotation = rotation_from_state(current_state.copy())
            shaper = self._pose_trajectories.get(side)
            if shaper is None:
                shaper = PoseTrajectory(current_state[:3], measured_rotation, config)
            dt = now - self._pose_reference_at[side] if side in self._pose_reference_at else 1.0 / self.control_hz
            reanchored = False
            if (not math.isfinite(float(dt)) or dt <= 0 or dt > config.max_update_dt_s) and allow_reanchor:
                # A delayed inference is a new reference epoch.  Continuing
                # the old Cartesian trajectory would integrate stale velocity
                # for hundreds of milliseconds and then reject the target as
                # a clock fault.  Re-anchor at fresh measured feedback and
                # take one bounded control-period step instead.
                shaper = PoseTrajectory(current_state[:3], measured_rotation, config)
                dt = 1.0 / max(self.control_hz, 1.0)
                reanchored = True
            xyz, rotation, metrics = shaper.update(
                target_xyz, Rotation.from_euler("xyz", target_rpy_deg, degrees=True).as_matrix(),
                dt=dt, feedback_xyz=current_state[:3], feedback_rotation=measured_rotation)
            for index, axis in enumerate("xyz"):
                if axis == "y":
                    bounds = delivery_workspace_y(self.args, side)
                else:
                    bounds = getattr(self.args, "workspace_" + axis, None)
                if bounds is not None and not bounds[0] <= xyz[index] <= bounds[1]:
                    raise ValueError("smoothed pose reference left the configured workspace")
            self._pose_trajectories[side] = shaper
            self._pose_reference_at[side] = now
            if reanchored:
                metrics["reference_reanchored_after_delay"] = True
            metrics["reference_rotation6d"] = matrix_to_rotation6d(rotation).tolist()
            metrics["raw_rpy_deg"] = np.asarray(target_rpy_deg).tolist()
            return xyz, Rotation.from_matrix(rotation).as_euler("xyz", degrees=True), metrics
        except ValueError as exc:
            raise ExecutionBlocked(str(exc)) from exc

    def _servo_publish_target(self, qpos: np.ndarray,
                              prepared: dict[str, tuple[np.ndarray, float]],
                              sides: tuple[str, ...], *, hold: bool,
                              context: dict[str, Any], protocol: PolicyProtocol | None = None) -> dict[str, Any]:
        if self.native_loop is not None:
            raise ServoError("native EEF owns actuators; joint publication prohibited")
        if self.servo_loop is None:
            if not self._servo_stopped_safely:
                raise ServoError("previous servo did not stop safely; reinitialize the robot controller")
            lower, upper = self._trajectory_limits(qpos)

            def feedback() -> np.ndarray:
                # Independently refresh hardware state on every servo cycle.
                require_can_send_healthy(self.pipers, error_type=RuntimeHardwareFault)
                for side in sides:
                    status = arm_status_dict(self.pipers[side])
                    if not self._piper_can_joint_mode_ready(status):
                        raise ServoError(f"{side} Piper left healthy CAN/MOVE_J mode")
                    if driver_enable_status_dict(self.pipers[side])["ready"] is False:
                        raise ServoError(f"{side} Piper driver enable was lost")
                return np.concatenate([read_output_qpos(
                    self.pipers[side], max_feedback_age_s=float(getattr(
                        self.args, "max_feedback_age_s", DEFAULT_FEEDBACK_MAX_AGE_S
                    ))) for side in sides])

            def publish(command: np.ndarray) -> dict[str, Any]:
                traces: dict[str, dict[str, Any]] = {side: {} for side in sides}
                for side in sides:
                    self.pipers[side].ModeCtrl(0x01, 0x01, int(getattr(self.args, "speed_pct", DEFAULT_INFERENCE_SPEED_PCT)), 0x00)
                    traces[side]["mode_ctrl_monotonic"] = time.monotonic()
                for index, side in enumerate(sides):
                    raw = np.rint(command[index * 7:index * 7 + 6] * RAD_FACTOR).astype(np.int64)
                    self.pipers[side].JointCtrl(*map(int, raw))
                    traces[side].update(piper_jointctrl_units=raw.tolist(),
                                        joint_ctrl_at=time.time(), joint_ctrl_monotonic=time.monotonic())
                for index, side in enumerate(sides):
                    raw = round(float(command[index * 7 + 6]) * GRIPPER_FACTOR)
                    self.pipers[side].GripperCtrl(raw, int(getattr(self.args, "gripper_effort", 1000)), 0x01, 0)
                    traces[side].update(piper_gripperctrl_units=raw,
                                        gripper_ctrl_at=time.time(), gripper_ctrl_monotonic=time.monotonic())
                return traces

            self.servo_loop = ServoLoop(qpos, lower, upper, self._servo_config(), feedback,
                                        publish, self.servo_sample_callback)
        target = np.asarray(qpos, dtype=np.float64).copy()
        for index, side in enumerate(sides):
            joints, gripper = prepared[side]
            target[index * 7:index * 7 + 6] = joints
            target[index * 7 + 6] = gripper
        discard_chunk = False
        if (self.planner_prefix_enabled and not hold and protocol is not None
                and self.planner_prefix_fallback_generation != context["generation"]
                and not self.servo_loop.has_chunk(context["generation"])):
            rows = [r for r in self.pending_actions if r.generation == context["generation"]]
            if not rows:
                raise ServoError("no queued chunk available for RTC reference planning")
            planned_targets = []
            for row in rows:
                value = np.empty_like(target)
                for index, side in enumerate(sides):
                    sl = slice(index*7, index*7+7)
                    try:
                        joints, gripper = build_checked_joint_target(
                            qpos[sl], np.asarray(row.absolute_target)[sl],
                            joint_limits=effective_limits(self.pipers[side], JOINT_LIMITS_RAD),
                            max_joint_step_rad=float("inf"),
                            max_gripper_step=float("inf"),  # Future endpoint; rate bounded by the plan.
                            joint_limit_tolerance_rad=float(getattr(self.args, "joint_limit_tolerance_rad",
                                                                   DEFAULT_JOINT_LIMIT_TOLERANCE_RAD)),
                            gripper_range_tolerance=float(getattr(self.args, "gripper_range_tolerance",
                                                                 DEFAULT_GRIPPER_RANGE_TOLERANCE)),
                            gripper_semantics=protocol.gripper_semantics)
                    except ExecutionBlocked as exc:
                        raise ChunkPlanRejected(
                            f"candidate chunk {row.generation} {side} source_index={row.source_index}: {exc}"
                        ) from exc
                    value[index*7:index*7+6] = joints
                    value[index*7+6] = gripper
                planned_targets.append(value)
            try:
                self.servo_loop.install_chunk(
                    np.asarray(planned_targets), np.array([r.target_monotonic for r in rows]),
                    np.array([r.source_index for r in rows]), generation=context["generation"])
                self.planner_prefix_fallback_reason = None
                self.planner_prefix_fallback_generation = None
            except ChunkPlanRejected as exc:
                # Stream this generation through bounded single-target
                # Ruckig. Keep the client-prefix protocol: without an installed
                # plan the next request carries planned_prefix=None and the
                # policy predicts afresh from measured state. Never promote
                # the rejected raw model curve to a server RTC prefix.
                logging.warning(
                    "RTC chunk splice rejected; using single-target Ruckig for generation %s; "
                    "client planner prefix protocol retained: %s", context["generation"], exc
                )
                self.planner_prefix_fallback_reason = str(exc)[:500]
                self.planner_prefix_fallback_generation = int(context["generation"])
                discard_chunk = True
        if discard_chunk:
            return self.servo_loop.submit(target, hold=hold, context=context, discard_chunk=True)
        return self.servo_loop.submit(target, hold=hold, context=context)

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
        if self.inference_trigger_mode == "chunk_step":
            self._chunk_step_launched_generation = self.active_generation
        self._inference_launch_times.append(float(launch.launched_monotonic))

    def chunk_step_launch_due(self) -> bool:
        """Trigger once after the configured source row was actually sent."""
        target = self.last_safe_target
        return bool(
            self.active_generation > 0
            and self._chunk_step_launched_generation != self.active_generation
            and target is not None
            and target.generation == self.active_generation
            and target.source_index is not None
            and target.source_index + 1 >= self.inference_trigger_step
        )

    def async_launch_due(
        self, schedule: PeriodicSchedule, now: float, *, in_flight: bool = False
    ) -> bool:
        if self.ik_recovery_fault or self.ik_recovery_target is not None:
            return False
        if self.native_loop is not None and not self.native_inference_ready():
            return False
        if self.tracking_lag_active and self.tracking_lag_required_after_monotonic is None:
            return False
        if self.inference_trigger_mode == "periodic":
            slot_due = schedule.due(now)
            if slot_due and in_flight:
                # Coalesce all missed slots into one pending request. Capture
                # the observation only after the worker is free so the next
                # request always uses the latest camera frames and feedback.
                schedule.pending_latest = True
                self.record_launch_deferred()
            return not in_flight and (slot_due or schedule.pending_latest)
        if self.chunk_step_launch_due():
            return True
        recovering = (
            self.active_generation == 0
            or not self.pending_actions
            or self.rejected_result is not None
            or self.tracking_lag_active
            or self.waiting_fresh_after_enable
        )
        return recovering and schedule.due(now)

    def record_inference_completion(self, arrived_monotonic: float) -> None:
        """Record every completed response, including rejected action chunks."""
        self._inference_completion_times.append(float(arrived_monotonic))

    def record_launch_deferred(self) -> None:
        self.inference_launch_deferred_count += 1

    def record_control_tick(self, *, overrun: bool = False) -> None:
        self.control_tick_count += 1
        if overrun:
            self.control_overrun_count += 1

    @staticmethod
    def _finite_timing_value(value: Any, *, allow_negative: bool = False) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result) or (result < 0 and not allow_negative):
            return None
        return result

    def _record_client_transport_timing(
        self, result: dict[str, Any], *, generation: int
    ) -> None:
        if not isinstance(result, dict):
            return
        raw = result.get("_client_transport_timing")
        if not isinstance(raw, dict):
            self.last_client_transport_timing = {}
            self.last_client_timing_source = None
            self.last_client_one_way_clock = None
            self.last_client_one_way_clock_sync_required = None
            self.last_transport_generation = int(generation)
            self.last_transport_first_command_generation = None
            return
        allowed = {
            "client_pack_ms",
            "client_send_ms",
            "client_wait_response_ms",
            "client_unpack_ms",
            "client_transport_total_ms",
            "request_bytes",
            "response_bytes",
            "response_to_admission_ms",
            "client_codec_ms",
            "server_unpack_ms",
            "unattributed_rpc_ms",
            "camera_capture_ms",
            "observation_upload_ms",
            "model_inference_ms",
            "result_download_ms",
            "network_transport_total_ms",
            "round_trip_ms",
            "request_sent_at",
            "server_request_received_at",
            "server_model_completed_at",
            "server_response_ready_at",
            "response_received_at",
            "response_received_monotonic",
            "inference_generation",
            "client_observation_upload_ms",
            "client_result_download_ms",
            "client_network_transport_total_ms",
            "non_model_rtt_ms",
        }
        cleaned = {
            key: value
            for key in allowed
            if (value := self._finite_timing_value(raw.get(key))) is not None
        }
        if not cleaned:
            self.last_client_transport_timing = {}
            self.last_client_timing_source = None
            self.last_client_one_way_clock = None
            self.last_client_one_way_clock_sync_required = None
            self.last_transport_generation = int(generation)
            self.last_transport_first_command_generation = None
            return
        cleaned["generation"] = int(generation)
        cleaned["timing_valid"] = True
        self.last_client_timing_source = str(raw.get("timing_source") or "") or None
        self.last_client_one_way_clock = str(raw.get("one_way_timing_clock") or "") or None
        sync_required = raw.get("one_way_timing_requires_clock_sync")
        self.last_client_one_way_clock_sync_required = (
            bool(sync_required) if isinstance(sync_required, bool) else None
        )
        self.last_client_transport_timing = cleaned
        self.last_transport_generation = int(generation)
        self.last_transport_first_command_generation = None

    def metadata(
        self,
        *,
        rtc_metadata: Mapping[str, Any] | None = None,
        compact: bool = False,
    ) -> dict[str, Any]:
        timing_snapshot_at = time.time()
        timing_snapshot_monotonic = time.monotonic()
        rtc_snapshot = (
            self._build_rtc_request_metadata(
                None,
                mark_config_sent=False,
                include_config=False,
            )
            if rtc_metadata is None
            else dict(rtc_metadata)
        )
        if compact:
            return {
                "timing_snapshot_at": timing_snapshot_at,
                "allow_execution": bool(getattr(self.args, "allow_execution", False)),
                "execution_state": self.state,
                "blocked_reason": self.blocked_reason,
                "tracking_lag_guard": {
                    "active": self.tracking_lag_active,
                    "trigger_count": self.tracking_lag_trigger_count,
                    "reference_duration_s": self.tracking_reference_lag_duration_s,
                    "evidence": (self.last_command_feedback or {}).get("tracking_lag_evidence"),
                    "sent_joint_error_rad": (self.last_command_feedback or {}).get("max_joint_abs_error_rad"),
                    "reference_joint_error_rad": (self.last_command_feedback or {}).get("reference_joint_error_rad"),
                    "policy_position_error_m": (self.last_command_feedback or {}).get("policy_position_error_m"),
                    "policy_rotation_error_rad": (self.last_command_feedback or {}).get("policy_rotation_error_rad"),
                },
                "policy_action_hz": self.policy_action_hz,
                "command_hz": self.control_hz,
                "control_hz": self.control_hz,
                "inference_hz": self.inference_hz,
                "configured_inference_hz": self.inference_hz,
                "inference_trigger_mode": self.inference_trigger_mode,
                "inference_trigger_step": self.inference_trigger_step,
                "expected_action_horizon": self.expected_action_horizon,
                "min_action_chunk_steps": self.min_action_chunk_steps,
                "action_chunk_steps": self.action_chunk_steps,
                "robot_enabled_sides": sorted(self.robot_enabled),
                "queued_action_count": len(self.pending_actions),
                "queued_action_index": self.queued_action_index,
                "last_queued_action_index": self.last_queued_action_index,
                "active_generation": self.active_generation,
                "inference_generation": self.inference_generation,
                "inference_latency_s": self.inference_latency_s,
                "inference_launch_count": self.inference_launch_count,
                "inference_result_hz": estimate_event_rate_hz(
                    self._inference_completion_times
                ),
                "queue_underrun": self.queue_underrun,
                "hold_active": self.hold_active,
                "unsafe_active": self.unsafe_active,
                "last_feedback_at": self.last_feedback_at,
                "last_command_at": self.last_command_at,
                "last_queue_drop_kind": self.last_queue_drop_kind,
                "last_queue_drop_reason": self.last_queue_drop_reason,
                # The next request carries the previous request's bounded
                # timing report to the Policy telemetry mirror.
                "client_transport_timing": dict(self.last_client_transport_timing),
                "timing_generation": self.last_transport_generation,
                "timing_source": self.last_client_timing_source,
                "one_way_timing_clock": self.last_client_one_way_clock,
                "one_way_timing_requires_clock_sync": self.last_client_one_way_clock_sync_required,
                "round_trip_ms": self.last_client_transport_timing.get("round_trip_ms"),
                "result_to_first_command_ms": self.last_client_transport_timing.get("result_to_first_command_ms"),
                "observation_to_first_command_ms": self.last_client_transport_timing.get("observation_to_first_command_ms"),
                "rtc": rtc_snapshot,
            }
        # Prefer the target that was actually sent on the most recent 20 Hz
        # control tick. ``last_safe_target`` is the fallback hold target and
        # can legitimately have ``hold=False`` while the controller is already
        # holding it after a queue underrun.
        if self.current_timed_target_action is not None:
            current_target = self._timed_target_telemetry(
                self.current_timed_target_action,
                now_monotonic=timing_snapshot_monotonic,
                now_wall=timing_snapshot_at,
            )
        else:
            current_target = self.current_timed_target
            if current_target is None:
                current_target = self._timed_target_telemetry(
                    self.last_safe_target,
                    now_monotonic=timing_snapshot_monotonic,
                    now_wall=timing_snapshot_at,
                )
        plan_target_times = [
            timing_snapshot_at
            + (float(target.target_monotonic) - timing_snapshot_monotonic)
            for target in self.pending_actions
        ]
        return {
            "timing_snapshot_at": timing_snapshot_at,
            "allow_execution": bool(getattr(self.args, "allow_execution", False)),
            "execution_state": self.state,
            "blocked_reason": self.blocked_reason,
            "last_command_at": self.last_command_at,
            "robot_arm_status": self.robot_status,
            "robot_enabled_sides": sorted(self.robot_enabled),
            "robot_driver_enable_status": self.robot_driver_enable_status,
            "robot_enable_hold": {
                "active": bool(self.arm_hold_targets),
                "sides": sorted(self.arm_hold_targets),
                "waiting_fresh_inference": self.waiting_fresh_after_enable,
                "staged_generation": self.enable_staged_generation,
                "settled": self.fresh_inference_required_after_monotonic is not None,
                "settled_at": self.enable_hold_settled_at,
                "hold_age_s": {
                    side: max(
                        0.0, timing_snapshot_monotonic - self.arm_hold_started_at[side]
                    )
                    for side in self.arm_hold_targets
                    if side in self.arm_hold_started_at
                },
                "stable_age_s": {
                    side: max(
                        0.0, timing_snapshot_monotonic - self.arm_hold_stable_since[side]
                    )
                    for side in self.arm_hold_targets
                    if side in self.arm_hold_stable_since
                },
            },
            "policy_action_hz": self.policy_action_hz,
            "command_hz": self.control_hz,
            "control_hz": self.control_hz,
            "inference_hz": self.inference_hz,
            "configured_inference_hz": self.inference_hz,
            "inference_trigger_mode": self.inference_trigger_mode,
            "inference_trigger_step": self.inference_trigger_step,
            "inference_launch_hz": estimate_event_rate_hz(self._inference_launch_times),
            "inference_result_hz": estimate_event_rate_hz(self._inference_completion_times),
            "inference_single_inflight_ceiling_hz": estimate_single_inflight_ceiling_hz(
                self.inference_latency_s
            ),
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
            "unsafe_drop_count": self.unsafe_drop_count,
            "expired_drop_count": self.expired_drop_count,
            "other_drop_count": self.other_drop_count,
            "last_queue_drop_reason": self.last_queue_drop_reason,
            "last_queue_drop_kind": self.last_queue_drop_kind,
            "unsafe_active": self.unsafe_active,
            "executed_plan_command_count": self.executed_plan_command_count,
            "inference_progress_steps": self.inference_progress_steps,
            "inference_timeline_resynced": self.inference_timeline_resynced,
            "timeline_resync_active": self.timeline_resync_active,
            "inference_launch_at": self.inference_launch_at,
            "inference_capture_at": self.inference_capture_at,
            "inference_capture_monotonic": self.inference_capture_monotonic,
            "inference_arrival_at": self.inference_arrival_at,
            "inference_arrival_monotonic": self.inference_arrival_monotonic,
            "inference_latency_s": self.inference_latency_s,
            "inference_skip_steps": self.inference_skip_steps,
            "inference_elapsed_prefix_steps": self.inference_elapsed_prefix_steps,
            "inference_blend_steps": self.inference_blend_steps,
            "inference_generation": self.inference_generation,
            "action_generation": self.active_generation,
            "rtc": {
                "enabled": self.rtc_enabled,
                "algorithm": "real_time_chunking_prefix_guidance",
                "session_id": str(getattr(self.args, "rtc_session_id", "")),
                "inference_delay_steps": self.rtc_inference_delay_steps,
                "execution_horizon": self.rtc_execution_horizon,
                "max_guidance_weight": self.rtc_max_guidance_weight,
                "prefix_attention_schedule": self.rtc_prefix_attention_schedule,
                **rtc_snapshot,
            },
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
            "last_safe_target": self._timed_target_telemetry(
                self.last_safe_target,
                now_monotonic=timing_snapshot_monotonic,
                now_wall=timing_snapshot_at,
            ),
            "timed_target": current_target,
            "current_timed_target": current_target,
            "plan_target_times": plan_target_times,
            "client_transport_timing": dict(self.last_client_transport_timing),
            "timing_generation": self.last_transport_generation,
            "transport_timing_generation": self.last_transport_generation,
            "transport_timing_valid": bool(self.last_client_transport_timing.get("timing_valid", False)),
            "client_timing_source": self.last_client_timing_source,
            "one_way_timing_clock": self.last_client_one_way_clock,
            "one_way_timing_requires_clock_sync": self.last_client_one_way_clock_sync_required,
            "camera_capture_ms": self.last_client_transport_timing.get("camera_capture_ms"),
            "observation_upload_ms": self.last_client_transport_timing.get("observation_upload_ms"),
            "client_observation_upload_ms": self.last_client_transport_timing.get("client_observation_upload_ms"),
            "model_inference_ms": self.last_client_transport_timing.get("model_inference_ms"),
            "result_download_ms": self.last_client_transport_timing.get("result_download_ms"),
            "client_result_download_ms": self.last_client_transport_timing.get("client_result_download_ms"),
            "network_transport_total_ms": self.last_client_transport_timing.get("network_transport_total_ms"),
            "client_network_transport_total_ms": self.last_client_transport_timing.get("client_network_transport_total_ms"),
            "non_model_rtt_ms": self.last_client_transport_timing.get("non_model_rtt_ms"),
            "round_trip_ms": self.last_client_transport_timing.get("round_trip_ms"),
            "request_sent_at": self.last_client_transport_timing.get("request_sent_at"),
            "server_request_received_at": self.last_client_transport_timing.get("server_request_received_at"),
            "server_model_completed_at": self.last_client_transport_timing.get("server_model_completed_at"),
            "server_response_ready_at": self.last_client_transport_timing.get("server_response_ready_at"),
            "response_received_at": self.last_client_transport_timing.get("response_received_at"),
            "result_to_first_command_ms": self.last_client_transport_timing.get("result_to_first_command_ms"),
            "observation_to_first_command_ms": self.last_client_transport_timing.get("observation_to_first_command_ms"),
            "rejected_result": self.rejected_result,
            "rejected_result_count": self.rejected_result_count,
            "control_tick_count": self.control_tick_count,
            "control_overrun_count": self.control_overrun_count,
            "command_sequence": self.command_sequence,
            "last_actuator_command": self.last_actuator_command,
            "last_command_feedback": self.last_command_feedback,
            "estimated_actuator_delay_s": self.estimated_actuator_delay_s,
            "native_eef": self.native_loop.metrics() if self.native_loop is not None else None,
            "eef_control_mode": self.eef_control_mode,
            "servo": (self.servo_loop.metrics() if self.servo_loop is not None else {
                "backend": self.servo_backend, "configured_hz": self._servo_config().hz,
            }),
            "trajectory_shaper": {
                "enabled": self.trajectory_shaping and self.servo_backend == "legacy",
                "lowpass_enabled": self.trajectory_lowpass,
                "tracking_enabled": self.trajectory_tracking,
                "speed_limit_enabled": self.trajectory_speed_limit,
                "acceleration_limit_enabled": self.trajectory_acceleration_limit,
                "jerk_limit_enabled": self.trajectory_jerk_limit,
                "lookahead_enabled": self.trajectory_lookahead,
                "blend_profile": self.blend_profile,
                "max_speed_rad_s": self.trajectory_max_speed_rad_s,
                "max_acceleration_rad_s2": self.trajectory_max_acceleration_rad_s2,
                "max_jerk_rad_s3": self.trajectory_max_jerk_rad_s3,
                "smoothing_cutoff_hz": self.trajectory_smoothing_cutoff_hz,
                "tracking_time_constant_s": self.trajectory_tracking_time_constant_s,
                "command_lookahead_rad": self.trajectory_command_lookahead_rad,
                "max_tracking_error_rad": self.trajectory_max_tracking_error_rad,
                "last_command_monotonic": self.last_command_monotonic,
                "state": (
                    self.joint_trajectory.metrics()
                    if self.joint_trajectory is not None
                    else None
                ),
            },
            "external_control_interlock": {
                "enabled": bool(
                    getattr(self.args, "reject_external_control_streams", False)
                ),
                "checked": self.external_control_checked,
                "threshold_hz": float(
                    getattr(
                        self.args,
                        "external_control_hz_threshold",
                        DEFAULT_EXTERNAL_CONTROL_HZ_THRESHOLD,
                    )
                ),
            },
            "tracking_lag_guard": {
                "active": self.tracking_lag_active,
                "reference_duration_s": self.tracking_reference_lag_duration_s,
                "reference_lag_since": self.tracking_reference_lag_since,
                "reference_lag_cycles": self.tracking_reference_lag_cycles,
                "position_threshold_m": self.tracking_lag_position_threshold_m,
                "rotation_threshold_rad": self.tracking_lag_rotation_threshold_rad,
                "waiting_for_settle": self.tracking_lag_active and self.tracking_lag_required_after_monotonic is None,
                "threshold_rad": self.tracking_lag_threshold_rad,
                "confirm_cycles": self.tracking_lag_confirm_cycles,
                "consecutive_cycles": self.tracking_lag_consecutive_cycles,
                "started_at": self.tracking_lag_started_at,
                "required_after_monotonic": (
                    self.tracking_lag_required_after_monotonic
                ),
                "peak_error_rad": self.tracking_lag_peak_error_rad,
                "trigger_count": self.tracking_lag_trigger_count,
                "trigger_generation": self.tracking_lag_trigger_generation,
                "recovered_generation": self.tracking_lag_recovered_generation,
            },
            "gripper_filter": {
                "lowpass_enabled": self.gripper_lowpass,
                "endpoint_filter_enabled": self.gripper_endpoint_filter,
                "lowpass_alpha": self.gripper_lowpass_alpha,
                "hysteresis": self.gripper_hysteresis,
                "confirm_steps": self.gripper_confirm_steps,
                "opening_fraction": dict(self._filtered_gripper_opening),
                "extreme_latch": dict(self._gripper_extreme_latch),
            },
            "safety_profile": SAFETY_PROFILE,
            "delivery_command_mode": "continuous_ik_joint",
            "continuous_ik": {
                "shrink_recovery": {
                    "enabled": self.servo_backend == "ruckig",
                    "fractions": [0.5, 0.25],
                    "moving": self.ik_recovery_target is not None,
                    "required_after_monotonic": self.ik_recovery_barrier,
                    "prefix_reset": self.ik_recovery_prefix_reset,
                    "braking_for_handoff": self.ik_recovery_braking,
                    "sent_tracking_cycles": self._recovery_tracking_cycles,
                    "fault": self.ik_recovery_fault,
                    "last_recovery": self.last_ik_recovery,
                    "last_requests": self.last_ik_requests,
                    "original_goal": (self.eef_goal_action.absolute_target.tolist()
                                      if self.eef_goal_action is not None else None),
                    "goal_errors": self.eef_goal_errors,
                    "goal_steps": self.eef_goal_step_count,
                    "pose_reference": self.eef_goal_pose_metrics,
                    "pose_reset_waiting_for_stop": self.eef_goal_pose_reset_pending,
                    "position_tolerance_m": self.eef_goal_position_tolerance_m,
                    "rotation_tolerance_rad": self.eef_goal_rotation_tolerance_rad,
                    "timeout_s": self.eef_goal_timeout_s,
                },
                "rate_limit_enabled": self.ik_rate_limit,
                "max_joint_step_rad": float(getattr(self.args, "ik_max_joint_step_rad", DEFAULT_IK_MAX_JOINT_STEP_RAD)),
                "search_joint_radius_rad": float(getattr(self.args, "ik_search_joint_radius_rad", DEFAULT_IK_SEARCH_JOINT_RADIUS_RAD)),
                    "reference_joint_radius_rad": float(getattr(self.args, "ik_reference_joint_radius_rad", DEFAULT_IK_REFERENCE_JOINT_RADIUS_RAD)),
                "joint_regularization_weight": float(getattr(self.args, "ik_joint_regularization_weight", DEFAULT_IK_JOINT_REGULARIZATION_WEIGHT)),
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
                "workspace_x_m": (list(getattr(self.args, "workspace_x"))
                                  if getattr(self.args, "workspace_x", None) is not None else None),
                "workspace_y_m": (list(getattr(self.args, "workspace_y"))
                                  if getattr(self.args, "workspace_y", None) is not None else None),
                "workspace_z_m": (list(getattr(self.args, "workspace_z"))
                                  if getattr(self.args, "workspace_z", None) is not None else None),
                "blend_targets_rechecked_each_control_step": True,
            },
        }

    def _build_rtc_request_metadata(
        self,
        protocol: PolicyProtocol | None,
        *,
        mark_config_sent: bool,
        include_config: bool = True,
        observation_monotonic: float | None = None,
    ) -> dict[str, Any]:
        """Describe prefix availability at the next inference launch.

        Ruckig joint execution uses a reserved writer plan, never raw model
        history. Legacy paths align server history by source offset and time.
        Runtime RTC configuration is sent only after a connection/config change.
        """
        offset = 0
        previous_generation = self.active_generation or None
        if self.pending_actions:
            same_generation = [
                target
                for target in self.pending_actions
                if target.generation == self.active_generation
            ]
            if same_generation:
                source_index = same_generation[0].source_index
                if source_index is not None:
                    offset = max(offset, int(source_index))
        if (
            self.last_safe_target is not None
            and self.last_safe_target.generation == self.active_generation
            and self.last_safe_target.source_index is not None
        ):
            offset = max(offset, int(self.last_safe_target.source_index) + 1)
        if (observation_monotonic is not None
                and self.active_chunk_capture_monotonic is not None):
            # Prefix offset is a time alignment, not the count of dispatched
            # references. Holds/drops must not rewind the old trajectory.
            elapsed = max(0.0, observation_monotonic - self.active_chunk_capture_monotonic)
            offset = max(offset, int(math.ceil(elapsed * self.policy_action_hz - 1e-6)))
        delay = self.rtc_inference_delay_steps
        if self.unsafe_active or self.ik_recovery_prefix_reset or self.tracking_lag_active:
            # A rejected plan is not an executable prefix. Reusing it would
            # clamp its unsafe targets into successive RTC generations.
            previous_generation = None
            offset = 0
        if self.planner_prefix_enabled or self.planner_prefix_fallback_reason is not None:
            # Ruckig execution and the raw model curve can have very different
            # positions at the same source index. Only the reserved writer
            # trajectory is a valid prefix, regardless of queue progress.
            previous_generation = None
            offset = 0
        if not 0 <= delay <= self.rtc_execution_horizon:
            raise ValueError("RTC requires 0 <= inference_delay_steps <= execution_horizon")
        latency_s = delay / self.policy_action_hz
        payload = {
            "enabled": bool(self.rtc_enabled),
            "inference_delay_steps": delay,
            "previous_chunk_offset_steps": offset,
            "previous_chunk_generation": previous_generation,
            "planner_prefix_enabled": bool(self.planner_prefix_enabled),
            "planner_prefix_fallback_reason": self.planner_prefix_fallback_reason,
            "planner_prefix_fallback_generation": self.planner_prefix_fallback_generation,
            "predicted_capture_to_result_s": float(latency_s),
        }
        if include_config and self._rtc_config_sent_revision != self.rtc_config_revision:
            payload["config_revision"] = int(self.rtc_config_revision)
            payload["config"] = {
                "enabled": bool(self.rtc_enabled),
                "execution_horizon": int(self.rtc_execution_horizon),
                "max_guidance_weight": float(self.rtc_max_guidance_weight),
                "prefix_attention_schedule": str(self.rtc_prefix_attention_schedule),
            }
            if mark_config_sent:
                self._rtc_config_sent_revision = self.rtc_config_revision
        return payload

    def rtc_request_metadata(self, protocol: PolicyProtocol | None, *,
                             observation_monotonic: float | None = None) -> dict[str, Any]:
        """Build request metadata and mark the RTC config as sent.

        This is intentionally the only method that mutates the sent-revision
        marker. Telemetry snapshots use the pure builder below so observing
        controller state cannot consume the first-request configuration.
        """
        payload = self._build_rtc_request_metadata(
            protocol,
            mark_config_sent=True,
            include_config=True,
            observation_monotonic=observation_monotonic,
        )
        if self.planner_prefix_enabled:
            payload["prefix_source"] = "client_planner"
            payload["planned_prefix"] = None
            payload["previous_chunk_generation"] = None
            if self.servo_loop is not None and observation_monotonic is not None:
                prefix = self.servo_loop.rtc_prefix(observation_monotonic,
                                                    self.rtc_inference_delay_steps, self.policy_action_hz)
                if prefix is not None:
                    positions = prefix.pop("positions")
                    for index in range(6, positions.shape[1], 7):
                        positions[:, index] = [
                            _gripper_value_from_opening_fraction(v / GRIPPER_MAX_M,
                                                                semantics=protocol.gripper_semantics)
                            for v in positions[:, index]]
                    prefix["actions"] = positions.tolist()
                    payload["planned_prefix"] = prefix
        return payload

    def _block(self, state: str, reason: str) -> bool:
        # A safety gate always cancels a partially executed recovery target.
        if self.ik_recovery_target is not None:
            self.ik_recovery_target = None
            self.ik_recovery_barrier = time.monotonic()
            if self.eef_goal_action is not None and not self.ik_recovery_fault:
                self.ik_recovery_fault = "original EEF goal interrupted by safety gate; stop and restart required"
        self.eef_goal_action = None
        self.ik_recovery_braking = False
        self.eef_goal_pose_reset_pending = False
        for side in self._pose_trajectories:
            self._ik_warm_starts.pop(side, None)
        self._pose_trajectories.clear()
        self._pose_reference_at.clear()
        if self.native_loop is not None:
            self.native_loop.hold(reason)
        if self.servo_loop is not None:
            if state == "ready" or (state == "blocked" and self.ik_recovery_fault):
                self.servo_loop.hold()
            else:
                self.servo_loop.pause()
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

    @staticmethod
    def _piper_can_joint_mode_ready(status: dict[str, Any]) -> bool:
        return (
            status.get("ctrl_mode") == PIPER_CTRL_MODE_CAN
            and status.get("mode_feed") == PIPER_MOVE_MODE_J
            and status.get("arm_status") == 0
            and status.get("err_code") == 0
            and status.get("feedback_fresh") is not False
        )

    def _send_enable_hold(
        self,
        side: str,
        piper: Any,
        hold_joints: np.ndarray,
        hold_gripper_m: float,
    ) -> None:
        """Refresh CAN joint mode and the measured enable-time hold target."""
        if not self.close_servo():
            raise ExecutionBlocked("servo did not settle before enable-time actuator handoff")
        raw_joints = np.rint(hold_joints * RAD_FACTOR).astype(np.int64)
        piper.ModeCtrl(
            PIPER_CTRL_MODE_CAN,
            PIPER_MOVE_MODE_J,
            int(getattr(self.args, "speed_pct", DEFAULT_INFERENCE_SPEED_PCT)),
            0x00,
        )
        piper.JointCtrl(*map(int, raw_joints))
        piper.GripperCtrl(
            round(float(hold_gripper_m) * GRIPPER_FACTOR),
            int(getattr(self.args, "gripper_effort", 1000)),
            0x01,
            0,
        )

    def _refresh_known_enable_holds(self, *, exclude_side: str | None = None) -> None:
        """Keep an already-enabled sibling arm held during a second arm handshake."""
        for known_side, hold_joints in self.arm_hold_targets.items():
            if known_side == exclude_side:
                continue
            piper = self.pipers.get(known_side)
            hold_gripper_m = self.arm_hold_gripper_targets.get(known_side)
            if piper is None or hold_gripper_m is None:
                continue
            self._send_enable_hold(
                known_side, piper, hold_joints, hold_gripper_m
            )

    def _enable_robot(self, side: str, piper: Any, hold_qpos: np.ndarray) -> None:
        """Enable Piper only after CAN joint mode accepts a repeated pose hold.

        ``EnablePiper()`` reports the enable state sampled before its own enable
        frame is sent.  A single subsequent ``JointCtrl`` can therefore arrive
        while feedback still says STANDBY and be ignored exactly when the motor
        brake is released.  Repeat enable, mode, and measured-pose hold commands
        until feedback confirms CAN/MOVE_J for consecutive cycles.
        """
        hold_qpos = np.asarray(hold_qpos, dtype=np.float64)
        if hold_qpos.shape != (7,) or not np.all(np.isfinite(hold_qpos)):
            raise ExecutionBlocked(f"{side} Piper hold qpos is not finite 7D")
        limits = effective_limits(piper, JOINT_LIMITS_RAD)
        lower = limits[:, 0]
        upper = limits[:, 1]
        excess = np.maximum(lower - hold_qpos[:6], hold_qpos[:6] - upper)
        if float(np.max(excess)) > IK_FEEDBACK_LIMIT_TOLERANCE_RAD:
            raise ExecutionBlocked(
                f"{side} Piper measured joints are too far outside limits to hold safely"
            )
        # Hold the measured pose exactly.  Clipping a calibrated zero-offset
        # feedback value to the SDK's nominal limits would itself create an
        # unsolicited enable-time motion.
        hold_joints = hold_qpos[:6].copy()
        hold_gripper_m = float(np.clip(hold_qpos[6], 0.0, GRIPPER_EXECUTION_MAX_M))
        enable_timeout_s = float(getattr(self.args, "enable_timeout_s", 3.0))
        deadline = time.monotonic() + enable_timeout_s
        ready_cycles = 0
        last_status: dict[str, Any] | None = None
        last_status_timestamp: float | None = None
        last_driver_timestamp: float | None = None
        last_driver_status: dict[str, Any] | None = None
        # SDK getters expose cached frames. A MOVE_P rejection received before
        # this request is not an acknowledgement of the new MOVE_J hold.
        # Keep one barrier for the whole handshake, never advance it on retry.
        requested_at = time.time()
        logged_previous_mode_rejection = False
        while time.monotonic() < deadline:
            enabled_feedback = bool(piper.EnablePiper())
            self._send_enable_hold(side, piper, hold_joints, hold_gripper_m)
            self._refresh_known_enable_holds(exclude_side=side)
            last_status = arm_status_dict(piper)
            driver_status = driver_enable_status_dict(piper)
            last_driver_status = driver_status
            self.robot_driver_enable_status[side] = driver_status
            status_timestamp = last_status.get("feedback_timestamp")
            post_request_status = (
                status_timestamp is None  # Compatibility with timestamp-less offline fakes.
                or float(status_timestamp) > requested_at
            )
            previous_mode_rejection = (
                not post_request_status
                and last_status.get("feedback_fresh") is True
                and last_status.get("ctrl_mode") == PIPER_CTRL_MODE_CAN
                and last_status.get("mode_feed") == 0  # MOVE_P
                and last_status.get("arm_status") == 4  # Target joint limit.
                and last_status.get("err_code") == 0
                and np.all(hold_joints >= lower)
                and np.all(hold_joints <= upper)
            )
            if (
                last_status.get("feedback_fresh") is True
                and not previous_mode_rejection
                and (
                    last_status["arm_status"]
                    not in {
                        PIPER_ARM_STATUS_NORMAL,
                        PIPER_ARM_STATUS_JOINT_BRAKE_NOT_RELEASED,
                    }
                    or last_status["err_code"] != 0
                )
            ) or driver_status["faulted"] is True:
                raise ExecutionBlocked(
                    f"{side} Piper became unhealthy during enable: "
                    f"arm_status={last_status} driver_status={driver_status} "
                    f"requested_at={requested_at:.6f} post_request_status={post_request_status}"
                )
            if previous_mode_rejection and not logged_previous_mode_rejection:
                logging.info(
                    "%s waiting for post-request MOVE_J feedback; cached MOVE_P limit rejection "
                    "at %.6f predates request %.6f", side, status_timestamp, requested_at,
                )
                logged_previous_mode_rejection = True
            status_advanced = (
                status_timestamp is None
                or last_status_timestamp is None
                or float(status_timestamp) > float(last_status_timestamp) + 1e-9
            )
            driver_timestamp = driver_status.get("feedback_timestamp")
            driver_advanced = (
                driver_timestamp is None
                or last_driver_timestamp is None
                or float(driver_timestamp) > float(last_driver_timestamp) + 1e-9
            )
            mode_ready = self._piper_can_joint_mode_ready(last_status)
            drivers_ready = (
                enabled_feedback
                if driver_status["ready"] is None
                else bool(driver_status["ready"])
            )
            if (
                drivers_ready
                and mode_ready
                and post_request_status
                and status_advanced
                and driver_advanced
            ):
                ready_cycles += 1
                if ready_cycles >= PIPER_ENABLE_CONFIRM_CYCLES:
                    confirmed_at = time.monotonic()
                    self.arm_hold_targets[side] = hold_joints.copy()
                    self.arm_hold_gripper_targets[side] = hold_gripper_m
                    self.arm_hold_started_at[side] = confirmed_at
                    self.arm_hold_stable_since.pop(side, None)
                    self.robot_enabled.add(side)
                    return
            elif not mode_ready or not drivers_ready:
                ready_cycles = 0
            if status_timestamp is not None:
                last_status_timestamp = float(status_timestamp)
            if driver_timestamp is not None:
                last_driver_timestamp = float(driver_timestamp)
            time.sleep(PIPER_ENABLE_RETRY_S)
        raise ExecutionBlocked(
            f"{side} Piper enable timed out after {enable_timeout_s:.1f}s; "
            f"last_status={last_status} last_driver_status={last_driver_status}"
        )

    def _maintain_post_enable_hold(
        self,
        sides: tuple[str, ...],
        qpos_m: np.ndarray,
        statuses: dict[str, dict[str, Any]],
        *,
        now_monotonic: float,
        driver_statuses: Mapping[str, dict[str, Any]] | None = None,
    ) -> bool:
        """Refresh enable holds before timed-plan ageing and wait for fresh data."""
        hold_sides = [side for side in sides if side in self.arm_hold_targets]
        if not hold_sides:
            return False
        if self.enable_staged_generation is not None:
            return False

        waiting_for_hold: list[str] = []
        for index, side in enumerate(sides):
            if side not in self.arm_hold_targets:
                continue
            status = statuses[side]
            driver_status = (
                dict(driver_statuses[side])
                if driver_statuses is not None and side in driver_statuses
                else driver_enable_status_dict(self.pipers[side])
            )
            self.robot_driver_enable_status[side] = driver_status
            if (
                status["arm_status"] != 0
                or status["err_code"] != 0
                or status.get("feedback_fresh") is False
                or driver_status["ready"] is False
            ):
                raise ExecutionBlocked(
                    f"{side} Piper became unhealthy during enable hold: "
                    f"arm_status={status} driver_status={driver_status}"
                )
            hold_target = self.arm_hold_targets[side]
            hold_gripper_m = self.arm_hold_gripper_targets[side]
            self._send_enable_hold(
                side, self.pipers[side], hold_target, hold_gripper_m
            )
            qpos_slice = np.asarray(qpos_m, dtype=np.float64)[
                index * 7 : (index + 1) * 7
            ]
            hold_age = now_monotonic - self.arm_hold_started_at[side]
            hold_error = float(np.max(np.abs(qpos_slice[:6] - hold_target)))
            mode_ready = self._piper_can_joint_mode_ready(status)
            within_tolerance = hold_error <= float(
                getattr(
                    self.args,
                    "arm_hold_tolerance_rad",
                    DEFAULT_ARM_HOLD_TOLERANCE_RAD,
                )
            )
            if mode_ready and within_tolerance and driver_status["ready"] is not False:
                self.arm_hold_stable_since.setdefault(side, now_monotonic)
            else:
                self.arm_hold_stable_since.pop(side, None)
            stable_since = self.arm_hold_stable_since.get(side)
            stable_age = (
                now_monotonic - stable_since if stable_since is not None else 0.0
            )
            if (
                not mode_ready
                or not within_tolerance
                or stable_since is None
                or stable_age < float(getattr(self.args, "arm_settle_s", 0.0))
            ):
                waiting_for_hold.append(
                    f"{side}: ctrl={status['ctrl_mode']} mode={status['mode_feed']} "
                    f"age={hold_age:.2f}s stable={stable_age:.2f}s "
                    f"joint_error={hold_error:.4f}rad"
                )

        if waiting_for_hold:
            self.state = "armed"
            self.blocked_reason = (
                "waiting for enable hold to settle: " + "; ".join(waiting_for_hold)
            )
            return True

        if self.fresh_inference_required_after_monotonic is None:
            self.fresh_inference_required_after_monotonic = now_monotonic
            self.enable_hold_settled_at = time.time()
            # Any result captured before this barrier describes the robot before
            # its controller was confirmed stable.  Keep holding and require a
            # post-settle observation instead of activating that old timeline.
            self.discard_pending_actions(
                "post-enable hold settled; discarded pre-settle trajectory",
                kind="other",
            )
        self.waiting_fresh_after_enable = True
        self.state = "armed"
        self.blocked_reason = (
            "Piper enable hold is stable; waiting for inference captured after settle"
        )
        return True

    def _cancel_staged_enable_plan(self, reason: str, *, kind: str = "other") -> bool:
        if self.enable_staged_generation is None or not self.arm_hold_targets:
            return False
        self.enable_staged_generation = None
        self.waiting_fresh_after_enable = True
        self.fresh_inference_required_after_monotonic = time.monotonic()
        self.enable_hold_settled_at = time.time()
        self.discard_pending_actions(reason, kind=kind)
        self.state = "armed"
        self.blocked_reason = f"{reason}; continuing measured-pose hold"
        return True

    def _commit_staged_enable_plan(self, generation: int) -> None:
        if self.enable_staged_generation != int(generation):
            return
        self.enable_staged_generation = None
        self.waiting_fresh_after_enable = False
        self.arm_hold_targets.clear()
        self.arm_hold_gripper_targets.clear()
        self.arm_hold_started_at.clear()
        self.arm_hold_stable_since.clear()

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
        now_wall: float | None = None,
    ) -> dict[str, Any] | None:
        if target is None:
            return None
        now_monotonic = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        now_wall = time.time() if now_wall is None else float(now_wall)
        target_age_s = now_monotonic - float(target.target_monotonic)
        return {
            "target_monotonic": float(target.target_monotonic),
            "target_at": now_wall - target_age_s,
            "target_age_s": target_age_s,
            "target_time_error_s": target_age_s,
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
        # Target timestamps are monotonic within one decoded chunk.  Keep the
        # lookup logarithmic so a long action horizon does not add a linear
        # scan to every 20 Hz control tick.
        low = 0
        high = len(targets)
        while low < high:
            middle = (low + high) // 2
            if targets[middle].target_monotonic + 1e-9 >= execution_time:
                high = middle
            else:
                low = middle + 1
        return low if low < len(targets) else None

    def _retime_actions_from(
        self,
        actions: list[DecodedQueuedAction],
        start_monotonic: float,
    ) -> list[DecodedQueuedAction]:
        """Place retained rows on a fresh action timeline.

        If the old plan didn't actually progress during inference, retaining
        observation-relative timestamps would make the control tick discard the
        same prefix that result acceptance intentionally kept.
        """
        period_s = 1.0 / self.policy_action_hz
        return [
            replace(
                action,
                target_monotonic=float(start_monotonic) + (index + 1) * period_s,
            )
            for index, action in enumerate(actions)
        ]

    def _record_queue_drop(self, count: int, reason: str, *, kind: str) -> None:
        if kind not in {"unsafe", "expired", "other", "hardware_fault"}:
            raise ValueError(f"unsupported queue drop kind: {kind!r}")
        count = max(0, int(count))
        self.dropped_action_count += count
        if kind in {"unsafe", "hardware_fault"}:
            self.unsafe_drop_count += count
            self.unsafe_active = True
        elif kind == "expired":
            self.expired_drop_count += count
        else:
            self.other_drop_count += count
        self.last_queue_drop_reason = str(reason)[:500]
        self.last_queue_drop_kind = kind

    @staticmethod
    def _gripper_slot(protocol: PolicyProtocol, arm_index: int) -> int:
        return arm_index * (10 if protocol.schema == "delivery" else 7) + (
            9 if protocol.schema == "delivery" else 6
        )

    @staticmethod
    def _opening_to_wire(opening: float, semantics: str) -> float:
        if semantics == GRIPPER_OPENING_METRES:
            return _gripper_value_from_opening_fraction(opening, semantics=semantics)
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
        if not self.gripper_endpoint_filter:
            return desired
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
            ceiling = GRIPPER_EXECUTION_MAX_M / GRIPPER_MAX_M if protocol.gripper_semantics == GRIPPER_OPENING_METRES else 1.0
            current = float(np.clip(qpos_slice[6] / GRIPPER_MAX_M, 0.0, ceiling))
            previous = self._filtered_gripper_opening.get(side, current)
            # Metre targets are physical positions, not normalized endpoints.
            confirmed = desired if protocol.gripper_semantics == GRIPPER_OPENING_METRES else self._confirmed_gripper_desired(side, desired, previous)
            alpha = self.gripper_lowpass_alpha if self.gripper_lowpass else 1.0
            filtered = previous + alpha * (confirmed - previous)
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
            filtered = float(np.clip(filtered, 0.0, ceiling))
            self._filtered_gripper_opening[side] = filtered
            output[slot] = self._opening_to_wire(filtered, protocol.gripper_semantics)
        return replace(queued, absolute_target=output)

    def _delivery_ik_kwargs(self, side):
        return {
            "minimum_motion": bool(getattr(self.args, "ik_minimum_motion", True)),
            "reference_joint_radius_rad": float(getattr(self.args, "ik_reference_joint_radius_rad",
                                                        DEFAULT_IK_REFERENCE_JOINT_RADIUS_RAD)),
            "max_joint_step_rad": float(
                getattr(self.args, "ik_max_joint_step_rad", DEFAULT_IK_MAX_JOINT_STEP_RAD)
                if self.ik_rate_limit and self.servo_backend != "ruckig"
                else getattr(
                    self.args, "ik_search_joint_radius_rad", DEFAULT_IK_SEARCH_JOINT_RADIUS_RAD
                )
            ),
            "search_joint_radius_rad": float(
                getattr(
                    self.args,
                    "ik_search_joint_radius_rad",
                    DEFAULT_IK_SEARCH_JOINT_RADIUS_RAD,
                )
            ),
            "joint_regularization_weight": float(
                getattr(
                    self.args,
                    "ik_joint_regularization_weight",
                    DEFAULT_IK_JOINT_REGULARIZATION_WEIGHT,
                )
            ),
            "position_tolerance_m": float(
                getattr(
                    self.args,
                    "ik_position_tolerance_m",
                    DEFAULT_IK_POSITION_TOLERANCE_M,
                )
            ),
            "rotation_tolerance_rad": float(
                getattr(
                    self.args,
                    "ik_rotation_tolerance_rad",
                    DEFAULT_IK_ROTATION_TOLERANCE_RAD,
                )
            ),
            "max_nfev": int(getattr(self.args, "ik_max_nfev", DEFAULT_IK_MAX_NFEV)),
            "joint_limits": effective_limits(
                self.pipers[self.arm_side if side is None else side], JOINT_LIMITS_RAD
            ),
        }

    def _solve_delivery_ik(
        self,
        current_joints_rad: np.ndarray,
        target_xyz_m: np.ndarray,
        target_rpy_deg: np.ndarray,
        *,
        side: str | None = None,
        warm_start_joints_rad: np.ndarray | None = None,
    ) -> PiperIKSolveResult:
        """Run IK with diagnostics while preserving compatibility with test/custom solvers."""
        if self.ik_solver is None:
            self.ik_solver = PiperContinuousIK()
        kwargs = self._delivery_ik_kwargs(side)
        if not isinstance(self.ik_solver, PiperContinuousIK):
            kwargs.pop("minimum_motion", None)
        solve_with_diagnostics = getattr(self.ik_solver, "solve_with_diagnostics", None)
        if callable(solve_with_diagnostics):
            solver_kwargs = dict(kwargs)
            # Built-in numerical IK can safely warm-start from the previous
            # solution.  Keep custom/test solvers on their legacy signature.
            if (
                warm_start_joints_rad is not None
                and isinstance(self.ik_solver, PiperContinuousIK)
            ):
                solver_kwargs["initial_joints_rad"] = warm_start_joints_rad
            result = solve_with_diagnostics(
                current_joints_rad,
                target_xyz_m,
                target_rpy_deg,
                **solver_kwargs,
            )
            if not isinstance(result, PiperIKSolveResult):
                raise ExecutionBlocked(
                    "continuous IK diagnostics solver returned an invalid result"
                )
            return result

        command = np.asarray(
            self.ik_solver.solve(
                current_joints_rad,
                target_xyz_m,
                target_rpy_deg,
                **kwargs,
            ),
            dtype=np.float64,
        )
        if command.shape != (6,) or not np.all(np.isfinite(command)):
            raise ExecutionBlocked(
                f"continuous IK command must be finite 6D, got {command.shape}"
            )
        # Custom/test solvers using the legacy interface cannot expose the
        # unconstrained solution separately; retain a valid, explicitly unknown
        # diagnostic record instead of inventing a different joint target.
        return PiperIKSolveResult(
            solution_joints_rad=command.copy(),
            command_joints_rad=command.copy(),
            rate_limited=None,
        )

    def _solve_ik_batch(self, requests, *, timeout_s):
        """Same atomic geometric checks with or without worker processes."""
        from bimanual_vla.deployment.ik_workers import IKWorkerError, solve_ik_request
        # Preserve exact attempted geometry, including the failed fraction.
        # Timing-based reconstruction of a discarded queue row is ambiguous.
        self.last_ik_requests = {
            side: dict(joints_rad=np.asarray(args[0]).tolist(),
                       xyz_m=np.asarray(args[1]).tolist(),
                       rpy_deg=np.asarray(args[2]).tolist(),
                       fraction=kwargs.get("recovery_fraction", 1.),
                       hold_position=kwargs.get("hold_position", False),
                       progress_goal=([np.asarray(v).tolist() for v in kwargs["progress_goal"]]
                                      if kwargs.get("progress_goal") is not None else None))
            for side, (args, kwargs) in requests.items()
        }
        if self.ik_workers is not None:
            return self.ik_workers.solve(requests, timeout_s=timeout_s)
        if self.ik_solver is None:
            self.ik_solver = PiperContinuousIK()
        deadline = time.monotonic() + timeout_s
        results = {}
        for side, (args, kwargs) in requests.items():
            results[side] = solve_ik_request(self.ik_solver, args, kwargs)
            if time.monotonic() > deadline:
                raise IKWorkerError("in-process IK deadline exceeded")
        return results

    def _eef_arrival_errors(self, measured, goal, sides):
        errors = {}
        for index, side in enumerate(sides):
            current = np.asarray(measured[index*10:index*10+10], dtype=float)
            desired = np.asarray(goal[index*10:index*10+10], dtype=float)
            position = float(np.linalg.norm(current[:3] - desired[:3]))
            rotation = float(Rotation.from_matrix(
                rotation_from_state(desired.copy()) @ rotation_from_state(current.copy()).T).magnitude())
            errors[side] = dict(position_m=position, rotation_rad=rotation,
                arrived=(position <= self.eef_goal_position_tolerance_m
                         and rotation <= self.eef_goal_rotation_tolerance_rad))
        return errors

    def _eef_goal_openings(self, action, protocol, sides):
        return np.array([GRIPPER_MAX_M * _opening_fraction(
            action.absolute_target[index*10+9], semantics=protocol.gripper_semantics,
            tolerance=float(getattr(self.args, "gripper_range_tolerance", DEFAULT_GRIPPER_RANGE_TOLERANCE)))
            for index, _ in enumerate(sides)])

    def _check_eef_goal_workspace(self, goal, sides):
        for index, side in enumerate(sides):
            self._check_eef_xyz_workspace(goal[index*10:index*10+3], side)

    def _check_eef_xyz_workspace(self, xyz, side):
        for axis_index, axis in enumerate(("x", "y", "z")):
            bounds = (delivery_workspace_y(self.args, side) if axis == "y"
                      else getattr(self.args, "workspace_" + axis, None))
            if bounds is not None and not bounds[0] <= xyz[axis_index] <= bounds[1]:
                raise ExecutionBlocked(f"{side} EEF goal exceeds workspace {axis}")

    def _pose_reference_settled(self, side, goal):
        shaper = self._pose_trajectories.get(side)
        return bool(shaper is not None
                    and np.linalg.norm(shaper.xyz-goal[:3]) < 1e-5
                    and Rotation.from_matrix(rotation_from_state(goal.copy()) @ shaper.rotation.T).magnitude() < 1e-4
                    and np.linalg.norm(shaper.linear_velocity) < 1e-4
                    and np.linalg.norm(shaper.angular_velocity) < 1e-3)

    def _recovery_needs_braking(self):
        # An existing writer may carry velocity/acceleration or actuator lag.
        # Even zero generated velocity alone is not proof of measured settling.
        return self.servo_loop is not None

    def _advance_ik_recovery(self, qpos, now, *, raw_delivery_state, protocol, feedback_at=None):
        """Reach the retained original goal, then release grippers and reobserve."""
        from bimanual_vla.deployment.ik_workers import IKWorkerError, solve_recovering
        target = self.ik_recovery_target
        goal = self.eef_goal_action
        sides = ("left", "right") if protocol.arm_mode == "bimanual" else (protocol.arm_side,)
        metrics = self.servo_loop.metrics() if self.servo_loop is not None else {}
        if metrics.get("fault"):
            raise RuntimeHardwareFault(f"servo failed during EEF goal approach: {metrics['fault']}")
        if (goal is None or self.servo_loop is None or metrics.get("paused") or metrics.get("stopping")
                or now - self.ik_recovery_started > self.eef_goal_timeout_s):
            self.ik_recovery_fault = (f"original EEF goal did not settle within {self.eef_goal_timeout_s:g}s; "
                                      "grippers held; stop and restart required")
            return self._block("blocked", self.ik_recovery_fault)
        indices = [i for i in range(len(target)) if i % 7 != 6]
        grippers = [i*7+6 for i in range(len(sides))]
        state = metrics["state"]
        feedback_at = float(now if feedback_at is None else feedback_at)
        joint_feedback = np.asarray(qpos)[indices]
        sent_error = float(np.max(np.abs(joint_feedback-np.asarray(state["position"]))))
        if self.ik_recovery_braking:
            previous = self._recovery_settle_feedback
            stamps = tuple(float(self.pipers[side].GetArmJointMsgs().time_stamp) for side in sides)
            fresh = (all(math.isfinite(stamp) and stamp > 0 for stamp in stamps)
                     and (previous is None or (feedback_at > previous[0]
                          and all(a > b for a, b in zip(stamps, previous[2])))))
            if fresh:
                dt = feedback_at-previous[0] if previous is not None else 0.
                stable = (previous is not None and 0 < dt <= max(.2, 3/self.control_hz)
                    and np.max(np.abs(joint_feedback-previous[1]))/dt < .01
                    and np.max(np.abs(state["velocity"])) < 1e-4
                    and np.max(np.abs(state["acceleration"])) < 1e-3 and sent_error < .01)
                self._recovery_stable_since = (
                    (self._recovery_stable_since if self._recovery_stable_since is not None else feedback_at)
                    if stable else None)
                self._recovery_settle_feedback = (feedback_at, joint_feedback.copy(), stamps)
            if not fresh or self._recovery_stable_since is None or feedback_at-self._recovery_stable_since < .1:
                self.state = "recovering"
                self.blocked_reason = "IK recovery braking; waiting for fresh measured settling"
                return False
            # Discard the pre-brake IK candidate. Re-solve the retained original
            # goal from the actual settled pose, with no artificial state reset.
            self.ik_recovery_braking = False
            target = np.asarray(qpos).copy()
            target[grippers] = self.eef_goal_held_grippers
            self.ik_recovery_target = target
            self._ik_warm_starts.clear()
            self._pose_trajectories.clear()
            self._pose_reference_at.clear()
            self.eef_goal_pose_reset_pending = False
            self.eef_goal_step_count = 0
            self._recovery_tracking_cycles = 0
            self.blocked_reason = ""
            logging.info("IK recovery hold settled; recomputing retained goal from measured pose")
        elif self._recovery_tracking_at is None or feedback_at > self._recovery_tracking_at:
            # Recovery submits bypass normal policy command traces. Retain the
            # same fast sent-command guard here instead of waiting for the
            # independent 0.35 rad servo hard fault.
            if (self._recovery_tracking_at is not None
                    and feedback_at-self._recovery_tracking_at > max(.2, 3/self.control_hz)):
                self._recovery_tracking_cycles = 0
            self._recovery_tracking_cycles = (self._recovery_tracking_cycles+1
                if sent_error > self.tracking_lag_threshold_rad else 0)
            if self._recovery_tracking_cycles >= self.tracking_lag_confirm_cycles:
                self.ik_recovery_fault = (f"IK recovery sent-command tracking lag {sent_error:.6f}rad; "
                    "braking with grippers held; stop and restart required")
                return self._block("blocked", self.ik_recovery_fault)
        self._recovery_tracking_at = feedback_at
        settled = (np.max(np.abs(np.asarray(state["position"])-target[indices])) < 1e-5
                   and np.max(np.abs(state["velocity"])) < 1e-4
                   and np.max(np.abs(state["acceleration"])) < 1e-3
                   and np.max(np.abs(np.asarray(qpos)[indices]-target[indices])) < .01)
        self.eef_goal_errors = self._eef_arrival_errors(raw_delivery_state, goal.absolute_target, sides)
        arrived = all(value["arrived"] for value in self.eef_goal_errors.values())
        pose_enabled = bool(getattr(self.args, "pose_trajectory", False))
        pose_settled = {side: self._pose_reference_settled(side, goal.absolute_target[index*10:index*10+10])
                        for index, side in enumerate(sides)} if pose_enabled else {}
        context = dict(generation=goal.generation, source_index=goal.source_index,
                       queue_index=goal.queue_index, eef_goal=True, goal_step=self.eef_goal_step_count)
        try:
            if settled and arrived and (not pose_enabled or all(pose_settled.values())):
                # The original pose, not an intermediate IK point, authorizes
                # gripper filtering. Keep the original pose while the writer
                # ramps the requested opening; contact need not reach zero.
                self.eef_goal_releasing = True
                filtered = self._filter_gripper_target(goal, qpos, protocol)
                target[grippers] = self._eef_goal_openings(filtered, protocol, sides)
                sample = self.servo_loop.submit(target, hold=False, context=context)
                desired = self._eef_goal_openings(goal, protocol, sides)
                self.state = "gripper_release"
                if (np.max(np.abs(target[grippers]-desired)) <= .0005
                        and np.max(np.abs(np.asarray(sample["qpos"])[grippers]-desired)) <= .001):
                    self._eef_committed_grippers.update(zip(sides, desired.tolist()))
                    self.ik_recovery_target = None
                    self.eef_goal_action = None
                    self.ik_recovery_barrier = time.monotonic()
                    self.state = "holding"
                    self.blocked_reason = "original EEF goal reached; waiting for fresh observation"
                    logging.info("Original EEF goal reached; gripper command published; fresh observation required")
                return False

            # Always hold the captured opening while approaching. Do not let
            # discarded policy rows or a fresh model generation release it.
            if self.eef_goal_releasing:
                raise ExecutionBlocked("original pose drifted during gripper release")
            target[grippers] = self.eef_goal_held_grippers
            self.servo_loop.submit(target, hold=False, context=context)
            # A persistent smooth reference must advance every control tick,
            # not restart from rest at every intermediate joint waypoint.
            # A shrunken IK reference breaks that continuity: only re-anchor
            # once its joint writer and measured feedback have settled.
            if not settled and (not pose_enabled or self.eef_goal_pose_reset_pending):
                return False
            if self.eef_goal_pose_reset_pending:
                self._pose_trajectories.clear()
                self._pose_reference_at.clear()
                self.eef_goal_pose_reset_pending = False
            requests = {}
            for index, side in enumerate(sides):
                current = np.asarray(raw_delivery_state[index*10:index*10+10], dtype=float)
                original = np.asarray(goal.absolute_target[index*10:index*10+10], dtype=float)
                local, _ = project_delivery_target_step(current, original,
                    max_translation_step_m=float(getattr(self.args, "max_translation_step_m", DEFAULT_MAX_TRANSLATION_STEP_M)),
                    max_rotation_step_rad=float(getattr(self.args, "max_rotation_step_rad", DEFAULT_MAX_ROTATION_STEP_RAD)))
                self._check_eef_xyz_workspace(local[:3], side)
                kwargs = self._delivery_ik_kwargs(side)
                kwargs["initial_joints_rad"] = self._ik_warm_starts.get(side)
                # Intermediate recovery references must continue advancing,
                # even when they are inside the normal streaming pose tube.
                kwargs["minimum_motion"] = False
                if settled:
                    # This is a stopped, feedback-confirmed recovery leg, not
                    # the next 30 Hz waypoint. It may use the measured search
                    # radius; the servo still enforces dynamics and tracking.
                    kwargs["reference_joint_radius_rad"] = kwargs["search_joint_radius_rad"]
                target_xyz = local[:3]
                target_rpy = Rotation.from_matrix(rotation_from_state(local.copy())).as_euler("xyz", degrees=True)
                if pose_enabled:
                    target_xyz, target_rpy, self.eef_goal_pose_metrics[side] = self._shape_delivery_pose(
                        side, current, target_xyz, target_rpy)
                else:
                    kwargs["progress_goal"] = (original[:3], Rotation.from_matrix(
                        rotation_from_state(original.copy())).as_euler("xyz", degrees=True))
                # An already arrived arm must not jiggle just to satisfy a
                # relative-progress test for the other arm's next local step.
                joints = np.asarray(qpos[index*7:index*7+6])
                limits = kwargs["joint_limits"]
                kwargs["hold_position"] = bool(self.eef_goal_errors[side]["arrived"]
                    and (not pose_enabled or pose_settled.get(side, False))
                    and np.all(joints >= limits[:, 0]) and np.all(joints <= limits[:, 1]))
                requests[side] = ((joints, target_xyz, target_rpy), kwargs)
            results = solve_recovering(self._solve_ik_batch, requests)
            if time.monotonic() - self.ik_recovery_started > self.eef_goal_timeout_s:
                raise IKWorkerError("original EEF goal deadline exceeded during IK")
            next_target = target.copy()
            for index, side in enumerate(sides):
                if "recovery" in results[side]:
                    self._check_eef_xyz_workspace(results[side]["recovery"]["xyz_m"], side)
                next_target[index*7:index*7+6] = results[side]["solution_joints_rad"]
            self.servo_loop.submit(next_target, hold=False, context=context)
            self.ik_recovery_target = next_target
            self._ik_warm_starts.update({side: results[side]["solution_joints_rad"].copy() for side in sides})
            if pose_enabled and any("recovery" in result for result in results.values()):
                self.eef_goal_pose_reset_pending = True
            self.eef_goal_step_count += 1
            self.last_ik_recovery = {s: r.get("goal_progress", r.get("recovery")) for s, r in results.items()}
            self.state = "recovering"
            return False
        except (IKWorkerError, ExecutionBlocked, EOFError, OSError) as exc:
            self.ik_recovery_fault = f"original EEF goal approach failed: {exc}; stop and restart required"
            return self._block("blocked", self.ik_recovery_fault)
        except ServoError as exc:
            raise RuntimeHardwareFault(f"servo failed during EEF goal approach: {exc}") from exc

    def _complete_pending_command_feedback(
        self,
        raw_delivery_state: np.ndarray,
        qpos_m: np.ndarray,
        protocol: PolicyProtocol,
        *,
        feedback_at: float,
    ) -> None:
        """Attach the next control-cycle feedback to the preceding command trace."""
        pending = self._pending_feedback_command
        if pending is None:
            return
        self._pending_feedback_command = None
        current_sides = (
            ("left", "right") if protocol.arm_mode == "bimanual" else (protocol.arm_side,)
        )
        pending_sides = pending.get("sides")
        if not isinstance(pending_sides, dict) or set(pending_sides) != set(current_sides):
            self.last_command_feedback = {
                **pending,
                "feedback_status": "unavailable",
                "feedback_error": "command/feedback arm layout changed before the next cycle",
                "feedback_at": float(feedback_at),
            }
            return

        qpos = np.asarray(qpos_m, dtype=np.float64)
        delivery = np.asarray(raw_delivery_state, dtype=np.float64)
        expected_qpos_dim = 7 * len(current_sides)
        expected_delivery_dim = 10 * len(current_sides)
        qpos_valid = qpos.shape == (expected_qpos_dim,) and np.all(np.isfinite(qpos))
        delivery_valid = (
            protocol.schema == "joint"
            or (
                delivery.shape == (expected_delivery_dim,)
                and np.all(np.isfinite(delivery))
            )
        )
        if not qpos_valid or not delivery_valid:
            requirement = f"finite {expected_qpos_dim}D qpos"
            if protocol.schema != "joint":
                requirement += f" and {expected_delivery_dim}D delivery state"
            self.last_command_feedback = {
                **pending,
                "feedback_status": "unavailable",
                "feedback_error": f"next-cycle feedback must be {requirement}",
                "feedback_at": float(feedback_at),
            }
            return

        completed_sides: dict[str, Any] = {}
        max_joint_errors: list[float] = []
        reference_joint_errors: list[float] = []
        policy_position_errors: list[float] = []
        policy_rotation_errors: list[float] = []
        gripper_errors: list[float] = []
        eef_translation_errors: list[float] = []
        eef_rotation_errors: list[float] = []
        run_eef_diagnostics = (
            self.command_sequence <= 1
            or self.control_tick_count % max(1, self.feedback_diagnostics_stride) == 0
        )
        for index, side in enumerate(current_sides):
            issued = pending_sides[side]
            issued = issued if isinstance(issued, dict) else {}
            measured_qpos = qpos[index * 7 : (index + 1) * 7]
            measured_delivery = (
                delivery[index * 10 : (index + 1) * 10]
                if protocol.schema != "joint"
                else None
            )
            commanded_joints = np.asarray(
                issued.get("commanded_joints_rad"), dtype=np.float64
            )
            commanded_gripper_m = self._finite_timing_value(
                issued.get("commanded_gripper_m")
            )
            # Keep the pre-planner joint difference for diagnosis only. IK
            # branches/near-singular motion can differ in joint space while
            # the actual EEF still tracks; measure policy lag in task space.
            reference_error = None
            policy_position_error = policy_rotation_error = None
            if protocol.schema == "delivery":
                reference = np.asarray(issued.get("full_ik_solution_joints_rad"), dtype=float)
                if reference.shape == (6,) and np.isfinite(reference).all():
                    reference_error = float(np.max(np.abs(reference - measured_qpos[:6])))
                    reference_joint_errors.append(reference_error)
                policy_target = np.asarray(issued.get("policy_eef_target"), dtype=float)
                if policy_target.shape == (10,) and np.isfinite(policy_target).all():
                    policy_position_error = float(np.linalg.norm(policy_target[:3] - measured_delivery[:3]))
                    policy_rotation_error = float(Rotation.from_matrix(
                        rotation_from_state(policy_target.copy()) @
                        rotation_from_state(measured_delivery.copy()).T).magnitude())
                    policy_position_errors.append(policy_position_error)
                    policy_rotation_errors.append(policy_rotation_error)
            if commanded_joints.shape == (6,) and np.all(np.isfinite(commanded_joints)):
                joint_error = measured_qpos[:6] - commanded_joints
                joint_abs_error = np.abs(joint_error)
                max_joint_error = float(np.max(joint_abs_error))
                rms_joint_error = float(np.sqrt(np.mean(joint_error**2)))
                max_joint_errors.append(max_joint_error)
                joint_error_values: list[float] | None = joint_error.tolist()
                joint_abs_error_values: list[float] | None = joint_abs_error.tolist()
            else:
                max_joint_error = None
                rms_joint_error = None
                joint_error_values = None
                joint_abs_error_values = None

            if commanded_gripper_m is not None:
                gripper_error_m = float(measured_qpos[6] - commanded_gripper_m)
                gripper_errors.append(abs(gripper_error_m))
            else:
                gripper_error_m = None

            eef_translation_error_m = None
            eef_rotation_error_rad = None
            measured_eef_rpy_deg = None
            measured_rotation = None
            if measured_delivery is not None and run_eef_diagnostics:
                try:
                    measured_rotation = rotation_from_state(measured_delivery.copy())
                    measured_eef_rpy_deg = Rotation.from_matrix(measured_rotation).as_euler(
                        "xyz", degrees=True
                    ).tolist()
                except (ExecutionBlocked, ValueError, FloatingPointError):
                    measured_rotation = None
                pre_ik = issued.get("pre_ik_eef_target")
                if isinstance(pre_ik, dict):
                    absolute_target = np.asarray(
                        pre_ik.get("absolute_target"), dtype=np.float64
                    )
                    if absolute_target.shape == (10,) and np.all(np.isfinite(absolute_target)):
                        eef_translation_error_m = float(
                            np.linalg.norm(measured_delivery[:3] - absolute_target[:3])
                        )
                        try:
                            if measured_rotation is not None:
                                eef_rotation_error_rad = float(
                                    Rotation.from_matrix(
                                        rotation_from_state(absolute_target.copy())
                                        @ measured_rotation.T
                                    ).magnitude()
                                )
                        except (ExecutionBlocked, ValueError, FloatingPointError):
                            eef_rotation_error_rad = None
                        eef_translation_errors.append(eef_translation_error_m)
                        if eef_rotation_error_rad is not None:
                            eef_rotation_errors.append(eef_rotation_error_rad)

            completed_sides[side] = {
                **issued,
                "next_cycle_feedback": {
                    "feedback_at": float(feedback_at),
                    "joints_rad": measured_qpos[:6].tolist(),
                    "gripper_opening_m": float(measured_qpos[6]),
                    "eef_state": (
                        measured_delivery.tolist() if measured_delivery is not None else None
                    ),
                    "eef_xyz_m": (
                        measured_delivery[:3].tolist()
                        if measured_delivery is not None
                        else None
                    ),
                    "eef_rotation6d": (
                        measured_delivery[3:9].tolist()
                        if measured_delivery is not None
                        else None
                    ),
                    "eef_rpy_deg": measured_eef_rpy_deg,
                    "gripper_opening_fraction": float(
                        np.clip(measured_qpos[6] / GRIPPER_MAX_M, 0.0, 1.0)
                    ),
                },
                "command_feedback_error": {
                    "reference_joint_abs_error_rad": reference_error,
                    "policy_position_error_m": policy_position_error,
                    "policy_rotation_error_rad": policy_rotation_error,
                    "joint_error_definition": "feedback_minus_command",
                    "joint_error_rad": joint_error_values,
                    "joint_abs_error_rad": joint_abs_error_values,
                    "max_joint_abs_error_rad": max_joint_error,
                    "rms_joint_error_rad": rms_joint_error,
                    "gripper_error_definition": "feedback_minus_command",
                    "gripper_error_m": gripper_error_m,
                    "eef_translation_error_m": eef_translation_error_m,
                    "eef_rotation_error_rad": eef_rotation_error_rad,
                },
            }

        command_at = self._finite_timing_value(pending.get("command_at"))
        max_joint_error = max(max_joint_errors) if max_joint_errors else None
        reference_joint_error = max(reference_joint_errors, default=0.)
        policy_position_error = max(policy_position_errors, default=0.)
        policy_rotation_error = max(policy_rotation_errors, default=0.)
        guard_joint_error = max_joint_error or 0.
        self.last_command_feedback = {
            **pending,
            "feedback_status": "complete",
            "feedback_cycle_offset": 1,
            "feedback_at": float(feedback_at),
            "command_to_feedback_ms": (
                max(0.0, (float(feedback_at) - command_at) * 1000.0)
                if command_at is not None
                else None
            ),
            "max_joint_abs_error_rad": max_joint_error,
            "tracking_guard_joint_error_rad": guard_joint_error,
            "reference_joint_error_rad": reference_joint_error,
            "policy_position_error_m": policy_position_error,
            "policy_rotation_error_rad": policy_rotation_error,
            "max_gripper_abs_error_m": max(gripper_errors) if gripper_errors else None,
            "max_eef_translation_error_m": (
                max(eef_translation_errors) if eef_translation_errors else None
            ),
            "max_eef_rotation_error_rad": (
                max(eef_rotation_errors) if eef_rotation_errors else None
            ),
            "sides": completed_sides,
        }

        # Sent commands and policy EEF references have different timing.
        # Keep the fast actuator guard, but allow bounded planner acceleration
        # before calling an actual EEF mismatch sustained. Never re-time
        # the RTC suffix or reset this evidence merely on a chunk boundary.
        if bool(pending.get("hold")) and not self.tracking_lag_active:
            self.tracking_lag_consecutive_cycles = 0
            self.tracking_reference_lag_since = None
            self.tracking_reference_lag_cycles = 0
            self.tracking_lag_peak_error_rad = 0.
        if (not bool(pending.get("hold")) and (max_joint_error is not None or policy_position_errors or policy_rotation_errors)
                and (self.tracking_lag_last_feedback_at is None
                     or feedback_at > self.tracking_lag_last_feedback_at)):
            if (self.tracking_lag_last_feedback_at is not None
                    and feedback_at - self.tracking_lag_last_feedback_at > max(.2, 3 / self.control_hz)
                    and not self.tracking_lag_active):
                self.tracking_lag_consecutive_cycles = 0
                self.tracking_reference_lag_since = None
                self.tracking_reference_lag_cycles = 0
            self.tracking_lag_last_feedback_at = feedback_at
            if (max_joint_error or 0.) > self.tracking_lag_threshold_rad:
                self.tracking_lag_consecutive_cycles += 1
            elif not self.tracking_lag_active:
                self.tracking_lag_consecutive_cycles = 0
            reference_lag = (policy_position_error > self.tracking_lag_position_threshold_m
                             or policy_rotation_error > self.tracking_lag_rotation_threshold_rad)
            if reference_lag:
                if self.tracking_reference_lag_since is None:
                    self.tracking_reference_lag_since = float(feedback_at)
                self.tracking_reference_lag_cycles += 1
            elif not self.tracking_lag_active:
                self.tracking_reference_lag_since = None
                self.tracking_reference_lag_cycles = 0
            reference_duration = (float(feedback_at) - self.tracking_reference_lag_since
                                  if self.tracking_reference_lag_since is not None else 0.)
            sent_trigger = self.tracking_lag_consecutive_cycles >= self.tracking_lag_confirm_cycles
            reference_trigger = (self.tracking_reference_lag_cycles >= self.tracking_lag_confirm_cycles
                                 and reference_duration >= self.tracking_reference_lag_duration_s)
            if reference_lag or self.tracking_lag_consecutive_cycles:
                self.tracking_lag_peak_error_rad = max(self.tracking_lag_peak_error_rad, guard_joint_error)
            elif not self.tracking_lag_active:
                self.tracking_lag_peak_error_rad = 0.0
            self.last_command_feedback["tracking_lag_evidence"] = dict(
                sent_cycles=self.tracking_lag_consecutive_cycles,
                reference_cycles=self.tracking_reference_lag_cycles,
                reference_duration_s=reference_duration,
                source="sent_joint" if sent_trigger else "policy_reference" if reference_trigger else None,
            )

            if (
                not self.tracking_lag_active
                and (sent_trigger or reference_trigger)
            ):
                self.tracking_lag_active = True
                self.tracking_lag_started_at = float(feedback_at)
                self.tracking_lag_required_after_monotonic = time.monotonic()
                if self.servo_loop is not None:
                    self.servo_loop.hold()
                    self.tracking_lag_required_after_monotonic = None
                if self.native_loop is not None:
                    self.native_loop.hold("tracking_lag")
                    self._native_epoch = self.native_loop.metrics()["epoch"]
                    self.tracking_lag_required_after_monotonic = None
                self.tracking_lag_settle_cycles = 0
                self.tracking_lag_settle_joints = None
                self.tracking_lag_settle_feedback_at = None
                self.ik_recovery_target = None
                self.eef_goal_action = None
                self.eef_goal_pose_reset_pending = False
                self.ik_recovery_prefix_reset = True
                self._pose_trajectories.clear()
                self._pose_reference_at.clear()
                self._ik_warm_starts.clear()
                self._filtered_gripper_opening.clear()
                self._gripper_extreme_candidate.clear()
                self._gripper_extreme_count.clear()
                self._gripper_extreme_latch.clear()
                self.tracking_lag_trigger_count += 1
                generation = pending.get("generation")
                self.tracking_lag_trigger_generation = (
                    int(generation) if generation is not None else self.active_generation
                )
                suffix_count = len(self.pending_actions)
                reason = (
                    "tracking lag guard froze timed trajectory after "
                    f"source={'sent_joint' if sent_trigger else 'policy_reference'} "
                    f"sent_cycles={self.tracking_lag_consecutive_cycles} "
                    f"reference_duration={reference_duration:.3f}s; "
                    f"sent_joint={max_joint_error or 0.:.6f}rad "
                    f"reference_joint={reference_joint_error:.6f}rad position={policy_position_error:.6f}m "
                    f"rotation={policy_rotation_error:.6f}rad; waiting for stable hold"
                )
                if suffix_count:
                    self._record_queue_drop(suffix_count, reason, kind="other")
                    self.pending_actions.clear()
                else:
                    self.last_queue_drop_reason = reason
                    self.last_queue_drop_kind = "other"
                self.queued_action_index = None
                self.timeline_resync_active = False
                self.hold_active = self.last_safe_target is not None
                if self.hold_active and self.hold_started_at is None:
                    self.hold_started_at = float(feedback_at)
                self.state = "holding" if self.hold_active else "blocked"
                self.blocked_reason = reason
                logging.warning(
                    "%s; peak error %.6f rad",
                    reason,
                    self.tracking_lag_peak_error_rad,
                )

    def _update_tracking_lag_settle(self, qpos, now, *, feedback_at):
        """Open the fresh-observation barrier only after the braking hold settles."""
        if self.tracking_lag_required_after_monotonic is not None or self.servo_loop is None:
            return
        previous_feedback_at = self.tracking_lag_settle_feedback_at
        if previous_feedback_at is not None and feedback_at <= previous_feedback_at:
            return
        if (previous_feedback_at is not None
                and feedback_at - previous_feedback_at > max(.2, 3 / self.control_hz)):
            self.tracking_lag_settle_cycles = 0
            self.tracking_lag_settle_joints = None
        self.tracking_lag_settle_feedback_at = feedback_at
        metrics = self.servo_loop.metrics()
        if metrics.get("fault") or metrics.get("paused") or metrics.get("stopping"):
            raise RuntimeHardwareFault("servo unavailable while settling tracking-lag hold")
        state = metrics.get("state", {})
        joints = np.delete(np.asarray(qpos), np.arange(6, len(qpos), 7))
        position = np.asarray(state.get("position"), dtype=float)
        velocity = np.asarray(state.get("velocity"), dtype=float)
        acceleration = np.asarray(state.get("acceleration"), dtype=float)
        stable = (all(value.shape == joints.shape and np.isfinite(value).all()
                      for value in (position, velocity, acceleration))
                  and np.max(np.abs(position-joints)) < .01
                  and np.max(np.abs(velocity)) < 1e-4
                  and np.max(np.abs(acceleration)) < 1e-3
                  and self.tracking_lag_settle_joints is not None
                  and np.max(np.abs(joints-self.tracking_lag_settle_joints)) < .001)
        self.tracking_lag_settle_joints = joints.copy()
        self.tracking_lag_settle_cycles = self.tracking_lag_settle_cycles + 1 if stable else 0
        if self.tracking_lag_settle_cycles >= self.tracking_lag_confirm_cycles:
            self.tracking_lag_required_after_monotonic = float(now)
            self.blocked_reason = "tracking-lag hold settled; waiting for a post-settle observation"
            logging.info(self.blocked_reason)

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
        if self.rtc_enabled and latency_s * self.policy_action_hz > self.rtc_inference_delay_steps:
            logging.warning(
                "RTC delay exceeded: generation=%s actual=%.1f ms (%.2f steps), "
                "configured delay=%s, horizon=%s; selecting by actual elapsed time",
                launch.generation, latency_s * 1000, latency_s * self.policy_action_hz,
                self.rtc_inference_delay_steps, self.rtc_execution_horizon,
            )
        self.record_inference_completion(arrived_monotonic)
        self._record_client_transport_timing(result, generation=launch.generation)
        self.inference_old_remaining = len(self.pending_actions)
        self.inference_skip_steps = 0
        self.inference_elapsed_prefix_steps = 0
        self.inference_blend_steps = 0
        self.inference_progress_steps = max(
            0,
            self.executed_plan_command_count
            - int(getattr(launch, "executed_plan_command_count", 0)),
        )
        self.inference_timeline_resynced = False
        if not math.isfinite(latency_s) or latency_s < 0:
            return self._reject_result(
                launch.generation, f"invalid capture-to-arrival latency {latency_s!r}", arrived_at
            )
        if not isinstance(result, dict) or "actions" not in result:
            return self._reject_result(launch.generation, "result has no actions", arrived_at)

        if self.ik_recovery_fault:
            return self._reject_result(launch.generation, self.ik_recovery_fault, arrived_at)
        if self.native_loop is not None and not self.native_inference_ready():
            return self._reject_result(launch.generation, "native EEF hold/fault invalidated inference; waiting for stable feedback", arrived_at)
        if self.ik_recovery_target is not None:
            return self._reject_result(launch.generation, "IK recovery still moving; waiting for measured arrival", arrived_at)
        if (self.ik_recovery_barrier is not None
                and float(launch.captured_monotonic) <= self.ik_recovery_barrier):
            return self._reject_result(launch.generation,
                "discarded pre-recovery inference; waiting for a post-recovery observation", arrived_at)

        if self.planner_prefix_enabled:
            snapshot = getattr(launch, "observation_snapshot", None)
            request_rtc = snapshot.rtc_metadata if snapshot is not None else {}
            prefix = request_rtc.get("planned_prefix")
            response_rtc = result.get("rtc", {})
            if response_rtc.get("prefix_source") != "client_planner":
                return self._reject_result(launch.generation, "policy did not acknowledge planner RTC prefix", arrived_at)
            if prefix is not None:
                plan_id = prefix["plan_id"]
                if (response_rtc.get("client_plan_id") != plan_id or self.servo_loop is None
                        or not self.servo_loop.prefix_is_valid(plan_id)):
                    return self._reject_result(launch.generation,
                                               "RTC planner prefix invalidated by hold, fault or plan replacement", arrived_at)

        if self.waiting_fresh_after_enable:
            barrier = self.fresh_inference_required_after_monotonic
            if barrier is None:
                return self._reject_result(
                    launch.generation,
                    "discarded inference captured before post-enable hold settled",
                    arrived_at,
                )
            if float(launch.captured_monotonic) < barrier:
                return self._reject_result(
                    launch.generation,
                    "discarded pre-settle inference; waiting for a post-enable observation",
                    arrived_at,
                )

        if self.tracking_lag_active:
            barrier = self.tracking_lag_required_after_monotonic
            if barrier is None:
                return self._reject_result(launch.generation,
                    "tracking-lag hold still settling; waiting for a post-settle observation", arrived_at)
            if float(launch.captured_monotonic) <= barrier:
                return self._reject_result(
                    launch.generation,
                    "discarded inference captured before tracking-lag guard triggered; "
                    "waiting for a post-trigger observation after hold settled",
                    arrived_at,
                )

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
        retime_from_arrival = False
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
            self.inference_elapsed_prefix_steps = future_index
            # RTC keeps the observation time axis; non-RTC recovery may retime.
            # Time passing alone isn't sufficient evidence that a delayed
            # prefix was executed.  In cold start, hold, blocked, or underrun
            # recovery, the robot may not have advanced at all.  Skip no more
            # rows than the old plan actually sent after this observation was
            # captured.  If execution lagged behind wall time, place the
            # retained rows on a fresh timeline so execute_next won't discard
            # them again on the following control tick.
            skip_steps = (future_index if self.rtc_enabled else
                          min(future_index, self.inference_progress_steps))
            retime_from_arrival = skip_steps < future_index
        self.inference_skip_steps = skip_steps
        fresh_actions = decoded[skip_steps:]
        if (
            bool(getattr(self.args, "trajectory_spike_suppression", False))
            and self.servo_backend == "legacy"
            and protocol.schema == "joint"
            and len(fresh_actions) >= 5
        ):
            # Reuse the GUI/client RTC execution horizon. Do not let the remote
            # tail of a 50-step chunk affect commands that will be replaced by
            # the next receding-horizon inference.
            spike_horizon = int(
                getattr(
                    self.args,
                    "rtc_execution_horizon",
                    DEFAULT_RTC_EXECUTION_HORIZON,
                )
            )
            absolute = np.stack(
                [item.absolute_target for item in fresh_actions], axis=0
            ).astype(np.float32, copy=False)
            suppressed = suppress_local_joint_spikes(
                absolute,
                action_hz=self.policy_action_hz,
                horizon_steps=spike_horizon,
            )
            if np.any(np.abs(suppressed - absolute) > 1e-7):
                fresh_actions = [
                    replace(item, absolute_target=suppressed[index].copy())
                    for index, item in enumerate(fresh_actions)
                ]
        if retime_from_arrival:
            fresh_actions = self._retime_actions_from(
                fresh_actions, arrived_monotonic
            )
            self.inference_timeline_resynced = True
            # Pending rows weren't consumed on schedule, so they aren't a valid
            # trajectory to blend against.  The last command that really
            # reached the robot is the only safe transition anchor.
            old_actions = (
                [self.last_safe_target] if self.last_safe_target is not None else []
            )
        else:
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
                    blend_profile=self.blend_profile,
                )
            except ExecutionBlocked as exc:
                return self._reject_result(launch.generation, str(exc), arrived_at)
            self.inference_blend_steps = blend_steps
        else:
            candidate = list(fresh_actions)
        if not candidate:
            return self._reject_result(launch.generation, "result has no executable rows", arrived_at)

        candidate = apply_gripper_open_lookahead(
            candidate,
            protocol,
            lookahead_steps=self.gripper_open_lookahead_steps,
            tolerance=float(
                getattr(
                    self.args,
                    "gripper_range_tolerance",
                    DEFAULT_GRIPPER_RANGE_TOLERANCE,
                )
            ),
        )

        # Synchronous mode intentionally throttles execution to a fixed prefix
        # of the accepted chunk. The gripper lookahead above already saw the
        # full fresh chunk, so truncating here does not weaken that decision.
        # Once this short queue drains, ``execute_next`` holds the last safe
        # target (the existing queue-underrun path) until the next inference
        # is launched and accepted -- see the launch gating in the control
        # loop, which only launches a new request once ``pending_actions`` is
        # empty when ``async_inference`` is disabled.
        if not self.async_inference:
            candidate = candidate[: self.sync_wait_steps]

        # All decoding/blending checks finished. The control thread
        # performs one atomic list replacement; inference never mutates this queue.
        self.pending_actions = candidate
        self.timeline_resync_active = retime_from_arrival
        self.active_generation = launch.generation
        self.ik_recovery_barrier = None
        self.ik_recovery_prefix_reset = False
        self.active_chunk_capture_monotonic = launch.captured_monotonic
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
        self.last_hold_duration_s = (
            arrived_at - self.hold_started_at if self.hold_started_at is not None else None
        )
        self.hold_started_at = None
        if self.arm_hold_targets:
            # This is only a staged plan until its first checked command is
            # actually published.  Keep refreshing the physical hold if that
            # first row later fails safety, IK, or queue timing.
            self.enable_staged_generation = launch.generation
        self.waiting_fresh_after_enable = False
        if self.tracking_lag_active:
            self.tracking_lag_active = False
            self.tracking_lag_consecutive_cycles = 0
            self.tracking_reference_lag_since = None
            self.tracking_reference_lag_cycles = 0
            self.tracking_lag_required_after_monotonic = None
            self.tracking_lag_recovered_generation = int(launch.generation)
            self.tracking_lag_peak_error_rad = 0.0
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
        self.record_inference_completion(completion.arrived_monotonic)
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

    def discard_pending_actions(self, reason: str, *, kind: str = "other") -> None:
        count = len(self.pending_actions)
        self._record_queue_drop(count, reason, kind=kind)
        self.pending_actions.clear()
        self.queued_action_index = None
        self.timeline_resync_active = False

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
        feedback_snapshot: PiperFeedbackSnapshot | None = None,
    ) -> bool:
        """Execute one time-selected target or hold the last safe absolute target."""
        feedback_at = time.time() if feedback_captured_at is None else float(feedback_captured_at)
        self.last_feedback_at = feedback_at
        self._complete_pending_command_feedback(
            raw_delivery_state,
            qpos_m,
            protocol,
            feedback_at=feedback_at,
        )
        now_monotonic = time.monotonic()
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
        if self.eef_control_mode == "piper_eef":
            return self._execute_native(raw_delivery_state, qpos_m, protocol, feedback_at, now_monotonic)

        if (
            not self.external_control_checked
            and bool(getattr(self.args, "reject_external_control_streams", False))
        ):
            for side in sides:
                check_external_control_streams(
                    self.pipers[side],
                    threshold_hz=float(
                        getattr(
                            self.args,
                            "external_control_hz_threshold",
                            DEFAULT_EXTERNAL_CONTROL_HZ_THRESHOLD,
                        )
                    ),
                )
            self.external_control_checked = True

        safety_hold_only = False
        # Enable and post-enable hold are handled before timed-plan selection.
        # Otherwise the queue advances by wall time while the controller is still
        # switching out of STANDBY, and the first real command can jump deep into
        # a chunk even though no earlier target reached the robot.
        snapshot_matches = (
            feedback_snapshot is not None
            and set(feedback_snapshot.statuses) == set(sides)
            and np.array_equal(np.asarray(feedback_snapshot.qpos_m), np.asarray(qpos_m))
            and np.array_equal(
                np.asarray(feedback_snapshot.delivery_state),
                np.asarray(raw_delivery_state),
            )
        )
        try:
            statuses = (
                {side: dict(feedback_snapshot.statuses[side]) for side in sides}
                if snapshot_matches and feedback_snapshot is not None
                else {side: arm_status_dict(self.pipers[side]) for side in sides}
            )
            self.robot_status = (
                statuses if protocol.arm_mode == "bimanual" else statuses[sides[0]]
            )
            bad_status = {
                side: status
                for side, status in statuses.items()
                if (
                    side in self.robot_enabled
                    or side in self.arm_hold_targets
                )
                and (
                    status["arm_status"] != PIPER_ARM_STATUS_NORMAL
                    or status["err_code"] != 0
                    or status.get("feedback_fresh") is False
                )
            }
            if bad_status:
                raise RuntimeHardwareFault(f"Piper status is not normal: {bad_status}")

            if self.arm_hold_targets:
                driver_statuses = (
                    {
                        side: dict(feedback_snapshot.driver_statuses[side])
                        for side in sides
                        if side in self.arm_hold_targets
                        and feedback_snapshot is not None
                        and side in feedback_snapshot.driver_statuses
                    }
                    if snapshot_matches and feedback_snapshot is not None
                    else {
                        side: driver_enable_status_dict(self.pipers[side])
                        for side in sides
                        if side in self.arm_hold_targets
                    }
                )
                self.robot_driver_enable_status.update(driver_statuses)
                staged_hold_invalid = bool(
                    self.enable_staged_generation is not None
                    and any(
                        not self._piper_can_joint_mode_ready(statuses[side])
                        or driver_statuses[side]["ready"] is False
                        for side in driver_statuses
                    )
                )
                if staged_hold_invalid:
                    self._cancel_staged_enable_plan(
                        "staged post-enable plan lost CAN/MOVE_J or driver enable"
                    )
                if self.enable_staged_generation is None:
                    if self._maintain_post_enable_hold(
                        sides,
                        qpos_m,
                        statuses,
                        now_monotonic=now_monotonic,
                        driver_statuses=driver_statuses,
                    ):
                        return False

            allow_execution = bool(getattr(self.args, "allow_execution", False))
            gate_reason = None
            gate_state = None
            if not allow_execution:
                gate_reason = "local --allow-execution is absent"
                gate_state = "client_disabled"
            elif self.state == "client_disabled":
                gate_reason = self.blocked_reason or "local --allow-execution is absent"
                gate_state = "client_disabled"
            elif self.state == "blocked":
                gate_reason = self.blocked_reason or "execution is blocked"
                gate_state = "blocked"

            if gate_reason is not None:
                if self._cancel_staged_enable_plan(
                    gate_reason,
                    kind="expired" if "expired" in gate_reason else "other",
                ):
                    self._maintain_post_enable_hold(
                        sides,
                        qpos_m,
                        statuses,
                        now_monotonic=now_monotonic,
                        driver_statuses=driver_statuses,
                    )
                    return False
                self.discard_pending_actions(
                    gate_reason,
                    kind="expired" if "expired" in gate_reason else "other",
                )
                if self.last_safe_target is not None and self.robot_enabled:
                    safety_hold_only = True
                    self.state = "holding"
                    self.blocked_reason = (
                        f"policy execution disabled ({gate_reason}); holding last safe target"
                    )
                else:
                    return self._block(gate_state or "blocked", gate_reason)

            active_driver_statuses = (
                {
                    side: dict(feedback_snapshot.driver_statuses[side])
                    for side in sides
                    if side in self.robot_enabled
                    and side not in self.arm_hold_targets
                    and feedback_snapshot is not None
                    and side in feedback_snapshot.driver_statuses
                }
                if snapshot_matches and feedback_snapshot is not None
                else {
                    side: driver_enable_status_dict(self.pipers[side])
                    for side in sides
                    if side in self.robot_enabled and side not in self.arm_hold_targets
                }
            )
            self.robot_driver_enable_status.update(active_driver_statuses)
            lost_control_mode = [
                side
                for side in sides
                if side in self.robot_enabled
                and side not in self.arm_hold_targets
                and (
                    not self._piper_can_joint_mode_ready(statuses[side])
                    or active_driver_statuses[side]["ready"] is False
                )
            ]
            for side in lost_control_mode:
                logging.warning(
                    "%s Piper lost control readiness; measured-pose enable hold requires safe servo handoff; "
                    "arm_status=%s driver_status=%s",
                    side, statuses[side], active_driver_statuses[side],
                )
                self.robot_enabled.discard(side)
            if lost_control_mode:
                self.external_control_checked = False
                self.joint_trajectory = None
                for side in lost_control_mode:
                    self._ik_warm_starts.pop(side, None)
                self.last_command_monotonic = None

            missing_enabled = [side for side in sides if side not in self.robot_enabled]
            if missing_enabled:
                if not self.pending_actions and not safety_hold_only:
                    return self._mark_queue_underrun(holding=False)
                for side in missing_enabled:
                    side_index = sides.index(side)
                    hold_qpos = np.asarray(qpos_m, dtype=np.float64)[
                        side_index * 7 : (side_index + 1) * 7
                    ]
                    self._enable_robot(side, self.pipers[side], hold_qpos)
                statuses = {side: arm_status_dict(self.pipers[side]) for side in sides}
                self.robot_status = (
                    statuses if protocol.arm_mode == "bimanual" else statuses[sides[0]]
                )
                self.discard_pending_actions(
                    "Piper enabled; discarded pre-enable trajectory and waiting for stable hold",
                    kind="other",
                )
                self.enable_staged_generation = None
                self.waiting_fresh_after_enable = True
                self.fresh_inference_required_after_monotonic = None
                self.enable_hold_settled_at = None
                self.state = "armed"
                self.blocked_reason = "Piper enabled; stabilizing measured-pose hold"
                return False

        except ExecutionBlocked as exc:
            self.discard_pending_actions(str(exc), kind="other")
            if isinstance(exc, (RuntimeHardwareFault, PiperFeedbackStaleError)):
                self._block("blocked", str(exc))
                raise
            return self._block("blocked", str(exc))
        except Exception as exc:
            logging.exception("Piper enable/hold handshake failed")
            self.discard_pending_actions(
                f"Piper enable/hold handshake failed: {exc}", kind="other"
            )
            raise RuntimeHardwareFault("Piper enable/hold handshake failed") from exc

        if self.ik_recovery_fault:
            return self._block("blocked", self.ik_recovery_fault)
        if self.tracking_lag_active:
            self._update_tracking_lag_settle(qpos_m, now_monotonic, feedback_at=feedback_at)
            return False
        if self.ik_recovery_target is not None:
            if safety_hold_only or self.tracking_lag_active:
                return self._block("ready", "IK recovery cancelled by safety hold")
            return self._advance_ik_recovery(qpos_m, now_monotonic,
                                            raw_delivery_state=raw_delivery_state, protocol=protocol,
                                            feedback_at=feedback_at)

        execution_time = self._estimated_execution_time(now_monotonic)
        if self.pending_actions:
            future_index = self._first_future_target_index(
                self.pending_actions, execution_time
            )
            if future_index is None:
                reason = (
                    "active timed plan exhausted before the estimated actuator execution time"
                )
                self._record_queue_drop(
                    len(self.pending_actions), reason, kind="expired"
                )
                logging.debug(
                    "Execution catch-up: dropped entire %d-row queue as stale "
                    "(execution_time=%.6f)",
                    len(self.pending_actions),
                    execution_time,
                )
                self.pending_actions.clear()
                self.queued_action_index = None
                self.timeline_resync_active = False
            elif future_index:
                reason = (
                    f"dropped {future_index} targets older than execution_time={execution_time:.6f}"
                )
                self._record_queue_drop(future_index, reason, kind="expired")
                logging.debug(
                    "Execution catch-up: dropped %d/%d stale rows from the front "
                    "of the active queue (execution_time=%.6f)",
                    future_index,
                    len(self.pending_actions),
                    execution_time,
                )
                del self.pending_actions[:future_index]

        if self.enable_staged_generation is not None and not self.pending_actions:
            self._cancel_staged_enable_plan(
                "staged post-enable plan expired before its first command",
                kind="expired",
            )
            self._maintain_post_enable_hold(
                sides, qpos_m, statuses, now_monotonic=now_monotonic
            )
            return False

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
        target_prevalidation = False
        recovery = None
        recovery_target = None
        original_queued = queued
        projected_any = False
        goal_gate = False
        try:
            target_prevalidation = True
            if not self.planner_prefix_enabled:
                queued = self._filter_gripper_target(queued, qpos_m, protocol)
            prepared: dict[str, tuple[Any, ...]] = {}
            command_pipeline: dict[str, dict[str, Any]] = {}
            ik_requests = {}
            ik_results = {}
            for index, side in enumerate(sides):
                qpos_slice = np.asarray(qpos_m, dtype=np.float64)[
                    index * 7 : (index + 1) * 7
                ]
                if protocol.schema == "delivery":
                    # A queue underrun or an explicit safety hold must brake
                    # the guarded writer from the measured state.  Re-running
                    # IK for the previous Cartesian target here is unsafe: a
                    # target that was already rejected can fail every tick,
                    # while the Ruckig writer is trying to settle.  Holding
                    # measured joints also avoids adding IK latency to the
                    # recovery path.
                    if holding or safety_hold_only:
                        self._pose_trajectories.pop(side, None)
                        self._pose_reference_at.pop(side, None)
                        prepared[side] = (
                            qpos_slice[:6].copy(),
                            float(qpos_slice[6]),
                        )
                        command_pipeline[side] = {
                            "control_path": "delivery_safety_hold",
                            "pose_trajectory": None,
                            "pre_ik_eef_target": None,
                            "full_ik_solution_joints_rad": None,
                            "commanded_joints_rad": qpos_slice[:6].tolist(),
                            "commanded_gripper_m": float(qpos_slice[6]),
                            "ik_rate_limited": None,
                            "ik_diagnostics": None,
                        }
                        continue
                    current_state = np.asarray(raw_delivery_state, dtype=np.float64)[
                        index * 10 : (index + 1) * 10
                    ]
                    target = np.asarray(
                        queued.absolute_target[index * 10 : (index + 1) * 10],
                        dtype=np.float64,
                    )
                    self._check_eef_xyz_workspace(target[:3], side)
                    # Receding-horizon chunks may jump farther than one local
                    # IK solve can safely cover.  Execute the bounded first
                    # Cartesian step. Normal RTC timing continues; sustained
                    # error to the original target is checked by the lag guard.
                    if self.servo_backend == "ruckig":
                        target, projected = project_delivery_target_step(
                            current_state,
                            target,
                            max_translation_step_m=float(
                                getattr(self.args, "max_translation_step_m", DEFAULT_MAX_TRANSLATION_STEP_M)
                            ),
                            max_rotation_step_rad=float(
                                getattr(self.args, "max_rotation_step_rad", DEFAULT_MAX_ROTATION_STEP_RAD)
                            ),
                        )
                    else:
                        projected = False
                    projected_any = projected_any or projected
                    checked = _check_delivery_absolute_target(
                        current_state,
                        float(qpos_slice[6]),
                        target,
                        gripper_semantics=protocol.gripper_semantics,
                        max_translation_step_m=(float("inf") if self.servo_backend == "ruckig" else float(
                            getattr(
                                self.args,
                                "max_translation_step_m",
                                DEFAULT_MAX_TRANSLATION_STEP_M,
                            )
                        )),
                        max_rotation_step_rad=(float("inf") if self.servo_backend == "ruckig" else float(
                            getattr(
                                self.args,
                                "max_rotation_step_rad",
                                DEFAULT_MAX_ROTATION_STEP_RAD,
                            )
                        )),
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
                        workspace_x=(
                            tuple(self.args.workspace_x)
                            if getattr(self.args, "workspace_x", None) is not None
                            else None
                        ),
                        workspace_y=delivery_workspace_y(self.args, side),
                        workspace_z=(
                            tuple(self.args.workspace_z)
                            if getattr(self.args, "workspace_z", None) is not None
                            else None
                        ),
                    )
                    target_xyz, target_rpy_deg, target_gripper_m, opening_fraction = checked
                    if holding or safety_hold_only:
                        # A stationary/underrun hold must not integrate a hidden
                        # Cartesian reference while the actuator writer brakes.
                        self._pose_trajectories.pop(side, None)
                        self._pose_reference_at.pop(side, None)
                        pose_metrics = None
                    elif self.rtc_enabled:
                        # Point-to-point stopping envelopes change the RTC time
                        # contract. Keep normal references continuous; the
                        # optional shaper is used only after recovery invalidates
                        # the old prefix, in _advance_ik_recovery.
                        self._pose_trajectories.pop(side, None)
                        self._pose_reference_at.pop(side, None)
                        pose_metrics = None
                    else:
                        target_xyz, target_rpy_deg, pose_metrics = self._shape_delivery_pose(
                            side, current_state, target_xyz, target_rpy_deg,
                            allow_reanchor=True)
                    reference_target = target.copy()
                    if pose_metrics is not None:
                        reference_target[:3] = target_xyz
                        reference_target[3:9] = pose_metrics["reference_rotation6d"]
                        pose_metrics["raw_policy_absolute_target"] = target.tolist()
                    if (self.ik_workers is not None or (self.servo_backend == "ruckig"
                            and (self.ik_solver is None or isinstance(self.ik_solver, PiperContinuousIK)))):
                        kwargs = self._delivery_ik_kwargs(side)
                        kwargs["initial_joints_rad"] = self._ik_warm_starts.get(side)
                        ik_requests[side] = ((qpos_slice[:6], target_xyz, target_rpy_deg), kwargs)
                    else:
                        ik_results[side] = self._solve_delivery_ik(
                            qpos_slice[:6], target_xyz, target_rpy_deg, side=side,
                            warm_start_joints_rad=self._ik_warm_starts.get(side))
                    prepared[side] = (None, target_gripper_m)
                    command_pipeline[side] = {
                        "control_path": "delivery_continuous_ik_joint",
                        "policy_eef_target": original_queued.absolute_target[index*10:index*10+10].tolist(),
                        "pose_trajectory_bypass_reason": ("rtc_continuous_tracking" if self.rtc_enabled
                            and getattr(self.args, "pose_trajectory", False) else None),
                        "pose_trajectory": pose_metrics,
                        "command_input_feedback": {
                            "feedback_at": float(feedback_at),
                            "joints_rad": qpos_slice[:6].tolist(),
                            "gripper_opening_m": float(qpos_slice[6]),
                            "eef_state": current_state.tolist(),
                            "eef_xyz_m": current_state[:3].tolist(),
                            "eef_rotation6d": current_state[3:9].tolist(),
                        },
                        "pre_ik_eef_target": {
                            "absolute_target": reference_target.tolist(),
                            "projected_for_local_ik": bool(projected),
                            "xyz_m": target_xyz.tolist(),
                            "rotation6d": reference_target[3:9].tolist(),
                            "rpy_deg": np.asarray(target_rpy_deg, dtype=np.float64).tolist(),
                            "gripper_opening_fraction": float(opening_fraction),
                            "gripper_opening_m": float(target_gripper_m),
                        },
                        "commanded_gripper_m": float(target_gripper_m),
                    }
                elif protocol.schema == "joint":
                    target = np.asarray(
                        queued.absolute_target[index * 7 : (index + 1) * 7],
                        dtype=np.float64,
                    )
                    target_joints, target_gripper_m = build_checked_joint_target(
                        qpos_slice,
                        target,
                        joint_limits=effective_limits(self.pipers[side], JOINT_LIMITS_RAD),
                        max_joint_step_rad=(float("inf") if self.servo_backend == "ruckig"
                                            else float(getattr(self.args, "max_joint_step_rad", 0.3))),
                        max_gripper_step=(float("inf") if self.planner_prefix_enabled else
                                          getattr(self.args, "max_joint_gripper_step", None)),
                        max_gripper_step_m=getattr(self.args, "max_joint_gripper_step_m", None),
                        joint_limit_tolerance_rad=float(
                            getattr(
                                self.args,
                                "joint_limit_tolerance_rad",
                                DEFAULT_JOINT_LIMIT_TOLERANCE_RAD,
                            )
                        ),
                        gripper_range_tolerance=float(
                            getattr(
                                self.args,
                                "gripper_range_tolerance",
                                DEFAULT_GRIPPER_RANGE_TOLERANCE,
                            )
                        ),
                        gripper_semantics=protocol.gripper_semantics,
                    )
                    prepared[side] = (target_joints, target_gripper_m)
                    command_pipeline[side] = {
                        "control_path": "joint_absolute",
                        "command_input_feedback": {
                            "feedback_at": float(feedback_at),
                            "joints_rad": qpos_slice[:6].tolist(),
                            "gripper_opening_m": float(qpos_slice[6]),
                        },
                        "pre_ik_eef_target": None,
                        "full_ik_solution_joints_rad": None,
                        "commanded_joints_rad": target_joints.tolist(),
                        "commanded_gripper_m": float(target_gripper_m),
                        "ik_rate_limited": None,
                        "ik_diagnostics": None,
                    }
                else:
                    raise ExecutionBlocked(f"unsupported execution schema: {protocol.schema}")

            if ik_requests:
                from bimanual_vla.deployment.ik_workers import (
                    IKWorkerError, IKRecoveryExhausted, solve_recovering)
                try:
                    if self.servo_backend == "ruckig":
                        results = solve_recovering(self._solve_ik_batch, ik_requests)
                    else:
                        results = self.ik_workers.solve(ik_requests)
                except IKRecoveryExhausted as exc:
                    self.ik_recovery_fault = str(exc) + "; stop and restart required"
                    self.ik_recovery_prefix_reset = True
                    raise ExecutionBlocked(self.ik_recovery_fault) from exc
                except (IKWorkerError, EOFError, OSError) as exc:
                    raise ExecutionBlocked(f"parallel IK: {exc}") from exc
                recovery = {side: result.pop("recovery") for side, result in results.items()
                            if "recovery" in result} or None
                if recovery:
                    # Roll back speculative gripper filtering and pose integration.
                    (self._filtered_gripper_opening, self._gripper_extreme_candidate,
                     self._gripper_extreme_count, self._gripper_extreme_latch) = filter_snapshot
                    self._pose_trajectories.clear()
                    self._pose_reference_at.clear()
                    for index, side in enumerate(sides):
                        prepared[side] = (None, float(qpos_m[index * 7 + 6]))
                        command_pipeline[side]["ik_recovery"] = recovery[side]
                        # Interpolation is about FK, which can differ slightly
                        # from reported EEF feedback; recheck workspace bounds.
                        for axis_index, axis in enumerate(("x", "y", "z")):
                            bounds = (delivery_workspace_y(self.args, side) if axis == "y"
                                      else getattr(self.args, "workspace_" + axis, None))
                            if bounds is not None and not bounds[0] <= recovery[side]["xyz_m"][axis_index] <= bounds[1]:
                                raise ExecutionBlocked("shrunk IK target exceeds configured workspace")
                ik_results = {side: PiperIKSolveResult(**result) for side, result in results.items()}
            # Nothing is published until every arm has a validated solution.
            for side, ik_result in ik_results.items():
                target_joints = (ik_result.solution_joints_rad if self.servo_backend == "ruckig"
                                 else ik_result.command_joints_rad).copy()
                prepared[side] = (target_joints, prepared[side][1])
                command_pipeline[side].update({
                    "full_ik_solution_joints_rad": ik_result.solution_joints_rad.tolist(),
                    "commanded_joints_rad": target_joints.tolist(),
                    "commanded_gripper_m": float(prepared[side][1]),
                    "ik_rate_limited": ik_result.rate_limited and self.servo_backend != "ruckig",
                    "ik_diagnostics": {
                        "solve_elapsed_s": ik_result.solve_elapsed_s,
                        "selection_method": ik_result.selection_method,
                        "minimum_motion_elapsed_s": ik_result.minimum_motion_elapsed_s,
                        "minimum_motion_timed_out": ik_result.minimum_motion_timed_out,
                        "parallel_batch_wall_ms": (self.ik_workers.last_wall_ms
                                                   if self.ik_workers is not None else None),
                        "solution_position_error_m": ik_result.solution_position_error_m,
                        "solution_rotation_error_rad": ik_result.solution_rotation_error_rad,
                        "solution_max_joint_step_rad": ik_result.solution_max_joint_step_rad,
                        "solution_reference_step_rad": ik_result.solution_reference_step_rad,
                        "command_position_error_m": ik_result.command_position_error_m,
                        "command_rotation_error_rad": ik_result.command_rotation_error_rad,
                        "optimizer_success": ik_result.optimizer_success,
                        "optimizer_status": ik_result.optimizer_status,
                        "optimizer_nfev": ik_result.optimizer_nfev,
                        "tolerance_relaxed": ik_result.tolerance_relaxed,
                        "accepted_position_tolerance_m": ik_result.accepted_position_tolerance_m,
                        "accepted_rotation_tolerance_rad": ik_result.accepted_rotation_tolerance_rad,
                    },
                })

            if protocol.schema == "delivery" and self.servo_backend == "ruckig" and not (holding or safety_hold_only):
                # A gripper transition or one bounded projection is not a task
                # endpoint. Only actual IK failure/shrink enters goal recovery.
                goal_gate = bool(recovery)
            if goal_gate:
                self._check_eef_goal_workspace(original_queued.absolute_target, sides)
                (self._filtered_gripper_opening, self._gripper_extreme_candidate,
                 self._gripper_extreme_count, self._gripper_extreme_latch) = filter_snapshot
                for index, side in enumerate(sides):
                    prepared[side] = (prepared[side][0], float(qpos_m[index*7+6]))
                    command_pipeline[side]["commanded_gripper_m"] = float(qpos_m[index*7+6])
                    command_pipeline[side]["original_goal_retained"] = original_queued.absolute_target[index*10:index*10+10].tolist()
                recovery_target = np.concatenate([
                    np.append(prepared[side][0], prepared[side][1]) for side in sides])
            recovery_braking = bool(goal_gate and self._recovery_needs_braking())
            if recovery_braking:
                # The geometric candidate was computed before braking. Retain
                # it for diagnostics only; do not make it an active waypoint.
                braking_position = np.asarray(self.servo_loop.metrics()["state"]["position"])
                for index, side in enumerate(sides):
                    prepared[side] = (braking_position[index*6:index*6+6].copy(),
                                      float(qpos_m[index*7+6]))
            self._shape_prepared_targets(
                qpos_m,
                prepared,
                command_pipeline,
                sides,
                now_monotonic=now_monotonic,
            )
            command_started_at = time.time()
            command_started_monotonic = time.monotonic()
            servo_sample = None
            if self.servo_backend == "ruckig":
                # Servo faults are actuator failures, not skippable queue rows.
                target_prevalidation = False
                servo_sample = self._servo_publish_target(
                    np.asarray(qpos_m), prepared, sides, hold=holding or safety_hold_only or recovery_braking,
                    context={"generation": queued.generation, "queue_index": queued.queue_index,
                             "source_index": queued.source_index},
                    protocol=protocol,
                )
                actual = np.asarray(servo_sample["qpos"])
                for index, side in enumerate(sides):
                    command_pipeline[side]["servo_reference_joints_rad"] = prepared[side][0].tolist()
                    prepared[side] = (actual[index * 7:index * 7 + 6], float(actual[index * 7 + 6]))
                    command_pipeline[side].update(servo_sample["sides"][side])
                    command_pipeline[side]["commanded_joints_rad"] = prepared[side][0].tolist()
                    command_pipeline[side]["servo_backend"] = "ruckig"
                    command_pipeline[side]["servo_sample_sequence"] = servo_sample["sequence"]
            target_prevalidation = False

            # Blend and hold never bypass safety: both arms are fully prevalidated
            # against the same fresh feedback before either arm receives a command.
            wire_commands: dict[str, tuple[np.ndarray, int]] = {}
            for side in sides:
                target_joints, target_gripper_m = prepared[side]
                raw_joints = np.rint(target_joints * RAD_FACTOR).astype(np.int64)
                raw_gripper = round(target_gripper_m * GRIPPER_FACTOR)
                wire_commands[side] = (raw_joints, int(raw_gripper))
                command_pipeline[side]["piper_jointctrl_units"] = raw_joints.tolist()
                command_pipeline[side]["piper_gripperctrl_units"] = int(raw_gripper)
                command_pipeline[side]["piper_speed_pct"] = int(
                    getattr(self.args, "speed_pct", DEFAULT_INFERENCE_SPEED_PCT)
                )
                command_pipeline[side]["piper_gripper_effort"] = int(
                    getattr(self.args, "gripper_effort", 1000)
                )

            if servo_sample is None:
                # Publish bimanual commands in phases.  This avoids completing the
                # full Mode/Joint/Gripper sequence for one arm before the other arm
                # receives its JointCtrl target, and exposes the remaining skew.
                for side in sides:
                    piper = self.pipers[side]
                    started_at = time.time()
                    started_monotonic = time.monotonic()
                    piper.ModeCtrl(
                        0x01, 0x01, int(getattr(self.args, "speed_pct", DEFAULT_INFERENCE_SPEED_PCT)), 0x00
                    )
                    command_pipeline[side]["mode_ctrl_started_at"] = started_at
                    command_pipeline[side][
                        "mode_ctrl_started_monotonic"
                    ] = started_monotonic
                    command_pipeline[side]["mode_ctrl_at"] = time.time()
                    command_pipeline[side]["mode_ctrl_monotonic"] = time.monotonic()

                for side in sides:
                    piper = self.pipers[side]
                    raw_joints, _ = wire_commands[side]
                    started_at = time.time()
                    started_monotonic = time.monotonic()
                    piper.JointCtrl(*map(int, raw_joints))
                    command_pipeline[side]["joint_ctrl_started_at"] = started_at
                    command_pipeline[side][
                        "joint_ctrl_started_monotonic"
                    ] = started_monotonic
                    command_pipeline[side]["joint_ctrl_at"] = time.time()
                    command_pipeline[side]["joint_ctrl_monotonic"] = time.monotonic()

                for side in sides:
                    piper = self.pipers[side]
                    _, raw_gripper = wire_commands[side]
                    started_at = time.time()
                    started_monotonic = time.monotonic()
                    piper.GripperCtrl(
                        int(raw_gripper), int(getattr(self.args, "gripper_effort", 1000)), 0x01, 0
                    )
                    command_pipeline[side]["gripper_ctrl_started_at"] = started_at
                    command_pipeline[side][
                        "gripper_ctrl_started_monotonic"
                    ] = started_monotonic
                    command_pipeline[side]["gripper_ctrl_at"] = time.time()
                    command_pipeline[side]["gripper_ctrl_monotonic"] = time.monotonic()
        except ChunkPlanDeferred as exc:
            # Preparation has not changed the writer's plan. Keep its verified
            # trajectory (including terminal stop), and retry the queued chunk
            # or a fresher inference on the next control tick. Hard-pausing here
            # freezes nonzero planner velocity and prevents all future splices.
            (
                self._filtered_gripper_opening,
                self._gripper_extreme_candidate,
                self._gripper_extreme_count,
                self._gripper_extreme_latch,
            ) = filter_snapshot
            self.state = "ready"
            self.blocked_reason = f"chunk planning deferred: {exc}"
            now = time.monotonic()
            if now - getattr(self, "_last_chunk_defer_log_at", -float("inf")) >= 2.0:
                logging.warning("New chunk deferred; existing guarded writer retained: %s", exc)
                self._last_chunk_defer_log_at = now
            return False
        except (ExecutionBlocked, ServoError) as exc:
            staged_enable_failure = bool(
                self.enable_staged_generation == queued.generation
                and self.arm_hold_targets
            )
            (
                self._filtered_gripper_opening,
                self._gripper_extreme_candidate,
                self._gripper_extreme_count,
                self._gripper_extreme_latch,
            ) = filter_snapshot
            # A guarded writer can be paused while it brakes after a feedback
            # or publication fault.  This is recoverable once the arm is
            # stationary; leaving the controller in the permanent ``blocked``
            # gate prevents the next fresh inference from restarting it.  Drop
            # the stale queue and remain ready/holding instead.  Hardware and
            # driver faults still take the fail-closed blocked path below.
            if (
                isinstance(exc, ServoError)
                and "moving servo was safety-paused" in str(exc)
            ):
                self.discard_pending_actions(str(exc), kind="unsafe")
                self.unsafe_active = True
                self.queued_action_index = None
                self.timeline_resync_active = False
                return self._block("ready", f"waiting for paused servo to settle: {exc}")
            if (target_prevalidation or isinstance(exc, ChunkPlanRejected)) and self.pending_actions:
                # Do not chase later cumulative targets after the robot failed
                # to execute this row.  They assume this target was reached and
                # can create a self-sustaining unsafe-drop loop.  Drop the bad
                # row, abandon the dependent suffix, and hold the last safe
                # command until a fresh observation produces a new plan.
                self.pending_actions.pop(0)
                self._record_queue_drop(1, str(exc), kind="unsafe")
                if self.pending_actions:
                    suffix_count = len(self.pending_actions)
                    self._record_queue_drop(
                        suffix_count,
                        f"abandoned {suffix_count} dependent targets after unsafe target: {exc}",
                        kind="other",
                    )
                    self.pending_actions.clear()
                # Preserve the unsafe event as the latest classification even
                # though the dependent suffix was abandoned for another reason.
                self.last_queue_drop_reason = str(exc)[:500]
                self.last_queue_drop_kind = "unsafe"
                self.unsafe_active = True
                self.queued_action_index = None
                self.timeline_resync_active = False
                if staged_enable_failure:
                    self._cancel_staged_enable_plan(
                        f"staged post-enable target was unsafe: {exc}",
                        kind="unsafe",
                    )
                    return False
                return self._block("blocked" if self.ik_recovery_fault else "ready",
                                   f"dropped unsafe queued target: {exc}")
            if staged_enable_failure:
                self._cancel_staged_enable_plan(
                    f"staged post-enable target was rejected: {exc}", kind="other"
                )
                return False
            self.discard_pending_actions(str(exc), kind="other")
            return self._block("blocked", str(exc))
        except Exception as exc:
            (
                self._filtered_gripper_opening,
                self._gripper_extreme_candidate,
                self._gripper_extreme_count,
                self._gripper_extreme_latch,
            ) = filter_snapshot
            logging.exception("robot command failed")
            if (
                self.enable_staged_generation == queued.generation
                and self.arm_hold_targets
            ):
                self._cancel_staged_enable_plan(
                    f"staged post-enable robot command failed: {exc}", kind="other"
                )
                raise RuntimeHardwareFault("Piper command failed during staged enable") from exc
            self.discard_pending_actions(f"robot command failed: {exc}", kind="other")
            raise RuntimeHardwareFault("Piper command failed") from exc

        self._commit_staged_enable_plan(queued.generation)
        if not holding:
            self.pending_actions.pop(0)
            self.last_safe_target = replace(queued, hold=False)
            self.executed_plan_command_count += 1
            if not self.pending_actions:
                self.timeline_resync_active = False
        self.last_queued_action_index = queued.queue_index
        self.last_command_monotonic = time.monotonic()
        self.queued_action_index = self.pending_actions[0].queue_index if self.pending_actions else None
        self.last_wire_action = queued.wire_action.tolist()
        decoded_absolute_target = self._target_telemetry(
            protocol, queued.absolute_target
        )
        self.last_decoded_absolute_target = decoded_absolute_target
        command_at = time.time()
        command_monotonic = time.monotonic()
        self.current_timed_target_action = replace(queued, hold=holding)
        self.current_timed_target = self._timed_target_telemetry(
            self.current_timed_target_action,
            now_monotonic=now_monotonic,
            now_wall=command_at,
        )
        self.command_sequence += 1

        def command_skew_ms(field: str) -> float | None:
            timestamps = [
                float(command_pipeline[side][field])
                for side in sides
                if field in command_pipeline[side]
            ]
            if len(timestamps) < 2:
                return None
            return (max(timestamps) - min(timestamps)) * 1000.0

        command_trace = {
            "trace_version": 1,
            "command_sequence": self.command_sequence,
            "generation": int(queued.generation),
            "source_index": queued.source_index,
            "queue_index": int(queued.queue_index),
            "correlation": {
                "generation": int(queued.generation),
                "source_index": queued.source_index,
                "queue_index": int(queued.queue_index),
                "inference_generation": self.inference_generation,
                "timing_generation": self.last_transport_generation,
            },
            "schema": protocol.schema,
            "arm_mode": protocol.arm_mode,
            "arm_side": protocol.arm_side,
            "action_semantics": protocol.action_semantics,
            "gripper_semantics": protocol.gripper_semantics,
            "wire_action": queued.wire_action.tolist(),
            "decoded_absolute_target": decoded_absolute_target,
            "target_timing": dict(self.current_timed_target or {}),
            "blended": bool(queued.blended),
            "blend_step": queued.blend_step,
            "hold": bool(holding or recovery_braking),
            "command_started_at": float(command_started_at),
            "command_at": float(command_at),
            "command_started_monotonic": float(command_started_monotonic),
            "command_monotonic": float(command_monotonic),
            "command_publish_duration_ms": max(
                0.0, (command_monotonic - command_started_monotonic) * 1000.0
            ),
            "modectrl_skew_ms": command_skew_ms("mode_ctrl_monotonic"),
            "jointctrl_skew_ms": command_skew_ms("joint_ctrl_monotonic"),
            "gripperctrl_skew_ms": command_skew_ms("gripper_ctrl_monotonic"),
            "queue_anchor_at": self.queue_anchor_at,
            "queue_loaded_at": self.queue_loaded_at,
            "sides": command_pipeline,
        }
        self.last_actuator_command = command_trace
        # A candidate rejected before publication is not the previous accepted
        # branch. Commit warm starts only after paired publication succeeds.
        self._ik_warm_starts.update({side: result.solution_joints_rad.copy()
                                    for side, result in ik_results.items()})
        # Use the quantized values actually sent through JointCtrl for jitter
        # metrics, excluding grippers and decoded/model targets.
        self.last_commanded_joint_positions = np.concatenate(
            [wire_commands[side][0].astype(np.float32) / RAD_FACTOR for side in sides]
        )
        self._pending_feedback_command = command_trace
        self.last_command_at = command_at
        self.unsafe_active = False
        if (
            queued.generation == self.last_transport_generation
            and self.last_transport_first_command_generation != queued.generation
        ):
            response_received_monotonic = self._finite_timing_value(
                self.last_client_transport_timing.get("response_received_monotonic")
            )
            if response_received_monotonic is not None:
                self.last_client_transport_timing["result_to_first_command_ms"] = max(
                    0.0, (command_monotonic - response_received_monotonic) * 1000.0
                )
            if self.inference_capture_monotonic is not None:
                self.last_client_transport_timing["observation_to_first_command_ms"] = max(
                    0.0, (command_monotonic - self.inference_capture_monotonic) * 1000.0
                )
            self.last_transport_first_command_generation = queued.generation
        if holding:
            self.hold_count += 1
            self.state = "holding"
        else:
            self.state = "executing"
            self.blocked_reason = ""
        if goal_gate:
            self.discard_pending_actions("original EEF goal retained; invalidated old RTC suffix", kind="other")
            self.ik_recovery_target = recovery_target
            self.eef_goal_action = original_queued
            self.eef_goal_held_grippers = np.array([prepared[side][1] for side in sides])
            self.ik_recovery_target[[i*7+6 for i in range(len(sides))]] = self.eef_goal_held_grippers
            self.ik_recovery_braking = recovery_braking
            self._recovery_settle_feedback = None
            self._recovery_stable_since = None
            self._recovery_tracking_at = None
            self._recovery_tracking_cycles = 0
            self.eef_goal_releasing = False
            self.eef_goal_pose_reset_pending = bool(recovery and getattr(self.args, "pose_trajectory", False))
            self.eef_goal_pose_metrics = {side: command_pipeline[side].get("pose_trajectory") for side in sides}
            self.eef_goal_step_count = 1
            self.eef_goal_errors = self._eef_arrival_errors(raw_delivery_state, original_queued.absolute_target, sides)
            self.ik_recovery_started = time.monotonic()
            self.ik_recovery_barrier = self.ik_recovery_started
            self.ik_recovery_prefix_reset = True
            self._ik_warm_starts.clear()
            self.last_ik_recovery = recovery
            self.state = "recovering"
            logging.info("Original EEF goal retained after IK recovery (shrink=%s projected=%s); "
                         "grippers held until arrival; old RTC suffix cleared; braking_handoff=%s",
                         next(iter(recovery.values()))["fraction"] if recovery else None,
                         projected_any, recovery_braking)
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
    snapshot: ObservationSnapshot,
    protocol: PolicyProtocol,
    instruction: str | None,
    source_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a policy payload using only a frozen control-thread snapshot."""
    # Snapshot arrays are immutable and are consumed read-only by the official
    # OpenPI client.  Avoid copying the state/images again on every request;
    # the fallback copy is retained for alternate mutable snapshot callers.
    state = np.asarray(snapshot.state, dtype=np.float32)
    if state.flags.writeable:
        state = np.array(state, copy=True)
    if state.shape != (protocol.state_dim,) or not np.all(np.isfinite(state)):
        raise RuntimeError(
            f"{protocol.arm_mode} {protocol.schema} observation state must be finite "
            f"{protocol.state_dim}D, got {state.shape}"
        )
    images = snapshot.images
    send_policy_telemetry = bool(getattr(args, "send_policy_telemetry", True))
    if protocol.arm_mode == "bimanual":
        observation_images = {
            key: (
                images[key]
                if isinstance(images[key], np.ndarray)
                and images[key].dtype == np.uint8
                and not images[key].flags.writeable
                else np.array(images[key], dtype=np.uint8, copy=True)
            )
            for key in protocol.camera_keys
        }
    else:
        wrist_key = next(key for key in protocol.camera_keys if "wrist" in key)
        observation_images = {
            "cam_high": (
                images["cam_high"]
                if isinstance(images["cam_high"], np.ndarray)
                and images["cam_high"].dtype == np.uint8
                and not images["cam_high"].flags.writeable
                else np.array(images["cam_high"], dtype=np.uint8, copy=True)
            ),
            wrist_key: (
                images["cam_wrist"]
                if isinstance(images["cam_wrist"], np.ndarray)
                and images["cam_wrist"].dtype == np.uint8
                and not images["cam_wrist"].flags.writeable
                else np.array(images["cam_wrist"], dtype=np.uint8, copy=True)
            ),
        }
    client_metadata = {
        # Keep RTC, request correlation, one-way timing and the local execution
        # authorization visible to the dashboard in the minimal mode.
        "captured_at": float(snapshot.captured_at),
        "source_name": source_name,
        "allow_execution": bool(getattr(args, "allow_execution", False)),
        "rtc": {
            **_thaw_snapshot_value(snapshot.rtc_metadata),
            "session_id": str(getattr(args, "rtc_session_id", "")),
            "inference_generation": int(snapshot.generation),
        },
    }
    if send_policy_telemetry:
        image_timestamps = snapshot.image_timestamps
        image_monotonic_timestamps = snapshot.image_monotonic_timestamps
        if protocol.arm_mode == "bimanual":
            can_names = {"left": args.left_can, "right": args.right_can}
            camera_devices = {
                "cam_high": str(args.cam_high_device),
                "cam_left_wrist": str(args.cam_left_wrist_device),
                "cam_right_wrist": str(args.cam_right_wrist_device),
            }
        else:
            can_names = {protocol.arm_side: args.can}
            camera_devices = {
                "cam_high": str(args.cam_high_device),
                wrist_key: str(args.cam_wrist_device),
            }
        client_metadata.update({
            "captured_monotonic": float(snapshot.captured_monotonic),
            "state_captured_at": float(snapshot.captured_at),
            "state_captured_monotonic": float(snapshot.captured_monotonic),
            "image_set_captured_monotonic": float(snapshot.image_captured_monotonic),
            "image_state_skew_ms": snapshot.image_state_skew_s * 1000.0,
            "arm_mode": protocol.arm_mode,
            "arm_side": protocol.arm_side,
            "can_names": can_names,
            "camera_devices": camera_devices,
            "image_captured_at": {
                key: float(
                    image_timestamps["cam_wrist"]
                    if protocol.arm_mode == "single" and "wrist" in key
                    else image_timestamps[key]
                )
                for key in protocol.camera_keys
            },
            "image_captured_monotonic": {
                key: float(
                    image_monotonic_timestamps["cam_wrist"]
                    if protocol.arm_mode == "single" and "wrist" in key
                    else image_monotonic_timestamps[key]
                )
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
            **_thaw_snapshot_value(snapshot.execution_metadata),
        })
    observation = {
        "state": state,
        "images": observation_images,
        "client_metadata": client_metadata,
    }
    if protocol.image_transport == "jpeg_chw_v1":
        observation["_transport"] = {
            "image_codec": "jpeg_chw_v1",
            "quality": int(protocol.image_jpeg_quality),
        }
    # Prompt text is session state on the Policy side. The client sends it on
    # the first request and only when the local prompt revision changes.
    if instruction is not None:
        observation["prompt"] = str(instruction)
    return observation


def request_metadata_for_inference(
    *,
    prompt_revision: int,
    request_sent_at: float,
    inference_generation: int,
    send_policy_telemetry: bool,
    camera_capture_started_at: float,
    camera_capture_finished_at: float,
    camera_selection_started_monotonic: float,
    camera_selection_finished_monotonic: float,
) -> dict[str, Any]:
    """Keep correlation/timing fields while gating optional camera diagnostics."""
    metadata: dict[str, Any] = {
        "prompt_revision": int(prompt_revision),
        "request_sent_at": float(request_sent_at),
        "inference_generation": int(inference_generation),
    }
    if send_policy_telemetry:
        metadata.update({
            "camera_capture_started_at": camera_capture_started_at,
            "camera_capture_finished_at": camera_capture_finished_at,
            "camera_selection_started_monotonic": camera_selection_started_monotonic,
            "camera_selection_finished_monotonic": camera_selection_finished_monotonic,
        })
    return metadata


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
        f"  local_allow={getattr(execution.args, 'allow_execution', False)} "
        f"client_state={execution.state} command_sent={command_sent} "
        f"reason={execution.blocked_reason or '-'}",
        flush=True,
    )


def build_client_transport_timing(
    *,
    request_sent_at: float,
    request_sent_monotonic: float,
    response_received_at: float,
    response_received_monotonic: float,
    server_timing: dict[str, Any] | None,
    camera_capture_ms: float | None,
    inference_generation: int,
    client_transport_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure and report transport timing from the client side.

    The client owns the send/receive boundaries and computes the two one-way
    values from server timestamps echoed in the response. Those values are
    still estimates when the two machines' wall clocks are not synchronized,
    so the payload explicitly records that measurement contract. RTT and all
    local intervals use monotonic clocks.
    """
    server_timing = server_timing if isinstance(server_timing, dict) else {}
    client_transport_timing = (
        client_transport_timing
        if isinstance(client_transport_timing, dict)
        else {}
    )

    def finite(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    def interval(end: Any, start: Any) -> float | None:
        end_value = finite(end)
        start_value = finite(start)
        if end_value is None or start_value is None or end_value < start_value:
            return None
        return (end_value - start_value) * 1000.0

    server_request_received_at = finite(server_timing.get("server_request_received_at"))
    server_response_ready_at = finite(server_timing.get("server_response_ready_at"))
    model_inference_ms = finite(server_timing.get("model_inference_ms"))
    upload_ms = interval(server_request_received_at, request_sent_at)
    download_ms = interval(response_received_at, server_response_ready_at)
    # A negative one-way leg proves the two wall clocks cannot be compared.
    # The other leg can then look hundreds of milliseconds too large too.
    if upload_ms is None or download_ms is None:
        upload_ms = download_ms = None
    round_trip_ms = max(
        0.0, (float(response_received_monotonic) - float(request_sent_monotonic)) * 1000.0
    )
    one_way_total_ms = (
        upload_ms + download_ms
        if upload_ms is not None and download_ms is not None
        else None
    )
    non_model_rtt_ms = (
        max(0.0, round_trip_ms - model_inference_ms)
        if model_inference_ms is not None
        else None
    )
    pack_ms = finite(client_transport_timing.get("pack_ms"))
    unpack_ms = finite(client_transport_timing.get("unpack_ms"))
    server_unpack_ms = finite(server_timing.get("server_unpack_ms"))
    client_codec_ms = pack_ms + unpack_ms if pack_ms is not None and unpack_ms is not None else None
    unattributed_rpc_ms = (
        max(0.0, non_model_rtt_ms - client_codec_ms - server_unpack_ms)
        if non_model_rtt_ms is not None and client_codec_ms is not None and server_unpack_ms is not None
        else None
    )
    return {
        # Preserve the official WebSocket client's local breakdown.  These
        # fields distinguish msgpack/image packing from actual network wait;
        # without them a large ``round_trip_ms`` is easy to misdiagnose as
        # model latency.
        "client_pack_ms": finite(client_transport_timing.get("pack_ms")),
        "client_send_ms": finite(client_transport_timing.get("send_ms")),
        "client_wait_response_ms": finite(
            client_transport_timing.get("wait_response_ms")
        ),
        "client_unpack_ms": finite(client_transport_timing.get("unpack_ms")),
        "client_transport_total_ms": finite(
            client_transport_timing.get("total_ms")
        ),
        "request_image_shapes": client_transport_timing.get("request_image_shapes"),
        "client_codec_ms": client_codec_ms,
        "server_unpack_ms": server_unpack_ms,
        # Still includes socket scheduling, server response work and transfer.
        # Never use this residual as a measurement of pure network latency.
        "unattributed_rpc_ms": unattributed_rpc_ms,
        "request_bytes": int(client_transport_timing["request_bytes"])
        if isinstance(client_transport_timing.get("request_bytes"), (int, float))
        and math.isfinite(float(client_transport_timing["request_bytes"]))
        and float(client_transport_timing["request_bytes"]) >= 0
        else None,
        "response_bytes": int(client_transport_timing["response_bytes"])
        if isinstance(client_transport_timing.get("response_bytes"), (int, float))
        and math.isfinite(float(client_transport_timing["response_bytes"]))
        and float(client_transport_timing["response_bytes"]) >= 0
        else None,
        "camera_capture_ms": finite(camera_capture_ms),
        "observation_upload_ms": upload_ms,
        "client_observation_upload_ms": upload_ms,
        "model_inference_ms": model_inference_ms,
        "result_download_ms": download_ms,
        "client_result_download_ms": download_ms,
        "network_transport_total_ms": one_way_total_ms,
        "client_network_transport_total_ms": one_way_total_ms,
        "non_model_rtt_ms": non_model_rtt_ms,
        "round_trip_ms": round_trip_ms,
        "request_sent_at": finite(request_sent_at),
        "server_request_received_at": server_request_received_at,
        "server_model_completed_at": finite(server_timing.get("server_model_completed_at")),
        "server_response_ready_at": server_response_ready_at,
        "response_received_at": finite(response_received_at),
        "response_received_monotonic": finite(response_received_monotonic),
        "inference_generation": int(inference_generation),
        "timing_source": "client_wall_clock_echo",
        "one_way_timing_clock": "wall_clock",
        "one_way_timing_requires_clock_sync": True,
    }


def return_pipers_to_initial(
    pipers: dict[str, Any],
    initial_qpos: np.ndarray,
    *,
    arm_mode: str,
    speed_pct: int,
    hz: float,
    timeout_s: float,
    max_feedback_age_s: float,
    max_gripper_speed_m_s: float = 0.08,
    max_joint_speed_rad_s: float = DEFAULT_TRAJECTORY_MAX_SPEED_RAD_S,
    max_joint_acceleration_rad_s2: float = DEFAULT_TRAJECTORY_MAX_ACCELERATION_RAD_S2,
    max_joint_jerk_rad_s3: float = DEFAULT_TRAJECTORY_MAX_JERK_RAD_S3,
    smoothing_cutoff_hz: float = DEFAULT_TRAJECTORY_SMOOTHING_CUTOFF_HZ,
    tracking_time_constant_s: float = DEFAULT_TRAJECTORY_TRACKING_TIME_CONSTANT_S,
    command_lookahead_rad: float = DEFAULT_TRAJECTORY_COMMAND_LOOKAHEAD_RAD,
    max_tracking_error_rad: float = DEFAULT_TRAJECTORY_MAX_TRACKING_ERROR_RAD,
    hold_tolerance_rad: float = DEFAULT_ARM_HOLD_TOLERANCE_RAD,
    servo_backend: str = "legacy",
    servo_hz: float = DEFAULT_SERVO_HZ,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Move all commanded arms back to their captured startup pose safely."""
    if arm_mode not in {"single", "bimanual"}:
        raise ValueError(f"unsupported arm_mode for return: {arm_mode!r}")
    expected_sides = {"left", "right"} if arm_mode == "bimanual" else None
    if expected_sides is not None:
        if set(pipers) != expected_sides:
            raise ExecutionBlocked(
                "bimanual return requires exactly left/right Piper handles; "
                f"got {sorted(pipers)}"
            )
    elif len(pipers) != 1:
        raise ExecutionBlocked(
            "single-arm return requires exactly one Piper handle; "
            f"got {sorted(pipers)}"
        )
    sides = ("left", "right") if arm_mode == "bimanual" else tuple(pipers)
    initial = np.asarray(initial_qpos, dtype=np.float32)
    if initial.shape != (7 * len(sides),) or not np.isfinite(initial).all():
        raise ExecutionBlocked(f"initial return pose must be finite {7 * len(sides)}D")
    if not math.isfinite(float(hz)) or hz <= 0:
        raise ValueError("return hz must be positive")
    if not math.isfinite(float(timeout_s)) or timeout_s <= 0:
        raise ValueError("return timeout must be positive")
    lower = np.full(initial.shape, -np.inf, dtype=np.float32)
    upper = np.full(initial.shape, np.inf, dtype=np.float32)
    for index, side in enumerate(sides):
        offset = index * 7
        limits = effective_limits(pipers[side], JOINT_LIMITS_RAD)
        lower[offset : offset + 6] = limits[:, 0]
        upper[offset : offset + 6] = limits[:, 1]
    max_joint_speed_rad_s = effective_speed(pipers, max_joint_speed_rad_s, speed_pct=speed_pct)
    home = initial.copy()
    for index in range(len(sides)):
        offset = index * 7
        home[offset : offset + 6] = np.clip(
            home[offset : offset + 6], lower[offset : offset + 6], upper[offset : offset + 6]
        )
    current = np.concatenate(
        [read_output_qpos(pipers[side], max_feedback_age_s=max_feedback_age_s) for side in sides]
    ).astype(np.float32)
    joint_indices = np.asarray([i for i in range(len(current)) if i % 7 != 6])
    if servo_backend == "ruckig":
        hz = max(float(hz), float(servo_hz))
        shaper = RuckigJointServo(current[joint_indices], lower[joint_indices], upper[joint_indices],
            ServoConfig(hz=hz, max_velocity=max_joint_speed_rad_s,
                        max_acceleration=max_joint_acceleration_rad_s2, max_jerk=max_joint_jerk_rad_s3,
                        max_tracking_error=max_tracking_error_rad))
        # Return is a long motion at the firmware's configured speed. Bound
        # each leg and wait for measured arrival before advancing, so a slow
        # actuator cannot accumulate the entire home-path tracking error.
        return_leg = min(0.1, max_tracking_error_rad * 0.25)
        return_arrival = min(0.02, max_tracking_error_rad * 0.1)
        def next_return_target():
            delta = home[joint_indices] - shaper.position
            scale = min(1.0, return_leg / max(float(np.max(np.abs(delta))), 1e-12))
            return shaper.position + delta * scale
        shaper.retarget(next_return_target())
    else:
        shaper = JerkLimitedJointTrajectory(
            current,
            lower,
            upper,
            max_speed_rad_s=max_joint_speed_rad_s,
            max_acceleration_rad_s2=max_joint_acceleration_rad_s2,
            max_jerk_rad_s3=max_joint_jerk_rad_s3,
            smoothing_cutoff_hz=smoothing_cutoff_hz,
            tracking_time_constant_s=tracking_time_constant_s,
            command_lookahead_rad=command_lookahead_rad,
            max_tracking_error_rad=max_tracking_error_rad,
        )
    gripper_indices = np.asarray(
        [index * 7 + 6 for index in range(len(sides))], dtype=np.int64
    )
    previous_gripper = current[gripper_indices].copy()
    period = 1.0 / float(hz)
    started = time.monotonic()
    last_tick = started
    next_tick = started
    sent = 0
    joint_error = float("inf")
    gripper_error = float("inf")

    def send_exact_home() -> None:
        """Leave the firmware with the exact captured target, without lookahead."""
        for side in sides:
            pipers[side].ModeCtrl(0x01, 0x01, int(speed_pct), 0x00)
        for index, side in enumerate(sides):
            offset = index * 7
            pipers[side].JointCtrl(
                *[
                    int(round(float(value) * RAD_FACTOR))
                    for value in home[offset : offset + 6]
                ]
            )
        for index, side in enumerate(sides):
            pipers[side].GripperCtrl(
                int(round(float(home[index * 7 + 6]) * GRIPPER_FACTOR)),
                1000,
                0x01,
                0,
            )

    while time.monotonic() - started < float(timeout_s):
        tick = time.monotonic()
        dt = (
            period
            if sent == 0
            else float(np.clip(tick - last_tick, 0.5 * period, 1.5 * period))
        )
        last_tick = tick
        current = np.concatenate(
            [read_output_qpos(pipers[side], max_feedback_age_s=max_feedback_age_s) for side in sides]
        ).astype(np.float32)
        if on_sample is not None and sent % max(1, round(hz / 10)) == 0:
            on_sample({"elapsed_s": tick - started, "feedback": current.tolist(),
                       "home": home.tolist(), "sent_commands": sent,
                       "servo": shaper.metrics() if servo_backend == "ruckig" else None})
        joint_error = float(np.max(np.abs(current[joint_indices] - home[joint_indices])))
        gripper_error = float(np.max(np.abs(current[gripper_indices] - home[gripper_indices])))
        trajectory_settled = (servo_backend != "ruckig" or (
            np.max(np.abs(shaper.position - home[joint_indices])) < 1e-5
            and np.max(np.abs(shaper.velocity)) < 1e-4
            and np.max(np.abs(shaper.acceleration)) < 1e-3))
        if trajectory_settled and joint_error <= float(hold_tolerance_rad) and gripper_error <= 0.001:
            send_exact_home()
            sent += 1
            time.sleep(min(0.15, 2.0 * period))
            final = np.concatenate(
                [
                    read_output_qpos(
                        pipers[side], max_feedback_age_s=max_feedback_age_s
                    )
                    for side in sides
                ]
            ).astype(np.float32)
            joint_error = float(
                np.max(np.abs(final[joint_indices] - home[joint_indices]))
            )
            gripper_error = float(
                np.max(np.abs(final[gripper_indices] - home[gripper_indices]))
            )
            if joint_error <= float(hold_tolerance_rad) and gripper_error <= 0.001:
                return {
                    "returned": True,
                    "elapsed_s": time.monotonic() - started,
                    "sent_commands": sent,
                    "joint_error_rad": joint_error,
                    "gripper_error_m": gripper_error,
                }
            current = final
        for side in sides:
            status = arm_status_dict(pipers[side])
            if status.get("arm_status") != PIPER_ARM_STATUS_NORMAL or status.get("err_code") != 0:
                raise ExecutionBlocked(f"{side} Piper unhealthy during return: {status}")
        if servo_backend == "ruckig":
            leg_settled = (np.max(np.abs(shaper.position - shaper.target)) < 1e-5
                           and np.max(np.abs(shaper.velocity)) < 1e-4
                           and np.max(np.abs(shaper.acceleration)) < 1e-3)
            if (leg_settled
                    and np.max(np.abs(current[joint_indices] - shaper.target)) <= return_arrival
                    and np.max(np.abs(shaper.target - home[joint_indices])) > 1e-5):
                shaper.retarget(next_return_target())
            shaped = current.copy()
            shaped[joint_indices] = shaper.update(current[joint_indices])
        else:
            shaped, _ = shaper.update(current, home, dt)
        gripper_target = rate_limit_grippers(
            current[gripper_indices],
            home[gripper_indices],
            max_speed_m_s=max_gripper_speed_m_s,
            dt=dt,
            previous_target_m=previous_gripper,
        )
        previous_gripper = gripper_target.copy()
        for side in sides:
            pipers[side].ModeCtrl(0x01, 0x01, int(speed_pct), 0x00)
        for index, side in enumerate(sides):
            offset = index * 7
            pipers[side].JointCtrl(
                *[int(round(float(value) * RAD_FACTOR)) for value in shaped[offset : offset + 6]]
            )
        for index, side in enumerate(sides):
            pipers[side].GripperCtrl(
                int(round(float(gripper_target[index]) * GRIPPER_FACTOR)),
                1000,
                0x01,
                0,
            )
        sent += 1
        if servo_backend == "ruckig":
            next_tick += period
            if next_tick < time.monotonic():
                next_tick = time.monotonic() + period
            sleep_s = next_tick - time.monotonic()
        else:
            sleep_s = started + sent * period - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
    raise TimeoutError(
        f"return-to-initial timed out after {timeout_s:.1f}s; "
        f"joint_error={joint_error:.4f}rad gripper_error={gripper_error:.4f}m"
    )


def clip_startup_pose_to_limits(
    pipers: dict[str, Any], pose: np.ndarray, *, arm_mode: str, arm_side: str,
    joint_margin_rad: float = 0.,
) -> tuple[np.ndarray, dict[str, dict[str, list[float]]]]:
    """Clip a saved/current startup pose to the effective Piper limits.

    The saved return pose can contain a small calibrated zero offset (for
    example J2=-0.034 rad) even though the driver advertises J2 >= 0.  This
    helper only prepares the target; it never sends a command.  Motion is
    performed by ``return_pipers_to_initial`` so the target is reached with
    the same bounded Ruckig path used by replay/return.
    """
    sides = ("left", "right") if arm_mode == "bimanual" else (arm_side,)
    if not math.isfinite(joint_margin_rad) or joint_margin_rad < 0:
        raise ValueError("startup joint margin must be finite and nonnegative")
    target = np.asarray(pose, dtype=np.float32).copy()
    expected = 7 * len(sides)
    if target.shape != (expected,) or not np.isfinite(target).all():
        raise ExecutionBlocked(f"startup pose must be finite {expected}D")
    changes: dict[str, dict[str, list[float]]] = {}
    for index, side in enumerate(sides):
        if side not in pipers:
            raise ExecutionBlocked(f"startup pose has no Piper for {side}")
        offset = index * 7
        limits = effective_limits(pipers[side], JOINT_LIMITS_RAD)
        lower, upper = limits[:, 0]+joint_margin_rad, limits[:, 1]-joint_margin_rad
        if np.any(lower >= upper):
            raise ExecutionBlocked(f"{side} driver range too narrow for startup clearance")
        before = target[offset : offset + 6].copy()
        after = np.clip(before, lower, upper)
        target[offset : offset + 6] = after
        if not np.array_equal(before, after):
            changes[side] = {
                "before_rad": before.astype(float).tolist(),
                "after_rad": after.astype(float).tolist(),
            }
    return target, changes


def save_return_pose(args, qpos, *, safe_to_return=False):
    path = getattr(args, "initial_pose_file", None)
    if not path:
        return
    path = Path(path)
    data = {"qpos": np.asarray(qpos).tolist(), "arm_mode": args.arm_mode,
            "arm_side": args.arm_side, "can": args.can, "left_can": args.left_can,
            "right_can": args.right_can, "safe_to_return": safe_to_return}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def load_return_pose(path, args):
    saved = json.loads(Path(path).read_text())
    keys = ("arm_mode", "left_can", "right_can") if args.arm_mode == "bimanual" else ("arm_mode", "arm_side", "can")
    for key in keys:
        if saved.get(key) != getattr(args, key):
            raise ExecutionBlocked(f"saved return pose does not match {key}")
    qpos = np.asarray(saved['qpos'], dtype=np.float64)
    if qpos.shape != ((14,) if args.arm_mode == 'bimanual' else (7,)) or not np.isfinite(qpos).all():
        raise ExecutionBlocked("saved return pose must contain a finite matching joint vector")
    return qpos


def wait_for_return_stationary(pipers, args):
    # A retry can follow a return error while firmware is still reaching
    # its last target. Never initialize a zero-velocity plan while moving.
    previous = np.concatenate([read_output_qpos(p, max_feedback_age_s=args.max_feedback_age_s)
                               for p in pipers.values()])
    stationary_since = time.monotonic()
    deadline = stationary_since + 5.0
    while time.monotonic() < deadline:
        time.sleep(.05)
        current = np.concatenate([read_output_qpos(p, max_feedback_age_s=args.max_feedback_age_s)
                                  for p in pipers.values()])
        if np.max(np.abs(current - previous)) > .001:
            stationary_since = time.monotonic()
            previous = current
        if time.monotonic() - stationary_since >= .3:
            break
    else:
        raise ExecutionBlocked("robot has not settled; manual return was not started")


def move_pipers_to_safe_startup_pose(
    pipers: dict[str, Any],
    initial_qpos: np.ndarray,
    args: argparse.Namespace,
    *,
    controller: "ExecutionController | None" = None,
) -> tuple[np.ndarray, dict[str, dict[str, list[float]]], dict[str, Any]]:
    """Approach the saved startup pose before a physical policy run.

    EEF policies are especially sensitive to the initial Cartesian reference.
    Use the same enable, Ruckig, feedback-arrival, and settling sequence as
    replay. The saved pose is clipped to queried driver limits. Native EEF
    handoff additionally needs clearance for firmware FK/IK quantization.
    """
    native = getattr(args, "eef_control_mode", "local_ik_joint") == "piper_eef"
    arrival_tolerance = float(getattr(args, "arm_hold_tolerance_rad", DEFAULT_ARM_HOLD_TOLERANCE_RAD))
    if native:
        arrival_tolerance = min(arrival_tolerance, float(np.deg2rad(.1)))
    target, clipped = clip_startup_pose_to_limits(
        pipers,
        initial_qpos,
        arm_mode=args.arm_mode,
        arm_side=args.arm_side,
        joint_margin_rad=NATIVE_STARTUP_MARGIN_RAD if native else 0.,
    )
    recover_return_faults(pipers, driver_enable_status_dict)
    if controller is None:
        controller = ExecutionController(pipers, args)
    sides = ("left", "right") if args.arm_mode == "bimanual" else (args.arm_side,)
    for side in sides:
        hold_qpos = read_output_qpos(
            pipers[side], max_feedback_age_s=args.max_feedback_age_s
        )
        logging.info(
            "Preparing %s Piper for EEF startup approach: enable and hold measured pose",
            side,
        )
        controller._enable_robot(side, pipers[side], hold_qpos)
    wait_for_return_stationary(pipers, args)
    logging.info(
        "EEF startup safe pose: moving to clipped target%s",
        " (limits clipped)" if clipped else "",
    )
    if native:
        logging.info("Native EEF startup: 1 degree joint-limit clearance; confirming measured arrival before MOVE_P")
    result = return_pipers_to_initial(
        pipers,
        target,
        arm_mode=args.arm_mode,
        speed_pct=int(getattr(args, "return_speed_pct", None) or args.speed_pct),
        hz=float(getattr(args, "return_hz", DEFAULT_RETURN_HZ)),
        timeout_s=float(getattr(args, "return_timeout_s", DEFAULT_RETURN_TIMEOUT_S)),
        max_feedback_age_s=float(args.max_feedback_age_s),
        servo_backend="ruckig",
        servo_hz=float(getattr(args, "servo_hz", DEFAULT_SERVO_HZ)),
        max_joint_speed_rad_s=float(
            getattr(args, "trajectory_max_speed_rad_s", DEFAULT_TRAJECTORY_MAX_SPEED_RAD_S)
        ),
        max_joint_acceleration_rad_s2=float(
            getattr(
                args,
                "trajectory_max_acceleration_rad_s2",
                DEFAULT_TRAJECTORY_MAX_ACCELERATION_RAD_S2,
            )
        ),
        max_joint_jerk_rad_s3=float(
            getattr(args, "trajectory_max_jerk_rad_s3", DEFAULT_TRAJECTORY_MAX_JERK_RAD_S3)
        ),
        max_tracking_error_rad=float(
            getattr(args, "trajectory_max_tracking_error_rad", DEFAULT_TRAJECTORY_MAX_TRACKING_ERROR_RAD)
        ),
        hold_tolerance_rad=arrival_tolerance,
    )
    wait_for_return_stationary(pipers, args)
    if native:
        measured_margins = {}
        for index, side in enumerate(sides):
            measured = read_output_qpos(pipers[side], max_feedback_age_s=args.max_feedback_age_s)[:6]
            limits = effective_limits(pipers[side], JOINT_LIMITS_RAD)
            margin = np.minimum(measured-limits[:, 0], limits[:, 1]-measured)
            if (np.any(margin < NATIVE_STARTUP_MIN_MARGIN_RAD)
                    or np.max(np.abs(measured-target[index*7:index*7+6])) > arrival_tolerance):
                raise ExecutionBlocked(f"{side} native startup did not reach interior joint pose; MOVE_P prohibited")
            measured_margins[side] = np.rad2deg(margin).tolist()
        result["native_startup_joint_margins_deg"] = measured_margins
    return target, clipped, result


def run_return_only(args):
    if not args.allow_execution or not args.initial_pose_file:
        raise ExecutionBlocked("manual return requires execution permission and a saved session pose")
    home = load_return_pose(args.initial_pose_file, args)
    interfaces = ({"left": args.left_can, "right": args.right_can}
                  if args.arm_mode == "bimanual" else {args.arm_side: args.can})
    pipers = {}
    try:
        for side, interface in interfaces.items():
            pipers[side] = connect_piper(interface)
        wait_for_return_stationary(pipers, args)
        recover_return_faults(pipers, driver_enable_status_dict)
        enable_controller = ExecutionController(pipers, args)
        for side, piper in pipers.items():
            hold_qpos = read_output_qpos(piper, max_feedback_age_s=args.max_feedback_age_s)
            logging.info("Preparing %s Piper for manual return: enable and hold current pose", side)
            enable_controller._enable_robot(side, piper, hold_qpos)
        wait_for_return_stationary(pipers, args)
        logging.info("Returning to saved session initial pose (no policy or cameras).")
        result = return_pipers_to_initial(
            pipers, home, arm_mode=args.arm_mode,
            speed_pct=args.return_speed_pct, hz=args.return_hz, timeout_s=args.return_timeout_s,
            max_feedback_age_s=args.max_feedback_age_s, servo_backend="ruckig", servo_hz=args.servo_hz,
            max_joint_speed_rad_s=args.trajectory_max_speed_rad_s,
            max_joint_acceleration_rad_s2=args.trajectory_max_acceleration_rad_s2,
            max_joint_jerk_rad_s3=args.trajectory_max_jerk_rad_s3,
            max_tracking_error_rad=args.trajectory_max_tracking_error_rad,
            hold_tolerance_rad=args.arm_hold_tolerance_rad)
        logging.info("Return to initial pose completed: %s", result)
    finally:
        for piper in pipers.values():
            piper.DisconnectPort()


def run_rtc_client(args: argparse.Namespace) -> None:
    """Run the physical robot RTC control loop, never a GUI inference preview.

    The loop keeps actuator feedback/command publication on the control thread
    while camera capture and Policy inference run asynchronously.  All action
    chunks enter the timestamped queue only after schema, freshness, execution
    authorization, workspace, IK, and Piper-status checks pass.
    """
    from bimanual_vla.device_guard import arm_role_request, camera_role_request, check_devices

    requested_arms = arm_role_request(
        args.arm_mode, args.arm_side,
        single=args.can, left=args.left_can, right=args.right_can,
    )
    requested_cameras = camera_role_request(
        args.arm_mode, args.arm_side,
        overhead=args.cam_high_device, wrist=args.cam_wrist_device,
        left_wrist=args.cam_left_wrist_device,
        right_wrist=args.cam_right_wrist_device,
    )
    resolved = check_devices(
        purpose="deployment",
        arm_mode=args.arm_mode,
        arm_side=args.arm_side,
        arms=requested_arms,
        cameras=requested_cameras,
        resolver=getattr(args, "rlsok_resolver", None),
    )
    if resolved is not None:
        args.can = resolved.arms.get(args.arm_side, args.can)
        args.left_can = resolved.arms.get("left", args.left_can)
        args.right_can = resolved.arms.get("right", args.right_can)
        args.cam_high_device = resolved.cameras["overhead"]
        args.cam_wrist_device = resolved.cameras.get(f"{args.arm_side}_wrist", args.cam_wrist_device)
        args.cam_left_wrist_device = resolved.cameras.get("left_wrist", args.cam_left_wrist_device)
        args.cam_right_wrist_device = resolved.cameras.get("right_wrist", args.cam_right_wrist_device)
    output_mode = getattr(args, "output_mode", "auto")
    for key in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        if args.host not in entries:
            entries.append(args.host)
        os.environ[key] = ",".join(entries)

    source_name = args.source_name or socket.gethostname()
    if not getattr(args, "rtc_session_id", None):
        args.rtc_session_id = uuid.uuid4().hex
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
        from bimanual_vla.collection.camera import select_video_device
        # Reviewed selectors must reach CameraCapture unchanged; its strict mode
        # rejects missing devices instead of replacing the resolver's decision.
        wrist_device = (args.cam_wrist_device if resolved is not None else
                        select_video_device(f"cam_{args.arm_side}_wrist", args.cam_wrist_device))
        camera_ids = {"cam_high": args.cam_high_device, "cam_wrist": wrist_device}
        pipers = {args.arm_side: connect_piper(args.can)}

    monitoring = MonitoringRecorder(
        args.monitoring_dir,
        args,
        enabled=not bool(getattr(args, "no_monitoring", False)),
        periodic_rate_hz=float(getattr(args, "monitoring_rate", 1.0)),
        level=str(getattr(args, "monitoring_level", "compact")),
    )
    monitoring.record(
        "piper_connected",
        can_interfaces={side: getattr(piper, "can_name", None) for side, piper in pipers.items()},
        arm_mode=args.arm_mode,
        arm_side=args.arm_side,
    )
    execution = ExecutionController(pipers, args)
    cameras = CameraCapture(
        cam_ids=camera_ids,
        fps=args.camera_fps,
        image_hw=IMAGE_HW,
        resize_mode="pad",  # Updated from checkpoint metadata on connection.
        capture_hw=CAMERA_SOURCE_HW,
        parallel_reads=True,
        strict_selectors=resolved is not None,
        policy_image_hw=(224, 224) if args.preresize_policy_images else None,
    )
    preview = CameraPreview(
        enabled=bool(getattr(args, "camera_preview", False)),
        fps=float(getattr(args, "camera_preview_fps", 8.0)),
    )
    cameras.set_preview_enabled(preview.enabled)
    worker = AsyncPolicyInference()
    runtime_device_guard = (
        RuntimeDeviceGuard(
            baseline=resolved, purpose="deployment",
            arm_mode=args.arm_mode, arm_side=args.arm_side,
            arms=requested_arms, cameras=requested_cameras,
            resolver=getattr(args, "rlsok_resolver", None),
        )
        if resolved is not None else None
    )
    recorder = DeploymentRunRecorder(
        args.record_root,
        video_fps=(args.record_video_fps or args.camera_fps),
        enabled=not args.no_recording,
    )
    policy = None
    protocol = None
    count = 0
    command_count = 0
    hardware_fault: str | None = None
    initial_qpos: np.ndarray | None = None
    return_attempted = False
    returned_to_initial = False
    once_result_accepted = False
    startup_safe_pose_done = False
    last_reconnect_attempt = 0.0
    prompt_revision = 0
    prompt_sent_revision: int | None = None
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
        sides_for_initial = (
            ("left", "right") if args.arm_mode == "bimanual" else (args.arm_side,)
        )
        initial_qpos = np.concatenate(
            [
                read_output_qpos(
                    pipers[side],
                    max_feedback_age_s=args.max_feedback_age_s,
                )
                for side in sides_for_initial
            ]
        ).astype(np.float32)
        monitoring.record("initial_pose_captured", qpos_m=initial_qpos)
        if getattr(args, "return_target_file", None):
            initial_qpos = load_return_pose(args.return_target_file, args)
            monitoring.record("fixed_return_pose_loaded", qpos_m=initial_qpos, path=args.return_target_file)
            logging.info("Using fixed saved return pose: %s", args.return_target_file)
        save_return_pose(args, initial_qpos)
        execution.servo_sample_callback = recorder.record_servo_tick
        recorder.start(
            {
                "source_name": source_name,
                "policy_host": args.host,
                "policy_port": int(args.port),
                "instruction": args.instruction,
                "arm_mode": args.arm_mode,
                "arm_side": args.arm_side,
                "can_interfaces": (
                    {"left": args.left_can, "right": args.right_can}
                    if args.arm_mode == "bimanual"
                    else {args.arm_side: args.can}
                ),
                "camera_devices": {key: str(value) for key, value in camera_ids.items()},
                "camera_checks": camera_checks,
                "control_hz": float(args.control_hz),
                "inference_hz": float(args.hz),
                "rtc_execution_horizon": int(args.rtc_execution_horizon),
                "trajectory_spike_suppression": bool(args.trajectory_spike_suppression),
                "runtime_configuration": execution.run_configuration(),
                "recording_video_source": "camera_stream",
                "camera_capture_fps": float(args.camera_fps),
            }
        )
        record_camera_stream = None
        if recorder.is_active:
            def record_camera_stream(
                images: dict[str, np.ndarray],
                image_timestamps: dict[str, float],
                captured_monotonic: float,
            ) -> None:
                try:
                    recorder.record_camera_frames(
                        images,
                        image_timestamps,
                        monotonic_timestamp=captured_monotonic,
                    )
                except Exception:
                    logging.exception("Failed to record background camera frames")

        try:
            cameras.start_background_capture(
                record_camera_stream, fps=args.camera_fps
            )
        except Exception as exc:
            raise RuntimeError(
                "RTC deployment requires the background camera stream for "
                "time-aligned observation snapshots"
            ) from exc
        logging.info(
            "Deployment recording: %s",
            recorder.run_dir if recorder.is_active else "disabled",
        )
        monitoring.record("camera_ready", camera_checks=camera_checks, camera_ids=camera_ids)
        if runtime_device_guard is not None:
            runtime_device_guard.start()
        logging.warning(
            "%s %s client: output_mode=%s control=%.3g Hz periodic/retry=%.3g Hz trigger=%s step=%d "
            "expected_chunk=%d minimum_chunk=%d. Robot commands require local --allow-execution and safety checks.",
            "EXECUTION-CAPABLE" if args.allow_execution else "SHADOW-ONLY",
            args.arm_mode,
            output_mode,
            args.control_hz,
            args.hz,
            execution.inference_trigger_mode,
            execution.inference_trigger_step,
            DEFAULT_OPENPI_CHUNK_STEPS,
            args.min_action_chunk_steps,
        )

        next_control_at = time.monotonic()
        launch_schedule = PeriodicSchedule(args.hz, next_at=next_control_at)

        def handle_inference_completion(completion: InferenceCompletion) -> bool:
            nonlocal policy, count, once_result_accepted
            assert protocol is not None
            refresh_reference = False
            if completion.error is not None:
                execution.reject_inference_completion(completion)
                try:
                    recorder.record_model_result(
                        launch=completion.launch,
                        result=None,
                        arrived_at=completion.arrived_at,
                        arrived_monotonic=completion.arrived_monotonic,
                        protocol=protocol,
                        accepted=False,
                        rejection=execution.rejected_result,
                        error=completion.error,
                    )
                except Exception:
                    logging.exception("Failed to record failed model inference")
                monitoring.record(
                    "inference_error",
                    generation=completion.launch.generation,
                    captured_at=completion.launch.captured_at,
                    launched_at=completion.launch.launched_at,
                    arrived_at=completion.arrived_at,
                    image_timestamps=completion.launch.image_timestamps,
                    error=repr(completion.error),
                    execution=execution.metadata(compact=monitoring.level != "full"),
                )
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
                refresh_reference = accepted and execution.servo_backend == "ruckig"
                try:
                    recorder.record_model_result(
                        launch=completion.launch,
                        result=completion.result,
                        arrived_at=completion.arrived_at,
                        arrived_monotonic=completion.arrived_monotonic,
                        protocol=protocol,
                        accepted=accepted,
                        rejection=None if accepted else execution.rejected_result,
                    )
                except Exception:
                    logging.exception("Failed to record model action chunk")
                monitoring.record(
                    "inference_result",
                    generation=completion.launch.generation,
                    captured_at=completion.launch.captured_at,
                    launched_at=completion.launch.launched_at,
                    arrived_at=completion.arrived_at,
                    image_timestamps=completion.launch.image_timestamps,
                    accepted=accepted,
                    # The raw action chunk is already persisted by
                    # DeploymentRunRecorder.  Deep JSON conversion
                    # of a 50x14 numpy array here used to run on the
                    # 20 Hz control thread and could consume a
                    # noticeable slice of the next tick.  Keep only
                    # bounded diagnostics in the low-latency event
                    # stream.
                    result_summary=(
                        {
                            "actions_shape": list(
                                np.asarray(
                                    completion.result.get("actions")
                                    if isinstance(completion.result, dict)
                                    else []
                                ).shape
                            ),
                            "transport_timing": (
                                completion.result.get("transport_timing")
                                if isinstance(completion.result, dict)
                                and isinstance(
                                    completion.result.get("transport_timing"),
                                    dict,
                                )
                                else None
                            ),
                            "rtc": (
                                completion.result.get("rtc")
                                if isinstance(completion.result, dict)
                                and isinstance(completion.result.get("rtc"), dict)
                                else None
                            ),
                        }
                        if isinstance(completion.result, dict)
                        else None
                    ),
                    # Keep result telemetry bounded on the control
                    # thread. The recorder still stores the full
                    # model response and command trace separately.
                    execution=execution.metadata(compact=True),
                )
                logging.debug(
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
                if not execution.async_inference:
                    # Synchronous mode never overlaps a request with
                    # execution of the previous chunk, so this is a
                    # clean, uncontaminated round-trip measurement.
                    # ``hold_s`` is the wall-clock time the robot
                    # actually sat still waiting for this result --
                    # it should track ``latency`` closely, since the
                    # next launch fires the instant the queue drains.
                    logging.info(
                        "Synchronous inference generation=%d round-trip "
                        "latency=%.3fs hold_s=%s accepted=%s queued=%d/%d steps",
                        completion.launch.generation,
                        execution.inference_latency_s or 0.0,
                        (
                            f"{execution.last_hold_duration_s:.3f}"
                            if execution.last_hold_duration_s is not None
                            else "n/a"
                        ),
                        accepted,
                        execution.pending_action_count,
                        execution.sync_wait_steps,
                    )
                count += 1
                if args.once and accepted:
                    once_result_accepted = True
            return refresh_reference

        manual_trigger = ManualInferenceTrigger(getattr(args, "manual_trigger_file", None))
        reference_refresh_tick = False
        while True:
            tick_started = time.monotonic()
            execution.record_control_tick(overrun=tick_started > next_control_at + control_period)
            if preview.enabled and not reference_refresh_tick:
                preview.update(cameras.latest_preview_images)
            command_sent = False
            completion: InferenceCompletion | None = None
            reading_hardware_feedback = False
            try:
                require_can_send_healthy(pipers, error_type=RuntimeHardwareFault)
                if runtime_device_guard is not None and runtime_device_guard.fault is not None:
                    raise RuntimeHardwareFault(runtime_device_guard.fault)
                try:
                    cameras.assert_background_healthy(
                        max_frame_age_s=max(0.5, 3.0 / float(args.camera_fps))
                    )
                except RuntimeError as exc:
                    raise RuntimeHardwareFault(f"camera stream unavailable: {exc}") from exc
                if preview.enabled:
                    preview.update(cameras.latest_preview_images)
                if (
                    policy is None
                    and not worker.in_flight
                    and (protocol is None or not execution.pending_action_count)
                ):
                    if tick_started - last_reconnect_attempt >= args.reconnect_delay:
                        last_reconnect_attempt = tick_started
                        try:
                            policy, protocol = connect_policy(
                                args.host,
                                args.port,
                                args.arm_side,
                                args.arm_mode,
                                output_mode,
                            )
                            if protocol.image_resize_mode not in {"pad", "stretch"}:
                                raise RuntimeError("unsupported policy image resize mode")
                            policy_image_hw = (
                                (protocol.image_size, protocol.image_size)
                                if protocol.image_resize_mode == "stretch" or args.preresize_policy_images
                                else None
                            )
                            if (cameras.resize_mode != protocol.image_resize_mode
                                    or cameras._policy_image_hw != policy_image_hw):
                                cameras.stop_background_capture()
                                cameras.resize_mode = protocol.image_resize_mode
                                cameras._policy_image_hw = policy_image_hw
                                cameras.start_background_capture(record_camera_stream, fps=args.camera_fps)
                            execution.configure_protocol(protocol)
                            execution.configure_ik_workers(protocol)
                            # Delivery/EEF policies need the Cartesian reference
                            # to start from a legal, settled joint pose.  Do
                            # this after the policy schema is known but before
                            # the first observation is sent.  Joint policies
                            # retain the historical measured-pose startup.
                            if (
                                protocol.schema == "delivery"
                                and not startup_safe_pose_done
                                and bool(getattr(args, "allow_execution", False))
                            ):
                                cameras.stop_background_capture()
                                try:
                                    (
                                        startup_target,
                                        startup_clipped,
                                        startup_result,
                                    ) = move_pipers_to_safe_startup_pose(
                                        pipers,
                                        initial_qpos,
                                        args,
                                        controller=execution,
                                    )
                                except Exception as exc:
                                    logging.exception("EEF startup safe-pose approach failed")
                                    raise RuntimeHardwareFault(f"EEF startup approach failed: {exc}") from exc
                                startup_safe_pose_done = True
                                initial_qpos = startup_target
                                monitoring.record(
                                    "startup_safe_pose_reached",
                                    qpos_m=startup_target,
                                    clipped=startup_clipped,
                                    result=startup_result,
                                )
                                recorder.update_metadata(
                                    {
                                        "startup_safe_pose": {
                                            "requested": True,
                                            "clipped": startup_clipped,
                                            "target_qpos_m": startup_target.tolist(),
                                            "result": startup_result,
                                        }
                                    }
                                )
                                # The startup approach already established a
                                # confirmed CAN/MOVE_J hold.  Let the normal
                                # execution controller treat it as settled so
                                # the first accepted EEF chunk is not discarded
                                # by a second enable handshake.
                                for side in (
                                    ("left", "right")
                                    if args.arm_mode == "bimanual"
                                    else (args.arm_side,)
                                ):
                                    execution.robot_enabled.add(side)
                                execution.arm_hold_targets.clear()
                                execution.arm_hold_gripper_targets.clear()
                                execution.arm_hold_started_at.clear()
                                execution.arm_hold_stable_since.clear()
                                execution.waiting_fresh_after_enable = False
                                execution.fresh_inference_required_after_monotonic = None
                                execution.enable_hold_settled_at = time.time()
                                execution.state = "ready"
                                execution.blocked_reason = ""
                                cameras.start_background_capture(
                                    record_camera_stream, fps=args.camera_fps
                                )
                            prompt_sent_revision = None
                            recorder.update_metadata({
                                "runtime_configuration": execution.run_configuration(),
                                "policy_protocol": {
                                    "requested_output_mode": output_mode,
                                    "schema": protocol.schema,
                                    "state_dim": protocol.state_dim,
                                    "action_dim": protocol.action_dim,
                                    "arm_mode": protocol.arm_mode,
                                    "arm_side": protocol.arm_side,
                                    "action_semantics": protocol.action_semantics,
                                    "camera_keys": list(protocol.camera_keys),
                                    "action_hz": protocol.action_hz,
                                    "gripper_semantics": protocol.gripper_semantics,
                                    "state_gripper_semantics": protocol.state_gripper_semantics,
                                    "contract_version": protocol.contract_version,
                                    "action_horizon": protocol.action_horizon,
                                    "image_resize_mode": protocol.image_resize_mode,
                                    "image_size": protocol.image_size,
                                    "image_transport": protocol.image_transport,
                                    "image_jpeg_quality": protocol.image_jpeg_quality,
                                }
                            })
                            control_period = 1.0 / execution.control_hz
                            launch_schedule = PeriodicSchedule(
                                execution.inference_hz, next_at=tick_started
                            )
                            monitoring.record(
                                "policy_connected",
                                requested_output_mode=output_mode,
                                protocol=vars(protocol),
                                metadata=execution.metadata(compact=monitoring.level != "full"),
                            )
                        except RuntimeHardwareFault:
                            raise
                        except Exception as exc:
                            execution._block("blocked", f"policy connection unavailable: {exc}")
                            monitoring.record(
                                "policy_connection_error",
                                error=repr(exc),
                                execution=execution.metadata(compact=monitoring.level != "full"),
                            )
                            logging.warning("Policy connection unavailable: %s", exc)

                if protocol is not None:
                    reading_hardware_feedback = True
                    sides = (
                        ("left", "right") if args.arm_mode == "bimanual" else (args.arm_side,)
                    )
                    if protocol.schema == "joint":
                        qpos = np.concatenate(
                            [
                                read_output_qpos(
                                    pipers[side],
                                    max_feedback_age_s=args.max_feedback_age_s,
                                )
                                for side in sides
                            ]
                        ).astype(np.float32)
                        # Joint policies and their command-feedback guard do not
                        # depend on EEF feedback.  Keep the absent modality
                        # explicit instead of blocking on GetArmEndPoseMsgs().
                        delivery_state = np.empty(0, dtype=np.float32)
                    else:
                        states = {
                            side: read_output_state(
                                pipers[side],
                                max_feedback_age_s=args.max_feedback_age_s,
                            )
                            for side in sides
                        }
                        delivery_state = np.concatenate(
                            [states[side][0] for side in sides]
                        ).astype(np.float32)
                        qpos = np.concatenate(
                            [states[side][1] for side in sides]
                        ).astype(np.float32)
                    observation_captured_at = time.time()
                    observation_captured_monotonic = time.monotonic()
                    feedback_snapshot = PiperFeedbackSnapshot(
                        qpos_m=qpos,
                        delivery_state=delivery_state,
                        statuses={
                            side: arm_status_dict(pipers[side]) for side in sides
                        },
                        driver_statuses={
                            side: driver_enable_status_dict(pipers[side])
                            for side in sides
                            if side in execution.robot_enabled
                            or side in execution.arm_hold_targets
                        },
                        captured_at=observation_captured_at,
                    )
                    reading_hardware_feedback = False

                    completion = worker.poll()
                    if completion is not None:
                        handle_inference_completion(completion)

                    # Install the just-accepted plan BEFORE reserving the next
                    # request's RTC prefix. When inference exceeds its launch
                    # period, doing this in reverse pins the old plan for yet
                    # another delay window (e.g. 7 steps / 350 ms).
                    # Feedback/camera timestamps remain anchored to the capture
                    # above; planning time is included in the request latency.
                    # Recheck immediately before the only robot command path.
                    if runtime_device_guard is not None and runtime_device_guard.fault is not None:
                        raise RuntimeHardwareFault(runtime_device_guard.fault)
                    try:
                        cameras.assert_background_healthy(
                            max_frame_age_s=max(0.5, 3.0 / float(args.camera_fps))
                        )
                    except RuntimeError as exc:
                        raise RuntimeHardwareFault(f"camera stream unavailable: {exc}") from exc
                    command_sent = execution.execute_next(
                        delivery_state,
                        qpos,
                        protocol,
                        feedback_captured_at=observation_captured_at,
                        feedback_snapshot=feedback_snapshot,
                    )

                    if execution.tracking_lag_active and not worker.in_flight:
                        # The lag guard cleared the old suffix.  Retry immediately
                        # rather than waiting for the normal 4 Hz launch cadence.
                        launch_schedule.next_at = min(
                            launch_schedule.next_at, tick_started
                        )

                    launch_candidate: tuple[
                        InferenceLaunch, Callable[[], InferenceWorkerResult]
                    ] | None = None
                    # Asynchronous launches can follow either wall-clock slots
                    # or a successfully published source row in each chunk.
                    # Synchronous mode instead launches only once the active
                    # chunk (truncated to ``sync_wait_steps`` on accept) has
                    # fully drained, so capture-to-result latency is never
                    # contaminated by an overlapping in-flight request. Within
                    # synchronous mode, manual stepping additionally requires a
                    # fresh value in the GUI-written trigger file before that
                    # drained state is allowed to launch -- the ``and`` short
                    # circuits so an early click made before the queue drains
                    # is not silently consumed and lost.
                    if (not execution.native_inference_ready()
                            or execution.ik_recovery_fault or execution.ik_recovery_target is not None
                            or (execution.tracking_lag_active
                                and execution.tracking_lag_required_after_monotonic is None)):
                        launch_due = False
                    elif execution.async_inference:
                        launch_due = execution.async_launch_due(
                            launch_schedule, tick_started, in_flight=worker.in_flight
                        )
                    else:
                        queue_drained = not execution.pending_action_count
                        # ``launch_schedule`` still applies as an upper bound
                        # on retry rate. Without it, a server that rejects
                        # requests *faster* than it fulfills them would retry as fast as the round
                        # trip allows, with no floor -- the same failure mode
                        # async mode's periodic schedule already prevents. A
                        # legitimate accepted chunk's natural drain time is
                        # normally well above one schedule period, so this
                        # floor rarely binds outside of a rejection loop.
                        rate_limited = queue_drained and launch_schedule.due(tick_started)
                        launch_due = (
                            rate_limited
                            if execution.auto_next_inference
                            else rate_limited and manual_trigger.consume()
                        )
                    if launch_due:
                        if worker.in_flight:
                            # Chunk-step mode waits for the current request;
                            # periodic mode records its coalesced slot above.
                            pass
                        elif policy is not None:
                            camera_selection_started_at = time.time()
                            camera_selection_started_monotonic = time.monotonic()
                            frame_set: CameraFrameSet | None = None
                            try:
                                try:
                                    frame_set = cameras.read_nearest(
                                        observation_captured_monotonic,
                                        copy=False,
                                    )
                                except RuntimeError as exc:
                                    raise RuntimeHardwareFault(
                                        f"camera observation unavailable: {exc}"
                                    ) from exc
                                camera_selected_monotonic = time.monotonic()
                                generation = execution.allocate_inference_generation()
                                rtc_snapshot = execution.rtc_request_metadata(
                                    protocol, observation_monotonic=observation_captured_monotonic)
                                rtc_prefix_finished_monotonic = time.monotonic()
                                if bool(getattr(args, "send_policy_telemetry", True)):
                                    execution_snapshot = execution.metadata(
                                        rtc_metadata=rtc_snapshot,
                                        compact=True,
                                    )
                                    execution_snapshot["trajectory_jitter"] = recorder.jitter_snapshot
                                else:
                                    # Local monitoring samples execution state
                                    # separately. Avoid constructing/freezing
                                    # it on the control thread for this request.
                                    execution_snapshot = {}
                                metadata_finished_monotonic = time.monotonic()
                                snapshot = make_observation_snapshot(
                                    generation=generation,
                                    raw_delivery_state=delivery_state,
                                    qpos_m=qpos,
                                    protocol=protocol,
                                    captured_at=observation_captured_at,
                                    captured_monotonic=observation_captured_monotonic,
                                    frame_set=frame_set,
                                    rtc_metadata=rtc_snapshot,
                                    execution_metadata=execution_snapshot,
                                    executed_plan_command_count=(
                                        execution.executed_plan_command_count
                                    ),
                                    max_image_state_skew_s=float(
                                        getattr(
                                            args,
                                            "max_image_state_skew_s",
                                            DEFAULT_MAX_IMAGE_STATE_SKEW_S,
                                        )
                                    ),
                                )
                                camera_selection_finished_at = time.time()
                                camera_selection_finished_monotonic = time.monotonic()
                                launch = InferenceLaunch(
                                    generation=generation,
                                    captured_at=snapshot.captured_at,
                                    captured_monotonic=snapshot.captured_monotonic,
                                    launched_at=camera_selection_finished_at,
                                    launched_monotonic=camera_selection_finished_monotonic,
                                    # The snapshot is immutable and already owns
                                    # these arrays; avoid a second hot-path copy.
                                    raw_delivery_state=snapshot.raw_delivery_state,
                                    qpos_m=snapshot.qpos_m,
                                    image_timestamps={
                                        key: float(value)
                                        for key, value in snapshot.image_timestamps.items()
                                    },
                                    executed_plan_command_count=(
                                        snapshot.executed_plan_command_count
                                    ),
                                    observation_snapshot=snapshot,
                                )

                                preparation_timing = {
                                    "feedback_to_snapshot_start_ms": (camera_selection_started_monotonic - observation_captured_monotonic) * 1000,
                                    "camera_select_ms": (camera_selected_monotonic - camera_selection_started_monotonic) * 1000,
                                    "rtc_prefix_build_ms": (rtc_prefix_finished_monotonic - camera_selected_monotonic) * 1000,
                                    "execution_metadata_ms": (metadata_finished_monotonic - rtc_prefix_finished_monotonic) * 1000,
                                    "snapshot_freeze_ms": (camera_selection_finished_monotonic - metadata_finished_monotonic) * 1000,
                                }

                                def infer_snapshot(
                                    *,
                                    policy_ref=policy,
                                    protocol_ref=protocol,
                                    snapshot_ref=snapshot,
                                    launch_ref=launch,
                                    preparation_timing_ref=preparation_timing,
                                    camera_started_at=camera_selection_started_at,
                                    camera_started_monotonic=(
                                        camera_selection_started_monotonic
                                    ),
                                    camera_finished_at=camera_selection_finished_at,
                                    camera_finished_monotonic=(
                                        camera_selection_finished_monotonic
                                    ),
                                ) -> InferenceWorkerResult:
                                    nonlocal prompt_sent_revision
                                    worker_started_monotonic = time.monotonic()
                                    send_prompt = prompt_sent_revision != prompt_revision
                                    observation = build_observation(
                                        snapshot=snapshot_ref,
                                        protocol=protocol_ref,
                                        instruction=args.instruction if send_prompt else None,
                                        source_name=source_name,
                                        args=args,
                                    )
                                    request_sent_at = time.time()
                                    request_sent_monotonic = time.monotonic()
                                    observation["client_metadata"].update(
                                        request_metadata_for_inference(
                                            prompt_revision=prompt_revision,
                                            request_sent_at=request_sent_at,
                                            inference_generation=launch_ref.generation,
                                            send_policy_telemetry=bool(getattr(
                                                args, "send_policy_telemetry", True
                                            )),
                                            camera_capture_started_at=camera_started_at,
                                            camera_capture_finished_at=camera_finished_at,
                                            camera_selection_started_monotonic=camera_started_monotonic,
                                            camera_selection_finished_monotonic=camera_finished_monotonic,
                                        )
                                    )
                                    result = dict(policy_ref.infer(observation))
                                    if send_prompt:
                                        prompt_sent_revision = prompt_revision
                                    response_received_at = time.time()
                                    response_received_monotonic = time.monotonic()
                                    server_timing = result.get("transport_timing")
                                    server_timing = (
                                        server_timing
                                        if isinstance(server_timing, dict)
                                        else {}
                                    )
                                    client_timing = build_client_transport_timing(
                                        request_sent_at=request_sent_at,
                                        request_sent_monotonic=request_sent_monotonic,
                                        response_received_at=response_received_at,
                                        response_received_monotonic=(
                                            response_received_monotonic
                                        ),
                                        server_timing=server_timing,
                                        client_transport_timing=(
                                            result.get("client_transport_timing")
                                            if isinstance(
                                                result.get("client_transport_timing"),
                                                dict,
                                            )
                                            else result.get("_client_transport_timing")
                                        ),
                                        camera_capture_ms=(
                                            camera_finished_monotonic
                                            - camera_started_monotonic
                                        )
                                        * 1000.0,
                                        inference_generation=launch_ref.generation,
                                    )
                                    client_timing.update(preparation_timing_ref)
                                    client_timing.update(
                                        request_worker_queue_ms=(worker_started_monotonic - launch_ref.launched_monotonic) * 1000,
                                        request_payload_build_ms=(request_sent_monotonic - worker_started_monotonic) * 1000,
                                        capture_to_rpc_ms=(request_sent_monotonic - snapshot_ref.captured_monotonic) * 1000,
                                    )
                                    return InferenceWorkerResult(
                                        result=result,
                                        image_timestamps={
                                            key: float(value)
                                            for key, value in snapshot_ref.image_timestamps.items()
                                        },
                                        client_timing=client_timing,
                                    )

                                launch_candidate = (launch, infer_snapshot)
                            except RuntimeHardwareFault:
                                raise
                            except Exception as exc:
                                # A failed snapshot (including a deferred RTC
                                # prefix) never sends its configuration revision.
                                execution._rtc_config_sent_revision = None
                                execution.record_launch_deferred()
                                launch_schedule.next_at = min(
                                    launch_schedule.next_at,
                                    tick_started + control_period,
                                )
                                monitoring.record(
                                    "observation_snapshot_error",
                                    captured_at=observation_captured_at,
                                    captured_monotonic=observation_captured_monotonic,
                                    camera_state_age_ms=(
                                        {
                                            key: round(
                                                (observation_captured_monotonic - float(stamp)) * 1000.0,
                                                2,
                                            )
                                            for key, stamp in frame_set.monotonic_timestamps.items()
                                        }
                                        if frame_set is not None else None
                                    ),
                                    error=repr(exc),
                                    execution=execution.metadata(compact=monitoring.level != "full"),
                                )
                                logging.warning(
                                    "Inference snapshot skipped: %s", exc
                                )

                    # Claim the periodic slot before constructing the
                    # payload.  Otherwise Python evaluates the expensive
                    # ``execution.metadata(...)`` argument on every 20 Hz
                    # tick even when the recorder is configured for 1 Hz.
                    if monitoring.periodic_due():
                        monitoring.record(
                            "control_tick",
                            command_sent=command_sent,
                            control_tick_count=execution.control_tick_count,
                            captured_at=observation_captured_at,
                            captured_monotonic=observation_captured_monotonic,
                            raw_delivery_state=delivery_state,
                            qpos_m=qpos,
                            execution=execution.metadata(
                                compact=monitoring.level != "full"
                            ),
                            trajectory_jitter=recorder.jitter_snapshot,
                            model_trajectory_jitter=recorder.model_jitter_snapshot,
                        )
                    if command_sent:
                        command_count += 1
                    command_target = execution.current_timed_target_action
                    try:
                        recorder.record_control_tick(
                            timestamp=observation_captured_at,
                            monotonic_timestamp=observation_captured_monotonic,
                            delivery_state=delivery_state,
                            qpos=qpos,
                            command_sent=command_sent,
                            action_dim=protocol.action_dim,
                            absolute_dim=(10 if protocol.schema == "delivery" else 7)
                            * (2 if protocol.arm_mode == "bimanual" else 1),
                            command_action=(
                                None
                                if not command_sent or execution.last_wire_action is None
                                else np.asarray(execution.last_wire_action, dtype=np.float32)
                            ),
                            command_absolute_target=(
                                None
                                if not command_sent or command_target is None
                                else np.asarray(command_target.absolute_target, dtype=np.float32)
                            ),
                            command_joints_rad=(
                                execution.last_commanded_joint_positions
                                if command_sent else None
                            ),
                            command_monotonic_timestamp=(
                                float(execution.last_actuator_command["command_monotonic"])
                                if command_sent and execution.last_actuator_command is not None
                                else None
                            ),
                            command_generation=(
                                None
                                if not command_sent or command_target is None
                                else command_target.generation
                            ),
                            command_queue_index=(
                                None if not command_sent else execution.last_queued_action_index
                            ),
                            command_hold=(
                                False
                                if not command_sent or command_target is None
                                else command_target.hold
                            ),
                            execution_state=execution.state,
                            blocked_reason=execution.blocked_reason,
                            pose_diagnostics=(
                                {side: {"pose_trajectory": trace["pose_trajectory"],
                                        "ik_diagnostics": trace.get("ik_diagnostics")}
                                 for side, trace in execution.last_actuator_command["sides"].items()
                                 if trace.get("pose_trajectory") is not None or trace.get("ik_diagnostics") is not None}
                                if command_sent and execution.last_actuator_command is not None
                                else None
                            ),
                        )
                    except Exception:
                        # Recording must never turn a safety-checked control tick
                        # into a robot-control failure.
                        logging.exception("Failed to record control tick")
                    if (
                        args.once
                        and once_result_accepted
                        and (not args.allow_execution or command_sent)
                    ):
                        return
                    if command_sent and args.max_commands is not None and command_count >= args.max_commands:
                        logging.warning(
                            "Reached --max-commands=%d; stopping after the checked command.",
                            args.max_commands,
                        )
                        return

                    if launch_candidate is not None:
                        launch, inference_task = launch_candidate
                        lag_barrier = execution.tracking_lag_required_after_monotonic
                        stale_for_lag_guard = execution.tracking_lag_active and (
                            lag_barrier is None
                            or float(launch.captured_monotonic) <= lag_barrier
                        )
                        if stale_for_lag_guard:
                            # This snapshot was frozen before execute_next() saw
                            # the third lagging feedback cycle.  Do not waste one
                            # serialized inference slot on a result that the
                            # post-trigger acceptance barrier must reject.
                            execution.record_launch_deferred()
                            launch_schedule.next_at = min(
                                launch_schedule.next_at,
                                tick_started + control_period,
                            )
                            monitoring.record(
                                "inference_launch_suppressed",
                                generation=launch.generation,
                                captured_monotonic=launch.captured_monotonic,
                                reason="snapshot predates tracking-lag trigger",
                                execution=execution.metadata(compact=monitoring.level != "full"),
                            )
                        elif worker.launch_callable(inference_task, launch):
                            execution.record_inference_launch(launch)
                            if execution.async_inference and execution.inference_trigger_mode == "periodic":
                                launch_schedule.mark_launched(tick_started)
                            elif execution.async_inference and execution.inference_trigger_mode == "chunk_step":
                                # The periodic clock is only a bounded retry
                                # interval for bootstrap, failure, and underrun.
                                launch_schedule.next_at = tick_started + launch_schedule.period_s
                        else:  # defensive; the control thread owns launch()
                            execution.record_launch_deferred()
                            if execution.async_inference and execution.inference_trigger_mode == "periodic":
                                launch_schedule.pending_latest = True

                # Launch retries happen on the next tick; no synchronous infer call.
            except (PiperFeedbackStaleError, RuntimeHardwareFault) as exc:
                hardware_fault = str(exc)
                execution.discard_pending_actions(hardware_fault, kind="hardware_fault")
                execution._block("blocked", hardware_fault)
                monitoring.record("runtime_hardware_fault", reason=hardware_fault)
                logging.error("Runtime hardware fault; ending execution: %s", hardware_fault)
                break
            except ExecutionBlocked as exc:
                execution._block("blocked", str(exc))
                monitoring.record(
                    "control_tick_blocked",
                    error=repr(exc),
                    execution=execution.metadata(compact=monitoring.level != "full"),
                )
                logging.warning("20 Hz feedback/safety check blocked: %s", exc)
            except Exception as exc:
                if reading_hardware_feedback:
                    hardware_fault = "Piper feedback read failed"
                    execution.discard_pending_actions(hardware_fault, kind="hardware_fault")
                    execution._block("blocked", hardware_fault)
                    monitoring.record("runtime_hardware_fault", reason=hardware_fault)
                    logging.error("Runtime hardware fault; ending execution: %s", hardware_fault)
                    break
                execution._block("blocked", f"control tick failed: {exc}")
                monitoring.record(
                    "control_tick_error",
                    error=repr(exc),
                    execution=execution.metadata(compact=monitoring.level != "full"),
                )
                logging.exception("20 Hz control tick failed")

            # A new plan may replace the active Ruckig reference between
            # regular 20 Hz ticks. Keep the original model clock phase; the
            # independent servo still owns every CAN publication at servo-hz.
            if not reference_refresh_tick:
                next_control_at += control_period
            now = time.monotonic()
            if now > next_control_at:
                missed = int(math.floor((now - next_control_at) / control_period)) + 1
                next_control_at += missed * control_period
            try:
                wake_at = next_control_at
                if execution.async_inference and execution.inference_trigger_mode == "periodic":
                    wake_at = launch_schedule.wakeup_deadline(
                        next_control_at, time.monotonic(), in_flight=worker.in_flight)
                reference_refresh_tick = worker.wait_until(wake_at, handle_inference_completion)
                # A request deadline is an extra fresh-observation tick, just
                # like a result wakeup; preserve the regular control clock.
                reference_refresh_tick = reference_refresh_tick or wake_at < next_control_at
            except Exception as exc:
                reference_refresh_tick = False
                execution._block("blocked", f"inference completion failed: {exc}")
                logging.exception("Inference completion failed between control ticks")
    except KeyboardInterrupt:
        logging.info("Stopping policy references; braking the active servo before shutdown.")
    finally:
        servo_stopped_safely = execution.close_servo()
        execution.close_ik_workers()
        if not servo_stopped_safely and hardware_fault is None:
            hardware_fault = "Servo stop could not be verified; automatic return suppressed"
        if initial_qpos is not None:
            save_return_pose(args, initial_qpos,
                             safe_to_return=servo_stopped_safely and hardware_fault is None)
        if not servo_stopped_safely:
            logging.error("Servo did not settle safely; automatic return is suppressed")
        if runtime_device_guard is not None:
            runtime_device_guard.close()
        worker.shutdown()
        try:
            recorder.stop(reason="client_stopped")
        except Exception:
            logging.exception("Failed to finalize deployment recording")
        close_policy(policy)
        preview.close()
        cameras.close()
        if (
            bool(getattr(args, "auto_return", True))
            and hardware_fault is None
            and initial_qpos is not None
            and command_count > 0
            and servo_stopped_safely
        ):
            return_attempted = True
            try:
                execution.prepare_native_return_handoff()
                return_summary = return_pipers_to_initial(
                    pipers,
                    initial_qpos,
                    arm_mode=args.arm_mode,
                    servo_backend=execution.servo_backend,
                    on_sample=lambda sample: monitoring.record("return_to_initial_sample", **sample),
                    servo_hz=float(getattr(args, "servo_hz", DEFAULT_SERVO_HZ)),
                    speed_pct=int(
                        getattr(args, "return_speed_pct", None) or args.speed_pct
                    ),
                    hz=float(getattr(args, "return_hz", DEFAULT_RETURN_HZ)),
                    timeout_s=float(
                        getattr(args, "return_timeout_s", DEFAULT_RETURN_TIMEOUT_S)
                    ),
                    max_feedback_age_s=float(args.max_feedback_age_s),
                    max_gripper_speed_m_s=0.08,
                    max_joint_speed_rad_s=float(
                        getattr(
                            args,
                            "trajectory_max_speed_rad_s",
                            DEFAULT_TRAJECTORY_MAX_SPEED_RAD_S,
                        )
                    ),
                    max_joint_acceleration_rad_s2=float(
                        getattr(
                            args,
                            "trajectory_max_acceleration_rad_s2",
                            DEFAULT_TRAJECTORY_MAX_ACCELERATION_RAD_S2,
                        )
                    ),
                    max_joint_jerk_rad_s3=float(
                        getattr(
                            args,
                            "trajectory_max_jerk_rad_s3",
                            DEFAULT_TRAJECTORY_MAX_JERK_RAD_S3,
                        )
                    ),
                    smoothing_cutoff_hz=float(
                        getattr(
                            args,
                            "trajectory_smoothing_cutoff_hz",
                            DEFAULT_TRAJECTORY_SMOOTHING_CUTOFF_HZ,
                        )
                    ),
                    tracking_time_constant_s=float(
                        getattr(
                            args,
                            "trajectory_tracking_time_constant_s",
                            DEFAULT_TRAJECTORY_TRACKING_TIME_CONSTANT_S,
                        )
                    ),
                    command_lookahead_rad=float(
                        getattr(
                            args,
                            "trajectory_command_lookahead_rad",
                            DEFAULT_TRAJECTORY_COMMAND_LOOKAHEAD_RAD,
                        )
                    ),
                    max_tracking_error_rad=float(
                        getattr(
                            args,
                            "trajectory_max_tracking_error_rad",
                            DEFAULT_TRAJECTORY_MAX_TRACKING_ERROR_RAD,
                        )
                    ),
                    hold_tolerance_rad=float(
                        getattr(args, "arm_hold_tolerance_rad", DEFAULT_ARM_HOLD_TOLERANCE_RAD)
                    ),
                )
                returned_to_initial = bool(return_summary.get("returned"))
                logging.info("Return to initial pose completed: %s", return_summary)
                monitoring.record("return_to_initial_completed", **return_summary)
            except Exception as exc:
                monitoring.record(
                    "return_to_initial_failed",
                    error=repr(exc),
                )
                logging.exception("Failed to return Piper to initial pose")
        for piper in pipers.values():
            try:
                piper.DisconnectPort()
            except Exception:
                logging.exception("Failed to disconnect Piper")
        monitoring.record(
            "deployment_finished",
            command_count=command_count,
            return_attempted=return_attempted,
            returned_to_initial=returned_to_initial,
            servo_stopped_safely=bool(servo_stopped_safely),
            hardware_fault=hardware_fault,
            trajectory_jitter=recorder.jitter_snapshot,
            model_trajectory_jitter=recorder.model_jitter_snapshot,
        )
        monitoring.close(reason="stopped")
        if return_attempted and not returned_to_initial:
            raise ExecutionBlocked("return to initial pose failed; see return diagnostics")
    if hardware_fault is not None:
        raise DeviceCheckRejected(hardware_fault)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rlsok-resolver",
        default=os.environ.get("BIMANUAL_VLA_RLSOK_RESOLVER", ""),
        help="project-side RLSOK v1.5.8 resolver executable; checked before CAN/camera open",
    )
    parser.add_argument("--host", default=os.environ.get("BIMANUAL_VLA_POLICY_HOST", DEFAULT_POLICY_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BIMANUAL_VLA_POLICY_PORT", DEFAULT_POLICY_PORT)))
    parser.add_argument("--arm-mode", choices=("single", "bimanual"), default="single")
    parser.add_argument(
        "--output-mode",
        "--policy-schema",
        dest="output_mode",
        choices=("auto", "joint", "delivery"),
        default="auto",
        help=(
            "policy output contract: auto follows server metadata; joint requires "
            "7D/14D joint targets; delivery requires EEF delivery targets. "
            "An explicit mode fails closed if the server advertises another schema."
        ),
    )
    parser.add_argument("--can", default=DEFAULT_CAN, help="single-arm CAN interface")
    parser.add_argument("--left-can", default=DEFAULT_LEFT_CAN)
    parser.add_argument("--right-can", default=DEFAULT_RIGHT_CAN)
    parser.add_argument("--arm-side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--cam-high-device", default=DEFAULT_HIGH_DEVICE)
    parser.add_argument("--cam-wrist-device", default=DEFAULT_WRIST_DEVICE, help="single-arm wrist camera")
    parser.add_argument(
        "--camera-preview",
        "--show-cameras",
        dest="camera_preview",
        action="store_true",
        help="show a low-resolution native-aspect live preview of the camera feeds",
    )
    parser.add_argument(
        "--camera-preview-fps",
        type=float,
        default=8.0,
        help="preview refresh rate; capture/inference rates are unchanged (default 8 FPS)",
    )
    parser.add_argument("--cam-left-wrist-device", default=DEFAULT_LEFT_WRIST_DEVICE)
    parser.add_argument("--cam-right-wrist-device", default=DEFAULT_RIGHT_WRIST_DEVICE)
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="camera acquisition rate (default 30 Hz; independent of inference launches)",
    )
    parser.add_argument(
        "--preresize-policy-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "precompute the server's exact PIL 256-to-224 resize on the camera "
            "thread for policy requests; recorded frames remain 256 pixels"
        ),
    )
    parser.add_argument(
        "--max-image-state-skew-ms",
        type=float,
        default=DEFAULT_MAX_IMAGE_STATE_SKEW_S * 1000.0,
        help=(
            "maximum allowed monotonic skew between Piper feedback and the "
            "nearest buffered multi-camera frame set"
        ),
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=DEFAULT_INFERENCE_HZ,
        help=(
            "periodic asynchronous launch frequency, or recovery retry rate in chunk_step mode "
            "(default 4 Hz / every 250 ms); "
            "robot control remains independently configured by --control-hz"
        ),
    )
    parser.add_argument(
        "--inference-trigger-mode",
        choices=("periodic", "chunk_step"),
        default="periodic",
        help="asynchronous request trigger: wall-clock schedule or executed chunk source step",
    )
    parser.add_argument(
        "--inference-trigger-step",
        type=int,
        default=DEFAULT_INFERENCE_TRIGGER_STEP,
        help="1-based chunk source step to request the next chunk in chunk_step mode (default 10)",
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
        "--rtc-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use model-side Real-Time Chunking when the Policy advertises it (default: enabled)",
    )
    parser.add_argument("--rtc-inference-delay-steps", type=int, default=8,
                        help="fixed model-side RTC delay prefix; never estimated from latency")
    parser.add_argument("--rtc-execution-horizon", type=int, default=DEFAULT_RTC_EXECUTION_HORIZON)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=DEFAULT_RTC_MAX_GUIDANCE_WEIGHT)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=("zeros", "ones", "linear", "exp"),
        default="linear",
    )
    parser.add_argument(
        "--rtc-client-blend-steps",
        type=int,
        default=0,
        choices=(0, 2, 3, 4),
        help="deprecated compatibility option; ignored (RTC client blending is disabled)",
    )
    parser.add_argument(
        "--async-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="asynchronous inference launch (default); --no-async-inference waits for chunk execution",
    )
    parser.add_argument(
        "--sync-wait-steps",
        type=int,
        default=8,
        help="steps to wait before next inference when async is disabled (default 8)",
    )
    parser.add_argument(
        "--auto-next-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "when async is disabled, automatically launch the next inference "
            "once the truncated chunk drains (default); --no-auto-next-inference "
            "instead holds and waits for --manual-trigger-file to change"
        ),
    )
    parser.add_argument(
        "--manual-trigger-file",
        type=str,
        default=None,
        help=(
            "path polled once per control tick for a manual next-inference "
            "request; a launch fires once the file's contents change. "
            "Required when --no-auto-next-inference is set"
        ),
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
        choices=(0, 2, 3, 4),
        default=DEFAULT_BLEND_STEPS,
        help="pose/joint old/new blend length; 0 disables client blending (default 3)",
    )
    parser.add_argument(
        "--blend-profile",
        choices=("linear", "smootherstep"),
        default="smootherstep",
        help="boundary interpolation profile; smootherstep has zero slope at both ends",
    )
    parser.add_argument("--servo-backend", choices=("off", "ruckig", "legacy"), default=None,
                        help="joint reference planner backend (not motor servo); default Ruckig unless shaping is disabled")
    parser.add_argument("--eef-control-mode", choices=("local_ik_joint", "piper_eef"), default="local_ik_joint",
                        help="delivery execution: local IK + JointCtrl, or firmware IK via MOVE P + EndPoseCtrl")
    parser.add_argument("--servo-hz", type=float, default=DEFAULT_SERVO_HZ,
                        help="actuator publication frequency, independent of policy action Hz")
    parser.add_argument("--servo-target-timeout-s", type=float, default=0.15)
    parser.add_argument("--servo-waypoint-velocity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pose-trajectory", action=argparse.BooleanOptionalAction, default=False,
                        help="host EEF target shaping; local IK requires Ruckig; normal RTC tracking bypasses it")
    for name, default in vars(PoseTrajectoryConfig()).items():
        parser.add_argument("--pose-" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument(
        "--trajectory-shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply velocity/acceleration/jerk-limited joint trajectory shaping",
    )
    parser.add_argument(
        "--trajectory-spike-suppression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "suppress isolated joint acceleration spikes inside the short RTC "
            "execution horizon (default: disabled)"
        ),
    )
    for option, description in (
        ("trajectory-lowpass", "joint reference low-pass filter"),
        ("trajectory-tracking", "second-order joint reference tracking"),
        ("trajectory-speed-limit", "joint shaper speed limit"),
        ("trajectory-acceleration-limit", "joint shaper acceleration limit"),
        ("trajectory-jerk-limit", "joint shaper jerk limit"),
        ("trajectory-lookahead", "velocity-based joint command lookahead"),
    ):
        parser.add_argument(
            f"--{option}", action=argparse.BooleanOptionalAction,
            default=True, help=f"enable {description} (default: enabled)",
        )
    parser.add_argument(
        "--trajectory-max-speed-rad-s",
        type=float,
        default=DEFAULT_TRAJECTORY_MAX_SPEED_RAD_S,
    )
    parser.add_argument(
        "--trajectory-max-acceleration-rad-s2",
        type=float,
        default=DEFAULT_TRAJECTORY_MAX_ACCELERATION_RAD_S2,
    )
    parser.add_argument(
        "--trajectory-max-jerk-rad-s3",
        type=float,
        default=DEFAULT_TRAJECTORY_MAX_JERK_RAD_S3,
    )
    parser.add_argument(
        "--trajectory-smoothing-cutoff-hz",
        type=float,
        default=DEFAULT_TRAJECTORY_SMOOTHING_CUTOFF_HZ,
    )
    parser.add_argument(
        "--trajectory-tracking-time-constant-s",
        type=float,
        default=DEFAULT_TRAJECTORY_TRACKING_TIME_CONSTANT_S,
    )
    parser.add_argument(
        "--trajectory-command-lookahead-rad",
        type=float,
        default=DEFAULT_TRAJECTORY_COMMAND_LOOKAHEAD_RAD,
    )
    parser.add_argument(
        "--trajectory-max-tracking-error-rad",
        type=float,
        default=DEFAULT_TRAJECTORY_MAX_TRACKING_ERROR_RAD,
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
        "--tracking-lag-threshold-rad",
        type=float,
        default=DEFAULT_TRACKING_LAG_THRESHOLD_RAD,
        help=(
            "freeze the timed suffix after sustained sent/reference-vs-measured joint "
            "error exceeds this infinity-norm threshold"
        ),
    )
    parser.add_argument(
        "--tracking-lag-confirm-cycles",
        type=int,
        default=DEFAULT_TRACKING_LAG_CONFIRM_CYCLES,
        help="consecutive high-error feedback cycles required to trigger the lag guard",
    )
    parser.add_argument("--tracking-lag-position-threshold-m", type=float, default=.05,
                        help="sustained original policy EEF target vs feedback position error")
    parser.add_argument("--tracking-lag-rotation-threshold-rad", type=float, default=.18,
                        help="sustained original policy EEF target vs feedback rotation error")
    parser.add_argument("--tracking-reference-lag-duration-s", type=float, default=.5,
                        help="continuous policy/reference mismatch duration before recovery; sent-joint guard remains immediate after confirm cycles")
    parser.add_argument("--eef-goal-position-tolerance-m", type=float, default=.005,
                        help="measured original-goal position tolerance before gripper release")
    parser.add_argument("--eef-goal-rotation-tolerance-rad", type=float, default=.05,
                        help="measured original-goal rotation tolerance before gripper release")
    parser.add_argument("--eef-goal-timeout-s", type=float, default=2.,
                        help="total original-goal approach and gripper-publication deadline")
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
        "--gripper-lowpass", action=argparse.BooleanOptionalAction,
        default=True, help="enable independent gripper low-pass filtering",
    )
    parser.add_argument(
        "--gripper-endpoint-filter", action=argparse.BooleanOptionalAction,
        default=True, help="enable gripper endpoint hysteresis and confirmation",
    )
    parser.add_argument(
        "--ik-rate-limit", action=argparse.BooleanOptionalAction,
        default=True, help="limit each delivery IK command to --ik-max-joint-step-rad",
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
    parser.add_argument(
        "--gripper-open-lookahead-steps",
        type=int,
        default=DEFAULT_GRIPPER_OPEN_LOOKAHEAD_STEPS,
        help="advance opening requests using a future action window; closing is never anticipated",
    )
    parser.add_argument("--instruction", default="pick up the cube")
    parser.add_argument("--source-name", default=None)
    parser.add_argument(
        "--monitoring-dir",
        default=os.environ.get("BIMANUAL_VLA_MONITORING_DIR", DEFAULT_MONITORING_DIR),
        help="local root for per-run monitoring_data/<session>/events.jsonl",
    )
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="disable JSONL monitoring telemetry without disabling deployment recording",
    )
    parser.add_argument(
        "--monitoring-rate",
        type=float,
        default=1.0,
        help="maximum periodic control telemetry rate in Hz (default 1 Hz)",
    )
    parser.add_argument(
        "--monitoring-level",
        choices=("compact", "full"),
        default="compact",
        help="periodic telemetry payload size; safety/error events remain detailed",
    )
    parser.add_argument(
        "--send-policy-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "include detailed client execution and device diagnostics in the main "
            "policy request (default: enabled); disable for transport A/B tests "
            "without disabling local monitoring or RTC"
        ),
    )
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="run one successful inference and exit")
    parser.add_argument(
        "--allow-execution",
        action="store_true",
        help="allow local robot execution, subject to per-command safety checks",
    )
    parser.add_argument("--max-action-age-s", type=float, default=2.0)
    parser.add_argument(
        "--ik-worker-processes", action=argparse.BooleanOptionalAction, default=True,
        help="solve EEF IK in persistent per-arm processes (default enabled; no hardware access)",
    )
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
        "--joint-limit-tolerance-rad",
        type=float,
        default=DEFAULT_JOINT_LIMIT_TOLERANCE_RAD,
        help="project model target overshoot within this margin onto joint limits before planning (default 0.10 rad)",
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
        help="maximum commanded per-joint change on one 20 Hz delivery control tick",
    )
    parser.add_argument(
        "--ik-search-joint-radius-rad",
        type=float,
        default=DEFAULT_IK_SEARCH_JOINT_RADIUS_RAD,
        help="near-current joint radius used to find the regularized EEF IK direction",
    )
    parser.add_argument(
        "--ik-reference-joint-radius-rad", type=float, default=DEFAULT_IK_REFERENCE_JOINT_RADIUS_RAD,
        help="IK branch search radius around the previous accepted solution; no post-solve rate clipping",
    )
    parser.add_argument(
        "--ik-minimum-motion", action=argparse.BooleanOptionalAction, default=True,
        help="prefer the nearest joint solution inside the IK pose tolerances",
    )
    parser.add_argument(
        "--ik-independent-arms", action=argparse.BooleanOptionalAction, default=True,
        help="local EEF IK: plan each arm independently on the shared servo clock",
    )
    parser.add_argument(
        "--ik-joint-regularization-weight",
        type=float,
        default=DEFAULT_IK_JOINT_REGULARIZATION_WEIGHT,
        help="penalty on unnecessary joint motion in delivery IK (0 disables it)",
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
        "--record-root",
        default=os.environ.get("BIMANUAL_VLA_RECORD_ROOT", "deployment_runs"),
        help="directory where one trajectory/model/video folder is created per run",
    )
    parser.add_argument(
        "--record-video-fps",
        type=float,
        default=None,
        help="nominal FPS for recorded MP4; default is --camera-fps",
    )
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help="disable deployment trajectory, model-command, and video recording",
    )
    parser.add_argument(
        "--arm-settle-s", type=float, default=0.75,
        help="minimum joint-hold settling time after enabling Piper",
    )
    parser.add_argument(
        "--arm-hold-tolerance-rad", type=float, default=DEFAULT_ARM_HOLD_TOLERANCE_RAD,
        help="maximum joint error before leaving the post-enable hold (default 0.05 rad)",
    )
    parser.add_argument(
        "--workspace-x", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="optional EEF x bounds; omitted means no Cartesian x envelope",
    )
    parser.add_argument(
        "--workspace-y", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="optional EEF y bounds; omitted means no Cartesian y envelope",
    )
    parser.add_argument(
        "--workspace-z", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="optional EEF z bounds; omitted means no Cartesian z envelope",
    )
    parser.add_argument(
        "--speed-pct", type=int, default=DEFAULT_INFERENCE_SPEED_PCT,
        help="Piper firmware speed percentage (default 100); trajectory speed is configured separately",
    )
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--enable-timeout-s", type=float, default=3.0)
    parser.add_argument(
        "--reject-external-control-streams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="refuse deployment when Piper reports another high-rate control stream",
    )
    parser.add_argument(
        "--external-control-hz-threshold",
        type=float,
        default=DEFAULT_EXTERNAL_CONTROL_HZ_THRESHOLD,
    )
    parser.add_argument(
        "--auto-return",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="smoothly return commanded arms to their startup pose before disconnect",
    )
    parser.add_argument("--initial-pose-file", default=None)
    parser.add_argument("--return-target-file", default=None)
    parser.add_argument("--return-only", action="store_true")
    parser.add_argument("--return-hz", type=float, default=DEFAULT_RETURN_HZ)
    parser.add_argument("--return-timeout-s", type=float, default=DEFAULT_RETURN_TIMEOUT_S)
    parser.add_argument(
        "--return-speed-pct",
        type=int,
        default=None,
        help="return speed percentage; default reuses --speed-pct",
    )
    args = parser.parse_args()
    args.servo_backend = args.servo_backend or ("ruckig" if args.trajectory_shaping else "off")
    args.trajectory_shaping = args.servo_backend != "off"
    try:
        PoseTrajectoryConfig(**{name: getattr(args, "pose_" + name)
                                for name in vars(PoseTrajectoryConfig())})
    except ValueError as exc:
        parser.error(str(exc))
    if args.pose_trajectory and args.servo_backend != "ruckig" and args.eef_control_mode != "piper_eef":
        parser.error("pose-trajectory requires servo-backend=ruckig")
    if args.pose_trajectory:
        if not math.isfinite(args.control_hz) or args.control_hz <= 0:
            parser.error("pose trajectory requires a positive finite control rate")
        if args.pose_max_update_dt_s < 1.0 / args.control_hz:
            parser.error("pose reference timeout must cover one control period")
    if args.servo_backend == "ruckig":
        try:
            ServoConfig(hz=args.servo_hz, target_timeout_s=args.servo_target_timeout_s)
        except ValueError as exc:
            parser.error(str(exc))
        if args.servo_hz < args.control_hz:
            parser.error("servo-hz must be at least control-hz")
    if not math.isfinite(float(args.monitoring_rate)) or args.monitoring_rate < 0:
        parser.error("--monitoring-rate must be finite and non-negative")
    if args.sync_wait_steps < 1:
        parser.error("--sync-wait-steps must be positive")
    if args.inference_trigger_step < 1:
        parser.error("--inference-trigger-step must be positive")
    if not args.auto_next_inference and not args.manual_trigger_file:
        parser.error("--no-auto-next-inference requires --manual-trigger-file")
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
    if args.return_speed_pct is None:
        args.return_speed_pct = args.speed_pct
    args.max_image_state_skew_s = args.max_image_state_skew_ms / 1000.0
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    if args.action_chunk_steps is not None:
        args.min_action_chunk_steps = args.action_chunk_steps
    positive = (
        args.hz,
        args.control_hz,
        args.camera_fps,
        args.camera_preview_fps,
        args.max_image_state_skew_ms,
        args.action_hz if args.action_hz is not None else 1.0,
        args.rtc_execution_horizon,
        args.rtc_max_guidance_weight,
        args.max_action_age_s,
        args.max_feedback_age_s,
        args.max_translation_step_m,
        args.max_rotation_step_rad,
        args.max_gripper_step,
        args.gripper_range_tolerance,
        args.max_joint_step_rad,
        args.tracking_lag_threshold_rad,
        args.tracking_reference_lag_duration_s,
        args.max_joint_gripper_step,
        args.ik_max_joint_step_rad,
        args.ik_search_joint_radius_rad,
        args.ik_reference_joint_radius_rad,
        args.ik_position_tolerance_m,
        args.ik_rotation_tolerance_rad,
        args.gripper_lowpass_alpha,
        args.gripper_hysteresis,
        args.trajectory_max_speed_rad_s,
        args.trajectory_max_acceleration_rad_s2,
        args.trajectory_max_jerk_rad_s3,
        args.trajectory_smoothing_cutoff_hz,
        args.trajectory_tracking_time_constant_s,
        args.trajectory_command_lookahead_rad if args.trajectory_command_lookahead_rad > 0 else 1.0,
        args.trajectory_max_tracking_error_rad,
        args.external_control_hz_threshold,
        args.return_hz,
        args.return_timeout_s,
        args.arm_settle_s,
        args.arm_hold_tolerance_rad,
        args.enable_timeout_s,
        args.record_video_fps if args.record_video_fps is not None else 1.0,
    )
    if any(value <= 0 for value in positive) or args.reconnect_delay < 0:
        parser.error("frequencies, freshness/safety limits, and timeout must be positive")
    if args.ik_search_joint_radius_rad < args.ik_max_joint_step_rad:
        parser.error("ik-search-joint-radius-rad must be at least ik-max-joint-step-rad")
    if not math.isfinite(args.ik_reference_joint_radius_rad):
        parser.error("ik-reference-joint-radius-rad must be finite")
    if args.ik_joint_regularization_weight < 0:
        parser.error("ik-joint-regularization-weight must be non-negative")
    if args.joint_limit_tolerance_rad < 0:
        parser.error("joint-limit-tolerance-rad must be non-negative")
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
    if args.trajectory_command_lookahead_rad < 0:
        parser.error("trajectory-command-lookahead-rad must be non-negative")
    if args.gripper_open_lookahead_steps < 0:
        parser.error("gripper-open-lookahead-steps must be non-negative")
    if not 1 <= args.return_speed_pct <= 100:
        parser.error("return-speed-pct must be in [1,100]")
    if args.gripper_lowpass_alpha > 1:
        parser.error("gripper-lowpass-alpha must be in (0,1]")
    if not 0 < args.gripper_hysteresis < 0.5:
        parser.error("gripper-hysteresis must be in (0,0.5)")
    if args.gripper_confirm_steps < 1:
        parser.error("gripper-confirm-steps must be positive")
    if args.tracking_lag_confirm_cycles < 1:
        parser.error("tracking-lag-confirm-cycles must be positive")
    if not 1 <= args.speed_pct <= 100:
        parser.error("speed-pct must be in [1,100]")
    if not 0 <= args.gripper_effort <= 5000:
        parser.error("gripper-effort must be in [0,5000]")
    for name in ("workspace_x", "workspace_y", "workspace_z"):
        bounds = getattr(args, name)
        if bounds is not None and bounds[0] >= bounds[1]:
            parser.error(f"{name.replace('_', '-')} MIN must be less than MAX")
    if args.arm_mode == "bimanual":
        if args.arm_side not in {"both", "right"}:
            parser.error("bimanual mode requires --arm-side both")
        args.arm_side = "both"
        try:
            resolve_can_pair(args.left_can, args.right_can)
        except ValueError as exc:
            parser.error(str(exc))
        if args.left_can == args.right_can:
            parser.error("--left-can and --right-can must differ in bimanual mode")
        explicit_cameras = [
            device for device in (
                args.cam_high_device, args.cam_left_wrist_device, args.cam_right_wrist_device
            ) if device.lower() != "auto"
        ]
        if len(set(explicit_cameras)) != len(explicit_cameras):
            parser.error("bimanual camera devices must be distinct")
    elif args.arm_side not in {"left", "right"}:
        parser.error("single mode requires --arm-side left or right")
    if not args.instruction.strip():
        parser.error("instruction must not be empty")
    args.instruction = args.instruction.strip()
    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    console_handler = NonBlockingConsoleHandler()
    root_logger.handlers = [console_handler]
    root_logger.setLevel(logging.INFO)
    try:
        with graceful_stop_signals():
            if args.return_only:
                run_return_only(args)
            else:
                run_rtc_client(args)
    except KeyboardInterrupt:
        logging.info("Client stopped during initialization or return")
    except DeviceCheckRejected as exc:
        parser.exit(2, f"{exc}\n")
    finally:
        root_logger.handlers = previous_handlers
        root_logger.setLevel(previous_level)
        console_handler.close()


# Backward-compatible import/entry-point name used by older scripts and tests.
run = run_rtc_client


if __name__ == "__main__":
    main()
