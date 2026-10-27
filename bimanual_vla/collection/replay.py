"""Explicit, guarded hardware replay of the current collector's NPZ format.

Recorded measured states are the reference path, not training action chunks.
Importing this module or inspecting a file never opens a CAN interface.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import fcntl
import json
import math
from pathlib import Path
import socket
import struct
import threading
import time
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.data.contract import (
    CONTRACT_VERSION, DEFAULT_COORDINATE_FRAME, GRIPPER_MAX_M,
    GRIPPER_OPENING_SEMANTICS, infer_episode_contract, rotation_matrix_from_state,
)
from bimanual_vla.deployment.pose_trajectory import PoseTrajectory, PoseTrajectoryConfig
from bimanual_vla.deployment.servo import RuckigJointServo, ServoConfig
from bimanual_vla.deployment.execution_defaults import (
    DEFAULT_ACTION_HZ, DEFAULT_SERVO_HZ, DEFAULT_SPEED_PCT, DEFAULT_FEEDBACK_MAX_AGE_S,
    DEFAULT_TRAJECTORY_MAX_SPEED_RAD_S, DEFAULT_TRAJECTORY_MAX_ACCELERATION_RAD_S2,
    DEFAULT_TRAJECTORY_MAX_JERK_RAD_S3, DEFAULT_TRAJECTORY_MAX_TRACKING_ERROR_RAD,
    DEFAULT_GRIPPER_SPEED_M_S,
)
from bimanual_vla.collection.replay_errors import ReplayBlocked, ReplayFeedbackPending, ReplayCancelled


@dataclass(frozen=True)
class ReplayEpisode:
    path: Path
    arm_mode: str
    sides: tuple[str, ...]
    targets: dict[str, np.ndarray]
    timestamps: dict[str, np.ndarray]

    @property
    def formats(self):
        return tuple(self.targets)


def load_replay_episode(path: str | Path) -> ReplayEpisode:
    """Load numeric fields only; reject legacy or ambiguous unit conventions."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        for key in ('contract_version', 'schema', 'arm_mode', 'arm_side', 'gripper_semantics', 'coordinate_frame'):
            if key not in data:
                raise ReplayBlocked(f'Missing current collector metadata: {key}')
        if int(data['contract_version'].item()) != CONTRACT_VERSION:
            raise ReplayBlocked('Hardware replay supports only the current v3 collector format')
        contract = infer_episode_contract(data)
        if contract.legacy_delivery_v2 or contract.legacy_joint_v2:
            raise ReplayBlocked('Legacy episodes are not supported by hardware replay')
        if str(data['gripper_semantics'].item()) != GRIPPER_OPENING_SEMANTICS:
            raise ReplayBlocked('Expected current opening-fraction gripper semantics')
        if contract.coordinate_frame != DEFAULT_COORDINATE_FRAME:
            raise ReplayBlocked('Replay requires poses in the corresponding slave_base frame')
        targets, timestamps = {}, {}
        sources = {}
        if contract.schema == 'joint':
            sources['joint'] = ('state', 'state_timestamp')
        else:
            sources['eef'] = ('state', 'state_timestamp')
            if 'joint_state' in data:
                sources['joint'] = ('joint_state', 'joint_state_timestamp')
            elif 'joint_qpos' in data:
                sources['joint'] = ('joint_qpos', 'state_timestamp')
        if 'eef_state' in data:
            sources['eef'] = ('eef_state', 'eef_state_timestamp')
        for mode, (key, ts_key) in sources.items():
            values = np.array(data[key], dtype=np.float64, copy=True)
            if ts_key not in data:
                raise ReplayBlocked(f'Missing {mode} sample clock: {ts_key}')
            clock = np.array(data[ts_key], dtype=np.float64, copy=True)
            width = 7 if mode == 'joint' else 10
            if (values.ndim != 2 or values.shape[1] != width * contract.arm_count
                    or len(values) < 2 or clock.shape != (len(values),)
                    or not np.isfinite(values).all() or not np.isfinite(clock).all()
                    or np.any(np.diff(clock) <= 0)):
                raise ReplayBlocked(f'Invalid {mode} states or sample timestamps')
            # EpisodeBuffer appends one synthetic terminal hold to each state array.
            if not np.array_equal(values[-1], values[-2]):
                raise ReplayBlocked(f'{mode} data lacks the current collector terminal hold')
            values, clock = values[:-1].copy(), clock[:-1].copy()
            openings = values[:, width - 1::width]
            if np.any((openings < 0) | (openings > 1)):
                raise ReplayBlocked(f'{mode} opening fractions must be within [0, 1]')
            openings *= GRIPPER_MAX_M
            if mode == 'eef':
                for offset in range(0, values.shape[1], 10):
                    a, b = values[:, offset + 3:offset + 6], values[:, offset + 6:offset + 9]
                    if (np.any(abs(np.linalg.norm(a, axis=1) - 1) > .005)
                            or np.any(abs(np.linalg.norm(b, axis=1) - 1) > .005)
                            or np.any(abs(np.sum(a * b, axis=1)) > .005)):
                        raise ReplayBlocked('Invalid recorded rotation6d')
            values.setflags(write=False)
            clock -= clock[0]
            clock.setflags(write=False)
            targets[mode], timestamps[mode] = values, clock
    sides = ('left', 'right') if contract.arm_mode == 'bimanual' else (contract.arm_side,)
    return ReplayEpisode(path.resolve(), contract.arm_mode, sides, targets, timestamps)


@dataclass(frozen=True)
class ReplayConfig:
    mode: str = 'joint'
    speed: float = .25
    joint_smoothing: bool = True
    eef_smoothing: bool = False
    enable_before_replay: bool = False
    auto_recover: bool = True
    control_hz: float = DEFAULT_ACTION_HZ
    hz: float = DEFAULT_SERVO_HZ
    speed_pct: int = DEFAULT_SPEED_PCT
    max_joint_speed: float = DEFAULT_TRAJECTORY_MAX_SPEED_RAD_S
    max_joint_acceleration: float = DEFAULT_TRAJECTORY_MAX_ACCELERATION_RAD_S2
    max_joint_jerk: float = DEFAULT_TRAJECTORY_MAX_JERK_RAD_S3
    max_tracking_error: float = DEFAULT_TRAJECTORY_MAX_TRACKING_ERROR_RAD
    max_gripper_speed: float = DEFAULT_GRIPPER_SPEED_M_S
    ik_rate_limit: bool = True
    ik_max_joint_step_rad: float = .02
    ik_search_joint_radius_rad: float = .30
    ik_joint_regularization_weight: float = 1.0
    ik_position_tolerance_m: float = .0015
    ik_rotation_tolerance_rad: float = .02
    ik_max_nfev: int = 100
    start_joint_tolerance: float = math.radians(3)
    start_position_tolerance: float = .01
    start_rotation_tolerance: float = .08
    start_gripper_tolerance: float = .003
    feedback_max_age: float = DEFAULT_FEEDBACK_MAX_AGE_S
    feedback_ready_timeout: float = 3.
    settle_timeout: float = 30.
    approach_joint_speed: float = .08
    approach_timeout: float = 120.
    pose: PoseTrajectoryConfig = field(default_factory=PoseTrajectoryConfig)

    def __post_init__(self):
        for name in ('enable_before_replay', 'auto_recover', 'ik_rate_limit'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be a boolean')
        if self.mode not in ('joint', 'eef'):
            raise ValueError('Replay mode must be joint or eef')
        if not .05 <= self.speed <= 1:
            raise ValueError('Replay speed must be between 0.05 and 1.0')
        if not 10 <= self.hz <= 100:
            raise ValueError('Replay control rate must be between 10 and 100 Hz')
        if not 1 <= self.control_hz <= self.hz:
            raise ValueError('Replay reference rate must be positive and not exceed command rate')
        if not isinstance(self.speed_pct, int) or not 1 <= self.speed_pct <= 100:
            raise ValueError('Firmware speed percentage must be an integer between 1 and 100')
        if self.eef_smoothing and self.pose.max_update_dt_s < 1 / self.control_hz:
            raise ValueError('Pose timeout must cover one reference period')
        for key, value in vars(self).items():
            if key in ('mode', 'joint_smoothing', 'eef_smoothing', 'enable_before_replay', 'auto_recover', 'ik_rate_limit', 'pose'):
                continue
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f'Replay {key} must be finite and positive')


class RemoteCommandMonitor:
    """Reject robot-originated control frames before and throughout replay.

    SocketCAN marks locally originated frames MSG_DONTROUTE. They include our
    own SDK writer, so they are excluded. This does not arbitrate other local
    programs; the operator must establish exclusive control as well.
    """
    IDS = (0x150, 0x151, 0x152, 0x155, 0x156, 0x157, 0x159, 0x191, 0x470)

    def __init__(self, interface):
        self.interface = interface
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            filters = b''.join(struct.pack('=II', i, socket.CAN_SFF_MASK | socket.CAN_EFF_FLAG)
                               for i in self.IDS)
            self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            self.sock.bind((interface,))
            self.sock.setblocking(False)
        except BaseException:
            self.sock.close()
            raise

    def check(self):
        # Bound work even when another local process floods the interface.
        for _ in range(2048):
            try:
                packet, _, flags, _ = self.sock.recvmsg(16)
            except BlockingIOError:
                return
            if not flags & socket.MSG_DONTROUTE:
                can_id = struct.unpack_from('=I', packet)[0] & socket.CAN_SFF_MASK
                raise ReplayBlocked(f'{self.interface}: external control frame 0x{can_id:X}; '
                                    'isolate master-arm CAN control before replay')
        raise ReplayBlocked(f'{self.interface}: command monitor overflow; refusing replay')

    def close(self):
        self.sock.close()


class PiperReplayHardware:
    """Device boundary, isolated from file loading, planning, and GUI code."""
    def __init__(self, sides, selectors, config):
        self.sides, self.selectors, self.config = sides, selectors, config
        self.arms, self.monitors, self.locks = {}, {}, []
        self.commanded = set()
        self.recovery_attempted = set()
        self.feedback_deadline = 0.

    def open(self):
        from bimanual_vla.collection.can_identity import resolve_can_selector
        from bimanual_vla.collection.output import connect
        interfaces = {s: resolve_can_selector(self.selectors[s]) for s in self.sides}
        if len(set(interfaces.values())) != len(self.sides):
            raise ReplayBlocked('Replay interfaces must be distinct')
        root = Path('~/.local/state/bimanual-vla/replay-locks').expanduser()
        root.mkdir(parents=True, exist_ok=True)
        try:
            for side in sorted(self.sides):
                lock = (root / (interfaces[side] + '.lock')).open('a')
                self.locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.monitors[side] = RemoteCommandMonitor(interfaces[side])
                self.arms[side] = connect(self.selectors[side])
        except BaseException:
            self.close()
            raise

    def check_ownership(self):
        for monitor in self.monitors.values():
            monitor.check()

    def limits(self):
        from bimanual_vla.deployment.client import JOINT_LIMITS_RAD
        from bimanual_vla.deployment.hardware_safety import effective_limits, query_limits
        limits = {}
        for side, arm in self.arms.items():
            self.check_ownership()
            query_limits(arm)
            limits[side] = effective_limits(arm, JOINT_LIMITS_RAD)
        return limits

    def read(self, *, require_ready=True, allow_recoverable_faults=False):
        from bimanual_vla.deployment.client import (
            ExecutionBlocked, arm_status_dict, driver_enable_status_dict, read_output_state,
        )
        result = {}
        deadlines = []
        for side, arm in self.arms.items():
            status, drivers = arm_status_dict(arm), driver_enable_status_dict(arm)
            if status['feedback_fresh'] and (status['arm_status'] != 0 or status['err_code'] != 0):
                raise ReplayBlocked(f'{side}: arm fault/status={status["arm_status"]}, '
                                    f'err_code={status["err_code"]}; no automatic fault recovery')
            recoverable = False
            if drivers['faulted'] and allow_recoverable_faults:
                from bimanual_vla.collection.replay_recovery import eligible_recovery_joints
                eligible_recovery_joints(arm, drivers, self.config.feedback_max_age)
                recoverable = True
            if drivers['faulted'] and not recoverable:
                raise ReplayBlocked(f'{side}: motor driver faults: {drivers["faults"]}; '
                                    'no automatic fault recovery')
            if not status['feedback_fresh'] or not (drivers['healthy'] or recoverable):
                raise ReplayFeedbackPending(f'{side}: waiting for fresh status and all six motor replies '
                    f'(status_age={status["feedback_age_s"]}, driver_age={drivers["feedback_age_s"]}, '
                    f'received={drivers["received"]})')
            if require_ready and not drivers['ready']:
                disabled = [i + 1 for i, enabled in enumerate(drivers['enabled']) if not enabled]
                raise ReplayBlocked(f'{side}: motors disabled at joints {disabled}; '
                    f'ctrl_mode={status["ctrl_mode"]}, mode_feed={status["mode_feed"]}. '
                    'Select "Enable motors while holding current pose" after isolating master control, '
                    'or enable with the approved controller. No automatic reset/fault recovery.')
            if require_ready and (status['ctrl_mode'] != 1 or status['mode_feed'] != 1):
                raise ReplayBlocked(f'{side}: not in CAN joint mode '
                                    f'(ctrl_mode={status["ctrl_mode"]}, mode_feed={status["mode_feed"]})')
            try:
                eef, qpos = read_output_state(arm, max_feedback_age_s=self.config.feedback_max_age)
            except ExecutionBlocked as exc:
                raise ReplayFeedbackPending(f'{side}: {exc}') from exc
            from bimanual_vla.deployment.gripper_limits import GRIPPER_MAX_M as physical_gripper_max
            if not np.isfinite(qpos).all() or not 0 <= qpos[6] <= physical_gripper_max:
                raise ReplayBlocked(f'{side}: invalid measured joint/gripper feedback')
            eef = np.asarray(eef, dtype=float).copy()
            eef[9] = qpos[6]  # replay's internal gripper unit is always metres
            result[side] = {'joint': np.asarray(qpos, dtype=float), 'eef': eef}
            deadlines.extend(m.time_stamp + self.config.feedback_max_age for m in (
                arm.GetArmJointMsgs(), arm.GetArmGripperMsgs(), arm.GetArmEndPoseMsgs()))
        self.feedback_deadline = min(deadlines)
        return result

    def recover_startup(self, stop, progress):
        from bimanual_vla.collection.replay_recovery import recover_startup_faults
        recover_startup_faults(self, stop, progress)

    def enable_holding(self, initial, stop, progress, *, limits=None):
        """Preload measured targets, then verify six-motor enable without moving to frame 0.

        Called only after stationary/exclusive-control, limits and path checks.
        Neither the homing/recovery code nor controller reference resets are used.
        """
        from bimanual_vla.deployment.client import arm_status_dict, driver_enable_status_dict
        anchors = prepare_replay_start(initial, limits, allow_boundary_alignment=True) if limits is not None else {
            s: initial[s]['joint'].copy() for s in self.sides}
        boundary = {s for s in self.sides if anchors[s][1] != initial[s]['joint'][1]}
        disabled = []
        for side, arm in self.arms.items():
            drivers = driver_enable_status_dict(arm)
            enabled = drivers['enabled']
            if any(enabled) and not all(enabled):
                raise ReplayBlocked(f'{side}: partially enabled motors; refusing automatic enable')
            if not all(enabled):
                disabled.append(side)
            elif arm_status_dict(arm)['ctrl_mode'] != 1 or arm_status_dict(arm)['mode_feed'] != 1:
                raise ReplayBlocked(f'{side}: enabled arm is not in CAN joint mode')
        if not disabled and not boundary:
            return
        progress({'state': 'enabling_hold', 'sides': disabled,
                  'hold_rad_m': {s: q.tolist() for s, q in anchors.items()}})
        if boundary:
            progress({'state': 'aligning_start_boundary', 'sides': sorted(boundary),
                      'measured_rad_m': {s: initial[s]['joint'].tolist() for s in boundary},
                      'target_rad_m': {s: anchors[s].tolist() for s in boundary}})
        started = time.monotonic()
        previous_stamps, ready_cycles = {}, 0
        preloaded, settled_since = False, None
        while True:
            if stop.is_set():
                raise ReplayCancelled()
            self.check_ownership()
            feedback = self.read(require_ready=False)
            for side in self.sides:
                q, origin = feedback[side]['joint'], initial[side]['joint']
                indices = [0, 2, 3, 4, 5] if side in boundary else list(range(6))
                if (np.max(abs(q[indices] - origin[indices])) > .01
                        or abs(q[6] - origin[6]) > .002
                        or (side in boundary and not origin[1] - .01 <= q[1] <= anchors[side][1] + .01)):
                    raise ReplayBlocked(f'{side}: measured pose drifted during enable hold')
            elapsed = time.monotonic() - started
            if elapsed > 10.:
                raise ReplayBlocked('Initial boundary alignment did not settle within 10s')
            states = {s: (arm_status_dict(a), driver_enable_status_dict(a))
                      for s, a in self.arms.items()}
            if elapsed < .3 and any(any(states[s][1]['enabled']) for s in disabled):
                raise ReplayBlocked('Motors enabled unexpectedly during disabled target preload')
            # Preload all six joint targets before the first mode/enable command.
            # Mark before writing so a partial write is stopped by run_replay.
            if not preloaded:
                if stop.is_set():
                    raise ReplayCancelled()
                self.check_ownership()
                if time.time() > self.feedback_deadline:
                    raise ReplayBlocked('Feedback expired before target preload')
                for side in self.sides:
                    self.commanded.add(side)
                    self.arms[side].JointCtrl(*[round(float(v) * 180000 / math.pi)
                                               for v in anchors[side][:6]])
                preloaded = True
            self.send(anchors, speed_pct=1, enable_gripper=False)
            if elapsed >= .3:
                all_ready, advanced = True, True
                for side, (status, drivers) in states.items():
                    stamp = (status['feedback_timestamp'], drivers['feedback_timestamp'])
                    old = previous_stamps.get(side, (-1., -1.))
                    advanced &= stamp[0] > old[0] and stamp[1] > old[1]
                    previous_stamps[side] = stamp
                    all_ready &= (drivers['ready'] and status['ctrl_mode'] == 1 and status['mode_feed'] == 1)
                if all_ready and advanced:
                    ready_cycles += 1
                    at_boundary = all(feedback[s]['joint'][1] >= anchors[s][1] - 1e-6
                                      and abs(feedback[s]['joint'][1] - anchors[s][1]) <= .0035
                                      for s in boundary)
                    settled_since = (settled_since or time.monotonic()) if at_boundary else None
                    if ready_cycles >= 3 and (not boundary or (settled_since is not None
                            and time.monotonic() - settled_since >= .3)):
                        self.read()  # Final strict readiness check before returning.
                        progress({'state': 'enabled_hold_ready'})
                        return
                elif not all_ready:
                    ready_cycles = 0
                    settled_since = None
                    if elapsed > 3.:
                        raise ReplayBlocked('Six-motor enable / CAN joint mode did not stabilize within 3s')
                for side in disabled:
                    self.check_ownership()
                    if stop.is_set():
                        raise ReplayCancelled()
                    if time.time() > self.feedback_deadline:
                        raise ReplayBlocked('Feedback expired before motor enable')
                    # Address only arm joints. The all-motor selector can also
                    # affect the gripper during startup on this rig.
                    for joint, enabled in enumerate(states[side][1]['enabled'], start=1):
                        if not enabled:
                            self.check_ownership()
                            if stop.is_set():
                                raise ReplayCancelled()
                            if time.time() > self.feedback_deadline:
                                raise ReplayBlocked('Feedback expired during joint enable')
                            self.arms[side].EnableArm(joint)
                self.send(anchors, speed_pct=1, enable_gripper=False)
            stop.wait(.02)

    def joint_speed_limit(self, requested):
        from bimanual_vla.deployment.hardware_safety import effective_speed
        return effective_speed(self.arms, requested, speed_pct=self.config.speed_pct)

    def send(self, commands, *, speed_pct=None, enable_gripper=True):
        self.check_ownership()
        if time.time() > self.feedback_deadline:
            raise ReplayBlocked('Feedback expired before command publication')
        for side in self.sides:
            # Mark before the first potentially partial write so errors stop this
            # side, without ever sending stop to an unrelated interface.
            self.commanded.add(side)
            self.arms[side].ModeCtrl(0x01, 0x01, self.config.speed_pct if speed_pct is None else speed_pct, 0x00)
        for side in self.sides:
            q = commands[side]
            self.arms[side].JointCtrl(*[round(float(v) * 180000 / math.pi) for v in q[:6]])
        for side in self.sides:
            if enable_gripper:
                self.arms[side].GripperCtrl(round(float(commands[side][6]) * 1e6), 1000, 0x01, 0)

    def stop_holding(self):
        """Replace the source target with fresh measured joints, keeping enable.

        Preserve the gripper target. Never reset, disable, switch modes or
        take control from another controller. Unhealthy feedback/control leaves
        the firmware target alone and reports that hold could not be requested.
        """
        from bimanual_vla.deployment.client import (
            JOINT_LIMITS_RAD, arm_status_dict, driver_enable_status_dict, read_output_qpos,
        )
        from bimanual_vla.deployment.hardware_safety import effective_limits
        errors = []
        for side in sorted(self.commanded):
            try:
                self.check_ownership()
                arm = self.arms[side]
                status, drivers = arm_status_dict(arm), driver_enable_status_dict(arm)
                if (not status['feedback_fresh'] or status['arm_status'] != 0 or status['err_code'] != 0
                        or status['ctrl_mode'] != 1 or status['mode_feed'] != 1
                        or not drivers['healthy'] or not drivers['ready'] or drivers['faulted']):
                    raise ReplayBlocked('healthy enabled CAN joint control not confirmed')
                joints = read_output_qpos(arm,
                    max_feedback_age_s=min(.1, self.config.feedback_max_age))[:6]
                bounds = effective_limits(arm, JOINT_LIMITS_RAD)
                if (joints.shape != (6,) or not np.isfinite(joints).all()
                        or np.any(joints < bounds[:, 0]) or np.any(joints > bounds[:, 1])):
                    raise ReplayBlocked('measured hold target outside joint limits')
                self.check_ownership()
                arm.JointCtrl(*[round(float(v) * 180000 / math.pi) for v in joints])
            except Exception as exc:
                errors.append(f'{side}: position hold not requested: {exc}; no disable/reset sent')
        return errors

    def close(self):
        for arm in self.arms.values():
            try:
                arm.DisconnectPort()
            except Exception:
                pass
        for monitor in self.monitors.values():
            monitor.close()
        for lock in self.locks:
            lock.close()
        self.arms.clear()
        self.monitors.clear()
        self.locks.clear()


def prepare_replay_start(initial, limits, *, allow_boundary_alignment=False):
    """Preserve measured state; plan only a small below-lower-bound J2 recovery.

    Reuse deployment's measured-feedback tolerance, without expanding command
    bounds or altering driver calibration. Every published target stays within
    the queried limits, including the explicitly logged startup target.
    """
    from bimanual_vla.deployment.client import IK_FEEDBACK_LIMIT_TOLERANCE_RAD
    anchors = {}
    for side, data in initial.items():
        q = np.asarray(data['joint'], dtype=float)
        bounds = limits[side]
        bad = np.flatnonzero((q[:6] < bounds[:, 0]) | (q[:6] > bounds[:, 1]))
        permitted = (allow_boundary_alignment and np.array_equal(bad, [1])
                     and bounds[1, 0] - IK_FEEDBACK_LIMIT_TOLERANCE_RAD <= q[1] < bounds[1, 0])
        if len(bad) and not permitted:
            details = ', '.join(f'J{i+1}={math.degrees(q[i]):.3f}deg '
                f'[{math.degrees(bounds[i, 0]):.3f}, {math.degrees(bounds[i, 1]):.3f}]' for i in bad)
            raise ReplayBlocked(f'{side}: measured starting joints outside limits: {details}; '
                                'no enabling, motion or clipping; reposition within limits first')
        anchors[side] = q.copy()
        if permitted:
            anchors[side][1] = bounds[1, 0]
    return anchors


def validate_start(episode, config, feedback):
    for i, side in enumerate(episode.sides):
        actual = feedback[side]
        if 'joint' in episode.targets:
            first = episode.targets['joint'][0, i * 7:(i + 1) * 7]
            if np.max(abs(actual['joint'][:6] - first[:6])) > config.start_joint_tolerance:
                raise ReplayBlocked(f'{side}: align joints with episode frame 0 before replay')
        width = 7 if config.mode == 'joint' else 10
        first = episode.targets[config.mode][0, i * width:(i + 1) * width]
        if abs(actual['joint'][6] - first[-1]) > config.start_gripper_tolerance:
            raise ReplayBlocked(f'{side}: align gripper with episode frame 0 before replay')
        if config.mode == 'eef':
            angle = Rotation.from_matrix(rotation_matrix_from_state(first) @
                                         rotation_matrix_from_state(actual['eef']).T).magnitude()
            if (np.linalg.norm(first[:3] - actual['eef'][:3]) > config.start_position_tolerance
                    or angle > config.start_rotation_tolerance):
                raise ReplayBlocked(f'{side}: align end-effector with episode frame 0 before replay')


class ReplayPlanner:
    def __init__(self, episode, config, limits, initial, ik=None):
        self.episode, self.config, self.limits = episode, config, limits
        self.joint, self.pose, self.warm = {}, {}, {}
        self.ik = ik
        if config.mode == 'eef' and self.ik is None:
            from bimanual_vla.deployment.client import PiperContinuousIK
            self.ik = PiperContinuousIK()
        for side in episode.sides:
            if config.joint_smoothing:
                self.joint[side] = RuckigJointServo(initial[side]['joint'][:6],
                    limits[side][:, 0], limits[side][:, 1], ServoConfig(
                        hz=config.hz, max_velocity=config.max_joint_speed,
                        max_acceleration=config.max_joint_acceleration, max_jerk=config.max_joint_jerk,
                        max_tracking_error=config.max_tracking_error))
            if config.mode == 'eef' and config.eef_smoothing:
                state = initial[side]['eef']
                self.pose[side] = PoseTrajectory(state[:3], rotation_matrix_from_state(state), config.pose)
        self.previous = {s: initial[s]['joint'].copy() for s in episode.sides}
        self.reference_targets, self.reference_raw = {}, {}
        self.ik_rate_limit = {}

    def prepare(self, index, feedback, *, reference_update=True):
        cfg = self.config
        width = 7 if cfg.mode == 'joint' else 10
        commands = {}
        for i, side in enumerate(self.episode.sides):
            measured = feedback[side]['joint']
            update = reference_update or side not in self.reference_targets
            if not update:
                raw = self.reference_raw[side]
                target = self.reference_targets[side].copy()
            else:
                raw = self.episode.targets[cfg.mode][index, i * width:(i + 1) * width]
                if cfg.mode == 'joint':
                    target = raw[:6].copy()
                else:
                    xyz, rotation = raw[:3], rotation_matrix_from_state(raw)
                    if side in self.pose:
                        xyz, rotation, _ = self.pose[side].update(xyz, rotation, dt=1 / cfg.control_hz,
                            feedback_xyz=feedback[side]['eef'][:3],
                            feedback_rotation=rotation_matrix_from_state(feedback[side]['eef']))
                    validate_eef_position(xyz)
                    rpy = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
                    max_step = (cfg.ik_max_joint_step_rad
                                if cfg.ik_rate_limit and not cfg.joint_smoothing
                                else cfg.ik_search_joint_radius_rad)
                    kwargs = dict(initial_joints_rad=self.warm.get(side),
                        max_joint_step_rad=max_step,
                        search_joint_radius_rad=cfg.ik_search_joint_radius_rad,
                        joint_regularization_weight=cfg.ik_joint_regularization_weight,
                        position_tolerance_m=cfg.ik_position_tolerance_m,
                        rotation_tolerance_rad=cfg.ik_rotation_tolerance_rad,
                        max_nfev=cfg.ik_max_nfev)
                    # Replay and inference must solve against the same
                    # per-arm effective limits (software intersected with the
                    # limits read from this physical driver).
                    from bimanual_vla.deployment.client import PiperContinuousIK
                    if isinstance(self.ik, PiperContinuousIK):
                        kwargs['joint_limits'] = self.limits[side]
                    diagnostics = getattr(self.ik, 'solve_with_diagnostics', None)
                    if callable(diagnostics):
                        result = diagnostics(measured[:6], xyz, rpy, **kwargs)
                        solution = np.asarray(result.solution_joints_rad, dtype=float)
                        target = np.asarray(result.command_joints_rad, dtype=float)
                        self.ik_rate_limit[side] = {
                            'limited': bool(result.rate_limited and not cfg.joint_smoothing),
                            'joint_limited': bool(result.rate_limited and not cfg.joint_smoothing),
                            'required_max_joint_step_rad': float(result.solution_max_joint_step_rad or 0.),
                            'command_max_joint_step_rad': float(np.max(abs(target - measured[:6]))),
                            'limit_rad': float(max_step),
                        }
                    else:
                        # Test/custom solvers retain their legacy signature;
                        # production inference uses the diagnostics path.
                        legacy_kwargs = dict(kwargs)
                        legacy_kwargs.pop('initial_joints_rad', None)
                        legacy_kwargs.pop('joint_regularization_weight', None)
                        target = np.asarray(self.ik.solve(measured[:6], xyz, rpy, **legacy_kwargs), dtype=float)
                        solution = target.copy()
                        self.ik_rate_limit[side] = {'limited': False, 'joint_limited': False}
                    self.warm[side] = solution.copy()
                self.reference_targets[side] = target.copy()
                self.reference_raw[side] = raw
            if not np.isfinite(target).all() or np.any(target < self.limits[side][:, 0]) or np.any(target > self.limits[side][:, 1]):
                bad = np.flatnonzero((target < self.limits[side][:, 0]) | (target > self.limits[side][:, 1]))
                detail = ', '.join(
                    f'J{i + 1}={target[i]:.6f} bounds=[{self.limits[side][i, 0]:.6f},{self.limits[side][i, 1]:.6f}]'
                    for i in bad
                )
                raise ReplayBlocked(f'{side}: replay target outside joint limits ({detail})')
            if side in self.joint:
                if update:
                    self.joint[side].retarget(target)
                target = self.joint[side].update(measured[:6])
            elif cfg.mode == 'joint' and np.max(abs(target - self.previous[side][:6])) > cfg.max_joint_speed / cfg.control_hz + 1e-7:
                speed = abs(target - self.previous[side][:6]) * cfg.control_hz
                joint = int(np.argmax(speed))
                raise ReplayBlocked(f'{side}: unsmoothed target exceeds joint speed; '
                    f'J{joint + 1} required={speed[joint]:.6f}rad/s '
                    f'limit={cfg.max_joint_speed:.6f}rad/s; reduce replay speed or enable joint smoothing')
            error = abs(target - measured[:6])
            if np.max(error) > cfg.max_tracking_error:
                joint = int(np.argmax(error))
                raise ReplayBlocked(f'{side}: replay tracking error exceeded limit; '
                    f'J{joint + 1} command={math.degrees(target[joint]):.3f}deg '
                    f'feedback={math.degrees(measured[joint]):.3f}deg '
                    f'error={math.degrees(error[joint]):.3f}deg '
                    f'limit={math.degrees(cfg.max_tracking_error):.3f}deg; '
                    'stopped, no automatic restart')
            opening = float(raw[-1])
            if cfg.joint_smoothing or cfg.mode == 'eef':
                delta = cfg.max_gripper_speed / cfg.hz
                opening = float(np.clip(opening, self.previous[side][6] - delta, self.previous[side][6] + delta))
            elif abs(opening - self.previous[side][6]) > cfg.max_gripper_speed / cfg.control_hz + 1e-7:
                raise ReplayBlocked(f'{side}: unsmoothed gripper target exceeds speed limit')
            if side in self.ik_rate_limit:
                limited = abs(opening - float(raw[-1])) > 1e-12
                self.ik_rate_limit[side]['gripper_limited'] = limited
                self.ik_rate_limit[side]['limited'] |= limited
            commands[side] = np.r_[target, opening]
        # Prepare both sides successfully before publishing either side.
        self.previous = {s: q.copy() for s, q in commands.items()}
        return commands

    def arrived(self, feedback):
        width = 7 if self.config.mode == 'joint' else 10
        for i, side in enumerate(self.episode.sides):
            target = self.episode.targets[self.config.mode][-1, i * width:(i + 1) * width]
            if abs(feedback[side]['joint'][6] - target[-1]) > .001:
                return False
            if self.config.mode == 'joint':
                if np.max(abs(feedback[side]['joint'][:6] - target[:6])) > .01:
                    return False
            else:
                pose = feedback[side]['eef']
                if (np.linalg.norm(pose[:3] - target[:3]) > .003 or
                        Rotation.from_matrix(rotation_matrix_from_state(target) @
                                             rotation_matrix_from_state(pose).T).magnitude() > .03):
                    return False
            if side in self.joint and (np.max(abs(self.joint[side].velocity)) > .005
                                      or np.max(abs(self.joint[side].acceleration)) > .02
                                      or np.max(abs(self.joint[side].position - self.joint[side].target)) > 1e-4):
                return False
            if side in self.pose and (np.linalg.norm(self.pose[side].xyz - target[:3]) > .0001
                                      or np.linalg.norm(self.pose[side].linear_velocity) > .001
                                      or np.linalg.norm(self.pose[side].angular_velocity) > .005):
                return False
        return True


def validate_eef_position(xyz):
    """Validate units/shape only; reachability is checked by IK and joint limits.

    The deployment task's XYZ box is not a manufacturer arm limit and must
    not reject recorded follower-base poses (including negative right-arm Y).
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ReplayBlocked('Invalid end-effector position; expected finite XYZ in metres')


def wait_for_replay_feedback(hardware, stop, timeout, *, allow_disabled=False):
    """Wait for initial CAN caches only; faults and external control fail closed."""
    deadline = time.monotonic() + timeout
    while True:
        if stop.is_set():
            raise ReplayCancelled()
        hardware.check_ownership()
        try:
            return hardware.read(require_ready=False) if allow_disabled else hardware.read()
        except ReplayFeedbackPending as exc:
            if time.monotonic() >= deadline:
                raise ReplayBlocked(f'No complete fresh CAN feedback within {timeout:.1f}s: {exc}') from exc
        stop.wait(.02)


def run_replay(episode: ReplayEpisode, config: ReplayConfig, selectors: dict[str, str], *,
               exclusive_control: bool = False, stop: threading.Event | None = None,
               progress: Callable[[dict], None] = lambda event: None,
               hardware_factory=PiperReplayHardware):
    """Approach the recorded first pose, then start the episode clock.

    Optional enable holds the measured pose; never reset, home, or switch roles.
    """
    if not exclusive_control:
        raise ReplayBlocked('Confirm that master control is isolated and robot control is exclusive')
    if config.mode not in episode.formats:
        raise ReplayBlocked(f'Episode has no recorded {config.mode} representation')
    if 'joint' not in episode.formats:
        raise ReplayBlocked('Automatic first-pose approach requires recorded joint feedback')
    if not config.joint_smoothing:
        config = replace(config, hz=config.control_hz)
    stop = stop or threading.Event()
    hardware = hardware_factory(episode.sides, selectors, config)

    def check_cancelled():
        if stop.is_set():
            raise ReplayCancelled()

    def sample(*, allow_disabled=False):
        check_cancelled()
        hardware.check_ownership()
        return hardware.read(require_ready=False) if allow_disabled else hardware.read()

    try:
        check_cancelled()
        hardware.open()
        progress({'state': 'preflight'})
        if config.auto_recover:
            hardware.recover_startup(stop, progress)
        progress({'state': 'waiting_feedback'})
        baseline = wait_for_replay_feedback(hardware, stop, config.feedback_ready_timeout,
                                           allow_disabled=config.enable_before_replay)
        # Observe long enough to catch idle/intermittent leader control frames.
        for _ in range(20):
            stop.wait(.05)
            actual = sample(allow_disabled=config.enable_before_replay)
            for side in episode.sides:
                if (np.max(abs(actual[side]['joint'][:6] - baseline[side]['joint'][:6])) > .003
                        or abs(actual[side]['joint'][6] - baseline[side]['joint'][6]) > .001):
                    raise ReplayBlocked(f'{side}: arm is moving during replay preflight')
        limits = hardware.limits()
        requested_speed = config.max_joint_speed
        config = replace(config, max_joint_speed=hardware.joint_speed_limit(requested_speed))
        progress({'state': 'limits_verified', 'requested_max_joint_speed_rad_s': requested_speed,
                  'firmware_speed_pct': config.speed_pct,
                  'effective_max_joint_speed_rad_s': config.max_joint_speed})
        for i, side in enumerate(episode.sides):
            if 'joint' in episode.targets:
                path = episode.targets['joint'][:, i * 7:i * 7 + 6]
                if np.any(path < limits[side][:, 0]) or np.any(path > limits[side][:, 1]):
                    raise ReplayBlocked(f'{side}: recorded joint path exceeds driver/software limits')
            if config.mode == 'eef':
                for point in episode.targets['eef'][:, i * 10:i * 10 + 3]:
                    validate_eef_position(point)
        initial = sample(allow_disabled=config.enable_before_replay)
        prepare_replay_start(initial, limits, allow_boundary_alignment=config.enable_before_replay)
        if config.enable_before_replay:
            hardware.enable_holding(initial, stop, progress, limits=limits)
            initial = sample()
            prepare_replay_start(initial, limits)
        approach_first_frame(episode, config, hardware, limits, initial, sample, stop, progress)
        initial = sample()
        validate_start(episode, config, initial)
        planner = ReplayPlanner(episode, config, limits, initial)
        clock = episode.timestamps[config.mode]
        duration = float(clock[-1]) / config.speed
        started = time.monotonic()
        previous_tick = started
        reference_tick = -1
        index = 0
        ticks, stable_since, settle_started = 0, None, None
        progress({'state': 'running', 'index': 0, 'frames': len(clock),
                  'control_hz': config.control_hz, 'command_hz': config.hz,
                  'adaptive_ik_timing': False})
        while True:
            feedback = sample()
            now = time.monotonic()
            if now - previous_tick > .15:
                raise ReplayBlocked('Replay control clock stalled; stopped instead of catching up')
            previous_tick = now
            elapsed = now - started
            current_reference_tick = math.floor(elapsed * config.control_hz + 1e-9)
            reference_update = current_reference_tick > reference_tick
            if reference_update:
                index = max(0, min(len(clock) - 1,
                    int(np.searchsorted(clock, elapsed * config.speed, side='right') - 1)))
                reference_tick = current_reference_tick
            commands = planner.prepare(index, feedback, reference_update=reference_update)
            check_cancelled()
            if time.monotonic() - now > min(.1, config.feedback_max_age):
                raise ReplayBlocked('Replay planning exceeded feedback deadline')
            hardware.send(commands)
            if ticks % max(1, round(config.hz / 10)) == 0:
                progress({'state': 'running', 'index': index, 'frames': len(clock),
                          'elapsed_s': elapsed, 'duration_s': duration,
                          'playback_elapsed_s': elapsed * config.speed,
                          'commands_rad_m': {s: q.tolist() for s, q in commands.items()},
                          'recorded_targets': {s: episode.targets[config.mode][index,
                              i * (7 if config.mode == 'joint' else 10):
                              (i + 1) * (7 if config.mode == 'joint' else 10)].tolist()
                              for i, s in enumerate(episode.sides)},
                          'recorded_representation': config.mode,
                          'ik_rate_limit': planner.ik_rate_limit,
                          'feedback_rad_m': {s: feedback[s]['joint'].tolist() for s in episode.sides}})
            if elapsed >= duration:
                settle_started = settle_started if settle_started is not None else now
            if elapsed >= duration and planner.arrived(feedback):
                stable_since = stable_since or now
                if now - stable_since >= .3:
                    progress({'state': 'completed', 'index': len(clock) - 1, 'frames': len(clock)})
                    return 'completed'
            else:
                stable_since = None
            if settle_started is not None and now - settle_started > config.settle_timeout:
                raise ReplayBlocked('Replay final pose did not settle before timeout')
            ticks += 1
            # Never replay missed ticks in a burst or accelerate after slow IK.
            stop.wait(max(0., 1 / config.hz - (time.monotonic() - now)))
    except ReplayCancelled:
        errors = hardware.stop_holding()
        progress({'state': 'stopped', 'stop_mode': 'hold_position', 'stop_errors': errors})
        return 'stopped'
    except BaseException:
        errors = hardware.stop_holding()
        progress({'state': 'holding_after_failure', 'stop_mode': 'hold_position', 'stop_errors': errors})
        if errors:
            progress({'state': 'stop_failed', 'errors': errors})
        raise
    finally:
        hardware.close()


def approach_first_frame(episode, config, hardware, limits, initial, sample, stop, progress):
    """Always use bounded joint motion for approach, independent of replay toggles."""
    planners, targets = {}, {}
    previous = {s: initial[s]['joint'].copy() for s in episode.sides}
    for i, side in enumerate(episode.sides):
        target = episode.targets['joint'][0, i * 7:(i + 1) * 7]
        targets[side] = target
        planners[side] = RuckigJointServo(initial[side]['joint'][:6], limits[side][:, 0],
            limits[side][:, 1], ServoConfig(hz=config.hz,
                max_velocity=min(config.approach_joint_speed, config.max_joint_speed),
                max_acceleration=min(.6, config.max_joint_acceleration),
                max_jerk=min(4., config.max_joint_jerk), max_tracking_error=config.max_tracking_error))
        planners[side].retarget(target[:6])
    started = previous_tick = time.monotonic()
    stable_since, ticks = None, 0
    progress({'state': 'approaching', 'elapsed_s': 0})
    while True:
        feedback = sample()
        now = time.monotonic()
        if now - previous_tick > .15:
            raise ReplayBlocked('First-pose approach control clock stalled')
        previous_tick = now
        if now - started > config.approach_timeout:
            raise ReplayBlocked('First-pose approach timed out')
        commands, arrived = {}, True
        for side in episode.sides:
            measured, target = feedback[side]['joint'], targets[side]
            q = planners[side].update(measured[:6])
            delta = config.max_gripper_speed / config.hz
            opening = np.clip(target[6], previous[side][6] - delta, previous[side][6] + delta)
            commands[side] = np.r_[q, opening]
            arrived &= bool(np.max(abs(measured[:6] - target[:6])) <= .01
                            and abs(measured[6] - target[6]) <= .001
                            and np.max(abs(q - target[:6])) < 1e-4
                            and np.max(abs(planners[side].velocity)) < .005
                            and np.max(abs(planners[side].acceleration)) < .02)
        if stop.is_set():
            raise ReplayCancelled()
        if time.monotonic() - now > min(.1, config.feedback_max_age):
            raise ReplayBlocked('First-pose approach feedback deadline exceeded')
        hardware.send(commands)
        previous = commands
        if ticks % max(1, round(config.hz / 10)) == 0:
            progress({'state': 'approaching', 'elapsed_s': now - started,
                      'commands_rad_m': {s: q.tolist() for s, q in commands.items()},
                      'feedback_rad_m': {s: feedback[s]['joint'].tolist() for s in episode.sides}})
        if arrived:
            stable_since = stable_since or now
            if now - stable_since >= .3:
                progress({'state': 'first_pose_reached'})
                return
        else:
            stable_since = None
        ticks += 1
        stop.wait(max(0., 1 / config.hz - (time.monotonic() - now)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--mode', choices=('joint', 'eef'), default='joint')
    parser.add_argument('--speed', type=float, default=.25)
    parser.add_argument('--joint-smoothing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eef-smoothing', action=argparse.BooleanOptionalAction, default=ReplayConfig().eef_smoothing)
    parser.add_argument('--control-hz', type=float, default=DEFAULT_ACTION_HZ,
                        help='Reference update rate, matching inference --control-hz')
    parser.add_argument('--servo-hz', type=float, default=DEFAULT_SERVO_HZ,
                        help='Command publication rate, matching inference --servo-hz')
    parser.add_argument('--auto-recover', action=argparse.BooleanOptionalAction, default=True,
                        help='Clear eligible cool, disabled thermal latches once during startup; never reset')
    parser.add_argument('--enable-before-replay', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable disabled motors while holding measured pose after preflight')
    parser.add_argument('--left-can', default='can0')
    parser.add_argument('--right-can', default='can1')
    parser.add_argument('--allow-execution', action='store_true')
    parser.add_argument('--exclusive-control', action='store_true')
    parser.add_argument('--config-file', type=Path, help='ReplayConfig JSON written by the GUI')
    parser.add_argument('--parent-pid', type=int, help='Stop if the owning GUI exits')
    args = parser.parse_args(argv)
    episode = load_replay_episode(args.episode)
    config = ReplayConfig(mode=args.mode, speed=args.speed, joint_smoothing=args.joint_smoothing,
                          eef_smoothing=args.eef_smoothing, enable_before_replay=args.enable_before_replay,
                          auto_recover=args.auto_recover, control_hz=args.control_hz, hz=args.servo_hz)
    if args.config_file:
        values = json.loads(args.config_file.read_text(encoding='utf-8'))
        pose = PoseTrajectoryConfig(**values.pop('pose', {}))
        config = ReplayConfig(**values, pose=pose)
    if not args.allow_execution:
        print(json.dumps({'path': str(episode.path), 'sides': episode.sides, 'formats': episode.formats,
                          'execution': False}, ensure_ascii=False))
        return
    import signal
    stopped = threading.Event()
    finished = threading.Event()
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())

    def watch_parent():
        import os
        while not finished.is_set():
            if os.getppid() != args.parent_pid:
                stopped.set()
                return
            finished.wait(.05)

    watcher = None
    if args.parent_pid is not None:
        import os
        if os.getppid() != args.parent_pid:
            stopped.set()
        watcher = threading.Thread(target=watch_parent, name='replay-owner-watch', daemon=True)
        watcher.start()
    try:
        run_replay(episode, config, {'left': args.left_can, 'right': args.right_can},
                   exclusive_control=args.exclusive_control, stop=stopped,
                   progress=lambda event: print(json.dumps(event), flush=True))
    except Exception as exc:
        import traceback
        print(json.dumps({'state': 'failed', 'error': str(exc),
                          'error_type': type(exc).__name__, 'traceback': traceback.format_exc()}), flush=True)
        raise SystemExit(1) from exc
    finally:
        finished.set()
        if watcher is not None:
            watcher.join(timeout=.2)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
