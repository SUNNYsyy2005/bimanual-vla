"""Exclusive Piper MOVE P publisher. No local IK and no joint trajectory API.

Firmware owns the Cartesian path and braking response. A measured-pose hold
requests stopping; only fresh feedback can establish that it settled. Software
cannot guarantee holding after transport loss. Faults never auto-disable arms.
"""
from dataclasses import dataclass
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.deployment.pose_trajectory import _rotation, _vector
from bimanual_vla.deployment.gripper_limits import GRIPPER_MAX_M

# A boundary joint pose need not survive the firmware FK -> integer Cartesian
# feedback -> IK round trip. These are startup clearances, not new hard limits.
NATIVE_STARTUP_MARGIN_RAD = float(np.deg2rad(1.0))
NATIVE_STARTUP_MIN_MARGIN_RAD = float(np.deg2rad(0.5))


class NativeEEFError(RuntimeError):
    pass


@dataclass(frozen=True)
class EEFReference:
    xyz: np.ndarray
    rotation: np.ndarray
    gripper_m: float

    def checked(self):
        xyz = _vector(self.xyz, (3,), "native target")
        rotation = _rotation(self.rotation)
        opening = float(self.gripper_m)
        if not np.isfinite(opening) or not 0 <= opening <= GRIPPER_MAX_M:
            raise NativeEEFError("invalid native gripper opening")
        return EEFReference(xyz, rotation, opening)

    def wire(self):
        value = self.checked()
        pose = np.r_[value.xyz * 1e6,
                     Rotation.from_matrix(value.rotation).as_euler("xyz", degrees=True) * 1000]
        if np.any(np.abs(pose) > np.iinfo(np.int32).max):
            raise NativeEEFError("native pose exceeds signed 32-bit wire range")
        return tuple(int(v) for v in np.rint(pose))


class NativeEEFLoop:
    """A single writer with paired validation, expiry hold and feedback watchdog.

    read_feedback returns per-side reference/qpos/status/stamp. It must enforce
    CAN health, stream freshness, enabled drivers and measured joint limits.
    status is checked here too, including timestamped mode acknowledgement.
    No transport method is called from submit/hold; only this owner's thread
    writes once start's synchronous handoff has completed.
    """
    def __init__(self, pipers, read_feedback, *, hz=30., speed_pct=10,
                 gripper_effort=1000, target_timeout_s=.15, mode_timeout_s=2.,
                 settle_s=.2, max_feedback_age_s=.5, hold_timeout_s=3., check_target=None, check_transport=None):
        for value in (hz, target_timeout_s, mode_timeout_s, settle_s, max_feedback_age_s, hold_timeout_s):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("native EEF timing must be finite and positive")
        if not 1 <= speed_pct <= 100 or target_timeout_s < 2/hz:
            raise ValueError("invalid native speed or reference timeout")
        self.pipers = dict(pipers)
        self.read_feedback = read_feedback
        self.check_target = check_target or (lambda side, value: None)
        self.check_transport = check_transport or (lambda: None)
        self.hz, self.speed_pct = hz, int(speed_pct)
        self.gripper_effort = int(gripper_effort)
        self.target_timeout_s, self.mode_timeout_s = target_timeout_s, mode_timeout_s
        self.settle_s, self.max_feedback_age_s = settle_s, max_feedback_age_s
        self.hold_timeout_s = hold_timeout_s
        self._hold_started_at = time.monotonic()
        self._condition = threading.Condition(threading.RLock())
        self._thread = None
        self._closed = self._stopping = False
        self._close_safe = False
        self._fault = None
        self._hold = True
        self._capture_hold = True
        self._settled = False
        self._stable_since = None
        self._previous = None
        self._target = {}
        self._at = None
        self._epoch = 0
        self._sequence = 0
        self._reason = "startup"
        self._grippers = {}
        self._errors = {}
        self._started = False
        self._revision = self._published_revision = 0
        self._last_wire = {}
        self._phase = "startup_precheck"
        self._fault_feedback = None

    def _feedback_fault(self, side, frame, reason):
        """Keep firmware rejection distinct from missing/stale CAN feedback."""
        detail = dict(side=side, phase=self._phase, status=dict(frame["status"]),
                      feedback_age_s=time.time()-float(frame["stamp"]),
                      measured_joints_rad=frame["qpos"][:6].tolist(),
                      measured_pose_units=frame["reference"].wire(),
                      last_sent_pose_units=self._last_wire.get(side),
                      joint_limits_rad=frame.get("joint_limits_rad"))
        self._fault_feedback = detail
        raise NativeEEFError(f"{side} native {reason}; {detail}")

    def _feedback(self, *, require_mode=True):
        frames = self.read_feedback()
        if set(frames) != set(self.pipers):
            raise NativeEEFError("native feedback arm layout mismatch")
        now = time.time()
        for side, frame in frames.items():
            frame["reference"] = frame["reference"].checked()
            frame["qpos"] = _vector(frame["qpos"], (7,), "native joint feedback")
            stamp = float(frame["stamp"])
            status = frame["status"]
            if (not np.isfinite(stamp) or not 0 < stamp or not -1 <= now-stamp <= self.max_feedback_age_s
                    or status.get("feedback_fresh") is not True):
                self._feedback_fault(side, frame, "feedback stale")
            if status.get("arm_status") != 0 or status.get("err_code") != 0:
                labels = {2: "IK has no solution", 3: "singularity", 4: "target joint limit exceeded"}
                label = labels.get(status.get("arm_status"), "arm/controller fault")
                self._feedback_fault(side, frame, f"firmware rejected target: {label}")
            if require_mode and (status.get("ctrl_mode") != 1 or status.get("mode_feed") != 0):
                self._feedback_fault(side, frame, "lost acknowledged CAN/MOVE_P mode")
        return frames

    def _checked_targets(self, targets):
        if set(targets) != set(self.pipers):
            raise NativeEEFError("native target arm layout mismatch")
        values = {s: v.checked() for s, v in targets.items()}
        for side, value in values.items():
            self.check_target(side, value)
            value.wire()
        return values

    def _publish(self, values):
        # Precompute both arms before any write. Bus failure can still cause
        # partial physical publication; the sticky transport fault must latch.
        wire = {s: v.wire() for s, v in values.items()}
        for side, piper in self.pipers.items():
            self.check_transport()
            piper.EndPoseCtrl(*wire[side])
            self._last_wire[side] = wire[side]
        for side, piper in self.pipers.items():
            self.check_transport()
            piper.GripperCtrl(round(values[side].gripper_m * 1e6), self.gripper_effort, 0x01, 0)
        self.check_transport()
        self._sequence += 1

    def start(self):
        if self._started or self._closed or self._fault:
            raise NativeEEFError("native writer cannot be restarted")
        requested_at = time.time()
        deadline = time.monotonic() + self.mode_timeout_s
        try:
            frames = self._feedback(require_mode=False)
            # Check both arms before switching either one out of joint control.
            for side, frame in frames.items():
                limits = _vector(frame.get("joint_limits_rad"), (6, 2), "native startup limits")
                margin = np.minimum(frame["qpos"][:6]-limits[:, 0],
                                    limits[:, 1]-frame["qpos"][:6])
                if np.any(margin < NATIVE_STARTUP_MIN_MARGIN_RAD):
                    self._feedback_fault(side, frame,
                        "startup joint clearance insufficient; joint startup approach required")
            held = {s: f["reference"] for s, f in frames.items()}
            self._target = held
            self._grippers = {s: v.gripper_m for s, v in held.items()}
            # A joint owner must already be stopped before calling start.
            # Repeat a measured-pose hold while waiting for fresh MOVE P ack.
            self._phase = "MOVE_P_handoff"
            while True:
                for piper in self.pipers.values():
                    self.check_transport()
                    piper.MotionCtrl_2(0x01, 0x00, self.speed_pct, 0x00)
                self._publish(held)
                frames = self._feedback(require_mode=False)
                if all(f["status"]["ctrl_mode"] == 1 and f["status"]["mode_feed"] == 0
                       and float(f["status"].get("feedback_timestamp") or 0) > requested_at
                       for f in frames.values()):
                    break
                if time.monotonic() >= deadline:
                    raise NativeEEFError("native MOVE_P acknowledgement timed out; return suppressed")
                time.sleep(min(.02, 1/self.hz))
            self._capture_hold = False
            self._hold_started_at = time.monotonic()
            # Mode acknowledgement can precede IK rejection of the first hold.
            # Prove a fresh stable hold before reporting native control active.
            self._phase = "startup_hold_confirmation"
            while not self._settled:
                self._tick()
                if not self._settled:
                    time.sleep(1/self.hz)
            self._started = True
            self._phase = "running"
            self._thread = threading.Thread(target=self._run, name="piper-native-eef", daemon=True)
            self._thread.start()
        except BaseException as exc:
            self._fault = str(exc) or type(exc).__name__
            raise

    def submit(self, targets):
        values = self._checked_targets(targets)
        with self._condition:
            if not self._started or self._closed or self._stopping or self._fault:
                raise NativeEEFError(self._fault or "native writer unavailable")
            if self._hold and not self._settled:
                raise NativeEEFError("native hold has not settled")
            self._target = values
            self._at = time.monotonic()
            self._hold = self._capture_hold = self._settled = False
            self._stable_since = None
            self._revision += 1
            revision = self._revision
            self._condition.notify_all()
            deadline = self._at + max(.15, 4/self.hz)
            while self._published_revision < revision and not self._fault and not self._closed:
                if time.monotonic() >= deadline:
                    self._fault = "native publisher missed submission deadline"
                    break
                self._condition.wait(max(0., deadline-time.monotonic()))
            if self._fault or self._closed or self._hold:
                raise NativeEEFError(self._fault or "native submission interrupted")
            return dict(sequence=self._sequence, grippers=dict(self._grippers))

    def hold(self, reason="requested"):
        with self._condition:
            if not self._hold:
                self._hold = self._capture_hold = True
                self._settled = False
                self._stable_since = None
                self._epoch += 1
                self._hold_started_at = time.monotonic()
            self._reason = reason
            self._condition.notify_all()

    def pause(self):
        with self._condition:
            self._fault = self._fault or "native publication safety-paused"
            self._condition.notify_all()

    def metrics(self):
        with self._condition:
            return dict(backend="piper_eef", mode="MOVE_P", fault=self._fault,
                        holding=self._hold, settled=self._settled, stopping=self._stopping,
                        epoch=self._epoch, sample_count=self._sequence, hold_reason=self._reason,
                        fault_feedback=self._fault_feedback,
                        feedback_errors=dict(self._errors))

    def _tick(self):
        frames = self._feedback()
        now = time.monotonic()
        if not self._hold and (self._at is None or now-self._at > self.target_timeout_s):
            self.hold("reference_timeout")
        if self._capture_hold:
            self._target = {s: f["reference"] for s, f in frames.items()}
            self._grippers = {s: v.gripper_m for s, v in self._target.items()}
            self._capture_hold = False
        self._errors = {s: dict(
            position_m=float(np.linalg.norm(f["reference"].xyz-self._target[s].xyz)),
            rotation_rad=float(Rotation.from_matrix(f["reference"].rotation @ self._target[s].rotation.T).magnitude()))
            for s, f in frames.items()}
        fresh = self._previous is not None and all(
            frames[s]["stamp"] > self._previous[s]["stamp"] for s in frames)
        if self._hold:
            stable = fresh and all(
                np.max(np.abs(f["qpos"][:6]-self._previous[s]["qpos"][:6])) / (f["stamp"]-self._previous[s]["stamp"]) < .01
                and np.linalg.norm(f["reference"].xyz-self._previous[s]["reference"].xyz) / (f["stamp"]-self._previous[s]["stamp"]) < .002
                and Rotation.from_matrix(f["reference"].rotation @ self._previous[s]["reference"].rotation.T).magnitude() / (f["stamp"]-self._previous[s]["stamp"]) < .02
                and self._errors[s]["position_m"] < .005
                and self._errors[s]["rotation_rad"] < .05
                and f["status"].get("motion_status") == 0
                for s, f in frames.items())
            if fresh:
                self._stable_since = (self._stable_since if self._stable_since is not None else now) if stable else None
                self._settled = bool(self._stable_since is not None and now-self._stable_since >= self.settle_s)
            if not self._settled and now-self._hold_started_at > self.hold_timeout_s:
                raise NativeEEFError("native measured hold did not settle before deadline")
        values = {}
        for side, target in self._target.items():
            opening = self._grippers[side] + float(np.clip(target.gripper_m-self._grippers[side], -.08/self.hz, .08/self.hz))
            self._grippers[side] = opening
            values[side] = EEFReference(target.xyz, target.rotation, opening)
        self._publish(values)
        self._published_revision = self._revision
        if self._previous is None or fresh:
            self._previous = frames

    def _run(self):
        with self._condition:
            while not self._closed and not self._fault:
                started = time.monotonic()
                try:
                    self._tick()
                except Exception as exc:
                    self._fault = f"{type(exc).__name__}: {exc}"
                    self._settled = False
                self._condition.notify_all()
                # Wakeups do not cause publication faster than the configured rate.
                deadline = started + 1/self.hz
                while not self._closed and not self._fault and time.monotonic() < deadline:
                    self._condition.wait(max(0., deadline-time.monotonic()))

    def close(self, *, stop_timeout_s=3.):
        with self._condition:
            if self._closed:
                return self._close_safe
            self._stopping = True
            self.hold("shutdown")
            # Even an already-held writer must prove a fresh stable interval
            # after Stop; a previous settled flag cannot authorize return.
            self._settled = False
            self._stable_since = None
            self._previous = None
            self._capture_hold = True
            self._hold_started_at = time.monotonic()
            deadline = time.monotonic() + stop_timeout_s
            while self._started and not self._settled and not self._fault and time.monotonic() < deadline:
                self._condition.wait(max(0., deadline-time.monotonic()))
            safe = self._started and self._settled and not self._fault
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.)
            if self._thread.is_alive():
                raise NativeEEFError("native writer did not terminate; ownership cannot transfer")
        self._close_safe = bool(safe)
        return self._close_safe
