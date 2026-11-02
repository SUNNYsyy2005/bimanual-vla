"""Continuous Quest IK -> JointCtrl collection writer; no Ruckig/RTC."""

from dataclasses import dataclass
import fcntl
import json
import logging
import queue
from pathlib import Path
import threading
import time

import numpy as np

from .input import QuestReader, adjusted_pose
from .ik import IKRejected, QuestIK


class InvalidCommand(IKRejected):
    """A command rejected before any SDK write; safe to report without exiting."""


def validate_command(side, target, limits):
    target = np.asarray(target, dtype=float)
    reason = None
    if target.shape != (7,):
        reason = f"expected 7 values, got shape {target.shape}"
    elif not np.isfinite(target).all():
        reason = "non-finite values"
    else:
        outside = np.flatnonzero((target[:6] < limits[:, 0]) | (target[:6] > limits[:, 1]))
        if outside.size:
            reason = "; ".join(
                f"J{i + 1}={np.rad2deg(target[i]):.9f} deg outside "
                f"[{np.rad2deg(limits[i, 0]):.9f}, {np.rad2deg(limits[i, 1]):.9f}] deg"
                for i in outside)
        elif not 0 <= target[6] <= 1:
            reason = f"gripper={target[6]:.9f} outside [0, 1]"
    if reason:
        raise InvalidCommand(f"{side}: invalid joint/gripper command: {reason}")


@dataclass
class GripPickup:
    opening: float
    previous: float
    picked: bool = False

    def update(self, trigger):
        desired = 1. - float(trigger)
        if abs(desired - self.opening) <= .02 or (self.previous - self.opening) * (desired - self.opening) <= 0:
            self.picked = True
        self.previous = desired
        return desired if self.picked else self.opening


class Clutch:
    """A fresh release is required after stale tracking or rejected motion."""

    def __init__(self):
        self.ready = False
        self.active = False
        self.hand_anchor = self.robot_anchor = self.last_hand = self.last_q = self.grip = None

    def disengage(self):
        self.ready = self.active = False

    def target(self, pressed, hand, measured, solver, trigger, *, held_opening=None):
        held_opening = float(measured[6]) if held_opening is None else float(held_opening)
        if not pressed:
            self.ready, self.active = True, False
            return None
        if not self.ready:
            return None
        if not self.active:
            self.hand_anchor = hand.copy()
            self.robot_anchor = solver.fk(measured[:6])
            self.last_hand = hand.copy()
            self.last_q = measured[:6].copy()
            self.grip = GripPickup(held_opening, 1 - trigger)
            solver.reset(measured[:6])
            self.active = True
            # First command is exactly the measured pose, with unchanged gripper.
            first = measured.copy()
            first[6] = held_opening
            return self.robot_anchor.copy(), first, True
        relative = np.linalg.inv(self.hand_anchor) @ hand
        target = self.robot_anchor @ relative
        delta = np.linalg.inv(self.last_hand) @ hand
        angle = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
        if np.linalg.norm(delta[:3, 3]) > .08 or angle > .5:
            raise IKRejected("Quest tracking jump/recenter; release and re-engage clutch")
        self.last_hand = hand.copy()
        opening = self.grip.update(trigger)
        q = np.asarray(solver.solve(target, opening * .07), dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise IKRejected("non-finite/invalid IK output")
        if np.max(np.abs(q - self.last_q)) > .20:
            raise IKRejected("IK branch jump >0.20 rad; release and re-engage clutch")
        if np.max(np.abs(q - measured[:6])) > .35:
            raise IKRejected("arm tracking error >0.35 rad; release and re-engage clutch")
        return target, np.r_[q, opening], False


class QuestHardware:
    """Share session feedback handles, own the sole VR motion-writing thread."""

    def __init__(self, arms, *, gripper_effort=1000):
        self.arms = arms
        self.locks, self.monitors, self.limits = [], {}, {}
        self.enabled = set()
        self.gripper_effort = gripper_effort
        self.last_targets = {}

    def open(self):
        from bimanual_vla.collection.replay import RemoteCommandMonitor
        from bimanual_vla.deployment.hardware_safety import install_can_send_guard, query_limits
        root = Path("~/.local/state/bimanual-vla/replay-locks").expanduser()
        root.mkdir(parents=True, exist_ok=True)
        for side, arm in sorted(self.arms.items()):
            interface = arm.GetCanName()
            lock = (root / (interface + ".lock")).open("a")
            self.locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.monitors[side] = RemoteCommandMonitor(interface)
            install_can_send_guard(arm)
            self.check_ownership()
            # Check observed streams before this writer starts transmitting.
            for name in ("GetArmJointCtrl", "GetArmGripperCtrl"):
                if float(getattr(getattr(arm, name)(), "Hz", 0) or 0) > 2:
                    raise RuntimeError(f"{side}: another control stream is active")
            self.limits[side], _ = query_limits(arm)

    def check_ownership(self):
        for monitor in self.monitors.values():
            monitor.check()

    def read(self):
        from bimanual_vla.collection.output import read_output_qpos
        from bimanual_vla.deployment.client import arm_status_dict, driver_enable_status_dict
        from bimanual_vla.deployment.hardware_safety import require_can_send_healthy
        self.check_ownership()
        require_can_send_healthy(self.arms)
        states = {}
        for side, arm in self.arms.items():
            status, driver = arm_status_dict(arm), driver_enable_status_dict(arm)
            if (not status["feedback_fresh"] or status["arm_status"] != 0 or status["err_code"]
                    or not driver["healthy"] or (side in self.enabled and not driver["ready"])):
                raise RuntimeError(f"{side}: unhealthy VR hardware: {status}; {driver}")
            if side in self.enabled and (status["ctrl_mode"] != 1 or status["mode_feed"] != 1):
                raise RuntimeError(f"{side}: lost CAN/MOVE_J mode")
            states[side] = read_output_qpos(arm)
        return states

    def wait_for_feedback(self, timeout_s=1.):
        """Let SDK receivers catch up after loading the native IK libraries."""
        from bimanual_vla.deployment.client import arm_status_dict, driver_enable_status_dict
        from bimanual_vla.deployment.hardware_safety import require_can_send_healthy
        deadline = time.monotonic()+timeout_s
        while True:
            self.check_ownership()
            require_can_send_healthy(self.arms)
            fresh = True
            for side, arm in self.arms.items():
                status, driver = arm_status_dict(arm), driver_enable_status_dict(arm)
                if ((status['feedback_fresh'] and (status['arm_status'] != 0 or status['err_code']))
                        or driver['faulted']):
                    raise RuntimeError(f'{side}: hardware fault during feedback initialization')
                fresh &= status['feedback_fresh'] and driver['healthy']
            if fresh or time.monotonic() >= deadline:
                return self.read()  # The usual full checks remain authoritative.
            time.sleep(.02)

    def require_ready(self, sides=None):
        """Admit existing CAN ownership; never release brakes or change modes."""
        from bimanual_vla.deployment.client import arm_status_dict, driver_enable_status_dict
        measured = self.read()
        for side in self.arms if sides is None else sides:
            arm = self.arms[side]
            status, driver = arm_status_dict(arm), driver_enable_status_dict(arm)
            if (not status['feedback_fresh'] or status['arm_status'] != 0 or status['err_code']
                    or not driver['ready'] or status['ctrl_mode'] != 1 or status['mode_feed'] != 1):
                raise RuntimeError(f'{side}: arm must already be enabled in CAN/MOVE_J; '
                                   'VR collection/reset does not automatically enable or switch modes')
        return measured

    def enable(self, side, measured):
        # Keep the controller API, but only adopt a ready arm. Enabling against
        # firmware's previous target is not a safe part of a VR/reset handoff.
        fresh = self.require_ready((side,))
        self.send(side, fresh[side])
        self.enabled.add(side)

    def send(self, side, target):
        arm = self.arms[side]
        target = np.asarray(target)
        validate_command(side, target, self.limits[side])
        self.check_ownership()
        arm.JointCtrl(*[round(float(q) * 180000 / np.pi) for q in target[:6]])
        arm.GripperCtrl(round(float(target[6]) * 70000), self.gripper_effort, 1, 0)
        self.last_targets[side] = target.copy()

    def emergency_stop(self):
        failures = []
        for side, arm in self.arms.items():
            try:
                arm.EmergencyStop(1)
            except Exception as exc:
                failures.append(f"{side}: {exc}")
        if failures:
            raise RuntimeError("; ".join(failures))

    def close(self):
        for monitor in self.monitors.values():
            monitor.close()
        for lock in self.locks:
            lock.close()
        self.monitors.clear()
        self.locks.clear()


class QuestController:
    def __init__(self, arms, model_dir, serial="", *, reader=None, hardware=None, solver_factory=QuestIK):
        self.hardware = hardware or QuestHardware(arms)
        self.reader = reader or QuestReader(serial)
        self.model_dir = model_dir
        self.solver_factory = solver_factory
        self.solvers = {}
        self.clutches = {side: Clutch() for side in arms}
        self._stop, self._estop = threading.Event(), threading.Event()
        self._estop_lock, self._estop_sent = threading.Lock(), False
        self._thread = None
        self.error = None
        self.status = "Waiting for fresh Quest tracking; release Y/B before engagement"
        self._record_lock = threading.Lock()
        self._events = None
        self._holds = {}
        self._hold_errors = {}
        self._last_sequence = None
        self.last_targets = {}
        self.collection_buttons = None  # RECAP must not consume collection shortcuts.
        self.ui_actions = queue.Queue()
        self._requests = queue.Queue()
        self._reset_busy = threading.Event()
        self._reset = None
        self.initial_joints = None
        self._feedback_lock = threading.Lock()
        self._feedback_sample = None

    def snapshot_feedback(self):
        self.check()
        with self._feedback_lock:
            sample = self._feedback_sample
            if sample is None or time.monotonic()-sample[0] > .15:
                raise RuntimeError('waiting for fresh robot feedback')
            return {s: q.copy() for s,q in sample[1].items()}

    def enable_collection_actions(self):
        from .actions import CollectionButtons
        self.collection_buttons = CollectionButtons()

    @property
    def resetting(self):
        return self._reset_busy.is_set()

    def request_reset(self, joints=None):
        self.check()
        if self.collection_buttons is None or self.initial_joints is None:
            raise RuntimeError('reset is available only in connected VR collection')
        if self.resetting:
            raise RuntimeError('reset is already running')
        sample = self.reader.snapshot()
        if sample is not None and (sample.buttons.get('Y') or sample.buttons.get('B')):
            raise RuntimeError('release Y/B before resetting')
        targets = self.initial_joints if joints is None else joints
        if set(targets) != set(self.clutches):
            raise ValueError('reset pose does not match connected arms')
        targets = {s: np.asarray(q, dtype=float).copy() for s, q in targets.items()}
        for s, q in targets.items():
            limits = self.hardware.limits[s]
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError(f'{s}: reset pose must contain six finite joints')
            # Match inference return handling: adjust the requested home, not
            # measured feedback or streamed teleoperation commands. The source
            # file/connection pose remains unchanged.
            targets[s] = np.clip(q, limits[:, 0], limits[:, 1])
            if not np.array_equal(q, targets[s]):
                logging.info('%s: reset target adjusted to driver limits: %s -> %s deg',
                             s, np.rad2deg(q).tolist(), np.rad2deg(targets[s]).tolist())
                self._event(event='reset_target_clipped', side=s,
                            before_rad=q.tolist(), after_rad=targets[s].tolist())
        # Queue first: the writer may see the busy flag on its next tick.
        self._requests.put(targets)
        self._reset_busy.set()

    def _step_reset(self, measured):
        """Runs only in the existing writer thread; recording is stopped by GUI."""
        from .actions import JointResetPlan
        if self._stop.is_set() or self._estop.is_set():
            return
        now = time.monotonic()
        if self._reset is None:
            targets = self._requests.get_nowait()
            # Validate BOTH arms before writing any hold. Never enable one arm
            # against an old firmware target while the other is being prepared.
            measured = self.hardware.require_ready()
            holds = {s: self.held_target(s, q) for s, q in measured.items()}
            for s, q in holds.items():
                validate_command(s, q, self.hardware.limits[s])
            self.hardware.enabled.update(measured)
            self._hold(measured)
            self._holds = holds
            now = time.monotonic()
            self._reset = dict(targets=targets, phase='settling', since=now,
                               stable=None, previous=measured, last=now)
        r = self._reset
        dt = now - r['last']
        if dt > .15:
            raise RuntimeError('reset writer stalled; reconnect after checking the arms')
        speed = max(np.max(abs(measured[s][:6]-r['previous'][s][:6])) / max(dt, 1e-6)
                    for s in measured)
        r['previous'], r['last'] = measured, now
        if r['phase'] in ('moving', 'arrival') and speed > .30:
            raise RuntimeError('reset measured joint speed exceeds 0.30 rad/s; reset aborted')
        if r['phase'] == 'moving':
            elapsed = now - r['since']
            next_holds = {}
            for s, plan in r['plans'].items():
                q = plan.at(elapsed)
                if np.max(abs(q-self._holds[s][:6])) > .10 * max(dt, 1e-6) + 1e-6:
                    raise RuntimeError(f'{s}: reset command exceeds per-cycle speed bound')
                if np.max(abs(measured[s][:6]-self._holds[s][:6])) > .05:
                    raise RuntimeError(f'{s}: reset tracking error exceeds 0.05 rad')
                if self.solvers[s].collides(q, self._holds[s][6]*.07):
                    raise RuntimeError(f'{s}: reset self collision')
                next_holds[s] = self._holds[s].copy()
                next_holds[s][:6] = q
            self._holds.update(next_holds)
            if elapsed >= max(p.duration for p in r['plans'].values()):
                r.update(phase='arrival', since=now, stable=None)
        else:
            error = max(np.max(abs(measured[s][:6]-self._holds[s][:6])) for s in measured)
            r['stable'] = (r['stable'] or now) if speed < .05 and error < .02 else None
            if now-r['since'] > 3.:
                raise RuntimeError('reset failed to settle within 3 seconds')
            if r['stable'] is not None and now-r['stable'] >= .25:
                if r['phase'] == 'arrival':
                    self._reset = None
                    self._reset_busy.clear()
                    self.collection_buttons.clear()
                    self.status = 'Reset complete; release Y/B before re-engaging'
                    self.ui_actions.put('reset_done')
                else:
                    plans = {s: JointResetPlan(self._holds[s][:6], q) for s,q in r['targets'].items()}
                    duration = max(p.duration for p in plans.values())
                    for plan in plans.values():
                        plan.duration = duration
                    # Reject known self collisions before starting the path.
                    for s, plan in plans.items():
                        if any(self.solvers[s].collides(plan.at(t), self._holds[s][6]*.07)
                               for t in np.linspace(0, plan.duration, 65)):
                            raise RuntimeError(f'{s}: reset path intersects robot geometry')
                    r.update(phase='moving', since=time.monotonic(), plans=plans,
                             last=time.monotonic())
        for s, q in self._holds.items():
            # Preflight/collision checks may take time. A stop raised during
            # those checks must prevent any further reset target writes.
            if self._stop.is_set() or self._estop.is_set():
                return
            self.hardware.send(s, q)
            self.last_targets[s] = q.copy()
        if self.resetting:
            self.status = 'Resetting; recording and VR pose control paused'

    def held_target(self, side, measured):
        target = measured.copy()
        if side in self.last_targets:
            target[6] = self.last_targets[side][6]
        return target

    def start(self):
        try:
            self.hardware.open()
            if self.collection_buttons is not None:
                self.initial_joints = {s: q[:6].copy() for s,q in self.hardware.read().items()}
            self.solvers = {side: self.solver_factory(self.model_dir, self.hardware.limits[side])
                            for side in self.clutches}
            self.hardware.wait_for_feedback()
            self.reader.start()
            self._thread = threading.Thread(target=self._run, name="quest-joint-writer", daemon=True)
            self._thread.start()
        except BaseException:
            try:
                self.reader.close()
            finally:
                self.hardware.close()
            raise

    def begin_episode(self):
        with self._record_lock:
            self._events = []

    def end_episode(self):
        with self._record_lock:
            events, self._events = self._events or [], None
        return json.dumps(events, separators=(",", ":"))

    def _event(self, **event):
        event["time"] = time.time()
        with self._record_lock:
            if self._events is not None:
                self._events.append(event)

    def _set_hold(self, side, measured):
        target = self.held_target(side, measured)
        try:
            validate_command(side, target, self.hardware.limits[side])
        except InvalidCommand as exc:
            # Neither clip measured joints nor replay an earlier target: either
            # could move the arm while reporting that it is being held.
            self._holds.pop(side, None)
            reason = f"{exc}; Cannot hold current pose; check joint position/zero. Return within limits, then release Y/B before re-engaging"
            if self._hold_errors.get(side) != reason:
                logging.warning("Quest hold unavailable: %s", reason)
                self._event(event="hold_unavailable", side=side, reason=reason)
            self._hold_errors[side] = reason
            return False
        if side not in self.hardware.enabled:
            # A rejected first engagement never adopted this arm, so it must
            # not create a new writer merely to send a hold.
            self._holds.pop(side, None)
            return False
        self._holds[side] = target
        self._hold_errors.pop(side, None)
        return True

    def _hold(self, measured):
        for side, clutch in self.clutches.items():
            if clutch.active:
                self._set_hold(side, measured[side])
            clutch.disengage()

    def _reject_motion(self, measured, exc):
        self._hold(measured)
        self.status = str(exc)
        logging.warning("Quest motion held/rejected: %s", exc)
        self._event(event="motion_rejected", reason=str(exc))

    def step(self):
        measured = self.hardware.read()
        with self._feedback_lock:
            self._feedback_sample = (time.monotonic(), {s:q.copy() for s,q in measured.items()})
        sample = self.reader.snapshot()
        if self.resetting:
            self._step_reset(measured)
            return
        if self.collection_buttons is not None:
            for action in self.collection_buttons.update(sample):
                self.ui_actions.put(action)
        if sample is None:
            self._hold(measured)
            self.status = 'No fresh VR tracking; wear the headset, dismiss overlays, and release Y/B'
        elif sample.source_time != self._last_sequence:
            self._last_sequence = sample.source_time
            commands = {}
            if not any(sample.buttons.get(key, False) for key in ('Y', 'B')):
                self.status = 'VR ready: hold Y (left) / B (right) to move; release to hold'
            elif not any(c.active for c in self.clutches.values()):
                self.status = 'Release Y/B first, then hold to re-anchor'
            try:
                for side, clutch in self.clutches.items():
                    hand, key = ("l", "Y") if side == "left" else ("r", "B")
                    if hand not in sample.poses:
                        self._hold(measured)
                        missing = [s for s in self.clutches
                                   if ('l' if s == 'left' else 'r') not in sample.poses]
                        self.status = ('Controller tracking lost: ' + ', '.join(missing)
                                       + '; open the teleop app, wake the controllers, and release Y/B')
                        commands.clear()
                        break
                    was_active = clutch.active
                    command = clutch.target(sample.buttons[key], adjusted_pose(sample.poses[hand]),
                                            measured[side], self.solvers[side], sample.buttons[side + "Trig"][0],
                                            held_opening=self.held_target(side, measured[side])[6])
                    if command is not None:
                        value = command[1]
                        validate_command(side, value, self.hardware.limits[side])
                        commands[side] = command
                    elif was_active:
                        self._set_hold(side, measured[side])
            except IKRejected as exc:
                self._reject_motion(measured, exc)
                commands.clear()
            # Check freshness, stop requests and hardware again AFTER solving.
            if (time.monotonic() - sample.received_monotonic > self.reader.max_age_s
                    or time.time() - sample.source_time > self.reader.max_age_s
                    or self._stop.is_set() or self._estop.is_set()):
                self._hold(measured)
                commands.clear()
            latest = self.reader.snapshot()
            if latest is None or any(
                ("l" if side == "left" else "r") not in latest.poses
                or not latest.buttons.get("Y" if side == "left" else "B", False)
                for side in commands
            ):
                self._hold(measured)
                commands.clear()
            self.hardware.read()
            for side, (pose, target, first) in commands.items():
                command_sample = sample
                if self._stop.is_set() or self._estop.is_set():
                    return
                if first:
                    if side not in self.hardware.enabled:
                        try:
                            self.hardware.enable(side, target)
                        except InvalidCommand as exc:
                            self._reject_motion(self.hardware.read(), exc)
                            break
                        # Enabling can take time; reanchor from fresh robot AND hand
                        # feedback instead of replaying the pre-enable snapshot.
                        fresh = self.hardware.read()[side]
                        latest = self.reader.snapshot()
                        hand, key = ("l", "Y") if side == "left" else ("r", "B")
                        clutch = self.clutches[side]
                        if (latest is None or hand not in latest.poses
                                or not latest.buttons.get(key) or self._stop.is_set() or self._estop.is_set()):
                            clutch.disengage()
                            self._set_hold(side, fresh)
                            continue
                        clutch.active = False
                        pose, target, _ = clutch.target(
                            True, adjusted_pose(latest.poses[hand]), fresh,
                            self.solvers[side], latest.buttons[side + "Trig"][0],
                            held_opening=self.held_target(side, fresh)[6])
                        command_sample = latest
                        measured[side] = fresh
                    self._event(event="engaged", side=side)
                try:
                    self.hardware.send(side, target)
                except InvalidCommand as exc:
                    self._reject_motion(self.hardware.read(), exc)
                    break
                self.last_targets[side] = target.copy()
                self.solvers[side].accept(target[:6])
                self.clutches[side].last_q = target[:6].copy()
                self._holds.pop(side, None)
                self._hold_errors.pop(side, None)
                self.status = f"VR active: {', '.join(s for s, c in self.clutches.items() if c.active)}"
                self._event(event="command", side=side, vr_time=command_sample.source_time,
                            vr_pose=command_sample.poses["l" if side == "left" else "r"].tolist(),
                            target_pose=pose.tolist(), joint_target=target.tolist(),
                            measured=measured[side].tolist())
        if self._stop.is_set() or self._estop.is_set():
            return
        for side, target in self._holds.items():
            self.hardware.send(side, target)
        if self._hold_errors:
            self.status = "; ".join(self._hold_errors.values())

    def _run(self):
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                if self._estop.is_set():
                    self._send_emergency_stop()
                    raise RuntimeError("VR emergency stop requested")
                self.step()
                self._stop.wait(max(0, 1 / 60 - (time.monotonic() - started)))
        except Exception as exc:
            self.error = exc
            self.status = str(exc)
            self._event(event="fault", reason=str(exc))
            logging.exception("Quest control stopped")
        finally:
            if self._estop.is_set():
                # Also covers a stop requested while step() was failing.
                self._send_emergency_stop()
            # Hold measured position only while fresh healthy feedback is available.
            if not self._estop.is_set():
                try:
                    measured = self.hardware.read()
                    for side in self.hardware.enabled:
                        if self._set_hold(side, measured[side]):
                            self.hardware.send(side, self._holds[side])
                except Exception:
                    logging.exception("Quest final hold unavailable")

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Quest control stopped: {self.error}") from self.error

    def _send_emergency_stop(self):
        with self._estop_lock:
            if self._estop_sent:
                return
            self.hardware.emergency_stop()
            self._estop_sent = True

    def emergency_stop(self):
        self._estop.set()
        # The writer thread sends the stop. If it has already exited (for
        # example after a fault), nothing else would, so send it here.
        if self._thread is None or not self._thread.is_alive():
            self._send_emergency_stop()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Quest writer did not stop; keep CAN connection open")
        try:
            self.reader.close()
        finally:
            self.hardware.close()
