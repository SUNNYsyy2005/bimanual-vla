"""Quest adapter called by the deployment control thread (no second writer)."""

import time

import numpy as np

from bimanual_vla.collection.quest.control import QuestController, QuestHardware
from bimanual_vla.deployment.hardware_safety import effective_limits, NOMINAL_JOINT_LIMITS_RAD


class RecapQuest:
    def __init__(self, arms, model_dir, serial, *, gripper_effort=1000):
        # The deployment process already owns the device guard and CAN handles.
        # Never call QuestHardware.open()/start() here: those belong to the
        # standalone collection writer, which must remain disconnected.
        self.hardware = QuestHardware(arms, gripper_effort=gripper_effort)
        self.hardware.limits = {side: effective_limits(arm, NOMINAL_JOINT_LIMITS_RAD) for side, arm in arms.items()}
        self.controller = QuestController(arms, model_dir, serial, hardware=self.hardware)
        self.controller.solvers = {side: self.controller.solver_factory(model_dir, self.hardware.limits[side])
                                   for side in arms}
        self.controller.reader.start()
        self.last_sent = {}
        self.sent_at = None
        self._send = self.hardware.send
        self.hardware.send = self._record_send

    def _record_send(self, side, target):
        self._send(side, target)
        self.last_sent[side] = np.asarray(target).copy()
        self.sent_at = time.monotonic()

    def connection_status(self):
        """Return lightweight headset and per-controller tracking state."""
        reader = self.controller.reader
        try:
            sample = reader.snapshot()
        except Exception as exc:
            return {
                "headset": "disconnected",
                "left_controller": "disconnected",
                "right_controller": "disconnected",
                "detail": str(exc),
            }
        if sample is None:
            process = getattr(reader, "_process", None)
            connected = process is not None and process.poll() is None
            state = "connected; waiting for tracking" if connected else "disconnected"
            return {
                "headset": "connected" if connected else "disconnected",
                "left_controller": "waiting" if connected else "disconnected",
                "right_controller": "waiting" if connected else "disconnected",
                "detail": state,
            }
        return {
            "headset": "connected",
            "left_controller": "tracked" if "l" in sample.poses else "not tracked",
            "right_controller": "tracked" if "r" in sample.poses else "not tracked",
            "detail": "fresh tracking",
        }

    def hold(self, targets):
        self.hardware.read()
        for side, target in targets.items():
            self.hardware.send(side, target)

    def settle(self, targets, *, timeout_s=3.0):
        """Verify measured stop; planner zero velocity alone is insufficient."""
        deadline = time.monotonic() + timeout_s
        previous = self.hardware.read()
        previous_at = time.monotonic()
        stable_since = None
        while time.monotonic() < deadline:
            self.hold(targets)
            time.sleep(.02)
            measured = self.hardware.read()
            now = time.monotonic()
            speed = max(np.max(np.abs(measured[s][:6] - previous[s][:6])) / (now - previous_at)
                        for s in measured)
            error = max(np.max(np.abs(measured[s][:6] - targets[s][:6])) for s in measured)
            stable_since = (stable_since or now) if speed < .05 and error < .03 else None
            if stable_since is not None and now - stable_since >= .25:
                return measured
            previous, previous_at = measured, now
        raise RuntimeError("RECAP handoff: measured arms did not settle within 3 seconds")

    def wait_for_tracking(self, *, timeout_s=0.8):
        """Wait briefly for a fresh pose containing every configured controller."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        missing = list(self.controller.clutches)
        while True:
            try:
                sample = self.controller.reader.snapshot()
            except Exception as exc:
                return False, str(exc)
            if sample is not None:
                missing = [
                    side for side in self.controller.clutches
                    if ("l" if side == "left" else "r") not in sample.poses
                ]
                if not missing:
                    return True, "VR tracking ready"
            if time.monotonic() >= deadline:
                if sample is None:
                    return False, "no fresh Quest tracking"
                return False, "missing controller tracking: " + ", ".join(missing)
            time.sleep(0.02)

    def engage(self, targets):
        ready, reason = self.wait_for_tracking(timeout_s=0.2)
        if not ready:
            raise RuntimeError("RECAP takeover requires fresh tracking of both hand controllers (" + reason + ")")
        sample = self.controller.reader.snapshot()
        if sample is None or any(("l" if s == "left" else "r") not in sample.poses for s in targets):
            raise RuntimeError("RECAP takeover requires fresh tracking of both hand controllers")
        self.hardware.enabled = set(targets)
        measured = self.settle(targets)
        self.controller.last_targets = {s: targets[s].copy() for s in targets}
        self.controller._holds = {s: self.controller.held_target(s, measured[s]) for s in targets}
        self.controller._last_sequence = None
        for clutch in self.controller.clutches.values():
            clutch.disengage()
        # Reanchoring occurs on a fresh release/press of Y/B, not using a hand
        # snapshot captured before the robot's braking movement.

    def step(self):
        before = self.sent_at
        self.controller.step()
        return self.sent_at != before

    def stop(self):
        measured = self.hardware.read()
        targets = {s: self.controller.held_target(s, measured[s]) for s in measured}
        for clutch in self.controller.clutches.values():
            clutch.disengage()
        self.settle(targets)
        self.controller._holds = targets
        return targets

    def close(self):
        self.controller.reader.close()
