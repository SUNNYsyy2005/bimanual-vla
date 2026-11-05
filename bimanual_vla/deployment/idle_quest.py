"""Unrecorded Quest teleoperation while the inference process is stopped."""

from bimanual_vla.collection.can_identity import resolve_can_selector
from bimanual_vla.collection.quest import DEFAULT_MODEL_DIR
from bimanual_vla.collection.quest.control import (
    DEFAULT_TELEOP_SPEED_PCT, QuestController, QuestHardware,
)
from bimanual_vla.device_guard import RuntimeDeviceGuard, check_devices


class IdleButtons:
    """One A press per release; reconnects and tracking gaps require a release."""

    def __init__(self):
        self.armed = False
        self.sequence = None

    def update(self, sample):
        if sample is None or "r" not in sample.poses or "A" not in sample.buttons:
            self.armed = False
            return []
        if sample.source_time == self.sequence:
            return []
        self.sequence = sample.source_time
        if not sample.buttons["A"]:
            self.armed = True
            return []
        armed, self.armed = self.armed, False
        if armed:
            return ["return_saved_home"]
        return []


class IdleQuestHardware(QuestHardware):
    def __init__(self, arms, guard=None):
        super().__init__(arms)
        self.guard = guard

    def check_ownership(self):
        super().check_ownership()
        if self.guard is not None and self.guard.fault:
            raise RuntimeError(self.guard.fault)

    def enable(self, side, measured):
        # Only an explicit Y/B engagement may enable the selected arm. Connecting
        # alone never releases brakes or changes the robot's control mode.
        fresh = self.prepare_for_reset(
            self.read(), speed_pct=DEFAULT_TELEOP_SPEED_PCT, sides=(side,))
        self.send(side, fresh[side])


class IdleQuestSession:
    """Own CAN feedback handles and exactly one writer, without a recorder."""

    def __init__(self, *, arms, cameras, arm_mode, arm_side, model_dir, serial, resolver=""):
        self.request = dict(purpose="deployment", arm_mode=arm_mode, arm_side=arm_side,
                            arms=dict(arms), cameras=dict(cameras), resolver=resolver)
        self.model_dir, self.serial = model_dir or DEFAULT_MODEL_DIR, serial
        self.arms = {}
        self.controller = None
        self.guard = None

    def start(self):
        from bimanual_vla.deployment.client import connect_piper

        selectors = {side: resolve_can_selector(value) for side, value in self.request["arms"].items()}
        if len(set(selectors.values())) != len(selectors):
            raise ValueError("Left and right CAN selectors resolve to the same interface")
        request = {**self.request, "arms": selectors}
        resolved = check_devices(**request)
        if resolved is not None:
            selectors = resolved.arms
            self.guard = RuntimeDeviceGuard(baseline=resolved, **request)
            self.guard.start()
        # The GUI retains this session even on a partial startup failure, and
        # closes it before accepting any other hardware owner.
        for side, selector in selectors.items():
            self.arms[side] = connect_piper(selector)
        hardware = IdleQuestHardware(self.arms, self.guard)
        self.controller = QuestController(self.arms, self.model_dir, self.serial, hardware=hardware)
        self.controller.collection_buttons = IdleButtons()
        self.controller.start()

    def connection_status(self):
        reader = self.controller.reader
        sample = reader.snapshot()
        process = reader._process
        connected = sample is not None or (process is not None and process.poll() is None)
        return {
            "vr_connection": "connected" if connected else "disconnected",
            "controller_tracking": {
                side: ("tracked" if sample is not None and hand in sample.poses else "not tracked")
                for side, hand in (("left", "l"), ("right", "r"))
            },
        }

    def close(self):
        # If the writer cannot stop, keep all handles/locks and propagate the
        # failure. The caller must not launch another controller in that case.
        if self.controller is not None:
            self.controller.close()
        if self.guard is not None:
            self.guard.close()
        for side in tuple(self.arms):
            self.arms[side].DisconnectPort()
            del self.arms[side]
