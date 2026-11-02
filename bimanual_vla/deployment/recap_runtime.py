"""Deployment integration for explicit VR takeover and policy resumption."""

import time

import numpy as np

from bimanual_vla.collection.quest import DEFAULT_MODEL_DIR
from .recap import RecapMailbox, RecapSession
from .recap_quest import RecapQuest


def _command_opening_fraction(opening_m):
    fraction = float(opening_m) / .07
    # A float32 0.07 m becomes 1.000000004 in fraction units. Admit only
    # representation noise at endpoints; malformed commands still fail closed.
    if not np.isfinite(fraction) or not -1e-6 <= fraction <= 1 + 1e-6:
        raise RuntimeError("invalid commanded gripper opening during RECAP handoff")
    return float(np.clip(fraction, 0., 1.))


class RecapRuntime:
    def __init__(self, execution, args, recorder, monitoring, reset_policy):
        self.execution, self.args = execution, args
        self.recorder, self.monitoring = recorder, monitoring
        self.reset_policy = reset_policy
        self.sides = tuple(execution.pipers)
        self.quest = RecapQuest(execution.pipers, args.recap_quest_model_dir or DEFAULT_MODEL_DIR,
                                args.recap_adb_serial, gripper_effort=args.gripper_effort)
        self.targets = None
        self.protocol = None
        self.mailbox = RecapMailbox(args.recap_control_dir, args.recap_session_id)
        self.session = RecapSession(stop_policy=self.stop_policy, start_human=self.start_human,
                                   stop_human=self.stop_human, start_policy=self.start_policy,
                                   event=self.event)
        self.publish()

    def event(self, **event):
        self.monitoring.record("recap_handoff", **event)
        self.recorder.update_metadata({"recap": dict(version=1, **self.session.status())})

    def publish(self):
        servo = self.execution.servo_loop
        has_command = servo is not None and servo.last_published_command() is not None
        self.mailbox.publish(self.session, vr_status=self.quest.controller.status,
                             recording=str(self.recorder.run_dir or ""),
                             ready=(self.protocol is not None and (self.targets is not None
                                    or has_command)))

    def stop_policy(self):
        controller = self.execution
        controller.discard_pending_actions("RECAP ownership handoff", kind="other")
        servo = controller.servo_loop
        if not controller.close_servo():
            raise RuntimeError("policy servo did not stop safely; RECAP takeover refused")
        controller.close_ik_workers()
        command = servo.last_published_command() if servo is not None else None
        self.quest.hardware.read()
        if command is not None:
            self.targets = {s: np.r_[command[i*7:i*7+6], _command_opening_fraction(command[i*7+6])]
                            for i, s in enumerate(self.sides)}
        elif self.targets is None:
            # No model command has ever been sent. Do not infer the holding
            # force/target from a contact-limited measured aperture.
            raise RuntimeError("RECAP takeover is available after the first policy command")
        self.quest.hardware.enabled = set(controller.robot_enabled)
        if set(self.sides) != self.quest.hardware.enabled:
            raise RuntimeError("RECAP requires both configured arms enabled and healthy")
        self.quest.settle(self.targets)

    def start_human(self):
        self.quest.engage(self.targets)

    def stop_human(self):
        self.targets = self.quest.stop()

    def start_policy(self):
        self.execution = self.reset_policy()
        openings = {side: float(target[6]) for side, target in self.targets.items()}
        self.execution.resume_gripper_targets = openings.copy()
        self.execution._filtered_gripper_opening.update(openings)

    def poll(self, protocol):
        self.protocol = protocol
        command = self.mailbox.poll()
        if command is not None:
            if protocol is None:
                raise RuntimeError("RECAP requires a connected policy and completed startup")
            self.session.request(command)
            self.recorder.update_metadata({"recap": dict(version=1, **self.session.status())})
        self.publish()
        return self.session.finished

    def accepts(self, completion):
        return self.session.accepts_policy(completion.launch.captured_monotonic)

    def step_human(self, protocol):
        sent = self.quest.step()
        from .client import read_output_state
        states = [read_output_state(self.execution.pipers[s], max_feedback_age_s=self.args.max_feedback_age_s)
                  for s in self.sides]
        active = any(c.active for c in self.quest.controller.clutches.values())
        commands = [self.quest.last_sent.get(s) for s in self.sides]
        complete = all(c is not None for c in commands)
        self.recorder.record_control_tick(
            timestamp=time.time(), monotonic_timestamp=time.monotonic(),
            delivery_state=np.concatenate([v[0] for v in states]), qpos=np.concatenate([v[1] for v in states]),
            command_sent=sent, action_dim=protocol.action_dim, absolute_dim=10 * len(states),
            command_joints_rad=np.concatenate([v[:6] for v in commands]) if complete else None,
            command_gripper_fraction=np.array([v[6] for v in commands]) if complete else None,
            command_monotonic_timestamp=self.quest.sent_at if sent else None,
            command_hold=not active, execution_state="recap_human", recap_owner="human",
            recap_epoch=self.session.epoch, recap_valid=sent and active,
        )

    def hold_until_policy(self):
        if self.execution.servo_loop is not None:
            # The resumed policy owns the arms. The handoff pose is stale from
            # now on: a later servo restart (e.g. reconnect) must not drive the
            # arms back to it, and a later takeover must start from the policy.
            self.targets = None
        elif self.targets is not None:
            self.quest.hold(self.targets)

    def close(self):
        try:
            if self.session.owner == "human":
                self.stop_human()
            if not self.session.finished:
                self.session.outcome = "aborted"
                self.session.finished = True
                self.session._transition("hold")
            self.recorder.update_metadata({"recap": dict(version=1, **self.session.status())})
            self.publish()
        finally:
            self.quest.close()
