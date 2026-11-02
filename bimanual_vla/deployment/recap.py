"""Exclusive RECAP ownership and local GUI command transport.

The runtime supplies blocking stop/settle operations. Ownership is granted only
after those operations return successfully; failures leave the session faulted.
This module never writes CAN and cannot bypass the controller's safety checks.
"""

import json
import os
from pathlib import Path
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class RecapSession:
    """Policy -> settled hold -> human -> settled hold -> fresh policy."""

    def __init__(self, *, stop_policy, start_human, stop_human, start_policy, event=None):
        self.stop_policy, self.start_human = stop_policy, start_human
        self.stop_human, self.start_policy = stop_human, start_policy
        self.event = event or (lambda **value: None)
        self.owner = "policy"
        self.epoch = 0
        self.barrier = time.monotonic()
        self.outcome = "unlabelled"
        self.error = None
        self.finished = False

    def _transition(self, owner):
        self.epoch += 1
        self.owner = owner
        self.barrier = time.monotonic()
        self.event(owner=owner, epoch=self.epoch, monotonic=self.barrier, outcome=self.outcome)

    def accepts_policy(self, captured_monotonic):
        return (not self.finished and not self.error and self.owner == "policy"
                and captured_monotonic > self.barrier)

    def request(self, command):
        if self.finished or self.error:
            raise RuntimeError("RECAP session is closed or faulted")
        if command not in ("takeover", "resume", "success", "failure", "aborted"):
            raise ValueError("unknown RECAP command")
        if command == "takeover" and self.owner == "human":
            return
        if command == "resume" and self.owner == "policy":
            return
        old = self.owner
        self._transition("transition")
        try:
            (self.stop_policy if old == "policy" else self.stop_human)()
            if command in ("success", "failure", "aborted"):
                self.outcome = command
                self.finished = True
                self._transition("hold")
            elif command == "takeover":
                self.start_human()
                self._transition("human")
            else:
                self.start_policy()
                # Install the barrier AFTER handoff, so pre-settle observations
                # and all in-flight policy results are rejected.
                self._transition("policy")
        except BaseException as exc:
            self.error = str(exc)
            self.outcome = "aborted"
            self._transition("hold")
            raise

    def status(self):
        return dict(owner=self.owner, epoch=self.epoch, outcome=self.outcome,
                    finished=self.finished, error=self.error)


class RecapMailbox:
    """One acknowledged GUI request at a time, scoped to this client session."""

    def __init__(self, directory, session_id):
        self.directory = Path(directory)
        self.session_id = session_id
        self.last_id = None

    def poll(self):
        try:
            data = json.loads((self.directory / "request.json").read_text())
        except FileNotFoundError:
            return None
        if data.get("session_id") != self.session_id:
            return None
        request_id = data.get("id")
        if not isinstance(request_id, str) or not request_id or request_id == self.last_id:
            return None
        self.last_id = request_id
        return data.get("command")

    def publish(self, session, **details):
        atomic_json(self.directory / "status.json", dict(
            session.status(), session_id=self.session_id, request_id=self.last_id,
            timestamp=time.time(), **details))
