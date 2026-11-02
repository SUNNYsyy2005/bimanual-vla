"""Read the official Quest APK protocol directly through ADB, without ROS."""

from dataclasses import dataclass
import re
import subprocess
import threading
import time

import numpy as np


@dataclass(frozen=True)
class Snapshot:
    sequence: int
    received_monotonic: float
    source_time: float
    poses: dict
    buttons: dict


def parse_payload(payload):
    transforms, raw_buttons = payload.strip().split("&")
    poses = {}
    for entry in transforms.split("|"):
        if not entry.strip():
            continue
        hand, values = entry.split(":")
        if hand not in ("l", "r"):
            raise ValueError("unknown hand")
        matrix = np.array([float(v) for v in values.split()]).reshape(4, 4)
        rotation = matrix[:3, :3]
        if (not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-4)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=.02)
                or not np.isclose(np.linalg.det(rotation), 1, atol=.02)):
            raise ValueError("invalid tracked transform")
        poses[hand] = matrix
    tokens = [token.strip() for token in raw_buttons.split(",")]
    buttons = {}
    for marker, names in (("L", ("X", "Y")), ("R", ("A", "B"))):
        if marker in tokens:
            buttons.update({name: name in tokens for name in names})
    for token in tokens:
        fields = token.split()
        if len(fields) > 1:
            values = tuple(float(v) for v in fields[1:])
            if not np.isfinite(values).all():
                raise ValueError("invalid analog button")
            buttons[fields[0]] = values
    for hand, side, clutch in (("l", "left", "Y"), ("r", "right", "B")):
        if hand in poses:
            trigger = buttons.get(side + "Trig", ())
            if clutch not in buttons or len(trigger) != 1 or not 0 <= trigger[0] <= 1:
                raise ValueError("missing hand buttons/trigger")
    return poses, buttons


def adjusted_pose(pose):
    # Same basis and controller orientation as official pub_pose.py.
    basis = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.]])
    controller = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]])
    return basis @ pose @ controller


class QuestReader:
    """No cached pose is restamped. Old queued logcat entries are rejected."""

    def __init__(self, serial="", max_age_s=.2, *, launch_app=True):
        self.serial = serial
        self.max_age_s = max_age_s
        self.launch_app = launch_app
        self._lock = threading.Lock()
        self._snapshot = None
        self._error = None
        self._process = None
        self._thread = None
        self._stopping = False
        self._clock_offset = 0.

    def start(self):
        from .devices import discover_devices, select_serial, launch_teleop_app
        self.serial = select_serial(discover_devices(), self.serial)
        if self.launch_app:
            launch_teleop_app(self.serial)
        base = ["adb", "-s", self.serial]
        # Android's wall clock can differ from the host. Measure it before streaming.
        before = time.time()
        remote = subprocess.check_output(base + ["shell", "date", "+%s.%N"], timeout=5, text=True)
        after = time.time()
        self._clock_offset = (before + after) / 2 - float(remote.strip())
        if after - before > self.max_age_s:
            raise RuntimeError("ADB clock probe too slow; check USB connection")
        self._process = subprocess.Popen(
            base + ["logcat", "-v", "epoch", "-T", "1", "wE9ryARX:I", "*:S"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._read, name="quest-adb", daemon=True)
        self._thread.start()

    def ingest(self, payload, source_time, now=None):
        now = time.time() if now is None else now
        if not -.05 <= now - source_time <= self.max_age_s:
            return False
        poses, buttons = parse_payload(payload)
        with self._lock:
            previous = self._snapshot
            if previous is not None and source_time <= previous.source_time:
                return False
            self._snapshot = Snapshot(
                1 if previous is None else previous.sequence + 1,
                time.monotonic(), source_time, poses, buttons)
        return True

    def _read(self):
        try:
            for line in self._process.stdout:
                match = re.match(r"\s*(\d+\.\d+)\s+.*?wE9ryARX\s*:\s*(.*)", line)
                if match:
                    try:
                        self.ingest(match[2], float(match[1]) + self._clock_offset)
                    except (ValueError, OverflowError):
                        # Invalid tracking invalidates the current snapshot immediately.
                        with self._lock:
                            self._snapshot = None
            if not self._stopping:
                self._error = RuntimeError("Quest ADB stream disconnected")
        except Exception as exc:
            if not self._stopping:
                self._error = exc

    def snapshot(self):
        if self._error:
            raise RuntimeError(str(self._error)) from self._error
        with self._lock:
            sample = self._snapshot
            if sample is None or time.monotonic() - sample.received_monotonic > self.max_age_s:
                return None
            return Snapshot(sample.sequence, sample.received_monotonic, sample.source_time,
                            {k: v.copy() for k, v in sample.poses.items()}, dict(sample.buttons))

    def close(self):
        self._stopping = True
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("Quest reader did not stop")
