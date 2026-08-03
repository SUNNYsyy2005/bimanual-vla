"""Trajectory recording, slow replay, and home reset."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from piper_env import PiperBimanualEnv
except ModuleNotFoundError:  # Allow contract/recorder tests without the hardware SDK.
    PiperBimanualEnv = Any  # type: ignore[misc,assignment]


class TrajectoryRecorder:
    """Record state/action/image streams with independent source timestamps."""

    def __init__(self):
        self._qpos: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._joint_qpos: list[np.ndarray] = []
        self._joint_qpos_presence: list[bool] = []
        self._state_timestamps: list[float] = []
        self._action_timestamps: list[float] = []
        self._images: dict[str, list[np.ndarray]] = {}
        self._image_timestamps: dict[str, list[float]] = {}

    def start(self):
        self._qpos.clear()
        self._actions.clear()
        self._joint_qpos.clear()
        self._joint_qpos_presence.clear()
        self._state_timestamps.clear()
        self._action_timestamps.clear()
        self._images.clear()
        self._image_timestamps.clear()

    def add(
        self,
        qpos: np.ndarray,
        action: np.ndarray,
        images: dict[str, np.ndarray],
        image_timestamps: dict[str, float] | None = None,
        *,
        state_timestamp: float | None = None,
        action_timestamp: float | None = None,
        joint_qpos: np.ndarray | None = None,
    ):
        state_ts = time.time() if state_timestamp is None else float(state_timestamp)
        action_ts = state_ts if action_timestamp is None else float(action_timestamp)
        if not np.isfinite(state_ts) or not np.isfinite(action_ts):
            raise ValueError("state_timestamp/action_timestamp must be finite")
        if self._state_timestamps and state_ts <= self._state_timestamps[-1]:
            raise ValueError("state_timestamp must be strictly increasing")
        if self._action_timestamps and action_ts <= self._action_timestamps[-1]:
            raise ValueError("action_timestamp must be strictly increasing")
        state = np.asarray(qpos, dtype=np.float32)
        command = np.asarray(action, dtype=np.float32)
        if not np.isfinite(state).all() or not np.isfinite(command).all():
            raise ValueError("state/action contains NaN or Inf")
        self._state_timestamps.append(state_ts)
        self._action_timestamps.append(action_ts)
        self._qpos.append(state.copy())
        self._actions.append(command.copy())
        diagnostic = None if joint_qpos is None else np.asarray(joint_qpos, dtype=np.float32)
        if diagnostic is not None and not np.isfinite(diagnostic).all():
            raise ValueError("joint_qpos contains NaN or Inf")
        self._joint_qpos_presence.append(diagnostic is not None)
        if diagnostic is not None:
            self._joint_qpos.append(diagnostic.copy())
        for key, img in images.items():
            self._images.setdefault(key, []).append(np.asarray(img, dtype=np.uint8).copy())
            ts = state_ts if image_timestamps is None else float(image_timestamps.get(key, state_ts))
            if not np.isfinite(ts):
                raise ValueError(f"image timestamp for {key} must be finite")
            self._image_timestamps.setdefault(key, []).append(ts)

    def to_numpy_dict(self, extras: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        state_timestamp = np.asarray(self._state_timestamps, dtype=np.float64)
        action_timestamp = np.asarray(self._action_timestamps, dtype=np.float64)
        data: dict[str, np.ndarray] = {
            "qpos": np.asarray(self._qpos, dtype=np.float32),
            "actions": np.asarray(self._actions, dtype=np.float32),
            # Compatibility alias. New consumers must use the explicit fields.
            "timestamps": state_timestamp.copy(),
            "state_timestamp": state_timestamp,
            "action_timestamp": action_timestamp,
        }
        if any(self._joint_qpos_presence) and not all(self._joint_qpos_presence):
            raise ValueError("joint_qpos must be present for every frame or omitted")
        if self._joint_qpos_presence and all(self._joint_qpos_presence):
            data["joint_qpos"] = np.asarray(self._joint_qpos, dtype=np.float32)
        for key, frames in self._images.items():
            data[f"images_{key}"] = np.asarray(frames, dtype=np.uint8)
        for key, timestamps in self._image_timestamps.items():
            data[f"image_timestamps_{key}"] = np.asarray(timestamps, dtype=np.float64)
        if extras:
            for key, value in extras.items():
                data[key] = np.asarray(value)
        return data

    def save(self, path: str | Path, extras: dict[str, Any] | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **self.to_numpy_dict(extras=extras))
        print(f"Saved {len(self._qpos)} steps → {path}")

    def __len__(self):
        return len(self._qpos)


class TrajectoryReplayer:
    def __init__(self, path: str | Path):
        data = np.load(str(path))
        self.qpos = data["qpos"]
        self.actions = data["actions"]
        self.timestamps = data["state_timestamp"] if "state_timestamp" in data else data["timestamps"]

    def run(self, env: PiperBimanualEnv, speed: float = 0.5, dry_run: bool = False):
        total = len(self.actions)
        print(f"Replaying {total} steps at {speed:.0%} speed. dry_run={dry_run}")
        for index in range(total):
            started = time.time()
            action = self.actions[index]
            if dry_run:
                print(f"  step {index:04d}: left_joints={action[:6].round(3)} gripper={action[6]:.3f}")
            else:
                env.step(action)
            if index < total - 1:
                delay = (self.timestamps[index + 1] - self.timestamps[index]) / speed
                sleep = delay - (time.time() - started)
                if sleep > 0:
                    time.sleep(sleep)
        print("Replay complete.")


def home_reset(env: PiperBimanualEnv, speed_pct: int = 15, wait_s: float = 3.0):
    env.left.set_speed_pct(speed_pct)
    env.right.set_speed_pct(speed_pct)
    env.go_home()
    print(f"Home reset sent. Waiting {wait_s}s for motion to complete...")
    time.sleep(wait_s)
    env.left.set_speed_pct(30)
    env.right.set_speed_pct(30)
    print("Home reset done.")
