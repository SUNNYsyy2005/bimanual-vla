"""Trajectory recording, slow replay, and home reset."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from piper_env import PiperBimanualEnv


class TrajectoryRecorder:
    def __init__(self):
        self._qpos: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._images: dict[str, list[np.ndarray]] = {}
        self._image_timestamps: dict[str, list[float]] = {}

    def start(self):
        self._qpos.clear()
        self._actions.clear()
        self._timestamps.clear()
        self._images.clear()
        self._image_timestamps.clear()

    def add(
        self,
        qpos: np.ndarray,
        action: np.ndarray,
        images: dict[str, np.ndarray],
        image_timestamps: dict[str, float] | None = None,
    ):
        now = time.time()
        self._timestamps.append(now)
        self._qpos.append(np.asarray(qpos, dtype=np.float32).copy())
        self._actions.append(np.asarray(action, dtype=np.float32).copy())
        for key, img in images.items():
            self._images.setdefault(key, []).append(np.asarray(img, dtype=np.uint8).copy())
            ts = now if image_timestamps is None else float(image_timestamps.get(key, now))
            self._image_timestamps.setdefault(key, []).append(ts)

    def to_numpy_dict(self, extras: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        data: dict[str, np.ndarray] = {
            "qpos": np.array(self._qpos, dtype=np.float32),
            "actions": np.array(self._actions, dtype=np.float32),
            "timestamps": np.array(self._timestamps, dtype=np.float64),
        }
        for key, frames in self._images.items():
            data[f"images_{key}"] = np.array(frames, dtype=np.uint8)
        for key, ts in self._image_timestamps.items():
            data[f"image_timestamps_{key}"] = np.array(ts, dtype=np.float64)
        if extras:
            for key, value in extras.items():
                data[key] = np.array(value)
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
        d = np.load(str(path))
        self.qpos = d["qpos"]
        self.actions = d["actions"]
        self.timestamps = d["timestamps"]

    def run(self, env: PiperBimanualEnv, speed: float = 0.5, dry_run: bool = False):
        T = len(self.actions)
        print(f"Replaying {T} steps at {speed:.0%} speed. dry_run={dry_run}")
        for i in range(T):
            t0 = time.time()
            action = self.actions[i]
            if dry_run:
                print(f"  step {i:04d}: left_joints={action[:6].round(3)} gripper={action[6]:.3f}")
            else:
                env.step(action)
            if i < T - 1:
                dt = (self.timestamps[i + 1] - self.timestamps[i]) / speed
                sleep = dt - (time.time() - t0)
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
