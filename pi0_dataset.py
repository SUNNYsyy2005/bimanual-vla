"""Small LeRobot v2.1 dataset writer used by the Piper collectors.

The writer stores camera streams as MP4 files and numeric observations/actions
in one parquet file per episode.  Parquet timestamps are generated from the
configured FPS, as required by LeRobot; original host timestamps are retained
only in the optional raw NPZ copy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LEROBOT_CODEBASE_VERSION = "v2.1"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_VIDEO_CODEC = "mp4v"

BIMANUAL_JOINT_NAMES = [
    "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
    "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper",
]


def single_arm_joint_names(side: str = "right") -> list[str]:
    side = side.lower()
    if side not in {"left", "right"}:
        raise ValueError(f"arm side must be left or right, got {side!r}")
    return [
        f"{side}_joint_1", f"{side}_joint_2", f"{side}_joint_3", f"{side}_joint_4",
        f"{side}_joint_5", f"{side}_joint_6", f"{side}_gripper",
    ]


def derive_absolute_actions(qpos: np.ndarray, action_offset: int = 1) -> np.ndarray:
    """Return future absolute joint targets with end-of-episode padding.

    For offset=1, action[t] is qpos[t+1].  The final action repeats the final
    state.  OpenPI's delta-joint transform may then subtract the current state
    from the first six joint dimensions while leaving the gripper absolute.
    """
    states = np.asarray(qpos, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"qpos must be rank 2, got shape={states.shape}")
    if len(states) == 0:
        raise ValueError("qpos is empty")
    if action_offset < 0:
        raise ValueError("action_offset must be >= 0")
    indices = np.minimum(np.arange(len(states)) + int(action_offset), len(states) - 1)
    return states[indices].copy()


class Pi0LeRobotDatasetWriter:
    """Append teleoperation episodes to a LeRobot v2.1 video dataset."""

    def __init__(
        self,
        root: str | Path,
        *,
        fps: int,
        robot_type: str,
        state_names: list[str],
        action_names: list[str] | None = None,
        camera_keys: list[str],
        image_hw: tuple[int, int] = (224, 224),
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        save_raw_npz: bool = True,
        val_ratio: float = 0.1,
        action_semantics: str = "absolute_joint_position",
        action_offset: int = 0,
    ):
        self.root = Path(root).expanduser()
        self.fps = int(fps)
        self.robot_type = str(robot_type)
        self.state_names = list(state_names)
        self.action_names = list(action_names or state_names)
        self.camera_keys = list(camera_keys)
        self.image_hw = tuple(int(x) for x in image_hw)
        self.chunk_size = int(chunk_size)
        self.save_raw_npz = bool(save_raw_npz)
        self.val_ratio = float(val_ratio)
        self.action_semantics = str(action_semantics)
        self.action_offset = int(action_offset)

        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not self.state_names or not self.action_names:
            raise ValueError("state_names and action_names must be non-empty")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be non-empty and unique")
        if len(self.image_hw) != 2 or min(self.image_hw) <= 0:
            raise ValueError("image_hw must be (height, width)")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.action_offset < 0:
            raise ValueError("action_offset must be >= 0")

        self.meta_dir = self.root / "meta"
        self.data_dir = self.root / "data"
        self.video_dir = self.root / "videos"
        self.raw_dir = self.root / "raw"
        for path in (self.meta_dir, self.data_dir, self.video_dir):
            path.mkdir(parents=True, exist_ok=True)
        if self.save_raw_npz:
            self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.info_path = self.meta_dir / "info.json"
        self.tasks_path = self.meta_dir / "tasks.jsonl"
        self.episodes_path = self.meta_dir / "episodes.jsonl"
        self.episodes_stats_path = self.meta_dir / "episodes_stats.jsonl"
        self.norm_stats_path = self.meta_dir / "openpi_norm_stats.json"

        self.tasks: dict[str, int] = {}
        self.info = self._load_or_init_info()
        self._load_existing_tasks()
        self._validate_existing_dataset()

    def append_episode(
        self,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
        images: dict[str, np.ndarray],
        task_name: str,
        instruction: str,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        capture_timestamps = np.asarray(timestamps, dtype=np.float64)
        self._validate_episode(states, actions, capture_timestamps, images)

        frame_count = len(states)
        canonical_timestamps = np.arange(frame_count, dtype=np.float32) / self.fps
        normalized_images = {
            key: self._ensure_rgb_hwc_uint8(images[key], expected_frames=frame_count)
            for key in self.camera_keys
        }
        task_name = str(task_name).strip() or "single_arm_task"
        instruction = str(instruction).strip() or task_name.replace("_", " ")

        episode_index = int(self.info["total_episodes"])
        global_offset = int(self.info["total_frames"])
        task_index = self._get_task_index(instruction)
        chunk_name = self._chunk_name(episode_index)
        chunk_data_dir = self.data_dir / chunk_name
        chunk_data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_data_dir / f"episode_{episode_index:06d}.parquet"

        if parquet_path.exists():
            raise FileExistsError(f"episode parquet already exists: {parquet_path}")

        if self.save_raw_npz:
            raw_path = self.raw_dir / f"episode_{episode_index:06d}.npz"
            raw_payload: dict[str, Any] = {
                "observation.state": states,
                "action": actions,
                "timestamp": canonical_timestamps,
                "capture_timestamps": capture_timestamps,
                "episode_index": np.full(frame_count, episode_index, dtype=np.int64),
                "frame_index": np.arange(frame_count, dtype=np.int64),
                "index": np.arange(global_offset, global_offset + frame_count, dtype=np.int64),
                "task_name": np.asarray(task_name),
                "instruction": np.asarray(instruction),
                "success": np.asarray(bool(success), dtype=np.bool_),
                "action_semantics": np.asarray(self.action_semantics),
                "action_offset": np.asarray(self.action_offset, dtype=np.int64),
            }
            for key, value in normalized_images.items():
                raw_payload[f"observation.images.{key}"] = value
            if metadata:
                for key, value in metadata.items():
                    raw_payload[f"meta.{key}"] = np.asarray(value)
            np.savez_compressed(raw_path, **raw_payload)

        self._write_episode_videos(episode_index, normalized_images)
        self._write_episode_parquet(
            parquet_path=parquet_path,
            episode_index=episode_index,
            task_index=task_index,
            global_offset=global_offset,
            states=states,
            actions=actions,
            timestamps=canonical_timestamps,
        )

        self._append_jsonl(
            self.episodes_path,
            {
                "episode_index": episode_index,
                "tasks": [instruction],
                "length": frame_count,
                "task_name": task_name,
                "success": bool(success),
                "action_semantics": self.action_semantics,
                "action_offset": self.action_offset,
                **(metadata or {}),
            },
        )
        self._append_jsonl(
            self.episodes_stats_path,
            {
                "episode_index": episode_index,
                "stats": {
                    "observation.state": self._stat_dict(states),
                    "action": self._stat_dict(actions),
                    "timestamp": self._stat_dict(canonical_timestamps[:, None]),
                },
            },
        )

        self.info["total_episodes"] = episode_index + 1
        self.info["total_frames"] = global_offset + frame_count
        self.info["total_tasks"] = len(self.tasks)
        self.info["total_videos"] = self.info["total_episodes"] * len(self.camera_keys)
        self.info["total_chunks"] = math.ceil(self.info["total_episodes"] / self.chunk_size)
        self.info["splits"] = self._build_splits(self.info["total_episodes"])
        self._write_info()
        self._recompute_openpi_norm_stats()
        return episode_index

    def _validate_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
        images: dict[str, np.ndarray],
    ) -> None:
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError("states/actions must be rank-2 arrays")
        if len(states) == 0:
            raise ValueError("episode is empty")
        if len(states) != len(actions):
            raise ValueError(f"state frames {len(states)} != action frames {len(actions)}")
        if states.shape[1] != len(self.state_names):
            raise ValueError(f"state dim {states.shape[1]} != expected {len(self.state_names)}")
        if actions.shape[1] != len(self.action_names):
            raise ValueError(f"action dim {actions.shape[1]} != expected {len(self.action_names)}")
        if timestamps.ndim != 1 or len(timestamps) != len(states):
            raise ValueError("timestamps must be rank 1 and match the number of frames")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("states/actions contain NaN or Inf")
        if not np.isfinite(timestamps).all():
            raise ValueError("timestamps contain NaN or Inf")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("capture timestamps must be strictly increasing")
        for key in self.camera_keys:
            if key not in images:
                raise ValueError(f"missing camera stream: {key}")
            self._ensure_rgb_hwc_uint8(images[key], expected_frames=len(states))

    def _load_or_init_info(self) -> dict[str, Any]:
        if self.info_path.exists():
            return json.loads(self.info_path.read_text(encoding="utf-8"))
        h, w = self.image_hw
        features: dict[str, Any] = {
            "observation.state": self._vector_feature(len(self.state_names), self.state_names),
            "action": self._vector_feature(len(self.action_names), self.action_names),
        }
        for key in self.camera_keys:
            features[f"observation.images.{key}"] = {
                "dtype": "video",
                "shape": [3, h, w],
                "names": ["channels", "height", "width"],
                "info": None,
            }
        features.update(
            {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            }
        )
        info = {
            "codebase_version": LEROBOT_CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 0,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
            "action_semantics": self.action_semantics,
            "action_offset": self.action_offset,
        }
        self.info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        return info

    def _validate_existing_dataset(self) -> None:
        expected = {
            "codebase_version": LEROBOT_CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "chunks_size": self.chunk_size,
            "action_semantics": self.action_semantics,
            "action_offset": self.action_offset,
        }
        for key, value in expected.items():
            actual = self.info.get(key)
            if actual != value:
                raise ValueError(f"existing dataset {key}={actual!r}, requested {value!r}")

        features = self.info.get("features", {})
        self._validate_vector_feature(features, "observation.state", self.state_names)
        self._validate_vector_feature(features, "action", self.action_names)
        actual_cameras = {
            key.removeprefix("observation.images.")
            for key, value in features.items()
            if value.get("dtype") in {"image", "video"}
        }
        if actual_cameras != set(self.camera_keys):
            raise ValueError(f"existing camera keys {sorted(actual_cameras)} != requested {sorted(self.camera_keys)}")
        expected_shape = [3, *self.image_hw]
        for key in self.camera_keys:
            actual_shape = features[f"observation.images.{key}"].get("shape")
            if actual_shape != expected_shape:
                raise ValueError(f"existing camera {key} shape {actual_shape} != requested {expected_shape}")

    @staticmethod
    def _validate_vector_feature(features: dict[str, Any], key: str, names: list[str]) -> None:
        feature = features.get(key)
        if feature is None:
            raise ValueError(f"existing dataset is missing feature {key}")
        if feature.get("dtype") != "float32" or feature.get("shape") != [len(names)] or feature.get("names") != names:
            raise ValueError(f"existing dataset feature {key} is incompatible: {feature}")

    def _write_info(self) -> None:
        self.info_path.write_text(json.dumps(self.info, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_existing_tasks(self) -> None:
        if not self.tasks_path.exists():
            return
        for line in self.tasks_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self.tasks[str(row["task"])] = int(row["task_index"])

    def _get_task_index(self, instruction: str) -> int:
        if instruction in self.tasks:
            return self.tasks[instruction]
        task_index = len(self.tasks)
        self.tasks[instruction] = task_index
        self._append_jsonl(self.tasks_path, {"task_index": task_index, "task": instruction})
        return task_index

    def _chunk_name(self, episode_index: int) -> str:
        return f"chunk-{episode_index // self.chunk_size:03d}"

    def _write_episode_videos(self, episode_index: int, images: dict[str, np.ndarray]) -> None:
        chunk_name = self._chunk_name(episode_index)
        h, w = self.image_hw
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
        for key, frames in images.items():
            out_dir = self.video_dir / chunk_name / f"observation.images.{key}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"episode_{episode_index:06d}.mp4"
            if out_path.exists():
                raise FileExistsError(f"episode video already exists: {out_path}")
            writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {out_path}")
            try:
                for frame in frames:
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            if not out_path.exists() or out_path.stat().st_size == 0:
                raise RuntimeError(f"empty video written: {out_path}")

    @staticmethod
    def _write_episode_parquet(
        *,
        parquet_path: Path,
        episode_index: int,
        task_index: int,
        global_offset: int,
        states: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        frame_count, state_dim = states.shape
        action_dim = actions.shape[1]
        table = pa.table(
            {
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), state_dim)),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), action_dim)),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(np.arange(frame_count, dtype=np.int64)),
                "episode_index": pa.array(np.full(frame_count, episode_index, dtype=np.int64)),
                "index": pa.array(np.arange(global_offset, global_offset + frame_count, dtype=np.int64)),
                "task_index": pa.array(np.full(frame_count, task_index, dtype=np.int64)),
            }
        )
        pq.write_table(table, parquet_path)

    def _recompute_openpi_norm_stats(self) -> None:
        state_batches: list[np.ndarray] = []
        action_batches: list[np.ndarray] = []
        for path in sorted(self.data_dir.glob("chunk-*/episode_*.parquet")):
            table = pq.read_table(path, columns=["observation.state", "action"])
            state_batches.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
            action_batches.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        if not state_batches:
            return
        payload = {
            "norm_stats": {
                "state": self._stat_dict(np.concatenate(state_batches, axis=0), include_count=False),
                "actions": self._stat_dict(np.concatenate(action_batches, axis=0), include_count=False),
            },
            "note": "Raw absolute-action statistics. Run OpenPI compute_norm_stats.py for the final transformed training statistics.",
        }
        self.norm_stats_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _build_splits(self, total_episodes: int) -> dict[str, str]:
        if total_episodes <= 0:
            return {}
        if self.val_ratio <= 0 or total_episodes < 10:
            return {"train": f"0:{total_episodes}"}
        val_count = max(1, int(round(total_episodes * self.val_ratio)))
        train_end = max(1, total_episodes - val_count)
        splits = {"train": f"0:{train_end}"}
        if train_end < total_episodes:
            splits["val"] = f"{train_end}:{total_episodes}"
        return splits

    @staticmethod
    def _vector_feature(dim: int, names: list[str]) -> dict[str, Any]:
        return {"dtype": "float32", "shape": [dim], "names": names}

    @staticmethod
    def _ensure_rgb_hwc_uint8(frames: np.ndarray, *, expected_frames: int | None = None) -> np.ndarray:
        arr = np.asarray(frames)
        if arr.ndim != 4:
            raise ValueError(f"expected 4D frames, got shape={arr.shape}")
        if expected_frames is not None and len(arr) != expected_frames:
            raise ValueError(f"camera frame count {len(arr)} != expected {expected_frames}")
        if arr.dtype != np.uint8:
            raise ValueError(f"camera frames must be uint8, got {arr.dtype}")
        if arr.shape[-1] == 3:
            out = arr
        elif arr.shape[1] == 3:
            out = arr.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"cannot infer RGB layout from shape={arr.shape}")
        return np.ascontiguousarray(out)

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _stat_dict(x: np.ndarray, *, include_count: bool = True) -> dict[str, list[float] | list[int]]:
        values = np.asarray(x, dtype=np.float32)
        stats: dict[str, list[float] | list[int]] = {
            "mean": np.mean(values, axis=0).astype(np.float32).tolist(),
            "std": np.std(values, axis=0).astype(np.float32).tolist(),
            "min": np.min(values, axis=0).astype(np.float32).tolist(),
            "max": np.max(values, axis=0).astype(np.float32).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).astype(np.float32).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).astype(np.float32).tolist(),
        }
        if include_count:
            stats["count"] = [len(values)]
        return stats
