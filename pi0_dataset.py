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
    return [
        f"{side}_joint_1", f"{side}_joint_2", f"{side}_joint_3", f"{side}_joint_4", f"{side}_joint_5", f"{side}_joint_6", f"{side}_gripper",
    ]


class Pi0LeRobotDatasetWriter:
    """Write teleop episodes in a LeRobot-v2.1-like layout for pi0.5/openpi training.

    Output layout:
      <root>/
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/observation.images.cam_high/episode_000000.mp4
        raw/episode_000000.npz
        meta/info.json
        meta/tasks.jsonl
        meta/episodes.jsonl
        meta/episodes_stats.jsonl
        meta/openpi_norm_stats.json
    """

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
    ):
        self.root = Path(root)
        self.fps = int(fps)
        self.robot_type = robot_type
        self.state_names = list(state_names)
        self.action_names = list(action_names or state_names)
        self.camera_keys = list(camera_keys)
        self.image_hw = tuple(image_hw)
        self.chunk_size = int(chunk_size)
        self.save_raw_npz = bool(save_raw_npz)
        self.val_ratio = float(val_ratio)

        self.meta_dir = self.root / "meta"
        self.data_dir = self.root / "data"
        self.video_dir = self.root / "videos"
        self.raw_dir = self.root / "raw"
        for p in (self.meta_dir, self.data_dir, self.video_dir):
            p.mkdir(parents=True, exist_ok=True)
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
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError("states/actions must be rank-2 arrays")
        if states.shape != actions.shape:
            raise ValueError(f"states shape {states.shape} != actions shape {actions.shape}")
        if states.shape[1] != len(self.state_names):
            raise ValueError(f"state dim {states.shape[1]} != expected {len(self.state_names)}")
        if actions.shape[1] != len(self.action_names):
            raise ValueError(f"action dim {actions.shape[1]} != expected {len(self.action_names)}")
        if len(timestamps) != len(states):
            raise ValueError("timestamps length must match number of frames")
        for key in self.camera_keys:
            if key not in images:
                raise ValueError(f"missing camera stream: {key}")
            if len(images[key]) != len(states):
                raise ValueError(f"camera {key} frame count mismatch")

        episode_index = int(self.info["total_episodes"])
        frame_count = int(len(states))
        global_offset = int(self.info["total_frames"])
        task_index = self._get_task_index(task_name)
        rel_timestamps = (timestamps - timestamps[0]).astype(np.float32)

        chunk_name = self._chunk_name(episode_index)
        chunk_data_dir = self.data_dir / chunk_name
        chunk_data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_data_dir / f"episode_{episode_index:06d}.parquet"

        if self.save_raw_npz:
            raw_path = self.raw_dir / f"episode_{episode_index:06d}.npz"
            raw_payload: dict[str, Any] = {
                "observation.state": states,
                "action": actions,
                "timestamp": rel_timestamps,
                "episode_index": np.full(frame_count, episode_index, dtype=np.int64),
                "frame_index": np.arange(frame_count, dtype=np.int64),
                "index": np.arange(global_offset, global_offset + frame_count, dtype=np.int64),
                "task_name": np.array(task_name),
                "instruction": np.array(instruction),
                "success": np.array(bool(success), dtype=np.bool_),
            }
            for key, value in images.items():
                raw_payload[f"observation.images.{key}"] = self._ensure_rgb_hwc_uint8(value)
            if metadata:
                for k, v in metadata.items():
                    raw_payload[f"meta.{k}"] = np.array(v)
            np.savez_compressed(raw_path, **raw_payload)

        normalized_images = {key: self._ensure_rgb_hwc_uint8(images[key]) for key in self.camera_keys}
        self._write_episode_videos(episode_index, normalized_images)
        self._write_episode_parquet(
            parquet_path=parquet_path,
            episode_index=episode_index,
            task_index=task_index,
            global_offset=global_offset,
            states=states,
            actions=actions,
            timestamps=rel_timestamps,
            task_name=task_name,
            instruction=instruction,
            success=success,
        )

        self._append_jsonl(
            self.episodes_path,
            {
                "episode_index": episode_index,
                "tasks": [task_name],
                "length": frame_count,
                "instruction": instruction,
                "success": bool(success),
            },
        )
        self._append_jsonl(
            self.episodes_stats_path,
            {
                "episode_index": episode_index,
                "tasks": [task_name],
                "length": frame_count,
                "stats": {
                    "observation.state": self._stat_dict(states),
                    "action": self._stat_dict(actions),
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

    def _load_or_init_info(self) -> dict[str, Any]:
        if self.info_path.exists():
            return json.loads(self.info_path.read_text())
        h, w = self.image_hw
        features: dict[str, Any] = {
            "action": self._vector_feature(len(self.action_names), self.action_names),
            "observation.state": self._vector_feature(len(self.state_names), self.state_names),
        }
        for key in self.camera_keys:
            features[f"observation.images.{key}"] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
                "info": None,
            }
        features.update(
            {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "task": {"dtype": "string", "shape": [1], "names": None},
                "instruction": {"dtype": "string", "shape": [1], "names": None},
                "success": {"dtype": "bool", "shape": [1], "names": None},
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
        }
        self.info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
        return info

    def _write_info(self):
        self.info_path.write_text(json.dumps(self.info, indent=2, ensure_ascii=False))

    def _load_existing_tasks(self):
        if not self.tasks_path.exists():
            return
        for line in self.tasks_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self.tasks[str(row["task"])] = int(row["task_index"])

    def _get_task_index(self, task_name: str) -> int:
        task_name = str(task_name)
        if task_name in self.tasks:
            return self.tasks[task_name]
        task_index = len(self.tasks)
        self.tasks[task_name] = task_index
        self._append_jsonl(self.tasks_path, {"task_index": task_index, "task": task_name})
        return task_index

    def _chunk_name(self, episode_index: int) -> str:
        return f"chunk-{episode_index // self.chunk_size:03d}"

    def _write_episode_videos(self, episode_index: int, images: dict[str, np.ndarray]):
        chunk_name = self._chunk_name(episode_index)
        h, w = self.image_hw
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
        for key, frames in images.items():
            out_dir = self.video_dir / chunk_name / f"observation.images.{key}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"episode_{episode_index:06d}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {out_path}")
            try:
                for frame in frames:
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()

    def _write_episode_parquet(
        self,
        *,
        parquet_path: Path,
        episode_index: int,
        task_index: int,
        global_offset: int,
        states: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
        task_name: str,
        instruction: str,
        success: bool,
    ):
        frame_count, state_dim = states.shape
        _, action_dim = actions.shape
        table = pa.table(
            {
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), state_dim)),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), action_dim)),
                "timestamp": pa.array(timestamps.astype(np.float32)),
                "frame_index": pa.array(np.arange(frame_count, dtype=np.int64)),
                "episode_index": pa.array(np.full(frame_count, episode_index, dtype=np.int64)),
                "index": pa.array(np.arange(global_offset, global_offset + frame_count, dtype=np.int64)),
                "task_index": pa.array(np.full(frame_count, task_index, dtype=np.int64)),
                "task": pa.array([task_name] * frame_count, type=pa.string()),
                "instruction": pa.array([instruction] * frame_count, type=pa.string()),
                "success": pa.array(np.full(frame_count, bool(success), dtype=np.bool_)),
            }
        )
        pq.write_table(table, parquet_path)

    def _recompute_openpi_norm_stats(self):
        if not self.save_raw_npz:
            return
        state_batches: list[np.ndarray] = []
        action_batches: list[np.ndarray] = []
        for path in sorted(self.raw_dir.glob("episode_*.npz")):
            with np.load(path) as data:
                state_batches.append(np.asarray(data["observation.state"], dtype=np.float32))
                action_batches.append(np.asarray(data["action"], dtype=np.float32))
        if not state_batches:
            return
        states = np.concatenate(state_batches, axis=0)
        actions = np.concatenate(action_batches, axis=0)
        payload = {
            "norm_stats": {
                "state": self._stat_dict(states),
                "actions": self._stat_dict(actions),
            }
        }
        self.norm_stats_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @staticmethod
    def _build_splits(total_episodes: int) -> dict[str, str]:
        if total_episodes <= 0:
            return {}
        if total_episodes < 10:
            return {"train": f"0:{total_episodes}"}
        val_count = max(1, int(round(total_episodes * 0.1)))
        train_end = max(1, total_episodes - val_count)
        splits = {"train": f"0:{train_end}"}
        if train_end < total_episodes:
            splits["val"] = f"{train_end}:{total_episodes}"
        return splits

    @staticmethod
    def _vector_feature(dim: int, names: list[str]) -> dict[str, Any]:
        return {"dtype": "float32", "shape": [dim], "names": names}

    @staticmethod
    def _ensure_rgb_hwc_uint8(frames: np.ndarray) -> np.ndarray:
        arr = np.asarray(frames)
        if arr.ndim != 4:
            raise ValueError(f"expected 4D frames, got shape={arr.shape}")
        if arr.shape[-1] == 3:
            out = arr
        elif arr.shape[1] == 3:
            out = arr.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"cannot infer image layout from shape={arr.shape}")
        return np.asarray(out, dtype=np.uint8)

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _stat_dict(x: np.ndarray) -> dict[str, list[float]]:
        x = np.asarray(x, dtype=np.float32)
        return {
            "mean": np.mean(x, axis=0).astype(np.float32).tolist(),
            "std": np.std(x, axis=0).astype(np.float32).tolist(),
            "min": np.min(x, axis=0).astype(np.float32).tolist(),
            "max": np.max(x, axis=0).astype(np.float32).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).astype(np.float32).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).astype(np.float32).tolist(),
        }
