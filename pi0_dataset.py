"""Small LeRobot v2.1 dataset writer used by the Piper collectors.

The writer stores camera streams as MP4 files and numeric observations/actions
in one parquet file per episode. ``timestamp`` is the canonical LeRobot frame
time; state, action, and per-camera device timestamps are retained as explicit
parquet columns and in the optional raw NPZ copy.
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

from piper_data_contract import (
    CONTRACT_VERSION,
    JOINT_SCHEMA,
    EpisodeContract,
)

LEROBOT_CODEBASE_VERSION = "v2.1"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_VIDEO_CODEC = "mp4v"
DEFAULT_ACTION_HORIZON = 50

LEGACY_V2 = "legacy_v2"
CANONICAL_CONTRACT_FORMAT = "canonical"
DELIVERY_ABSOLUTE_ACTION_FORMAT = "absolute_eef_target"
DELIVERY_LEGACY_ACTION_FORMAT = "step_delta"
DELIVERY_ABSOLUTE_ACTION_SEMANTICS = "absolute_eef_target"
DELIVERY_LEGACY_ACTION_SEMANTICS = (
    "eef_delta_base_xyz_left_rotvec_gripper_target"
)
GRIPPER_OPENING_FRACTION = "absolute_opening_fraction_0_closed_1_open"
GRIPPER_CLOSED_FRACTION_LEGACY = "absolute_closed_fraction_0_open_1_closed"
ROTATION6D_SEMANTICS = (
    "state_and_raw_action_rotation6d_first_two_columns_model_action_left_rotvec"
)
LEGACY_ROTATION_SEMANTICS = "state_rotation6d_action_left_rotvec_base_frame"
COORDINATE_FRAME = "slave_base"

EEF_STATE_NAMES = [
    "eef_x_base_m",
    "eef_y_base_m",
    "eef_z_base_m",
    "rotation6d_col0_x",
    "rotation6d_col0_y",
    "rotation6d_col0_z",
    "rotation6d_col1_x",
    "rotation6d_col1_y",
    "rotation6d_col1_z",
    "gripper_opening_fraction",
]
EEF_ABSOLUTE_ACTION_NAMES = [
    "target_eef_x_base_m",
    "target_eef_y_base_m",
    "target_eef_z_base_m",
    "target_rotation6d_col0_x",
    "target_rotation6d_col0_y",
    "target_rotation6d_col0_z",
    "target_rotation6d_col1_x",
    "target_rotation6d_col1_y",
    "target_rotation6d_col1_z",
    "target_gripper_opening_fraction",
]
EEF_LEGACY_STATE_NAMES = [*EEF_STATE_NAMES[:-1], "gripper_closed_fraction"]
EEF_LEGACY_ACTION_NAMES = [
    "delta_x_base_m",
    "delta_y_base_m",
    "delta_z_base_m",
    "delta_rx_base_rad",
    "delta_ry_base_rad",
    "delta_rz_base_rad",
    "gripper_target_closed_fraction",
]

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


def _per_arm_names(names: list[str], *, arm_mode: str, arm_side: str) -> list[str]:
    if arm_mode == "bimanual":
        return [f"left_{name}" for name in names] + [f"right_{name}" for name in names]
    side = arm_side if arm_side in {"left", "right"} else "right"
    return [f"{side}_{name}" for name in names]


def default_eef_names(
    *, arm_mode: str, arm_side: str, legacy: bool = False
) -> tuple[list[str], list[str]]:
    """Return design-document names for canonical or legacy delivery data."""
    state = EEF_LEGACY_STATE_NAMES if legacy else EEF_STATE_NAMES
    action = EEF_LEGACY_ACTION_NAMES if legacy else EEF_ABSOLUTE_ACTION_NAMES
    return (
        _per_arm_names(state, arm_mode=arm_mode, arm_side=arm_side),
        _per_arm_names(action, arm_mode=arm_mode, arm_side=arm_side),
    )


def classify_contract_dimensions(
    state_dim: int,
    raw_action_dim: int,
    *,
    schema: str | None = None,
    legacy_format: str | bool | None = None,
) -> dict[str, Any]:
    """Classify supported Piper layouts without relying on contract-class dimensions.

    Delivery ``10D state + 7D action`` (or ``20D + 14D``) is always the
    measured-step-delta ``legacy_v2`` layout. Canonical delivery is
    ``10D + 10D`` per arm and stores absolute EEF targets.
    """
    state_dim = int(state_dim)
    raw_action_dim = int(raw_action_dim)
    schema_text = str(schema or "").strip().lower()
    if schema_text in {"eef", "cartesian"}:
        schema_text = "delivery"
    legacy_requested = legacy_format is True or str(legacy_format or "").strip().lower() == LEGACY_V2

    if schema_text not in {"", "joint", "delivery"}:
        raise ValueError(f"unsupported schema {schema!r}")
    inferred_schema = schema_text
    if not inferred_schema:
        if (state_dim, raw_action_dim) in {(10, 7), (20, 14), (10, 10), (20, 20)}:
            inferred_schema = "delivery"
        elif (state_dim, raw_action_dim) in {(7, 7), (14, 14), (16, 16)}:
            inferred_schema = "joint"
        else:
            raise ValueError(
                f"unsupported Piper state/action dimensions {state_dim}/{raw_action_dim}"
            )

    if inferred_schema == "joint":
        if (state_dim, raw_action_dim) not in {(7, 7), (14, 14), (16, 16)}:
            raise ValueError(
                f"joint schema requires 7D/7D, 14D/14D, or Franka 16D/16D, got {state_dim}/{raw_action_dim}"
            )
        if legacy_requested:
            raise ValueError("legacy_v2 is reserved for delivery step-delta datasets")
        arm_count = 1 if state_dim == 7 else 2
        model_action_dim = raw_action_dim
        legacy = False
        action_format = "absolute_joint_target"
    else:
        if state_dim not in {10, 20}:
            raise ValueError(f"delivery schema requires 10D or 20D state, got {state_dim}")
        arm_count = state_dim // 10
        model_action_dim = arm_count * 7
        if raw_action_dim == state_dim:
            if legacy_requested:
                raise ValueError(
                    "legacy_v2 delivery must be 10D+7D or 20D+14D; "
                    f"got {state_dim}/{raw_action_dim}"
                )
            legacy = False
            action_format = DELIVERY_ABSOLUTE_ACTION_FORMAT
        elif raw_action_dim == arm_count * 7:
            legacy = True
            action_format = DELIVERY_LEGACY_ACTION_FORMAT
        else:
            raise ValueError(
                "delivery schema requires canonical 10D absolute action per arm or "
                f"legacy 7D step-delta per arm, got {state_dim}/{raw_action_dim}"
            )

    return {
        "schema": inferred_schema,
        "arm_count": arm_count,
        "arm_mode": "single" if arm_count == 1 else "bimanual",
        "state_dim": state_dim,
        "raw_action_dim": raw_action_dim,
        "model_action_dim": model_action_dim,
        "legacy": legacy,
        "legacy_format": LEGACY_V2 if legacy else None,
        "contract_format": LEGACY_V2 if legacy else CANONICAL_CONTRACT_FORMAT,
        "delivery_action_format": action_format if inferred_schema == "delivery" else None,
    }


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
        schema: str = JOINT_SCHEMA,
        arm_mode: str | None = None,
        arm_side: str = "right",
        action_semantics: str | None = None,
        action_source: str = "master_joint_feedback",
        action_alignment: str = "same_step_command",
        action_offset: int | None = None,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        raw_action_dim: int | None = None,
        model_action_dim: int | None = None,
        gripper_semantics: str | None = None,
        rotation_semantics: str | None = None,
        coordinate_frame: str = COORDINATE_FRAME,
        legacy_format: str | bool | None = None,
        source_frame: str = "",
    ):
        self.root = Path(root).expanduser()
        self.fps = int(fps)
        self.robot_type = str(robot_type)
        self.state_names = list(state_names)
        if action_names is None and str(schema).lower() == "delivery":
            inferred_mode_for_names = arm_mode or (
                "single" if len(self.state_names) == 10 else "bimanual"
            )
            inferred_side_for_names = "both" if inferred_mode_for_names == "bimanual" else arm_side
            _default_state_names, _default_action_names = default_eef_names(
                arm_mode=inferred_mode_for_names,
                arm_side=inferred_side_for_names,
                legacy=False,
            )
            self.action_names = list(_default_action_names)
        else:
            self.action_names = list(action_names or state_names)
        self.camera_keys = list(camera_keys)
        self.image_hw = tuple(int(x) for x in image_hw)
        self.chunk_size = int(chunk_size)
        self.save_raw_npz = bool(save_raw_npz)
        self.val_ratio = float(val_ratio)
        self.action_horizon = int(action_horizon)
        inferred_arm_mode = arm_mode
        if inferred_arm_mode is None:
            per_arm_state_dim = 10 if str(schema).lower() == "delivery" else 7
            if len(self.state_names) == per_arm_state_dim:
                inferred_arm_mode = "single"
            elif len(self.state_names) == 2 * per_arm_state_dim:
                inferred_arm_mode = "bimanual"
            else:
                raise ValueError(
                    f"cannot infer arm_mode from schema={schema!r} and "
                    f"state_dim={len(self.state_names)}"
                )
        dimensions = classify_contract_dimensions(
            len(self.state_names),
            len(self.action_names) if raw_action_dim is None else int(raw_action_dim),
            schema=schema,
            legacy_format=legacy_format,
        )
        contract_kwargs: dict[str, Any] = {
            "schema": schema,
            "arm_mode": inferred_arm_mode,
            "arm_side": arm_side,
            "camera_keys": tuple(self.camera_keys),
            "action_source": action_source,
            "action_alignment": action_alignment,
        }
        contract_fields = getattr(EpisodeContract, "__dataclass_fields__", {})
        optional_contract_values = {
            "action_offset": action_offset,
            "fps": self.fps,
            "action_horizon": self.action_horizon,
            "coordinate_frame": coordinate_frame,
            "source_frame": source_frame,
            "version": 2 if dimensions["legacy"] else CONTRACT_VERSION,
            "legacy_delivery_v2": bool(dimensions["legacy"]),
        }
        for key, value in optional_contract_values.items():
            if key in contract_fields:
                contract_kwargs[key] = value
        self.contract = EpisodeContract(**contract_kwargs)
        self.schema = self.contract.schema
        self.arm_mode = self.contract.arm_mode
        self.arm_side = self.contract.arm_side
        self.action_source = self.contract.action_source
        self.action_alignment = self.contract.action_alignment
        self.action_offset = self.contract.action_offset if action_offset is None else int(action_offset)

        if dimensions["arm_mode"] != self.arm_mode:
            raise ValueError(
                f"dimensions imply arm_mode={dimensions['arm_mode']!r}, got {self.arm_mode!r}"
            )
        if len(self.action_names) != dimensions["raw_action_dim"]:
            raise ValueError(
                f"action_names dim {len(self.action_names)} != raw_action_dim "
                f"{dimensions['raw_action_dim']}"
            )
        self.raw_action_dim = int(dimensions["raw_action_dim"])
        self.model_action_dim = int(
            dimensions["model_action_dim"] if model_action_dim is None else model_action_dim
        )
        if self.model_action_dim != int(dimensions["model_action_dim"]):
            raise ValueError(
                f"model_action_dim={self.model_action_dim} != expected "
                f"{dimensions['model_action_dim']}"
            )
        contract_model_names = list(getattr(self.contract, "model_action_names", ()))
        self.model_action_names = (
            contract_model_names
            if len(contract_model_names) == self.model_action_dim
            else [f"model_action_{index}" for index in range(self.model_action_dim)]
        )
        self.legacy = bool(dimensions["legacy"])
        self.legacy_format = dimensions["legacy_format"]
        self.contract_format = str(dimensions["contract_format"])
        self.delivery_action_format = dimensions["delivery_action_format"]
        self.contract_version = 2 if self.legacy else CONTRACT_VERSION
        self.source_frame = str(source_frame).strip()
        default_action_semantics = self.contract.action_semantics
        if self.schema == "delivery" and self.legacy:
            default_action_semantics = DELIVERY_LEGACY_ACTION_SEMANTICS
        self.action_semantics = str(action_semantics or default_action_semantics)
        self.model_action_semantics = str(
            getattr(self.contract, "model_action_semantics", self.action_semantics)
        )
        if self.schema == "delivery":
            looks_delta = "delta" in self.action_semantics.lower()
            if self.legacy and not looks_delta:
                raise ValueError("legacy_v2 delivery action_semantics must describe step deltas")
            if not self.legacy and looks_delta:
                raise ValueError(
                    "canonical delivery stores 10D absolute EEF targets; delta semantics are invalid"
                )
        if gripper_semantics is None:
            if self.schema == "delivery" and self.legacy:
                gripper_semantics = GRIPPER_CLOSED_FRACTION_LEGACY
            elif hasattr(self.contract, "gripper_semantics"):
                gripper_semantics = self.contract.gripper_semantics
            elif any("opening_m" in name for name in [*self.state_names, *self.action_names]):
                gripper_semantics = "absolute_opening_m"
            else:
                gripper_semantics = GRIPPER_OPENING_FRACTION
        self.gripper_semantics = str(gripper_semantics)
        if rotation_semantics is None:
            if self.schema != "delivery":
                rotation_semantics = "not_applicable"
            elif self.legacy:
                rotation_semantics = LEGACY_ROTATION_SEMANTICS
            elif hasattr(self.contract, "rotation_semantics"):
                rotation_semantics = self.contract.rotation_semantics
            else:
                rotation_semantics = ROTATION6D_SEMANTICS
        self.rotation_semantics = str(rotation_semantics)
        self.coordinate_frame = str(coordinate_frame).strip()

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
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if not self.coordinate_frame:
            raise ValueError("coordinate_frame must not be empty")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.action_offset < 0:
            raise ValueError("action_offset must be >= 0")
        if len(self.state_names) != self.contract.state_dim:
            raise ValueError(
                f"state_names dim {len(self.state_names)} != contract state_dim {self.contract.state_dim}"
            )
        if self.action_offset != self.contract.action_offset:
            raise ValueError(
                f"action_offset={self.action_offset} disagrees with "
                f"action_alignment={self.action_alignment!r} (expected {self.contract.action_offset})"
            )

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
        self.policy_contract_path = self.meta_dir / "policy_contract.json"

        self.tasks: dict[str, int] = {}
        self.info = self._load_or_init_info()
        self._load_existing_tasks()
        self._validate_existing_dataset()
        self._write_policy_contract()

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
        state_timestamps: np.ndarray | None = None,
        action_timestamps: np.ndarray | None = None,
        image_timestamps: dict[str, np.ndarray] | None = None,
    ) -> int:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        capture_timestamps = np.asarray(timestamps, dtype=np.float64)
        state_timestamps = np.asarray(
            capture_timestamps if state_timestamps is None else state_timestamps,
            dtype=np.float64,
        )
        action_timestamps = np.asarray(
            state_timestamps if action_timestamps is None else action_timestamps,
            dtype=np.float64,
        )
        image_timestamps = {
            key: np.asarray(
                state_timestamps
                if image_timestamps is None or key not in image_timestamps
                else image_timestamps[key],
                dtype=np.float64,
            )
            for key in self.camera_keys
        }
        self._validate_episode(
            states,
            actions,
            capture_timestamps,
            images,
            state_timestamps=state_timestamps,
            action_timestamps=action_timestamps,
            image_timestamps=image_timestamps,
        )

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
                "state_timestamp": state_timestamps,
                "action_timestamp": action_timestamps,
                "episode_index": np.full(frame_count, episode_index, dtype=np.int64),
                "frame_index": np.arange(frame_count, dtype=np.int64),
                "index": np.arange(global_offset, global_offset + frame_count, dtype=np.int64),
                "task_name": np.asarray(task_name),
                "instruction": np.asarray(instruction),
                "success": np.asarray(bool(success), dtype=np.bool_),
                "contract_version": np.asarray(self.contract_version, dtype=np.int64),
                "schema": np.asarray(self.schema),
                "arm_mode": np.asarray(self.arm_mode),
                "arm_side": np.asarray(self.arm_side),
                "robot_type": np.asarray(self.robot_type),
                "state_dim": np.asarray(len(self.state_names), dtype=np.int64),
                "action_dim": np.asarray(self.raw_action_dim, dtype=np.int64),
                "raw_action_dim": np.asarray(self.raw_action_dim, dtype=np.int64),
                "model_action_dim": np.asarray(self.model_action_dim, dtype=np.int64),
                "camera_keys": np.asarray(self.camera_keys),
                "state_names": np.asarray(self.state_names),
                "action_names": np.asarray(self.action_names),
                "model_action_names": np.asarray(self.model_action_names),
                "action_semantics": np.asarray(self.action_semantics),
                "model_action_semantics": np.asarray(self.model_action_semantics),
                "action_source": np.asarray(self.action_source),
                "action_alignment": np.asarray(self.action_alignment),
                "action_offset": np.asarray(self.action_offset, dtype=np.int64),
                "action_horizon": np.asarray(self.action_horizon, dtype=np.int64),
                "contract_format": np.asarray(self.contract_format),
                "legacy": np.asarray(self.legacy, dtype=np.bool_),
                "gripper_semantics": np.asarray(self.gripper_semantics),
                "rotation_semantics": np.asarray(self.rotation_semantics),
                "coordinate_frame": np.asarray(self.coordinate_frame),
                "source_frame": np.asarray(self.source_frame),
                "fps": np.asarray(self.fps, dtype=np.int64),
                "legacy_delivery_v2": np.asarray(self.legacy, dtype=np.bool_),
            }
            if self.legacy_format:
                raw_payload["legacy_format"] = np.asarray(self.legacy_format)
            if self.delivery_action_format:
                raw_payload["delivery_action_format"] = np.asarray(self.delivery_action_format)
            for key, value in normalized_images.items():
                raw_payload[f"observation.images.{key}"] = value
                raw_payload[f"image_timestamps_{key}"] = image_timestamps[key]
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
            state_timestamps=state_timestamps,
            action_timestamps=action_timestamps,
            image_timestamps=image_timestamps,
        )

        self._append_jsonl(
            self.episodes_path,
            {
                "episode_index": episode_index,
                "tasks": [instruction],
                "length": frame_count,
                "task_name": task_name,
                "success": bool(success),
                **self._contract_dict(),
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
                    "state_timestamp": self._stat_dict(state_timestamps[:, None]),
                    "action_timestamp": self._stat_dict(action_timestamps[:, None]),
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
        *,
        state_timestamps: np.ndarray,
        action_timestamps: np.ndarray,
        image_timestamps: dict[str, np.ndarray],
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
        for name, values in {
            "state_timestamp": state_timestamps,
            "action_timestamp": action_timestamps,
            **{f"image_timestamp.{key}": value for key, value in image_timestamps.items()},
        }.items():
            if values.ndim != 1 or len(values) != len(states):
                raise ValueError(f"{name} must be rank 1 and match the number of frames")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains NaN or Inf")
            if len(values) > 1 and np.any(np.diff(values) <= 0):
                raise ValueError(f"{name} must be strictly increasing")
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
            "state_timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "action_timestamp": {"dtype": "float64", "shape": [1], "names": None},
        }
        for key in self.camera_keys:
            features[f"observation.images.{key}"] = {
                "dtype": "video",
                "shape": [3, h, w],
                "names": ["channels", "height", "width"],
                "info": None,
            }
            features[f"image_timestamp.{key}"] = {
                "dtype": "float64",
                "shape": [1],
                "names": None,
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
            "timestamp_fields": {
                "canonical": "timestamp",
                "state": "state_timestamp",
                "action": "action_timestamp",
                "images": {key: f"image_timestamp.{key}" for key in self.camera_keys},
            },
            "features": features,
            **self._contract_dict(),
        }
        self.info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        return info

    def _validate_existing_dataset(self) -> None:
        expected = {
            "codebase_version": LEROBOT_CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "chunks_size": self.chunk_size,
            **self._contract_dict(),
        }
        backfilled = False
        for key, value in expected.items():
            if key not in self.info:
                # Older writer versions did not persist the full contract. The
                # feature/dimension checks below must still pass before this
                # metadata is committed.
                self.info[key] = value
                backfilled = True
                continue
            actual = self.info.get(key)
            if actual != value:
                raise ValueError(f"existing dataset {key}={actual!r}, requested {value!r}")

        features = self.info.get("features", {})
        self._validate_vector_feature(features, "observation.state", self.state_names)
        self._validate_vector_feature(features, "action", self.action_names)
        expected_timestamp_fields = {
            "canonical": "timestamp",
            "state": "state_timestamp",
            "action": "action_timestamp",
            "images": {key: f"image_timestamp.{key}" for key in self.camera_keys},
        }
        if self.info.get("timestamp_fields") != expected_timestamp_fields:
            raise ValueError(
                f"existing timestamp_fields={self.info.get('timestamp_fields')!r} "
                f"!= requested {expected_timestamp_fields!r}"
            )
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
        if backfilled:
            self._write_info()

    def _contract_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "schema": self.schema,
            "arm_mode": self.arm_mode,
            "arm_side": self.arm_side,
            "state_dim": len(self.state_names),
            "action_dim": self.raw_action_dim,
            "raw_action_dim": self.raw_action_dim,
            "model_action_dim": self.model_action_dim,
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "model_action_names": list(self.model_action_names),
            "camera_keys": list(self.camera_keys),
            "action_semantics": self.action_semantics,
            "model_action_semantics": self.model_action_semantics,
            "action_source": self.action_source,
            "action_alignment": self.action_alignment,
            "action_offset": self.action_offset,
            "action_horizon": self.action_horizon,
            "contract_format": self.contract_format,
            "legacy": self.legacy,
            "legacy_format": self.legacy_format,
            "delivery_action_format": self.delivery_action_format,
            "gripper_semantics": self.gripper_semantics,
            "rotation_semantics": self.rotation_semantics,
            "coordinate_frame": self.coordinate_frame,
            "source_frame": self.source_frame,
            "legacy_delivery_v2": self.legacy,
        }

    def _write_policy_contract(self) -> None:
        payload = {
            "version": self.contract_version,
            "robot_type": self.robot_type,
            **{key: value for key, value in self._contract_dict().items() if key != "contract_version"},
        }
        self.policy_contract_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
        state_timestamps: np.ndarray,
        action_timestamps: np.ndarray,
        image_timestamps: dict[str, np.ndarray],
    ) -> None:
        frame_count, state_dim = states.shape
        action_dim = actions.shape[1]
        columns: dict[str, Any] = {
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), state_dim)),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), action_dim)),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "state_timestamp": pa.array(state_timestamps, type=pa.float64()),
                "action_timestamp": pa.array(action_timestamps, type=pa.float64()),
                "frame_index": pa.array(np.arange(frame_count, dtype=np.int64)),
                "episode_index": pa.array(np.full(frame_count, episode_index, dtype=np.int64)),
                "index": pa.array(np.arange(global_offset, global_offset + frame_count, dtype=np.int64)),
                "task_index": pa.array(np.full(frame_count, task_index, dtype=np.int64)),
        }
        for key, values in image_timestamps.items():
            columns[f"image_timestamp.{key}"] = pa.array(values, type=pa.float64())
        table = pa.table(columns)
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
            "raw_action_dim": self.raw_action_dim,
            "model_action_dim": self.model_action_dim,
            "contract_format": self.contract_format,
            "note": (
                "Raw LeRobot action statistics. Canonical delivery stores absolute 10D EEF "
                "targets per arm; legacy_v2 stores 7D measured step deltas per arm. Run "
                "OpenPI compute_norm_stats.py for transformed model-action statistics."
            ),
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
