#!/usr/bin/env python3
"""Validate Piper raw NPZ files and LeRobot v2.1 datasets.

Supported layouts are joint 7D/14D, canonical delivery 10D/20D absolute
EEF targets, and metadata-free or marked ``legacy_v2`` delivery 7D/14D
step-delta actions. Shape, finite values, timestamps, camera frame counts,
contract metadata, episode indexes, and statistics are checked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# This checker is also launched as a standalone script by the 4x4090
# dashboard.  Make sibling project modules importable even when the service
# does not inherit a PYTHONPATH containing the repository root.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_lerobot import inspect_npz_episode
from pi0_dataset import (
    DEFAULT_ACTION_HORIZON,
    DELIVERY_ABSOLUTE_ACTION_FORMAT,
    DELIVERY_LEGACY_ACTION_FORMAT,
    GRIPPER_CLOSED_FRACTION_LEGACY,
    GRIPPER_OPENING_FRACTION,
    LEGACY_V2,
    ROTATION6D_SEMANTICS,
    classify_contract_dimensions,
)


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def _shape(info: dict[str, Any], key: str) -> int:
    feature = info.get("features", {}).get(key)
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        raise ValueError(f"feature {key} must have shape [dim]")
    return int(shape[0])


def _dataset_contract(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.features must be an object")
    if "observation.state" in features and "action" in features:
        state_key, action_key, column_layout = "observation.state", "action", "canonical_columns"
    elif "state" in features and "actions" in features:
        state_key, action_key, column_layout = "state", "actions", "legacy_columns"
    else:
        raise ValueError("missing observation.state/action or state/actions features")
    dimensions = classify_contract_dimensions(
        _shape(info, state_key),
        _shape(info, action_key),
        schema=info.get("schema"),
        legacy_format=info.get("legacy_format") or info.get("contract_format"),
    )
    arm_side = str(info.get("arm_side") or ("both" if dimensions["arm_mode"] == "bimanual" else "right"))
    if dimensions["arm_mode"] == "bimanual":
        arm_side = "both"
    camera_features = sorted(
        key
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") in {"image", "video"}
    )
    if column_layout == "legacy_columns":
        camera_keys = [
            "cam_high" if key == "image" else "cam_wrist" if key == "wrist_image" else key
            for key in camera_features
        ]
    else:
        camera_keys = [key.removeprefix("observation.images.") for key in camera_features]
    return {
        **dimensions,
        "state_key": state_key,
        "action_key": action_key,
        "column_layout": column_layout,
        "arm_side": arm_side,
        "camera_features": camera_features,
        "camera_keys": camera_keys,
    }


def _timestamp_fields(info: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    configured = info.get("timestamp_fields")
    if isinstance(configured, dict):
        return configured
    return {
        "canonical": "timestamp",
        "state": "state_timestamp",
        "action": "action_timestamp",
        "images": {key: f"image_timestamp.{key}" for key in contract["camera_keys"]},
    }


def _strict_increasing(values: np.ndarray, *, legacy: bool) -> bool:
    if len(values) <= 1:
        return True
    delta = np.diff(values)
    if np.all(delta > 0):
        return True
    return bool(legacy and delta[-1] == 0 and np.all(delta[:-1] > 0))


def _vector(table: pa.Table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


def _rotation6d_errors(values: np.ndarray, arm_count: int, *, prefix: str) -> list[str]:
    errors: list[str] = []
    for arm in range(arm_count):
        offset = arm * 10 + 3
        first = values[:, offset : offset + 3]
        second = values[:, offset + 3 : offset + 6]
        norm0 = np.linalg.norm(first, axis=1)
        norm1 = np.linalg.norm(second, axis=1)
        dot = np.sum(first * second, axis=1)
        if np.any(np.abs(norm0 - 1.0) > 0.1) or np.any(np.abs(norm1 - 1.0) > 0.1):
            errors.append(f"{prefix}: rotation6d columns are not unit length")
        if np.any(np.abs(dot) > 0.1):
            errors.append(f"{prefix}: rotation6d columns are not orthogonal")
    return errors


def _stat_dict(values: np.ndarray) -> dict[str, np.ndarray]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    return {
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0),
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
        "q01": np.quantile(data, 0.01, axis=0),
        "q99": np.quantile(data, 0.99, axis=0),
    }


def _check_stats(path: Path, table: pa.Table, row: dict[str, Any], keys: list[str]) -> list[str]:
    errors: list[str] = []
    stats = row.get("stats") if isinstance(row, dict) else None
    if not isinstance(stats, dict):
        return [f"{path}: missing episode statistics"]
    for key in keys:
        if key not in table.column_names or key not in stats:
            continue
        try:
            field_type = table.schema.field(key).type
            values = _vector(table, key) if (
                pa.types.is_list(field_type) or pa.types.is_fixed_size_list(field_type)
            ) else np.asarray(
                table[key].to_numpy(zero_copy_only=False), dtype=np.float64
            )
            expected = _stat_dict(values)
            actual = stats[key]
            for stat_name, expected_value in expected.items():
                if stat_name in actual and not np.allclose(
                    np.asarray(actual[stat_name], dtype=np.float64), expected_value, atol=1e-5, rtol=1e-5
                ):
                    errors.append(f"{path}: stale {key}.{stat_name} statistics")
                    break
            count = actual.get("count")
            if count is not None and int(np.asarray(count).reshape(-1)[0]) != table.num_rows:
                errors.append(f"{path}: stale {key}.count statistics")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{path}: malformed statistics for {key}")
    return errors


def check_npz(path: Path, action_offset: int | None = None) -> list[str]:
    try:
        episode = inspect_npz_episode(path)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    if action_offset is not None and int(episode["action_offset"]) != int(action_offset):
        return [f"action_offset={episode['action_offset']} != expected {action_offset}"]
    print(
        f"OK Piper NPZ: {path} | schema={episode['schema']} arm={episode['arm_mode']} "
        f"state/action={episode['state_dim']}/{episode['raw_action_dim']} "
        f"format={episode['contract_format']}"
    )
    return []


def _metadata_errors(info: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    features = info["features"]
    state_feature = features[contract["state_key"]]
    action_feature = features[contract["action_key"]]
    state_names = state_feature.get("names")
    action_names = action_feature.get("names")
    inferred = {
        "schema": contract["schema"],
        "arm_mode": contract["arm_mode"],
        "arm_side": contract["arm_side"],
        "state_dim": contract["state_dim"],
        "action_dim": contract["raw_action_dim"],
        "raw_action_dim": contract["raw_action_dim"],
        "model_action_dim": contract["model_action_dim"],
        "camera_keys": contract["camera_keys"],
        "contract_format": LEGACY_V2 if contract["legacy"] else "canonical",
        "legacy": bool(contract["legacy"]),
        "legacy_format": LEGACY_V2 if contract["legacy"] else None,
        "delivery_action_format": (
            DELIVERY_LEGACY_ACTION_FORMAT if contract["legacy"] else DELIVERY_ABSOLUTE_ACTION_FORMAT
            if contract["schema"] == "delivery" else None
        ),
    }
    for key, expected in inferred.items():
        if key in info and info[key] != expected:
            errors.append(f"metadata {key}={info[key]!r}, inferred {expected!r}")
    if contract["legacy"]:
        if int(info.get("contract_version", 2)) >= 3:
            errors.append("legacy_v2 delivery cannot declare contract_version>=3")
        if "legacy_delivery_v2" in info and info["legacy_delivery_v2"] is not True:
            errors.append("10D+7D/20D+14D delivery must set legacy_delivery_v2=true")
    elif info.get("legacy_delivery_v2") is True:
        errors.append("canonical data cannot set legacy_delivery_v2=true")

    expected_cameras = 3 if contract["arm_mode"] == "bimanual" else 2
    if len(contract["camera_features"]) != expected_cameras:
        errors.append(
            f"camera features {contract['camera_features']}, expected {expected_cameras} streams"
        )
    if state_names is not None and len(state_names) != contract["state_dim"]:
        errors.append("state feature names length does not match state_dim")
    if action_names is not None and len(action_names) != contract["raw_action_dim"]:
        errors.append("action feature names length does not match raw_action_dim")

    if contract["legacy"]:
        if info.get("gripper_semantics", GRIPPER_CLOSED_FRACTION_LEGACY) != GRIPPER_CLOSED_FRACTION_LEGACY:
            errors.append("legacy_v2 must use closed-fraction gripper semantics")
        semantics = str(info.get("action_semantics", "delta"))
        if "delta" not in semantics.lower():
            errors.append("legacy_v2 action_semantics must describe step deltas")
    else:
        required = (
            "schema", "arm_mode", "state_dim", "raw_action_dim", "model_action_dim",
            "state_names", "action_names", "camera_keys", "action_semantics",
            "action_source", "action_alignment", "action_offset", "gripper_semantics",
            "rotation_semantics", "coordinate_frame", "action_horizon", "timestamp_fields",
        )
        for key in required:
            if key not in info:
                errors.append(f"canonical metadata missing {key}")
        if state_names is None or info.get("state_names") != state_names:
            errors.append("canonical state_names must match observation.state feature names")
        if action_names is None or info.get("action_names") != action_names:
            errors.append("canonical action_names must match action feature names")
        if int(info.get("action_horizon", -1)) != DEFAULT_ACTION_HORIZON:
            errors.append(f"canonical action_horizon must be {DEFAULT_ACTION_HORIZON}")
        # The original real-Piper contract used 20 Hz, while the RoboTwin
        # simulation collection path records a uniform 25 Hz stream.  Both
        # are valid when the dataset metadata and timestamp columns agree;
        # never silently relabel one rate as the other.
        if int(info.get("fps", -1)) not in {20, 25}:
            errors.append("canonical Piper datasets must use fps=20 or fps=25")
        if info.get("gripper_semantics") != GRIPPER_OPENING_FRACTION:
            errors.append("canonical gripper semantics must be opening fraction (0 closed, 1 open)")
        if info.get("coordinate_frame") != "slave_base":
            errors.append("canonical coordinate_frame must be slave_base")
        if contract["schema"] == "delivery":
            if info.get("delivery_action_format") != DELIVERY_ABSOLUTE_ACTION_FORMAT:
                errors.append("canonical delivery action must be absolute_eef_target")
            if info.get("rotation_semantics") != ROTATION6D_SEMANTICS:
                errors.append("canonical delivery must use rotation6d first-two-columns semantics")
            if "delta" in str(info.get("action_semantics", "")).lower():
                errors.append("canonical 10D/20D delivery action cannot use delta semantics")
    return errors


def check_dataset(root: Path) -> list[str]:
    info = _json(root / "meta" / "info.json")
    if not isinstance(info, dict):
        return ["missing or invalid meta/info.json"]
    try:
        contract = _dataset_contract(info)
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    errors = _metadata_errors(info, contract)

    policy_contract = _json(root / "meta" / "policy_contract.json")
    if not contract["legacy"] and not isinstance(policy_contract, dict):
        errors.append("canonical dataset is missing meta/policy_contract.json")
    if isinstance(policy_contract, dict):
        for key in (
            "schema", "arm_mode", "arm_side", "state_dim", "raw_action_dim",
            "model_action_dim", "state_names", "action_names", "camera_keys",
            "action_semantics", "action_source", "action_alignment", "action_offset",
            "gripper_semantics", "rotation_semantics", "coordinate_frame",
            "contract_format", "legacy_format", "delivery_action_format", "action_horizon",
        ):
            if key in info and policy_contract.get(key) != info.get(key):
                errors.append(f"policy_contract {key} != info {key}")

    fps = float(info.get("fps", 0))
    if fps <= 0:
        errors.append(f"invalid fps={fps}")
    chunk_size = int(info.get("chunks_size", 1000))
    parquets = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquets) != int(info.get("total_episodes", -1)):
        errors.append(f"parquet count {len(parquets)} != total_episodes {info.get('total_episodes')}")
    episodes = {int(row["episode_index"]): row for row in _jsonl(root / "meta" / "episodes.jsonl") if "episode_index" in row}
    stats = {int(row["episode_index"]): row for row in _jsonl(root / "meta" / "episodes_stats.jsonl") if "episode_index" in row}
    timestamps = _timestamp_fields(info, contract)

    total_frames = 0
    expected_global = 0
    for expected_episode, path in enumerate(parquets):
        table = pq.read_table(path)
        required = {
            contract["state_key"], contract["action_key"], "timestamp", "frame_index",
            "episode_index", "index", "task_index",
        }
        if not contract["legacy"]:
            required.update((timestamps.get("state"), timestamps.get("action")))
            required.update((timestamps.get("images") or {}).values())
        for camera in contract["camera_features"]:
            if info["features"][camera].get("dtype") == "image":
                required.add(camera)
        missing = sorted(item for item in required if item and item not in table.column_names)
        if missing:
            errors.append(f"{path}: missing columns {missing}")
            continue

        states = _vector(table, contract["state_key"])
        actions = _vector(table, contract["action_key"])
        frame_count = table.num_rows
        total_frames += frame_count
        if states.shape != (frame_count, contract["state_dim"]):
            errors.append(f"{path}: state shape {states.shape} is invalid")
        if actions.shape != (frame_count, contract["raw_action_dim"]):
            errors.append(f"{path}: action shape {actions.shape} is invalid")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            errors.append(f"{path}: state/action contains NaN/Inf")
        if contract["schema"] == "delivery":
            errors.extend(_rotation6d_errors(states, contract["arm_count"], prefix=str(path)))
            if not contract["legacy"]:
                errors.extend(_rotation6d_errors(actions, contract["arm_count"], prefix=f"{path}: action"))
        gripper_stride = 7 if contract["legacy"] or contract["schema"] == "joint" else 10
        gripper_index = 6 if gripper_stride == 7 else 9
        for arm in range(contract["arm_count"]):
            values = actions[:, arm * gripper_stride + gripper_index]
            if np.min(values) < -1e-4 or np.max(values) > 1.0001:
                errors.append(f"{path}: action gripper is outside [0,1]")

        canonical_ts = np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
        expected_ts = np.arange(frame_count, dtype=np.float64) / fps if fps else np.zeros(frame_count)
        if not np.isfinite(canonical_ts).all() or not np.allclose(canonical_ts, expected_ts, atol=1e-5):
            errors.append(f"{path}: timestamp must equal frame_index/fps")
        for name in [timestamps.get("state"), timestamps.get("action"), *(timestamps.get("images") or {}).values()]:
            if not name or name not in table.column_names:
                continue
            values = np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float64)
            if not np.isfinite(values).all() or not _strict_increasing(values, legacy=contract["legacy"]):
                errors.append(f"{path}: {name} must be finite and strictly increasing")

        episode_index = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
        frame_index = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
        global_index = np.asarray(table["index"].to_numpy(zero_copy_only=False), dtype=np.int64)
        if not np.array_equal(episode_index, np.full(frame_count, expected_episode)):
            errors.append(f"{path}: episode_index is not contiguous/reindexed")
        if not np.array_equal(frame_index, np.arange(frame_count)):
            errors.append(f"{path}: frame_index is not contiguous")
        if not np.array_equal(global_index, np.arange(expected_global, expected_global + frame_count)):
            errors.append(f"{path}: global index is not contiguous")
        expected_global += frame_count

        row = episodes.get(expected_episode)
        if row is None or int(row.get("length", -1)) != frame_count:
            errors.append(f"{path}: episodes.jsonl length/index is stale")
        errors.extend(
            _check_stats(
                path,
                table,
                stats.get(expected_episode, {}),
                [contract["state_key"], contract["action_key"], "timestamp", "frame_index", "episode_index", "index", "task_index"],
            )
        )

        chunk = expected_episode // chunk_size
        for video_key in contract["camera_features"]:
            feature = info["features"][video_key]
            if feature.get("dtype") == "image":
                if len(table[video_key]) != frame_count:
                    errors.append(f"{path}: image column {video_key} frame count mismatch")
                continue
            try:
                video = root / str(info["video_path"]).format(
                    episode_chunk=chunk, video_key=video_key, episode_index=expected_episode
                )
            except KeyError:
                errors.append("video_path template is invalid")
                continue
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                errors.append(f"cannot open video {video}")
                continue
            video_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
            if video_frames != frame_count:
                errors.append(f"{video}: {video_frames} frames != parquet {frame_count}")

    if total_frames != int(info.get("total_frames", -1)):
        errors.append(f"computed total_frames {total_frames} != info {info.get('total_frames')}")
    if set(episodes) != set(range(len(parquets))):
        errors.append("episodes.jsonl indexes are not contiguous")
    if set(stats) != set(range(len(parquets))):
        errors.append("episodes_stats.jsonl indexes are not contiguous")
    if not errors:
        print(
            f"OK LeRobot v2.1: {root} | schema={contract['schema']} "
            f"arm={contract['arm_mode']}/{contract['arm_side']} "
            f"state/raw/model={contract['state_dim']}/{contract['raw_action_dim']}/{contract['model_action_dim']} "
            f"format={contract['contract_format']} episodes={len(parquets)} frames={total_frames} fps={fps}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="collected .npz or LeRobot dataset root")
    parser.add_argument("--action-offset", type=int, default=None)
    args = parser.parse_args()
    path = Path(args.path).expanduser()
    if not path.exists():
        print(f"FAILED: {path}\n  - path does not exist")
        return 1
    errors = check_npz(path, args.action_offset) if path.is_file() else check_dataset(path)
    if errors:
        print(f"FAILED: {path}")
        for error in errors:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
