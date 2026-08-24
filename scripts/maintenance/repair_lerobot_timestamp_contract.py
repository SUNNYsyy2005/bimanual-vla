#!/usr/bin/env python3
"""Repair a LeRobot v2.1 export with explicit Piper contract timestamps.

The generic LeRobot Aloha converter writes the canonical ``timestamp`` column,
but the Dashboard/Piper contract also requires state, action, and per-camera
capture timestamps.  This tool adds those columns in place without decoding or
rewriting the image/video payload semantics, then writes the contract metadata
used by the OpenPI/Piper validators.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def replace_or_append(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, array)
    return table.append_column(name, array)


def flatten_names(value: object, fallback: list[str]) -> list[str]:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return fallback


def repair(root: Path, fps: int) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    state_feature = features["observation.state"]
    action_feature = features["action"]
    fallback_names = [f"joint_{i}" for i in range(14)]
    state_names = flatten_names(state_feature.get("names"), fallback_names)
    action_names = flatten_names(action_feature.get("names"), state_names)
    if len(state_names) != 14 or len(action_names) != 14:
        raise ValueError(f"expected bimanual 14D/14D state/action names, got {len(state_names)}/{len(action_names)}")
    state_feature["names"] = state_names
    action_feature["names"] = action_names

    camera_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    timestamp_features = {
        "state_timestamp": {"dtype": "float64", "shape": [1], "names": None},
        "action_timestamp": {"dtype": "float64", "shape": [1], "names": None},
    }
    timestamp_features.update({
        f"image_timestamp.{key}": {"dtype": "float64", "shape": [1], "names": None}
        for key in camera_keys
    })
    features.update(timestamp_features)

    parquets = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquets:
        raise ValueError(f"no episode parquet files under {root / 'data'}")
    total_frames = 0
    for path in parquets:
        table = pq.read_table(path)
        if "timestamp" in table.column_names:
            timestamps = np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
        else:
            timestamps = np.arange(table.num_rows, dtype=np.float64) / float(fps)
            table = replace_or_append(table, "timestamp", pa.array(timestamps, type=pa.float32()))
        expected = np.arange(table.num_rows, dtype=np.float64) / float(fps)
        if timestamps.shape != expected.shape or not np.allclose(timestamps, expected, atol=1e-5):
            raise ValueError(f"{path}: timestamp is not frame_index/{fps}")
        action_timestamps = np.concatenate((timestamps[1:], [timestamps[-1] + 1.0 / float(fps)]))
        values = {
            "state_timestamp": timestamps,
            "action_timestamp": action_timestamps,
            **{f"image_timestamp.{key}": timestamps for key in camera_keys},
        }
        for name, array in values.items():
            table = replace_or_append(table, name, pa.array(array, type=pa.float64()))
        temp = path.with_name(f".{path.name}.repair-{uuid.uuid4().hex}.tmp")
        pq.write_table(table, temp)
        os.replace(temp, path)
        total_frames += table.num_rows
        print(f"repaired {path.name}: {table.num_rows} frames", flush=True)

    info.update({
        "contract_version": 3,
        "schema": "joint",
        "arm_mode": "bimanual",
        "arm_side": "both",
        "state_dim": 14,
        "action_dim": 14,
        "raw_action_dim": 14,
        "model_action_dim": 14,
        "state_names": state_names,
        "action_names": action_names,
        "model_action_names": action_names,
        "camera_keys": camera_keys,
        "action_semantics": "absolute_joint_position_opening_fraction",
        "model_action_semantics": "absolute_joint_position_opening_fraction",
        "action_source": "next_measured_joint_fallback",
        "action_alignment": "next_observation",
        "action_offset": 1,
        "action_horizon": 50,
        "contract_format": "canonical",
        "legacy": False,
        "legacy_format": None,
        "delivery_action_format": None,
        "gripper_semantics": "absolute_opening_fraction_0_closed_1_open",
        "rotation_semantics": "joint_positions_rad_first_6",
        "coordinate_frame": "slave_base",
        "source_frame": "slave_base",
        "timestamp_fields": {
            "canonical": "timestamp",
            "state": "state_timestamp",
            "action": "action_timestamp",
            "images": {key: f"image_timestamp.{key}" for key in camera_keys},
        },
        "total_episodes": len(parquets),
        "total_frames": total_frames,
        "total_chunks": len({p.parent.name for p in parquets}),
    })
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    policy_contract = {
        "version": 3,
        "robot_type": info.get("robot_type", "aloha"),
        **{key: info[key] for key in (
            "schema", "arm_mode", "arm_side", "state_dim", "raw_action_dim",
            "model_action_dim", "state_names", "action_names", "model_action_names",
            "camera_keys", "action_semantics", "model_action_semantics", "action_source",
            "action_alignment", "action_offset", "action_horizon", "contract_format",
            "legacy", "legacy_format", "delivery_action_format", "gripper_semantics",
            "rotation_semantics", "coordinate_frame", "source_frame",
        )},
    }
    (root / "meta" / "policy_contract.json").write_text(
        json.dumps(policy_contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"episodes": len(parquets), "frames": total_frames, "fps": fps, "root": str(root)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    repair(args.root.expanduser().resolve(), args.fps)


if __name__ == "__main__":
    main()
