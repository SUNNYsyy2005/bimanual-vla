#!/usr/bin/env python3
"""Validate a collected NPZ or a generated LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from pi0_dataset import derive_absolute_actions


def _check_images(name: str, frames: np.ndarray, frame_count: int, errors: list[str]) -> None:
    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        errors.append(f"{name}: dtype {arr.dtype}, expected uint8")
    if arr.ndim != 4:
        errors.append(f"{name}: shape {arr.shape}, expected rank 4")
        return
    if len(arr) != frame_count:
        errors.append(f"{name}: {len(arr)} frames, expected {frame_count}")
    if arr.shape[1] != 3 and arr.shape[-1] != 3:
        errors.append(f"{name}: cannot identify RGB channel in {arr.shape}")


def check_npz(path: Path, action_offset: int | None) -> list[str]:
    errors: list[str] = []
    with np.load(path, allow_pickle=False) as data:
        state_key = "qpos" if "qpos" in data.files else "observation.state"
        if state_key not in data.files:
            return ["missing qpos/observation.state"]
        states = np.asarray(data[state_key], dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 7 or len(states) == 0:
            errors.append(f"state shape {states.shape}, expected (T, 7), T>0")
        if not np.isfinite(states).all():
            errors.append("state contains NaN/Inf")

        action_key = "actions" if "actions" in data.files else "action"
        if action_key in data.files:
            actions = np.asarray(data[action_key], dtype=np.float32)
            if actions.shape != states.shape:
                errors.append(f"action shape {actions.shape} != state shape {states.shape}")
            if not np.isfinite(actions).all():
                errors.append("action contains NaN/Inf")
            offset = action_offset
            if offset is None and "action_offset" in data.files:
                offset = int(np.asarray(data["action_offset"]).item())
            if offset is not None and actions.shape == states.shape:
                expected = derive_absolute_actions(states, offset)
                if not np.array_equal(actions, expected):
                    max_err = float(np.max(np.abs(actions - expected)))
                    errors.append(f"action is not qpos[t+{offset}] padded at end; max error={max_err}")
        else:
            errors.append("missing action/actions (convert script can derive it)")

        timestamp_key = next((key for key in ("timestamps", "capture_timestamps", "timestamp") if key in data.files), None)
        if timestamp_key is None:
            errors.append("missing timestamps")
        else:
            timestamps = np.asarray(data[timestamp_key], dtype=np.float64)
            if timestamps.ndim != 1 or len(timestamps) != len(states):
                errors.append(f"{timestamp_key} shape {timestamps.shape}, expected ({len(states)},)")
            elif len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
                errors.append(f"{timestamp_key} is not strictly increasing")

        image_keys = [key for key in data.files if key.startswith("images_") or key.startswith("observation.images.")]
        high_keys = [key for key in image_keys if key.endswith("cam_high")]
        wrist_keys = [key for key in image_keys if "wrist" in key]
        if len(high_keys) != 1:
            errors.append(f"expected one cam_high stream, found {high_keys}")
        if len(wrist_keys) != 1:
            errors.append(f"expected one wrist stream, found {wrist_keys}")
        for key in high_keys[:1] + wrist_keys[:1]:
            _check_images(key, data[key], len(states), errors)

        if not errors:
            dt = np.diff(np.asarray(data[timestamp_key], dtype=np.float64))
            fps = 1.0 / np.median(dt) if len(dt) else float("nan")
            print(f"OK NPZ: {path} | T={len(states)} state={states.shape} capture_fps≈{fps:.2f}")
    return errors


def check_dataset(root: Path) -> list[str]:
    errors: list[str] = []
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return [f"missing {info_path}"]
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        errors.append(f"codebase_version={info.get('codebase_version')!r}, expected 'v2.1'")
    fps = int(info.get("fps", 0))
    if fps <= 0:
        errors.append(f"invalid fps={fps}")
    offset = int(info.get("action_offset", 0))
    semantics = info.get("action_semantics")
    if semantics != "absolute_next_joint_position":
        errors.append(f"action_semantics={semantics!r}, expected 'absolute_next_joint_position'")

    camera_keys = [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]
    if len(camera_keys) != 2 or "observation.images.cam_high" not in camera_keys or not any("wrist" in key for key in camera_keys):
        errors.append(f"expected cam_high + one wrist video feature, found {camera_keys}")

    parquets = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquets) != int(info.get("total_episodes", -1)):
        errors.append(f"parquet count {len(parquets)} != total_episodes {info.get('total_episodes')}")
    total_frames = 0
    for episode_index, path in enumerate(parquets):
        table = pq.read_table(path)
        required = {"observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"}
        missing = sorted(required - set(table.column_names))
        if missing:
            errors.append(f"{path}: missing columns {missing}")
            continue
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        timestamps = np.asarray(table["timestamp"], dtype=np.float32)
        total_frames += len(states)
        if states.ndim != 2 or states.shape[1] != 7:
            errors.append(f"{path}: state shape {states.shape}, expected (T, 7)")
        if actions.shape != states.shape:
            errors.append(f"{path}: action shape {actions.shape} != state shape {states.shape}")
        elif not np.array_equal(actions, derive_absolute_actions(states, offset)):
            errors.append(f"{path}: actions do not match shifted absolute qpos with offset={offset}")
        expected_ts = np.arange(len(states), dtype=np.float32) / fps if fps else np.zeros(len(states), np.float32)
        if not np.allclose(timestamps, expected_ts, atol=1e-6):
            errors.append(f"{path}: timestamps are not frame_index/fps")

        chunk = episode_index // int(info.get("chunks_size", 1000))
        for video_key in camera_keys:
            video = root / info["video_path"].format(
                episode_chunk=chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                errors.append(f"cannot open video {video}")
                continue
            frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
            if frames != len(states):
                errors.append(f"{video}: {frames} frames != parquet {len(states)}")

    if total_frames != int(info.get("total_frames", -1)):
        errors.append(f"computed total_frames {total_frames} != info {info.get('total_frames')}")
    if not errors:
        print(
            f"OK LeRobot v2.1: {root} | episodes={len(parquets)} frames={total_frames} "
            f"fps={fps} cameras={camera_keys} action_offset={offset}"
        )
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="collected .npz or LeRobot dataset root")
    ap.add_argument("--action-offset", type=int, default=None, help="expected NPZ action offset")
    args = ap.parse_args()
    path = Path(args.path).expanduser()
    if not path.exists():
        print(f"FAILED: {path}")
        print("  - path does not exist")
        return 1
    if path.is_file() and path.suffix.lower() != ".npz":
        print(f"FAILED: {path}")
        print("  - expected a .npz file or a LeRobot dataset directory")
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
