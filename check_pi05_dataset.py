#!/usr/bin/env python3
"""Validate Piper raw NPZ files and LeRobot v2.1 datasets.

Supported contracts:
- single joint: state/action 7D, cam_high + one wrist camera
- bimanual joint: state/action 14D, cam_high + left/right wrist cameras
- single delivery: state/action 10D/7D
- bimanual delivery: state/action 20D/14D
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

from pi0_dataset import derive_absolute_actions
from piper_data_contract import EpisodeContract
from validate_piper_data import EpisodeValidationError, validate_episode


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def check_npz(path: Path, action_offset: int | None) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        if {"state", "actions", "instruction", "success"}.issubset(data.files):
            try:
                validate_episode(path)
                print(f"OK Piper NPZ: {path}")
                return []
            except EpisodeValidationError as exc:
                return list(exc.errors)

        errors: list[str] = []
        state_key = next((key for key in ("qpos", "joint_qpos", "observation.state") if key in data.files), None)
        if state_key is None:
            return ["missing state/qpos/observation.state"]
        states = np.asarray(data[state_key], dtype=np.float32)
        if states.ndim != 2 or states.shape[1] not in {7, 14} or len(states) == 0:
            errors.append(f"state shape {states.shape}, expected (T, 7|14), T>0")
        if not np.isfinite(states).all():
            errors.append("state contains NaN/Inf")

        action_key = next((key for key in ("actions", "action") if key in data.files), None)
        if action_key is None:
            errors.append("missing action/actions")
        else:
            actions = np.asarray(data[action_key], dtype=np.float32)
            if actions.shape != states.shape:
                errors.append(f"action shape {actions.shape} != state shape {states.shape}")
            if not np.isfinite(actions).all():
                errors.append("action contains NaN/Inf")
            offset = action_offset
            if offset is None and "action_offset" in data.files:
                offset = int(np.asarray(data["action_offset"]).item())
            alignment = str(np.asarray(data["action_alignment"]).item()) if "action_alignment" in data.files else ""
            if offset is not None and alignment != "same_step_command" and actions.shape == states.shape:
                expected = derive_absolute_actions(states, offset)
                if not np.allclose(actions, expected, atol=1e-6):
                    errors.append(f"actions do not match shifted measured qpos with offset={offset}")

        camera_fields = [key for key in data.files if key.startswith("images_") or key.startswith("observation.images.")]
        expected_cameras = 3 if states.ndim == 2 and states.shape[1] == 14 else 2
        if len(camera_fields) < expected_cameras:
            errors.append(f"found {len(camera_fields)} camera arrays, expected at least {expected_cameras}")
        for key in camera_fields:
            frames = np.asarray(data[key])
            if frames.dtype != np.uint8 or frames.ndim != 4 or len(frames) != len(states):
                errors.append(f"{key}: invalid image array {frames.shape}/{frames.dtype}")

    if not errors:
        print(f"OK legacy joint NPZ: {path} | state={states.shape}")
    return errors


def _infer_contract(info: dict[str, Any]) -> tuple[EpisodeContract, str, str, str, list[str]]:
    features = info.get("features", {})
    legacy_delivery = (
        features.get("state", {}).get("shape") in ([10], [20])
        and features.get("actions", {}).get("shape") in ([7], [14])
    )
    if legacy_delivery:
        layout = "legacy"
        state_key, action_key = "state", "actions"
        state_dim = int(features[state_key]["shape"][0])
        camera_features = [key for key in ("image", "wrist_image") if key in features]
        camera_keys = ("cam_high", "cam_wrist")
    else:
        layout = "canonical"
        state_key, action_key = "observation.state", "action"
        shape = features.get(state_key, {}).get("shape")
        if not isinstance(shape, list) or len(shape) != 1:
            raise ValueError("missing one-dimensional observation.state feature")
        state_dim = int(shape[0])
        camera_features = sorted(
            key for key, value in features.items()
            if key.startswith("observation.images.") and value.get("dtype") in {"image", "video"}
        )
        camera_keys = tuple(key.removeprefix("observation.images.") for key in camera_features)

    schema = str(info.get("schema") or ("delivery" if state_dim in {10, 20} else "joint"))
    per_arm = 10 if schema == "delivery" else 7
    arm_mode = str(info.get("arm_mode") or ("bimanual" if state_dim == 2 * per_arm else "single"))
    arm_side = str(info.get("arm_side") or ("both" if arm_mode == "bimanual" else "right"))
    contract = EpisodeContract(
        schema=schema,
        arm_mode=arm_mode,
        arm_side=arm_side,
        camera_keys=camera_keys,
        action_source=str(info.get("action_source") or ("next_measured_eef" if schema == "delivery" else "next_measured_qpos")),
        action_alignment=str(info.get("action_alignment") or ("next_observation" if int(info.get("action_offset", 0)) == 1 else "same_step_command")),
    )
    return contract, layout, state_key, action_key, camera_features


def check_dataset(root: Path) -> list[str]:
    errors: list[str] = []
    info = _json(root / "meta" / "info.json")
    if not isinstance(info, dict):
        return ["missing or invalid meta/info.json"]
    if info.get("codebase_version") != "v2.1":
        errors.append(f"codebase_version={info.get('codebase_version')!r}, expected 'v2.1'")
    try:
        contract, layout, state_key, action_key, camera_features = _infer_contract(info)
    except (TypeError, ValueError) as exc:
        return [f"unsupported dataset contract: {exc}"]

    features = info.get("features", {})
    expected_dims = {
        state_key: contract.state_dim,
        action_key: contract.action_dim,
    }
    for key, dim in expected_dims.items():
        feature = features.get(key, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [dim]:
            errors.append(f"feature {key}={feature}, expected float32 [{dim}]")

    for key, expected in {
        "schema": contract.schema,
        "arm_mode": contract.arm_mode,
        "arm_side": contract.arm_side,
        "state_dim": contract.state_dim,
        "action_dim": contract.action_dim,
        "camera_keys": list(contract.camera_keys),
        "action_semantics": contract.action_semantics,
        "action_source": contract.action_source,
        "action_alignment": contract.action_alignment,
        "action_offset": contract.action_offset,
    }.items():
        if key in info and info[key] != expected:
            errors.append(f"metadata {key}={info[key]!r}, expected {expected!r}")

    if len(camera_features) != len(contract.camera_keys):
        errors.append(f"camera features {camera_features}, expected {list(contract.camera_keys)}")

    policy_contract = _json(root / "meta" / "policy_contract.json")
    if isinstance(policy_contract, dict):
        for key in ("schema", "arm_mode", "arm_side", "state_dim", "action_dim", "camera_keys", "action_semantics", "action_source", "action_alignment", "action_offset"):
            if policy_contract.get(key) != info.get(key):
                errors.append(f"policy_contract {key}={policy_contract.get(key)!r} != info {info.get(key)!r}")

    fps = float(info.get("fps", 0))
    if fps <= 0:
        errors.append(f"invalid fps={fps}")
    chunk_size = int(info.get("chunks_size", 1000))
    parquets = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquets) != int(info.get("total_episodes", -1)):
        errors.append(f"parquet count {len(parquets)} != total_episodes {info.get('total_episodes')}")

    total_frames = 0
    for path in parquets:
        table = pq.read_table(path)
        required = {state_key, action_key, "timestamp", "frame_index", "episode_index", "index", "task_index"}
        if layout == "legacy":
            required.update(camera_features)
        missing = sorted(required - set(table.column_names))
        if missing:
            errors.append(f"{path}: missing columns {missing}")
            continue
        states = np.asarray(table[state_key].to_pylist(), dtype=np.float32)
        actions = np.asarray(table[action_key].to_pylist(), dtype=np.float32)
        timestamps = np.asarray(table["timestamp"], dtype=np.float32)
        total_frames += len(states)
        if states.shape != (len(states), contract.state_dim):
            errors.append(f"{path}: state shape {states.shape}, expected (T,{contract.state_dim})")
        if actions.shape != (len(states), contract.action_dim):
            errors.append(f"{path}: action shape {actions.shape}, expected (T,{contract.action_dim})")
        elif contract.schema == "joint" and contract.action_alignment == "next_observation":
            expected = derive_absolute_actions(states, contract.action_offset)
            if not np.allclose(actions, expected, atol=1e-6):
                errors.append(f"{path}: next-observation joint actions are misaligned")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            errors.append(f"{path}: state/action contains NaN/Inf")
        expected_ts = np.arange(len(states), dtype=np.float32) / fps if fps else np.zeros(len(states), np.float32)
        if not np.allclose(timestamps, expected_ts, atol=1e-6):
            errors.append(f"{path}: timestamps are not frame_index/fps")

        episode_index = int(path.stem.removeprefix("episode_"))
        chunk = episode_index // chunk_size
        for video_key in camera_features:
            feature = features[video_key]
            if feature.get("dtype") != "video":
                continue
            video = root / info["video_path"].format(
                episode_chunk=chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                errors.append(f"cannot open video {video}")
                continue
            frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
            if frame_count != len(states):
                errors.append(f"{video}: {frame_count} frames != parquet {len(states)}")

    if total_frames != int(info.get("total_frames", -1)):
        errors.append(f"computed total_frames {total_frames} != info {info.get('total_frames')}")
    if not errors:
        print(
            f"OK LeRobot v2.1: {root} | schema={contract.schema} "
            f"arm={contract.arm_mode}/{contract.arm_side} state={contract.state_dim} "
            f"action={contract.action_dim} cameras={list(contract.camera_keys)} "
            f"episodes={len(parquets)} frames={total_frames} fps={fps} layout={layout}"
        )
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="collected .npz or LeRobot dataset root")
    ap.add_argument("--action-offset", type=int, default=None, help="expected legacy NPZ action offset")
    args = ap.parse_args()
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
