#!/usr/bin/env python3
"""Load a local single-arm or bimanual Piper dataset as OpenPI will."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _dataset_info(dataset_id: str) -> dict:
    root = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
    path = root / dataset_id / "meta" / "info.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    args = parser.parse_args()
    info = _dataset_info(args.dataset_id)
    metadata = LeRobotDatasetMetadata(args.dataset_id)
    dataset = LeRobotDataset(args.dataset_id)
    if len(dataset) <= 0:
        raise ValueError("dataset is empty")

    features = info.get("features", {})
    legacy = "state" in features and "actions" in features
    if legacy:
        state_key, action_key = "state", "actions"
        camera_fields = [key for key in ("image", "wrist_image") if key in features]
        state_dim = int(features[state_key]["shape"][0])
        action_dim = int(features[action_key]["shape"][0])
        camera_keys = ["cam_high", "cam_wrist"]
    else:
        state_key, action_key = "observation.state", "action"
        state_dim = int(features[state_key]["shape"][0])
        action_dim = int(features[action_key]["shape"][0])
        camera_fields = sorted(
            key for key, value in features.items()
            if key.startswith("observation.images.") and value.get("dtype") in {"image", "video"}
        )
        camera_keys = [key.removeprefix("observation.images.") for key in camera_fields]

    schema = str(info.get("schema") or ("delivery" if state_dim in {10, 20} else "joint"))
    per_arm = 10 if schema == "delivery" else 7
    arm_mode = str(info.get("arm_mode") or ("bimanual" if state_dim == 2 * per_arm else "single"))
    expected_state = per_arm * (2 if arm_mode == "bimanual" else 1)
    expected_action = 7 * (2 if arm_mode == "bimanual" else 1)
    expected_cameras = 3 if arm_mode == "bimanual" else 2
    if state_dim != expected_state or action_dim != expected_action:
        raise ValueError(
            f"metadata dims {state_dim}/{action_dim} disagree with {schema}/{arm_mode} "
            f"expected {expected_state}/{expected_action}"
        )
    if len(camera_fields) != expected_cameras:
        raise ValueError(f"camera fields {camera_fields}, expected {expected_cameras}")

    indexes = sorted({0, len(dataset) - 1})
    samples = []
    for index in indexes:
        sample = dataset[index]
        missing = sorted(key for key in {state_key, action_key, *camera_fields} if key not in sample)
        if missing:
            raise ValueError(f"sample {index}: missing fields {missing}")
        state = np.asarray(sample[state_key])
        action = np.asarray(sample[action_key])
        if state.shape[-1] != state_dim or action.shape[-1] != action_dim:
            raise ValueError(
                f"sample {index}: state={state.shape}, action={action.shape}, "
                f"expected last dims {state_dim}/{action_dim}"
            )
        samples.append({"index": index, "state_shape": list(state.shape), "action_shape": list(action.shape)})
    print(json.dumps({
        "dataset_id": args.dataset_id,
        "schema": schema,
        "arm_mode": arm_mode,
        "arm_side": info.get("arm_side", "both" if arm_mode == "bimanual" else "right"),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "camera_keys": camera_keys,
        "layout": "legacy" if legacy else "canonical",
        "frames": len(dataset),
        "fps": metadata.fps,
        "samples": samples,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
