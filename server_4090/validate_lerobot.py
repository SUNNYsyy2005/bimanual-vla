#!/usr/bin/env python3
"""Validate an installed LeRobot dataset and exercise the loader.

This accepts both canonical Piper delivery (10D/10D or 20D/20D absolute EEF)
and the observed metadata-free/marked ``legacy_v2`` layout (10D/7D or
20D/14D step delta). The dimensions, contract marker, timestamps and video
checks are performed before sampling the LeRobot loader.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from check_pi05_dataset import _dataset_contract, check_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _dataset_root(dataset_id: str) -> Path:
    root = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
    return root / dataset_id


def _dataset_info(dataset_id: str) -> dict:
    path = _dataset_root(dataset_id) / "meta" / "info.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    args = parser.parse_args()
    root = _dataset_root(args.dataset_id)
    info = _dataset_info(args.dataset_id)
    errors = check_dataset(root)
    if errors:
        raise ValueError("dataset contract validation failed:\n  - " + "\n  - ".join(errors))
    contract = _dataset_contract(info)

    metadata = LeRobotDatasetMetadata(args.dataset_id)
    dataset = LeRobotDataset(args.dataset_id)
    if len(dataset) <= 0:
        raise ValueError("dataset is empty")

    features = info.get("features", {})
    state_key, action_key = contract["state_key"], contract["action_key"]
    camera_fields = contract["camera_features"]
    indexes = sorted({0, len(dataset) - 1})
    samples = []
    for index in indexes:
        sample = dataset[index]
        missing = sorted(key for key in {state_key, action_key, *camera_fields} if key not in sample)
        if missing:
            raise ValueError(f"sample {index}: missing fields {missing}")
        state = np.asarray(sample[state_key])
        action = np.asarray(sample[action_key])
        if state.shape[-1] != contract["state_dim"] or action.shape[-1] != contract["raw_action_dim"]:
            raise ValueError(
                f"sample {index}: state={state.shape}, action={action.shape}, "
                f"expected last dims {contract['state_dim']}/{contract['raw_action_dim']}"
            )
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"sample {index}: state/action contains NaN/Inf")
        samples.append(
            {
                "index": index,
                "state_shape": list(state.shape),
                "action_shape": list(action.shape),
                "camera_fields": camera_fields,
            }
        )
    print(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "schema": contract["schema"],
                "arm_mode": contract["arm_mode"],
                "arm_side": contract["arm_side"],
                "state_dim": contract["state_dim"],
                "raw_action_dim": contract["raw_action_dim"],
                "model_action_dim": contract["model_action_dim"],
                "contract_format": contract["contract_format"],
                "legacy_format": contract["legacy_format"],
                "camera_keys": contract["camera_keys"],
                "layout": contract["column_layout"],
                "frames": len(dataset),
                "fps": metadata.fps,
                "samples": samples,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
