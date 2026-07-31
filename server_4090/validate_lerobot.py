#!/usr/bin/env python3
"""Load a local LeRobot dataset exactly as OpenPI's data loader will."""

from __future__ import annotations

import argparse
import json
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    args = parser.parse_args()
    metadata = LeRobotDatasetMetadata(args.dataset_id)
    dataset = LeRobotDataset(args.dataset_id)
    if len(dataset) <= 0:
        raise ValueError("dataset is empty")
    indexes = sorted({0, len(dataset) - 1})
    samples = []
    for index in indexes:
        sample = dataset[index]
        if "state" in sample and "actions" in sample:
            schema = "delivery"
            state_key, action_key = "state", "actions"
            state_dim = 10
            required_images = {"image", "wrist_image"}
        else:
            schema = "joint"
            state_key, action_key = "observation.state", "action"
            state_dim = 7
            wrist_keys = [
                key for key in sample
                if key.startswith("observation.images.") and "wrist" in key
            ]
            if not wrist_keys:
                raise ValueError(f"sample {index}: missing wrist image field")
            required_images = {
                "observation.images.cam_high",
                wrist_keys[0],
            }
        missing = sorted(key for key in {state_key, action_key, *required_images} if key not in sample)
        if missing:
            raise ValueError(f"sample {index}: missing fields {missing}")
        state = np.asarray(sample[state_key])
        action = np.asarray(sample[action_key])
        if state.shape[-1] != state_dim or action.shape[-1] != 7:
            raise ValueError(
                f"sample {index}: state={state.shape}, action={action.shape}, "
                f"expected last dims {state_dim}/7"
            )
        samples.append({"index": index, "state_shape": list(state.shape), "action_shape": list(action.shape)})
    print(json.dumps({"dataset_id": args.dataset_id, "schema": schema, "frames": len(dataset), "fps": metadata.fps, "samples": samples}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
