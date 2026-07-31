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
        state = np.asarray(sample["observation.state"])
        action = np.asarray(sample["action"])
        if state.shape[-1] != 7 or action.shape[-1] != 7:
            raise ValueError(f"sample {index}: state={state.shape}, action={action.shape}, expected last dim 7")
        samples.append({"index": index, "state_shape": list(state.shape), "action_shape": list(action.shape)})
    print(json.dumps({"dataset_id": args.dataset_id, "frames": len(dataset), "fps": metadata.fps, "samples": samples}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
