"""Exercise the real RECAP dataset reader before allocating a training GPU."""
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rlinf.data.datasets.recap.value_dataset import ValueDataset

root = Path(os.environ["TASK_ROOT"])
records = []
for split in ["train", "heldout"]:
    path = root / f"datasets/SYNTHETIC_recap_{split}"
    sidecar = pq.read_table(path / "meta/returns_synthetic_20261101.parquet").to_pydict()
    keys = list(zip(sidecar["episode_index"], sidecar["frame_index"]))
    assert len(set(keys)) == len(keys)
    for episode in set(sidecar["episode_index"]):
        indices = [i for i, ep in enumerate(sidecar["episode_index"]) if ep == episode]
        last = max(indices, key=lambda i: sidecar["frame_index"][i])
        assert sidecar["reward"][last] == (0 if episode % 2 == 0 else -20)
    dataset = ValueDataset(str(path), robot_type="libero", model_type="pi05", action_dim=32,
                           action_horizon=10, tag="synthetic_20261101", return_min=-37, return_max=0)
    sample = dataset[0]
    assert -1 <= sample["target_values"] <= 0
    assert "SYNTHETIC" in sample["prompt"]
    assert {"base_0_rgb", "left_wrist_0_rgb"}.issubset(sample["images"])
    assert np.isfinite(np.asarray(sample["images"]["base_0_rgb"])).all()
    assert len(dataset) == len(keys)
    records.append({"split": split, "frames": len(dataset), "returns_match_outcomes": True,
                    "actual_value_dataset_reader": "passed"})
(root / "records/data-contract.json").write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
