"""Create clearly marked synthetic LeRobot data for RECAP integration tests."""
import json
import os
import inspect
import time
from pathlib import Path

import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

root = Path(os.environ["TASK_ROOT"])
records = []
for split, episodes, seed in [("train", 4, 20261101), ("heldout", 2, 20261102)]:
    target = root / "datasets" / f"SYNTHETIC_recap_{split}"
    if target.exists():
        marker = target / "SYNTHETIC_DATASET.json"
        if marker.exists():
            records.append(json.loads(marker.read_text()))
            continue
        assert target.resolve().is_relative_to((root / "datasets").resolve())
        incomplete = root / "datasets/incomplete"
        incomplete.mkdir(exist_ok=True)
        target.rename(incomplete / f"{target.name}-{time.time_ns()}")
    features = {
        "image": {"dtype": "image", "shape": (64, 64, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (64, 64, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": [f"state_{i}" for i in range(8)]},
        "actions": {"dtype": "float32", "shape": (7,), "names": [f"action_{i}" for i in range(7)]},
        "is_success": {"dtype": "bool", "shape": (1,), "names": ["synthetic_success"]},
    }
    dataset = LeRobotDataset.create(f"synthetic/recap_{split}", fps=10, root=target,
                                    robot_type="synthetic_libero", features=features, use_videos=False)
    rng = np.random.default_rng(seed)
    states, actions = [], []
    for episode in range(episodes):
        length = 12 + episode * 2
        success = episode % 2 == 0
        for frame in range(length):
            state = rng.normal(0, 0.1, 8).astype(np.float32)
            action = rng.normal(0, 0.05, 7).astype(np.float32)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[..., 0] = (episode * 55 + (0 if split == "train" else 27)) % 256
            image[..., 1] = frame * 12 % 256
            image[16:40, 12 + frame % 12:32 + frame % 12, 2] = 255
            sample = {"image": image, "wrist_image": np.flip(image, axis=1).copy(),
                      "state": state, "actions": action, "is_success": np.array([success], dtype=bool)}
            prompt = "SYNTHETIC TEST ONLY: move the colored block"
            if "task" in inspect.signature(dataset.add_frame).parameters:
                dataset.add_frame(sample, task=prompt)
            else:
                dataset.add_frame({**sample, "task": prompt})
            states.append(state)
            actions.append(action)
        dataset.save_episode()
    norm = {}
    for key, values in [("state", states), ("actions", actions)]:
        values = np.stack(values)
        norm[key] = {"mean": values.mean(0).tolist(), "std": values.std(0).clip(1e-6).tolist(),
                     "q01": np.quantile(values, 0.01, axis=0).tolist(),
                     "q99": np.quantile(values, 0.99, axis=0).tolist()}
    (target / "meta" / "norm_stats.json").write_text(json.dumps({"norm_stats": norm}, indent=2))
    record = {"synthetic": True, "purpose": "integration_test_only", "split": split,
              "seed": seed, "path": str(target), "episodes": episodes, "frames": len(states),
              "success_episodes": (episodes + 1) // 2, "failure_episodes": episodes // 2}
    (target / "SYNTHETIC_DATASET.json").write_text(json.dumps(record, indent=2))
    records.append(record)
(root / "records" / "datasets.json").write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
