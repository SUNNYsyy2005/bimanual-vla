"""Export collected NPZ episodes to LeRobot v2.1.

The exporter keeps only successful episodes by default and writes the exact
10D state / 7D action schema required by the Piper delivery format.

Example:
    python export_lerobot.py \
      --input-dir episodes_output_arm \
      --repo-id piper/piper_v1 \
      --root piper/piper_v1
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np


FEATURES = {
    "image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (10,),
        "names": [
            "eef_x_base_m", "eef_y_base_m", "eef_z_base_m",
            "rotation6d_col0_x", "rotation6d_col0_y", "rotation6d_col0_z",
            "rotation6d_col1_x", "rotation6d_col1_y", "rotation6d_col1_z",
            "gripper_closed_fraction",
        ],
    },
    "actions": {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "delta_x_base_m", "delta_y_base_m", "delta_z_base_m",
            "delta_rx_base_rad", "delta_ry_base_rad", "delta_rz_base_rad",
            "gripper_target_closed_fraction",
        ],
    },
}


def _load_successful(paths: list[Path]):
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if not bool(data["success"].item()):
                continue
            required = {"state", "actions", "image", "wrist_image", "task"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path}: missing fields {sorted(missing)}")
            state = np.asarray(data["state"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            image = np.asarray(data["image"], dtype=np.uint8)
            wrist = np.asarray(data["wrist_image"], dtype=np.uint8)
            if state.ndim != 2 or state.shape[1] != 10:
                raise ValueError(f"{path}: state shape must be (T,10), got {state.shape}")
            if actions.shape != (len(state), 7):
                raise ValueError(f"{path}: actions shape must be ({len(state)},7), got {actions.shape}")
            if image.shape != (len(state), 256, 256, 3):
                raise ValueError(f"{path}: image shape mismatch: {image.shape}")
            if wrist.shape != image.shape:
                raise ValueError(f"{path}: wrist_image shape mismatch: {wrist.shape}")
            if not np.isfinite(state).all() or not np.isfinite(actions).all():
                raise ValueError(f"{path}: non-finite state/action")
            if not (0.0 <= state[:, 9]).all() or not (state[:, 9] <= 1.0).all():
                raise ValueError(f"{path}: state gripper fraction out of range")
            if not (0.0 <= actions[:, 6]).all() or not (actions[:, 6] <= 1.0).all():
                raise ValueError(f"{path}: action gripper fraction out of range")
            if not np.allclose(actions[-1, :6], 0.0, atol=1e-6):
                raise ValueError(f"{path}: terminal action is not a no-op")
            yield path, state, actions, image, wrist, str(data["task"].item())


def run(args):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        codebase_version = "v2.1"
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
            codebase_version = CODEBASE_VERSION
        except ImportError as exc:
            raise SystemExit(
                "LeRobot is not installed in this environment. Install the project "
                "environment containing lerobot, then rerun this exporter."
            ) from exc

    if codebase_version != "v2.1":
        raise SystemExit(
            f"Installed LeRobot creates datasets with codebase_version={codebase_version}, "
            "but this delivery requires LeRobot v2.1. Use the OpenPI-pinned "
            "LeRobot environment before exporting."
        )

    if args.fps != 20:
        raise SystemExit("The Piper delivery format requires fps=20.")

    paths = sorted(Path(args.input_dir).glob("ep_*.npz"))
    if not paths:
        raise SystemExit(f"No episodes found in {args.input_dir}")

    create_kwargs = {
        "repo_id": args.repo_id,
        "robot_type": "piper",
        "fps": args.fps,
        "features": FEATURES,
        "use_videos": False,
    }
    if args.root is not None and "root" in inspect.signature(LeRobotDataset.create).parameters:
        create_kwargs["root"] = args.root
    dataset = LeRobotDataset.create(**create_kwargs)

    count = 0
    frames = 0
    for path, state, actions, image, wrist, task in _load_successful(paths):
        for i in range(len(state)):
            dataset.add_frame({
                "image": image[i],
                "wrist_image": wrist[i],
                "state": state[i],
                "actions": actions[i],
                "task": task,
            })
        dataset.save_episode()
        count += 1
        frames += len(state)
        print(f"Exported {path} ({len(state)} frames)")

    print(f"Export complete: episodes={count}, frames={frames}, fps={args.fps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="episodes_piper_v21")
    ap.add_argument("--repo-id", default="piper/piper_v1")
    ap.add_argument("--root", default="piper/piper_v1")
    ap.add_argument("--fps", type=int, default=20)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
