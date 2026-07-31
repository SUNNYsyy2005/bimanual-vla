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

from validate_piper_data import (
    EpisodeStats,
    EpisodeValidationError,
    format_dataset_report,
    format_episode_report,
    validate_episode,
    validate_gripper_coverage,
    validate_instruction_consistency,
)


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


def _validate_inputs(paths: list[Path], target_fps: float) -> list[EpisodeStats]:
    successful: list[EpisodeStats] = []
    coverage_candidates: list[EpisodeStats] = []
    failures: list[str] = []
    for path in paths:
        try:
            stats = validate_episode(path, target_fps=target_fps)
            print(format_episode_report(stats))
            if stats.success:
                successful.append(stats)
                coverage_candidates.append(stats)
        except EpisodeValidationError as exc:
            failures.append(str(exc))
            if exc.stats is not None and exc.stats.success:
                coverage_candidates.append(exc.stats)

    coverage_error = None
    try:
        validate_gripper_coverage(coverage_candidates)
        validate_instruction_consistency(coverage_candidates)
    except ValueError as exc:
        coverage_error = str(exc)
    if failures:
        details = "Input validation failed:\n\n" + "\n\n".join(failures)
        if coverage_error:
            details += f"\n\nDataset validation failed: {coverage_error}"
        raise SystemExit(details)
    if coverage_error:
        raise SystemExit(f"Input validation failed: {coverage_error}")
    print(format_dataset_report(successful))
    return successful


def _load_episode_arrays(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["state"], dtype=np.float32),
            np.asarray(data["actions"], dtype=np.float32),
            np.asarray(data["image"], dtype=np.uint8),
            np.asarray(data["wrist_image"], dtype=np.uint8),
        )


def run(args):
    if args.fps != 20:
        raise SystemExit("The Piper delivery format requires fps=20.")

    paths = sorted(Path(args.input_dir).glob("ep_*.npz"))
    if not paths:
        raise SystemExit(f"No episodes found in {args.input_dir}")

    successful = _validate_inputs(paths, target_fps=args.fps)
    if args.validate_only:
        print("Validation complete; no LeRobot dataset was written.")
        return

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
    for stats in successful:
        state, actions, image, wrist = _load_episode_arrays(stats.path)
        for i in range(len(state)):
            dataset.add_frame({
                "image": image[i],
                "wrist_image": wrist[i],
                "state": state[i],
                "actions": actions[i],
            }, task=stats.instruction, timestamp=i / args.fps)
        dataset.save_episode()
        count += 1
        frames += len(state)
        print(
            f"Exported {stats.path} ({len(state)} frames, "
            f"instruction={stats.instruction!r})"
        )

    print(f"Export complete: episodes={count}, frames={frames}, fps={args.fps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="episodes_piper_v21")
    ap.add_argument("--repo-id", default="piper/piper_v1")
    ap.add_argument("--root", default="piper/piper_v1")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="validate NPZ episodes without writing a LeRobot dataset",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
