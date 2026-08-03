"""Export collected NPZ episodes to LeRobot v2.1.

The exporter keeps only successful episodes by default.  Single-arm delivery
episodes use the established server-compatible ``state`` / ``actions`` plus
``image`` / ``wrist_image`` layout.  Other contracts use canonical LeRobot
``observation.state`` / ``action`` fields and video camera features.

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

from piper_data_contract import DELIVERY_SCHEMA, LEROBOT_FEATURES, SINGLE_ARM
from validate_piper_data import (
    EpisodeStats,
    EpisodeValidationError,
    format_dataset_report,
    format_episode_report,
    validate_episode,
    validate_gripper_coverage,
    validate_instruction_consistency,
)


FEATURES = LEROBOT_FEATURES


def _validate_inputs(
    paths: list[Path],
    target_fps: float,
    allow_incomplete_gripper_coverage: bool = False,
) -> list[EpisodeStats]:
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

    dataset_errors = []
    try:
        validate_gripper_coverage(coverage_candidates)
    except ValueError as exc:
        if allow_incomplete_gripper_coverage:
            print(f"WARNING: ignoring gripper coverage check: {exc}")
        else:
            dataset_errors.append(str(exc))
    try:
        validate_instruction_consistency(coverage_candidates)
    except ValueError as exc:
        dataset_errors.append(str(exc))
    if failures:
        details = "Input validation failed:\n\n" + "\n\n".join(failures)
        if dataset_errors:
            details += f"\n\nDataset validation failed: {'; '.join(dataset_errors)}"
        raise SystemExit(details)
    if dataset_errors:
        raise SystemExit(f"Input validation failed: {'; '.join(dataset_errors)}")
    print(format_dataset_report(successful))
    return successful


def _load_episode(path: Path, contract):
    with np.load(path, allow_pickle=False) as data:
        state_key = "state" if "state" in data.files else "observation.state"
        action_key = "actions" if "actions" in data.files else "action"
        states = np.asarray(data[state_key], dtype=np.float32)
        actions = np.asarray(data[action_key], dtype=np.float32)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        images = {
            key: np.asarray(data[contract.image_field(key)], dtype=np.uint8)
            for key in contract.camera_keys
        }
        task_name = str(np.asarray(data["task"]).item()) if "task" in data.files else path.stem
        instruction = str(np.asarray(data["instruction"]).item())
        success = bool(np.asarray(data["success"]).item())
    return {
        "states": states,
        "actions": actions,
        "timestamps": timestamps,
        "images": images,
        "task_name": task_name,
        "instruction": instruction,
        "success": success,
        "metadata": {"source_npz": path.name},
    }


def _export_legacy_single_delivery(
    successful: list[EpisodeStats],
    output_root: Path,
    *,
    fps: int,
) -> tuple[int, int]:
    """Write the delivery layout accepted by the deployed dataset server."""
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
            "but this delivery requires LeRobot v2.1."
        )

    create_kwargs = {
        "repo_id": f"piper/{output_root.name}",
        "robot_type": "piper",
        "fps": fps,
        "features": FEATURES,
        "use_videos": False,
    }
    if "root" in inspect.signature(LeRobotDataset.create).parameters:
        create_kwargs["root"] = output_root
    dataset = LeRobotDataset.create(**create_kwargs)

    count = 0
    frames = 0
    for stats in successful:
        with np.load(stats.path, allow_pickle=False) as data:
            states = np.asarray(data["state"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            images = np.asarray(data["image"], dtype=np.uint8)
            wrist_images = np.asarray(data["wrist_image"], dtype=np.uint8)
        for index in range(len(states)):
            dataset.add_frame(
                {
                    "image": images[index],
                    "wrist_image": wrist_images[index],
                    "state": states[index],
                    "actions": actions[index],
                },
                task=stats.instruction,
                timestamp=index / fps,
            )
        dataset.save_episode()
        count += 1
        frames += len(states)
        print(
            f"Exported {stats.path} -> episode {count - 1:06d} "
            f"({len(states)} frames, instruction={stats.instruction!r})"
        )
    return count, frames


def export_dataset(
    input_dir: str | Path,
    root: str | Path,
    *,
    fps: int = 20,
    allow_incomplete_gripper_coverage: bool = False,
    validate_only: bool = False,
) -> Path | None:
    """Validate GUI NPZ episodes and export successful ones to LeRobot v2.1."""
    input_root = Path(input_dir).expanduser()
    paths = sorted(input_root.glob("ep_*.npz"))
    if not paths:
        raise SystemExit(f"No episodes found in {input_root}")

    successful = _validate_inputs(
        paths,
        target_fps=fps,
        allow_incomplete_gripper_coverage=allow_incomplete_gripper_coverage,
    )
    if not successful:
        raise SystemExit("No successful episodes are available for export")
    if validate_only:
        print("Validation complete; no LeRobot dataset was written.")
        return None

    from pi0_dataset import Pi0LeRobotDatasetWriter
    from piper_data_contract import infer_episode_contract

    first_path = successful[0].path
    with np.load(first_path, allow_pickle=False) as data:
        contract = infer_episode_contract(data)
    output_root = Path(root).expanduser()
    if contract.schema == DELIVERY_SCHEMA and contract.arm_mode == SINGLE_ARM:
        count, frames = _export_legacy_single_delivery(
            successful,
            output_root,
            fps=fps,
        )
        print(
            f"Export complete: root={output_root} schema={contract.schema} "
            f"arm={contract.arm_mode}/{contract.arm_side} layout=legacy "
            f"episodes={count} frames={frames} fps={fps}"
        )
        return output_root

    writer = Pi0LeRobotDatasetWriter(
        output_root,
        fps=fps,
        robot_type=contract.robot_type,
        state_names=list(contract.state_names),
        action_names=list(contract.action_names),
        camera_keys=list(contract.camera_keys),
        image_hw=(224, 224),
        schema=contract.schema,
        arm_mode=contract.arm_mode,
        arm_side=contract.arm_side,
        action_source=contract.action_source,
        action_alignment=contract.action_alignment,
        action_offset=contract.action_offset,
    )

    count = 0
    frames = 0
    for stats in successful:
        with np.load(stats.path, allow_pickle=False) as data:
            episode_contract = infer_episode_contract(data)
        if episode_contract != contract:
            raise SystemExit(
                f"episode contract mismatch: {stats.path}: {episode_contract} != {contract}"
            )
        episode = _load_episode(stats.path, contract)
        index = writer.append_episode(**episode)
        count += 1
        frames += len(episode["states"])
        print(
            f"Exported {stats.path} -> episode {index:06d} "
            f"({len(episode['states'])} frames, instruction={stats.instruction!r})"
        )

    print(
        f"Export complete: root={output_root} schema={contract.schema} "
        f"arm={contract.arm_mode}/{contract.arm_side} episodes={count} "
        f"frames={frames} fps={fps}"
    )
    return output_root


def run(args):
    return export_dataset(
        args.input_dir,
        args.root or args.repo_id,
        fps=args.fps,
        allow_incomplete_gripper_coverage=args.allow_incomplete_gripper_coverage,
        validate_only=args.validate_only,
    )

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
    ap.add_argument(
        "--allow-incomplete-gripper-coverage",
        action="store_true",
        help="export even when successful episodes do not cover fully open and closed gripper states",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
