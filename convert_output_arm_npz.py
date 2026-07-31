#!/usr/bin/env python3
"""Convert old/new single-arm collector NPZ files to LeRobot v2.1.

Old episodes that only contain qpos are upgraded by deriving future absolute
joint actions: action[t] = qpos[min(t + offset, T - 1)].
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import numpy as np

from pi0_dataset import Pi0LeRobotDatasetWriter, derive_absolute_actions, single_arm_joint_names


def _expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in patterns:
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("ep_*.npz")))
            continue
        matches = [Path(path) for path in glob.glob(str(candidate))]
        paths.extend(matches or [candidate])
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _array(data, keys: tuple[str, ...], *, required: bool = True):
    for key in keys:
        if key in data.files:
            return np.asarray(data[key]), key
    if required:
        raise KeyError(f"none of these fields exist: {keys}")
    return None, None


def _text(data, key: str, default: str) -> str:
    if key not in data.files:
        return default
    return str(np.asarray(data[key]).item())


def _bool(data, key: str, default: bool) -> bool:
    if key not in data.files:
        return default
    return bool(np.asarray(data[key]).item())


def load_episode(path: Path, *, arm_side: str, fps: int, action_offset: int, use_existing_actions: bool) -> dict:
    wrist_key = f"cam_{arm_side}_wrist"
    with np.load(path, allow_pickle=False) as data:
        states, state_field = _array(data, ("qpos", "observation.state"))
        states = np.asarray(states, dtype=np.float32)

        existing_actions, action_field = _array(data, ("actions", "action"), required=False)
        if use_existing_actions and existing_actions is not None:
            actions = np.asarray(existing_actions, dtype=np.float32)
            action_source = action_field
        else:
            actions = derive_absolute_actions(states, action_offset)
            action_source = f"derived_from_{state_field}"

        timestamps, timestamp_field = _array(
            data,
            ("timestamps", "capture_timestamps", "timestamp"),
            required=False,
        )
        if timestamps is None:
            timestamps = np.arange(len(states), dtype=np.float64) / fps
            timestamp_field = "generated_from_fps"
        timestamps = np.asarray(timestamps, dtype=np.float64)

        high, high_field = _array(
            data,
            ("images_cam_high", "observation.images.cam_high"),
        )
        wrist, wrist_field = _array(
            data,
            (
                f"images_{wrist_key}",
                f"observation.images.{wrist_key}",
                "images_cam_wrist",
                "observation.images.cam_wrist",
                "images_cam_right_wrist",
                "observation.images.cam_right_wrist",
                "images_cam_left_wrist",
                "observation.images.cam_left_wrist",
            ),
        )
        return {
            "states": states,
            "actions": actions,
            "timestamps": timestamps,
            "images": {
                "cam_high": np.asarray(high),
                wrist_key: np.asarray(wrist),
            },
            "task_name": _text(data, "task_name", "single_arm_task"),
            "instruction": _text(data, "instruction", "single arm task"),
            "success": _bool(data, "success", True),
            "metadata": {
                "source_file": str(path.resolve()),
                "source_state_field": state_field,
                "source_action_field": action_source,
                "source_timestamp_field": timestamp_field,
                "source_high_camera_field": high_field,
                "source_wrist_camera_field": wrist_field,
                "arm_side": arm_side,
                "action_offset": action_offset,
            },
        }


def summarize_episode(path: Path, episode: dict, action_offset: int) -> None:
    states = episode["states"]
    actions = episode["actions"]
    timestamps = episode["timestamps"]
    dt = np.diff(timestamps)
    delta = actions[:, :6] - states[:, :6]
    fps = float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else float("nan")
    exact = np.array_equal(actions, derive_absolute_actions(states, action_offset))
    cameras = ", ".join(f"{key}:{value.shape}" for key, value in episode["images"].items())
    print(
        f"{path}: T={len(states)}, state={states.shape}, action={actions.shape}, "
        f"capture_fps≈{fps:.2f}, derived_action_exact={exact}, "
        f"joint_delta_abs_max={np.max(np.abs(delta)):.6f}, {cameras}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="NPZ files, directories, or glob patterns")
    ap.add_argument("--dataset-root", default="pi0_dataset_single")
    ap.add_argument("--arm-side", choices=("left", "right"), default="right")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--robot-type", default=None)
    ap.add_argument("--action-offset", type=int, default=1)
    ap.add_argument("--task-name", default=None, help="override task_name for every input")
    ap.add_argument("--instruction", default=None, help="override language instruction for every input")
    ap.add_argument("--mark-failure", action="store_true", help="override success=False")
    ap.add_argument(
        "--use-existing-actions",
        action="store_true",
        help="trust action/actions already in NPZ; default safely re-derives from qpos",
    )
    ap.add_argument("--check-only", action="store_true", help="validate and print without writing")
    args = ap.parse_args()

    if args.fps <= 0 or args.action_offset < 0:
        ap.error("--fps must be positive and --action-offset must be >= 0")
    paths = _expand_inputs(args.inputs)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input files: {missing}")
    if not paths:
        raise FileNotFoundError("no NPZ files matched")

    writer = None
    if not args.check_only:
        writer = Pi0LeRobotDatasetWriter(
            args.dataset_root,
            fps=args.fps,
            robot_type=args.robot_type or f"piper_single_arm_{args.arm_side}",
            state_names=single_arm_joint_names(args.arm_side),
            action_names=single_arm_joint_names(args.arm_side),
            camera_keys=["cam_high", f"cam_{args.arm_side}_wrist"],
            image_hw=(224, 224),
            save_raw_npz=False,
            action_semantics="absolute_next_joint_position",
            action_offset=args.action_offset,
        )

    converted = 0
    for path in paths:
        episode = load_episode(
            path,
            arm_side=args.arm_side,
            fps=args.fps,
            action_offset=args.action_offset,
            use_existing_actions=args.use_existing_actions,
        )
        if args.task_name:
            episode["task_name"] = args.task_name
        if args.instruction:
            episode["instruction"] = args.instruction
        if args.mark_failure:
            episode["success"] = False
        summarize_episode(path, episode, args.action_offset)
        if writer is not None:
            index = writer.append_episode(**episode)
            print(f"  -> LeRobot episode {index:06d}")
            converted += 1

    if writer is not None:
        print(f"Converted {converted} episode(s) -> {Path(args.dataset_root).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
