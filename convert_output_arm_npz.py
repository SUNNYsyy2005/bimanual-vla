#!/usr/bin/env python3
"""Convert Piper joint NPZ episodes to canonical LeRobot v2.1.

Supported contracts:

* single-arm joint: 7D state / 7D action, cam_high + one wrist camera
* bimanual joint: 14D state / 14D action, cam_high + left/right wrist cameras

Legacy episodes containing only measured qpos are upgraded with
``action[t] = qpos[min(t + 1, T - 1)]`` and are explicitly marked as
``next_measured_qpos`` / ``next_observation`` rather than commanded teleop data.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import numpy as np

from pi0_dataset import Pi0LeRobotDatasetWriter, derive_absolute_actions
from piper_data_contract import BIMANUAL, JOINT_SCHEMA, SINGLE_ARM, EpisodeContract


def _expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in patterns:
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("ep_*.npz")))
            continue
        matches = [Path(path) for path in glob.glob(str(candidate))]
        paths.extend(matches or [candidate])
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _array(data, keys: tuple[str, ...], *, required: bool = True):
    for key in keys:
        if key and key in data.files:
            return np.asarray(data[key]), key
    if required:
        raise KeyError(f"none of these fields exist: {keys}")
    return None, None


def _paired_array(
    data,
    combined_keys: tuple[str, ...],
    left_keys: tuple[str, ...],
    right_keys: tuple[str, ...],
    *,
    required: bool = True,
):
    combined, field = _array(data, combined_keys, required=False)
    if combined is not None:
        return combined, field
    left, left_field = _array(data, left_keys, required=False)
    right, right_field = _array(data, right_keys, required=False)
    if left is not None and right is not None:
        if len(left) != len(right):
            raise ValueError(f"left/right frame mismatch: {len(left)} != {len(right)}")
        return np.concatenate([left, right], axis=-1), f"{left_field}+{right_field}"
    if required:
        raise KeyError(
            f"none of combined fields {combined_keys} or complete left/right fields exist"
        )
    return None, None


def _text(data, key: str, default: str) -> str:
    if key not in data.files:
        return default
    return str(np.asarray(data[key]).item())


def _bool(data, key: str, default: bool) -> bool:
    if key not in data.files:
        return default
    return bool(np.asarray(data[key]).item())


def _camera_array(data, camera_key: str, *, arm_side: str):
    aliases = {
        "cam_high": ("images_cam_high", "observation.images.cam_high", "image"),
        "cam_left_wrist": (
            "images_cam_left_wrist",
            "observation.images.cam_left_wrist",
            "left_wrist_image",
        ),
        "cam_right_wrist": (
            "images_cam_right_wrist",
            "observation.images.cam_right_wrist",
            "right_wrist_image",
        ),
    }
    candidates = list(aliases[camera_key])
    if camera_key == f"cam_{arm_side}_wrist":
        candidates.extend(("images_cam_wrist", "observation.images.cam_wrist", "wrist_image"))
    return _array(data, tuple(candidates))


def load_episode(
    path: Path,
    *,
    contract: EpisodeContract,
    fps: int,
    action_offset: int,
    use_existing_actions: bool,
) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if contract.arm_mode == BIMANUAL:
            states, state_field = _paired_array(
                data,
                ("qpos", "joint_qpos", "observation.state", "state"),
                ("left_qpos", "qpos_left", "left_joint_qpos"),
                ("right_qpos", "qpos_right", "right_joint_qpos"),
            )
            existing_actions, action_field = _paired_array(
                data,
                ("actions", "action"),
                ("left_actions", "actions_left", "left_action"),
                ("right_actions", "actions_right", "right_action"),
                required=False,
            )
        else:
            states, state_field = _array(
                data, ("qpos", "joint_qpos", "observation.state", "state")
            )
            existing_actions, action_field = _array(
                data, ("actions", "action"), required=False
            )

        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != contract.state_dim:
            raise ValueError(
                f"{path}: joint state shape {states.shape} != (*, {contract.state_dim})"
            )
        if use_existing_actions:
            if existing_actions is None:
                raise ValueError(f"{path}: --use-existing-actions requested but action/actions is missing")
            actions = np.asarray(existing_actions, dtype=np.float32)
            action_source = action_field
        else:
            actions = derive_absolute_actions(states, action_offset)
            action_source = f"derived_from_{state_field}"
        if actions.ndim != 2 or actions.shape != (len(states), contract.action_dim):
            raise ValueError(
                f"{path}: joint action shape {actions.shape} != ({len(states)}, {contract.action_dim})"
            )

        timestamps, timestamp_field = _array(
            data, ("timestamps", "capture_timestamps", "timestamp"), required=False
        )
        if timestamps is None:
            timestamps = np.arange(len(states), dtype=np.float64) / fps
            timestamp_field = "generated_from_fps"
        timestamps = np.asarray(timestamps, dtype=np.float64)

        images: dict[str, np.ndarray] = {}
        image_fields: dict[str, str] = {}
        for camera_key in contract.camera_keys:
            image, field = _camera_array(data, camera_key, arm_side=contract.arm_side)
            images[camera_key] = np.asarray(image)
            image_fields[camera_key] = str(field)

        return {
            "states": states,
            "actions": actions,
            "timestamps": timestamps,
            "images": images,
            "task_name": _text(data, "task_name", _text(data, "task", "piper_joint_task")),
            "instruction": _text(data, "instruction", "Piper joint teleoperation task"),
            "success": _bool(data, "success", True),
            "metadata": {
                "source_file": str(path.resolve()),
                "source_state_field": state_field,
                "source_action_field": action_source,
                "source_timestamp_field": timestamp_field,
                "source_camera_fields": image_fields,
                "arm_mode": contract.arm_mode,
                "arm_side": contract.arm_side,
                "action_offset": action_offset,
            },
        }


def summarize_episode(path: Path, episode: dict, action_offset: int) -> None:
    states = episode["states"]
    actions = episode["actions"]
    timestamps = episode["timestamps"]
    dt = np.diff(timestamps)
    joint_indices = [index for index in range(states.shape[1]) if index % 7 != 6]
    delta = actions[:, joint_indices] - states[:, joint_indices]
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
    ap.add_argument("--dataset-root", default="pi0_dataset_joint")
    ap.add_argument("--arm-mode", choices=(SINGLE_ARM, BIMANUAL), default=SINGLE_ARM)
    ap.add_argument("--arm-side", choices=("left", "right", "both"), default="right")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--robot-type", default=None)
    ap.add_argument("--action-offset", type=int, default=1)
    ap.add_argument("--task-name", default=None, help="override task_name for every input")
    ap.add_argument("--instruction", default=None, help="override language instruction for every input")
    ap.add_argument("--mark-failure", action="store_true", help="override success=False")
    ap.add_argument(
        "--use-existing-actions",
        action="store_true",
        help="use recorded action/actions as same-step commands instead of deriving next measured qpos",
    )
    ap.add_argument(
        "--existing-action-source",
        default="master_joint_feedback",
        help="action_source metadata used with --use-existing-actions",
    )
    ap.add_argument("--check-only", action="store_true", help="validate and print without writing")
    args = ap.parse_args()

    if args.fps <= 0:
        ap.error("--fps must be positive")
    if args.arm_mode == BIMANUAL:
        args.arm_side = "both"
    elif args.arm_side == "both":
        ap.error("single arm mode requires --arm-side left or right")
    if args.use_existing_actions:
        effective_offset = 0
        action_source = args.existing_action_source.strip()
        action_alignment = "same_step_command"
        if not action_source:
            ap.error("--existing-action-source must not be empty")
    else:
        if args.action_offset != 1:
            ap.error("derived next-observation actions require --action-offset 1")
        effective_offset = 1
        action_source = "next_measured_qpos"
        action_alignment = "next_observation"

    contract = EpisodeContract(
        schema=JOINT_SCHEMA,
        arm_mode=args.arm_mode,
        arm_side=args.arm_side,
        action_source=action_source,
        action_alignment=action_alignment,
    )
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
            robot_type=args.robot_type or contract.robot_type,
            state_names=list(contract.state_names),
            action_names=list(contract.action_names),
            camera_keys=list(contract.camera_keys),
            image_hw=(224, 224),
            save_raw_npz=False,
            schema=contract.schema,
            arm_mode=contract.arm_mode,
            arm_side=contract.arm_side,
            action_semantics=contract.action_semantics,
            action_source=contract.action_source,
            action_alignment=contract.action_alignment,
            action_offset=contract.action_offset,
        )

    converted = 0
    for path in paths:
        episode = load_episode(
            path,
            contract=contract,
            fps=args.fps,
            action_offset=effective_offset,
            use_existing_actions=args.use_existing_actions,
        )
        if args.task_name:
            episode["task_name"] = args.task_name
        if args.instruction:
            episode["instruction"] = args.instruction
        if args.mark_failure:
            episode["success"] = False
        summarize_episode(path, episode, effective_offset)
        if writer is not None:
            index = writer.append_episode(**episode)
            print(f"  -> LeRobot episode {index:06d}")
            converted += 1

    if writer is not None:
        print(f"Converted {converted} episode(s) -> {Path(args.dataset_root).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
