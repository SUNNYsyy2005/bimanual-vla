#!/usr/bin/env python3
"""Convert Piper NPZ episodes to canonical or explicitly legacy LeRobot v2.1.

Supported contracts:

* single/bimanual joint: 7D/7D or 14D/14D absolute joint targets
* canonical delivery: 10D/10D or 20D/20D absolute EEF targets
* legacy_v2 delivery: 10D/7D or 20D/14D measured step deltas

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
from export_lerobot import inspect_npz_episode


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
        state_timestamps, _ = _array(
            data, ("state_timestamp", "state_timestamps"), required=False
        )
        action_timestamps, _ = _array(
            data, ("action_timestamp", "action_timestamps"), required=False
        )
        state_timestamps = np.asarray(
            timestamps if state_timestamps is None else state_timestamps, dtype=np.float64
        )
        action_timestamps = np.asarray(
            state_timestamps if action_timestamps is None else action_timestamps, dtype=np.float64
        )

        images: dict[str, np.ndarray] = {}
        image_fields: dict[str, str] = {}
        image_timestamps: dict[str, np.ndarray] = {}
        for camera_key in contract.camera_keys:
            image, field = _camera_array(data, camera_key, arm_side=contract.arm_side)
            images[camera_key] = np.asarray(image)
            image_fields[camera_key] = str(field)
            values, _ = _array(
                data,
                (
                    f"image_timestamps_{camera_key}",
                    f"image_timestamp.{camera_key}",
                    f"image_timestamps.{camera_key}",
                ),
                required=False,
            )
            image_timestamps[camera_key] = np.asarray(
                state_timestamps if values is None else values, dtype=np.float64
            )

        return {
            "states": states,
            "actions": actions,
            "timestamps": timestamps,
            "state_timestamps": state_timestamps,
            "action_timestamps": action_timestamps,
            "image_timestamps": image_timestamps,
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


def summarize_episode(path: Path, episode: dict, action_offset: int = 0) -> None:
    states = np.asarray(episode["states"])
    actions = np.asarray(episode["actions"])
    timestamps = np.asarray(episode["timestamps"])
    dt = np.diff(timestamps)
    measured_fps = float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else float("nan")
    metadata = episode.get("metadata", {})
    schema = metadata.get("schema", "joint" if states.shape[1] in {7, 14} else "delivery")
    format_name = metadata.get("contract_format", "canonical")
    details = ""
    if schema == "joint" and states.shape == actions.shape:
        exact = np.array_equal(actions, derive_absolute_actions(states, action_offset))
        details = f", derived_action_exact={exact}"
    cameras = ", ".join(f"{key}:{value.shape}" for key, value in episode["images"].items())
    print(
        f"{path}: T={len(states)}, state={states.shape}, action={actions.shape}, "
        f"schema={schema}, format={format_name}, capture_fps≈{measured_fps:.2f}{details}, {cameras}"
    )


def _writer_kwargs_from_inspected(spec: dict) -> dict:
    return {
        "robot_type": spec["robot_type"],
        "state_names": list(spec["state_names"]),
        "action_names": list(spec["action_names"]),
        "camera_keys": list(spec["camera_keys"]),
        "schema": spec["schema"],
        "arm_mode": spec["arm_mode"],
        "arm_side": spec["arm_side"],
        "action_semantics": spec["action_semantics"],
        "action_source": spec["action_source"],
        "action_alignment": spec["action_alignment"],
        "action_offset": spec["action_offset"],
        "action_horizon": spec["action_horizon"],
        "raw_action_dim": spec["raw_action_dim"],
        "model_action_dim": spec["model_action_dim"],
        "gripper_semantics": spec["gripper_semantics"],
        "rotation_semantics": spec["rotation_semantics"],
        "coordinate_frame": spec["coordinate_frame"],
        "legacy_format": spec["legacy_format"],
    }


def _episode_from_inspected(spec: dict) -> dict:
    metadata = dict(spec.get("metadata", {}))
    metadata.update(
        {
            "schema": spec["schema"],
            "contract_format": spec["contract_format"],
            "legacy_format": spec["legacy_format"],
            "raw_action_dim": spec["raw_action_dim"],
            "model_action_dim": spec["model_action_dim"],
        }
    )
    return {
        key: spec[key]
        for key in (
            "states", "actions", "timestamps", "state_timestamps", "action_timestamps",
            "image_timestamps", "images", "task_name", "instruction", "success",
        )
    } | {"metadata": metadata}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="NPZ files, directories, or glob patterns")
    ap.add_argument("--dataset-root", default="pi0_dataset")
    ap.add_argument("--schema", choices=("auto", "joint", "delivery"), default="auto")
    ap.add_argument("--arm-mode", choices=("auto", SINGLE_ARM, BIMANUAL), default="auto")
    ap.add_argument("--arm-side", choices=("left", "right", "both"), default="right")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--robot-type", default=None)
    ap.add_argument("--action-offset", type=int, default=1)
    ap.add_argument("--task-name", default=None, help="override task_name for every input")
    ap.add_argument("--instruction", default=None, help="override language instruction for every input")
    ap.add_argument("--mark-failure", action="store_true", help="override success=False")
    ap.add_argument(
        "--use-existing-actions",
        action="store_true",
        help="for joint data, use recorded actions instead of deriving next measured qpos",
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
    paths = _expand_inputs(args.inputs)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input files: {missing}")
    if not paths:
        raise FileNotFoundError("no NPZ files matched")

    inspected: list[dict] = []
    for path in paths:
        try:
            spec = inspect_npz_episode(path, fps=args.fps)
        except ValueError:
            # Measured-qpos-only files remain supported through the original
            # joint conversion path; delivery always requires recorded action.
            with np.load(path, allow_pickle=False) as raw:
                if "action" in raw.files or "actions" in raw.files:
                    raise
            if args.schema == "delivery":
                raise
            arm_mode = BIMANUAL if args.arm_mode == BIMANUAL else SINGLE_ARM
            arm_side = "both" if arm_mode == BIMANUAL else args.arm_side
            if arm_side == "both" and arm_mode != BIMANUAL:
                ap.error("single arm mode requires --arm-side left or right")
            if args.use_existing_actions:
                action_source = args.existing_action_source.strip()
                action_alignment, effective_offset = "same_step_command", 0
            else:
                action_source = "next_measured_qpos"
                action_alignment, effective_offset = "next_observation", args.action_offset
                if effective_offset != 1:
                    ap.error("derived next-observation actions require --action-offset 1")
            contract = EpisodeContract(
                schema=JOINT_SCHEMA,
                arm_mode=arm_mode,
                arm_side=arm_side,
                action_source=action_source,
                action_alignment=action_alignment,
            )
            episode = load_episode(
                path,
                contract=contract,
                fps=args.fps,
                action_offset=effective_offset,
                use_existing_actions=args.use_existing_actions,
            )
            spec = {
                "path": path,
                "schema": "joint",
                "arm_mode": contract.arm_mode,
                "arm_side": contract.arm_side,
                "state_dim": contract.state_dim,
                "raw_action_dim": contract.action_dim,
                "model_action_dim": contract.action_dim,
                "legacy_format": None,
                "contract_format": "canonical",
                "robot_type": args.robot_type or contract.robot_type,
                "state_names": list(contract.state_names),
                "action_names": list(contract.action_names),
                "camera_keys": list(contract.camera_keys),
                "action_semantics": contract.action_semantics,
                "action_source": contract.action_source,
                "action_alignment": contract.action_alignment,
                "action_offset": contract.action_offset,
                "action_horizon": 50,
                "gripper_semantics": "absolute_opening_fraction_0_closed_1_open",
                "rotation_semantics": "not_applicable",
                "coordinate_frame": "slave_base",
                **episode,
            }
        if args.schema != "auto" and spec["schema"] != args.schema:
            raise ValueError(f"{path}: detected schema={spec['schema']}, requested {args.schema}")
        if args.arm_mode != "auto" and spec["arm_mode"] != args.arm_mode:
            raise ValueError(f"{path}: detected arm_mode={spec['arm_mode']}, requested {args.arm_mode}")
        if args.robot_type:
            spec["robot_type"] = args.robot_type
        inspected.append(spec)

    reference = _writer_kwargs_from_inspected(inspected[0])
    for item in inspected[1:]:
        if _writer_kwargs_from_inspected(item) != reference:
            raise ValueError(f"{item['path']}: data contract differs from the first episode")

    writer = None
    if not args.check_only:
        writer = Pi0LeRobotDatasetWriter(
            args.dataset_root,
            fps=args.fps,
            image_hw=(224, 224),
            save_raw_npz=False,
            **reference,
        )

    converted = 0
    for path, spec in zip(paths, inspected, strict=True):
        episode = _episode_from_inspected(spec)
        if args.task_name:
            episode["task_name"] = args.task_name
        if args.instruction:
            episode["instruction"] = args.instruction
        if args.mark_failure:
            episode["success"] = False
        summarize_episode(path, episode, spec["action_offset"])
        if writer is not None:
            index = writer.append_episode(**episode)
            print(f"  -> LeRobot episode {index:06d}")
            converted += 1

    if writer is not None:
        print(f"Converted {converted} episode(s) -> {Path(args.dataset_root).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
