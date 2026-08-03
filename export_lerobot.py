"""Export collected Piper NPZ episodes to LeRobot v2.1.

Supported homogeneous dataset contracts:

* joint: single-arm 7D/7D or bimanual 14D/14D absolute targets;
* canonical delivery: 10D/10D or 20D/20D absolute EEF targets;
* ``legacy_v2`` delivery: 10D/7D or 20D/14D measured step deltas.

The legacy layout is retained without rewriting action values and is explicitly
marked ``legacy_v2``. It must never be inferred as canonical absolute EEF data.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pi0_dataset import (
    DEFAULT_ACTION_HORIZON,
    DELIVERY_ABSOLUTE_ACTION_SEMANTICS,
    DELIVERY_LEGACY_ACTION_SEMANTICS,
    GRIPPER_CLOSED_FRACTION_LEGACY,
    GRIPPER_OPENING_FRACTION,
    LEGACY_ROTATION_SEMANTICS,
    LEGACY_V2,
    ROTATION6D_SEMANTICS,
    Pi0LeRobotDatasetWriter,
    classify_contract_dimensions,
    default_eef_names,
    single_arm_joint_names,
    BIMANUAL_JOINT_NAMES,
)
from piper_data_contract import LEROBOT_FEATURES


FEATURES = LEROBOT_FEATURES


def _scalar(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.shape != ():
        return default
    return value.item()


def _text(data: Mapping[str, Any], key: str, default: str = "") -> str:
    value = _scalar(data, key, default)
    return str(value).strip()


def _array(data: Mapping[str, Any], keys: tuple[str, ...], *, required: bool = True):
    for key in keys:
        if key and key in data:
            return np.asarray(data[key]), key
    if required:
        raise ValueError(f"none of these fields exist: {keys}")
    return None, None


def _camera_field(data: Mapping[str, Any], camera_key: str, arm_side: str) -> tuple[np.ndarray, str]:
    candidates = {
        "cam_high": ("images_cam_high", "observation.images.cam_high", "image"),
        "cam_wrist": ("images_cam_wrist", "observation.images.cam_wrist", "wrist_image"),
        "cam_left_wrist": (
            "images_cam_left_wrist", "observation.images.cam_left_wrist", "left_wrist_image",
        ),
        "cam_right_wrist": (
            "images_cam_right_wrist", "observation.images.cam_right_wrist", "right_wrist_image",
        ),
    }
    values = list(candidates.get(camera_key, (f"images_{camera_key}", f"observation.images.{camera_key}")))
    if camera_key == f"cam_{arm_side}_wrist":
        values.extend(("images_cam_wrist", "observation.images.cam_wrist", "wrist_image"))
    array, field = _array(data, tuple(values))
    return np.asarray(array), str(field)


def _camera_keys(data: Mapping[str, Any], *, arm_mode: str, arm_side: str) -> list[str]:
    if "camera_keys" in data:
        values = np.asarray(data["camera_keys"])
        if values.ndim == 1:
            keys = [str(item).strip() for item in values.tolist() if str(item).strip()]
            if keys:
                return keys
    if arm_mode == "bimanual":
        return ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    if any(key in data for key in ("wrist_image", "images_cam_wrist", "observation.images.cam_wrist")):
        return ["cam_high", "cam_wrist"]
    return ["cam_high", f"cam_{arm_side}_wrist"]


def _names(data: Mapping[str, Any], key: str, fallback: list[str], expected: int) -> list[str]:
    if key in data:
        values = np.asarray(data[key])
        if values.ndim == 1:
            result = [str(item) for item in values.tolist()]
            if len(result) == expected:
                return result
    return list(fallback)


def _strict_timestamps(values: np.ndarray, *, name: str, frames: int, legacy: bool) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (frames,):
        raise ValueError(f"{name} shape {result.shape} != ({frames},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if len(result) > 1:
        delta = np.diff(result)
        valid = bool(np.all(delta > 0))
        # Old EpisodeBuffer files may contain one terminal hold row with the
        # final device timestamp repeated. Preserve those files as legacy_v2.
        if legacy and len(delta) and delta[-1] == 0 and np.all(delta[:-1] > 0):
            valid = True
        if not valid:
            raise ValueError(f"{name} must be strictly increasing")
    return result


def _legacy_matches_next_measured(states: np.ndarray, actions: np.ndarray, arm_count: int) -> bool:
    try:
        import piper_data_contract as contract_module

        builder = getattr(
            contract_module,
            "build_legacy_delivery_step_actions",
            contract_module.build_delivery_actions,
        )
        expected = builder(states, arm_count=arm_count)
    except Exception:
        return False
    return expected.shape == actions.shape and bool(np.allclose(actions, expected, atol=1e-5, rtol=1e-5))


def inspect_npz_episode(path: str | Path, *, fps: int = 20) -> dict[str, Any]:
    """Load and validate one raw NPZ using shape/metadata duck typing."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        state, state_field = _array(data, ("observation.state", "state", "qpos", "joint_qpos"))
        action, action_field = _array(data, ("action", "actions"))
        states = np.asarray(state, dtype=np.float32)
        actions = np.asarray(action, dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2 or len(states) == 0:
            raise ValueError(f"{path}: states/actions must be non-empty rank-2 arrays")
        if len(states) != len(actions):
            raise ValueError(f"{path}: state frames {len(states)} != action frames {len(actions)}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"{path}: state/action contains NaN/Inf")

        dimensions = classify_contract_dimensions(
            states.shape[1],
            actions.shape[1],
            schema=_text(data, "schema"),
            legacy_format=_text(data, "legacy_format") or _text(data, "contract_format"),
        )
        arm_mode = _text(data, "arm_mode", dimensions["arm_mode"]) or dimensions["arm_mode"]
        if arm_mode != dimensions["arm_mode"]:
            raise ValueError(f"{path}: arm_mode={arm_mode!r} disagrees with dimensions")
        arm_side = _text(data, "arm_side", "both" if arm_mode == "bimanual" else "right")
        if arm_mode == "bimanual":
            arm_side = "both"
        elif arm_side not in {"left", "right"}:
            raise ValueError(f"{path}: invalid arm_side={arm_side!r}")

        if dimensions["schema"] == "delivery":
            fallback_state_names, fallback_action_names = default_eef_names(
                arm_mode=arm_mode, arm_side=arm_side, legacy=dimensions["legacy"]
            )
        elif arm_mode == "bimanual":
            fallback_state_names = list(BIMANUAL_JOINT_NAMES)
            fallback_action_names = list(BIMANUAL_JOINT_NAMES)
        else:
            fallback_state_names = single_arm_joint_names(arm_side)
            fallback_action_names = single_arm_joint_names(arm_side)
        state_names = _names(data, "state_names", fallback_state_names, states.shape[1])
        action_names = _names(data, "action_names", fallback_action_names, actions.shape[1])

        legacy_next_measured = dimensions["legacy"] and _legacy_matches_next_measured(
            states, actions, dimensions["arm_count"]
        )
        action_source = _text(data, "action_source")
        action_alignment = _text(data, "action_alignment")
        action_offset = _scalar(data, "action_offset", None)
        if dimensions["legacy"]:
            action_source = action_source or (
                "next_measured_eef" if legacy_next_measured else "legacy_recorded_eef_delta"
            )
            action_alignment = action_alignment or "next_observation"
            action_offset = 1 if action_offset is None else int(action_offset)
        else:
            if not action_source and dimensions["schema"] == "joint":
                action_source = "recorded_joint_target"
            if not action_alignment and dimensions["schema"] == "joint":
                action_alignment = "same_step_command"
            if not action_source:
                raise ValueError(f"{path}: canonical data requires explicit action_source metadata")
            if not action_alignment:
                raise ValueError(f"{path}: canonical data requires explicit action_alignment metadata")
            action_offset = int(0 if action_offset is None else action_offset)

        camera_keys = _camera_keys(data, arm_mode=arm_mode, arm_side=arm_side)
        expected_cameras = 3 if arm_mode == "bimanual" else 2
        if len(camera_keys) != expected_cameras:
            raise ValueError(f"{path}: camera_keys={camera_keys}, expected {expected_cameras} cameras")
        images: dict[str, np.ndarray] = {}
        image_fields: dict[str, str] = {}
        for camera_key in camera_keys:
            image, field = _camera_field(data, camera_key, arm_side)
            if image.ndim != 4 or image.dtype != np.uint8 or len(image) != len(states):
                raise ValueError(
                    f"{path}: {field} image shape/dtype {image.shape}/{image.dtype} is invalid"
                )
            images[camera_key] = image
            image_fields[camera_key] = field

        timestamp, timestamp_field = _array(
            data, ("timestamps", "capture_timestamps", "timestamp", "state_timestamp"), required=False
        )
        if timestamp is None:
            timestamp = np.arange(len(states), dtype=np.float64) / float(fps)
            timestamp_field = "generated_from_fps"
        timestamps = _strict_timestamps(
            timestamp, name=str(timestamp_field), frames=len(states), legacy=dimensions["legacy"]
        )
        state_timestamp, _ = _array(data, ("state_timestamp", "state_timestamps"), required=False)
        action_timestamp, _ = _array(data, ("action_timestamp", "action_timestamps"), required=False)
        state_timestamps = _strict_timestamps(
            timestamps if state_timestamp is None else state_timestamp,
            name="state_timestamp",
            frames=len(states),
            legacy=dimensions["legacy"],
        )
        action_timestamps = _strict_timestamps(
            state_timestamps if action_timestamp is None else action_timestamp,
            name="action_timestamp",
            frames=len(states),
            legacy=dimensions["legacy"],
        )
        image_timestamps: dict[str, np.ndarray] = {}
        for camera_key in camera_keys:
            value, _ = _array(
                data,
                (
                    f"image_timestamps_{camera_key}",
                    f"image_timestamp.{camera_key}",
                    f"image_timestamps.{camera_key}",
                ),
                required=False,
            )
            image_timestamps[camera_key] = _strict_timestamps(
                state_timestamps if value is None else value,
                name=f"image_timestamps_{camera_key}",
                frames=len(states),
                legacy=dimensions["legacy"],
            )

        gripper_indices = (
            [index * 10 + 9 for index in range(dimensions["arm_count"])]
            if dimensions["schema"] == "delivery" and not dimensions["legacy"]
            else [index * 7 + 6 for index in range(dimensions["arm_count"])]
        )
        for index in gripper_indices:
            if index < actions.shape[1] and (np.min(actions[:, index]) < -1e-4 or np.max(actions[:, index]) > 1.0001):
                raise ValueError(f"{path}: action gripper dimension {index} is outside [0,1]")

        action_semantics = _text(data, "action_semantics") or (
            DELIVERY_LEGACY_ACTION_SEMANTICS
            if dimensions["legacy"]
            else DELIVERY_ABSOLUTE_ACTION_SEMANTICS
            if dimensions["schema"] == "delivery"
            else "absolute_joint_position"
        )
        if dimensions["schema"] == "delivery":
            if dimensions["legacy"] and "delta" not in action_semantics.lower():
                raise ValueError(f"{path}: legacy_v2 action semantics must describe deltas")
            if not dimensions["legacy"] and "delta" in action_semantics.lower():
                raise ValueError(f"{path}: canonical 10D delivery action cannot be marked delta")

        metadata = {
            "source_npz": path.name,
            "source_state_field": state_field,
            "source_action_field": action_field,
            "source_timestamp_field": timestamp_field,
            "source_camera_fields": image_fields,
            "legacy_next_measured_verified": bool(legacy_next_measured),
        }
        return {
            **dimensions,
            "path": path,
            "states": states,
            "actions": actions,
            "timestamps": timestamps,
            "state_timestamps": state_timestamps,
            "action_timestamps": action_timestamps,
            "image_timestamps": image_timestamps,
            "images": images,
            "state_names": state_names,
            "action_names": action_names,
            "camera_keys": camera_keys,
            "arm_side": arm_side,
            "robot_type": _text(
                data,
                "robot_type",
                "piper_bimanual" if arm_mode == "bimanual" else f"piper_single_arm_{arm_side}",
            ),
            "action_semantics": action_semantics,
            "action_source": action_source,
            "action_alignment": action_alignment,
            "action_offset": int(action_offset),
            "action_horizon": int(_scalar(data, "action_horizon", DEFAULT_ACTION_HORIZON)),
            "gripper_semantics": _text(data, "gripper_semantics") or (
                GRIPPER_CLOSED_FRACTION_LEGACY if dimensions["legacy"] else GRIPPER_OPENING_FRACTION
            ),
            "rotation_semantics": _text(data, "rotation_semantics") or (
                LEGACY_ROTATION_SEMANTICS if dimensions["legacy"] else ROTATION6D_SEMANTICS
                if dimensions["schema"] == "delivery" else "not_applicable"
            ),
            "coordinate_frame": _text(data, "coordinate_frame", "slave_base"),
            "task_name": _text(data, "task_name", _text(data, "task", path.stem)),
            "instruction": _text(data, "instruction", path.stem.replace("_", " ")),
            "success": bool(_scalar(data, "success", True)),
            "metadata": metadata,
        }


def _homogeneous_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        spec[key]
        for key in (
            "schema", "arm_mode", "arm_side", "state_dim", "raw_action_dim",
            "model_action_dim", "state_names", "action_names", "camera_keys",
            "action_semantics", "action_source", "action_alignment", "action_offset",
            "gripper_semantics", "rotation_semantics", "coordinate_frame", "legacy_format",
        )
    )


def _validate_inputs(
    paths: list[Path],
    target_fps: float,
    allow_incomplete_gripper_coverage: bool = False,
) -> list[dict[str, Any]]:
    del allow_incomplete_gripper_coverage  # Shape/range checks are per episode; coverage is not fabricated.
    episodes: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in paths:
        try:
            episode = inspect_npz_episode(path, fps=int(target_fps))
            episodes.append(episode)
            print(
                f"OK {path}: schema={episode['schema']} arm={episode['arm_mode']} "
                f"state/action={episode['state_dim']}/{episode['raw_action_dim']} "
                f"format={episode['contract_format']} frames={len(episode['states'])}"
            )
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("Input validation failed:\n  - " + "\n  - ".join(failures))
    if not episodes:
        raise SystemExit("No valid successful episodes are available for export")
    reference = _homogeneous_key(episodes[0])
    mismatches = [str(item["path"]) for item in episodes[1:] if _homogeneous_key(item) != reference]
    if mismatches:
        raise SystemExit(f"Input episodes do not share one data contract: {mismatches}")
    return [item for item in episodes if item["success"]]


def _load_episode(path: Path, contract=None, *, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compatibility wrapper used by tests and older callers."""
    del contract
    item = spec or inspect_npz_episode(path)
    return {
        key: item[key]
        for key in (
            "states", "actions", "timestamps", "state_timestamps", "action_timestamps",
            "image_timestamps", "images", "task_name", "instruction", "success", "metadata",
        )
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
    paths = sorted({*input_root.glob("ep_*.npz"), *input_root.glob("episode_*.npz")})
    if not paths:
        raise SystemExit(f"No episodes found in {input_root}")

    episodes = _validate_inputs(paths, fps, allow_incomplete_gripper_coverage)
    if not episodes:
        raise SystemExit("No successful episodes are available for export")
    if validate_only:
        print("Validation complete; no LeRobot dataset was written.")
        return None

    first = episodes[0]
    output_root = Path(root).expanduser()
    writer = Pi0LeRobotDatasetWriter(
        output_root,
        fps=fps,
        robot_type=first["robot_type"],
        state_names=list(first["state_names"]),
        action_names=list(first["action_names"]),
        camera_keys=list(first["camera_keys"]),
        image_hw=(224, 224),
        schema=first["schema"],
        arm_mode=first["arm_mode"],
        arm_side=first["arm_side"],
        action_semantics=first["action_semantics"],
        action_source=first["action_source"],
        action_alignment=first["action_alignment"],
        action_offset=first["action_offset"],
        action_horizon=first["action_horizon"],
        raw_action_dim=first["raw_action_dim"],
        model_action_dim=first["model_action_dim"],
        gripper_semantics=first["gripper_semantics"],
        rotation_semantics=first["rotation_semantics"],
        coordinate_frame=first["coordinate_frame"],
        legacy_format=first["legacy_format"],
    )

    frames = 0
    for count, item in enumerate(episodes, start=1):
        index = writer.append_episode(**_load_episode(item["path"], spec=item))
        frames += len(item["states"])
        print(
            f"Exported {item['path']} -> episode {index:06d} "
            f"({len(item['states'])} frames, instruction={item['instruction']!r})"
        )

    print(
        f"Export complete: root={output_root} schema={first['schema']} "
        f"arm={first['arm_mode']}/{first['arm_side']} format={first['contract_format']} "
        f"episodes={len(episodes)} frames={frames} fps={fps}"
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
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument(
        "--allow-incomplete-gripper-coverage",
        action="store_true",
        help="kept for CLI compatibility; per-episode gripper range is always checked",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
