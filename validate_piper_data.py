"""Validate raw Piper NPZ episodes against the training delivery contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


TARGET_FPS = 20.0
FPS_TOLERANCE = 0.05
MAX_SYNC_S = 0.050
ROTATION_NORM_TOLERANCE = 1e-3
ROTATION_DOT_TOLERANCE = 1e-3
TRANSLATION_TOLERANCE_M = 1e-5
ROTATION_ACTION_TOLERANCE_RAD = 1e-4
GRIPPER_TOLERANCE = 1e-5
GRIPPER_OPEN_THRESHOLD = 0.1
GRIPPER_CLOSED_THRESHOLD = 0.9
GRIPPER_TRANSITION_THRESHOLD = 0.01

REQUIRED_FIELDS = {
    "state",
    "actions",
    "timestamps",
    "image",
    "wrist_image",
    "instruction",
    "success",
    "image_timestamps_cam_high",
    "image_timestamps_cam_wrist",
}


class EpisodeValidationError(ValueError):
    def __init__(self, path: Path, errors: list[str], stats=None):
        self.path = path
        self.errors = errors
        self.stats = stats
        details = "\n".join(f"  - {error}" for error in errors)
        super().__init__(f"{path}: validation failed\n{details}")


@dataclass
class EpisodeStats:
    path: Path
    success: bool
    task: str | None
    instruction: str
    frames: int
    real_frames: int
    duration_s: float
    actual_fps: float
    dt_s: np.ndarray
    sync_high_s: np.ndarray
    sync_wrist_s: np.ndarray
    action_norms: np.ndarray
    translation_norms: np.ndarray
    rotation_norms: np.ndarray
    no_op_count: int
    no_op_total: int
    gripper_values: np.ndarray
    gripper_transition_count: int
    frozen_high_count: int
    frozen_wrist_count: int


def _require_exact_dtype(array: np.ndarray, dtype, name: str, errors: list[str]):
    expected = np.dtype(dtype)
    if array.dtype != expected:
        errors.append(f"{name} dtype must be {expected}, got {array.dtype}")


def _is_finite(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.number) and bool(np.isfinite(array).all())


def _read_string_scalar(data, name: str, required: bool, errors: list[str]) -> str | None:
    if name not in data.files:
        if required:
            errors.append(f"missing field: {name}")
        return None
    value = np.asarray(data[name])
    if value.shape != () or value.dtype.kind != "U":
        errors.append(f"{name} must be a Unicode string scalar, got shape={value.shape} dtype={value.dtype}")
        return None
    text = str(value.item()).strip()
    if required and not text:
        errors.append(f"{name} must not be empty")
    return text


def _frozen_transition_count(images: np.ndarray, real_frames: int) -> int:
    return sum(
        np.array_equal(images[index], images[index - 1])
        for index in range(1, real_frames)
    )


def _rotation_matrices(state: np.ndarray) -> np.ndarray:
    column_0 = np.asarray(state[:, 3:6], dtype=np.float64)
    column_1 = np.asarray(state[:, 6:9], dtype=np.float64)
    column_2 = np.cross(column_0, column_1)
    return np.stack((column_0, column_1, column_2), axis=2)


def _percentiles(values: np.ndarray) -> tuple[float, float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return float(p50), float(p95), float(p99), float(np.max(values))


def validate_episode(path: str | Path, target_fps: float = TARGET_FPS) -> EpisodeStats:
    path = Path(path)
    errors: list[str] = []
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")

    with np.load(path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_FIELDS.difference(data.files))
        if missing:
            errors.append(f"missing required fields: {missing}")
            raise EpisodeValidationError(path, errors)

        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"])
        timestamps = np.asarray(data["timestamps"])
        image = np.asarray(data["image"])
        wrist_image = np.asarray(data["wrist_image"])
        image_ts_high = np.asarray(data["image_timestamps_cam_high"])
        image_ts_wrist = np.asarray(data["image_timestamps_cam_wrist"])
        instruction = _read_string_scalar(data, "instruction", True, errors) or ""
        task = _read_string_scalar(data, "task", False, errors)
        if "_" in instruction:
            errors.append(
                "instruction must be natural language, not an internal snake_case task ID"
            )

        success_array = np.asarray(data["success"])
        if success_array.shape != () or success_array.dtype != np.dtype(np.bool_):
            errors.append(
                "success must be a bool scalar, "
                f"got shape={success_array.shape} dtype={success_array.dtype}"
            )
            success = False
        else:
            success = bool(success_array.item())

        _require_exact_dtype(state, np.float32, "state", errors)
        _require_exact_dtype(actions, np.float32, "actions", errors)
        _require_exact_dtype(timestamps, np.float64, "timestamps", errors)
        _require_exact_dtype(image, np.uint8, "image", errors)
        _require_exact_dtype(wrist_image, np.uint8, "wrist_image", errors)
        _require_exact_dtype(image_ts_high, np.float64, "image_timestamps_cam_high", errors)
        _require_exact_dtype(image_ts_wrist, np.float64, "image_timestamps_cam_wrist", errors)

        if state.ndim != 2 or state.shape[1:] != (10,):
            errors.append(f"state shape must be (T,10), got {state.shape}")
            frame_count = state.shape[0] if state.ndim else 0
        else:
            frame_count = len(state)

        expected_shapes = {
            "actions": ((frame_count, 7), actions.shape),
            "timestamps": ((frame_count,), timestamps.shape),
            "image": ((frame_count, 256, 256, 3), image.shape),
            "wrist_image": ((frame_count, 256, 256, 3), wrist_image.shape),
            "image_timestamps_cam_high": ((frame_count,), image_ts_high.shape),
            "image_timestamps_cam_wrist": ((frame_count,), image_ts_wrist.shape),
        }
        for name, (expected, actual) in expected_shapes.items():
            if actual != expected:
                errors.append(f"{name} shape must be {expected}, got {actual}")

        if "joint_qpos" in data.files:
            joint_qpos = np.asarray(data["joint_qpos"])
            _require_exact_dtype(joint_qpos, np.float32, "joint_qpos", errors)
            if joint_qpos.shape != (frame_count, 7):
                errors.append(
                    f"joint_qpos shape must be ({frame_count},7), got {joint_qpos.shape}"
                )
            elif not _is_finite(joint_qpos):
                errors.append("joint_qpos contains NaN or Inf")

        shapes_valid = all(expected == actual for expected, actual in expected_shapes.values())
        state_shape_valid = state.ndim == 2 and state.shape == (frame_count, 10)
        if frame_count < 3:
            errors.append("episode must contain at least two real frames and one terminal frame")

        if state_shape_valid and not _is_finite(state):
            errors.append("state contains NaN or Inf")
        if actions.shape == (frame_count, 7) and not _is_finite(actions):
            errors.append("actions contains NaN or Inf")
        if timestamps.shape == (frame_count,) and not _is_finite(timestamps):
            errors.append("timestamps contains NaN or Inf")
        if image_ts_high.shape == (frame_count,) and not _is_finite(image_ts_high):
            errors.append("image_timestamps_cam_high contains NaN or Inf")
        if image_ts_wrist.shape == (frame_count,) and not _is_finite(image_ts_wrist):
            errors.append("image_timestamps_cam_wrist contains NaN or Inf")

        timing_valid = (
            frame_count >= 3
            and timestamps.shape == (frame_count,)
            and _is_finite(timestamps)
        )
        if timing_valid:
            full_dt = np.diff(timestamps)
            if np.any(full_dt <= 0):
                errors.append("timestamps must be strictly increasing, including the terminal timestamp")
            real_timestamps = timestamps[:-1]
            dt_s = np.diff(real_timestamps)
            if np.any(dt_s <= 0):
                actual_fps = 0.0
                duration_s = 0.0
            else:
                actual_fps = float(1.0 / np.mean(dt_s))
                duration_s = float(real_timestamps[-1] - real_timestamps[0])
                relative_error = abs(actual_fps - target_fps) / target_fps
                if relative_error > FPS_TOLERANCE:
                    errors.append(
                        f"actual FPS {actual_fps:.3f} is outside {target_fps:.0f} +/-5%"
                    )
        else:
            dt_s = np.empty(0, dtype=np.float64)
            actual_fps = 0.0
            duration_s = 0.0

        image_timing_valid = _is_finite(image_ts_high) and _is_finite(image_ts_wrist)
        if timing_valid and shapes_valid and image_timing_valid:
            sync_high_s = np.abs(image_ts_high[:-1] - timestamps[:-1])
            sync_wrist_s = np.abs(image_ts_wrist[:-1] - timestamps[:-1])
            if np.max(sync_high_s) > MAX_SYNC_S:
                errors.append(
                    "image/state sync exceeds 50 ms for image: "
                    f"max={np.max(sync_high_s) * 1000:.2f} ms"
                )
            if np.max(sync_wrist_s) > MAX_SYNC_S:
                errors.append(
                    "image/state sync exceeds 50 ms for wrist_image: "
                    f"max={np.max(sync_wrist_s) * 1000:.2f} ms"
                )
        else:
            sync_high_s = np.empty(0, dtype=np.float64)
            sync_wrist_s = np.empty(0, dtype=np.float64)

        real_frames = max(0, frame_count - 1)
        frozen_high_count = 0
        frozen_wrist_count = 0
        if shapes_valid and real_frames:
            black_high = np.flatnonzero(np.max(image.reshape(frame_count, -1), axis=1) == 0)
            black_wrist = np.flatnonzero(np.max(wrist_image.reshape(frame_count, -1), axis=1) == 0)
            if len(black_high):
                errors.append(f"image contains all-black frames: {black_high[:10].tolist()}")
            if len(black_wrist):
                errors.append(f"wrist_image contains all-black frames: {black_wrist[:10].tolist()}")

            frozen_high_count = _frozen_transition_count(image, real_frames)
            frozen_wrist_count = _frozen_transition_count(wrist_image, real_frames)
            transition_count = max(0, real_frames - 1)
            if transition_count and frozen_high_count == transition_count:
                errors.append("image is frozen for the entire non-terminal episode")
            if transition_count and frozen_wrist_count == transition_count:
                errors.append("wrist_image is frozen for the entire non-terminal episode")
            if np.array_equal(image[:real_frames], wrist_image[:real_frames]):
                errors.append("image and wrist_image are identical; check the two camera mappings")

        action_shape_valid = actions.shape == (frame_count, 7)
        state_numeric_valid = frame_count > 0 and state_shape_valid and _is_finite(state)
        action_numeric_valid = frame_count > 0 and action_shape_valid and _is_finite(actions)
        rotation_state_valid = False
        if state_numeric_valid:
            column_0 = np.asarray(state[:, 3:6], dtype=np.float64)
            column_1 = np.asarray(state[:, 6:9], dtype=np.float64)
            norm_error_0 = np.max(np.abs(np.linalg.norm(column_0, axis=1) - 1.0))
            norm_error_1 = np.max(np.abs(np.linalg.norm(column_1, axis=1) - 1.0))
            dot_error = np.max(np.abs(np.sum(column_0 * column_1, axis=1)))
            if max(norm_error_0, norm_error_1) > ROTATION_NORM_TOLERANCE:
                errors.append(
                    "rotation6D column norm error exceeds 1e-3: "
                    f"max={max(norm_error_0, norm_error_1):.3e}"
                )
            if dot_error > ROTATION_DOT_TOLERANCE:
                errors.append(
                    f"rotation6D column dot product exceeds 1e-3: max={dot_error:.3e}"
                )
            rotation_state_valid = (
                max(norm_error_0, norm_error_1) <= ROTATION_NORM_TOLERANCE
                and dot_error <= ROTATION_DOT_TOLERANCE
            )

            state_gripper = state[:, 9]
            if np.any((state_gripper < 0.0) | (state_gripper > 1.0)):
                errors.append("state gripper_closed_fraction is outside [0,1]")

        if action_numeric_valid:
            action_gripper = actions[:, 6]
            if np.any((action_gripper < 0.0) | (action_gripper > 1.0)):
                errors.append("actions gripper target is outside [0,1]")

        if state_numeric_valid and action_numeric_valid and frame_count >= 2:
            expected_translation = state[1:, :3] - state[:-1, :3]
            translation_error = float(
                np.max(np.abs(actions[:-1, :3] - expected_translation))
            )
            if translation_error > TRANSLATION_TOLERANCE_M:
                errors.append(
                    "translation action reconstruction error exceeds 1e-5 m: "
                    f"max={translation_error:.3e}"
                )

            if rotation_state_valid:
                rotation_matrices = _rotation_matrices(state)
                rotation_delta = rotation_matrices[1:] @ np.swapaxes(
                    rotation_matrices[:-1], 1, 2
                )
                expected_rotation = Rotation.from_matrix(rotation_delta).as_rotvec()
                rotation_error = float(
                    np.max(
                        np.linalg.norm(
                            actions[:-1, 3:6] - expected_rotation,
                            axis=1,
                        )
                    )
                )
                if rotation_error > ROTATION_ACTION_TOLERANCE_RAD:
                    errors.append(
                        "rotation action reconstruction error exceeds 1e-4 rad: "
                        f"max={rotation_error:.3e}"
                    )

            gripper_error = float(np.max(np.abs(actions[:-1, 6] - state[1:, 9])))
            if gripper_error > GRIPPER_TOLERANCE:
                errors.append(
                    "gripper action reconstruction error exceeds 1e-5: "
                    f"max={gripper_error:.3e}"
                )

            terminal_motion = float(np.max(np.abs(actions[-1, :6])))
            if terminal_motion > 1e-6:
                errors.append(
                    f"terminal action first 6 values must be zero, max={terminal_motion:.3e}"
                )
            terminal_gripper_error = abs(float(actions[-1, 6] - state[-1, 9]))
            if terminal_gripper_error > GRIPPER_TOLERANCE:
                errors.append(
                    "terminal action must hold the final gripper target, "
                    f"error={terminal_gripper_error:.3e}"
                )

            measured_actions = actions[:-1]
            translation_norms = np.linalg.norm(measured_actions[:, :3], axis=1)
            rotation_norms = np.linalg.norm(measured_actions[:, 3:6], axis=1)
            action_norms = np.linalg.norm(measured_actions[:, :6], axis=1)
            gripper_change = np.abs(measured_actions[:, 6] - state[:-1, 9])
            no_op = (
                (translation_norms <= 1e-6)
                & (rotation_norms <= 1e-6)
                & (gripper_change <= GRIPPER_TOLERANCE)
            )
            no_op_count = int(np.count_nonzero(no_op))
            no_op_total = len(no_op)
            gripper_values = np.asarray(state[:-1, 9], dtype=np.float32).copy()
            gripper_transition_count = int(
                np.count_nonzero(
                    np.abs(np.diff(gripper_values)) > GRIPPER_TRANSITION_THRESHOLD
                )
            )
        else:
            action_norms = np.empty(0, dtype=np.float64)
            translation_norms = np.empty(0, dtype=np.float64)
            rotation_norms = np.empty(0, dtype=np.float64)
            no_op_count = 0
            no_op_total = 0
            gripper_values = np.empty(0, dtype=np.float32)
            gripper_transition_count = 0

    stats = EpisodeStats(
        path=path,
        success=success,
        task=task,
        instruction=instruction,
        frames=frame_count,
        real_frames=real_frames,
        duration_s=duration_s,
        actual_fps=actual_fps,
        dt_s=dt_s,
        sync_high_s=sync_high_s,
        sync_wrist_s=sync_wrist_s,
        action_norms=action_norms,
        translation_norms=translation_norms,
        rotation_norms=rotation_norms,
        no_op_count=no_op_count,
        no_op_total=no_op_total,
        gripper_values=gripper_values,
        gripper_transition_count=gripper_transition_count,
        frozen_high_count=frozen_high_count,
        frozen_wrist_count=frozen_wrist_count,
    )
    if errors:
        raise EpisodeValidationError(path, errors, stats=stats)
    return stats


def validate_gripper_coverage(stats: list[EpisodeStats]):
    if not stats:
        raise ValueError("no successful episodes are available for export")
    gripper = np.concatenate([item.gripper_values for item in stats])
    transition_count = sum(item.gripper_transition_count for item in stats)
    missing = []
    if not np.any(gripper <= GRIPPER_OPEN_THRESHOLD):
        missing.append("open states (closed_fraction <= 0.1)")
    if not np.any(gripper >= GRIPPER_CLOSED_THRESHOLD):
        missing.append("closed states (closed_fraction >= 0.9)")
    if transition_count == 0:
        missing.append("gripper transitions (step change > 0.01)")
    if missing:
        raise ValueError("successful dataset lacks " + ", ".join(missing))


def validate_instruction_consistency(stats: list[EpisodeStats]):
    instructions_by_task: dict[str, set[str]] = {}
    for item in stats:
        if item.task:
            instructions_by_task.setdefault(item.task, set()).add(item.instruction)
    inconsistent = {
        task: sorted(instructions)
        for task, instructions in instructions_by_task.items()
        if len(instructions) > 1
    }
    if inconsistent:
        raise ValueError(
            "internal task IDs map to inconsistent instructions: "
            f"{inconsistent}"
        )


def format_episode_report(stats: EpisodeStats) -> str:
    no_op_ratio = stats.no_op_count / max(1, stats.no_op_total)
    p50, p95, p99, maximum = _percentiles(stats.action_norms)
    _, high_p95, _, high_max = _percentiles(stats.sync_high_s)
    _, wrist_p95, _, wrist_max = _percentiles(stats.sync_wrist_s)
    high_mean = float(np.mean(stats.sync_high_s))
    wrist_mean = float(np.mean(stats.sync_wrist_s))
    gripper = stats.gripper_values
    open_count = int(np.count_nonzero(gripper <= GRIPPER_OPEN_THRESHOLD))
    closed_count = int(np.count_nonzero(gripper >= GRIPPER_CLOSED_THRESHOLD))
    warning = " WARNING:no-op>50%" if no_op_ratio > 0.5 else ""
    return (
        f"PASS {stats.path}: frames={stats.frames} real={stats.real_frames} "
        f"duration={stats.duration_s:.2f}s fps={stats.actual_fps:.3f}\n"
        f"  sync_ms image mean/p95/max={high_mean * 1000:.2f}/"
        f"{high_p95 * 1000:.2f}/{high_max * 1000:.2f}, wrist_image="
        f"{wrist_mean * 1000:.2f}/{wrist_p95 * 1000:.2f}/{wrist_max * 1000:.2f}\n"
        f"  action_norm p50/p95/p99/max={p50:.6f}/{p95:.6f}/{p99:.6f}/"
        f"{maximum:.6f}, no_op={no_op_ratio:.1%}{warning}\n"
        f"  gripper min/max={np.min(gripper):.3f}/{np.max(gripper):.3f}, "
        f"open={open_count}, closed={closed_count}, "
        f"transitions={stats.gripper_transition_count}\n"
        f"  frozen_transitions image={stats.frozen_high_count}, "
        f"wrist_image={stats.frozen_wrist_count}"
    )


def format_dataset_report(stats: list[EpisodeStats]) -> str:
    action_norms = np.concatenate([item.action_norms for item in stats])
    gripper = np.concatenate([item.gripper_values for item in stats])
    p50, p95, p99, maximum = _percentiles(action_norms)
    no_op_count = sum(item.no_op_count for item in stats)
    no_op_total = sum(item.no_op_total for item in stats)
    return (
        f"Dataset PASS: episodes={len(stats)} frames={sum(item.frames for item in stats)} "
        f"real_duration={sum(item.duration_s for item in stats):.2f}s\n"
        f"  fps min/max={min(item.actual_fps for item in stats):.3f}/"
        f"{max(item.actual_fps for item in stats):.3f}\n"
        f"  action_norm p50/p95/p99/max={p50:.6f}/{p95:.6f}/{p99:.6f}/"
        f"{maximum:.6f}, no_op={no_op_count / max(1, no_op_total):.1%}\n"
        f"  gripper min/max={np.min(gripper):.3f}/{np.max(gripper):.3f}, "
        f"transitions={sum(item.gripper_transition_count for item in stats)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="episodes_piper_v21")
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob("ep_*.npz"))
    if not paths:
        raise SystemExit(f"No episodes found in {args.input_dir}")

    successful: list[EpisodeStats] = []
    coverage_candidates: list[EpisodeStats] = []
    failures: list[str] = []
    for path in paths:
        try:
            stats = validate_episode(path, target_fps=args.target_fps)
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
        details = "\n\n".join(failures)
        if coverage_error:
            details += f"\n\nDataset validation failed: {coverage_error}"
        raise SystemExit(details)
    if coverage_error:
        raise SystemExit(f"Dataset validation failed: {coverage_error}")
    print(format_dataset_report(successful))


if __name__ == "__main__":
    main()
