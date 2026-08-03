"""Validate raw Piper NPZ episodes for single-arm/bimanual π0.5 training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from piper_data_contract import (
    DELIVERY_ACTION_SEMANTICS,
    DELIVERY_SCHEMA,
    GRIPPER_MAX_M,
    IMAGE_HW,
    JOINT_ACTION_SEMANTICS,
    JOINT_SCHEMA,
    LEGACY_NEXT_JOINT_ACTION_SEMANTICS,
    EpisodeContract,
    infer_episode_contract,
)


TARGET_FPS = 20.0
FPS_TOLERANCE = 0.05
MAX_SYNC_S = 0.050
ROTATION_NORM_TOLERANCE = 1e-3
ROTATION_DOT_TOLERANCE = 1e-3
TRANSLATION_TOLERANCE_M = 1e-5
ROTATION_ACTION_TOLERANCE_RAD = 1e-4
GRIPPER_TOLERANCE = 1e-5
JOINT_ACTION_TOLERANCE = 1e-6
GRIPPER_OPEN_THRESHOLD = 0.1
GRIPPER_CLOSED_THRESHOLD = 0.9
GRIPPER_TRANSITION_THRESHOLD = 0.01


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
    schema: str = DELIVERY_SCHEMA
    arm_mode: str = "single"
    arm_side: str = "right"
    state_dim: int = 10
    action_dim: int = 7
    camera_keys: tuple[str, ...] = ("cam_high", "cam_wrist")
    action_semantics: str = DELIVERY_ACTION_SEMANTICS
    action_source: str = "next_measured_eef"
    action_alignment: str = "next_observation"


class _NpzMapping(Mapping[str, Any]):
    """Expose NpzFile as a normal Mapping for contract inference."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self):
        return iter(self.data.files)

    def __len__(self) -> int:
        return len(self.data.files)


def _require_exact_dtype(array: np.ndarray, dtype, name: str, errors: list[str]) -> None:
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
        errors.append(
            f"{name} must be a Unicode string scalar, got shape={value.shape} dtype={value.dtype}"
        )
        return None
    text = str(value.item()).strip()
    if required and not text:
        errors.append(f"{name} must not be empty")
    return text


def _read_scalar(data, name: str, default: Any = None) -> Any:
    if name not in data.files:
        return default
    value = np.asarray(data[name])
    return value.item() if value.shape == () else default


def _frozen_transition_count(images: np.ndarray, real_frames: int) -> int:
    return sum(
        np.array_equal(images[index], images[index - 1])
        for index in range(1, real_frames)
    )


def _percentiles(values: np.ndarray) -> tuple[float, float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return float(p50), float(p95), float(p99), float(np.max(values))


def _image_candidates(contract: EpisodeContract, camera_key: str) -> tuple[str, ...]:
    candidates = [
        contract.image_field(camera_key),
        f"images_{camera_key}",
        f"observation.images.{camera_key}",
    ]
    if camera_key == "cam_high":
        candidates.append("image")
    elif camera_key == "cam_wrist":
        candidates.append("wrist_image")
    return tuple(dict.fromkeys(candidates))


def _load_image(data, contract: EpisodeContract, camera_key: str, errors: list[str]):
    for field in _image_candidates(contract, camera_key):
        if field in data.files:
            return np.asarray(data[field]), field
    errors.append(
        f"missing camera {camera_key}; expected one of {_image_candidates(contract, camera_key)}"
    )
    return np.empty((0, *IMAGE_HW, 3), dtype=np.uint8), contract.image_field(camera_key)


def _validate_contract_metadata(data, contract: EpisodeContract, errors: list[str]) -> None:
    expected_scalars = {
        "schema": contract.schema,
        "arm_mode": contract.arm_mode,
        "arm_side": contract.arm_side,
        "state_dim": contract.state_dim,
        "action_dim": contract.action_dim,
        "action_alignment": contract.action_alignment,
        "action_offset": contract.action_offset,
    }
    for key, expected in expected_scalars.items():
        if key in data.files:
            actual = _read_scalar(data, key)
            if actual != expected:
                errors.append(f"metadata {key}={actual!r}, inferred/expected {expected!r}")
    if "camera_keys" in data.files:
        values = np.asarray(data["camera_keys"])
        actual = tuple(str(item) for item in values.tolist()) if values.ndim == 1 else ()
        if actual != contract.camera_keys:
            errors.append(f"metadata camera_keys={actual}, expected {contract.camera_keys}")
    if "action_semantics" in data.files:
        actual = str(_read_scalar(data, "action_semantics", ""))
        accepted = {contract.action_semantics}
        if contract.schema == JOINT_SCHEMA:
            accepted.add(LEGACY_NEXT_JOINT_ACTION_SEMANTICS)
        if actual not in accepted:
            errors.append(
                f"metadata action_semantics={actual!r}, expected one of {sorted(accepted)}"
            )


def _delivery_rotation_matrices(arm_states: np.ndarray) -> np.ndarray:
    column_0 = np.asarray(arm_states[:, 3:6], dtype=np.float64)
    column_1 = np.asarray(arm_states[:, 6:9], dtype=np.float64)
    column_2 = np.cross(column_0, column_1)
    return np.stack((column_0, column_1, column_2), axis=2)


def _empty_stats(path: Path, contract: EpisodeContract, instruction: str = "") -> EpisodeStats:
    empty = np.empty(0, dtype=np.float64)
    return EpisodeStats(
        path=path,
        success=False,
        task=None,
        instruction=instruction,
        frames=0,
        real_frames=0,
        duration_s=0.0,
        actual_fps=0.0,
        dt_s=empty,
        sync_high_s=empty,
        sync_wrist_s=empty,
        action_norms=empty,
        translation_norms=empty,
        rotation_norms=empty,
        no_op_count=0,
        no_op_total=0,
        gripper_values=np.empty(0, dtype=np.float32),
        gripper_transition_count=0,
        frozen_high_count=0,
        frozen_wrist_count=0,
        schema=contract.schema,
        arm_mode=contract.arm_mode,
        arm_side=contract.arm_side,
        state_dim=contract.state_dim,
        action_dim=contract.action_dim,
        camera_keys=contract.camera_keys,
        action_semantics=contract.action_semantics,
        action_source=contract.action_source,
        action_alignment=contract.action_alignment,
    )


def validate_episode(path: str | Path, target_fps: float = TARGET_FPS) -> EpisodeStats:
    path = Path(path)
    errors: list[str] = []
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")

    with np.load(path, allow_pickle=False) as data:
        if "state" not in data.files or "actions" not in data.files:
            missing = [key for key in ("state", "actions") if key not in data.files]
            raise EpisodeValidationError(path, [f"missing required fields: {missing}"])
        try:
            contract = infer_episode_contract(_NpzMapping(data))
        except (KeyError, TypeError, ValueError) as exc:
            raise EpisodeValidationError(path, [f"invalid episode contract: {exc}"]) from exc
        _validate_contract_metadata(data, contract, errors)

        core_required = {"state", "actions", "timestamps", "instruction", "success"}
        missing_core = sorted(core_required.difference(data.files))
        if missing_core:
            errors.append(f"missing required fields: {missing_core}")

        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"])
        timestamps = np.asarray(data["timestamps"]) if "timestamps" in data.files else np.empty(0)
        instruction = _read_string_scalar(data, "instruction", True, errors) or ""
        task = _read_string_scalar(data, "task", False, errors)
        if "_" in instruction:
            errors.append("instruction must be natural language, not an internal snake_case task ID")

        if "success" in data.files:
            success_array = np.asarray(data["success"])
            if success_array.shape != () or success_array.dtype != np.dtype(np.bool_):
                errors.append(
                    "success must be a bool scalar, "
                    f"got shape={success_array.shape} dtype={success_array.dtype}"
                )
                success = False
            else:
                success = bool(success_array.item())
        else:
            success = False

        _require_exact_dtype(state, np.float32, "state", errors)
        _require_exact_dtype(actions, np.float32, "actions", errors)
        _require_exact_dtype(timestamps, np.float64, "timestamps", errors)

        if state.ndim != 2 or state.shape[1:] != (contract.state_dim,):
            errors.append(f"state shape must be (T,{contract.state_dim}), got {state.shape}")
            frame_count = state.shape[0] if state.ndim else 0
        else:
            frame_count = len(state)
        if actions.shape != (frame_count, contract.action_dim):
            errors.append(
                f"actions shape must be ({frame_count},{contract.action_dim}), got {actions.shape}"
            )
        if timestamps.shape != (frame_count,):
            errors.append(f"timestamps shape must be ({frame_count},), got {timestamps.shape}")

        images: dict[str, np.ndarray] = {}
        image_timestamps: dict[str, np.ndarray] = {}
        image_fields: dict[str, str] = {}
        for camera_key in contract.camera_keys:
            image, image_field = _load_image(data, contract, camera_key, errors)
            images[camera_key] = image
            image_fields[camera_key] = image_field
            timestamp_field = contract.timestamp_field(camera_key)
            if timestamp_field not in data.files:
                errors.append(f"missing field: {timestamp_field}")
                image_ts = np.empty(0)
            else:
                image_ts = np.asarray(data[timestamp_field])
            image_timestamps[camera_key] = image_ts
            _require_exact_dtype(image, np.uint8, image_field, errors)
            _require_exact_dtype(image_ts, np.float64, timestamp_field, errors)
            if image.shape != (frame_count, *IMAGE_HW, 3):
                errors.append(
                    f"{image_field} shape must be ({frame_count},{IMAGE_HW[0]},{IMAGE_HW[1]},3), "
                    f"got {image.shape}"
                )
            if image_ts.shape != (frame_count,):
                errors.append(
                    f"{timestamp_field} shape must be ({frame_count},), got {image_ts.shape}"
                )

        if "joint_qpos" in data.files:
            joint_qpos = np.asarray(data["joint_qpos"])
            _require_exact_dtype(joint_qpos, np.float32, "joint_qpos", errors)
            if joint_qpos.shape != (frame_count, contract.joint_dim):
                errors.append(
                    f"joint_qpos shape must be ({frame_count},{contract.joint_dim}), got {joint_qpos.shape}"
                )
            elif not _is_finite(joint_qpos):
                errors.append("joint_qpos contains NaN or Inf")

        terminal_padding = bool(_read_scalar(data, "terminal_padding", True))
        min_frames = 3 if terminal_padding else 2
        if frame_count < min_frames:
            errors.append(
                "episode must contain at least two real frames"
                + (" and one terminal frame" if terminal_padding else "")
            )
        real_frames = max(0, frame_count - 1) if terminal_padding else frame_count

        if state.shape == (frame_count, contract.state_dim) and not _is_finite(state):
            errors.append("state contains NaN or Inf")
        if actions.shape == (frame_count, contract.action_dim) and not _is_finite(actions):
            errors.append("actions contains NaN or Inf")
        if timestamps.shape == (frame_count,) and not _is_finite(timestamps):
            errors.append("timestamps contains NaN or Inf")
        for camera_key, image_ts in image_timestamps.items():
            if image_ts.shape == (frame_count,) and not _is_finite(image_ts):
                errors.append(f"{contract.timestamp_field(camera_key)} contains NaN or Inf")

        timing_valid = (
            frame_count >= min_frames
            and timestamps.shape == (frame_count,)
            and _is_finite(timestamps)
        )
        if timing_valid:
            full_dt = np.diff(timestamps)
            if np.any(full_dt <= 0):
                errors.append("timestamps must be strictly increasing")
            real_timestamps = timestamps[:real_frames]
            dt_s = np.diff(real_timestamps)
            if len(dt_s) and np.all(dt_s > 0):
                actual_fps = float(1.0 / np.mean(dt_s))
                duration_s = float(real_timestamps[-1] - real_timestamps[0])
                relative_error = abs(actual_fps - target_fps) / target_fps
                if relative_error > FPS_TOLERANCE:
                    errors.append(
                        f"actual FPS {actual_fps:.3f} is outside {target_fps:.0f} +/-5%"
                    )
            else:
                actual_fps = 0.0
                duration_s = 0.0
        else:
            dt_s = np.empty(0, dtype=np.float64)
            actual_fps = 0.0
            duration_s = 0.0

        sync_by_camera: dict[str, np.ndarray] = {}
        frozen_by_camera: dict[str, int] = {}
        shapes_valid = True
        for camera_key in contract.camera_keys:
            image = images[camera_key]
            image_ts = image_timestamps[camera_key]
            valid_shape = image.shape == (frame_count, *IMAGE_HW, 3)
            valid_ts = image_ts.shape == (frame_count,) and _is_finite(image_ts)
            shapes_valid &= valid_shape and valid_ts
            if timing_valid and valid_ts:
                sync = np.abs(image_ts[:real_frames] - timestamps[:real_frames])
                sync_by_camera[camera_key] = sync
                if len(sync) and np.max(sync) > MAX_SYNC_S:
                    errors.append(
                        f"image/state sync exceeds 50 ms for {camera_key}: "
                        f"max={np.max(sync) * 1000:.2f} ms"
                    )
            else:
                sync_by_camera[camera_key] = np.empty(0, dtype=np.float64)
            frozen_by_camera[camera_key] = 0
            if valid_shape and real_frames:
                black = np.flatnonzero(np.max(image.reshape(frame_count, -1), axis=1) == 0)
                if len(black):
                    errors.append(
                        f"{camera_key} contains all-black frames: {black[:10].tolist()}"
                    )
                frozen = _frozen_transition_count(image, real_frames)
                frozen_by_camera[camera_key] = frozen
                transition_count = max(0, real_frames - 1)
                if transition_count and frozen == transition_count:
                    errors.append(f"{camera_key} is frozen for the entire non-terminal episode")

        if shapes_valid and real_frames:
            high = images["cam_high"][:real_frames]
            for camera_key in contract.camera_keys:
                if camera_key != "cam_high" and np.array_equal(high, images[camera_key][:real_frames]):
                    errors.append(
                        f"cam_high and {camera_key} are identical; check camera mappings"
                    )

        numeric_valid = (
            frame_count > 0
            and state.shape == (frame_count, contract.state_dim)
            and actions.shape == (frame_count, contract.action_dim)
            and _is_finite(state)
            and _is_finite(actions)
        )
        action_norms = np.empty(0, dtype=np.float64)
        translation_norms = np.empty(0, dtype=np.float64)
        rotation_norms = np.empty(0, dtype=np.float64)
        no_op_count = 0
        no_op_total = 0
        gripper_values = np.empty(0, dtype=np.float32)
        gripper_transition_count = 0

        if numeric_valid:
            gripper_chunks: list[np.ndarray] = []
            for index in contract.gripper_state_indices:
                values = state[:real_frames, index]
                if contract.schema == JOINT_SCHEMA:
                    values = np.clip(1.0 - values / GRIPPER_MAX_M, 0.0, 1.0)
                gripper_chunks.append(np.asarray(values, dtype=np.float32))
            if gripper_chunks:
                gripper_values = np.concatenate(gripper_chunks)
                gripper_transition_count = sum(
                    int(np.count_nonzero(np.abs(np.diff(values)) > GRIPPER_TRANSITION_THRESHOLD))
                    for values in gripper_chunks
                )

            if contract.schema == DELIVERY_SCHEMA:
                motion_norms: list[np.ndarray] = []
                translations: list[np.ndarray] = []
                rotations: list[np.ndarray] = []
                no_op_parts: list[np.ndarray] = []
                for arm_index in range(contract.arm_count):
                    ss = arm_index * 10
                    aa = arm_index * 7
                    arm_state = state[:, ss : ss + 10]
                    arm_action = actions[:, aa : aa + 7]
                    column_0 = np.asarray(arm_state[:, 3:6], dtype=np.float64)
                    column_1 = np.asarray(arm_state[:, 6:9], dtype=np.float64)
                    norm_error_0 = float(np.max(np.abs(np.linalg.norm(column_0, axis=1) - 1.0)))
                    norm_error_1 = float(np.max(np.abs(np.linalg.norm(column_1, axis=1) - 1.0)))
                    dot_error = float(np.max(np.abs(np.sum(column_0 * column_1, axis=1))))
                    if max(norm_error_0, norm_error_1) > ROTATION_NORM_TOLERANCE:
                        errors.append(
                            f"arm {arm_index} rotation6D column norm error exceeds 1e-3"
                        )
                    if dot_error > ROTATION_DOT_TOLERANCE:
                        errors.append(
                            f"arm {arm_index} rotation6D columns are not orthogonal"
                        )
                    if terminal_padding and frame_count >= 2:
                        expected_translation = arm_state[1:, :3] - arm_state[:-1, :3]
                        error = float(np.max(np.abs(arm_action[:-1, :3] - expected_translation)))
                        if contract.action_alignment == "next_observation" and error > TRANSLATION_TOLERANCE_M:
                            errors.append(
                                f"arm {arm_index} translation action reconstruction error "
                                f"exceeds 1e-5 m: max={error:.3e}"
                            )
                        matrices = _delivery_rotation_matrices(arm_state)
                        delta = matrices[1:] @ np.swapaxes(matrices[:-1], 1, 2)
                        expected_rotation = Rotation.from_matrix(delta).as_rotvec()
                        rotation_error = float(
                            np.max(np.linalg.norm(arm_action[:-1, 3:6] - expected_rotation, axis=1))
                        )
                        if contract.action_alignment == "next_observation" and rotation_error > ROTATION_ACTION_TOLERANCE_RAD:
                            errors.append(
                                f"arm {arm_index} rotation action reconstruction error "
                                f"exceeds 1e-4 rad: max={rotation_error:.3e}"
                            )
                        gripper_error = float(
                            np.max(np.abs(arm_action[:-1, 6] - arm_state[1:, 9]))
                        )
                        if contract.action_alignment == "next_observation" and gripper_error > GRIPPER_TOLERANCE:
                            errors.append(
                                f"arm {arm_index} gripper action reconstruction error "
                                f"exceeds 1e-5: max={gripper_error:.3e}"
                            )
                        if np.max(np.abs(arm_action[-1, :6])) > 1e-6:
                            errors.append(f"arm {arm_index} terminal delivery motion must be zero")
                        if abs(float(arm_action[-1, 6] - arm_state[-1, 9])) > GRIPPER_TOLERANCE:
                            errors.append(f"arm {arm_index} terminal gripper action must hold")
                    measured = arm_action[:real_frames]
                    translation = np.linalg.norm(measured[:, :3], axis=1)
                    rotation = np.linalg.norm(measured[:, 3:6], axis=1)
                    gripper_change = np.abs(measured[:, 6] - arm_state[:real_frames, 9])
                    translations.append(translation)
                    rotations.append(rotation)
                    motion_norms.append(np.linalg.norm(measured[:, :6], axis=1))
                    no_op_parts.append(
                        (translation <= 1e-6)
                        & (rotation <= 1e-6)
                        & (gripper_change <= GRIPPER_TOLERANCE)
                    )
                translation_norms = np.linalg.norm(np.stack(translations, axis=1), axis=1)
                rotation_norms = np.linalg.norm(np.stack(rotations, axis=1), axis=1)
                action_norms = np.linalg.norm(np.stack(motion_norms, axis=1), axis=1)
                no_op = np.logical_and.reduce(no_op_parts) if no_op_parts else np.empty(0, bool)
            else:
                measured = actions[:real_frames]
                current = state[:real_frames]
                delta = measured - current
                joint_delta_parts = []
                gripper_delta_parts = []
                for arm_index in range(contract.arm_count):
                    start = arm_index * 7
                    joint_delta_parts.append(delta[:, start : start + 6])
                    gripper_delta_parts.append(np.abs(delta[:, start + 6]))
                joints = np.concatenate(joint_delta_parts, axis=1)
                grippers = np.stack(gripper_delta_parts, axis=1)
                action_norms = np.linalg.norm(joints, axis=1)
                translation_norms = action_norms.copy()
                rotation_norms = np.zeros_like(action_norms)
                no_op = (action_norms <= JOINT_ACTION_TOLERANCE) & np.all(
                    grippers <= JOINT_ACTION_TOLERANCE, axis=1
                )
                if terminal_padding:
                    if contract.action_alignment == "next_observation" and not np.allclose(
                        actions[:-1], state[1:], atol=JOINT_ACTION_TOLERANCE, rtol=0
                    ):
                        error = float(np.max(np.abs(actions[:-1] - state[1:])))
                        errors.append(
                            "joint actions do not match the next measured observation: "
                            f"max error={error:.3e}"
                        )
                    if not np.allclose(actions[-1], state[-1], atol=JOINT_ACTION_TOLERANCE, rtol=0):
                        errors.append("terminal joint action must hold the final joint state")
            no_op_count = int(np.count_nonzero(no_op))
            no_op_total = len(no_op)

        wrist_sync = [
            sync_by_camera[key]
            for key in contract.camera_keys
            if key != "cam_high" and len(sync_by_camera[key])
        ]
        sync_wrist_s = np.concatenate(wrist_sync) if wrist_sync else np.empty(0, dtype=np.float64)
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
            sync_high_s=sync_by_camera.get("cam_high", np.empty(0, dtype=np.float64)),
            sync_wrist_s=sync_wrist_s,
            action_norms=action_norms,
            translation_norms=translation_norms,
            rotation_norms=rotation_norms,
            no_op_count=no_op_count,
            no_op_total=no_op_total,
            gripper_values=gripper_values,
            gripper_transition_count=gripper_transition_count,
            frozen_high_count=frozen_by_camera.get("cam_high", 0),
            frozen_wrist_count=sum(
                value for key, value in frozen_by_camera.items() if key != "cam_high"
            ),
            schema=contract.schema,
            arm_mode=contract.arm_mode,
            arm_side=contract.arm_side,
            state_dim=contract.state_dim,
            action_dim=contract.action_dim,
            camera_keys=contract.camera_keys,
            action_semantics=contract.action_semantics,
            action_source=contract.action_source,
            action_alignment=contract.action_alignment,
        )

    if success and no_op_total > 0 and no_op_count == no_op_total:
        errors.append(
            "successful episode contains no robot motion or gripper change "
            "(100% no-op); check Piper CAN feedback before recording"
        )
    if errors:
        raise EpisodeValidationError(path, errors, stats=stats)
    return stats


def validate_gripper_coverage(stats: list[EpisodeStats]) -> None:
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


def validate_instruction_consistency(stats: list[EpisodeStats]) -> None:
    instructions_by_task: dict[str, set[str]] = {}
    contracts = {
        (
            item.schema,
            item.arm_mode,
            item.arm_side,
            item.state_dim,
            item.action_dim,
            item.camera_keys,
            item.action_semantics,
            item.action_alignment,
        )
        for item in stats
    }
    if len(contracts) > 1:
        raise ValueError(f"episodes mix incompatible contracts: {sorted(contracts)!r}")
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
    high_mean = float(np.mean(stats.sync_high_s)) if len(stats.sync_high_s) else 0.0
    wrist_mean = float(np.mean(stats.sync_wrist_s)) if len(stats.sync_wrist_s) else 0.0
    gripper = stats.gripper_values
    gripper_min = float(np.min(gripper)) if len(gripper) else 0.0
    gripper_max = float(np.max(gripper)) if len(gripper) else 0.0
    open_count = int(np.count_nonzero(gripper <= GRIPPER_OPEN_THRESHOLD))
    closed_count = int(np.count_nonzero(gripper >= GRIPPER_CLOSED_THRESHOLD))
    warning = " WARNING:no-op>50%" if no_op_ratio > 0.5 else ""
    return (
        f"PASS {stats.path}: schema={stats.schema} arm={stats.arm_mode}/{stats.arm_side} "
        f"state={stats.state_dim} action={stats.action_dim} frames={stats.frames} "
        f"real={stats.real_frames} duration={stats.duration_s:.2f}s fps={stats.actual_fps:.3f}\n"
        f"  cameras={list(stats.camera_keys)} sync_ms high mean/p95/max={high_mean * 1000:.2f}/"
        f"{high_p95 * 1000:.2f}/{high_max * 1000:.2f}, wrists="
        f"{wrist_mean * 1000:.2f}/{wrist_p95 * 1000:.2f}/{wrist_max * 1000:.2f}\n"
        f"  action={stats.action_semantics} source={stats.action_source} "
        f"alignment={stats.action_alignment}\n"
        f"  action_norm p50/p95/p99/max={p50:.6f}/{p95:.6f}/{p99:.6f}/"
        f"{maximum:.6f}, no_op={no_op_ratio:.1%}{warning}\n"
        f"  gripper closed_fraction min/max={gripper_min:.3f}/{gripper_max:.3f}, "
        f"open={open_count}, closed={closed_count}, transitions={stats.gripper_transition_count}\n"
        f"  frozen_transitions high={stats.frozen_high_count}, wrists={stats.frozen_wrist_count}"
    )


def format_dataset_report(stats: list[EpisodeStats]) -> str:
    if not stats:
        return "Dataset PASS: episodes=0 frames=0"
    action_norms = np.concatenate([item.action_norms for item in stats])
    gripper = np.concatenate([item.gripper_values for item in stats])
    p50, p95, p99, maximum = _percentiles(action_norms)
    no_op_count = sum(item.no_op_count for item in stats)
    no_op_total = sum(item.no_op_total for item in stats)
    first = stats[0]
    return (
        f"Dataset PASS: schema={first.schema} arm={first.arm_mode}/{first.arm_side} "
        f"episodes={len(stats)} frames={sum(item.frames for item in stats)} "
        f"real_duration={sum(item.duration_s for item in stats):.2f}s\n"
        f"  fps min/max={min(item.actual_fps for item in stats):.3f}/"
        f"{max(item.actual_fps for item in stats):.3f}\n"
        f"  action_norm p50/p95/p99/max={p50:.6f}/{p95:.6f}/{p99:.6f}/"
        f"{maximum:.6f}, no_op={no_op_count / max(1, no_op_total):.1%}\n"
        f"  gripper min/max={np.min(gripper):.3f}/{np.max(gripper):.3f}, "
        f"transitions={sum(item.gripper_transition_count for item in stats)}"
    )


def main() -> None:
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

    dataset_error = None
    try:
        validate_gripper_coverage(coverage_candidates)
        validate_instruction_consistency(coverage_candidates)
    except ValueError as exc:
        dataset_error = str(exc)
    if failures:
        details = "\n\n".join(failures)
        if dataset_error:
            details += f"\n\nDataset validation failed: {dataset_error}"
        raise SystemExit(details)
    if dataset_error:
        raise SystemExit(f"Dataset validation failed: {dataset_error}")
    print(format_dataset_report(successful))


if __name__ == "__main__":
    main()
