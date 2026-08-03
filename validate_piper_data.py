"""Validate raw Piper NPZ episodes across explicit legacy-v2 and new-v3 paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from piper_action_conventions import rotation6d_to_matrix
from piper_data_contract import (
    CONTRACT_VERSION,
    DEFAULT_ACTION_HORIZON,
    DEFAULT_FPS,
    DELIVERY_ACTION_SEMANTICS,
    DELIVERY_MEASURED_ACTION_SOURCE,
    DELIVERY_SCHEMA,
    GRIPPER_MAX_M,
    IMAGE_HW,
    JOINT_SCHEMA,
    EpisodeContract,
    infer_episode_contract,
)

TARGET_FPS = float(DEFAULT_FPS)
FPS_TOLERANCE = 0.05
MAX_SYNC_S = 0.050
ROTATION_NORM_TOLERANCE = 1e-3
ROTATION_DOT_TOLERANCE = 1e-3
TRANSLATION_TOLERANCE_M = 1e-5
ROTATION_ACTION_TOLERANCE_RAD = 1e-4
GRIPPER_TOLERANCE = 1e-5
JOINT_ACTION_TOLERANCE = 1e-6
GRIPPER_CLOSED_THRESHOLD = 0.1
GRIPPER_OPEN_THRESHOLD = 0.9
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
    action_dim: int = 10
    model_action_dim: int = 7
    camera_keys: tuple[str, ...] = ("cam_high", "cam_right_wrist")
    action_semantics: str = DELIVERY_ACTION_SEMANTICS
    action_source: str = DELIVERY_MEASURED_ACTION_SOURCE
    action_alignment: str = "next_observation"
    contract_version: int = CONTRACT_VERSION
    legacy_layout: bool = False


class _NpzMapping(Mapping[str, Any]):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self):
        return iter(self.data.files)

    def __len__(self) -> int:
        return len(self.data.files)


def _read_scalar(data, name: str, default: Any = None) -> Any:
    if name not in data.files:
        return default
    value = np.asarray(data[name])
    return value.item() if value.shape == () else default


def _read_string_scalar(data, name: str, required: bool, errors: list[str]) -> str | None:
    if name not in data.files:
        if required:
            errors.append(f"missing field: {name}")
        return None
    value = np.asarray(data[name])
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        errors.append(f"{name} must be a string scalar, got shape={value.shape} dtype={value.dtype}")
        return None
    text = str(value.item()).strip()
    if required and not text:
        errors.append(f"{name} must not be empty")
    return text


def _require_dtype(value: np.ndarray, dtype, name: str, errors: list[str]) -> None:
    if value.dtype != np.dtype(dtype):
        errors.append(f"{name} dtype must be {np.dtype(dtype)}, got {value.dtype}")


def _finite(value: np.ndarray) -> bool:
    return np.issubdtype(value.dtype, np.number) and bool(np.isfinite(value).all())


def _image_candidates(contract: EpisodeContract, key: str) -> tuple[str, ...]:
    candidates = [contract.image_field(key), f"images_{key}", f"observation.images.{key}"]
    if key == "cam_high":
        candidates.append("image")
    elif key == "cam_wrist":
        candidates.append("wrist_image")
    return tuple(dict.fromkeys(candidates))


def _load_image(data, contract: EpisodeContract, key: str, errors: list[str]) -> tuple[np.ndarray, str]:
    for field in _image_candidates(contract, key):
        if field in data.files:
            return np.asarray(data[field]), field
    errors.append(f"missing camera {key}; expected one of {_image_candidates(contract, key)}")
    return np.empty((0, *IMAGE_HW, 3), dtype=np.uint8), contract.image_field(key)


def _percentiles(values: np.ndarray) -> tuple[float, float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return float(p50), float(p95), float(p99), float(np.max(values))


def _rotation_errors(values: np.ndarray) -> tuple[float, float, float]:
    col0, col1 = values[:, 3:6], values[:, 6:9]
    return (
        float(np.max(np.abs(np.linalg.norm(col0, axis=1) - 1.0))),
        float(np.max(np.abs(np.linalg.norm(col1, axis=1) - 1.0))),
        float(np.max(np.abs(np.sum(col0 * col1, axis=1)))),
    )


def _validate_rotation6d(values: np.ndarray, label: str, errors: list[str]) -> None:
    norm0, norm1, dot = _rotation_errors(np.asarray(values, dtype=np.float64))
    if max(norm0, norm1) > ROTATION_NORM_TOLERANCE:
        errors.append(f"{label} rotation6D column norm error exceeds 1e-3")
    if dot > ROTATION_DOT_TOLERANCE:
        errors.append(f"{label} rotation6D columns are not orthogonal")


def _validate_contract_metadata(data, contract: EpisodeContract, errors: list[str]) -> None:
    expected = {
        "schema": contract.schema,
        "arm_mode": contract.arm_mode,
        "arm_side": contract.arm_side,
        "state_dim": contract.state_dim,
        "action_dim": contract.raw_action_dim,
        "action_alignment": contract.action_alignment,
        "action_offset": contract.action_offset,
    }
    if contract.version >= CONTRACT_VERSION:
        expected.update(
            {
                "contract_version": CONTRACT_VERSION,
                "raw_action_dim": contract.raw_action_dim,
                "model_action_dim": contract.model_action_dim,
                "action_semantics": contract.action_semantics,
                "model_action_semantics": contract.model_action_semantics,
                "gripper_semantics": contract.gripper_semantics,
                "rotation_semantics": contract.rotation_semantics,
                "coordinate_frame": contract.coordinate_frame,
                "fps": contract.fps,
                "action_horizon": contract.action_horizon,
            }
        )
        required = {
            "state_names",
            "action_names",
            "model_action_names",
            "action_source",
            "source_frame",
            *expected.keys(),
        }
        for key in sorted(required):
            if key not in data.files:
                errors.append(f"missing v3 metadata: {key}")
    for key, wanted in expected.items():
        if key in data.files:
            actual = _read_scalar(data, key)
            if actual != wanted:
                errors.append(f"metadata {key}={actual!r}, expected {wanted!r}")
    for key, wanted in (
        ("camera_keys", contract.camera_keys),
        ("state_names", contract.state_names),
        ("action_names", contract.action_names),
        ("model_action_names", contract.model_action_names),
    ):
        if key in data.files:
            values = np.asarray(data[key])
            actual = tuple(str(item) for item in values.tolist()) if values.ndim == 1 else ()
            if actual != tuple(wanted):
                errors.append(f"metadata {key}={actual}, expected {tuple(wanted)}")


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
        action_dim=contract.raw_action_dim,
        model_action_dim=contract.model_action_dim,
        camera_keys=contract.camera_keys,
        action_semantics=contract.action_semantics,
        action_source=contract.action_source,
        action_alignment=contract.action_alignment,
        contract_version=contract.version,
        legacy_layout=contract.legacy_delivery_v2 or contract.legacy_joint_v2,
    )


def validate_episode(path: str | Path, target_fps: float = TARGET_FPS) -> EpisodeStats:
    path = Path(path)
    errors: list[str] = []
    with np.load(path, allow_pickle=False) as data:
        try:
            contract = infer_episode_contract(_NpzMapping(data))
        except Exception as exc:
            fallback = EpisodeContract()
            raise EpisodeValidationError(path, [f"cannot infer episode contract: {exc}"], _empty_stats(path, fallback)) from exc
        _validate_contract_metadata(data, contract, errors)
        instruction = _read_string_scalar(data, "instruction", True, errors) or ""
        task = _read_string_scalar(data, "task", False, errors)
        if task is None:
            task = _read_string_scalar(data, "task_name", False, errors)
        success_value = np.asarray(data["success"]) if "success" in data.files else np.asarray(False)
        if "success" not in data.files:
            errors.append("missing field: success")
        elif success_value.shape != () or success_value.dtype != np.bool_:
            errors.append(f"success must be bool scalar, got shape={success_value.shape} dtype={success_value.dtype}")
        success = bool(success_value.item()) if success_value.shape == () else False

        state = np.asarray(data["state"]) if "state" in data.files else np.empty((0, contract.state_dim))
        actions = np.asarray(data["actions"]) if "actions" in data.files else np.empty((0, contract.raw_action_dim))
        if "state" not in data.files:
            errors.append("missing field: state")
        if "actions" not in data.files:
            errors.append("missing field: actions")
        frame_count = int(state.shape[0]) if state.ndim == 2 else 0
        _require_dtype(state, np.float32, "state", errors)
        _require_dtype(actions, np.float32, "actions", errors)
        if state.shape != (frame_count, contract.state_dim):
            errors.append(f"state shape must be (T,{contract.state_dim}), got {state.shape}")
        if actions.shape != (frame_count, contract.raw_action_dim):
            errors.append(f"actions shape must be ({frame_count},{contract.raw_action_dim}), got {actions.shape}")
        if state.size and not _finite(state):
            errors.append("state contains NaN or Inf")
        if actions.size and not _finite(actions):
            errors.append("actions contains NaN or Inf")

        terminal_padding = bool(_read_scalar(data, "terminal_padding", True))
        real_frames = max(0, frame_count - 1) if terminal_padding else frame_count
        if real_frames < 2:
            errors.append("episode must contain at least two real frames")

        state_field = "state_timestamp" if "state_timestamp" in data.files else "timestamps"
        if contract.version >= CONTRACT_VERSION and "state_timestamp" not in data.files:
            errors.append("missing field: state_timestamp")
        state_ts = np.asarray(data[state_field]) if state_field in data.files else np.empty(0)
        _require_dtype(state_ts, np.float64, state_field, errors)
        if state_ts.shape != (frame_count,):
            errors.append(f"{state_field} shape must be ({frame_count},), got {state_ts.shape}")
        if contract.version >= CONTRACT_VERSION and "action_timestamp" not in data.files:
            errors.append("missing field: action_timestamp")
        if "action_timestamp" in data.files:
            action_ts = np.asarray(data["action_timestamp"])
            _require_dtype(action_ts, np.float64, "action_timestamp", errors)
            if action_ts.shape != (frame_count,):
                errors.append(f"action_timestamp shape must be ({frame_count},), got {action_ts.shape}")
        else:
            action_ts = state_ts.copy()
        if state_ts.shape == (frame_count,) and _finite(state_ts):
            if np.any(np.diff(state_ts) <= 0):
                errors.append("state timestamps must be strictly increasing")
            real_ts = state_ts[:real_frames]
            dt_s = np.diff(real_ts)
            if len(dt_s) and np.all(dt_s > 0):
                actual_fps = float(1.0 / np.mean(dt_s))
                duration_s = float(real_ts[-1] - real_ts[0])
                if abs(actual_fps - target_fps) / target_fps > FPS_TOLERANCE:
                    errors.append(f"actual FPS {actual_fps:.3f} is outside {target_fps:.0f} +/-5%")
            else:
                actual_fps = duration_s = 0.0
        else:
            if state_ts.size and not _finite(state_ts):
                errors.append(f"{state_field} contains NaN or Inf")
            dt_s = np.empty(0, dtype=np.float64)
            actual_fps = duration_s = 0.0
        if action_ts.shape == (frame_count,) and _finite(action_ts):
            if contract.version >= CONTRACT_VERSION and np.any(np.diff(action_ts) <= 0):
                errors.append("action timestamps must be strictly increasing")
            if state_ts.shape == action_ts.shape and real_frames:
                anchor = np.minimum(np.arange(real_frames) + contract.action_offset, frame_count - 1)
                skew = np.abs(action_ts[:real_frames] - state_ts[anchor])
                if contract.action_alignment == "same_step_command" and np.max(skew) > MAX_SYNC_S:
                    errors.append(f"state/action command sync exceeds 50 ms: max={np.max(skew)*1000:.2f} ms")
        elif action_ts.size:
            errors.append("action_timestamp contains NaN or Inf")

        images: dict[str, np.ndarray] = {}
        image_ts_by_key: dict[str, np.ndarray] = {}
        sync_by_key: dict[str, np.ndarray] = {}
        frozen_by_key: dict[str, int] = {}
        for key in contract.camera_keys:
            image, field = _load_image(data, contract, key, errors)
            images[key] = image
            _require_dtype(image, np.uint8, field, errors)
            if image.shape != (frame_count, *IMAGE_HW, 3):
                errors.append(f"{field} shape must be ({frame_count},{IMAGE_HW[0]},{IMAGE_HW[1]},3), got {image.shape}")
            ts_field = contract.timestamp_field(key)
            image_ts = np.asarray(data[ts_field]) if ts_field in data.files else np.empty(0)
            if ts_field not in data.files:
                errors.append(f"missing field: {ts_field}")
            _require_dtype(image_ts, np.float64, ts_field, errors)
            if image_ts.shape != (frame_count,):
                errors.append(f"{ts_field} shape must be ({frame_count},), got {image_ts.shape}")
            image_ts_by_key[key] = image_ts
            if image_ts.shape == (frame_count,) and _finite(image_ts) and state_ts.shape == (frame_count,):
                if contract.version >= CONTRACT_VERSION and np.any(np.diff(image_ts) <= 0):
                    errors.append(f"{ts_field} must be strictly increasing")
                sync = np.abs(image_ts[:real_frames] - state_ts[:real_frames])
                sync_by_key[key] = sync
                if len(sync) and np.max(sync) > MAX_SYNC_S:
                    errors.append(f"image/state sync exceeds 50 ms for {key}: max={np.max(sync)*1000:.2f} ms")
            else:
                sync_by_key[key] = np.empty(0, dtype=np.float64)
            frozen = 0
            if image.shape == (frame_count, *IMAGE_HW, 3) and real_frames:
                black = np.flatnonzero(np.max(image.reshape(frame_count, -1), axis=1) == 0)
                if len(black):
                    errors.append(f"{key} contains all-black frames: {black[:10].tolist()}")
                frozen = sum(np.array_equal(image[index], image[index - 1]) for index in range(1, real_frames))
                if real_frames > 1 and frozen == real_frames - 1:
                    errors.append(f"{key} is frozen for the entire non-terminal episode")
            frozen_by_key[key] = frozen
        if "cam_high" in images and images["cam_high"].shape == (frame_count, *IMAGE_HW, 3):
            for key, image in images.items():
                if key != "cam_high" and image.shape == images["cam_high"].shape and np.array_equal(image[:real_frames], images["cam_high"][:real_frames]):
                    errors.append(f"cam_high and {key} are identical; check camera mappings")

        if "joint_qpos" in data.files:
            joint_qpos = np.asarray(data["joint_qpos"])
            _require_dtype(joint_qpos, np.float32, "joint_qpos", errors)
            if joint_qpos.shape != (frame_count, contract.joint_dim):
                errors.append(f"joint_qpos shape must be ({frame_count},{contract.joint_dim}), got {joint_qpos.shape}")
            elif not _finite(joint_qpos):
                errors.append("joint_qpos contains NaN or Inf")

        action_norms = np.empty(0, dtype=np.float64)
        translation_norms = np.empty(0, dtype=np.float64)
        rotation_norms = np.empty(0, dtype=np.float64)
        no_op = np.empty(0, dtype=np.bool_)
        gripper_parts: list[np.ndarray] = []
        if state.shape == (frame_count, contract.state_dim) and actions.shape == (frame_count, contract.raw_action_dim) and _finite(state) and _finite(actions):
            for index in contract.gripper_state_indices:
                values = state[:real_frames, index].astype(np.float32)
                if contract.legacy_delivery_v2:
                    values = 1.0 - values
                elif contract.legacy_joint_v2:
                    values = np.clip(values / GRIPPER_MAX_M, 0.0, 1.0)
                gripper_parts.append(values)
            gripper_values = np.concatenate(gripper_parts) if gripper_parts else np.empty(0, dtype=np.float32)
            for arm, values in enumerate(gripper_parts):
                if np.min(values) < -GRIPPER_TOLERANCE or np.max(values) > 1 + GRIPPER_TOLERANCE:
                    errors.append(f"arm {arm} normalized gripper state is outside [0,1]")

            translations: list[np.ndarray] = []
            rotations: list[np.ndarray] = []
            no_ops: list[np.ndarray] = []
            if contract.schema == DELIVERY_SCHEMA:
                for arm in range(contract.arm_count):
                    ss = arm * 10
                    state_arm = state[:, ss : ss + 10]
                    _validate_rotation6d(state_arm, f"arm {arm} state", errors)
                    if contract.legacy_delivery_v2:
                        aa = arm * 7
                        action_arm = actions[:, aa : aa + 7]
                        if terminal_padding and frame_count > 1:
                            expected_xyz = state_arm[1:, :3] - state_arm[:-1, :3]
                            xyz_error = float(np.max(np.abs(action_arm[:-1, :3] - expected_xyz)))
                            if xyz_error > TRANSLATION_TOLERANCE_M:
                                errors.append(f"arm {arm} legacy translation action reconstruction error: {xyz_error:.3e}")
                            expected_rot = []
                            for current, nxt in zip(state_arm[:-1], state_arm[1:]):
                                expected_rot.append(Rotation.from_matrix(rotation6d_to_matrix(nxt[3:9]) @ rotation6d_to_matrix(current[3:9]).T).as_rotvec())
                            rot_error = float(np.max(np.abs(action_arm[:-1, 3:6] - np.asarray(expected_rot))))
                            if rot_error > ROTATION_ACTION_TOLERANCE_RAD:
                                errors.append(f"arm {arm} legacy rotation action reconstruction error: {rot_error:.3e}")
                            grip_error = float(np.max(np.abs(action_arm[:-1, 6] - state_arm[1:, 9])))
                            if grip_error > GRIPPER_TOLERANCE:
                                errors.append(f"arm {arm} legacy gripper action reconstruction error: {grip_error:.3e}")
                        translation = np.linalg.norm(action_arm[:real_frames, :3], axis=1)
                        rotation = np.linalg.norm(action_arm[:real_frames, 3:6], axis=1)
                        grip_change = np.abs((1.0 - action_arm[:real_frames, 6]) - gripper_parts[arm])
                    else:
                        aa = arm * 10
                        action_arm = actions[:, aa : aa + 10]
                        _validate_rotation6d(action_arm, f"arm {arm} raw action", errors)
                        if np.min(action_arm[:, 9]) < -GRIPPER_TOLERANCE or np.max(action_arm[:, 9]) > 1 + GRIPPER_TOLERANCE:
                            errors.append(f"arm {arm} action gripper opening fraction is outside [0,1]")
                        if terminal_padding and contract.action_offset == 1:
                            pose_error = float(np.max(np.abs(action_arm[:-1, :9] - state_arm[1:, :9])))
                            if pose_error > TRANSLATION_TOLERANCE_M:
                                errors.append(f"arm {arm} fallback absolute EEF pose does not match next measured state: {pose_error:.3e}")
                            if "gripper_command_present" in data.files:
                                present = np.asarray(data["gripper_command_present"])[:-1, arm].astype(bool)
                                missing = ~present
                                if np.any(missing):
                                    grip_error = float(np.max(np.abs(action_arm[:-1, 9][missing] - state_arm[1:, 9][missing])))
                                    if grip_error > GRIPPER_TOLERANCE:
                                        errors.append(f"arm {arm} fallback gripper does not match next measured state: {grip_error:.3e}")
                            elif not np.allclose(action_arm[:-1, 9], state_arm[1:, 9], atol=GRIPPER_TOLERANCE, rtol=0):
                                errors.append(f"arm {arm} fallback gripper does not match next measured state")
                        translation = np.linalg.norm(action_arm[:real_frames, :3] - state_arm[:real_frames, :3], axis=1)
                        rotation_values = []
                        for current, target in zip(state_arm[:real_frames], action_arm[:real_frames]):
                            rotation_values.append(np.linalg.norm(Rotation.from_matrix(rotation6d_to_matrix(target[3:9]) @ rotation6d_to_matrix(current[3:9]).T).as_rotvec()))
                        rotation = np.asarray(rotation_values)
                        grip_change = np.abs(action_arm[:real_frames, 9] - state_arm[:real_frames, 9])
                    translations.append(translation)
                    rotations.append(rotation)
                    no_ops.append((translation <= TRANSLATION_TOLERANCE_M) & (rotation <= ROTATION_ACTION_TOLERANCE_RAD) & (grip_change <= GRIPPER_TOLERANCE))
                translation_norms = np.linalg.norm(np.stack(translations, axis=1), axis=1)
                rotation_norms = np.linalg.norm(np.stack(rotations, axis=1), axis=1)
                action_norms = np.sqrt(translation_norms**2 + rotation_norms**2)
                no_op = np.all(np.stack(no_ops, axis=1), axis=1)
            else:
                current = state[:real_frames]
                target = actions[:real_frames]
                differences: list[np.ndarray] = []
                grip_changes: list[np.ndarray] = []
                for arm in range(contract.arm_count):
                    offset = arm * 7
                    differences.append(target[:, offset : offset + 6] - current[:, offset : offset + 6])
                    if contract.legacy_joint_v2:
                        grip_changes.append(np.abs(target[:, offset + 6] - current[:, offset + 6]) / GRIPPER_MAX_M)
                        if np.min(current[:, offset + 6]) < -GRIPPER_TOLERANCE or np.max(current[:, offset + 6]) > GRIPPER_MAX_M + GRIPPER_TOLERANCE:
                            errors.append(f"arm {arm} legacy joint gripper opening_m is outside [0,{GRIPPER_MAX_M}]")
                    else:
                        grip_changes.append(np.abs(target[:, offset + 6] - current[:, offset + 6]))
                        if np.min(current[:, offset + 6]) < -GRIPPER_TOLERANCE or np.max(current[:, offset + 6]) > 1 + GRIPPER_TOLERANCE:
                            errors.append(f"arm {arm} joint gripper opening fraction is outside [0,1]")
                joints = np.concatenate(differences, axis=1)
                action_norms = np.linalg.norm(joints, axis=1)
                translation_norms = action_norms.copy()
                rotation_norms = np.zeros_like(action_norms)
                no_op = (action_norms <= JOINT_ACTION_TOLERANCE) & np.all(np.stack(grip_changes, axis=1) <= JOINT_ACTION_TOLERANCE, axis=1)
                if terminal_padding and contract.action_offset == 1 and not np.allclose(actions[:-1], state[1:], atol=JOINT_ACTION_TOLERANCE, rtol=0):
                    errors.append(f"joint fallback actions do not match next measured observation: max error={np.max(np.abs(actions[:-1]-state[1:])):.3e}")
        else:
            gripper_values = np.empty(0, dtype=np.float32)

        gripper_transition_count = sum(int(np.count_nonzero(np.abs(np.diff(values)) > GRIPPER_TRANSITION_THRESHOLD)) for values in gripper_parts)
        wrist_sync = [value for key, value in sync_by_key.items() if key != "cam_high" and len(value)]
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
            sync_high_s=sync_by_key.get("cam_high", np.empty(0, dtype=np.float64)),
            sync_wrist_s=np.concatenate(wrist_sync) if wrist_sync else np.empty(0, dtype=np.float64),
            action_norms=action_norms,
            translation_norms=translation_norms,
            rotation_norms=rotation_norms,
            no_op_count=int(np.count_nonzero(no_op)),
            no_op_total=len(no_op),
            gripper_values=gripper_values,
            gripper_transition_count=gripper_transition_count,
            frozen_high_count=frozen_by_key.get("cam_high", 0),
            frozen_wrist_count=sum(value for key, value in frozen_by_key.items() if key != "cam_high"),
            schema=contract.schema,
            arm_mode=contract.arm_mode,
            arm_side=contract.arm_side,
            state_dim=contract.state_dim,
            action_dim=contract.raw_action_dim,
            model_action_dim=contract.model_action_dim,
            camera_keys=contract.camera_keys,
            action_semantics=contract.action_semantics,
            action_source=contract.action_source,
            action_alignment=contract.action_alignment,
            contract_version=contract.version,
            legacy_layout=contract.legacy_delivery_v2 or contract.legacy_joint_v2,
        )

    if success and stats.no_op_total and stats.no_op_count == stats.no_op_total:
        errors.append("successful episode contains no robot motion or gripper change (100% no-op); check Piper CAN feedback before recording")
    if errors:
        raise EpisodeValidationError(path, errors, stats=stats)
    return stats


def validate_gripper_coverage(stats: list[EpisodeStats]) -> None:
    if not stats:
        raise ValueError("no successful episodes are available for export")
    gripper = np.concatenate([item.gripper_values for item in stats])
    missing = []
    if not np.any(gripper <= GRIPPER_CLOSED_THRESHOLD):
        missing.append("closed states (opening_fraction <= 0.1)")
    if not np.any(gripper >= GRIPPER_OPEN_THRESHOLD):
        missing.append("open states (opening_fraction >= 0.9)")
    if sum(item.gripper_transition_count for item in stats) == 0:
        missing.append("gripper transitions (step change > 0.01)")
    if missing:
        raise ValueError("successful dataset lacks " + ", ".join(missing))


def validate_instruction_consistency(stats: list[EpisodeStats]) -> None:
    contracts = {
        (item.schema, item.arm_mode, item.arm_side, item.state_dim, item.action_dim, item.model_action_dim, item.camera_keys, item.action_semantics, item.action_alignment, item.contract_version)
        for item in stats
    }
    if len(contracts) > 1:
        raise ValueError(f"episodes mix incompatible contracts: {sorted(contracts)!r}")
    instructions_by_task: dict[str, set[str]] = {}
    for item in stats:
        if item.task:
            instructions_by_task.setdefault(item.task, set()).add(item.instruction)
    inconsistent = {task: sorted(values) for task, values in instructions_by_task.items() if len(values) > 1}
    if inconsistent:
        raise ValueError(f"internal task IDs map to inconsistent instructions: {inconsistent}")


def format_episode_report(stats: EpisodeStats) -> str:
    no_op_ratio = stats.no_op_count / max(1, stats.no_op_total)
    p50, p95, p99, maximum = _percentiles(stats.action_norms)
    gripper_min = float(np.min(stats.gripper_values)) if len(stats.gripper_values) else 0.0
    gripper_max = float(np.max(stats.gripper_values)) if len(stats.gripper_values) else 0.0
    layout = "legacy-v2" if stats.legacy_layout else "v3"
    return (
        f"PASS {stats.path}: {layout} schema={stats.schema} arm={stats.arm_mode}/{stats.arm_side} "
        f"state={stats.state_dim} raw_action={stats.action_dim} model_action={stats.model_action_dim} "
        f"frames={stats.frames} real={stats.real_frames} fps={stats.actual_fps:.3f}\n"
        f"  action={stats.action_semantics} source={stats.action_source} alignment={stats.action_alignment}\n"
        f"  action_norm p50/p95/p99/max={p50:.6f}/{p95:.6f}/{p99:.6f}/{maximum:.6f}, no_op={no_op_ratio:.1%}\n"
        f"  gripper opening_fraction min/max={gripper_min:.3f}/{gripper_max:.3f}, transitions={stats.gripper_transition_count}"
    )


def format_dataset_report(stats: list[EpisodeStats]) -> str:
    if not stats:
        return "Dataset PASS: episodes=0 frames=0"
    action_norms = np.concatenate([item.action_norms for item in stats])
    p50, p95, p99, maximum = _percentiles(action_norms)
    first = stats[0]
    return (
        f"Dataset PASS: schema={first.schema} arm={first.arm_mode}/{first.arm_side} episodes={len(stats)} "
        f"frames={sum(item.frames for item in stats)}\n"
        f"  action_norm p50/p95/p99/max={p50:.6f}/{p95:.6f}/{p99:.6f}/{maximum:.6f}"
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
    failures: list[str] = []
    for path in paths:
        try:
            stats = validate_episode(path, target_fps=args.target_fps)
            print(format_episode_report(stats))
            if stats.success:
                successful.append(stats)
        except EpisodeValidationError as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("\n\n".join(failures))
    validate_gripper_coverage(successful)
    validate_instruction_consistency(successful)
    print(format_dataset_report(successful))


if __name__ == "__main__":
    main()
