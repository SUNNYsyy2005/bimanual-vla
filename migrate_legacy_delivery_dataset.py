#!/usr/bin/env python3
"""Migrate legacy Piper delivery LeRobot datasets to the canonical v3 contract.

Source layout:
    10D state + 7D one-step EEF delta per arm, closed-fraction gripper.

Target layout:
    10D state + 10D absolute EEF target per arm, opening-fraction gripper.

The migration never writes into the source tree. Legacy actions are decoded
against the current measured state; they remain explicitly labelled as
``next_measured_eef_fallback`` and are not represented as actual robot commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from pi0_dataset import (
    CANONICAL_CONTRACT_FORMAT,
    DELIVERY_ABSOLUTE_ACTION_FORMAT,
    LEGACY_V2,
    classify_contract_dimensions,
)
from piper_action_conventions import matrix_to_rotation6d, rotation6d_to_matrix
from piper_data_contract import (
    CONTRACT_VERSION,
    DEFAULT_ACTION_HORIZON,
    DEFAULT_COORDINATE_FRAME,
    DEFAULT_FPS,
    DELIVERY_SCHEMA,
    LEGACY_DELIVERY_ACTION_SEMANTICS,
    LEGACY_GRIPPER_CLOSED_SEMANTICS,
    EpisodeContract,
)


MIGRATION_VERSION = 1
MIGRATION_NOTE = (
    "Decoded each legacy one-step EEF delta against that frame's measured state; "
    "converted closed fraction to opening fraction; labels remain next-measured "
    "fallback targets and are not actual commanded actions."
)


@dataclass(frozen=True)
class LegacyLayout:
    state_key: str
    action_key: str
    arm_count: int
    arm_mode: str
    arm_side: str
    camera_field_map: dict[str, str]
    camera_keys: tuple[str, ...]
    source_contract: dict[str, Any]


class MigrationError(ValueError):
    pass


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError:
        return []


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
        return dst
    except OSError:
        return shutil.copy2(src, dst)


def _feature_dim(info: dict[str, Any], key: str) -> int:
    feature = info.get("features", {}).get(key)
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        raise MigrationError(f"feature {key} must have shape [dim]")
    return int(shape[0])


def _camera_key_for_feature(feature_key: str, *, arm_side: str) -> str:
    key = feature_key.removeprefix("observation.images.").removeprefix("images_")
    aliases = {
        "image": "cam_high",
        "cam_high": "cam_high",
        "wrist_image": f"cam_{arm_side}_wrist",
        "cam_wrist": f"cam_{arm_side}_wrist",
        "left_wrist_image": "cam_left_wrist",
        "right_wrist_image": "cam_right_wrist",
        "cam_left_wrist": "cam_left_wrist",
        "cam_right_wrist": "cam_right_wrist",
    }
    if key not in aliases:
        raise MigrationError(f"unsupported legacy camera feature {feature_key!r}")
    return aliases[key]


def detect_legacy_layout(info: dict[str, Any]) -> LegacyLayout:
    features = info.get("features")
    if not isinstance(features, dict):
        raise MigrationError("meta/info.json has no features object")
    if "state" in features and "actions" in features:
        state_key, action_key = "state", "actions"
    elif "observation.state" in features and "action" in features:
        state_key, action_key = "observation.state", "action"
    else:
        raise MigrationError("source is missing state/actions or observation.state/action")

    state_dim = _feature_dim(info, state_key)
    action_dim = _feature_dim(info, action_key)
    try:
        dimensions = classify_contract_dimensions(
            state_dim,
            action_dim,
            schema=info.get("schema"),
            legacy_format=info.get("legacy_format") or info.get("contract_format"),
        )
    except ValueError as exc:
        raise MigrationError(str(exc)) from exc
    if dimensions["schema"] != DELIVERY_SCHEMA or not dimensions["legacy"]:
        raise MigrationError(
            "source must be legacy delivery 10D+7D or 20D+14D step-delta data"
        )
    if int(info.get("fps", 0)) != DEFAULT_FPS:
        raise MigrationError(f"source fps must be {DEFAULT_FPS}")
    if int(info.get("contract_version", 2)) >= CONTRACT_VERSION:
        raise MigrationError("legacy delivery source cannot declare contract_version>=3")
    semantics = str(info.get("action_semantics") or LEGACY_DELIVERY_ACTION_SEMANTICS)
    if "delta" not in semantics.lower():
        raise MigrationError("source action_semantics does not describe one-step deltas")
    gripper_semantics = str(
        info.get("gripper_semantics") or LEGACY_GRIPPER_CLOSED_SEMANTICS
    )
    if gripper_semantics != LEGACY_GRIPPER_CLOSED_SEMANTICS:
        raise MigrationError("source must use legacy closed-fraction gripper semantics")

    arm_mode = dimensions["arm_mode"]
    arm_side = str(
        info.get("arm_side")
        or ("both" if arm_mode == "bimanual" else "right")
    )
    if arm_mode == "bimanual":
        arm_side = "both"
    elif arm_side not in {"left", "right"}:
        raise MigrationError(f"invalid single-arm arm_side={arm_side!r}")

    camera_features = [
        key
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") in {"image", "video"}
    ]
    expected_cameras = 3 if arm_mode == "bimanual" else 2
    if len(camera_features) != expected_cameras:
        raise MigrationError(
            f"source has {len(camera_features)} camera features, expected {expected_cameras}"
        )
    mapping: dict[str, str] = {}
    for old_key in camera_features:
        camera_key = _camera_key_for_feature(
            old_key, arm_side="right" if arm_side == "both" else arm_side
        )
        mapping[old_key] = f"observation.images.{camera_key}"
    camera_keys = tuple(
        ["cam_high"]
        + (["cam_left_wrist", "cam_right_wrist"] if arm_mode == "bimanual" else [f"cam_{arm_side}_wrist"])
    )
    if set(mapping.values()) != {f"observation.images.{key}" for key in camera_keys}:
        raise MigrationError(
            f"camera features do not map to required canonical cameras {camera_keys}"
        )

    source_contract = {
        "contract_version": int(info.get("contract_version", 2)),
        "schema": "delivery",
        "arm_mode": arm_mode,
        "arm_side": arm_side,
        "state_dim": state_dim,
        "raw_action_dim": action_dim,
        "model_action_dim": dimensions["model_action_dim"],
        "state_names": features[state_key].get("names"),
        "action_names": features[action_key].get("names"),
        "camera_fields": camera_features,
        "fps": int(info["fps"]),
        "action_semantics": semantics,
        "action_source": info.get("action_source", "next_measured_eef"),
        "action_alignment": info.get("action_alignment", "next_observation"),
        "action_offset": int(info.get("action_offset", 1)),
        "gripper_semantics": gripper_semantics,
        "rotation_semantics": info.get(
            "rotation_semantics", "state_rotation6d_action_left_rotvec_base_frame"
        ),
        "coordinate_frame": info.get("coordinate_frame", DEFAULT_COORDINATE_FRAME),
        "legacy_format": LEGACY_V2,
    }
    return LegacyLayout(
        state_key=state_key,
        action_key=action_key,
        arm_count=dimensions["arm_count"],
        arm_mode=arm_mode,
        arm_side=arm_side,
        camera_field_map=mapping,
        camera_keys=camera_keys,
        source_contract=source_contract,
    )


def decode_legacy_delivery_episode(
    states: np.ndarray,
    legacy_actions: np.ndarray,
    *,
    arm_count: int,
    verify_next_measured: bool = True,
    atol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert closed-fraction states and one-step deltas to canonical arrays."""
    states = np.asarray(states, dtype=np.float32)
    legacy_actions = np.asarray(legacy_actions, dtype=np.float32)
    expected_state_dim = 10 * int(arm_count)
    expected_action_dim = 7 * int(arm_count)
    if states.ndim != 2 or states.shape[1] != expected_state_dim or len(states) == 0:
        raise MigrationError(
            f"state must have shape (T,{expected_state_dim}), got {states.shape}"
        )
    if legacy_actions.shape != (len(states), expected_action_dim):
        raise MigrationError(
            f"legacy action must have shape ({len(states)},{expected_action_dim}), "
            f"got {legacy_actions.shape}"
        )
    if not np.isfinite(states).all() or not np.isfinite(legacy_actions).all():
        raise MigrationError("state/action contains NaN or Inf")

    canonical_states = states.copy()
    absolute_targets = np.empty((len(states), expected_state_dim), dtype=np.float32)
    for arm in range(arm_count):
        ss, aa = arm * 10, arm * 7
        state_closed = states[:, ss + 9]
        action_closed = legacy_actions[:, aa + 6]
        if (
            np.min(state_closed) < -atol
            or np.max(state_closed) > 1.0 + atol
            or np.min(action_closed) < -atol
            or np.max(action_closed) > 1.0 + atol
        ):
            raise MigrationError("legacy closed-fraction gripper values are outside [0,1]")
        canonical_states[:, ss + 9] = 1.0 - state_closed
        for frame in range(len(states)):
            current = states[frame, ss : ss + 10]
            action = legacy_actions[frame, aa : aa + 7]
            current_rotation = rotation6d_to_matrix(current[3:9])
            target_rotation = Rotation.from_rotvec(action[3:6]).as_matrix() @ current_rotation
            absolute_targets[frame, ss : ss + 3] = current[:3] + action[:3]
            absolute_targets[frame, ss + 3 : ss + 9] = matrix_to_rotation6d(
                target_rotation
            ).astype(np.float32)
            absolute_targets[frame, ss + 9] = 1.0 - action[6]

    # The terminal row is an explicit hold regardless of legacy padding noise.
    absolute_targets[-1] = canonical_states[-1]
    if verify_next_measured and len(states) > 1 and not np.allclose(
        absolute_targets[:-1], canonical_states[1:], atol=atol, rtol=atol
    ):
        difference = float(
            np.max(np.abs(absolute_targets[:-1] - canonical_states[1:]))
        )
        raise MigrationError(
            "legacy actions are not next-measured one-step deltas; refusing to "
            f"label migrated targets as next_measured_eef_fallback (max diff={difference:.6g})"
        )
    return canonical_states, absolute_targets


def _replace_or_append(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index >= 0:
        return table.set_column(index, pa.field(name, array.type), array)
    return table.append_column(name, array)


def _huggingface_value(dtype: pa.DataType) -> dict[str, str]:
    if pa.types.is_boolean(dtype):
        name = "bool"
    elif pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_string(dtype) or pa.types.is_binary(dtype):
        name = str(dtype)
    else:
        raise MigrationError(f"unsupported HuggingFace scalar dtype: {dtype}")
    return {"dtype": name, "_type": "Value"}


def canonical_huggingface_schema_metadata(
    table: pa.Table,
    *,
    image_columns: set[str] | frozenset[str],
) -> dict[bytes, bytes]:
    """Build datasets-compatible Arrow metadata after canonical column rewrites."""
    features: dict[str, Any] = {}
    for field in table.schema:
        if field.name in image_columns:
            features[field.name] = {"_type": "Image"}
        elif pa.types.is_fixed_size_list(field.type):
            features[field.name] = {
                "feature": _huggingface_value(field.type.value_type),
                "length": field.type.list_size,
                "_type": "Sequence",
            }
        elif pa.types.is_list(field.type):
            features[field.name] = {
                "feature": _huggingface_value(field.type.value_type),
                "length": -1,
                "_type": "Sequence",
            }
        else:
            features[field.name] = _huggingface_value(field.type)
    metadata = dict(table.schema.metadata or {})
    metadata[b"huggingface"] = json.dumps(
        {"info": {"features": features}}, separators=(",", ":")
    ).encode("utf-8")
    return metadata


def _numeric_values(table: pa.Table, name: str) -> np.ndarray | None:
    field_type = table.schema.field(name).type
    try:
        if pa.types.is_list(field_type) or pa.types.is_fixed_size_list(field_type):
            values = np.asarray(table[name].to_pylist(), dtype=np.float64)
        elif (
            pa.types.is_integer(field_type)
            or pa.types.is_floating(field_type)
            or pa.types.is_boolean(field_type)
        ):
            values = np.asarray(
                table[name].to_numpy(zero_copy_only=False), dtype=np.float64
            )
        else:
            return None
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        return None
    return values


def _stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [len(values)],
    }


def _is_image_timestamp_feature(name: str) -> bool:
    """Return whether a scalar timestamp key trips LeRobot's image-stat check.

    LeRobot v2.1 currently classifies episode stats as image stats when the
    feature name merely contains ``image``.  A numeric key such as
    ``image_timestamp.cam_high`` therefore gets incorrectly required to have
    shape ``(3, 1, 1)``.  Timestamp columns remain in parquet/info metadata,
    but must be omitted from ``episodes_stats.jsonl`` until that upstream
    check uses the declared feature dtype instead of the key name.
    """
    return name.startswith(("image_timestamp.", "image_timestamps.", "image_timestamps_"))


def _episode_stats(
    table: pa.Table,
    previous: dict[str, Any],
    column_map: dict[str, str],
) -> dict[str, Any]:
    previous_stats = previous.get("stats", {}) if isinstance(previous, dict) else {}
    result = {
        mapped: value
        for key, value in previous_stats.items()
        if (mapped := column_map.get(key, key)) in table.column_names
        and not _is_image_timestamp_feature(mapped)
    }
    for name in table.column_names:
        if _is_image_timestamp_feature(name):
            continue
        values = _numeric_values(table, name)
        if values is not None:
            result[name] = _stats(values)
    return result


def _source_timestamp(
    table: pa.Table,
    candidates: tuple[str, ...],
    fallback: np.ndarray,
) -> np.ndarray:
    for name in candidates:
        if name in table.column_names:
            values = np.asarray(
                table[name].to_numpy(zero_copy_only=False), dtype=np.float64
            )
            if values.shape == fallback.shape and np.isfinite(values).all():
                return values
    return fallback.copy()


def _canonical_metadata(
    source: Path,
    source_info: dict[str, Any],
    layout: LegacyLayout,
) -> tuple[dict[str, Any], EpisodeContract]:
    contract = EpisodeContract(
        schema=DELIVERY_SCHEMA,
        arm_mode=layout.arm_mode,
        arm_side=layout.arm_side,
        camera_keys=layout.camera_keys,
        action_source="next_measured_eef_fallback",
        action_alignment="next_observation",
        action_offset=1,
        fps=DEFAULT_FPS,
        action_horizon=DEFAULT_ACTION_HORIZON,
        coordinate_frame=DEFAULT_COORDINATE_FRAME,
        version=CONTRACT_VERSION,
    )
    migrated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "schema": contract.schema,
        "arm_mode": contract.arm_mode,
        "arm_side": contract.arm_side,
        "state_dim": contract.state_dim,
        "action_dim": contract.raw_action_dim,
        "raw_action_dim": contract.raw_action_dim,
        "model_action_dim": contract.model_action_dim,
        "state_names": list(contract.state_names),
        "action_names": list(contract.action_names),
        "model_action_names": list(contract.model_action_names),
        "camera_keys": list(contract.camera_keys),
        "action_semantics": contract.action_semantics,
        "model_action_semantics": contract.model_action_semantics,
        "action_source": "next_measured_eef_fallback",
        "action_alignment": "next_observation",
        "action_offset": 1,
        "action_horizon": DEFAULT_ACTION_HORIZON,
        "gripper_semantics": contract.gripper_semantics,
        "rotation_semantics": contract.rotation_semantics,
        "coordinate_frame": contract.coordinate_frame,
        "source_frame": source_info.get("source_frame", ""),
        "contract_format": CANONICAL_CONTRACT_FORMAT,
        "legacy": False,
        "legacy_format": None,
        "legacy_delivery_v2": False,
        "delivery_action_format": DELIVERY_ABSOLUTE_ACTION_FORMAT,
        "fps": DEFAULT_FPS,
        "source_dataset": source.name,
        "source_dataset_path": str(source.resolve()),
        "source_contract": layout.source_contract,
        "migration_version": MIGRATION_VERSION,
        "migration_note": MIGRATION_NOTE,
        "migrated_at": migrated_at,
        "timestamp_provenance": (
            "state/image timestamps preserve source columns when available, otherwise "
            "use LeRobot frame_index/fps; action timestamp uses next-observation time"
        ),
    }
    return metadata, contract


def _rewrite_features(
    source_info: dict[str, Any],
    layout: LegacyLayout,
    contract: EpisodeContract,
) -> dict[str, Any]:
    source_features = source_info["features"]
    column_map = {
        layout.state_key: "observation.state",
        layout.action_key: "action",
        **layout.camera_field_map,
    }
    features: dict[str, Any] = {}
    for key, value in source_features.items():
        new_key = column_map.get(key, key)
        features[new_key] = dict(value) if isinstance(value, dict) else value
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [contract.state_dim],
        "names": list(contract.state_names),
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [contract.raw_action_dim],
        "names": list(contract.action_names),
    }
    for name in ["state_timestamp", "action_timestamp"]:
        features[name] = {"dtype": "float64", "shape": [1], "names": None}
    for camera_key in contract.camera_keys:
        features[f"image_timestamp.{camera_key}"] = {
            "dtype": "float64",
            "shape": [1],
            "names": None,
        }
    return features


def _relocate_videos(
    candidate: Path,
    source_info: dict[str, Any],
    layout: LegacyLayout,
    episode_indexes: list[int],
) -> None:
    template = source_info.get("video_path")
    if not isinstance(template, str) or not template:
        return
    chunk_size = int(source_info.get("chunks_size", 1000))
    for old_key, new_key in layout.camera_field_map.items():
        feature = source_info["features"].get(old_key, {})
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            continue
        for episode_index in episode_indexes:
            values = {
                "episode_chunk": episode_index // chunk_size,
                "episode_index": episode_index,
            }
            old_path = candidate / template.format(video_key=old_key, **values)
            new_path = candidate / template.format(video_key=new_key, **values)
            if old_path == new_path or not old_path.exists():
                continue
            if new_path.exists():
                raise MigrationError(f"video destination already exists: {new_path}")
            new_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_path, new_path)


def _rewrite_candidate(
    source: Path,
    candidate: Path,
    source_info: dict[str, Any],
    layout: LegacyLayout,
) -> dict[str, Any]:
    metadata, contract = _canonical_metadata(source, source_info, layout)
    parquets = sorted((candidate / "data").glob("chunk-*/episode_*.parquet"))
    if not parquets:
        raise MigrationError("source dataset contains no episode parquet files")
    previous_stats = {
        int(row["episode_index"]): row
        for row in _read_jsonl(candidate / "meta" / "episodes_stats.jsonl")
        if "episode_index" in row
    }
    column_map = {
        layout.state_key: "observation.state",
        layout.action_key: "action",
        **layout.camera_field_map,
    }
    stats_rows: list[dict[str, Any]] = []
    episode_lengths: dict[int, int] = {}
    total_frames = 0
    episode_indexes: list[int] = []

    for path in parquets:
        table = pq.read_table(path)
        missing = [
            key
            for key in (layout.state_key, layout.action_key, "timestamp", "episode_index")
            if key not in table.column_names
        ]
        if missing:
            raise MigrationError(f"{path}: missing columns {missing}")
        states = np.asarray(table[layout.state_key].to_pylist(), dtype=np.float32)
        legacy_actions = np.asarray(
            table[layout.action_key].to_pylist(), dtype=np.float32
        )
        canonical_states, absolute_targets = decode_legacy_delivery_episode(
            states,
            legacy_actions,
            arm_count=layout.arm_count,
            verify_next_measured=True,
        )
        renamed = [column_map.get(name, name) for name in table.column_names]
        if len(set(renamed)) != len(renamed):
            raise MigrationError(f"{path}: canonical column rename would collide")
        table = table.rename_columns(renamed)
        table = _replace_or_append(
            table,
            "observation.state",
            pa.array(
                canonical_states.tolist(),
                type=pa.list_(pa.float32(), canonical_states.shape[1]),
            ),
        )
        table = _replace_or_append(
            table,
            "action",
            pa.array(
                absolute_targets.tolist(),
                type=pa.list_(pa.float32(), absolute_targets.shape[1]),
            ),
        )

        canonical_timestamp = np.asarray(
            table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64
        )
        if canonical_timestamp.shape != (table.num_rows,) or not np.isfinite(
            canonical_timestamp
        ).all():
            raise MigrationError(f"{path}: invalid timestamp column")
        expected_timestamp = np.arange(table.num_rows, dtype=np.float64) / DEFAULT_FPS
        if not np.allclose(canonical_timestamp, expected_timestamp, atol=1e-5):
            raise MigrationError(f"{path}: timestamp is not frame_index/fps")
        state_timestamp = _source_timestamp(
            table,
            ("state_timestamp", "state_timestamps"),
            canonical_timestamp,
        )
        action_timestamp = _source_timestamp(
            table,
            ("action_timestamp", "action_timestamps"),
            np.concatenate(
                (
                    state_timestamp[1:],
                    [state_timestamp[-1] + 1.0 / DEFAULT_FPS],
                )
            ),
        )
        table = _replace_or_append(
            table,
            "state_timestamp",
            pa.array(state_timestamp, type=pa.float64()),
        )
        table = _replace_or_append(
            table,
            "action_timestamp",
            pa.array(action_timestamp, type=pa.float64()),
        )
        for camera_key in contract.camera_keys:
            fallback = state_timestamp
            image_timestamp = _source_timestamp(
                table,
                (
                    f"image_timestamp.{camera_key}",
                    f"image_timestamps.{camera_key}",
                    f"image_timestamps_{camera_key}",
                ),
                fallback,
            )
            table = _replace_or_append(
                table,
                f"image_timestamp.{camera_key}",
                pa.array(image_timestamp, type=pa.float64()),
            )

        episode_values = np.asarray(
            table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64
        )
        if len(set(episode_values.tolist())) != 1:
            raise MigrationError(f"{path}: episode_index varies within one parquet")
        episode_index = int(episode_values[0])
        episode_indexes.append(episode_index)
        episode_lengths[episode_index] = table.num_rows
        total_frames += table.num_rows
        stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": _episode_stats(
                    table,
                    previous_stats.get(episode_index, {}),
                    column_map,
                ),
            }
        )

        # Replace stale HuggingFace schema metadata so image structs are
        # decoded as PIL images instead of reaching torch as raw dictionaries.
        image_columns = {
            f"observation.images.{key}"
            for key in contract.camera_keys
            if f"observation.images.{key}" in table.column_names
        }
        table = table.replace_schema_metadata(
            canonical_huggingface_schema_metadata(
                table,
                image_columns=image_columns,
            )
        )
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        pq.write_table(table, temporary)
        os.replace(temporary, path)

    if sorted(episode_indexes) != list(range(len(parquets))):
        raise MigrationError("episode indexes must be contiguous from zero")
    _relocate_videos(candidate, source_info, layout, episode_indexes)

    info = dict(source_info)
    info.update(metadata)
    info["features"] = _rewrite_features(source_info, layout, contract)
    info["timestamp_fields"] = {
        "canonical": "timestamp",
        "state": "state_timestamp",
        "action": "action_timestamp",
        "images": {
            key: f"image_timestamp.{key}" for key in contract.camera_keys
        },
    }
    info["total_episodes"] = len(parquets)
    info["total_frames"] = total_frames
    info["total_chunks"] = math.ceil(
        len(parquets) / int(info.get("chunks_size", 1000))
    )
    _atomic_json(candidate / "meta" / "info.json", info)

    episodes = _read_jsonl(candidate / "meta" / "episodes.jsonl")
    episode_rows: list[dict[str, Any]] = []
    for row in episodes:
        index = int(row["episode_index"])
        updated = dict(row)
        updated["length"] = episode_lengths[index]
        updated.update(
            {
                key: value
                for key, value in metadata.items()
                if key
                in {
                    "contract_version",
                    "schema",
                    "arm_mode",
                    "arm_side",
                    "state_dim",
                    "action_dim",
                    "raw_action_dim",
                    "model_action_dim",
                    "state_names",
                    "action_names",
                    "model_action_names",
                    "camera_keys",
                    "action_semantics",
                    "model_action_semantics",
                    "action_source",
                    "action_alignment",
                    "action_offset",
                    "action_horizon",
                    "gripper_semantics",
                    "rotation_semantics",
                    "coordinate_frame",
                    "contract_format",
                    "legacy",
                    "legacy_format",
                    "legacy_delivery_v2",
                    "delivery_action_format",
                    "source_dataset",
                    "source_contract",
                    "migration_version",
                    "migration_note",
                }
            }
        )
        episode_rows.append(updated)
    if len(episode_rows) != len(parquets):
        raise MigrationError("episodes.jsonl count does not match parquet count")
    _atomic_jsonl(candidate / "meta" / "episodes.jsonl", episode_rows)
    _atomic_jsonl(
        candidate / "meta" / "episodes_stats.jsonl",
        sorted(stats_rows, key=lambda row: int(row["episode_index"])),
    )

    policy_contract = {
        "version": CONTRACT_VERSION,
        "robot_type": info.get("robot_type", contract.robot_type),
        **{
            key: value
            for key, value in metadata.items()
            if key not in {"migrated_at", "timestamp_provenance"}
        },
    }
    _atomic_json(candidate / "meta" / "policy_contract.json", policy_contract)
    (candidate / "meta" / "openpi_norm_stats.json").unlink(missing_ok=True)
    raw_dir = candidate / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    return info


def verify_migrated_dataset(root: Path) -> None:
    info = _read_json(root / "meta" / "info.json")
    if not isinstance(info, dict):
        raise MigrationError("target is missing meta/info.json")
    if info.get("schema") != "delivery" or info.get("contract_version") != CONTRACT_VERSION:
        raise MigrationError("target does not declare canonical delivery contract v3")
    if info.get("action_source") != "next_measured_eef_fallback":
        raise MigrationError("target action_source is not next_measured_eef_fallback")
    state_dim = int(info.get("state_dim", 0))
    action_dim = int(info.get("raw_action_dim", 0))
    if state_dim not in {10, 20} or action_dim != state_dim:
        raise MigrationError("target canonical state/action dimensions are invalid")
    for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if len(states) > 1 and not np.allclose(actions[:-1], states[1:], atol=1e-4, rtol=1e-4):
            raise MigrationError(f"{path}: migrated actions do not match next measured state")
        if not np.allclose(actions[-1], states[-1], atol=1e-6, rtol=1e-6):
            raise MigrationError(f"{path}: terminal action is not a hold")
        for arm in range(state_dim // 10):
            for values, name in ((states, "state"), (actions, "action")):
                gripper = values[:, arm * 10 + 9]
                if np.min(gripper) < -1e-5 or np.max(gripper) > 1.00001:
                    raise MigrationError(f"{path}: {name} opening fraction outside [0,1]")


def migrate_dataset(
    source: str | Path,
    target: str | Path,
    *,
    verify: bool = False,
) -> Path:
    source = Path(source).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    if target.exists():
        raise FileExistsError(f"target already exists: {target}")
    if source == target or source in target.parents or target in source.parents:
        raise MigrationError("source and target must be separate, non-nested directories")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise MigrationError(f"source dataset contains a symlink: {path}")
    source_info = _read_json(source / "meta" / "info.json")
    if not isinstance(source_info, dict):
        raise MigrationError("source is missing meta/info.json")
    layout = detect_legacy_layout(source_info)

    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = target.parent / f".{target.name}.migrating-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, candidate, copy_function=_link_or_copy)
        info = _rewrite_candidate(source, candidate, source_info, layout)
        verify_migrated_dataset(candidate)
        if verify:
            from check_pi05_dataset import check_dataset

            errors = check_dataset(candidate)
            if errors:
                raise MigrationError(
                    "canonical checker failed:\n  - " + "\n  - ".join(errors)
                )
        os.replace(candidate, target)
    except BaseException:
        shutil.rmtree(candidate, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "source": str(source),
                "target": str(target),
                "episodes": info["total_episodes"],
                "frames": info["total_frames"],
                "state_dim": info["state_dim"],
                "raw_action_dim": info["raw_action_dim"],
                "model_action_dim": info["model_action_dim"],
                "action_source": info["action_source"],
                "verified": bool(verify),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy_v2 Piper delivery LeRobot data to canonical v3"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run the full canonical dataset checker before publishing the target",
    )
    args = parser.parse_args()
    migrate_dataset(args.source, args.target, verify=args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
