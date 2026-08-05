#!/usr/bin/env python3
"""Atomic episode-level editing and merging for local LeRobot datasets.

The editor never changes frame payloads.  It may rewrite parquet index columns and
optional raw NPZ metadata when episodes are merged, removed, or renumbered.
Videos are hard-linked when possible and are never re-encoded.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Callable, Iterator
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pi0_dataset import (
    DEFAULT_ACTION_HORIZON,
    DELIVERY_ABSOLUTE_ACTION_FORMAT,
    DELIVERY_LEGACY_ACTION_FORMAT,
    GRIPPER_CLOSED_FRACTION_LEGACY,
    GRIPPER_OPENING_FRACTION,
    LEGACY_V2,
    LEGACY_ROTATION_SEMANTICS,
    ROTATION6D_SEMANTICS,
    classify_contract_dimensions,
    default_eef_names,
)


EPISODE_FILE = re.compile(r"episode_(\d+)\.parquet$")
DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RESERVED_EPISODE_FIELDS = {
    "episode_index",
    "tasks",
    "length",
    "task_name",
    "success",
    "action_semantics",
    "action_offset",
    "instruction",
    "task_index",
    "contract_version",
    "contract_format",
    "legacy",
    "legacy_format",
    "schema",
    "arm_mode",
    "arm_side",
    "state_dim",
    "action_dim",
    "raw_action_dim",
    "model_action_dim",
    "state_names",
    "action_names",
    "camera_keys",
    "delivery_action_format",
    "action_source",
    "action_alignment",
    "action_horizon",
    "gripper_semantics",
    "rotation_semantics",
    "coordinate_frame",
}
COMPATIBILITY_FIELDS = (
    "codebase_version",
    "robot_type",
    "fps",
    "chunks_size",
    "features",
    "schema",
    "arm_mode",
    "arm_side",
    "state_dim",
    "action_dim",
    "raw_action_dim",
    "model_action_dim",
    "state_names",
    "action_names",
    "camera_keys",
    "contract_format",
    "legacy_format",
    "delivery_action_format",
    "action_semantics",
    "action_source",
    "action_alignment",
    "action_offset",
    "action_horizon",
    "gripper_semantics",
    "rotation_semantics",
    "coordinate_frame",
)
POLICY_CONFIG_NAMES = (
    "pi05_piper_single_arm_lora",
    "pi05_piper_bimanual_lora",
)
DATASET_ORIGINS = {"real", "simulation", "unknown"}
DATASET_ORIGIN_METADATA_FILENAME = "dashboard_dataset_origin.json"


class DatasetValidationError(ValueError):
    def __init__(self, phase: str, message: str, output: str = ""):
        super().__init__(message)
        self.phase = phase
        self.output = output


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def normalize_dataset_origin(value: Any, *, allow_unknown: bool = True) -> str:
    origin = str(value or "unknown").strip().lower()
    aliases = {
        "real_robot": "real",
        "real-robot": "real",
        "hardware": "real",
        "sim": "simulation",
        "synthetic": "simulation",
        "unknown": "unknown",
    }
    origin = aliases.get(origin, origin)
    allowed = DATASET_ORIGINS if allow_unknown else DATASET_ORIGINS - {"unknown"}
    if origin not in allowed:
        raise ValueError(f"dataset origin must be one of {sorted(allowed)}")
    return origin


def read_dataset_origin_marker(dataset_path: Path) -> dict[str, Any] | None:
    value = _read_json(
        Path(dataset_path) / "meta" / DATASET_ORIGIN_METADATA_FILENAME
    )
    if not isinstance(value, dict):
        return None
    try:
        origin = normalize_dataset_origin(value.get("origin"))
    except ValueError:
        return None
    return {**value, "origin": origin}


def write_dataset_origin_marker(
    dataset_path: Path,
    origin: Any,
    *,
    source: str,
) -> dict[str, Any]:
    normalized = normalize_dataset_origin(origin)
    payload = {
        "version": 1,
        "origin": normalized,
        "source": str(source),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(
        Path(dataset_path) / "meta" / DATASET_ORIGIN_METADATA_FILENAME, payload
    )
    return payload


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, path)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _clone_tree(source: Path, destination: Path) -> None:
    def copy_file(src: str, dst: str) -> str:
        try:
            os.link(src, dst)
            return dst
        except OSError:
            return shutil.copy2(src, dst)

    shutil.copytree(source, destination, copy_function=copy_file)


def _safe_dataset_id(value: str) -> str:
    value = str(value or "")
    if not DATASET_ID.fullmatch(value) or value in {".", ".."} or ".." in value:
        raise ValueError("invalid dataset id: use letters, numbers, dot, underscore, or dash")
    return value


def _video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        return []
    return sorted(
        key for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


def _image_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        return []
    return sorted(
        key for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "image"
    )


def _image_cell(table: pa.Table, image_key: str, frame_index: int) -> dict[str, Any]:
    column_index = table.schema.get_field_index(image_key)
    if column_index < 0:
        return {}
    value = table.column(column_index)[frame_index].as_py()
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes": bytes(value)}
    return {}


def _safe_relative_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip())
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path


def _external_image_path(
    root: Path,
    info: dict[str, Any],
    image_key: str,
    episode_index: int,
    frame_index: int,
    stored_path: Any = None,
) -> Path | None:
    chunk = episode_index // int(info.get("chunks_size", 1000))
    episode_name = f"episode_{episode_index:06d}"
    default_name = f"frame_{frame_index:06d}.png"
    stored = _safe_relative_path(stored_path)
    frame_name = stored.name if stored is not None else default_name
    root_resolved = root.resolve()
    candidates = [
        root / "images" / image_key / episode_name / default_name,
        root / "images" / f"chunk-{chunk:03d}" / image_key / episode_name / default_name,
    ]
    if stored is not None:
        candidates.extend(
            [
                root / stored,
                root / "images" / stored,
                root / "images" / image_key / episode_name / frame_name,
                root / "images" / f"chunk-{chunk:03d}" / image_key / episode_name / frame_name,
            ]
        )
    for episode_dir in (
        root / "images" / image_key / episode_name,
        root / "images" / f"chunk-{chunk:03d}" / image_key / episode_name,
    ):
        candidates.extend(sorted(episode_dir.glob(f"frame_{frame_index:06d}.*")))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_relative_to(root_resolved) and resolved.is_file():
            return resolved
    return None


def _image_mimetype(blob: bytes, stored_path: Any = None) -> str:
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if blob.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if blob.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if blob.startswith(b"RIFF") and blob[8:12] == b"WEBP":
        return "image/webp"
    guessed, _encoding = mimetypes.guess_type(str(stored_path or ""))
    return guessed or "application/octet-stream"


def _format_episode_path(root: Path, info: dict[str, Any], key: str, episode_index: int, **extra: Any) -> Path:
    template = info.get(key)
    if not isinstance(template, str) or not template:
        raise ValueError(f"dataset meta/info.json is missing {key}")
    values = {
        "episode_index": episode_index,
        "episode_chunk": episode_index // int(info.get("chunks_size", 1000)),
        **extra,
    }
    return root / template.format(**values)


def _parquet_paths(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        match = EPISODE_FILE.search(path.name)
        if not match:
            continue
        index = int(match.group(1))
        if index in result:
            raise ValueError(f"duplicate episode parquet index {index} in {root}")
        result[index] = path
    return result


def _metadata_by_index(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if "episode_index" not in row:
            continue
        index = int(row["episode_index"])
        if index in result:
            raise ValueError(f"duplicate episode metadata index {index} in {path}")
        result[index] = row
    return result


def _tasks_by_index(root: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in _read_jsonl(root / "meta" / "tasks.jsonl"):
        if "task_index" in row and "task" in row:
            result[int(row["task_index"])] = str(row["task"])
    return result


def _replace_column(table: pa.Table, name: str, values: Any) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"episode parquet is missing required column {name}")
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def _numeric_column(table: pa.Table, name: str) -> np.ndarray | None:
    field_type = table.schema.field(name).type
    try:
        if pa.types.is_list(field_type) or pa.types.is_fixed_size_list(field_type):
            values = np.asarray(table[name].to_pylist(), dtype=np.float64)
        elif pa.types.is_integer(field_type) or pa.types.is_floating(field_type) or pa.types.is_boolean(field_type):
            values = np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float64)
        else:
            return None
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        return None
    return values


def _column_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": np.mean(values, axis=0).astype(np.float32).tolist(),
        "std": np.std(values, axis=0).astype(np.float32).tolist(),
        "min": np.min(values, axis=0).astype(np.float32).tolist(),
        "max": np.max(values, axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float32).tolist(),
        "count": [len(values)],
    }


def _recompute_episode_stats(table: pa.Table, existing: dict[str, Any]) -> dict[str, Any]:
    """Refresh every numeric parquet statistic after reindex/task edits."""
    result = dict(existing.get("stats", {})) if isinstance(existing, dict) else {}
    for name in table.column_names:
        values = _numeric_column(table, name)
        if values is not None:
            result[name] = _column_stats(values)
    return {"stats": result}


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _build_splits(original: Any, old_total: int, new_total: int) -> dict[str, str]:
    if new_total <= 0:
        return {}
    if not isinstance(original, dict) or "val" not in original:
        return {"train": f"0:{new_total}"}
    try:
        start_text, end_text = str(original["val"]).split(":", 1)
        old_val = max(0, int(end_text) - int(start_text))
    except (TypeError, ValueError):
        return {"train": f"0:{new_total}"}
    if old_total <= 1 or old_val <= 0 or new_total <= 1:
        return {"train": f"0:{new_total}"}
    val_count = min(new_total - 1, max(1, round(new_total * old_val / old_total)))
    train_end = new_total - val_count
    return {"train": f"0:{train_end}", "val": f"{train_end}:{new_total}"}


def _contract_metadata(info: dict[str, Any]) -> dict[str, Any]:
    """Return normalized contract metadata while accepting old info.json files."""
    features = info.get("features", {})
    if not isinstance(features, dict):
        return {}
    if "observation.state" in features and "action" in features:
        state_key, action_key = "observation.state", "action"
    elif "state" in features and "actions" in features:
        state_key, action_key = "state", "actions"
    else:
        return {}
    try:
        state_dim = int(features[state_key]["shape"][0])
        action_dim = int(features[action_key]["shape"][0])
        dimensions = classify_contract_dimensions(
            state_dim,
            action_dim,
            schema=info.get("schema"),
            legacy_format=info.get("legacy_format") or info.get("contract_format"),
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return {}
    arm_mode = dimensions["arm_mode"]
    arm_side = str(info.get("arm_side") or ("both" if arm_mode == "bimanual" else "right"))
    if arm_mode == "bimanual":
        arm_side = "both"
    fallback_state, fallback_action = default_eef_names(
        arm_mode=arm_mode, arm_side=arm_side, legacy=dimensions["legacy"]
    )
    state_names = features[state_key].get("names")
    action_names = features[action_key].get("names")
    if not isinstance(state_names, list) or len(state_names) != state_dim:
        state_names = fallback_state if dimensions["schema"] == "delivery" else None
    if not isinstance(action_names, list) or len(action_names) != action_dim:
        action_names = fallback_action if dimensions["schema"] == "delivery" else None
    legacy = bool(dimensions["legacy"])
    return {
        **dimensions,
        "arm_side": arm_side,
        "state_names": state_names,
        "action_names": action_names,
        "action_semantics": info.get(
            "action_semantics",
            "eef_delta_base_xyz_left_rotvec_gripper_target" if legacy else "absolute_joint_position",
        ),
        "action_source": info.get("action_source", "next_measured_eef" if legacy else ""),
        "action_alignment": info.get("action_alignment", "next_observation" if legacy else ""),
        "action_offset": int(info.get("action_offset", 1 if legacy else 0)),
        "action_horizon": int(info.get("action_horizon", DEFAULT_ACTION_HORIZON)),
        "gripper_semantics": info.get(
            "gripper_semantics", GRIPPER_CLOSED_FRACTION_LEGACY if legacy else GRIPPER_OPENING_FRACTION
        ),
        "rotation_semantics": info.get(
            "rotation_semantics", LEGACY_ROTATION_SEMANTICS if legacy else ROTATION6D_SEMANTICS
        ),
        "coordinate_frame": info.get("coordinate_frame", "slave_base"),
    }


def _compatibility_signature(info: dict[str, Any]) -> dict[str, Any]:
    metadata = _contract_metadata(info)
    features = info.get("features", {})
    feature_signature = {}
    for key, value in features.items():
        if not isinstance(value, dict):
            continue
        names = value.get("names")
        if key in {"observation.state", "state"} and metadata.get("state_names") is not None:
            names = metadata["state_names"]
        elif key in {"action", "actions"} and metadata.get("action_names") is not None:
            names = metadata["action_names"]
        feature_signature[key] = {
            "dtype": value.get("dtype"),
            "shape": value.get("shape"),
            "names": names,
        }
    result = {
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "chunks_size": info.get("chunks_size"),
        "features": feature_signature,
    }
    for key in (
        "schema", "arm_mode", "arm_side", "state_dim", "raw_action_dim", "model_action_dim",
        "contract_format", "legacy_format", "delivery_action_format", "action_semantics",
        "action_source", "action_alignment", "action_offset", "action_horizon",
        "gripper_semantics", "rotation_semantics", "coordinate_frame",
    ):
        if key in metadata:
            result[key] = metadata[key]
    # Missing names in metadata-free legacy files are tolerated; if both sides
    # provide names they must agree exactly.
    for key in ("state_names", "action_names"):
        value = metadata.get(key)
        if value is not None:
            result[key] = value
    return result


def _validate_compatible(target_info: dict[str, Any], source_info: dict[str, Any]) -> None:
    target = _compatibility_signature(target_info)
    source = _compatibility_signature(source_info)
    mismatches = [key for key in sorted(set(target) | set(source)) if target.get(key) != source.get(key)]
    if mismatches:
        raise ValueError("datasets are incompatible; differing fields: " + ", ".join(mismatches))


class DatasetEditor:
    def __init__(
        self,
        *,
        dataset_root: Path,
        assets_base_dir: Path,
        validate_staging: Callable[[Path], str],
        validate_installed: Callable[[str], str],
        assert_idle: Callable[[str], None] | None = None,
    ):
        self.dataset_root = dataset_root
        self.assets_base_dir = assets_base_dir
        self.validate_staging = validate_staging
        self.validate_installed = validate_installed
        self.assert_idle = assert_idle or (lambda _dataset_id: None)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _dataset_path(self, dataset_id: str) -> Path:
        return self.dataset_root / dataset_id

    @contextlib.contextmanager
    def _lock(self, dataset_id: str) -> Iterator[None]:
        with self._locks_for(dataset_id):
            yield

    @contextlib.contextmanager
    def _locks_for(self, *dataset_ids: str) -> Iterator[None]:
        names = sorted(set(dataset_ids))
        with self._global_lock:
            locks = [self._locks.setdefault(name, threading.Lock()) for name in names]
        with contextlib.ExitStack() as stack:
            for lock in locks:
                stack.enter_context(lock)
            yield

    def _validated_commit(self, dataset_id: str, candidate: Path) -> tuple[str, dict[str, Any]]:
        """Validate and install a rebuilt tree, removing abandoned candidates."""
        try:
            structural = self.validate_staging(candidate)
            return structural, self._commit(dataset_id, candidate)
        finally:
            if candidate.exists():
                shutil.rmtree(candidate)

    def details(self, dataset_id: str, *, offset: int = 0, limit: int = 200) -> dict[str, Any]:
        root = self._dataset_path(dataset_id)
        info = _read_json(root / "meta" / "info.json")
        if not isinstance(info, dict):
            raise FileNotFoundError(f"dataset is not installed: {dataset_id}")
        display_info = dict(info)
        normalized_contract = _contract_metadata(info)
        for key, value in normalized_contract.items():
            display_info.setdefault(key, value)
        parquet = _parquet_paths(root)
        episodes = _metadata_by_index(root / "meta" / "episodes.jsonl")
        tasks = _tasks_by_index(root)
        indexes = sorted(parquet)
        selected = indexes[offset:offset + limit]
        rows = []
        video_keys = _video_keys(info)
        image_keys = _image_keys(info)
        for index in selected:
            row = dict(episodes.get(index, {}))
            task_values = row.get("tasks")
            if not isinstance(task_values, list):
                task_values = []
            if not task_values:
                table = pq.read_table(parquet[index], columns=["task_index"])
                task_ids = np.asarray(table["task_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
                task_values = _unique_strings([tasks.get(int(item), f"task_{int(item)}") for item in task_ids])
            length = row.get("length")
            if length is None:
                length = pq.read_metadata(parquet[index]).num_rows
            parameters = {
                key: value for key, value in row.items()
                if key not in RESERVED_EPISODE_FIELDS
            }
            rows.append(
                {
                    "episode_index": index,
                    "length": int(length),
                    "instruction": str(task_values[0]) if task_values else "",
                    "tasks": task_values,
                    "task_name": row.get("task_name"),
                    "success": row.get("success"),
                    "parameters": parameters,
                    "video_keys": video_keys,
                    "image_keys": image_keys,
                    "media": [
                        *(
                            {"key": key, "type": "video", "frames": int(length), "fps": info.get("fps")}
                            for key in video_keys
                        ),
                        *(
                            {"key": key, "type": "image", "frames": int(length), "fps": info.get("fps")}
                            for key in image_keys
                        ),
                    ],
                }
            )
        return {
            "id": dataset_id,
            "dataset_origin_marker": read_dataset_origin_marker(root),
            "info": display_info,
            "tasks": [{"task_index": key, "task": value} for key, value in sorted(tasks.items())],
            "episodes": rows,
            "offset": offset,
            "limit": limit,
            "total": len(indexes),
        }

    def video_path(self, dataset_id: str, episode_index: int, video_key: str) -> Path:
        root = self._dataset_path(dataset_id)
        info = _read_json(root / "meta" / "info.json")
        if not isinstance(info, dict):
            raise FileNotFoundError(f"dataset is not installed: {dataset_id}")
        if video_key not in _video_keys(info):
            raise ValueError(f"unknown video key: {video_key}")
        path = _format_episode_path(root, info, "video_path", episode_index, video_key=video_key)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def image_path(self, dataset_id: str, episode_index: int, image_key: str, frame_index: int) -> Path:
        source, _mimetype = self.image_source(dataset_id, episode_index, image_key, frame_index)
        if isinstance(source, Path):
            return source
        raise FileNotFoundError(
            f"image is embedded in parquet and has no external path: dataset={dataset_id} "
            f"episode={episode_index} key={image_key} frame={frame_index}"
        )

    def image_source(
        self, dataset_id: str, episode_index: int, image_key: str, frame_index: int
    ) -> tuple[Path | io.BytesIO, str | None]:
        root = self._dataset_path(dataset_id)
        info = _read_json(root / "meta" / "info.json")
        if not isinstance(info, dict):
            raise FileNotFoundError(f"dataset is not installed: {dataset_id}")
        if image_key not in _image_keys(info):
            raise ValueError(f"unknown image key: {image_key}")
        parquet = _parquet_paths(root)
        parquet_path = parquet.get(episode_index)
        if parquet_path is None:
            raise FileNotFoundError(f"episode does not exist: {episode_index}")
        frame_count = pq.read_metadata(parquet_path).num_rows
        if not 0 <= frame_index < frame_count:
            raise ValueError(f"frame index must be in [0, {max(0, frame_count - 1)}]")

        table = None
        cell: dict[str, Any] = {}
        try:
            table = pq.read_table(parquet_path, columns=[image_key])
            cell = _image_cell(table, image_key, frame_index)
        except (KeyError, IndexError, OSError, TypeError, ValueError, pa.ArrowInvalid):
            cell = {}
        stored_path = cell.get("path")
        result = _external_image_path(
            root, info, image_key, episode_index, frame_index, stored_path
        )
        if result is not None:
            return result, None
        blob = cell.get("bytes")
        if isinstance(blob, (bytes, bytearray, memoryview)) and blob:
            data = bytes(blob)
            return io.BytesIO(data), _image_mimetype(data, stored_path)
        raise FileNotFoundError(
            f"image frame not found: dataset={dataset_id} episode={episode_index} "
            f"key={image_key} frame={frame_index}"
        )

    def set_dataset_origin(
        self, dataset_id: str, origin: Any, *, source: str = "dashboard"
    ) -> dict[str, Any]:
        dataset_id = _safe_dataset_id(dataset_id)
        normalized = normalize_dataset_origin(origin)
        with self._lock(dataset_id):
            target = self._dataset_path(dataset_id)
            if not target.is_dir():
                raise FileNotFoundError(f"dataset is not installed: {dataset_id}")
            marker = write_dataset_origin_marker(
                target, normalized, source=source
            )
        return {
            "operation": "set_dataset_origin",
            "dataset_id": dataset_id,
            "dataset_origin": normalized,
            "marker": marker,
            "path": str(target),
        }

    def rename_dataset(self, old_id: str, new_id: str) -> dict[str, Any]:
        old_id = _safe_dataset_id(old_id)
        new_id = _safe_dataset_id(new_id)
        if old_id == new_id:
            raise ValueError("new dataset id must differ from the current id")
        with self._locks_for(old_id, new_id):
            self.assert_idle(old_id)
            source = self._dataset_path(old_id)
            target = self._dataset_path(new_id)
            if not source.is_dir():
                raise FileNotFoundError(f"dataset is not installed: {old_id}")
            if target.exists():
                raise FileExistsError(f"dataset already exists: {new_id}")

            asset_moves = [
                (
                    self.assets_base_dir / config_name / old_id,
                    self.assets_base_dir / config_name / new_id,
                )
                for config_name in POLICY_CONFIG_NAMES
            ]
            conflicts = [str(target_assets) for _, target_assets in asset_moves if target_assets.exists()]
            if conflicts:
                raise FileExistsError(
                    f"norm stats directory already exists for: {new_id}: {conflicts}"
                )

            moved_assets: list[tuple[Path, Path]] = []
            os.replace(source, target)
            try:
                loader_output = self.validate_installed(new_id)
                for source_assets, target_assets in asset_moves:
                    if not source_assets.exists():
                        continue
                    target_assets.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source_assets, target_assets)
                    moved_assets.append((source_assets, target_assets))
            except Exception:
                for source_assets, target_assets in reversed(moved_assets):
                    if target_assets.exists() and not source_assets.exists():
                        os.replace(target_assets, source_assets)
                if target.exists() and not source.exists():
                    os.replace(target, source)
                raise
            return {
                "operation": "rename_dataset",
                "old_dataset_id": old_id,
                "dataset_id": new_id,
                "path": str(target),
                "loader_validation": loader_output,
                "norm_stats_moved": bool(moved_assets),
                "norm_stats_paths_moved": [str(target) for _, target in moved_assets],
                "warning": "historical checkpoints and task records still reference the old dataset id",
            }

    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        dataset_id = _safe_dataset_id(dataset_id)
        with self._lock(dataset_id):
            self.assert_idle(dataset_id)
            target = self._dataset_path(dataset_id)
            if not target.is_dir():
                raise FileNotFoundError(f"dataset is not installed: {dataset_id}")

            suffix = f"deleted-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            dataset_trash = self.dataset_root / f".{dataset_id}.{suffix}"
            asset_moves = []
            for config_name in POLICY_CONFIG_NAMES:
                assets = self.assets_base_dir / config_name / dataset_id
                assets_trash = assets.parent / f".{dataset_id}.{suffix}"
                asset_moves.append((assets, assets_trash))
            moved_assets: list[tuple[Path, Path]] = []
            os.replace(target, dataset_trash)
            try:
                for assets, assets_trash in asset_moves:
                    if not assets.exists():
                        continue
                    os.replace(assets, assets_trash)
                    moved_assets.append((assets, assets_trash))
                shutil.rmtree(dataset_trash)
                for _, assets_trash in moved_assets:
                    shutil.rmtree(assets_trash)
            except Exception:
                if dataset_trash.exists() and not target.exists():
                    os.replace(dataset_trash, target)
                for assets, assets_trash in reversed(moved_assets):
                    if assets_trash.exists() and not assets.exists():
                        os.replace(assets_trash, assets)
                raise
            return {
                "operation": "delete_dataset",
                "dataset_id": dataset_id,
                "dataset_deleted": True,
                "norm_stats_deleted": bool(moved_assets),
                "norm_stats_paths_deleted": [str(path) for path, _ in moved_assets],
                "warning": "historical checkpoints, models, and task records were not deleted",
            }

    def install_upload(
        self,
        dataset_id: str,
        extracted: Path,
        *,
        overwrite: bool,
        merge: bool,
        dataset_origin: Any = "real",
    ) -> dict[str, Any]:
        if overwrite and merge:
            raise ValueError("overwrite and merge are mutually exclusive")
        normalized_origin = normalize_dataset_origin(dataset_origin, allow_unknown=False)
        write_dataset_origin_marker(extracted, normalized_origin, source="upload")
        with self._lock(dataset_id):
            target = self._dataset_path(dataset_id)
            source_validation = self.validate_staging(extracted)
            operation = "install"
            candidate = extracted
            final_validation = source_validation
            if target.exists():
                self.assert_idle(dataset_id)
                existing_marker = read_dataset_origin_marker(target)
                existing_origin = (existing_marker or {}).get("origin")
                if merge and existing_origin not in {None, "unknown", normalized_origin}:
                    raise ValueError(
                        f"cannot merge {normalized_origin} upload into {existing_origin} dataset {dataset_id}"
                    )
                if merge:
                    operation = "merge"
                    candidate = self._rebuild_candidate(dataset_id, [(target, None), (extracted, None)])
                    write_dataset_origin_marker(
                        candidate, normalized_origin, source="upload_merge"
                    )
                    final_validation, result = self._validated_commit(dataset_id, candidate)
                elif overwrite:
                    operation = "overwrite"
                else:
                    raise FileExistsError(
                        f"dataset already exists: {target}; use --merge to append episodes or --overwrite to replace it"
                    )
            elif merge:
                operation = "install"
            if operation != "merge":
                result = self._commit(dataset_id, candidate)
            result.update(
                {
                    "operation": operation,
                    "dataset_origin": normalized_origin,
                    "source_validation": source_validation,
                    "structural_validation": final_validation,
                }
            )
            if candidate != extracted and extracted.exists():
                shutil.rmtree(extracted)
            return result

    def merge_existing(self, target_id: str, source_id: str) -> dict[str, Any]:
        if target_id == source_id:
            raise ValueError("source and target dataset must be different")
        target = self._dataset_path(target_id)
        source = self._dataset_path(source_id)
        if not target.is_dir() or not source.is_dir():
            raise FileNotFoundError("source or target dataset is not installed")
        with self._locks_for(target_id, source_id):
            self.assert_idle(target_id)
            candidate = self._rebuild_candidate(target_id, [(target, None), (source, None)])
            structural, result = self._validated_commit(target_id, candidate)
            result.update({"operation": "merge", "source_dataset_id": source_id, "structural_validation": structural})
            return result

    def update_episode(self, dataset_id: str, episode_index: int, updates: dict[str, Any]) -> dict[str, Any]:
        target = self._dataset_path(dataset_id)
        if not target.is_dir():
            raise FileNotFoundError(f"dataset is not installed: {dataset_id}")
        normalized = self._normalize_updates(updates)
        with self._lock(dataset_id):
            self.assert_idle(dataset_id)
            if episode_index not in _parquet_paths(target):
                raise FileNotFoundError(f"episode {episode_index} does not exist in {dataset_id}")
            candidate = self._rebuild_candidate(dataset_id, [(target, {episode_index: normalized})])
            structural, result = self._validated_commit(dataset_id, candidate)
            result.update({"operation": "update_episode", "episode_index": episode_index, "structural_validation": structural})
            return result

    def delete_episodes(self, dataset_id: str, episode_indexes: list[int]) -> dict[str, Any]:
        target = self._dataset_path(dataset_id)
        if not target.is_dir():
            raise FileNotFoundError(f"dataset is not installed: {dataset_id}")
        selected = sorted(set(int(item) for item in episode_indexes))
        if not selected:
            raise ValueError("episode_indexes must not be empty")
        with self._lock(dataset_id):
            self.assert_idle(dataset_id)
            existing = set(_parquet_paths(target))
            missing = sorted(set(selected) - existing)
            if missing:
                raise FileNotFoundError(f"episodes do not exist: {missing}")
            if len(selected) >= len(existing):
                raise ValueError("refusing to delete every episode; delete the dataset explicitly instead")
            candidate = self._rebuild_candidate(dataset_id, [(target, None)], excluded=set(selected))
            structural, result = self._validated_commit(dataset_id, candidate)
            result.update(
                {
                    "operation": "delete_episodes",
                    "deleted_episode_indexes": selected,
                    "structural_validation": structural,
                }
            )
            return result

    @staticmethod
    def _normalize_updates(updates: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(updates, dict):
            raise ValueError("episode update must be a JSON object")
        result: dict[str, Any] = {}
        if "instruction" in updates:
            instruction = str(updates["instruction"]).strip()
            if not instruction or len(instruction) > 4096:
                raise ValueError("instruction must contain 1-4096 characters")
            result["instruction"] = instruction
        if "task_name" in updates:
            if updates["task_name"] is None:
                result["task_name"] = None
            else:
                task_name = str(updates["task_name"]).strip()
                if len(task_name) > 256:
                    raise ValueError("task_name must not exceed 256 characters")
                result["task_name"] = task_name or None
        if "success" in updates:
            if updates["success"] is not None and not isinstance(updates["success"], bool):
                raise ValueError("success must be a JSON boolean or null")
            result["success"] = updates["success"]
        metadata = updates.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        if len(metadata) > 100:
            raise ValueError("metadata contains too many fields")
        blocked = RESERVED_EPISODE_FIELDS.intersection(metadata)
        if blocked:
            raise ValueError("metadata cannot modify reserved fields: " + ", ".join(sorted(blocked)))
        json.dumps(metadata, ensure_ascii=False)
        result["metadata"] = metadata
        remove = updates.get("remove_metadata", [])
        if not isinstance(remove, list) or any(not isinstance(item, str) for item in remove):
            raise ValueError("remove_metadata must be a list of field names")
        blocked = RESERVED_EPISODE_FIELDS.intersection(remove)
        if blocked:
            raise ValueError("cannot remove reserved fields: " + ", ".join(sorted(blocked)))
        result["remove_metadata"] = remove
        if set(result) == {"metadata", "remove_metadata"} and not metadata and not remove:
            raise ValueError("no episode fields were provided")
        return result

    def _rebuild_candidate(
        self,
        dataset_id: str,
        sources: list[tuple[Path, dict[int, dict[str, Any]] | None]],
        *,
        excluded: set[int] | None = None,
    ) -> Path:
        excluded = excluded or set()
        base = sources[0][0]
        base_info = _read_json(base / "meta" / "info.json")
        if not isinstance(base_info, dict):
            raise ValueError(f"invalid target dataset: {base}")
        base_contract = _contract_metadata(base_info)
        for source, _updates in sources[1:]:
            source_info = _read_json(source / "meta" / "info.json")
            if not isinstance(source_info, dict):
                raise ValueError(f"invalid source dataset: {source}")
            _validate_compatible(base_info, source_info)

        candidate = self.dataset_root / f".{dataset_id}.editing-{uuid.uuid4().hex}"
        _clone_tree(base, candidate)
        for name in ("data", "videos", "images", "raw"):
            path = candidate / name
            if path.exists():
                shutil.rmtree(path)
        for name in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "openpi_norm_stats.json"):
            (candidate / "meta" / name).unlink(missing_ok=True)
        (candidate / "data").mkdir(parents=True, exist_ok=True)
        (candidate / "videos").mkdir(parents=True, exist_ok=True)

        task_to_index: dict[str, int] = {}
        task_rows: list[dict[str, Any]] = []
        episode_rows: list[dict[str, Any]] = []
        stats_rows: list[dict[str, Any]] = []
        total_frames = 0
        total_videos = 0
        new_episode_index = 0

        try:
            for source_number, (source, updates) in enumerate(sources):
                info = _read_json(source / "meta" / "info.json")
                assert isinstance(info, dict)
                parquets = _parquet_paths(source)
                episodes = _metadata_by_index(source / "meta" / "episodes.jsonl")
                stats = _metadata_by_index(source / "meta" / "episodes_stats.jsonl")
                source_tasks = _tasks_by_index(source)
                for old_episode_index, parquet_path in sorted(parquets.items()):
                    if source_number == 0 and old_episode_index in excluded:
                        continue
                    update = (updates or {}).get(old_episode_index, {})
                    table = pq.read_table(parquet_path)
                    frame_count = table.num_rows
                    if frame_count <= 0:
                        raise ValueError(f"episode is empty: {parquet_path}")
                    for required in ("frame_index", "episode_index", "index", "task_index"):
                        if table.schema.get_field_index(required) < 0:
                            raise ValueError(f"{parquet_path} is missing required column {required}")

                    old_task_ids = np.asarray(table["task_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
                    source_row = dict(episodes.get(old_episode_index, {}))
                    fallback_tasks = source_row.get("tasks")
                    if not isinstance(fallback_tasks, list):
                        fallback_tasks = []
                    instruction_override = update.get("instruction")
                    task_texts: list[str] = []
                    mapped_task_ids: list[int] = []
                    for old_task_id in old_task_ids:
                        if instruction_override is not None:
                            task_text = instruction_override
                        else:
                            task_text = source_tasks.get(int(old_task_id), "")
                            if not task_text and len(fallback_tasks) == 1:
                                task_text = str(fallback_tasks[0])
                            if not task_text:
                                raise ValueError(
                                    f"cannot resolve task_index={int(old_task_id)} in {source}/meta/tasks.jsonl"
                                )
                        if task_text not in task_to_index:
                            task_to_index[task_text] = len(task_to_index)
                            task_rows.append({"task_index": task_to_index[task_text], "task": task_text})
                        task_texts.append(task_text)
                        mapped_task_ids.append(task_to_index[task_text])

                    table = _replace_column(table, "frame_index", np.arange(frame_count, dtype=np.int64))
                    table = _replace_column(
                        table, "episode_index", np.full(frame_count, new_episode_index, dtype=np.int64)
                    )
                    table = _replace_column(
                        table, "index", np.arange(total_frames, total_frames + frame_count, dtype=np.int64)
                    )
                    table = _replace_column(table, "task_index", np.asarray(mapped_task_ids, dtype=np.int64))
                    destination_parquet = _format_episode_path(
                        candidate, base_info, "data_path", new_episode_index
                    )
                    destination_parquet.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(table, destination_parquet)

                    for video_key in _video_keys(base_info):
                        source_video = _format_episode_path(
                            source, info, "video_path", old_episode_index, video_key=video_key
                        )
                        if not source_video.is_file():
                            raise FileNotFoundError(source_video)
                        destination_video = _format_episode_path(
                            candidate, base_info, "video_path", new_episode_index, video_key=video_key
                        )
                        _link_or_copy(source_video, destination_video)
                        total_videos += 1

                    self._copy_episode_images(
                        source,
                        info,
                        table,
                        old_episode_index=old_episode_index,
                        candidate=candidate,
                        new_episode_index=new_episode_index,
                    )

                    source_row["episode_index"] = new_episode_index
                    source_row["length"] = frame_count
                    source_row["tasks"] = _unique_strings(task_texts)
                    if base_contract.get("legacy"):
                        source_row.update(
                            {
                                "schema": base_contract["schema"],
                                "arm_mode": base_contract["arm_mode"],
                                "arm_side": base_contract["arm_side"],
                                "state_dim": base_contract["state_dim"],
                                "action_dim": base_contract["raw_action_dim"],
                                "raw_action_dim": base_contract["raw_action_dim"],
                                "model_action_dim": base_contract["model_action_dim"],
                                "contract_format": LEGACY_V2,
                                "legacy": True,
                                "legacy_format": LEGACY_V2,
                                "legacy_delivery_v2": True,
                                "delivery_action_format": DELIVERY_LEGACY_ACTION_FORMAT,
                                "action_semantics": base_contract["action_semantics"],
                                "action_source": base_contract["action_source"],
                                "action_alignment": base_contract["action_alignment"],
                                "action_offset": base_contract["action_offset"],
                                "action_horizon": base_contract["action_horizon"],
                                "gripper_semantics": base_contract["gripper_semantics"],
                                "rotation_semantics": base_contract["rotation_semantics"],
                                "coordinate_frame": base_contract["coordinate_frame"],
                            }
                        )
                    if "task_name" in update:
                        if update["task_name"] is None:
                            source_row.pop("task_name", None)
                        else:
                            source_row["task_name"] = update["task_name"]
                    if "success" in update:
                        if update["success"] is None:
                            source_row.pop("success", None)
                        else:
                            source_row["success"] = update["success"]
                    for key in update.get("remove_metadata", []):
                        source_row.pop(key, None)
                    source_row.update(update.get("metadata", {}))
                    episode_rows.append(source_row)

                    source_stats = _recompute_episode_stats(
                        table, stats.get(old_episode_index, {"stats": {}})
                    )
                    source_stats["episode_index"] = new_episode_index
                    stats_rows.append(source_stats)

                    source_raw = source / "raw" / f"episode_{old_episode_index:06d}.npz"
                    if source_raw.is_file():
                        destination_raw = candidate / "raw" / f"episode_{new_episode_index:06d}.npz"
                        self._rewrite_raw(
                            source_raw,
                            destination_raw,
                            frame_count=frame_count,
                            episode_index=new_episode_index,
                            global_offset=total_frames,
                            task_indexes=np.asarray(mapped_task_ids, dtype=np.int64),
                            episode_row=source_row,
                            update=update,
                        )

                    total_frames += frame_count
                    new_episode_index += 1

            if new_episode_index <= 0:
                raise ValueError("resulting dataset would be empty")
            _atomic_jsonl(candidate / "meta" / "tasks.jsonl", task_rows)
            _atomic_jsonl(candidate / "meta" / "episodes.jsonl", episode_rows)
            _atomic_jsonl(candidate / "meta" / "episodes_stats.jsonl", stats_rows)
            updated_info = dict(base_info)
            old_total = int(base_info.get("total_episodes", len(_parquet_paths(base))))
            chunk_size = int(base_info.get("chunks_size", 1000))
            updated_info.update(
                {
                    "total_episodes": new_episode_index,
                    "total_frames": total_frames,
                    "total_tasks": len(task_rows),
                    "total_videos": total_videos,
                    "total_chunks": math.ceil(new_episode_index / chunk_size),
                    "splits": _build_splits(base_info.get("splits"), old_total, new_episode_index),
                }
            )
            if base_contract.get("legacy"):
                legacy_metadata = {
                    key: value
                    for key, value in base_contract.items()
                    if key
                    in {
                        "schema", "arm_mode", "arm_side", "state_dim", "raw_action_dim",
                        "model_action_dim", "state_names", "action_names", "contract_format",
                        "legacy", "legacy_format", "delivery_action_format", "action_semantics",
                        "action_source", "action_alignment", "action_offset", "action_horizon",
                        "gripper_semantics", "rotation_semantics", "coordinate_frame",
                    }
                }
                legacy_metadata["action_dim"] = base_contract["raw_action_dim"]
                legacy_metadata["contract_version"] = 2
                legacy_metadata["legacy_delivery_v2"] = True
                updated_info.update(legacy_metadata)
                state_key = "state" if "state" in updated_info.get("features", {}) else "observation.state"
                action_key = "actions" if "actions" in updated_info.get("features", {}) else "action"
                if base_contract.get("state_names") is not None:
                    updated_info["features"][state_key]["names"] = base_contract["state_names"]
                if base_contract.get("action_names") is not None:
                    updated_info["features"][action_key]["names"] = base_contract["action_names"]
            _atomic_json(candidate / "meta" / "info.json", updated_info)
            policy_contract = _read_json(candidate / "meta" / "policy_contract.json", {})
            if not isinstance(policy_contract, dict):
                policy_contract = {}
            if base_contract:
                policy_contract.update(
                    {
                        "version": updated_info.get("contract_version", policy_contract.get("version", 2)),
                        "robot_type": updated_info.get("robot_type"),
                        **{
                            key: updated_info[key]
                            for key in (
                                "schema", "arm_mode", "arm_side", "state_dim", "action_dim",
                                "raw_action_dim", "model_action_dim", "state_names", "action_names",
                                "camera_keys", "contract_format", "legacy", "legacy_format",
                                "legacy_delivery_v2", "delivery_action_format", "action_semantics", "action_source",
                                "action_alignment", "action_offset", "action_horizon",
                                "gripper_semantics", "rotation_semantics", "coordinate_frame",
                            )
                            if key in updated_info
                        },
                    }
                )
                _atomic_json(candidate / "meta" / "policy_contract.json", policy_contract)
            return candidate
        except Exception:
            if candidate.exists():
                shutil.rmtree(candidate)
            raise

    @staticmethod
    def _copy_episode_images(
        source: Path,
        info: dict[str, Any],
        table: pa.Table,
        *,
        old_episode_index: int,
        candidate: Path,
        new_episode_index: int,
    ) -> None:
        for image_key in _image_keys(info):
            if table.schema.get_field_index(image_key) < 0:
                raise ValueError(f"episode parquet is missing image column {image_key}")
            for frame_index in range(table.num_rows):
                cell = _image_cell(table, image_key, frame_index)
                external = _external_image_path(
                    source,
                    info,
                    image_key,
                    old_episode_index,
                    frame_index,
                    cell.get("path"),
                )
                if external is None:
                    blob = cell.get("bytes")
                    if isinstance(blob, (bytes, bytearray, memoryview)) and blob:
                        continue
                    raise FileNotFoundError(
                        f"image frame payload is missing: dataset={source.name} "
                        f"episode={old_episode_index} key={image_key} frame={frame_index}"
                    )
                suffix = external.suffix.lower() or Path(str(cell.get("path") or "")).suffix.lower() or ".png"
                stored_relative = _safe_relative_path(cell.get("path"))
                destination = candidate / stored_relative if stored_relative is not None else (
                    candidate
                    / "images"
                    / image_key
                    / f"episode_{new_episode_index:06d}"
                    / f"frame_{frame_index:06d}{suffix}"
                )
                _link_or_copy(external, destination)

    @staticmethod
    def _rewrite_raw(
        source: Path,
        destination: Path,
        *,
        frame_count: int,
        episode_index: int,
        global_offset: int,
        task_indexes: np.ndarray,
        episode_row: dict[str, Any],
        update: dict[str, Any],
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with np.load(source, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        payload["episode_index"] = np.full(frame_count, episode_index, dtype=np.int64)
        payload["frame_index"] = np.arange(frame_count, dtype=np.int64)
        payload["index"] = np.arange(global_offset, global_offset + frame_count, dtype=np.int64)
        if "task_index" in payload:
            payload["task_index"] = task_indexes
        tasks = episode_row.get("tasks", [])
        if tasks:
            payload["instruction"] = np.asarray(str(tasks[0]))
        if "task_name" in update and "task_name" not in episode_row:
            payload.pop("task_name", None)
        elif "task_name" in episode_row:
            payload["task_name"] = np.asarray(str(episode_row["task_name"]))
        if "success" in update and "success" not in episode_row:
            payload.pop("success", None)
        elif "success" in episode_row:
            payload["success"] = np.asarray(bool(episode_row["success"]), dtype=np.bool_)
        for key in update.get("remove_metadata", []):
            payload.pop(f"meta.{key}", None)
        for key, value in update.get("metadata", {}).items():
            payload[f"meta.{key}"] = DatasetEditor._raw_metadata_array(value)
        temp = destination.with_name(destination.name + f".tmp-{uuid.uuid4().hex}.npz")
        np.savez_compressed(temp, **payload)
        os.replace(temp, destination)

    @staticmethod
    def _raw_metadata_array(value: Any) -> np.ndarray:
        """Store JSON metadata without creating pickle-dependent object arrays."""
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = np.asarray(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        if array.dtype.hasobject:
            array = np.asarray(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        return array

    def _commit(self, dataset_id: str, candidate: Path) -> dict[str, Any]:
        target = self._dataset_path(dataset_id)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = self.dataset_root / f".{dataset_id}.backup-{stamp}-{uuid.uuid4().hex[:8]}"
        failed = self.dataset_root / f".{dataset_id}.failed-{uuid.uuid4().hex}"
        had_target = target.exists()
        if had_target:
            os.replace(target, backup)
        try:
            os.replace(candidate, target)
            loader_output = self.validate_installed(dataset_id)
        except Exception:
            if target.exists():
                os.replace(target, failed)
            if had_target and backup.exists():
                os.replace(backup, target)
            raise
        invalidated = self._invalidate_norm_stats(dataset_id)
        info = _read_json(target / "meta" / "info.json", {})
        return {
            "dataset_id": dataset_id,
            "path": str(target),
            "backup": str(backup) if had_target else None,
            "loader_validation": loader_output,
            "norm_stats_invalidated": bool(invalidated),
            "norm_stats_invalidated_paths": invalidated,
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
        }

    def _invalidate_norm_stats(self, dataset_id: str) -> list[str]:
        invalidated: list[str] = []
        for config_name in POLICY_CONFIG_NAMES:
            norm_path = self.assets_base_dir / config_name / dataset_id / "norm_stats.json"
            if not norm_path.is_file():
                continue
            destination = norm_path.with_name(
                f"norm_stats.invalidated-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
            )
            try:
                os.replace(norm_path, destination)
            except OSError:
                norm_path.unlink(missing_ok=True)
                invalidated.append(str(norm_path))
            else:
                invalidated.append(str(destination))
        return invalidated
