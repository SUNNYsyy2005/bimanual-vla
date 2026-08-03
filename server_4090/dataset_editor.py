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
}
COMPATIBILITY_FIELDS = (
    "codebase_version",
    "robot_type",
    "fps",
    "chunks_size",
    "features",
    "action_semantics",
    "action_offset",
)
POLICY_CONFIG_NAMES = (
    "pi05_piper_single_arm_lora",
    "pi05_piper_bimanual_lora",
)


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


def _validate_compatible(target_info: dict[str, Any], source_info: dict[str, Any]) -> None:
    mismatches = []
    for key in COMPATIBILITY_FIELDS:
        if target_info.get(key) != source_info.get(key):
            mismatches.append(key)
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
            "info": info,
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

    def install_upload(self, dataset_id: str, extracted: Path, *, overwrite: bool, merge: bool) -> dict[str, Any]:
        if overwrite and merge:
            raise ValueError("overwrite and merge are mutually exclusive")
        with self._lock(dataset_id):
            target = self._dataset_path(dataset_id)
            source_validation = self.validate_staging(extracted)
            operation = "install"
            candidate = extracted
            final_validation = source_validation
            if target.exists():
                self.assert_idle(dataset_id)
                if merge:
                    operation = "merge"
                    candidate = self._rebuild_candidate(dataset_id, [(target, None), (extracted, None)])
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

                    source_stats = dict(stats.get(old_episode_index, {"stats": {}}))
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
            _atomic_json(candidate / "meta" / "info.json", updated_info)
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
                destination = (
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
