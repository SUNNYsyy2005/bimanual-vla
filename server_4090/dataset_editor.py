#!/usr/bin/env python3
"""Atomic episode-level editing and merging for local LeRobot datasets.

The editor never changes frame payloads.  It may rewrite parquet index columns and
optional raw NPZ metadata when episodes are merged, removed, or renumbered.
Videos are hard-linked when possible and are never re-encoded.
"""

from __future__ import annotations

import contextlib
import json
import math
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


def _video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        return []
    return sorted(
        key for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


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
                    "task_name": row.get("task_name", ""),
                    "success": row.get("success"),
                    "parameters": parameters,
                    "video_keys": _video_keys(info),
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
            task_name = str(updates["task_name"]).strip()
            if len(task_name) > 256:
                raise ValueError("task_name must not exceed 256 characters")
            result["task_name"] = task_name
        if "success" in updates:
            if not isinstance(updates["success"], bool):
                raise ValueError("success must be a JSON boolean")
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
        for name in ("data", "videos", "raw"):
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

                    source_row["episode_index"] = new_episode_index
                    source_row["length"] = frame_count
                    source_row["tasks"] = _unique_strings(task_texts)
                    if "task_name" in update:
                        source_row["task_name"] = update["task_name"]
                    if "success" in update:
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
        if "task_name" in episode_row:
            payload["task_name"] = np.asarray(str(episode_row["task_name"]))
        if "success" in episode_row:
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
            "norm_stats_invalidated": invalidated,
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
        }

    def _invalidate_norm_stats(self, dataset_id: str) -> str | None:
        norm_path = self.assets_base_dir / "pi05_piper_single_arm_lora" / dataset_id / "norm_stats.json"
        if not norm_path.is_file():
            return None
        destination = norm_path.with_name(
            f"norm_stats.invalidated-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
        )
        try:
            os.replace(norm_path, destination)
        except OSError:
            norm_path.unlink(missing_ok=True)
            return str(norm_path)
        return str(destination)
