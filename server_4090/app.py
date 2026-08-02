#!/usr/bin/env python3
"""Authenticated dashboard for dataset upload, π0.5 fine-tuning, and serving."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import threading
import time
from typing import Any
from urllib.request import urlopen
import uuid

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

try:
    from .dataset_editor import DatasetEditor, DatasetValidationError
except ImportError:  # app.py is normally executed directly by start_server.sh
    from dataset_editor import DatasetEditor, DatasetValidationError


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TASK_TYPES = {"norm", "train", "policy"}
PROCESS_STATES = {"starting", "running", "stopping"}
WAITING_STATES = {"waiting_norm", "waiting_gpu"}
TERMINAL_STATES = {"completed", "failed", "lost", "stopped"}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def safe_name(value: Any, label: str = "name") -> str:
    value = str(value or "")
    if not SAFE_NAME.fullmatch(value) or value in {".", ".."} or ".." in value:
        raise ValueError(f"invalid {label}: use letters, numbers, dot, underscore, or dash")
    return value


def safe_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return parsed


def resolve_under(value: str | Path, roots: list[Path], *, must_exist: bool = True) -> Path:
    candidate = Path(value).expanduser().resolve()
    if not any(candidate == root or candidate.is_relative_to(root) for root in roots):
        raise ValueError(f"path is outside allowed roots: {candidate}")
    if must_exist and not candidate.exists():
        raise ValueError(f"path does not exist: {candidate}")
    return candidate


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # A zombie still responds to signal 0 but no longer represents a usable task.
        return stat.rsplit(")", 1)[1].strip().split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def process_matches_task(pid: int, task: dict[str, Any]) -> bool:
    """Prevent stale task files from targeting an unrelated reused PID."""
    command = task.get("command")
    if not isinstance(command, list) or len(command) < 3:
        return False
    try:
        running = [
            item.decode("utf-8", errors="surrogateescape")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        ]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    if len(running) < 3:
        return False
    try:
        same_entrypoint = Path(running[1]).resolve() == Path(str(command[1])).resolve()
    except (OSError, RuntimeError):
        same_entrypoint = running[1] == str(command[1])
    return same_entrypoint and running[2] == str(command[2])


def process_cmdline(pid: int) -> list[str]:
    try:
        return [
            item.decode("utf-8", errors="surrogateescape")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        ]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []


def _cmd_arg(command: list[str], flag: str) -> str | None:
    prefix = flag + "="
    for item in command:
        if item.startswith(prefix):
            return item[len(prefix):]
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _is_policy_command(command: list[str]) -> bool:
    joined = " ".join(command)
    return (
        "serve_policy.py" in joined
        or ("openpi_single_arm.py" in joined and "serve" in command)
    )


def _policy_port_from_command(command: list[str]) -> int | None:
    value = _cmd_arg(command, "--port")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError(f"invalid config JSON: {path}")
    defaults = {
        "host": "0.0.0.0",
        "port": 8090,
        "allowed_gpu_ids": [0, 1, 2, 3],
        "allow_busy_gpus": False,
        "max_upload_gib": 500,
        "max_chunk_mib": 64,
        "policy_port_min": 8000,
        "policy_port_max": 8099,
        "robot_observation_max_age_s": 3.0,
        "task_monitor_interval_s": 2.0,
    }
    defaults.update(config)
    for key in (
        "openpi_repo",
        "openpi_python",
        "dataset_root",
        "workspace_root",
        "assets_base_dir",
        "checkpoint_base_dir",
        "base_checkpoint",
    ):
        defaults[key] = str(Path(defaults[key]).expanduser().resolve())
    defaults["checkpoint_allowed_roots"] = [
        str(Path(item).expanduser().resolve()) for item in defaults.get("checkpoint_allowed_roots", [])
    ]
    return defaults


class TaskManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.root = Path(config["workspace_root"]) / "tasks"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processes: dict[str, subprocess.Popen] = {}
        self.monitor_interval_s = float(config.get("task_monitor_interval_s", 2.0))
        self.monitor_wakeup = threading.Event()
        self.monitor_stop = threading.Event()
        self.monitor_thread: threading.Thread | None = None
        if self.monitor_interval_s > 0:
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="task-dependency-monitor",
                daemon=True,
            )
            self.monitor_thread.start()

    def _path(self, task_id: str) -> Path:
        return self.root / safe_name(task_id, "task id") / "task.json"

    def _log_path(self, task_id: str) -> Path:
        return self.root / safe_name(task_id, "task id") / "task.log"

    def _monitor_loop(self) -> None:
        while not self.monitor_stop.is_set():
            self.monitor_wakeup.wait(self.monitor_interval_s)
            self.monitor_wakeup.clear()
            if self.monitor_stop.is_set():
                return
            try:
                self.list()
            except Exception:
                logging.getLogger(__name__).exception("task dependency monitor failed")

    def close(self) -> None:
        self.monitor_stop.set()
        self.monitor_wakeup.set()
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=max(1.0, self.monitor_interval_s + 0.5))

    def _append_log(self, task: dict[str, Any], message: str) -> None:
        with self._log_path(task["id"]).open("a", encoding="utf-8") as handle:
            handle.write(f"[{now_iso()}] {message}\n")

    def _new_task(
        self,
        task_type: str,
        command: list[str],
        metadata: dict[str, Any],
        *,
        state: str,
    ) -> dict[str, Any]:
        if task_type not in TASK_TYPES:
            raise ValueError("unsupported task type")
        task_id = f"{task_type}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        task_dir = self.root / task_id
        task_dir.mkdir(parents=True)
        task = {
            "id": task_id,
            "type": task_type,
            "state": state,
            "created_at": now_iso(),
            "command": command,
            "metadata": dict(metadata),
            "log_path": str(task_dir / "task.log"),
        }
        atomic_json(task_dir / "task.json", task)
        return task

    def _launch(
        self,
        task: dict[str, Any],
        *,
        env: dict[str, str],
        raise_on_error: bool,
    ) -> dict[str, Any]:
        task_id = task["id"]
        task["state"] = "starting"
        task["launch_attempted_at"] = now_iso()
        task.pop("waiting_reason", None)
        atomic_json(self._path(task_id), task)
        log_handle = self._log_path(task_id).open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                task["command"],
                cwd=self.config["openpi_repo"],
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            task.update(
                {
                    "state": "failed",
                    "start_error": str(exc),
                    "finished_at": now_iso(),
                }
            )
            atomic_json(self._path(task_id), task)
            self._append_log(task, f"process launch failed: {exc}")
            if raise_on_error:
                raise
            return task
        finally:
            log_handle.close()
        task.update({"state": "running", "pid": process.pid, "started_at": now_iso()})
        atomic_json(self._path(task_id), task)
        self.processes[task_id] = process
        self.monitor_wakeup.set()
        return task

    def _fail_dependency(self, task: dict[str, Any], reason: str) -> dict[str, Any]:
        task.update(
            {
                "state": "failed",
                "dependency_failed": True,
                "dependency_error": reason,
                "finished_at": now_iso(),
            }
        )
        atomic_json(self._path(task["id"]), task)
        self._append_log(task, reason)
        return task

    def _gpu_wait_reason(self, task: dict[str, Any]) -> str | None:
        if self.config.get("allow_busy_gpus", False):
            return None
        gpu_ids = [int(item) for item in task.get("metadata", {}).get("gpu_ids", [])]
        if not gpu_ids:
            return "queued training task has no GPU ids"
        requested = set(gpu_ids)
        managed_busy: dict[int, list[str]] = {}
        for path in self.root.glob("*/task.json"):
            other = read_json(path)
            if not isinstance(other, dict) or other.get("id") == task["id"]:
                continue
            if other.get("state") not in PROCESS_STATES or other.get("type") not in {"train", "policy"}:
                continue
            overlap = requested.intersection(other.get("metadata", {}).get("gpu_ids", []))
            for gpu_id in overlap:
                managed_busy.setdefault(gpu_id, []).append(other["id"])
        if managed_busy:
            return f"waiting for managed task(s) on GPU(s): {managed_busy}"
        inventory = {gpu["index"]: gpu for gpu in gpu_inventory()}
        external_busy = {
            gpu_id: inventory.get(gpu_id, {}).get("processes", [])
            for gpu_id in gpu_ids
            if inventory.get(gpu_id, {}).get("processes")
        }
        if external_busy:
            return f"waiting for busy GPU(s): {external_busy}"
        return None

    def _refresh_waiting(self, task: dict[str, Any]) -> dict[str, Any]:
        dependency = task.get("dependency")
        if not isinstance(dependency, dict):
            return self._fail_dependency(task, "queued training task has no dependency record")
        dependency_id = dependency.get("task_id")
        artifact = dependency.get("artifact")
        if not dependency_id or not artifact:
            return self._fail_dependency(task, "queued training dependency is incomplete")

        dependency_task = read_json(self._path(str(dependency_id)))
        if isinstance(dependency_task, dict):
            dependency_task = self._refresh(dependency_task)
            task["dependency_state"] = dependency_task.get("state")
        else:
            task["dependency_state"] = "missing"

        dependency_state = task.get("dependency_state")
        artifact_exists = Path(str(artifact)).is_file()
        dependency_ready = dependency_state == "completed" or (
            dependency_state == "lost" and artifact_exists
        )
        if not dependency_ready:
            if dependency_state in TERMINAL_STATES or dependency_state == "missing":
                detail = None
                if isinstance(dependency_task, dict):
                    detail = dependency_task.get("start_error") or dependency_task.get("lost_reason")
                    if detail is None and dependency_task.get("returncode") is not None:
                        detail = f"returncode={dependency_task['returncode']}"
                suffix = f": {detail}" if detail else ""
                return self._fail_dependency(
                    task,
                    f"normalization dependency {dependency_id} ended as {dependency_state} without {artifact}{suffix}",
                )
            if task.get("state") != "waiting_norm":
                task["state"] = "waiting_norm"
                task.pop("waiting_reason", None)
            atomic_json(self._path(task["id"]), task)
            return task

        if not artifact_exists:
            return self._fail_dependency(task, f"normalization dependency {dependency_id} completed but missing {artifact}")

        if dependency_state == "lost":
            task["dependency_recovered_from_artifact"] = True

        wait_reason = self._gpu_wait_reason(task)
        if wait_reason:
            changed = task.get("state") != "waiting_gpu" or task.get("waiting_reason") != wait_reason
            task["state"] = "waiting_gpu"
            task["waiting_reason"] = wait_reason
            if changed:
                atomic_json(self._path(task["id"]), task)
            return task
        task["dependency_resolved_at"] = now_iso()
        task.pop("waiting_reason", None)
        self._append_log(
            task,
            f"normalization dependency {dependency_id} is ready; starting training",
        )
        return self._launch(
            task,
            env=build_environment(self.config, task.get("metadata", {}).get("gpu_ids", [])),
            raise_on_error=False,
        )

    def _refresh(self, task: dict[str, Any]) -> dict[str, Any]:
        if task.get("state") in WAITING_STATES:
            return self._refresh_waiting(task)
        if task.get("state") not in PROCESS_STATES:
            return task
        task_id = task["id"]
        process = self.processes.get(task_id)
        if process is not None:
            rc = process.poll()
            if rc is None:
                task["state"] = "running" if task["state"] != "stopping" else "stopping"
                return task
            was_stopping = task["state"] == "stopping"
            task["state"] = "stopped" if was_stopping else ("completed" if rc == 0 else "failed")
            task["returncode"] = rc
            task["finished_at"] = now_iso()
            atomic_json(self._path(task_id), task)
            self.processes.pop(task_id, None)
            return task
        pid = int(task.get("pid", 0))
        if pid and pid_alive(pid) and process_matches_task(pid, task):
            return task
        task["state"] = "stopped" if task.get("state") == "stopping" else "lost"
        if pid and pid_alive(pid):
            task["lost_reason"] = "PID is alive but no longer matches the recorded task command"
        task["finished_at"] = now_iso()
        atomic_json(self._path(task_id), task)
        return task

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            tasks = []
            for path in self.root.glob("*/task.json"):
                task = read_json(path)
                if isinstance(task, dict):
                    tasks.append(self._refresh(task))
            return sorted(tasks, key=lambda item: item.get("created_at", ""), reverse=True)

    def get(self, task_id: str) -> dict[str, Any]:
        with self.lock:
            task = read_json(self._path(task_id))
            if not isinstance(task, dict):
                raise FileNotFoundError(task_id)
            return self._refresh(task)

    def start(
        self,
        task_type: str,
        command: list[str],
        *,
        env: dict[str, str],
        metadata: dict[str, Any],
        raise_on_error: bool = True,
    ) -> dict[str, Any]:
        with self.lock:
            task = self._new_task(task_type, command, metadata, state="starting")
            return self._launch(task, env=env, raise_on_error=raise_on_error)

    def create_waiting_train(
        self,
        command: list[str],
        *,
        metadata: dict[str, Any],
        norm_task: dict[str, Any],
        norm_path: Path,
    ) -> dict[str, Any]:
        with self.lock:
            metadata = {
                **metadata,
                "depends_on": norm_task["id"],
                "dependency_type": "norm",
                "norm_path": str(norm_path),
            }
            task = self._new_task("train", command, metadata, state="waiting_norm")
            task["queued_at"] = now_iso()
            task["dependency"] = {
                "task_id": norm_task["id"],
                "type": "norm",
                "artifact": str(norm_path),
            }
            task["dependency_state"] = norm_task.get("state")
            atomic_json(self._path(task["id"]), task)
            self._append_log(task, f"waiting for normalization dependency {norm_task['id']}")
            self.monitor_wakeup.set()
            return task

    def _active_policy_pids(self) -> set[int]:
        pids: set[int] = set()
        for path in self.root.glob("*/task.json"):
            task = read_json(path)
            if not isinstance(task, dict):
                continue
            if task.get("type") != "policy" or task.get("state") not in PROCESS_STATES:
                continue
            if task.get("metadata", {}).get("external"):
                continue
            try:
                pids.add(int(task.get("pid", 0)))
            except (TypeError, ValueError):
                continue
        pids.discard(0)
        return pids

    def adopt_external_policy(
        self,
        *,
        pid: int,
        command: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = f"policy-external-{pid}"
        task_dir = self.root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        task = read_json(task_dir / "task.json")
        if not isinstance(task, dict):
            task = {
                "id": task_id,
                "type": "policy",
                "state": "running",
                "pid": pid,
                "created_at": now_iso(),
                "started_at": None,
                "discovered_at": now_iso(),
                "command": command,
                "metadata": metadata,
                "log_path": str(task_dir / "task.log"),
            }
            atomic_json(task_dir / "task.json", task)
            self._append_log(task, "adopted external Policy process discovered on a managed port")
            return task
        task["command"] = command
        task["metadata"] = {**task.get("metadata", {}), **metadata}
        task["pid"] = pid
        if pid_alive(pid) and process_matches_task(pid, task) and task.get("state") not in {"stopping"}:
            task["state"] = "running"
            task.pop("finished_at", None)
            task.pop("returncode", None)
            task.pop("lost_reason", None)
        task["last_discovered_at"] = now_iso()
        atomic_json(task_dir / "task.json", task)
        return self._refresh(task)

    def discover_external_policies(self) -> list[dict[str, Any]]:
        with self.lock:
            active_pids = self._active_policy_pids()
            adopted = []
            for candidate in discover_external_policy_candidates(self.config, ignored_pids=active_pids):
                adopted.append(self.adopt_external_policy(**candidate))
            return adopted

    def stop(self, task_id: str, *, force: bool = False) -> dict[str, Any]:
        with self.lock:
            task = self.get(task_id)
            if task["state"] in WAITING_STATES:
                task["state"] = "stopped"
                task["stop_requested_at"] = now_iso()
                task["finished_at"] = now_iso()
                task["stop_reason"] = "queued task cancelled before process launch"
                atomic_json(self._path(task_id), task)
                self._append_log(task, task["stop_reason"])
                return task
            if task["state"] not in PROCESS_STATES:
                return task
            pid = int(task["pid"])
            process = self.processes.get(task_id)
            if process is None and not process_matches_task(pid, task):
                task["state"] = "lost"
                task["lost_reason"] = "refused to signal a PID that does not match the recorded task command"
                task["finished_at"] = now_iso()
                atomic_json(self._path(task_id), task)
                return task
            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, sig)
                else:
                    os.kill(pid, sig)
            except ProcessLookupError:
                pass
            task["state"] = "stopping"
            task["stop_requested_at"] = now_iso()
            task["stop_signal"] = "SIGKILL" if force else "SIGTERM"
            atomic_json(self._path(task_id), task)
            return task

    def log_tail(self, task_id: str, max_bytes: int = 64 * 1024) -> str:
        path = self._log_path(task_id)
        if not path.exists():
            return ""
        with path.open("rb") as handle:
            size = path.stat().st_size
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")


class UploadManager:
    def __init__(self, config: dict[str, Any], dataset_editor: DatasetEditor):
        self.config = config
        self.dataset_editor = dataset_editor
        self.root = Path(config["workspace_root"]) / "uploads"
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_root = Path(config["dataset_root"])
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.locks: dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()

    def _lock(self, upload_id: str) -> threading.Lock:
        with self.global_lock:
            return self.locks.setdefault(upload_id, threading.Lock())

    def _dir(self, upload_id: str) -> Path:
        return self.root / safe_name(upload_id, "upload id")

    def _state(self, upload_id: str) -> dict[str, Any]:
        state = read_json(self._dir(upload_id) / "upload.json")
        if not isinstance(state, dict):
            raise FileNotFoundError(upload_id)
        return state

    @staticmethod
    def _part_path(upload_dir: Path, index: int) -> Path:
        return upload_dir / "chunks" / f"{index:08d}.part"

    def _received(self, state: dict[str, Any], upload_dir: Path) -> list[int]:
        result = []
        count = int(state["chunk_count"])
        size = int(state["size"])
        chunk_size = int(state["chunk_size"])
        for index in range(count):
            path = self._part_path(upload_dir, index)
            expected = min(chunk_size, size - index * chunk_size)
            if path.exists() and path.stat().st_size == expected:
                result.append(index)
        return result

    def initialize(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_name = safe_name(payload.get("dataset_name"), "dataset name")
        size = safe_int(payload.get("size"), "size", 1, int(self.config["max_upload_gib"] * 1024**3))
        chunk_size = safe_int(
            payload.get("chunk_size"),
            "chunk_size",
            1024 * 1024,
            int(self.config["max_chunk_mib"] * 1024**2),
        )
        sha256 = str(payload.get("sha256", "")).lower()
        if not HEX_SHA256.fullmatch(sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        overwrite = bool(payload.get("overwrite", False))
        merge = bool(payload.get("merge", False))
        if overwrite and merge:
            raise ValueError("overwrite and merge are mutually exclusive")
        upload_id = hashlib.sha256(f"{dataset_name}\0{size}\0{sha256}".encode()).hexdigest()[:32]
        upload_dir = self._dir(upload_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / "chunks").mkdir(exist_ok=True)
        state_path = upload_dir / "upload.json"
        state = read_json(state_path)
        expected = {
            "id": upload_id,
            "dataset_name": dataset_name,
            "size": size,
            "sha256": sha256,
            "chunk_size": chunk_size,
            "chunk_count": (size + chunk_size - 1) // chunk_size,
            "overwrite": overwrite,
            "merge": merge,
        }
        if isinstance(state, dict):
            for key in ("dataset_name", "size", "sha256", "chunk_size"):
                if state.get(key) != expected[key]:
                    raise ValueError(f"existing upload metadata mismatch for {key}")
            state["overwrite"] = overwrite
            state["merge"] = merge
        else:
            state = {**expected, "created_at": now_iso(), "state": "uploading"}
        atomic_json(state_path, state)
        return {**state, "received": self._received(state, upload_dir)}

    def status(self, upload_id: str) -> dict[str, Any]:
        state = self._state(upload_id)
        return {**state, "received": self._received(state, self._dir(upload_id))}

    def put_chunk(self, upload_id: str, index: int, body, content_length: int, chunk_sha: str) -> dict[str, Any]:
        with self._lock(upload_id):
            state = self._state(upload_id)
            count = int(state["chunk_count"])
            if not 0 <= index < count:
                raise ValueError(f"chunk index must be in [0, {count - 1}]")
            expected = min(int(state["chunk_size"]), int(state["size"]) - index * int(state["chunk_size"]))
            if content_length != expected:
                raise ValueError(f"chunk length {content_length} != expected {expected}")
            if not HEX_SHA256.fullmatch(chunk_sha):
                raise ValueError("missing or invalid X-Chunk-SHA256")
            path = self._part_path(self._dir(upload_id), index)
            temp = path.with_suffix(".incoming")
            digest = hashlib.sha256()
            written = 0
            with temp.open("wb") as output:
                while written < expected:
                    block = body.read(min(8 * 1024 * 1024, expected - written))
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    written += len(block)
                output.flush()
                os.fsync(output.fileno())
            if written != expected or digest.hexdigest() != chunk_sha:
                temp.unlink(missing_ok=True)
                raise ValueError("chunk size or SHA256 mismatch")
            os.replace(temp, path)
            return {"index": index, "size": written, "sha256": chunk_sha}

    @staticmethod
    def _extract_tar(archive: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        seen: set[str] = set()
        with tarfile.open(archive, mode="r:") as tar:
            members = tar.getmembers()
            if len(members) > 1_000_000:
                raise ValueError("archive contains too many members")
            for member in members:
                name = PurePosixPath(member.name)
                if name.is_absolute() or not name.parts or any(part in {"", ".", ".."} for part in name.parts):
                    raise ValueError(f"unsafe archive path: {member.name!r}")
                if member.name in seen:
                    raise ValueError(f"duplicate archive path: {member.name!r}")
                seen.add(member.name)
                target = destination.joinpath(*name.parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f"archive path escapes destination: {member.name!r}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isreg():
                    raise ValueError(f"links/devices are not allowed in archive: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name!r}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                os.chmod(target, 0o644)

    def complete(self, upload_id: str) -> dict[str, Any]:
        with self._lock(upload_id):
            upload_dir = self._dir(upload_id)
            state = self._state(upload_id)
            received = self._received(state, upload_dir)
            if len(received) != int(state["chunk_count"]):
                raise ValueError(f"upload incomplete: {len(received)}/{state['chunk_count']} chunks")
            state["state"] = "assembling"
            atomic_json(upload_dir / "upload.json", state)
            archive = upload_dir / "dataset.tar"
            temp_archive = upload_dir / "dataset.tar.assembling"
            digest = hashlib.sha256()
            total = 0
            with temp_archive.open("wb") as output:
                for index in range(int(state["chunk_count"])):
                    part = self._part_path(upload_dir, index)
                    with part.open("rb") as source:
                        while block := source.read(8 * 1024 * 1024):
                            output.write(block)
                            digest.update(block)
                            total += len(block)
                output.flush()
                os.fsync(output.fileno())
            if total != int(state["size"]) or digest.hexdigest() != state["sha256"]:
                temp_archive.unlink(missing_ok=True)
                state["state"] = "failed"
                state["error"] = "assembled archive size or SHA256 mismatch"
                atomic_json(upload_dir / "upload.json", state)
                raise ValueError(state["error"])
            os.replace(temp_archive, archive)

            dataset_name = state["dataset_name"]
            staging = self.dataset_root / f".{dataset_name}.installing-{uuid.uuid4().hex}"
            try:
                self._extract_tar(archive, staging)
                result = self.dataset_editor.install_upload(
                    dataset_name,
                    staging,
                    overwrite=bool(state.get("overwrite")),
                    merge=bool(state.get("merge")),
                )
                state.pop("error", None)
                state.pop("failed_at", None)
                state.update(result)
                state.update({"state": "installed", "installed_at": now_iso(), "path": result["path"]})
                atomic_json(upload_dir / "upload.json", state)
                return state
            except Exception as exc:
                if staging.exists():
                    shutil.rmtree(staging)
                if isinstance(exc, DatasetValidationError):
                    state[f"{exc.phase}_validation"] = exc.output[-16_000:]
                state.update({"state": "failed", "error": str(exc), "failed_at": now_iso()})
                atomic_json(upload_dir / "upload.json", state)
                raise


class PolicyTelemetryStore:
    """Read telemetry and manage the fail-closed execution gate per Policy."""

    def __init__(self, config: dict[str, Any]):
        self.root = Path(config["workspace_root"]) / "policy_telemetry"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_age_s = float(config.get("robot_observation_max_age_s", 3.0))
        if self.max_age_s <= 0:
            raise ValueError("robot_observation_max_age_s must be positive")
        # A Dashboard restart must never preserve an armed browser-side gate.
        for session_dir in self.root.iterdir():
            if session_dir.is_dir() and SAFE_NAME.fullmatch(session_dir.name):
                self._write_control(session_dir, mode="shadow", updated_by="dashboard_restart")

    def create_session(self) -> tuple[str, Path]:
        session = uuid.uuid4().hex
        directory = self.root / session
        directory.mkdir(parents=True, exist_ok=False)
        self._write_control(directory, mode="shadow", updated_by="policy_created")
        return session, directory

    def _session_dir(self, session: str) -> Path:
        return self.root / safe_name(session, "telemetry session")

    @staticmethod
    def _effective_control(value: Any, *, session: str) -> dict[str, Any]:
        now = time.time()
        control = value if isinstance(value, dict) else {}
        requested_mode = control.get("mode") if control.get("mode") in {"shadow", "execute"} else "shadow"
        expires_at = control.get("expires_at")
        try:
            expires_at = float(expires_at) if expires_at is not None else None
        except (TypeError, ValueError):
            expires_at = None
        expired = requested_mode == "execute" and (expires_at is None or expires_at <= now)
        return {
            "mode": "shadow" if expired else requested_mode,
            "requested_mode": requested_mode,
            "revision": int(control.get("revision", 0)),
            "updated_at": control.get("updated_at"),
            "updated_by": control.get("updated_by"),
            "expires_at": expires_at,
            "expired": expired,
            "task_id": control.get("task_id"),
            "session_id": session,
        }

    def _write_control(
        self,
        session_dir: Path,
        *,
        mode: str,
        updated_by: str,
        expires_at: float | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        path = session_dir / "execution_control.json"
        previous = read_json(path, {})
        if not isinstance(previous, dict):
            previous = {}
        payload = {
            "mode": mode,
            "revision": int(previous.get("revision", 0)) + 1,
            "updated_at": time.time(),
            "updated_by": updated_by,
            "expires_at": expires_at if mode == "execute" else None,
            "task_id": task_id if task_id is not None else previous.get("task_id"),
            "session_id": session_dir.name,
        }
        atomic_json(path, payload)
        return self._effective_control(payload, session=session_dir.name)

    def control_for_task(self, task: dict[str, Any]) -> dict[str, Any]:
        if task.get("type") != "policy":
            raise ValueError("execution control is only available for policy tasks")
        session = task.get("metadata", {}).get("telemetry_session")
        if not session:
            return self._effective_control({}, session="")
        session = safe_name(session, "telemetry session")
        value = read_json(self._session_dir(session) / "execution_control.json", {})
        return self._effective_control(value, session=session)

    def set_control(
        self,
        task: dict[str, Any],
        *,
        mode: str,
        expires_in_s: int | None = None,
        updated_by: str = "dashboard",
    ) -> dict[str, Any]:
        if task.get("type") != "policy":
            raise ValueError("execution control is only available for policy tasks")
        if mode not in {"shadow", "execute"}:
            raise ValueError("mode must be shadow or execute")
        if mode == "execute" and task.get("state") != "running":
            raise ValueError("only a running policy can be armed for execution")
        session = task.get("metadata", {}).get("telemetry_session")
        if not session:
            raise ValueError("this policy predates execution control; restart or switch it first")
        expires_at = None
        if mode == "execute":
            if expires_in_s is None:
                expires_in_s = 300
            expires_at = time.time() + safe_int(expires_in_s, "expires_in_s", 10, 3600)
        return self._write_control(
            self._session_dir(str(session)),
            mode=mode,
            updated_by=updated_by,
            expires_at=expires_at,
            task_id=str(task["id"]),
        )

    def bind_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Attach the task id to a newly-created fail-closed control file."""
        return self.set_control(task, mode="shadow", updated_by="policy_started")

    def summary_for_task(self, task: dict[str, Any]) -> dict[str, Any] | None:
        metadata = task.get("metadata", {})
        session = metadata.get("telemetry_session")
        if not session:
            return None
        session_dir = self._session_dir(str(session))
        payload = read_json(session_dir / "latest.json")
        connections = read_json(session_dir / "connections.json")
        payload = payload if isinstance(payload, dict) else {}
        connections = connections if isinstance(connections, dict) else {}
        control = self.control_for_task(task)
        received_at = payload.get("received_at")
        age_s = max(0.0, time.time() - float(received_at)) if received_at is not None else None
        process_active = task.get("state") in {"starting", "running", "stopping"}
        client_connected = process_active and bool(connections.get("client_connected", False))
        client_allow = bool(payload.get("client_allow_execution", False))
        client_state = str(payload.get("client_execution_state", "unknown"))
        dual_gate_open = bool(
            process_active
            and client_connected
            and age_s is not None
            and age_s <= self.max_age_s
            and control["mode"] == "execute"
            and client_allow
        )
        return {
            **payload,
            "task_id": task["id"],
            "policy_port": metadata.get("port"),
            "telemetry_session": session,
            "age_s": round(age_s, 3) if age_s is not None else None,
            "fresh": age_s is not None and age_s <= self.max_age_s,
            "max_age_s": self.max_age_s,
            "client_connected": client_connected,
            "active_clients": int(connections.get("active_clients", 0)) if process_active else 0,
            "client_addresses": connections.get("client_addresses", []) if process_active else [],
            "connection_event": connections.get("event"),
            "connection_updated_at": connections.get("updated_at"),
            "execution_control": control,
            "client_allow_execution": client_allow,
            "client_execution_state": client_state,
            "dual_gate_open": dual_gate_open,
        }

    def latest(self, task_list: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates = []
        for task in task_list:
            if task.get("type") != "policy" or task.get("state") not in {"running", "stopping"}:
                continue
            summary = self.summary_for_task(task)
            if summary is not None and summary.get("received_at") is not None:
                candidates.append(summary)
        if not candidates:
            return None
        return max(candidates, key=lambda item: float(item.get("received_at", 0.0)))

    def image_path(self, session: str, view: str) -> Path:
        if view not in {"cam_high", "cam_wrist"}:
            raise ValueError("view must be cam_high or cam_wrist")
        path = self._session_dir(session) / f"{view}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"no policy telemetry image: {view}")
        return path


def gpu_inventory() -> list[dict[str, Any]]:
    gpu_cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    proc_cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_lines = subprocess.check_output(gpu_cmd, text=True, timeout=10).splitlines()
        process_lines = subprocess.check_output(proc_cmd, text=True, timeout=10).splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    processes: dict[str, list[dict[str, Any]]] = {}
    for line in process_lines:
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) == 4:
            processes.setdefault(parts[0], []).append(
                {"pid": int(parts[1]), "name": parts[2], "memory_mib": int(parts[3])}
            )
    gpus = []
    for line in gpu_lines:
        parts = [part.strip() for part in line.split(",", 4)]
        if len(parts) == 5:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mib": int(parts[3]),
                    "memory_used_mib": int(parts[4]),
                    "processes": processes.get(parts[1], []),
                }
            )
    return gpus


def listening_processes_by_port(port_min: int, port_max: int) -> dict[int, set[int]]:
    try:
        output = subprocess.check_output(["ss", "-H", "-ltnp"], text=True, timeout=5, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    result: dict[int, set[int]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        local = parts[3]
        try:
            port = int(local.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            continue
        if not port_min <= port <= port_max:
            continue
        pids = {int(match.group(1)) for match in re.finditer(r"pid=(\d+)", line)}
        if pids:
            result.setdefault(port, set()).update(pids)
    return result


def discover_external_policy_candidates(
    config: dict[str, Any],
    *,
    ignored_pids: set[int] | None = None,
) -> list[dict[str, Any]]:
    ignored_pids = ignored_pids or set()
    port_min = int(config.get("policy_port_min", 8000))
    port_max = int(config.get("policy_port_max", 8099))
    gpu_by_pid: dict[int, list[int]] = {}
    for gpu in gpu_inventory():
        gpu_id = int(gpu["index"])
        for process in gpu.get("processes", []):
            try:
                gpu_by_pid.setdefault(int(process["pid"]), []).append(gpu_id)
            except (TypeError, ValueError):
                continue
    candidates = []
    for port, pids in listening_processes_by_port(port_min, port_max).items():
        for pid in sorted(pids):
            if pid in ignored_pids:
                continue
            command = process_cmdline(pid)
            if not command or not _is_policy_command(command):
                continue
            command_port = _policy_port_from_command(command)
            metadata = {
                "external": True,
                "adopted": True,
                "source": "listening_policy_port",
                "port": command_port or port,
                "gpu_ids": sorted(gpu_by_pid.get(pid, [])),
                "schema": _cmd_arg(command, "--schema") or "external",
                "dataset_id": _cmd_arg(command, "--dataset-id") or "external",
                "arm_side": _cmd_arg(command, "--arm-side"),
                "checkpoint": _cmd_arg(command, "--checkpoint") or _cmd_arg(command, "--policy.dir"),
                "policy_config": _cmd_arg(command, "--policy.config"),
                "ws_url": f"ws://{socket.gethostname()}:{command_port or port}",
            }
            telemetry_dir = _cmd_arg(command, "--telemetry-dir")
            if telemetry_dir:
                metadata["telemetry_dir"] = telemetry_dir
                metadata["telemetry_session"] = Path(telemetry_dir).name
            candidates.append({"pid": pid, "command": command, "metadata": metadata})
    return candidates


def build_environment(config: dict[str, Any], gpu_ids: list[int] | None) -> dict[str, str]:
    env = os.environ.copy()
    openpi_env_lib = str(Path(config["openpi_python"]).resolve().parent.parent / "lib")
    inherited_ld = env.get("LD_LIBRARY_PATH", "")
    env.update(
        {
            "XDG_CACHE_HOME": str(Path.home() / ".cache"),
            "HF_HOME": str(Path.home() / ".cache" / "huggingface"),
            "HF_LEROBOT_HOME": config["dataset_root"],
            "LD_LIBRARY_PATH": openpi_env_lib + ((":" + inherited_ld) if inherited_ld else ""),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # JAX defaults to a 75% preallocation pool, which is too tight for
            # π0.5 LoRA initialization on a 24 GiB RTX 4090.
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(config.get("xla_memory_fraction", 0.95)),
        }
    )
    if gpu_ids is None:
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("JAX_PLATFORMS", None)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    return env


def create_app(config_path: Path) -> Flask:
    config = load_config(config_path)
    token = os.environ.get("BIMANUAL_VLA_SERVER_TOKEN", "")
    if len(token) < 20:
        raise RuntimeError("set BIMANUAL_VLA_SERVER_TOKEN to a random value of at least 20 characters")
    login_user = os.environ.get("BIMANUAL_VLA_LOGIN_USER", "")
    login_password = os.environ.get("BIMANUAL_VLA_LOGIN_PASSWORD", "")
    app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = int(config["max_chunk_mib"] * 1024**2) + 1024 * 1024
    tasks = TaskManager(config)
    dataset_root = Path(config["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)

    def assert_dataset_idle(dataset_id: str) -> None:
        active = [
            task for task in tasks.list()
            if task.get("metadata", {}).get("dataset_id") == dataset_id
            and task.get("state") not in TERMINAL_STATES
        ]
        if active:
            summary = ", ".join(f"{task['id']} ({task.get('state')})" for task in active)
            raise ValueError(f"dataset is in use by active task(s): {summary}")

    def validate_staging_dataset(path: Path) -> str:
        checker = [config["openpi_python"], str(REPO_DIR / "check_pi05_dataset.py"), str(path)]
        result = subprocess.run(
            checker,
            capture_output=True,
            text=True,
            timeout=3600,
            env=build_environment(config, None),
        )
        output = (result.stdout + result.stderr)[-16_000:]
        if result.returncode != 0:
            raise DatasetValidationError("structural", "dataset structural validation failed", output)
        return output

    def validate_installed_dataset(dataset_id: str) -> str:
        result = subprocess.run(
            [config["openpi_python"], str(APP_DIR / "validate_lerobot.py"), dataset_id],
            cwd=config["openpi_repo"],
            capture_output=True,
            text=True,
            timeout=3600,
            env=build_environment(config, None),
        )
        output = (result.stdout + result.stderr)[-16_000:]
        if result.returncode != 0:
            raise DatasetValidationError("loader", "LeRobot/OpenPI loader validation failed", output)
        return output

    dataset_editor = DatasetEditor(
        dataset_root=dataset_root,
        assets_base_dir=Path(config["assets_base_dir"]),
        validate_staging=validate_staging_dataset,
        validate_installed=validate_installed_dataset,
        assert_idle=assert_dataset_idle,
    )
    uploads = UploadManager(config, dataset_editor)
    observations = PolicyTelemetryStore(config)
    allowed_gpus = set(map(int, config["allowed_gpu_ids"]))
    checkpoint_roots = [Path(item) for item in config["checkpoint_allowed_roots"]]
    openpi_helper = str(APP_DIR / "openpi_single_arm.py")
    checkpoint_size_cache: dict[str, tuple[float, int]] = {}

    @app.before_request
    def authenticate():
        if request.path in {"/", "/healthz", "/api/auth/token"}:
            return None
        supplied = request.headers.get("Authorization", "")
        if supplied.startswith("Bearer "):
            supplied = supplied[7:]
        else:
            supplied = request.headers.get("X-API-Token", "")
        if not hmac.compare_digest(supplied, token):
            return jsonify({"error": "unauthorized"}), 401
        return None

    @app.errorhandler(Exception)
    def handle_error(exc: Exception):
        if isinstance(exc, HTTPException):
            return jsonify({"error": exc.description, "type": type(exc).__name__}), exc.code
        status = 400 if isinstance(exc, (ValueError, FileExistsError, FileNotFoundError)) else 500
        if status == 500:
            app.logger.exception("request failed")
        return jsonify({"error": str(exc), "type": type(exc).__name__}), status

    @app.get("/")
    def index():
        return render_template("index.html", server_port=config["port"])

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "time": now_iso()})

    @app.post("/api/auth/token")
    def issue_token():
        """Exchange the Dashboard login credentials for the existing Bearer token.

        The login endpoint intentionally accepts credentials only in the request
        body, never in the URL.  The returned token remains the same token used
        by the existing Dashboard API and upload clients.
        """
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            payload = request.form.to_dict()
        supplied_user = str(payload.get("username", ""))
        supplied_password = str(payload.get("password", ""))
        if not login_user or not login_password:
            return jsonify({"error": "Dashboard login credentials are not configured"}), 503
        valid_user = hmac.compare_digest(supplied_user, login_user)
        valid_password = hmac.compare_digest(supplied_password, login_password)
        if not (valid_user and valid_password):
            return jsonify({"error": "invalid username or password"}), 401
        return jsonify({"token": token, "token_type": "Bearer", "username": login_user})

    def list_datasets() -> list[dict[str, Any]]:
        datasets = []
        for directory in sorted(dataset_root.iterdir() if dataset_root.exists() else []):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            info = read_json(directory / "meta" / "info.json")
            if not isinstance(info, dict):
                continue
            features = info.get("features", {})
            if features.get("state", {}).get("shape") == [10] and features.get("actions", {}).get("shape") == [7]:
                schema = "delivery"
                state_shape = features["state"]["shape"]
                action_shape = features["actions"]["shape"]
                camera_keys = [key for key in ("image", "wrist_image") if key in features]
                action_semantics = "eef_delta_base_xyz_left_rotvec_gripper_target"
                action_offset = 0
            else:
                schema = "joint"
                state_shape = features.get("observation.state", {}).get("shape")
                action_shape = features.get("action", {}).get("shape")
                camera_keys = [
                    key for key, value in features.items()
                    if value.get("dtype") in {"image", "video"} and key.startswith("observation.images.")
                ]
                action_semantics = info.get("action_semantics") or "absolute_joint_position"
                action_offset = info.get("action_offset")
            datasets.append(
                {
                    "id": directory.name,
                    "path": str(directory),
                    "episodes": info.get("total_episodes"),
                    "frames": info.get("total_frames"),
                    "fps": info.get("fps"),
                    "robot_type": info.get("robot_type"),
                    "schema": schema,
                    "state_shape": state_shape,
                    "action_shape": action_shape,
                    "cameras": camera_keys,
                    "action_semantics": action_semantics,
                    "action_offset": action_offset,
                    "norm_stats_ready": (
                        Path(config["assets_base_dir"])
                        / "pi05_piper_single_arm_lora"
                        / directory.name
                        / "norm_stats.json"
                    ).is_file(),
                    "mtime": directory.stat().st_mtime,
                }
            )
        return datasets

    def list_checkpoints() -> list[dict[str, Any]]:
        config_root = Path(config["checkpoint_base_dir"]) / "pi05_piper_single_arm_lora"
        checkpoints = []
        if not config_root.exists():
            return checkpoints
        for exp_dir in config_root.iterdir():
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            for step_dir in exp_dir.iterdir():
                if not step_dir.is_dir() or not step_dir.name.isdigit():
                    continue
                if not (step_dir / "params").is_dir() or not (step_dir / "_CHECKPOINT_METADATA").is_file():
                    continue
                dataset_ids = sorted(
                    path.parent.name
                    for path in (step_dir / "assets").glob("*/norm_stats.json")
                    if path.is_file()
                )
                mtime = step_dir.stat().st_mtime
                cache_key = str(step_dir.resolve())
                cached = checkpoint_size_cache.get(cache_key)
                if cached is None or cached[0] != mtime:
                    size_bytes = sum(path.stat().st_size for path in step_dir.rglob("*") if path.is_file())
                    checkpoint_size_cache[cache_key] = (mtime, size_bytes)
                else:
                    size_bytes = cached[1]
                checkpoints.append(
                    {
                        "path": cache_key,
                        "experiment": exp_dir.name,
                        "step": int(step_dir.name),
                        "dataset_ids": dataset_ids,
                        "mtime": mtime,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                        "size_gib": round(size_bytes / 1024**3, 2),
                    }
                )
        return sorted(checkpoints, key=lambda item: (item["mtime"], item["step"]), reverse=True)

    @app.get("/api/status")
    def status():
        tasks.discover_external_policies()
        task_list = tasks.list()
        for task in task_list:
            if task["type"] != "policy":
                continue
            task["telemetry"] = observations.summary_for_task(task)
            task["policy_healthy"] = False
            if task["state"] == "running":
                port = task.get("metadata", {}).get("port")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                        task["policy_healthy"] = response.status == 200
                except Exception:
                    pass
        latest_observation = observations.latest(task_list)
        return jsonify(
            {
                "datasets": list_datasets(),
                "checkpoints": list_checkpoints(),
                "robot_observation": latest_observation,
                "tasks": task_list,
                "gpus": gpu_inventory(),
                "config": {
                    "dataset_root": config["dataset_root"],
                    "checkpoint_base_dir": config["checkpoint_base_dir"],
                    "base_checkpoint": config["base_checkpoint"],
                    "allowed_gpu_ids": sorted(allowed_gpus),
                    "allow_busy_gpus": config["allow_busy_gpus"],
                    "policy_port_range": [config["policy_port_min"], config["policy_port_max"]],
                    "robot_observation_max_age_s": observations.max_age_s,
                },
            }
        )

    @app.get("/api/robot/observation")
    def get_robot_observation():
        return jsonify({"observation": observations.latest(tasks.list())})

    @app.get("/api/policy-telemetry/<session>/image/<view>")
    def get_policy_telemetry_image(session: str, view: str):
        return send_file(observations.image_path(session, view), mimetype="image/jpeg", max_age=0, conditional=False)

    @app.post("/api/uploads/init")
    def upload_init():
        return jsonify(uploads.initialize(request.get_json(force=True)))

    @app.get("/api/uploads/<upload_id>")
    def upload_status(upload_id: str):
        return jsonify(uploads.status(upload_id))

    @app.put("/api/uploads/<upload_id>/chunks/<int:index>")
    def upload_chunk(upload_id: str, index: int):
        content_length = request.content_length
        if content_length is None:
            raise ValueError("Content-Length is required")
        chunk_sha = request.headers.get("X-Chunk-SHA256", "").lower()
        return jsonify(uploads.put_chunk(upload_id, index, request.stream, content_length, chunk_sha))

    @app.post("/api/uploads/<upload_id>/complete")
    def upload_complete(upload_id: str):
        return jsonify(uploads.complete(upload_id))

    @app.get("/api/datasets/<dataset_id>")
    def dataset_details(dataset_id: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        offset = safe_int(request.args.get("offset", 0), "offset", 0, 10**9)
        limit = safe_int(request.args.get("limit", 200), "limit", 1, 500)
        return jsonify(dataset_editor.details(dataset_id, offset=offset, limit=limit))

    @app.get("/api/datasets/<dataset_id>/episodes/<int:episode_index>/video/<video_key>")
    def dataset_episode_video(dataset_id: str, episode_index: int, video_key: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        path = dataset_editor.video_path(dataset_id, episode_index, video_key)
        return send_file(path, mimetype="video/mp4", conditional=True, max_age=0)

    @app.patch("/api/datasets/<dataset_id>/episodes/<int:episode_index>")
    def update_dataset_episode(dataset_id: str, episode_index: int):
        dataset_id = safe_name(dataset_id, "dataset id")
        result = dataset_editor.update_episode(dataset_id, episode_index, request.get_json(force=True))
        return jsonify(result)

    @app.post("/api/datasets/<dataset_id>/episodes/delete")
    def delete_dataset_episodes(dataset_id: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        payload = request.get_json(force=True)
        indexes = payload.get("episode_indexes") if isinstance(payload, dict) else None
        if not isinstance(indexes, list):
            raise ValueError("episode_indexes must be a list")
        return jsonify(dataset_editor.delete_episodes(dataset_id, indexes))

    @app.post("/api/datasets/<dataset_id>/merge")
    def merge_installed_dataset(dataset_id: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        payload = request.get_json(force=True)
        source_id = safe_name(payload.get("source_dataset_id") if isinstance(payload, dict) else None, "source dataset id")
        return jsonify(dataset_editor.merge_existing(dataset_id, source_id))

    def parse_dataset(payload: dict[str, Any]) -> tuple[str, str, str]:
        dataset_id = safe_name(payload.get("dataset_id"), "dataset id")
        arm_side = str(payload.get("arm_side", "right"))
        if arm_side not in {"left", "right"}:
            raise ValueError("arm_side must be left or right")
        dataset_path = dataset_root / dataset_id
        info = read_json(dataset_path / "meta" / "info.json")
        if not isinstance(info, dict):
            raise ValueError(f"dataset is not installed: {dataset_id}")
        features = info.get("features", {})
        if (
            features.get("state", {}).get("shape") == [10]
            and features.get("actions", {}).get("shape") == [7]
            and {"image", "wrist_image"}.issubset(features)
        ):
            return dataset_id, arm_side, "delivery"

        state_shape = features.get("observation.state", {}).get("shape")
        action_shape = features.get("action", {}).get("shape")
        cameras = [
            key for key, value in features.items()
            if value.get("dtype") in {"image", "video"}
        ]
        expected_wrist = f"observation.images.cam_{arm_side}_wrist"
        if state_shape != [7] or action_shape != [7]:
            raise ValueError(
                "unsupported dataset schema: expected delivery state/actions [10]/[7] "
                f"or legacy joint [7]/[7], got {state_shape}/{action_shape}"
            )
        if "observation.images.cam_high" not in cameras or expected_wrist not in cameras:
            raise ValueError(f"legacy joint schema requires cam_high + {expected_wrist}, got {cameras}")
        return dataset_id, arm_side, "joint"

    def parse_gpus(
        payload: dict[str, Any],
        *,
        one_only: bool = False,
        ignored_pids: set[int] | None = None,
        check_busy: bool = True,
    ) -> list[int]:
        raw = payload.get("gpu_ids", [0])
        if isinstance(raw, str):
            raw = [item.strip() for item in raw.split(",") if item.strip()]
        if not isinstance(raw, list) or not raw:
            raise ValueError("gpu_ids must be a non-empty list")
        gpu_ids = [safe_int(item, "GPU id", 0, 128) for item in raw]
        if len(set(gpu_ids)) != len(gpu_ids) or not set(gpu_ids).issubset(allowed_gpus):
            raise ValueError(f"GPU ids must be unique and within {sorted(allowed_gpus)}")
        if one_only and len(gpu_ids) != 1:
            raise ValueError("policy serving requires exactly one GPU")
        if not check_busy:
            return gpu_ids
        ignored_pids = ignored_pids or set()
        inventory = {gpu["index"]: gpu for gpu in gpu_inventory()}
        busy = {
            gpu_id: [
                process
                for process in inventory.get(gpu_id, {}).get("processes", [])
                if int(process.get("pid", -1)) not in ignored_pids
            ]
            for gpu_id in gpu_ids
        }
        busy = {gpu_id: procs for gpu_id, procs in busy.items() if procs}
        if busy and not config["allow_busy_gpus"]:
            raise ValueError(f"refusing busy GPU(s): {busy}")
        return gpu_ids

    def norm_stats_path(dataset_id: str) -> Path:
        return Path(config["assets_base_dir"]) / "pi05_piper_single_arm_lora" / dataset_id / "norm_stats.json"

    def build_norm_command(
        dataset_id: str,
        arm_side: str,
        schema: str,
        *,
        batch_size: int,
        num_workers: int,
        max_frames: int | None = None,
    ) -> list[str]:
        command = [
            config["openpi_python"], openpi_helper, "norm",
            "--dataset-id", dataset_id,
            "--arm-side", arm_side,
            "--schema", schema,
            "--assets-base-dir", config["assets_base_dir"],
            "--checkpoint-base-dir", config["checkpoint_base_dir"],
            "--base-checkpoint", config["base_checkpoint"],
            "--batch-size", str(batch_size),
            "--num-workers", str(num_workers),
        ]
        if max_frames is not None:
            command += ["--max-frames", str(max_frames)]
        return command

    @app.post("/api/tasks/norm")
    def start_norm():
        payload = request.get_json(force=True)
        dataset_id, arm_side, schema = parse_dataset(payload)
        batch_size = safe_int(payload.get("batch_size", 16), "batch_size", 1, 1024)
        num_workers = safe_int(payload.get("num_workers", 2), "num_workers", 1, 64)
        max_frames = payload.get("max_frames")
        parsed_max_frames = (
            None if max_frames in (None, "") else safe_int(max_frames, "max_frames", 1, 10**9)
        )
        command = build_norm_command(
            dataset_id,
            arm_side,
            schema,
            batch_size=batch_size,
            num_workers=num_workers,
            max_frames=parsed_max_frames,
        )
        task = tasks.start(
            "norm", command,
            env=build_environment(config, None),
            metadata={
                "dataset_id": dataset_id,
                "arm_side": arm_side,
                "schema": schema,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "max_frames": parsed_max_frames,
                "norm_path": str(norm_stats_path(dataset_id)),
                "automatic": False,
            },
        )
        return jsonify(task), 201

    @app.post("/api/tasks/train")
    def start_train():
        payload = request.get_json(force=True)
        dataset_id, arm_side, schema = parse_dataset(payload)
        norm_path = norm_stats_path(dataset_id)
        gpu_ids = parse_gpus(payload, check_busy=norm_path.is_file())
        exp_name = safe_name(payload.get("exp_name"), "experiment name")
        batch_size = safe_int(payload.get("batch_size", 8), "batch_size", 1, 1024)
        if batch_size % len(gpu_ids):
            raise ValueError("batch_size must be divisible by the number of selected GPUs")
        fsdp_devices = safe_int(payload.get("fsdp_devices", 1), "fsdp_devices", 1, len(gpu_ids))
        if len(gpu_ids) % fsdp_devices:
            raise ValueError("selected GPU count must be divisible by fsdp_devices")
        num_workers = safe_int(payload.get("num_workers", 2), "num_workers", 1, 64)
        steps = safe_int(payload.get("num_train_steps", 30_000), "num_train_steps", 1, 10_000_000)
        save_interval = safe_int(payload.get("save_interval", 1_000), "save_interval", 1, steps)
        mode = str(payload.get("mode", "auto"))
        if mode not in {"auto", "new", "resume", "overwrite"}:
            raise ValueError("mode must be auto, new, resume, or overwrite")
        checkpoint_dir = (
            Path(config["checkpoint_base_dir"])
            / "pi05_piper_single_arm_lora"
            / exp_name
        )
        if mode == "new" and checkpoint_dir.exists():
            raise FileExistsError(
                f"checkpoint directory already exists: {checkpoint_dir}; "
                "choose auto/resume to continue it, or overwrite to replace it"
            )
        # Upstream --resume is safe for both cases: it resumes the latest
        # checkpoint when the directory exists, and starts a fresh run when it
        # does not. Use it as the non-destructive Dashboard default.
        effective_mode = "resume" if mode == "auto" else mode
        command = [
            config["openpi_python"], openpi_helper, "train",
            "--dataset-id", dataset_id,
            "--arm-side", arm_side,
            "--schema", schema,
            "--exp-name", exp_name,
            "--assets-base-dir", config["assets_base_dir"],
            "--checkpoint-base-dir", config["checkpoint_base_dir"],
            "--base-checkpoint", config["base_checkpoint"],
            "--batch-size", str(batch_size),
            "--num-workers", str(num_workers),
            "--num-train-steps", str(steps),
            "--save-interval", str(save_interval),
            "--fsdp-devices", str(fsdp_devices),
        ]
        if effective_mode != "new":
            command.append(f"--{effective_mode}")
        if bool(payload.get("wandb_enabled", False)):
            command.append("--wandb-enabled")
        metadata = {
            "dataset_id": dataset_id,
            "arm_side": arm_side,
            "schema": schema,
            "exp_name": exp_name,
            "gpu_ids": gpu_ids,
            "steps": steps,
            "fsdp_devices": fsdp_devices,
            "mode": mode,
            "effective_mode": effective_mode,
            "checkpoint_dir": str(checkpoint_dir),
        }
        with tasks.lock:
            # Re-check while holding the task lock so simultaneous submissions
            # cannot create duplicate automatic norm jobs for the same dataset.
            if norm_path.is_file():
                gpu_ids = parse_gpus(payload)
                metadata["gpu_ids"] = gpu_ids
                task = tasks.start(
                    "train",
                    command,
                    env=build_environment(config, gpu_ids),
                    metadata=metadata,
                )
                return jsonify(task), 201

            norm_task = next(
                (
                    item
                    for item in tasks.list()
                    if item.get("type") == "norm"
                    and item.get("state") in {"starting", "running"}
                    and item.get("metadata", {}).get("dataset_id") == dataset_id
                    and item.get("metadata", {}).get("arm_side") == arm_side
                    and item.get("metadata", {}).get("schema") == schema
                ),
                None,
            )
            if norm_task is None:
                norm_batch_size = safe_int(payload.get("norm_batch_size", 16), "norm_batch_size", 1, 1024)
                norm_num_workers = safe_int(payload.get("norm_num_workers", 2), "norm_num_workers", 1, 64)
                norm_task = tasks.start(
                    "norm",
                    build_norm_command(
                        dataset_id,
                        arm_side,
                        schema,
                        batch_size=norm_batch_size,
                        num_workers=norm_num_workers,
                    ),
                    env=build_environment(config, None),
                    metadata={
                        "dataset_id": dataset_id,
                        "arm_side": arm_side,
                        "schema": schema,
                        "batch_size": norm_batch_size,
                        "num_workers": norm_num_workers,
                        "max_frames": None,
                        "norm_path": str(norm_path),
                        "automatic": True,
                    },
                    raise_on_error=False,
                )
            task = tasks.create_waiting_train(
                command,
                metadata=metadata,
                norm_task=norm_task,
                norm_path=norm_path,
            )
            return jsonify(task), 202

    @app.post("/api/tasks/policy")
    def start_policy():
        payload = request.get_json(force=True)
        dataset_id, arm_side, schema = parse_dataset(payload)
        port = safe_int(payload.get("port", 8000), "port", config["policy_port_min"], config["policy_port_max"])
        checkpoint = resolve_under(payload.get("checkpoint", ""), checkpoint_roots)
        if not (checkpoint / "params").exists():
            raise ValueError(f"checkpoint has no params directory: {checkpoint}")
        checkpoint_norm = checkpoint / "assets" / dataset_id / "norm_stats.json"
        if not checkpoint_norm.exists():
            raise ValueError(
                f"checkpoint is not associated with dataset {dataset_id}: missing {checkpoint_norm}"
            )
        replace_task_id = str(payload.get("replace_task_id", "")).strip()
        old_task: dict[str, Any] | None = None
        old_active = False
        ignored_pids: set[int] = set()
        if replace_task_id:
            old_task = tasks.get(safe_name(replace_task_id, "replacement task id"))
            if old_task.get("type") != "policy":
                raise ValueError("replace_task_id must refer to a policy task")
            old_active = old_task.get("state") in {"starting", "running", "stopping"}
            if old_active and old_task.get("pid"):
                ignored_pids.add(int(old_task["pid"]))

        # Validate the target resources before disrupting a working Policy. The
        # process being replaced may legitimately own the requested GPU/port.
        gpu_ids = parse_gpus(payload, one_only=True, ignored_pids=ignored_pids)
        old_port = old_task.get("metadata", {}).get("port") if old_task else None
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            port_busy = sock.connect_ex(("127.0.0.1", port)) == 0
        if port_busy and not (old_active and int(old_port or -1) == port):
            raise ValueError(f"port {port} is already in use")

        if old_active and old_task is not None:
            observations.set_control(old_task, mode="shadow", updated_by="policy_replacement")
            tasks.stop(old_task["id"])
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if tasks.get(old_task["id"]).get("state") not in {"starting", "running", "stopping"}:
                    break
                time.sleep(0.2)
            else:
                tasks.stop(old_task["id"], force=True)
                force_deadline = time.monotonic() + 5.0
                while time.monotonic() < force_deadline:
                    if tasks.get(old_task["id"]).get("state") not in {"starting", "running", "stopping"}:
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError(f"timed out force-stopping policy {old_task['id']}")

        # Recheck after shutdown to avoid a race with another process taking the port.
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"port {port} is already in use")
        telemetry_session, telemetry_dir = observations.create_session()
        command = [
            config["openpi_python"], openpi_helper, "serve",
            "--dataset-id", dataset_id,
            "--arm-side", arm_side,
            "--schema", schema,
            "--assets-base-dir", config["assets_base_dir"],
            "--checkpoint-base-dir", config["checkpoint_base_dir"],
            "--base-checkpoint", config["base_checkpoint"],
            "--checkpoint", str(checkpoint),
            "--port", str(port),
            "--telemetry-dir", str(telemetry_dir),
        ]
        default_prompt = str(payload.get("default_prompt", "")).strip()
        if default_prompt:
            if len(default_prompt) > 500:
                raise ValueError("default_prompt is too long")
            command += ["--default-prompt", default_prompt]
        task = tasks.start(
            "policy", command,
            env=build_environment(config, gpu_ids),
            metadata={
                "dataset_id": dataset_id, "arm_side": arm_side, "schema": schema,
                "checkpoint": str(checkpoint), "gpu_ids": gpu_ids, "port": port,
                "ws_url": f"ws://{request.host.split(':')[0]}:{port}",
                "telemetry_session": telemetry_session, "telemetry_dir": str(telemetry_dir),
                "replaced_task_id": replace_task_id or None,
            },
        )
        observations.bind_task(task)
        return jsonify(task), 201

    @app.get("/api/tasks/<task_id>/execution-control")
    def get_execution_control(task_id: str):
        task = tasks.get(task_id)
        return jsonify({"task_id": task["id"], "execution_control": observations.control_for_task(task)})

    @app.post("/api/tasks/<task_id>/execution-control")
    def set_execution_control(task_id: str):
        task = tasks.get(task_id)
        payload = request.get_json(force=True)
        mode = str(payload.get("mode", "")).strip().lower()
        if mode == "execute" and str(payload.get("confirm_task_id", "")) != task["id"]:
            raise ValueError("confirm_task_id must exactly match the policy task id")
        if mode == "execute":
            telemetry = observations.summary_for_task(task)
            if telemetry is None or not telemetry.get("client_connected"):
                raise ValueError("execution requires a connected robot client")
            if not telemetry.get("fresh"):
                raise ValueError("execution requires fresh robot telemetry")
            if not telemetry.get("client_allow_execution"):
                raise ValueError("robot client was not started with --allow-execution")
        control = observations.set_control(
            task,
            mode=mode,
            expires_in_s=payload.get("expires_in_s"),
        )
        return jsonify({"task_id": task["id"], "execution_control": control})

    @app.post("/api/tasks/<task_id>/stop")
    def stop_task(task_id: str):
        payload = request.get_json(silent=True) or {}
        task = tasks.get(task_id)
        if task.get("type") == "policy" and task.get("metadata", {}).get("telemetry_session"):
            observations.set_control(task, mode="shadow", updated_by="policy_stop")
        return jsonify(tasks.stop(task_id, force=bool(payload.get("force", False))))

    @app.get("/api/tasks/<task_id>")
    def get_task(task_id: str):
        return jsonify(tasks.get(task_id))

    @app.get("/api/tasks/<task_id>/log")
    def task_log(task_id: str):
        max_bytes = safe_int(request.args.get("max_bytes", 64 * 1024), "max_bytes", 1024, 1024 * 1024)
        task = tasks.get(task_id)
        return jsonify({"task": task, "log": tasks.log_tail(task_id, max_bytes)})


    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=APP_DIR / "config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    app = create_app(args.config)
    app.run(host=config["host"], port=int(config["port"]), threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
