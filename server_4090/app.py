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
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any, Iterable
from urllib.request import urlopen
import uuid

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import piper_action_conventions as _piper_action_conventions
except ImportError:
    _piper_action_conventions = None


def _action_constant(name: str, default: str) -> str:
    return str(getattr(_piper_action_conventions, name, default))


DELIVERY_STEP_ACTION_CONVENTION = _action_constant(
    "DELIVERY_STEP_ACTION_CONVENTION", "step"
)
DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION = _action_constant(
    "DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION", "chunk_origin"
)
DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION = "absolute_eef_target"
DELIVERY_LEGACY_STEP_ACTION_SEMANTICS = _action_constant(
    "DELIVERY_STEP_ACTION_SEMANTICS",
    "eef_delta_base_xyz_left_rotvec_gripper_target",
)
DELIVERY_LEGACY_CHUNK_ACTION_SEMANTICS = _action_constant(
    "DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS",
    "eef_delta_chunk_origin_base_xyz_left_rotvec_gripper_target",
)
DELIVERY_RAW_ACTION_SEMANTICS = _action_constant(
    "DELIVERY_RAW_ACTION_SEMANTICS", "absolute_eef_target"
)
DELIVERY_MODEL_ACTION_SEMANTICS = _action_constant(
    "DELIVERY_MODEL_ACTION_SEMANTICS",
    "eef_delta_chunk_origin_base_xyz_left_rotvec_gripper_opening_target",
)
JOINT_RAW_ACTION_SEMANTICS = _action_constant(
    "JOINT_ACTION_SEMANTICS", "absolute_joint_position_opening_fraction"
)
JOINT_MODEL_ACTION_SEMANTICS = _action_constant(
    "JOINT_MODEL_ACTION_SEMANTICS",
    "joint_delta_chunk_origin_first_6_absolute_gripper_target",
)
NEW_GRIPPER_SEMANTICS = _action_constant(
    "NEW_GRIPPER_SEMANTICS", "absolute_opening_fraction_0_closed_1_open"
)
LEGACY_DELIVERY_GRIPPER_SEMANTICS = _action_constant(
    "LEGACY_GRIPPER_SEMANTICS", "absolute_closed_fraction_0_open_1_closed"
)
LEGACY_JOINT_GRIPPER_SEMANTICS = "absolute_opening_metres"
JOINT_RAW_ACTION_CONVENTION = "absolute_joint_target"
CURRENT_CONTRACT_VERSION = 3
LEGACY_CONTRACT_VERSION = 2
MIN_POLICY_ACTION_HORIZON = 16
MODEL_ACTION_START_OFFSET_STEPS = 1
ACTION_CONTRACT_MARKER_VERSION = 3
SSH_COMMAND = [
    "ssh",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
]

try:
    from .dataset_editor import (
        DATASET_ORIGINS,
        DatasetEditor,
        DatasetValidationError,
        normalize_dataset_origin,
        read_dataset_origin_marker,
    )
    from .episode_split import (
        DEFAULT_SPLIT_SEED,
        DEFAULT_TEST_RATIO,
        NORM_CONFIG_FILENAME,
        NORM_CONFIG_VERSION,
        EpisodeSplit,
        load_episode_split,
        norm_split_matches,
        normalize_contract_fingerprint,
        resolve_episode_split,
    )
except ImportError:  # app.py is normally executed directly by start_server.sh
    from dataset_editor import (
        DATASET_ORIGINS,
        DatasetEditor,
        DatasetValidationError,
        normalize_dataset_origin,
        read_dataset_origin_marker,
    )
    from episode_split import (
        DEFAULT_SPLIT_SEED,
        DEFAULT_TEST_RATIO,
        NORM_CONFIG_FILENAME,
        NORM_CONFIG_VERSION,
        EpisodeSplit,
        load_episode_split,
        norm_split_matches,
        normalize_contract_fingerprint,
        resolve_episode_split,
    )


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TASK_TYPES = {"norm", "train", "eval", "policy", "transfer"}
PROCESS_STATES = {"starting", "running", "stopping"}
WAITING_STATES = {"waiting_norm", "waiting_gpu"}
TERMINAL_STATES = {"completed", "failed", "lost", "stopped", "skipped"}
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
TRAIN_STEP = re.compile(r"\bStep\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
TRAIN_METRIC = re.compile(
    r"([A-Za-z][A-Za-z0-9_.-]*)\s*=\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
DASHBOARD_SLURM_JOB_ID = re.compile(r"\[dashboard\]\s+slurm_job_id=(\d+)")
DASHBOARD_SLURM_LOG_STREAM_MARKER = "[dashboard] slurm log stream"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def parse_training_metrics(log_text: str, *, max_points: int = 1200) -> dict[str, Any]:
    """Extract OpenPI's ``Step N: key=value`` progress records from a task log."""
    clean = ANSI_ESCAPE.sub("", log_text).replace("\r", "\n")
    by_step: dict[int, dict[str, float | int]] = {}
    for line in clean.splitlines():
        match = TRAIN_STEP.search(line.strip())
        if not match:
            continue
        point: dict[str, float | int] = {"step": int(match.group(1))}
        for key, value in TRAIN_METRIC.findall(match.group(2)):
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                point[key] = number
        if len(point) > 1:
            by_step[int(point["step"])] = point

    all_points = [by_step[step] for step in sorted(by_step)]
    total_points = len(all_points)
    series = sorted({key for point in all_points for key in point if key != "step"})
    summary: dict[str, dict[str, float]] = {}
    for key in series:
        values = [float(point[key]) for point in all_points if key in point]
        if values:
            summary[key] = {"latest": values[-1], "min": min(values), "max": max(values)}

    points = all_points
    if max_points > 1 and total_points > max_points:
        indexes = sorted({round(index * (total_points - 1) / (max_points - 1)) for index in range(max_points)})
        points = [all_points[index] for index in indexes]
    return {
        "points": points,
        "series": series,
        "summary": summary,
        "total_points": total_points,
        "sampled_points": len(points),
    }


def is_complete_checkpoint(path: Path | str) -> bool:
    """Return whether *path* has finalized parameter-checkpoint markers.

    This deliberately keeps the historical Dashboard definition used for
    listing/evaluation: some weight-only checkpoints do not contain optimizer
    state.  Full-state resume uses :func:`is_full_state_checkpoint` below.
    """
    path = Path(path).expanduser()
    return bool(
        path.is_dir()
        and path.name.isdigit()
        and (path / "_CHECKPOINT_METADATA").is_file()
        and (path / "params" / "_METADATA").is_file()
    )


def is_full_state_checkpoint(path: Path | str) -> bool:
    path = Path(path).expanduser()
    return bool(is_complete_checkpoint(path) and (path / "train_state" / "_METADATA").is_file())


def full_state_checkpoint_steps(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return []
    return sorted(
        (int(child.name), child.resolve())
        for child in checkpoint_dir.iterdir()
        if is_full_state_checkpoint(child)
    )


def complete_checkpoint_steps(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Return complete numeric Orbax checkpoints without expensive size scans."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for child in checkpoint_dir.iterdir():
        if not is_complete_checkpoint(child):
            continue
        checkpoints.append((int(child.name), child.resolve()))
    return sorted(checkpoints)


def select_idle_eval_gpu(
    task_list: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    *,
    allowed_gpu_ids: set[int],
    minimum_free_mib: int,
) -> int | None:
    """Choose a truly idle GPU, including the process-start reservation window."""
    reserved = {
        int(gpu_id)
        for task in task_list
        if task.get("state") in PROCESS_STATES
        and task.get("type") in {"train", "eval", "policy"}
        for gpu_id in task.get("metadata", {}).get("gpu_ids", [])
    }
    candidates: list[tuple[int, int]] = []
    for gpu in inventory:
        gpu_id = int(gpu.get("index", -1))
        if gpu_id not in allowed_gpu_ids or gpu_id in reserved:
            continue
        if gpu.get("compute_available") is not True or gpu.get("processes"):
            continue
        free_mib = int(gpu.get("memory_total_mib", 0)) - int(gpu.get("memory_used_mib", 0))
        if free_mib < minimum_free_mib:
            continue
        candidates.append((free_mib, gpu_id))
    if not candidates:
        return None
    # Prefer the emptiest candidate, then a stable physical GPU id.
    return max(candidates, key=lambda item: (item[0], -item[1]))[1]


def merge_eval_metrics(
    training_metrics: dict[str, Any],
    eval_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge completed held-out metrics into the same step-indexed chart points."""
    by_step = {int(point["step"]): dict(point) for point in training_metrics.get("points", [])}
    original_summary = dict(training_metrics.get("summary", {}))
    original_total_points = int(training_metrics.get("total_points", len(by_step)))
    eval_runs: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for task in sorted(eval_tasks, key=lambda item: int(item.get("metadata", {}).get("checkpoint_step", 0))):
        metadata = task.get("metadata", {})
        state = str(task.get("state", "unknown"))
        counts[state] = counts.get(state, 0) + 1
        run: dict[str, Any] = {
            "task_id": task.get("id"),
            "state": state,
            "checkpoint_step": metadata.get("checkpoint_step"),
            "checkpoint": metadata.get("checkpoint"),
            "gpu_ids": metadata.get("gpu_ids", []),
            "skip_reason": task.get("skip_reason") or metadata.get("skip_reason"),
            "finished_at": task.get("finished_at"),
        }
        result_path = metadata.get("result_path")
        result = read_json(Path(result_path)) if result_path else None
        if isinstance(result, dict):
            run["result"] = result
            step = int(result.get("checkpoint_step", metadata.get("checkpoint_step", 0)))
            point = by_step.setdefault(step, {"step": step})
            for key, value in result.items():
                if key.startswith("eval_") and key != "eval_loss_per_dim" and isinstance(value, (int, float)):
                    number = float(value)
                    if math.isfinite(number):
                        point[key] = number
        eval_runs.append(run)

    all_points = [by_step[step] for step in sorted(by_step)]
    series = sorted({key for point in all_points for key in point if key != "step"})
    summary: dict[str, dict[str, float]] = original_summary
    for key in (item for item in series if item.startswith("eval_")):
        values = [float(point[key]) for point in all_points if key in point]
        if values:
            summary[key] = {"latest": values[-1], "min": min(values), "max": max(values)}
    eval_points = [point for point in all_points if any(key.startswith("eval_") for key in point)]
    model_points = [point for point in eval_points if "eval_loss_model" in point]
    eval_summary: dict[str, Any] = {"counts": counts}
    if model_points:
        latest = model_points[-1]
        best = min(model_points, key=lambda point: float(point["eval_loss_model"]))
        eval_summary.update({
            "latest_step": int(latest["step"]),
            "latest_loss": float(latest["eval_loss_model"]),
            "best_step": int(best["step"]),
            "best_loss": float(best["eval_loss_model"]),
        })
    training_metrics.update({
        "points": all_points,
        "series": series,
        "summary": summary,
        "total_points": original_total_points,
        "sampled_points": len(all_points),
        "eval_points": eval_points,
        "eval_runs": eval_runs,
        "eval_summary": eval_summary,
    })
    return training_metrics


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


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_temporal_action_contract(
    metadata: dict[str, Any],
    *,
    legacy_delivery: bool,
) -> tuple[int, int]:
    alignment = str(metadata.get("action_alignment") or "").strip().lower()
    source = str(metadata.get("action_source") or "").strip().lower()
    expected: int | None = None
    if alignment.startswith("same_step_command"):
        expected = 0
    elif alignment in {"next_observation", "next_measured", "next_measured_fallback"}:
        expected = 1
    elif "next_measured" in source:
        expected = 1

    raw = metadata.get("action_offset")
    if raw is None:
        action_offset = expected if expected is not None else 1 if legacy_delivery else 0
    else:
        try:
            action_offset = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("action_offset must be integer 0 or 1") from exc
    if action_offset not in {0, 1}:
        raise ValueError("action_offset must be 0 or 1")
    if expected is not None and action_offset != expected:
        raise ValueError(
            f"action_offset={action_offset} conflicts with action_alignment={alignment!r}; expected {expected}"
        )

    raw_model = metadata.get(
        "model_action_start_offset",
        metadata.get("model_action_start_offset_steps", MODEL_ACTION_START_OFFSET_STEPS),
    )
    try:
        model_start = int(raw_model)
    except (TypeError, ValueError) as exc:
        raise ValueError("model_action_start_offset must be integer 1") from exc
    if model_start != MODEL_ACTION_START_OFFSET_STEPS:
        raise ValueError(
            f"model_action_start_offset must be {MODEL_ACTION_START_OFFSET_STEPS}, got {model_start}"
        )
    return action_offset, model_start


def complete_action_contract_fingerprint(contract: dict[str, Any]) -> dict[str, int | str]:
    fingerprint = normalize_contract_fingerprint(contract)
    try:
        action_offset = int(contract.get("action_offset"))
        model_start = int(
            contract.get("model_action_start_offset", contract.get("model_action_start_offset_steps"))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "action contract requires action_offset and model_action_start_offset"
        ) from exc
    if action_offset not in {0, 1}:
        raise ValueError("action_offset must be 0 or 1")
    if model_start != MODEL_ACTION_START_OFFSET_STEPS:
        raise ValueError(
            f"model_action_start_offset must be {MODEL_ACTION_START_OFFSET_STEPS}"
        )
    return {
        **fingerprint,
        "action_offset": action_offset,
        "model_action_start_offset": model_start,
    }


def norm_extended_contract_matches(
    norm_config: Any, contract: dict[str, int | str]
) -> bool:
    return bool(
        isinstance(norm_config, dict)
        and norm_config.get("version") == NORM_CONFIG_VERSION
        and all(norm_config.get(key) == value for key, value in contract.items())
    )


def policy_horizon_status(
    telemetry: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the fail-closed async execution-horizon contract."""
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    action_horizon = _positive_int_or_none(
        telemetry.get("action_horizon", metadata.get("action_horizon"))
    )
    client_horizon = _positive_int_or_none(telemetry.get("client_action_horizon"))
    advertised_minimum = _positive_int_or_none(
        telemetry.get("client_minimum_horizon", telemetry.get("minimum_horizon"))
    )
    minimum_horizon = max(MIN_POLICY_ACTION_HORIZON, advertised_minimum or 0)
    client_matches = (
        None
        if client_horizon is None or action_horizon is None
        else client_horizon == action_horizon
    )
    ready = bool(
        action_horizon is not None
        and action_horizon >= minimum_horizon
        and client_matches is not False
    )
    if action_horizon is None:
        error = "policy metadata is missing a valid action_horizon"
    elif action_horizon < minimum_horizon:
        error = (
            f"action_horizon={action_horizon} is below the execution minimum "
            f"{minimum_horizon}"
        )
    elif client_matches is False:
        error = (
            f"client action_horizon={client_horizon} does not match policy "
            f"action_horizon={action_horizon}"
        )
    else:
        error = None
    return {
        "action_horizon": action_horizon,
        "minimum_horizon": minimum_horizon,
        "client_action_horizon": client_horizon,
        "horizon_contract_match": client_matches,
        "horizon_execution_ready": ready,
        "horizon_error": error,
    }


def policy_time_contract_status(
    telemetry: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}

    def value(key: str) -> Any:
        return telemetry.get(key, metadata.get(key))

    try:
        action_offset = int(value("action_offset"))
    except (TypeError, ValueError):
        action_offset = None
    try:
        model_start = int(value("model_action_start_offset"))
    except (TypeError, ValueError):
        model_start = None
    try:
        wire_start = int(value("action_start_offset_steps"))
    except (TypeError, ValueError):
        wire_start = None
    try:
        action_hz = float(value("action_hz"))
    except (TypeError, ValueError):
        action_hz = None
    try:
        time_step = float(value("action_time_step_s"))
    except (TypeError, ValueError):
        time_step = None

    errors = []
    if action_offset not in {0, 1}:
        errors.append("action_offset must be 0 or 1")
    if model_start != MODEL_ACTION_START_OFFSET_STEPS:
        errors.append("model_action_start_offset must be 1")
    if wire_start != MODEL_ACTION_START_OFFSET_STEPS:
        errors.append("action_start_offset_steps must be 1")
    if action_hz is None or not math.isfinite(action_hz) or action_hz <= 0:
        errors.append("action_hz must be positive")
    if time_step is None or not math.isfinite(time_step) or time_step <= 0:
        errors.append("action_time_step_s must be positive")
    elif action_hz is not None and math.isfinite(action_hz) and action_hz > 0 and not math.isclose(
        time_step, 1.0 / action_hz, rel_tol=1e-6, abs_tol=1e-9
    ):
        errors.append("action_time_step_s does not equal 1/action_hz")
    return {
        "action_offset": action_offset,
        "model_action_start_offset": model_start,
        "action_start_offset_steps": wire_start,
        "action_time_step_s": time_step,
        "time_contract_ready": not errors,
        "time_contract_error": "; ".join(errors) if errors else None,
    }


def require_policy_execution_time_contract(telemetry: dict[str, Any] | None) -> None:
    status = policy_time_contract_status(telemetry)
    if not status["time_contract_ready"]:
        raise ValueError(
            "execution requires model/wire actions to start at t_obs + 1/fps: "
            + str(status["time_contract_error"])
        )


def require_policy_execution_horizon(telemetry: dict[str, Any] | None) -> None:
    status = policy_horizon_status(telemetry)
    if not status["horizon_execution_ready"]:
        raise ValueError(f"execution requires action_horizon >= {MIN_POLICY_ACTION_HORIZON}: {status['horizon_error']}")


def safe_float(value: Any, label: str, minimum: float, maximum: float, *, maximum_inclusive: bool = True) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    upper_ok = parsed <= maximum if maximum_inclusive else parsed < maximum
    if not math.isfinite(parsed) or parsed < minimum or not upper_ok:
        bracket = "]" if maximum_inclusive else ")"
        raise ValueError(f"{label} must be in [{minimum}, {maximum}{bracket}")
    return parsed


MODEL_VARIANTS = {"pi05", "pi0"}


def infer_model_variant(path: Path) -> str | None:
    path = Path(path)

    def from_name(value: str) -> str | None:
        name = value.lower()
        if re.match(r"^pi(?:05|0\.5)(?:_|-|$)", name):
            return "pi05"
        if re.match(r"^pi0(?:_|-|$)", name):
            return "pi0"
        return None

    # A complete checkpoint is normally ``<config>/<experiment>/<step>``.
    # Inspect the config directory first so an experiment name such as
    # ``from_pi05_transfer`` cannot override the actual model family.
    if path.name.isdigit() and len(path.parents) >= 2:
        configured = from_name(path.parent.parent.name)
        if configured is not None:
            return configured

    # Then inspect from the checkpoint itself towards its parents.  The OpenPI
    # checkout currently lives below a directory named ``pi05`` on 4x4090, so
    # nearest recognized names must win over distant repository directories.
    for part in reversed(path.parts):
        inferred = from_name(part)
        if inferred is not None:
            return inferred
    return None


def policy_config_name(arm_mode: str, model_variant: str = "pi05") -> str:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"unsupported model_variant: {model_variant!r}")
    if arm_mode not in {"single", "bimanual"}:
        raise ValueError(f"unsupported arm_mode: {arm_mode!r}")
    suffix = "single_arm" if arm_mode == "single" else "bimanual"
    return f"{model_variant}_piper_{suffix}_lora"


def training_checkpoint_identity(
    path: Path, checkpoint_base_dir: Path
) -> dict[str, Any] | None:
    """Describe a standard ``config/experiment/step`` training checkpoint."""
    path = Path(path).expanduser().resolve()
    checkpoint_base_dir = Path(checkpoint_base_dir).expanduser().resolve()
    try:
        relative = path.relative_to(checkpoint_base_dir)
    except ValueError:
        return None
    if len(relative.parts) != 3 or not relative.parts[2].isdigit():
        return None
    config_name, experiment, step_text = relative.parts
    for model_variant in sorted(MODEL_VARIANTS):
        for arm_mode in ("single", "bimanual"):
            if config_name == policy_config_name(arm_mode, model_variant):
                return {
                    "config_name": config_name,
                    "experiment": experiment,
                    "checkpoint_step": int(step_text),
                    "model_variant": model_variant,
                    "arm_mode": arm_mode,
                }
    return None


def training_experiment_catalog(checkpoint_base_dir: Path) -> list[dict[str, Any]]:
    """List existing experiment directories, including runs without a complete step."""
    checkpoint_base_dir = Path(checkpoint_base_dir).expanduser().resolve()
    experiments: dict[str, dict[str, Any]] = {}
    for model_variant in sorted(MODEL_VARIANTS):
        for arm_mode in ("single", "bimanual"):
            config_name = policy_config_name(arm_mode, model_variant)
            config_root = checkpoint_base_dir / config_name
            if not config_root.is_dir():
                continue
            for experiment_dir in config_root.iterdir():
                if not experiment_dir.is_dir() or experiment_dir.name.startswith("."):
                    continue
                complete = complete_checkpoint_steps(experiment_dir)
                entry = experiments.setdefault(
                    experiment_dir.name,
                    {
                        "name": experiment_dir.name,
                        "model_variants": set(),
                        "arm_modes": set(),
                        "config_names": set(),
                        "checkpoint_count": 0,
                        "latest_step": None,
                        "mtime": 0.0,
                    },
                )
                entry["model_variants"].add(model_variant)
                entry["arm_modes"].add(arm_mode)
                entry["config_names"].add(config_name)
                entry["checkpoint_count"] += len(complete)
                if complete:
                    latest_step = complete[-1][0]
                    entry["latest_step"] = max(entry["latest_step"] or 0, latest_step)
                entry["mtime"] = max(entry["mtime"], experiment_dir.stat().st_mtime)
    result = []
    for entry in experiments.values():
        result.append(
            {
                **entry,
                "model_variants": sorted(entry["model_variants"]),
                "arm_modes": sorted(entry["arm_modes"]),
                "config_names": sorted(entry["config_names"]),
                "updated_at": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(entry["mtime"])
                ),
            }
        )
    return sorted(result, key=lambda item: (-item["mtime"], item["name"]))


def dataset_origin_info(
    dataset_id: str, dataset_path: Path, info: dict[str, Any]
) -> dict[str, Any]:
    """Classify datasets while allowing an explicit Dashboard marker to win."""
    if not isinstance(info, dict):
        info = {}
    marker = read_dataset_origin_marker(dataset_path)
    if marker is not None:
        return {
            "dataset_origin": marker["origin"],
            "dataset_origin_source": "marker",
            "dataset_origin_marker": marker,
        }

    for key in ("dataset_origin", "data_origin", "source_domain"):
        if info.get(key) is None:
            continue
        try:
            origin = normalize_dataset_origin(info.get(key))
        except ValueError:
            continue
        return {
            "dataset_origin": origin,
            "dataset_origin_source": f"info.{key}",
            "dataset_origin_marker": None,
        }
    if isinstance(info.get("simulation"), bool):
        return {
            "dataset_origin": "simulation" if info["simulation"] else "real",
            "dataset_origin_source": "info.simulation",
            "dataset_origin_marker": None,
        }

    name = dataset_id.lower()
    robot_type = str(info.get("robot_type", "")).strip().lower()
    real_name = bool(re.search(r"(?:^|[._-])real(?:[._-]|$)", name) or name == "my_dataset")
    if robot_type == "piper" or real_name:
        return {
            "dataset_origin": "real",
            "dataset_origin_source": "heuristic",
            "dataset_origin_marker": None,
        }
    simulation_name = bool(
        re.search(r"(?:^|[._-])(sim|synth|synthetic|smoke|robottwin)(?:[._-]|$)", name)
    )
    if (
        simulation_name
        or robot_type == "aloha"
        or (robot_type.startswith("piper_single_arm") and bool(info.get("video_path")))
    ):
        return {
            "dataset_origin": "simulation",
            "dataset_origin_source": "heuristic",
            "dataset_origin_marker": None,
        }
    if "piper" in robot_type:
        return {
            "dataset_origin": "real",
            "dataset_origin_source": "heuristic",
            "dataset_origin_marker": None,
        }
    return {
        "dataset_origin": "unknown",
        "dataset_origin_source": "unclassified",
        "dataset_origin_marker": None,
    }


def describe_dataset_schema(info: dict[str, Any]) -> dict[str, Any]:
    """Describe and validate raw/model Piper action contracts for the UI."""
    try:
        dataset_origin = normalize_dataset_origin(info.get("dataset_origin", "unknown"))
    except ValueError:
        dataset_origin = "unknown"
    is_simulation_dataset = dataset_origin == "simulation"
    features = info.get("features", {})
    if not isinstance(features, dict):
        features = {}
    metadata: dict[str, Any] = {}
    for key in ("data_contract", "contract", "piper_contract"):
        value = info.get(key)
        if isinstance(value, dict):
            metadata.update(value)
    metadata.update(info)

    def feature_for(*keys: str) -> tuple[str | None, dict[str, Any]]:
        for key in keys:
            value = features.get(key)
            if isinstance(value, dict):
                return key, value
        return None, {}

    def last_dim(feature: dict[str, Any]) -> int | None:
        shape = feature.get("shape")
        try:
            return int(shape[-1]) if isinstance(shape, (list, tuple)) and shape else None
        except (TypeError, ValueError):
            return None

    state_key, state_feature = feature_for("observation.state", "state")
    action_key, action_feature = feature_for("action", "actions")
    state_shape = state_feature.get("shape")
    action_shape = action_feature.get("shape")
    state_dim = last_dim(state_feature)
    raw_action_dim = last_dim(action_feature)
    dataset_layout = (
        "canonical"
        if state_key == "observation.state" and action_key == "action"
        else "legacy"
        if state_key == "state" and action_key == "actions"
        else "unknown"
    )

    layouts = {
        (7, 7): ("joint", "single", False),
        (14, 14): ("joint", "bimanual", False),
        # Franka Panda: 7 joints + 1 gripper per arm.
        (16, 16): ("joint", "bimanual", False),
        (10, 7): ("delivery", "single", True),
        (20, 14): ("delivery", "bimanual", True),
        (10, 10): ("delivery", "single", False),
        (20, 20): ("delivery", "bimanual", False),
    }
    inferred = layouts.get((state_dim, raw_action_dim))
    inferred_schema = inferred[0] if inferred else "custom"
    inferred_arm_mode = inferred[1] if inferred else "unknown"
    legacy_delivery = bool(inferred and inferred[2])
    schema = str(metadata.get("schema") or inferred_schema).lower()
    arm_mode = str(metadata.get("arm_mode") or inferred_arm_mode).lower()
    arm_side = "both" if arm_mode == "bimanual" else str(metadata.get("arm_side") or "right").lower()
    arm_count = 2 if arm_mode == "bimanual" else 1
    model_action_dim = (
        raw_action_dim
        if schema == "joint" and arm_mode in {"single", "bimanual"}
        else 7 * arm_count if arm_mode in {"single", "bimanual"} else None
    )

    media = sorted(
        (
            {"key": key, "type": value.get("dtype")}
            for key, value in features.items()
            if isinstance(value, dict) and value.get("dtype") in {"image", "video"}
        ),
        key=lambda item: (str(item["type"]), str(item["key"])),
    )
    media_keys = {str(item["key"]) for item in media}
    if dataset_layout == "legacy":
        required_media = {"image", "wrist_image"}
    elif arm_mode == "bimanual":
        required_media = {
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        }
    else:
        wrist_candidates = {
            "observation.images.cam_wrist",
            f"observation.images.cam_{arm_side}_wrist",
        }
        required_media = {"observation.images.cam_high"}
        if not (media_keys & wrist_candidates):
            required_media.add(f"observation.images.cam_{arm_side}_wrist")

    errors: list[str] = []
    if inferred is None:
        errors.append("unsupported state/raw-action dimensions")
    elif schema != inferred_schema or arm_mode != inferred_arm_mode:
        errors.append(
            f"schema/arm metadata {schema}/{arm_mode} conflicts with dimensions"
        )
    if dataset_layout == "legacy" and not (
        legacy_delivery and arm_mode == "single"
    ):
        errors.append("legacy state/actions layout only supports single-arm delivery v2")
    if legacy_delivery and dataset_layout == "canonical" and not (
        str(metadata.get("legacy_format") or "").lower() == "legacy_v2"
        or str(metadata.get("raw_action_convention") or metadata.get("action_convention") or "").lower()
        in {"step", "one_step", "one_step_delta", "step_delta"}
    ):
        errors.append("canonical 10D/7D delivery requires explicit legacy_v2/step metadata")
    if not required_media.issubset(media_keys):
        errors.append("missing required camera media")
    if arm_mode == "single" and arm_side not in {"left", "right"}:
        errors.append("single-arm arm_side must be left/right")
    if arm_mode == "bimanual" and arm_side != "both":
        errors.append("bimanual arm_side must be both")

    raw_action_semantics: str | None = None
    model_action_semantics: str | None = None
    raw_action_convention: str | None = None
    model_action_convention: str | None = None
    raw_gripper_semantics: str | None = None
    model_gripper_semantics: str | None = None
    contract_version: int | None = None
    if inferred is not None:
        try:
            declared_version = int(metadata["contract_version"]) if metadata.get("contract_version") is not None else None
        except (TypeError, ValueError):
            declared_version = None
            errors.append("contract_version must be an integer")
        declared_raw_dim = metadata.get("raw_action_dim")
        declared_model_dim = metadata.get("model_action_dim")
        try:
            if declared_raw_dim is not None and int(declared_raw_dim) != raw_action_dim:
                errors.append("raw_action_dim metadata conflicts with feature")
            if declared_model_dim is not None and int(declared_model_dim) != model_action_dim:
                errors.append("model_action_dim metadata conflicts with model contract")
        except (TypeError, ValueError):
            errors.append("raw/model action dimensions must be integers")

        if schema == "delivery":
            if legacy_delivery:
                contract_version = LEGACY_CONTRACT_VERSION
                raw_action_semantics = str(
                    metadata.get("raw_action_semantics")
                    or metadata.get("action_semantics")
                    or DELIVERY_LEGACY_STEP_ACTION_SEMANTICS
                )
                raw_action_convention = DELIVERY_STEP_ACTION_CONVENTION
                model_action_semantics = DELIVERY_LEGACY_CHUNK_ACTION_SEMANTICS
                raw_gripper_semantics = LEGACY_DELIVERY_GRIPPER_SEMANTICS
                model_gripper_semantics = raw_gripper_semantics
            else:
                contract_version = declared_version or CURRENT_CONTRACT_VERSION
                if contract_version < CURRENT_CONTRACT_VERSION:
                    errors.append("absolute-EEF raw delivery requires contract_version>=3")
                raw_action_semantics = str(
                    metadata.get("raw_action_semantics")
                    or metadata.get("action_semantics")
                    or DELIVERY_RAW_ACTION_SEMANTICS
                )
                raw_action_convention = DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION
                model_action_semantics = DELIVERY_MODEL_ACTION_SEMANTICS
                raw_gripper_semantics = NEW_GRIPPER_SEMANTICS
                model_gripper_semantics = NEW_GRIPPER_SEMANTICS
            model_action_convention = DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
        elif schema == "joint":
            declared_gripper = str(
                metadata.get("raw_gripper_semantics")
                or metadata.get("gripper_semantics")
                or ""
            )
            names = " ".join(
                map(
                    str,
                    [
                        *(state_feature.get("names") or []),
                        *(action_feature.get("names") or []),
                    ],
                )
            ).lower()
            meter_aliases = {
                LEGACY_JOINT_GRIPPER_SEMANTICS,
                "absolute_opening_m",
                "opening_m",
            }
            fraction_aliases = {
                NEW_GRIPPER_SEMANTICS,
                "absolute_opening_fraction",
                "opening_fraction",
            }
            if declared_gripper in meter_aliases or "gripper_opening_m" in names:
                contract_version = LEGACY_CONTRACT_VERSION
                raw_gripper_semantics = LEGACY_JOINT_GRIPPER_SEMANTICS
                raw_action_semantics = str(
                    metadata.get("raw_action_semantics")
                    or metadata.get("action_semantics")
                    or "absolute_joint_position"
                )
            elif declared_gripper in fraction_aliases or "gripper_opening_fraction" in names:
                contract_version = max(declared_version or CURRENT_CONTRACT_VERSION, CURRENT_CONTRACT_VERSION)
                raw_gripper_semantics = NEW_GRIPPER_SEMANTICS
                raw_action_semantics = str(
                    metadata.get("raw_action_semantics")
                    or metadata.get("action_semantics")
                    or JOINT_RAW_ACTION_SEMANTICS
                )
            elif declared_version is not None:
                contract_version = (
                    LEGACY_CONTRACT_VERSION
                    if declared_version <= LEGACY_CONTRACT_VERSION
                    else declared_version
                )
                raw_gripper_semantics = (
                    LEGACY_JOINT_GRIPPER_SEMANTICS
                    if contract_version == LEGACY_CONTRACT_VERSION
                    else NEW_GRIPPER_SEMANTICS
                )
                raw_action_semantics = str(
                    metadata.get("raw_action_semantics")
                    or metadata.get("action_semantics")
                    or (
                        "absolute_joint_position"
                        if contract_version == LEGACY_CONTRACT_VERSION
                        else JOINT_RAW_ACTION_SEMANTICS
                    )
                )
            elif is_simulation_dataset:
                # RoboTwin / sim LeRobot exports commonly contain canonical joint
                # 7D/14D Piper or 16D Franka rows without real-robot contract metadata.
                # Keep real datasets fail-closed, but allow simulation datasets to
                # default to the current v3 opening-fraction joint convention.
                contract_version = CURRENT_CONTRACT_VERSION
                raw_gripper_semantics = NEW_GRIPPER_SEMANTICS
                raw_action_semantics = str(
                    metadata.get("raw_action_semantics")
                    or metadata.get("action_semantics")
                    or JOINT_RAW_ACTION_SEMANTICS
                )
            else:
                errors.append(
                    "joint 7D/14D/16D requires contract_version or gripper semantics to distinguish v2 metres from v3 fraction"
                )
            raw_action_convention = JOINT_RAW_ACTION_CONVENTION
            model_action_convention = DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
            joint_dims_per_arm = int(raw_action_dim // arm_count) - 1
            model_action_semantics = (
                JOINT_MODEL_ACTION_SEMANTICS
                if joint_dims_per_arm == 6
                else f"joint_delta_chunk_origin_first_{joint_dims_per_arm}_absolute_gripper_target"
            )
            # New training converts legacy joint metres to fractions before norm.
            model_gripper_semantics = NEW_GRIPPER_SEMANTICS

    action_offset: int | None = None
    model_action_start_offset: int | None = None
    try:
        action_offset, model_action_start_offset = resolve_temporal_action_contract(
            metadata, legacy_delivery=legacy_delivery
        )
    except ValueError as exc:
        errors.append(str(exc))

    contract_fingerprint: dict[str, int | str] | None = None
    if not errors and contract_version is not None:
        contract_fingerprint = complete_action_contract_fingerprint(
            {
                "contract_version": contract_version,
                "raw_action_dim": raw_action_dim,
                "model_action_dim": model_action_dim,
                "raw_action_semantics": raw_action_semantics,
                "model_action_semantics": model_action_semantics,
                "raw_action_convention": raw_action_convention,
                "model_action_convention": model_action_convention,
                "gripper_semantics": model_gripper_semantics,
                "raw_gripper_semantics": raw_gripper_semantics,
                "wire_gripper_semantics": model_gripper_semantics,
                "action_offset": action_offset,
                "model_action_start_offset": model_action_start_offset,
            }
        )

    if inferred is None:
        schema_label = f"通用格式 {state_dim or '?'}D/{raw_action_dim or '?'}D"
    else:
        arm_label = "单臂" if arm_mode == "single" else "双臂"
        if schema == "delivery" and legacy_delivery:
            schema_label = f"{arm_label} Delivery legacy v2 · raw {raw_action_dim}D step → model {model_action_dim}D"
        elif schema == "delivery":
            schema_label = f"{arm_label} Delivery v3 · raw {raw_action_dim}D absolute EEF → model {model_action_dim}D"
        else:
            version_label = "legacy v2" if contract_version == LEGACY_CONTRACT_VERSION else "v3"
            schema_label = f"{arm_label} Joint {version_label} · raw {raw_action_dim}D → model {model_action_dim}D"

    camera_keys = [
        key.removeprefix("observation.images.") for key in sorted(media_keys)
    ]
    model_contract_supported = not errors and contract_fingerprint is not None
    # Legacy Delivery v2 (raw 7D step deltas) is still a valid real-robot
    # training contract.  It uses the explicit step/chunk-origin compatibility
    # path below; only datasets whose dimensions/metadata cannot form a
    # contract should be blocked.  Canonical v3 remains the preferred format,
    # but v2 must not be mislabeled as preview-only because 8_3_64eps is an
    # intentionally supported training dataset.
    training_supported = model_contract_supported
    training_error = None
    return {
        "schema": schema,
        "schema_label": schema_label,
        "arm_mode": arm_mode,
        "arm_layout": "bimanual" if arm_mode == "bimanual" else "single_arm" if arm_mode == "single" else "unknown",
        "arm_side": arm_side,
        "dataset_layout": dataset_layout,
        "contract_version": contract_version,
        "contract_error": "; ".join(errors) if errors else None,
        "contract_fingerprint": contract_fingerprint,
        "legacy_delivery_v2": legacy_delivery,
        "legacy_joint_v2": schema == "joint" and contract_version == LEGACY_CONTRACT_VERSION,
        "state_key": state_key,
        "action_key": action_key,
        "state_shape": state_shape,
        "action_shape": action_shape,
        "state_dim": state_dim,
        "action_dim": raw_action_dim,
        "raw_action_dim": raw_action_dim,
        "model_action_dim": model_action_dim,
        "camera_keys": camera_keys,
        "cameras": [str(item["key"]) for item in media],
        "media": media,
        "training_schema": schema if training_supported else None,
        "model_contract_supported": model_contract_supported,
        "training_supported": training_supported,
        "training_error": training_error,
        "action_semantics": raw_action_semantics,
        "raw_action_semantics": raw_action_semantics,
        "model_action_semantics": model_action_semantics,
        "wire_action_semantics": (
            JOINT_RAW_ACTION_SEMANTICS if schema == "joint" else model_action_semantics
        ),
        "raw_action_convention": raw_action_convention,
        "model_action_convention": model_action_convention,
        "wire_action_convention": (
            JOINT_RAW_ACTION_CONVENTION if schema == "joint" else model_action_convention
        ),
        "gripper_semantics": model_gripper_semantics,
        "raw_gripper_semantics": raw_gripper_semantics,
        "model_gripper_semantics": model_gripper_semantics,
        "wire_gripper_semantics": model_gripper_semantics,
        "action_source": info.get("action_source"),
        "action_alignment": info.get("action_alignment"),
        "action_offset": action_offset,
        "model_action_start_offset": model_action_start_offset,
        "model_action_start_offset_steps": model_action_start_offset,
    }

def action_contract_for_model(
    dataset_contract: dict[str, Any],
    *,
    delivery_action_convention: str | None = None,
    model_gripper_semantics: str | None = None,
) -> dict[str, Any]:
    """Return a complete norm/train/serve contract derived from dataset raw data."""
    if not dataset_contract.get(
        "model_contract_supported", dataset_contract.get("training_supported")
    ):
        raise ValueError(dataset_contract.get("contract_error") or "unsupported dataset contract")
    contract = dict(dataset_contract)
    schema = str(contract["schema"])
    if schema == "delivery":
        convention = delivery_action_convention or DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
        if convention == DELIVERY_STEP_ACTION_CONVENTION:
            if not contract.get("legacy_delivery_v2"):
                raise ValueError("step convention is only valid for legacy delivery v2")
            model_semantics = contract["raw_action_semantics"]
        elif convention == DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION:
            model_semantics = (
                DELIVERY_LEGACY_CHUNK_ACTION_SEMANTICS
                if contract.get("legacy_delivery_v2")
                else DELIVERY_MODEL_ACTION_SEMANTICS
            )
        else:
            raise ValueError(f"unsupported delivery action convention: {convention!r}")
        contract["model_action_convention"] = convention
        contract["model_action_semantics"] = model_semantics
        contract["wire_action_convention"] = convention
        contract["wire_action_semantics"] = model_semantics
    else:
        gripper = model_gripper_semantics or NEW_GRIPPER_SEMANTICS
        if gripper not in {NEW_GRIPPER_SEMANTICS, LEGACY_JOINT_GRIPPER_SEMANTICS}:
            raise ValueError(f"unsupported joint model gripper semantics: {gripper!r}")
        if not contract.get("legacy_joint_v2") and gripper != NEW_GRIPPER_SEMANTICS:
            raise ValueError("joint v3 checkpoints must use opening-fraction grippers")
        contract["model_gripper_semantics"] = gripper
        contract["wire_gripper_semantics"] = gripper
        contract["gripper_semantics"] = gripper
        contract["wire_action_semantics"] = (
            JOINT_RAW_ACTION_SEMANTICS
            if gripper == NEW_GRIPPER_SEMANTICS
            else contract["raw_action_semantics"]
        )
    contract["gripper_semantics"] = contract["model_gripper_semantics"]
    contract["contract_fingerprint"] = complete_action_contract_fingerprint(contract)
    return contract


def action_contract_command_args(contract: dict[str, Any]) -> list[str]:
    args = [
        "--contract-version", str(contract["contract_version"]),
        "--raw-action-dim", str(contract["raw_action_dim"]),
        "--model-action-dim", str(contract["model_action_dim"]),
        "--raw-action-semantics", str(contract["raw_action_semantics"]),
        "--model-action-semantics", str(contract["model_action_semantics"]),
        "--raw-action-convention", str(contract["raw_action_convention"]),
        "--model-action-convention", str(contract["model_action_convention"]),
        "--raw-gripper-semantics", str(contract["raw_gripper_semantics"]),
        "--gripper-semantics", str(contract["model_gripper_semantics"]),
        "--model-gripper-semantics", str(contract["model_gripper_semantics"]),
        "--action-offset", str(contract["action_offset"]),
        "--model-action-start-offset", str(contract["model_action_start_offset"]),
    ]
    if contract["schema"] == "delivery":
        args += [
            "--delivery-action-convention",
            str(contract["model_action_convention"]),
        ]
    return args


def checkpoint_action_contract_marker(path: Path) -> Path:
    path = Path(path)
    experiment_dir = path.parent if path.name.isdigit() else path
    return (
        experiment_dir.parent
        / ".policy_action_conventions"
        / f"{experiment_dir.name}.json"
    )


def checkpoint_action_contract(path: Path) -> dict[str, Any] | None:
    value = read_json(checkpoint_action_contract_marker(path))
    return value if isinstance(value, dict) else None


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


def _process_match_command(task: dict[str, Any]) -> list[Any] | None:
    launch_command = task.get("launch_command")
    if isinstance(launch_command, list) and len(launch_command) >= 3:
        return launch_command
    command = task.get("command")
    if isinstance(command, list) and len(command) >= 3:
        return command
    return None


def process_matches_task(pid: int, task: dict[str, Any]) -> bool:
    """Prevent stale task files from targeting an unrelated reused PID."""
    command = _process_match_command(task)
    if command is None:
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


def process_children_by_parent() -> dict[int, set[int]]:
    """Return a lightweight /proc process tree indexed by parent PID."""
    children: dict[int, set[int]] = {}
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except (FileNotFoundError, PermissionError):
        return children
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            child_pid = int(entry.name)
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat.rsplit(")", 1)[1].strip().split()
            # /proc/<pid>/stat fields after comm start with state then ppid.
            if len(fields) < 2:
                continue
            parent_pid = int(fields[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
        children.setdefault(parent_pid, set()).add(child_pid)
    return children


def process_descendant_pids(
    root_pid: int,
    children_by_parent: dict[int, set[int]] | None = None,
) -> set[int]:
    """Return all live descendants of ``root_pid`` by walking /proc PPid links."""
    if root_pid <= 0:
        return set()
    children = children_by_parent if children_by_parent is not None else process_children_by_parent()
    descendants: set[int] = set()
    pending = list(children.get(root_pid, set()))
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children.get(pid, set()) - descendants)
    descendants.discard(root_pid)
    return descendants


def process_fd_path(pid: int, fd: int) -> str | None:
    try:
        target = os.readlink(f"/proc/{pid}/fd/{fd}")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    if target.startswith("/") and Path(target).is_file():
        return target
    return None


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


def normalize_ssh_host(value: Any) -> str:
    return str(value or "").strip()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError(f"invalid config JSON: {path}")
    defaults = {
        "host": "0.0.0.0",
        "port": 8090,
        "allowed_gpu_ids": [0, 1, 2, 3],
        "allow_busy_gpus": False,
        "small_gpu_process_memory_mib": 512,
        "small_gpu_process_total_mib": 1024,
        "xla_memory_fraction": 0.90,
        "training_min_free_gpu_mib": 23_000,
        "evaluation_min_free_gpu_mib": 23_000,
        "evaluation_xla_memory_fraction": 0.85,
        # Policy inference is much lighter than training. Keep training/eval
        # exclusive, but allow Policy to share with small stable workloads.
        "policy_allow_busy_gpus": True,
        "policy_min_free_gpu_mib": 12_000,
        "policy_xla_memory_fraction": 0.60,
        "policy_xla_preallocate": False,
        "max_upload_gib": 500,
        "max_chunk_mib": 64,
        "policy_port_min": 8000,
        "policy_port_max": 8099,
        "robot_observation_max_age_s": 3.0,
        "task_monitor_interval_s": 2.0,
        "dashboard_profile": "real",
        "dashboard_title": "Bimanual-VLA · 4×4090 控制台",
        "upload_default_origin": None,
        "visible_dataset_origins": None,
        "enable_policy": True,
        "cluster_targets": {},
        "local_storage_locations": {},
        "cache_root": str(Path.home() / ".cache"),
        "eval_video_roots": [],
        "cluster_resources_script": str(REPO_DIR / "scripts" / "query_h100_h200_resources.sh"),
        "transfer_parallelism": 4,
        "auto_sync_cluster_dataset": True,
        "nas_dataset_staging_root": "/DATA/NAS/GPUServer/sunny/dashboard_dataset_sync",
        "nas_checkpoint_staging_root": "/DATA/NAS/GPUServer/sunny/dashboard_checkpoint_sync",
    }
    defaults.update(config)
    profile = str(defaults.get("dashboard_profile") or "real").lower()
    defaults["dashboard_profile"] = profile
    if defaults.get("upload_default_origin") is None:
        defaults["upload_default_origin"] = "simulation" if profile == "simulation" else "real"
    if defaults.get("visible_dataset_origins") is None:
        defaults["visible_dataset_origins"] = ["simulation"] if profile == "simulation" else ["real", "unknown"]
    defaults["visible_dataset_origins"] = [
        normalize_dataset_origin(item) for item in defaults.get("visible_dataset_origins", [])
    ]
    for key in (
        "openpi_repo",
        "openpi_python",
        "dataset_root",
        "workspace_root",
        "cache_root",
        "assets_base_dir",
        "checkpoint_base_dir",
        "base_checkpoint",
    ):
        defaults[key] = str(Path(defaults[key]).expanduser().resolve())
    checkpoint_allowed_roots = [
        str(Path(item).expanduser().resolve()) for item in defaults.get("checkpoint_allowed_roots", [])
    ]
    # Keep configured paths usable after symlink resolution.  A common layout
    # keeps ~/.cache/openpi on one NVMe mount while pi05_base is a symlink to a
    # dedicated model directory on another mount; resolving only the configured
    # parent root otherwise rejects the configured base checkpoint itself.
    for required_root in (defaults["checkpoint_base_dir"], defaults["base_checkpoint"]):
        if required_root not in checkpoint_allowed_roots:
            checkpoint_allowed_roots.append(required_root)
    defaults["checkpoint_allowed_roots"] = checkpoint_allowed_roots
    defaults["eval_video_roots"] = [
        str(Path(item).expanduser().resolve()) for item in defaults.get("eval_video_roots", [])
    ]
    defaults["cluster_resources_script"] = str(
        Path(defaults["cluster_resources_script"]).expanduser().resolve()
    )
    if defaults.get("nas_dataset_staging_root"):
        defaults["nas_dataset_staging_root"] = str(defaults["nas_dataset_staging_root"])
    if defaults.get("nas_checkpoint_staging_root"):
        defaults["nas_checkpoint_staging_root"] = str(defaults["nas_checkpoint_staging_root"])
    try:
        defaults["transfer_parallelism"] = max(1, min(16, int(defaults.get("transfer_parallelism", 4))))
    except (TypeError, ValueError):
        defaults["transfer_parallelism"] = 4
    normalized_local_storage = {}
    for name, storage in dict(defaults.get("local_storage_locations", {})).items():
        if not isinstance(storage, dict):
            continue
        item = dict(storage)
        for path_key in ("dataset_root", "checkpoint_base_dir"):
            if item.get(path_key):
                item[path_key] = str(Path(item[path_key]).expanduser().resolve())
        item["kind"] = str(item.get("kind") or "local_archive")
        item["available"] = bool(item.get("available", True))
        normalized_local_storage[str(name)] = item
    defaults["local_storage_locations"] = normalized_local_storage

    normalized_targets = {}
    for name, target in dict(defaults.get("cluster_targets", {})).items():
        if not isinstance(target, dict):
            continue
        item = dict(target)
        for host_key in ("host", "submit_host"):
            if item.get(host_key):
                item[host_key] = normalize_ssh_host(item[host_key])
        for path_key in (
            "workdir",
            "openpi_repo",
            "dashboard_repo",
            "dataset_root",
            "assets_base_dir",
            "checkpoint_base_dir",
            "base_checkpoint",
            "eval_video_roots",
            "inventory_cache_path",
            "inventory_source_path",
            "nas_dataset_staging_root",
        ):
            if item.get(path_key):
                if path_key == "eval_video_roots" and isinstance(item[path_key], list):
                    item[path_key] = [str(value) for value in item[path_key]]
                else:
                    item[path_key] = str(item[path_key])
        normalized_targets[str(name)] = item
    defaults["cluster_targets"] = normalized_targets
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
        self.training_metric_probe_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
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
                with self.lock:
                    current = self._refresh_all_locked()
                    self._reconcile_auto_evals_locked(current)
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

    def _exit_path(self, task_id: str) -> Path:
        return self.root / safe_name(task_id, "task id") / "exit.json"

    def _runner_command(self, task: dict[str, Any]) -> list[str]:
        return [
            sys.executable,
            str(APP_DIR / "task_runner.py"),
            "--exit-json",
            str(self._exit_path(task["id"])),
            "--cwd",
            str(self.config["openpi_repo"]),
            "--",
            *[str(item) for item in task["command"]],
        ]

    def _systemd_task_backend_enabled(self) -> bool:
        backend = str(self.config.get("task_launch_backend", "auto") or "auto").lower()
        if backend in {"direct", "popen", "subprocess"}:
            return False
        if backend in {"systemd", "systemd_user", "systemd-user"}:
            return True
        return bool(os.environ.get("INVOCATION_ID") and shutil.which("systemd-run") and shutil.which("systemctl"))

    def _systemd_unit_name(self, task_id: str) -> str:
        safe = safe_name(task_id, "task id").replace("_", "-").replace(".", "-")
        return f"bimanual-vla-task-{safe}.service"

    @staticmethod
    def _systemd_show(unit: str) -> dict[str, str]:
        if not unit:
            return {}
        try:
            result = subprocess.run(
                [
                    "systemctl", "--user", "show", unit,
                    "-p", "LoadState",
                    "-p", "ActiveState",
                    "-p", "SubState",
                    "-p", "Result",
                    "-p", "ExecMainCode",
                    "-p", "ExecMainStatus",
                    "-p", "MainPID",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:
            return {"_query_error": repr(exc)}
        values: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        if result.returncode != 0 and not values:
            detail = (result.stderr or result.stdout or "systemctl show failed").strip()
            return {"_query_error": detail}
        return values

    def _launch_direct_runner(
        self,
        task: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> subprocess.Popen:
        log_handle = self._log_path(task["id"]).open("ab", buffering=0)
        try:
            return subprocess.Popen(
                task["launch_command"],
                cwd=self.config["openpi_repo"],
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()

    def _launch_systemd_runner(
        self,
        task: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> int:
        unit = self._systemd_unit_name(task["id"])
        task["systemd_unit"] = unit
        log_path = str(self._log_path(task["id"]))
        command = [
            "systemd-run",
            "--user",
            f"--unit={unit[:-8] if unit.endswith('.service') else unit}",
            f"--property=WorkingDirectory={self.config['openpi_repo']}",
            f"--property=StandardOutput=append:{log_path}",
            f"--property=StandardError=append:{log_path}",
            "--property=KillMode=control-group",
            "--property=ManagedOOMPreference=omit",
            "--property=OOMScoreAdjust=-500",
        ]
        for key, value in sorted(env.items()):
            if "=" in key or "\x00" in key or "\x00" in str(value):
                continue
            command.append(f"--setenv={key}={value}")
        command.extend(task["launch_command"])
        result = subprocess.run(
            command,
            cwd=self.config["openpi_repo"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "systemd-run failed").strip()
            raise RuntimeError(detail)
        status = self._systemd_show(unit)
        try:
            return int(status.get("MainPID", "0") or 0)
        except ValueError:
            return 0

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
        task["exit_path"] = str(self._exit_path(task_id))
        task["launch_command"] = self._runner_command(task)
        task.pop("waiting_reason", None)
        if task.get("type") == "transfer":
            progress_path = self.root / safe_name(task_id, "task id") / "progress.json"
            task["progress_path"] = str(progress_path)
            env = dict(env)
            env["DASHBOARD_TASK_PROGRESS_PATH"] = str(progress_path)
            env["DASHBOARD_TASK_ID"] = task_id
        task.pop("start_error", None)
        try:
            self._exit_path(task_id).unlink(missing_ok=True)
        except OSError:
            pass
        atomic_json(self._path(task_id), task)

        try:
            if self._systemd_task_backend_enabled():
                try:
                    pid = self._launch_systemd_runner(task, env=env)
                    task.update({
                        "state": "running",
                        "pid": pid,
                        "started_at": now_iso(),
                        "launch_backend": "systemd_user_service",
                    })
                    atomic_json(self._path(task_id), task)
                    self.monitor_wakeup.set()
                    return task
                except Exception as exc:
                    self._append_log(task, f"systemd task launch failed; falling back to direct runner: {exc}")

            task["launch_backend"] = "direct_runner"
            process = self._launch_direct_runner(task, env=env)
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
        metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
        allow_busy_gpus = bool(
            metadata.get("allow_busy_gpus", self.config.get("allow_busy_gpus", False))
        )
        gpu_ids = [int(item) for item in metadata.get("gpu_ids", [])]
        if not gpu_ids:
            return "queued training task has no GPU ids"
        requested = set(gpu_ids)
        managed_busy: dict[int, list[str]] = {}
        for path in self.root.glob("*/task.json"):
            other = read_json(path)
            if not isinstance(other, dict) or other.get("id") == task["id"]:
                continue
            if other.get("state") not in PROCESS_STATES or other.get("type") not in {"train", "eval", "policy"}:
                continue
            overlap = requested.intersection(other.get("metadata", {}).get("gpu_ids", []))
            for gpu_id in overlap:
                managed_busy.setdefault(gpu_id, []).append(other["id"])
        if managed_busy:
            return f"waiting for managed task(s) on GPU(s): {managed_busy}"
        inventory = {gpu["index"]: gpu for gpu in gpu_inventory()}
        unavailable = {
            gpu_id: inventory.get(gpu_id, {}).get("health_issue") or "GPU compute unavailable"
            for gpu_id in gpu_ids
            if inventory.get(gpu_id, {}).get("compute_available") is False
        }
        if unavailable:
            return f"waiting for unavailable GPU(s): {unavailable}"
        external_busy = {
            gpu_id: inventory.get(gpu_id, {}).get("processes", [])
            for gpu_id in gpu_ids
            if inventory.get(gpu_id, {}).get("processes")
        }
        if external_busy and not allow_busy_gpus:
            return f"waiting for busy GPU(s): {external_busy}"
        minimum_free_mib = int(
            metadata.get(
                "minimum_free_gpu_mib",
                self.config.get("training_min_free_gpu_mib", 23_000),
            )
        )
        low_memory = gpu_memory_shortfalls(inventory, gpu_ids, minimum_free_mib)
        if low_memory:
            return f"waiting for GPU free memory: {low_memory}"
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

        norm_config = read_json(Path(str(artifact)).parent / NORM_CONFIG_FILENAME)
        metadata = task.setdefault("metadata", {})
        try:
            expected_contract = complete_action_contract_fingerprint(metadata)
        except ValueError:
            expected_contract = {}
        if expected_contract and (
            not isinstance(norm_config, dict)
            or norm_config.get("version") != NORM_CONFIG_VERSION
            or any(norm_config.get(key) != value for key, value in expected_contract.items())
        ):
            return self._fail_dependency(
                task,
                "normalization dependency raw/model action contract does not match training",
            )
        expected_convention = metadata.get("delivery_action_convention")
        if not expected_contract and expected_convention is not None and (
            not isinstance(norm_config, dict)
            or norm_config.get("delivery_action_convention") != expected_convention
        ):
            return self._fail_dependency(
                task,
                "normalization dependency action convention does not match training: "
                f"expected {expected_convention!r}",
            )
        if isinstance(norm_config, dict):
            metadata["norm_config"] = norm_config
            metadata["norm_batch_size"] = norm_config.get("effective_batch_size")

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
            env=build_environment(
                self.config,
                task.get("metadata", {}).get("gpu_ids", []),
                xla_memory_fraction=task.get("metadata", {}).get("xla_memory_fraction"),
            ),
            raise_on_error=False,
        )

    def _read_exit_info(self, task: dict[str, Any]) -> dict[str, Any] | None:
        exit_path = task.get("exit_path")
        if not exit_path:
            return None
        value = read_json(Path(str(exit_path)))
        if isinstance(value, dict) and isinstance(value.get("returncode"), int):
            return value
        return None

    def _finish_from_returncode(
        self,
        task: dict[str, Any],
        returncode: int,
        *,
        exit_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        was_stopping = task.get("state") == "stopping"
        task["state"] = "stopped" if was_stopping else ("completed" if returncode == 0 else "failed")
        task["returncode"] = int(returncode)
        if exit_info:
            task["exit_info"] = exit_info
            task["finished_at"] = exit_info.get("finished_at") or now_iso()
            if exit_info.get("child_pid"):
                task["child_pid"] = exit_info.get("child_pid")
        else:
            task["finished_at"] = now_iso()
        atomic_json(self._path(task["id"]), task)
        self.processes.pop(task["id"], None)
        return task

    @staticmethod
    def _systemd_status_snapshot(status: dict[str, str]) -> dict[str, str]:
        return {
            key: str(status[key])
            for key in (
                "LoadState",
                "ActiveState",
                "SubState",
                "Result",
                "ExecMainCode",
                "ExecMainStatus",
                "MainPID",
            )
            if status.get(key) not in (None, "")
        }

    def _remember_systemd_status(self, task: dict[str, Any], status: dict[str, str]) -> bool:
        """Persist the small, stable subset needed to explain unit ownership."""
        snapshot = self._systemd_status_snapshot(status)
        if not snapshot:
            return False
        changed = task.get("systemd_status") != snapshot
        if changed:
            task["systemd_status"] = snapshot
            task["systemd_status_updated_at"] = now_iso()
        result = status.get("Result")
        if task.get("systemd_result") != result:
            task["systemd_result"] = result
            changed = True
        return changed

    @staticmethod
    def _systemd_returncode(status: dict[str, str]) -> int:
        """Convert systemd's ExecMainCode/ExecMainStatus into a task rc."""
        try:
            code = int(status.get("ExecMainStatus", "") or 0)
        except (TypeError, ValueError):
            code = 0
        # systemctl show commonly exposes CLD_KILLED/CLD_DUMPED as the
        # numeric values 2/3 rather than the symbolic names.  Normalize both
        # forms to the subprocess convention (negative signal number).
        exec_code = str(status.get("ExecMainCode", "") or "").lower()
        if exec_code in {"killed", "dumped", "2", "3"} and code > 0:
            return -code
        if status.get("Result") == "success" and code == 0:
            return 0
        return code or 1

    def _systemd_terminal_reason(self, task: dict[str, Any], status: dict[str, str]) -> str:
        details = []
        for key in ("Result", "ActiveState", "SubState", "ExecMainCode", "ExecMainStatus"):
            value = status.get(key)
            if value not in (None, ""):
                details.append(f"{key}={value}")
        detail = ", ".join(details) or "no terminal status fields"
        return (
            f"systemd unit {task.get('systemd_unit')} ended without exit.json; "
            f"{detail}"
        )

    def _finish_from_systemd_status(
        self,
        task: dict[str, Any],
        status: dict[str, str],
    ) -> dict[str, Any]:
        """Finish an adopted systemd task when the runner could not write exit.json."""
        self._remember_systemd_status(task, status)
        exit_info = self._read_exit_info(task)
        if exit_info is not None:
            return self._finish_from_returncode(
                task,
                int(exit_info["returncode"]),
                exit_info=exit_info,
            )

        active = status.get("ActiveState")
        result = status.get("Result")
        failed = active == "failed" or result not in (None, "", "success")
        returncode = self._systemd_returncode(status)
        was_stopping = task.get("state") == "stopping"
        task["state"] = "stopped" if was_stopping else ("failed" if failed else "completed")
        task["returncode"] = int(returncode)
        task["finished_at"] = now_iso()
        task["lost_reason"] = self._systemd_terminal_reason(task, status)
        task["systemd_exit_json_missing"] = True
        atomic_json(self._path(task["id"]), task)
        self.processes.pop(task["id"], None)
        return task

    def _refresh_systemd_task(self, task: dict[str, Any]) -> dict[str, Any] | None:
        unit = str(task.get("systemd_unit") or "")
        if not unit:
            return None
        status = self._systemd_show(unit)
        if status.get("_query_error"):
            # A transient systemctl/user-bus failure is not evidence that the
            # task disappeared.  Keep the durable record adoptable on the
            # next refresh instead of incorrectly changing it to ``lost``.
            reason = str(status["_query_error"])
            if task.get("systemd_status_error") != reason:
                task["systemd_status_error"] = reason
                atomic_json(self._path(task["id"]), task)
            return task
        if not status:
            return None
        status_changed = self._remember_systemd_status(task, status)
        active = status.get("ActiveState")
        try:
            main_pid = int(status.get("MainPID", "0") or 0)
        except ValueError:
            main_pid = 0
        if main_pid > 0 and task.get("pid") != main_pid:
            task["pid"] = main_pid
            status_changed = True
        if active in {"active", "activating", "reloading"}:
            previous_state = task.get("state")
            task["state"] = "running" if previous_state != "stopping" else "stopping"
            if previous_state == "lost":
                task["systemd_recovered_at"] = now_iso()
                task["systemd_recovered_from"] = "lost"
                task.pop("finished_at", None)
                task.pop("returncode", None)
                task.pop("lost_reason", None)
                task.pop("systemd_exit_json_missing", None)
                status_changed = True
            if status_changed:
                atomic_json(self._path(task["id"]), task)
            return task
        if active in {"inactive", "failed"}:
            return self._finish_from_systemd_status(task, status)
        if status.get("LoadState") in {"not-found", "bad-setting", "error"}:
            # The unit is known to systemd but no longer exists/is loadable.
            # Treat this as a terminal systemd outcome rather than falling
            # through to PID heuristics and calling it an unexplained loss.
            return self._finish_from_systemd_status(task, status)
        if status_changed:
            atomic_json(self._path(task["id"]), task)
        return None

    def _refresh(self, task: dict[str, Any]) -> dict[str, Any]:
        if task.get("state") in WAITING_STATES:
            return self._refresh_waiting(task)
        task_id = task["id"]
        state = task.get("state")
        if task.get("type") == "transfer":
            progress_path = task.get("progress_path") or (self.root / safe_name(task_id, "task id") / "progress.json")
            task["progress_path"] = str(progress_path)
            progress = read_json(Path(progress_path))
            if isinstance(progress, dict):
                task["progress"] = progress
        systemd_unit = str(task.get("systemd_unit") or "")
        can_recover_systemd = bool(systemd_unit and (state in PROCESS_STATES or state == "lost"))
        if can_recover_systemd:
            systemd_result = self._refresh_systemd_task(task)
            if systemd_result is not None:
                return systemd_result
        if state not in PROCESS_STATES:
            return task
        exit_info = self._read_exit_info(task)
        if exit_info is not None:
            return self._finish_from_returncode(task, int(exit_info["returncode"]), exit_info=exit_info)

        process = self.processes.get(task_id)
        if process is not None:
            rc = process.poll()
            if rc is None:
                task["state"] = "running" if task["state"] != "stopping" else "stopping"
                return task
            return self._finish_from_returncode(task, int(rc), exit_info=self._read_exit_info(task))
        pid = int(task.get("pid", 0) or 0)
        if pid and pid_alive(pid) and process_matches_task(pid, task):
            return task
        if task.get("metadata", {}).get("external") and task.get("type") == "eval" and not (pid and pid_alive(pid)):
            task["state"] = "stopped" if task.get("state") == "stopping" else "completed"
            task["external_exit_unknown"] = True
        else:
            task["state"] = "stopped" if task.get("state") == "stopping" else "lost"
        if pid and pid_alive(pid):
            task["lost_reason"] = "PID is alive but no longer matches the recorded task command"
        elif task.get("launch_backend"):
            task["lost_reason"] = "task runner disappeared before writing an exit status"
        task["finished_at"] = now_iso()
        atomic_json(self._path(task_id), task)
        return task

    def _decorate_eval_result(self, task: dict[str, Any]) -> dict[str, Any]:
        if task.get("type") != "eval":
            return task
        result_path = task.get("metadata", {}).get("result_path")
        result = read_json(Path(result_path)) if result_path else None
        if isinstance(result, dict):
            task["result"] = result
        return task

    def _refresh_all_locked(self) -> list[dict[str, Any]]:
        tasks = []
        for path in self.root.glob("*/task.json"):
            task = read_json(path)
            if isinstance(task, dict):
                tasks.append(self._decorate_eval_result(self._refresh(task)))
        return sorted(tasks, key=lambda item: item.get("created_at", ""), reverse=True)

    def _auto_eval_settings(self, train_task: dict[str, Any]) -> dict[str, Any]:
        metadata = train_task.get("metadata", {})
        explicit = metadata.get("auto_eval")
        save_interval = max(1, int(metadata.get("save_interval", 1000)))
        if isinstance(explicit, dict):
            settings = dict(explicit)
        else:
            # Backward compatibility: active training tasks created before this
            # feature evaluate durable 5000-step checkpoints automatically.
            settings = {
                "enabled": bool(
                    metadata.get(
                        "eval_enabled",
                        train_task.get("state") in PROCESS_STATES,
                    )
                ),
                "every_steps": int(metadata.get("eval_interval_steps", max(5000, save_interval))),
                "batch_size": int(metadata.get("eval_batch_size", 1)),
                "num_workers": int(metadata.get("eval_num_workers", 2)),
                "max_batches": int(metadata.get("eval_max_batches", 50)),
                "seed": int(metadata.get("eval_seed", metadata.get("split_seed", 0))),
                "minimum_free_gpu_mib": int(
                    metadata.get(
                        "eval_minimum_free_gpu_mib",
                        self.config.get("evaluation_min_free_gpu_mib", 23_000),
                    )
                ),
                "xla_memory_fraction": float(
                    metadata.get(
                        "eval_xla_memory_fraction",
                        self.config.get("evaluation_xla_memory_fraction", 0.85),
                    )
                ),
            }
        settings.setdefault("enabled", True)
        settings.setdefault("every_steps", max(5000, save_interval))
        settings.setdefault("batch_size", 1)
        settings.setdefault("num_workers", 2)
        settings.setdefault("max_batches", 50)
        settings.setdefault("seed", int(metadata.get("split_seed", 0)))
        settings.setdefault(
            "minimum_free_gpu_mib",
            int(self.config.get("evaluation_min_free_gpu_mib", 23_000)),
        )
        settings.setdefault(
            "xla_memory_fraction",
            float(self.config.get("evaluation_xla_memory_fraction", 0.85)),
        )
        return settings

    def _auto_eval_command(
        self,
        train_task: dict[str, Any],
        checkpoint: Path,
        result_path: Path,
        settings: dict[str, Any],
    ) -> list[str]:
        metadata = train_task.get("metadata", {})
        contract = {
            **metadata,
            "schema": metadata["schema"],
            "raw_gripper_semantics": metadata["raw_gripper_semantics"],
            "model_gripper_semantics": metadata["gripper_semantics"],
        }
        return [
            self.config["openpi_python"],
            str(APP_DIR / "eval_heldout_loss.py"),
            "--checkpoint", str(checkpoint),
            "--result-json", str(result_path),
            "--dataset-id", str(metadata["dataset_id"]),
            "--arm-mode", str(metadata["arm_mode"]),
            "--arm-side", str(metadata["arm_side"]),
            "--schema", str(metadata["schema"]),
            "--model-variant", str(metadata.get("model_variant", "pi05")),
            "--assets-base-dir", self.config["assets_base_dir"],
            "--checkpoint-base-dir", self.config["checkpoint_base_dir"],
            "--base-checkpoint", str(metadata.get("base_checkpoint", self.config["base_checkpoint"])),
            "--batch-size", str(settings["batch_size"]),
            "--num-workers", str(settings["num_workers"]),
            "--max-batches", str(settings["max_batches"]),
            "--eval-seed", str(settings["seed"]),
        ] + action_contract_command_args(contract)

    def _create_auto_eval_locked(
        self,
        train_task: dict[str, Any],
        checkpoint_step: int,
        checkpoint: Path,
        settings: dict[str, Any],
        *,
        gpu_id: int | None,
        skip_reason: str | None = None,
    ) -> dict[str, Any]:
        train_metadata = train_task.get("metadata", {})
        metadata = {
            "parent_train_task_id": train_task["id"],
            "depends_on": train_task["id"],
            "automatic": True,
            "trigger": "checkpoint_complete",
            "dedupe_key": f"{train_task['id']}:{checkpoint_step}",
            "dataset_id": train_metadata.get("dataset_id"),
            "arm_mode": train_metadata.get("arm_mode"),
            "arm_side": train_metadata.get("arm_side"),
            "schema": train_metadata.get("schema"),
            "model_variant": train_metadata.get("model_variant"),
            "checkpoint": str(checkpoint),
            "checkpoint_step": checkpoint_step,
            "gpu_ids": [] if gpu_id is None else [gpu_id],
            "test_ratio": train_metadata.get("test_ratio"),
            "split_seed": train_metadata.get("split_seed"),
            "test_episodes": train_metadata.get("test_episodes"),
            "test_episode_indexes": train_metadata.get("test_episode_indexes", []),
            "batch_size": int(settings["batch_size"]),
            "num_workers": int(settings["num_workers"]),
            "max_batches": int(settings["max_batches"]),
            "eval_seed": int(settings["seed"]),
            "xla_memory_fraction": float(settings["xla_memory_fraction"]),
        }
        task = self._new_task(
            "eval",
            [],
            metadata,
            state="skipped" if skip_reason else "starting",
        )
        result_path = self.root / task["id"] / "result.json"
        task["metadata"]["result_path"] = str(result_path)
        task["command"] = self._auto_eval_command(
            train_task, checkpoint, result_path, settings
        )
        if skip_reason:
            task["skip_reason"] = skip_reason
            task["metadata"]["skip_reason"] = skip_reason
            task["finished_at"] = now_iso()
            atomic_json(self._path(task["id"]), task)
            self._append_log(task, f"evaluation skipped: {skip_reason}")
            return task
        atomic_json(self._path(task["id"]), task)
        self._append_log(
            task,
            f"starting held-out evaluation for checkpoint step {checkpoint_step} on GPU {gpu_id}",
        )
        return self._launch(
            task,
            env=build_environment(
                self.config,
                [int(gpu_id)],
                xla_memory_fraction=float(settings["xla_memory_fraction"]),
            ),
            raise_on_error=False,
        )

    def _reconcile_auto_evals_locked(self, task_list: list[dict[str, Any]]) -> None:
        existing_keys = {
            task.get("metadata", {}).get("dedupe_key")
            for task in task_list
            if task.get("type") == "eval"
        }
        allowed_gpu_ids = set(map(int, self.config.get("allowed_gpu_ids", [])))
        for train_task in sorted(
            (task for task in task_list if task.get("type") == "train"),
            key=lambda item: item.get("created_at", ""),
        ):
            if train_task.get("state") not in PROCESS_STATES | {"completed"}:
                continue
            metadata = train_task.get("metadata", {})
            settings = self._auto_eval_settings(train_task)
            if not settings.get("enabled") or int(metadata.get("test_episodes", 0)) <= 0:
                continue
            every_steps = int(settings["every_steps"])
            if every_steps <= 0:
                continue
            eval_after_step = int(metadata.get("eval_after_step", 0))
            checkpoints = [
                (step, path)
                for step, path in complete_checkpoint_steps(Path(metadata.get("checkpoint_dir", "")))
                if step > eval_after_step and step % every_steps == 0
                and f"{train_task['id']}:{step}" not in existing_keys
            ]
            if not checkpoints:
                continue
            # At most one new eval decision per train per monitor cycle.
            checkpoint_step, checkpoint = checkpoints[0]
            active_eval = next((
                task for task in task_list
                if task.get("type") == "eval"
                and task.get("state") in PROCESS_STATES
                and task.get("metadata", {}).get("parent_train_task_id") == train_task["id"]
            ), None)
            if active_eval is not None:
                created = self._create_auto_eval_locked(
                    train_task,
                    checkpoint_step,
                    checkpoint,
                    settings,
                    gpu_id=None,
                    skip_reason=f"previous_eval_still_running:{active_eval['id']}",
                )
            else:
                inventory = gpu_inventory()
                gpu_id = select_idle_eval_gpu(
                    task_list,
                    inventory,
                    allowed_gpu_ids=allowed_gpu_ids,
                    minimum_free_mib=int(settings["minimum_free_gpu_mib"]),
                )
                created = self._create_auto_eval_locked(
                    train_task,
                    checkpoint_step,
                    checkpoint,
                    settings,
                    gpu_id=gpu_id,
                    skip_reason="no_idle_gpu" if gpu_id is None else None,
                )
            task_list.append(created)
            existing_keys.add(created.get("metadata", {}).get("dedupe_key"))

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return self._refresh_all_locked()

    def get(self, task_id: str) -> dict[str, Any]:
        with self.lock:
            task = read_json(self._path(task_id))
            if not isinstance(task, dict):
                raise FileNotFoundError(task_id)
            return self._decorate_eval_result(self._refresh(task))

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

    def _active_task_pids(self, task_types: set[str] | None = None, *, include_external: bool = False) -> set[int]:
        pids: set[int] = set()
        for path in self.root.glob("*/task.json"):
            task = read_json(path)
            if not isinstance(task, dict):
                continue
            if task_types is not None and task.get("type") not in task_types:
                continue
            if task.get("state") not in PROCESS_STATES:
                continue
            if not include_external and task.get("metadata", {}).get("external"):
                continue
            try:
                pids.add(int(task.get("pid", 0)))
            except (TypeError, ValueError):
                continue
        pids.discard(0)
        return pids

    def _active_managed_policy_process_tree(self) -> tuple[set[int], dict[int, str]]:
        """Return active managed Policy root/descendant PIDs.

        Dashboard-launched tasks run through ``task_runner.py``.  The listening
        Policy server is therefore a child of the recorded task PID.  External
        discovery must ignore those children, otherwise one Dashboard-started
        Policy is also adopted as a second "external" Policy row.
        """
        pids: set[int] = set()
        descendant_owner: dict[int, str] = {}
        children_by_parent: dict[int, set[int]] | None = None
        for path in self.root.glob("*/task.json"):
            task = read_json(path)
            if not isinstance(task, dict):
                continue
            if task.get("type") != "policy" or task.get("state") not in PROCESS_STATES:
                continue
            if task.get("metadata", {}).get("external"):
                continue
            try:
                root_pid = int(task.get("pid", 0) or 0)
            except (TypeError, ValueError):
                continue
            if root_pid <= 0:
                continue
            pids.add(root_pid)
            if not (pid_alive(root_pid) and process_matches_task(root_pid, task)):
                continue
            if children_by_parent is None:
                children_by_parent = process_children_by_parent()
            for pid in process_descendant_pids(root_pid, children_by_parent):
                pids.add(pid)
                descendant_owner.setdefault(pid, str(task.get("id", path.parent.name)))
        pids.discard(0)
        return pids, descendant_owner

    def _active_policy_pids(self) -> set[int]:
        pids, _ = self._active_managed_policy_process_tree()
        return pids

    def _prune_external_policy_duplicates(self, descendant_owner: dict[int, str]) -> list[str]:
        """Remove bogus external Policy rows that are actually managed children."""
        if not descendant_owner:
            return []
        removed: list[str] = []
        for path in list(self.root.glob("*/task.json")):
            task = read_json(path)
            if not isinstance(task, dict):
                continue
            if task.get("type") != "policy" or task.get("state") not in PROCESS_STATES:
                continue
            metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
            if not metadata.get("external"):
                continue
            try:
                pid = int(task.get("pid", 0) or 0)
            except (TypeError, ValueError):
                continue
            owner_id = descendant_owner.get(pid)
            if not owner_id:
                continue
            task_id = str(task.get("id", path.parent.name))
            try:
                self._append_log(
                    task,
                    f"removing duplicate external Policy record; pid {pid} is managed by {owner_id}",
                )
            except OSError:
                pass
            self.processes.pop(task_id, None)
            shutil.rmtree(path.parent, ignore_errors=True)
            removed.append(task_id)
        return removed

    def _active_eval_pids(self) -> set[int]:
        return self._active_task_pids({"eval"}, include_external=False)

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
            active_pids, descendant_owner = self._active_managed_policy_process_tree()
            self._prune_external_policy_duplicates(descendant_owner)
            adopted = []
            for candidate in discover_external_policy_candidates(self.config, ignored_pids=active_pids):
                adopted.append(self.adopt_external_policy(**candidate))
            return adopted

    def adopt_external_eval(
        self,
        *,
        pid: int,
        command: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = f"eval-external-{pid}"
        task_dir = self.root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        task = read_json(task_dir / "task.json")
        if not isinstance(task, dict):
            task = {
                "id": task_id,
                "type": "eval",
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
            self._append_log(task, "adopted external RoboTwin eval process discovered on a managed GPU")
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

    def discover_external_evals(self) -> list[dict[str, Any]]:
        with self.lock:
            active_pids = self._active_eval_pids()
            adopted = []
            for candidate in discover_external_eval_candidates(self.config, ignored_pids=active_pids):
                adopted.append(self.adopt_external_eval(**candidate))
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
            unit = str(task.get("systemd_unit") or "")
            if unit:
                sig = "SIGKILL" if force else "SIGTERM"
                cmd = ["systemctl", "--user", "kill", "--kill-who=all", "-s", sig, unit]
                if not force:
                    cmd = ["systemctl", "--user", "stop", unit]
                try:
                    subprocess.run(cmd, check=False, timeout=30)
                except Exception as exc:
                    task["stop_signal_error"] = str(exc)
                task["state"] = "stopping"
                task["stop_requested_at"] = now_iso()
                task["stop_signal"] = "SIGKILL" if force else "SIGTERM"
                atomic_json(self._path(task_id), task)
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

    def delete(self, task_id: str) -> dict[str, Any]:
        """Delete a terminal task record and its log without touching outputs/checkpoints."""
        with self.lock:
            task = self.get(task_id)
            state = task.get("state")
            if state not in TERMINAL_STATES:
                raise ValueError(f"cannot delete active task {task_id} in state {state}")

            process = self.processes.get(task_id)
            if process is not None and process.poll() is None:
                raise ValueError(f"cannot delete task {task_id} while its process is still alive")

            active_dependents = []
            for path in self.root.glob("*/task.json"):
                dependent = read_json(path)
                if not isinstance(dependent, dict) or dependent.get("id") == task_id:
                    continue
                dependency_id = (
                    dependent.get("metadata", {}).get("depends_on")
                    or dependent.get("dependency", {}).get("task_id")
                )
                if dependency_id != task_id:
                    continue
                if dependent.get("state") not in TERMINAL_STATES:
                    active_dependents.append(str(dependent.get("id", path.parent.name)))
            if active_dependents:
                names = ", ".join(sorted(active_dependents))
                raise ValueError(f"cannot delete task {task_id}; active dependent task(s): {names}")

            task_dir = self._path(task_id).parent
            self.processes.pop(task_id, None)
            shutil.rmtree(task_dir)
            return {"deleted": True, "task": task}

    def delete_many(self, task_ids: list[str]) -> dict[str, Any]:
        """Delete multiple terminal task records and their logs as one validated batch.

        Outputs, checkpoints, and any external log paths are intentionally left
        untouched, matching :meth:`delete`.  Validate the complete selection
        before removing anything so a batch containing an active task or an
        active dependency cannot result in a partially deleted selection.
        """
        if not isinstance(task_ids, list):
            raise ValueError("task_ids must be a list")
        if not task_ids:
            raise ValueError("task_ids must not be empty")
        if len(task_ids) > 200:
            raise ValueError("cannot delete more than 200 tasks at once")

        normalized_ids: list[str] = []
        seen: set[str] = set()
        for raw_task_id in task_ids:
            if not isinstance(raw_task_id, str) or not raw_task_id.strip():
                raise ValueError("task_ids must contain non-empty strings")
            task_id = raw_task_id
            if task_id in seen:
                continue
            seen.add(task_id)
            normalized_ids.append(task_id)

        with self.lock:
            selected_ids = set(normalized_ids)
            selected_tasks: dict[str, dict[str, Any]] = {}
            for task_id in normalized_ids:
                task = self.get(task_id)
                state = task.get("state")
                if state not in TERMINAL_STATES:
                    raise ValueError(f"cannot delete active task {task_id} in state {state}")

                process = self.processes.get(task_id)
                if process is not None and process.poll() is None:
                    raise ValueError(f"cannot delete task {task_id} while its process is still alive")
                selected_tasks[task_id] = task

            active_dependents: dict[str, list[str]] = {}
            for path in self.root.glob("*/task.json"):
                dependent = read_json(path)
                if not isinstance(dependent, dict):
                    continue
                dependent_id = str(dependent.get("id", path.parent.name))
                if dependent_id in selected_ids:
                    continue
                dependency_id = (
                    dependent.get("metadata", {}).get("depends_on")
                    or dependent.get("dependency", {}).get("task_id")
                )
                if dependency_id not in selected_ids:
                    continue
                if dependent.get("state") not in TERMINAL_STATES:
                    active_dependents.setdefault(str(dependency_id), []).append(dependent_id)
            if active_dependents:
                details = "; ".join(
                    f"{task_id}: {', '.join(sorted(dependents))}"
                    for task_id, dependents in sorted(active_dependents.items())
                )
                raise ValueError(f"cannot delete selected tasks; active dependent task(s): {details}")

            task_dirs = {
                task_id: self._path(task_id).parent
                for task_id in normalized_ids
            }
            for task_id, task_dir in task_dirs.items():
                if not task_dir.is_dir():
                    raise FileNotFoundError(task_id)

            for task_id, task_dir in task_dirs.items():
                self.processes.pop(task_id, None)
                shutil.rmtree(task_dir)

            return {
                "deleted": True,
                "deleted_count": len(normalized_ids),
                "task_ids": normalized_ids,
                "tasks": [selected_tasks[task_id] for task_id in normalized_ids],
            }

    @staticmethod
    def _task_command_tokens(task: dict[str, Any]) -> list[str]:
        command = task.get("command", [])
        tokens = [str(item) for item in command] if isinstance(command, list) else []
        # Slurm tasks with an automatic dataset sync are wrapped as
        # ``bash -lc '<sync> && python slurm_job_runner.py ...'``.  Decode the
        # shell layer just enough to recover Dashboard runner flags such as
        # ``--job-name``; failures simply fall back to metadata-derived names.
        if len(tokens) >= 3 and tokens[0] in {"bash", "/bin/bash"} and "-lc" in tokens:
            try:
                script = tokens[tokens.index("-lc") + 1]
                tokens.extend(shlex.split(script))
            except Exception:
                pass
        return tokens

    @classmethod
    def _task_command_option(cls, task: dict[str, Any], option: str) -> str | None:
        tokens = cls._task_command_tokens(task)
        for index, token in enumerate(tokens):
            if token == option and index + 1 < len(tokens):
                return tokens[index + 1]
            prefix = option + "="
            if token.startswith(prefix):
                return token[len(prefix):]
        return None

    def _slurm_job_name(self, task: dict[str, Any]) -> str | None:
        explicit = self._task_command_option(task, "--job-name")
        if explicit:
            return safe_name(explicit, "slurm job name")
        metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
        exp_name = metadata.get("exp_name")
        if task.get("type") == "train" and exp_name:
            prefix = "sim" if self.config.get("dashboard_profile") == "simulation" else "real"
            return safe_name(f"{prefix}_train_{exp_name}", "slurm job name")
        dataset_id = metadata.get("dataset_id")
        if task.get("type") == "eval" and dataset_id:
            return safe_name(f"sim_eval_{dataset_id}", "slurm job name")
        return None

    @staticmethod
    def _slurm_job_id_from_log(log_text: str) -> str | None:
        matches = DASHBOARD_SLURM_JOB_ID.findall(log_text or "")
        return matches[-1] if matches else None

    def _remote_slurm_log_targets(
        self,
        task: dict[str, Any],
        local_log: str,
    ) -> tuple[str, dict[str, str]] | None:
        metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
        target_name = metadata.get("slurm_target") or metadata.get("cluster_target")
        if not target_name or str(metadata.get("runtime") or "") != "slurm":
            return None
        targets = self.config.get("cluster_targets", {})
        target = targets.get(str(target_name))
        if not isinstance(target, dict):
            return None
        submit_host = str(metadata.get("slurm_submit_host") or target.get("submit_host") or "")
        job_id = self._slurm_job_id_from_log(local_log)
        job_name = self._slurm_job_name(task)
        workdir = str(target.get("workdir") or "")
        if not submit_host or not job_id or not job_name or not workdir:
            return None
        log_dir = str(target.get("log_dir") or f"{workdir.rstrip('/')}/logs/dashboard_slurm").rstrip("/")
        return submit_host, {
            "stdout": f"{log_dir}/{job_name}_{job_id}.out",
            "stderr": f"{log_dir}/{job_name}_{job_id}.err",
        }

    @staticmethod
    def _remote_tail_command(paths: dict[str, str], max_bytes_per_file: int) -> str:
        parts = []
        for label, path in paths.items():
            quoted_label = shlex.quote(str(label))
            quoted_path = shlex.quote(path)
            parts.append(
                "if [ -f {path} ]; then "
                "printf '\\n[dashboard] remote slurm %s: %s\\n' {label} {path}; "
                "tail -c {max_bytes} {path}; "
                "fi".format(
                    label=quoted_label,
                    path=quoted_path,
                    max_bytes=int(max_bytes_per_file),
                )
            )
        return "; ".join(parts)

    def _remote_slurm_log_tail(
        self,
        task: dict[str, Any],
        local_log: str,
        max_bytes: int,
    ) -> str:
        target_info = self._remote_slurm_log_targets(task, local_log)
        if target_info is None:
            return ""
        host, paths = target_info
        per_file = max(4096, min(max_bytes, max_bytes // max(1, len(paths))))
        try:
            result = subprocess.run(
                [*SSH_COMMAND, host, self._remote_tail_command(paths, per_file)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
            )
        except Exception as exc:
            return f"\n[dashboard] remote slurm log tail failed: {exc}\n"
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1000:]
            return f"\n[dashboard] remote slurm log tail failed rc={result.returncode}: {detail}\n"
        return result.stdout

    def log_tail(self, task_id: str, max_bytes: int = 64 * 1024, *, include_remote_slurm: bool = False) -> str:
        task = read_json(self._path(task_id))
        path = self._log_path(task_id)
        external_log = None
        if isinstance(task, dict):
            external_log = task.get("metadata", {}).get("external_log_path")
        if external_log:
            candidate = Path(str(external_log))
            if candidate.is_file():
                path = candidate
        if not path.exists():
            return ""
        with path.open("rb") as handle:
            size = path.stat().st_size
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
        if include_remote_slurm and isinstance(task, dict) and DASHBOARD_SLURM_LOG_STREAM_MARKER not in text:
            remote_text = self._remote_slurm_log_tail(task, text, max_bytes)
            if remote_text:
                text = (text + "\n" + remote_text)[-max_bytes:]
        return text

    def _task_log_path_for_read(self, task: dict[str, Any]) -> Path:
        path = self._log_path(str(task["id"]))
        external_log = task.get("metadata", {}).get("external_log_path")
        if external_log:
            candidate = Path(str(external_log))
            if candidate.is_file():
                return candidate
        return path

    @staticmethod
    def _first_training_metric_step(path: Path, max_scan_bytes: int = 64 * 1024 * 1024) -> int | None:
        """Find an early Step metric without materializing a large log in memory."""
        scanned = 0
        try:
            with path.open("rb") as handle:
                for raw_line in handle:
                    scanned += len(raw_line)
                    if scanned > max_scan_bytes:
                        return None
                    line = ANSI_ESCAPE.sub("", raw_line.decode("utf-8", errors="replace")).strip()
                    match = TRAIN_STEP.search(line)
                    if not match:
                        continue
                    if TRAIN_METRIC.findall(match.group(2)):
                        return int(match.group(1))
        except (OSError, ValueError):
            return None
        return None

    def training_metrics_probe(self, task: dict[str, Any], max_bytes: int = 512 * 1024) -> dict[str, Any] | None:
        """Return a cached, lightweight hint for whether a train task has curves.

        The full metrics endpoint may read multi-MiB logs and, for Slurm tasks,
        may SSH to a submit host.  The Dashboard task table refreshes often, so
        this probe intentionally reads only the local/external log tail and
        caches by size/mtime.  Slurm tasks without local streamed metrics are
        marked ``unknown`` instead of ``no_metrics`` to avoid encouraging users
        to delete a task that may still have remote curves.
        """
        if task.get("type") != "train" or not task.get("id"):
            return None
        task_id = str(task["id"])
        metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
        runtime = str(metadata.get("runtime") or "")
        path = self._task_log_path_for_read(task)
        if not path.exists():
            status = "unknown" if runtime == "slurm" else "no_metrics"
            return {
                "status": status,
                "has_points": None if status == "unknown" else False,
                "total_points": 0,
                "latest_step": 0,
                "source": "missing_log",
            }
        try:
            stat = path.stat()
        except OSError:
            return {
                "status": "unknown",
                "has_points": None,
                "total_points": 0,
                "latest_step": 0,
                "source": "stat_failed",
            }
        cache_key = (
            str(path),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            runtime,
            int(max_bytes),
        )
        cached = self.training_metric_probe_cache.get(task_id)
        if cached and cached[0] == cache_key:
            return dict(cached[1])
        try:
            with path.open("rb") as handle:
                handle.seek(max(0, stat.st_size - max_bytes))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            payload = {
                "status": "unknown",
                "has_points": None,
                "total_points": 0,
                "latest_step": 0,
                "source": "read_failed",
            }
            self.training_metric_probe_cache[task_id] = (cache_key, payload)
            return dict(payload)
        metrics = parse_training_metrics(text, max_points=2)
        total_points = int(metrics.get("total_points", 0) or 0)
        points = metrics.get("points") or []
        latest_step = int(points[-1].get("step", 0)) if points else 0
        partial = False
        if total_points == 0 and stat.st_size > max_bytes:
            first_step = self._first_training_metric_step(path)
            if first_step is not None:
                total_points = 1
                latest_step = first_step
                partial = True
        if total_points > 0:
            status = "has_metrics"
            has_points: bool | None = True
        elif runtime == "slurm" and DASHBOARD_SLURM_LOG_STREAM_MARKER not in text:
            status = "unknown"
            has_points = None
        else:
            status = "no_metrics"
            has_points = False
        payload = {
            "status": status,
            "has_points": has_points,
            "total_points": total_points,
            "latest_step": latest_step,
            "series": list(metrics.get("series", []))[:12],
            "source": "local_file_scan" if partial else "local_tail",
            "tail_bytes": min(int(stat.st_size), int(max_bytes)),
        }
        if partial:
            payload["partial"] = True
        self.training_metric_probe_cache[task_id] = (cache_key, payload)
        return dict(payload)


class UploadManager:
    def __init__(self, config: dict[str, Any], dataset_editor: DatasetEditor):
        self.config = config
        self.dataset_editor = dataset_editor
        self.root = Path(config["workspace_root"]) / "uploads"
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_root = Path(config["dataset_root"])
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        for origin in ("real", "simulation"):
            (self.root / origin).mkdir(parents=True, exist_ok=True)
        self.locks: dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()

    def _lock(self, upload_id: str) -> threading.Lock:
        with self.global_lock:
            return self.locks.setdefault(upload_id, threading.Lock())

    def _dir(self, upload_id: str) -> Path:
        upload_id = safe_name(upload_id, "upload id")
        prefix = upload_id.split("-", 1)[0]
        if prefix in {"real", "simulation"}:
            return self.root / prefix / upload_id
        legacy = self.root / upload_id
        if legacy.exists():
            return legacy
        for origin in ("real", "simulation"):
            candidate = self.root / origin / upload_id
            if candidate.exists():
                return candidate
        return legacy

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
        dataset_origin = normalize_dataset_origin(
            payload.get("dataset_origin", "real"), allow_unknown=False
        )
        digest = hashlib.sha256(
            f"{dataset_origin}\0{dataset_name}\0{size}\0{sha256}".encode()
        ).hexdigest()[:24]
        upload_id = f"{dataset_origin}-{digest}"
        upload_dir = self.root / dataset_origin / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / "chunks").mkdir(exist_ok=True)
        state_path = upload_dir / "upload.json"
        state = read_json(state_path)
        expected = {
            "id": upload_id,
            "dataset_name": dataset_name,
            "dataset_origin": dataset_origin,
            "size": size,
            "sha256": sha256,
            "chunk_size": chunk_size,
            "chunk_count": (size + chunk_size - 1) // chunk_size,
            "overwrite": overwrite,
            "merge": merge,
        }
        if isinstance(state, dict):
            for key in ("dataset_name", "dataset_origin", "size", "sha256", "chunk_size"):
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
                requested_origin = normalize_dataset_origin(
                    state.get("dataset_origin", "real"), allow_unknown=False
                )
                target = self.dataset_root / dataset_name
                if bool(state.get("merge")) and target.is_dir():
                    target_info = read_json(target / "meta" / "info.json", {})
                    existing_origin = dataset_origin_info(
                        dataset_name, target, target_info
                    )["dataset_origin"]
                    if existing_origin not in {"unknown", requested_origin}:
                        raise ValueError(
                            f"cannot merge {requested_origin} upload into "
                            f"{existing_origin} dataset {dataset_name}"
                        )
                result = self.dataset_editor.install_upload(
                    dataset_name,
                    staging,
                    overwrite=bool(state.get("overwrite")),
                    merge=bool(state.get("merge")),
                    dataset_origin=requested_origin,
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
        runtime = read_json(session_dir / "runtime.json")
        payload = payload if isinstance(payload, dict) else {}
        connections = connections if isinstance(connections, dict) else {}
        runtime = runtime if isinstance(runtime, dict) else {}
        control = self.control_for_task(task)
        received_at = payload.get("received_at")
        now = time.time()
        age_s = max(0.0, now - float(received_at)) if received_at is not None else None
        # ``target_time_error_s`` is signed: positive means the active command
        # is behind its target timestamp, negative means the target is still in
        # the future. Prefer the new explicit field, then preserve compatibility
        # with older telemetry payloads that only stored ``target_age_s`` or the
        # millisecond form.
        timed_target = payload.get("client_timed_target")
        timed_target = timed_target if isinstance(timed_target, dict) else {}
        target_age_at_snapshot = payload.get("client_target_age_s")
        if target_age_at_snapshot is None:
            target_age_at_snapshot = payload.get("client_target_time_error_ms")
            if target_age_at_snapshot is not None:
                try:
                    target_age_at_snapshot = float(target_age_at_snapshot) / 1000.0
                except (TypeError, ValueError):
                    target_age_at_snapshot = None
        if target_age_at_snapshot is None:
            target_age_at_snapshot = timed_target.get("target_time_error_s", timed_target.get("target_age_s"))
        timing_snapshot_at = payload.get("client_timing_snapshot_at")
        current_target_age_s = None
        try:
            target_age_at_snapshot = float(target_age_at_snapshot)
            if math.isfinite(target_age_at_snapshot):
                current_target_age_s = target_age_at_snapshot
                if timing_snapshot_at is not None:
                    elapsed_since_snapshot = now - float(timing_snapshot_at)
                    if math.isfinite(elapsed_since_snapshot) and abs(elapsed_since_snapshot) <= 10.0:
                        current_target_age_s += elapsed_since_snapshot
        except (TypeError, ValueError):
            current_target_age_s = None
        process_active = task.get("state") in {"starting", "running", "stopping"}
        client_connected = process_active and bool(connections.get("client_connected", False))
        client_allow = bool(payload.get("client_allow_execution", False))
        client_state = str(payload.get("client_execution_state", "unknown"))
        runtime_in_flight = bool(runtime.get("in_flight", False))
        reported_in_flight = payload.get("client_in_flight")
        client_in_flight = (
            runtime_in_flight
            if reported_in_flight is None
            else bool(reported_in_flight) or runtime_in_flight
        )
        horizon_status = policy_horizon_status(payload, metadata)
        time_contract_status = policy_time_contract_status(payload, metadata)
        dual_gate_open = bool(
            process_active
            and client_connected
            and age_s is not None
            and age_s <= self.max_age_s
            and control["mode"] == "execute"
            and client_allow
            and horizon_status["horizon_execution_ready"]
            and time_contract_status["time_contract_ready"]
        )
        return {
            **payload,
            "task_id": task["id"],
            "policy_port": metadata.get("port"),
            "telemetry_session": session,
            "age_s": round(age_s, 3) if age_s is not None else None,
            "fresh": age_s is not None and age_s <= self.max_age_s,
            "client_current_target_age_s": (
                round(current_target_age_s, 4)
                if current_target_age_s is not None
                else None
            ),
            "client_current_target_time_error_ms": (
                round(current_target_age_s * 1000.0, 2)
                if current_target_age_s is not None
                else None
            ),
            "max_age_s": self.max_age_s,
            "client_connected": client_connected,
            "active_clients": int(connections.get("active_clients", 0)) if process_active else 0,
            "client_addresses": connections.get("client_addresses", []) if process_active else [],
            "connection_event": connections.get("event"),
            "connection_updated_at": connections.get("updated_at"),
            "execution_control": control,
            "client_allow_execution": client_allow,
            "client_execution_state": client_state,
            "client_in_flight": client_in_flight,
            "policy_in_flight": runtime_in_flight,
            "policy_active_inferences": _positive_int_or_none(runtime.get("active_inferences")) or 0,
            "policy_inference_started_at": runtime.get("last_inference_started_at"),
            "policy_inference_finished_at": runtime.get("last_inference_finished_at"),
            **horizon_status,
            **time_contract_status,
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
        allowed = {"cam_high", "cam_wrist", "cam_left_wrist", "cam_right_wrist"}
        if view not in allowed:
            raise ValueError(f"view must be one of {sorted(allowed)}")
        path = self._session_dir(session) / f"{view}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"no policy telemetry image: {view}")
        return path


def _nvidia_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (AttributeError, TypeError, ValueError):
        return None


def gpu_memory_shortfalls(
    inventory: dict[int, dict[str, Any]],
    gpu_ids: list[int],
    minimum_free_mib: int,
    *,
    ignored_pids: set[int] | None = None,
) -> dict[int, dict[str, int]]:
    if minimum_free_mib <= 0:
        return {}
    ignored_pids = ignored_pids or set()
    shortfalls: dict[int, dict[str, int]] = {}
    for gpu_id in gpu_ids:
        gpu = inventory.get(gpu_id, {})
        reclaimable_mib = sum(
            max(0, int(process.get("memory_mib") or 0))
            for process in gpu.get("processes", [])
            if int(process.get("pid", -1)) in ignored_pids
        )
        free_mib = max(
            0,
            int(gpu.get("memory_total_mib", 0))
            - max(0, int(gpu.get("memory_used_mib", 0)) - reclaimable_mib),
        )
        if free_mib < minimum_free_mib:
            shortfalls[gpu_id] = {
                "free_mib": free_mib,
                "required_mib": minimum_free_mib,
            }
    return shortfalls


def blocking_gpu_processes(
    processes: list[dict[str, Any]],
    *,
    small_process_memory_mib: int,
    small_process_total_mib: int,
) -> list[dict[str, Any]]:
    """Return GPU occupants that should prevent a new task from starting.

    Low-memory simulation or visualization processes can share a GPU when
    their individual and aggregate memory footprints stay below the limits.
    Unknown memory usage remains blocking.
    """
    if not processes:
        return []
    if small_process_memory_mib <= 0 or small_process_total_mib <= 0:
        return list(processes)
    total = 0
    for process in processes:
        memory_mib = process.get('memory_mib')
        if not isinstance(memory_mib, int) or memory_mib < 0:
            return list(processes)
        if memory_mib > small_process_memory_mib:
            return list(processes)
        total += memory_mib
    return [] if total <= small_process_total_mib else list(processes)


def process_owner_map(pids: Iterable[int]) -> dict[int, str]:
    unique_pids = sorted({int(pid) for pid in pids if int(pid) > 0})
    if not unique_pids:
        return {}
    try:
        proc = subprocess.run(
            ["ps", "-o", "pid=,user=", "-p", ",".join(str(pid) for pid in unique_pids)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return {}
    owners: dict[int, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid = _nvidia_int(parts[0])
        if pid is not None and parts[1].strip():
            owners[pid] = parts[1].strip()
    return owners


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
    raw_processes: list[tuple[str, int, str | None, int | None]] = []
    unavailable_uuids: set[str] = set()
    for line in process_lines:
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        pid = _nvidia_int(parts[1])
        if pid is None:
            # NVIDIA occasionally reports stale/driver-only compute contexts as
            # ``uuid, [N/A], [N/A], [N/A]``. In practice CUDA no longer exposes
            # that physical GPU, so keep the Dashboard alive but mark it unsafe.
            unavailable_uuids.add(parts[0])
            continue
        raw_processes.append(
            (
                parts[0],
                pid,
                None if parts[2] == "[N/A]" else parts[2],
                _nvidia_int(parts[3]),
            )
        )
    process_owners = process_owner_map(pid for _, pid, _, _ in raw_processes)
    processes: dict[str, list[dict[str, Any]]] = {}
    for uuid, pid, name, memory_mib in raw_processes:
        processes.setdefault(uuid, []).append(
            {
                "pid": pid,
                "user": process_owners.get(pid),
                "name": name,
                "memory_mib": memory_mib,
            }
        )
    gpus = []
    for line in gpu_lines:
        parts = [part.strip() for part in line.split(",", 4)]
        if len(parts) != 5:
            continue
        index = _nvidia_int(parts[0])
        if index is None:
            continue
        gpus.append(
            {
                "index": index,
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": _nvidia_int(parts[3]) or 0,
                "memory_used_mib": _nvidia_int(parts[4]) or 0,
                "processes": processes.get(parts[1], []),
                "compute_available": parts[1] not in unavailable_uuids,
                "health_issue": (
                    None
                    if parts[1] not in unavailable_uuids
                    else "nvidia-smi reports an unavailable compute context ([N/A])"
                ),
            }
        )
    return gpus


def cuda_visible_devices(gpu_ids: list[int], inventory: list[dict[str, Any]] | None = None) -> str:
    """Address physical GPUs by UUID so broken/missing ordinals cannot remap ids."""
    inventory = gpu_inventory() if inventory is None else inventory
    by_index = {int(gpu["index"]): gpu for gpu in inventory}
    uuids = [str(by_index.get(gpu_id, {}).get("uuid", "")) for gpu_id in gpu_ids]
    if uuids and all(uuids):
        return ",".join(uuids)
    return ",".join(map(str, gpu_ids))


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
                "model_variant": _cmd_arg(command, "--model-variant") or "pi05",
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


def _is_external_eval_command(command: list[str]) -> bool:
    if len(command) < 2:
        return False
    joined = " ".join(command)
    return "eval_policy.py" in joined and "script/eval_policy.py" in joined or any(
        Path(item).name == "eval_policy.py" for item in command[1:3]
    )


def discover_external_eval_candidates(
    config: dict[str, Any],
    *,
    ignored_pids: set[int] | None = None,
) -> list[dict[str, Any]]:
    ignored_pids = ignored_pids or set()
    candidates = []
    seen: set[int] = set()
    for gpu in gpu_inventory():
        gpu_id = int(gpu["index"])
        for process in gpu.get("processes", []):
            try:
                pid = int(process["pid"])
            except (TypeError, ValueError):
                continue
            if pid in seen or pid in ignored_pids:
                continue
            command = process_cmdline(pid)
            if not command or not _is_external_eval_command(command):
                continue
            seen.add(pid)
            gpu_ids = sorted(
                int(item["index"])
                for item in gpu_inventory()
                for proc in item.get("processes", [])
                if int(proc.get("pid", -1)) == pid
            )
            stdout_path = process_fd_path(pid, 1) or process_fd_path(pid, 2)
            checkpoint_id = _cmd_arg(command, "--checkpoint_id")
            metadata = {
                "external": True,
                "adopted": True,
                "source": "gpu_eval_policy_process",
                "gpu_ids": gpu_ids or [gpu_id],
                "runtime": "external_4x4090",
                "execution_target": "local_4090",
                "dataset_id": _cmd_arg(command, "--train_config_name") or _cmd_arg(command, "--task_name") or "external_eval",
                "task_name": _cmd_arg(command, "--task_name"),
                "task_config": _cmd_arg(command, "--task_config"),
                "train_config_name": _cmd_arg(command, "--train_config_name"),
                "model_name": _cmd_arg(command, "--model_name"),
                "exp_name": _cmd_arg(command, "--model_name") or _cmd_arg(command, "--ckpt_setting") or "external_eval",
                "checkpoint_id": checkpoint_id,
                "checkpoint_step": int(checkpoint_id) if checkpoint_id and checkpoint_id.isdigit() else checkpoint_id,
                "ckpt_setting": _cmd_arg(command, "--ckpt_setting"),
                "seed": _cmd_arg(command, "--seed"),
                "policy_name": _cmd_arg(command, "--policy_name"),
                "instruction_type": _cmd_arg(command, "--instruction_type"),
                "pi0_step": _cmd_arg(command, "--pi0_step"),
            }
            if stdout_path:
                metadata["external_log_path"] = stdout_path
            candidates.append({"pid": pid, "command": command, "metadata": metadata})
    return candidates


def _nccl_version(path: Path) -> int | None:
    try:
        import ctypes

        lib = ctypes.CDLL(str(path))
        version = ctypes.c_int()
        if lib.ncclGetVersion(ctypes.byref(version)) != 0:
            return None
        return int(version.value)
    except Exception:
        return None


_NCCL_PRELOAD_CACHE: str | None | bool = False


def _compatible_nccl_preload_path(config: dict[str, Any]) -> str | None:
    """Pick an NCCL runtime new enough for JAX's ncclConfig_t usage.

    The 4x4090 OpenPI env currently bundles NCCL 2.14.3.  JAX 0.5 calls
    ``ncclCommInitRankConfig`` with newer config fields; NCCL 2.14 rejects the
    unset blocking sentinel with ``Invalid config blocking attribute value``.
    Prefer a user-space CUDA-12 NCCL from sibling envs/caches via LD_PRELOAD
    instead of reinstalling the training environment.
    """

    global _NCCL_PRELOAD_CACHE
    if _NCCL_PRELOAD_CACHE is not False:
        return _NCCL_PRELOAD_CACHE or None

    configured = config.get("nccl_preload_path")
    candidate_paths: list[Path] = []
    if configured:
        candidate_paths.append(Path(str(configured)).expanduser())
    openpi_env = Path(config["openpi_python"]).resolve().parents[1]
    conda_root = openpi_env.parent
    py_versions = ("python3.12", "python3.11", "python3.10", "python3.9")
    preferred_envs = ("openpi_eval_cu121", "RoboTwin2", "tsq-pilot")
    for env_name in preferred_envs:
        for py_version in py_versions:
            candidate_paths.append(
                conda_root
                / env_name
                / "lib"
                / py_version
                / "site-packages"
                / "nvidia"
                / "nccl"
                / "lib"
                / "libnccl.so.2"
            )
    candidate_paths.extend(
        sorted(Path.home().glob(".cache/uv/archive-v0/*/nvidia/nccl/lib/libnccl.so.2"))
    )
    candidate_paths.append(Path("/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib/libnccl.so.2"))

    candidates: list[tuple[int, int, Path]] = []
    seen: set[Path] = set()
    for priority, path in enumerate(candidate_paths):
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        version = _nccl_version(resolved)
        if version is not None and version >= 22000:
            candidates.append((priority, version, resolved))
    # Prefer earlier, known CUDA-12-era candidates (notably the CU121 eval env)
    # over newer CUDA-13 cache/system libraries.  CUDA-13 NCCL may load on this
    # host but has produced CUDA OOM in tiny JAX pmap smoke tests.
    cuda12_candidates = [item for item in candidates if item[1] < 22900]
    selected = min(cuda12_candidates or candidates, default=None, key=lambda item: item[0])
    _NCCL_PRELOAD_CACHE = str(selected[2]) if selected else None
    return _NCCL_PRELOAD_CACHE or None


def build_environment(
    config: dict[str, Any],
    gpu_ids: list[int] | None,
    *,
    xla_memory_fraction: float | None = None,
    xla_preallocate: bool | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    for sensitive_key in ("BIMANUAL_VLA_SERVER_TOKEN", "BIMANUAL_VLA_LOGIN_PASSWORD"):
        env.pop(sensitive_key, None)
    openpi_env_lib = str(Path(config["openpi_python"]).resolve().parent.parent / "lib")
    cache_root = Path(config.get("cache_root") or (Path.home() / ".cache")).expanduser().resolve()
    inherited_ld = env.get("LD_LIBRARY_PATH", "")
    nccl_preload = _compatible_nccl_preload_path(config)
    ld_parts = []
    if nccl_preload:
        ld_parts.append(str(Path(nccl_preload).parent))
    ld_parts.append(openpi_env_lib)
    if inherited_ld:
        ld_parts.append(inherited_ld)
    env.update(
        {
            "XDG_CACHE_HOME": str(cache_root),
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_LEROBOT_HOME": config["dataset_root"],
            "LD_LIBRARY_PATH": ":".join(ld_parts),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # UUID-based visibility below keeps Dashboard physical ids stable
            # even when a failed GPU disappears from CUDA ordinal enumeration.
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            # 0.95 leaves too little non-XLA memory for NCCL on 24 GiB 4090s.
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(
                xla_memory_fraction
                if xla_memory_fraction is not None
                else config.get("xla_memory_fraction", 0.90)
            ),
        }
    )
    if nccl_preload and gpu_ids is not None:
        inherited_preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = str(nccl_preload) + ((":" + inherited_preload) if inherited_preload else "")
    if xla_preallocate is not None:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if xla_preallocate else "false"
    if gpu_ids is None:
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("JAX_PLATFORMS", None)
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices(gpu_ids)
    # Validation scripts import repository-local helpers.  Do not depend on
    # the service manager's working directory or inherited PYTHONPATH.
    repo_path = str(REPO_DIR)
    env["PYTHONPATH"] = repo_path + os.pathsep + env.get("PYTHONPATH", "")
    return env


def create_app(config_path: Path) -> Flask:
    config = load_config(config_path)
    token = os.environ.get("BIMANUAL_VLA_SERVER_TOKEN", "")
    if len(token) < 20:
        raise RuntimeError("set BIMANUAL_VLA_SERVER_TOKEN to a random value of at least 20 characters")
    login_user = os.environ.get("BIMANUAL_VLA_LOGIN_USER", "")
    login_password = os.environ.get("BIMANUAL_VLA_LOGIN_PASSWORD", "")
    app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
    dashboard_template_path = APP_DIR / "templates" / "index.html"
    dashboard_build_id = hashlib.sha256(dashboard_template_path.read_bytes()).hexdigest()[:12]
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
            cwd=str(REPO_DIR),
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
        if not supplied:
            # HTML <video> elements cannot attach Authorization headers.  Allow
            # token query auth for read-only media URLs generated by the
            # authenticated Dashboard page.
            supplied = request.args.get("token", "")
        if not supplied:
            # Also allow a same-site HttpOnly cookie.  This keeps old/opened
            # media URLs working because browser <video> range requests cannot
            # attach Authorization headers; API responses set this cookie after
            # a normal authenticated request.
            supplied = request.cookies.get("bimanual_vla_token", "")
        if not hmac.compare_digest(supplied, token):
            return jsonify({"error": "unauthorized"}), 401
        request.environ["BIMANUAL_VLA_AUTH_OK"] = "1"
        if supplied != request.cookies.get("bimanual_vla_token", ""):
            request.environ["BIMANUAL_VLA_SET_COOKIE"] = "1"
        return None

    @app.after_request
    def attach_auth_cookie(response):
        if request.environ.get("BIMANUAL_VLA_SET_COOKIE"):
            response.set_cookie(
                "bimanual_vla_token",
                token,
                max_age=30 * 24 * 60 * 60,
                httponly=True,
                samesite="Lax",
            )
        if request.path == "/":
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

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
        return render_template(
            "index.html",
            server_port=config["port"],
            dashboard_profile=config.get("dashboard_profile", "real"),
            dashboard_title=config.get("dashboard_title", "Bimanual-VLA · 4×4090 控制台"),
            upload_default_origin=config.get("upload_default_origin", "real"),
            visible_dataset_origins=config.get("visible_dataset_origins", ["real", "unknown"]),
            enable_policy=bool(config.get("enable_policy", True)),
            dashboard_build=dashboard_build_id,
        )

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "time": now_iso(), "dashboard_build": dashboard_build_id})

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
        response = jsonify({"token": token, "token_type": "Bearer", "username": login_user})
        response.set_cookie(
            "bimanual_vla_token",
            token,
            max_age=30 * 24 * 60 * 60,
            httponly=True,
            samesite="Lax",
        )
        return response

    def list_datasets() -> list[dict[str, Any]]:
        datasets = []
        for directory in sorted(dataset_root.iterdir() if dataset_root.exists() else []):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            info = read_json(directory / "meta" / "info.json")
            if not isinstance(info, dict):
                continue
            origin = dataset_origin_info(directory.name, directory, info)
            visible_origins = set(config.get("visible_dataset_origins", ["real", "unknown"]))
            if origin.get("dataset_origin") not in visible_origins:
                continue
            schema = describe_dataset_schema({**info, **origin})
            split_info = read_json(directory / "meta" / "train_test_split.json")
            norm_ready_by_model: dict[str, bool] = {}
            norm_config_by_model: dict[str, dict[str, Any] | None] = {}
            if schema["arm_mode"] in {"single", "bimanual"}:
                for model_variant in sorted(MODEL_VARIANTS):
                    norm_dir = (
                        Path(config["assets_base_dir"])
                        / policy_config_name(schema["arm_mode"], model_variant)
                        / directory.name
                    )
                    norm_split = read_json(norm_dir / "episode_split.json")
                    norm_config = read_json(norm_dir / NORM_CONFIG_FILENAME)
                    norm_config_by_model[model_variant] = norm_config if isinstance(norm_config, dict) else None
                    default_contract = (
                        action_contract_for_model(schema)
                        if schema.get("training_supported")
                        else None
                    )
                    expected_contract = (
                        default_contract["contract_fingerprint"]
                        if default_contract is not None
                        else {}
                    )
                    norm_ready_by_model[model_variant] = bool(
                        (norm_dir / "norm_stats.json").is_file()
                        and isinstance(split_info, dict)
                        and norm_split == split_info
                        and isinstance(norm_config, dict)
                        and norm_config.get("version") == NORM_CONFIG_VERSION
                        and expected_contract
                        and all(
                            norm_config.get(key) == value
                            for key, value in expected_contract.items()
                        )
                    )
            default_model_variant = infer_model_variant(Path(config["base_checkpoint"])) or "pi05"
            datasets.append(
                {
                    "id": directory.name,
                    "path": str(directory),
                    "episodes": info.get("total_episodes"),
                    "frames": info.get("total_frames"),
                    "fps": info.get("fps"),
                    "robot_type": info.get("robot_type"),
                    **origin,
                    **schema,
                    "episode_split": split_info if isinstance(split_info, dict) else None,
                    "train_episodes": split_info.get("num_train_episodes") if isinstance(split_info, dict) else None,
                    "test_episodes": split_info.get("num_test_episodes") if isinstance(split_info, dict) else None,
                    "norm_stats_ready": norm_ready_by_model.get(default_model_variant, False),
                    "norm_stats_by_model": norm_ready_by_model,
                    "norm_config_by_model": norm_config_by_model,
                    "norm_model_variants": [
                        variant for variant, ready in norm_ready_by_model.items() if ready
                    ],
                    "mtime": directory.stat().st_mtime,
                    "locations": [
                        {
                            "target": "local_4090",
                            "label": "4×4090",
                            "kind": "local",
                            "path": str(directory),
                            "root": str(dataset_root),
                            "origin": origin.get("dataset_origin", "unknown"),
                            "episodes": info.get("total_episodes"),
                            "frames": info.get("total_frames"),
                            "fps": info.get("fps"),
                            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(directory.stat().st_mtime)),
                        }
                    ],
                    "targets": ["local_4090"],
                }
            )
        return datasets

    def visible_dataset_origin_set() -> set[str]:
        return set(config.get("visible_dataset_origins", ["real", "unknown"]))

    def dataset_origin_for_id(dataset_id: str) -> str:
        dataset_path = dataset_root / dataset_id
        info = read_json(dataset_path / "meta" / "info.json")
        if isinstance(info, dict):
            return dataset_origin_info(dataset_id, dataset_path, info).get("dataset_origin", "unknown")
        return "unknown"

    def checkpoint_dataset_ids(step_dir: Path) -> list[str]:
        return sorted(
            path.parent.name
            for path in (step_dir / "assets").glob("*/norm_stats.json")
            if path.is_file()
        )

    def checkpoint_dataset_origins(dataset_ids: list[str]) -> dict[str, str]:
        return {dataset_id: dataset_origin_for_id(dataset_id) for dataset_id in dataset_ids}

    def checkpoint_matches_visible_datasets(step_dir: Path) -> tuple[bool, list[str], dict[str, str]]:
        dataset_ids = checkpoint_dataset_ids(step_dir)
        origins = checkpoint_dataset_origins(dataset_ids)
        visible = visible_dataset_origin_set()
        # A checkpoint without embedded norm assets has unknown provenance.  Keep
        # it only on dashboards that explicitly show unknown-origin datasets;
        # simulation dashboards therefore won't show real/unknown legacy weights.
        if not dataset_ids:
            return "unknown" in visible, dataset_ids, origins
        return any(origin in visible for origin in origins.values()), dataset_ids, origins

    def list_base_models() -> list[dict[str, Any]]:
        candidates: set[Path] = {Path(config["base_checkpoint"]).resolve()}
        for root in checkpoint_roots:
            if not root.exists():
                continue
            for params_dir in root.rglob("params"):
                if params_dir.is_dir() and any(
                    (params_dir / marker).exists()
                    for marker in ("manifest.ocdbt", "_METADATA", "_CHECKPOINT_METADATA")
                ):
                    candidates.add(params_dir.parent.resolve())
        default_path = Path(config["base_checkpoint"]).resolve()
        checkpoint_base_dir = Path(config["checkpoint_base_dir"]).resolve()
        models = []
        for path in sorted(candidates, key=str):
            if not (path / "params").is_dir():
                continue
            model_variant = infer_model_variant(path)
            if model_variant is None:
                continue
            identity = training_checkpoint_identity(path, checkpoint_base_dir)
            foundation = bool(
                path == default_path
                or path.is_relative_to(Path.home() / ".cache/openpi")
            )
            dataset_ids: list[str] = []
            dataset_origins: dict[str, str] = {}
            if not foundation:
                visible_checkpoint, dataset_ids, dataset_origins = checkpoint_matches_visible_datasets(path)
                if not visible_checkpoint:
                    continue
            models.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "model_variant": model_variant,
                    "default": path == default_path,
                    "foundation": foundation,
                    "source": "pretrained" if foundation else "checkpoint",
                    "experiment": identity.get("experiment") if identity else None,
                    "checkpoint_step": identity.get("checkpoint_step") if identity else None,
                    "arm_mode": identity.get("arm_mode") if identity else None,
                    "config_name": identity.get("config_name") if identity else None,
                    "dataset_ids": dataset_ids,
                    "dataset_origins": dataset_origins,
                }
            )
        return sorted(
            models,
            key=lambda item: (
                not item["default"],
                item["model_variant"],
                item["experiment"] or "",
                -(item["checkpoint_step"] or 0),
                item["path"],
            ),
        )

    def resolve_base_model(payload: dict[str, Any]) -> tuple[Path, str]:
        requested_path = payload.get("base_checkpoint") or config["base_checkpoint"]
        path = resolve_under(requested_path, checkpoint_roots)
        if not (path / "params").is_dir():
            raise ValueError(f"base checkpoint has no params directory: {path}")
        inferred = infer_model_variant(path)
        model_variant = str(payload.get("model_variant") or inferred or "pi05")
        if model_variant not in MODEL_VARIANTS:
            raise ValueError(f"model_variant must be one of {sorted(MODEL_VARIANTS)}")
        if inferred is not None and inferred != model_variant:
            raise ValueError(
                f"base checkpoint {path} appears to be {inferred}, but model_variant={model_variant}"
            )
        default_path = Path(config["base_checkpoint"]).resolve()
        foundation = bool(path == default_path or path.is_relative_to(Path.home() / ".cache/openpi"))
        if not foundation:
            visible_checkpoint, dataset_ids, dataset_origins = checkpoint_matches_visible_datasets(path)
            if not visible_checkpoint:
                raise ValueError(
                    "checkpoint provenance is hidden on this Dashboard: "
                    f"datasets={dataset_ids or ['unknown']} origins={dataset_origins or {'unknown': 'unknown'}}"
                )
        return path, model_variant

    def resolve_complete_resume_checkpoint(value: Any) -> Path:
        """Resolve an exact checkpoint step or the latest complete step in an experiment."""
        path = resolve_under(value, checkpoint_roots)
        if is_full_state_checkpoint(path):
            return path.resolve()
        steps = full_state_checkpoint_steps(path)
        if steps:
            return steps[-1][1]
        raise ValueError(
            f"resume checkpoint is not a finalized Orbax checkpoint and contains no complete steps: {path}"
        )

    def validate_resume_checkpoint_variant(path: Path, model_variant: str) -> None:
        inferred = infer_model_variant(path)
        if inferred is not None and inferred != model_variant:
            raise ValueError(
                f"resume checkpoint {path} appears to be {inferred}, but model_variant={model_variant}"
            )
        visible_checkpoint, dataset_ids, dataset_origins = checkpoint_matches_visible_datasets(path)
        if not visible_checkpoint:
            raise ValueError(
                "resume checkpoint provenance is hidden on this Dashboard: "
                f"datasets={dataset_ids or ['unknown']} origins={dataset_origins or {'unknown': 'unknown'}}"
            )

    def validate_resume_checkpoint_contract(path: Path, model_contract: dict[str, Any]) -> None:
        marker = checkpoint_action_contract(path)
        if marker is None:
            raise ValueError(
                "resume checkpoint has no action-contract marker; refusing full-state "
                f"resume because its dataset/action semantics are unverified: {path}"
            )
        expected = model_contract["contract_fingerprint"].items()
        mismatches = {
            key: {"checkpoint": marker.get(key), "training": value}
            for key, value in expected
            if marker.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "resume checkpoint action contract does not match the requested real dataset: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )

    def list_checkpoints() -> list[dict[str, Any]]:
        checkpoints = []
        for model_variant in ("pi05", "pi0"):
            for arm_mode in ("single", "bimanual"):
                config_name = policy_config_name(arm_mode, model_variant)
                config_root = Path(config["checkpoint_base_dir"]) / config_name
                if not config_root.exists():
                    continue
                for exp_dir in config_root.iterdir():
                    if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                        continue
                    for step_dir in exp_dir.iterdir():
                        if not step_dir.is_dir() or not step_dir.name.isdigit():
                            continue
                        if not (step_dir / "params").is_dir() or not (step_dir / "_CHECKPOINT_METADATA").is_file():
                            continue
                        visible_checkpoint, dataset_ids, dataset_origins = checkpoint_matches_visible_datasets(step_dir)
                        if not visible_checkpoint:
                            continue
                        mtime = step_dir.stat().st_mtime
                        cache_key = str(step_dir.resolve())
                        cached = checkpoint_size_cache.get(cache_key)
                        if cached is None or cached[0] != mtime:
                            size_bytes = sum(path.stat().st_size for path in step_dir.rglob("*") if path.is_file())
                            checkpoint_size_cache[cache_key] = (mtime, size_bytes)
                        else:
                            size_bytes = cached[1]
                        action_contract = checkpoint_action_contract(step_dir)
                        checkpoints.append(
                            {
                                "path": cache_key,
                                "action_contract": action_contract,
                                "contract_version": action_contract.get("contract_version") if action_contract else None,
                                "raw_action_dim": action_contract.get("raw_action_dim") if action_contract else None,
                                "model_action_dim": action_contract.get("model_action_dim") if action_contract else None,
                                "model_action_convention": (
                                    action_contract.get("model_action_convention")
                                    if action_contract
                                    else None
                                ),
                                "gripper_semantics": action_contract.get("gripper_semantics") if action_contract else None,
                                "config_name": config_name,
                                "model_variant": model_variant,
                                "arm_mode": arm_mode,
                                "experiment": exp_dir.name,
                                "step": int(step_dir.name),
                                "full_state_available": is_full_state_checkpoint(step_dir),
                                "restore_capabilities": (["full_state"] if is_full_state_checkpoint(step_dir) else ["weights_only"]),
                                "dataset_ids": dataset_ids,
                                "dataset_origins": dataset_origins,
                                "mtime": mtime,
                                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                                "size_gib": round(size_bytes / (1024**3), 3),
                            }
                        )
        return sorted(checkpoints, key=lambda item: (item["mtime"], item["step"]), reverse=True)

    def checkpoint_active_references(path: Path) -> list[dict[str, Any]]:
        path = Path(path).expanduser().resolve()
        references: list[dict[str, Any]] = []
        for task in tasks.list():
            if task.get("state") in TERMINAL_STATES:
                continue
            metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
            task_references: list[dict[str, Any]] = []
            for key in ("checkpoint", "base_checkpoint", "resume_checkpoint"):
                value = metadata.get(key)
                if not value:
                    continue
                try:
                    ref_path = Path(str(value)).expanduser().resolve()
                except Exception:
                    continue
                if ref_path == path:
                    task_references.append({"kind": key, "path": str(ref_path)})
            checkpoint_dir = metadata.get("checkpoint_dir")
            if checkpoint_dir:
                try:
                    checkpoint_dir_path = Path(str(checkpoint_dir)).expanduser().resolve()
                except Exception:
                    checkpoint_dir_path = None
                if checkpoint_dir_path is not None and (path == checkpoint_dir_path or path.is_relative_to(checkpoint_dir_path)):
                    task_references.append({"kind": "checkpoint_dir", "path": str(checkpoint_dir_path)})
            if task_references:
                references.append(
                    {
                        "task_id": str(task.get("id", "")),
                        "type": task.get("type"),
                        "state": task.get("state"),
                        "references": task_references,
                    }
                )
        return references

    def prune_empty_checkpoint_parents(path: Path) -> None:
        checkpoint_root = Path(config["checkpoint_base_dir"]).expanduser().resolve()
        current = Path(path).expanduser().resolve().parent
        while current != checkpoint_root and checkpoint_root in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


    @app.get("/api/status")
    def status():
        tasks.discover_external_policies()
        tasks.discover_external_evals()
        task_list = tasks.list()
        for task in task_list:
            if task.get("type") == "transfer":
                progress = transfer_progress_for_task(task)
                if progress is not None:
                    task["progress"] = progress
            elif task.get("type") == "train":
                probe = tasks.training_metrics_probe(task)
                if probe is not None:
                    task["training_metrics"] = probe
        for task in task_list:
            if task["type"] != "policy":
                continue
            metadata = task.setdefault("metadata", {})
            if not metadata.get("model_variant"):
                metadata["model_variant"] = (
                    _cmd_arg(task.get("command", []), "--model-variant")
                    or infer_model_variant(Path(str(metadata.get("checkpoint", ""))))
                    or "pi05"
                )
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
        checkpoints = list_checkpoints()
        visible_experiments = {item.get("experiment") for item in checkpoints if item.get("experiment")}
        experiments = [
            item for item in training_experiment_catalog(Path(config["checkpoint_base_dir"]))
            if item.get("name") in visible_experiments
        ]
        return jsonify(
            {
                "datasets": list_datasets(),
                "checkpoints": checkpoints,
                "experiments": experiments,
                "base_models": list_base_models(),
                "robot_observation": latest_observation,
                "tasks": task_list,
                "gpus": gpu_inventory(),
                "config": {
                    "dashboard_profile": config.get("dashboard_profile", "real"),
                    "dashboard_title": config.get("dashboard_title", "Bimanual-VLA Dashboard"),
                    "dashboard_build": dashboard_build_id,
                    "upload_default_origin": config.get("upload_default_origin", "real"),
                    "visible_dataset_origins": config.get("visible_dataset_origins", ["real", "unknown"]),
                    "enable_policy": bool(config.get("enable_policy", True)),
                    "local_storage_locations": config.get("local_storage_locations", {}),
                    "cluster_targets": {
                        name: {
                            key: value
                            for key, value in target.items()
                            if key not in {"password", "token", "secret"}
                        }
                        for name, target in config.get("cluster_targets", {}).items()
                    },
                    "eval_video_roots": config.get("eval_video_roots", []),
                    "dataset_root": config["dataset_root"],
                    "workspace_root": config["workspace_root"],
                    "cache_root": config.get("cache_root"),
                    "upload_roots": {
                        origin: str(Path(config["workspace_root"]) / "uploads" / origin)
                        for origin in ("real", "simulation")
                    },
                    "dataset_origins": sorted(DATASET_ORIGINS),
                    "checkpoint_base_dir": config["checkpoint_base_dir"],
                    "base_checkpoint": config["base_checkpoint"],
                    "allowed_gpu_ids": sorted(allowed_gpus),
                    "allow_busy_gpus": config["allow_busy_gpus"],
                    "xla_memory_fraction": config.get("xla_memory_fraction", 0.90),
                    "training_min_free_gpu_mib": config.get("training_min_free_gpu_mib", 23_000),
                    "policy_allow_busy_gpus": config.get("policy_allow_busy_gpus", True),
                    "policy_min_free_gpu_mib": config.get("policy_min_free_gpu_mib", 12_000),
                    "policy_xla_memory_fraction": config.get("policy_xla_memory_fraction", 0.60),
                    "policy_xla_preallocate": config.get("policy_xla_preallocate", False),
                    "transfer_parallelism": config.get("transfer_parallelism", 4),
                    "nas_dataset_staging_root": config.get("nas_dataset_staging_root"),
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

    @app.patch("/api/datasets/<dataset_id>")
    def rename_dataset(dataset_id: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        payload = request.get_json(force=True)
        new_dataset_id = safe_name(
            payload.get("new_dataset_id") if isinstance(payload, dict) else None,
            "new dataset id",
        )
        return jsonify(dataset_editor.rename_dataset(dataset_id, new_dataset_id))

    @app.patch("/api/datasets/<dataset_id>/origin")
    def set_dataset_origin(dataset_id: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        payload = request.get_json(force=True)
        origin = normalize_dataset_origin(
            payload.get("dataset_origin") if isinstance(payload, dict) else None
        )
        return jsonify(
            dataset_editor.set_dataset_origin(dataset_id, origin, source="dashboard")
        )

    @app.delete("/api/datasets/<dataset_id>")
    def delete_dataset(dataset_id: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        payload = request.get_json(silent=True)
        confirmation = payload.get("confirm_dataset_id") if isinstance(payload, dict) else None
        if confirmation != dataset_id:
            raise ValueError("confirm_dataset_id must exactly match the dataset id")
        return jsonify(dataset_editor.delete_dataset(dataset_id))

    @app.get("/api/datasets/<dataset_id>/episodes/<int:episode_index>/video/<video_key>")
    def dataset_episode_video(dataset_id: str, episode_index: int, video_key: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        raw = str(request.args.get("raw", "")).lower() in {"1", "true", "yes", "on"}
        path = (
            dataset_editor.video_path(dataset_id, episode_index, video_key)
            if raw
            else dataset_editor.browser_video_path(dataset_id, episode_index, video_key)
        )
        return send_file(path, mimetype="video/mp4", conditional=True, max_age=0)

    @app.get("/api/datasets/<dataset_id>/episodes/<int:episode_index>/image/<image_key>/<int:frame_index>")
    def dataset_episode_image(dataset_id: str, episode_index: int, image_key: str, frame_index: int):
        dataset_id = safe_name(dataset_id, "dataset id")
        source, mimetype = dataset_editor.image_source(dataset_id, episode_index, image_key, frame_index)
        return send_file(source, mimetype=mimetype, conditional=True, max_age=3600)

    @app.patch("/api/datasets/<dataset_id>/episodes/<int:episode_index>")
    def update_dataset_episode(dataset_id: str, episode_index: int):
        dataset_id = safe_name(dataset_id, "dataset id")
        result = dataset_editor.update_episode(dataset_id, episode_index, request.get_json(force=True))
        return jsonify(result)

    @app.get("/api/datasets/<dataset_id>/event-semantics")
    def get_dataset_event_semantics(dataset_id: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        return jsonify(dataset_editor.event_semantics(dataset_id))

    @app.post("/api/datasets/<dataset_id>/event-semantics")
    def save_dataset_event_semantics(dataset_id: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        return jsonify(dataset_editor.save_event_semantics(dataset_id, request.get_json(force=True)))

    @app.get("/api/datasets/<dataset_id>/episodes/<int:episode_index>/event-overrides")
    def get_dataset_episode_event_overrides(dataset_id: str, episode_index: int):
        dataset_id = safe_name(dataset_id, "dataset id")
        return jsonify(dataset_editor.event_overrides(dataset_id, episode_index))

    @app.post("/api/datasets/<dataset_id>/episodes/<int:episode_index>/event-overrides")
    def save_dataset_episode_event_overrides(dataset_id: str, episode_index: int):
        dataset_id = safe_name(dataset_id, "dataset id")
        return jsonify(dataset_editor.save_event_overrides(dataset_id, episode_index, request.get_json(force=True)))

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
        target_path = dataset_root / dataset_id
        source_path = dataset_root / source_id
        target_info = read_json(target_path / "meta" / "info.json", {})
        source_info = read_json(source_path / "meta" / "info.json", {})
        target_origin = dataset_origin_info(dataset_id, target_path, target_info).get("dataset_origin")
        source_origin = dataset_origin_info(source_id, source_path, source_info).get("dataset_origin")
        if (
            target_origin != "unknown"
            and source_origin != "unknown"
            and target_origin != source_origin
        ):
            raise ValueError(
                f"cannot merge {source_origin} dataset {source_id} into {target_origin} dataset {dataset_id}"
            )
        result = dataset_editor.merge_existing(dataset_id, source_id)
        if target_origin == "unknown" and source_origin != "unknown":
            dataset_editor.set_dataset_origin(
                dataset_id, source_origin, source="merge_inherited"
            )
            result["dataset_origin"] = source_origin
        return jsonify(result)

    def parse_dataset(payload: dict[str, Any]) -> tuple[str, str, str, str, dict[str, Any]]:
        dataset_id = safe_name(payload.get("dataset_id"), "dataset id")
        dataset_path = dataset_root / dataset_id
        info = read_json(dataset_path / "meta" / "info.json")
        if not isinstance(info, dict):
            raise ValueError(f"dataset is not installed: {dataset_id}")
        origin = dataset_origin_info(dataset_id, dataset_path, info)
        contract = describe_dataset_schema({**info, **origin})
        if not contract["training_supported"]:
            raise ValueError(
                "unsupported dataset contract: "
                f"layout={contract['dataset_layout']} schema={contract['schema']} "
                f"arm_mode={contract['arm_mode']} dims={contract['state_shape']}/{contract['action_shape']} "
                f"cameras={contract['cameras']} error={contract.get('contract_error') or '-'}"
            )
        arm_mode = str(contract["arm_mode"])
        schema = str(contract["schema"])
        if arm_mode == "bimanual":
            arm_side = "both"
        else:
            arm_side = str(contract.get("arm_side") or payload.get("arm_side", "right"))
            requested_side = str(payload.get("arm_side", arm_side))
            if requested_side in {"left", "right"} and requested_side != arm_side:
                raise ValueError(
                    f"requested arm_side={requested_side} conflicts with dataset arm_side={arm_side}"
                )
            if arm_side not in {"left", "right"}:
                raise ValueError("single-arm dataset arm_side must be left or right")
        return dataset_id, arm_mode, arm_side, schema, contract


    def parse_gpus(
        payload: dict[str, Any],
        *,
        one_only: bool = False,
        ignored_pids: set[int] | None = None,
        check_busy: bool = True,
        minimum_free_mib: int = 0,
        allow_busy: bool | None = None,
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
        unavailable = {
            gpu_id: inventory.get(gpu_id, {}).get("health_issue") or "GPU compute unavailable"
            for gpu_id in gpu_ids
            if inventory.get(gpu_id, {}).get("compute_available") is False
        }
        if unavailable:
            raise ValueError(f"GPU(s) are unavailable for CUDA compute: {unavailable}")
        busy = {
            gpu_id: blocking_gpu_processes(
                [
                    process
                    for process in inventory.get(gpu_id, {}).get("processes", [])
                    if int(process.get("pid", -1)) not in ignored_pids
                ],
                small_process_memory_mib=int(
                    config.get("small_gpu_process_memory_mib", 512)
                ),
                small_process_total_mib=int(
                    config.get("small_gpu_process_total_mib", 1024)
                ),
            )
            for gpu_id in gpu_ids
        }
        busy = {gpu_id: procs for gpu_id, procs in busy.items() if procs}
        effective_allow_busy = (
            bool(config.get("allow_busy_gpus", False))
            if allow_busy is None
            else bool(allow_busy)
        )
        if busy and not effective_allow_busy:
            raise ValueError(f"refusing busy GPU(s): {busy}")
        if minimum_free_mib > 0:
            low_memory = gpu_memory_shortfalls(
                inventory,
                gpu_ids,
                minimum_free_mib,
                ignored_pids=ignored_pids,
            )
            if low_memory:
                raise ValueError(f"GPU(s) do not have enough free memory: {low_memory}")
        return gpu_ids

    def json_arg(value: Any) -> str:
        return base64.urlsafe_b64encode(
            json.dumps(value, ensure_ascii=False).encode()
        ).decode()

    def runtime_config_for_target(target_name: str | None) -> dict[str, Any] | None:
        name = str(target_name or "local_4090")
        if name in {"", "local", "local_4090", "4x4090"}:
            return None
        targets = config.get("cluster_targets", {})
        if name not in targets:
            raise ValueError(f"unknown cluster execution target: {name}")
        target = {**targets[name]}
        required = [
            "submit_host",
            "partition",
            "workdir",
            "openpi_python",
            "dataset_root",
            "assets_base_dir",
            "checkpoint_base_dir",
            "base_checkpoint",
        ]
        missing = [key for key in required if not target.get(key)]
        if missing:
            raise ValueError(f"cluster target {name} missing required fields: {missing}")
        target.setdefault("name", name)
        target.setdefault("openpi_repo", target.get("workdir"))
        target.setdefault("dashboard_repo", target.get("workdir"))
        target.setdefault("remote_job_dir", str(PurePosixPath(str(target["workdir"])) / "logs" / "dashboard_slurm"))
        return target

    def openpi_helper_for(runtime_config: dict[str, Any]) -> str:
        repo = runtime_config.get("dashboard_repo") or str(REPO_DIR)
        return str(PurePosixPath(str(repo)) / "server_4090" / "openpi_single_arm.py")

    def eval_helper_for(runtime_config: dict[str, Any]) -> str:
        repo = runtime_config.get("dashboard_repo") or str(REPO_DIR)
        return str(PurePosixPath(str(repo)) / "server_4090" / "eval_heldout_loss.py")

    def translate_runtime_path(path: str | Path, runtime_config: dict[str, Any]) -> str:
        value = str(path)
        replacements = [
            (config.get("checkpoint_base_dir"), runtime_config.get("checkpoint_base_dir")),
            (config.get("assets_base_dir"), runtime_config.get("assets_base_dir")),
            (config.get("dataset_root"), runtime_config.get("dataset_root")),
        ]
        for local_root, remote_root in replacements:
            if not local_root or not remote_root:
                continue
            local = str(local_root).rstrip("/")
            if value == local or value.startswith(local + "/"):
                return str(remote_root).rstrip("/") + value[len(local):]
        if value == str(config.get("base_checkpoint")) and runtime_config.get("base_checkpoint"):
            return str(runtime_config["base_checkpoint"])
        return value

    def build_train_command(
        runtime_config: dict[str, Any],
        dataset_id: str,
        arm_mode: str,
        arm_side: str,
        schema: str,
        *,
        base_checkpoint: str | Path,
        model_variant: str,
        exp_name: str,
        batch_size: int,
        num_workers: int,
        steps: int,
        save_interval: int,
        keep_period: int | None,
        fsdp_devices: int,
        split: EpisodeSplit,
        model_contract: dict[str, Any],
        effective_mode: str,
        resume_checkpoint: str | Path | None = None,
        wandb_enabled: bool = False,
    ) -> list[str]:
        command = [
            str(runtime_config["openpi_python"]), openpi_helper_for(runtime_config), "train",
            "--dataset-id", dataset_id,
            "--arm-mode", arm_mode,
            "--arm-side", arm_side,
            "--schema", schema,
            "--model-variant", model_variant,
            "--exp-name", exp_name,
            "--assets-base-dir", str(runtime_config["assets_base_dir"]),
            "--checkpoint-base-dir", str(runtime_config["checkpoint_base_dir"]),
            "--base-checkpoint", str(base_checkpoint),
            "--batch-size", str(batch_size),
            "--num-workers", str(num_workers),
            "--num-train-steps", str(steps),
            "--save-interval", str(save_interval),
            "--fsdp-devices", str(fsdp_devices),
            "--test-ratio", str(split.test_ratio),
            "--split-seed", str(split.seed),
        ] + action_contract_command_args(model_contract)
        if keep_period is not None:
            command += ["--keep-period", str(keep_period)]
        if resume_checkpoint is not None:
            command += ["--resume-checkpoint", str(resume_checkpoint)]
        elif effective_mode != "new":
            command.append(f"--{effective_mode}")
        if wandb_enabled:
            command.append("--wandb-enabled")
        return command

    def build_eval_command(
        runtime_config: dict[str, Any],
        *,
        checkpoint: str | Path,
        result_path: str | Path,
        dataset_id: str,
        arm_mode: str,
        arm_side: str,
        schema: str,
        model_variant: str,
        base_checkpoint: str | Path,
        batch_size: int,
        num_workers: int,
        max_batches: int,
        eval_seed: int,
        model_contract: dict[str, Any],
    ) -> list[str]:
        contract = {
            **model_contract,
            **model_contract.get("contract_fingerprint", {}),
            "schema": schema,
            "raw_gripper_semantics": model_contract["raw_gripper_semantics"],
            "model_gripper_semantics": model_contract["model_gripper_semantics"],
        }
        return [
            str(runtime_config["openpi_python"]), eval_helper_for(runtime_config),
            "--checkpoint", str(checkpoint),
            "--result-json", str(result_path),
            "--dataset-id", dataset_id,
            "--arm-mode", arm_mode,
            "--arm-side", arm_side,
            "--schema", schema,
            "--model-variant", model_variant,
            "--assets-base-dir", str(runtime_config["assets_base_dir"]),
            "--checkpoint-base-dir", str(runtime_config["checkpoint_base_dir"]),
            "--base-checkpoint", str(base_checkpoint),
            "--batch-size", str(batch_size),
            "--num-workers", str(num_workers),
            "--max-batches", str(max_batches),
            "--eval-seed", str(eval_seed),
        ] + action_contract_command_args(contract)

    def slurm_runner_command(
        *,
        target_name: str,
        target_config: dict[str, Any],
        commands: list[list[str]],
        labels: list[str],
        job_name: str,
    ) -> list[str]:
        return [
            config["openpi_python"],
            str(APP_DIR / "slurm_job_runner.py"),
            "--target-json", json_arg({**target_config, "name": target_name}),
            "--commands-json", json_arg(commands),
            "--command-labels-json", json_arg(labels),
            "--job-name", safe_name(job_name, "slurm job name"),
        ]

    def norm_stats_path(dataset_id: str, arm_mode: str, model_variant: str) -> Path:
        return (
            Path(config["assets_base_dir"])
            / policy_config_name(arm_mode, model_variant)
            / dataset_id
            / "norm_stats.json"
        )

    def parse_episode_split(
        payload: dict[str, Any], dataset_id: str, dataset_contract: dict[str, Any]
    ) -> EpisodeSplit:
        test_ratio = safe_float(
            payload.get("test_ratio", DEFAULT_TEST_RATIO),
            "test_ratio",
            0.0,
            1.0,
            maximum_inclusive=False,
        )
        split_seed = safe_int(
            payload.get("split_seed", DEFAULT_SPLIT_SEED),
            "split_seed",
            0,
            2**31 - 1,
        )
        contract = action_contract_for_model(dataset_contract)
        return resolve_episode_split(
            dataset_root,
            dataset_id,
            test_ratio=test_ratio,
            seed=split_seed,
            contract=contract["contract_fingerprint"],
        )

    def training_episode_split(
        payload: dict[str, Any],
        dataset_id: str,
        dataset_contract: dict[str, Any],
        *,
        model_contract: dict[str, Any] | None = None,
    ) -> tuple[EpisodeSplit, str]:
        contract = model_contract or action_contract_for_model(dataset_contract)
        fingerprint = contract["contract_fingerprint"]
        persisted = load_episode_split(
            dataset_root, dataset_id, contract=fingerprint
        )
        explicit_ratio = payload.get("test_ratio") not in (None, "")
        explicit_seed = payload.get("split_seed") not in (None, "")
        if not explicit_ratio and not explicit_seed and persisted is not None:
            return persisted, "persisted"

        test_ratio = safe_float(
            payload.get("test_ratio")
            if explicit_ratio
            else persisted.test_ratio if persisted is not None else DEFAULT_TEST_RATIO,
            "test_ratio",
            0.0,
            1.0,
            maximum_inclusive=False,
        )
        split_seed = safe_int(
            payload.get("split_seed")
            if explicit_seed
            else persisted.seed if persisted is not None else DEFAULT_SPLIT_SEED,
            "split_seed",
            0,
            2**31 - 1,
        )
        split = resolve_episode_split(
            dataset_root,
            dataset_id,
            test_ratio=test_ratio,
            seed=split_seed,
            contract=fingerprint,
        )
        return split, "request" if explicit_ratio or explicit_seed else "default"


    def build_norm_command(
        dataset_id: str,
        arm_mode: str,
        arm_side: str,
        schema: str,
        *,
        base_checkpoint: Path | str,
        model_variant: str,
        batch_size: int,
        num_workers: int,
        split: EpisodeSplit,
        model_contract: dict[str, Any],
        max_frames: int | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> list[str]:
        runtime_config = runtime_config or config
        command = [
            str(runtime_config["openpi_python"]), openpi_helper_for(runtime_config), "norm",
            "--dataset-id", dataset_id,
            "--arm-mode", arm_mode,
            "--arm-side", arm_side,
            "--schema", schema,
            "--model-variant", model_variant,
            "--assets-base-dir", str(runtime_config["assets_base_dir"]),
            "--checkpoint-base-dir", str(runtime_config["checkpoint_base_dir"]),
            "--base-checkpoint", str(base_checkpoint),
            "--batch-size", str(batch_size),
            "--num-workers", str(num_workers),
            "--test-ratio", str(split.test_ratio),
            "--split-seed", str(split.seed),
        ]
        command += action_contract_command_args(model_contract)
        if max_frames is not None:
            command += ["--max-frames", str(max_frames)]
        return command

    VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}

    def eval_video_roots() -> list[Path]:
        return [Path(item) for item in config.get("eval_video_roots", []) if item]

    def encode_video_id(path: Path) -> str:
        return base64.urlsafe_b64encode(str(path.resolve()).encode()).decode().rstrip("=")

    def decode_video_id(video_id: str) -> Path:
        padded = video_id + "=" * (-len(video_id) % 4)
        try:
            path = Path(base64.urlsafe_b64decode(padded.encode()).decode()).resolve()
        except Exception as exc:
            raise FileNotFoundError("invalid video id") from exc
        roots = [root.resolve() for root in eval_video_roots()]
        if not any(path == root or root in path.parents for root in roots):
            raise FileNotFoundError("video is outside configured roots")
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            raise FileNotFoundError("video not found")
        return path

    def prune_empty_eval_video_parents(path: Path) -> None:
        roots = [root.resolve() for root in eval_video_roots()]
        if not roots:
            return
        current = Path(path).expanduser().resolve().parent
        while current not in roots:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    EVENT_OVERRIDE_SCHEMA = "event_overrides.v1"
    EVENT_LABELS_BY_TASK = {
        "handover_mic": {
            0: "E0 未稳定抓住 mic",
            1: "E1 donor 稳定抓住 mic",
            2: "E2 mic 稳定抬离支撑面",
            3: "E3 receiver 稳定抓住 mic",
            4: "E4 donor 释放且 receiver 独立持有",
        }
    }

    def event_override_path_for_video(path: Path) -> Path:
        return path.parent / "_event_overrides" / f"{path.stem}.json"

    def event_labels_for_task(task_name: Any) -> dict[str, str]:
        text = str(task_name or "").lower()
        for key, labels in EVENT_LABELS_BY_TASK.items():
            if key in text:
                return {str(index): label for index, label in labels.items()}
        return {}

    def safe_event_int(value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    def normalize_event_edits(raw_edits: Any, *, max_step: int | None = None) -> list[dict[str, Any]]:
        if not isinstance(raw_edits, list):
            return []
        by_start: dict[int, dict[str, Any]] = {}
        for raw in raw_edits:
            if not isinstance(raw, dict):
                continue
            start = safe_event_int(raw.get("start_step", raw.get("step", raw.get("start_frame", raw.get("frame")))))
            if start is None:
                continue
            start = max(0, start)
            if max_step is not None:
                start = min(start, max_step)
            marker: dict[str, Any] = {"start_step": int(start)}
            current = safe_event_int(raw.get("current_event"))
            max_event = safe_event_int(raw.get("max_event_reached", raw.get("max_event")))
            if current is not None:
                marker["current_event"] = current
            if max_event is not None:
                marker["max_event_reached"] = max_event
            note = raw.get("note")
            if isinstance(note, str) and note.strip():
                marker["note"] = note.strip()[:500]
            if "current_event" in marker or "max_event_reached" in marker:
                by_start[int(start)] = marker
        normalized = [by_start[start] for start in sorted(by_start)]
        running_max: int | None = None
        for marker in normalized:
            current = safe_event_int(marker.get("current_event"))
            declared_max = safe_event_int(marker.get("max_event_reached"))
            candidates = [value for value in (running_max, current, declared_max) if value is not None]
            if candidates:
                running_max = max(candidates)
                marker["max_event_reached"] = running_max
        return normalized


    def normalize_event_timeline(raw_timeline: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_timeline, list):
            return []
        timeline: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_timeline):
            if not isinstance(raw, dict):
                continue
            step = safe_event_int(raw.get("step", raw.get("frame", index)))
            if step is None:
                step = index
            item: dict[str, Any] = {"step": max(0, int(step))}
            frame = safe_event_int(raw.get("frame"))
            if frame is not None:
                item["frame"] = max(0, int(frame))
            current = safe_event_int(raw.get("current_event"))
            max_event = safe_event_int(raw.get("max_event_reached", raw.get("max_event")))
            if current is not None:
                item["current_event"] = current
            if max_event is not None:
                item["max_event_reached"] = max_event
            if "current_event" in item or "max_event_reached" in item:
                timeline.append(item)
        timeline.sort(key=lambda item: int(item.get("step", 0)))
        return timeline

    def read_video_event_overrides(path: Path, *, max_step: int | None = None) -> dict[str, Any] | None:
        raw = read_json(event_override_path_for_video(path))
        if not isinstance(raw, dict):
            return None
        return {**raw, "schema": raw.get("schema") or EVENT_OVERRIDE_SCHEMA, "edits": normalize_event_edits(raw.get("edits", []), max_step=max_step)}

    def load_episode_video_sidecar(path: Path, episode_index: int) -> tuple[dict[str, Any] | None, str | None]:
        candidates = sorted(
            [path.parent / "_episode_results.jsonl", *path.parent.glob("_episode_results.rescore*.jsonl")],
            key=lambda item: item.stat().st_mtime if item.exists() else 0,
            reverse=True,
        )
        seen: set[Path] = set()
        fallback: tuple[dict[str, Any], str] | None = None
        for jsonl_path in candidates:
            if jsonl_path in seen or not jsonl_path.is_file():
                continue
            seen.add(jsonl_path)
            rows = []
            try:
                for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    try:
                        item_episode_index = int(item.get("episode_index", -1))
                    except Exception:
                        item_episode_index = -1
                    if item_episode_index == episode_index and item.get("counted_in_success_rate", True):
                        rows.append(item)
            except Exception:
                rows = []
            if rows:
                item = rows[-1]
                if fallback is None:
                    fallback = (item, jsonl_path.name)
                if item.get("event_timeline") or "current_event" in item or "max_event_reached" in item:
                    return item, jsonl_path.name
        if fallback is not None:
            return fallback
        return None, None

    def event_track_from_video_sidecar(path: Path, row: dict[str, Any] | None, sidecar_name: str | None, *, task_name: Any = None) -> dict[str, Any] | None:
        row = row or {}
        timeline = normalize_event_timeline(row.get("event_timeline"))
        current = safe_event_int(row.get("current_event"))
        max_event = safe_event_int(row.get("max_event_reached", row.get("max_event")))
        if not timeline and (current is not None or max_event is not None):
            timeline = [{"step": 0}]
            if current is not None:
                timeline[0]["current_event"] = current
            if max_event is not None:
                timeline[0]["max_event_reached"] = max_event
        timeline_max = max((int(item.get("step", 0)) for item in timeline), default=0)
        override_raw = read_video_event_overrides(path, max_step=None)
        override_max = max((int(item.get("start_step", 0)) for item in (override_raw or {}).get("edits", [])), default=0)
        max_step = timeline_max if timeline_max > 0 else (override_max if override_max > 0 else None)
        overrides = read_video_event_overrides(path, max_step=max_step)
        has_overrides = bool(overrides and overrides.get("edits"))
        if not timeline and not has_overrides:
            return None
        labels = event_labels_for_task(task_name or row.get("task") or row.get("task_name"))
        override_edits = (overrides or {}).get("edits") or []
        override_current = safe_event_int(override_edits[-1].get("current_event")) if override_edits else None
        override_max_values = [
            value
            for edit in override_edits
            for value in (
                safe_event_int(edit.get("current_event")),
                safe_event_int(edit.get("max_event_reached")),
            )
            if value is not None
        ]
        override_reached = max(override_max_values) if override_max_values else None
        return {
            "available": bool(timeline or has_overrides),
            "source": sidecar_name or ("manual_override" if has_overrides else None),
            "event_version": row.get("event_version") if isinstance(row, dict) else None,
            "timeline": timeline,
            "current_event": override_current if override_current is not None else (current if current is not None else (timeline[-1].get("current_event") if timeline else None)),
            "max_event_reached": max(
                [
                    value
                    for value in (
                        override_reached,
                        max_event,
                        timeline[-1].get("max_event_reached") if timeline else None,
                    )
                    if value is not None
                ],
                default=None,
            ),
            "max_step": max_step,
            "labels": labels,
            "overrides": overrides,
            "override_count": len(overrides.get("edits") or []) if overrides else 0,
            "marker_semantics": "start_frame_until_next_marker" if has_overrides else None,
            "editable": True,
        }

    def save_video_event_override(video_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        path = decode_video_id(video_id)
        episode_match = re.search(r"episode(\d+)\.(?:mp4|webm|mov|mkv|avi|gif)$", path.name, re.IGNORECASE)
        row: dict[str, Any] | None = None
        max_step: int | None = None
        track: dict[str, Any] | None = None
        if episode_match is not None:
            row, sidecar_name = load_episode_video_sidecar(path, int(episode_match.group(1)))
            if row:
                track = event_track_from_video_sidecar(path, row, sidecar_name, task_name=row.get("task") or row.get("task_name"))
                max_step = safe_event_int((track or {}).get("max_step"))
        edit_values = payload.get("edits")
        if edit_values is None:
            step = payload.get("step", payload.get("frame"))
            edit = {"start_step": step}
            if "current_event" in payload:
                edit["current_event"] = payload.get("current_event")
            if "max_event_reached" in payload or "max_event" in payload:
                edit["max_event_reached"] = payload.get("max_event_reached", payload.get("max_event"))
            if "note" in payload:
                edit["note"] = payload.get("note")
            edit_values = [edit]
        replace = bool(payload.get("replace", False))
        existing = [] if replace else ((read_video_event_overrides(path, max_step=max_step) or {}).get("edits") or [])
        edits = normalize_event_edits([*existing, *(edit_values or [])], max_step=max_step)
        if max_step is None:
            inferred_max = max((int(edit.get("start_step", 0)) for edit in edits), default=0)
            max_step = inferred_max if inferred_max > 0 else None
        # Keep handover_mic strict (0..4) and otherwise accept a conservative
        # generic range; this lets non-event datasets keep using the dashboard.
        task_text = " ".join(str((row or {}).get(key, "")) for key in ("task", "task_name", "event_version"))
        max_allowed = 4 if "handover_mic" in task_text.lower() or "mic" in str(path).lower() else 99
        for edit in edits:
            for key in ("current_event", "max_event_reached"):
                if key in edit and not (0 <= int(edit[key]) <= max_allowed):
                    raise ValueError(f"{key} must be in [0, {max_allowed}]")
        output = {
            "schema": EVENT_OVERRIDE_SCHEMA,
            "video": path.name,
            "video_id": video_id,
            "relative_dir": path.parent.name,
            "event_max_value": max_allowed,
            "marker_semantics": "start_frame_until_next_marker",
            "max_step": max_step,
            "updated_at": now_iso(),
            "source": "dashboard_eval_video",
            "edits": edits,
        }
        out_path = event_override_path_for_video(path)
        atomic_json(out_path, output)
        return {**output, "path": str(out_path), "override_count": len(edits)}

    REMOTE_EVAL_VIDEO_INVENTORY_SCRIPT = r'''
import json, os, re
from pathlib import Path, PurePosixPath
suffixes = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}
roots = [Path(item).expanduser() for item in json.loads(os.environ.get("EVAL_VIDEO_ROOTS", "[]"))]

def infer_metadata(root, rel, path):
    parts = list(PurePosixPath(rel).parts)
    name = parts[-1] if parts else rel
    experiment = next((part for part in parts if "eval" in part.lower() or "ckpt" in part.lower() or "cp" in part.lower()), None)
    if experiment is None and len(parts) >= 2:
        experiment = parts[-2]
    task_name = parts[0] if parts else "-"
    success = None
    score = None
    episode_status = None
    episode_seed = None

    episode_match = re.search(r"episode(\d+)\.(?:mp4|webm|mov|mkv|avi|gif)$", name, re.IGNORECASE)
    if episode_match is not None:
        episode_index = int(episode_match.group(1))
        jsonl_path = path.parent / "_episode_results.jsonl"
        if jsonl_path.is_file():
            rows = []
            try:
                for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    try:
                        item_episode_index = int(item.get("episode_index", -1))
                    except Exception:
                        item_episode_index = -1
                    if item_episode_index == episode_index and item.get("counted_in_success_rate", True):
                        rows.append(item)
            except Exception:
                rows = []
            if rows:
                item = rows[-1]
                score_value = item.get("stage_reward")
                if isinstance(score_value, (int, float)):
                    score = float(score_value)
                episode_status = str(item.get("status", "")) or None
                episode_seed = item.get("seed")
                if bool(item.get("success")):
                    success = "success"
                else:
                    status_value = str(item.get("status", "failure")).lower()
                    if status_value == "success":
                        success = "success"
                    elif status_value in {"failure", "failed", "policy_error", "expert_failed", "expert_invalid", "expert_unstable"} or "success" in item:
                        success = "failed"

    if success is None:
        root_resolved = root.resolve()
        for parent in [path.parent, *path.parents]:
            try:
                parent_resolved = parent.resolve()
                if root_resolved not in [parent_resolved, *parent_resolved.parents]:
                    break
            except Exception:
                pass
            result_file = parent / "_result.txt"
            if not result_file.is_file():
                continue
            text = result_file.read_text(encoding="utf-8", errors="replace")[-4000:]
            lowered = text.lower()
            if "success" in lowered and "fail" not in lowered:
                success = "success"
            elif "fail" in lowered or "failure" in lowered:
                success = "failed"
            else:
                for token in reversed(text.replace(",", " ").split()):
                    try:
                        score = float(token)
                        success = "success" if score >= 0.5 else "failed"
                        break
                    except ValueError:
                        continue
            if success:
                break
    if success is None:
        lowered = rel.lower()
        if "success" in lowered:
            success = "success"
        elif "fail" in lowered or "failed" in lowered:
            success = "failed"
        else:
            success = "unknown"
    return {
        "task": task_name,
        "experiment": experiment or "-",
        "success": success,
        "score": score,
        "episode_status": episode_status,
        "episode_seed": episode_seed,
    }

rows = []
for root in roots:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        stat = path.stat()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        rows.append({
            "name": path.name,
            "relative_path": rel,
            "path": str(path),
            "root": str(root),
            "size_mib": round(stat.st_size / (1024**2), 2),
            "mtime": stat.st_mtime,
            **infer_metadata(root, rel, path),
        })
print(json.dumps(rows, ensure_ascii=False))
'''

    def remote_eval_video_roots(target: dict[str, Any]) -> list[str]:
        roots = target.get("eval_video_roots")
        if isinstance(roots, list) and roots:
            return [str(item) for item in roots]
        candidates = []
        if target.get("openpi_repo"):
            candidates.append(str(PurePosixPath(str(target["openpi_repo"])) / "outputs"))
        if target.get("workdir"):
            candidates.append(str(PurePosixPath(str(target["workdir"])) / "eval_videos"))
        return candidates

    def eval_video_metadata(root: Path | str, relative_path: str, *, source: str = "local_4090") -> dict[str, Any]:
        parts = list(PurePosixPath(relative_path).parts)
        name = parts[-1] if parts else relative_path
        experiment = next((part for part in parts if "eval" in part.lower() or "ckpt" in part.lower() or "cp" in part.lower()), None)
        if experiment is None and len(parts) >= 2:
            experiment = parts[-2]
        task_name = parts[0] if parts else "-"
        success: str | None = None
        score: float | None = None
        episode_status: str | None = None
        episode_seed: Any | None = None
        instruction: str | None = None
        model_name: str | None = None
        checkpoint_id: str | None = None
        video_layout: str | None = None
        video_views: list[Any] | None = None
        event_track: dict[str, Any] | None = None
        root_path = Path(root)
        path = root_path / relative_path

        # Prefer per-episode RoboTwin jsonl metadata when available.  The old
        # fallback below only inspected a run-level _result.txt, which is often
        # absent until the whole eval finishes and cannot represent individual
        # episode videos; that made running continuous eval videos show up as
        # "unknown" even though _episode_results.jsonl already had statuses.
        episode_match = re.search(r"episode(\d+)\.(?:mp4|webm|mov|mkv|avi|gif)$", name, re.IGNORECASE)
        if episode_match is not None:
            episode_index = int(episode_match.group(1))
            try:
                item, sidecar_name = load_episode_video_sidecar(path, episode_index)
                if item:
                    score_value = item.get("stage_reward")
                    if isinstance(score_value, (int, float)):
                        score = float(score_value)
                    episode_status = str(item.get("status", "")) or None
                    episode_seed = item.get("seed")
                    instruction_value = item.get("instruction")
                    instruction = str(instruction_value).strip() if instruction_value is not None else None
                    model_value = item.get("model_name")
                    model_name = str(model_value).strip() if model_value is not None else None
                    checkpoint_value = item.get("checkpoint_id")
                    checkpoint_id = str(checkpoint_value).strip() if checkpoint_value is not None else None
                    layout_value = item.get("video_layout")
                    video_layout = str(layout_value).strip() if layout_value is not None else None
                    views_value = item.get("video_views")
                    video_views = list(views_value) if isinstance(views_value, list) else None
                    item_task = item.get("task_name", item.get("task"))
                    if item_task:
                        task_name = str(item_task)
                    if bool(item.get("success")):
                        success = "success"
                    else:
                        status_value = str(item.get("status", "failure")).lower()
                        if status_value in {"success"}:
                            success = "success"
                        elif status_value in {"failure", "failed", "policy_error", "expert_failed", "expert_invalid", "expert_unstable"} or "success" in item:
                            success = "failed"
                    event_track = event_track_from_video_sidecar(path, item, sidecar_name, task_name=task_name)
            except Exception:
                pass

        if success is None:
            for parent in [path.parent, *path.parents]:
                try:
                    if root_path.resolve() not in [parent.resolve(), *parent.resolve().parents]:
                        break
                except Exception:
                    pass
                result_file = parent / "_result.txt"
                if not result_file.is_file():
                    continue
                text = result_file.read_text(encoding="utf-8", errors="replace")[-4000:]
                lowered = text.lower()
                if "success" in lowered and "fail" not in lowered:
                    success = "success"
                elif "fail" in lowered or "failure" in lowered:
                    success = "failed"
                else:
                    for token in reversed(text.replace(",", " ").split()):
                        try:
                            score = float(token)
                            success = "success" if score >= 0.5 else "failed"
                            break
                        except ValueError:
                            continue
                if success:
                    break
        if success is None:
            lowered = relative_path.lower()
            if "success" in lowered:
                success = "success"
            elif "fail" in lowered or "failed" in lowered:
                success = "failed"
            else:
                success = "unknown"
        return {
            "task": task_name,
            "experiment": experiment or "-",
            "success": success,
            "score": score,
            "episode_status": episode_status,
            "episode_seed": episode_seed,
            "instruction": instruction,
            "model_name": model_name,
            "checkpoint_id": checkpoint_id,
            "video_layout": video_layout,
            "video_views": video_views,
            "event_track": event_track,
            "source": source,
            "display_name": name,
        }

    def list_remote_eval_videos() -> tuple[list[dict[str, Any]], dict[str, str]]:
        videos: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        for name, target in config.get("cluster_targets", {}).items():
            if target.get("access_mode") == "slurm_only":
                errors[name] = (
                    target.get("inventory_note")
                    or "slurm-only target: direct SSH video scan is disabled; sync videos through a Slurm staging job/NAS first"
                )
                continue
            host = target.get("submit_host")
            roots = remote_eval_video_roots(target)
            if not host or not roots:
                continue
            command = (
                "EVAL_VIDEO_ROOTS="
                + shlex.quote(json.dumps(roots, ensure_ascii=False))
                + " python3 - <<'REMOTE_EVAL_VIDEO_PY'\n"
                + REMOTE_EVAL_VIDEO_INVENTORY_SCRIPT
                + "\nREMOTE_EVAL_VIDEO_PY"
            )
            try:
                result = subprocess.run(
                    [*SSH_COMMAND, str(host), command],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
            except Exception as exc:
                errors[name] = str(exc)
                continue
            if result.returncode != 0:
                errors[name] = (result.stderr or result.stdout)[-2000:]
                continue
            try:
                rows = json.loads(result.stdout or "[]")
            except json.JSONDecodeError as exc:
                errors[name] = f"invalid JSON: {exc}: {(result.stdout or '')[-500:]}"
                continue
            for row in rows if isinstance(rows, list) else []:
                root = str(row.get("root", ""))
                rel = str(row.get("relative_path", ""))
                if root not in roots or not rel or PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts:
                    continue
                metadata = eval_video_metadata(root, rel, source=name)
                # For remote videos, per-episode jsonl/_result.txt lives on the
                # remote host, so metadata inferred by the remote inventory
                # script must override local path fallbacks.  Otherwise every
                # remote episode without a local copy is shown as "unknown".
                for key in ("task", "experiment", "success", "score", "episode_status", "episode_seed"):
                    if key in row and row.get(key) is not None:
                        metadata[key] = row.get(key)
                videos.append({
                    **row,
                    **metadata,
                    "id": base64.urlsafe_b64encode(f"{name}\0{root}\0{rel}".encode()).decode().rstrip("="),
                    "source": name,
                    "host": host,
                    "remote": True,
                    "playable": False,
                    "syncable": True,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(row.get("mtime") or 0))),
                    "sync_url": "/api/eval-videos/sync",
                    "deletable": False,
                })
        return videos, errors

    @app.get("/api/eval-videos")
    def list_eval_videos():
        limit = safe_int(request.args.get("limit", 200), "limit", 1, 1000)
        include_remote = str(request.args.get("include_remote", "")).lower() in {"1", "true", "yes"}
        task_filter = str(request.args.get("task", "")).strip().lower()
        experiment_filter = str(request.args.get("experiment", "")).strip().lower()
        success_filter = str(request.args.get("success", "")).strip().lower()
        query_filter = str(request.args.get("q", "")).strip().lower()
        videos = []
        for root in eval_video_roots():
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                stat = path.stat()
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    rel = path.name
                videos.append({
                    "id": encode_video_id(path),
                    **eval_video_metadata(root, rel),
                    "name": path.name,
                    "relative_path": rel,
                    "root": str(root),
                    "size_mib": round(stat.st_size / (1024**2), 2),
                    "mtime": stat.st_mtime,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                    "url": f"/api/eval-videos/{encode_video_id(path)}",
                    "remote": False,
                    "playable": True,
                    "syncable": False,
                    "deletable": True,
                })
        remote_errors = {}
        if include_remote:
            remote_videos, remote_errors = list_remote_eval_videos()
            videos.extend(remote_videos)
        def video_matches(item: dict[str, Any]) -> bool:
            if task_filter and task_filter not in str(item.get("task", "")).lower():
                return False
            if experiment_filter and experiment_filter not in str(item.get("experiment", "")).lower():
                return False
            if success_filter and success_filter not in {"all", str(item.get("success", "unknown")).lower()}:
                return False
            if query_filter:
                haystack = " ".join(str(item.get(key, "")) for key in ("relative_path", "name", "task", "experiment", "source")).lower()
                if query_filter not in haystack:
                    return False
            return True
        videos = [item for item in videos if video_matches(item)]
        videos.sort(key=lambda item: item["mtime"], reverse=True)
        facets = {
            "tasks": sorted({str(item.get("task") or "-") for item in videos}),
            "experiments": sorted({str(item.get("experiment") or "-") for item in videos}),
            "success": {key: sum(1 for item in videos if item.get("success") == key) for key in ("success", "failed", "unknown")},
        }
        return jsonify({
            "videos": videos[:limit],
            "total": len(videos),
            "facets": facets,
            "roots": [str(root) for root in eval_video_roots()],
            "remote_errors": remote_errors,
        })

    @app.get("/api/eval-videos/<video_id>")
    def get_eval_video(video_id: str):
        return send_file(decode_video_id(video_id), conditional=True, max_age=0)

    @app.get("/api/eval-videos/<video_id>/event-overrides")
    def get_eval_video_event_overrides(video_id: str):
        path = decode_video_id(video_id)
        return jsonify(read_video_event_overrides(path) or {"schema": EVENT_OVERRIDE_SCHEMA, "edits": []})

    @app.post("/api/eval-videos/<video_id>/event-overrides")
    def post_eval_video_event_overrides(video_id: str):
        return jsonify(save_video_event_override(video_id, request.get_json(force=True)))

    @app.post("/api/eval-videos/batch-delete")
    def delete_eval_videos_batch():
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        raw_video_ids = payload.get("video_ids")
        if not isinstance(raw_video_ids, list) or not raw_video_ids:
            raise ValueError("video_ids must be a non-empty list")
        if len(raw_video_ids) > 200:
            raise ValueError("cannot delete more than 200 videos at once")

        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        roots = [root.resolve() for root in eval_video_roots()]
        for raw_video_id in raw_video_ids:
            if not isinstance(raw_video_id, str) or not raw_video_id.strip():
                raise ValueError("video_ids must contain non-empty strings")
            video_id = raw_video_id.strip()
            if video_id in seen:
                continue
            seen.add(video_id)
            path = decode_video_id(video_id)
            root = next((root for root in roots if path == root or root in path.parents), None)
            if root is None:
                raise FileNotFoundError(f"video root not found for {path}")
            relative_path = path.relative_to(root).as_posix()
            selected.append(
                {
                    "video_id": video_id,
                    "path": path,
                    "root": root,
                    "relative_path": relative_path,
                    "metadata": eval_video_metadata(root, relative_path),
                }
            )

        for item in selected:
            if not item["path"].is_file():
                raise FileNotFoundError(str(item["path"]))

        deleted: list[dict[str, Any]] = []
        for item in selected:
            path = item["path"]
            path.unlink()
            prune_empty_eval_video_parents(path)
            deleted.append(
                {
                    "id": item["video_id"],
                    "path": str(path),
                    "root": str(item["root"]),
                    "relative_path": item["relative_path"],
                    **item["metadata"],
                }
            )

        return jsonify(
            {
                "deleted": True,
                "deleted_count": len(deleted),
                "video_ids": [item["id"] for item in deleted],
                "deleted_videos": deleted,
            }
        )

    @app.post("/api/eval-videos/sync")
    def sync_eval_video():
        payload = request.get_json(force=True)
        source = str(payload.get("source", ""))
        root = str(payload.get("root", ""))
        relative_path = str(payload.get("relative_path", ""))
        overwrite = bool(payload.get("overwrite", False))
        parallelism = safe_int(payload.get("parallelism", config.get("transfer_parallelism", 4)), "parallelism", 1, 16)
        targets = config.get("cluster_targets", {})
        if source not in targets:
            raise ValueError(f"unknown video source target: {source}")
        target = targets[source]
        if target.get("access_mode") == "slurm_only":
            raise ValueError(
                "remote video source is Slurm-only; copy/export the video via a Slurm staging job or NAS before Dashboard playback"
            )
        roots = remote_eval_video_roots(target)
        rel = PurePosixPath(relative_path)
        if root not in roots:
            raise ValueError("remote video root is not configured")
        if rel.is_absolute() or ".." in rel.parts or Path(rel.name).suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("invalid remote video relative_path")
        local_roots = eval_video_roots()
        if not local_roots:
            raise ValueError("no local eval_video_roots configured")
        command = [
            config["openpi_python"],
            str(APP_DIR / "video_transfer_runner.py"),
            "--source-name", source,
            "--source-host", str(target.get("submit_host")),
            "--source-root", root,
            "--relative-path", relative_path,
            "--target-root", str(local_roots[0]),
            "--parallelism", str(parallelism),
        ]
        if overwrite:
            command.append("--overwrite")
        task = tasks.start(
            "transfer",
            command,
            env=build_environment(config, None),
            metadata={
                "transfer_kind": "eval_video",
                "source": source,
                "target": "local_4090",
                "source_path": str(PurePosixPath(root) / rel),
                "target_path": str(local_roots[0] / source / Path(*rel.parts)),
                "overwrite": overwrite,
                "parallelism": parallelism,
                "slurm_involved": slurm_involved,
                "nas_dataset_staging_root": config.get("nas_dataset_staging_root"),
            },
        )
        return jsonify(task), 201

    def dataset_location_configs() -> dict[str, dict[str, Any]]:
        locations = {
            "local_4090": {
                "name": "local_4090",
                "label": "4×4090 NVMe（活动）",
                "kind": "local",
                "host": None,
                "dataset_root": config["dataset_root"],
                "available": True,
                "training_enabled": True,
            }
        }
        for name, storage in config.get("local_storage_locations", {}).items():
            if not storage.get("dataset_root"):
                continue
            locations[name] = {
                "name": name,
                "label": storage.get("label") or name,
                "kind": storage.get("kind") or "local_archive",
                "host": None,
                "dataset_root": storage.get("dataset_root"),
                "available": bool(storage.get("available", True)),
                "training_enabled": bool(storage.get("training_enabled", False)),
            }
        for name, target in config.get("cluster_targets", {}).items():
            if not target.get("dataset_root"):
                continue
            locations[name] = {
                "name": name,
                "label": target.get("label") or name,
                "kind": "slurm_only" if target.get("access_mode") == "slurm_only" else "ssh",
                "access_mode": target.get("access_mode", "ssh"),
                "inventory_note": target.get("inventory_note"),
                "inventory_cache_path": target.get("inventory_cache_path"),
                "inventory_cache_host": target.get("inventory_cache_host", target.get("submit_host")),
                "inventory_source_path": target.get("inventory_source_path"),
                "inventory_source_host": target.get("inventory_source_host", target.get("submit_host")),
                "nas_dataset_staging_root": target.get("nas_dataset_staging_root") or config.get("nas_dataset_staging_root"),
                "host": target.get("submit_host"),
                "submit_host": target.get("submit_host"),
                "partition": target.get("partition"),
                "node": target.get("node"),
                "gpu_type": target.get("gpu_type"),
                "dataset_root": target.get("dataset_root"),
                "available": True,
            }
        return locations

    def checkpoint_location_configs(config_name: str) -> dict[str, dict[str, Any]]:
        config_name = safe_name(config_name, "checkpoint config name")
        locations = {
            "local_4090": {
                "name": "local_4090",
                "label": "4×4090 NVMe（活动）",
                "kind": "local",
                "host": None,
                "dataset_root": str(PurePosixPath(str(config["checkpoint_base_dir"]).rstrip("/")) / config_name),
                "available": True,
                "serving_enabled": True,
            }
        }
        for name, storage in config.get("local_storage_locations", {}).items():
            if not storage.get("checkpoint_base_dir"):
                continue
            locations[name] = {
                "name": name,
                "label": storage.get("label") or name,
                "kind": storage.get("kind") or "local_archive",
                "host": None,
                "dataset_root": str(
                    PurePosixPath(str(storage["checkpoint_base_dir"]).rstrip("/")) / config_name
                ),
                "available": bool(storage.get("available", True)),
                "serving_enabled": bool(storage.get("serving_enabled", False)),
            }
        for name, target in config.get("cluster_targets", {}).items():
            if not target.get("checkpoint_base_dir"):
                continue
            locations[name] = {
                "name": name,
                "label": target.get("label") or name,
                "kind": "slurm_only" if target.get("access_mode") == "slurm_only" else "ssh",
                "access_mode": target.get("access_mode", "ssh"),
                "host": target.get("submit_host"),
                "submit_host": target.get("submit_host"),
                "partition": target.get("partition"),
                "node": target.get("node"),
                "gpu_type": target.get("gpu_type"),
                "workdir": target.get("workdir"),
                "remote_job_dir": target.get("remote_job_dir"),
                "log_dir": target.get("log_dir"),
                "dataset_root": str(PurePosixPath(str(target["checkpoint_base_dir"]).rstrip("/")) / config_name),
                "nas_dataset_staging_root": target.get("nas_checkpoint_staging_root") or config.get("nas_checkpoint_staging_root"),
                "available": True,
            }
        return locations

    def local_dataset_inventory(location: dict[str, Any]) -> list[dict[str, Any]]:
        root = Path(str(location["dataset_root"])).expanduser()
        rows = []
        for directory in sorted(root.iterdir() if root.exists() else []):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            info = read_json(directory / "meta" / "info.json")
            if not isinstance(info, dict):
                continue
            marker = read_json(directory / "meta" / "dashboard_dataset_origin.json")
            origin = dataset_origin_info(directory.name, directory, info).get("dataset_origin", "unknown")
            rows.append({
                "id": directory.name,
                "origin": origin,
                "path": str(directory),
                "episodes": info.get("total_episodes"),
                "frames": info.get("total_frames"),
                "fps": info.get("fps"),
                "robot_type": info.get("robot_type"),
                "mtime": directory.stat().st_mtime,
                "marker": marker if isinstance(marker, dict) else None,
            })
        return rows

    REMOTE_DATASET_INVENTORY_SCRIPT = r'''
import json, os
from pathlib import Path
root = Path(os.environ["DATASET_ROOT"]).expanduser()
def read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None
def origin_for(dataset_id, info, marker):
    if isinstance(marker, dict) and marker.get("origin") in {"real", "simulation", "unknown"}:
        return marker.get("origin")
    for key in ("dataset_origin", "data_origin", "source_domain"):
        value = str(info.get(key, "")).lower() if isinstance(info, dict) else ""
        if value in {"real", "robot", "real_robot", "physical"}: return "real"
        if value in {"simulation", "sim", "synthetic", "synthetic_sim"}: return "simulation"
    if isinstance(info, dict) and isinstance(info.get("simulation"), bool):
        return "simulation" if info["simulation"] else "real"
    name = dataset_id.lower(); robot_type = str((info or {}).get("robot_type") or "").lower()
    if robot_type == "piper":
        return "real"
    if __import__("re").search(r"(?:^|[._-])real(?:[._-]|$)", name) or name == "my_dataset":
        return "real"
    simulation_name = any(token in name for token in ("sim", "synth", "synthetic", "smoke", "robottwin"))
    if simulation_name or robot_type in {"aloha", "sim", "simulation"} or (robot_type.startswith("piper_single_arm") and bool((info or {}).get("video_path"))):
        return "simulation"
    if "piper" in robot_type:
        return "real"
    return "unknown"
rows=[]
if root.exists():
    for directory in sorted(root.iterdir(), key=lambda path: path.name):
        if not directory.is_dir() or directory.name.startswith("."): continue
        info = read_json(directory / "meta" / "info.json")
        if not isinstance(info, dict): continue
        marker = read_json(directory / "meta" / "dashboard_dataset_origin.json")
        stat = directory.stat()
        rows.append({
            "id": directory.name,
            "origin": origin_for(directory.name, info, marker),
            "path": str(directory),
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
            "fps": info.get("fps"),
            "robot_type": info.get("robot_type"),
            "mtime": stat.st_mtime,
            "marker": marker if isinstance(marker, dict) else None,
        })
print(json.dumps(rows, ensure_ascii=False))
'''

    def remote_dataset_inventory(location: dict[str, Any], *, timeout: int = 30) -> tuple[list[dict[str, Any]], str | None]:
        if location.get("kind") in {"local", "local_archive"}:
            return local_dataset_inventory(location), None
        if location.get("kind") == "slurm_only":
            cache_path = location.get("inventory_cache_path")
            cache_host = location.get("inventory_cache_host")
            if not cache_path:
                return [], (
                    location.get("inventory_note")
                    or "slurm-only target: no inventory_cache_path configured; submit a staging/setup job through login-server"
                )
            stdout = ""
            source_label = str(cache_path)

            def read_inventory_source() -> tuple[str, str] | tuple[None, None]:
                source_path = location.get("inventory_source_path")
                source_host = location.get("inventory_source_host") or location.get("submit_host")
                if not source_path:
                    return None, None
                if not source_host or str(source_host) in {"local", "local_4090", "4x4090"}:
                    candidate = Path(str(source_path)).expanduser()
                    if candidate.is_file() and candidate.stat().st_size > 0:
                        return candidate.read_text(encoding="utf-8"), str(candidate)
                    return None, None
                command = f"test -s {shlex.quote(str(source_path))} && cat {shlex.quote(str(source_path))}"
                try:
                    result = subprocess.run(
                        [*SSH_COMMAND, str(source_host), command],
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=min(timeout, 20),
                    )
                except Exception:
                    return None, None
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout, f"{source_host}:{source_path}"
                return None, None

            if not cache_host or str(cache_host) in {"local", "local_4090", "4x4090"}:
                local_cache = Path(str(cache_path)).expanduser()
                if local_cache.is_file() and local_cache.stat().st_size > 0:
                    stdout = local_cache.read_text(encoding="utf-8")
                    source_label = str(local_cache)
                else:
                    source_stdout, source_label_value = read_inventory_source()
                    if source_stdout:
                        local_cache.parent.mkdir(parents=True, exist_ok=True)
                        local_cache.write_text(source_stdout, encoding="utf-8")
                        stdout = source_stdout
                        source_label = source_label_value or str(local_cache)
                    else:
                        note = location.get("inventory_note") or "slurm-only local inventory cache is not ready"
                        return [], f"{note}; cache={local_cache}"
            else:
                command = f"test -s {shlex.quote(str(cache_path))} && cat {shlex.quote(str(cache_path))}"
                try:
                    result = subprocess.run(
                        [*SSH_COMMAND, str(cache_host), command],
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=min(timeout, 12),
                    )
                except Exception as exc:
                    return [], str(exc)
                if result.returncode != 0:
                    source_stdout, source_label_value = read_inventory_source()
                    if source_stdout:
                        stdout = source_stdout
                        source_label = source_label_value or f"{cache_host}:{cache_path}"
                    else:
                        note = location.get("inventory_note") or "slurm-only inventory cache is not ready"
                        return [], f"{note}; cache={cache_path}; {(result.stderr or result.stdout)[-1000:]}"
                else:
                    stdout = result.stdout
                    source_label = f"{cache_host}:{cache_path}"
            try:
                payload = json.loads(stdout or "{}")
            except json.JSONDecodeError as exc:
                return [], f"invalid JSON inventory cache from {source_label}: {exc}"
            if isinstance(payload, dict):
                rows = payload.get("datasets", [])
                if not rows and payload.get("roots"):
                    note = location.get("inventory_note") or "slurm-only inventory cache contains no datasets"
                    return [], f"{note}; roots={payload.get('roots')}"
                return rows if isinstance(rows, list) else [], None
            if isinstance(payload, list):
                return payload, None
            return [], f"unsupported inventory cache format from {source_label}"
        host = location.get("host")
        if not host:
            return [], "missing ssh host"
        env = f"DATASET_ROOT={shlex.quote(str(location['dataset_root']))}"
        command = f"{env} python3 - <<'REMOTE_DATASET_PY'\n{REMOTE_DATASET_INVENTORY_SCRIPT}\nREMOTE_DATASET_PY"
        try:
            result = subprocess.run(
                [*SSH_COMMAND, str(host), command],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except Exception as exc:
            return [], str(exc)
        if result.returncode != 0:
            return [], (result.stderr or result.stdout)[-2000:]
        try:
            rows = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            return [], f"invalid JSON from {host}: {exc}: {(result.stdout or '')[-500:]}"
        return rows if isinstance(rows, list) else [], None

    def grouped_dataset_locations(*, origin_filter: str | None = None) -> dict[str, Any]:
        locations = dataset_location_configs()
        groups: dict[str, dict[str, Any]] = {}
        errors = {}
        for name, location in locations.items():
            rows, error = remote_dataset_inventory(location)
            if error:
                errors[name] = error
            for row in rows:
                origin = normalize_dataset_origin(row.get("origin", "unknown"))
                if origin_filter and origin != origin_filter:
                    continue
                dataset_id = str(row.get("id"))
                entry = groups.setdefault(dataset_id, {"id": dataset_id, "locations": []})
                entry["locations"].append({
                    **row,
                    "origin": origin,
                    "target": name,
                    "label": location.get("label") or name,
                    "host": location.get("host"),
                    "root": location.get("dataset_root"),
                    "kind": location.get("kind"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(row.get("mtime") or 0))),
                })
        datasets = []
        for item in groups.values():
            item["duplicate_count"] = len(item["locations"])
            item["origins"] = sorted({loc.get("origin", "unknown") for loc in item["locations"]})
            item["targets"] = sorted({loc.get("target") for loc in item["locations"] if loc.get("target")})
            datasets.append(item)
        datasets.sort(key=lambda item: item["id"])
        return {"datasets": datasets, "locations": locations, "errors": errors}

    @app.get("/api/dataset-locations")
    def dataset_locations():
        origin = request.args.get("origin")
        if origin:
            origin = normalize_dataset_origin(origin)
        return jsonify(grouped_dataset_locations(origin_filter=origin))

    @app.get("/api/cluster-resources")
    def cluster_resources():
        """Return a read-only H100/H200 Slurm resource snapshot.

        This never starts a service or opens a port on H100/H200.  It runs the
        existing query helper locally on 4x4090, which itself uses SSH to the
        Slurm login node when needed.
        """
        script = Path(config.get("cluster_resources_script", ""))
        if not script.is_file():
            raise FileNotFoundError(f"cluster resources script not found: {script}")
        show_all = str(request.args.get("all_jobs", "")).lower() in {"1", "true", "yes"}
        native = str(request.args.get("native", "")).lower() in {"1", "true", "yes"}
        command = [str(script), "--compact"]
        if show_all:
            command.append("--all-jobs")
        if native:
            command.append("--native")
        started = time.time()
        result = subprocess.run(
            command,
            cwd=str(REPO_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        return jsonify({
            "command": command,
            "returncode": result.returncode,
            "ok": result.returncode == 0,
            "elapsed_s": round(time.time() - started, 3),
            "output": result.stdout[-64_000:],
            "note": "H100/H200 are queried via SSH/Slurm only; no remote Dashboard port is opened.",
        })

    @app.post("/api/datasets/<dataset_id>/sync")
    def sync_dataset(dataset_id: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        payload = request.get_json(force=True)
        source_name = str(payload.get("source", "local_4090"))
        target_name = str(payload.get("target", ""))
        if not target_name:
            raise ValueError("target is required")
        overwrite = bool(payload.get("overwrite", False))
        parallelism = safe_int(payload.get("parallelism", config.get("transfer_parallelism", 4)), "parallelism", 1, 16)
        locations = dataset_location_configs()
        if source_name not in locations:
            raise ValueError(f"unknown source location: {source_name}")
        if target_name not in locations:
            raise ValueError(f"unknown target location: {target_name}")
        if source_name == target_name:
            raise ValueError("source and target must differ")
        slurm_involved = (
            locations[source_name].get("kind") == "slurm_only"
            or locations[target_name].get("kind") == "slurm_only"
        )
        runner = "slurm_dataset_sync_runner.py" if slurm_involved else "dataset_transfer_runner.py"
        command = [
            config["openpi_python"],
            str(APP_DIR / runner),
            "--dataset-id", dataset_id,
            "--source-json", json_arg(locations[source_name]),
            "--target-json", json_arg(locations[target_name]),
            "--parallelism", str(parallelism),
        ]
        if slurm_involved and config.get("nas_dataset_staging_root"):
            command += ["--nas-staging-root", str(config["nas_dataset_staging_root"])]
        if overwrite:
            command.append("--overwrite")
        if bool(payload.get("skip_existing", False)):
            command.append("--skip-existing")
        task = tasks.start(
            "transfer",
            command,
            env=build_environment(config, None),
            metadata={
                "dataset_id": dataset_id,
                "source": source_name,
                "target": target_name,
                "overwrite": overwrite,
                "source_path": str(PurePosixPath(str(locations[source_name]["dataset_root"])) / dataset_id),
                "target_path": str(PurePosixPath(str(locations[target_name]["dataset_root"])) / dataset_id),
                "parallelism": parallelism,
            },
        )
        return jsonify(task), 201

    @app.post("/api/checkpoints/sync")
    def sync_checkpoint():
        payload = request.get_json(force=True)
        source_name = str(payload.get("source", ""))
        target_name = str(payload.get("target", "local_4090") or "local_4090")
        if not source_name:
            raise ValueError("source is required")
        if payload.get("config_name"):
            config_name = safe_name(payload.get("config_name"), "checkpoint config name")
        else:
            arm_mode = str(payload.get("arm_mode", "single"))
            model_variant = str(payload.get("model_variant", "pi05"))
            if arm_mode not in {"single", "bimanual"}:
                raise ValueError("arm_mode must be single or bimanual")
            if model_variant not in MODEL_VARIANTS:
                raise ValueError(f"model_variant must be one of {sorted(MODEL_VARIANTS)}")
            config_name = policy_config_name(arm_mode, model_variant)
        exp_name = safe_name(payload.get("exp_name"), "experiment name")
        overwrite = bool(payload.get("overwrite", False))
        skip_existing = bool(payload.get("skip_existing", not overwrite))
        parallelism = safe_int(payload.get("parallelism", config.get("transfer_parallelism", 4)), "parallelism", 1, 16)
        locations = checkpoint_location_configs(config_name)
        if source_name not in locations:
            raise ValueError(f"unknown checkpoint source location: {source_name}")
        if target_name not in locations:
            raise ValueError(f"unknown checkpoint target location: {target_name}")
        if source_name == target_name:
            raise ValueError("source and target must differ")
        slurm_involved = (
            locations[source_name].get("kind") == "slurm_only"
            or locations[target_name].get("kind") == "slurm_only"
        )
        runner = "slurm_dataset_sync_runner.py" if slurm_involved else "dataset_transfer_runner.py"
        command = [
            config["openpi_python"],
            str(APP_DIR / runner),
            "--dataset-id", exp_name,
            "--source-json", json_arg(locations[source_name]),
            "--target-json", json_arg(locations[target_name]),
            "--parallelism", str(parallelism),
        ]
        if slurm_involved and config.get("nas_checkpoint_staging_root"):
            command += ["--nas-staging-root", str(config["nas_checkpoint_staging_root"])]
        if overwrite:
            command.append("--overwrite")
        if skip_existing:
            command.append("--skip-existing")
        # Action-contract markers live beside the experiment directory rather
        # than inside the numeric Orbax step.  Direct transfers can copy the
        # small marker after the checkpoint manifest; Slurm/NAS transfers use
        # their own staging path and are handled by the Slurm sync runner.
        if not slurm_involved:
            command += ["--marker-name", exp_name]
        task = tasks.start(
            "transfer",
            command,
            env=build_environment(config, None),
            metadata={
                "transfer_kind": "checkpoint",
                "config_name": config_name,
                "exp_name": exp_name,
                "source": source_name,
                "target": target_name,
                "overwrite": overwrite,
                "skip_existing": skip_existing,
                "source_path": str(PurePosixPath(str(locations[source_name]["dataset_root"])) / exp_name),
                "target_path": str(PurePosixPath(str(locations[target_name]["dataset_root"])) / exp_name),
                "parallelism": parallelism,
            },
        )
        return jsonify(task), 201

    @app.post("/api/checkpoints/batch-delete")
    def delete_checkpoints_batch():
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        raw_paths = payload.get("checkpoint_paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError("checkpoint_paths must be a non-empty list")
        if len(raw_paths) > 200:
            raise ValueError("cannot delete more than 200 checkpoints at once")

        checkpoint_root = Path(config["checkpoint_base_dir"]).expanduser().resolve()
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_path in raw_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("checkpoint_paths must contain non-empty strings")
            path = Path(raw_path).expanduser().resolve()
            try:
                path.relative_to(checkpoint_root)
            except ValueError:
                raise ValueError(f"checkpoint is outside checkpoint_base_dir: {path}")
            if not path.is_dir():
                raise FileNotFoundError(str(path))
            if not is_complete_checkpoint(path):
                raise ValueError(f"checkpoint is not complete: {path}")
            identity = training_checkpoint_identity(path, checkpoint_root)
            if identity is None:
                raise ValueError(f"checkpoint is not a standard training checkpoint: {path}")
            visible_checkpoint, dataset_ids, dataset_origins = checkpoint_matches_visible_datasets(path)
            if not visible_checkpoint:
                raise ValueError(f"checkpoint is hidden on this dashboard and cannot be deleted here: {path}")
            cache_key = str(path)
            if cache_key in seen:
                continue
            seen.add(cache_key)
            selected.append({
                **identity,
                "path": path,
                "path_str": cache_key,
                "dataset_ids": dataset_ids,
                "dataset_origins": dataset_origins,
            })

        with tasks.lock:
            active_references: dict[str, list[dict[str, Any]]] = {}
            for item in selected:
                references = checkpoint_active_references(item["path"])
                if references:
                    active_references[item["path_str"]] = references
            if active_references:
                details = "; ".join(
                    f"{path}: " + ", ".join(
                        f"{ref['task_id']}[{ref['type']}:{ref['state']}]({', '.join(r['kind'] for r in ref['references'])})"
                        for ref in refs
                    )
                    for path, refs in sorted(active_references.items())
                )
                raise ValueError(f"cannot delete selected checkpoints; active task(s) still reference them: {details}")

            deleted: list[dict[str, Any]] = []
            for item in selected:
                path = item["path"]
                if not path.is_dir():
                    raise FileNotFoundError(str(path))
                shutil.rmtree(path)
                checkpoint_size_cache.pop(item["path_str"], None)
                prune_empty_checkpoint_parents(path)
                deleted.append(
                    {
                        "path": item["path_str"],
                        "experiment": item["experiment"],
                        "step": item["checkpoint_step"],
                        "model_variant": item["model_variant"],
                        "arm_mode": item["arm_mode"],
                        "dataset_ids": item["dataset_ids"],
                    }
                )

        return jsonify({
            "deleted": True,
            "deleted_count": len(deleted),
            "checkpoint_paths": [item["path"] for item in deleted],
            "deleted_checkpoints": deleted,
        })

    def collection_root() -> Path:
        path = Path(config["workspace_root"]) / "collection_sessions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def collection_path(session_id: str) -> Path:
        return collection_root() / f"{safe_name(session_id, 'collection session id')}.json"

    def list_collection_sessions() -> list[dict[str, Any]]:
        rows = []
        for path in collection_root().glob("*.json"):
            value = read_json(path)
            if isinstance(value, dict):
                rows.append(value)
        return sorted(rows, key=lambda item: item.get("created_at", ""), reverse=True)

    @app.get("/api/collection-sessions")
    def get_collection_sessions():
        return jsonify({"sessions": list_collection_sessions()})

    @app.post("/api/collection-sessions")
    def create_collection_session():
        payload = request.get_json(force=True)
        dataset_id = safe_name(payload.get("dataset_id") or payload.get("name"), "dataset id")
        target = str(payload.get("target", "local_4090"))
        locations = dataset_location_configs()
        if target not in locations:
            raise ValueError(f"unknown collection target: {target}")
        origin = normalize_dataset_origin(payload.get("dataset_origin", config.get("upload_default_origin", "simulation")), allow_unknown=False)
        session_id = f"collect-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        server_url = payload.get("server") or f"http://192.168.101.9:{config['port']}"
        upload_command = (
            f"python upload_dataset_4090.py LEROBOT_OR_GUI_NPZ_DIR --name {dataset_id} "
            f"--dataset-origin {origin} --server {server_url} --token TOKEN --workers 4 --merge"
        )
        session = {
            "id": session_id,
            "dataset_id": dataset_id,
            "dataset_origin": origin,
            "target": target,
            "target_path": str(PurePosixPath(str(locations[target]["dataset_root"])) / dataset_id),
            "status": str(payload.get("status", "created")),
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "metadata": payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
            "upload_command": upload_command,
        }
        atomic_json(collection_path(session_id), session)
        return jsonify(session), 201

    @app.get("/api/collection-sessions/<session_id>")
    def get_collection_session(session_id: str):
        value = read_json(collection_path(session_id))
        if not isinstance(value, dict):
            raise FileNotFoundError(session_id)
        return jsonify(value)

    @app.patch("/api/collection-sessions/<session_id>")
    def update_collection_session(session_id: str):
        path = collection_path(session_id)
        value = read_json(path)
        if not isinstance(value, dict):
            raise FileNotFoundError(session_id)
        payload = request.get_json(force=True)
        for key in ("status", "upload_task_id", "notes", "dataset_id"):
            if key in payload:
                value[key] = payload[key]
        if isinstance(payload.get("metadata"), dict):
            value["metadata"] = {**value.get("metadata", {}), **payload["metadata"]}
        value["updated_at"] = now_iso()
        atomic_json(path, value)
        return jsonify(value)

    @app.post("/api/tasks/norm")
    def start_norm():
        payload = request.get_json(force=True)
        dataset_id, arm_mode, arm_side, schema, dataset_contract = parse_dataset(payload)
        model_contract = action_contract_for_model(dataset_contract)
        base_checkpoint, model_variant = resolve_base_model(payload)
        split = parse_episode_split(payload, dataset_id, dataset_contract)
        batch_size = safe_int(payload.get("batch_size", 16), "batch_size", 1, 1024)
        num_workers = safe_int(payload.get("num_workers", 2), "num_workers", 1, 64)
        max_frames = payload.get("max_frames")
        parsed_max_frames = (
            None if max_frames in (None, "") else safe_int(max_frames, "max_frames", 1, 10**9)
        )
        command = build_norm_command(
            dataset_id,
            arm_mode,
            arm_side,
            schema,
            base_checkpoint=base_checkpoint,
            model_variant=model_variant,
            batch_size=batch_size,
            num_workers=num_workers,
            split=split,
            model_contract=model_contract,
            max_frames=parsed_max_frames,
        )
        task = tasks.start(
            "norm", command,
            env=build_environment(config, None),
            metadata={
                "dataset_id": dataset_id,
                "arm_mode": arm_mode,
                "arm_side": arm_side,
                "schema": schema,
                **model_contract["contract_fingerprint"],
                "raw_gripper_semantics": model_contract["raw_gripper_semantics"],
                "wire_gripper_semantics": model_contract["wire_gripper_semantics"],
                "delivery_action_convention": (
                    model_contract["model_action_convention"] if schema == "delivery" else None
                ),
                "model_variant": model_variant,
                "base_checkpoint": str(base_checkpoint),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "max_frames": parsed_max_frames,
                "test_ratio": split.test_ratio,
                "split_seed": split.seed,
                "split_source": "request",
                "train_episodes": len(split.train_episodes),
                "test_episodes": len(split.test_episodes),
                "test_episode_indexes": list(split.test_episodes),
                "norm_path": str(norm_stats_path(dataset_id, arm_mode, model_variant)),
                "automatic": False,
            },
        )
        return jsonify(task), 201

    @app.post("/api/tasks/train")
    def start_train():
        payload = request.get_json(force=True)
        dataset_id, arm_mode, arm_side, schema, dataset_contract = parse_dataset(payload)
        base_checkpoint, model_variant = resolve_base_model(payload)
        exp_name = safe_name(payload.get("exp_name"), "experiment name")
        mode = str(payload.get("mode", "auto"))
        if mode not in {"auto", "new", "resume", "overwrite"}:
            raise ValueError("mode must be auto, new, resume, or overwrite")
        checkpoint_dir = (
            Path(config["checkpoint_base_dir"])
            / policy_config_name(arm_mode, model_variant)
            / exp_name
        )
        if mode == "new" and checkpoint_dir.exists():
            raise FileExistsError(
                f"checkpoint directory already exists: {checkpoint_dir}; "
                "choose auto/resume to continue it, or overwrite to replace it"
            )
        target_complete_steps = full_state_checkpoint_steps(checkpoint_dir)
        saved_steps = bool(target_complete_steps)
        target_has_artifacts = checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir())
        # ``auto`` must not turn an empty/incomplete experiment into a silent
        # foundation-model run.  We decide between new/in-place/external resume
        # after reading the dataset norm manifest below.
        effective_mode = ("resume" if saved_steps else "new") if mode == "auto" else mode
        resume_checkpoint: Path | None = None
        resume_kind = "none"
        requested_resume_checkpoint = payload.get("resume_checkpoint")

        # Auto-resume is allowed to recover old checkpoints only through an
        # explicit compatibility choice. The generated command still carries
        # the resolved convention/semantics, so it is never silent at runtime.
        model_convention = (
            DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
            if schema == "delivery"
            else None
        )
        model_gripper = (
            dataset_contract.get("model_gripper_semantics")
            if schema == "joint"
            else None
        )
        marker = checkpoint_action_contract(checkpoint_dir) if saved_steps else None
        if effective_mode == "resume" and saved_steps:
            if marker is not None:
                if schema == "delivery":
                    model_convention = marker.get("model_action_convention") or marker.get(
                        "delivery_action_convention"
                    )
                else:
                    model_gripper = marker.get("gripper_semantics") or marker.get(
                        "model_gripper_semantics"
                    )
            elif schema == "delivery" and dataset_contract.get("legacy_delivery_v2"):
                model_convention = DELIVERY_STEP_ACTION_CONVENTION
            elif schema == "joint" and dataset_contract.get("legacy_joint_v2"):
                model_gripper = LEGACY_JOINT_GRIPPER_SEMANTICS
            else:
                raise ValueError(
                    "existing checkpoint has no action-contract marker; it cannot be resumed "
                    "without a verified legacy dataset/convention"
                )
        model_contract = action_contract_for_model(
            dataset_contract,
            delivery_action_convention=model_convention,
            model_gripper_semantics=model_gripper,
        )
        if effective_mode == "resume" and saved_steps and marker is not None:
            marker_version = int(marker.get("version", 1))
            legacy_temporal_compat = bool(
                marker_version < ACTION_CONTRACT_MARKER_VERSION
                and (
                    dataset_contract.get("legacy_delivery_v2")
                    or dataset_contract.get("legacy_joint_v2")
                )
            )
            if marker_version < ACTION_CONTRACT_MARKER_VERSION and not legacy_temporal_compat:
                raise ValueError(
                    "checkpoint action marker predates action_offset/model_action_start_offset"
                )
            expected_items = (
                normalize_contract_fingerprint(model_contract["contract_fingerprint"]).items()
                if legacy_temporal_compat
                else model_contract["contract_fingerprint"].items()
            )
            mismatches = {
                key: {"checkpoint": marker.get(key), "training": value}
                for key, value in expected_items
                if marker.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    "checkpoint action/time contract does not match requested training: "
                    + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
                )
        split, split_source = training_episode_split(
            payload,
            dataset_id,
            dataset_contract,
            model_contract=model_contract,
        )
        norm_path = norm_stats_path(dataset_id, arm_mode, model_variant)
        norm_ready = norm_split_matches(
            norm_path.parent,
            split,
            contract=model_contract["contract_fingerprint"],
        )
        saved_norm_config = (
            read_json(norm_path.parent / NORM_CONFIG_FILENAME) if norm_ready else None
        )
        norm_ready = norm_ready and norm_extended_contract_matches(
            saved_norm_config, model_contract["contract_fingerprint"]
        )
        if not norm_ready:
            saved_norm_config = None

        # A resume request must name a finalized checkpoint.  If the target
        # experiment only contains an interrupted Orbax temp directory, use an
        # explicitly requested checkpoint or the complete checkpoint recorded
        # when this dataset was normalized.  Never fall back to pi05_base for a
        # resume request.
        if effective_mode == "resume" and not saved_steps:
            candidate = requested_resume_checkpoint
            if candidate in (None, ""):
                raise ValueError(
                    f"experiment {exp_name!r} has no complete checkpoint; "
                    "resume_checkpoint is required. The interrupted Orbax temp save "
                    "cannot be used as a full-state source."
                )
            resume_checkpoint = resolve_complete_resume_checkpoint(candidate)
            validate_resume_checkpoint_variant(resume_checkpoint, model_variant)
            resume_kind = "external_full_state"
            # The base checkpoint argument is still required by the OpenPI
            # config schema, but it is only a structural fallback; the helper
            # restores train_state/params from resume_checkpoint before any
            # training step.
            base_checkpoint = resume_checkpoint
            effective_mode = "resume_external"
        elif requested_resume_checkpoint not in (None, ""):
            if effective_mode != "resume":
                raise ValueError("resume_checkpoint can only be used with mode=resume or mode=auto")
            resume_checkpoint = resolve_complete_resume_checkpoint(requested_resume_checkpoint)
            validate_resume_checkpoint_variant(resume_checkpoint, model_variant)
            resume_kind = "external_full_state"
            base_checkpoint = resume_checkpoint
            effective_mode = "resume_external"
        elif saved_steps:
            resume_checkpoint = target_complete_steps[-1][1]
            resume_kind = "in_place_full_state"

        # Resolve the external source before validating its action contract.
        # The old ordering checked ``resume_kind`` before the external branch
        # assigned it, so an explicitly supplied checkpoint could bypass the
        # schema/action-semantic compatibility check entirely.
        if resume_kind == "external_full_state" and resume_checkpoint is not None:
            validate_resume_checkpoint_contract(resume_checkpoint, model_contract)

        # An auto request against an interrupted directory is treated as the
        # same explicit external-resume flow above.  A genuinely new directory
        # remains a new run and may use the selected foundation/weight checkpoint.
        if mode == "auto" and target_has_artifacts and not saved_steps and resume_checkpoint is None:
            raise ValueError(
                f"experiment {exp_name!r} contains artifacts but no complete checkpoint; "
                "provide resume_checkpoint or remove/archive the interrupted run"
            )
        if resume_kind == "external_full_state" and target_has_artifacts and not saved_steps:
            raise ValueError(
                f"target experiment {exp_name!r} contains an incomplete checkpoint; "
                "use a new target experiment name for external full-state resume "
                "so the failed run remains available for audit"
            )

        execution_target = str(payload.get("execution_target", "local_4090") or "local_4090")
        cluster_target_config = runtime_config_for_target(execution_target)
        is_cluster_target = cluster_target_config is not None
        minimum_free_gpu_mib = int(config.get("training_min_free_gpu_mib", 23_000))
        allow_busy_raw = payload.get("allow_busy_gpus", config.get("allow_busy_gpus", False))
        allow_busy_gpus = (
            allow_busy_raw
            if isinstance(allow_busy_raw, bool)
            else str(allow_busy_raw).strip().lower() in {"1", "true", "yes", "on"}
        )
        if is_cluster_target:
            raw_cluster_gpus = payload.get("cluster_gpus", payload.get("gpu_count", payload.get("gpu_ids", 1)))
            if isinstance(raw_cluster_gpus, str) and "," in raw_cluster_gpus:
                cluster_gpu_count = len([item for item in raw_cluster_gpus.split(",") if item.strip()])
            else:
                cluster_gpu_count = safe_int(raw_cluster_gpus, "cluster_gpus", 1, 8)
            gpu_ids = list(range(cluster_gpu_count))
            cluster_target_config["gpu_count"] = cluster_gpu_count
        else:
            gpu_ids = parse_gpus(
                payload,
                check_busy=norm_ready,
                minimum_free_mib=minimum_free_gpu_mib,
                allow_busy=allow_busy_gpus,
            )
        batch_size = safe_int(payload.get("batch_size", 2), "batch_size", 1, 1024)
        if batch_size % len(gpu_ids):
            raise ValueError("batch_size must be divisible by the number of selected GPUs")
        fsdp_devices = safe_int(payload.get("fsdp_devices", 1), "fsdp_devices", 1, len(gpu_ids))
        if len(gpu_ids) % fsdp_devices:
            raise ValueError("selected GPU count must be divisible by fsdp_devices")
        num_workers = safe_int(payload.get("num_workers", 2), "num_workers", 1, 64)
        xla_memory_fraction = safe_float(
            payload.get("xla_memory_fraction", config.get("xla_memory_fraction", 0.90)),
            "xla_memory_fraction",
            0.50,
            0.95,
        )
        steps = safe_int(payload.get("num_train_steps", 30_000), "num_train_steps", 1, 10_000_000)
        save_interval = safe_int(payload.get("save_interval", 1_000), "save_interval", 1, steps)
        keep_period_raw = payload.get("keep_period", 5_000)
        keep_period = 0 if str(keep_period_raw).strip().lower() in {"", "none", "null", "0", "false", "no", "off"} else safe_int(keep_period_raw, "keep_period", 1, 10_000_000)
        eval_enabled_raw = payload.get("eval_enabled", True)
        eval_enabled = (
            eval_enabled_raw
            if isinstance(eval_enabled_raw, bool)
            else str(eval_enabled_raw).strip().lower() not in {"0", "false", "no", "off", ""}
        )
        eval_interval_steps = safe_int(
            payload.get("eval_interval_steps", 5_000),
            "eval_interval_steps",
            1,
            10_000_000,
        )
        eval_batch_size = safe_int(payload.get("eval_batch_size", 1), "eval_batch_size", 1, 64)
        eval_num_workers = safe_int(payload.get("eval_num_workers", 2), "eval_num_workers", 0, 16)
        eval_max_batches = safe_int(payload.get("eval_max_batches", 50), "eval_max_batches", 1, 100_000)
        eval_seed = safe_int(
            payload.get("eval_seed", split.seed),
            "eval_seed",
            0,
            2**31 - 1,
        )
        eval_xla_memory_fraction = safe_float(
            payload.get(
                "eval_xla_memory_fraction",
                config.get("evaluation_xla_memory_fraction", 0.85),
            ),
            "eval_xla_memory_fraction",
            0.50,
            0.95,
        )
        eval_disabled_reason = None
        if eval_enabled and not split.test_episodes:
            eval_enabled = False
            eval_disabled_reason = "test_split_is_empty"
        if eval_enabled and eval_interval_steps % save_interval:
            raise ValueError("eval_interval_steps must be divisible by save_interval")
        # Upstream Orbax currently retains 5000-step checkpoints. Restricting
        # asynchronous eval to durable checkpoints prevents a save/delete race.
        if eval_enabled and eval_interval_steps % 5_000:
            raise ValueError("eval_interval_steps must be a multiple of 5000")
        existing_complete_steps = complete_checkpoint_steps(checkpoint_dir)
        eval_after_step = (
            existing_complete_steps[-1][0]
            if effective_mode == "resume" and existing_complete_steps
            else 0
        )
        command = build_train_command(
            config,
            dataset_id,
            arm_mode,
            arm_side,
            schema,
            base_checkpoint=base_checkpoint,
            model_variant=model_variant,
            exp_name=exp_name,
            batch_size=batch_size,
            num_workers=num_workers,
            steps=steps,
            save_interval=save_interval,
            keep_period=keep_period,
            fsdp_devices=fsdp_devices,
            split=split,
            model_contract=model_contract,
            effective_mode=("resume" if resume_kind == "in_place_full_state" else effective_mode),
            resume_checkpoint=(resume_checkpoint if resume_kind == "external_full_state" else None),
            wandb_enabled=bool(payload.get("wandb_enabled", False)),
        )
        cluster_dataset_sync_command: list[str] | None = None
        remote_checkpoint_dir: str | None = None
        if is_cluster_target:
            auto_sync_raw = payload.get("auto_sync_dataset", config.get("auto_sync_cluster_dataset", True))
            auto_sync_dataset = (
                auto_sync_raw
                if isinstance(auto_sync_raw, bool)
                else str(auto_sync_raw).strip().lower() not in {"0", "false", "no", "off", ""}
            )
            if auto_sync_dataset:
                locations = dataset_location_configs()
                if execution_target not in locations:
                    raise ValueError(f"dataset sync target is not configured: {execution_target}")
                source_location = locations["local_4090"]
                target_location = locations[execution_target]
                target_rows, target_inventory_error = remote_dataset_inventory(target_location)
                target_has_dataset = any(str(row.get("id")) == dataset_id for row in target_rows)
                if not target_has_dataset:
                    if target_inventory_error:
                        app.logger.warning(
                            "cluster dataset inventory failed for %s before train; will attempt sync: %s",
                            execution_target,
                            target_inventory_error,
                        )
                    slurm_sync = target_location.get("kind") == "slurm_only"
                    sync_runner = "slurm_dataset_sync_runner.py" if slurm_sync else "dataset_transfer_runner.py"
                    cluster_dataset_sync_command = [
                        config["openpi_python"],
                        str(APP_DIR / sync_runner),
                        "--dataset-id", dataset_id,
                        "--source-json", json_arg(source_location),
                        "--target-json", json_arg(target_location),
                        "--parallelism", str(config.get("transfer_parallelism", 4)),
                        "--skip-existing",
                    ]
                    if slurm_sync and config.get("nas_dataset_staging_root"):
                        cluster_dataset_sync_command += ["--nas-staging-root", str(config["nas_dataset_staging_root"])]
            remote_checkpoint_dir = str(
                PurePosixPath(str(cluster_target_config["checkpoint_base_dir"]).rstrip("/"))
                / policy_config_name(arm_mode, model_variant)
                / exp_name
            )
            remote_base_checkpoint = translate_runtime_path(base_checkpoint, cluster_target_config)
            remote_norm_batch_size = safe_int(payload.get("norm_batch_size", 16), "norm_batch_size", 1, 1024)
            remote_norm_num_workers = safe_int(payload.get("norm_num_workers", 2), "norm_num_workers", 1, 64)
            remote_norm_command = build_norm_command(
                dataset_id,
                arm_mode,
                arm_side,
                schema,
                base_checkpoint=remote_base_checkpoint,
                model_variant=model_variant,
                batch_size=remote_norm_batch_size,
                num_workers=remote_norm_num_workers,
                split=split,
                model_contract=model_contract,
                runtime_config=cluster_target_config,
            )
            remote_train_command = build_train_command(
                cluster_target_config,
                dataset_id,
                arm_mode,
                arm_side,
                schema,
                base_checkpoint=remote_base_checkpoint,
                model_variant=model_variant,
                exp_name=exp_name,
                batch_size=batch_size,
                num_workers=num_workers,
                steps=steps,
                save_interval=save_interval,
                keep_period=keep_period,
                fsdp_devices=fsdp_devices,
                split=split,
                model_contract=model_contract,
                effective_mode=("resume" if resume_kind == "in_place_full_state" else effective_mode),
                resume_checkpoint=(
                    translate_runtime_path(resume_checkpoint, cluster_target_config)
                    if resume_kind == "external_full_state" and resume_checkpoint is not None
                    else None
                ),
                wandb_enabled=bool(payload.get("wandb_enabled", False)),
            )
            slurm_command = slurm_runner_command(
                target_name=execution_target,
                target_config={
                    **cluster_target_config,
                    "xla_memory_fraction": xla_memory_fraction,
                },
                commands=[remote_norm_command, remote_train_command],
                labels=["norm", "train"],
                job_name=f"{'sim' if config.get('dashboard_profile') == 'simulation' else 'real'}_train_{exp_name}",
            )
            if cluster_dataset_sync_command is not None:
                command = [
                    "bash",
                    "-lc",
                    "set -euo pipefail; "
                    + shlex.join([str(item) for item in cluster_dataset_sync_command])
                    + " && "
                    + shlex.join([str(item) for item in slurm_command]),
                ]
            else:
                command = slurm_command
        metadata = {
            "dataset_id": dataset_id,
            "arm_mode": arm_mode,
            "arm_side": arm_side,
            "schema": schema,
            **model_contract["contract_fingerprint"],
            "raw_gripper_semantics": model_contract["raw_gripper_semantics"],
            "wire_gripper_semantics": model_contract["wire_gripper_semantics"],
            "delivery_action_convention": (
                model_contract["model_action_convention"] if schema == "delivery" else None
            ),
            "model_variant": model_variant,
            "base_checkpoint": str(base_checkpoint),
            "exp_name": exp_name,
            "gpu_ids": gpu_ids,
            "execution_target": execution_target,
            "runtime": "slurm" if is_cluster_target else "local_4090",
            "cluster_target": execution_target if is_cluster_target else None,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "steps": steps,
            "save_interval": save_interval,
            "keep_period": keep_period,
            "fsdp_devices": fsdp_devices,
            "xla_memory_fraction": xla_memory_fraction,
            "minimum_free_gpu_mib": minimum_free_gpu_mib,
            "allow_busy_gpus": allow_busy_gpus,
            "mode": mode,
            "effective_mode": effective_mode,
            "resume_kind": resume_kind,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "resume_step": int(resume_checkpoint.name) if resume_checkpoint is not None and resume_checkpoint.name.isdigit() else None,
            "checkpoint_dir": str(checkpoint_dir),
            "remote_checkpoint_dir": remote_checkpoint_dir,
            "auto_sync_dataset": bool(cluster_dataset_sync_command is not None),
            "checkpoint_sync_payload": (
                {
                    "source": execution_target,
                    "target": "local_4090",
                    "config_name": policy_config_name(arm_mode, model_variant),
                    "exp_name": exp_name,
                    "overwrite": False,
                    "skip_existing": True,
                }
                if is_cluster_target
                else None
            ),
            "test_ratio": split.test_ratio,
            "split_seed": split.seed,
            "split_source": split_source,
            "train_episodes": len(split.train_episodes),
            "test_episodes": len(split.test_episodes),
            "test_episode_indexes": list(split.test_episodes),
            "norm_config": saved_norm_config if isinstance(saved_norm_config, dict) else None,
            "norm_batch_size": (
                saved_norm_config.get("effective_batch_size")
                if isinstance(saved_norm_config, dict)
                else None
            ),
            "eval_after_step": eval_after_step,
            "auto_eval": {
                "enabled": eval_enabled,
                "disabled_reason": eval_disabled_reason,
                "every_steps": eval_interval_steps,
                "batch_size": eval_batch_size,
                "num_workers": eval_num_workers,
                "max_batches": eval_max_batches,
                "seed": eval_seed,
                "minimum_free_gpu_mib": int(
                    config.get("evaluation_min_free_gpu_mib", 23_000)
                ),
                "xla_memory_fraction": eval_xla_memory_fraction,
                "split": "test",
            },
        }
        with tasks.lock:
            if is_cluster_target:
                task = tasks.start(
                    "train",
                    command,
                    env=build_environment(config, None),
                    metadata={
                        **metadata,
                        "slurm_target": execution_target,
                        "slurm_submit_host": cluster_target_config.get("submit_host"),
                        "slurm_partition": cluster_target_config.get("partition"),
                        "slurm_node": cluster_target_config.get("node"),
                        "remote_dataset_root": cluster_target_config.get("dataset_root"),
                        "remote_checkpoint_base_dir": cluster_target_config.get("checkpoint_base_dir"),
                    },
                )
                return jsonify(task), 201

            if norm_ready:
                gpu_ids = parse_gpus(
                    payload,
                    minimum_free_mib=minimum_free_gpu_mib,
                    allow_busy=allow_busy_gpus,
                )
                metadata["gpu_ids"] = gpu_ids
                task = tasks.start(
                    "train",
                    command,
                    env=build_environment(
                        config,
                        gpu_ids,
                        xla_memory_fraction=xla_memory_fraction,
                    ),
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
                    and item.get("metadata", {}).get("arm_mode") == arm_mode
                    and item.get("metadata", {}).get("arm_side") == arm_side
                    and item.get("metadata", {}).get("schema") == schema
                    and item.get("metadata", {}).get("contract_version")
                    == model_contract["contract_version"]
                    and item.get("metadata", {}).get("raw_action_dim")
                    == model_contract["raw_action_dim"]
                    and item.get("metadata", {}).get("model_action_dim")
                    == model_contract["model_action_dim"]
                    and item.get("metadata", {}).get("model_action_convention")
                    == model_contract["model_action_convention"]
                    and item.get("metadata", {}).get("gripper_semantics")
                    == model_contract["model_gripper_semantics"]
                    and item.get("metadata", {}).get("model_variant") == model_variant
                    and item.get("metadata", {}).get("base_checkpoint") == str(base_checkpoint)
                    and float(item.get("metadata", {}).get("test_ratio", -1)) == split.test_ratio
                    and int(item.get("metadata", {}).get("split_seed", -1)) == split.seed
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
                        arm_mode,
                        arm_side,
                        schema,
                        base_checkpoint=base_checkpoint,
                        model_variant=model_variant,
                        batch_size=norm_batch_size,
                        num_workers=norm_num_workers,
                        split=split,
                        model_contract=model_contract,
                    ),
                    env=build_environment(config, None),
                    metadata={
                        "dataset_id": dataset_id,
                        "arm_mode": arm_mode,
                        "arm_side": arm_side,
                        "schema": schema,
                        **model_contract["contract_fingerprint"],
                        "raw_gripper_semantics": model_contract["raw_gripper_semantics"],
                        "wire_gripper_semantics": model_contract["wire_gripper_semantics"],
                        "delivery_action_convention": (
                            model_contract["model_action_convention"] if schema == "delivery" else None
                        ),
                        "model_variant": model_variant,
                        "base_checkpoint": str(base_checkpoint),
                        "batch_size": norm_batch_size,
                        "num_workers": norm_num_workers,
                        "max_frames": None,
                        "test_ratio": split.test_ratio,
                        "split_seed": split.seed,
                        "split_source": split_source,
                        "train_episodes": len(split.train_episodes),
                        "test_episodes": len(split.test_episodes),
                        "test_episode_indexes": list(split.test_episodes),
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
    @app.post("/api/tasks/eval")
    def start_eval():
        payload = request.get_json(force=True)
        dataset_id, arm_mode, arm_side, schema, dataset_contract = parse_dataset(payload)
        base_checkpoint, model_variant = resolve_base_model(payload)
        model_contract = action_contract_for_model(dataset_contract)
        execution_target = str(payload.get("execution_target", "local_4090") or "local_4090")
        cluster_target_config = runtime_config_for_target(execution_target)
        is_cluster_target = cluster_target_config is not None
        checkpoint_raw = payload.get("checkpoint")
        checkpoint = resolve_under(checkpoint_raw, checkpoint_roots)
        if not (checkpoint / "params").exists():
            raise ValueError("checkpoint does not contain params")
        batch_size = safe_int(payload.get("batch_size", 1), "batch_size", 1, 64)
        num_workers = safe_int(payload.get("num_workers", 2), "num_workers", 0, 16)
        max_batches = safe_int(payload.get("max_batches", 50), "max_batches", 1, 100_000)
        split, _ = training_episode_split(payload, dataset_id, dataset_contract, model_contract=model_contract)
        eval_seed = safe_int(payload.get("eval_seed", split.seed), "eval_seed", 0, 2**31 - 1)
        xla_memory_fraction = safe_float(
            payload.get("xla_memory_fraction", config.get("evaluation_xla_memory_fraction", 0.85)),
            "xla_memory_fraction",
            0.50,
            0.95,
        )
        result_path = tasks.root / f"manual-eval-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}" / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        gpu_ids: list[int]
        if is_cluster_target:
            raw_cluster_gpus = payload.get("cluster_gpus", payload.get("gpu_count", 1))
            cluster_gpu_count = safe_int(raw_cluster_gpus, "cluster_gpus", 1, 8)
            cluster_target_config["gpu_count"] = cluster_gpu_count
            gpu_ids = list(range(cluster_gpu_count))
            remote_checkpoint = translate_runtime_path(checkpoint, cluster_target_config)
            remote_base_checkpoint = translate_runtime_path(base_checkpoint, cluster_target_config)
            remote_result_path = str(PurePosixPath(str(cluster_target_config.get("remote_job_dir"))) / f"eval_{uuid.uuid4().hex[:8]}_result.json")
            remote_eval_command = build_eval_command(
                cluster_target_config,
                checkpoint=remote_checkpoint,
                result_path=remote_result_path,
                dataset_id=dataset_id,
                arm_mode=arm_mode,
                arm_side=arm_side,
                schema=schema,
                model_variant=model_variant,
                base_checkpoint=remote_base_checkpoint,
                batch_size=batch_size,
                num_workers=num_workers,
                max_batches=max_batches,
                eval_seed=eval_seed,
                model_contract=model_contract,
            )
            command = slurm_runner_command(
                target_name=execution_target,
                target_config={**cluster_target_config, "xla_memory_fraction": xla_memory_fraction},
                commands=[remote_eval_command],
                labels=["eval"],
                job_name=f"sim_eval_{dataset_id}",
            )
            env = build_environment(config, None)
        else:
            gpu_ids = parse_gpus(
                payload,
                one_only=True,
                minimum_free_mib=int(config.get("evaluation_min_free_gpu_mib", 23_000)),
            )
            command = build_eval_command(
                config,
                checkpoint=checkpoint,
                result_path=result_path,
                dataset_id=dataset_id,
                arm_mode=arm_mode,
                arm_side=arm_side,
                schema=schema,
                model_variant=model_variant,
                base_checkpoint=base_checkpoint,
                batch_size=batch_size,
                num_workers=num_workers,
                max_batches=max_batches,
                eval_seed=eval_seed,
                model_contract=model_contract,
            )
            env = build_environment(config, gpu_ids, xla_memory_fraction=xla_memory_fraction)
        task = tasks.start(
            "eval",
            command,
            env=env,
            metadata={
                "dataset_id": dataset_id,
                "arm_mode": arm_mode,
                "arm_side": arm_side,
                "schema": schema,
                **model_contract["contract_fingerprint"],
                "raw_gripper_semantics": model_contract["raw_gripper_semantics"],
                "wire_gripper_semantics": model_contract["wire_gripper_semantics"],
                "model_variant": model_variant,
                "base_checkpoint": str(base_checkpoint),
                "checkpoint": str(checkpoint),
                "gpu_ids": gpu_ids,
                "execution_target": execution_target,
                "runtime": "slurm" if is_cluster_target else "local_4090",
                "cluster_target": execution_target if is_cluster_target else None,
                "result_path": str(result_path),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "max_batches": max_batches,
                "eval_seed": eval_seed,
                "xla_memory_fraction": xla_memory_fraction,
                "manual": True,
            },
        )
        return jsonify(task), 201

    @app.post("/api/tasks/policy")
    def start_policy():
        payload = request.get_json(force=True)
        raw_rtc_enabled = payload.get("rtc_enabled", True)
        if isinstance(raw_rtc_enabled, bool):
            rtc_enabled = raw_rtc_enabled
        elif isinstance(raw_rtc_enabled, (int, float)) and raw_rtc_enabled in (0, 1):
            rtc_enabled = bool(raw_rtc_enabled)
        elif isinstance(raw_rtc_enabled, str) and raw_rtc_enabled.strip().lower() in {
            "true", "1", "yes", "on",
        }:
            rtc_enabled = True
        elif isinstance(raw_rtc_enabled, str) and raw_rtc_enabled.strip().lower() in {
            "false", "0", "no", "off",
        }:
            rtc_enabled = False
        else:
            raise ValueError("rtc_enabled must be a boolean")
        rtc_execution_horizon = safe_int(
            payload.get("rtc_execution_horizon", 8),
            "rtc_execution_horizon",
            1,
            50,
        )
        rtc_max_guidance_weight = safe_float(
            payload.get("rtc_max_guidance_weight", 5.0),
            "rtc_max_guidance_weight",
            0.001,
            100.0,
        )
        rtc_prefix_attention_schedule = str(
            payload.get("rtc_prefix_attention_schedule", "linear") or "linear"
        ).strip().lower()
        if rtc_prefix_attention_schedule not in {"zeros", "ones", "linear", "exp"}:
            raise ValueError(
                "rtc_prefix_attention_schedule must be one of zeros, ones, linear, exp"
            )
        policy_target = str(payload.get("execution_target", "local_4090") or "local_4090")
        if policy_target not in {"", "local", "local_4090", "4x4090"}:
            raise ValueError("Policy serving is only supported on the 4×4090 host; train/sync checkpoints back before serving")
        dataset_id, arm_mode, arm_side, schema, dataset_contract = parse_dataset(payload)
        port = safe_int(payload.get("port", 8000), "port", config["policy_port_min"], config["policy_port_max"])
        checkpoint = resolve_under(payload.get("checkpoint", ""), checkpoint_roots)
        if not (checkpoint / "params").exists():
            raise ValueError(f"checkpoint has no params directory: {checkpoint}")
        inferred_variant = infer_model_variant(checkpoint)
        requested_variant = str(payload.get("model_variant") or inferred_variant or "pi05")
        if requested_variant not in MODEL_VARIANTS:
            raise ValueError(f"model_variant must be one of {sorted(MODEL_VARIANTS)}")
        if inferred_variant is not None and inferred_variant != requested_variant:
            raise ValueError(
                f"checkpoint {checkpoint} appears to be {inferred_variant}, "
                f"but model_variant={requested_variant}"
            )
        model_variant = requested_variant
        expected_config = policy_config_name(arm_mode, model_variant)
        if expected_config not in checkpoint.parts:
            raise ValueError(
                f"checkpoint is not a {model_variant}/{arm_mode} checkpoint: "
                f"expected path component {expected_config!r}"
            )
        checkpoint_norm = checkpoint / "assets" / dataset_id / "norm_stats.json"
        if not checkpoint_norm.exists():
            raise ValueError(
                f"checkpoint is not associated with dataset {dataset_id}: missing {checkpoint_norm}"
            )
        marker = checkpoint_action_contract(checkpoint)
        if marker is not None:
            model_convention = marker.get("model_action_convention") or marker.get(
                "delivery_action_convention"
            )
            model_gripper = marker.get("gripper_semantics") or marker.get(
                "model_gripper_semantics"
            )
        elif schema == "delivery" and dataset_contract.get("legacy_delivery_v2"):
            model_convention = DELIVERY_STEP_ACTION_CONVENTION
            model_gripper = None
        elif schema == "joint" and dataset_contract.get("legacy_joint_v2"):
            model_convention = None
            model_gripper = LEGACY_JOINT_GRIPPER_SEMANTICS
        else:
            raise ValueError(
                "checkpoint has no complete action-contract marker and is not a verified "
                "legacy-v2 checkpoint"
            )
        model_contract = action_contract_for_model(
            dataset_contract,
            delivery_action_convention=model_convention,
            model_gripper_semantics=model_gripper,
        )
        if marker is not None:
            marker_version = int(marker.get("version", 1))
            legacy_temporal_compat = bool(
                marker_version < ACTION_CONTRACT_MARKER_VERSION
                and (
                    (
                        dataset_contract.get("legacy_delivery_v2")
                        and marker.get("model_action_convention", marker.get("delivery_action_convention"))
                        == model_contract.get("model_action_convention")
                    )
                    or (
                        dataset_contract.get("legacy_joint_v2")
                        and (marker.get("gripper_semantics") or marker.get("model_gripper_semantics"))
                        == model_contract.get("model_gripper_semantics")
                    )
                )
            )
            if marker_version < ACTION_CONTRACT_MARKER_VERSION and not legacy_temporal_compat:
                raise ValueError(
                    "checkpoint action marker predates action_offset/model_action_start_offset; "
                    "retrain or explicitly migrate the verified checkpoint contract"
                )
            fingerprint_items = (
                normalize_contract_fingerprint(model_contract["contract_fingerprint"]).items()
                if legacy_temporal_compat
                else model_contract["contract_fingerprint"].items()
            )
            mismatches = {
                key: {"checkpoint": marker.get(key), "dataset": value}
                for key, value in fingerprint_items
                if marker.get(key) != value
            }
            if marker.get("dataset_id") not in (None, dataset_id):
                mismatches["dataset_id"] = {
                    "checkpoint": marker.get("dataset_id"),
                    "dataset": dataset_id,
                }
            if mismatches:
                raise ValueError(
                    "checkpoint action contract does not match selected dataset: "
                    + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
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
                old_root_pid = int(old_task["pid"])
                ignored_pids.add(old_root_pid)
                ignored_pids.update(
                    process_descendant_pids(old_root_pid, process_children_by_parent())
                )

        policy_allow_busy_gpus = bool(config.get("policy_allow_busy_gpus", True))
        policy_min_free_gpu_mib = safe_int(
            config.get("policy_min_free_gpu_mib", 12_000),
            "policy_min_free_gpu_mib",
            0,
            1_000_000,
        )
        policy_xla_memory_fraction = safe_float(
            config.get("policy_xla_memory_fraction", 0.60),
            "policy_xla_memory_fraction",
            0.25,
            0.90,
        )
        policy_xla_preallocate = bool(config.get("policy_xla_preallocate", False))

        # Validate the target resources before disrupting a working Policy. The
        # process being replaced may legitimately own the requested GPU/port;
        # its memory is treated as reclaimable during this first check.
        gpu_ids = parse_gpus(
            payload,
            one_only=True,
            ignored_pids=ignored_pids,
            minimum_free_mib=policy_min_free_gpu_mib,
            allow_busy=policy_allow_busy_gpus,
        )
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

        # Recheck after shutdown. The process state can become terminal slightly
        # before the kernel releases its listening socket, so wait briefly for
        # the exact port instead of failing a valid model switch.
        port_deadline = time.monotonic() + 5.0
        while True:
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                if sock.connect_ex(("127.0.0.1", port)) != 0:
                    break
            if time.monotonic() >= port_deadline:
                raise ValueError(f"port {port} is still in use after stopping the previous policy")
            time.sleep(0.1)

        # Recheck after replacement shutdown so a process that raced onto the
        # selected GPU cannot bypass the free-memory guard.
        gpu_ids = parse_gpus(
            payload,
            one_only=True,
            minimum_free_mib=policy_min_free_gpu_mib,
            allow_busy=policy_allow_busy_gpus,
        )

        telemetry_session, telemetry_dir = observations.create_session()
        command = [
            config["openpi_python"], openpi_helper, "serve",
            "--dataset-id", dataset_id,
            "--arm-mode", arm_mode,
            "--arm-side", arm_side,
            "--schema", schema,
            "--model-variant", model_variant,
            "--assets-base-dir", config["assets_base_dir"],
            "--checkpoint-base-dir", config["checkpoint_base_dir"],
            "--base-checkpoint", config["base_checkpoint"],
            "--checkpoint", str(checkpoint),
            "--port", str(port),
            "--telemetry-dir", str(telemetry_dir),
            "--rtc-enabled" if rtc_enabled else "--no-rtc-enabled",
            "--rtc-execution-horizon", str(rtc_execution_horizon),
            "--rtc-max-guidance-weight", str(rtc_max_guidance_weight),
            "--rtc-prefix-attention-schedule", rtc_prefix_attention_schedule,
        ] + action_contract_command_args(model_contract)
        default_prompt = str(payload.get("default_prompt", "")).strip()
        if default_prompt:
            if len(default_prompt) > 500:
                raise ValueError("default_prompt is too long")
            command += ["--default-prompt", default_prompt]
        task = tasks.start(
            "policy", command,
            env=build_environment(
                config,
                gpu_ids,
                xla_memory_fraction=policy_xla_memory_fraction,
                xla_preallocate=policy_xla_preallocate,
            ),
            metadata={
                "dataset_id": dataset_id,
                "arm_mode": arm_mode,
                "arm_side": arm_side,
                "schema": schema,
                **model_contract["contract_fingerprint"],
                "raw_gripper_semantics": model_contract["raw_gripper_semantics"],
                "wire_gripper_semantics": model_contract["wire_gripper_semantics"],
                "delivery_action_convention": (
                    model_contract["model_action_convention"] if schema == "delivery" else None
                ),
                "model_variant": model_variant,
                "checkpoint": str(checkpoint),
                "gpu_ids": gpu_ids,
                "allow_busy_gpus": policy_allow_busy_gpus,
                "minimum_free_gpu_mib": policy_min_free_gpu_mib,
                "xla_memory_fraction": policy_xla_memory_fraction,
                "xla_preallocate": policy_xla_preallocate,
                "port": port,
                "ws_url": f"ws://{request.host.split(':')[0]}:{port}",
                "telemetry_session": telemetry_session,
                "telemetry_dir": str(telemetry_dir),
                "rtc_enabled": rtc_enabled,
                "rtc_execution_horizon": rtc_execution_horizon,
                "rtc_max_guidance_weight": rtc_max_guidance_weight,
                "rtc_prefix_attention_schedule": rtc_prefix_attention_schedule,
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
            require_policy_execution_horizon(telemetry)
            require_policy_execution_time_contract(telemetry)
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

    @app.delete("/api/tasks/<task_id>")
    def delete_task(task_id: str):
        return jsonify(tasks.delete(task_id))

    @app.post("/api/tasks/batch-delete")
    def delete_tasks_batch():
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return jsonify(tasks.delete_many(payload.get("task_ids")))


    transfer_progress_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def transfer_progress_for_task(task: dict[str, Any]) -> dict[str, Any] | None:
        """Return runner progress, with a conservative legacy-task fallback.

        New transfer runners publish ``progress.json`` and TaskManager attaches
        it to the task.  A transfer created by an older Dashboard build has no
        sidecar, so while it is still running we derive a read-only estimate
        from the durable transfer log and the target manifest.  The fallback
        is throttled to one scan per task every four seconds and never affects
        the transfer process itself.
        """
        if task.get("type") != "transfer":
            return None
        existing = task.get("progress")
        if isinstance(existing, dict):
            return existing
        task_id = str(task.get("id") or "")
        if not task_id:
            return None
        now = time.time()
        cached = transfer_progress_cache.get(task_id)
        if cached and now - cached[0] < 4.0:
            return cached[1]
        metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
        total_bytes = 0
        total_files = 0
        total_shards = 0
        try:
            log = tasks.log_tail(task_id, 256 * 1024)
            match = re.search(r"files=(\d+)\s+bytes=(\d+)\s+shards=(\d+)", log)
            if match:
                total_files, total_bytes, total_shards = (int(item) for item in match.groups())
        except Exception:
            log = ""
        target_path = str(metadata.get("target_path") or "")
        target_name = str(metadata.get("target") or "")
        target_cfg = dataset_location_configs().get(target_name, {})
        completed_bytes = 0
        completed_files = 0
        scan_error = None
        if target_path:
            command = f"test -d {shlex.quote(target_path)} && cd {shlex.quote(target_path)} && find . -type f -printf '%s\\t%P\\0' | sort -z"
            host = target_cfg.get("host") or target_cfg.get("submit_host")
            try:
                if host:
                    result = subprocess.run(
                        [*SSH_COMMAND, str(host), command],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=20,
                    )
                else:
                    result = subprocess.run(
                        ["bash", "-lc", command],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=20,
                    )
                if result.returncode != 0:
                    raise RuntimeError((result.stderr or result.stdout).decode(errors="replace")[-1000:])
                rows = 0
                for entry in result.stdout.split(b"\\0"):
                    if not entry:
                        continue
                    size_text, sep, rel_bytes = entry.partition(b"\\t")
                    if not sep:
                        continue
                    rel = rel_bytes.decode(errors="replace")
                    if any(part.startswith(".") for part in rel.split("/")):
                        continue
                    try:
                        completed_bytes += max(0, int(size_text.decode()))
                        rows += 1
                    except ValueError:
                        continue
                completed_files = rows
            except Exception as exc:
                scan_error = str(exc)[-1000:]
        previous = cached[1] if cached else None
        previous_bytes = int(previous.get("completed_bytes", 0)) if previous else completed_bytes
        previous_at = float(previous.get("_sampled_at", now)) if previous else now
        elapsed = max(0.001, now - previous_at)
        speed = max(0.0, (completed_bytes - previous_bytes) / elapsed)
        if previous:
            old_speed = float(previous.get("speed_bytes_per_sec", 0) or 0)
            speed = speed if not old_speed else (0.7 * old_speed + 0.3 * speed)
        remaining = max(0, total_bytes - completed_bytes)
        progress = (completed_bytes / total_bytes) if total_bytes else None
        payload: dict[str, Any] = {
            "state": str(task.get("state", "unknown")),
            "progress": None if progress is None else round(min(1.0, progress), 6),
            "completed_bytes": completed_bytes,
            "total_bytes": total_bytes,
            "completed_files": completed_files,
            "total_files": total_files,
            "completed_shards": 0,
            "total_shards": total_shards,
            "parallelism": int(metadata.get("parallelism", 1) or 1),
            "speed_bytes_per_sec": round(speed, 2),
            "eta_seconds": round(remaining / speed, 1) if speed > 0 and remaining else (0.0 if remaining == 0 and total_bytes else None),
            "dataset_id": metadata.get("dataset_id") or metadata.get("exp_name"),
            "source": metadata.get("source"),
            "target": target_name,
            "source_path": metadata.get("source_path"),
            "target_path": target_path,
            "updated_at": now_iso(),
            "_sampled_at": now,
        }
        if scan_error:
            payload["scan_error"] = scan_error
        transfer_progress_cache[task_id] = (now, payload)
        return payload

    def task_status_summary(task: dict[str, Any], *, include_metrics: bool = False) -> dict[str, Any]:
        metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
        state = str(task.get("state", "unknown"))
        summary = {
            "id": task.get("id"),
            "type": task.get("type"),
            "state": state,
            "active": state in PROCESS_STATES or state in WAITING_STATES,
            "terminal": state in TERMINAL_STATES,
            "pid": task.get("pid"),
            "created_at": task.get("created_at"),
            "queued_at": task.get("queued_at"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
            "returncode": task.get("returncode"),
            "waiting_reason": task.get("waiting_reason"),
            "skip_reason": task.get("skip_reason"),
            "start_error": task.get("start_error"),
            "lost_reason": task.get("lost_reason"),
            "dependency": task.get("dependency"),
            "dependency_state": task.get("dependency_state"),
            "progress": transfer_progress_for_task(task),
            "result": task.get("result") if task.get("type") == "eval" else None,
            "metadata": {
                key: metadata.get(key)
                for key in (
                    "dataset_id",
                    "exp_name",
                    "model_variant",
                    "arm_mode",
                    "arm_side",
                    "schema",
                    "execution_target",
                    "runtime",
                    "cluster_target",
                    "slurm_target",
                    "slurm_submit_host",
                    "slurm_partition",
                    "slurm_node",
                    "gpu_ids",
                    "batch_size",
                    "fsdp_devices",
                    "steps",
                    "save_interval",
                    "keep_period",
                    "checkpoint",
                    "checkpoint_step",
                    "checkpoint_dir",
                    "base_checkpoint",
                    "result_path",
                    "source",
                    "external",
                    "external_log_path",
                    "task_name",
                    "task_config",
                    "train_config_name",
                    "model_name",
                    "checkpoint_id",
                    "ckpt_setting",
                    "policy_name",
                    "instruction_type",
                    "pi0_step",
                    "target",
                    "source_path",
                    "target_path",
                    "overwrite",
                    "skip_existing",
                    "parallelism",
                    "transfer_kind",
                    "test_ratio",
                    "split_seed",
                    "train_episodes",
                    "test_episodes",
                )
                if key in metadata
            },
        }
        if task.get("type") == "train":
            probe = tasks.training_metrics_probe(task)
            if probe is not None:
                summary["training_metrics"] = probe
        if include_metrics and task.get("type") == "train":
            metrics = parse_training_metrics(
                tasks.log_tail(str(task["id"]), 16 * 1024 * 1024, include_remote_slurm=True),
                max_points=1200,
            )
            planned_steps = metadata.get("steps")
            latest_step = int(metrics["points"][-1]["step"]) if metrics.get("points") else 0
            summary["metrics"] = {
                "summary": metrics.get("summary", {}),
                "latest_step": latest_step,
                "planned_steps": planned_steps,
                "progress": (
                    min(1.0, latest_step / int(planned_steps))
                    if planned_steps and int(planned_steps) > 0
                    else None
                ),
            }
        return summary

    @app.get("/api/tasks")
    def list_task_statuses():
        task_type = request.args.get("type")
        state = request.args.get("state")
        dataset_id = request.args.get("dataset_id")
        exp_name = request.args.get("exp_name")
        active_only = str(request.args.get("active", "")).lower() in {"1", "true", "yes"}
        terminal_only = str(request.args.get("terminal", "")).lower() in {"1", "true", "yes"}
        include_metrics = str(request.args.get("include_metrics", "")).lower() in {"1", "true", "yes"}
        limit = safe_int(request.args.get("limit", 200), "limit", 1, 1000)
        tasks.discover_external_policies()
        tasks.discover_external_evals()
        task_list = tasks.list()
        filtered = []
        for task in task_list:
            metadata = task.get("metadata", {}) if isinstance(task.get("metadata"), dict) else {}
            if task_type and task.get("type") != task_type:
                continue
            if state and task.get("state") != state:
                continue
            if dataset_id and metadata.get("dataset_id") != dataset_id:
                continue
            if exp_name and metadata.get("exp_name") != exp_name:
                continue
            is_active = task.get("state") in PROCESS_STATES or task.get("state") in WAITING_STATES
            is_terminal = task.get("state") in TERMINAL_STATES
            if active_only and not is_active:
                continue
            if terminal_only and not is_terminal:
                continue
            filtered.append(task)
        filtered = filtered[:limit]
        counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        for task in filtered:
            counts[str(task.get("type", "unknown"))] = counts.get(str(task.get("type", "unknown")), 0) + 1
            state_counts[str(task.get("state", "unknown"))] = state_counts.get(str(task.get("state", "unknown")), 0) + 1
        return jsonify({
            "tasks": [task_status_summary(task, include_metrics=include_metrics) for task in filtered],
            "count": len(filtered),
            "counts_by_type": counts,
            "counts_by_state": state_counts,
            "filters": {
                "type": task_type,
                "state": state,
                "dataset_id": dataset_id,
                "exp_name": exp_name,
                "active": active_only,
                "terminal": terminal_only,
                "limit": limit,
                "include_metrics": include_metrics,
            },
        })

    @app.get("/api/tasks/<task_id>/status")
    def get_task_status(task_id: str):
        include_metrics = str(request.args.get("include_metrics", "")).lower() in {"1", "true", "yes"}
        return jsonify({"task": task_status_summary(tasks.get(task_id), include_metrics=include_metrics)})

    @app.get("/api/tasks/<task_id>")
    def get_task(task_id: str):
        return jsonify(tasks.get(task_id))

    @app.get("/api/tasks/<task_id>/log")
    def task_log(task_id: str):
        max_bytes = safe_int(request.args.get("max_bytes", 64 * 1024), "max_bytes", 1024, 1024 * 1024)
        task = tasks.get(task_id)
        return jsonify({"task": task, "log": tasks.log_tail(task_id, max_bytes, include_remote_slurm=True)})

    @app.get("/api/tasks/<task_id>/metrics")
    def task_metrics(task_id: str):
        task = tasks.get(task_id)
        if task.get("type") != "train":
            raise ValueError("metrics are only available for train tasks")
        max_points = safe_int(request.args.get("max_points", 1200), "max_points", 50, 5000)
        result = parse_training_metrics(
            tasks.log_tail(task_id, 16 * 1024 * 1024, include_remote_slurm=True),
            max_points=max_points,
        )
        eval_tasks = [
            item
            for item in tasks.list()
            if item.get("type") == "eval"
            and item.get("metadata", {}).get("parent_train_task_id") == task_id
        ]
        result = merge_eval_metrics(result, eval_tasks)
        planned_steps = task.get("metadata", {}).get("steps")
        latest_step = int(result["points"][-1]["step"]) if result["points"] else 0
        result.update(
            {
                "task": task,
                "planned_steps": planned_steps,
                "latest_step": latest_step,
                "progress": (
                    min(1.0, latest_step / int(planned_steps))
                    if planned_steps and int(planned_steps) > 0
                    else None
                ),
            }
        )
        return jsonify(result)


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
