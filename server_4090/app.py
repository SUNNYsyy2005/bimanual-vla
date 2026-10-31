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
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
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
    from bimanual_vla.data import action_conventions as _piper_action_conventions
except ImportError:
    _piper_action_conventions = None

from bimanual_vla.data.arm_geometry import normalize_arm_base_offset, normalize_robot_type
from bimanual_vla.training.settings import FinetuningSettings


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
    from .rlinf_dashboard import build_launch_spec, checkpoints_for_run, evaluation_profile, metrics_for_run, native_gpu_placement, native_log_for_run, public_launch_config, public_runs, scan_runs, upgrade_native_profiles
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
    from rlinf_dashboard import build_launch_spec, checkpoints_for_run, evaluation_profile, metrics_for_run, native_gpu_placement, native_log_for_run, public_launch_config, public_runs, scan_runs, upgrade_native_profiles


try:
    from . import eval_video_catalog as video_catalog
    from . import experiment_artifacts
except ImportError:
    import eval_video_catalog as video_catalog
    import experiment_artifacts




APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TASK_TYPES = {"norm", "train", "eval", "policy", "transfer", "dataset_alignment"}
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


def artifact_checkpoint_identity(path: Path) -> dict[str, Any] | None:
    """Read SFT identity after physical layout migration, independent of names."""
    path = Path(path)
    step_match=re.fullmatch(r"(?:step[-_])?(\d+)",path.name)
    if not step_match:
        return None
    for ancestor in path.parents:
        run = read_json(ancestor / "run.json", {}) or {}
        if run.get("schema") != "experiment_run.v1" or run.get("backend") != "openpi":
            continue
        origin = run.get("origin", {})
        config_name = origin.get("config_name")
        if not config_name and origin.get("legacy_path"):
            config_name = Path(origin["legacy_path"]).parent.name
        for model_variant in sorted(MODEL_VARIANTS):
            for arm_mode in ("single", "bimanual"):
                if config_name == policy_config_name(arm_mode, model_variant):
                    return {"config_name": config_name, "experiment": run["experiment"],
                            "checkpoint_step": int(step_match[1]), "model_variant": model_variant,
                            "arm_mode": arm_mode}
    return None


def infer_model_variant(path: Path) -> str | None:
    path = Path(path)
    registered = artifact_checkpoint_identity(path)
    if registered:
        return registered["model_variant"]

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
    registered = artifact_checkpoint_identity(path)
    if registered:
        return registered
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
    normalized_robot_type = normalize_robot_type(robot_type)
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
        or normalized_robot_type in {"aloha-agilex", "arx-x5", "franka-panda"}
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
    try:
        normalized_arm_base_offset = normalize_arm_base_offset(info.get("arm_base_offset"))
    except ValueError:
        normalized_arm_base_offset = None
    from server_4090.model_identity import normalize_robot
    return {
        "robot_type": normalize_robot(metadata.get("robot_model") or metadata.get("robot_type")),
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
        "arm_base_offset": info.get("arm_base_offset"),
        "arm_base_offset_m": list(normalized_arm_base_offset) if normalized_arm_base_offset is not None else None,
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
    for ancestor in (path, *path.parents):
        marker = ancestor / "metadata" / "action-contract.json"
        if (ancestor / "run.json").is_file() and marker.is_file():
            return marker
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


def lexical_absolute_path(value: str | Path) -> Path:
    """Normalize a path without resolving directory symlinks.

    Dashboard checkpoint layouts may intentionally use symlinks to move large
    model trees between disks.  The directory browser must keep the logical
    path so users can select it even when its resolved target is elsewhere.
    """
    return Path(os.path.abspath(os.path.expanduser(str(value))))


def lexical_path_is_within(path: Path, root: Path) -> bool:
    path = lexical_absolute_path(path)
    root = lexical_absolute_path(root)
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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


def expand_config_path(value: Any) -> str:
    """Expand home and environment references in configurable filesystem paths."""
    return os.path.expandvars(os.path.expanduser(str(value)))


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
        "training_min_free_gpu_mib": 22_500,
        "evaluation_min_free_gpu_mib": 23_000,
        "evaluation_xla_memory_fraction": 0.85,
        # Policy inference is much lighter than training. Keep training/eval
        # exclusive, but allow Policy to share with small stable workloads.
        "policy_allow_busy_gpus": True,
        "policy_min_free_gpu_mib": 12_000,
        "policy_xla_memory_fraction": 0.60,
        "policy_xla_preallocate": False,
        # Server-owned RTC capability ceiling. The actual runtime values are
        # selected by each robot client and can change without restarting the
        # Policy process.
        "policy_rtc_enabled": True,
        "policy_inference_diagnostics_enabled": True,
        "policy_rtc_max_execution_horizon": 12,
        "policy_rtc_max_guidance_weight": 5.0,
        "max_upload_gib": 500,
        "max_chunk_mib": 64,
        "policy_port_min": 8000,
        "policy_port_max": 8099,
        "robot_observation_max_age_s": 3.0,
        "task_monitor_interval_s": 2.0,
        # Keep long-running training outside the Dashboard's terminal/session
        # lifecycle. Norm/eval/policy retain the existing auto backend.
        "training_task_launch_backend": "nohup",
        "dashboard_profile": "real",
        "dashboard_title": "Bimanual-VLA · 4×4090 控制台",
        "upload_default_origin": None,
        "visible_dataset_origins": None,
        "enable_policy": True,
        "cluster_targets": {},
        "local_storage_locations": {},
        "cache_root": str(Path.home() / ".cache"),
        "dataset_read_roots": [],
        "eval_video_roots": [],
        "cluster_resources_script": str(REPO_DIR / "scripts" / "query_h100_h200_resources.sh"),
        "transfer_parallelism": 4,
        "auto_sync_cluster_dataset": True,
        "nas_dataset_staging_root": str(Path.home() / "nas" / "dashboard_dataset_sync"),
        "nas_checkpoint_staging_root": str(Path.home() / "nas" / "dashboard_checkpoint_sync"),
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
        defaults[key] = str(Path(expand_config_path(defaults[key])).resolve())
    raw_dataset_read_roots = defaults.get("dataset_read_roots", [])
    if isinstance(raw_dataset_read_roots, str):
        raw_dataset_read_roots = [raw_dataset_read_roots]
    dataset_read_roots = [
        str(Path(expand_config_path(item)).resolve())
        for item in raw_dataset_read_roots
        if str(item).strip()
    ]
    if defaults["dataset_root"] not in dataset_read_roots:
        dataset_read_roots.insert(0, defaults["dataset_root"])
    defaults["dataset_read_roots"] = list(dict.fromkeys(dataset_read_roots))
    checkpoint_allowed_roots = [
        str(Path(expand_config_path(item)).resolve()) for item in defaults.get("checkpoint_allowed_roots", [])
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
        str(Path(expand_config_path(item)).resolve()) for item in defaults.get("eval_video_roots", [])
    ]
    defaults["cluster_resources_script"] = str(
        Path(expand_config_path(defaults["cluster_resources_script"])).resolve()
    )
    if defaults.get("nas_dataset_staging_root"):
        defaults["nas_dataset_staging_root"] = expand_config_path(defaults["nas_dataset_staging_root"])
    if defaults.get("nas_checkpoint_staging_root"):
        defaults["nas_checkpoint_staging_root"] = expand_config_path(defaults["nas_checkpoint_staging_root"])
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
                item[path_key] = str(Path(expand_config_path(item[path_key])).resolve())
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
            "dataset_read_roots",
            "assets_base_dir",
            "checkpoint_base_dir",
            "base_checkpoint",
            "eval_video_roots",
            "inventory_cache_path",
            "inventory_source_path",
            "nas_dataset_staging_root",
            "nas_checkpoint_staging_root",
            "cache_root",
            "log_dir",
            "remote_job_dir",
            "conda_sh",
        ):
            if item.get(path_key):
                if path_key in {"eval_video_roots", "dataset_read_roots"} and isinstance(item[path_key], list):
                    item[path_key] = [expand_config_path(value) for value in item[path_key]]
                else:
                    item[path_key] = expand_config_path(item[path_key])
        for path_key in ("openpi_python",):
            if item.get(path_key):
                item[path_key] = expand_config_path(item[path_key])
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
        dataset_id = metadata.get("dataset_id")
        if dataset_id:
            deleting = metadata.get("transfer_kind") == "dataset_delete"
            for path in self.root.glob("*/task.json"):
                other = read_json(path, {}) or {}
                other_metadata = other.get("metadata", {})
                if other.get("state") not in TERMINAL_STATES and other_metadata.get("dataset_id") == dataset_id:
                    if deleting or other_metadata.get("transfer_kind") == "dataset_delete":
                        raise ValueError(f"dataset {dataset_id} is in use by active task {other.get('id')}")
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
        if task_type == "train":
            experiment_artifacts.record_task(self.config, task)
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

    def _task_launch_backend(self, task: dict[str, Any]) -> str:
        """Return the configured launcher for a task.

        Training can be launched with ``nohup`` so it is detached from the
        Dashboard's terminal/session.  Other task types keep the existing
        systemd/direct selection unless a global backend is explicitly set.
        """
        key = "training_task_launch_backend" if task.get("type") == "train" else "task_launch_backend"
        backend = self.config.get(key, self.config.get("task_launch_backend", "auto"))
        return str(backend or "auto").strip().lower()

    def _systemd_task_backend_enabled(self, task: dict[str, Any]) -> bool:
        backend = self._task_launch_backend(task)
        if backend in {"direct", "popen", "subprocess"}:
            return False
        if backend in {"nohup", "detached"}:
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

    def _launch_nohup_runner(
        self,
        task: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> subprocess.Popen:
        """Launch a task with nohup and a detached process session.

        ``task_runner.py`` still owns the child process and writes the durable
        exit.json file.  Keeping the runner command (rather than the nohup
        executable) in ``launch_command`` also lets Dashboard recovery match
        the post-exec Python process after a Dashboard restart.
        """
        nohup = shutil.which("nohup")
        if not nohup:
            raise RuntimeError("nohup executable is required for the configured task backend")
        log_handle = self._log_path(task["id"]).open("ab", buffering=0)
        try:
            return subprocess.Popen(
                [nohup, *task["launch_command"]],
                cwd=self.config["openpi_repo"],
                env=env,
                stdin=subprocess.DEVNULL,
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
            backend = self._task_launch_backend(task)
            if self._systemd_task_backend_enabled(task):
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

            if backend in {"nohup", "detached"}:
                task["launch_backend"] = "nohup"
                process = self._launch_nohup_runner(task, env=env)
            else:
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
                self.config.get("training_min_free_gpu_mib", 22_500),
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
                dataset_root_override=task.get("metadata", {}).get("dataset_root_path"),
                dataset_read_only=bool(task.get("metadata", {}).get("dataset_read_only", False)),
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
        callback = getattr(self, 'storage_change_callback', None)
        if callback:
            try: callback(task)
            except Exception as exc: self._append_log(task, 'Storage index update pending: '+str(exc))
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
        callback = getattr(self, 'storage_change_callback', None)
        if callback:
            try: callback(task)
            except Exception as exc: self._append_log(task, 'Storage index update pending: '+str(exc))
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
        if task.get('metadata',{}).get('external_training') and task.get('metadata',{}).get('runtime')=='slurm':
            return task
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
            "dataset_root_path": train_metadata.get("dataset_root_path"),
            "dataset_read_only": train_metadata.get("dataset_read_only", False),
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
                dataset_root_override=task.get("metadata", {}).get("dataset_root_path"),
                dataset_read_only=bool(task.get("metadata", {}).get("dataset_read_only", False)),
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
            metadata=task.get('metadata') or {}
            if metadata.get('external_training') and metadata.get('runtime')=='slurm':
                job=str(metadata.get('slurm_job_id',''))
                if not job.isdigit(): raise ValueError('Invalid external Slurm identity')
                result=subprocess.run(['ssh','-T',metadata['slurm_submit_host'],shlex.join(['scancel',job])],capture_output=True,text=True,timeout=10)
                if result.returncode: raise ValueError(result.stderr[-500:])
                task['state']='stopping';atomic_json(self._path(task_id),task);return task
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
        result = self.delete_many([task_id])
        return {"deleted": True, "task": result["tasks"][0]}

    def delete_many(self, task_ids: list[str], *, validate_only=False) -> dict[str, Any]:
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
                pid=int(task.get('pid') or 0)
                if pid and pid_alive(pid) and process_matches_task(pid,task):
                    raise ValueError('cannot delete task while its recorded process is alive')
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

            if validate_only:
                return {"tasks": list(selected_tasks.values()), "task_ids": normalized_ids}
            callback = getattr(self, 'before_history_delete', None)
            if callback:
                callback(list(selected_tasks.values()))
            from server_4090.storage_worker import history_cleanup
            for task_id, task_dir in task_dirs.items():
                self.processes.pop(task_id, None)
                history_cleanup(dict(run_paths=[str(task_dir)], run={'state': selected_tasks[task_id]['state']}))
                self._path(task_id).unlink()
                if getattr(self,'live_mirror',None):self.live_mirror.forget('task:'+task_id)
                # Unknown files (including nested rollout/weights) are preserved.
                try: task_dir.rmdir()
                except OSError: pass

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
        if metadata.get('external_training') and metadata.get('remote_stdout'):
            return submit_host, {k:v for k,v in {'stdout':metadata.get('remote_stdout'),'stderr':metadata.get('remote_stderr')}.items() if v and v!='/dev/null'}
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
            "client_allow_execution": client_allow,
            "client_execution_state": client_state,
            "client_in_flight": client_in_flight,
            "policy_in_flight": runtime_in_flight,
            "policy_active_inferences": _positive_int_or_none(runtime.get("active_inferences")) or 0,
            "policy_inference_started_at": runtime.get("last_inference_started_at"),
            "policy_inference_finished_at": runtime.get("last_inference_finished_at"),
            **horizon_status,
            **time_contract_status,
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


def gpu_inventory(*, strict: bool = False) -> list[dict[str, Any]]:
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
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logging.getLogger(__name__).warning("GPU inventory query failed: %s", exc)
        if strict:
            raise RuntimeError(f"GPU inventory query failed: {exc}") from exc
        return []
    process_query_error = None
    try:
        process_lines = subprocess.check_output(proc_cmd, text=True, timeout=10).splitlines()
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        process_lines = []
        process_query_error = f"GPU process query failed: {exc}"
        logging.getLogger(__name__).warning("%s", process_query_error)
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
                "compute_available": process_query_error is None and parts[1] not in unavailable_uuids,
                "health_issue": process_query_error or (
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
    system_site_paths = sysconfig.get_paths()
    for site_key in ("platlib", "purelib"):
        site_path = system_site_paths.get(site_key)
        if site_path:
            candidate_paths.append(
                Path(site_path) / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
            )

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
    dataset_root_override: str | Path | None = None,
    dataset_read_only: bool = False,
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
            "HF_LEROBOT_HOME": str(dataset_root_override or config["dataset_root"]),
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
    if dataset_read_only:
        env["BIMANUAL_VLA_DATASET_READ_ONLY"] = "1"
    else:
        env.pop("BIMANUAL_VLA_DATASET_READ_ONLY", None)
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
    config_file = Path(config_path).expanduser().resolve()
    config = upgrade_native_profiles(load_config(config_file))
    from server_4090.remote_checkpoints import RemoteCheckpointInventory
    remote_checkpoints = RemoteCheckpointInventory(config.get("cluster_targets", {}))
    token = os.environ.get("BIMANUAL_VLA_SERVER_TOKEN", "")
    if len(token) < 20:
        raise RuntimeError("set BIMANUAL_VLA_SERVER_TOKEN to a random value of at least 20 characters")
    login_user = os.environ.get("BIMANUAL_VLA_LOGIN_USER", "")
    login_password = os.environ.get("BIMANUAL_VLA_LOGIN_PASSWORD", "")
    app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
    from server_4090.json_response import FiniteJSONProvider
    app.json = FiniteJSONProvider(app)
    dashboard_template_path = APP_DIR / "templates" / "index.html"
    dashboard_build_id = hashlib.sha256(dashboard_template_path.read_bytes()).hexdigest()[:12]
    app.config["MAX_CONTENT_LENGTH"] = int(config["max_chunk_mib"] * 1024**2) + 1024 * 1024
    tasks = TaskManager(config)
    from server_4090.history_management import HistoryManager
    histories = HistoryManager(config, tasks)
    tasks.before_history_delete = histories.before_tasks_deleted
    rlinf_history_path = Path(config["workspace_root"]) / "rlinf_deleted_runs.json"

    from server_4090.dashboard_snapshot import Snapshot
    rlinf_list_snapshot = Snapshot(Path(config["workspace_root"]) / "rlinf-runs-index.json",
                                    lambda: scan_runs(config, tasks.list()), ttl=30)

    live_rlinf_overlay_path = Path(config['workspace_root']) / 'rlinf-discovered-runs.json'
    live_rlinf_overlay = read_json(live_rlinf_overlay_path, {}) or {}
    live_rlinf_overlay_lock = threading.Lock()
    rlinf_discovery_snapshots = {}
    rlinf_discovery_lock = threading.Lock()

    def cached_rlinf_runs():
        with rlinf_list_snapshot.lock:
            base = list(rlinf_list_snapshot.data or [])
        with live_rlinf_overlay_lock:
            combined = dict(live_rlinf_overlay)
        combined.update({r['id']: r for r in base})
        return list(combined.values())

    from server_4090.rlinf_status import RunStatusCache
    rlinf_status_cache = RunStatusCache(config, Path(config['workspace_root']) / 'rlinf-job-status.json', cached_rlinf_runs)

    def rlinf_identity_keys(run):
        from server_4090.rlinf_run_dedup import identity_keys
        return identity_keys(run)

    def visible_rlinf_runs(snapshot=None):
        with tasks.lock:
            deleted = set(read_json(rlinf_history_path, []) or [])
            task_list = tasks.list()
        snapshot = snapshot or rlinf_list_snapshot.get()
        current_tasks = {task["id"]: task for task in task_list}
        runs = []
        with live_rlinf_overlay_lock:
            combined = dict(live_rlinf_overlay)
        combined.update({r['id']: r for r in snapshot['data'] or []})
        job_snapshot = rlinf_status_cache.snapshot.get()
        # A configured live mirror and a discovered node path can be one run.
        # Prefer the refreshed configured mirror, matching node AND path aliases.
        seen = set()
        from server_4090.rlinf_run_dedup import deduplicate_runs
        candidates, _ = deduplicate_runs(combined.values())
        for run in candidates:
            # Legacy discovery errors have no observation time. Fresh scans
            # republish current failures; an old exception is not current health.
            if run.get('state') == 'unavailable' and not run.get('checked_at'):
                continue
            keys = rlinf_identity_keys(run)
            if keys & seen:
                continue
            seen.update(keys)
            if run["id"] in deleted:
                continue
            run = dict(run)
            linked = current_tasks.get(run.get("task_id"))
            if linked:
                run["state"] = linked["state"]
            elif run.get('job_id'):
                run = rlinf_status_cache.apply(run, job_snapshot)
            elif time.time() - float(run.get('_status_checked_at') or (snapshot['updated_at'] if run['id'] not in live_rlinf_overlay else 0)) > 120 and run.get("state") in {"running", "queued"}:
                run["state"] = "observed"
                run["status_reason"] = "Cached status is stale; background refresh pending"
            from server_4090.rlinf_identity import annotate_run
            runs.append(annotate_run(config, run))
        return runs

    def selected_rlinf_run(run_id):
        if run_id in set(read_json(rlinf_history_path, []) or []):
            return None
        run = live_rlinf_overlay.get(run_id)
        from server_4090.rlinf_identity import annotate_run
        run = dict(run) if run else rlinf_list_snapshot.find(run_id)
        if run:
            keys = rlinf_identity_keys(run)
            from server_4090.rlinf_run_dedup import deduplicate_runs
            candidates, _ = deduplicate_runs(cached_rlinf_runs())
            for current in candidates:
                if keys & rlinf_identity_keys(current):
                    run = dict(current, id=run_id)
                    break
            run = rlinf_status_cache.apply(run, rlinf_status_cache.snapshot.get())
            if run.get('task_id') and tasks._path(run['task_id']).is_file():
                linked = read_json(tasks._path(run['task_id']), {})
                if linked.get('state'):
                    run = dict(run, state=linked['state'])
        return annotate_run(config, run) if run else None

    def delete_rlinf_records(run_ids):
        if not isinstance(run_ids, list) or not run_ids or len(run_ids) > 200:
            raise ValueError("run_ids must contain between 1 and 200 run IDs")
        if any(not isinstance(item, str) or not item for item in run_ids):
            raise ValueError("run_ids must contain nonempty strings")
        runs = {run["id"]: run for run in visible_rlinf_runs()}
        ids = list(dict.fromkeys(run_ids))
        if any(item not in runs or runs[item].get("state") == "unavailable" for item in ids):
            raise ValueError("unknown or unavailable RLinf run")
        for ident in ids: runs[ident] = check_rlinf_run_now(runs[ident])
        with tasks.lock:
            if any(task.get("type") == "eval" and task.get("state") not in TERMINAL_STATES and task.get("metadata", {}).get("run_id") in ids for task in tasks.list()):
                raise ValueError("该运行仍有正在执行的 rollout 评测，请先停止评测")
            task_ids = list(dict.fromkeys(runs[item]["task_id"] for item in ids if runs[item].get("task_id")))
            if task_ids: tasks.delete_many(task_ids, validate_only=True)
            operations = [op for ident in ids for op in histories.run_operations(runs[ident])]
            histories.cleanup(operations)
            histories.remember([runs[ident] for ident in ids])
            if task_ids:
                tasks.delete_many(task_ids)
            deleted = set(read_json(rlinf_history_path, []) or [])
            atomic_json(rlinf_history_path, sorted(deleted | set(ids)))
        for ident in ids:live_mirror.forget('rlinf:'+ident)
        rlinf_list_snapshot.invalidate()
        artifact_history_snapshot.invalidate()
        artifact_replicas.mark_dirty()
        return {"deleted": True, "run_ids": ids, "task_ids": task_ids}
    dataset_root = Path(config["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_read_roots = [Path(item) for item in config.get("dataset_read_roots", [])]
    dataset_read_roots.extend(Path(item["dataset_root"]) for item in config.get("local_storage_locations", {}).values() if item.get("available", True) and item.get("dataset_root"))
    if dataset_root not in dataset_read_roots:
        dataset_read_roots.insert(0, dataset_root)
    dataset_read_roots = [dataset_root, *[item for item in dict.fromkeys(dataset_read_roots) if item != dataset_root]]

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
        checker = [
            config["openpi_python"],
            "-m",
            "bimanual_vla.data.check",
            str(path),
        ]
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
        dataset_read_roots=dataset_read_roots,
        assets_base_dir=Path(config["assets_base_dir"]),
        validate_staging=validate_staging_dataset,
        validate_installed=validate_installed_dataset,
        assert_idle=assert_dataset_idle,
    )
    from server_4090.dataset_timing_api import register_timing_routes
    register_timing_routes(app, config=config, tasks=tasks, editor=dataset_editor,
                           environment=lambda: build_environment(config, None),
                           assert_idle=assert_dataset_idle)
    uploads = UploadManager(config, dataset_editor)
    observations = PolicyTelemetryStore(config)
    allowed_gpus = set(map(int, config["allowed_gpu_ids"]))
    checkpoint_roots = [Path(item) for item in config["checkpoint_allowed_roots"]]
    checkpoint_roots.extend(Path(item) for item in config.get("experiment_artifact_read_roots", []))
    openpi_helper = str(APP_DIR / "openpi_single_arm.py")
    checkpoint_size_cache: dict[str, tuple[float, int]] = {}
    settings_lock = threading.RLock()
    auth_users_file = Path(
        os.environ.get(
            "BIMANUAL_VLA_AUTH_USERS_FILE",
            str(Path.home() / ".config" / "bimanual-vla" / "users.json"),
        )
    ).expanduser().resolve()

    def _password_digest(password: str, salt: bytes | None = None) -> tuple[str, str]:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 240_000)
        return salt.hex(), digest.hex()

    def _load_auth_users() -> list[dict[str, str]]:
        raw = read_json(auth_users_file)
        users = raw.get("users") if isinstance(raw, dict) else raw
        if not isinstance(users, list):
            return []
        result: list[dict[str, str]] = []
        for item in users:
            if not isinstance(item, dict):
                continue
            username = str(item.get("username", "")).strip()
            salt = str(item.get("salt", "")).strip()
            digest = str(item.get("password_hash", "")).strip()
            if username and salt and digest:
                result.append({"username": username, "salt": salt, "password_hash": digest})
        return result

    auth_users = _load_auth_users()
    if not auth_users and login_user and login_password:
        salt, digest = _password_digest(login_password)
        auth_users = [{"username": login_user, "salt": salt, "password_hash": digest}]

    def _persist_auth_users() -> None:
        auth_users_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(auth_users_file, {"version": 1, "users": auth_users})

    def _verify_auth_password(username: str, password: str) -> bool:
        for user in auth_users:
            if user["username"] != username:
                continue
            try:
                digest = hashlib.pbkdf2_hmac(
                    "sha256", password.encode("utf-8"), bytes.fromhex(user["salt"]), 240_000
                ).hex()
            except ValueError:
                return False
            return hmac.compare_digest(digest, user["password_hash"])
        return False

    def _auth_usernames() -> list[str]:
        names = [user["username"] for user in auth_users]
        if not names and login_user:
            names.append(login_user)
        return sorted(set(names), key=str.casefold)

    def _token_file_path() -> Path:
        return Path(
            os.environ.get(
                "BIMANUAL_VLA_TOKEN_FILE",
                str(Path.home() / ".config" / "bimanual-vla" / "server.env"),
            )
        ).expanduser().resolve()

    def _write_server_env(*, token: str) -> None:
        path = _token_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        current_user = login_user or (auth_users[0]["username"] if auth_users else "dashboard")
        lines = [
            f"export BIMANUAL_VLA_SERVER_TOKEN={shlex.quote(token)}",
            f"export BIMANUAL_VLA_LOGIN_USER={shlex.quote(current_user)}",
            f"export BIMANUAL_VLA_LOGIN_PASSWORD={shlex.quote(login_password or secrets.token_urlsafe(24))}",
        ]
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)

    settings_path_keys = (
        "dataset_root",
        "checkpoint_base_dir",
        "base_checkpoint",
        "assets_base_dir",
        "workspace_root",
        "cache_root",
    )

    def dataset_path_for_id(dataset_id: str) -> Path:
        dataset_id = safe_name(dataset_id, "dataset id")
        roots = [Path(item) for item in config.get("dataset_read_roots", [])]
        roots.extend(Path(item["dataset_root"]) for item in config.get("local_storage_locations", {}).values() if item.get("available", True) and item.get("dataset_root"))
        storage = Path(config["dataset_root"])
        roots = [storage, *[root for root in roots if root != storage]]
        for root in roots:
            candidate = root / dataset_id
            if candidate.is_dir() and (candidate / "meta" / "info.json").is_file():
                return candidate
        return storage / dataset_id

    def dataset_root_for_id(dataset_id: str) -> tuple[Path, bool]:
        path = dataset_path_for_id(dataset_id)
        storage = Path(config["dataset_root"]).expanduser().resolve()
        try:
            path.resolve().relative_to(storage)
            writable = True
        except ValueError:
            writable = False
        return path, writable

    def dataset_env_kwargs(dataset_id: str) -> dict[str, Any]:
        path, writable = dataset_root_for_id(dataset_id)
        return {
            "dataset_root_override": path.parent,
            "dataset_read_only": not writable,
        }
    def settings_browser_roots() -> list[Path]:
        """Return safe starting points for the authenticated directory browser."""
        candidates: list[Path] = []
        configured = config.get("directory_browser_roots", [])
        if isinstance(configured, list):
            candidates.extend(Path(str(item)).expanduser() for item in configured if str(item).strip())
        for key in settings_path_keys:
            value = config.get(key)
            if value:
                path = Path(str(value)).expanduser()
                candidates.extend((path, path.parent))
        for key in ("dataset_read_roots", "checkpoint_allowed_roots", "eval_video_roots"):
            values = config.get(key, [])
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                for value in values:
                    if str(value).strip():
                        path = Path(str(value)).expanduser()
                        candidates.extend((path, path.parent))
        candidates.append(Path.home())
        for anchor in (Path("/mnt"), Path("/DATA")):
            if anchor.exists():
                candidates.append(anchor)
        result: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                path = lexical_absolute_path(candidate)
            except (OSError, RuntimeError, ValueError):
                continue
            if not path.exists() or not path.is_dir():
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            result.append(path)
        return result

    def settings_payload() -> dict[str, Any]:
        paths = {key: str(config.get(key, "")) for key in settings_path_keys}
        paths["dataset_read_roots"] = [
            str(item) for item in config.get("dataset_read_roots", [config.get("dataset_root", "")])
        ]
        paths["checkpoint_allowed_roots"] = [
            str(item) for item in config.get("checkpoint_allowed_roots", [])
        ]
        paths["eval_video_roots"] = [
            str(item) for item in config.get("eval_video_roots", [])
        ]
        readonly = {
            "openpi_repo": str(config.get("openpi_repo", "")),
            "openpi_python": str(config.get("openpi_python", "")),
            "config_file": str(config_file),
        }
        browser_roots = [
            {"path": str(path), "label": str(path)} for path in settings_browser_roots()
        ]
        return {
            "paths": paths,
            "readonly": readonly,
            "auth": {
                "users": _auth_usernames(),
                "token_configured": bool(token),
                "token_hint": (token[:4] + "…" + token[-4:]) if len(token) >= 10 else "configured",
                "users_file": str(auth_users_file),
            },
            "browser_roots": browser_roots,
            "restart_required_for": [
                "dataset_root",
                "dataset_read_roots",
                "assets_base_dir",
                "workspace_root",
                "cache_root",
                "eval_video_roots",
            ],
        }

    def settings_path(value: Any, label: str, *, require_params: bool = False) -> str:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError(f"{label} cannot be empty")
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"{label} must be an existing directory: {path}")
        if require_params and not (path / "params").is_dir():
            raise ValueError(f"{label} must contain a params directory: {path}")
        return str(path)

    def settings_path_list(value: Any, label: str) -> list[str]:
        if isinstance(value, str):
            values = [line.strip() for line in value.splitlines() if line.strip()]
        elif isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
        else:
            raise ValueError(f"{label} must be a list or one path per line")
        result: list[str] = []
        seen: set[str] = set()
        for item in values:
            normalized = settings_path(item, label)
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        if not result:
            raise ValueError(f"{label} cannot be empty")
        return result

    def persist_settings(patch: dict[str, Any]) -> None:
        with settings_lock:
            current = read_json(config_file)
            if not isinstance(current, dict):
                raise ValueError(f"invalid Dashboard config: {config_file}")
            current.update(patch)
            temporary = config_file.with_name(f".{config_file.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(current, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, config_file)

    @app.get("/api/settings")
    def get_settings():
        return jsonify(settings_payload())

    @app.put("/api/settings")
    def update_settings():
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        requested = payload.get("paths", payload)
        if not isinstance(requested, dict):
            raise ValueError("paths must be a JSON object")

        normalized: dict[str, Any] = {}
        for key in settings_path_keys:
            if key in requested:
                normalized[key] = settings_path(
                    requested[key],
                    key,
                    require_params=key == "base_checkpoint",
                )
            else:
                normalized[key] = str(config.get(key, ""))
        if "dataset_read_roots" in requested:
            dataset_roots = settings_path_list(
                requested["dataset_read_roots"], "dataset_read_roots"
            )
        else:
            dataset_roots = [
                str(item) for item in config.get("dataset_read_roots", [config["dataset_root"]])
            ]
        if normalized["dataset_root"] not in dataset_roots:
            dataset_roots.insert(0, normalized["dataset_root"])
        normalized["dataset_read_roots"] = list(dict.fromkeys(dataset_roots))
        if "checkpoint_allowed_roots" in requested:
            allowed_roots = settings_path_list(
                requested["checkpoint_allowed_roots"], "checkpoint_allowed_roots"
            )
        else:
            allowed_roots = [
                str(item) for item in config.get("checkpoint_allowed_roots", [])
            ]
        for required in (normalized["checkpoint_base_dir"], normalized["base_checkpoint"]):
            if required not in allowed_roots:
                allowed_roots.append(required)
        normalized["checkpoint_allowed_roots"] = allowed_roots

        if "eval_video_roots" in requested:
            raw_eval_roots = requested["eval_video_roots"]
            if isinstance(raw_eval_roots, str) and not raw_eval_roots.strip():
                normalized["eval_video_roots"] = []
            elif isinstance(raw_eval_roots, list) and not raw_eval_roots:
                normalized["eval_video_roots"] = []
            else:
                normalized["eval_video_roots"] = settings_path_list(
                    raw_eval_roots, "eval_video_roots"
                )
        else:
            normalized["eval_video_roots"] = [
                str(item) for item in config.get("eval_video_roots", [])
            ]

        changed = {
            key for key in (*settings_path_keys, "dataset_read_roots", "checkpoint_allowed_roots", "eval_video_roots")
            if normalized.get(key) != config.get(key)
        }
        persist_patch = {
            key: normalized[key]
            for key in (*settings_path_keys, "dataset_read_roots", "checkpoint_allowed_roots", "eval_video_roots")
        }
        persist_settings(persist_patch)
        config.update(persist_patch)
        checkpoint_roots[:] = [Path(item) for item in normalized["checkpoint_allowed_roots"]]
        restart_fields = {
            "dataset_root",
            "dataset_read_roots",
            "assets_base_dir",
            "workspace_root",
            "cache_root",
            "eval_video_roots",
        }
        restart_required = bool(changed & restart_fields)
        return jsonify(
            {
                "ok": True,
                "settings": settings_payload(),
                "changed": sorted(changed),
                "restart_required": restart_required,
                "restart_required_for": sorted(changed & restart_fields),
            }
        )

    @app.get("/api/fs/directories")
    def list_server_directories():
        roots = settings_browser_roots()
        if not roots:
            raise ValueError("no browsable server directories are configured")
        requested = request.args.get("path", "").strip()
        path = lexical_absolute_path(requested) if requested else roots[0]
        if not any(lexical_path_is_within(path, root) for root in roots):
            raise ValueError("directory is outside configured browser roots")
        if not path.is_dir():
            raise FileNotFoundError(str(path))
        entries = []
        for entry in sorted(path.iterdir(), key=lambda item: item.name.lower()):
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            if not is_dir or entry.name.startswith("."):
                continue
            entries.append(
                {
                    "name": entry.name,
                    "path": str(lexical_absolute_path(entry)),
                    "symlink": entry.is_symlink(),
                }
            )
        parent = path.parent if any(lexical_path_is_within(path.parent, root) for root in roots) else None
        return jsonify(
            {
                "path": str(path),
                "parent": str(parent) if parent else None,
                "roots": [{"path": str(root), "label": str(root)} for root in roots],
                "entries": entries,
            }
        )

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
    def compress_rlinf_response(response):
        from server_4090.response_compression import compress_json_response
        if request.path.startswith('/api/rlinf/'):
            return compress_json_response(response, request)
        return response

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
        status = 400 if isinstance(exc, (ValueError, FileExistsError, FileNotFoundError, PermissionError)) else 500
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
            checkpoint_destinations=[dict(id=k, label=v["label"]) for k,v in artifact_replica_locations(config).items()],
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
        valid_user = supplied_user in _auth_usernames()
        valid_password = (
            _verify_auth_password(supplied_user, supplied_password)
            if auth_users
            else hmac.compare_digest(supplied_password, login_password)
        )
        if not (valid_user and valid_password):
            return jsonify({"error": "invalid username or password"}), 401
        response = jsonify({"token": token, "token_type": "Bearer", "username": supplied_user})
        response.set_cookie(
            "bimanual_vla_token",
            token,
            max_age=30 * 24 * 60 * 60,
            httponly=True,
            samesite="Lax",
        )
        return response

    @app.get("/api/auth/settings")
    def get_auth_settings():
        return jsonify(settings_payload().get("auth", {}))

    @app.post("/api/auth/users")
    def add_auth_user():
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        if not SAFE_NAME.fullmatch(username):
            raise ValueError("username must contain only letters, numbers, dot, underscore or hyphen")
        if len(password) < 8:
            raise ValueError("password must contain at least 8 characters")
        with settings_lock:
            if any(item["username"] == username for item in auth_users):
                raise ValueError(f"user already exists: {username}")
            salt, digest = _password_digest(password)
            auth_users.append({"username": username, "salt": salt, "password_hash": digest})
            _persist_auth_users()
        return jsonify({"ok": True, "username": username, "users": _auth_usernames()}), 201

    @app.put("/api/auth/users/<username>")
    def update_auth_user(username: str):
        username = str(username).strip()
        payload = request.get_json(force=True)
        password = str(payload.get("password", "")) if isinstance(payload, dict) else ""
        if len(password) < 8:
            raise ValueError("password must contain at least 8 characters")
        with settings_lock:
            user = next((item for item in auth_users if item["username"] == username), None)
            if user is None:
                raise FileNotFoundError(username)
            salt, digest = _password_digest(password)
            user.update({"salt": salt, "password_hash": digest})
            _persist_auth_users()
        return jsonify({"ok": True, "username": username, "users": _auth_usernames()})

    @app.delete("/api/auth/users/<username>")
    def delete_auth_user(username: str):
        username = str(username).strip()
        with settings_lock:
            if len(auth_users) <= 1:
                raise ValueError("cannot delete the last Dashboard user")
            before = len(auth_users)
            auth_users[:] = [item for item in auth_users if item["username"] != username]
            if len(auth_users) == before:
                raise FileNotFoundError(username)
            _persist_auth_users()
        return jsonify({"ok": True, "deleted": username, "users": _auth_usernames()})

    @app.post("/api/auth/token/rotate")
    def rotate_auth_token():
        nonlocal token
        new_token = secrets.token_urlsafe(36)
        _write_server_env(token=new_token)
        token = new_token
        return jsonify({"ok": True, "token": new_token, "restart_required": True})

    @app.post("/api/dashboard/restart")
    def restart_dashboard():
        if not shutil.which("systemctl"):
            raise RuntimeError("systemctl is not available; restart the Dashboard service manually")
        service = (
            "bimanual-vla-sim-dashboard.service"
            if config.get("dashboard_profile") == "simulation"
            else "bimanual-vla-dashboard.service"
        )
        subprocess.Popen(
            ["systemctl", "--user", "restart", service],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return jsonify({"ok": True, "service": service, "restart_requested": True}), 202

    def list_datasets() -> list[dict[str, Any]]:
        from server_4090.task_scenes import dataset_scene, bindings
        scene_bindings=bindings(config)
        datasets = []
        storage_root = Path(config["dataset_root"]).expanduser().resolve()
        roots = [Path(item).expanduser().resolve() for item in config.get("dataset_read_roots", [])]
        roots.extend(Path(item["dataset_root"]).expanduser().resolve() for item in config.get("local_storage_locations", {}).values() if item.get("available", True) and item.get("dataset_root"))
        roots = list(dict.fromkeys(roots))
        roots = [storage_root, *[root for root in roots if root != storage_root]]
        selected: dict[str, tuple[Path, bool]] = {}
        for root in roots:
            for directory in sorted(root.iterdir() if root.exists() else []):
                if not directory.is_dir() or directory.name.startswith("."):
                    continue
                if not (directory / "meta" / "info.json").is_file():
                    continue
                # The storage root wins when the same dataset id is present in
                # both storage and an external read-only archive.
                selected.setdefault(
                    directory.name,
                    (directory, root == storage_root),
                )
        for directory, writable in sorted(selected.values(), key=lambda item: item[0].name.lower()):
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
            local_location = next((item for item in dataset_location_configs().values() if item.get("kind") in {"local", "local_archive"} and Path(item["dataset_root"]).expanduser().resolve() == directory.parent.resolve()), {})
            local_target = local_location.get("name", "local_4090_read")
            datasets.append(
                {
                    "id": directory.name,
                    "task_scene": dataset_scene(directory.name,info,scene_bindings),
                    "path": str(directory),
                    "root": str(directory.parent),
                    "storage": writable,
                    "read_only": not writable,
                    "remote_only": False,
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
                            "target": local_target,
                            "label": local_location.get("label", "4×4090 本机读取目录"),
                            "kind": local_location.get("kind", "local"),
                            "host": None,
                            "path": str(directory),
                            "root": str(directory.parent),
                            "read_only": not writable,
                            "origin": origin.get("dataset_origin", "unknown"),
                            "episodes": info.get("total_episodes"),
                            "frames": info.get("total_frames"),
                            "fps": info.get("fps"),
                            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(directory.stat().st_mtime)),
                        }
                    ],
                    "targets": [local_target],
                }
            )
        return datasets

    def visible_dataset_origin_set() -> set[str]:
        return set(config.get("visible_dataset_origins", ["real", "unknown"]))

    def dataset_origin_for_id(dataset_id: str) -> str:
        dataset_path = dataset_path_for_id(dataset_id)
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

    def list_base_models(checkpoints: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Original pretrained models only; checkpoints remain in their own catalog."""
        from server_4090.model_catalog import is_foundation_path
        default_path=Path(config['base_checkpoint']).resolve()
        trained={Path(item['path']).resolve() for item in checkpoints or [] if item.get('path')}
        candidates={default_path}
        for root in checkpoint_roots:
            if not root.is_dir():continue
            candidates.update(p.resolve() for p in [root,*root.iterdir()] if is_foundation_path(p) and p.is_dir())
        models=[]
        for path in candidates:
            if path in trained or not is_foundation_path(path) or not (path/'params').is_dir():continue
            variant=infer_model_variant(path)
            if variant is None:continue
            models.append(dict(path=str(path),name=path.name,model_variant=variant,default=path==default_path,
                               foundation=True,source='pretrained',backend='foundation',format='orbax',
                               experiment=None,checkpoint_step=None,arm_mode=None,config_name=None,
                               dataset_ids=[],dataset_origins={},action_contract=None))
        return sorted(models,key=lambda m:(not m['default'],m['model_variant'],m['path']))

    def resolve_base_model(payload: dict[str, Any]) -> tuple[Path, str]:
        requested_path = payload.get("base_checkpoint") or config["base_checkpoint"]
        target = runtime_config_for_target(payload.get("execution_target"))
        # A configured foundation model may exist only on the shared cluster
        # filesystem. Validate there without requiring a redundant local copy.
        remote_foundation = bool(
            target and target.get("access_mode") in {"login_shared_slurm", "slurm_only"}
            and str(requested_path) == str(config["base_checkpoint"])
        )
        path = resolve_under(requested_path, checkpoint_roots, must_exist=not remote_foundation)
        if remote_foundation and target.get("access_mode") == "slurm_only":
            # Independent compute-node storage is validated inside the Slurm
            # allocation before norm/train, never against the Dashboard disk.
            pass
        elif remote_foundation:
            remote_params = str(PurePosixPath(target["base_checkpoint"]) / "params")
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 target["submit_host"], "test -d " + shlex.quote(remote_params)],
                capture_output=True, text=True, timeout=20, check=False,
            )
            if result.returncode:
                raise ValueError(f"cluster base checkpoint is unavailable: {remote_params}")
        elif not (path / "params").is_dir():
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

    def resolve_norm_model(payload: dict[str, Any]) -> tuple[Path, str]:
        """Resolve norm's data contract without requiring a weight selection."""
        model_variant = str(payload.get("model_variant") or "pi05")
        if model_variant not in MODEL_VARIANTS:
            raise ValueError(f"model_variant must be one of {sorted(MODEL_VARIANTS)}")
        return Path(config["base_checkpoint"]).expanduser().resolve(), model_variant

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

    def measure_checkpoint_sizes():
        result = {}
        roots = list(dict.fromkeys([config['checkpoint_base_dir'], *config.get('checkpoint_allowed_roots', [])]))
        paths = {m['path'] for m in artifact_inventory.models('openpi')}
        for root in roots:
            for variant in ('pi05', 'pi0'):
                for arm in ('single', 'bimanual'):
                    base = Path(root) / policy_config_name(arm, variant)
                    if not base.is_dir(): continue
                    for exp in base.iterdir():
                        if exp.is_dir():
                            paths.update(str(p.resolve()) for p in exp.iterdir() if p.name.isdigit() and p.is_dir())
        from server_4090.storage_worker import discover_models
        canonical=[dict(path=str(r),layout='canonical') for r in artifact_inventory.roots() if r.name in {'simulation-artifacts','experiment_artifacts'}]
        paths.update(m['path'] for m in discover_models(dict(roots=canonical,backend='openpi'))['models'])
        for raw in sorted(paths):
            path = Path(raw).resolve()
            total = 0
            for current, dirs, files in os.walk(path, followlinks=False):
                dirs[:] = [d for d in dirs if not (Path(current)/d).is_symlink()]
                for name in files:
                    file = Path(current)/name
                    if not file.is_symlink():
                        try: total += file.stat().st_size
                        except FileNotFoundError: pass
            result[str(path)] = total
        return result

    checkpoint_sizes_snapshot = Snapshot(Path(config['workspace_root']) / 'checkpoint-sizes-index.json', measure_checkpoint_sizes, ttl=900)

    def list_checkpoints() -> list[dict[str, Any]]:
        checkpoints = []
        storage_root = Path(config["checkpoint_base_dir"]).expanduser().resolve()
        roots = [Path(item).expanduser().resolve() for item in config.get("checkpoint_allowed_roots", [])]
        if storage_root not in roots:
            roots.insert(0, storage_root)
        roots += [p for p in artifact_inventory.roots() if p not in roots]
        candidates = []
        from server_4090.storage_worker import discover_models
        live_roots=[dict(path=str(r),layout='canonical') for r in artifact_inventory.roots() if r.name in {'simulation-artifacts','experiment_artifacts'} or str(r)==config.get('experiment_artifact_root')]
        live_roots += [dict(path=str(r),layout='legacy') for r in roots]
        for entry in discover_models(dict(roots=live_roots,backend='openpi'))['models']:
            step_dir=Path(entry['path'])
            root=next((r for r in roots if step_dir.resolve().is_relative_to(r)),None)
            if root is not None:
                candidates.append((step_dir,entry['model_variant'],entry['arm_mode'],entry['config_name'],entry['experiment'],root,entry))
        for entry in artifact_inventory.models("openpi"):
            step_dir = Path(entry["path"])
            if entry.get("complete") is not True or not step_dir.name.isdigit():
                continue
            identity = artifact_checkpoint_identity(step_dir)
            if not identity:
                continue
            root = next((r for r in roots if step_dir.resolve().is_relative_to(r)), None)
            if root is not None:
                candidates.append((step_dir, identity["model_variant"], identity["arm_mode"], identity["config_name"], identity["experiment"], root, entry))
        for model_variant in ("pi05", "pi0"):
            for arm_mode in ("single", "bimanual"):
                config_name = policy_config_name(arm_mode, model_variant)
                for root in roots:
                    config_root = root / config_name
                    if not config_root.is_dir():
                        continue
                    for exp_dir in config_root.iterdir():
                        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                            continue
                        if (exp_dir / "checkpoints/index.json").is_file():
                            continue
                        for step_dir in exp_dir.iterdir():
                            if step_dir.is_dir() and step_dir.name.isdigit():
                                candidates.append((step_dir, model_variant, arm_mode, config_name, exp_dir.name, root, {}))
        with artifact_inventory.connect() as db:
            indexed_sizes = {row['path']: row['size'] for row in db.execute(
                "SELECT path,json_extract(data,'$.size_bytes') size FROM replicas WHERE status='present'")}
        measured_sizes = checkpoint_sizes_snapshot.get()['data'] or {}
        seen_paths: set[str] = set()
        for step_dir, model_variant, arm_mode, config_name, experiment, root, entry in candidates:
            if not (step_dir / "params").is_dir() or not (step_dir / "_CHECKPOINT_METADATA").is_file():
                continue
            visible_checkpoint, dataset_ids, dataset_origins = checkpoint_matches_visible_datasets(step_dir)
            if not visible_checkpoint:
                continue
            mtime = step_dir.stat().st_mtime
            cache_key = str(step_dir.resolve())
            if cache_key in seen_paths:
                continue
            seen_paths.add(cache_key)
            cached = checkpoint_size_cache.get(cache_key)
            size_bytes = entry.get("size_bytes") or indexed_sizes.get(cache_key) or measured_sizes.get(cache_key) or (cached[1] if cached and cached[0] == mtime else None)
            action_contract = checkpoint_action_contract(step_dir)
            checkpoints.append(
                {
                "path": cache_key,
                "backend": "openpi",
                "format": "orbax",
                "size_bytes": size_bytes,
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
                "finetuning": (action_contract or {}).get("finetuning") or FinetuningSettings.legacy().to_dict(),
                "config_name": config_name,
                "model_variant": model_variant,
                "arm_mode": arm_mode,
                "experiment": experiment,
                "step": int(entry.get("step") if entry.get("step") is not None else step_dir.name),
                "full_state_available": is_full_state_checkpoint(step_dir),
                "restore_capabilities": (["full_state"] if is_full_state_checkpoint(step_dir) else ["weights_only"]),
                "dataset_ids": dataset_ids,
                "dataset_origins": dataset_origins,
                "root": str(root),
                "storage": root == storage_root,
                "read_only": root != storage_root,
                "mtime": mtime,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                "size_gib": round(size_bytes / (1024**3), 3) if size_bytes is not None else None,
                }
            )
        from server_4090.model_identity import enrich
        return enrich(sorted(checkpoints, key=lambda item: (item["mtime"], item["step"]), reverse=True), config)

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


    @app.get("/api/gpus")
    def local_gpu_resources():
        gpus = gpu_inventory(strict=True)
        if not gpus:
            raise RuntimeError("nvidia-smi returned no GPU records")
        response = jsonify({"gpus": gpus})
        response.headers["Cache-Control"] = "no-store"
        return response

    def build_status_inventory():
        checkpoints = list_checkpoints()
        base_models = list_base_models(checkpoints)
        visible_experiments = {item.get("experiment") for item in checkpoints if item.get("experiment")}
        experiment_rows: dict[str, dict[str, Any]] = {}
        experiment_roots = [Path(config["checkpoint_base_dir"])] + [
            Path(item) for item in config.get("checkpoint_allowed_roots", [])
            if Path(item) != Path(config["checkpoint_base_dir"])
        ]
        for experiment_root in experiment_roots:
            for item in training_experiment_catalog(experiment_root):
                if item.get("name") not in visible_experiments:
                    continue
                current = experiment_rows.get(item["name"])
                if current is None:
                    experiment_rows[item["name"]] = dict(item)
                    continue
                for key in ("model_variants", "arm_modes", "config_names"):
                    current[key] = sorted(set(current.get(key, [])) | set(item.get(key, [])))
                current["checkpoint_count"] = int(current.get("checkpoint_count", 0)) + int(item.get("checkpoint_count", 0))
                current["latest_step"] = max(current.get("latest_step") or 0, item.get("latest_step") or 0) or None
                current["mtime"] = max(float(current.get("mtime", 0)), float(item.get("mtime", 0)))
                current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current["mtime"]))
        experiments = sorted(experiment_rows.values(), key=lambda item: (-float(item.get("mtime", 0)), item["name"]))
        return dict(checkpoints=checkpoints, base_models=base_models, experiments=experiments, datasets=list_datasets())

    status_inventory_snapshot = Snapshot(Path(config["workspace_root"]) / "status-inventory-index.json", build_status_inventory, ttl=30)

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
        if request.args.get("compact") == "1":
            cached_inventory = status_inventory_snapshot.get()
            inventory = cached_inventory["data"] if cached_inventory["ready"] else build_status_inventory()
        else:
            inventory = build_status_inventory()
        from server_4090.task_scenes import annotate, bindings, dataset_scene
        scene_bindings=bindings(config)
        checkpoints = [annotate(c,scene_bindings) for c in inventory["checkpoints"]]
        scene_datasets=[dict(d,task_scene=dataset_scene(d["id"],d,scene_bindings)) for d in inventory["datasets"]]
        from server_4090.model_catalog import is_foundation_path
        # Also filter snapshots written by older versions that mixed both categories.
        base_models = [m for m in inventory["base_models"] if m.get("foundation") and is_foundation_path(m.get("path")) and not m.get("experiment")]
        experiments = inventory["experiments"]
        return jsonify(
            {
                "datasets": scene_datasets,
                "checkpoints": checkpoints,
                "experiments": experiments,
                "base_models": base_models,
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
                    "dataset_read_roots": config.get("dataset_read_roots", [config["dataset_root"]]),
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
                    "training_min_free_gpu_mib": config.get("training_min_free_gpu_mib", 22_500),
                    "policy_allow_busy_gpus": config.get("policy_allow_busy_gpus", True),
                    "policy_min_free_gpu_mib": config.get("policy_min_free_gpu_mib", 12_000),
                    "policy_xla_memory_fraction": config.get("policy_xla_memory_fraction", 0.60),
                    "policy_xla_preallocate": config.get("policy_xla_preallocate", False),
                    "policy_rtc_enabled": config.get("policy_rtc_enabled", True),
                    "policy_rtc_max_execution_horizon": config.get(
                        "policy_rtc_max_execution_horizon", 12
                    ),
                    "policy_rtc_max_guidance_weight": config.get(
                        "policy_rtc_max_guidance_weight", 5.0
                    ),
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
        result = uploads.complete(upload_id)
        artifact_replicas.mark_dirty()
        return jsonify(result)

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
        result = dataset_editor.rename_dataset(dataset_id, new_dataset_id)
        artifact_replicas.mark_dirty()
        return jsonify(result)

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

    @app.get("/api/datasets")
    def dataset_main_list():
        return jsonify({"datasets": list_datasets()})

    @app.get("/api/transfers")
    def transfer_task_list():
        # This table must not wait for GPU discovery, checkpoint inventories,
        # or legacy destination-directory scans in the full status endpoint.
        return jsonify({"tasks": [task for task in tasks.list() if task.get("type") == "transfer"]})

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

    @app.get("/api/datasets/<dataset_id>/episodes/<int:episode_index>/analysis")
    def dataset_episode_analysis(dataset_id: str, episode_index: int):
        dataset_id = safe_name(dataset_id, "dataset id")
        max_points = safe_int(request.args.get("max_points", 1200), "max_points", 50, 5000)
        return jsonify(dataset_editor.episode_analysis(dataset_id, episode_index, max_points=max_points))

    @app.get("/api/datasets/<dataset_id>/episodes/<int:episode_index>/frames/<int:frame_index>")
    @app.get("/api/datasets/<dataset_id>/episodes/<int:episode_index>/frame/<int:frame_index>")
    @app.get("/api/datasets/<dataset_id>/episodes/<int:episode_index>/frame-data/<int:frame_index>")
    def dataset_episode_frame(dataset_id: str, episode_index: int, frame_index: int):
        dataset_id = safe_name(dataset_id, "dataset id")
        return jsonify(dataset_editor.episode_frame(dataset_id, episode_index, frame_index))

    @app.post("/api/datasets/<dataset_id>/episodes/<int:episode_index>/crop")
    @app.post("/api/datasets/<dataset_id>/episodes/<int:episode_index>/trim")
    def crop_dataset_episode(dataset_id: str, episode_index: int):
        dataset_id = safe_name(dataset_id, "dataset id")
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("crop request must be a JSON object")
        start = safe_int(payload.get("start_frame"), "start_frame", 0, 10**9)
        end = safe_int(payload.get("end_frame"), "end_frame", 0, 10**9)
        return jsonify(dataset_editor.crop_episode(dataset_id, episode_index, start, end))

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
        target_path = dataset_path_for_id(dataset_id)
        source_path = dataset_path_for_id(source_id)
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
        dataset_path = dataset_path_for_id(dataset_id)
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
        finetuning: FinetuningSettings | None = None,
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
        command += (finetuning or FinetuningSettings()).command_args()
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
        command = [
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
        if runtime_config.get("submit_host"):
            from server_4090.evaluation_snapshot import snapshot_command
            return snapshot_command(command, APP_DIR)
        return command

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
        dataset_path, writable = dataset_root_for_id(dataset_id)
        return resolve_episode_split(
            dataset_path.parent,
            dataset_id,
            test_ratio=test_ratio,
            seed=split_seed,
            contract=contract["contract_fingerprint"],
            persist=writable,
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
        dataset_path, writable = dataset_root_for_id(dataset_id)
        persisted = load_episode_split(
            dataset_path.parent, dataset_id, contract=fingerprint
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
            dataset_path.parent,
            dataset_id,
            test_ratio=test_ratio,
            seed=split_seed,
            contract=fingerprint,
            persist=writable,
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
        return list(dict.fromkeys([video_catalog.catalog_root(config), *[Path(item) for item in config.get("eval_video_roots", []) if item]]))

    def encode_video_id(path: Path) -> str:
        return base64.urlsafe_b64encode(str(path.absolute()).encode()).decode().rstrip("=")

    def decode_video_id(video_id: str) -> Path:
        padded = video_id + "=" * (-len(video_id) % 4)
        try:
            path = Path(os.path.abspath(base64.urlsafe_b64decode(padded.encode()).decode()))
        except Exception as exc:
            raise FileNotFoundError("invalid video id") from exc
        roots = [root.absolute() for root in eval_video_roots()]
        if not any(path == root or root in path.parents for root in roots):
            raise FileNotFoundError("video is outside configured roots")
        allowed_targets = [*roots, *[Path(s["root"]).expanduser().resolve() for s in video_catalog.configured_sources(config) if not s.get("ssh_host")]]
        allowed_targets.extend(Path(r).expanduser().resolve() for r in config.get("experiment_artifact_read_roots", []))
        resolved = path.resolve()
        if not any(resolved == r or r in resolved.parents for r in allowed_targets):
            raise FileNotFoundError("video target is outside configured roots")
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            raise FileNotFoundError("video not found")
        return path

    def prune_empty_eval_video_parents(path: Path) -> None:
        roots = [root.absolute() for root in eval_video_roots()]
        if not roots:
            return
        current = Path(path).expanduser().absolute().parent
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
        stored = video_catalog.read_object(video_catalog.metadata_path(path))
        if stored.get("episode_record"):
            row = stored["episode_record"]
            track = event_track_from_video_sidecar(path, row, stored.get("episode_sidecar"), task_name=str(stored.get("task", "")).replace("-", "_"))
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

    REMOTE_EVAL_VIDEO_INVENTORY_SCRIPT = Path(video_catalog.__file__).read_text(encoding="utf-8").split('\nif __name__ == "__main__":')[0] + "\n" + r'''
rows = []
for root in json.loads(os.environ.get("EVAL_VIDEO_ROOTS", "[]")):
    rows.extend(scan_source({"root": root}))
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
        path = Path(root) / relative_path
        metadata = video_catalog.classify(root, relative_path, {"storage_source": source})
        row = metadata.pop("episode_record", {})
        sidecar = metadata.pop("episode_sidecar", None)
        metadata["event_track"] = event_track_from_video_sidecar(path, row, sidecar, task_name=str(metadata.get("task", "")).replace("-", "_"))
        metadata["has_events"] = bool(metadata["event_track"])
        metadata["source"] = source
        return metadata

    def catalog_remote_videos():
        inventory = video_catalog.read_object(video_catalog.catalog_root(config) / "remote-inventory.json", max_bytes=128 * 1024 * 1024)
        rows = []
        for source, group in inventory.get("sources", {}).items():
            for original in group.get("videos", []):
                item = dict(original)
                item.pop("episode_record", None)
                item.pop("episode_sidecar", None)
                item.update(source=source, remote=True, playable=False, syncable=True, deletable=False,
                            id=base64.urlsafe_b64encode(f"{source}\0{item['root']}\0{item['relative_path']}".encode()).decode().rstrip("="),
                            updated_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item["mtime"])))
                rows.append(item)
        return rows, inventory.get("errors", {})

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
                + " " + shlex.quote(str(target.get("openpi_python") or target.get("python") or "/DATA/sync/sunny/miniconda3/bin/python")) + " - <<'REMOTE_EVAL_VIDEO_PY'\n"
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
                for key in ("score", "episode_status", "episode_seed"):
                    if key in row and row.get(key) is not None:
                        metadata[key] = row.get(key)
                videos.append({
                    **metadata,
                    **{k: v for k, v in row.items() if k not in {"episode_record", "episode_sidecar"}},
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

    from server_4090.artifact_inventory import Inventory as ArtifactInventory
    artifact_inventory = ArtifactInventory(config)

    def artifact_assert_idle(path):
        path = Path(path).resolve()
        if checkpoint_active_references(path):
            raise ValueError("Active tasks reference this checkpoint")
        for task in tasks.list():
            if task.get("state") in TERMINAL_STATES:
                continue
            for key in ("artifact_run_dir", "checkpoint", "checkpoint_dir", "rlinf_output_dir", "output_dir", "base_checkpoint", "resume_checkpoint"):
                value = (task.get("metadata") or {}).get(key)
                if not value:
                    continue
                ref = Path(str(value)).expanduser().resolve()
                if ref == path or ref in path.parents or path in ref.parents:
                    raise ValueError("Active task references this artifact")

    from server_4090.storage_catalog import Replicas, locations as artifact_replica_locations
    artifact_replicas = Replicas(config)

    def storage_task_finished(task):
        metadata = task.get('metadata') or {}
        if str(metadata.get('transfer_kind', '')).startswith('artifact_') and not str(metadata.get('transfer_kind', '')).startswith('artifact_checkpoint_'):
            return
        if task.get('type') not in {'train', 'eval', 'transfer', 'collection', 'norm'}:
            return
        paths = [metadata[k] for k in ('artifact_run_dir', 'output_dir', 'checkpoint_dir', 'dataset_path') if metadata.get(k)]
        artifact_replicas.invalidate_local_paths(paths)
        artifact_replicas.mark_dirty()
        status_inventory_snapshot.invalidate()
        artifact_history_snapshot.invalidate()
        rlinf_list_snapshot.invalidate()

    tasks.storage_change_callback = storage_task_finished

    def refresh_changed_storage(paths):
        status_inventory_snapshot.invalidate()
        artifact_history_snapshot.invalidate()
        rlinf_list_snapshot.invalidate()
        nodes = artifact_replicas.invalidate_local_paths(paths)
        if nodes: artifact_replicas.mark_dirty(nodes)
        for node in nodes:
            tasks.start("transfer", [config["openpi_python"], "-m", "server_4090.storage_catalog", "--config", str(config_file),
                "--action", "refresh", "--node", node, "--quick"], env=build_environment(config, None),
                metadata={"transfer_kind": "artifact_index_after_mutation", "gpu_required": False})

    @app.get("/api/checkpoints/discover")
    def discover_live_checkpoints():
        from server_4090.storage_worker import discover_models
        from server_4090.storage_catalog import storage_worker
        backend=request.args.get('backend','openpi')
        if backend not in {'openpi','rlinf'}: raise ValueError('Unknown backend')
        node=request.args.get('node','local_4090')
        available=artifact_replica_locations(config)
        if node not in available: raise ValueError('Unknown storage node')
        location=available[node]
        if location['local']:
            roots=[dict(path=str(r),layout='canonical') for r in artifact_inventory.roots() if r.name in {'simulation-artifacts','experiment_artifacts'} or str(r)==config.get('experiment_artifact_root')]
            roots += [dict(path=p,layout='legacy') for p in dict.fromkeys([config['checkpoint_base_dir'],*config.get('checkpoint_allowed_roots',[])])]
        else:
            roots=[dict(path=p,layout='canonical') for p in location['artifact_roots']]
            roots += [dict(path=p,layout='legacy') for p in dict.fromkeys([location['checkpoint_base_dir'],*location.get('checkpoint_read_roots',[])])]
        query=dict(action='discover_models',roots=roots,backend=backend,experiment=request.args.get('experiment',''))
        data=discover_models(query) if location['local'] else storage_worker(location,query)
        if backend=='openpi' and location['local']:
            data['models']=[row for row in data['models'] if checkpoint_matches_visible_datasets(Path(row['path']))[0]]
        for row in data['models']:
            row['node']=node
            if backend=='openpi' and location['local']:
                contract=checkpoint_action_contract(Path(row['path'])) or row.get('action_contract') or {}
                row['action_contract']=contract
                row['finetuning']=contract.get('finetuning') or FinetuningSettings.legacy().to_dict()
        from server_4090.task_scenes import annotate, bindings
        scene_bindings=bindings(config)
        from server_4090.model_identity import enrich
        data['models']=enrich([annotate(m,scene_bindings) for m in data['models']], config)
        response=jsonify(data);response.headers['Cache-Control']='no-store'
        return response

    @app.post("/api/checkpoints/inspect")
    def inspect_selected_checkpoint():
        from server_4090.selected_resource import inspect_checkpoint
        payload=request.get_json(force=True)
        path=Path(str(payload.get('path',''))).expanduser()
        allowed=[Path(p).resolve() for p in config.get('checkpoint_allowed_roots', [])]
        allowed += artifact_inventory.roots()
        allowed.append(Path(config['base_checkpoint']).resolve())
        resolved=path.resolve()
        if not path.is_absolute() or not any(resolved==r or r in resolved.parents for r in allowed):
            raise ValueError('Checkpoint is outside configured model roots')
        result=inspect_checkpoint(path)
        response=jsonify(result);response.headers['Cache-Control']='no-store'
        return response

    @app.post("/api/task-scenes/bind")
    def bind_task_scene():
        from server_4090.task_scenes import bind
        payload=request.get_json(force=True)
        scene=bind(config,payload.get('kind'),payload.get('key'),payload.get('task_scene'))
        status_inventory_snapshot.invalidate()
        return jsonify(task_scene=scene)

    @app.get("/api/model-catalog")
    def model_catalog():
        from server_4090.model_catalog import indexed_models
        return jsonify(models=indexed_models(artifact_replicas, request.args.get("backend"), request.args.get("q", "")))

    @app.post("/api/checkpoints/operate")
    def operate_selected_checkpoint():
        from server_4090.checkpoint_operations import source_location
        payload=request.get_json(force=True)
        action=payload.get('action');backend=payload.get('backend')
        if action not in {'copy','delete'} or backend not in {'openpi','rlinf'}:raise ValueError('Invalid checkpoint operation')
        path=str(payload.get('path') or '')
        location,_=source_location(config,payload.get('node'),path)
        if payload.get('confirm_path')!=path:raise ValueError('Confirm the exact checkpoint path')
        if location['local']:artifact_assert_idle(path)
        target=payload.get('target')
        if action=='copy' and (target not in artifact_replica_locations(config) or target==location['name']):raise ValueError('Choose another configured destination')
        for task in tasks.list():
            metadata=task.get('metadata',{})
            if task.get('state') not in TERMINAL_STATES and metadata.get('source_path')==path and metadata.get('source')==location['name']:
                raise ValueError('An operation on this checkpoint is already running')
        command=[config['openpi_python'],'-m','server_4090.checkpoint_operations','--config',str(config_file),
                 '--node',location['name'],'--path',path,'--backend',backend,'--action',action]
        if action=='copy':command+=['--target',target]
        return jsonify(tasks.start('transfer',command,env=build_environment(config,None),metadata={
            'transfer_kind':'artifact_checkpoint_'+action,'source':location['name'],'source_path':path,
            'target':target,'checkpoint_dir':path,'backend':backend,'gpu_required':False})),202

    @app.get("/api/storage-locations")
    def storage_locations_only():
        return jsonify(locations=[dict(id=k,label=v['label'],local=v['local'],artifact_root=v.get('artifact_root'))
            for k,v in artifact_replica_locations(config).items()])

    @app.get("/api/storage-items")
    def list_artifact_replicas():
        return jsonify(artifact_replicas.page(int(request.args.get("offset", 0)), int(request.args.get("limit", 25)),
            request.args.get("kind", "all"), False, request.args.get("q", ""), request.args.get("node", "all"), request.args.get("area", "all"), request.args.get("backend", "all")))

    @app.get("/api/storage-items/<item_id>/files")
    def storage_item_files(item_id):
        item = artifact_replicas.get(item_id)
        offset = int(request.args.get("offset", 0)); limit = int(request.args.get("limit", 100))
        if not 0 <= offset <= 10000000 or not 1 <= limit <= 200:
            raise ValueError("Invalid file page")
        files = item.get("files", [])
        return jsonify(files=files[offset:offset + limit], total=len(files), offset=offset, limit=limit,
                       path=item["path"], updated_at=item["updated_at"], sampled=bool(item.get("sampled_file_list")))

    @app.post("/api/storage-items/refresh")
    def refresh_artifact_replicas():
        source = (request.get_json(silent=True) or {}).get("source", "all")
        if source != "all" and source not in artifact_replica_locations(config):
            raise ValueError("Unknown storage source")
        scope = ["--local-only"] if source == "all" else ["--node", source]
        for task in tasks.list():
            if task.get("state") not in TERMINAL_STATES and task.get("metadata", {}).get("transfer_kind") == "artifact_replica_inventory":
                return jsonify(task), 202
        return jsonify(tasks.start("transfer", [config["openpi_python"], "-m", "server_4090.storage_catalog",
            "--config", str(config_file), "--action", "refresh", *scope], env=build_environment(config, None),
            metadata={"transfer_kind": "artifact_replica_inventory", "gpu_required": False})), 202

    @app.post("/api/storage-items/operate")
    def operate_artifact_replica():
        payload = request.get_json(force=True)
        action = payload.get("action")
        if action not in {"copy", "delete"}:
            raise ValueError("action must be copy or delete")
        copy = artifact_replicas.get(payload.get("copy_id"))
        if copy["kind"] != "dataset" and payload.get("area") != ("training" if copy["kind"] in {"model", "logs", "metrics", "metadata"} else "rollout"):
            raise ValueError("Use the corresponding training or rollout management area")
        if action == "delete" and payload.get("confirm_path") != copy["path"]:
            raise ValueError("Confirm the exact replica path")
        available = artifact_replica_locations(config)
        if copy["node"] not in available:
            raise ValueError("Source storage is no longer configured")
        if available[copy["node"]]["local"]:
            artifact_assert_idle(copy["path"])
        command = [config["openpi_python"], "-m", "server_4090.storage_catalog", "--config", str(config_file),
                   "--action", action, "--copy-id", copy["id"]]
        if action == "copy":
            if payload.get("target") not in available:
                raise ValueError("Unknown target storage")
            target = available[payload["target"]]
            target_root = target.get("dataset_root") if copy["kind"] == "dataset" else target["artifact_root"]
            if not target_root:
                raise ValueError("Target does not accept this artifact type")
            if target["local"]:
                artifact_assert_idle(Path(target_root) / copy["relative_path"])
            command.extend(["--target", payload["target"]])
        kinds = payload.get("kinds")
        if kinds is not None:
            if not kinds or copy["kind"] != "rollout" or not isinstance(kinds, list) or not set(kinds) <= {"video", "trajectory", "image", "feature"}:
                raise ValueError("Invalid selected file types")
            command.extend(["--kinds", json.dumps(kinds)])
        for task in tasks.list():
            if task.get("state") not in TERMINAL_STATES and task.get("metadata", {}).get("replica_id") == copy["id"]:
                raise ValueError("An operation on this replica is already running")
        return jsonify(tasks.start("transfer", command, env=build_environment(config, None), metadata={
            "transfer_kind": "artifact_replica_" + action, "replica_id": copy["id"], "source_path": copy["path"],
            "replica_target": payload.get("target"), "source": copy["node"], "target": payload.get("target"), "gpu_required": False})), 202

    @app.get("/api/rollout-artifacts/groups")
    def rollout_artifact_groups():
        return jsonify(artifact_inventory.groups())

    @app.get("/api/rollout-artifacts")
    def rollout_artifact_page():
        return jsonify(artifact_inventory.page(request.args.getlist("scope"),
            int(request.args.get("offset", 0)), int(request.args.get("limit", 25)),
            request.args.get("phase", "all"), request.args.get("kind", "all")))

    @app.get("/api/training-artifacts/models")
    def training_artifact_models():
        return jsonify(models=artifact_inventory.models(request.args.get("backend")), **artifact_inventory.status())

    @app.post("/api/rollout-artifacts/batch-delete")
    def rollout_artifact_delete():
        payload = request.get_json(force=True)
        with tasks.lock:
            result = artifact_inventory.delete_files(payload.get("ids"), artifact_assert_idle)
        refresh_changed_storage([artifact_inventory.get(r["id"])["path"] for r in result["results"] if r["deleted"]])
        return jsonify(result)

    @app.post("/api/training-artifacts/models/batch-delete")
    def training_artifact_delete():
        payload = request.get_json(force=True)
        with tasks.lock:
            result = artifact_inventory.delete_files(payload.get("ids"), artifact_assert_idle, models=True)
        refresh_changed_storage([artifact_inventory.get(r["id"])["path"] for r in result["results"] if r["deleted"]])
        return jsonify(result)

    @app.post("/api/artifact-index/refresh")
    def refresh_artifact_inventory():
        for task in tasks.list():
            if task.get("state") not in TERMINAL_STATES and task.get("metadata", {}).get("transfer_kind") == "artifact_inventory":
                return jsonify(task), 202
        return jsonify(tasks.start("transfer", [config["openpi_python"], "-m", "server_4090.artifact_inventory",
            "--config", str(config_file)], env=build_environment(config, None),
            metadata={"transfer_kind": "artifact_inventory", "gpu_required": False})), 202

    @app.post("/api/artifact-runs")
    def allocate_artifact_run():
        payload = request.get_json(force=True)
        fields = experiment_artifacts.identity(payload.get("video_source"), payload.get("task"), payload.get("experiment"),
                                               payload.get("run_id") or "run-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        backend, kind = payload.get("backend"), payload.get("kind")
        if not isinstance(backend, str) or not SAFE_NAME.fullmatch(backend) or not isinstance(kind, str) or not SAFE_NAME.fullmatch(kind):
            raise ValueError("backend and kind must be safe names")
        root = Path(config.get("experiment_artifact_root") or Path(config["workspace_root"]) / "experiment_artifacts")
        directory = experiment_artifacts.run_directory(root, fields)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise ValueError("artifact run already exists; use a new run_id") from None
        row = experiment_artifacts.manifest(fields, backend=backend, kind=kind, state="created")
        for name in row["artifacts"].values():
            (directory / name).mkdir(exist_ok=True)
        video_catalog.atomic_write(directory / "run.json", row)
        artifact_replicas.mark_dirty(["local_4090"])
        artifact_history_snapshot.invalidate()
        return jsonify(**row, path=str(directory), manifest_path=str(directory / "run.json")), 201

    def build_artifact_run_rows():
        path = Path(config["workspace_root"]) / "artifact-run-inventory.json"
        data = video_catalog.read_object(path, max_bytes=128 * 1024 * 1024)
        rows = list(data.get("runs", []))
        root = Path(config.get("experiment_artifact_root") or Path(config["workspace_root"]) / "experiment_artifacts")
        for path in root.glob("*/*/*/*/run.json"):
            row = video_catalog.read_object(path)
            if row.get("schema") == experiment_artifacts.SCHEMA:
                rows.append({**row, "path": str(path.parent)})
        for task in tasks.list():
            if task.get("type") != "train":
                continue
            meta = task.get("metadata", {})
            if meta.get("artifact_manifest"):
                row = video_catalog.read_object(Path(meta["artifact_manifest"]))
                if row:
                    rows.append({**row, "state": task["state"], "task_id": task["id"], "path": meta.get("artifact_run_dir")})
        unique = {row.get("path") or row.get("run_id"): row for row in rows}
        rows = list(unique.values())
        return {"runs": rows, "updated_at": data.get("updated_at")}

    artifact_history_snapshot = Snapshot(Path(config["workspace_root"]) / "artifact-history-index.json",
                                         build_artifact_run_rows, ttl=30)

    @app.post("/api/artifact-runs/delete-history")
    def delete_artifact_history():
        payload = request.get_json(force=True)
        rows = build_artifact_run_rows()['runs']
        run = next((r for r in rows if r.get('path') == payload.get('path')), None)
        if not run: raise ValueError('Unknown run history')
        task_id = run.get('task_id') or run.get('origin', {}).get('dashboard_id')
        if task_id and tasks._path(task_id).is_file():
            tasks.delete_many([task_id], validate_only=True)
        histories.cleanup(histories.run_operations(run))
        if task_id and tasks._path(task_id).is_file(): tasks.delete(task_id)
        histories.remember([run])
        artifact_history_snapshot.invalidate()
        artifact_replicas.mark_dirty()
        return jsonify(deleted=True, run_id=run.get('run_id'))

    @app.get("/api/artifact-runs")
    def artifact_runs():
        snapshot = artifact_history_snapshot.get()
        data = snapshot["data"] or {"runs": []}
        deleted = histories.deleted()
        rows = [row for row in data["runs"] if (row.get('path') or row.get('run_id')) not in deleted]
        for field in ("source", "task", "backend", "kind", "state", "experiment"):
            value = request.args.get(field, "")
            if value and value != "all":
                rows = [r for r in rows if r.get(field) == value]
        return jsonify(schema=experiment_artifacts.SCHEMA, runs=rows, total=len(rows), updated_at=snapshot["updated_at"], ready=snapshot["ready"], refreshing=snapshot["refreshing"])

    try:
        from .eval_video_browser import VideoBrowser
    except ImportError:
        from eval_video_browser import VideoBrowser
    video_browser = VideoBrowser(video_catalog.catalog_root(config))

    @app.get("/api/eval-videos/groups")
    def eval_video_groups():
        return jsonify(video_browser.groups())

    @app.get("/api/eval-videos")
    def list_eval_videos():
        if request.args.get("browse") == "1":
            result = video_browser.page(request.args)
            for item in result["videos"]:
                if item.get("remote"):
                    continue
                try:
                    decode_video_id(item["id"])
                    # Only this page gets per-episode event/annotation details.
                    detail = eval_video_metadata(item["root"], item["relative_path"])
                    routing = {"id", "group_id", "root", "relative_path", "url", "source", "remote", "playable", "syncable", "deletable",
                               "video_source", "task", "experiment", "experiment_label", "mtime", "updated_at",
                               "artifact_experiment", "artifact_experiment_label", "artifact_role", "training_run_id",
                               "training_run_ref", "lineage_evidence"}
                    item.update({key: value for key, value in detail.items() if key not in routing})
                except (FileNotFoundError, OSError):
                    item.update(playable=False, deletable=False, missing=True)
            return jsonify(result)
        limit = safe_int(request.args.get("limit", 200), "limit", 1, 1000)
        offset = safe_int(request.args.get("offset", 0), "offset", 0, 10000000)
        include_remote = str(request.args.get("include_remote", "")).lower() in {"1", "true", "yes"}
        destination = video_catalog.catalog_root(config).absolute()
        registry = video_catalog.read_object(destination / "registry.json", max_bytes=128 * 1024 * 1024)
        videos, seen = [], set()
        for root in eval_video_roots():
            if not root.exists():
                continue
            for path in video_catalog.walk_videos(root):
                identity = str(path.absolute())
                if identity in registry or identity in seen:
                    continue
                seen.add(identity)
                stat = path.stat()
                rel = path.relative_to(root).as_posix()
                videos.append({
                    **eval_video_metadata(root, rel), "id": encode_video_id(path), "name": path.name,
                    "relative_path": rel, "root": str(root), "size_mib": round(stat.st_size / 1024**2, 2),
                    "mtime": stat.st_mtime, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                    "url": f"/api/eval-videos/{encode_video_id(path)}", "remote": False, "playable": True,
                    "syncable": False, "deletable": True,
                    "file_identity": dict(host=socket.gethostname(), device=stat.st_dev, inode=stat.st_ino,
                                          size=stat.st_size, mtime_ns=stat.st_mtime_ns),
                    "_file_signature": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns],
                })
        remote_videos, remote_errors = catalog_remote_videos()
        # Hide cached originals once their canonical local copy is available.
        synced = {(v.get("provenance", {}).get("storage_source"), v.get("provenance", {}).get("root"), v.get("provenance", {}).get("relative_path")) for v in videos}
        videos.extend(v for v in remote_videos if (v["source"], v["root"], v["relative_path"]) not in synced)
        if include_remote:
            extra, errors = list_remote_eval_videos()
            remote_errors.update(errors)
            videos.extend(extra)
        # Facets cover the inventory, independent of active filters.
        facets = {key: sorted({str(v.get(field) or "unknown") for v in videos}) for key, field in
                  (("video_sources", "video_source"), ("tasks", "task"), ("experiments", "experiment"), ("phases", "phase"), ("storage_sources", "source"))}
        def video_matches(item):
            events = request.args.get("events", "all")
            if events in {"with", "without"} and bool(item.get("has_events")) != (events == "with"):
                return False
            for key in ("video_source", "phase", "source", "success", "safe_success"):
                value = str(request.args.get(key, "")).strip().lower()
                if value and value != "all" and value != str(item.get(key, "unknown")).lower():
                    return False
            for key in ("task", "experiment", "run_id", "checkpoint_id"):
                value = str(request.args.get(key, "")).strip().lower().replace("_", "-")
                haystack = " ".join(str(item.get(k, "")) for k in (key, "experiment_label") if k == key or key == "experiment").lower().replace("_", "-")
                if value and value not in haystack:
                    return False
            query = str(request.args.get("q", "")).strip().lower()
            return not query or query in " ".join(str(item.get(k, "")) for k in ("relative_path", "catalog_relative_path", "name", "task", "experiment", "experiment_label", "run_id", "checkpoint_id", "source", "video_source", "instruction")).lower()
        from server_4090.eval_video_identity import deduplicate_videos, cached_identities
        filtered = cached_identities([v for v in videos if video_matches(v)], destination)
        # The legacy endpoint shares identity rules; content hashing remains
        # exclusively in the background publisher.
        videos = deduplicate_videos(filtered)
        videos.sort(key=lambda v: (-v["mtime"], v["id"]))
        facets["success"] = {k: sum(v.get("success") == k for v in videos) for k in ("success", "failed", "unknown")}
        return jsonify(videos=videos[offset:offset + limit], total=len(videos), duplicate_count=len(filtered)-len(videos), offset=offset, limit=limit,
                       facets=facets, roots=[str(r) for r in eval_video_roots()], remote_errors=remote_errors,
                       catalog_status=video_catalog.read_object(destination / "status.json"))

    @app.post("/api/eval-videos/catalog/refresh")
    def refresh_eval_video_catalog():
        return jsonify(tasks.start("transfer", [config["openpi_python"], str(APP_DIR / "eval_video_catalog_runner.py"),
                       "--config", str(config_file), "--apply"], env=build_environment(config, None),
                       metadata={"transfer_kind": "eval_video_catalog", "gpu_required": False})), 201

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
        roots = [root.absolute() for root in eval_video_roots()]
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
            video_catalog.metadata_path(path).unlink(missing_ok=True)
            event_override_path_for_video(path).unlink(missing_ok=True)
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

        video_browser.discard([item["id"] for item in deleted])
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
        parallelism = safe_int(payload.get("parallelism", config.get("transfer_parallelism", 4)), "parallelism", 1, 16)
        rel = PurePosixPath(relative_path)
        if rel.is_absolute() or ".." in rel.parts or not rel.name or Path(rel.name).suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("invalid remote video relative_path")
        catalog_sources = {s["storage_source"]: s for s in video_catalog.configured_sources(config) if s.get("ssh_host")}
        target = config.get("cluster_targets", {}).get(source)
        metadata = None
        if source in catalog_sources:
            item = catalog_sources[source]
            if root != item["root"] or item.get("access_mode") == "slurm_only":
                raise ValueError("remote video root is not configured for SSH inventory")
            host = item["ssh_host"]
            inventory = video_catalog.read_object(video_catalog.catalog_root(config) / "remote-inventory.json", max_bytes=128 * 1024 * 1024)
            metadata = next((r for r in inventory.get("sources", {}).get(source, {}).get("videos", [])
                             if r["root"] == root and r["relative_path"] == relative_path), None)
            if metadata is None:
                raise ValueError("video is absent from the configured inventory; refresh the catalog")
        elif target and target.get("access_mode") != "slurm_only" and root in remote_eval_video_roots(target):
            host = target.get("submit_host")
            remote, _ = list_remote_eval_videos()
            metadata = next((r for r in remote if r["source"] == source and r["root"] == root and r["relative_path"] == relative_path), None)
        else:
            raise ValueError("unknown or Slurm-only video source/root")
        if not metadata or not host:
            raise ValueError("remote video was not found")
        metadata = {**metadata, "provenance": {"storage_source": source, "root": root, "relative_path": relative_path}}
        metadata["catalog_relative_path"] = video_catalog.catalog_relative_path(metadata, rel.name)
        destination = video_catalog.catalog_root(config)
        target_path = destination / metadata["catalog_relative_path"]
        command = [config["openpi_python"], str(APP_DIR / "video_transfer_runner.py"),
                   "--source-name", source, "--source-host", str(host), "--source-root", root,
                   "--relative-path", relative_path, "--target-root", str(destination),
                   "--catalog-relative-path", metadata["catalog_relative_path"], "--metadata-json", json.dumps(metadata, ensure_ascii=False),
                   "--parallelism", str(parallelism)]
        if payload.get("overwrite", False):
            command.append("--overwrite")
        return jsonify(tasks.start("transfer", command, env=build_environment(config, None), metadata={
            "transfer_kind": "eval_video", "source": source, "target": "local_4090",
            "source_path": str(PurePosixPath(root) / rel), "target_path": str(target_path),
            "parallelism": parallelism, "slurm_involved": False})), 201

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
                "inventory_python": target.get("inventory_python") or target.get("openpi_python"),
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
                "openpi_python": target.get("openpi_python"),
                "workdir": target.get("workdir"),
                "log_dir": target.get("log_dir"),
                "remote_job_dir": target.get("remote_job_dir"),
                "cache_root": target.get("cache_root"),
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
                "openpi_python": target.get("openpi_python"),
                "inventory_python": target.get("inventory_python"),
                "log_transport": target.get("log_transport", "sattach"),
                "cache_root": target.get("cache_root") or str(PurePosixPath(str(target.get("workdir") or "/tmp")).parent / "cache"),
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
                if not isinstance(rows, list):
                    return [], f"invalid dataset list in inventory cache from {source_label}"
                return rows, None
            if isinstance(payload, list):
                return payload, None
            return [], f"unsupported inventory cache format from {source_label}"
        host = location.get("host")
        if not host:
            return [], "missing ssh host"
        env = f"DATASET_ROOT={shlex.quote(str(location['dataset_root']))}"
        inventory_python = location.get("inventory_python") or location.get("openpi_python")
        if not inventory_python:
            return [], "missing configured inventory Python interpreter"
        command = f"{env} {shlex.quote(str(inventory_python))} - <<'REMOTE_DATASET_PY'\n{REMOTE_DATASET_INVENTORY_SCRIPT}\nREMOTE_DATASET_PY"
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

    def grouped_dataset_locations(*, origin_filter: str | None = None, source: str | None = None) -> dict[str, Any]:
        locations = dataset_location_configs()
        if source and source not in locations:
            raise ValueError("Unknown dataset source")
        groups: dict[str, dict[str, Any]] = {}
        errors = {}
        cache_root = Path(config["workspace_root"]) / "dataset-location-cache"
        for name, location in locations.items():
            cache = cache_root / (safe_name(name, "source") + ".json")
            if location.get("kind") in {"local", "local_archive"} or name == source:
                rows, error = remote_dataset_inventory(location)
                if not error and location.get("kind") not in {"local", "local_archive"}:
                    cache_root.mkdir(parents=True, exist_ok=True)
                    temporary = cache.with_suffix("." + uuid.uuid4().hex + ".tmp")
                    temporary.write_text(json.dumps(rows), encoding="utf-8")
                    temporary.replace(cache)
            else:
                rows, error = [], None
                fixed = cache
                configured = location.get("inventory_cache_path")
                if not fixed.is_file() and configured and location.get("inventory_cache_host") in {None, "local", "local_4090", "4x4090"}:
                    fixed = Path(configured).expanduser()
                if fixed.is_file():
                    try:
                        payload = json.loads(fixed.read_text(encoding="utf-8"))
                        rows = payload.get("datasets", []) if isinstance(payload, dict) else payload
                    except (OSError, ValueError):
                        rows = []
                if not rows:
                    with artifact_replicas.inventory.connect() as db:
                        indexed = db.execute("SELECT data FROM replicas WHERE node=? AND status IN ('present','stale') AND json_extract(data,'$.kind')='dataset'", (name,)).fetchall()
                    for indexed_row in indexed:
                        item = json.loads(indexed_row['data'])
                        rows.append(dict(id=Path(item['path']).name, path=item['path'], origin=item.get('origin', 'unknown'), mtime=item.get('updated_at', 0)))
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
        return {"datasets": datasets, "local_datasets": list_datasets(), "locations": locations, "errors": errors}

    @app.get("/api/dataset-locations")
    def dataset_locations():
        origin = request.args.get("origin")
        if origin:
            origin = normalize_dataset_origin(origin)
        return jsonify(grouped_dataset_locations(origin_filter=origin, source=request.args.get("source")))

    @app.post("/api/datasets/<dataset_id>/delete-copy")
    def delete_dataset_copy(dataset_id: str):
        dataset_id = safe_name(dataset_id, "dataset id")
        payload = request.get_json(force=True)
        if not isinstance(payload, dict) or payload.get("confirm_dataset_id") != dataset_id:
            raise ValueError("confirm_dataset_id must exactly match the dataset id")
        target_name = str(payload.get("target", ""))
        location = dataset_location_configs().get(target_name)
        if not location or not location.get("available", True):
            raise ValueError("unknown or unavailable dataset location")
        expected_episodes = payload.get("expected_episodes")
        if expected_episodes is not None:
            expected_episodes = safe_int(expected_episodes, "expected_episodes", 1, 1000000)
        with tasks.lock:
            assert_dataset_idle(dataset_id)
            if target_name == "local_4090":
                # Explicitly delete the storage copy even if another read root
                # with the same ID happens to appear earlier in the config.
                target_path = Path(config["dataset_root"]) / dataset_id
                if target_path.is_symlink() or target_path.resolve().parent != Path(config["dataset_root"]).resolve():
                    raise ValueError("dataset path escapes configured storage root")
                if not (target_path / "meta/info.json").is_file():
                    raise FileNotFoundError("local dataset copy is not installed")
                if expected_episodes is not None and read_json(target_path / "meta/info.json", {}).get("total_episodes") != expected_episodes:
                    raise ValueError("dataset episode count changed; deletion refused")
                return jsonify({**dataset_editor.delete_dataset(dataset_id), "target": target_name})
            if location.get("kind") not in {"local", "local_archive"}:
                if not location.get("openpi_python") or not location.get("partition") or not location.get("node"):
                    raise ValueError("remote dataset deletion requires a configured Slurm target and Conda Python")
            command = [config["openpi_python"], str(APP_DIR / "dataset_delete_runner.py"),
                       "--dataset-id", dataset_id, "--location-json", json_arg(location)]
            if expected_episodes is not None:
                command.extend(["--expected-episodes", str(expected_episodes)])
            task = tasks.start("transfer", command, env=build_environment(config, None), metadata={
                "dataset_id": dataset_id, "transfer_kind": "dataset_delete", "target": target_name,
                "target_path": str(Path(location["dataset_root"]) / dataset_id) if location.get("kind") in {"local", "local_archive"} else str(PurePosixPath(location["dataset_root"]) / dataset_id),
                "execution_target": target_name, "runtime": "local_4090", "source": target_name,
            })
        return jsonify({"task": task, "dataset_id": dataset_id, "target": target_name}), 202


    @app.get("/api/cluster-resources")
    def cluster_resources():
        from server_4090.extension_nodes import resource_snapshot
        started = time.time()
        result = resource_snapshot(config.get("cluster_targets", {}),
            all_jobs=str(request.args.get("all_jobs", "")).lower() in {"1", "true", "yes"},
            native=str(request.args.get("native", "")).lower() in {"1", "true", "yes"})
        result["elapsed_s"] = round(time.time() - started, 3)
        return jsonify(result)

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

    extension_nodes_lock = threading.Lock()

    @app.get("/api/settings/extension-nodes")
    def get_extension_nodes():
        return jsonify({"nodes": config.get("cluster_targets", {})})

    def check_node_unused(name):
        for task in tasks.list():
            if task.get("state") not in PROCESS_STATES | WAITING_STATES:
                continue
            metadata = task.get("metadata", {})
            if name in [metadata.get(key) for key in ("execution_target", "cluster_target", "slurm_target", "source", "target")]:
                raise ValueError(f"node {name} is used by active task {task['id']}; edit after it finishes")

    @app.route("/api/settings/extension-nodes/<name>", methods=["PUT", "DELETE"])
    def save_extension_node(name):
        nonlocal remote_checkpoints
        from server_4090.extension_nodes import validate_node
        with extension_nodes_lock:
            targets = dict(config.get("cluster_targets", {}))
            check_node_unused(name)
            if request.method == "DELETE":
                if name not in targets:
                    raise ValueError("unknown extension node")
                del targets[name]
            else:
                payload = request.get_json(force=True)
                if not isinstance(payload, dict):
                    raise ValueError("node settings must be an object")
                targets[name] = validate_node(name, {**targets.get(name, {}), **payload})
            persist_settings({"cluster_targets": targets})
            config["cluster_targets"] = targets
            previous = remote_checkpoints
            remote_checkpoints = RemoteCheckpointInventory(targets)
            previous.pool.shutdown(wait=False)
        return jsonify({"ok": True, "nodes": targets, "restart_required": False})

    @app.get("/api/checkpoints/remote")
    def list_remote_checkpoints():
        locations = remote_checkpoints.snapshot(force=request.args.get("refresh") == "1", source=request.args.get("source"))
        from server_4090.model_catalog import indexed_models
        fixed_models = indexed_models(artifact_replicas, 'openpi')
        for location in locations:
            if not location['checkpoints']:
                location['checkpoints'] = [dict(m, config_name=Path(m['path']).parent.parent.name,
                    model_variant='pi05' if 'pi05' in m['path'] else 'pi0')
                    for m in fixed_models if m.get('node') == location['source']]
                location['index_mode'] = 'fixed'
        # Keep real/simulation Dashboard provenance filtering consistent.
        for location in locations:
            location["checkpoints"] = [row for row in location["checkpoints"]
                if any(origin in visible_dataset_origin_set() for origin in checkpoint_dataset_origins(row["dataset_ids"]).values())
                or (not row["dataset_ids"] and "unknown" in visible_dataset_origin_set())]
        return jsonify({"locations": locations})

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
        if payload.get("step") is not None:
            if target_name != "local_4090" or source_name not in config.get("cluster_targets", {}):
                raise ValueError("single-step sync supports cluster to local_4090")
            step = safe_int(payload["step"], "checkpoint step", 0, 10**9)
            row = remote_checkpoints.selected(source_name, config_name, exp_name, step)
            contract = row.get("action_contract")
            if not contract:
                raise ValueError("checkpoint has no action contract; cannot prepare policy deployment")
            source, target = dict(locations[source_name]), dict(locations[target_name])
            marker_path = str(PurePosixPath(target["dataset_root"]) / ".policy_action_conventions" / (exp_name + ".json"))
            source["dataset_root"] = str(PurePosixPath(row["root"]) / config_name / exp_name)
            target["dataset_root"] = str(PurePosixPath(target["dataset_root"]) / exp_name)
            if source.get("kind") == "slurm_only":
                staging = source.get("nas_dataset_staging_root") or config["nas_checkpoint_staging_root"]
                source["nas_dataset_staging_root"] = str(PurePosixPath(staging) / source_name / config_name / exp_name)
            for existing in tasks.list():
                meta = existing.get("metadata", {})
                if (existing.get("type") == "transfer" and existing.get("state") in PROCESS_STATES
                        and meta.get("transfer_kind") == "checkpoint"
                        and (meta.get("source"), meta.get("config_name"), meta.get("exp_name"), meta.get("checkpoint_step"))
                        == (source_name, config_name, exp_name, step)):
                    return jsonify(existing), 200
            transfer_request = dict(source=source, target=target, step=step, contract=contract,
                                    marker_path=marker_path, parallelism=parallelism,
                                    nas_staging_root=config.get("nas_checkpoint_staging_root"))
            task = tasks.start("transfer", [config["openpi_python"], str(APP_DIR / "checkpoint_sync_runner.py"),
                               "--request-json", json_arg(transfer_request)], env=build_environment(config, None),
                               metadata={"transfer_kind": "checkpoint", "config_name": config_name,
                                         "exp_name": exp_name, "checkpoint_step": step,
                                         "source": source_name, "target": target_name,
                                         "source_path": str(PurePosixPath(source["dataset_root"]) / str(step)),
                                         "target_path": str(PurePosixPath(target["dataset_root"]) / str(step)),
                                         "parallelism": parallelism})
            return jsonify(task), 201
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
                status_inventory_snapshot.invalidate()
                artifact_inventory.invalidate_missing(path)
                artifact_replicas.invalidate_local_paths([path])
                artifact_replicas.mark_dirty()
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
            f"bin/bimanual-vla data-upload LEROBOT_OR_GUI_NPZ_DIR --name {dataset_id} "
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
        base_checkpoint, model_variant = resolve_norm_model(payload)
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
            env=build_environment(config, None, **dataset_env_kwargs(dataset_id)),
            metadata={
                "dataset_id": dataset_id,
                "dataset_root_path": str(dataset_root_for_id(dataset_id)[0].parent),
                "dataset_read_only": not dataset_root_for_id(dataset_id)[1],
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

    def validate_weight_identity(path, dataset_contract):
        from server_4090.model_catalog import is_foundation_path
        from server_4090.model_identity import enrich, mismatch
        if is_foundation_path(path):
            return
        marker=checkpoint_action_contract(Path(path)) or {}
        assets=Path(path)/'assets'
        ids=[p.name for p in assets.iterdir() if p.is_dir()] if assets.is_dir() else []
        if marker.get('dataset_id') and marker['dataset_id'] not in ids:ids.append(marker['dataset_id'])
        model=enrich([dict(action_contract=marker,dataset_ids=ids)],config)[0]
        reason=mismatch(model,dataset_contract)
        if reason:raise ValueError('所选 checkpoint 不兼容：'+reason)

    @app.post("/api/tasks/train")
    def start_train():
        payload = request.get_json(force=True)
        dataset_id, arm_mode, arm_side, schema, dataset_contract = parse_dataset(payload)
        base_checkpoint, model_variant = resolve_base_model(payload)
        from server_4090.task_scenes import dataset_scene, bindings
        scene=dataset_scene(dataset_id,read_json(dataset_root_for_id(dataset_id)[0]/"meta/info.json") or {},bindings(config))
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
        if effective_mode in {'new','overwrite'}:
            validate_weight_identity(base_checkpoint, action_contract_for_model(dataset_contract))
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

        validate_weight_identity(resume_checkpoint or (checkpoint_dir if saved_steps and effective_mode == 'resume' else base_checkpoint), model_contract)
        settings_source = checkpoint_action_contract(resume_checkpoint) if resume_checkpoint is not None else marker
        requested_finetuning = {key: payload.get(key) for key in FinetuningSettings.__dataclass_fields__}
        weights = requested_finetuning["rtc_training_delay_weights"]
        if isinstance(weights, str):
            requested_finetuning["rtc_training_delay_weights"] = json.loads(weights) if weights.strip() else None
        for key in ("llm_lora", "rtc_training", "train_vision", "train_llm"):
            value = requested_finetuning[key]
            if value is not None and not isinstance(value, bool):
                if str(value).strip().lower() not in {"true", "false", "1", "0"}:
                    raise ValueError(f"{key} must be a boolean")
                requested_finetuning[key] = str(value).strip().lower() in {"true", "1"}
        if requested_finetuning["rtc_training_max_delay"] is not None:
            requested_finetuning["rtc_training_max_delay"] = safe_int(
                requested_finetuning["rtc_training_max_delay"], "rtc_training_max_delay", 1, 49)
        is_resume = resume_kind != "none"
        finetuning = FinetuningSettings.resolve(requested_finetuning,
            (settings_source or {}).get("finetuning"), legacy=is_resume, resume=is_resume)
        finetuning.validate_model(model_variant)
        if not is_resume:
            initialization_marker = checkpoint_action_contract(base_checkpoint)
            if initialization_marker:
                initial_settings = FinetuningSettings.resolve({}, initialization_marker.get("finetuning"), legacy=True)
                if (finetuning.action_expert_tuning != initial_settings.action_expert_tuning
                        or finetuning.llm_lora != initial_settings.llm_lora):
                    raise ValueError("initialization checkpoint uses a different LoRA architecture; choose a base model or convert adapters")

        execution_target = str(payload.get("execution_target", "local_4090") or "local_4090")
        cluster_target_config = runtime_config_for_target(execution_target)
        is_cluster_target = cluster_target_config is not None
        minimum_free_gpu_mib = int(config.get("training_min_free_gpu_mib", 22_500))
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
            finetuning=finetuning,
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
                finetuning=finetuning,
            )
            slurm_command = slurm_runner_command(
                target_name=execution_target,
                target_config={
                    **cluster_target_config,
                    "xla_memory_fraction": xla_memory_fraction,
                },
                # Explicit retry option: run_train still checks the saved split,
                # normalization manifest and action contract before training.
                commands=([remote_train_command] if payload.get("reuse_norm_stats") is True
                          else [remote_norm_command, remote_train_command]),
                labels=(["train"] if payload.get("reuse_norm_stats") is True else ["norm", "train"]),
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
            "dataset_root_path": str(dataset_root_for_id(dataset_id)[0].parent),
            "dataset_read_only": not dataset_root_for_id(dataset_id)[1],
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
            "finetuning": finetuning.to_dict(),
            "base_checkpoint": str(base_checkpoint),
            "exp_name": exp_name,
            "task_scene": scene or None,
            "action_contract": dict(model_contract, model_variant=model_variant),
            "artifact_identity": payload.get("artifact_identity") or {"video_source": "robotwin" if "handover_mic" in dataset_id or "put_bottles" in dataset_id else "unknown", "task": video_catalog.slug(scene or "unknown"), "experiment": video_catalog.slug(exp_name)},
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
            "reuse_norm_stats": bool(is_cluster_target and payload.get("reuse_norm_stats") is True),
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
                remote_norm_batch_size if is_cluster_target else (
                    saved_norm_config.get("effective_batch_size")
                    if isinstance(saved_norm_config, dict) else None
                )
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
                    env=build_environment(config, None, **dataset_env_kwargs(dataset_id)),
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
                        **dataset_env_kwargs(dataset_id),
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
                        "dataset_root_path": str(dataset_root_for_id(dataset_id)[0].parent),
                        "dataset_read_only": not dataset_root_for_id(dataset_id)[1],
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
        model_contract = action_contract_for_model(dataset_contract)
        execution_target = str(payload.get("execution_target", "local_4090") or "local_4090")
        cluster_target_config = runtime_config_for_target(execution_target)
        is_cluster_target = cluster_target_config is not None
        checkpoint_raw = payload.get("checkpoint")
        checkpoint = resolve_under(checkpoint_raw, checkpoint_roots)
        if not (checkpoint / "params").exists():
            raise ValueError("checkpoint does not contain params")
        validate_weight_identity(checkpoint, model_contract)
        # Evaluation restores the selected complete checkpoint. No separate
        # foundation/initialization model is required, including Slurm preflight.
        base_checkpoint, model_variant = resolve_base_model({**payload, "base_checkpoint": str(checkpoint)})
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
            remote_base_checkpoint = remote_checkpoint
            cluster_target_config["base_checkpoint"] = remote_checkpoint
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
            env = build_environment(
                config,
                gpu_ids,
                xla_memory_fraction=xla_memory_fraction,
                **dataset_env_kwargs(dataset_id),
            )
        task = tasks.start(
            "eval",
            command,
            env=env,
            metadata={
                "dataset_id": dataset_id,
                "dataset_root_path": str(dataset_root_for_id(dataset_id)[0].parent),
                "dataset_read_only": not dataset_root_for_id(dataset_id)[1],
                "arm_mode": arm_mode,
                "arm_side": arm_side,
                "schema": schema,
                **model_contract["contract_fingerprint"],
                "raw_gripper_semantics": model_contract["raw_gripper_semantics"],
                "wire_gripper_semantics": model_contract["wire_gripper_semantics"],
                "model_variant": model_variant,
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
        # RTC runtime values and prompt are client-owned. The Dashboard only
        # starts a Policy with the server's capability ceiling; clients send
        # their selected values at runtime and can change them without
        # restarting this process.
        rtc_enabled = bool(config.get("policy_rtc_enabled", True))
        inference_diagnostics_enabled = bool(
            config.get("policy_inference_diagnostics_enabled", True)
        )
        rtc_execution_horizon = safe_int(
            config.get("policy_rtc_max_execution_horizon", 12),
            "policy_rtc_max_execution_horizon",
            1,
            50,
        )
        rtc_max_guidance_weight = safe_float(
            config.get("policy_rtc_max_guidance_weight", 5.0),
            "policy_rtc_max_guidance_weight",
            0.001,
            100.0,
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
            "--inference-diagnostics" if inference_diagnostics_enabled else "--no-inference-diagnostics",
            "--rtc-execution-horizon", str(rtc_execution_horizon),
            "--rtc-max-guidance-weight", str(rtc_max_guidance_weight),
        ] + action_contract_command_args(model_contract)
        task = tasks.start(
            "policy", command,
            env=build_environment(
                config,
                gpu_ids,
                xla_memory_fraction=policy_xla_memory_fraction,
                xla_preallocate=policy_xla_preallocate,
                **dataset_env_kwargs(dataset_id),
            ),
            metadata={
                "dataset_id": dataset_id,
                "dataset_root_path": str(dataset_root_for_id(dataset_id)[0].parent),
                "dataset_read_only": not dataset_root_for_id(dataset_id)[1],
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
                "rtc_server_enabled": rtc_enabled,
                "inference_diagnostics_enabled": inference_diagnostics_enabled,
                "rtc_server_max_execution_horizon": rtc_execution_horizon,
                "rtc_server_max_guidance_weight": rtc_max_guidance_weight,
                "replaced_task_id": replace_task_id or None,
            },
        )
        observations.bind_task(task)
        return jsonify(task), 201

    @app.get("/api/tasks/<task_id>/execution-control")
    def get_execution_control(task_id: str):
        task = tasks.get(task_id)
        return jsonify({"task_id": task["id"], "owner": "client", "execution_control": None})

    @app.post("/api/tasks/<task_id>/execution-control")
    def set_execution_control(task_id: str):
        tasks.get(task_id)
        return jsonify({"error": "execution is controlled by the robot client"}), 410

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
        if metadata.get("transfer_kind") == "dataset_delete":
            return {"stage": "delete_dataset", "progress": 1.0 if task.get("state") == "completed" else None}
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
        if target_path and task.get("state") not in TERMINAL_STATES and target_cfg.get("kind") != "slurm_only":
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
                    "finetuning",
                    "cluster_gpus",
                    "auto_eval",
                    "norm_batch_size",
                    "num_workers",
                    "xla_memory_fraction",
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
        task=tasks.get(task_id)
        checked={'checked_at':time.time()}
        if request.args.get('live')=='1' and task.get('metadata',{}).get('runtime')=='slurm':
            from server_4090.selected_resource import job_status
            meta=task['metadata']; target=config.get('cluster_targets',{}).get(meta.get('slurm_target') or meta.get('cluster_target'),{})
            job_id=meta.get('slurm_job_id') or tasks._slurm_job_id_from_log(tasks.log_tail(task_id))
            if job_id:
                checked.update(job_status(meta.get('slurm_submit_host') or target.get('submit_host'),job_id))
                if meta.get('external_training') and checked['state']!='unknown':
                    task['state']=checked['state'];atomic_json(tasks._path(task_id),task)
        result=task_status_summary(task,include_metrics=include_metrics)
        result.update(checked)
        response=jsonify(task=result);response.headers['Cache-Control']='no-store'
        return response

    @app.get("/api/tasks/<task_id>")
    def get_task(task_id: str):
        return jsonify(tasks.get(task_id))

    from server_4090.live_mirror import LiveMirror
    live_mirror=LiveMirror(Path(config['workspace_root'])/'live-log-mirrors',interval=3)
    tasks.live_mirror=live_mirror
    app.extensions["live_mirror"] = live_mirror

    def task_for_live_read(task_id):
        # Status polling owns refresh; curve/log reads must not wait on global task locks.
        task = read_json(tasks._path(task_id))
        if not isinstance(task, dict):
            raise FileNotFoundError(task_id)
        return task

    def mirrored_task_log(task_id):
        return live_mirror.get('task:'+task_id+':log',
            lambda: dict(log=tasks.log_tail(task_id,1024*1024,include_remote_slurm=True)),
            seed=dict(log=tasks.log_tail(task_id,131072)))

    @app.get("/api/tasks/<task_id>/log")
    def task_log(task_id: str):
        max_bytes = safe_int(request.args.get("max_bytes", 64 * 1024), "max_bytes", 1024, 1024 * 1024)
        task = task_for_live_read(task_id)
        result=mirrored_task_log(task_id)
        return jsonify({**result,"task":task,"log":result.get('log','')[-max_bytes:]})

    @app.get("/api/tasks/<task_id>/metrics")
    def task_metrics(task_id: str):
        task = task_for_live_read(task_id)
        if task.get("type") != "train":
            raise ValueError("metrics are only available for train tasks")
        max_points = safe_int(request.args.get("max_points", 1200), "max_points", 50, 5000)
        return jsonify(live_mirror.get('task:'+task_id+':metrics:'+str(max_points),
            lambda: build_task_metrics(task,max_points),
            seed=dict(parse_training_metrics(tasks.log_tail(task_id,256*1024),max_points=max_points),task=task)))

    def build_task_metrics(task,max_points):
        task_id=task['id']
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
        return result

    @app.post("/api/training/discover")
    def discover_training_processes():
        from server_4090.training_discovery import local_training_processes, cluster_training_processes
        payload=request.get_json(silent=True) or {}
        target_id=payload.get('target','local_4090')
        if target_id!='local_4090':
            target=config.get('cluster_targets',{}).get(target_id)
            if not target: raise ValueError('Unknown training target')
            rows=cluster_training_processes(target_id,target)
        else: rows=None
        with tasks.lock:
            existing=tasks.list()
            known={int(t.get('pid') or 0) for t in existing}
            known_jobs={(str((t.get('metadata') or {}).get('cluster_target') or (t.get('metadata') or {}).get('slurm_target')),str((t.get('metadata') or {}).get('slurm_job_id'))) for t in existing if not (t.get('metadata') or {}).get('external_training')}
            if rows is None: rows=local_training_processes(config,known)
            else: rows=[r for r in rows if (target_id,str(r['metadata'].get('slurm_job_id'))) not in known_jobs]
            for row in rows:
                task_path=tasks._path(row['id']);task_path.parent.mkdir(parents=True,exist_ok=True)
                row['log_path']=str(task_path.parent/'task.log')
                previous=read_json(task_path,{}) or {}
                atomic_json(task_path,{**previous,**row})
        return jsonify(tasks=rows,checked_at=time.time())

    @app.get("/api/training/discovery-sources")
    def training_discovery_sources():
        from server_4090.training_discovery import experiment_parent_sources
        parents=experiment_parent_sources(config)
        sources=[dict(id='live-all',label='发现可直接读取目录的新实验（节点内目录请单独选择）',target='可直接读取的实验父目录')]
        sources += [dict(id=s['id'],label=s.get('label',s['id']),target=s.get('target'),root=s.get('root'),requires_slurm=bool(s.get('storage_location'))) for s in parents]
        sources += [dict(id=s['id'],label=s.get('label',s['id']),target=s.get('target'),root=s.get('root'))
                 for s in config.get('rlinf_dashboard',{}).get('sources',[]) if s.get('enabled',True)]
        return jsonify(sources=sources)

    def rlinf_discovery_snapshot(source_id='live-all'):
        from server_4090.training_discovery import experiment_parent_sources
        import hashlib
        with rlinf_discovery_lock:
            if source_id in rlinf_discovery_snapshots:
                return rlinf_discovery_snapshots[source_id]
            parents=experiment_parent_sources(config)
            sources=[s for s in parents if not s.get('storage_location')] if source_id=='live-all' else [s for s in parents+config.get('rlinf_dashboard',{}).get('sources',[]) if s.get('id')==source_id and s.get('enabled',True)]
            if not sources: raise ValueError('请选择需要实时发现的实验父目录')
            path=Path(config['workspace_root'])/'rlinf-discovery'/ (hashlib.sha256(source_id.encode()).hexdigest()+'-summary.json')
            snapshot=Snapshot(path,lambda: discover_rlinf_sources(sources),ttl=15)
            rlinf_discovery_snapshots[source_id]=snapshot
            return snapshot

    @app.post("/api/rlinf/runs/discover")
    def discover_rlinf_source():
        payload=request.get_json(force=True)
        snapshot=rlinf_discovery_snapshot(payload.get('source','live-all')).get()
        data=snapshot['data'] or {}
        deleted=set(read_json(rlinf_history_path,[]) or [])
        return jsonify(runs=public_runs([r for r in data.get('runs',[]) if r['id'] not in deleted]),checked_at=snapshot['updated_at'],
                       ready=snapshot['ready'],refreshing=snapshot['refreshing'],error=snapshot['error'])

    def discover_rlinf_sources(sources):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        task_rows=tasks.list()
        def read_source(source):
            scoped=dict(config,rlinf_dashboard=dict(config.get('rlinf_dashboard',{}),sources=[source]))
            return source['id'], scan_runs(scoped,task_rows)
        rows=[]
        base_runs=rlinf_list_snapshot.get()['data'] or []
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(read_source,source) for source in sources]):
                source_id, fresh = future.result()
                rows.extend(fresh)
                publish_discovered_rlinf_runs(fresh,base_runs,refreshed_sources=[source_id])
        return dict(runs=public_runs(publish_discovered_rlinf_runs(rows,base_runs,persist=True,refreshed_sources=[s['id'] for s in sources])),scopes=[s['root'] for s in sources])

    def publish_discovered_rlinf_runs(rows,base_runs,persist=False,refreshed_sources=()):
        deleted=set(read_json(rlinf_history_path,[]) or [])
        def identities(row):
            target=str(row.get('target','')).lower()
            paths=[row.get('path'),row.get('_physical_path'),*row.get('_aliases',[])]
            return {(target,p) for p in paths if p}
        # Canonical metadata and a legacy symlink can describe the same run.
        # Prefer the canonical record and preserve established IDs for selections.
        known={key:r for r in base_runs+list(live_rlinf_overlay.values()) for key in identities(r)}
        unique=[];seen=set()
        for row in sorted(rows,key=lambda r:not r.get('_canonical')):
            keys=identities(row)
            if keys & seen:continue
            seen.update(keys)
            previous=next((known[key] for key in keys if key in known),None)
            if previous:
                preserve={'id','launch_profile','task_id','evaluation_cohort'}
                if not row.get('_canonical') and not str(previous.get('source_id','')).startswith('live-'):preserve.update({'name','source_id','_source'})
                row.update({k:v for k,v in previous.items() if k in preserve and v is not None})
            if row['id'] not in deleted:unique.append(row)
        rows=unique
        with live_rlinf_overlay_lock:
            refreshed = set(refreshed_sources)
            for ident, old in list(live_rlinf_overlay.items()):
                if ident.startswith('source-error-') and old.get('source_id') in refreshed:
                    del live_rlinf_overlay[ident]
            live_rlinf_overlay.update({r['id']:r for r in rows})
            if persist:atomic_json(live_rlinf_overlay_path,live_rlinf_overlay)
        return rows

    def selected_rlinf_options(target_id,profile_id,run_id=None,task_scene=None):
        from server_4090.model_catalog import rlinf_options
        from server_4090.model_identity import mismatch, rlinf_run_contract
        expected=config.get('rlinf_dashboard',{}).get('targets',{}).get(target_id,{}).get('profiles',{}).get(profile_id,{}).get('action_contract',{})
        options=rlinf_options(config,artifact_replicas,target_id,profile_id,task_scene)
        if run_id:
            run=next((r for r in visible_rlinf_runs()+list(histories.deleted().values()) if r.get('id')==run_id),None)
            if not run: raise ValueError('Unknown checkpoint run')
            if '_source' not in run:
                run['_source']=next((s for s in config.get('rlinf_dashboard',{}).get('sources',[]) if s.get('id')==run.get('source_id')),{})
            from server_4090.task_scenes import normalize
            scene=normalize(run.get('task_scene') or run.get('task') or run.get('task_name'))
            matched=(str(run.get('execution_target') or run.get('target','')).lower()==target_id and (not normalize(task_scene) or not scene or scene==normalize(task_scene)))
            reason=mismatch({'action_contract':rlinf_run_contract(config,run)},expected)
            for cp in checkpoints_for_run(run):
                options.append(dict(cp,action_contract=rlinf_run_contract(config,run),id='run:'+run_id+':'+cp['id'],compatible=matched and not reason,foundation=False,task_scene=scene,task_scenes=[scene] if scene else [],
                                    label=run['name']+' / step '+str(cp['step']),reason=reason or (None if matched else '运行所在服务器 / 任务场景尚未匹配')))
        return options

    @app.get("/api/rlinf/model-options")
    def rlinf_model_options():
        return jsonify(models=selected_rlinf_options(request.args.get('target'),request.args.get('profile'),request.args.get('run_id'),request.args.get('task_scene')))

    @app.get("/api/rlinf/config")
    def rlinf_dashboard_config():
        return jsonify(public_launch_config(config))

    @app.get("/api/rlinf/runs")
    def rlinf_dashboard_runs():
        # Run roots may live on the 4x4090 host, H100's shared filesystem, or
        # a NAS-backed H200 export.  The source map is server-owned; clients
        # can select a run but can never submit an arbitrary filesystem path.
        snapshot = rlinf_list_snapshot.get()
        runs = visible_rlinf_runs(snapshot)
        return jsonify({"runs": public_runs(runs), "index_updated_at": snapshot["updated_at"],
                        "refreshing": snapshot["refreshing"], "ready": snapshot["ready"], "error": snapshot["error"]})

    def check_rlinf_run_now(run):
        from server_4090.rlinf_status import job_identity, job_statuses
        if run.get('task_id') and tasks._path(run['task_id']).is_file():
            task=tasks.get(run['task_id']);run=dict(run,state=task['state'])
        job_id=run.get('job_id') or run.get('origin',{}).get('job_id')
        if job_id:
            identity = job_identity(config, run)
            if not identity: raise ValueError('Cannot resolve Slurm submit host')
            status = job_statuses(identity[0], [job_id])[str(job_id)]
            rlinf_status_cache.remember(run, status)
            run = dict(run, **status)
        return dict(run, checked_at=time.time())

    @app.get("/api/rlinf/runs/<run_id>/status")
    def rlinf_selected_status(run_id):
        run=selected_rlinf_run(run_id)
        if not run: raise ValueError('Unknown RLinf run')
        response=jsonify(run=public_runs([check_rlinf_run_now(run)])[0]);response.headers['Cache-Control']='no-store'
        return response

    def mirrored_rlinf_metrics(run):
        from server_4090.rlinf_dashboard import live_metrics_for_run, metrics_for_run
        def build():
            if (run.get('_source') or {}).get('storage_location'):
                raise ValueError('该来源尚无 Dashboard 日志同步；保留已同步曲线，不提交 Slurm 扫描')
            return live_metrics_for_run(run)
        seed=metrics_for_run([dict(run,_event_files=[])],run['id'])
        return live_mirror.get('rlinf:'+run['id']+':metrics',build,seed=seed)

    def mirrored_rlinf_log(run):
        def build():
            if run.get('task_id'):return dict(log=tasks.log_tail(run['task_id'],131072,include_remote_slurm=True))
            if (run.get('_source') or {}).get('storage_location'):
                raise ValueError('该来源尚无 Dashboard 日志同步，不提交 Slurm 扫描')
            return native_log_for_run(run)
        return live_mirror.get('rlinf:'+run['id']+':log',build,seed=dict(log=''))

    @app.get("/api/rlinf/runs/<run_id>/metrics")
    def rlinf_dashboard_run_metrics(run_id: str):
        run=selected_rlinf_run(run_id)
        if not run:raise ValueError('Unknown RLinf run')
        result = mirrored_rlinf_metrics(run)
        # The metric mirror owns curves; current task status comes from the list.
        if result.get('run'):
            result['run'].update({key: run[key] for key in ('state', 'status_reason', 'checked_at', 'slurm_state', 'finished_at') if key in run})
        return jsonify(result)

    @app.get("/api/rlinf/runs/<run_id>/log")
    def rlinf_run_log(run_id: str):
        run=selected_rlinf_run(run_id)
        if not run or run.get('state')=='unavailable':raise ValueError('unknown or unavailable RLInf run')
        return jsonify(run=public_runs([run])[0],**mirrored_rlinf_log(run))

    @app.get("/api/rlinf/runs/<run_id>/checkpoints")
    def rlinf_run_checkpoints(run_id: str):
        runs = visible_rlinf_runs() + list(histories.deleted().values())
        run = next((item for item in runs if item.get("id") == run_id), None)
        if not run or run.get("state") == "unavailable":
            raise ValueError("unknown or unavailable RLInf run")
        if '_source' not in run:
            run['_source'] = next((s for s in config.get('rlinf_dashboard', {}).get('sources', []) if s.get('id') == run.get('source_id')), {})
        try:
            target_id, _, profile_id, _ = evaluation_profile(config, run)
        except ValueError as exc:
            return jsonify({"checkpoints": [], "error": str(exc)})
        from server_4090.model_identity import mismatch, rlinf_run_contract
        expected=config['rlinf_dashboard']['targets'][target_id]['profiles'][profile_id].get('action_contract',{})
        reason=mismatch({'action_contract':rlinf_run_contract(config,run)},expected)
        return jsonify({"checkpoints": [dict(cp,compatible=not reason,reason=reason) for cp in checkpoints_for_run(run)], "target": target_id, "profile": profile_id})

    @app.get("/api/rlinf/model-runs")
    def rlinf_model_runs():
        runs=visible_rlinf_runs()+list(histories.deleted().values())
        unique={r.get('id'):r for r in runs if r.get('id') and r.get('state')!='unavailable'}
        return jsonify(runs=public_runs(list(unique.values())))

    @app.get("/api/rlinf/evaluations")
    def rlinf_evaluation_tasks():
        return jsonify({"tasks": [task for task in tasks.list() if task.get("type") == "eval" and task.get("metadata", {}).get("framework") == "RLinf"]})

    @app.post("/api/rlinf/evaluate")
    def evaluate_rlinf_checkpoint():
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        run = next((item for item in visible_rlinf_runs()+list(histories.deleted().values()) if item.get("id") == payload.get("run_id")), None)
        if run and '_source' not in run:
            run['_source']=next((s for s in config.get('rlinf_dashboard',{}).get('sources',[]) if s.get('id')==run.get('source_id')),{})
        if not run or run.get("state") == "unavailable":
            raise ValueError("unknown or unavailable RLInf run")
        target_id, target, profile_id, profile = evaluation_profile(config, run)
        checkpoint = next((item for item in checkpoints_for_run(run) if item["id"] == payload.get("checkpoint_id")), None)
        if not checkpoint:
            raise ValueError("请选择实际存在且权重完整的 RLInf checkpoint")
        from server_4090.model_identity import mismatch, rlinf_run_contract
        reason=mismatch({'action_contract':rlinf_run_contract(config,run)},profile.get('action_contract',{}))
        if reason:raise ValueError('评测模型不兼容：'+reason)
        episodes = safe_int(payload.get("episodes", 5), "episodes", 1, 1000)
        horizon = safe_int(payload.get("max_steps", 512), "max_steps", 1, 2000)
        seed = safe_int(payload.get("seed", 43), "seed", 0, 2147483647)
        record_video = payload.get("record_video", False)
        if not isinstance(record_video, bool):
            raise ValueError("record_video must be a boolean")
        roots = eval_video_roots()
        if not roots:
            raise ValueError("no evaluation artifact/video root configured")
        eval_name = f"step_{checkpoint['step']}_{uuid.uuid4().hex[:12]}"
        local_output = roots[0] / "rlinf" / target_id / eval_name
        kind = target.get("kind", "local")
        output = str(local_output) if kind == "local" else str(PurePosixPath(profile["run_root"]) / "dashboard_evaluations" / eval_name)
        context = video_catalog.classify(Path(run["path"]).parent, Path(run["path"]).name + "/context.mp4", {
            "video_source": "libero-safety", "phase": "eval", "experiment": run.get("source_id") or run.get("source") or target_id,
            "experiment_label": run.get("name"), "run_id": Path(run["path"]).name,
            "checkpoint_id": f"step-{checkpoint['step']}"})
        context = {k: context[k] for k in ("video_source", "task", "experiment", "experiment_label", "run_id", "phase", "checkpoint_id")}
        settings = profile["evaluation"]
        project = str(target["project_root"])
        entrypoint = settings.get("entrypoint", project + "/third_party/wovr/evaluations/eval_embodied_agent.py")
        runner = str(APP_DIR / "rlinf_eval_runner.py")
        inner = [str(target.get("python", "python3")), runner, "--entrypoint", str(entrypoint),
                 "--config-path", str(settings.get("config_path", project + "/configs/rlinf")),
                 "--config-name", str(settings.get("config_name", "libero_safety_grpo_pi05_smoke")),
                 "--checkpoint", checkpoint["path"], "--step", str(checkpoint["step"]),
                 "--output", output, "--episodes", str(episodes), "--max-steps", str(horizon),
                 "--seed", str(seed), "--record-video", str(record_video).lower(), "--video-context-json", json.dumps(context, ensure_ascii=False),
                 *[str(item) for item in settings.get("overrides", [])]]
        profile_env = {str(key): str(value) for key, value in profile.get("environment", {}).items()}
        if kind == "local":
            gpu_ids = parse_gpus(payload, one_only=True, minimum_free_mib=int(target.get("minimum_free_gpu_mib", 18000)), allow_busy=bool(target.get("allow_busy_gpus", False)))
            env = build_environment(config, gpu_ids)
            env.update(profile_env)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            inner.extend(native_gpu_placement(gpu_ids, evaluation=True))
            command = ["bash", "-lc", "cd " + shlex.quote(project) + " && exec " + shlex.join(inner)]
        elif kind == "slurm":
            # Ship this exact helper in the generated batch command so H200
            # node-local checkouts do not need another deployment or NAS mount
            # just to obtain the small dashboard evaluation wrapper.
            inner = [str(target.get("python", "python3")), "-c", (APP_DIR / "rlinf_eval_runner.py").read_text(encoding="utf-8"), *inner[2:]]
            exports = " ".join(f"export {key}={shlex.quote(value)};" for key, value in profile_env.items() if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key))
            cluster = {**target, "gpu_count": 1, "workdir": project, "openpi_python": config["openpi_python"]}
            slurm = slurm_runner_command(target_name=target_id, target_config=cluster,
                commands=[["bash", "-lc", exports + " exec " + shlex.join(inner)]], labels=["rlinf_eval"], job_name="rlinf_eval_" + eval_name)
            command = [config["openpi_python"], str(APP_DIR / "rlinf_eval_collect.py"), "--command-json", json_arg(slurm),
                       "--host", str(target["submit_host"]), "--remote-output", output, "--local-output", str(local_output)]
            env = build_environment(config, None)
            gpu_ids = []
        else:
            raise ValueError("unknown RLInf evaluation target kind")
        return jsonify(tasks.start("eval", command, env=env, metadata={
            "framework": "RLinf", "eval_kind": "rollout", "execution_target": target_id,
            "runtime": "local_4090" if kind == "local" else "slurm", "cluster_target": target_id if kind == "slurm" else None,
            "launch_profile": profile_id, "run_id": run["id"], "run_name": run["name"],
            "parent_train_task_id": run.get("task_id"), "checkpoint": checkpoint["path"], "checkpoint_step": checkpoint["step"],
            "episodes": episodes, "max_steps": horizon, "seed": seed, "record_video": record_video,
            "gpu_ids": gpu_ids, "result_path": str(local_output / "result.json"), "video_dir": str(local_output),
        })), 201

    @app.delete("/api/rlinf/runs/<run_id>")
    def delete_rlinf_run(run_id: str):
        return jsonify(delete_rlinf_records([run_id]))

    @app.post("/api/rlinf/runs/batch-delete")
    def delete_rlinf_runs_batch():
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return jsonify(delete_rlinf_records(payload.get("run_ids")))

    @app.post("/api/rlinf/launch")
    def launch_rlinf_training():
        payload = request.get_json(force=True)
        spec = build_launch_spec(config, payload)
        selected_model=payload.get('initial_model', 'profile-default')
        if selected_model != 'profile-default':
            from server_4090.model_catalog import rlinf_options
            options=selected_rlinf_options(spec['target_id'],spec['profile_id'],payload.get('initial_run'),payload.get('weight_task_scene'))
            model=next((m for m in options if m['id']==selected_model and m.get('compatible')),None)
            if not model: raise ValueError('所选初始化模型不兼容或已不在模型目录中')
            if not re.fullmatch(r'/[A-Za-z0-9_./-]+',model['path']): raise ValueError('Unsupported model path')
            if spec['target'].get('kind','local')=='local':
                from server_4090.selected_resource import inspect_checkpoint
                if not inspect_checkpoint(model['path'])['available']: raise ValueError('所选模型的关键文件已不可用')
            overrides=['actor.model.model_path='+model['path'],'rollout.model.model_path='+model['path']]
            if spec['command'][:2]==['bash','-lc']:
                spec['command'][2]=re.sub(r'(?:actor|rollout)\.model\.model_path=\S+', '', spec['command'][2])+' '+shlex.join(overrides)
            else:
                spec['command']=[arg for arg in spec['command'] if not arg.startswith(('actor.model.model_path=','rollout.model.model_path='))]+overrides
        target = spec["target"]
        profile = spec["profile"]
        output_dir = spec["output_dir"]
        kind = str(target.get("kind", "local"))
        if kind == "local":
            # Refuse to overwrite any run.  The parent is created here while
            # the method-specific directory remains absent for the trainer's
            # own exist_ok=False guard.
            output_path = Path(output_dir).expanduser()
            if output_path.exists():
                raise FileExistsError(f"RLinf output already exists: {output_dir}")
            gpu_ids = parse_gpus(
                {"gpu_ids": payload.get("gpu_ids", target.get("default_gpu_ids", []))},
                minimum_free_mib=int(target.get("minimum_free_gpu_mib", 0)),
                allow_busy=bool(target.get("allow_busy_gpus", False)),
            )
            env = build_environment(config, gpu_ids)
            env.update(spec["environment"])
            env["SAFE_RL_WM_DASHBOARD_RUN_DIR"] = output_dir
            if profile.get("backend") == "rlinf":
                if int(spec["values"]["total_envs"]) % len(gpu_ids):
                    raise ValueError("采样环境总数（每组个数 × 组数）必须能被 GPU 数量整除")
                env.pop("CUDA_VISIBLE_DEVICES", None)
                placement = native_gpu_placement(gpu_ids)
                # Hydra overrides belong inside the existing bash profile command.
                if spec["command"][:2] == ["bash", "-lc"]:
                    spec["command"][2] += " " + shlex.join(placement)
                else:
                    spec["command"].extend(placement)
            shell = "cd " + shlex.quote(spec["project_root"]) + " && exec " + shlex.join(spec["command"])
            command = ["bash", "-lc", shell]
            runtime = "local_4090"
            metadata_gpu_ids = gpu_ids
            target_config = None
        elif kind == "slurm":
            required = ("submit_host", "partition", "project_root", "run_root")
            missing = [key for key in required if not target.get(key)]
            if missing:
                raise ValueError(f"Slurm RLinf target is missing {missing}")
            gpu_count = safe_int(payload.get("gpu_count", target.get("gpu_count", 1)), "gpu_count", 1, 8)
            shell_env = {
                **{str(k): str(v) for k, v in target.get("environment", {}).items()},
                **spec["environment"],
                "SAFE_RL_WM_DASHBOARD_RUN_DIR": output_dir,
            }
            exports = " ".join(
                f"export {key}={shlex.quote(value)};"
                for key, value in sorted(shell_env.items())
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            )
            remote_shell = (
                "mkdir -p " + shlex.quote(str(PurePosixPath(output_dir).parent)) + " && "
                "(set -o noclobber; : > " + shlex.quote(
                    str(PurePosixPath(output_dir).parent / f".dashboard-reserved-{spec['algorithm']}")
                ) + ") && test ! -e " + shlex.quote(output_dir) + " && cd "
                + shlex.quote(spec["project_root"]) + " && "
                + exports + " exec " + shlex.join(spec["command"])
            )
            cluster = dict(target)
            cluster.update({
                "workdir": str(target["project_root"]),
                "gpu_count": gpu_count,
                "openpi_python": config["openpi_python"],
                "dataset_root": target.get("dataset_root", target.get("cache_root", "/tmp")),
                "cache_root": target.get("cache_root", "/tmp"),
            })
            labels = ["rlinf"]
            command = slurm_runner_command(
                target_name=str(payload.get("target")),
                target_config=cluster,
                commands=[["bash", "-lc", remote_shell]],
                labels=labels,
                job_name=f"rlinf_{spec['algorithm']}_{spec['run_name']}",
            )
            env = build_environment(config, None)
            runtime = "slurm"
            metadata_gpu_ids = []
            target_config = cluster
        else:
            raise ValueError("RLinf target kind must be local or slurm")

        metadata = {
            "framework": "RLinf",
            "artifact_identity": payload.get("artifact_identity") or {"video_source": "libero-safety", "task": "unknown", "experiment": video_catalog.slug(spec["run_name"])},
            "algorithm": spec["algorithm"],
            "launch_profile": spec["profile_id"],
            "action_contract": profile.get('action_contract'),
            "run_name": spec["run_name"],
            "rlinf_output_dir": output_dir,
            "rlinf_run_root": spec["run_root"],
            "execution_target": payload.get("target"),
            "runtime": runtime,
            "gpu_ids": metadata_gpu_ids,
            "cluster_target": payload.get("target") if kind == "slurm" else None,
            "cluster_gpu_count": target_config.get("gpu_count") if target_config else None,
            "test_episodes": 0,
            "auto_eval": {"enabled": False},
        }
        if kind == "local":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            reservation = output_path.parent / f".dashboard-reserved-{spec['algorithm']}"
            try:
                with reservation.open("x", encoding="utf-8") as handle:
                    handle.write(f"{now_iso()}\n")
            except FileExistsError as exc:
                raise FileExistsError(f"RLinf run already reserved: {output_dir}") from exc
        return jsonify(tasks.start("train", command, env=env, metadata=metadata)), 201


    if config.get('live_metrics_sync',{}).get('enabled',False):
        def keep_active_logs_synced():
            while True:
                try:
                    rlinf_discovery_snapshot().get()
                    for run in [r for r in visible_rlinf_runs() if r.get('state')=='running'][:16]:
                        mirrored_rlinf_metrics(run);mirrored_rlinf_log(run)
                    for task in [t for t in tasks.list() if t.get('type')=='train' and t.get('state')=='running'][:16]:
                        mirrored_task_log(task['id'])
                        live_mirror.get('task:'+task['id']+':metrics:1200',lambda t=task:build_task_metrics(t,1200))
                except Exception:
                    app.logger.exception('Live log synchronization registration failed')
                time.sleep(3)
        threading.Thread(target=keep_active_logs_synced,daemon=True,name='dashboard-active-log-sync').start()

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
