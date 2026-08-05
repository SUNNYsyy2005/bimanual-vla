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
from typing import Any
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


def complete_checkpoint_steps(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Return complete numeric Orbax checkpoints without expensive size scans."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for child in checkpoint_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        if not (child / "_CHECKPOINT_METADATA").is_file():
            continue
        if not (child / "params" / "_METADATA").is_file():
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
    if robot_type == "piper":
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
    model_action_dim = 7 * arm_count if arm_mode in {"single", "bimanual"} else None

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
                # 7D/14D rows without the real-robot action-contract metadata.
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
                    "joint 7D/14D requires contract_version or gripper semantics to distinguish v2 metres from v3 fraction"
                )
            raw_action_convention = JOINT_RAW_ACTION_CONVENTION
            model_action_convention = DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
            model_action_semantics = JOINT_MODEL_ACTION_SEMANTICS
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
    training_supported = model_contract_supported and not legacy_delivery
    training_error = (
        "旧版 Delivery v2 仅保留预览/迁移兼容；请迁移为 canonical v3 后再训练"
        if legacy_delivery and model_contract_supported
        else None
    )
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
        "xla_memory_fraction": 0.90,
        "training_min_free_gpu_mib": 23_000,
        "evaluation_min_free_gpu_mib": 23_000,
        "evaluation_xla_memory_fraction": 0.85,
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
        "eval_video_roots": [],
        "cluster_resources_script": str(REPO_DIR / "scripts" / "query_h100_h200_resources.sh"),
        "transfer_parallelism": 4,
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
        "assets_base_dir",
        "checkpoint_base_dir",
        "base_checkpoint",
    ):
        defaults[key] = str(Path(defaults[key]).expanduser().resolve())
    defaults["checkpoint_allowed_roots"] = [
        str(Path(item).expanduser().resolve()) for item in defaults.get("checkpoint_allowed_roots", [])
    ]
    defaults["eval_video_roots"] = [
        str(Path(item).expanduser().resolve()) for item in defaults.get("eval_video_roots", [])
    ]
    defaults["cluster_resources_script"] = str(
        Path(defaults["cluster_resources_script"]).expanduser().resolve()
    )
    try:
        defaults["transfer_parallelism"] = max(1, min(16, int(defaults.get("transfer_parallelism", 4))))
    except (TypeError, ValueError):
        defaults["transfer_parallelism"] = 4
    normalized_targets = {}
    for name, target in dict(defaults.get("cluster_targets", {})).items():
        if not isinstance(target, dict):
            continue
        item = dict(target)
        for path_key in (
            "workdir",
            "openpi_repo",
            "dashboard_repo",
            "dataset_root",
            "assets_base_dir",
            "checkpoint_base_dir",
            "base_checkpoint",
            "eval_video_roots",
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
        if external_busy:
            return f"waiting for busy GPU(s): {external_busy}"
        minimum_free_mib = int(self.config.get("training_min_free_gpu_mib", 23_000))
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
) -> dict[int, dict[str, int]]:
    if minimum_free_mib <= 0:
        return {}
    shortfalls: dict[int, dict[str, int]] = {}
    for gpu_id in gpu_ids:
        gpu = inventory.get(gpu_id, {})
        free_mib = max(
            0,
            int(gpu.get("memory_total_mib", 0)) - int(gpu.get("memory_used_mib", 0)),
        )
        if free_mib < minimum_free_mib:
            shortfalls[gpu_id] = {
                "free_mib": free_mib,
                "required_mib": minimum_free_mib,
            }
    return shortfalls


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
        processes.setdefault(parts[0], []).append(
            {
                "pid": pid,
                "name": None if parts[2] == "[N/A]" else parts[2],
                "memory_mib": _nvidia_int(parts[3]),
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


def build_environment(
    config: dict[str, Any],
    gpu_ids: list[int] | None,
    *,
    xla_memory_fraction: float | None = None,
) -> dict[str, str]:
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
    if gpu_ids is None:
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("JAX_PLATFORMS", None)
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices(gpu_ids)
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
        if not supplied:
            # HTML <video> elements cannot attach Authorization headers.  Allow
            # token query auth for read-only media URLs generated by the
            # authenticated Dashboard page.
            supplied = request.args.get("token", "")
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
        return render_template(
            "index.html",
            server_port=config["port"],
            dashboard_profile=config.get("dashboard_profile", "real"),
            dashboard_title=config.get("dashboard_title", "Bimanual-VLA · 4×4090 控制台"),
            upload_default_origin=config.get("upload_default_origin", "real"),
            visible_dataset_origins=config.get("visible_dataset_origins", ["real", "unknown"]),
            enable_policy=bool(config.get("enable_policy", True)),
        )

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
                                "dataset_ids": dataset_ids,
                                "dataset_origins": dataset_origins,
                                "mtime": mtime,
                                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                                "size_gib": round(size_bytes / (1024**3), 3),
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
                    "upload_default_origin": config.get("upload_default_origin", "real"),
                    "visible_dataset_origins": config.get("visible_dataset_origins", ["real", "unknown"]),
                    "enable_policy": bool(config.get("enable_policy", True)),
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
        path = dataset_editor.video_path(dataset_id, episode_index, video_key)
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
        if minimum_free_mib > 0 and not config["allow_busy_gpus"]:
            low_memory = gpu_memory_shortfalls(inventory, gpu_ids, minimum_free_mib)
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
        fsdp_devices: int,
        split: EpisodeSplit,
        model_contract: dict[str, Any],
        effective_mode: str,
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
        if effective_mode != "new":
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

    REMOTE_EVAL_VIDEO_INVENTORY_SCRIPT = r'''
import json, os
from pathlib import Path
suffixes = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}
roots = [Path(item).expanduser() for item in json.loads(os.environ.get("EVAL_VIDEO_ROOTS", "[]"))]
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

    def list_remote_eval_videos() -> tuple[list[dict[str, Any]], dict[str, str]]:
        videos: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        for name, target in config.get("cluster_targets", {}).items():
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
                    ["ssh", "-o", "BatchMode=yes", str(host), command],
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
                videos.append({
                    **row,
                    "id": base64.urlsafe_b64encode(f"{name}\0{root}\0{rel}".encode()).decode().rstrip("="),
                    "source": name,
                    "host": host,
                    "remote": True,
                    "playable": False,
                    "syncable": True,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(row.get("mtime") or 0))),
                    "sync_url": "/api/eval-videos/sync",
                })
        return videos, errors

    @app.get("/api/eval-videos")
    def list_eval_videos():
        limit = safe_int(request.args.get("limit", 200), "limit", 1, 1000)
        include_remote = str(request.args.get("include_remote", "")).lower() in {"1", "true", "yes"}
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
                })
        remote_errors = {}
        if include_remote:
            remote_videos, remote_errors = list_remote_eval_videos()
            videos.extend(remote_videos)
        videos.sort(key=lambda item: item["mtime"], reverse=True)
        return jsonify({
            "videos": videos[:limit],
            "roots": [str(root) for root in eval_video_roots()],
            "remote_errors": remote_errors,
        })

    @app.get("/api/eval-videos/<video_id>")
    def get_eval_video(video_id: str):
        return send_file(decode_video_id(video_id), conditional=True, max_age=0)

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
            },
        )
        return jsonify(task), 201

    def dataset_location_configs() -> dict[str, dict[str, Any]]:
        locations = {
            "local_4090": {
                "name": "local_4090",
                "kind": "local",
                "host": None,
                "dataset_root": config["dataset_root"],
                "available": True,
            }
        }
        for name, target in config.get("cluster_targets", {}).items():
            if not target.get("dataset_root"):
                continue
            locations[name] = {
                "name": name,
                "kind": "ssh",
                "host": target.get("submit_host"),
                "partition": target.get("partition"),
                "node": target.get("node"),
                "gpu_type": target.get("gpu_type"),
                "dataset_root": target.get("dataset_root"),
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
    if any(token in name for token in ("sim", "synth", "cube", "dustbin", "bottle", "lift_pot")) or robot_type in {"aloha", "sim", "simulation"}:
        return "simulation"
    if "piper" in robot_type or "real" in name or "my_dataset" in name:
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
        if location.get("kind") == "local":
            return local_dataset_inventory(location), None
        host = location.get("host")
        if not host:
            return [], "missing ssh host"
        env = f"DATASET_ROOT={shlex.quote(str(location['dataset_root']))}"
        command = f"{env} python3 - <<'REMOTE_DATASET_PY'\n{REMOTE_DATASET_INVENTORY_SCRIPT}\nREMOTE_DATASET_PY"
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", str(host), command],
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
        command = [
            config["openpi_python"],
            str(APP_DIR / "dataset_transfer_runner.py"),
            "--dataset-id", dataset_id,
            "--source-json", json_arg(locations[source_name]),
            "--target-json", json_arg(locations[target_name]),
            "--parallelism", str(parallelism),
        ]
        if overwrite:
            command.append("--overwrite")
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
        effective_mode = "resume" if mode == "auto" else mode
        saved_steps = any(
            child.is_dir() and child.name.isdigit() and (child / "params").is_dir()
            for child in checkpoint_dir.iterdir()
        ) if checkpoint_dir.is_dir() else False

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
        execution_target = str(payload.get("execution_target", "local_4090") or "local_4090")
        cluster_target_config = runtime_config_for_target(execution_target)
        is_cluster_target = cluster_target_config is not None
        minimum_free_gpu_mib = int(config.get("training_min_free_gpu_mib", 23_000))
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
            fsdp_devices=fsdp_devices,
            split=split,
            model_contract=model_contract,
            effective_mode=effective_mode,
            wandb_enabled=bool(payload.get("wandb_enabled", False)),
        )
        if is_cluster_target:
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
                fsdp_devices=fsdp_devices,
                split=split,
                model_contract=model_contract,
                effective_mode=effective_mode,
                wandb_enabled=bool(payload.get("wandb_enabled", False)),
            )
            command = slurm_runner_command(
                target_name=execution_target,
                target_config={
                    **cluster_target_config,
                    "xla_memory_fraction": xla_memory_fraction,
                },
                commands=[remote_norm_command, remote_train_command],
                labels=["norm", "train"],
                job_name=f"sim_train_{exp_name}",
            )
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
            "fsdp_devices": fsdp_devices,
            "xla_memory_fraction": xla_memory_fraction,
            "minimum_free_gpu_mib": minimum_free_gpu_mib,
            "mode": mode,
            "effective_mode": effective_mode,
            "checkpoint_dir": str(checkpoint_dir),
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
                gpu_ids = parse_gpus(payload, minimum_free_mib=minimum_free_gpu_mib)
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
        ] + action_contract_command_args(model_contract)
        default_prompt = str(payload.get("default_prompt", "")).strip()
        if default_prompt:
            if len(default_prompt) > 500:
                raise ValueError("default_prompt is too long")
            command += ["--default-prompt", default_prompt]
        task = tasks.start(
            "policy", command,
            env=build_environment(config, gpu_ids),
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
                "port": port,
                "ws_url": f"ws://{request.host.split(':')[0]}:{port}",
                "telemetry_session": telemetry_session,
                "telemetry_dir": str(telemetry_dir),
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
                    "checkpoint",
                    "checkpoint_step",
                    "checkpoint_dir",
                    "base_checkpoint",
                    "result_path",
                    "source",
                    "target",
                    "source_path",
                    "target_path",
                    "overwrite",
                    "test_ratio",
                    "split_seed",
                    "train_episodes",
                    "test_episodes",
                )
                if key in metadata
            },
        }
        if include_metrics and task.get("type") == "train":
            metrics = parse_training_metrics(
                tasks.log_tail(str(task["id"]), 16 * 1024 * 1024),
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
        return jsonify({"task": task, "log": tasks.log_tail(task_id, max_bytes)})

    @app.get("/api/tasks/<task_id>/metrics")
    def task_metrics(task_id: str):
        task = tasks.get(task_id)
        if task.get("type") != "train":
            raise ValueError("metrics are only available for train tasks")
        max_points = safe_int(request.args.get("max_points", 1200), "max_points", 50, 5000)
        result = parse_training_metrics(tasks.log_tail(task_id, 16 * 1024 * 1024), max_points=max_points)
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
