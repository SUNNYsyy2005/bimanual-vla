"""Read-only analysis helpers for collected episodes and deployment runs.

The GUI deliberately keeps this module independent of Tk.  It can therefore
be tested with small synthetic files and reused by command-line tooling later.
Raw files are never modified by any function in this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_NAMES = tuple(
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
)


def _scalar(value: Any, default: Any = None) -> Any:
    """Convert a numpy scalar/0-d array to a JSON-friendly Python value."""
    if value is None:
        return default
    try:
        value = value.item()
    except AttributeError:
        pass
    return value


def _finite_vector(value: Any, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        return np.full((len(array), width), np.nan, dtype=np.float64)
    return array


def _names(value: Any, width: int) -> tuple[str, ...]:
    try:
        values = tuple(str(item) for item in np.asarray(value).reshape(-1).tolist())
    except (TypeError, ValueError):
        values = ()
    if len(values) == width:
        return values
    if width == len(DEFAULT_NAMES):
        return DEFAULT_NAMES
    return tuple(f"dim_{index + 1}" for index in range(width))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


@dataclass(frozen=True)
class AnalysisData:
    """Normalized read-only representation of one data source."""

    path: Path
    kind: str
    label: str
    timestamps: np.ndarray
    measured: np.ndarray
    desired: np.ndarray
    command_sent: np.ndarray
    command_hold: np.ndarray
    generations: np.ndarray
    queue_indices: np.ndarray
    blocked_reasons: tuple[str, ...]
    execution_states: tuple[str, ...]
    names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    command_records: tuple[dict[str, Any], ...] = ()

    @property
    def sample_count(self) -> int:
        return int(len(self.timestamps))

    @property
    def duration_s(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return max(0.0, float(self.timestamps[-1] - self.timestamps[0]))


def _deployment_label(path: Path) -> str:
    return path.name


def load_analysis_data(source: str | Path) -> AnalysisData:
    """Load one deployment run directory or one ``ep_XXXX.npz`` file."""
    path = Path(source).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".npz":
        return _load_episode(path)
    if path.is_dir() and (path / "trajectory.npz").is_file():
        return _load_deployment(path)
    raise ValueError(f"unsupported analysis source: {path}")


def _load_episode(path: Path) -> AnalysisData:
    with np.load(path, allow_pickle=False) as archive:
        measured_raw = archive["state"] if "state" in archive else (
            archive["joint_qpos"] if "joint_qpos" in archive else None
        )
        desired_raw = archive["actions"] if "actions" in archive else measured_raw
        if measured_raw is None or desired_raw is None:
            raise ValueError(f"episode has no state/actions arrays: {path}")
        measured = np.asarray(measured_raw, dtype=np.float64)
        desired = np.asarray(desired_raw, dtype=np.float64)
        width = int(measured.shape[1]) if measured.ndim == 2 else 0
        if width <= 0:
            raise ValueError(f"episode state must be a 2-D array: {path}")
        measured = _finite_vector(measured, width)
        desired = _finite_vector(desired, width)
        timestamps = np.asarray(
            archive["state_timestamp"] if "state_timestamp" in archive else archive["timestamps"],
            dtype=np.float64,
        ).reshape(-1)
        names = _names(archive["state_names"] if "state_names" in archive else None, width)
        metadata = {
            key: _scalar(archive[key])
            for key in ("task", "instruction", "schema", "arm_mode", "arm_side", "fps", "success")
            if key in archive
        }
    count = min(len(timestamps), len(measured), len(desired))
    timestamps, measured, desired = timestamps[:count], measured[:count], desired[:count]
    return AnalysisData(
        path=path,
        kind="episode",
        label=f"{path.parent.name}/{path.name}",
        timestamps=timestamps,
        measured=measured,
        desired=desired,
        command_sent=np.isfinite(desired).all(axis=1),
        command_hold=np.zeros(count, dtype=bool),
        generations=np.full(count, -1, dtype=np.int64),
        queue_indices=np.full(count, -1, dtype=np.int64),
        blocked_reasons=tuple("" for _ in range(count)),
        execution_states=tuple("recorded" for _ in range(count)),
        names=names,
        metadata=metadata,
    )


def _load_deployment(path: Path) -> AnalysisData:
    with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
        timestamps = np.asarray(archive["timestamp"], dtype=np.float64).reshape(-1)
        measured = np.asarray(archive["qpos"], dtype=np.float64)
        desired = np.asarray(archive["command_action"], dtype=np.float64)
        count = min(len(timestamps), len(measured), len(desired))
        timestamps, measured, desired = timestamps[:count], measured[:count], desired[:count]
        command_sent = np.asarray(
            archive["command_sent"] if "command_sent" in archive else np.isfinite(desired).all(axis=1),
            dtype=bool,
        )[:count]
        command_hold = np.asarray(
            archive["command_hold"] if "command_hold" in archive else np.zeros(count),
            dtype=bool,
        )[:count]
        generations = np.asarray(
            archive["command_generation"] if "command_generation" in archive else np.full(count, -1),
            dtype=np.int64,
        )[:count]
        queue_indices = np.asarray(
            archive["command_queue_index"] if "command_queue_index" in archive else np.full(count, -1),
            dtype=np.int64,
        )[:count]
    metadata_path = path / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    rows = _load_jsonl(path / "trajectory.jsonl")
    blocked = tuple(str(row.get("blocked_reason") or "") for row in rows[:count])
    states = tuple(str(row.get("execution_state") or "") for row in rows[:count])
    if len(blocked) < count:
        blocked += ("",) * (count - len(blocked))
    if len(states) < count:
        states += ("",) * (count - len(states))
    command_records = tuple(_load_jsonl(path / "model_commands.jsonl"))
    width = int(measured.shape[1]) if measured.ndim == 2 else 0
    return AnalysisData(
        path=path,
        kind="deployment",
        label=_deployment_label(path),
        timestamps=timestamps,
        measured=measured,
        desired=desired,
        command_sent=command_sent,
        command_hold=command_hold,
        generations=generations,
        queue_indices=queue_indices,
        blocked_reasons=blocked,
        execution_states=states,
        names=_names(None, width),
        metadata=metadata,
        command_records=command_records,
    )


def scan_analysis_sources(roots: Iterable[str | Path]) -> list[Path]:
    """Find deployment runs and regular episode files under the given roots."""
    found: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if root.is_file() and root.suffix.lower() == ".npz":
            found.add(root.resolve())
            continue
        if not root.is_dir():
            continue
        if (root / "trajectory.npz").is_file():
            found.add(root.resolve())
            continue
        for trajectory in root.rglob("trajectory.npz"):
            if trajectory.parent.is_dir():
                found.add(trajectory.parent.resolve())
        for episode in root.rglob("ep_*.npz"):
            if episode.is_file() and "model_commands" not in episode.parts:
                found.add(episode.resolve())
    return sorted(found, key=lambda value: str(value))


def selection_indices(data: AnalysisData, start_s: float = 0.0, end_s: float | None = None) -> tuple[int, int]:
    """Convert relative seconds into an inclusive sample range."""
    count = data.sample_count
    if count == 0:
        return 0, -1
    duration = data.duration_s
    start = max(0.0, min(float(start_s), duration))
    end = duration if end_s is None else max(start, min(float(end_s), duration))
    relative = data.timestamps - data.timestamps[0]
    start_index = int(np.searchsorted(relative, start, side="left"))
    end_index = int(np.searchsorted(relative, end, side="right")) - 1
    return max(0, min(start_index, count - 1)), max(0, min(end_index, count - 1))


def _stat(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "median": None, "p95": None, "mean": None, "max": None, "std": None}
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _timing_values(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        timing = record.get("_client_transport_timing") or {}
        value = timing.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return values


def _action_row_count(record: dict[str, Any], default_horizon: int = 0) -> int:
    shape = record.get("action_shape")
    if isinstance(shape, (list, tuple)) and shape:
        try:
            return max(0, int(shape[0]))
        except (TypeError, ValueError):
            pass
    return max(0, int(default_horizon))


def compute_metrics(data: AnalysisData, start_index: int = 0, end_index: int | None = None) -> dict[str, Any]:
    """Compute summary statistics for an inclusive selected sample range."""
    if data.sample_count == 0:
        return {"sample_count": 0, "duration_s": 0.0}
    end = data.sample_count - 1 if end_index is None else max(start_index, min(end_index, data.sample_count - 1))
    start = max(0, min(start_index, end))
    sl = slice(start, end + 1)
    timestamps = data.timestamps[sl]
    relative_duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    tick_ms = np.diff(timestamps) * 1000.0
    measured = data.measured[sl]
    desired = data.desired[sl]
    valid_desired = np.isfinite(desired).all(axis=1)
    action_delta = np.linalg.norm(np.diff(desired[valid_desired], axis=0), axis=1) if valid_desired.sum() > 1 else []
    qpos_delta = np.linalg.norm(np.diff(measured, axis=0), axis=1) if len(measured) > 1 else []
    error = desired[valid_desired] - measured[valid_desired]
    error_norm = np.linalg.norm(error, axis=1) if len(error) else []
    blocked = Counter(reason for reason in data.blocked_reasons[sl] if reason)
    states = Counter(state for state in data.execution_states[sl] if state)
    commands = [
        record
        for record in data.command_records
        if start <= int(np.searchsorted(data.timestamps, float(record.get("captured_at", -np.inf)), side="left")) <= end
    ]
    model_intervals = np.diff([float(record.get("captured_at")) for record in commands if record.get("captured_at") is not None]) * 1000.0
    rejected_commands = sum(1 for record in commands if record.get("accepted") is False)
    model_action_rows = sum(
        _action_row_count(
            record,
            int((data.metadata.get("policy_protocol") or {}).get("action_horizon") or 0),
        )
        for record in commands
    )
    rejected_action_rows = sum(
        _action_row_count(
            record,
            int((data.metadata.get("policy_protocol") or {}).get("action_horizon") or 0),
        )
        for record in commands
        if record.get("accepted") is False
    )
    unsafe_drops = sum(count for reason, count in blocked.items() if reason.startswith("dropped unsafe"))
    timing = {
        key: _stat(_timing_values(commands, key))
        for key in ("camera_capture_ms", "observation_upload_ms", "model_inference_ms", "result_download_ms", "round_trip_ms")
    }
    return {
        "sample_count": int(end - start + 1),
        "start_s": float(timestamps[0] - data.timestamps[0]),
        "end_s": float(timestamps[-1] - data.timestamps[0]),
        "duration_s": relative_duration,
        "control_hz": float(1000.0 / np.median(tick_ms)) if len(tick_ms) and np.median(tick_ms) > 0 else None,
        "tick_interval_ms": _stat(tick_ms),
        "command_sent": int(np.count_nonzero(data.command_sent[sl])),
        "command_sent_fraction": float(np.mean(data.command_sent[sl])) if len(data.command_sent[sl]) else 0.0,
        "hold_count": int(np.count_nonzero(data.command_hold[sl])),
        "blocked": dict(blocked),
        "execution_states": dict(states),
        "model_command_count": len(commands),
        "model_action_rows": int(model_action_rows),
        "executed_control_actions": int(np.count_nonzero(data.command_sent[sl])),
        "rejected_action_count": int(rejected_commands),
        "rejected_action_rows": int(rejected_action_rows),
        "unsafe_drop_count": int(unsafe_drops),
        "discarded_action_count": int(rejected_action_rows + unsafe_drops),
        "model_interval_ms": _stat(model_intervals),
        "latency": timing,
        "action_step_norm": _stat(action_delta),
        "qpos_step_norm": _stat(qpos_delta),
        "action_error_norm": _stat(error_norm),
    }


def compute_end_effector_positions(
    data: AnalysisData,
    start_index: int = 0,
    end_index: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute Piper XYZ trajectories from joint-space state/action data.

    Piper's SDK exposes the same forward-kinematics implementation used by the
    runtime.  Joint data is expected in left/right blocks of 7 values; delivery
    (10D pose) data is intentionally skipped because it is already a pose
    representation and does not need a second FK conversion.
    """
    if data.measured.ndim != 2 or data.measured.shape[1] not in (7, 14):
        return {}
    try:
        from piper_sdk import C_PiperForwardKinematics

        fk = C_PiperForwardKinematics()
    except Exception:
        return {}
    end = data.sample_count - 1 if end_index is None else min(end_index, data.sample_count - 1)
    start = max(0, min(start_index, end))
    result: dict[str, np.ndarray] = {}
    sides = ("left", "right") if data.measured.shape[1] == 14 else (str(data.metadata.get("arm_side") or "right"),)
    for source_name, source in (("measured", data.measured), ("target", data.desired)):
        for side_index, side in enumerate(sides):
            block = source[start : end + 1, side_index * 7 : side_index * 7 + 6]
            positions = np.full((len(block), 3), np.nan, dtype=np.float64)
            for index, joints in enumerate(block):
                if not np.all(np.isfinite(joints)):
                    continue
                try:
                    pose = np.asarray(fk.CalFK(joints.tolist())[-1], dtype=np.float64)
                    positions[index] = pose[:3] / 1000.0
                except Exception:
                    continue
            result[f"{side}_{source_name}"] = positions
    return result
