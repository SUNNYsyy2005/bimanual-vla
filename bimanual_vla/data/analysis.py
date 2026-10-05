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

from bimanual_vla.data.arm_geometry import (
    normalize_arm_base_offset,
    normalize_arm_base_rotations,
    normalize_robot_type,
)
from bimanual_vla.data.episode_analysis import compute_eef_trajectory
from bimanual_vla.deployment.jitter import TrajectoryJitterMonitor, model_joint_positions
from bimanual_vla.deployment.trajectory import suppress_local_joint_spikes


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
    except (AttributeError, ValueError):
        pass
    return value


def _metadata_value(value: Any) -> Any:
    value = _scalar(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.ndarray):
        return value.tolist()
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
    jitter_events: tuple[dict[str, Any], ...] = ()
    command_joints_rad: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    command_monotonic_timestamp: np.ndarray = field(default_factory=lambda: np.empty(0))
    model_joint_cache: dict[str, np.ndarray | None] = field(default_factory=dict, compare=False, repr=False)

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
            key: _metadata_value(archive[key])
            for key in (
                "task",
                "instruction",
                "schema",
                "arm_mode",
                "arm_side",
                "fps",
                "success",
                "robot_type",
                "dataset_origin",
                "arm_base_offset",
                "arm_base_rotations",
            )
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
        command_joints = np.asarray(
            archive["command_joints_rad"] if "command_joints_rad" in archive else np.empty((0, 0)),
            dtype=np.float64,
        )
        command_times = np.asarray(
            archive["command_monotonic_timestamp"] if "command_monotonic_timestamp" in archive else np.empty((0,)),
            dtype=np.float64,
        )
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
    jitter_events = _load_jsonl(path / "trajectory_jitter.jsonl")
    if not any(event.get("stream") == "command_sent" and isinstance(event.get("timestamp"), (int, float)) for event in jitter_events):
        jitter_events.extend(_reconstruct_sent_jitter(
            timestamps, command_sent, command_hold, generations, queue_indices,
            command_joints, command_times, float(metadata.get("control_hz") or 20.0),
        ))
        jitter_events = [
            event for event in jitter_events
            if event.get("stream") in {"model_raw", "command_sent"}
        ]
    if not any(event.get("stream") == "model_raw" for event in jitter_events):
        jitter_events.extend(_reconstruct_model_jitter(path, command_records, metadata))
    jitter_events = [
        event for event in jitter_events
        if isinstance(event.get("timestamp"), (int, float)) and np.isfinite(event["timestamp"])
    ]
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
        jitter_events=tuple(sorted(jitter_events, key=lambda event: float(event["timestamp"]))),
        command_joints_rad=command_joints,
        command_monotonic_timestamp=command_times,
    )


def _reconstruct_sent_jitter(
    timestamps: np.ndarray, sent: np.ndarray, holds: np.ndarray,
    generations: np.ndarray, indices: np.ndarray, joints: np.ndarray,
    command_times: np.ndarray, control_hz: float,
) -> list[dict[str, Any]]:
    if len(joints) != len(timestamps) or len(command_times) != len(timestamps):
        return []
    monitor = TrajectoryJitterMonitor(control_hz)
    events: list[dict[str, Any]] = []
    last_command_timestamp: float | None = None
    for row in range(len(timestamps)):
        valid = (
            bool(sent[row]) and not bool(holds[row]) and generations[row] >= 0
            and indices[row] >= 0 and np.isfinite(command_times[row])
            and np.isfinite(joints[row]).all()
        )
        emitted = monitor.observe(
            joints_rad=joints[row] if valid else None,
            generation=int(generations[row]) if valid else None,
            queue_index=int(indices[row]) if valid else None,
            command_at=float(command_times[row]) if valid else None,
            hold=bool(holds[row]),
        )
        for event in emitted:
            event_timestamp = (
                last_command_timestamp
                if event["event"] == "chunk_completed" and last_command_timestamp is not None
                else float(timestamps[row])
            )
            events.append({**event, "stream": "command_sent", "timestamp": event_timestamp})
        if valid:
            last_command_timestamp = float(timestamps[row])
    final = monitor.finish()
    if final is not None and last_command_timestamp is not None:
        events.append({**final, "stream": "command_sent", "timestamp": last_command_timestamp})
    return events


def _reconstruct_model_jitter(
    path: Path, records: tuple[dict[str, Any], ...], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    monitor: TrajectoryJitterMonitor | None = None
    tick = 0
    last_timestamp: float | None = None
    events: list[dict[str, Any]] = []
    for record in records:
        if record.get("accepted") is not True:
            continue
        relative = record.get("action_file")
        if not isinstance(relative, str):
            continue
        command_file = (path / relative).resolve()
        if not command_file.is_relative_to(path.resolve()) or not command_file.is_file():
            continue
        try:
            with np.load(command_file, allow_pickle=False) as archive:
                actions = np.asarray(archive["raw_actions"])
        except (OSError, KeyError, ValueError):
            continue
        protocol = record.get("protocol") or {}
        joints = model_joint_positions(
            actions,
            schema=str(protocol.get("schema") or ""),
            arm_mode=str(protocol.get("arm_mode") or ""),
        )
        if joints is None:
            continue
        if monitor is None:
            monitor = TrajectoryJitterMonitor(
                float(protocol.get("action_hz") or metadata.get("control_hz") or 20.0),
                basis="accepted_raw_model_full_horizon_joint_targets_excluding_gripper",
            )
        timestamp = float(record.get("captured_at") or record.get("arrived_at") or 0.0)
        generation = int(record.get("generation") or 0)
        for index, position in enumerate(joints):
            tick += 1
            for event in monitor.observe(
                joints_rad=position, generation=generation, queue_index=index,
                command_at=tick / monitor.control_hz,
            ):
                event_timestamp = (
                    last_timestamp
                    if event["event"] == "chunk_completed" and last_timestamp is not None
                    else timestamp
                )
                events.append({**event, "stream": "model_raw", "timestamp": event_timestamp})
        last_timestamp = timestamp
    if monitor is not None and last_timestamp is not None:
        final = monitor.finish()
        if final is not None:
            events.append({**final, "stream": "model_raw", "timestamp": last_timestamp})
    return events


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


_JITTER_EVENT_FIELDS = {
    "intra_accel_mean_rad_per_step2": ("chunk_completed", "intra_accel_mean_rad_per_step2"),
    "boundary_jump_mean_rad_l2": ("chunk_boundary", "position_jump_rad_l2"),
    "boundary_momentum_cosine_mean": ("chunk_boundary_momentum", "momentum_cosine"),
}


def _jitter_summary(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    streams: dict[str, list[dict[str, Any]]] = {"model_raw": [], "command_sent": []}
    for event in events:
        stream = event.get("stream")
        if stream in streams:
            streams[stream].append(event)
    result: dict[str, dict[str, float | int | None]] = {}
    for stream, items in streams.items():
        values: dict[str, float | int | None] = {}
        for metric, (event_type, field) in _JITTER_EVENT_FIELDS.items():
            matched = [event for event in items if event.get("event") == event_type]
            if metric == "intra_accel_mean_rad_per_step2":
                pairs = [
                    (float(event[field]), int(event.get("intra_accel_samples") or 0))
                    for event in matched if event.get(field) is not None
                ]
                pairs = [(value, count) for value, count in pairs if np.isfinite(value) and count > 0]
                count = sum(count for _, count in pairs)
                values[metric] = sum(value * samples for value, samples in pairs) / count if count else None
                values["intra_accel_samples"] = count
            else:
                samples = [float(event[field]) for event in matched if event.get(field) is not None]
                samples = [value for value in samples if np.isfinite(value)]
                values[metric] = float(np.mean(samples)) if samples else None
                values["boundary_jump_samples" if metric == "boundary_jump_mean_rad_l2" else "boundary_momentum_samples"] = len(samples)
        result[stream] = values
    return result


def jitter_plot_series(
    data: AnalysisData, start_index: int, end_index: int, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return event times and sparse model/sent values for one jitter chart."""
    if metric not in _JITTER_EVENT_FIELDS or end_index < start_index:
        return (np.empty(0), np.empty(0), np.empty(0))
    event_type, field = _JITTER_EVENT_FIELDS[metric]
    start_at, end_at = data.timestamps[start_index], data.timestamps[end_index]
    selected = [
        event for event in data.jitter_events
        if event.get("event") == event_type
        and event.get("stream") in {"model_raw", "command_sent"}
        and start_at <= float(event["timestamp"]) <= end_at
        and event.get(field) is not None
        and np.isfinite(float(event[field]))
    ]
    x = np.asarray([float(event["timestamp"]) - data.timestamps[0] for event in selected])
    model = np.asarray([
        float(event[field]) if event["stream"] == "model_raw" else np.nan
        for event in selected
    ])
    sent = np.asarray([
        float(event[field]) if event["stream"] == "command_sent" else np.nan
        for event in selected
    ])
    return x, model, sent


def motion_joint_names(data: AnalysisData) -> tuple[str, ...]:
    """Joint labels for motion charts; gripper openings are deliberately excluded."""
    width = data.command_joints_rad.shape[1] if data.command_joints_rad.ndim == 2 else 0
    protocol = data.metadata.get("policy_protocol") or {}
    if not protocol:
        protocol = next((r.get("protocol") or {} for r in data.command_records if r.get("accepted") is True), {})
    arm_mode = protocol.get("arm_mode") or data.metadata.get("arm_mode")
    if width not in (6, 12):
        width = 12 if arm_mode == "bimanual" else 6
    if width == 12:
        return tuple(f"{side}_j{joint}" for side in ("left", "right") for joint in range(1, 7))
    side = str(protocol.get("arm_side") or data.metadata.get("arm_side") or "arm")
    return tuple(f"{side}_j{joint}" for joint in range(1, 7))


def _model_chunk_joints(data: AnalysisData, record: dict[str, Any]) -> np.ndarray | None:
    relative = record.get("action_file")
    if not isinstance(relative, str):
        return None
    if relative in data.model_joint_cache:
        return data.model_joint_cache[relative]
    path = (data.path / relative).resolve()
    if not path.is_relative_to(data.path.resolve()) or not path.is_file():
        data.model_joint_cache[relative] = None
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            actions = np.asarray(archive["raw_actions"])
        protocol = record.get("protocol") or {}
        joints = model_joint_positions(
            actions, schema=str(protocol.get("schema") or ""),
            arm_mode=str(protocol.get("arm_mode") or ""),
        )
    except (OSError, KeyError, TypeError, ValueError):
        joints = None
    data.model_joint_cache[relative] = joints
    return joints


def sent_chunk_switch_indices(
    data: AnalysisData, start_index: int, end_index: int
) -> np.ndarray:
    """Rows where a sent command starts a different chunk, including range edges."""
    if data.kind != "deployment" or end_index < start_index:
        return np.empty(0, dtype=np.int64)
    count = min(data.sample_count, len(data.command_sent), len(data.command_hold),
                len(data.generations), len(data.queue_indices))
    previous_generation: int | None = None
    switches: list[int] = []
    for row in range(min(count, end_index + 1)):
        if not (data.command_sent[row] and not data.command_hold[row]
                and data.generations[row] >= 0 and data.queue_indices[row] >= 0):
            continue
        generation = int(data.generations[row])
        if previous_generation is not None and generation != previous_generation and row >= start_index:
            switches.append(row)
        previous_generation = generation
    return np.asarray(switches, dtype=np.int64)


def trajectory_chunk_switch_times(
    data: AnalysisData, start_index: int, end_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Prediction arrival and actual command-switch times relative to the run."""
    if data.kind != "deployment" or data.sample_count == 0 or end_index < start_index:
        return np.empty(0), np.empty(0)
    start_at, end_at = data.timestamps[start_index], data.timestamps[end_index]
    origin = data.timestamps[0]
    model_records: list[tuple[float, int]] = []
    for record in data.command_records:
        if record.get("accepted") is not True:
            continue
        try:
            arrived = float(record.get("arrived_at") or record.get("captured_at"))
            generation = int(record.get("generation"))
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(arrived):
            continue
        model_records.append((arrived, generation))
    model_switches: list[float] = []
    previous_generation: int | None = None
    for arrived, generation in sorted(model_records):
        if previous_generation is not None and generation != previous_generation and start_at <= arrived <= end_at:
            model_switches.append(arrived - origin)
        previous_generation = generation
    sent_rows = sent_chunk_switch_indices(data, start_index, end_index)
    sent_switches = data.timestamps[sent_rows] - origin
    return np.asarray(model_switches), np.asarray(sent_switches)

def _analysis_spike_horizon_steps(
    data: AnalysisData, record: dict[str, Any]
) -> int:
    """Return the recorded RTC execution horizon used by spike suppression."""
    candidates = (
        record.get("execution_control"),
        record.get("_client_transport_timing"),
        record.get("transport_timing"),
        data.metadata,
    )
    for source in candidates:
        if not isinstance(source, dict):
            continue
        for key in (
            "rtc_execution_horizon",
            "execution_horizon",
            "client_execution_horizon",
        ):
            try:
                value = int(source.get(key))
            except (TypeError, ValueError, OverflowError):
                continue
            if value > 0:
                return value
    return 8


def _analysis_skipped_prefix(record: dict[str, Any]) -> int:
    """Recover the client prefix skip when it was recorded; otherwise use zero."""
    candidates = (
        record.get("_client_transport_timing"),
        record.get("transport_timing"),
        record.get("execution_control"),
    )
    for source in candidates:
        if not isinstance(source, dict):
            continue
        for key in (
            "client_skipped_prefix",
            "inference_skip_steps",
            "skipped_prefix_steps",
            "skip_prefix_steps",
        ):
            try:
                value = int(source.get(key))
            except (TypeError, ValueError, OverflowError):
                continue
            if value >= 0:
                return value
    return 0


def _smoothed_model_chunk_joints(
    data: AnalysisData,
    record: dict[str, Any],
    joints: np.ndarray,
    action_hz: float,
) -> np.ndarray:
    """Replay the client short-horizon spike suppression for dashboard plots."""
    values = np.asarray(joints, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        return values.copy()
    skip = min(_analysis_skipped_prefix(record), len(values))
    if len(values) - skip < 5:
        return values.copy()
    output = values.copy()
    output[skip:] = suppress_local_joint_spikes(
        output[skip:].astype(np.float32),
        action_hz=float(action_hz),
        horizon_steps=_analysis_spike_horizon_steps(data, record),
        joint_indices=range(output.shape[1]),
    ).astype(np.float64)
    return output



def policy_trajectory_series(
    data: AnalysisData, start_index: int, end_index: int, *,
    order: int, joint_index: int | None = None, smoothed: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return times, current forecast, superseded forecast and switch times.

    Every accepted model horizon remains visible. Samples before the next
    prediction arrival are solid/current; older forecasts beyond that arrival
    are returned separately so the GUI can draw them dashed.
    """
    if order not in (0, 1, 2):
        raise ValueError("unsupported trajectory order")
    if data.sample_count == 0 or end_index < start_index:
        empty = np.empty(0)
        return empty, empty, empty, empty
    start_at, end_at = data.timestamps[start_index], data.timestamps[end_index]
    origin = data.timestamps[0]
    x: list[float] = []
    current: list[float] = []
    superseded: list[float] = []
    switches: list[float] = []

    def select(values: np.ndarray) -> np.ndarray:
        if joint_index is None:
            return np.linalg.norm(values, axis=1)
        if joint_index < 0 or joint_index >= values.shape[1]:
            return np.full(len(values), np.nan)
        return values[:, joint_index]

    records: list[tuple[float, int, float, np.ndarray]] = []
    for record in data.command_records:
        if record.get("accepted") is not True:
            continue
        try:
            arrived = float(record.get("arrived_at") or record.get("captured_at"))
            generation = int(record.get("generation"))
            protocol = record.get("protocol") or {}
            hz = float(protocol.get("action_hz") or data.metadata.get("control_hz") or 0.0)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(arrived) or not np.isfinite(hz) or hz <= 0:
            continue
        joints = _model_chunk_joints(data, record)
        if joints is not None:
            if smoothed:
                joints = _smoothed_model_chunk_joints(data, record, joints, hz)
            records.append((arrived, generation, hz, joints))
    records.sort(key=lambda item: item[0])

    previous_generation: int | None = None
    for chunk_index, (arrived, generation, hz, joints) in enumerate(records):
        if previous_generation is not None and generation != previous_generation and start_at <= arrived <= end_at:
            switches.append(arrived - origin)
        previous_generation = generation
        if arrived > end_at or arrived + (len(joints) - 1) / hz < start_at:
            continue
        values = joints
        if order == 1:
            values = np.full(joints.shape, np.nan, dtype=np.float64)
            if len(joints) > 1:
                values[1:] = np.diff(joints, axis=0) * hz
        elif order == 2:
            values = np.full(joints.shape, np.nan, dtype=np.float64)
            if len(joints) > 2:
                values[2:] = np.diff(joints, n=2, axis=0) * hz * hz
        chunk_time = arrived + np.arange(len(joints)) / hz
        selected = (chunk_time >= start_at) & (chunk_time <= end_at)
        if not np.any(selected):
            continue
        if x:
            x.append(float(chunk_time[selected][0] - origin))
            current.append(np.nan)
            superseded.append(np.nan)
        next_arrival = records[chunk_index + 1][0] if chunk_index + 1 < len(records) else np.inf
        selected_times = chunk_time[selected]
        selected_values = select(values[selected])
        is_superseded = selected_times >= next_arrival
        x.extend((selected_times - origin).tolist())
        current.extend(np.where(is_superseded, np.nan, selected_values).tolist())
        superseded.extend(np.where(is_superseded, selected_values, np.nan).tolist())
    return np.asarray(x), np.asarray(current), np.asarray(superseded), np.asarray(switches)


def trajectory_motion_series(
    data: AnalysisData, start_index: int, end_index: int, *,
    stream: str, order: int, joint_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (relative time, joint/L2 derivative, chunk switch times).

    Model samples retain each complete predicted horizon; published commands
    use wall time and monotonic timestamps for derivative intervals. NaNs
    separate chunks, holds, skipped rows, and irregular control ticks.
    """
    if order not in (1, 2) or stream not in {"model_raw", "command_sent"}:
        raise ValueError("unsupported motion chart")
    if data.sample_count == 0 or end_index < start_index:
        return np.empty(0), np.empty(0), np.empty(0)
    start_at, end_at = data.timestamps[start_index], data.timestamps[end_index]
    origin = data.timestamps[0]
    x: list[float] = []
    y: list[float] = []
    switches: list[float] = []

    def select(values: np.ndarray) -> np.ndarray:
        if joint_index is None:
            return np.linalg.norm(values, axis=1)
        if joint_index < 0 or joint_index >= values.shape[1]:
            return np.full(len(values), np.nan)
        return values[:, joint_index]

    if stream == "model_raw":
        model_x, current, superseded, model_switches = policy_trajectory_series(
            data, start_index, end_index, order=order, joint_index=joint_index,
        )
        values = np.where(np.isfinite(current), current, superseded)
        return model_x, values, model_switches

    joints = data.command_joints_rad
    command_times = data.command_monotonic_timestamp
    count = min(data.sample_count, len(joints), len(command_times), len(data.command_sent),
                len(data.command_hold), len(data.generations), len(data.queue_indices))
    try:
        hz = float(data.metadata.get("control_hz") or 0.0)
    except (TypeError, ValueError):
        hz = 0.0
    nominal = 1.0 / hz if np.isfinite(hz) and hz > 0 else None
    previous_row: int | None = None
    previous_generation: int | None = None
    previous_velocity: np.ndarray | None = None
    previous_dt: float | None = None
    for row in range(min(count, end_index + 1)):
        position = joints[row]
        valid = (
            data.command_sent[row] and not data.command_hold[row]
            and data.generations[row] >= 0 and data.queue_indices[row] >= 0
            and np.isfinite(command_times[row]) and np.isfinite(position).all()
        )
        value = np.nan
        if valid:
            generation = int(data.generations[row])
            if previous_generation is not None and generation != previous_generation and row >= start_index:
                switches.append(float(data.timestamps[row] - origin))
            contiguous = (
                previous_row == row - 1 and generation == previous_generation
                and data.queue_indices[row] == data.queue_indices[previous_row] + 1
            ) if previous_row is not None else False
            dt = float(command_times[row] - command_times[previous_row]) if contiguous else np.nan
            regular = contiguous and dt > 0 and (nominal is None or 0.5 * nominal <= dt <= 1.5 * nominal)
            velocity = (position - joints[previous_row]) / dt if regular else None
            if order == 1 and velocity is not None:
                value = float(select(velocity[np.newaxis, :])[0])
            elif order == 2 and velocity is not None and previous_velocity is not None and previous_dt is not None:
                acceleration = (velocity - previous_velocity) / ((dt + previous_dt) / 2.0)
                value = float(select(acceleration[np.newaxis, :])[0])
            previous_velocity = velocity
            previous_dt = dt if velocity is not None else None
            previous_row = row
            previous_generation = generation
        else:
            previous_row = None
            previous_velocity = None
            previous_dt = None
        if row >= start_index:
            x.append(float(data.timestamps[row] - origin))
            y.append(value)
    return np.asarray(x), np.asarray(y), np.asarray(switches)


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
    jitter_events = [
        event for event in data.jitter_events
        if timestamps[0] <= float(event["timestamp"]) <= timestamps[-1]
    ]
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
        "trajectory_jitter": _jitter_summary(jitter_events),
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
    """Compute per-arm XYZ trajectories using the declared robot geometry."""
    if data.measured.ndim != 2 or data.measured.shape[1] not in (7, 8, 10, 14, 16, 20):
        return {}
    end = data.sample_count - 1 if end_index is None else min(end_index, data.sample_count - 1)
    start = max(0, min(start_index, end))
    arm_side = str(data.metadata.get("arm_side") or "right")
    robot_type, dataset_origin = end_effector_source_context(data)
    arm_base_offset = data.metadata.get("arm_base_offset")
    arm_base_rotations = data.metadata.get("arm_base_rotations")
    if arm_base_offset is not None:
        arm_base_offset = normalize_arm_base_offset(arm_base_offset)
    if arm_base_rotations is not None:
        arm_base_rotations = normalize_arm_base_rotations(arm_base_rotations)
    measured, _method = compute_eef_trajectory(
        data.measured[start : end + 1],
        names=data.names,
        arm_side=arm_side,
        robot_type=robot_type,
        dataset_origin=dataset_origin,
        arm_base_offset=arm_base_offset,
        arm_base_rotations=arm_base_rotations,
    )
    desired, _method = compute_eef_trajectory(
        data.desired[start : end + 1],
        names=data.names,
        arm_side=arm_side,
        robot_type=robot_type,
        dataset_origin=dataset_origin,
        arm_base_offset=arm_base_offset,
        arm_base_rotations=arm_base_rotations,
    )
    result: dict[str, np.ndarray] = {}
    for side, values in measured.items():
        result[f"{side}_measured"] = np.asarray(values.get("position", []), dtype=np.float64)
    for side, values in desired.items():
        result[f"{side}_target"] = np.asarray(values.get("position", []), dtype=np.float64)
    return result


def end_effector_source_context(data: AnalysisData) -> tuple[str | None, str | None]:
    """Resolve recording provenance before interpreting left/right arm axes."""
    robot_type = data.metadata.get("robot_type")
    dataset_origin = data.metadata.get("dataset_origin")
    width = data.measured.shape[1] if data.measured.ndim == 2 else 0
    # Deployment runs are recorded by the physical Piper client. Older runs
    # omitted these two metadata fields, unlike collected episode files.
    if robot_type is None and data.kind == "deployment" and width in {7, 14}:
        robot_type = "piper"
    if dataset_origin is None and normalize_robot_type(robot_type) == "piper" and width in {14, 20}:
        dataset_origin = "real"
    return robot_type, dataset_origin
