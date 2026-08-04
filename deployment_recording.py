"""Persistent recording for real-robot policy deployment runs.

The deployment client has three different time bases/streams that are useful in
post-hoc analysis:

* measured arm feedback and the command selected on every control tick;
* complete model action chunks, including chunks that were rejected or became
  stale before execution; and
* camera frames used by inference, with their source timestamps.

``DeploymentRunRecorder`` keeps the control path non-blocking.  Small metadata
records are written by a background worker, while the final trajectory NPZ is
written when the run is closed.  Video files are deliberately accompanied by a
JSONL timestamp index because MP4 containers normally expose only a nominal
constant frame rate, whereas camera capture and policy requests are not exactly
periodic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from queue import Full, Queue
import threading
import time
from typing import Any, Mapping

import numpy as np


LOGGER = logging.getLogger(__name__)
RECORDING_VERSION = 1


_STOP = object()


def _json_safe(value: Any) -> Any:
    """Convert scalar/nested telemetry values into JSON without huge arrays."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _as_finite_array(value: Any, *, dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError("recorded array contains NaN or Inf")
    return np.array(array, copy=True)


@dataclass(frozen=True)
class _VideoFrame:
    camera_key: str
    frame: np.ndarray
    timestamp: float
    monotonic_timestamp: float | None
    frame_group: str | None


class DeploymentRunRecorder:
    """Record one deployment run into a self-contained directory.

    The default layout is::

        deployment_runs/<UTC timestamp>_<pid>/
          metadata.json
          trajectory.npz
          trajectory.jsonl
          model_commands.jsonl
          model_commands/command_000001_gen_00000042.npz
          videos/<camera>.mp4
          videos/timestamps.jsonl

    ``trajectory.npz`` contains one row per control tick.  Missing commands are
    represented by NaN rows and ``command_sent == False``; this preserves exact
    alignment with measured feedback even while the client is shadowed, waiting
    for a fresh chunk, or safety-blocked.
    """

    def __init__(
        self,
        root: str | Path = "deployment_runs",
        *,
        video_fps: float = 4.0,
        enabled: bool = True,
        queue_size: int = 4096,
    ) -> None:
        if video_fps <= 0 or not np.isfinite(video_fps):
            raise ValueError("video_fps must be positive and finite")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self.root = Path(root)
        self.video_fps = float(video_fps)
        self.enabled = bool(enabled)
        self.queue_size = int(queue_size)

        self.run_dir: Path | None = None
        self._metadata: dict[str, Any] = {}
        self._trajectory_rows: list[dict[str, Any]] = []
        self._model_command_count = 0
        self._control_tick_count = 0
        self._dropped_event_count = 0
        self._queue: Queue[Any] | None = None
        self._worker: threading.Thread | None = None
        self._closed = True
        self._trajectory_file: Any | None = None
        self._model_file: Any | None = None
        self._video_index_file: Any | None = None
        self._video_writers: dict[str, Any] = {}
        self._video_paths: dict[str, Path] = {}
        self._video_frame_fallback_dirs: dict[str, Path] = {}
        self._video_frame_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        return bool(self.enabled and self.run_dir is not None and not self._closed)

    def start(self, metadata: Mapping[str, Any] | None = None) -> Path | None:
        """Start a new run and return its directory, or ``None`` when disabled."""
        if not self.enabled:
            return None
        if self.is_active:
            raise RuntimeError("deployment recorder is already active")
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = self.root / f"{stamp}_{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "model_commands").mkdir()
        (run_dir / "videos").mkdir()
        (run_dir / "video_frames").mkdir()

        self.run_dir = run_dir
        self._metadata = {
            "recording_version": RECORDING_VERSION,
            "started_at": time.time(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "video_nominal_fps": self.video_fps,
            "video_timestamp_index": "videos/timestamps.jsonl",
            **_json_safe(dict(metadata or {})),
        }
        self._trajectory_rows.clear()
        self._model_command_count = 0
        self._control_tick_count = 0
        self._dropped_event_count = 0
        self._video_writers.clear()
        self._video_paths.clear()
        self._video_frame_fallback_dirs.clear()
        self._video_frame_counts.clear()
        self._trajectory_file = (run_dir / "trajectory.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._model_file = (run_dir / "model_commands.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._video_index_file = (run_dir / "videos" / "timestamps.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._queue = Queue(maxsize=self.queue_size)
        self._closed = False
        self._worker = threading.Thread(
            target=self._writer_loop,
            name="deployment-recorder",
            daemon=True,
        )
        self._worker.start()
        self._write_metadata()
        LOGGER.info("Deployment recording started: %s", run_dir)
        return run_dir

    def update_metadata(self, values: Mapping[str, Any]) -> None:
        """Merge connection/protocol metadata and update ``metadata.json``."""
        if not self.is_active:
            return
        with self._lock:
            self._metadata.update(_json_safe(dict(values)))
            self._write_metadata()

    def record_control_tick(
        self,
        *,
        timestamp: float,
        monotonic_timestamp: float,
        delivery_state: np.ndarray,
        qpos: np.ndarray,
        command_sent: bool,
        action_dim: int,
        absolute_dim: int,
        command_action: np.ndarray | None = None,
        command_absolute_target: np.ndarray | None = None,
        command_generation: int | None = None,
        command_queue_index: int | None = None,
        command_hold: bool = False,
        execution_state: str | None = None,
        blocked_reason: str | None = None,
    ) -> None:
        """Record measured state and command outcome for one control tick."""
        if not self.is_active:
            return
        state = _as_finite_array(delivery_state)
        joints = _as_finite_array(qpos)
        raw = np.full(int(action_dim), np.nan, dtype=np.float32)
        absolute = np.full(int(absolute_dim), np.nan, dtype=np.float32)
        if command_action is not None:
            candidate = _as_finite_array(command_action)
            if candidate.shape != raw.shape:
                raise ValueError(f"command_action must have shape {raw.shape}, got {candidate.shape}")
            raw[:] = candidate
        if command_absolute_target is not None:
            candidate = _as_finite_array(command_absolute_target)
            if candidate.shape != absolute.shape:
                raise ValueError(
                    f"command_absolute_target must have shape {absolute.shape}, got {candidate.shape}"
                )
            absolute[:] = candidate
        row = {
            "timestamp": float(timestamp),
            "monotonic_timestamp": float(monotonic_timestamp),
            "delivery_state": state,
            "qpos": joints,
            "command_action": raw,
            "command_absolute_target": absolute,
            "command_sent": bool(command_sent),
            "command_generation": -1 if command_generation is None else int(command_generation),
            "command_queue_index": -1 if command_queue_index is None else int(command_queue_index),
            "command_hold": bool(command_hold),
            "execution_state": "" if execution_state is None else str(execution_state),
            "blocked_reason": "" if blocked_reason is None else str(blocked_reason)[:500],
        }
        self._trajectory_rows.append(row)
        self._control_tick_count += 1
        self._enqueue(("control", row))

    def record_camera_frames(
        self,
        images: Mapping[str, np.ndarray],
        timestamps: Mapping[str, float],
        *,
        monotonic_timestamp: float | None = None,
        frame_group: str | None = None,
    ) -> None:
        """Queue RGB CHW camera frames and their source timestamps for video."""
        if not self.is_active:
            return
        for key, value in images.items():
            frame = np.asarray(value, dtype=np.uint8)
            if frame.ndim != 3:
                raise ValueError(f"camera frame {key!r} must be 3D, got {frame.shape}")
            if frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
            if frame.shape[-1] != 3:
                raise ValueError(f"camera frame {key!r} must have 3 RGB channels, got {frame.shape}")
            ts = float(timestamps.get(key, time.time()))
            if not np.isfinite(ts):
                raise ValueError(f"camera timestamp for {key!r} must be finite")
            self._enqueue(
                (
                    "video",
                    _VideoFrame(
                        camera_key=str(key),
                        frame=np.array(frame, copy=True),
                        timestamp=ts,
                        monotonic_timestamp=(
                            None
                            if monotonic_timestamp is None
                            else float(monotonic_timestamp)
                        ),
                        frame_group=None if frame_group is None else str(frame_group),
                    ),
                )
            )

    def record_model_result(
        self,
        *,
        launch: Any,
        result: Mapping[str, Any] | None,
        arrived_at: float,
        arrived_monotonic: float,
        protocol: Any,
        accepted: bool | None,
        rejection: Any = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Persist a complete model response, including rejected responses."""
        if not self.is_active:
            return
        self._model_command_count += 1
        sequence = self._model_command_count
        generation = int(getattr(launch, "generation", sequence))
        raw_actions: np.ndarray | None = None
        if result is not None and "actions" in result:
            try:
                raw_actions = np.asarray(result["actions"], dtype=np.float32)
                if not np.all(np.isfinite(raw_actions)):
                    raw_actions = None
            except (TypeError, ValueError):
                raw_actions = None

        filename = f"command_{sequence:06d}_gen_{generation:08d}.npz"
        relative_file = Path("model_commands") / filename
        arrays: dict[str, np.ndarray] = {
            "raw_actions": (
                np.asarray(raw_actions, dtype=np.float32)
                if raw_actions is not None
                else np.empty((0,), dtype=np.float32)
            ),
            "raw_delivery_state": _as_finite_array(getattr(launch, "raw_delivery_state")),
            "qpos": _as_finite_array(getattr(launch, "qpos_m")),
        }
        payload = {
            "sequence": sequence,
            "generation": generation,
            "captured_at": float(getattr(launch, "captured_at")),
            "captured_monotonic": float(getattr(launch, "captured_monotonic")),
            "launched_at": float(getattr(launch, "launched_at")),
            "launched_monotonic": float(getattr(launch, "launched_monotonic")),
            "arrived_at": float(arrived_at),
            "arrived_monotonic": float(arrived_monotonic),
            "image_timestamps": _json_safe(dict(getattr(launch, "image_timestamps", {}) or {})),
            "action_file": str(relative_file),
            "action_shape": None if raw_actions is None else list(raw_actions.shape),
            "accepted": None if accepted is None else bool(accepted),
            "rejection": _json_safe(rejection),
            "error": None if error is None else str(error),
            "protocol": _json_safe({
                "schema": getattr(protocol, "schema", None),
                "arm_mode": getattr(protocol, "arm_mode", None),
                "arm_side": getattr(protocol, "arm_side", None),
                "state_dim": getattr(protocol, "state_dim", None),
                "action_dim": getattr(protocol, "action_dim", None),
                "action_semantics": getattr(protocol, "action_semantics", None),
                "gripper_semantics": getattr(protocol, "gripper_semantics", None),
                "camera_keys": list(getattr(protocol, "camera_keys", ()) or ()),
                "action_hz": getattr(protocol, "action_hz", None),
                "contract_version": getattr(protocol, "contract_version", None),
            }),
        }
        if result is not None:
            for key in ("execution_control", "transport_timing", "_client_transport_timing"):
                if key in result:
                    payload[key] = _json_safe(result[key])
        self._enqueue(("model", payload, arrays))

    def stop(self, *, reason: str = "stopped") -> Path | None:
        """Flush all records, close MP4 writers, and return the run directory."""
        if not self.is_active:
            return self.run_dir
        assert self._queue is not None
        self._enqueue(("metadata", {"stopped_reason": str(reason)[:200]}), force=True)
        self._enqueue(_STOP, force=True)
        if self._worker is not None:
            self._worker.join(timeout=30.0)
            if self._worker.is_alive():
                LOGGER.error("Timed out flushing deployment recording: %s", self.run_dir)
        for writer in self._video_writers.values():
            try:
                writer.release()
            except Exception:
                LOGGER.exception("Failed to close deployment video writer")
        self._video_writers.clear()
        self._write_trajectory_npz()
        self._metadata.update(
            {
                "finished_at": time.time(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "stopped_reason": str(reason)[:200],
                "control_tick_count": self._control_tick_count,
                "model_command_count": self._model_command_count,
                "dropped_event_count": self._dropped_event_count,
                "video_cameras": sorted(self._video_paths),
                "video_fallback_cameras": sorted(self._video_frame_fallback_dirs),
            }
        )
        self._write_metadata()
        for stream in (self._trajectory_file, self._model_file, self._video_index_file):
            if stream is not None:
                stream.close()
        self._trajectory_file = None
        self._model_file = None
        self._video_index_file = None
        self._queue = None
        self._worker = None
        self._closed = True
        LOGGER.info("Deployment recording saved: %s", self.run_dir)
        return self.run_dir

    def _enqueue(self, event: Any, *, force: bool = False) -> None:
        queue = self._queue
        if queue is None:
            return
        try:
            if force:
                queue.put(event, timeout=5.0)
            else:
                queue.put_nowait(event)
        except Full:
            self._dropped_event_count += 1
            if self._dropped_event_count == 1 or self._dropped_event_count % 100 == 0:
                LOGGER.warning("Deployment recording queue full; dropped %d events", self._dropped_event_count)

    def _write_metadata(self) -> None:
        if self.run_dir is None:
            return
        path = self.run_dir / "metadata.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(_json_safe(self._metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp.replace(path)

    def _write_trajectory_npz(self) -> None:
        if self.run_dir is None:
            return
        rows = self._trajectory_rows
        if not rows:
            data = {
                "timestamp": np.empty((0,), dtype=np.float64),
                "monotonic_timestamp": np.empty((0,), dtype=np.float64),
                "delivery_state": np.empty((0, 0), dtype=np.float32),
                "qpos": np.empty((0, 0), dtype=np.float32),
                "command_action": np.empty((0, 0), dtype=np.float32),
                "command_absolute_target": np.empty((0, 0), dtype=np.float32),
                "command_sent": np.empty((0,), dtype=np.bool_),
                "command_generation": np.empty((0,), dtype=np.int64),
                "command_queue_index": np.empty((0,), dtype=np.int64),
                "command_hold": np.empty((0,), dtype=np.bool_),
            }
        else:
            data = {
                "timestamp": np.asarray([row["timestamp"] for row in rows], dtype=np.float64),
                "monotonic_timestamp": np.asarray(
                    [row["monotonic_timestamp"] for row in rows], dtype=np.float64
                ),
                "delivery_state": np.stack([row["delivery_state"] for row in rows]),
                "qpos": np.stack([row["qpos"] for row in rows]),
                "command_action": np.stack([row["command_action"] for row in rows]),
                "command_absolute_target": np.stack(
                    [row["command_absolute_target"] for row in rows]
                ),
                "command_sent": np.asarray([row["command_sent"] for row in rows], dtype=np.bool_),
                "command_generation": np.asarray(
                    [row["command_generation"] for row in rows], dtype=np.int64
                ),
                "command_queue_index": np.asarray(
                    [row["command_queue_index"] for row in rows], dtype=np.int64
                ),
                "command_hold": np.asarray([row["command_hold"] for row in rows], dtype=np.bool_),
            }
        destination = self.run_dir / "trajectory.npz"
        temp = destination.with_suffix(".npz.tmp")
        # np.savez appends .npz when handed a path string. Use an open handle so
        # the temporary name stays exactly as requested for the atomic replace.
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **data)
        temp.replace(destination)

    def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            event = self._queue.get()
            try:
                if event is _STOP:
                    return
                kind = event[0]
                if kind == "control":
                    self._write_control_json(event[1])
                elif kind == "video":
                    self._write_video_frame(event[1])
                elif kind == "model":
                    self._write_model(event[1], event[2])
                elif kind == "metadata":
                    self._metadata.update(_json_safe(event[1]))
                    self._write_metadata()
            except Exception:
                LOGGER.exception("Failed to write deployment recording event")
            finally:
                self._queue.task_done()

    def _write_control_json(self, row: Mapping[str, Any]) -> None:
        if self._trajectory_file is None:
            return
        payload = {
            key: (_json_safe(value.tolist()) if isinstance(value, np.ndarray) else _json_safe(value))
            for key, value in row.items()
        }
        self._trajectory_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_model(self, payload: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        if self.run_dir is None or self._model_file is None:
            return
        relative_file = Path(str(payload["action_file"]))
        target = self.run_dir / relative_file
        temp = target.with_suffix(".npz.tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temp.replace(target)
        self._model_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _ensure_video_writer(self, frame: _VideoFrame) -> Any | None:
        if self.run_dir is None:
            return None
        if frame.camera_key in self._video_writers:
            return self._video_writers[frame.camera_key]
        height, width = frame.frame.shape[:2]
        video_path = self.run_dir / "videos" / f"{frame.camera_key}.mp4"
        try:
            import cv2

            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (int(width), int(height)),
            )
            if writer.isOpened():
                self._video_writers[frame.camera_key] = writer
                self._video_paths[frame.camera_key] = video_path
                return writer
            writer.release()
        except Exception:
            LOGGER.exception("Failed to initialize MP4 writer for camera %s", frame.camera_key)
        fallback = self.run_dir / "video_frames" / frame.camera_key
        fallback.mkdir(parents=True, exist_ok=True)
        self._video_frame_fallback_dirs[frame.camera_key] = fallback
        return None

    def _write_video_frame(self, frame: _VideoFrame) -> None:
        if self.run_dir is None or self._video_index_file is None:
            return
        import cv2

        writer = self._ensure_video_writer(frame)
        frame_index = self._video_frame_counts.get(frame.camera_key, 0)
        self._video_frame_counts[frame.camera_key] = frame_index + 1
        rgb = np.ascontiguousarray(frame.frame)
        if writer is not None:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            relative_path = self._video_paths[frame.camera_key].relative_to(self.run_dir)
            storage = str(relative_path)
        else:
            fallback = self._video_frame_fallback_dirs[frame.camera_key]
            image_path = fallback / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            storage = str(image_path.relative_to(self.run_dir))
        self._video_index_file.write(
            json.dumps(
                {
                    "camera": frame.camera_key,
                    "frame_index": frame_index,
                    "timestamp": frame.timestamp,
                    "monotonic_timestamp": frame.monotonic_timestamp,
                    "frame_group": frame.frame_group,
                    "storage": storage,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
