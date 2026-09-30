"""Streaming jitter metrics for joint commands actually published to Piper."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def model_joint_positions(actions: np.ndarray, *, schema: str, arm_mode: str) -> np.ndarray | None:
    """Select only absolute joint targets; gripper fractions have different units."""
    if schema != "joint" or arm_mode not in {"single", "bimanual"}:
        return None
    values = np.asarray(actions, dtype=np.float64)
    arm_count = 2 if arm_mode == "bimanual" else 1
    if (
        values.ndim != 2 or values.shape[1] != arm_count * 7
        or not len(values) or not np.isfinite(values).all()
    ):
        return None
    columns = [index for arm in range(arm_count) for index in range(arm * 7, arm * 7 + 6)]
    return values[:, columns]


class TrajectoryJitterMonitor:
    """Measure contiguous, fixed-rate command samples without storing chunks.

    Holds and missed/skipped command rows break velocity and acceleration
    continuity. A generation switch still measures the position jump between
    the last published row of the old chunk and the first of the new chunk.
    """

    def __init__(self, control_hz: float, *, basis: str = "published_joint_commands_excluding_gripper_and_holds") -> None:
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be positive and finite")
        self.control_hz = float(control_hz)
        self.basis = str(basis)
        self._tick = 0
        self._generation: int | None = None
        self._last_position: np.ndarray | None = None
        self._previous_position: np.ndarray | None = None
        self._last_queue_index: int | None = None
        self._last_command_at: float | None = None
        self._last_tick: int | None = None
        self._last_velocity: np.ndarray | None = None
        self._pending_momentum: tuple[int, np.ndarray, np.ndarray] | None = None
        self._chunk_samples = 0
        self._chunk_accel_sum = 0.0
        self._chunk_accel_count = 0
        self._accel_sum = 0.0
        self._accel_count = 0
        self._jump_sum = 0.0
        self._jump_count = 0
        self._cosine_sum = 0.0
        self._cosine_count = 0
        self._stationary_boundary_count = 0
        self._latest_boundary: dict[str, Any] | None = None

    def _regular_interval(self, command_at: float) -> bool:
        if self._last_command_at is None:
            return False
        elapsed = command_at - self._last_command_at
        period = 1.0 / self.control_hz
        return 0.5 * period <= elapsed <= 1.5 * period

    def observe(
        self,
        *,
        joints_rad: np.ndarray | None,
        generation: int | None,
        queue_index: int | None,
        command_at: float | None,
        hold: bool = False,
    ) -> list[dict[str, Any]]:
        """Consume one control tick and return completed metric events."""
        self._tick += 1
        if joints_rad is None or generation is None or queue_index is None or hold:
            self._previous_position = None
            self._last_velocity = None
            self._pending_momentum = None
            return []
        position = np.asarray(joints_rad, dtype=np.float64)
        if position.ndim != 1 or not position.size or not np.isfinite(position).all():
            raise ValueError("joints_rad must be a finite nonempty vector")
        if command_at is None or not math.isfinite(command_at):
            raise ValueError("command_at must be finite for a published command")
        generation = int(generation)
        queue_index = int(queue_index)
        events: list[dict[str, Any]] = []
        previous_generation = self._generation
        changed = previous_generation is not None and generation != previous_generation
        adjacent = self._last_tick == self._tick - 1 and self._regular_interval(command_at)
        if changed:
            events.append({
                "event": "chunk_completed",
                "generation": previous_generation,
                "last_command_monotonic": self._last_command_at,
                "command_samples": self._chunk_samples,
                "intra_accel_mean_rad_per_step2": (
                    self._chunk_accel_sum / self._chunk_accel_count
                    if self._chunk_accel_count else None
                ),
                "intra_accel_samples": self._chunk_accel_count,
            })
            if self._last_position is not None and self._last_position.shape == position.shape:
                jump = float(np.linalg.norm(position - self._last_position))
                self._jump_sum += jump
                self._jump_count += 1
                self._latest_boundary = {
                    "previous_generation": previous_generation,
                    "generation": generation,
                    "boundary_monotonic": float(command_at),
                    "position_jump_rad_l2": jump,
                    "momentum_cosine": None,
                }
                events.append({"event": "chunk_boundary", **self._latest_boundary})
                if adjacent and self._last_velocity is not None:
                    self._pending_momentum = (
                        generation, self._last_velocity.copy(), position.copy()
                    )
                else:
                    self._pending_momentum = None
            self._chunk_samples = 0
            self._chunk_accel_sum = 0.0
            self._chunk_accel_count = 0
            self._previous_position = None
            self._last_velocity = None
        elif previous_generation is None:
            self._generation = generation

        consecutive = (
            not changed
            and adjacent
            and self._last_queue_index is not None
            and queue_index == self._last_queue_index + 1
            and self._last_position is not None
            and self._last_position.shape == position.shape
        )
        if consecutive:
            velocity = position - self._last_position
            if self._pending_momentum is not None:
                pending_generation, end_velocity, first_position = self._pending_momentum
                if pending_generation == generation and np.array_equal(first_position, self._last_position):
                    denominator = float(np.linalg.norm(end_velocity) * np.linalg.norm(velocity))
                    if denominator > 1e-8:
                        cosine = float(np.dot(end_velocity, velocity) / denominator)
                        cosine = max(-1.0, min(1.0, cosine))
                        self._cosine_sum += cosine
                        self._cosine_count += 1
                        assert self._latest_boundary is not None
                        self._latest_boundary["momentum_cosine"] = cosine
                        events.append({
                            "event": "chunk_boundary_momentum",
                            **self._latest_boundary,
                            "second_command_monotonic": float(command_at),
                        })
                    else:
                        self._stationary_boundary_count += 1
                self._pending_momentum = None
            if self._previous_position is not None:
                acceleration = float(np.linalg.norm(position - 2.0 * self._last_position + self._previous_position))
                self._accel_sum += acceleration
                self._accel_count += 1
                self._chunk_accel_sum += acceleration
                self._chunk_accel_count += 1
            self._previous_position = self._last_position
            self._last_velocity = velocity
        else:
            self._previous_position = None
            self._last_velocity = None
            if not changed:
                self._pending_momentum = None

        self._generation = generation
        self._last_position = position.copy()
        self._last_queue_index = queue_index
        self._last_command_at = float(command_at)
        self._last_tick = self._tick
        self._chunk_samples += 1
        return events

    def summary(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "nominal_control_hz": self.control_hz,
            "intra_accel_mean_rad_per_step2": (
                self._accel_sum / self._accel_count if self._accel_count else None
            ),
            "intra_accel_samples": self._accel_count,
            "boundary_jump_mean_rad_l2": (
                self._jump_sum / self._jump_count if self._jump_count else None
            ),
            "boundary_jump_samples": self._jump_count,
            "boundary_momentum_cosine_mean": (
                self._cosine_sum / self._cosine_count if self._cosine_count else None
            ),
            "boundary_momentum_samples": self._cosine_count,
            "stationary_boundary_count": self._stationary_boundary_count,
            "latest_boundary": dict(self._latest_boundary) if self._latest_boundary else None,
        }

    def finish(self) -> dict[str, Any] | None:
        """Return the final chunk's per-chunk acceleration at run shutdown."""
        if self._generation is None:
            return None
        event = {
            "event": "chunk_completed",
            "generation": self._generation,
            "last_command_monotonic": self._last_command_at,
            "command_samples": self._chunk_samples,
            "intra_accel_mean_rad_per_step2": (
                self._chunk_accel_sum / self._chunk_accel_count
                if self._chunk_accel_count else None
            ),
            "intra_accel_samples": self._chunk_accel_count,
        }
        self._generation = None
        return event
