"""Low-level trajectory shaping helpers for Piper deployment.

The policy remains a position reference.  These helpers turn that reference
into a bounded, continuous joint trajectory before a Piper ``JointCtrl`` is
published.  They are intentionally dependency-light so they can also be
used by offline tests and the controlled return-to-home path.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


class TrajectoryTrackingError(RuntimeError):
    """The shaped trajectory has fallen too far behind measured feedback."""


class JerkLimitedJointTrajectory:
    """Stateful velocity/acceleration/jerk limited position controller.

    ``qpos`` is laid out as one 7D block per arm: six joints followed by a
    gripper value.  Grippers are deliberately excluded from the joint state;
    callers should rate-limit them independently.
    """

    def __init__(
        self,
        initial: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        max_speed_rad_s: float = 0.30,
        max_acceleration_rad_s2: float = 0.80,
        max_jerk_rad_s3: float = 4.0,
        smoothing_cutoff_hz: float = 3.0,
        tracking_time_constant_s: float = 0.25,
        command_lookahead_rad: float = 0.02,
        max_tracking_error_rad: float = 0.35,
        joint_indices: Iterable[int] | None = None,
    ) -> None:
        initial = np.asarray(initial, dtype=np.float32)
        lower = np.asarray(lower, dtype=np.float32)
        upper = np.asarray(upper, dtype=np.float32)
        if initial.ndim != 1 or lower.shape != initial.shape or upper.shape != initial.shape:
            raise ValueError("initial/lower/upper must be matching 1D arrays")
        if not np.isfinite(initial).all():
            raise ValueError("initial joint state must be finite")
        if joint_indices is None:
            joint_indices = [
                index
                for arm_start in range(0, len(initial), 7)
                for index in range(arm_start, min(arm_start + 6, len(initial)))
            ]
        self.joint_indices = np.asarray(list(joint_indices), dtype=np.int64)
        if self.joint_indices.ndim != 1 or not len(self.joint_indices):
            raise ValueError("joint_indices must contain at least one index")
        if np.any(self.joint_indices < 0) or np.any(self.joint_indices >= len(initial)):
            raise ValueError("joint_indices contain an out-of-range index")
        for name, value in (
            ("max_speed_rad_s", max_speed_rad_s),
            ("max_acceleration_rad_s2", max_acceleration_rad_s2),
            ("max_jerk_rad_s3", max_jerk_rad_s3),
            ("smoothing_cutoff_hz", smoothing_cutoff_hz),
            ("tracking_time_constant_s", tracking_time_constant_s),
            ("max_tracking_error_rad", max_tracking_error_rad),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(float(command_lookahead_rad)) or float(command_lookahead_rad) < 0:
            raise ValueError("command_lookahead_rad must be non-negative")
        self.lower = lower
        self.upper = upper
        self.max_speed = float(max_speed_rad_s)
        self.max_acceleration = float(max_acceleration_rad_s2)
        self.max_jerk = float(max_jerk_rad_s3)
        self.cutoff_hz = float(smoothing_cutoff_hz)
        self.tracking_time_constant = float(tracking_time_constant_s)
        self.lookahead = float(command_lookahead_rad)
        self.max_tracking_error = float(max_tracking_error_rad)
        self.reset(initial)

    def reset(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != self.lower.shape or not np.isfinite(state).all():
            raise ValueError("trajectory reset state has an invalid shape or value")
        self.filtered_reference = state.copy()
        self.position = state.copy()
        self.velocity = np.zeros_like(state, dtype=np.float32)
        self.acceleration = np.zeros_like(state, dtype=np.float32)

    def update(
        self,
        feedback: np.ndarray,
        proposed: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, list[int]]:
        feedback = np.asarray(feedback, dtype=np.float32)
        proposed = np.asarray(proposed, dtype=np.float32)
        if feedback.shape != self.lower.shape or proposed.shape != self.lower.shape:
            raise ValueError("feedback and proposed states must match trajectory dimensions")
        if not np.isfinite(feedback).all() or not np.isfinite(proposed).all():
            raise ValueError("feedback and proposed states must be finite")
        dt = float(np.clip(dt, 0.001, 0.1))
        indices = self.joint_indices

        alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_hz * dt)
        self.filtered_reference[indices] += alpha * (
            proposed[indices] - self.filtered_reference[indices]
        )
        self.filtered_reference[indices] = np.clip(
            self.filtered_reference[indices], self.lower[indices], self.upper[indices]
        )

        error = self.filtered_reference[indices] - self.position[indices]
        natural_frequency = 1.0 / self.tracking_time_constant
        desired_acceleration = np.clip(
            natural_frequency**2 * error
            - 2.0 * natural_frequency * self.velocity[indices],
            -self.max_acceleration,
            self.max_acceleration,
        )
        acceleration_step = np.clip(
            desired_acceleration - self.acceleration[indices],
            -self.max_jerk * dt,
            self.max_jerk * dt,
        )
        self.acceleration[indices] = np.clip(
            self.acceleration[indices] + acceleration_step,
            -self.max_acceleration,
            self.max_acceleration,
        )
        self.velocity[indices] = np.clip(
            self.velocity[indices] + self.acceleration[indices] * dt,
            -self.max_speed,
            self.max_speed,
        )
        self.position[indices] += self.velocity[indices] * dt

        before_limit = self.position[indices].copy()
        self.position[indices] = np.clip(
            self.position[indices], self.lower[indices], self.upper[indices]
        )
        position_limited = np.abs(self.position[indices] - before_limit) > 1e-7
        if np.any(position_limited):
            limited_indices = indices[position_limited]
            self.velocity[limited_indices] = 0.0
            self.acceleration[limited_indices] = 0.0

        tracking_error = np.abs(self.position[indices] - feedback[indices])
        if np.any(tracking_error > self.max_tracking_error):
            worst = int(indices[int(np.argmax(tracking_error))])
            raise TrajectoryTrackingError(
                f"trajectory tracking error exceeded limit: joint={worst} "
                f"error={float(np.max(tracking_error)):.5f}rad"
            )

        lookahead = self.lookahead * self.velocity[indices] / self.max_speed
        shaped = self.position[indices] + lookahead
        before_firmware_limit = shaped.copy()
        shaped = np.clip(shaped, self.lower[indices], self.upper[indices])

        target = feedback.copy()
        target[indices] = shaped
        clipped = position_limited | (
            np.abs(shaped - before_firmware_limit) > 1e-7
        )
        return target.astype(np.float32), indices[clipped].tolist()

    def metrics(self) -> dict[str, list[float]]:
        """Return finite state for telemetry without exposing mutable arrays."""
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "acceleration": self.acceleration.tolist(),
        }


def smootherstep(value: float) -> float:
    """Quintic interpolation with zero slope and curvature at both ends."""
    x = float(np.clip(value, 0.0, 1.0))
    return x**3 * (x * (x * 6.0 - 15.0) + 10.0)


def gripper_open_lookahead(
    opening: np.ndarray,
    future_openings: np.ndarray,
    *,
    lookahead_steps: int,
) -> np.ndarray:
    """Advance opening requests but never advance a closing request."""
    current = np.asarray(opening, dtype=np.float64)
    future = np.asarray(future_openings, dtype=np.float64)
    if current.ndim != 1 or future.ndim != 2 or future.shape[1] != current.size:
        raise ValueError("opening/future_openings have incompatible shapes")
    if not np.isfinite(current).all() or not np.isfinite(future).all():
        raise ValueError("opening/future_openings must be finite")
    if lookahead_steps <= 0 or not len(future):
        return current.copy()
    return np.maximum(current, np.max(future[: int(lookahead_steps) + 1], axis=0))


def rate_limit_grippers(
    current_m: np.ndarray,
    proposed_m: np.ndarray,
    *,
    max_speed_m_s: float,
    dt: float,
    previous_target_m: np.ndarray | None = None,
    max_command_lead_m: float = 0.012,
) -> np.ndarray:
    """Rate-limit opening in metres while bounding command lead over feedback."""
    current = np.asarray(current_m, dtype=np.float64)
    proposed = np.asarray(proposed_m, dtype=np.float64)
    if current.shape != proposed.shape or current.ndim != 1:
        raise ValueError("current_m and proposed_m must be matching 1D arrays")
    previous = current if previous_target_m is None else np.asarray(previous_target_m, dtype=np.float64)
    if previous.shape != current.shape:
        raise ValueError("previous_target_m must match current_m")
    step = float(max_speed_m_s) * max(0.001, float(dt))
    lead = max(float(max_command_lead_m), step)
    result = previous + np.clip(proposed - previous, -step, step)
    result = np.clip(result, current - lead, current + lead)
    return np.clip(result, 0.0, 0.07).astype(np.float32)
