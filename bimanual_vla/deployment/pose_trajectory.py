"""Causal Cartesian reference shaping before IK; no actuator ownership.

Translation is in metres and rotations are matrices in the robot base frame.
Angular velocity is a spatial (base-frame) vector. The shortest SO(3) error is
used, never Euler-angle interpolation. This is an online reference shaper at
the policy/control rate, not a timed chunk planner or a motor feedback loop.
The downstream joint Ruckig writer remains responsible for physical joint
velocity/acceleration/jerk bounds. Pose velocities are interval reference
increments divided by dt; acceleration limits bound their successive changes.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseTrajectoryConfig:
    # Historical 20 Hz bounds retained during the stopping-law correction.
    # Dataset-specific faster settings remain offline candidates, not validated
    # hardware limits. Shaping remains opt-in.
    max_linear_speed_m_s: float = 0.20
    max_linear_acceleration_m_s2: float = 1.20
    max_angular_speed_rad_s: float = 0.75
    max_angular_acceleration_rad_s2: float = 8.0
    response_time_s: float = 0.06
    max_position_tracking_error_m: float = 0.10
    max_rotation_tracking_error_rad: float = 0.80
    max_update_dt_s: float = 0.15

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"pose trajectory {name} must be finite and positive")


def _vector(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"invalid pose trajectory {name}")
    return result.copy()


def _rotation(value):
    result = _vector(value, (3, 3), "rotation")
    if (not np.allclose(result.T @ result, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(result), 1.0, atol=1e-6)):
        raise ValueError("pose trajectory rotation must be in SO(3)")
    return result


def _reference_velocity(error, previous, *, dt, speed_limit, acceleration_limit, response_time):
    """Bound a geodesic reference step by response, slew and stopping distance.

    A feasible aligned step stays on the segment to the target. Reserve enough
    remaining distance to stop: s*dt + s**2/(2*a) <= distance. The intersection
    with ||s*direction-previous|| <= a*dt preserves the vector slew limit.

    A suddenly changed target can make that intersection empty (e.g. a goal
    inside the stopping distance). Brake the existing velocity instead of
    teleporting the reference or pretending instantaneous stopping is possible.
    Supports (..., 3) arrays for offline production-code replay.
    """
    distance = np.linalg.norm(error, axis=-1)
    old_speed = np.linalg.norm(previous, axis=-1)
    direction = error / np.maximum(distance[..., None], 1e-15)
    parallel = np.sum(previous * direction, axis=-1)
    perpendicular_sq = np.maximum(0., old_speed**2 - parallel**2)
    budget = acceleration_limit * dt
    radius = np.sqrt(np.maximum(0., budget**2 - perpendicular_sq))
    lower = np.maximum(0., parallel - radius)
    # Stable positive root of s*dt + s**2/(2*a) = distance.
    stopping_speed = 2 * acceleration_limit * distance / (
        np.sqrt(budget**2 + 2 * acceleration_limit * distance) + budget)
    upper = np.minimum(np.minimum(parallel + radius, speed_limit), stopping_speed)
    # Exact first-order response has 0 <= gain <= 1 at every valid update dt.
    desired_speed = -math.expm1(-dt / response_time) * distance / dt
    feasible = ((distance > 1e-12) & (perpendicular_sq <= budget**2)
                & (upper >= lower))
    speed = np.minimum(np.maximum(desired_speed, lower), np.maximum(upper, 0.))
    braking_velocity = previous * np.maximum(0., 1 - budget / np.maximum(old_speed, 1e-15))[..., None]
    velocity = np.where(feasible[..., None], direction * speed[..., None], braking_velocity)
    return velocity, {
        "speed_limited": desired_speed > speed_limit,
        "acceleration_limited": np.linalg.norm(direction * desired_speed[..., None] - previous, axis=-1) > budget,
        "stopping_distance_limited": desired_speed > stopping_speed,
        "braking_for_target_change": ~feasible & (old_speed > 1e-12),
    }


class PoseTrajectory:
    """Retain position and twist across target changes, with norm bounds.

    Feedback initializes the reference and then only guards tracking. Replacing
    the generated state with noisy measurements every tick defeats continuity.
    Discontinuities, stale clocks and invalid feedback raise before mutation.
    Restart explicitly from stationary measured feedback after a pause/fault.
    """

    def __init__(self, xyz, rotation, config=None):
        self.config = config or PoseTrajectoryConfig()
        self.xyz = _vector(xyz, (3,), "position")
        self.rotation = _rotation(rotation)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.last_metrics = {}

    def update(self, xyz, rotation, *, dt, feedback_xyz, feedback_rotation):
        target = _vector(xyz, (3,), "target position")
        target_rotation = _rotation(rotation)
        measured = _vector(feedback_xyz, (3,), "feedback position")
        measured_rotation = _rotation(feedback_rotation)
        cfg = self.config
        if not math.isfinite(float(dt)) or not 0 < dt <= cfg.max_update_dt_s:
            raise ValueError("pose trajectory update clock is stale or invalid")
        position_tracking = float(np.linalg.norm(self.xyz - measured))
        rotation_tracking = float(np.linalg.norm(Rotation.from_matrix(
            self.rotation @ measured_rotation.T).as_rotvec()))
        if (position_tracking > cfg.max_position_tracking_error_m
                or rotation_tracking > cfg.max_rotation_tracking_error_rad):
            raise ValueError("pose trajectory feedback tracking error exceeded limit")
        error = target - self.xyz
        angular_error = Rotation.from_matrix(target_rotation @ self.rotation.T).as_rotvec()
        velocity, linear_flags = _reference_velocity(
            error, self.linear_velocity, dt=dt, speed_limit=cfg.max_linear_speed_m_s,
            acceleration_limit=cfg.max_linear_acceleration_m_s2, response_time=cfg.response_time_s)
        angular_velocity, angular_flags = _reference_velocity(
            angular_error, self.angular_velocity, dt=dt, speed_limit=cfg.max_angular_speed_rad_s,
            acceleration_limit=cfg.max_angular_acceleration_rad_s2, response_time=cfg.response_time_s)
        velocity = _vector(velocity, (3,), "candidate linear velocity")
        angular_velocity = _vector(angular_velocity, (3,), "candidate angular velocity")
        next_xyz = self.xyz + velocity * dt
        next_rotation = Rotation.from_rotvec(angular_velocity * dt).as_matrix() @ self.rotation
        next_xyz = _vector(next_xyz, (3,), "candidate position")
        next_rotation = _rotation(next_rotation)
        next_position_tracking = float(np.linalg.norm(next_xyz - measured))
        next_rotation_tracking = float(np.linalg.norm(Rotation.from_matrix(
            next_rotation @ measured_rotation.T).as_rotvec()))
        if (next_position_tracking > cfg.max_position_tracking_error_m
                or next_rotation_tracking > cfg.max_rotation_tracking_error_rad):
            raise ValueError("pose trajectory candidate feedback tracking error exceeded limit")
        self.last_metrics = {
            "algorithm": "stopping_distance_reference_v2",
            "dt_s": float(dt),
            "raw_xyz_m": target.tolist(),
            "reference_xyz_m": next_xyz.tolist(),
            "position_lag_m": float(np.linalg.norm(target - next_xyz)),
            "rotation_lag_rad": float(np.linalg.norm(Rotation.from_matrix(
                target_rotation @ next_rotation.T).as_rotvec())),
            "position_tracking_error_m": position_tracking,
            "rotation_tracking_error_rad": rotation_tracking,
            "candidate_position_tracking_error_m": next_position_tracking,
            "candidate_rotation_tracking_error_rad": next_rotation_tracking,
            "linear_velocity_m_s": velocity.tolist(),
            "angular_velocity_rad_s": angular_velocity.tolist(),
            "linear_acceleration_m_s2": ((velocity - self.linear_velocity) / dt).tolist(),
            "angular_acceleration_rad_s2": ((angular_velocity - self.angular_velocity) / dt).tolist(),
            **{"linear_" + key: bool(value) for key, value in linear_flags.items()},
            **{"angular_" + key: bool(value) for key, value in angular_flags.items()},
        }
        self.xyz, self.rotation = next_xyz, next_rotation
        self.linear_velocity, self.angular_velocity = velocity, angular_velocity
        return next_xyz.copy(), next_rotation.copy(), dict(self.last_metrics)
