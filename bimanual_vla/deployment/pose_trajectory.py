"""Causal Cartesian reference shaping before IK; no actuator ownership.

Translation is in metres and rotations are matrices in the robot base frame.
Angular velocity is a spatial (base-frame) vector. The shortest SO(3) error is
used, never Euler-angle interpolation. This is an online reference shaper at
the policy/control rate, not a timed chunk planner or a motor feedback loop.
The downstream joint Ruckig writer remains responsible for joint jerk limits.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseTrajectoryConfig:
    # 20 Hz fix demonstrations: keep lag small without the tau=0.05 noise gain.
    # FK geometry tuning only; hardware and task success require validation.
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


def _bounded(value, limit):
    norm = float(np.linalg.norm(value))
    return value * min(1.0, limit / max(norm, 1e-15))


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
        # An acceleration-limited velocity tracker. At short distances the
        # proportional velocity converges continuously to zero. Braking may
        # overshoot a reversing target; workspace checks remain mandatory.
        desired_v = _bounded(error / cfg.response_time_s, cfg.max_linear_speed_m_s)
        desired_w = _bounded(angular_error / cfg.response_time_s, cfg.max_angular_speed_rad_s)
        velocity = self.linear_velocity + _bounded(
            desired_v - self.linear_velocity, cfg.max_linear_acceleration_m_s2 * dt)
        angular_velocity = self.angular_velocity + _bounded(
            desired_w - self.angular_velocity, cfg.max_angular_acceleration_rad_s2 * dt)
        next_xyz = self.xyz + 0.5 * (self.linear_velocity + velocity) * dt
        next_rotation = Rotation.from_rotvec(
            0.5 * (self.angular_velocity + angular_velocity) * dt).as_matrix() @ self.rotation
        next_position_tracking = float(np.linalg.norm(next_xyz - measured))
        next_rotation_tracking = float(np.linalg.norm(Rotation.from_matrix(
            next_rotation @ measured_rotation.T).as_rotvec()))
        if (next_position_tracking > cfg.max_position_tracking_error_m
                or next_rotation_tracking > cfg.max_rotation_tracking_error_rad):
            raise ValueError("pose trajectory candidate feedback tracking error exceeded limit")
        self.last_metrics = {
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
            "linear_speed_limited": bool(np.linalg.norm(error / cfg.response_time_s) > cfg.max_linear_speed_m_s),
            "angular_speed_limited": bool(np.linalg.norm(angular_error / cfg.response_time_s) > cfg.max_angular_speed_rad_s),
            "linear_acceleration_limited": bool(np.linalg.norm(desired_v - self.linear_velocity) > cfg.max_linear_acceleration_m_s2 * dt),
            "angular_acceleration_limited": bool(np.linalg.norm(desired_w - self.angular_velocity) > cfg.max_angular_acceleration_rad_s2 * dt),
        }
        self.xyz, self.rotation = next_xyz, next_rotation
        self.linear_velocity, self.angular_velocity = velocity, angular_velocity
        return next_xyz.copy(), next_rotation.copy(), dict(self.last_metrics)
