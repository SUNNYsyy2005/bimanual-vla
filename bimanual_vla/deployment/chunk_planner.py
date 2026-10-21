"""Local multi-point reference planning; no hardware or cloud API calls.

The result is an immutable, clocked sequence of Ruckig states. RTC samples and
the actuator writer consume this same sequence, rather than separate rollouts.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from .servo import (JOINT_POSITION_NUMERICAL_TOLERANCE_RAD,
                    RuckigJointPlanner, ServoConfig, ServoError)


@dataclass(frozen=True)
class ChunkPlan:
    times: np.ndarray
    positions: np.ndarray  # joint rad, gripper metres
    velocities: np.ndarray  # joints only
    accelerations: np.ndarray
    generation: int
    source_times: np.ndarray
    source_indices: np.ndarray
    sample_generations: np.ndarray | None = None
    sample_source_indices: np.ndarray | None = None

    def __post_init__(self):
        if self.sample_generations is None:
            object.__setattr__(self, "sample_generations", np.full(len(self.times), self.generation))
        if self.sample_source_indices is None:
            indices = np.clip(np.searchsorted(self.source_times, self.times, side="left"),
                              0, len(self.source_times) - 1)
            object.__setattr__(self, "sample_source_indices", self.source_indices[indices])
        for key in ("times", "positions", "velocities", "accelerations",
                    "source_times", "source_indices", "sample_generations", "sample_source_indices"):
            value = np.array(getattr(self, key), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, key, value)

    def index_at(self, at: float) -> int:
        return int(np.clip(np.searchsorted(self.times, at, side="right") - 1,
                           0, len(self.times) - 1))

    def context_at(self, at: float) -> dict:
        # Installation can retain an old prefix. Label the state actually
        # published, not the newest chunk merely waiting to take over.
        index = self.index_at(at)
        source = int(self.sample_source_indices[index])
        return dict(generation=int(self.sample_generations[index]),
                    source_index=source, queue_index=source)


def compile_chunk(initial, velocity, acceleration, lower, upper, config: ServoConfig,
                  targets, target_times, source_indices, *, start: float,
                  generation: int) -> ChunkPlan:
    """Preview a smoothed multi-point curve and materialize bounded 100 Hz states.

    A three-point spatial filter and shape-preserving interpolation suppress
    isolated zigzags. The terminal velocity comes from the *future curve*, not
    a backward difference of asynchronously received targets. Ruckig remains
    responsible for q/v/a continuity and analytic position-extrema checks.
    """
    initial = np.asarray(initial, dtype=float)
    targets = np.asarray(targets, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    source_indices = np.asarray(source_indices, dtype=int)
    if (initial.ndim != 1 or len(initial) not in (7, 14) or not np.isfinite(initial).all()
            or targets.shape != (len(target_times), len(initial))
            or source_indices.shape != target_times.shape or len(target_times) < 2
            or not np.isfinite(targets).all() or not np.isfinite(target_times).all()
            or np.any(np.diff(target_times) <= 0) or target_times[-1] <= start):
        raise ServoError("invalid or expired timed chunk for reference planning")
    # Spliced states come directly from Ruckig: use the same joint numerical
    # tolerance as its extrema/output guard. Grippers remain in metres.
    tolerance = np.full(initial.shape, JOINT_POSITION_NUMERICAL_TOLERANCE_RAD)
    tolerance[6::7] = 1e-9
    if (np.any(initial < np.asarray(lower) - tolerance)
            or np.any(initial > np.asarray(upper) + tolerance)):
        raise ServoError("initial chunk state exceeds physical position bounds")
    joints = np.array([i for i in range(len(initial)) if i % 7 != 6])
    grippers = np.array([i for i in range(len(initial)) if i % 7 == 6])
    planner = RuckigJointPlanner(initial[joints], np.asarray(lower)[joints],
                                 np.asarray(upper)[joints], config)
    planner.set_state(initial[joints], velocity, acceleration)
    if (np.any(targets < np.asarray(lower) - tolerance)
            or np.any(targets > np.asarray(upper) + tolerance)):
        raise ServoError("chunk target exceeds physical position bounds")
    smooth = targets[:, joints].copy()
    if len(smooth) >= 3:
        smooth[1:-1] = (smooth[:-2] + 2 * smooth[1:-1] + smooth[2:]) / 4
    curve = PchipInterpolator(target_times, smooth, axis=0, extrapolate=False)
    derivative = curve.derivative()
    period = 1 / config.hz
    reference_period = float(np.median(np.diff(target_times)))
    retarget_ticks = max(1, round(reference_period / period))
    preview = 2 * reference_period
    count = int(np.ceil((target_times[-1] - start) / period))
    # Include the starting state, so a splice preserves q/v/a exactly.
    ts, qs, vs, accs = [start], [initial.copy()], [planner.velocity.copy()], [planner.acceleration.copy()]
    grip = initial[grippers].copy()
    for tick in range(1, count + 1):
        at = start + tick * period
        if (tick - 1) % retarget_ticks == 0:
            future = float(np.clip(at + preview, target_times[0], target_times[-1]))
            terminal_v = derivative(future) if future < target_times[-1] else np.zeros(len(joints))
            planner.retarget(curve(future), terminal_v)
        command = initial.copy()
        command[joints] = planner.advance()
        desired_grip = np.array([np.interp(at, target_times, targets[:, i]) for i in grippers])
        grip += np.clip(desired_grip - grip, -config.max_gripper_velocity * period,
                        config.max_gripper_velocity * period)
        command[grippers] = grip
        ts.append(at); qs.append(command); vs.append(planner.velocity.copy()); accs.append(planner.acceleration.copy())
    # The finite chunk ends with a bounded stop, never a stale-target timeout
    # while a valid chunk still has references to execute.
    planner.brake()
    planner._check_planned_bounds()
    for _ in range(int(config.hz * 3)):
        if max(np.max(abs(planner.velocity)), np.max(abs(planner.acceleration))) < 1e-8:
            break
        command = qs[-1].copy()
        command[joints] = planner.advance()
        ts.append(ts[-1] + period); qs.append(command); vs.append(planner.velocity.copy()); accs.append(planner.acceleration.copy())
    else:
        raise ServoError("chunk planner could not produce a bounded terminal stop")
    return ChunkPlan(np.asarray(ts), np.asarray(qs), np.asarray(vs), np.asarray(accs),
                     generation, target_times, source_indices)
