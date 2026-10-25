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


def splice_ready_chunk(prepared: ChunkPlan, initial, velocity, acceleration,
                       lower, upper, config: ServoConfig, *, start: float) -> ChunkPlan:
    """Join a prepared curve from the old plan's state at the actual handoff.

    Heavy full-horizon compilation has already finished. A local, analytically
    checked Ruckig transition matches q/v/a at both ends. The prepared suffix
    keeps its original observation clock; it is never shifted to hide latency.
    """
    initial = np.asarray(initial, dtype=float)
    joints = np.array([i for i in range(len(initial)) if i % 7 != 6])
    grippers = np.array([i for i in range(len(initial)) if i % 7 == 6])
    planner = RuckigJointPlanner(initial[joints], np.asarray(lower)[joints],
                                 np.asarray(upper)[joints], config)
    planner.set_state(initial[joints], velocity, acceleration)
    # Join on an existing sample clock. Continuous duration avoids rounding a
    # floating point 0.10 s interval up to 0.11 s and missing the suffix clock.
    planner.input.duration_discretization = planner._ruckig.DurationDiscretization.Continuous
    index = int(np.searchsorted(prepared.times, start + .05, side="left"))
    for _ in range(16):
        if index >= len(prepared.times):
            break
        end = float(prepared.times[index])
        duration = end - start
        planner.input.target_position = prepared.positions[index, joints].tolist()
        planner.input.target_velocity = prepared.velocities[index].tolist()
        planner.input.target_acceleration = prepared.accelerations[index].tolist()
        planner.input.minimum_duration = duration
        try:
            trajectory = planner._check_planned_bounds()
        except ServoError:
            index += max(1, round(.05 * config.hz))
            continue
        if trajectory.duration > duration + 1e-8:
            index = max(index + max(1, round(.05 * config.hz)),
                        int(np.searchsorted(prepared.times, start + trajectory.duration)))
            continue
        clocks = start + np.arange(int(np.ceil(duration * config.hz))) / config.hz
        clocks = clocks[clocks < end - 1e-9]
        samples = [trajectory.at_time(float(at - start)) for at in clocks]
        q, v, a = (np.asarray(x) for x in zip(*samples))
        positions = np.tile(initial, (len(clocks), 1))
        positions[:, joints] = q
        # Exact old state at the splice; exact prepared state at the join.
        positions[0] = initial
        v[0], a[0] = velocity, acceleration
        all_clocks = np.concatenate([clocks, prepared.times[index:]])
        all_positions = np.concatenate([positions, prepared.positions[index:]])
        # Grippers have their own speed limit, not joint q/v/a synchronization.
        # Both the prepared curve and this cone are rate bounded; clipping
        # between the cone bounds preserves the rate bound throughout.
        desired = np.stack([np.interp(all_clocks, prepared.times, prepared.positions[:, i])
                            for i in grippers], axis=1)
        reach = ((all_clocks - start) * config.max_gripper_velocity)[:, None]
        all_positions[:, grippers] = np.clip(desired, initial[grippers] - reach, initial[grippers] + reach)
        return ChunkPlan(all_clocks, all_positions,
                         np.concatenate([v, prepared.velocities[index:]]),
                         np.concatenate([a, prepared.accelerations[index:]]),
                         prepared.generation, prepared.source_times, prepared.source_indices)
    raise ServoError("ready chunk has no bounded q/v/a-continuous splice")


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
    period = 1 / config.hz
    reference_period = float(np.median(np.diff(target_times)))
    retarget_ticks = max(1, round(reference_period / period))
    preview = 2 * reference_period
    count = int(np.ceil((target_times[-1] - start) / period))
    # Include the starting state, so a splice preserves q/v/a exactly.
    ts, qs, vs, accs = [start], [initial.copy()], [planner.velocity.copy()], [planner.acceleration.copy()]
    grip = initial[grippers].copy()
    ticks = np.arange(1, count + 1)
    clocks = start + ticks * period
    future = np.clip(clocks[::retarget_ticks] + preview, target_times[0], target_times[-1])
    future_q, future_v = curve(future), curve.derivative()(future)
    future_v[future >= target_times[-1]] = 0
    desired_grips = np.stack([np.interp(clocks, target_times, targets[:, i]) for i in grippers], axis=1)
    for block, begin in enumerate(range(0, count, retarget_ticks)):
        planner.retarget(future_q[block], future_v[block])
        end = min(begin + retarget_ticks, count)
        positions, velocities, accelerations = planner.sample_checked(end - begin)
        for k, at in enumerate(clocks[begin:end]):
            command = initial.copy()
            command[joints] = positions[k]
            grip += np.clip(desired_grips[begin+k] - grip, -config.max_gripper_velocity * period,
                            config.max_gripper_velocity * period)
            command[grippers] = grip
            ts.append(at); qs.append(command); vs.append(velocities[k]); accs.append(accelerations[k])
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
