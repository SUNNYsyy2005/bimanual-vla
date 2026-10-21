"""Joint reference planning and a guarded actuator publication clock.

Piper owns the motor feedback loops. This module generates position references;
measured feedback initializes planning and guards publication, not a second PID.
Legacy ``servo`` API/CLI names remain for compatibility.

Reference: https://github.com/pantor/ruckig (MIT), version 0.12.2.
Only the local state-to-state API is used. Community intermediate waypoints
use a cloud service and must never be used on this physical control path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import logging
import threading
import time
from typing import Any, Callable

import numpy as np


# Absolute numerical slack for Ruckig states/extrema, not a motion allowance.
# 1e-7 rad is ~0.00000573 degrees: < 1/170 of Piper's 0.001-degree
# command quantum. Keep q/v/a intact instead of clipping generated states.
JOINT_POSITION_NUMERICAL_TOLERANCE_RAD = 1e-7


class ServoError(RuntimeError):
    """A reference, feedback sample, or generated command cannot be executed."""


class ChunkPlanDeferred(ServoError):
    """No new plan was installed; the healthy writer may retain its old plan."""


class ChunkPlanRejected(ServoError):
    """An unpublished candidate is unsafe; reject it without faulting the writer."""


@dataclass(frozen=True)
class ServoConfig:
    hz: float = 100.0
    max_velocity: float = 0.60
    max_acceleration: float = 3.0
    max_jerk: float = 20.0
    max_tracking_error: float = 0.35
    target_timeout_s: float = 0.15
    max_gripper_velocity: float = 0.08
    follow_waypoint_velocity: bool = False

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name == "follow_waypoint_velocity":
                continue
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"servo {name} must be finite and positive")
        if self.target_timeout_s < 2.0 / self.hz:
            raise ValueError("servo target timeout must cover at least two servo cycles")


def check_reference_tracking(feedback: np.ndarray, reference: np.ndarray, limit: float) -> None:
    """Execution safety check; never modifies the planner's q/v/a state."""
    feedback = np.asarray(feedback, dtype=np.float64)
    if feedback.shape != reference.shape or not np.isfinite(feedback).all():
        raise ServoError("invalid servo feedback")
    if np.max(np.abs(feedback - reference)) > limit:
        index = int(np.argmax(np.abs(feedback - reference)))
        raise ServoError("servo trajectory tracking error exceeded configured limit: "
                         f"arm_index={index // 6} joint={index % 6 + 1} "
                         f"feedback={feedback[index]:.6f} target={reference[index]:.6f} "
                         f"error={abs(feedback[index] - reference[index]):.6f}rad "
                         f"limit={limit:.6f}rad")


class RuckigJointPlanner:
    """Synchronized S curves retaining q/v/a across every target replacement.

    The caller initializes this trajectory from feedback and monitors tracking.
    Subsequent cycles use Ruckig's generated state; substituting measured velocity
    or resetting to zero acceleration each cycle destroys the motion bounds.
    Joint bounds are checked on every output, never enforced by clipping it.
    """

    def __init__(self, initial: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                 config: ServoConfig) -> None:
        try:
            import ruckig
        except ImportError as exc:
            raise ServoError("Ruckig servo requires ruckig==0.12.2 in the robot Conda environment") from exc
        self._ruckig = ruckig
        self.config = config
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.position = np.asarray(initial, dtype=np.float64).copy()
        if self.position.ndim != 1 or self.lower.shape != self.position.shape or self.upper.shape != self.position.shape:
            raise ValueError("servo initial/lower/upper must be matching joint vectors")
        if (not np.isfinite(self.position).all() or np.isnan(self.lower).any()
                or np.isnan(self.upper).any() or np.any(self.lower > self.upper)):
            raise ValueError("invalid servo state or bounds")
        self._check_bounds(self.position, stage="initial state")
        self.velocity = np.zeros_like(self.position)
        self.acceleration = np.zeros_like(self.position)
        self.generator = ruckig.Ruckig(len(self.position), 1.0 / config.hz)
        self.input = ruckig.InputParameter(len(self.position))
        self.output = ruckig.OutputParameter(len(self.position))
        self.input.current_position = self.position.tolist()
        self.input.current_velocity = self.velocity.tolist()
        self.input.current_acceleration = self.acceleration.tolist()
        self.input.max_velocity = [config.max_velocity] * len(self.position)
        self.input.max_acceleration = [config.max_acceleration] * len(self.position)
        self.input.max_jerk = [config.max_jerk] * len(self.position)
        self.input.synchronization = ruckig.Synchronization.Time
        self.input.duration_discretization = ruckig.DurationDiscretization.Discrete
        self.target = self.position.copy()
        self.retarget(self.target)
        self.braking = False

    def _check_bounds(self, position: np.ndarray, *, stage: str = "generated output") -> None:
        tolerance = JOINT_POSITION_NUMERICAL_TOLERANCE_RAD
        invalid = (~np.isfinite(position) | (position < self.lower - tolerance)
                   | (position > self.upper + tolerance))
        if invalid.any():
            index = int(np.flatnonzero(invalid)[0])
            raise ServoError(f"{stage} exceeds physical position bounds: "
                             f"arm_index={index // 6} joint={index % 6 + 1} "
                             f"q={position[index]:.9f}rad "
                             f"bounds=[{self.lower[index]:.9f}, {self.upper[index]:.9f}] "
                             f"numerical_tolerance={tolerance:.1e}rad")

    def _check_planned_bounds(self) -> None:
        trajectory = self._ruckig.Trajectory(len(self.position))
        try:
            result = self.generator.calculate(self.input, trajectory)
        except Exception as exc:
            raise ServoError(f"Ruckig trajectory planning failed: {exc}") from exc
        if int(result) < 0:
            raise ServoError(f"Ruckig trajectory planning returned {result}")
        minima = np.array([e.min for e in trajectory.position_extrema])
        maxima = np.array([e.max for e in trajectory.position_extrema])
        # Ruckig 0.12.2 position_extrema can omit a velocity zero crossing
        # inside a constant-acceleration segment (j=0). Check polynomial
        # segments explicitly, including any initial braking profile.
        for section in trajectory.profiles:
            for axis, profile in enumerate(section):
                for part in (profile.brake, profile):
                    for dt, p, v, a, j in zip(part.t, part.p, part.v, part.a, part.j):
                        if dt <= 0:
                            continue
                        candidates = [0., dt]
                        if j == 0:
                            if a != 0:
                                candidates.append(-v / a)
                        else:
                            discriminant = a*a - 2*j*v
                            if discriminant >= 0:
                                root = math.sqrt(discriminant)
                                candidates.extend(((-a-root)/j, (-a+root)/j))
                        for at in candidates:
                            if 0 <= at <= dt:
                                q = p + at*(v + at*(a/2 + at*j/6))
                                minima[axis] = min(minima[axis], q)
                                maxima[axis] = max(maxima[axis], q)
        self._check_bounds(minima, stage="planned minimum")
        self._check_bounds(maxima, stage="planned maximum")

    def retarget(self, position: np.ndarray, velocity: np.ndarray | None = None) -> None:
        target = np.asarray(position, dtype=np.float64)
        if target.shape != self.position.shape:
            raise ServoError("servo target dimensions changed")
        self._check_bounds(target, stage="target")
        terminal_velocity = np.zeros_like(target) if velocity is None else np.asarray(velocity, dtype=np.float64)
        if terminal_velocity.shape != target.shape or not np.isfinite(terminal_velocity).all():
            raise ServoError("invalid servo terminal velocity")
        # Avoid a nonzero terminal speed towards a nearby mechanical stop.
        distance = np.minimum(target - self.lower, self.upper - target)
        braking_speed = np.minimum(
            np.sqrt(np.maximum(0.0, 2 * self.config.max_acceleration * distance)),
            np.cbrt(np.maximum(0.0, self.config.max_jerk * distance**2)),
        )
        limit = np.minimum(self.config.max_velocity, braking_speed)
        terminal_velocity = np.clip(terminal_velocity, -limit, limit)
        self.input.control_interface = self._ruckig.ControlInterface.Position
        self.input.target_position = target.tolist()
        self.input.target_velocity = terminal_velocity.tolist()
        self.input.target_acceleration = [0.0] * len(target)
        # A legal endpoint and a bounded terminal speed do not guarantee a
        # legal synchronized trajectory: it may first move away from the goal.
        # Check analytic extrema before publishing any part of the new plan.
        try:
            self._check_planned_bounds()
        except ServoError:
            if not np.any(terminal_velocity):
                raise
            self.input.target_velocity = [0.0] * len(target)
            self._check_planned_bounds()
        self.target = target.copy()
        self.braking = False

    def brake(self) -> None:
        """Generate a bounded stop from the current velocity and acceleration."""
        self.input.control_interface = self._ruckig.ControlInterface.Velocity
        self.input.target_velocity = [0.0] * len(self.position)
        self.input.target_acceleration = [0.0] * len(self.position)
        self.braking = True

    def set_state(self, position, velocity, acceleration) -> None:
        """Adopt an already validated planned state, including its derivatives."""
        position, velocity, acceleration = (np.asarray(x, dtype=float).copy()
                                             for x in (position, velocity, acceleration))
        if any(x.shape != self.position.shape or not np.isfinite(x).all()
               for x in (position, velocity, acceleration)):
            raise ServoError("invalid planned q/v/a state")
        self._check_bounds(position)
        if (np.max(abs(velocity)) > self.config.max_velocity + 1e-7
                or np.max(abs(acceleration)) > self.config.max_acceleration + 1e-7):
            raise ServoError("planned derivative exceeds motion limits")
        self.position, self.velocity, self.acceleration = position, velocity, acceleration
        self.input.current_position = position.tolist()
        self.input.current_velocity = velocity.tolist()
        self.input.current_acceleration = acceleration.tolist()
        self.generator.reset()

    def advance(self) -> np.ndarray:
        """Generate one reference tick from retained q/v/a, without feedback control."""
        try:
            result = self.generator.update(self.input, self.output)
        except Exception as exc:
            raise ServoError(f"Ruckig trajectory generation failed: {exc}") from exc
        if int(result) < 0:
            raise ServoError(f"Ruckig trajectory generation returned {result}")
        position = np.asarray(self.output.new_position, dtype=np.float64)
        self._check_bounds(position)
        self.position = position.copy()
        self.velocity = np.asarray(self.output.new_velocity, dtype=np.float64).copy()
        self.acceleration = np.asarray(self.output.new_acceleration, dtype=np.float64).copy()
        self.output.pass_to_input(self.input)
        return self.position.copy()

    def metrics(self) -> dict[str, Any]:
        return {"position": self.position.tolist(), "velocity": self.velocity.tolist(),
                "acceleration": self.acceleration.tolist(), "braking": self.braking}


class RuckigJointServo(RuckigJointPlanner):
    """Compatibility adapter for existing guarded planning/return callers."""

    def update(self, feedback: np.ndarray) -> np.ndarray:
        check_reference_tracking(feedback, self.position, self.config.max_tracking_error)
        return self.advance()


class ServoLoop:
    """Guarded reference planner and single actuator writer; not a motor servo.

    ``read_feedback`` must check freshness and drive status before returning a
    7D/14D vector. ``publish`` receives the generated vector and returns actual
    per-arm wire/timing telemetry. A submit waits for its first publication so
    the action queue never counts a merely enqueued target as executed.
    """

    def __init__(self, initial: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                 config: ServoConfig, read_feedback: Callable[[], np.ndarray],
                 publish: Callable[[np.ndarray], dict[str, Any]],
                 on_sample: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.config = config
        initial = np.asarray(initial, dtype=np.float64)
        if initial.ndim != 1 or len(initial) not in (7, 14):
            raise ValueError("Piper servo requires a 7D/14D initial state")
        self.joint_indices = np.array([i for i in range(len(initial)) if i % 7 != 6])
        self.gripper_indices = np.array([i for i in range(len(initial)) if i % 7 == 6])
        self.planner = RuckigJointPlanner(initial[self.joint_indices],
                                     np.asarray(lower)[self.joint_indices],
                                     np.asarray(upper)[self.joint_indices], config)
        self.servo = self.planner  # Existing metrics and shutdown API compatibility.
        self.read_feedback = read_feedback
        self.publish = publish
        self.on_sample = on_sample
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._paused = True
        self._fault: str | None = None
        self._target = initial.copy()
        self._grippers = initial[self.gripper_indices].copy()
        self._last_reference: np.ndarray | None = None
        self._reference_at: float | None = None
        self._target_velocity = np.zeros(len(self.joint_indices))
        self._revision = 0
        self._applied_revision = 0
        self._published_revision = 0
        self._context: dict[str, Any] = {}
        self._hold = False
        self._stopping = False
        self._last_sample: dict[str, Any] | None = None
        self._sample_count = 0
        self._overruns = 0
        self._plan = None
        self._plan_revision = 0
        self._plan_epoch = 0
        self._committed_until = 0.0
        # Issued prefix reservations survive suffix-only plan revisions.
        # Bound history by revisions; eviction rejects old requests safely.
        self._prefix_reservations: dict[str, float] = {}
        self._plan_last_index = 0
        self._timing_invalidations = 0
        self._chunk_prepare_lead_s = .05
        self._plan_timing: dict[str, Any] = {}
        self._thread = threading.Thread(target=self._run, name="piper-ruckig-servo", daemon=True)
        self._thread.start()

    def install_chunk(self, targets, target_times, source_indices, *, generation: int) -> None:
        """Prepare outside the writer lock; preserve the already committed prefix."""
        from .chunk_planner import ChunkPlan, compile_chunk
        with self._condition:
            if self._closed or self._fault or self._stopping:
                raise ServoError(self._fault or "planner is unavailable")
            if self._paused and self._sample_count and not self._stationary():
                raise ServoError("moving servo was safety-paused; restart after the robot is stationary")
            if self._plan is not None and self._plan.generation == generation:
                return
            old = self._plan
            epoch, revision = self._plan_epoch, self._plan_revision
            now = time.monotonic()
            # Preparation is bounded and never blocks the actuator writer.
            boundary = max(now + self._chunk_prepare_lead_s, self._committed_until)
            terminal_gap = False
            if old is not None:
                index = int(np.searchsorted(old.times, boundary, side="left"))
                if index >= len(old.times):
                    # The compiled plan includes a terminal stop. Its final
                    # pose remains valid while waiting for the next chunk.
                    index = len(old.times) - 1
                    if (np.max(np.abs(old.velocities[index])) > 1e-6
                            or np.max(np.abs(old.accelerations[index])) > 1e-6):
                        raise ServoError("old plan has no stationary terminal state")
                    terminal_gap = True
                else:
                    boundary = float(old.times[index])
                initial = old.positions[index].copy()
                velocity, acceleration = old.velocities[index], old.accelerations[index]
            else:
                if not self._stationary():
                    raise ChunkPlanDeferred("chunk initialization is waiting for healthy braking to finish")
                initial = self._target.copy()
                initial[self.joint_indices] = self.planner.position
                initial[self.gripper_indices] = self._grippers
                velocity, acceleration = self.planner.velocity.copy(), self.planner.acceleration.copy()
            lower = np.full(len(initial), -np.inf); upper = np.full(len(initial), np.inf)
            lower[self.joint_indices], upper[self.joint_indices] = self.planner.lower, self.planner.upper
            lower[self.gripper_indices] = 0.0
            # Gripper units here are metres, as in read_output_qpos/JointCtrl.
            from bimanual_vla.data.contract import GRIPPER_MAX_M
            upper[self.gripper_indices] = GRIPPER_MAX_M
            if len(target_times) and target_times[-1] <= boundary:
                raise ChunkPlanDeferred("timed chunk expired before its splice boundary; waiting for fresh inference")
        prepare_started = time.monotonic()
        try:
            new = compile_chunk(initial, velocity, acceleration, lower, upper, self.config,
                                targets, target_times, source_indices, start=boundary, generation=generation)
            computed_at = time.monotonic()
        except ServoError as exc:
            # This computation has not installed or published any candidate
            # state. A bad candidate is different from a fault in the active
            # writer. Never downgrade a concurrent hardware/tracking fault.
            with self._condition:
                if (self._fault or self._closed or self._stopping
                        or (self._paused and self._sample_count and not self._stationary())):
                    raise ServoError(self._fault or "planner unavailable during chunk preparation") from exc
            raise ChunkPlanRejected(f"candidate chunk {generation} rejected: {exc}") from exc
        with self._condition:
            # Include lock reacquisition. A past outlier must not permanently
            # reserve hundreds of milliseconds of obsolete trajectory.
            prepared_at = time.monotonic()
            preparation_s = prepared_at - prepare_started
            self._chunk_prepare_lead_s = min(.5, max(
                2.0 / self.config.hz,
                1.25 * preparation_s + 1.0 / self.config.hz,
                .5 * self._chunk_prepare_lead_s))
            if self._fault or self._closed or self._stopping:
                raise ServoError(self._fault or "planner became unavailable during chunk preparation")
            if self._paused and self._sample_count and not self._stationary():
                raise ServoError("moving servo was safety-paused; restart after the robot is stationary")
            if epoch != self._plan_epoch or revision != self._plan_revision:
                raise ChunkPlanDeferred("planner changed while preparing a chunk; retry with current state")
            if time.monotonic() >= boundary:
                raise ChunkPlanDeferred("chunk planning missed its splice deadline; old plan retained")
            if self._committed_until > boundary + 1e-9:
                raise ChunkPlanDeferred("new RTC prefix was reserved during chunk planning; old plan retained")
            if old is not None:
                # Samples through the boundary are unchanged, including v/a.
                keep_from = max(0, self._plan_last_index - 2)
                new_start = 0 if terminal_gap else 1
                self._plan = ChunkPlan(
                    np.concatenate([old.times[keep_from:index+1], new.times[new_start:]]),
                    np.concatenate([old.positions[keep_from:index+1], new.positions[new_start:]]),
                    np.concatenate([old.velocities[keep_from:index+1], new.velocities[new_start:]]),
                    np.concatenate([old.accelerations[keep_from:index+1], new.accelerations[new_start:]]),
                    generation, new.source_times, new.source_indices,
                    np.concatenate([old.sample_generations[keep_from:index+1], new.sample_generations[new_start:]]),
                    np.concatenate([old.sample_source_indices[keep_from:index+1], new.sample_source_indices[new_start:]]))
                self._plan_last_index -= keep_from
            else:
                self._plan = new
                self._plan_last_index = 0
            self._plan_revision += 1
            # Source times remain anchored to the original observation. The
            # splice (including preparation and the reserved RTC prefix), not
            # result arrival, determines the new trajectory's source index.
            source_index = int(np.clip(np.searchsorted(target_times, boundary, side="left"),
                                       0, len(source_indices)-1))
            self._plan_timing = {
                "generation": generation,
                "prepare_started_monotonic": prepare_started,
                "computed_monotonic": computed_at,
                "prepared_monotonic": prepared_at,
                "installed_monotonic": time.monotonic(),
                "splice_monotonic": boundary,
                "committed_until_monotonic": self._committed_until,
                "preparation_ms": preparation_s * 1000,
                "compile_ms": (computed_at - prepare_started) * 1000,
                "commit_lock_wait_ms": (prepared_at - computed_at) * 1000,
                "source_index_at_splice": int(source_indices[source_index]),
                "source_target_monotonic_at_splice": float(target_times[source_index]),
                "first_published_monotonic": None,
                "first_published_source_index": None,
            }
            self._paused = False
            self._hold = False
            self._condition.notify_all()

    def rtc_prefix(self, capture: float, steps: int, action_hz: float) -> dict[str, Any] | None:
        """Reserve samples from the same immutable plan consumed by the writer."""
        with self._condition:
            if (self._plan is None or self._paused or self._hold or self._fault
                    or self._stopping or steps <= 0):
                return None
            times = capture + np.arange(1, steps + 1) / action_hz
            if times[0] < self._plan.times[0] or times[-1] > self._plan.times[-1]:
                return None
            indices = np.searchsorted(self._plan.times, times, side="right") - 1
            self._committed_until = max(self._committed_until, float(times[-1]))
            plan_id = f"{self._plan_epoch}:{self._plan_revision}"
            self._prefix_reservations[plan_id] = max(
                self._prefix_reservations.get(plan_id, 0.0), float(times[-1]))
            if len(self._prefix_reservations) > 128:
                del self._prefix_reservations[next(iter(self._prefix_reservations))]
            return {"plan_id": plan_id,
                    "sample_offsets_s": ((np.arange(steps)+1)/action_hz).tolist(),
                    "positions": self._plan.positions[indices].copy(),
                    "action_hz": float(action_hz)}

    def prefix_is_valid(self, plan_id: str) -> bool:
        """Validate a reservation, not equality with the latest suffix revision.

        install_chunk copies old q/v/a through _committed_until, and rejects
        commits that race a newer reservation. Thus issued prefixes remain
        valid across successful installs, even after published samples have
        been pruned. Holds, faults and clock gaps revoke all reservations.
        Merely sharing an epoch is insufficient: the token must be issued.
        """
        with self._condition:
            reserved_until = self._prefix_reservations.get(plan_id)
            return bool(self._plan is not None and not self._paused and not self._hold
                        and not self._fault and not self._stopping and not self._closed
                        and reserved_until is not None
                        and reserved_until <= self._committed_until)

    def has_chunk(self, generation: int) -> bool:
        with self._condition:
            return bool(self._plan is not None and self._plan.generation == generation
                        and not self._fault and not self._paused and not self._hold)

    def _invalidate_plan(self) -> None:
        self._plan = None
        self._plan_timing = {}
        self._invalidate_prefixes()

    def _invalidate_prefixes(self) -> None:
        """Revoke promises under the writer lock, optionally retaining the plan."""
        self._plan_epoch += 1
        self._committed_until = 0.0
        self._prefix_reservations.clear()

    def submit(self, target: np.ndarray, *, hold: bool = False,
               context: dict[str, Any] | None = None) -> dict[str, Any]:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != self._target.shape or not np.isfinite(target).all():
            raise ServoError("invalid submitted servo reference")
        with self._condition:
            if self._closed or self._fault or self._stopping:
                raise ServoError(self._fault or "servo loop is closed")
            if self._paused and self._sample_count and not self._stationary():
                raise ServoError("moving servo was safety-paused; restart after the robot is stationary")
            now = time.monotonic()
            if hold:
                self._invalidate_plan()
            velocity = np.zeros(len(self.joint_indices))
            if (self.config.follow_waypoint_velocity and not hold
                    and self._last_reference is not None and self._reference_at is not None
                    and not self._paused and not self._hold):
                elapsed = now - self._reference_at
                if 0 < elapsed <= self.config.target_timeout_s:
                    velocity = (target[self.joint_indices] - self._last_reference) / elapsed
                    velocity = np.clip(velocity, -self.config.max_velocity, self.config.max_velocity)
            self._target_velocity = velocity
            self._target = target.copy()
            self._last_reference = target[self.joint_indices].copy()
            self._reference_at = now
            self._hold = bool(hold)
            self._context = dict(context or {})
            self._paused = False
            self._revision += 1
            revision = self._revision
            self._condition.notify_all()
            deadline = now + max(0.1, 4.0 / self.config.hz)
            while self._published_revision < revision and not self._fault and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._paused = True
                    raise ServoError("servo did not publish the submitted target within its deadline")
                self._condition.wait(timeout=remaining)
            if self._fault or self._closed:
                raise ServoError(self._fault or "servo closed before target publication")
            assert self._last_sample is not None
            return dict(self._last_sample)

    def pause(self) -> None:
        # The same lock covers publication: after returning, no in-flight writer
        # can race an enable handshake, return-to-home, or disconnection.
        with self._condition:
            self._paused = True
            self._invalidate_plan()
            self._last_reference = None

    def hold(self) -> None:
        """Reject further travel while keeping the healthy braking writer active."""
        with self._condition:
            self._hold = True
            self._invalidate_plan()
            self._last_reference = None
            self._condition.notify_all()

    def close(self, *, stop_timeout_s: float = 3.0) -> bool:
        """Brake before ending a healthy writer; report whether it settled.

        A safety pause/fault prohibits further publication. In that case a
        moving generated state is not declared stationary, and the caller must
        not start a new trajectory initialized with zero velocity.
        """
        with self._condition:
            self._stopping = True
            self._invalidate_plan()
            deadline = time.monotonic() + stop_timeout_s
            if not self._closed and not self._paused and not self._fault:
                self._hold = True
                self._condition.notify_all()
                while not self._stationary() and not self._fault and not self._paused:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
            settled = self._stationary() and not self._fault
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            raise ServoError("servo writer did not terminate; actuator ownership cannot be transferred")
        return settled

    def _stationary(self) -> bool:
        return bool(np.max(np.abs(self.servo.velocity)) < 1e-6
                    and np.max(np.abs(self.servo.acceleration)) < 1e-6)

    def metrics(self) -> dict[str, Any]:
        with self._condition:
            return {"backend": "ruckig", "role": "joint_reference_planner",
                    "configured_hz": self.config.hz,
                    "sample_count": self._sample_count, "overruns": self._overruns,
                    "paused": self._paused, "fault": self._fault,
                    "stopping": self._stopping,
                    "plan_id": f"{self._plan_epoch}:{self._plan_revision}",
                    "chunk_planner": self._plan is not None,
                    "committed_until": self._committed_until,
                    "timing_invalidations": self._timing_invalidations,
                    "preparation_budget_ms": self._chunk_prepare_lead_s * 1000,
                    "plan_timing": dict(self._plan_timing),
                    "state": self.servo.metrics()}

    def _run(self) -> None:
        period = 1.0 / self.config.hz
        next_at = time.monotonic()
        while True:
            planned_next_at = None
            with self._condition:
                if self._closed:
                    return
                if self._paused:
                    self._condition.wait(timeout=period)
                    next_at = time.monotonic()
                    continue
                try:
                    feedback = np.asarray(self.read_feedback(), dtype=np.float64).copy()
                    now = time.monotonic()
                    if feedback.shape != self._target.shape or not np.isfinite(feedback).all():
                        raise ServoError("invalid servo feedback")
                    check_reference_tracking(feedback[self.joint_indices], self.planner.position,
                                             self.config.max_tracking_error)
                    expired = (self._plan is None and
                               (self._reference_at is None or now - self._reference_at > self.config.target_timeout_s))
                    braking = self._hold or expired
                    command = feedback.copy()
                    if self._plan is not None and not braking:
                        plan = self._plan
                        # Sample the time-indexed trajectory at the current
                        # clock. Sending only one queued 10 ms row per slow
                        # CAN cycle accumulates permanent lag. Never burst
                        # historical rows to catch up.
                        index = max(self._plan_last_index, plan.index_at(now))
                        skipped = index - self._plan_last_index - 1
                        if skipped > 1:
                            # A scheduling gap invalidates pending prefix
                            # promises, not the whole healthy actuator writer.
                            self._invalidate_prefixes()
                            self._timing_invalidations += 1
                            if self._timing_invalidations == 1 or self._timing_invalidations % 50 == 0:
                                logging.warning("Planner publication skipped %d elapsed ticks; "
                                                "pending RTC prefix invalidated, continuing timed plan", skipped)
                        self._plan_last_index = index
                        command = plan.positions[index].copy()
                        check_reference_tracking(feedback[self.joint_indices], command[self.joint_indices],
                                                 self.config.max_tracking_error)
                        self.planner.set_state(command[self.joint_indices], plan.velocities[index],
                                               plan.accelerations[index])
                        self.planner.braking = False
                        self._grippers = command[self.gripper_indices].copy()
                        self._context = plan.context_at(now)
                    elif braking:
                        if not self.servo.braking:
                            self.servo.brake()
                    elif self._applied_revision != self._revision:
                        self.servo.retarget(self._target[self.joint_indices], self._target_velocity)
                    self._applied_revision = self._revision
                    if self._plan is None or braking:
                        command[self.joint_indices] = self.planner.advance()
                    # A hold/expired reference retains the last published gripper.
                    if not braking and self._plan is None:
                        self._grippers += np.clip(self._target[self.gripper_indices] - self._grippers,
                                                  -self.config.max_gripper_velocity * period,
                                                  self.config.max_gripper_velocity * period)
                    command[self.gripper_indices] = self._grippers
                    trace = self.publish(command)
                    published_at = time.monotonic()
                    if (self._plan_timing and not braking
                            and now >= self._plan_timing["splice_monotonic"]
                            and self._context.get("generation") == self._plan_timing["generation"]
                            and self._plan_timing["first_published_monotonic"] is None):
                        self._plan_timing["first_published_monotonic"] = published_at
                        self._plan_timing["first_published_source_index"] = self._context["source_index"]
                    self._sample_count += 1
                    sample = {"sequence": self._sample_count, "timestamp": time.time(),
                              "monotonic_timestamp": time.monotonic(), "qpos": command.tolist(),
                              "command_qpos": command.tolist(), "feedback_qpos": feedback.tolist(),
                              "feedback_read_monotonic": now,
                              "plan_id": (f"{self._plan_epoch}:{self._plan_revision}"
                                          if self._plan is not None else None),
                              "planned_monotonic": (float(self._plan.times[self._plan_last_index])
                                                    if self._plan is not None else None),
                              "plan_timing": dict(self._plan_timing),
                              "role": "joint_reference_planner",
                              "reference": self._target.tolist(), "expired": expired,
                              "context": dict(self._context), "sides": trace,
                              **self.servo.metrics()}
                    self._last_sample = sample
                    self._published_revision = self._revision
                    self._condition.notify_all()
                    if self.on_sample is not None:
                        self.on_sample(sample)
                    if self._plan is not None:
                        if now < self._plan.times[0]:
                            planned_next_at = min(now + period, float(self._plan.times[0]))
                        elif self._plan_last_index + 1 < len(self._plan.times):
                            planned_next_at = float(self._plan.times[self._plan_last_index + 1])
                except Exception as exc:
                    self._fault = str(exc)
                    self._paused = True
                    self._invalidate_plan()
                    self._condition.notify_all()
            next_at = next_at + period if planned_next_at is None else planned_next_at
            now = time.monotonic()
            if now > next_at:
                self._overruns += 1
                # Never replay missed samples in a burst.
                next_at = now + period
            with self._condition:
                if self._closed:
                    return
                # New references may wake the condition early. Retarget only
                # on the next actuator tick, never publish an extra short tick.
                while not self._closed and not self._paused and time.monotonic() < next_at:
                    self._condition.wait(timeout=next_at - time.monotonic())
