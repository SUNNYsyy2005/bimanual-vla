"""Persistent, hardware-free per-arm IK processes with bounded pending work."""
import dataclasses
import multiprocessing as mp
import time


class IKWorkerError(RuntimeError):
    pass


class IKUnreachableError(IKWorkerError):
    """A valid local geometric problem has no acceptable solution."""


class IKRecoveryExhausted(IKWorkerError):
    """Neither bounded Cartesian step made acceptable progress."""


def solve_ik_request(solver, args, kwargs):
    """Solve a hardware-free request, optionally shrinking it about measured FK."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from bimanual_vla.deployment.hardware_safety import NOMINAL_JOINT_LIMITS_RAD

    kwargs = dict(kwargs)
    fraction = kwargs.pop("recovery_fraction", 1.0)
    progress_goal = kwargs.pop("progress_goal", None)
    hold_position = kwargs.pop("hold_position", False)
    if fraction not in (1.0, .5, .25):
        raise ValueError("unsupported IK recovery fraction")
    current, goal_xyz, goal_rpy = (np.asarray(v, dtype=np.float64) for v in args)
    if hold_position:
        from bimanual_vla.deployment.client import PiperIKSolveResult
        if current.shape != (6,) or not np.isfinite(current).all():
            raise ValueError("invalid stationary-arm joints")
        limits = np.asarray(kwargs.get("joint_limits", NOMINAL_JOINT_LIMITS_RAD))
        if np.any(current < limits[:, 0]) or np.any(current > limits[:, 1]):
            raise IKUnreachableError("stationary-arm feedback exceeds driver limits")
        xyz, rot = solver.pose(current)
        diagnostic = dict(fraction=fraction, progress=False, stationary=True,
                          xyz_m=xyz.tolist(), rpy_deg=Rotation.from_matrix(rot).as_euler('xyz', degrees=True).tolist())
        payload = dataclasses.asdict(PiperIKSolveResult(
            solution_joints_rad=current.copy(), command_joints_rad=current.copy(), rate_limited=False))
        payload["goal_progress"] = diagnostic
        if fraction < 1:
            payload["recovery"] = diagnostic
        return payload
    xyz, rpy = goal_xyz, goal_rpy
    if fraction < 1 or progress_goal is not None:
        # Recovery must advance toward a retained goal. A nearest-pose tube
        # could keep the measured pose when a shrunken intermediate target is
        # already inside tolerance, preventing otherwise valid progress.
        kwargs["minimum_motion"] = False
    if fraction < 1:
        origin_xyz, origin_rot = solver.pose(current)
        goal_rot = Rotation.from_euler("xyz", goal_rpy, degrees=True).as_matrix()
        xyz = origin_xyz + fraction * (goal_xyz - origin_xyz)
        rot = Rotation.from_rotvec(fraction * Rotation.from_matrix(
            goal_rot @ origin_rot.T).as_rotvec()).as_matrix() @ origin_rot
        rpy = Rotation.from_matrix(rot).as_euler("xyz", degrees=True)
        # Keep the last accepted branch through a failed full-step attempt.
        # The controller clears it only after a measured, settled handoff.
        if kwargs.get("initial_joints_rad") is None:
            kwargs["initial_joints_rad"] = current.copy()
    result = solver.solve_with_diagnostics(current, xyz, rpy, **kwargs)
    limits = np.asarray(kwargs.get("joint_limits", NOMINAL_JOINT_LIMITS_RAD))
    solution = np.asarray(result.solution_joints_rad)
    # The solver's feedback tolerance must never extend actuator hard limits.
    if (not np.isfinite(solution).all() or np.any(solution < limits[:, 0])
            or np.any(solution > limits[:, 1])):
        raise IKUnreachableError(
            f"IK solution exceeds driver joint position limits: q={solution.tolist()} "
            f"bounds={limits.tolist()}")
    payload = dataclasses.asdict(result)
    if fraction < 1 or progress_goal is not None:
        origin_xyz, origin_rot = solver.pose(current)
        if progress_goal is not None:
            goal_xyz, goal_rpy = (np.asarray(v, dtype=np.float64) for v in progress_goal)
        goal_rot = Rotation.from_euler("xyz", goal_rpy, degrees=True).as_matrix()
        actual_xyz, actual_rot = solver.pose(solution)
        before_pos = float(np.linalg.norm(goal_xyz - origin_xyz))
        before_rot = float(Rotation.from_matrix(goal_rot @ origin_rot.T).magnitude())
        after_pos = float(np.linalg.norm(goal_xyz - actual_xyz))
        after_rot = float(Rotation.from_matrix(goal_rot @ actual_rot.T).magnitude())
        before = before_pos / kwargs["position_tolerance_m"] + before_rot / kwargs["rotation_tolerance_rad"]
        after = after_pos / kwargs["position_tolerance_m"] + after_rot / kwargs["rotation_tolerance_rad"]
        stationary = (before_pos <= kwargs["position_tolerance_m"]
                      and before_rot <= kwargs["rotation_tolerance_rad"]
                      and np.max(np.abs(solution-current)) < 1e-6)
        progress = (before-after >= max(.01 * before, 1e-3)
                    and after_pos <= max(before_pos, kwargs["position_tolerance_m"]) + 1e-6
                    and after_rot <= max(before_rot, kwargs["rotation_tolerance_rad"]) + 1e-5)
        if not stationary and not progress:
            raise IKUnreachableError(
                "IK step makes no safe progress toward original target: "
                f"position={before_pos:.6f}->{after_pos:.6f}m "
                f"rotation={before_rot:.6f}->{after_rot:.6f}rad score={before:.4f}->{after:.4f}")
        diagnostic = dict(fraction=fraction, xyz_m=xyz.tolist(), rpy_deg=rpy.tolist(),
            goal_score_before=before, goal_score_after=after, progress=bool(progress),
            goal_position_before_m=before_pos, goal_position_after_m=after_pos,
            goal_rotation_before_rad=before_rot, goal_rotation_after_rad=after_rot)
        if fraction < 1:
            payload["recovery"] = diagnostic
        if progress_goal is not None:
            payload["goal_progress"] = diagnostic
    return payload


def solve_recovering(solve_batch, requests, *, timeout_s=.1):
    """Retry only geometry failures; one common fraction and total deadline."""
    deadline = time.monotonic() + timeout_s
    last_error = None
    for fraction in (1., .5, .25):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise IKWorkerError("IK recovery deadline exceeded")
        batch = {side: (args, dict(kwargs, recovery_fraction=fraction))
                 for side, (args, kwargs) in requests.items()}
        try:
            result = solve_batch(batch, timeout_s=remaining)
            if time.monotonic() > deadline:
                raise IKWorkerError("IK recovery deadline exceeded; result discarded")
            if fraction < 1 and not any(v["recovery"]["progress"] for v in result.values()):
                raise IKUnreachableError("neither arm makes progress")
            return result
        except IKUnreachableError as exc:
            last_error = exc
    raise IKRecoveryExhausted(f"IK recovery exhausted (1/2, 1/4): {last_error}")


def _worker(connection):
    # Imported after spawn: never inherit camera threads or CAN descriptors.
    from bimanual_vla.deployment.client import PiperContinuousIK
    import scipy.optimize  # Warm the fallback before accepting live targets.

    solver = PiperContinuousIK()
    connection.send(("ready", None))
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            batch, args, kwargs = request
            try:
                response = (batch, solve_ik_request(solver, args, kwargs), None)
            except Exception as exc:
                response = (batch, None, {"geometric": isinstance(exc, IKUnreachableError),
                                          "message": f"{type(exc).__name__}: {exc}"})
            connection.send(response)
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class PiperIKWorkers:
    """Only one in-flight batch; late replies are discarded before reuse.

    Results are returned atomically for both arms. Workers cannot publish robot
    commands, and no next batch is enqueued behind unfinished computation.
    """
    def __init__(self, sides, *, startup_timeout_s=15., worker_target=_worker):
        self.connections = {}
        self.processes = {}
        self.pending = set()
        self.batch = 0
        self.closed = False
        self.last_wall_ms = None
        context = mp.get_context("spawn")
        try:
            for side in sides:
                parent, child = context.Pipe()
                process = context.Process(target=worker_target, args=(child,),
                                          name=f"piper-ik-{side}", daemon=True)
                self.connections[side] = parent
                self.processes[side] = process
                process.start()
                child.close()
            deadline = time.monotonic() + startup_timeout_s
            for side, connection in self.connections.items():
                if not connection.poll(max(0., deadline-time.monotonic())):
                    raise IKWorkerError(f"{side} IK worker startup timed out")
                if connection.recv() != ("ready", None):
                    raise IKWorkerError(f"{side} IK worker failed startup")
        except BaseException:
            self.close()
            raise

    def solve(self, requests, *, timeout_s=.1):
        if self.closed:
            raise IKWorkerError("IK workers are closed")
        if set(requests) != set(self.connections):
            raise IKWorkerError("IK batch must contain exactly the configured arms")
        for side, process in self.processes.items():
            if not process.is_alive():
                raise IKWorkerError(f"{side} IK worker exited")
        # Drain an expired batch; never interpret it as a new target's answer.
        for side in tuple(self.pending):
            if self.connections[side].poll():
                self.connections[side].recv()
                self.pending.remove(side)
        if self.pending:
            raise IKWorkerError("previous IK batch still running; no target queued")
        self.batch += 1
        started = time.monotonic()
        deadline = started + timeout_s
        try:
            for side, (args, kwargs) in requests.items():
                self.connections[side].send((self.batch, args, kwargs))
                self.pending.add(side)
            results, errors, geometric = {}, [], []
            for side in requests:
                connection = self.connections[side]
                if not connection.poll(max(0., deadline-time.monotonic())):
                    raise IKWorkerError("IK batch deadline exceeded; late targets discarded")
                batch, result, error = connection.recv()
                self.pending.remove(side)
                if batch != self.batch:
                    raise IKWorkerError("IK worker returned a stale batch")
                if error:
                    geometric.append(isinstance(error, dict) and error.get("geometric") is True)
                    errors.append(f"{side}: {error['message'] if isinstance(error, dict) else error}")
                else:
                    results[side] = result
            if errors:
                error_type = IKUnreachableError if all(geometric) else IKWorkerError
                raise error_type("; ".join(errors))
            return results
        finally:
            self.last_wall_ms = (time.monotonic()-started)*1000

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections.values():
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes.values():
            if process.pid is not None:
                process.join(timeout=.2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.)
        for connection in self.connections.values():
            connection.close()
