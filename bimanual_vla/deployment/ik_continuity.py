"""Bounded local joint-motion minimization inside a Cartesian tolerance tube."""
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class MinimumMotionResult:
    joints: np.ndarray | None
    evaluations: int
    elapsed_s: float
    timed_out: bool


def bounded_regularized_solve(task, seed, anchor, lower, upper, *, penalty,
                              max_iterations=12, converged=.05):
    """Bounded Levenberg-Marquardt toward the target centre.

    Minimizes ``|e(q)|^2 + penalty*|q-anchor|^2`` subject to
    ``lower <= q <= upper``, where ``task`` returns the tolerance-scaled 6D
    pose error and its Jacobian. Driver limits and the per-tick step boxes
    constrain every linearized step instead of clipping its result, so a joint
    pinned at a limit lets the others compensate. The motion penalty is never
    dropped: near the wrist singularity it stops J4/J6 from swinging far to
    chase a residual that barely moves the end effector.

    Returns the best iterate and the number of task evaluations.
    """
    from scipy.optimize import lsq_linear

    lower, upper, anchor = (np.asarray(x, dtype=float) for x in (lower, upper, anchor))
    q = np.clip(np.asarray(seed, dtype=float), lower, upper)
    penalty = max(float(penalty), 0.)
    identity = np.eye(len(q))

    def cost(error, joints):
        return float(error @ error + penalty * np.sum((joints - anchor) ** 2))

    error, jacobian = task(q)
    best, evaluations, damping = cost(error, q), 1, 1e-3
    for _ in range(max_iterations):
        if np.max(np.abs(error)) <= converged:
            break
        gradient = jacobian.T @ error + penalty * (q - anchor)
        normal = jacobian.T @ jacobian + (penalty + damping) * identity
        try:
            step = np.linalg.solve(normal, -gradient)
        except np.linalg.LinAlgError:
            step = np.full(len(q), np.inf)
        if not (np.all(np.isfinite(step)) and np.all(q + step >= lower) and np.all(q + step <= upper)):
            # Active bound: solve the same damped step as bounded least squares.
            system = np.vstack((jacobian, np.sqrt(penalty) * identity, np.sqrt(damping) * identity))
            rhs = -np.concatenate((error, np.sqrt(penalty) * (q - anchor), np.zeros(len(q))))
            step = lsq_linear(system, rhs, bounds=(lower - q, upper - q), method="bvls").x
        candidate = np.clip(q + step, lower, upper)
        new_error, new_jacobian = task(candidate)
        evaluations += 1
        value = cost(new_error, candidate)
        if np.isfinite(value) and value < best:
            improvement = best - value
            q, error, jacobian, best = candidate, new_error, new_jacobian, value
            damping = max(damping / 3., 1e-6)
            if improvement <= 1e-9 * (1. + best):
                break
        else:
            damping *= 10.
            if damping > 1e6:
                break
    return q, evaluations


def minimum_motion_candidate(task, anchor, lower, upper, position_tolerance,
                             rotation_tolerance, *, max_iterations=24, budget_s=.008):
    """Prefer the nearest feasible local pose, not a lower residual at any cost.

    ``task`` returns the 6D pose residual and its analytic joint Jacobian.
    Every accepted iterate must satisfy BOTH pose norms and all joint bounds.
    An iteration/time cap returns the best feasible iterate, or no candidate;
    it never relaxes a tolerance or expands the caller's search neighborhood.
    """
    from scipy.optimize import minimize

    started = time.perf_counter()
    anchor, lower, upper = (np.asarray(x, dtype=float) for x in (anchor, lower, upper))
    scales = np.array([position_tolerance]*3 + [rotation_tolerance]*3)
    if (anchor.shape != lower.shape or anchor.shape != upper.shape or anchor.ndim != 1
            or not all(np.isfinite(x).all() for x in (anchor, lower, upper))
            or np.any(lower >= upper) or not np.isfinite(scales).all() or np.any(scales <= 0)
            or not np.isfinite(budget_s) or budget_s <= 0 or max_iterations <= 0):
        raise ValueError("invalid minimum-motion IK bounds, tolerances or budget")
    seed = np.clip(anchor, lower, upper)
    best, best_cost, evaluations = None, float('inf'), 0
    cached_q = cached_error = cached_jacobian = None

    class BudgetExceeded(Exception):
        pass

    def evaluate(q):
        nonlocal cached_q, cached_error, cached_jacobian, best, best_cost, evaluations
        if time.perf_counter()-started >= budget_s:
            raise BudgetExceeded
        if cached_q is None or not np.array_equal(q, cached_q):
            error, jacobian = task(q)
            cached_q = np.asarray(q).copy()
            cached_error, cached_jacobian = error/scales, jacobian/scales[:, None]
            evaluations += 1
            if (np.isfinite(cached_error).all() and np.all(q >= lower) and np.all(q <= upper)
                    and np.linalg.norm(cached_error[:3]) <= 1.
                    and np.linalg.norm(cached_error[3:]) <= 1.):
                cost = float(np.sum((q-anchor)**2))
                if cost < best_cost:
                    best, best_cost = q.copy(), cost
        return cached_error, cached_jacobian

    def constraints(q):
        e, _ = evaluate(q)
        # A small interior margin avoids returning numerically outside the
        # advertised pose tube. Acceptance above still checks the actual norms.
        return np.array([.9999-e[:3]@e[:3], .9999-e[3:]@e[3:]])

    def constraint_jacobian(q):
        e, j = evaluate(q)
        return np.stack((-2*e[:3]@j[:3], -2*e[3:]@j[3:]))

    timed_out = False
    try:
        evaluate(seed)
        if best is None or best_cost > 1e-20:
            result = minimize(lambda q: .5*np.sum(((q-anchor)/.08)**2), seed,
                jac=lambda q: (q-anchor)/.08**2, bounds=list(zip(lower, upper)),
                constraints=[dict(type='ineq', fun=constraints, jac=constraint_jacobian)],
                method='SLSQP', options=dict(maxiter=max_iterations, ftol=1e-8))
            evaluate(result.x)
    except BudgetExceeded:
        timed_out = True
    return MinimumMotionResult(best, evaluations, time.perf_counter()-started, timed_out)
