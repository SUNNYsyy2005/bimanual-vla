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
