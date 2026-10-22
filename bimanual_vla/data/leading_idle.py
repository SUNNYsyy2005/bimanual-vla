"""Conservative leading-hold detection for absolute Piper joint datasets."""
import numpy as np


def leading_hold_cut(state, action, *, joint_threshold=.001,
                     gripper_threshold=.002, keep_before_motion=2):
    """Keep two frames before either arm/gripper first leaves its initial value.

    Inspect both measured and commanded motion. Never remove internal stops or
    an entirely stationary episode. Thresholds apply to cumulative displacement,
    so a slow ramp cannot be mistaken for an indefinitely stationary prefix.
    """
    q, a = np.asarray(state, dtype=float), np.asarray(action, dtype=float)
    if (q.ndim != 2 or q.shape[1] not in (7, 14) or a.shape != q.shape
            or not len(q) or not np.isfinite(q).all() or not np.isfinite(a).all()):
        raise ValueError("expected finite matching absolute joint state/action arrays of width 7 or 14")
    if (not np.isfinite(joint_threshold) or joint_threshold <= 0
            or not np.isfinite(gripper_threshold) or gripper_threshold <= 0
            or int(keep_before_motion) != keep_before_motion or keep_before_motion < 1):
        raise ValueError("positive thresholds and at least one context frame are required")
    limits = np.full(q.shape[1], joint_threshold)
    limits[6::7] = gripper_threshold
    changed = np.any(abs(q - q[0]) > limits, axis=1) | np.any(abs(a - a[0]) > limits, axis=1)
    # A different initial action already asks for motion: keep this transition.
    if np.any(abs(a[0] - q[0]) > limits):
        changed[0] = True
    starts = np.flatnonzero(changed)
    onset = int(starts[0]) if len(starts) else None
    return dict(start_frame=max(0, onset - int(keep_before_motion)) if onset is not None else 0,
                motion_onset_frame=onset, all_idle=onset is None)


def boundary_hold_range(state, action, **kwargs):
    """Inclusive range retaining context around motion, including internal stops.

    Use cumulative displacement from each endpoint, rather than a velocity
    threshold that can discard slow movement. Entirely idle episodes are kept.
    """
    leading = leading_hold_cut(state, action, **kwargs)
    trailing = leading_hold_cut(np.asarray(state)[::-1], np.asarray(action)[::-1], **kwargs)
    count = len(state)
    start = leading["start_frame"]
    end = max(start, count - 1 - trailing["start_frame"])
    return dict(start_frame=start, end_frame=end, removed_leading=start,
                removed_trailing=count-1-end, all_idle=leading["all_idle"])
