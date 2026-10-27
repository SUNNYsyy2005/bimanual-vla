"""Unit-aware motion warnings; these statistical flags are not safety limits."""
from __future__ import annotations

import re
import numpy as np


def diagnose_motion(state, timestamps, *, names, arm_side="right", fps=None):
    from .episode_analysis import detect_motion_anomalies, idle_runs
    from .action_conventions import rotation6d_to_matrix

    n, width = state.shape
    result = dict(accelerations=np.zeros_like(state), jerk_norm=np.zeros(n),
                  velocity_reversal=np.zeros((n, width), bool), jitter=np.zeros((n, width), bool),
                  abrupt_change=np.zeros(n, bool), anomaly_score=np.zeros(n))
    groups = []
    labels = [str(x).lower() for x in names]
    specs = []
    if width in (10, 20) and any("eef_x" in x for x in labels):
        from scipy.spatial.transform import Rotation
        sides = ("left", "right") if width == 20 else (arm_side,)
        for arm, side in enumerate(sides):
            start = arm * 10
            specs.append((side, "translation", "m", list(range(start, start+3)), state[:, start:start+3], .01, .5, 5.))
            try:
                rotations = np.array([rotation6d_to_matrix(row) for row in state[:, start+3:start+9]])
                if not np.isfinite(rotations).all():
                    raise ValueError("nonfinite orientation")
                # Spatial relative rotations avoid rotation6D coordinate derivatives
                # and Euler wrap discontinuities. Integrate increments solely to
                # feed the finite-difference detector in physical angular units.
                increments = Rotation.from_matrix(rotations[1:] @ rotations[:-1].transpose(0,2,1)).as_rotvec() if n > 1 else np.zeros((0,3))
                angular_path = np.vstack([np.zeros((1,3)), np.cumsum(increments, axis=0)]) if n else np.zeros((0,3))
                specs.append((side, "rotation", "rad", list(range(start+3,start+6)), angular_path, .05, 2., 12.))
            except ValueError:
                groups.append(dict(side=side, kind="rotation", checked=False, reason="invalid rotation6D"))
    else:
        for side, prefixes in (("left", ("left", "fl")), ("right", ("right", "fr"))):
            cols = [i for i,label in enumerate(labels) if any(label.startswith(p+"_") for p in prefixes) and re.search(r"joint_?\d",label)]
            if cols:
                specs.append((side,"joint","rad",cols,state[:,cols],.05,2.,12.))
    for side, kind, unit, cols, values, reversal, accel, jerk in specs:
        if not np.isfinite(values).all():
            groups.append(dict(side=side,kind=kind,checked=False,reason="nonfinite state"))
            continue
        a = detect_motion_anomalies(values, timestamps, fps=fps,
                                    velocity_reversal_threshold=reversal,
                                    acceleration_threshold=accel, jerk_threshold=jerk)
        # Angular axes are group diagnostics, not derivatives of rotation6D fields.
        if kind != "rotation":
            result['accelerations'][:,cols] = a['accelerations']
        for key in ('velocity_reversal','jitter'):
            result[key][:,cols] = a[key]
        result['abrupt_change'] |= a['abrupt_change']
        result['anomaly_score'] = np.maximum(result['anomaly_score'], a['anomaly_score'])
        groups.append(dict(side=side,kind=kind,unit=unit,checked=True,
                           acceleration_unit=unit+"/s²",jerk_unit=unit+"/s³",
                           acceleration_limit=float(a['acceleration_limit']),jerk_limit=float(a['jerk_limit']),
                           reversal_speed_threshold=reversal,
                           reversal_frames=int(a['velocity_reversal'].any(1).sum()),
                           jitter_frames=int(a['jitter'].any(1).sum()),abrupt_frames=int(a['abrupt_change'].sum()),
                           warning_runs=idle_runs(a["anomaly"]),
                           acceleration_norm=a['acceleration_norm'],jerk_norm=a['jerk_norm'],
                           abrupt_change=a['abrupt_change']))
    rad_groups = [g for g in groups if g.get("checked") and g["kind"] == "joint"]
    result["acceleration_norm"] = np.maximum.reduce([g["acceleration_norm"] for g in rad_groups]) if rad_groups else np.full(n, np.nan)
    result["jerk_norm"] = np.maximum.reduce([g["jerk_norm"] for g in rad_groups]) if rad_groups else np.full(n, np.nan)
    result['diagnostics'] = dict(version=3, checked=bool(specs) and all(g['checked'] for g in groups),
                                 interpretation="Statistical review warnings, not hardware safety limits; grippers excluded.",
                                 groups=groups)
    return result


def motion_payload(diagnostics, *, index=None):
    """Keep the JSON API unit-labelled; never sum translation and rotation."""
    out = {k:v for k,v in diagnostics.items() if k != 'groups'}
    out['groups'] = []
    for group in diagnostics['groups']:
        item = {k:v for k,v in group.items() if not isinstance(v,np.ndarray)}
        if index is not None:
            for key in ('acceleration_norm','jerk_norm','abrupt_change'):
                if key in group:
                    item[key] = group[key][index].item()
        out['groups'].append(item)
    return out
