"""Explain tracking-lag freezes in a run and replay the windowed guard.

Usage: python -I guard_replay.py RUN_DIR

For every freeze: the logged reason, then the error peaks in the preceding
second with and without the latency windows.  The EEF check compares the next
feedback with the closest policy target within the window; the joint check
compares it with the servo command published at each control tick.  Also
reports, per arm, measured-vs-target delay over time (growth would mean
accumulating lag) and how far each arm reaches from its base.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bimanual_vla.deployment.client import (  # noqa: E402
    DEFAULT_TRACKING_JOINT_LAG_WINDOW_S as JOINT_WINDOW,
    DEFAULT_TRACKING_REFERENCE_LAG_WINDOW_S as EEF_WINDOW)

RUN = Path(sys.argv[1])
ARM = [i for i in range(14) if i % 7 != 6]
rows = [json.loads(line) for line in open(RUN/'trajectory.jsonl')]
t0 = rows[0]['monotonic_timestamp']
ts = np.array([r['monotonic_timestamp'] - t0 for r in rows])
state = np.array([r['delivery_state'] for r in rows])
target = np.array([[np.nan if v is None else v for v in (r['command_absolute_target'] or [None]*20)]
                   for r in rows], float)
executing = np.array([r['execution_state'] == 'executing' for r in rows])
freezes = [i for i in range(1, len(rows))
           if rows[i]['execution_state'] == 'holding' and rows[i-1]['execution_state'] != 'holding']

servo = [json.loads(line) for line in open(RUN/'servo_commands.jsonl')]
servo_t = np.array([s['monotonic_timestamp'] for s in servo])
servo_pos = np.array([s['position'] for s in servo])


def eef_error(i, window):
    """Next-feedback distance to the closest executed target within ``window``."""
    errors = []
    for side in range(2):
        candidates = [j for j in range(i + 1) if executing[j] and ts[i] - ts[j] <= window + 1e-9
                      and np.isfinite(target[j, 0])]
        errors.append(min(np.linalg.norm(target[j, side*10:side*10+3] - state[i+1, side*10:side*10+3])
                          for j in candidates))
    return max(errors)


commands = []  # (row index, servo command at the control tick)
for i, r in enumerate(rows):
    if r['execution_state'] == 'executing' and r['command_monotonic_timestamp']:
        k = np.searchsorted(servo_t, r['command_monotonic_timestamp'], side='right') - 1
        commands.append((i, servo_pos[k]))


def joint_error(n, window):
    i = commands[n][0]
    measured = np.array(rows[i+1]['qpos'])[ARM]
    candidates = [c for m, c in commands[:n+1] if ts[i] - ts[m] <= window + 1e-9]
    return max(min(np.abs(measured[s*6:s*6+6] - c[s*6:s*6+6]).max() for c in candidates) for s in range(2))


print(f"{len(freezes)} tracking-lag freezes in {ts[-1]:.1f} s")
for f in freezes:
    before = [i for i in range(len(rows) - 1) if executing[i] and ts[f] - 1 <= ts[i] < ts[f]]
    joints = [n for n, (i, _) in enumerate(commands[:-1]) if ts[f] - 1 <= ts[i] < ts[f]]
    resume = next((ts[i] for i in range(f, len(rows)) if executing[i]), ts[-1])
    print(f"\n{ts[f]:6.2f} s, stopped {resume - ts[f]:.2f} s: {rows[f]['blocked_reason'][:110]}")
    if before:
        print("  EEF peak  newest {:.1f} cm | {:.0f} ms window {:.1f} cm".format(
            max(eef_error(i, 0.) for i in before) * 100, EEF_WINDOW * 1000,
            max(eef_error(i, EEF_WINDOW) for i in before) * 100))
    if joints:
        print("  joint peak newest {:.3f} rad | {:.0f} ms window {:.3f} rad".format(
            max(joint_error(n, 0.) for n in joints), JOINT_WINDOW * 1000,
            max(joint_error(n, JOINT_WINDOW) for n in joints)))

grid = np.arange(0, ts[-1], .01)
valid = executing & np.isfinite(target[:, 0])
for side, name in enumerate('LR'):
    delays = []
    for start in np.arange(2, ts[-1] - 4, 8):
        g = grid[(grid >= start) & (grid < start + 8)]
        on = np.interp(g, ts, executing.astype(float)) > .99
        measured = np.stack([np.interp(g, ts, state[:, side*10+k]) for k in range(3)], 1)
        if np.linalg.norm(np.diff(measured, axis=0), axis=1).sum() < .05:
            delays.append('still')
            continue
        best = min(((np.mean(np.linalg.norm(np.stack([np.interp(g - d, ts[valid], target[valid, side*10+k])
                                                       for k in range(3)], 1)[on] - measured[on], axis=1)), d)
                    for d in np.arange(0, .6, .01)))
        delays.append(f"{best[1]*1000:.0f} ms")
    reach = np.linalg.norm(state[:, side*10:side*10+3], axis=1)
    want = np.linalg.norm(target[:, side*10:side*10+3], axis=1)
    print(f"\n{name}: measured-vs-target delay per 8 s window {delays}; "
          f"max reach measured {np.nanmax(reach)*100:.1f} cm, target {np.nanmax(want)*100:.1f} cm")
