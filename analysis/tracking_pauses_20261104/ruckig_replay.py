"""Replay a run's logged IK references through the servo's Ruckig planner.

Usage: python -I ruckig_replay.py RUN_DIR

Reports the reference -> Ruckig -> feedback delays, then re-plans the logged
references (zero terminal velocity, as in the EEF path) with higher limits,
with finite-difference terminal velocity, and with a target lead (each tick
retargets to the reference logged tau seconds later).  The zero-lead replay
reproduces the logged Ruckig output exactly, which validates the model.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bimanual_vla.deployment.servo import IndependentArmJointPlanner, ServoConfig  # noqa: E402

RUN = Path(sys.argv[1])
ARM = [i for i in range(14) if i % 7 != 6]
meta = json.loads((RUN/'metadata.json').read_text())
limits = meta['runtime_configuration']['effective_joint_limits_rad']
LOWER = np.r_[np.array(limits['left'])[:, 0], np.array(limits['right'])[:, 0]]
UPPER = np.r_[np.array(limits['left'])[:, 1], np.array(limits['right'])[:, 1]]
CONFIG = ServoConfig(**{k: v for k, v in meta['runtime_configuration']['servo'].items()
                        if k in ServoConfig.__dataclass_fields__})
S = [json.loads(line) for line in open(RUN/'servo_commands.jsonl')]
T = np.array([s['monotonic_timestamp'] for s in S])
BRAKING = np.array([bool(s.get('braking')) for s in S])
REF = np.array([[s['reference'][i] for i in ARM] for s in S])
POS = np.array([s['position'] for s in S])
VEL = np.array([s['velocity'] for s in S])
ACC = np.array([s['acceleration'] for s in S])
FB = np.array([[s['feedback_qpos'][i] for i in ARM] for s in S])
FB_T = np.array([s['feedback_read_monotonic'] for s in S])

segments, start = [], None
for i in range(len(S)):
    if not BRAKING[i] and start is None:
        start = i
    if (BRAKING[i] or i == len(S) - 1) and start is not None:
        if i - start > 100:
            segments.append((start, i))
        start = None


def delay(a_t, a, b_t, b, lo=0., hi=.4):
    """Shift of ``a`` that best explains ``b`` (least squares over all joints)."""
    grid = np.arange(b_t[0] + 1, b_t[-1] - 1, .005)
    best = (0., np.inf)
    for d in np.arange(lo, hi, .005):
        err = sum(np.mean((np.interp(grid - d, a_t, a[:, j]) - np.interp(grid, b_t, b[:, j]))**2)
                  for j in range(a.shape[1]))
        if err < best[1]:
            best = (d, err)
    return best[0]


def replan(config, *, lead=0., velocity=False):
    out_t, out_p, out_r, out_a = [], [], [], []
    for a, b in segments:
        planner = IndependentArmJointPlanner(POS[a], LOWER, UPPER, config)
        planner.set_state(POS[a], VEL[a], ACC[a])
        last = last_at = None
        for i in range(a + 1, b):
            j = min(np.searchsorted(T, T[i] + lead, side='right') - 1, b - 1)
            target = REF[j]
            if last is None or not np.array_equal(target, last):
                speed = np.zeros(12)
                if velocity and last is not None and 0 < T[i] - last_at <= config.target_timeout_s:
                    speed = np.clip((target - last) / (T[i] - last_at), -config.max_velocity, config.max_velocity)
                planner.retarget(target, speed)
                last, last_at = target.copy(), T[i]
            planner.advance()
            out_t.append(T[i]); out_p.append(planner.position.copy())
            out_r.append(REF[i]); out_a.append(planner.acceleration.copy())
    return tuple(np.array(x) for x in (out_t, out_p, out_r, out_a))


def report(name, t, p, r, a):
    lag = delay(t, r, t, p, -.2, .45)
    print(f"{name:34s} lag {lag*1000:4.0f} ms  p95 |ref-cmd| {np.percentile(np.abs(p-r).max(1), 95):.3f} rad  "
          f"mean |acc| {np.abs(a).mean():.2f} rad/s2")


print(f"IK reference -> Ruckig command : {delay(T, REF, T, POS)*1000:.0f} ms")
print(f"Ruckig command -> feedback      : {delay(T, POS, FB_T, FB)*1000:.0f} ms")
print(f"IK reference -> feedback        : {delay(T, REF, FB_T, FB)*1000:.0f} ms")
report('replay as run', *replan(CONFIG))
report('finite-difference terminal velocity', *replan(CONFIG, velocity=True))
for v, acc, jerk in ((1.2, 12, 120), (1.2, 16, 200), (2.0, 16, 200)):
    report(f'limits v={v} a={acc} j={jerk}', *replan(replace(CONFIG, max_velocity=v, max_acceleration=acc, max_jerk=jerk)))
for lead in (.1, .15, .2):
    report(f'target lead {lead*1000:.0f} ms', *replan(CONFIG, lead=lead))
