# Tracking-lag pauses and Ruckig delay (2026-11-04)

Offline replays of the 09:51 bimanual EEF run
(`deployment_runs/20261104T095105.597618+0800_679521`, 45.8 s, 5 tracking-lag
freezes of 0.5–0.9 s each). Conclusions and the resulting parameters are in
`docs/deployment/RTC_CLIENT_GUIDE.md` ("连续跟踪与持续失配", "IK 目标提前量").

```bash
PY=/home/user/miniconda3/envs/dual_arm/bin/python
RUN=deployment_runs/20261104T095105.597618+0800_679521
$PY -I analysis/tracking_pauses_20261104/guard_replay.py $RUN    # freezes, windowed guard, delay drift, reach
$PY -I analysis/tracking_pauses_20261104/ruckig_replay.py $RUN   # delay split, Ruckig variants
```

Results for this run:

| | value |
|---|---|
| IK reference → Ruckig command → feedback | 190 ms + 65 ms (260 ms total) |
| EEF peak before the 3 lag freezes, newest target / 300 ms window | 6.5–9.9 cm / 1.2–3.6 cm |
| sent-joint peak before 13.39 s, newest / 100 ms window | 0.111 / 0.043 rad |
| right arm max reach measured / target | 63.5 / 69.5 cm (Piper ~626 mm) |
| Ruckig lag: as run / a=16 j=200 / target lead 150 ms / 200 ms | 195 / 120 / 60 / 10 ms |
| mean planner acceleration: as run / a=16 j=200 / lead 200 ms | 1.14 / 2.37 / 1.12 rad/s² |

The zero-lead replay reproduces the logged Ruckig output exactly. Lead replays
approximate the future IK solution by the reference logged later and do not
model firmware dynamics; confirm on the robot by rerunning both scripts on the
next run (`ik_target_lead` in each command trace records the row used).
Left-arm delay stays 200–320 ms over the run (no accumulation); the right arm's
410–590 ms in the second half is the unreachable target, not growing lag.

## 10:33 run (lead 0.15 s, windows 0.10/0.30 s)

`deployment_runs/20261104T103348.257344+0800_698468`, 27.8 s, 2 freezes.
Measured-vs-target delay fell to 60–220 ms (left) and 40–100 ms (right), but
both freezes were sustained lag: the right arm leaving the start pose (J2 at
its 0 limit) and the left arm reconfiguring after J1 reached its 2.618 rad
limit. In both, the IK reference led the measured EEF by only 4.0–4.8 cm while
the policy target led by 10–11 cm: the 5 cm per-tick projection capped EEF
tracking near 18 cm/s. Raising the joint speed cap does not help (the
reference itself is capped).

```bash
# SIDE 0=left 1=right, window START END, then projection bounds in metres
$PY -I analysis/tracking_pauses_20261104/closed_loop_replay.py $RUN 1 0.2 1.58 .05 .08 .10
$PY -I analysis/tracking_pauses_20261104/closed_loop_replay.py $RUN 0 13.0 15.16 .05 .08 .10
```

| projection | right 0.2–1.58 s | left 13.0–15.16 s |
|---|---|---|
| 5 cm | 18.6 cm/s, guard error max 5.1 cm | servo bound fault at 14.11 s (J1 limit) |
| 8 cm | 24.9 cm/s, 2.9 cm | 26.7 cm/s, 4.0 cm |
| 10 cm | 26.4 cm/s, 2.9 cm | 23.4 cm/s, 3.8 cm |

The closed loop is kinematic (pure 65 ms firmware delay, no dynamics); its
5 cm J1-limit fault did not occur on the robot, where J1 approached the limit
more slowly. It does show that a fast approach to a joint limit followed by an
IK reversal can exceed the Ruckig plan bound check, independent of projection.
