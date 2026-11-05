# End-effector jitter while the arm should be still (2026-11-05)

Runs on the 8099 policy server (3 denoising steps), 2026-11-05 09:51–10:37.
Conclusions and the client change are in `docs/deployment/RTC_CLIENT_GUIDE.md`
("EEF chunk 内零相位平滑").

```bash
PY=/home/user/miniconda3/envs/dual_arm/bin/python
DATA=episodes_piper_v21/pick_place_clean_v3_20261025
RUNS=$(ls -d deployment_runs/20261105T09*/ deployment_runs/20261105T1028*/ deployment_runs/20261105T1029*/ deployment_runs/20261105T103*/)
D=analysis/stationary_jitter_20261105
$PY -I $D/jitter_layers.py $RUNS                  # jitter per layer in still windows
$PY -I $D/jitter_bands.py $RUNS                   # drift / 0.5-3 Hz wobble / 3-15 Hz per layer
$PY -I $D/wobble_source.py $RUNS                  # wobble inside one chunk vs across switches
$PY -I $D/policy_jitter.py $DATA $RUNS            # chunk noise vs demonstrations
$PY -I $D/smoothing_choice.py $DATA $RUNS         # short window: noise vs demo distortion
$PY -I $D/smoothing_sweep.py $DATA $RUNS          # plain vs still-gated long windows
$PY -I $D/rotation_gate.py $DATA $RUNS            # angular threshold of the still gate
$PY -I $D/verify_client_smoothing.py $DATA $RUNS  # the client's smooth_delivery_targets
$PY -I $D/chunk_switch.py $RUNS                   # target jump at chunk switches
# closed loop (IK -> Ruckig -> 65 ms delay), first 1 s excluded; STILL=0 is short window only
while read name side a b; do run=$(ls -d deployment_runs/${name}*/); for s in 0 31; do
  STILL=$s $PY -I $D/deadband_hold.py loop $run $side $a $b; done; done < $D/still_windows.txt
```

| measurement | value |
|---|---|
| policy step noise, still / demonstrations still | 3.6–4.3 mm / 0.13 mm |
| measured EEF in still windows: < 0.5 Hz / 0.5–3 Hz / 3–15 Hz (median) | 0.73 / 1.54 / 0.42 mm, 4–13 reversals/s |
| 0.5–3 Hz wobble: executed sequence / one chunk alone | 2.32 / 2.08 mm |
| still-chunk wobble: raw / 7 rows / 7 + 31 gated | 2.89 / 1.89 / 0.97 mm |
| demo moved by 7 + 31 gated: settled / near stop / all rows (max) | 1.75 / 2.56 / 2.64 mm |
| closed-loop 0.5–3 Hz wobble median, 17 still windows: raw / 7 / 7 + 31 | 1.21 / 1.04 / 0.66 mm |
| closed loop with a constant target | 0.00 mm (no self-oscillation) |

Not adopted: Ruckig phase synchronization (falls back to time synchronization
on every retarget), larger IK regularization (more wobble), long windows gated
by the centre-row speed (demo stop positions moved 3.8–9.5 mm), a 1.5–2 mm
deadband follower on top (little gain, uses the 3 mm budget; see
`deadband_hold.py`), and a filter spanning executed history plus the chunk
ahead (no better than per-chunk).
