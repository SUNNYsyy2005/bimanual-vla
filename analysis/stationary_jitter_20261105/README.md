# End-effector jitter while the arm should be still (2026-11-05)

Runs on the 8099 policy server (3 denoising steps), 2026-11-05 09:51–10:37.
Conclusions and the client change are in `docs/deployment/RTC_CLIENT_GUIDE.md`
("EEF chunk 内零相位平滑").

```bash
PY=/home/user/miniconda3/envs/dual_arm/bin/python
DATA=episodes_piper_v21/pick_place_clean_v3_20261025
RUNS=$(ls -d deployment_runs/20261105T09*/ deployment_runs/20261105T1028*/ deployment_runs/20261105T1029*/ deployment_runs/20261105T103*/)
$PY -I analysis/stationary_jitter_20261105/jitter_layers.py $RUNS            # jitter per layer in still windows
$PY -I analysis/stationary_jitter_20261105/policy_jitter.py $DATA $RUNS      # chunk noise vs demonstrations
$PY -I analysis/stationary_jitter_20261105/smoothing_choice.py $DATA $RUNS   # filter choice: noise vs demo distortion
$PY -I analysis/stationary_jitter_20261105/chunk_switch.py $RUNS             # target jump at chunk switches
$PY -I analysis/stationary_jitter_20261105/closed_loop_chunk_smoothing.py RUN SIDE START END 0 5 7
```

| measurement | value |
|---|---|
| policy step noise, still / demonstrations still | 3.6–4.3 mm / 0.13 mm |
| rotation noise, policy / demonstrations | 0.5–0.6° / 0.6° |
| jitter by layer in still windows: target, IK ref, Ruckig, measured | 3–5, 2–4.5, 0.5–1.9, 0.5–2 mm |
| SG 7 rows: remaining policy noise | 1.59 mm |
| SG 7 rows on demos: settled max / slow-next-to-stop p99, max | 0.38 / 0.59, 1.36 mm |
| chunk switch jump (still, smoothed) p50 / p90 | 1.0–1.9 / 2–4 mm |
| policy target drift per 333 ms in still windows (smoothed) / demos | 2.5–5.4 / 0.00 mm |

Not helpful: Ruckig phase synchronization (falls back to time synchronization
on every retarget), 9+ row windows (corner distortion > 1 mm).
