# Bounded Levenberg-Marquardt IK replacement (2026-11-02)

Closed-loop kinematic replays comparing the old minimum-motion IK (commit
e324d7d) with the bounded Levenberg-Marquardt solver. The simulated arm follows
each IK solution at <= 1.2 rad/s with one control period of feedback delay; no
dynamics, contact or cameras. Results are summarised in
`docs/deployment/LOCAL_IK_PARAMETERS.md`.

```bash
./prepare_old_solver.sh
PY=/home/user/miniconda3/envs/dual_arm/bin/python
# demonstration targets (pick_place_clean_v3_20261025, every 9th episode)
$PY -I lag_replay.py old 0.12 0.30
$PY -I lag_replay.py new 0.20 0.40
# recorded 2026-11-02 policy targets, with joint/EEF step ratios
$PY -I policy_replay_ratio.py old 0.12 0.30
$PY -I policy_replay_ratio.py new 0.20 0.40
```
