"""Closed-loop replay with actuator lag: measured joints chase IK targets at <=1.2 rad/s, 1-tick delay."""
import sys, glob, time
import numpy as np
sys.path.insert(0, "/home/user/project/bimanual-vla"); sys.path.insert(0, "/home/user/project/bimanual-vla/analysis/ik_bounded_lm_20261102/old")
from scipy.spatial.transform import Rotation
from bimanual_vla.deployment.client import rotation_from_state, ExecutionBlocked
from bimanual_vla.deployment.hardware_safety import NOMINAL_JOINT_LIMITS_RAD
which, ref_r, search_r = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
if which == "old":
    import old_client as mod
else:
    from bimanual_vla.deployment import client as mod
solver = mod.PiperContinuousIK()
kw = dict(max_joint_step_rad=search_r, search_joint_radius_rad=search_r, reference_joint_radius_rad=ref_r,
          joint_regularization_weight=.25, position_tolerance_m=.003, rotation_tolerance_rad=.02,
          max_nfev=100, joint_limits=NOMINAL_JOINT_LIMITS_RAD, minimum_motion=True)
files = sorted(glob.glob("/home/user/project/bimanual-vla/episodes_piper_v21/pick_place_clean_v3_20261025/ep_*.npz"))[::9]
vmax = 1.2 / 30
n = blocked = relaxed = big = 0; steps, lagmax, perr, rerr, times = [], [], [], [], []
for f in files:
    with np.load(f) as z: s = z["state"].astype(float); qt = z["joint_qpos"].astype(float)
    for arm in (0, 1):
        actual = qt[0, arm*7:arm*7+6].copy(); delayed = actual.copy(); warm = None; prev_solution = actual.copy()
        for t in range(1, len(s)):
            g = s[t, arm*10:arm*10+10]; n += 1
            rpy = Rotation.from_matrix(rotation_from_state(g.copy())).as_euler("xyz", degrees=True)
            t0 = time.perf_counter()
            try:
                r = solver.solve_with_diagnostics(delayed, g[:3], rpy, initial_joints_rad=warm, **kw)
                target = r.solution_joints_rad; relaxed += bool(r.tolerance_relaxed)
                st = float(np.max(np.abs(target - prev_solution))); steps.append(st); big += st > .097
                xyz, R = solver.pose(target); perr.append(np.linalg.norm(xyz-g[:3])); rerr.append(Rotation.from_matrix(rotation_from_state(g.copy()) @ R.T).magnitude())
                warm = target; prev_solution = target
            except ExecutionBlocked:
                blocked += 1; target = actual.copy(); warm = None   # hold, re-anchor at measured
            times.append(time.perf_counter() - t0)
            delayed = actual.copy()
            actual = actual + np.clip(target - actual, -vmax, vmax)
            lagmax.append(float(np.max(np.abs(target - actual))))
steps = np.array(steps)
print(f"[{which} ref={ref_r} search={search_r}] ticks={n} BLOCKED={blocked} ({blocked/n*100:.2f}%) relaxed={relaxed} ({relaxed/n*100:.1f}%) "
      f"big(>0.097)={big} max_step={steps.max():.3f} | pos mean={np.mean(perr)*1000:.2f}mm p99={np.percentile(perr,99)*1000:.1f}mm "
      f"rot mean={np.mean(rerr):.4f} p99={np.percentile(rerr,99):.4f} | solve ms p50={np.median(times)*1000:.1f} p99={np.percentile(times,99)*1000:.1f}")
