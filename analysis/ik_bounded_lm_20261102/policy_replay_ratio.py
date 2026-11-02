"""Replay recorded 2026-11-02 policy EEF targets through old/new IK with a lagged simulated arm."""
import sys, glob, json, time
import numpy as np
sys.path.insert(0, "/home/user/project/bimanual-vla"); sys.path.insert(0, "/home/user/project/bimanual-vla/analysis/ik_bounded_lm_20261102/old")
from scipy.spatial.transform import Rotation
from bimanual_vla.deployment import client as new
from bimanual_vla.data.contract import build_delivery_state
which, ref_r, search_r = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
mod = __import__("old_client") if which == "old" else new
solver = mod.PiperContinuousIK()
lim = np.asarray(json.load(open(glob.glob("/home/user/project/bimanual-vla/deployment_runs/20261102T122243*/metadata.json")[0]))["runtime_configuration"]["effective_joint_limits_rad"]["left"])
kw = dict(max_joint_step_rad=search_r, search_joint_radius_rad=search_r, reference_joint_radius_rad=ref_r,
          joint_regularization_weight=.25, position_tolerance_m=.003, rotation_tolerance_rad=.02, max_nfev=100,
          joint_limits=lim, minimum_motion=True)
vmax = 1.2 / 30; total = blocked = relaxed = big = 0; reasons = {}; ratios = []
for run in sorted(glob.glob("/home/user/project/bimanual-vla/deployment_runs/20261102T*/trajectory.jsonl")):
    rows = [json.loads(l) for l in open(run)]
    rows = [r for r in rows if r.get("command_absolute_target") and r["command_absolute_target"][0] is not None]
    if not rows: continue
    for arm in (0, 1):
        actual = np.asarray(rows[0]["qpos"][arm*7:arm*7+6], float); actual = np.clip(actual, lim[:, 0], lim[:, 1])
        delayed = actual.copy(); warm = None; prev = actual.copy(); run_blocked = 0
        for r in rows:
            g = np.asarray(r["command_absolute_target"][arm*10:arm*10+10], float); total += 1
            xyz, R = solver.pose(delayed)
            state = build_delivery_state(xyz, R, .035).astype(float)
            projected, _ = new.project_delivery_target_step(state, g, max_translation_step_m=new.DEFAULT_MAX_TRANSLATION_STEP_M,
                                                            max_rotation_step_rad=new.DEFAULT_MAX_ROTATION_STEP_RAD)
            rpy = Rotation.from_matrix(new.rotation_from_state(projected.copy())).as_euler("xyz", degrees=True)
            try:
                res = solver.solve_with_diagnostics(delayed, projected[:3], rpy, initial_joints_rad=warm, **kw)
                target = res.solution_joints_rad; relaxed += bool(res.tolerance_relaxed)
                st = float(np.max(np.abs(target - prev))); big += st > .097
                pa, Ra = solver.pose(prev); pb, Rb = solver.pose(target)
                cart = np.linalg.norm(pb - pa) / .3 + Rotation.from_matrix(Rb @ Ra.T).magnitude()
                ratios.append((st, cart, st / max(cart, 1e-4)))
                prev = target; warm = target
            except new.ExecutionBlocked as exc:
                blocked += 1; run_blocked += 1; target = actual.copy(); warm = None
                k = str(exc).split(":")[0][:60]; reasons[k] = reasons.get(k, 0) + 1
            delayed = actual.copy(); actual = actual + np.clip(target - actual, -vmax, vmax)
        print(f"  {run.split('/')[-2][:15]} arm={'LR'[arm]} ticks={len(rows)} blocked={run_blocked}")
print(f"[{which} ref={ref_r} search={search_r}] total={total} BLOCKED={blocked} relaxed={relaxed} big_steps(>0.097)={big} reasons={reasons}")
r = np.array(ratios)
sel = r[:, 0] > .05
print(f"  joint-step/EEF-step ratio (steps>0.05 rad, n={sel.sum()}): median={np.median(r[sel,2]):.2f} p95={np.percentile(r[sel,2],95):.2f} max={r[sel,2].max():.2f}; "
      f"steps>0.05 with EEF change <0.01: {int(np.sum(sel & (r[:,1] < .01)))}  max_step={r[:,0].max():.3f}")
