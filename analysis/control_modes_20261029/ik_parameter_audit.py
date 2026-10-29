"""Small recorded-state IK sweeps. Never imports or opens a CAN interface.

Observed joints are replayed as feedback, not simulated after changed commands;
this tests geometric continuity/latency, not closed-loop task success.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.deployment.client import PiperContinuousIK, rotation_from_state, project_delivery_target_step
from bimanual_vla.deployment.hardware_safety import NOMINAL_JOINT_LIMITS_RAD
from bimanual_vla.deployment.ik_workers import solve_ik_request, IKUnreachableError

ROOT = Path(__file__).resolve().parents[2]


def quantiles(values):
    return dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(values, [50, 95, 99, 100]).tolist())) if values else {}


def samples():
    stats = {'train': [], 'validation': []}
    segments = []
    provenance = []
    for i, path in enumerate(sorted((ROOT/'episodes_piper_v21/pick_place_clean_v3_20261025').glob('ep_*.npz'))):
        split = 'train' if i < 81 else 'validation'
        with np.load(path, allow_pickle=False) as z:
            q = z['joint_qpos'][:-1].reshape(-1, 2, 7)[:, :, :6].astype(float)
            continuous = np.diff(z['cleaning_source_frame_indices'][:-1]) == 1
            step = np.max(abs(np.diff(q, axis=0)), axis=2)
            stats[split].extend(step[continuous].ravel().tolist())
            if i not in (0, 20, 40, 60, 81, 85):
                continue
            state = z['state'][:-1].reshape(-1, 2, 10).astype(float)
            # Peak-motion neighborhoods deliberately stress normal demonstrations.
            center = int(np.argmax(np.where(continuous, step.max(axis=1), 0)))
            start, end = max(0, center-15), min(len(q)-1, center+15)
            provenance.append(dict(file=str(path.relative_to(ROOT)), split=split, start=start, end=end))
            for arm in range(2):
                sequence = []
                for k in range(start, end):
                    if not continuous[k]:
                        if sequence:
                            segments.append((split, sequence)); sequence = []
                        continue
                    sequence.append((q[k, arm], state[k+1, arm]))
                if sequence:
                    segments.append((split, sequence))
    run = ROOT/'deployment_runs/20261029T094625.632498+0800_782639'
    rows = [json.loads(line) for line in (run/'trajectory.jsonl').open()]
    # Recorded failure vicinity, distinct from the demonstration parameter selection.
    for arm in range(2):
        sequence = []
        for row in rows[510:539]:
            goal = np.asarray(row['command_absolute_target'][arm*10:arm*10+10], float)
            if not np.isfinite(goal).all():
                continue
            current = np.asarray(row['delivery_state'][arm*10:arm*10+10])
            goal, _ = project_delivery_target_step(current, goal, max_translation_step_m=.05, max_rotation_step_rad=.18)
            sequence.append((np.asarray(row['qpos'][arm*7:arm*7+6]), goal))
        segments.append(('failure_window', sequence))
    return {k: dict(n=len(v), step_rad=quantiles(v)) for k, v in stats.items()}, segments, provenance


def sweep(segments, radius, weight):
    solver = PiperContinuousIK()
    result = {}
    for split, sequence in segments:
        item = result.setdefault(split, dict(total=0, accepted=0, failed=0, invalid_feedback=0,
            relaxed=0, joint_step_rad=[], position_error_mm=[], rotation_error_rad=[], solve_ms=[]))
        previous = None
        for current, goal in sequence:
            item['total'] += 1
            if np.any(current < NOMINAL_JOINT_LIMITS_RAD[:, 0]-.06) or np.any(current > NOMINAL_JOINT_LIMITS_RAD[:, 1]+.06):
                item['invalid_feedback'] += 1; previous = None; continue
            started = time.perf_counter()
            try:
                solved = solve_ik_request(solver, (current, goal[:3],
                    Rotation.from_matrix(rotation_from_state(goal.copy())).as_euler('xyz', degrees=True)),
                    dict(initial_joints_rad=previous, max_joint_step_rad=.3, search_joint_radius_rad=.3,
                         reference_joint_radius_rad=radius, joint_regularization_weight=weight,
                         position_tolerance_m=.0015, rotation_tolerance_rad=.02, max_nfev=100))
            except IKUnreachableError:
                item['failed'] += 1
                previous = None  # Marks an interrupted segment, not an automatic live retry.
            else:
                item['accepted'] += 1
                item['relaxed'] += int(solved['tolerance_relaxed'])
                if previous is not None:
                    item['joint_step_rad'].append(float(np.max(abs(solved['solution_joints_rad']-previous))))
                previous = solved['solution_joints_rad']
                item['position_error_mm'].append(solved['solution_position_error_m']*1000)
                item['rotation_error_rad'].append(solved['solution_rotation_error_rad'])
            item['solve_ms'].append((time.perf_counter()-started)*1000)
    for item in result.values():
        for key in ('joint_step_rad', 'position_error_mm', 'rotation_error_rad', 'solve_ms'):
            item[key] = quantiles(item[key])
    return result


def main():
    stats, segments, provenance = samples()
    report = dict(dataset_steps=stats, sampled_segments=provenance, profiles=[])
    for radius, weight in ((.08, 1.), (.12, 1.), (.20, 1.), (.30, 1.), (.12, .25), (.12, 4.)):
        item = dict(reference_radius_rad=radius, regularization_weight=weight,
                    results=sweep(segments, radius, weight))
        report['profiles'].append(item)
        print(json.dumps(item), flush=True)
    Path(__file__).with_name('ik_parameter_results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
