"""Closed-loop replay of local IK tracking against a run's policy targets.

Usage: python -I closed_loop_replay.py RUN_DIR SIDE START END [STEP_M ...]

Re-runs the streaming path for one arm on the logged head-of-queue policy
targets: IK target lead -> per-tick Cartesian projection (STEP_M, rotation
scaled by the same factor from 0.05 m / 0.18 rad) -> bounded LM IK
(best effort, run's tolerances/radii) -> zero-terminal Ruckig at 100 Hz ->
pure firmware delay. Reports EEF speed and the windowed EEF guard error,
and whether the reference trigger (>5 cm for >=0.5 s) would fire.
Kinematic model only: no dynamics, contact, cameras or policy reaction.
"""
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bimanual_vla.data.action_conventions import matrix_to_rotation6d  # noqa: E402
from bimanual_vla.deployment.client import (  # noqa: E402
    DEFAULT_IK_TARGET_LEAD_S, DEFAULT_TRACKING_REFERENCE_LAG_WINDOW_S, PiperContinuousIK,
    project_delivery_target_step, rotation_from_state)
from bimanual_vla.deployment.servo import RuckigJointPlanner, ServoConfig, ServoError  # noqa: E402

RUN, SIDE, START, END = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
STEPS = [float(x) for x in sys.argv[5:]] or [.05, .08, .10]
FIRMWARE_DELAY_S = .065
meta = json.loads((RUN/'metadata.json').read_text())
config = meta['runtime_configuration']
limits = np.array(config['effective_joint_limits_rad']['left' if SIDE == 0 else 'right'])
servo = replace(ServoConfig(**{k: v for k, v in config['servo'].items() if k in ServoConfig.__dataclass_fields__}),
                independent_arms=False)
ik_config = config['ik_execution']
rows = [json.loads(line) for line in open(RUN/'trajectory.jsonl')]
t0 = rows[0]['monotonic_timestamp']
logged = [(r['monotonic_timestamp'] - t0, np.array(r['command_absolute_target'][SIDE*10:SIDE*10+10]))
          for r in rows if r['execution_state'] == 'executing' and r['command_absolute_target'][0] is not None]
target_t = np.array([x[0] for x in logged]); targets = np.array([x[1] for x in logged])
start_row = next(r for r in rows if r['monotonic_timestamp'] - t0 >= START)
servo_rows = [json.loads(line) for line in open(RUN/'servo_commands.jsonl')]
servo_t = np.array([s['monotonic_timestamp'] - t0 for s in servo_rows])
k0 = np.searchsorted(servo_t, START)
ik = PiperContinuousIK()
kwargs = dict(best_effort=True, position_tolerance_m=ik_config['position_tolerance_m'],
              rotation_tolerance_rad=ik_config['rotation_tolerance_rad'],
              reference_joint_radius_rad=ik_config['reference_joint_radius_rad'],
              search_joint_radius_rad=ik_config['search_joint_radius_rad'],
              max_joint_step_rad=ik_config['search_joint_radius_rad'],
              joint_regularization_weight=ik_config['joint_regularization_weight'],
              max_nfev=100, joint_limits=limits)


def target_at(time_s):
    i = int(np.clip(np.searchsorted(target_t, time_s, side='right') - 1, 0, len(target_t) - 1))
    return targets[i]


def state_of(q):
    xyz, rotation = ik.pose(q)
    return np.r_[xyz, matrix_to_rotation6d(rotation), 0.]


for step in STEPS:
    planner = RuckigJointPlanner(np.array(servo_rows[k0]['position'][SIDE*6:SIDE*6+6]),
                                 limits[:, 0], limits[:, 1], servo)
    planner.set_state(np.array(servo_rows[k0]['position'][SIDE*6:SIDE*6+6]),
                      np.array(servo_rows[k0]['velocity'][SIDE*6:SIDE*6+6]),
                      np.array(servo_rows[k0]['acceleration'][SIDE*6:SIDE*6+6]))
    history = deque([(START, planner.position.copy())])  # Ruckig output for the firmware delay
    warm = None
    issued = deque()
    lag_since, trips, speeds, errors = None, [], [], []
    previous_xyz = None
    fault = None
    time_s = START
    while time_s < END:
        delayed = [q for at, q in history if at <= time_s - FIRMWARE_DELAY_S]
        measured = delayed[-1] if delayed else history[0][1]
        current = state_of(measured)
        if previous_xyz is not None:
            speeds.append(np.linalg.norm(current[:3] - previous_xyz) * 30)
        previous_xyz = current[:3]
        # Guard: closest head target issued within the window.
        while issued and time_s - issued[0][0] > DEFAULT_TRACKING_REFERENCE_LAG_WINDOW_S + 1e-9:
            issued.popleft()
        if issued:
            error = min(np.linalg.norm(goal[:3] - current[:3]) for _, goal in issued)
            errors.append(error)
            if error > .05:
                lag_since = time_s if lag_since is None else lag_since
                if time_s - lag_since >= .5 and not trips:
                    trips.append(time_s)
            else:
                lag_since = None
        head = target_at(time_s)
        issued.append((time_s, head))
        goal = target_at(time_s + DEFAULT_IK_TARGET_LEAD_S).copy()
        goal[9] = 0.
        projected, _ = project_delivery_target_step(current, goal, max_translation_step_m=step,
                                                    max_rotation_step_rad=.18 * step / .05)
        rpy = Rotation.from_matrix(rotation_from_state(projected.copy())).as_euler('xyz', degrees=True)
        result = ik.solve_with_diagnostics(measured, projected[:3], rpy, initial_joints_rad=warm, **kwargs)
        warm = result.solution_joints_rad.copy()
        try:
            planner.retarget(warm)
        except ServoError as exc:
            fault = f'servo bound fault at {time_s:.2f} s ({str(exc)[:60]})'
            break
        for _ in range(int(round(servo.hz / 30))):
            planner.advance()
            history.append((time_s, planner.position.copy()))
            time_s += 1 / servo.hz
        while len(history) > 2 and history[1][0] <= time_s - FIRMWARE_DELAY_S - .05:
            history.popleft()
    speeds, errors = np.array(speeds), np.array(errors)
    print(f"projection {step*100:4.1f} cm: EEF speed p90 {np.percentile(speeds, 90)*100:4.1f} max {speeds.max()*100:4.1f} cm/s; "
          f"guard error max {errors.max()*100:4.1f} cm, >5 cm {np.mean(errors > .05)*100:4.1f}% of ticks; "
          f"reference trigger {'at %.2f s' % trips[0] if trips else 'none'}" + (f'; {fault}' if fault else ''))
