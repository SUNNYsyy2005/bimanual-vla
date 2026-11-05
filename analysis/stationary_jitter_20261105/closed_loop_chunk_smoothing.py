"""Closed-loop EEF jitter with the client's chunk smoothing, from re-decoded chunks.

Usage: python -I closed_loop_chunk_smoothing.py RUN_DIR SIDE START END [SHORT[+LONG] ...]

Every executed chunk is re-decoded with decode_action_queue from its logged
raw actions and inference anchor, with eef_smoothing_window=SHORT and eef_still_window=LONG. Each
control tick takes the executed (generation, queue index) row from the run's
trajectory, and the row 0.15 s ahead in the same chunk as the IK lead target,
then streams them through bounded LM IK -> Ruckig -> 65 ms firmware delay.
Reports Ruckig-command EEF motion per band (jitter_bands.py) for one arm.
"""
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimanual_vla.data.action_conventions import matrix_to_rotation6d  # noqa: E402
from bimanual_vla.deployment.client import (  # noqa: E402
    DEFAULT_IK_TARGET_LEAD_S, PiperContinuousIK, decode_action_queue, project_delivery_target_step,
    rotation_from_state, validate_policy_metadata)
from bimanual_vla.deployment.servo import RuckigJointPlanner, ServoConfig  # noqa: E402
from jitter_bands import band_rms  # noqa: E402
from jitter_layers import pose_series  # noqa: E402

RUN, SIDE, START, END = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
WINDOWS = sys.argv[5:] or ['0', '7', '7+31']
rc = json.loads((RUN/'metadata.json').read_text())['runtime_configuration']
limits = np.array(rc['effective_joint_limits_rad']['left' if SIDE == 0 else 'right'])
servo = replace(ServoConfig(**{k: v for k, v in rc['servo'].items() if k in ServoConfig.__dataclass_fields__}),
                independent_arms=False)
ic = rc['ik_execution']
kw = dict(best_effort=True, position_tolerance_m=ic['position_tolerance_m'], rotation_tolerance_rad=ic['rotation_tolerance_rad'],
          reference_joint_radius_rad=ic['reference_joint_radius_rad'], search_joint_radius_rad=ic['search_joint_radius_rad'],
          max_joint_step_rad=ic['search_joint_radius_rad'], joint_regularization_weight=ic['joint_regularization_weight'],
          max_nfev=100, joint_limits=limits)
records = [json.loads(line) for line in open(RUN/'model_commands.jsonl')]
records = [r for r in records if r.get('accepted')]
protocol = validate_policy_metadata(dict(records[0]['protocol'], transport='openpi_websocket_v1'), 'both', 'bimanual')
rows = [json.loads(line) for line in open(RUN/'trajectory.jsonl')]
t0 = rows[0]['monotonic_timestamp']
executed = [(r['monotonic_timestamp'] - t0, r['command_generation'], r['command_queue_index']) for r in rows
            if r['execution_state'] == 'executing' and r['command_generation'] > 0
            and START - 1 <= r['monotonic_timestamp'] - t0 <= END + 1]
times = np.array([e[0] for e in executed])
lead_rows = int(np.ceil(DEFAULT_IK_TARGET_LEAD_S * protocol.action_hz - 1e-9))
start_row = min(rows, key=lambda r: abs(r['monotonic_timestamp'] - t0 - START))
q0 = np.array(start_row['qpos'][SIDE*7:SIDE*7+6])
ik = PiperContinuousIK()
for config in WINDOWS:
    window, _, still = config.partition('+')
    window, still = int(window), int(still or 0)
    chunks = {}
    for record in records:
        z = np.load(RUN/record['action_file'])
        _, decoded = decode_action_queue(z['raw_actions'], protocol, z['raw_delivery_state'], z['qpos'],
                                         generation=record['generation'], observation_capture_monotonic=0.,
                                         eef_smoothing_window=window, eef_still_window=still)
        chunks[record['generation']] = np.array([d.absolute_target[SIDE*10:SIDE*10+10] for d in decoded])

    def targets_at(time_s):
        _, generation, index = executed[int(np.clip(np.searchsorted(times, time_s, side='right') - 1, 0, len(times) - 1))]
        chunk = chunks[generation]
        return chunk[min(index, len(chunk) - 1)], chunk[min(index + lead_rows, len(chunk) - 1)]

    planner = RuckigJointPlanner(q0, limits[:, 0], limits[:, 1], servo)
    history = deque([(START, q0.copy())]); warm = None; time_s = START; command = []
    while time_s < END:
        delayed = [q for at, q in history if at <= time_s - .065]
        q = delayed[-1] if delayed else history[0][1]
        xyz, rotation = ik.pose(q)
        _, goal = targets_at(time_s)
        goal = goal.copy(); goal[9] = 0.
        projected, _ = project_delivery_target_step(np.r_[xyz, matrix_to_rotation6d(rotation), 0.], goal,
                                                    max_translation_step_m=.1, max_rotation_step_rad=.36)
        rpy = Rotation.from_matrix(rotation_from_state(projected.copy())).as_euler('xyz', degrees=True)
        warm = ik.solve_with_diagnostics(q, projected[:3], rpy, initial_joints_rad=warm, **kw).solution_joints_rad.copy()
        planner.retarget(warm)
        for _ in range(3):
            planner.advance(); command.append(planner.position.copy()); history.append((time_s, planner.position.copy()))
            time_s += .01
    xyz, _ = pose_series(np.array(command))
    bands = band_rms(xyz)
    print(f"{config:6s}: Ruckig command EEF " + " | ".join(f"{name} {value:4.2f} mm" for name, value in bands.items()))
