"""Deadband (backlash) hold on the smoothed EEF target while the plan is still.

Usage:
  python -I deadband_hold.py demo DATASET_DIR             # deviation on demonstrations
  python -I deadband_hold.py loop RUN_DIR SIDE START END  # closed-loop band RMS

Per control tick, with the still-gate weight w of the head row (the client's
smooth_delivery_targets gate): the held pose moves only when the smoothed
target leaves a ball of HOLD_M (or HOLD_RAD in rotation), and then just far
enough to sit on its edge; the commanded target is smoothed + w*(held - smoothed).
The hold persists across chunk switches and resets when w reaches 0.
"""
from collections import deque
from dataclasses import replace
import glob
import json
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimanual_vla.data.action_conventions import matrix_to_rotation6d  # noqa: E402
from bimanual_vla.deployment.client import (  # noqa: E402
    PiperContinuousIK, decode_action_queue, project_delivery_target_step, rotation_from_state,
    smooth_delivery_targets, validate_policy_metadata)

import os
HOLD_M = float(os.environ.get('HOLD_M', .0015))
HOLD_RAD, V, W_ROT = np.radians(.5), .01, .1
LONG = int(os.environ.get('STILL', 31))
RECENTER_S = float(os.environ.get('RECENTER_S', 1.))
ALWAYS = os.environ.get('ALWAYS', '1') == '1'
V_FULL, V_OFF = float(os.environ.get('V_FULL', .03)), float(os.environ.get('V_OFF', .06))


def speed_weight(speed):
    return float(np.clip((V_OFF - speed) / (V_OFF - V_FULL), 0., 1.))


def still_weight(pose):
    """Gate weight per row of a [T, 9] pose block (as in smooth_delivery_targets)."""
    if len(pose) < LONG:
        return np.zeros(len(pose))
    rate = savgol_filter(pose, LONG, 2, deriv=1, delta=1/30, axis=0, mode='interp')
    half = LONG // 2
    worst = lambda x: sliding_window_view(np.pad(x, half, mode='edge'), LONG).max(axis=1)
    wp = np.clip((2 * V - worst(np.linalg.norm(rate[:, :3], axis=1))) / V, 0, 1)
    wr = np.clip((2 * W_ROT - worst(np.maximum(np.linalg.norm(rate[:, 3:6], axis=1),
                                                np.linalg.norm(rate[:, 6:9], axis=1)))) / W_ROT, 0, 1)
    return np.minimum(wp, wr)


class Deadband:
    def __init__(self):
        self.pos = self.rot = None

    def __call__(self, target, weight):
        rot = rotation_from_state(target.copy())
        if self.pos is None:
            self.pos, self.rot = target[:3].copy(), rot
        # Slow re-centring inside the ball removes the static offset at a stop.
        pull = min(1., (1 / 30) / RECENTER_S) if RECENTER_S > 0 else 0.
        self.pos = self.pos + pull * (target[:3] - self.pos)
        self.rot = Rotation.from_rotvec(pull * Rotation.from_matrix(rot @ self.rot.T).as_rotvec()).as_matrix() @ self.rot
        offset = target[:3] - self.pos
        distance = np.linalg.norm(offset)
        if distance > HOLD_M:
            self.pos = self.pos + offset * (1 - HOLD_M / distance)
        relative = Rotation.from_matrix(rot @ self.rot.T).as_rotvec()
        angle = np.linalg.norm(relative)
        if angle > HOLD_RAD:
            self.rot = Rotation.from_rotvec(relative * (1 - HOLD_RAD / angle)).as_matrix() @ self.rot
        out = target.copy()
        out[:3] = target[:3] + weight * (self.pos - target[:3])
        blended = Rotation.from_rotvec(weight * Rotation.from_matrix(self.rot @ rot.T).as_rotvec()).as_matrix() @ rot
        out[3:9] = matrix_to_rotation6d(blended)
        return out


def demo_check(dataset):
    worst = {'settled': 0., 'near stop': 0., 'all': 0.}; rot_worst = 0.; tail = []
    for f in sorted(glob.glob(str(Path(dataset)/'*.npz'))):
        s = np.asarray(np.load(f)['state'], float)
        for side in range(2):
            traj = s[:, side*10:side*10+10]
            hold = Deadband()
            # Stream: at row k the controller sees the last 15 rows and the chunk ahead.
            for k in range(15, len(traj) - 35, 1):
                block = traj[k - 10:k + 40]  # a 50-row chunk executed from its row 10
                smoothed = smooth_delivery_targets(block[:, :10], 1, 7, still_window=LONG, action_hz=30.)
                speed = np.linalg.norm(smoothed[11, :3] - smoothed[9, :3]) * 15
                out = hold(smoothed[10], speed_weight(speed))
                moved = np.linalg.norm(out[:3] - traj[k, :3])
                speed = np.linalg.norm(traj[min(k + 3, len(traj) - 1), :3] - traj[max(k - 3, 0), :3]) * 5
                here = np.linalg.norm(traj[k + 1, :3] - traj[k, :3]) * 30
                worst['all'] = max(worst['all'], moved); tail.append(moved)
                if speed < .02 * 6 / 6 and here < .02:
                    worst['settled'] = max(worst['settled'], moved)
                elif here < .05:
                    worst['near stop'] = max(worst['near stop'], moved)
                rot_worst = max(rot_worst, np.degrees(Rotation.from_matrix(
                    rotation_from_state(out.copy()) @ rotation_from_state(traj[k].copy()).T).magnitude()))
    print("demonstration deviation max: " + ", ".join(f"{k} {v*1000:.2f} mm" for k, v in worst.items()) +
          f"; p99.9 {np.percentile(tail, 99.9)*1000:.2f} mm; rotation max {rot_worst:.2f} deg")


if sys.argv[1] == 'demo':
    demo_check(sys.argv[2])
    sys.exit()

from jitter_bands import band_rms  # noqa: E402
from jitter_layers import pose_series  # noqa: E402
from bimanual_vla.deployment.servo import RuckigJointPlanner, ServoConfig  # noqa: E402

RUN, SIDE, START, END = Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
rc = json.loads((RUN/'metadata.json').read_text())['runtime_configuration']
limits = np.array(rc['effective_joint_limits_rad']['left' if SIDE == 0 else 'right'])
servo = replace(ServoConfig(**{k: v for k, v in rc['servo'].items() if k in ServoConfig.__dataclass_fields__}),
                independent_arms=False)
ic = rc['ik_execution']
kw = dict(best_effort=True, position_tolerance_m=ic['position_tolerance_m'], rotation_tolerance_rad=ic['rotation_tolerance_rad'],
          reference_joint_radius_rad=ic['reference_joint_radius_rad'], search_joint_radius_rad=ic['search_joint_radius_rad'],
          max_joint_step_rad=ic['search_joint_radius_rad'], joint_regularization_weight=ic['joint_regularization_weight'],
          max_nfev=100, joint_limits=limits)
records = [r for r in (json.loads(line) for line in open(RUN/'model_commands.jsonl')) if r.get('accepted')]
protocol = validate_policy_metadata(dict(records[0]['protocol'], transport='openpi_websocket_v1'), 'both', 'bimanual')
raw, smooth = {}, {}
for record in records:
    z = np.load(RUN/record['action_file'])
    _, decoded = decode_action_queue(z['raw_actions'], protocol, z['raw_delivery_state'], z['qpos'],
                                     observation_capture_monotonic=0.)
    raw[record['generation']] = np.array([d.absolute_target[SIDE*10:SIDE*10+10] for d in decoded])
    smooth[record['generation']] = smooth_delivery_targets(raw[record['generation']], 1, 7, still_window=LONG, action_hz=30.)
rows = [json.loads(line) for line in open(RUN/'trajectory.jsonl')]
t0 = rows[0]['monotonic_timestamp']
executed = [(r['monotonic_timestamp'] - t0, r['command_generation'], r['command_queue_index']) for r in rows
            if r['execution_state'] == 'executing' and r['command_generation'] in raw
            and START - 1 <= r['monotonic_timestamp'] - t0 <= END + 1]
times = np.array([e[0] for e in executed])
ik = PiperContinuousIK()
q0 = np.array(min(rows, key=lambda r: abs(r['monotonic_timestamp'] - t0 - START))['qpos'][SIDE*7:SIDE*7+6])
for variant in ('off', 'smoothed', 'smoothed + deadband'):
    planner = RuckigJointPlanner(q0, limits[:, 0], limits[:, 1], servo)
    history = deque([(START, q0.copy())]); warm = None; time_s = START; command = []
    hold = Deadband(); last = None; held_ticks = ticks = 0
    while time_s < END:
        i = int(np.clip(np.searchsorted(times, time_s, side='right') - 1, 0, len(times) - 1))
        _, generation, index = executed[i]
        chunk = raw[generation] if variant == 'off' else smooth[generation]
        goal = chunk[min(index + 5, len(chunk) - 1)].copy()
        if variant == 'smoothed + deadband':
            if last != i:
                ahead = chunk[min(index + 6, len(chunk) - 1), :3]; behind = chunk[min(index + 4, len(chunk) - 1), :3]
                held_goal = hold(goal, speed_weight(np.linalg.norm(ahead - behind) * 15))
                last = i
            goal = held_goal.copy()
        goal[9] = 0.
        delayed = [q for at, q in history if at <= time_s - .065]
        q = delayed[-1] if delayed else history[0][1]
        xyz, rotation = ik.pose(q)
        projected, _ = project_delivery_target_step(np.r_[xyz, matrix_to_rotation6d(rotation), 0.], goal,
                                                    max_translation_step_m=.1, max_rotation_step_rad=.36)
        rpy = Rotation.from_matrix(rotation_from_state(projected.copy())).as_euler('xyz', degrees=True)
        warm = ik.solve_with_diagnostics(q, projected[:3], rpy, initial_joints_rad=warm, **kw).solution_joints_rad.copy()
        planner.retarget(warm)
        for _ in range(3):
            planner.advance(); command.append(planner.position.copy()); history.append((time_s, planner.position.copy()))
            time_s += .01
    xyz, _ = pose_series(np.array(command)[100:])
    bands = band_rms(xyz)
    note = f" | fully still {held_ticks}/{ticks} ticks" if ticks else ""
    print(f"{variant:20s}: " + " | ".join(f"{name} {value:4.2f} mm" for name, value in bands.items()) + note)
