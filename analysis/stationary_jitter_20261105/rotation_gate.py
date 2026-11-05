"""Rotation side of the stillness-gated smoother (position gate 31 rows, 1 cm/s).

Usage: python -I rotation_gate.py DATASET_DIR RUN_DIR [RUN_DIR ...]

Stillness requires both the position speed and the angular-speed estimate
(largest rate of either rotation-6D column) to stay low over the whole window.
Reports, per angular threshold, demonstration rotation change (settled / all)
and position change, and the 0.5-3 Hz rotation/position wobble of still chunks.
"""
import glob
import json
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, detrend, savgol_filter, sosfiltfilt
from scipy.spatial.transform import Rotation

DT = 1 / 30
SOS = butter(2, (.5, 3.), 'bandpass', fs=30., output='sos')
WINDOW, V = 31, .01


def to_rot(r6):
    c0 = r6[:, :3] / np.linalg.norm(r6[:, :3], axis=1, keepdims=True)
    c1 = r6[:, 3:] - c0 * np.sum(c0 * r6[:, 3:], axis=1, keepdims=True)
    c1 /= np.linalg.norm(c1, axis=1, keepdims=True)
    return np.stack([c0, c1, np.cross(c0, c1)], 2)


def gated(block, w_rot):
    """block: [T, 9] xyz + rotation-6D."""
    base = savgol_filter(block, 7, 2, axis=0, mode='interp')
    strong = savgol_filter(block, WINDOW, 2, axis=0, mode='interp')
    rate = savgol_filter(block, WINDOW, 2, deriv=1, delta=DT, axis=0, mode='interp')
    half = WINDOW // 2
    worst = lambda x: sliding_window_view(np.pad(x, half, mode='edge'), WINDOW).max(axis=1)
    wp = np.clip((2 * V - worst(np.linalg.norm(rate[:, :3], axis=1))) / V, 0, 1)
    ang = np.maximum(np.linalg.norm(rate[:, 3:6], axis=1), np.linalg.norm(rate[:, 6:9], axis=1))
    wr = np.clip((2 * w_rot - worst(ang)) / w_rot, 0, 1) if w_rot else np.ones_like(wp)
    w = np.minimum(wp, wr)[:, None]
    return w * strong + (1 - w) * base


def wobble_rot(block):
    R = Rotation.from_matrix(to_rot(block[:, 3:9]))
    rv = (R * R.mean().inv()).as_rotvec()
    return np.degrees(np.sqrt(np.mean(np.sum(sosfiltfilt(SOS, detrend(rv, axis=0), axis=0, padlen=10)**2, axis=1))))


def wobble_pos(block):
    return 1000 * np.sqrt(np.mean(np.sum(sosfiltfilt(SOS, detrend(block[:, :3], axis=0), axis=0, padlen=10)**2, axis=1)))


demo = []
for f in sorted(glob.glob(str(Path(sys.argv[1])/'*.npz'))):
    s = np.asarray(np.load(f)['state'], float)
    for side in range(2):
        blk = s[:, side*10:side*10+9]
        demo.extend(blk[k:k+50] for k in range(0, len(blk) - 50, 10))
still = []
for path in sys.argv[2:]:
    run = Path(path)
    for line in open(run/'model_commands.jsonl'):
        record = json.loads(line)
        if record.get('accepted'):
            actions = np.asarray(np.load(run/record['action_file'])['raw_actions'], float)
            for side in range(2):
                pos = actions[:, side*7:side*7+3]
                rot = Rotation.from_rotvec(actions[:, side*7+3:side*7+6]).as_matrix()
                if np.linalg.norm(pos[30] - pos[10]) < .01:
                    still.append(np.c_[pos, rot[:, :, 0], rot[:, :, 1]])
print(f"{len(demo)} demo windows, {len(still)} still policy chunks")
raw_rot = np.median([wobble_rot(c[10:31]) for c in still]); raw_pos = np.median([wobble_pos(c[10:31]) for c in still])
print(f"raw policy wobble: {raw_pos:.2f} mm / {raw_rot:.2f} deg")
for w_rot in (None, .03, .05, .1, .2):
    ang_all, ang_settled, pos_all = [], [], []
    for d in demo:
        out = gated(d, w_rot)
        Ro, Rs = to_rot(d[:, 3:9]), to_rot(out[:, 3:9])
        ang = np.degrees(Rotation.from_matrix(np.einsum('nij,nkj->nik', Rs, Ro)).magnitude())[10:-5]
        pos = np.linalg.norm(out[:, :3] - d[:, :3], axis=1)[10:-5]
        speed = np.r_[np.linalg.norm(np.diff(d[:, :3], axis=0), axis=1) / DT, 0][10:-5]
        ang_all.append(ang); pos_all.append(pos); ang_settled.append(ang[speed < .005])
    ang_all = np.concatenate(ang_all); pos_all = np.concatenate(pos_all); ang_settled = np.concatenate(ang_settled)
    outs = [gated(c, w_rot)[10:31] for c in still]
    print(f"angular gate {'none' if w_rot is None else f'{w_rot:.2f} rad/s'}: still wobble {np.median([wobble_pos(o) for o in outs]):.2f} mm / "
          f"{np.median([wobble_rot(o) for o in outs]):.2f} deg | demo change: position max {pos_all.max()*1000:.2f} mm, "
          f"rotation max {ang_all.max():.2f} deg (p99.9 {np.percentile(ang_all, 99.9):.2f}), settled rotation max {ang_settled.max():.2f} deg")
