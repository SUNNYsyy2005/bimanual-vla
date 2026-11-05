"""Check the client's smooth_delivery_targets against the offline numbers.

Usage: python -I verify_client_smoothing.py DATASET_DIR RUN_DIR [RUN_DIR ...]
"""
import glob
import json
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, detrend, sosfiltfilt
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bimanual_vla.data.action_conventions import matrix_to_rotation6d  # noqa: E402
from bimanual_vla.deployment.client import (  # noqa: E402
    DEFAULT_EEF_CHUNK_SMOOTHING_STEPS as SHORT, DEFAULT_EEF_STILL_SMOOTHING_STEPS as LONG, rotation_from_state,
    smooth_delivery_targets)

SOS = butter(2, (.5, 3.), 'bandpass', fs=30., output='sos')
CONFIGS = {'off': dict(window=0), 'short only': dict(window=SHORT),
           'short + still (default)': dict(window=SHORT, still_window=LONG)}


def run(block, cfg):
    return smooth_delivery_targets(block, 1, cfg['window'], still_window=cfg.get('still_window', 0), action_hz=30.)


def rotations(block):
    return Rotation.from_matrix(np.array([rotation_from_state(r.copy()) for r in block]))


demo = []
for f in sorted(glob.glob(str(Path(sys.argv[1])/'*.npz'))):
    s = np.asarray(np.load(f)['state'], float)
    for side in range(2):
        blk = s[:, side*10:side*10+10]
        demo.extend(blk[k:k+50] for k in range(0, len(blk) - 50, 10))
still = []
for path in sys.argv[2:]:
    for line in open(Path(path)/'model_commands.jsonl'):
        record = json.loads(line)
        if record.get('accepted'):
            a = np.asarray(np.load(Path(path)/record['action_file'])['raw_actions'], float)
            for side in range(2):
                pos = a[:, side*7:side*7+3] + [.3, 0, .3]
                if np.linalg.norm(pos[30] - pos[10]) < .01:
                    r6 = matrix_to_rotation6d(Rotation.from_rotvec(a[:, side*7+3:side*7+6]).as_matrix())
                    still.append(np.c_[pos, r6, a[:, side*7+6]])
for name, cfg in CONFIGS.items():
    pos_max = settled = near = 0.; ang = []
    for d in demo:
        out = run(d, cfg)
        moved = np.linalg.norm(out[:, :3] - d[:, :3], axis=1)[10:-5]
        speed = np.r_[np.linalg.norm(np.diff(d[:, :3], axis=0), axis=1) * 30, 0]
        lm = sliding_window_view(np.pad(speed, 3, mode='edge'), 7).max(axis=1)[10:-5]
        pos_max = max(pos_max, moved.max())
        if (lm < .02).any(): settled = max(settled, moved[lm < .02].max())
        sel = (lm >= .02) & (speed[10:-5] < .05)
        if sel.any(): near = max(near, moved[sel].max())
        ang.append(np.degrees((rotations(out) * rotations(d).inv()).magnitude())[10:-5])
    ang = np.concatenate(ang)
    wp, wr = [], []
    for c in still:
        out = run(c, cfg)[10:31]
        wp.append(1000 * np.sqrt(np.mean(np.sum(sosfiltfilt(SOS, detrend(out[:, :3], axis=0), axis=0, padlen=10)**2, axis=1))))
        R = rotations(out); rv = (R * R.mean().inv()).as_rotvec()
        wr.append(np.degrees(np.sqrt(np.mean(np.sum(sosfiltfilt(SOS, detrend(rv, axis=0), axis=0, padlen=10)**2, axis=1)))))
    print(f"{name:24s}: still wobble {np.median(wp):4.2f} mm / {np.median(wr):4.2f} deg | demo moved max settled {settled*1000:4.2f} "
          f"near stop {near*1000:4.2f} all {pos_max*1000:4.2f} mm, rotation p99.9 {np.percentile(ang, 99.9):4.2f} max {ang.max():4.2f} deg")
