"""Sweep chunk smoothers: 0.5-3 Hz wobble removed vs demonstrated motion moved.

Usage: python -I smoothing_sweep.py DATASET_DIR RUN_DIR [RUN_DIR ...]

Variants (all zero-phase over one 50-row chunk at 30 Hz):
  sg W/O              Savitzky-Golay window W, order O on every row
  gate W v            SG(W,2) where the plan is still, SG(7,2) elsewhere; stillness
                      is the largest SG(W) speed estimate anywhere in the W-row
                      window around the row (so a stop corner inside the window
                      falls back to SG 7), blended linearly between v and 2v.
Demonstration rows: settled (all rows within +-3 move < 2 cm/s), near stop (slow
row next to faster motion), all rows. Wobble: 0.5-3 Hz band RMS of executed rows
10..30 of policy chunks that are still (< 1 cm net over rows 10..30).
"""
import glob
import json
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, detrend, savgol_filter, sosfiltfilt

DT = 1 / 30
SOS = butter(2, (.5, 3.), 'bandpass', fs=30., output='sos')


def sg(pos, window, order=2):
    return savgol_filter(pos, window, order, axis=0, mode='interp')


def gate(pos, window, v):
    base = sg(pos, 7)
    strong = sg(pos, window)
    speed = np.linalg.norm(savgol_filter(pos, window, 2, deriv=1, delta=DT, axis=0, mode='interp'), axis=1)
    half = window // 2
    worst = sliding_window_view(np.pad(speed, half, mode='edge'), window).max(axis=1)
    w = np.clip((2 * v - worst) / v, 0., 1.)[:, None]
    return w * strong + (1 - w) * base


PLAIN = [] if __import__('os').environ.get('GATES') else [(11, 2), (13, 2), (15, 2), (21, 4), (25, 4), (31, 4)]
GATES = [(int(x.split(':')[0]), float(x.split(':')[1])) for x in __import__('os').environ.get('GATES', '21:.005,21:.01,21:.02,31:.005,31:.01,31:.02').split(',')]
VARIANTS = [('sg 7/2', lambda p: sg(p, 7))] + \
    [(f'sg {w}/{o}', (lambda p, w=w, o=o: sg(p, w, o))) for w, o in PLAIN] + \
    [(f'gate {w} {v*100:.1f}cm/s', (lambda p, w=w, v=v: gate(p, w, v))) for w, v in GATES]

demo = []
for f in sorted(glob.glob(str(Path(sys.argv[1])/'*.npz'))):
    s = np.asarray(np.load(f)['state'], float)
    for side in range(2):
        xyz = s[:, side*10:side*10+3]
        demo.extend(xyz[k:k+50] for k in range(0, len(xyz) - 50, 10))
still = []
for path in sys.argv[2:]:
    run = Path(path)
    for line in open(run/'model_commands.jsonl'):
        record = json.loads(line)
        if record.get('accepted'):
            actions = np.asarray(np.load(run/record['action_file'])['raw_actions'], float)
            for side in range(2):
                pos = actions[:, side*7:side*7+3]
                if np.linalg.norm(pos[30] - pos[10]) < .01:
                    still.append(pos)
speeds = [np.r_[np.linalg.norm(np.diff(d, axis=0), axis=1) / DT, 0] for d in demo]
local_max = [sliding_window_view(np.pad(s, 3, mode='edge'), 7).max(axis=1) for s in speeds]
print(f"{len(demo)} demonstration windows, {len(still)} still policy chunks")
print(f"{'variant':20s} wobble mm | demo moved max: settled / near stop / all rows (p99.9 all)")
for name, f in VARIANTS:
    settled = near = every = 0.; tail = []
    for d, s, lm in zip(demo, speeds, local_max):
        moved = np.linalg.norm(f(d) - d, axis=1)[10:-5]
        s, lm = s[10:-5], lm[10:-5]
        every = max(every, moved.max()); tail.append(moved)
        if (lm < .02).any():
            settled = max(settled, moved[lm < .02].max())
        sel = (lm >= .02) & (s < .05)
        if sel.any():
            near = max(near, moved[sel].max())
    wob = [1000 * np.sqrt(np.mean(np.sum(sosfiltfilt(SOS, detrend(f(c)[10:31], axis=0), axis=0, padlen=10)**2, axis=1)))
           for c in still]
    print(f"{name:20s} {np.median(wob):5.2f}    | {settled*1000:5.2f} / {near*1000:5.2f} / {every*1000:5.2f} ({np.percentile(np.concatenate(tail), 99.9)*1000:4.2f})")
