"""Pick the chunk smoothing filter: jitter removed vs demonstrated motion moved.

Usage: python -I smoothing_choice.py DATASET_DIR RUN_DIR [RUN_DIR ...]

Demonstration distortion = |smoothed - original| on 50-row demo windows (rows
after the RTC delay, as executed), overall p99/max and the maximum during slow
motion (< 5 cm/s), where fine positioning happens. Policy jitter = residual of
the smoothed chunk from its own 7-row centred average (as in policy_jitter.py).
"""
import glob
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import savgol_filter

DELAY = 10


def jitter(pos):
    k = np.ones(7) / 7
    pad = np.pad(pos, ((3, 3), (0, 0)), mode='edge')
    smooth = np.stack([np.convolve(pad[:, j], k, mode='valid') for j in range(3)], 1)
    return np.linalg.norm(pos - smooth, axis=1)[3:-3]


demo = []
for f in sorted(glob.glob(str(Path(sys.argv[1])/'*.npz'))):
    s = np.asarray(np.load(f)['state'], float)
    for side in range(2):
        xyz = s[:, side*10:side*10+3]
        demo.extend(xyz[k:k+50] for k in range(0, len(xyz) - 50, 10))
policy = []
for path in sys.argv[2:]:
    run = Path(path)
    for line in open(run/'model_commands.jsonl'):
        record = json.loads(line)
        if record.get('accepted'):
            actions = np.asarray(np.load(run/record['action_file'])['raw_actions'], float)
            policy.extend(actions[:, side*7:side*7+3] for side in range(2))
print(f"{len(demo)} demonstration windows, {len(policy)} policy chunks")
print("baseline policy jitter %.2f mm" % (np.sqrt(np.mean(np.concatenate([jitter(c)[DELAY:] for c in policy])**2))*1000))
for window, order in ((5, 2), (7, 2), (7, 3), (9, 2), (9, 3), (11, 3), (13, 3), (11, 4), (13, 4)):
    shift, slow = [], []
    for d in demo:
        moved = np.linalg.norm(savgol_filter(d, window, order, axis=0, mode='interp') - d, axis=1)[DELAY:-5]
        speed = (np.linalg.norm(np.diff(d, axis=0), axis=1) * 30)[DELAY:-4]
        shift.append(moved); slow.append(moved[speed < .05])
    shift = np.concatenate(shift); slow = np.concatenate(slow)
    pj = np.concatenate([jitter(savgol_filter(c, window, order, axis=0, mode='interp'))[DELAY:] for c in policy])
    print(f"SG {window:2d}/{order}: policy jitter {np.sqrt(np.mean(pj**2))*1000:4.2f} mm | demo moved p99 {np.percentile(shift, 99)*1000:4.2f} "
          f"p99.9 {np.percentile(shift, 99.9)*1000:4.2f} max {shift.max()*1000:4.2f} mm | slow (<5 cm/s) max {slow.max()*1000:4.2f} p99.9 {np.percentile(slow, 99.9)*1000:4.2f} mm")

print("\nsettled = every row within the filter window moves < 2 cm/s (the arm is at its fine position)")
for window in (5, 7, 9):
    settled, transient = [], []
    half = window // 2
    for d in demo:
        moved = np.linalg.norm(savgol_filter(d, window, 2, axis=0, mode='interp') - d, axis=1)
        speed = np.r_[np.linalg.norm(np.diff(d, axis=0), axis=1) * 30, 0]
        for i in range(DELAY, len(d) - 5):
            block = speed[max(0, i - half):i + half + 1]
            (settled if block.max() < .02 else transient if speed[i] < .05 else []).append(moved[i])
    settled = np.array(settled); transient = np.array(transient)
    print(f"SG {window}/2: settled max {settled.max()*1000:4.2f} p99.9 {np.percentile(settled, 99.9)*1000:4.2f} mm (n={len(settled)}) | "
          f"slow rows next to faster motion: max {transient.max()*1000:4.2f} p99 {np.percentile(transient, 99)*1000:4.2f} mm")
