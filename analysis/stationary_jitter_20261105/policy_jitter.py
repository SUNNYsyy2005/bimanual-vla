"""Policy-chunk jitter per run versus denoising steps, and the demonstration baseline.

Usage: python -I policy_jitter.py DATASET_DIR RUN_DIR [RUN_DIR ...]

Jitter = per-step deviation of each 50-step chunk's EEF position/rotation from
its own centred 7-step moving average (no lag: the whole chunk is known).
Chunks are split by how far the chunk moves overall (still: < 2 cm net).
"""
import glob
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def hp(x, n=7):
    k = np.ones(n) / n
    pad = np.pad(x, ((n // 2, n // 2), (0, 0)), mode='edge')
    return x - np.stack([np.convolve(pad[:, j], k, mode='valid') for j in range(x.shape[1])], 1)


def chunk_stats(chunks):
    """chunks: list of (positions [T,3], rotvecs [T,3])."""
    still, moving = [], []
    for pos, rv in chunks:
        res = np.r_[np.linalg.norm(hp(pos), axis=1)[3:-3]]
        rot = np.r_[np.linalg.norm(hp(np.unwrap(rv, axis=0)), axis=1)[3:-3]]
        (still if np.linalg.norm(pos[-1] - pos[0]) < .02 else moving).append((res, rot))
    def summary(items):
        if not items:
            return 'n/a'
        p = np.concatenate([a for a, _ in items]); r = np.concatenate([b for _, b in items])
        return f"{np.sqrt(np.mean(p**2))*1000:4.2f} mm / {np.degrees(np.sqrt(np.mean(r**2))):4.2f} deg (n={len(items)})"
    return summary(still), summary(moving)


dataset = Path(sys.argv[1])
demo = []
for f in sorted(glob.glob(str(dataset/'*.npz')))[::3]:
    z = np.load(f)
    s = np.asarray(z['state'], float)
    for side in range(2):
        block = s[:, side*10:side*10+10]
        rot = np.stack([block[:, 3:6], block[:, 6:9]], 2)
        c0 = rot[:, :, 0] / np.linalg.norm(rot[:, :, 0], axis=1, keepdims=True)
        c1 = rot[:, :, 1] - c0 * np.sum(c0 * rot[:, :, 1], axis=1, keepdims=True)
        c1 /= np.linalg.norm(c1, axis=1, keepdims=True)
        rv = Rotation.from_matrix(np.stack([c0, c1, np.cross(c0, c1)], 2)).as_rotvec()
        for k in range(0, len(block) - 50, 50):
            demo.append((block[k:k+50, :3], rv[k:k+50]))
still, moving = chunk_stats(demo)
print(f"{'demonstrations':28s} steps  -  still {still} | moving {moving}")

for path in sys.argv[2:]:
    run = Path(path)
    steps = set()
    chunks = []
    for line in open(run/'model_commands.jsonl'):
        record = json.loads(line)
        if not record.get('accepted'):
            continue
        z = np.load(run/record['action_file'])
        actions = np.asarray(z['raw_actions'], float)
        for side in range(2):
            chunks.append((actions[:, side*7:side*7+3], actions[:, side*7+3:side*7+6]))
    # Session monitoring starts a few seconds before the run directory.
    stamp = run.name[:8] + '_' + run.name[9:15]
    sessions = [d for d in sorted(glob.glob('monitoring_data/*/')) if Path(d).name[:15] <= stamp]
    for line in (open(sessions[-1] + 'events.jsonl') if sessions else ()):
        event = json.loads(line)
        if event['event_type'] == 'inference_result':
            steps.add(((event.get('result_summary') or {}).get('rtc') or {}).get('denoising_steps'))
    still, moving = chunk_stats(chunks)
    print(f"{run.name[:15]:28s} steps {sorted(s for s in steps if s is not None)} - still {still} | moving {moving}")
