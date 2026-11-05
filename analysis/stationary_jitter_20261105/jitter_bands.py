"""Frequency bands of EEF motion in still windows, layer by layer.

Usage: python -I jitter_bands.py RUN_DIR [RUN_DIR ...]

Still windows as in jitter_layers.py. Every layer is mapped to EEF position
by forward kinematics, resampled at 100 Hz (policy targets are held between
control ticks), linearly detrended and split with zero-phase Butterworth
filters into drift (< 0.5 Hz), wobble (0.5-3 Hz) and jitter (3-15 Hz). Also
reports how often the measured EEF reverses direction along its main axis.
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jitter_layers import ARM, pose_series, smooth  # noqa: E402

HZ = 100.
BANDS = {'drift <0.5Hz': (None, .5), 'wobble 0.5-3Hz': (.5, 3.), 'jitter 3-15Hz': (3., 15.)}


def band_rms(x):
    x = detrend(x, axis=0)
    out = {}
    for name, (lo, hi) in BANDS.items():
        sos = (butter(2, hi, 'lowpass', fs=HZ, output='sos') if lo is None else
               butter(2, (lo, hi), 'bandpass', fs=HZ, output='sos'))
        y = sosfiltfilt(sos, x, axis=0)
        out[name] = 1000 * np.sqrt(np.mean(np.sum(y**2, axis=1)))
    return out


def windows(run):
    rows = [json.loads(line) for line in open(run/'trajectory.jsonl')]
    t0 = rows[0]['monotonic_timestamp']
    rt = np.array([r['monotonic_timestamp'] - t0 for r in rows])
    ok = np.array([r['execution_state'] == 'executing' and r['command_absolute_target'][0] is not None for r in rows])
    state = np.array([r['delivery_state'] for r in rows])
    target = np.array([r['command_absolute_target'] if r['command_absolute_target'][0] is not None else [np.nan]*20
                       for r in rows], float)
    for side in range(2):
        xyz = state[:, side*10:side*10+3]
        speed = np.r_[0, np.linalg.norm(np.diff(smooth(xyz, 15), axis=0), axis=1) / np.maximum(np.diff(rt), 1e-3)]
        still = ok & (speed < .015)
        start = None
        for i in range(len(rows) + 1):
            if i < len(rows) and still[i]:
                start = i if start is None else start
                continue
            if start is not None and rt[i-1] - rt[start] >= 1.5:
                yield side, rt[start], rt[i-1], rt[start:i], target[start:i, side*10:side*10+3]
            start = None


def main():
    totals = {}
    for path in sys.argv[1:]:
        run = Path(path)
        servo = [json.loads(line) for line in open(run/'servo_commands.jsonl')]
        t0 = json.loads(open(run/'trajectory.jsonl').readline())['monotonic_timestamp']
        st = np.array([s['monotonic_timestamp'] - t0 for s in servo])
        ref = np.array([[s['reference'][i] for i in ARM] for s in servo])
        cmd = np.array([s['position'] for s in servo])
        fb = np.array([[s['feedback_qpos'][i] for i in ARM] for s in servo])
        for side, a, b, tt, tg in windows(run):
            m = (st >= a) & (st <= b)
            grid = st[m]
            held = tg[np.clip(np.searchsorted(tt, grid, side='right') - 1, 0, len(tt) - 1)]
            layers = {'target': held}
            for name, joints in (('ik_ref', ref), ('ruckig', cmd), ('measured', fb)):
                layers[name] = pose_series(joints[m][:, side*6:side*6+6])[0]
            bands = {name: band_rms(x) for name, x in layers.items()}
            meas = detrend(layers['measured'], axis=0)
            axis = np.linalg.svd(meas - meas.mean(0), full_matrices=False)[2][0]
            v = np.diff(sosfiltfilt(butter(2, 15, 'lowpass', fs=HZ, output='sos'), meas @ axis))
            reversals = np.sum(np.diff(np.sign(v[np.abs(v) > 1e-6])) != 0) / (b - a)
            print(f"{run.name[:15]} {'LR'[side]} {a:5.1f}-{b:5.1f}s reversals {reversals:4.1f}/s | " + " | ".join(
                f"{band}: " + " ".join(f"{bands[l][band]:4.2f}" for l in layers) for band in BANDS))
            for l in layers:
                for band in BANDS:
                    totals.setdefault((l, band), []).append(bands[l][band])
    print("\nmedian over windows (mm RMS)  layers: target / IK reference / Ruckig command / measured")
    for band in BANDS:
        print(f"  {band:15s}: " + " / ".join(f"{np.median(totals[(l, band)]):4.2f}" for l in ('target', 'ik_ref', 'ruckig', 'measured')))


if __name__ == '__main__':
    main()
