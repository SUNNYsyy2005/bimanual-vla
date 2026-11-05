"""Where does end-effector jitter enter while the arm should be still?

Usage: python -I jitter_layers.py RUN_DIR [RUN_DIR ...]

Stationary windows: executing rows where the measured EEF moves < 1.5 cm/s
(0.5 s smoothing) for at least 1 s. In each window, the high-pass residual
(signal minus 0.4 s centred moving average) is measured for every layer:
policy head target -> IK reference -> Ruckig command -> measured feedback,
all mapped to EEF position/rotation with the same forward kinematics.
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bimanual_vla.deployment.client import PiperContinuousIK, rotation_from_state  # noqa: E402

IK = PiperContinuousIK()
HZ = 100.
ARM = [i for i in range(14) if i % 7 != 6]


def smooth(x, n):
    kernel = np.ones(n) / n
    pad = np.pad(x, ((n // 2, n - 1 - n // 2), (0, 0)), mode='edge')
    return np.stack([np.convolve(pad[:, k], kernel, mode='valid') for k in range(x.shape[1])], 1)


def pose_series(joints):
    xyz, rot = [], []
    for q in joints:
        p, r = IK.pose(q)
        xyz.append(p); rot.append(r)
    return np.array(xyz), np.array(rot)


def residual(xyz, rot, n):
    """High-pass position (m) and rotation (rad) magnitude per sample."""
    pos = np.linalg.norm(xyz - smooth(xyz, n), axis=1)
    rv = Rotation.from_matrix(rot).as_rotvec()
    rv = np.unwrap(rv, axis=0)
    mean = Rotation.from_rotvec(smooth(rv, n))
    ang = (Rotation.from_matrix(rot) * mean.inv()).magnitude()
    return pos, ang


def analyse(run):
    rows = [json.loads(line) for line in open(run/'trajectory.jsonl')]
    servo = [json.loads(line) for line in open(run/'servo_commands.jsonl')]
    t0 = rows[0]['monotonic_timestamp']
    rt = np.array([r['monotonic_timestamp'] - t0 for r in rows])
    st = np.array([s['monotonic_timestamp'] - t0 for s in servo])
    executing = np.array([r['execution_state'] == 'executing' for r in rows])
    state = np.array([r['delivery_state'] for r in rows])
    target = np.array([[np.nan if v is None else v for v in (r['command_absolute_target'] or [None]*20)]
                       for r in rows], float)
    braking = np.array([bool(s.get('braking')) for s in servo])
    out = []
    for side in range(2):
        xyz = state[:, side*10:side*10+3]
        speed = np.r_[0, np.linalg.norm(np.diff(smooth(xyz, 15), axis=0), axis=1) / np.maximum(np.diff(rt), 1e-3)]
        still = executing & (speed < .015) & np.isfinite(target[:, side*10])
        start = None
        for i in range(len(rows) + 1):
            if i < len(rows) and still[i]:
                start = i if start is None else start
                continue
            if start is not None and rt[i-1] - rt[start] >= 1.:
                out.append((side, rt[start], rt[i-1]))
            start = None
    results = []
    for side, a, b in out:
        m = (st >= a) & (st <= b) & ~braking
        if m.sum() < 80:
            continue
        sl = slice(side*6, side*6+6)
        ref = np.array([[s['reference'][i] for i in ARM] for s, keep in zip(servo, m) if keep])[:, sl]
        cmd = np.array([s['position'] for s, keep in zip(servo, m) if keep])[:, sl]
        fb = np.array([[s['feedback_qpos'][i] for i in ARM] for s, keep in zip(servo, m) if keep])[:, sl]
        layers = {}
        for name, joints in (('ik_reference', ref), ('ruckig_cmd', cmd), ('measured', fb)):
            xyz, rot = pose_series(joints)
            layers[name] = residual(xyz, rot, 40) + (joints - smooth(joints, 40),)
        rm = (rt >= a) & (rt <= b) & np.isfinite(target[:, side*10])
        tg = target[rm][:, side*10:side*10+10]
        trot = np.array([rotation_from_state(x.copy()) for x in tg])
        tpos, tang = residual(tg[:, :3], trot, 12)
        # chunk boundary steps of the head target inside the window
        gens = np.array([r['command_generation'] for r, keep in zip(rows, rm) if keep])
        steps = np.linalg.norm(np.diff(tg[:, :3], axis=0), axis=1)
        switch = gens[1:] != gens[:-1]
        fbj = layers['measured'][2]
        freq = np.fft.rfftfreq(len(fbj), 1 / HZ)
        spec = np.abs(np.fft.rfft(layers['measured'][0] - layers['measured'][0].mean()))
        band = (freq > .5)
        results.append(dict(
            side='LR'[side], start=a, end=b,
            target_mm=1000*np.sqrt(np.mean(tpos**2)), target_deg=np.degrees(np.sqrt(np.mean(tang**2))),
            ref_mm=1000*np.sqrt(np.mean(layers['ik_reference'][0]**2)), ref_deg=np.degrees(np.sqrt(np.mean(layers['ik_reference'][1]**2))),
            cmd_mm=1000*np.sqrt(np.mean(layers['ruckig_cmd'][0]**2)), cmd_deg=np.degrees(np.sqrt(np.mean(layers['ruckig_cmd'][1]**2))),
            meas_mm=1000*np.sqrt(np.mean(layers['measured'][0]**2)), meas_deg=np.degrees(np.sqrt(np.mean(layers['measured'][1]**2))),
            joint_deg=np.degrees(np.sqrt(np.mean(fbj**2, axis=0))),
            step_in_mm=1000*np.median(steps[~switch]) if (~switch).any() else np.nan,
            step_switch_mm=1000*np.median(steps[switch]) if switch.any() else np.nan,
            peak_hz=freq[band][np.argmax(spec[band])] if band.any() else np.nan))
    return results


if __name__ == '__main__':
  for path in sys.argv[1:]:
    run = Path(path)
    for r in analyse(run):
        print(f"{run.name[:15]} {r['side']} {r['start']:6.1f}-{r['end']:6.1f}s | target {r['target_mm']:4.1f}mm {r['target_deg']:4.2f}° | "
              f"IKref {r['ref_mm']:4.1f}mm {r['ref_deg']:4.2f}° | cmd {r['cmd_mm']:4.1f}mm {r['cmd_deg']:4.2f}° | "
              f"meas {r['meas_mm']:4.1f}mm {r['meas_deg']:4.2f}° peak {r['peak_hz']:4.1f}Hz | "
              f"step in-chunk {r['step_in_mm']:4.1f} switch {r['step_switch_mm']:4.1f}mm | joint RMS° {np.round(r['joint_deg'],2).tolist()}")
