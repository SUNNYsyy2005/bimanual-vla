"""Is the 0.5-3 Hz target wobble inside chunks or created by switching chunks?

Usage: python -I wobble_source.py RUN_DIR [RUN_DIR ...]

For each still window (jitter_bands.py), compares the 0.5-3 Hz band RMS of
(a) the executed head-target sequence, which switches chunk every ~0.5 s, and
(b) each chunk's own rows (decoded with the client's smoothing) over the
same span, i.e. what execution would be without switching. Also reports how
far consecutive chunks disagree about the same future instant.
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimanual_vla.deployment.client import (  # noqa: E402
    DEFAULT_EEF_CHUNK_SMOOTHING_STEPS, decode_action_queue, validate_policy_metadata)
from jitter_bands import windows  # noqa: E402

SOS = butter(2, (.5, 3.), 'bandpass', fs=30., output='sos')


def wobble(x):
    return 1000 * np.sqrt(np.mean(np.sum(sosfiltfilt(SOS, detrend(x, axis=0), axis=0, padlen=min(len(x) - 1, 15))**2, axis=1)))


summary = []
for path in sys.argv[1:]:
    run = Path(path)
    records = [r for r in (json.loads(line) for line in open(run/'model_commands.jsonl')) if r.get('accepted')]
    protocol = validate_policy_metadata(dict(records[0]['protocol'], transport='openpi_websocket_v1'), 'both', 'bimanual')
    chunks, start = {}, {}
    for record in records:
        z = np.load(run/record['action_file'])
        _, decoded = decode_action_queue(z['raw_actions'], protocol, z['raw_delivery_state'], z['qpos'],
                                         observation_capture_monotonic=record['captured_monotonic'],
                                         eef_smoothing_window=DEFAULT_EEF_CHUNK_SMOOTHING_STEPS)
        chunks[record['generation']] = (np.array([d.target_monotonic for d in decoded]),
                                        np.array([d.absolute_target for d in decoded]))
    rows = [json.loads(line) for line in open(run/'trajectory.jsonl')]
    t0 = rows[0]['monotonic_timestamp']
    for side, a, b, tt, tg in windows(run):
        sel = [r for r in rows if a <= r['monotonic_timestamp'] - t0 <= b and r['command_generation'] in chunks]
        if len(sel) < 30:
            continue
        executed = np.array([r['command_absolute_target'][side*10:side*10+3] for r in sel])
        # Each chunk alone over the window span (rows within [a, b] on its own timeline).
        own, disagreement = [], []
        gens = sorted({r['command_generation'] for r in sel})
        for g in gens:
            times, targets = chunks[g]
            m = (times - t0 >= a) & (times - t0 <= b)
            if m.sum() >= 30:
                own.append(wobble(targets[m][:, side*10:side*10+3]))
        for g_old, g_new in zip(gens, gens[1:]):
            to, xo = chunks[g_old]; tn, xn = chunks[g_new]
            common = np.intersect1d(np.round(to, 4), np.round(tn, 4))
            if len(common):
                io = np.isin(np.round(to, 4), common); inn = np.isin(np.round(tn, 4), common)
                disagreement.append(np.median(np.linalg.norm(xo[io][:, side*10:side*10+3] - xn[inn][:, side*10:side*10+3], axis=1)))
        if own:
            summary.append((wobble(executed), np.median(own), np.median(disagreement) if disagreement else np.nan))
            print(f"{run.name[:15]} {'LR'[side]} {a:5.1f}-{b:5.1f}s: executed wobble {summary[-1][0]:4.2f} mm | single chunk {summary[-1][1]:4.2f} mm | "
                  f"consecutive chunks disagree {summary[-1][2]:4.1f} mm (median over overlap)")
s = np.array(summary)
print(f"\nmedian: executed {np.median(s[:,0]):.2f} mm, single chunk {np.median(s[:,1]):.2f} mm, chunk disagreement {np.nanmedian(s[:,2]):.1f} mm")
