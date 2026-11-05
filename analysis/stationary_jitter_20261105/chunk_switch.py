"""Target discontinuity when execution switches to a new chunk.

Usage: python -I chunk_switch.py RUN_DIR [RUN_DIR ...]

At each executed generation change, compares the new chunk's first executed
target with the old chunk's target for the same time (its next row), both
re-decoded with the client's default chunk smoothing. Split by whether the
arm was still (measured EEF < 1.5 cm/s). For scale, also the in-chunk
step between consecutive executed rows.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bimanual_vla.deployment.client import (  # noqa: E402
    DEFAULT_EEF_CHUNK_SMOOTHING_STEPS, decode_action_queue, validate_policy_metadata)

for path in sys.argv[1:]:
    run = Path(path)
    records = [json.loads(line) for line in open(run/'model_commands.jsonl')]
    records = [r for r in records if r.get('accepted')]
    protocol = validate_policy_metadata(dict(records[0]['protocol'], transport='openpi_websocket_v1'), 'both', 'bimanual')
    chunks = {}
    for record in records:
        z = np.load(run/record['action_file'])
        _, decoded = decode_action_queue(z['raw_actions'], protocol, z['raw_delivery_state'], z['qpos'],
                                         observation_capture_monotonic=0.,
                                         eef_smoothing_window=DEFAULT_EEF_CHUNK_SMOOTHING_STEPS)
        chunks[record['generation']] = np.array([d.absolute_target for d in decoded])
    rows = [json.loads(line) for line in open(run/'trajectory.jsonl')]
    executed = [r for r in rows if r['execution_state'] == 'executing' and r['command_generation'] in chunks]
    switch = {'still': [], 'moving': []}; inner = {'still': [], 'moving': []}
    for a, b in zip(executed, executed[1:]):
        if b['monotonic_timestamp'] - a['monotonic_timestamp'] > .1:
            continue
        for side in range(2):
            speed = np.linalg.norm(np.array(b['delivery_state'][side*10:side*10+3]) - a['delivery_state'][side*10:side*10+3]) / \
                max(b['monotonic_timestamp'] - a['monotonic_timestamp'], 1e-3)
            kind = 'still' if speed < .015 else 'moving'
            old, new = chunks[a['command_generation']], chunks[b['command_generation']]
            target_b = new[min(b['command_queue_index'], len(new) - 1), side*10:side*10+3]
            if a['command_generation'] == b['command_generation']:
                target_a = old[min(a['command_queue_index'], len(old) - 1), side*10:side*10+3]
                inner[kind].append(np.linalg.norm(target_b - target_a))
            else:
                same_time = old[min(a['command_queue_index'] + 1, len(old) - 1), side*10:side*10+3]
                switch[kind].append(np.linalg.norm(target_b - same_time))
    blend = json.loads((run/'metadata.json').read_text()).get('runtime_configuration', {}).get('blend_steps')
    text = " | ".join(f"{k}: switch jump p50 {np.median(switch[k])*1000:4.1f} p90 {np.percentile(switch[k], 90)*1000:4.1f} mm (n={len(switch[k])}), "
                      f"in-chunk step p50 {np.median(inner[k])*1000:4.1f} mm" for k in ('still', 'moving') if switch[k])
    print(f"{run.name[:15]} blend={blend} {text}")
