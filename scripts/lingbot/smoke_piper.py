"""Synthetic websocket check. Does not connect to CAN or command a robot."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bimanual_vla.deployment.lingbot_piper import CAMERAS, DEFAULT_TASK
from bimanual_vla.deployment.client import validate_policy_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.101.9')
    parser.add_argument('--port', type=int, default=8101)
    parser.add_argument('--task', default=DEFAULT_TASK)
    args = parser.parse_args()
    policy = WebsocketClientPolicy(args.host, args.port)
    meta = policy.get_server_metadata()
    validate_policy_metadata(meta, 'both', 'bimanual', 'joint')
    if meta.get('policy_family') != 'lingbot_piper' or meta.get('rtc_supported') is not False:
        raise RuntimeError('Wrong policy family or RTC setting')
    print(json.dumps({'checkpoint_source': meta.get('checkpoint_source'),
                      'rtc_supported': meta['rtc_supported']}, ensure_ascii=False), flush=True)
    obs = {'state': np.array([0, .5, -.5, 0, 0, 0, .5] * 2, np.float32),
           'images': {key: np.zeros((3, 256, 256), np.uint8) for key in CAMERAS},
           'prompt': args.task, 'client_metadata': {'rtc': {'enabled': False}}}
    for i in range(2):
        start = time.monotonic()
        result = policy.infer(obs)
        actions = np.asarray(result['actions'])
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise RuntimeError(f'Invalid actions: {actions.shape}')
        print(json.dumps({'request': i + 1, 'latency_s': time.monotonic() - start,
                          'action_shape': list(actions.shape), 'finite': True,
                          'gripper_range': [float(actions[:, [6, 13]].min()), float(actions[:, [6, 13]].max())]}), flush=True)
        obs.pop('prompt', None)
    print('PASS: transport and model inference only; task success and real hardware motion are untested.')


if __name__ == '__main__':
    main()
