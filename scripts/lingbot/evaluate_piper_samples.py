"""Read-only evaluation on prepared heldout Piper samples; never opens CAN."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--host', default='192.168.101.9')
    p.add_argument('--port', type=int, default=8101)
    p.add_argument('--lingbot-root', type=Path)
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    data = np.load(args.samples, allow_pickle=False)
    sample_info = json.loads(args.samples.with_suffix('.json').read_text())
    cameras = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
    if args.checkpoint:
        if not args.lingbot_root:
            p.error('--checkpoint requires --lingbot-root')
        sys.path.insert(0, str(args.lingbot_root.resolve()))
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
        policy = LingbotVLAv2Server(str(args.checkpoint.absolute()),
                                   robot_norm_path=str(args.checkpoint / 'piper_bimanual_norm.json'),
                                   use_length=50, chunk_ret=True, use_bf16=True, use_compile=False)
        policy.vla.model.flexpi_params['num_steps'] = 4
        policy.infer({'reset': True, 'robo_name': 'piper_bimanual'})
        source = json.loads((args.checkpoint / 'export_source.json').read_text())
        def infer(obs):
            native = {'observation.state': obs['state'], 'task': obs['prompt']}
            native.update({f'observation.images.{k}': obs['images'][k].transpose(1, 2, 0).copy() for k in cameras})
            return policy.infer(native)['action']
    else:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        policy = WebsocketClientPolicy(args.host, args.port)
        meta = policy.get_server_metadata()
        if meta.get('policy_family') != 'lingbot_piper' or meta.get('rtc_supported') is not False:
            raise RuntimeError('Expected LingBot Piper with RTC disabled')
        source = meta['checkpoint_source']
        def infer(obs):
            return policy.infer(obs)['actions']
    predictions, latencies = [], []
    for i, sample in enumerate(sample_info['samples']):
        obs = {'state': data['state'][i], 'images': {k: data[k][i] for k in cameras},
               'prompt': sample['task'], 'image_format': 'piper_training_rgb256',
               'client_metadata': {'rtc': {'enabled': False}}}
        start = time.monotonic()
        actions = np.asarray(infer(obs), np.float32)
        latency = time.monotonic() - start
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise RuntimeError(f'Invalid action chunk {actions.shape}')
        predictions.append(actions); latencies.append(latency)
        print(json.dumps({'sample': i, **sample, 'latency_s': latency}), flush=True)
    predicted = np.stack(predictions)
    target = data['actions']
    joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    limits = np.deg2rad(np.tile([[-150,150],[0,180],[-170,0],[-100,100],[-70,70],[-180,180]],(2,1)))
    lo, hi = limits[:, 0], limits[:, 1]
    report = {'checkpoint_source': source, 'split': sample_info['split'], 'samples': sample_info['samples'],
              'sample_count': len(predictions), 'rtc_enabled': False, 'action_shape': list(predicted.shape),
              'cold_latency_s': latencies[0], 'warm_latency_p50_s': float(np.median(latencies[1:])),
              'warm_latency_p95_s': float(np.percentile(latencies[1:],95)),
              'predicted_joint_limit_violation_values': int(((predicted[:,:,joints] < lo) | (predicted[:,:,joints] > hi)).sum()),
              'target_joint_limit_violation_values': int(((target[:,:,joints] < lo) | (target[:,:,joints] > hi)).sum()),
              'predicted_gripper_range': [float(predicted[:,:,[6,13]].min()),float(predicted[:,:,[6,13]].max())],
              'limitation': 'Nine heldout observations only; no closed-loop rollout or real hardware execution.'}
    for n in (1,25,50):
        report[f'joint_mae_deg_first_{n}'] = float(np.rad2deg(np.abs(predicted[:,:n,joints]-target[:,:n,joints])).mean())
        report[f'hold_baseline_joint_mae_deg_first_{n}'] = float(np.rad2deg(np.abs(data['state'][:,None,joints]-target[:,:n,joints])).mean())
        report[f'gripper_mae_first_{n}'] = float(np.abs(predicted[:,:n,[6,13]]-target[:,:n,[6,13]]).mean())
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.report.with_suffix('.npz'), predicted=predicted, target=target, state=data['state'], latency_s=latencies)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
