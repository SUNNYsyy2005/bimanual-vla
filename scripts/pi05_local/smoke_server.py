"""Validate local policy RPCs on dataset images, without connecting hardware."""
import argparse
import json
from pathlib import Path
import time
import uuid

import numpy as np

from common import configure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='/home/user/src/openpi-5060')
    p.add_argument('--dataset', default='/home/user/datasets/lerobot/fix_nopad_trimstart_20261021')
    p.add_argument('--port', type=int, default=8099)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--output', default='analysis/pi05_local_8099/smoke_test.json')
    args = p.parse_args()
    configure(args.source)
    from evaluate import Frames, CAMERAS
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy('127.0.0.1', args.port)
    metadata = client.get_server_metadata()
    assert metadata['checkpoint_step'] == 15000 and metadata['rtc_mode'] == 'trained'
    assert metadata['wire_action_convention'] == 'absolute_joint_target'
    frames = Frames(args.dataset, [12], 2, 42)
    session = 'local-smoke-' + uuid.uuid4().hex
    results = []

    def request(index, generation, enabled=False, delay=0, **extra):
        state, _, images = frames.get(index)
        rtc = dict(session_id=session, inference_generation=generation,
                   inference_delay_steps=delay, config_revision=generation,
                   config=dict(enabled=enabled, execution_horizon=12), **extra)
        payload = dict(state=state,
            images={k: images[v].transpose(2, 0, 1) for k, v in CAMERAS.items()},
            prompt='put the bottle on the balck pad', client_metadata=dict(rtc=rtc))
        start = time.perf_counter()
        result = client.infer(payload)
        elapsed = time.perf_counter() - start
        actions = result['actions']
        assert actions.shape == (50, 14) and np.isfinite(actions).all()
        results.append(dict(generation=generation, wall_s=elapsed, actions_shape=list(actions.shape),
            rtc=result['rtc'], policy_timing=result.get('policy_timing'),
            server_timing=result.get('server_timing')))
        return actions

    try:
        request(0, 1)
        assert not results[-1]['rtc']['enabled']
        first = request(0, 2, enabled=True)
        second = request(1, 3, enabled=True, delay=5,
                         previous_chunk_generation=2, previous_chunk_offset_steps=4)
        assert results[-1]['rtc']['enabled'] and results[-1]['rtc']['reanchored']
        np.testing.assert_allclose(second[:5], first[4:9], atol=2e-6, rtol=0)
        server_error = float(np.max(np.abs(second[:5] - first[4:9])))
        planned = second[:3].copy()
        third = request(0, 4, enabled=True, delay=3, prefix_source='client_planner', planned_prefix=dict(
            plan_id='local-smoke-plan', action_hz=20., sample_offsets_s=(np.arange(3) + 1) / 20., actions=planned))
        assert results[-1]['rtc']['enabled'] and results[-1]['rtc']['prefix_source'] == 'client_planner'
        np.testing.assert_allclose(third[:3], planned, atol=2e-6, rtol=0)
        for generation in range(5, 5 + args.repeats):
            request(generation % 2, generation, enabled=True, delay=5,
                    previous_chunk_generation=generation - 1, previous_chunk_offset_steps=4)
            assert results[-1]['rtc']['enabled'] and results[-1]['rtc']['reanchored']
        report = dict(success=True, endpoint=f'ws://127.0.0.1:{args.port}',
            metadata=metadata, source='canonical held-out episode 12, no hardware connection',
            server_previous_prefix_max_abs_error=server_error,
            client_planner_prefix_max_abs_error=float(np.max(np.abs(third[:3] - planned))), requests=results)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(success=True, requests=len(results),
            cold_s=results[0]['wall_s'], warm_median_s=float(np.median([r['wall_s'] for r in results[1:]])),
            prefix_max_abs_error=server_error, output=str(output)), indent=2))
    finally:
        client._ws.close()


if __name__ == '__main__':
    main()
