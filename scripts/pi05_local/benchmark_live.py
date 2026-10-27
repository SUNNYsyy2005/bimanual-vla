"""Read-only live camera/CAN-to-local-policy latency measurement. No arm commands."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np

from common import configure


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                minimum=float(a.min()), maximum=float(a.max()), mean=float(a.mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='/home/user/src/openpi-5060')
    p.add_argument('--left-can', default='can1')
    p.add_argument('--right-can', default='can0')
    p.add_argument('--port', type=int, default=8099)
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--output', default='analysis/pi05_local_8099/live_latency.json')
    args = p.parse_args()
    configure(args.source)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import cv2
    from piper_sdk import C_PiperInterface_V2
    from bimanual_vla.collection.camera import CameraCapture
    from bimanual_vla.deployment.client import (read_output_qpos, policy_observation_state,
                                              validate_policy_metadata)
    from bimanual_vla.deployment.transport import TimedWebsocketClientPolicy

    class MeasuredCameras(CameraCapture):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.measurements = deque(maxlen=5000)
            self.read_durations = {}
            self.measurement_lock = threading.Lock()

        def _read_frame(self, cap):
            start = time.monotonic()
            result = super()._read_frame(cap)
            self.read_durations[id(cap)] = (time.monotonic() - start) * 1000
            return result

        def _read_direct(self):
            start = time.monotonic()
            result = super()._read_direct()
            end = time.monotonic()
            stamp = float(np.median(list(self._last_direct_monotonic_timestamps.values())))
            with self.measurement_lock:
                self.measurements.append(dict(start=start, end=end, frame_monotonic=stamp,
                    capture_and_prepare_ms=(end - start) * 1000,
                    camera_read_ms={k: self.read_durations[id(cap)] for k, cap in self._caps.items()},
                    prepare_after_last_read_ms=(end - max(self._last_direct_monotonic_timestamps.values())) * 1000))
            return result

    prefs = json.loads((Path.home() / '.config/bimanual-vla/collect_gui_preferences.json').read_text())
    selectors = dict(cam_high='auto', cam_left_wrist=prefs['left_wrist_device'],
                     cam_right_wrist=prefs['right_wrist_device'])
    if prefs.get('swap_wrist_cameras'):
        selectors['cam_left_wrist'], selectors['cam_right_wrist'] = selectors['cam_right_wrist'], selectors['cam_left_wrist']
    cameras = MeasuredCameras(cam_ids=selectors, fps=30, image_hw=(256, 256),
        capture_hw=(240, 424), parallel_reads=True, policy_image_hw=(224, 224), resize_mode='stretch')
    pipers = {}
    client = None
    rows = []
    started = time.monotonic()
    try:
        for side, can_name in [('left', args.left_can), ('right', args.right_can)]:
            arm = C_PiperInterface_V2(can_name, judge_flag=False, can_auto_init=False)
            arm.CreateCanBus(can_name=can_name, bustype='socketcan', expected_bitrate=1000000, judge_flag=False)
            # Receive and parse only: skip SDK initialization queries and all enable/control commands.
            arm.ConnectPort(can_init=True, piper_init=False)
            pipers[side] = arm
        time.sleep(1)
        for arm in pipers.values():
            read_output_qpos(arm)
        cameras.open()
        selected = {key: dict(selector=str(cameras._ids[key]),
            width=c.get(cv2.CAP_PROP_FRAME_WIDTH), height=c.get(cv2.CAP_PROP_FRAME_HEIGHT),
            fps=c.get(cv2.CAP_PROP_FPS)) for key, c in cameras._caps.items()}
        print('Cameras:', json.dumps(selected), flush=True)
        cameras.start_background_capture(fps=30)
        time.sleep(2)
        client = TimedWebsocketClientPolicy('127.0.0.1', args.port)
        metadata = client.get_server_metadata()
        protocol = validate_policy_metadata(metadata, arm_side='right', arm_mode='bimanual', output_mode='joint')
        session = 'latency-live-' + uuid.uuid4().hex
        for i in range(args.samples + 1):
            cycle = time.monotonic()
            qpos = np.concatenate([read_output_qpos(pipers[side]) for side in ('left', 'right')])
            state = policy_observation_state(np.empty(0, dtype=np.float32), qpos, protocol)
            state_ready = time.monotonic()
            frames = cameras.read_nearest(state_ready)
            frame_ready = time.monotonic()
            rtc = dict(session_id=session, inference_generation=i + 1,
                       config=dict(enabled=True, execution_horizon=10), config_revision=1,
                       previous_chunk_generation=i, previous_chunk_offset_steps=7,
                       inference_delay_steps=7 if i else 0)
            observation = dict(state=state, images=frames.policy_images,
                prompt='put the bottle on the balck pad',
                client_metadata=dict(source_name='read-only-live-latency', allow_execution=False,
                                     rtc=rtc, captured_at=time.time()))
            infer_start = time.monotonic()
            result = client.infer(observation)
            received = time.monotonic()
            actions = np.asarray(result['actions'])
            assert actions.shape == (50, 14) and np.isfinite(actions).all()
            ct = result['client_transport_timing']
            st = result.get('transport_timing', {})
            oldest = min(frames.monotonic_timestamps.values())
            newest = max(frames.monotonic_timestamps.values())
            with cameras.measurement_lock:
                capture = min(cameras.measurements, key=lambda r: abs(r['frame_monotonic'] - frames.captured_monotonic)).copy()
            row = dict(index=i, cold=(i == 0), state_read_ms=(state_ready - cycle) * 1000,
                select_and_copy_ms=(frame_ready - state_ready) * 1000,
                payload_build_ms=(infer_start - frame_ready) * 1000,
                rpc_ms=(received - infer_start) * 1000, request_cycle_ms=(received - cycle) * 1000,
                image_host_read_age_at_send_ms=(infer_start - oldest) * 1000,
                image_host_read_to_actions_ms=(received - oldest) * 1000,
                image_read_start_to_actions_ms=(received - capture['start']) * 1000,
                camera_set_skew_ms=(newest - oldest) * 1000,
                capture=capture, client_transport=ct, server_transport=st,
                server_policy_ms=st.get('server_policy_ms'),
                server_sampler_ms=result.get('policy_timing', {}).get('infer_ms'),
                rtc_enabled=result['rtc']['enabled'], rtc_delay_steps=result['rtc']['inference_delay_steps'],
                image_shapes={k: list(v.shape) for k, v in frames.policy_images.items()})
            rows.append(row)
            print(f"{i}: cycle={row['request_cycle_ms']:.1f}ms frame_read_to_actions={row['image_host_read_to_actions_ms']:.1f}ms rtc={row['rtc_enabled']}", flush=True)
        warm = rows[1:]
        fields = ('state_read_ms', 'select_and_copy_ms', 'payload_build_ms', 'rpc_ms', 'request_cycle_ms',
                  'image_host_read_age_at_send_ms', 'image_host_read_to_actions_ms',
                  'image_read_start_to_actions_ms', 'camera_set_skew_ms', 'server_policy_ms', 'server_sampler_ms')
        with cameras.measurement_lock:
            captures = [dict(r) for r in cameras.measurements if r['start'] >= started + 3]
        report = dict(status='completed', endpoint=f'ws://127.0.0.1:{args.port}', metadata=metadata,
            camera_devices=selected, can_interfaces=dict(left=args.left_can, right=args.right_can),
            scope='Live cameras and passive CAN feedback through local policy action return; no motion',
            limitation='Camera timestamps are host read completion, not sensor exposure; hidden sensor/USB/driver buffering is unmeasured',
            warm_samples=len(warm), summary_ms={k: summary([r[k] for r in warm]) for k in fields},
            transport_summary_ms={k: summary([r['client_transport'][k] for r in warm])
                for k in ('pack_ms', 'send_ms', 'wait_response_ms', 'unpack_ms')},
            capture_summary_ms={k: summary([r[k] for r in captures])
                for k in ('capture_and_prepare_ms', 'prepare_after_last_read_ms')},
            camera_read_summary_ms={k: summary([r['camera_read_ms'][k] for r in captures]) for k in selected},
            captures=captures, rows=rows)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print('SUMMARY', json.dumps(report['summary_ms'], indent=2), flush=True)
        print('RESULT', path, flush=True)
    finally:
        if client is not None:
            client._ws.close()
        cameras.close()
        for arm in pipers.values():
            arm.DisconnectPort()


if __name__ == '__main__':
    main()
