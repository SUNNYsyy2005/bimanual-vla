"""Profile independent policy requests from a saved observation; no hardware I/O."""
import json
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), '/home/user/src/openpi-5060/packages/openpi-client/src']
from bimanual_vla.deployment.transport import TimedWebsocketClientPolicy

p = next((ROOT/'deployment_runs').glob('20261022T095336*'))
m = list(map(json.loads, (p/'model_commands.jsonl').open()))[45]
with np.load(p/m['action_file']) as data:
    state = data['qpos'].astype(np.float32)
    state[6::7] = np.clip(state[6::7]/.07, 0, 1)
    prefix = data['planned_prefix_actions'].copy()
    offsets = data['planned_prefix_offsets_s'].copy()
frames = list(map(json.loads, (p/'videos/timestamps.jsonl').open()))
images = {}
for name, stamp in m['image_timestamps'].items():
    row = min((r for r in frames if r['camera']==name), key=lambda r: abs(r['timestamp']-stamp))
    cap = cv2.VideoCapture(str(p/'videos'/f'{name}.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_index']); ok, image = cap.read(); cap.release()
    assert ok
    images[name] = cv2.cvtColor(cv2.resize(image, (224,224)), cv2.COLOR_BGR2RGB).transpose(2,0,1)
session = 'offline-profile-' + uuid.uuid4().hex
client = TimedWebsocketClientPolicy('127.0.0.1', 8099)
rows = []
try:
    for g in range(1,9):
        rtc = dict(session_id=session, inference_generation=g, config_revision=1,
                   config=dict(enabled=True, execution_horizon=10), inference_delay_steps=len(prefix),
                   prefix_source='client_planner', planned_prefix=dict(plan_id=f'profile:{g}',
                   action_hz=20., sample_offsets_s=offsets, actions=prefix))
        if g == 1:
            rtc['inference_delay_steps'] = 0
            rtc['planned_prefix'] = None
        observation = dict(state=state, images=images, prompt='put the bottle on the balck pad',
                           client_metadata=dict(source_name='saved-observation-profile',
                                                allow_execution=False, rtc=rtc))
        start = time.monotonic(); result = client.infer(observation)
        rows.append(dict(wall_ms=(time.monotonic()-start)*1000,
                         phases=result.get('model_phase_timing'),
                         transport=result.get('client_transport_timing'),
                         server=result.get('transport_timing'), rtc=result.get('rtc')))
finally:
    client._ws.close()
output = Path(__file__).with_name('saved_observation_profile.json')
output.write_text(json.dumps(dict(scope='8 independent requests, saved decoded video and state, no hardware',
                                  rows=rows), indent=2)+'\n')
for key in rows[-1]['phases'] or {}:
    values=[r['phases'][key] for r in rows[1:] if isinstance(r['phases'].get(key),(int,float))]
    if values: print(key, 'median_ms', round(float(np.median(values)),3))
