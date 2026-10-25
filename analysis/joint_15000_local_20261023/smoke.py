"""Read-only websocket verification for the relabeled single-left-arm local 15k policy."""
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/pi05_local'))
from common import configure
configure('/home/user/src/openpi-5060')
from openpi_client.websocket_client_policy import WebsocketClientPolicy
root=Path(__file__).parent
sample=np.load(root/'sample.npz')
c=WebsocketClientPolicy('127.0.0.1',8099)
m=c.get_server_metadata()
assert m['checkpoint_step']==15000 and m['state_dim']==7 and m['arm_side']=='left' and m['training_arm_side']=='right'
assert m['rtc_supported'] is False and m['rtc_mode']=='off'
assert m['wire_action_convention']=='absolute_joint_target'
assert m['camera_keys']==['cam_high','cam_left_wrist']
assert m['experiment']=='joint_trim_nopad_no_rtc_b4_20261022'
rows=[];predictions=[]
try:
 for i in range(6):
  obs=dict(state=sample['state'],images={k:sample['cam_right_wrist' if k=='cam_left_wrist' else k] for k in m['camera_keys']})
  if i:obs['prompt']=str(sample['prompt'])
  t=time.perf_counter();result=c.infer(obs);elapsed=time.perf_counter()-t
  actions=np.asarray(result['actions']);assert actions.shape==(50,7) and np.isfinite(actions).all()
  rows.append(dict(seconds=elapsed,shape=list(actions.shape),joint_mae_rad=float(abs(actions[:,:6]-sample['truth'][:,:6]).mean())))
  predictions.append(actions)
finally:c._ws.close()
np.savez_compressed(root/'smoke_predictions.npz',predictions=predictions,truth=sample['truth'])
report=dict(success=True,metadata=m,episode=int(sample['episode']),frame=int(sample['frame']),requests=rows,warm_median_s=float(np.median([r['seconds'] for r in rows[1:]])),note='One heldout observation, repeated stochastic outputs; not a benchmark or robot execution')
(root/'smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(success=True,requests=len(rows),cold_s=rows[0]['seconds'],warm_median_s=report['warm_median_s'])))
