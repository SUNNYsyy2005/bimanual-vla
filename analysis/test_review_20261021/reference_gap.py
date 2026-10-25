"""Compare latest accepted prediction and issued commands on the same clock."""
import json
from pathlib import Path
import numpy as np
from review import J,read,stat
results=[]
for tag in ['105741','105846','110000','110058']:
 p=next(Path('deployment_runs').glob('*T'+tag+'*'));sv=read(p/'servo_commands.jsonl');tr=read(p/'trajectory.jsonl');ms=[m for m in read(p/'model_commands.jsonl') if m.get('accepted') and m.get('action_file')];t0=tr[0]['monotonic_timestamp'];tt=np.array([r['monotonic_timestamp'] for r in tr]);mt=np.array([m['arrived_monotonic'] for m in ms]);raw={m['generation']:np.load(p/m['action_file'])['raw_actions'][:,J] for m in ms};gaps=[];times=[];feedback=[];examples=[]
 for s in sv:
  at=max(v['joint_ctrl_monotonic'] for v in s['sides'].values());i=np.searchsorted(mt,at,side='right')-1;ti=np.searchsorted(tt,at,side='right')-1
  if i<0 or ti<0 or ti>=len(tr) or s['braking'] or not tr[ti]['command_sent'] or tr[ti]['command_hold']:continue
  if at-t0<8 or max(abs(np.array(s['velocity'])))<=.05:continue
  m=ms[i];a=raw[m['generation']];source=(at-m['captured_monotonic'])*20-1
  if not 0<=source<len(a)-1:continue
  left=int(np.floor(source));ref=a[left]+(source-left)*(a[left+1]-a[left]);cmd=np.array(s['command_qpos'])[J];error=float(max(abs(cmd-ref)));gaps.append(error);times.append(at);feedback.append(float(max(abs(np.array(s['feedback_qpos'])[J]-ref))))
  examples.append(dict(at=at-t0,generation=m['generation'],joint=int(J[np.argmax(abs(cmd-ref))]),raw=float(ref[np.argmax(abs(cmd-ref))]),command=float(cmd[np.argmax(abs(cmd-ref))]),error=error))
 result=dict(time=tag,latest_prediction_to_command_same_time_rad=stat(gaps),fraction_above_0_1=float(np.mean(np.array(gaps)>.1)),largest=sorted(examples,key=lambda x:-x['error'])[:3]);results.append(result);print(json.dumps(result))
Path(__file__).with_name('reference_gap_results.json').write_text(json.dumps(results,indent=2))
