"""Quantify commanded retreats following actual generation handoffs, read-only.
Requires timing.py output. No inference or hardware access.
"""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).parent
J=np.array([0,1,2,3,4,5,7,8,9,10,11,12])
results=[]
for tag in ['104131','104306','104416','104534','104653','104728']:
 rows=json.loads((OUT/f'samples_{tag}.json').read_text())
 t=np.array([x['at_s'] for x in rows]);q=np.array([x['command'] for x in rows])[:,J];v=np.array([x['velocity'] for x in rows]);g=np.array([x['generation'] for x in rows])
 starts=np.flatnonzero(g[1:]!=g[:-1])+1;events=[]
 for a in starts:
  if t[a]<2 or t[a]-t[a-1]>.03:continue
  b=np.searchsorted(t,t[a]+.25,side='right');end=next((i for i in starts if i>a),len(t));b=min(b,end)
  if b-a<5:continue
  before=(t>=t[a]-.05)&(t<t[a]);incoming=np.median(v[before],axis=0);moving=abs(incoming)>=.02
  if not moving.any():continue
  signed=q[a:b]*np.sign(incoming)
  # Maximum backwards excursion after any earlier point within this window,
  # measured in the incoming direction, not raw command/feedback distance.
  drop=(np.maximum.accumulate(signed,axis=0)-signed).max(axis=0);drop[~moving]=0
  j=int(drop.argmax());events.append(dict(generation=int(g[a]),at_s=float(t[a]),window_s=float(t[b-1]-t[a]),joint_index=int(J[j]),retreat_rad=float(drop[j])))
 p=next(Path('deployment_runs').glob('20261022T'+tag+'*'));ms=[json.loads(x) for x in (p/'model_commands.jsonl').open()];raw=[]
 for m in ms:
  if not m.get('accepted') or not m.get('action_file'):continue
  f=np.load(p/m['action_file']);d=int(m['rtc_request']['inference_delay_steps'])
  if 'planned_prefix_actions' not in f or d<2:continue
  pre=f['planned_prefix_actions'];a=f['raw_actions'];inc=(pre[-1,J]-pre[-2,J])/.05;moving=abs(inc)>=.02
  if not moving.any():continue
  disp=(a[d:d+3,J]-pre[-1,J])*np.sign(inc)
  drop=np.maximum(0,-disp.min(axis=0));drop[~moving]=0
  raw.append(dict(generation=m['generation'],max_retreat=float(drop.max())))
 result=dict(time=tag,definition='Actual handoff: at least one incoming joint |v|>=0.02 rad/s; maximum backwards command excursion within 250 ms or next handoff, excluding gaps/startup.',eligible_handoffs=len(events),command_counts={str(th):sum(x['retreat_rad']>=th for x in events) for th in [.005,.01,.02]},raw_eligible=len(raw),raw_counts={str(th):sum(x['max_retreat']>=th for x in raw) for th in [.005,.01,.02]},examples=sorted(events,key=lambda x:-x['retreat_rad'])[:5])
 results.append(result);print(json.dumps(result))
(OUT/'switch_alignment.json').write_text(json.dumps(results,indent=2)+'\n')
