"""Inspect recorded raw chunk interiors, independently of servo limits."""
import json
from pathlib import Path
import numpy as np
J=[0,1,2,3,4,5,7,8,9,10,11,12]
OUT=Path('analysis/chunk_smoothness_20261019')
def stats(x):
 a=np.concatenate(x)
 return dict(count=a.size,abs_p50_p95_p99_max=np.percentile(abs(a),[50,95,99,100]).tolist())
def main():
 groups={k:[] for k in ('v','a','jerk','selected_same_v','selected_switch_v')}; records=[]; worst=None; skipped=0
 for run in sorted(Path('deployment_runs').glob('20261008*')):
  log=run/'model_commands.jsonl'
  if not log.exists():continue
  for line in log.open():
   r=json.loads(line)
   if not r.get('accepted') or not r.get('action_file'):continue
   hz=(r.get('protocol') or {}).get('action_hz')
   if not hz or (r.get('protocol') or {}).get('action_semantics')!='absolute_joint_position_opening_fraction':skipped+=1;continue
   with np.load(run/r['action_file']) as d:q=d['raw_actions'][:,J].astype(float)
   if len(q)<4 or not np.isfinite(q).all():skipped+=1;continue
   v=np.diff(q,axis=0)*hz;a=np.diff(v,axis=0)*hz;j=np.diff(a,axis=0)*hz
   for key,val in zip(('v','a','jerk'),(v,a,j)):groups[key].append(val.ravel())
   # Significant reversal: both adjacent interval speeds exceed .1 rad/s.
   rev=(v[:-1]*v[1:]<0)&(abs(v[:-1])>.1)&(abs(v[1:])>.1)
   rec=dict(run=run.name,generation=r['generation'],file=r['action_file'],max_speed=float(abs(v).max()),max_acceleration=float(abs(a).max()),reversal_count=int(rev.sum()),step=int(np.unravel_index(abs(a).argmax(),a.shape)[0]+1),joint=int(J[np.unravel_index(abs(a).argmax(),a.shape)[1]]))
   records.append(rec)
   if worst is None or rec['max_acceleration']>worst[0]['max_acceleration']:worst=(rec,q,hz)
  traj=run/'trajectory.jsonl';prev=None
  if not traj.exists():continue
  for line in traj.open():
   r=json.loads(line);valid=r.get('command_sent') and not r.get('command_hold')
   if valid:
    q=np.asarray(r['command_action'],dtype=float)[J];t=r['command_monotonic_timestamp']
    valid=np.isfinite(q).all() and t is not None
   if not valid:prev=None;continue
   if prev is not None:
    dt=t-prev[0]
    if 0<dt<=.125:groups['selected_switch_v' if r['command_generation']!=prev[2] else 'selected_same_v'].append(((q-prev[1])/dt).ravel())
   prev=(t,q,r['command_generation'])
 result=dict(chunks=len(records),runs=len({r['run'] for r in records}),skipped=skipped,metrics={k:stats(v) for k,v in groups.items()},chunks_with_speed_above_1=sum(r['max_speed']>1 for r in records),chunks_with_acceleration_above_4=sum(r['max_acceleration']>4 for r in records),chunks_with_significant_reversal=sum(r['reversal_count']>0 for r in records),worst_chunks=sorted(records,key=lambda r:r['max_acceleration'],reverse=True)[:20])
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2)[:6500])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rec,q,hz=worst; col=J.index(rec['joint']);t=np.arange(len(q))/hz
 fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
 ax[0].plot(t,q[:,col],'.-');ax[0].set_ylabel('Joint position (rad)')
 ax[1].plot((t[1:]+t[:-1])/2,np.diff(q[:,col])*hz,'.-');ax[1].set_ylabel('Velocity (rad/s)')
 ax[2].plot(t[1:-1],np.diff(q[:,col],n=2)*hz**2,'.-');ax[2].set_ylabel('Acceleration (rad/s²)');ax[2].set_xlabel('Chunk time (s)')
 fig.suptitle(f"Raw chunk interior: {rec['run']}\ngeneration {rec['generation']}, joint index {rec['joint']} (zero based)")
 fig.savefig(OUT/'worst_chunk.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
