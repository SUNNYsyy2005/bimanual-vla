"""Compare prior published/selected commands to feedback sampled afterwards."""
import json
from pathlib import Path
import numpy as np
J=[0,1,2,3,4,5,7,8,9,10,11,12]
OUT=Path('analysis/execution_lag_20261019')
def main():
 rows=[];boundaries=[];configs=[]
 for path in sorted(Path('deployment_runs').glob('20261008*/trajectory.jsonl')):
  meta=json.load((path.parent/'metadata.json').open());configs.append(dict(run=path.parent.name,configuration=meta.get('runtime_configuration',{})))
  prev=None;prevprev=None
  for line in path.open():
   r=json.loads(line)
   valid=r.get('command_sent') and not r.get('command_hold') and len(r.get('command_joints_rad') or [])==12
   if not valid:prev=None;prevprev=None;continue
   q=np.array(r['qpos'],float)[J];target=np.array(r['command_action'],float)[J];sent=np.array(r['command_joints_rad'],float)
   ft=r['monotonic_timestamp'];ct=r['command_monotonic_timestamp']
   if not all(np.isfinite(x).all() for x in (q,target,sent)):prev=None;prevprev=None;continue
   cur=dict(q=q,target=target,sent=sent,ft=ft,ct=ct,gen=r['command_generation'],index=r['command_queue_index'])
   if prev is not None and 0<ft-prev['ct']<=.125:
    rec=dict(run=path.parent.name,t=ft,generation=cur['gen'],prior_generation=prev['gen'],feedback_after_previous_command_ms=(ft-prev['ct'])*1000,
      shaping_max_rad=float(abs(prev['target']-prev['sent']).max()),
      sent_feedback_max_rad=float(abs(prev['sent']-q).max()),
      target_feedback_max_rad=float(abs(prev['target']-q).max()))
    rows.append(rec)
    if cur['gen']!=prev['gen']:
     e=prev['target']-q;newgap=target-q;jump=target-prev['target']
     rec=dict(rec,new_target_feedback_max_rad=float(abs(newgap).max()),selected_target_jump_max_rad=float(abs(jump).max()),
      new_closer_to_old_than_feedback=bool(np.linalg.norm(jump)<np.linalg.norm(newgap)),
      error_projection_on_new_gap=float(np.dot(e,newgap)/(np.linalg.norm(e)*np.linalg.norm(newgap))) if np.linalg.norm(e)*np.linalg.norm(newgap)>1e-10 else None)
     if prevprev is not None and prevprev['gen']==prev['gen']:
      motion=prev['target']-prevprev['target'];norm=np.linalg.norm(motion)
      if norm>0.005:
       rec['behind_along_old_motion_rad']=float(np.dot(e,motion)/norm)
     boundaries.append(rec)
   prevprev=prev;prev=cur
 def stats(records):
  keys=['shaping_max_rad','sent_feedback_max_rad','target_feedback_max_rad','feedback_after_previous_command_ms','new_target_feedback_max_rad','selected_target_jump_max_rad','behind_along_old_motion_rad']
  return {k:dict(n=len(a),p50_p95_p99=np.percentile(a,[50,95,99]).tolist()) for k in keys if (a:=[r[k] for r in records if k in r])}
 moving=[r for r in boundaries if 'behind_along_old_motion_rad' in r]
 result=dict(runs=len(configs),comparisons=len(rows),switches=len(boundaries),all=stats(rows),switch=stats(boundaries),
   switches_old_gap_over_005=sum(r['target_feedback_max_rad']>.05 for r in boundaries),
   switches_new_closer_old=sum(r['new_closer_to_old_than_feedback'] for r in boundaries),
   moving_switches=len(moving),moving_switches_behind=sum(r['behind_along_old_motion_rad']>.01 for r in moving),
   examples=sorted(boundaries,key=lambda r:r['target_feedback_max_rad'],reverse=True)[:20],configs=configs)
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 (OUT/'switches.json').write_text(json.dumps(boundaries,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('examples','configs')},indent=2));print('example',json.dumps(result['examples'][0]))
if __name__=='__main__':main()
