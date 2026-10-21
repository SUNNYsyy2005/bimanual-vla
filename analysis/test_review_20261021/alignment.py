"""Check clocked RTC prefix against actual published planner states."""
import json
from pathlib import Path
import numpy as np
from review import read,stat,J
results=[]
for tag in ['100325','105741','105846','110000','110058']:
 p=next(Path('deployment_runs').glob('*T'+tag+'*'));sv=[s for s in read(p/'servo_commands.jsonl') if s.get('planned_monotonic') is not None];tr=read(p/'trajectory.jsonl');ms=read(p/'model_commands.jsonl');lookup={m['generation']:m for m in ms};t0=tr[0]['monotonic_timestamp']
 pt=np.array([s['planned_monotonic'] for s in sv]);pq=np.array([s['command_qpos'] for s in sv])[:,J];errs=[];gaps=[];examples=[];boundaries=[];revisions=[];seam_velocity=[];prev=None
 for m in ms:
  if not m.get('action_file'):continue
  a=np.load(p/m['action_file'])['raw_actions'][:,J]
  if (m.get('rtc',{}).get('prefix_source')=='client_planner'
      and m.get('rtc',{}).get('enabled') and m.get('rtc',{}).get('client_plan_id')):
   d=m['rtc_request']['inference_delay_steps'];tt=m['captured_monotonic']+(np.arange(d)+1)/20
   ix=np.searchsorted(pt,tt,side='right')-1
   if 2<=d<len(a):seam_velocity.append(float(max(abs((a[d]-2*a[d-1]+a[d-2])*20))))
   for k,i in enumerate(ix):
    if i<0 or tt[k]-pt[i]>.01001:continue
    e=float(max(abs(a[k]-pq[i])));errs.append(e);gaps.append(tt[k]-pt[i]);examples.append(dict(generation=m['generation'],index=k,error=e,time=tt[k]-t0))
   b=m.get('rtc',{}).get('position_boundary_jump_l2')
   if b is not None:boundaries.append(b)
  if prev is not None:
   old,oa=prev;future=m['captured_monotonic']+.5
   ot=old['captured_monotonic']+(np.arange(len(oa))+1)/20
   if future<=ot[-1]:
    oq=np.array([np.interp(future,ot,oa[:,j]) for j in range(12)])
    revisions.append(float(max(abs(a[9]-oq))))
  prev=m,a
 residual=[];first=[];seen=set()
 for r in tr:
  if not r['command_sent'] or r['command_hold'] or r['command_generation'] not in lookup:continue
  m=lookup[r['command_generation']];idx=r['command_queue_index'];off=(m['captured_monotonic']+(idx+1)/20-r['monotonic_timestamp'])*1000;residual.append(off)
  if m['generation'] not in seen:seen.add(m['generation']);first.append(dict(generation=m['generation'],index=idx,elapsed_steps=(r['monotonic_timestamp']-m['captured_monotonic'])*20,source_time_minus_control_ms=off))
 result=dict(time=tag,prefix_command_error=stat(errs),prefix_gap_ms=stat(np.array(gaps)*1000),boundary_l2=stat(boundaries),prefix_to_suffix_velocity_change=stat(seam_velocity),same_future_raw_revision_rad=stat(revisions),source_time_minus_control_ms=stat(residual),first_selected=first,worst_prefix=sorted(examples,key=lambda x:-x['error'])[:4]);results.append(result)
 print(tag,{k:v for k,v in result.items() if k not in ['first_selected','worst_prefix']})
Path(__file__).with_name('alignment_results.json').write_text(json.dumps(results,indent=2))
