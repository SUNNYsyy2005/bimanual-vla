"""Offline review; all positions are absolute wire joints in radians.
Feedback is paired only with commands already published. No hardware imports.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
OUT=Path(__file__).parent
J=np.array([0,1,2,3,4,5,7,8,9,10,11,12])
NAMES=[f'{s}{j}' for s in ('L','R') for j in range(1,7)]
def read(p): return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []
def stat(v):
 v=np.asarray(v,dtype=float);v=v[np.isfinite(v)]
 return dict(n=len(v),p50=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(max(v))) if len(v) else dict(n=0)
def audit(p):
 meta=json.loads((p/'metadata.json').read_text());cfg=meta['runtime_configuration']
 tr=read(p/'trajectory.jsonl');ms=read(p/'model_commands.jsonl');sv=read(p/'servo_commands.jsonl')
 if not tr:return None
 t=np.array([r['monotonic_timestamp'] for r in tr]); t0=t[0]
 active=np.array([r['command_sent'] and not r['command_hold'] for r in tr]);hold=np.array([r['command_hold'] for r in tr]);q=np.asarray([r['qpos'] for r in tr],float)[:,J]
 r=dict(run=p.name,time=p.name[9:15],port=meta['policy_port'],servo=cfg['servo']['backend'],servo_config=cfg['servo'],async_=cfg.get('async_inference'),rtc_modes=dict(Counter(m.get('rtc',{}).get('mode','none') for m in ms)),prefix_sources=dict(Counter(m.get('rtc',{}).get('prefix_source','none') for m in ms)),instances=sorted({m.get('transport_timing',{}).get('policy_instance_id','unrecorded') for m in ms}),duration=t[-1]-t[0],active_seconds=float(np.sum(np.diff(t)*active[:-1])),hold_seconds=float(np.sum(np.diff(t)*hold[:-1])),blocked=dict(Counter(x['blocked_reason'] for x in tr if x['blocked_reason'])),rejections=dict(Counter(str(m.get('rejection') or m.get('error')) for m in ms if not m.get('accepted'))))
 r['latency_ms']=stat([(m['arrived_monotonic']-m['captured_monotonic'])*1000 for m in ms]);r['image_age_ms']=stat([(m['captured_at']-v)*1000 for m in ms for v in m.get('image_timestamps',{}).values()]);r['capture_interval']=stat(np.diff([m['captured_monotonic'] for m in ms]))
 used={}
 for x in tr:
  if x['command_sent'] and not x['command_hold']: used.setdefault(x['command_generation'],set()).add(x['command_queue_index'])
 raw_vel=[];raw_acc=[];zig=[];full_vel=[];full_acc=[];raw_map={}
 for m in ms:
  if not m.get('action_file'):continue
  with np.load(p/m['action_file']) as f:a=f['raw_actions'][:,J]
  if a.shape[0]<3:continue
  raw_map[m['generation']]=a
  v=np.diff(a,axis=0)*20;acc=np.diff(v,axis=0)*20
  full_vel.extend(abs(v).max(axis=1));full_acc.extend(abs(acc).max(axis=1))
  inds=sorted(used.get(m['generation'],set()))
  for i in inds:
   if not 1<=i<len(a)-1:continue
   raw_vel.append(max(abs(v[i])));raw_acc.append(max(abs(acc[i-1])))
   b=a[i]-a[i-1];c=a[i+1]-a[i]
   for j in np.flatnonzero((b*c<0)&(np.minimum(abs(b),abs(c))>=.005)):
    zig.append(dict(generation=m['generation'],index=i,joint=NAMES[j],amplitude=float(min(abs(b[j]),abs(c[j]))),q=a[i-1:i+2,j].tolist()))
 r['raw_used_speed']=stat(raw_vel);r['raw_used_accel']=stat(raw_acc);r['raw_full_speed']=stat(full_vel);r['raw_full_accel']=stat(full_acc);r['raw_used_zigzags']=sorted(zig,key=lambda x:-x['amplitude'])[:8];r['raw_used_zigzag_count']=len(zig)
 if sv:
  ct=np.array([max(y['joint_ctrl_monotonic'] for y in x['sides'].values()) for x in sv]);cq=np.array([x['command_qpos'] for x in sv])[:,J]
  ft=np.array([x['feedback_read_monotonic'] for x in sv]);fq=np.array([x['feedback_qpos'] for x in sv])[:,J]
  ix=np.searchsorted(ct,ft,side='right')-1;safe=ix>=0;ix=np.maximum(ix,0)
  err=abs(fq-cq[ix]);speed=np.max(abs(np.array([x['velocity'] for x in sv])),axis=1)
  moving=safe&(speed>.05)&np.array([not x.get('braking') for x in sv])&(ft<=t[-1])
  r['moving_tracking_maxjoint_rad']=stat(err[moving].max(axis=1));r['moving_error_over_0_1_fraction']=float(np.mean(err[moving].max(axis=1)>.1)) if moving.any() else None
  r['speed']=stat(speed[moving]);r['at_speed_cap_fraction']=float(np.mean(speed[moving]>=cfg['servo']['max_velocity']*.99)) if moving.any() else None
  r['accel']=stat(np.max(abs(np.array([x['acceleration'] for x in sv])),axis=1)[moving]);r['publish_interval_ms']=stat(np.diff(ct)*1000)
  late=moving&(ft-t0>8);r['late_moving_tracking_rad']=stat(err[late].max(axis=1))
  r['planner_samples']=sum(bool(x.get('plan_id')) for x in sv)
 else:
  sent=[x for x in tr if x['command_sent']];ct=np.array([x['command_monotonic_timestamp'] for x in sent],float);cq=np.array([x['command_joints_rad'] for x in sent],float).reshape(-1,12)
  ix=np.searchsorted(ct,t,side='right')-1;safe=ix>=0;ix=np.maximum(ix,0)
  if len(ct):
   e=abs(q-cq[ix]).max(axis=1);moving=active&safe
   r['moving_tracking_maxjoint_rad']=stat(e[moving]);r['late_moving_tracking_rad']=stat(e[moving&(t-t0>8)]);r['moving_error_over_0_1_fraction']=float(np.mean(e[moving]>.1)) if moving.any() else None
  r['planner_samples']=0
 if len(ct)<3:return r
 # Same 50 ms grid for comparing commanded variation. Exclude holds, faults,
 # enable and shutdown; analytic Ruckig accelerations above are separate.
 grid=np.arange(max(ct[0],t0),min(ct[-1],t[-1]),.05)
 gq=np.stack([np.interp(grid,ct,cq[:,j]) for j in range(12)],axis=1)
 ti=np.clip(np.searchsorted(t,grid,side='right')-1,0,len(t)-1);good=active[ti]
 gv=np.diff(gq,axis=0)/.05;ga=np.diff(gv,axis=0)/.05
 r['command_50ms_speed']=stat(abs(gv).max(axis=1)[good[:-1]&good[1:]])
 r['command_50ms_accel']=stat(abs(ga).max(axis=1)[good[:-2]&good[1:-1]&good[2:]])
 # Reference-to-command distance is NOT a physical actuator tracking error.
 ref=np.array([x['command_absolute_target'] for x in tr],float)[:,J]
 qc=np.stack([np.interp(t,ct,cq[:,j]) for j in range(12)],axis=1)
 r['selected_ref_to_command_rad']=stat(abs(ref-qc).max(axis=1)[active])
 events=[]
 for j in range(12):
  pk,_=find_peaks(gq[:,j],prominence=.02,distance=3);vl,_=find_peaks(-gq[:,j],prominence=.02,distance=3);ext=sorted(set(pk)|set(vl))
  for a,b,c in zip(ext,ext[1:],ext[2:]):
   if (grid[c]-grid[a]>1.5 or not good[a:c+1].all()
       or (gq[b,j]-gq[a,j])*(gq[c,j]-gq[b,j])>=0):continue
   amp=min(abs(gq[b,j]-gq[a,j]),abs(gq[c,j]-gq[b,j]))
   if amp<.02:continue
   events.append(dict(joint=NAMES[j],times=(grid[[a,b,c]]-t0).tolist(),q=gq[[a,b,c],j].tolist(),amplitude=float(amp),generations=[tr[ti[k]]['command_generation'] for k in [a,b,c]]))
 r['command_oscillations']=sorted(events,key=lambda x:-x['amplitude'])[:6];r['command_oscillation_count']=len(events)
 return r
if __name__=='__main__':
 rs=[audit(p) for p in sorted(Path('deployment_runs').glob('20261021*')) if json.loads((p/'metadata.json').read_text())['policy_port']==8099];rs=[r for r in rs if r]
 (OUT/'results.json').write_text(json.dumps(rs,indent=2)+'\n')
 def f(r,k):return round(r.get(k,{}).get('p95',0),4)
 for r in rs:print(r['time'],r['port'],r['servo'],str(r['rtc_modes']),round(r['active_seconds'],1),round(r['hold_seconds'],1),'track',f(r,'moving_tracking_maxjoint_rad'),'refcmd',f(r,'selected_ref_to_command_rad'),'rawA',f(r,'raw_used_accel'),'cmdA',f(r,'command_50ms_accel'),'zig',r['raw_used_zigzag_count'],'osc',r.get('command_oscillation_count'))
