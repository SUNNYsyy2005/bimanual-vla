"""Additional timing, boundary and prior-command tracking checks."""
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'test_review_20261021'))
from review import read,stat,J
OUT=Path(__file__).parent
rs=json.loads((OUT/'timing_results.json').read_text());out=[]
for r in rs:
 tag=r['time'];p=next(Path('deployment_runs').glob('20261022T'+tag+'*'));ms=read(p/'model_commands.jsonl');tr=read(p/'trajectory.jsonl');sv=read(p/'servo_commands.jsonl');t0=tr[0]['monotonic_timestamp']
 md=json.loads((p/'metadata.json').read_text());row=dict(time=tag,host=md['policy_host'],port=md['policy_port'],configured_hz=md['inference_hz'],rtc=md['runtime_configuration']['rtc'])
 row['timing']={k:stat([m.get('_client_transport_timing',{}).get(k,np.nan) for m in ms]) for k in ['feedback_to_snapshot_start_ms','camera_select_ms','rtc_prefix_build_ms','snapshot_freeze_ms','request_worker_queue_ms','capture_to_rpc_ms','response_to_admission_ms','unattributed_rpc_ms']}
 row['sampler_ms']=stat([m.get('rtc',{}).get('sampler_wall_ms',np.nan) for m in ms])
 row['camera_at_send_age_ms']=stat([(m['launched_at']-v)*1000 for m in ms for v in m['image_timestamps'].values()])
 prefix_errors=[];boundary=[]
 for m in ms:
  if not m.get('accepted') or not m.get('action_file'):continue
  f=np.load(p/m['action_file']);a=f['raw_actions'];d=int(m['rtc_request']['inference_delay_steps']);cap=m['captured_monotonic']-t0
  if 'planned_prefix_actions' not in f or len(f['planned_prefix_actions'])!=d or not d:continue
  pre=f['planned_prefix_actions'];prefix_errors.append(float(abs(a[:d]-pre).max()))
  # Fixed diagnostic: right J2 suffix first 3 rows vs last frozen prefix,
  # restricted to an increasing prefix and capture >=2 s.
  if cap>=2 and pre[-1,8]-pre[-2,8]>.0005:
   retreat=float(pre[-1,8]-min(a[d:d+3,8]));boundary.append(dict(generation=m['generation'],cap_s=cap,retreat_rad=retreat,q=a[d-1:d+4,8].tolist()))
 row['raw_prefix_absolute_error']=stat(prefix_errors);row['r2_increasing_prefix_count']=len(boundary);row['r2_suffix_retreat_ge_0_01']=sum(x['retreat_rad']>=.01 for x in boundary);row['r2_suffix_retreat_ge_0_02']=sum(x['retreat_rad']>=.02 for x in boundary);row['r2_suffix_examples']=sorted(boundary,key=lambda x:-x['retreat_rad'])[:3]
 row['nonstartup_rejections']=[m.get('rejection') or m.get('error') for m in ms if not m.get('accepted') and 'settle' not in str(m.get('rejection'))]
 row['hold_intervals']=[]
 for i,t in enumerate(tr):
  if t['command_hold'] and (i==0 or not tr[i-1]['command_hold']):start=t['monotonic_timestamp']-t0
  if t['command_hold'] and (i==len(tr)-1 or not tr[i+1]['command_hold']):row['hold_intervals'].append([start,t['monotonic_timestamp']-t0,t['blocked_reason']])
 # Find maximum aligned feedback error against the last already issued command.
 ct=np.array([max(v['joint_ctrl_monotonic'] for v in x['sides'].values()) for x in sv]);ft=np.array([x['feedback_read_monotonic'] for x in sv]);cmd=np.array([x['command_qpos'] for x in sv]);fb=np.array([x['feedback_qpos'] for x in sv]);ix=np.maximum(0,np.searchsorted(ct,ft,side='right')-1)
 e=abs(cmd[ix][:,J]-fb[:,J]);ok=(ft>=ct[0])&(ft<=tr[-1]['monotonic_timestamp'])&np.array([not x['braking'] for x in sv]);e[~ok]=0;i,j=np.unravel_index(e.argmax(),e.shape)
 row['max_feedback_error']=dict(at_s=float(ft[i]-t0),joint=int(J[j]),error=float(e[i,j]),generation=sv[i]['context'].get('generation'))
 row['long_waits']=[x for x in r['first'] if x['wait_ms']>=300]
 # Compare planned splice with that request's own prefix commitment.
 models={x['generation']:x for x in ms};matches=[]
 for plan in r['plans']:
  m=models[plan['generation']];own=m['captured_monotonic']+m['rtc_request']['inference_delay_steps']/20
  matches.append(0 <= plan['splice_monotonic']-own <= .011)
 row['own_prefix_boundary_splices']=dict(match=sum(matches),total=len(matches))
 row['actual_handoff_interval_ms']=stat(np.diff([x['at_s'] for x in r['first']])*1000)
 out.append(row)
 print(tag,'prefixerr',row['raw_prefix_absolute_error'],'suffixretreat',row['r2_suffix_retreat_ge_0_01'],'/',len(boundary),'ownsplice',row['own_prefix_boundary_splices'],'maxtrack',row['max_feedback_error'],'holds',row['hold_intervals'],'long',row['long_waits'])
(OUT/'details.json').write_text(json.dumps(out,indent=2)+'\n')
