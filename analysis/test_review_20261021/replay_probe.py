"""Validate frozen-chunk replays against the recorded published portion."""
import json
from pathlib import Path
import numpy as np
from bimanual_vla.deployment.chunk_planner import compile_chunk
from bimanual_vla.deployment.servo import ServoConfig
from review import read,J
p=next(Path('deployment_runs').glob('*T110058*'));sv=read(p/'servo_commands.jsonl');tr=read(p/'trajectory.jsonl');ms=read(p/'model_commands.jsonl');cfg=json.loads((p/'metadata.json').read_text())['runtime_configuration'];t0=tr[0]['monotonic_timestamp']
sc=ServoConfig(**{k:v for k,v in cfg['servo'].items() if k!='backend'});lims=np.array(cfg['effective_joint_limits_rad']['left']+cfg['effective_joint_limits_rad']['right']);lo=np.zeros(14);hi=np.full(14,.07);lo[J]=lims[:,0];hi[J]=lims[:,1]
results=[]
for gen in [29,30,31,32,33]:
 m=next(m for m in ms if m['generation']==gen);sample=[s for s in sv if s['context'].get('generation')==gen];skip=next(r['command_queue_index'] for r in tr if r['command_generation']==gen and r['command_sent']);a=np.load(p/m['action_file'])['raw_actions'][skip:].astype(float);a[:,6::7]*=.07;a=np.clip(a,lo,hi);ts=m['captured_monotonic']+(np.arange(skip,50)+1)/20
 candidates=[]
 for s in sample[:30]:
  start=s['planned_monotonic']
  try:plan=compile_chunk(s['command_qpos'],s['velocity'],s['acceleration'],lo,hi,sc,a,ts,np.arange(skip,50),start=start,generation=gen)
  except Exception:continue
  observed=[x for x in sample if start<=x['planned_monotonic']]
  if len(observed)<18:continue
  error=max(max(abs(plan.positions[int(round((x['planned_monotonic']-start)*sc.hz))][J]-np.array(x['command_qpos'])[J])) for x in observed)
  candidates.append((error,s,plan))
 matches=[x for x in candidates if x[0]<1e-6]
 print('matching_starts',gen,[(round(x[1]['planned_monotonic']-t0,3),round(x[0],10)) for x in matches])
 error,s,plan=(matches[0] if matches else min(candidates,key=lambda x:x[0]));start=s['planned_monotonic'];item=dict(generation=gen,start=start-t0,matched_error=error,initial_q=s['position'][1],initial_v=s['velocity'][1],initial_a=s['acceleration'][1],reference_times=(ts-t0).tolist(),reference_q=a[:,1].tolist(),plan_times=(plan.times-t0).tolist(),plan_q=plan.positions[:,1].tolist());observed=[x for x in sv if x.get('planned_monotonic') is not None and start<=x['planned_monotonic']<=start+.9]
 differences=[float(max(abs(plan.positions[int(round((x['planned_monotonic']-start)*sc.hz))][J]-np.array(x['command_qpos'])[J]))) for x in observed]
 first_diff=next(((x['planned_monotonic']-t0,e,x['context']['generation']) for x,e in zip(observed,differences) if e>1e-5),None)
 item['first_divergence']=first_diff
 item['matched_to_next_change_s']=first_diff[0]-item['start'] if first_diff else None
 print('diverges from frozen',gen,first_diff)
 results.append(item)
 print(gen,'skip',skip,'best error',error,'start',start-t0,'initial',item['initial_q'],item['initial_v'],item['initial_a'])
Path(__file__).with_name('replay_results.json').write_text(json.dumps(results,indent=2))
