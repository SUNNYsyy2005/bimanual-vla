"""Hold a recorded prediction fixed; compare timely handoff with recorded path.
This is a planner replay, not a closed-loop grasp-success simulation.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bimanual_vla.deployment.chunk_planner import compile_chunk
from bimanual_vla.deployment.servo import ServoConfig
from review import read,J
p=next(Path('deployment_runs').glob('*T110058*'));sv=read(p/'servo_commands.jsonl');tr=read(p/'trajectory.jsonl');ms=read(p/'model_commands.jsonl');cfg=json.loads((p/'metadata.json').read_text())['runtime_configuration'];t0=tr[0]['monotonic_timestamp'];sc=ServoConfig(**{k:v for k,v in cfg['servo'].items() if k!='backend'});lims=np.array(cfg['effective_joint_limits_rad']['left']+cfg['effective_joint_limits_rad']['right']);lo=np.zeros(14);hi=np.full(14,.07);lo[J]=lims[:,0];hi[J]=lims[:,1]
m=next(m for m in ms if m['generation']==31);capture=m['captured_monotonic'];arrival=m['arrived_monotonic'];end=capture+m['rtc_request']['inference_delay_steps']/20
s=next(s for s in sv if s.get('planned_monotonic') is not None and s['planned_monotonic']>=end)
skip=next(r['command_queue_index'] for r in tr if r['command_generation']==31 and r['command_sent']);raw=np.load(p/m['action_file'])['raw_actions'].astype(float);a=raw[skip:].copy();a[:,6::7]*=.07;a=np.clip(a,lo,hi);ts=capture+(np.arange(skip,50)+1)/20;start=s['planned_monotonic']
plan=compile_chunk(s['command_qpos'],s['velocity'],s['acceleration'],lo,hi,sc,a,ts,np.arange(skip,50),start=start,generation=31)
actual=[x for x in sv if x.get('planned_monotonic') is not None and start<=x['planned_monotonic']<=start+.45];actual_min=min(x['position'][1] for x in actual);mask=(plan.times<=start+.45);counter_min=min(plan.positions[mask,1]);summary=dict(run=p.name,generation=31,capture=capture-t0,arrival=arrival-t0,protected_prefix_end=end-t0,early_handoff=start-t0,initial_q=s['position'][1],initial_v=s['velocity'][1],initial_a=s['acceleration'][1],actual_min_next_450ms=actual_min,timely_handoff_min_next_450ms=counter_min,actual_retreat=s['position'][1]-actual_min,timely_retreat=s['position'][1]-counter_min,config=cfg['servo']);print(json.dumps(summary,indent=2));Path(__file__).with_name('handoff_counterfactual.json').write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(figsize=(10,4.5));t=np.array([s['planned_monotonic'] for s in sv if s.get('planned_monotonic') is not None]);q=np.array([s['position'][1] for s in sv if s.get('planned_monotonic') is not None]);fq=np.array([s['feedback_qpos'][1] for s in sv if s.get('planned_monotonic') is not None]);ax.plot(t-t0,q,color='black',lw=2,label='Recorded planner command');ax.plot(t-t0,fq,'--',color='gray',label='Measured feedback (unshifted)');ax.plot(capture-t0+(np.arange(50)+1)/20,raw[:,1],color='blue',lw=1,label='Raw prediction, gen 31');ax.plot(plan.times-t0,plan.positions[:,1],color='green',lw=2,label='Replay: handoff at protected-prefix end');ax.axvline(arrival-t0,color='purple',ls=':',label='New result available');ax.axvline(end-t0,color='green',ls=':',label='Protected-prefix end');ax.axvline(16.087445673,color='red',ls=':',label='Handoff reconstructed from replay');ax.set_xlim(15.65,16.3);ax.set_ylim(.17,.42);ax.set_xlabel('Seconds since recording began');ax.set_ylabel('Left joint 2 (rad)');ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(Path(__file__).with_name('handoff_counterfactual.png'),dpi=160)
