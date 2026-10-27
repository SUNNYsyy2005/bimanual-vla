"""Within-generation replay alignment and historical growing tracking errors."""
import json
from pathlib import Path
import numpy as np
from scripts.analyze_servo_replay import load_segments,replay,JOINTS,DT
OUT=Path('analysis/intra_chunk_tracking_20261019')

def evaluate(segment,result):
    # Compare output timestamp to the most recent causal target at THAT time.
    t=result['times'];q=result['q']
    index=np.searchsorted(segment['times'],t,side='right')-1
    gen=segment['generations'][index];ref=segment['targets'][index]
    starts=np.r_[0,np.flatnonzero(np.diff(segment['generations']))+1]
    start_for_event=np.maximum.accumulate(np.where(np.isin(np.arange(len(segment['times'])), starts),np.arange(len(segment['times'])),0))
    age=t-segment['times'][start_for_event[index]]
    interior=(t>=1.)&(age>=.05)
    boundary=(t>=1.)&(age<.05)
    error=q-ref
    same3=(gen[2:]==gen[:-2])&interior[2:]
    same4=(gen[3:]==gen[:-3])&interior[3:]
    return dict(error=error[interior],boundary_error=error[boundary],
        acceleration=np.diff(q,n=2,axis=0)[same3]/DT**2,
        jerk=np.diff(q,n=3,axis=0)[same4]/DT**3)

def historical():
    episodes=[];all_errors=[];pairs=0
    for path in sorted(Path('deployment_runs').glob('20261008*/trajectory.jsonl')):
        segments=[];current=[];prev=None
        for line in path.open():
            r=json.loads(line)
            valid=r.get('command_sent') and not r.get('command_hold') and len(r.get('command_joints_rad') or [])==12
            if not valid:
                if current:segments.append(current)
                current=[];prev=None;continue
            q=np.array(r['qpos'],float)[JOINTS];target=np.array(r['command_action'],float)[JOINTS];sent=np.array(r['command_joints_rad'],float)
            if not all(np.isfinite(a).all() for a in (q,target,sent)):prev=None;continue
            t=r['monotonic_timestamp']
            if prev is not None and 0<t-prev['ct']<=.125:
                current.append(dict(t=t,q=q,target=prev['target'],sent=prev['sent'],gen=prev['gen']))
            else:
                if current:segments.append(current)
                current=[]
            prev=dict(ct=r['command_monotonic_timestamp'],target=target,sent=sent,gen=r['command_generation'])
        if current:segments.append(current)
        for rows in segments:
            ts=np.array([r['t'] for r in rows]);q=np.array([r['q'] for r in rows]);target=np.array([r['target'] for r in rows]);sent=np.array([r['sent'] for r in rows]);gens=np.array([r['gen'] for r in rows]);err=target-q
            pairs+=len(rows);all_errors.extend(np.max(abs(err),axis=1).tolist())
            hits={}
            for b in range(len(rows)):
                e=np.searchsorted(ts,ts[b]+.5)
                if e>=len(rows) or e-b<8 or ts[e]-ts[b]>.65:continue
                for j in range(12):
                    sign=np.sign(err[b,j]);v=sign*err[b:e+1,j];dv=np.diff(v)
                    # Persistent signed lag, net +.05rad over >=.5s; >80% nondecreasing
                    # allowing .002rad noise, and final lag>.1rad.
                    if v[0]<.02 or v[-1]<.1 or v[-1]-v[0]<.05 or np.mean(dv>=-.002)<.8:continue
                    motion=sign*(target[e,j]-target[b,j])
                    if motion<.05:continue
                    hits.setdefault(j,[]).append((b,e))
            for j,intervals in hits.items():
                merged=[]
                for b,e in intervals:
                    if merged and b<=merged[-1][1]:merged[-1][1]=max(e,merged[-1][1])
                    else:merged.append([b,e])
                for b,e in merged:
                    er=err[b:e+1,j];sg=np.sign(er[0]);later=np.flatnonzero((ts>ts[e])&(ts<=ts[e]+2))
                    episodes.append(dict(run=path.parent.name,joint_index=JOINTS[j],start_monotonic=float(ts[b]),end_monotonic=float(ts[e]),duration_s=float(ts[e]-ts[b]),
                        generations=int(len(np.unique(gens[b:e+1]))),start_error_rad=float(abs(er[0])),end_error_rad=float(abs(er[-1])),peak_error_rad=float(abs(er).max()),
                        target_travel_rad=float(sg*(target[e,j]-target[b,j])),feedback_travel_rad=float(sg*(q[e,j]-q[b,j])),
                        end_shaping_gap_rad=float(sg*(target[e,j]-sent[e,j])),end_sent_feedback_gap_rad=float(sg*(sent[e,j]-q[e,j])),
                        next_2s_min_abs_error_rad=float(abs(err[later,j]).min()) if len(later) else None))
    return dict(comparisons=pairs,error_max_joint_p50_p95_p99=np.percentile(all_errors,[50,95,99]).tolist(),episodes=sorted(episodes,key=lambda r:r['peak_error_rad'],reverse=True))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    h=historical();(OUT/'historical_growth.json').write_text(json.dumps(h,indent=2)+'\n')
    print('historical growing episodes',len(h['episodes']),json.dumps(h['episodes'][:3]),flush=True)
    segs=load_segments(Path('deployment_runs'))
    configs=[(.45,2,20),(.6,2,20),(.6,3,20),(.6,3,60),(.6,3,120),(.6,3,240),(.6,4,60),(.6,4,120),(.6,6,120),(.6,6,240),(.6,3,60,'zero_velocity')]
    records=[]
    for config in configs:
        per=[];fail=[]
        for seg in segs:
            r=replay(seg,config)
            if 'failure' in r:fail.append(dict(run=seg['run'],**r));continue
            per.append(evaluate(seg,r))
        pooled={k:np.concatenate([r[k] for r in per]) for k in per[0]}
        error=pooled['error'];peak=np.max(abs(error),axis=1)
        record=dict(config=config,successful_runs=len(per),failures=fail,interior_samples=len(error),
            rmse_rad=float(np.sqrt(np.mean(error**2))),
            max_joint_error_p50_p95_p99_max=np.percentile(peak,[50,95,99,100]).tolist(),
            percent_max_error_over_005=float(np.mean(peak>.05)*100),
            boundary_max_joint_error_p95=float(np.percentile(np.max(abs(pooled['boundary_error']),axis=1),95)),
            acceleration_p99=float(np.percentile(abs(pooled['acceleration']),99)),
            jerk_p99=float(np.percentile(abs(pooled['jerk']),99)))
        records.append(record);(OUT/'replay.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k!='failures'}),flush=True)
if __name__=='__main__':main()
