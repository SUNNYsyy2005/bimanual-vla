"""Offline parameter sweep on the same recorded targets; never robot I/O."""
import json
from pathlib import Path
import numpy as np
from scripts.analyze_servo_replay import load_segments,replay
OUT=Path('analysis/servo_limits_100ms_20261019')

def window_lags(result):
    q,ref=result['q'],result['reference'];lags=[]
    # Non-overlapping 2 s reference windows, with .5 s output alignment margin.
    for start in range(100,len(q)-250,200):
        r=ref[start:start+200];active=np.std(r,axis=0)>.01
        if not active.any():continue
        scores=[np.mean((q[start+d:start+200+d,active]-r[:,active])**2) for d in range(51)]
        lags.append(int(np.argmin(scores))*10 + 10)  # output is stamped tick+DT
    return lags

def main():
    segments=load_segments(Path('deployment_runs'));OUT.mkdir(parents=True,exist_ok=True)
    configs=[(.6,3,20),(.6,3,60),(.6,3,120),(.6,3,240),(.6,6,120),(.6,6,240)]
    configs += [(v,a,j) for v in (.9,1.2) for a in (3,6,10) for j in (60,120,240)]
    configs += [(v,a,j) for v in (.6,.9) for a in (6,10,15) for j in (480,960)]
    configs += [(.6,10,240),(.6,15,240)]
    configs += [(.75,12,600),(.9,12,600),(.75,15,600),(.9,15,600)]
    records=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
    done={(r['v'],r['a'],r['j']) for r in records}
    for config in configs:
        if config in done:continue
        runs=[];windows=[];failures=[];samples=[]
        for seg in segments:
            r=replay(seg,config)
            if 'failure' in r:failures.append(dict(run=seg['run'],**r));continue
            w=window_lags(r);windows+=w;samples.append(r['samples'])
            runs.append(dict(run=seg['run'],lag_ms=(r['lag_ms']+10 if r['lag_ms'] is not None else None),window_lags_ms=w,bounds_violations=r['bounds_violations']))
        pooled={k:np.concatenate([s[k] for s in samples]) for k in samples[0]} if samples else {}
        lag=[r['lag_ms'] for r in runs if r['lag_ms'] is not None]
        rec=dict(v=config[0],a=config[1],j=config[2],successful_runs=len(runs),failures=failures,
            run_lag_p50_p95_max=np.percentile(lag,[50,95,100]).tolist() if lag else [],
            window_lag_p50_p95_max=np.percentile(windows,[50,95,100]).tolist() if windows else [],
            windows=len(windows),rmse=float(np.sqrt(np.mean(pooled['error']**2))) if pooled else None,
            acceleration_p99=float(np.percentile(abs(pooled['acceleration']),99)) if pooled else None,
            jerk_p99=float(np.percentile(abs(pooled['jerk']),99)) if pooled else None,runs=runs)
        records.append(rec);(OUT/'results.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps({k:v for k,v in rec.items() if k not in ('runs','failures')}),flush=True)
if __name__=='__main__':main()
