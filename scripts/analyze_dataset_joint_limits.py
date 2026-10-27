"""Offline joint derivative statistics; smooth estimates are retrospective, not a controller."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

ROOT = Path('episodes_piper_v21/fix')
OUT = Path('analysis/dataset_limits_20261019')
JOINTS = [0,1,2,3,4,5,7,8,9,10,11,12]

def summarize(v, a):
    return dict(velocity_abs_percentiles=np.percentile(abs(v), [50,95,99,99.9,100]).tolist(),
                acceleration_abs_percentiles=np.percentile(abs(a), [50,95,99,99.9,100]).tolist(),
                joint_velocity_above_03_percent=float(np.mean(abs(v)>.3)*100),
                joint_acceleration_above_08_percent=float(np.mean(abs(a)>.8)*100),
                frame_any_velocity_above_03_percent=float(np.mean(np.any(abs(v)>.3,axis=1))*100),
                frame_any_acceleration_above_08_percent=float(np.mean(np.any(abs(a)>.8,axis=1))*100),
                frame_any_acceleration_above_2_percent=float(np.mean(np.any(abs(a)>2,axis=1))*100),
                per_joint_velocity_p99=np.percentile(abs(v),99,axis=0).tolist(),
                per_joint_acceleration_p99=np.percentile(abs(a),99,axis=0).tolist(),
                braking_acceleration_p99=float(np.percentile(abs(a[v*a<0]),99)))

def main():
    pooled={}; runs=[]
    for path in sorted(ROOT.glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            t=d['timestamps'].copy();t-=t[0]
            if not np.all(np.diff(t)>0): raise ValueError(f'nonmonotonic timestamps: {path}')
            record=dict(run=path.stem,frames=len(t),duration_s=float(t[-1]),dt_min=float(np.diff(t).min()),dt_max=float(np.diff(t).max()),success=bool(d['success']))
            record['state_action_joint_max_difference']=float(np.max(abs(d['state'][:,JOINTS]-d['actions'][:,JOINTS])))
            runs.append(record)
            for source in ('state','actions'):
                q=d[source][:,JOINTS].astype(float)
                if not np.isfinite(q).all(): raise ValueError(path)
                # Raw backward interval differences located at interval midpoints.
                v=np.diff(q,axis=0)/np.diff(t)[:,None]
                a=np.diff(v,axis=0)/((np.diff(t)[1:]+np.diff(t)[:-1])/2)[:,None]
                pooled.setdefault(source+'_raw',[]).append(((v[1:]+v[:-1])/2,a))
                # Uniform interpolation then centered polynomial derivatives; exclude edges.
                dt=float(np.median(np.diff(t))); grid=np.arange(0,t[-1],dt)
                uniform=np.column_stack([np.interp(grid,t,col) for col in q.T])
                for w in (5,9):
                    v=savgol_filter(uniform,w,3,deriv=1,delta=dt,axis=0)
                    a=savgol_filter(uniform,w,3,deriv=2,delta=dt,axis=0)
                    edge=w//2
                    pooled.setdefault(source+f'_sg{w}',[]).append((v[edge:-edge],a[edge:-edge]))
    result=dict(source=str(ROOT),joint_indices=JOINTS,percentile_order=[50,95,99,99.9,100],runs=runs,metrics={})
    for name, pairs in pooled.items():
        result['metrics'][name]=summarize(np.concatenate([p[0] for p in pairs]),np.concatenate([p[1] for p in pairs]))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('runs',len(runs),'frames',sum(r['frames'] for r in runs),'seconds',sum(r['duration_s'] for r in runs))
    for name,s in result['metrics'].items(): print(name,json.dumps(s))
if __name__=='__main__': main()
