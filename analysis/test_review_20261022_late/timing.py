"""Read-only timing and same-clock reference audit of completed local recordings."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'test_review_20261021'))
from review import read, stat, J, audit

OUT = Path(__file__).parent
results = []
for tag in ['104131', '104306', '104416', '104534', '104653', '104728', '105028', '105113']:
    p = next(Path('deployment_runs').glob('20261022T' + tag + '*'))
    tr, ms, sv = [read(p / name) for name in ['trajectory.jsonl', 'model_commands.jsonl', 'servo_commands.jsonl']]
    tt = np.array([r['monotonic_timestamp'] for r in tr]); t0 = tt[0]
    models = {m['generation']: m for m in ms}
    raw = {m['generation']: np.load(p / m['action_file'])['raw_actions'] for m in ms if m.get('action_file')}
    ct = np.array([max(x['joint_ctrl_monotonic'] for x in s['sides'].values()) for s in sv])
    ti = np.clip(np.searchsorted(tt, ct, side='right') - 1, 0, len(tt)-1)
    valid = np.array([not s.get('braking') and bool(s.get('plan_id')) and tr[i]['command_sent'] and not tr[i]['command_hold'] for s,i in zip(sv,ti)]) & (ct <= tt[-1])
    plans, first, rows = {}, {}, []
    accepted = [m for m in ms if m.get('accepted') and m.get('action_file')]
    arrivals = np.array([m['arrived_monotonic'] for m in accepted])
    for s,t,ok in zip(sv,ct,valid):
        pt = s.get('plan_timing') or {}
        if pt: plans.setdefault(pt['generation'], pt)
        if not ok: continue
        g = s['context']['generation']; m = models[g]
        first.setdefault(g, dict(generation=g, at_s=t-t0, index=s['context']['source_index'], wait_ms=(t-m['arrived_monotonic'])*1000, total_ms=(t-m['captured_monotonic'])*1000))
        k = np.searchsorted(arrivals, t, side='right')-1
        if k < 0: continue
        latest = accepted[k]; a = raw[latest['generation']]
        source = (t-latest['captured_monotonic'])*20-1
        if not 0 <= source < len(a)-1: continue
        lo = int(source); ref = a[lo] + (source-lo)*(a[lo+1]-a[lo])
        cmd = np.array(s['command_qpos']); fb = np.array(s['feedback_qpos'])
        rows.append(dict(at_s=t-t0, generation=g, latest=latest['generation'], gap=float(max(abs(cmd[J]-ref[J]))), track=float(max(abs(cmd[J]-fb[J]))),
                         latest_ref=ref.tolist(),command=cmd.tolist(),feedback=fb.tolist(),velocity=s['velocity']))
    result = dict(time=tag, first=list(first.values()), plans=list(plans.values()))
    for key in ['wait_ms','total_ms','index']:
        result[key] = stat([r[key] for r in first.values()])
    for key in ['compile_ms','splice_compute_ms']:
        result[key] = stat([r[key] for r in plans.values() if key in r])
    result['scheduled_wait_ms'] = stat([(r['splice_monotonic']-models[g]['arrived_monotonic'])*1000 for g,r in plans.items()])
    result['wait_ge300'] = int(sum(r['wait_ms'] >= 300 for r in first.values()))
    result['capture_to_launch_ms'] = stat([(m['launched_monotonic']-m['captured_monotonic'])*1000 for m in ms])
    result['rpc_ms'] = stat([m.get('_client_transport_timing',{}).get('client_transport_total_ms',np.nan) for m in ms])
    result['server_policy_ms'] = stat([m.get('transport_timing',{}).get('server_policy_ms',np.nan) for m in ms])
    result['camera_at_launch_age_ms'] = stat([(m['launched_at']-v)*1000 for m in ms for v in m.get('image_timestamps',{}).values()])
    result['windows'] = []
    for start,end in [(3,10),(10,18),(18,28),(28,35),(35,55)]:
        w=[r for r in rows if start<=r['at_s']<end]
        if not w:continue
        result['windows'].append(dict(start=start,end=end,same_clock_model_to_command=stat([r['gap'] for r in w]),command_to_feedback=stat([r['track'] for r in w]),
            median_grip_model=np.median([np.array(r['latest_ref'])[[6,13]] for r in w],axis=0).tolist(),median_grip_command_m=np.median([np.array(r['command'])[[6,13]] for r in w],axis=0).tolist()))
    result['audit']=audit(p)
    (OUT/f'samples_{tag}.json').write_text(json.dumps(rows))
    results.append(result)
    print(tag, 'done', len(rows), flush=True)
(OUT/'timing_results.json').write_text(json.dumps(results,indent=2)+'\n')
