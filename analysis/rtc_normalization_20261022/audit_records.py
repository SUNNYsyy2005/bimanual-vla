"""Read-only per-joint normalization scale and state-token audit of saved runs."""
import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).parent
s=json.loads((OUT/'norm_stats.json').read_text())['norm_stats'];qlo=np.array(s['state']['q01']);qhi=np.array(s['state']['q99']);alo=np.array(s['actions']['q01']);ahi=np.array(s['actions']['q99']);mask=np.array([i%7!=6 for i in range(14)])
scale=(ahi-alo+1e-6)/2
result=dict(stats_sha256=hashlib.sha256((OUT/'norm_stats.json').read_bytes()).hexdigest(),joint_scales=[],runs=[])
for i in range(14):
 result['joint_scales'].append(dict(index=i,name=('L' if i<7 else 'R')+('gripper' if i%7==6 else str(i%7+1)),state_bin_width=float((qhi[i]-qlo[i]+1e-6)/256),action_physical_units_per_normalized_unit=float(scale[i]),normalized_delta_for_0_02=float(.02/scale[i])))
for tag in ['104131','104306','104416','104534','104653','104728']:
 p=next(Path('deployment_runs').glob('20261022T'+tag+'*'));ms=[json.loads(x) for x in (p/'model_commands.jsonl').open()];errors=[];tokenlist=[];retreat=[];normpre=[]
 for m in ms:
  if not m.get('accepted') or not m.get('action_file'):continue
  f=np.load(p/m['action_file']);state=f['qpos'].astype(float).copy();state[6::7]/=.07
  z=(state-qlo)/(qhi-qlo+1e-6)*2-1;bins=np.digitize(z,np.linspace(-1,1,257)[:-1])-1;tokenlist.append(bins)
  if 'planned_prefix_actions' not in f or len(f['planned_prefix_actions'])<2:retreat.append(False);continue
  pre=f['planned_prefix_actions'];d=len(pre);a=f['raw_actions'];delta=pre-state*np.asarray(mask)
  encoded=(delta-alo)/(ahi-alo+1e-6)*2-1;decoded=(encoded+1)/2*(ahi-alo+1e-6)+alo+state*mask
  errors.append(float(abs(pre-decoded).max()));normpre.extend(encoded.reshape(-1).tolist())
  direction=np.sign(pre[-1,8]-pre[-2,8]);retreat.append(bool(abs(pre[-1,8]-pre[-2,8])>=.001 and np.min((a[d:d+3,8]-pre[-1,8])*direction)<-.01))
 tokens=np.array(tokenlist);same=tokens[1:,8]==tokens[:-1,8];r=np.array(retreat)[1:]
 result['runs'].append(dict(time=tag,accepted=len(tokenlist),right_j2_same_bin_pairs=int(same.sum()),adjacent_pairs=len(same),right_j2_retreat_when_same_bin=dict(count=int(r[same].sum()),total=int(same.sum())),right_j2_retreat_when_changed_bin=dict(count=int(r[~same].sum()),total=int((~same).sum())),prefix_roundtrip_float64_max=max(errors),normalized_prefix_abs_max=float(max(abs(np.array(normpre)))),state_edge_bins_per_dim=np.sum((tokens<0)|(tokens>=255),axis=0).tolist()))
(OUT/'record_audit.json').write_text(json.dumps(result,indent=2)+'\n')
for x in result['joint_scales']:
 if x['index'] in [0,1,7,8,9]:print(x)
for r in result['runs']:print(r['time'],r['right_j2_retreat_when_same_bin'],r['right_j2_retreat_when_changed_bin'])
