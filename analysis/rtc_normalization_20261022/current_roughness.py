"""Read-only current deployed-run roughness; simple filter is diagnostic only."""
import json
from pathlib import Path
import numpy as np
J=[0,1,2,3,4,5,7,8,9,10,11,12];out=[]
for tag in ['104131','104306','104416','104534','104653','104728']:
 p=next(Path('deployment_runs').glob('20261022T'+tag+'*'));ms=[json.loads(x) for x in (p/'model_commands.jsonl').open()];values=[]
 for m in ms:
  if not m.get('accepted') or not m.get('action_file'):continue
  a=np.load(p/m['action_file'])['raw_actions'][:,J];d=m['rtc_request']['inference_delay_steps'];sm=a.copy();sm[1:-1]=(a[:-2]+2*a[1:-1]+a[2:])/4
  values.append([np.abs(np.diff(a,n=2,axis=0)).mean(),np.abs(np.diff(a[d:],n=2,axis=0)).mean(),np.abs(np.diff(a[10:],n=2,axis=0)).mean(),np.abs(np.diff(sm[d:],n=2,axis=0)).mean()])
 row=dict(time=tag,n=len(values),mean_abs_d2=dict(zip(['whole','free_suffix','rows10_plus','free_suffix_after_121_filter'],np.mean(values,axis=0).astype(float).tolist())));out.append(row);print(row)
Path(__file__).with_name('current_roughness.json').write_text(json.dumps(out,indent=2)+'\n')
