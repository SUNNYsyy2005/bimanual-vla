"""Recompute old checkpoint noise correlation from saved predictions; CPU only."""
from pathlib import Path
import json
import numpy as np
import jax
ROOT=Path('/home/sunny/bimanual-vla/analysis');J=[0,1,2,3,4,5,7,8,9,10,11,12];result=[]
for folder,name,preds in [('rtc_padding_fix_20261021','20000.json','20000_predictions'),('no_rtc_comparison_20261021','metrics.json','predictions')]:
 d=json.loads((ROOT/folder/name).read_text());rows=d['rows'] if isinstance(d,dict) else d
 for steps in [5,50]:
  entries=[]
  for r in rows:
   if r['split']!='heldout' or r['mode']!='no_prefix' or r['steps']!=steps:continue
   f=ROOT/folder/preds/f'{r["id"]}_n{steps}_s{r["seed"]}_no_prefix.npz';a=np.load(f)['prediction'][:,J]
   noise=np.asarray(jax.random.normal(jax.random.split(jax.random.key(r['seed']))[1],(1,50,32)))[0,:,J].T
   # NumPy advanced indexing moves J forward: above gives 12x50 before transpose.
   y=np.diff(a,n=2,axis=0);x=np.diff(noise,n=2,axis=0);cs=[];gs=[];res=[]
   assert x.shape==y.shape,(x.shape,y.shape)
   for j in range(12):
    gain=np.cov(y[:,j],x[:,j])[0,1]/np.var(x[:,j],ddof=1);cs.append(np.corrcoef(y[:,j],x[:,j])[0,1]);gs.append(gain)
    res.append(np.mean(abs(y[:,j]-np.mean(y[:,j])-gain*(x[:,j]-np.mean(x[:,j])))))
   entries.append([np.mean(cs),np.mean(gs),np.mean(abs(y)),np.mean(res)])
  result.append(dict(source=folder,steps=steps,n=len(entries),means=dict(zip(['per_joint_d2_noise_correlation','per_joint_gain_rad','mean_abs_d2','mean_abs_d2_after_linear_noise_projection'],np.mean(entries,axis=0).tolist()))))
print(json.dumps(result,indent=2))
