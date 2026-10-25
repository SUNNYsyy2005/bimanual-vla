"""CPU-only RTC normalization axis, roundtrip and RMSNorm parity audit.
Run in OpenPI environment with JAX_PLATFORMS=cpu; never loads policy weights.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from openpi import transforms
from openpi.shared import normalize
from openpi.models import gemma
from bimanual_vla.deployment.rtc_policy import _reanchor_normalized_actions
from bimanual_vla.training.openpi_model import install_per_action_adarms

parser=argparse.ArgumentParser();parser.add_argument('--stats-dir',required=True);parser.add_argument('--sample',required=True);args=parser.parse_args()
stats=normalize.load(Path(args.stats_dir));f=np.load(args.sample)
state=f['qpos'].astype(np.float32).copy();state[6::7]/=.07
absolute=f['raw_actions'].astype(np.float32);mask=tuple(i%7!=6 for i in range(14))
encode=transforms.compose([transforms.DeltaActions(mask),transforms.Normalize(stats,use_quantiles=True),transforms.PadStatesAndActions(32)])
policy=SimpleNamespace(_input_transform=encode)
full=encode({'state':state.copy(),'actions':absolute.copy()})['actions']
checks=[]
for d in [6,7,8,10,50]:
 encoded=encode({'state':state.copy(),'actions':absolute[:d].copy()})['actions']
 rtc=_reanchor_normalized_actions(policy,{'state':state.copy()},np.zeros((d,32),np.float32),absolute[:d].copy(),tuple([True]*14))
 decoded=transforms.Unnormalize(stats,use_quantiles=True)({'state':encode({'state':state.copy()})['state'],'actions':rtc})
 decoded=transforms.AbsoluteActions(mask)(decoded)['actions'][:,:14]
 checks.append(dict(length=d,prefix_vs_full_max=float(abs(encoded-full[:d]).max()),rtc_vs_training_max=float(abs(rtc-encoded).max()),absolute_roundtrip_max=float(abs(decoded-absolute[:d]).max()),pad_max=float(abs(rtc[:,14:]).max())))
# Perturbing unrelated steps cannot alter this row's normalized coordinates.
altered=absolute.copy();altered[1:]+=1
other=encode({'state':state.copy(),'actions':altered})['actions'][0]
axis_error=float(abs(other-full[0]).max())

Original=gemma.RMSNorm
x=jax.random.normal(jax.random.key(5),(2,10,16));cond=jax.random.normal(jax.random.key(6),(2,16))
original=Original();params=original.init(jax.random.key(7),x,cond)
params['params']['Dense_0']['kernel']=jax.random.normal(jax.random.key(8),(16,48))*.1
params['params']['Dense_0']['bias']=jax.random.normal(jax.random.key(9),(48,))*.03
install_per_action_adarms();adapted=gemma.RMSNorm();norm=[]
for input_name,input_dtype in [('float32',jnp.float32),('bfloat16',jnp.bfloat16)]:
 for param_name,param_dtype in [('float32',jnp.float32),('bfloat16',jnp.bfloat16)]:
  pp=jax.tree.map(lambda z:z.astype(param_dtype),params);xx=x.astype(input_dtype);cc=cond.astype(input_dtype)
  expected=original.apply(pp,xx,cc);actual=adapted.apply(pp,xx,cc)
  token_cond=jnp.broadcast_to(cc[:,None,:],(2,10,16));token=adapted.apply(pp,xx,token_cond)
  altered_x=xx.at[:,1:,:].multiply(10);altered_cond=token_cond.at[:,1:,:].multiply(3)
  isolated=adapted.apply(pp,altered_x,altered_cond)
  diff=np.asarray(actual[0],np.float32)-np.asarray(expected[0],np.float32)
  norm.append(dict(input_dtype=input_name,param_dtype=param_name,original_output_dtype=str(expected[0].dtype),adapted_output_dtype=str(actual[0].dtype),stock_difference_max=float(abs(diff).max()),stock_difference_rms=float(np.sqrt(np.mean(diff**2))),broadcast_vs_token_max=float(np.max(abs(np.asarray(token[0],np.float32)-np.asarray(actual[0],np.float32)))),other_tokens_influence_on_first=float(np.max(abs(np.asarray(isolated[0][:,0],np.float32)-np.asarray(token[0][:,0],np.float32))))))
# Plain (unconditioned language) norm also needs dtype parity.
plain_params=original.init(jax.random.key(10),x,None);plain=[]
for dtype in [jnp.float32,jnp.bfloat16]:
 xx=x.astype(dtype);a=adapted.apply(plain_params,xx,None)[0];b=original.apply(plain_params,xx,None)[0]
 plain.append(dict(input_dtype=str(xx.dtype),original_output_dtype=str(b.dtype),adapted_output_dtype=str(a.dtype),max_difference=float(np.max(abs(np.asarray(a,np.float32)-np.asarray(b,np.float32))))))
print(json.dumps(dict(backend=jax.default_backend(),norm_shapes={k:{n:list(np.asarray(getattr(v,n)).shape) for n in ['mean','std','q01','q99']} for k,v in stats.items()},action_roundtrips=checks,other_rows_influence_on_action=axis_error,adaptive_norm=norm,plain_norm=plain),indent=2))
