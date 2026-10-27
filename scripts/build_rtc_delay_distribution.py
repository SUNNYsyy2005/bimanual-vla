"""Build a reproducible training delay distribution from recorded first-sent targets."""
import json,math
from collections import Counter
from pathlib import Path

def main():
 rows=json.loads(Path('analysis/rtc_alignment_20261019/results.json').read_text())['records']
 maximum=10;counts=Counter();tail=0;cold=0
 for r in rows:
  if r['first_index']==0:
   # No source prefix skipped: retain a zero-prefix training case for startup/recovery.
   step=0;cold+=1
  else:
   step=math.ceil(r['capture_to_first_send_ms']/50)
   tail+=int(step>maximum)
  counts[min(step,maximum)]+=1
 weights=[counts[i] for i in range(maximum+1)]
 result=dict(source='analysis/rtc_alignment_20261019/results.json',sample_count=len(rows),action_hz=20,
  rtc_training_max_delay=maximum,rtc_training_delay_weights=weights,
  probabilities=[x/len(rows) for x in weights],zero_prefix_samples=cold,
  tail_clipped_samples=tail,tail_clipped_fraction=tail/len(rows),
  note='First index zero maps to zero-prefix startup/recovery. Other samples use ceil(capture-to-first-send/50ms); values above10 map to10. This preserves historical latency, not a promise for the optimized pipeline.')
 out=Path('analysis/fix_rtc_training_20261019');out.mkdir(parents=True,exist_ok=True)
 (out/'delay_distribution.json').write_text(json.dumps(result,indent=2)+'\n')
 payload=dict(execution_target='h100',cluster_gpus=1,dataset_id='fix_nopad_20261019',exp_name='fix_nopad_rtc_empirical_d10_20261019',mode='new',model_variant='pi05',
  batch_size=8,fsdp_devices=1,num_workers=4,num_train_steps=20000,save_interval=1000,keep_period=5000,test_ratio=.1,split_seed=42,
  action_expert_tuning='full',llm_lora=False,image_resize_mode='stretch',rtc_training=True,rtc_training_max_delay=maximum,rtc_training_delay_weights=weights,
  norm_batch_size=16,norm_num_workers=2,eval_enabled=True,eval_interval_steps=5000,eval_batch_size=1,eval_num_workers=2,eval_max_batches=50,eval_seed=42,wandb_enabled=False)
 (out/'train_request.json').write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
