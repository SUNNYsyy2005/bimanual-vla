"""Audit historical first-sent action indexes; request RTC payload is not recorded."""
import json, math
from pathlib import Path
import numpy as np
from collections import Counter
OUT=Path('analysis/rtc_alignment_20261019')
def main():
 records=[]
 for path in sorted(Path('deployment_runs').glob('20261008*/model_commands.jsonl')):
  trajectory=path.parent/'trajectory.jsonl'
  if not trajectory.exists():continue
  first={}
  for l in trajectory.open():
   r=json.loads(l)
   if r.get('command_sent') and not r.get('command_hold'):first.setdefault(r['command_generation'],r)
  meta=json.load((path.parent/'metadata.json').open());h=meta.get('rtc_execution_horizon')
  completed=[]
  for l in path.open():
   r=json.loads(l)
   if r.get('captured_monotonic') is None or r.get('arrived_monotonic') is None:continue
   cap=r['captured_monotonic'];arrival=r['arrived_monotonic'];lat=arrival-cap
   prior=[x for x in completed if x[0]<=r['launched_monotonic']]
   pred=math.ceil(prior[-1][1]*20) if prior else None
   completed.append((arrival,lat))
   if not r.get('accepted') or r['generation'] not in first:continue
   f=first[r['generation']];age=f['command_monotonic_timestamp']-cap
   records.append(dict(run=path.parent.name,generation=r['generation'],horizon=h,capture_to_admission_ms=lat*1000,capture_to_first_send_ms=age*1000,first_index=f['command_queue_index'],time_only_index=max(0,math.ceil(age*20)-1),estimated_delay_from_previous_completion=pred,actual_admission_steps=math.ceil(lat*20)))
 def summ(rs):
  return dict(n=len(rs),first_index_counts=dict(Counter(r['first_index'] for r in rs)),horizon_counts=dict(Counter(r['horizon'] for r in rs)),capture_to_admission_ms=np.percentile([r['capture_to_admission_ms'] for r in rs],[50,95,99]).tolist(),capture_to_first_send_ms=np.percentile([r['capture_to_first_send_ms'] for r in rs],[50,95,99]).tolist(),first_index_ge6=sum(r['first_index']>=6 for r in rs),first_index_ge8=sum(r['first_index']>=8 for r in rs),first_index_less_than_time_only=sum(r['first_index']<r['time_only_index'] for r in rs),previous_estimate_under_by2_or_more=sum(r['estimated_delay_from_previous_completion'] is not None and r['actual_admission_steps']-r['estimated_delay_from_previous_completion']>=2 for r in rs))
 latest=max(r['run'] for r in records)
 result=dict(all=summ(records),latest_run=latest,latest=summ([r for r in records if r['run']==latest]),records=records)
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
