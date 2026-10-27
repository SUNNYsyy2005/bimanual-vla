"""Copy one finalized checkpoint using the existing verified transfer runners."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request-json',required=True)
    args=parser.parse_args()
    request=json.loads(base64.urlsafe_b64decode(args.request_json))
    source,target=request['source'],request['target']
    step=str(request['step'])
    marker=Path(request['marker_path'])
    contract=request['contract']
    if marker.exists() and json.loads(marker.read_text())!=contract:
        raise ValueError('destination experiment action contract differs; choose a separate experiment')
    encoded=lambda value:base64.urlsafe_b64encode(json.dumps(value).encode()).decode()
    runner='slurm_dataset_sync_runner.py' if source.get('kind')=='slurm_only' else 'dataset_transfer_runner.py'
    cmd=[sys.executable,str(Path(__file__).with_name(runner)),'--dataset-id',step,
         '--source-json',encoded(source),'--target-json',encoded(target),
         '--parallelism',str(request['parallelism']),'--skip-existing']
    if source.get('kind')=='slurm_only' and request.get('nas_staging_root'):
        cmd+=['--nas-staging-root',request['nas_staging_root']]
    subprocess.run(cmd,check=True)
    dest=Path(target['dataset_root'])/step
    if not (dest/'_CHECKPOINT_METADATA').is_file() or not (dest/'params/_METADATA').is_file():
        raise ValueError('transferred checkpoint is incomplete')
    marker.parent.mkdir(parents=True,exist_ok=True)
    temp=marker.with_suffix('.incoming.json');temp.write_text(json.dumps(contract,indent=2)+'\n');temp.replace(marker)
    print('[dashboard] checkpoint and action contract ready for policy:',dest,flush=True)

if __name__=='__main__': main()
