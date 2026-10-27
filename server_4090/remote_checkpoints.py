"""Non-blocking, cached checkpoint inventory on configured Slurm targets."""
import concurrent.futures
import json
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path


def scan_checkpoints(root):
    if isinstance(root, list):
        existing = [item for item in root if Path(item).is_dir()]
        if not existing: raise FileNotFoundError(str(root))
        return sorted([row for item in existing for row in scan_checkpoints(item)], key=lambda row:(row['mtime'],row['step']), reverse=True)
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    rows = []
    for config in root.iterdir():
        if not config.is_dir() or config.name.startswith('.'):
            continue
        for exp in config.iterdir():
            if not exp.is_dir() or exp.name.startswith('.'):
                continue
            marker = config / '.policy_action_conventions' / (exp.name + '.json')
            contract = json.loads(marker.read_text()) if marker.is_file() else {}
            for step in exp.iterdir():
                if not step.name.isdigit() or not (step/'_CHECKPOINT_METADATA').is_file() or not (step/'params/_METADATA').is_file():
                    continue
                datasets = [p.name for p in (step/'assets').iterdir() if p.is_dir()] if (step/'assets').is_dir() else []
                if contract.get('dataset_id') and contract['dataset_id'] not in datasets:
                    datasets.append(contract['dataset_id'])
                rows.append(dict(root=str(root),path=str(step),config_name=config.name,experiment=exp.name,step=int(step.name),
                                 dataset_ids=datasets,action_contract=contract,finetuning=contract.get('finetuning'),
                                 model_variant='pi05' if config.name.startswith('pi05') else 'pi0',
                                 arm_mode=contract.get('arm_mode'),mtime=step.stat().st_mtime,
                                 full_state_available=(step/'train_state/_METADATA').is_file()))
    return sorted(rows,key=lambda r:(r['mtime'],r['step']),reverse=True)


def ssh(host, command, *, timeout=30, input=None):
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,command],
                            input=input,text=True,capture_output=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-1500:])
    return result.stdout


def probe(target):
    import inspect
    code = 'import json\nfrom pathlib import Path\n' + inspect.getsource(scan_checkpoints)
    code += '\nprint(json.dumps(scan_checkpoints(' + repr(list(dict.fromkeys([target['checkpoint_base_dir'], *target.get('checkpoint_read_roots', [])]))) + ')))\n'
    host = target['submit_host']
    python = target.get('inventory_python') or target['openpi_python']
    if target.get('access_mode') != 'slurm_only':
        return json.loads(ssh(host,shlex.quote(python)+' -',input=code))
    # Return small metadata through Slurm's job comment; no shared NAS is
    # required for reading independent H200 filesystems.
    code = code[:code.rindex('print(json.dumps(scan_checkpoints(')]
    code += "\nimport base64, zlib, os, subprocess\n"
    code += "try:\n    payload = {'checkpoints': scan_checkpoints(" + repr(list(dict.fromkeys([target['checkpoint_base_dir'], *target.get('checkpoint_read_roots', [])]))) + ")}\n"
    code += "except Exception as exc:\n    payload = {'error': str(exc)}\n"
    code += "encoded = base64.urlsafe_b64encode(zlib.compress(json.dumps(payload).encode())).decode()\n"
    code += "if len(encoded)>60000: raise ValueError('inventory exceeds Slurm metadata limit')\n"
    code += "subprocess.run(['scontrol','update','JobId='+os.environ['SLURM_JOB_ID'],'Comment=checkpoint_inventory:'+encoded],check=True)\n"
    candidates = [python, *target.get('python_fallbacks', [])]
    choose = 'probe_python=; for candidate in ' + ' '.join(shlex.quote(p) for p in candidates) + '; do if [ -x "$candidate" ]; then probe_python=$candidate; break; fi; done'
    log='/tmp/checkpoint-inventory-'+uuid.uuid4().hex+'.log'
    script = '\n'.join(['#!/bin/bash', '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=1',
                        '#SBATCH --mem=1G','#SBATCH --time=00:03:00',
                        '#SBATCH -p '+target['partition'], '#SBATCH -w '+target['node'],
                        '#SBATCH --output='+log,'#SBATCH --error='+log,
                        choose, "srun --ntasks=1 \"${probe_python:-"+python+"}\" - <<'CHECKPOINT_PROBE'",code,'CHECKPOINT_PROBE',
                        'rc=$?', 'if [ "$rc" -ne 0 ]; then',
                        '  msg=$(tail -c 2000 '+shlex.quote(log)+' | base64 -w0)',
                        '  scontrol update JobId="$SLURM_JOB_ID" Comment="checkpoint_error:$msg"',
                        'fi', 'exit "$rc"'])
    job = ssh(host,'sbatch --parsable',input=script).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('invalid inventory job id: '+job)
    deadline=time.monotonic()+240
    import base64, re, zlib
    while time.monotonic()<deadline:
        description=ssh(host,'scontrol show job -o '+job)
        match=re.search(r'Comment=checkpoint_inventory:([A-Za-z0-9_=\-]+)',description)
        if match:
            payload=json.loads(zlib.decompress(base64.urlsafe_b64decode(match[1])))
            if payload.get('error'): raise RuntimeError(payload['error'])
            return payload['checkpoints']
        error=re.search(r'Comment=checkpoint_error:([A-Za-z0-9+/=]+)',description)
        if error: raise RuntimeError(base64.b64decode(error[1]).decode(errors='replace'))
        if any('JobState='+state in description for state in ('FAILED','CANCELLED','TIMEOUT','COMPLETED')):
            raise RuntimeError('Inventory job '+job+' ended without metadata; node log '+log)
        time.sleep(3)
    ssh(host,'scancel '+job)
    raise TimeoutError('inventory job '+job+' exceeded 240s; retry refresh')



class RemoteCheckpointInventory:
    def __init__(self, targets):
        self.targets=targets
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.lock=threading.Lock()
        self.entries={}

    def snapshot(self, force=False):
        with self.lock:
            for name,target in self.targets.items():
                if not target.get('checkpoint_base_dir'): continue
                row=self.entries.setdefault(name,dict(source=name,label=target.get('label',name),checkpoints=[],state='idle',updated_at=None,error=None))
                if row['state']!='loading' and (force or row['updated_at'] is None or time.time()-row['updated_at']>600):
                    row['state']='loading'
                    self.pool.submit(self._refresh,name,dict(target))
            return json.loads(json.dumps(list(self.entries.values())))

    def _refresh(self,name,target):
        try:
            rows=probe(target)
            result=dict(checkpoints=rows,state='ready',error=None)
        except Exception as exc:
            result=dict(state='error',error=str(exc))
        with self.lock:
            self.entries[name].update(result,updated_at=time.time())

    def selected(self,source,config,experiment,step):
        with self.lock:
            entry=self.entries.get(source,{})
            if entry.get('state')!='ready': raise ValueError('refresh remote checkpoint list before syncing')
            for row in entry['checkpoints']:
                if (row['config_name'],row['experiment'],row['step'])==(config,experiment,step):
                    return dict(row)
        raise ValueError('checkpoint is not in the complete remote inventory')
