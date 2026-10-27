"""Non-blocking, cached checkpoint inventory on configured Slurm targets."""
import concurrent.futures
import json
import shlex
import subprocess
import threading
import time
import uuid
import hashlib
import os
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


def publish_inventory(payload):
    """Send metadata within Slurm's small Comment limit, with acknowledgments."""
    import base64, os, subprocess, time, zlib
    encoded=base64.urlsafe_b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    job=os.environ['SLURM_JOB_ID']
    chunks=[encoded[i:i+700] for i in range(0,len(encoded),700)]
    for index,chunk in enumerate(chunks):
        comment=f'checkpoint_part:{index}:{len(chunks)}:{chunk}'
        subprocess.run(['scontrol','update','JobId='+job,'Comment='+comment],check=True)
        deadline=time.monotonic()+50
        while time.monotonic()<deadline:
            description=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,check=True).stdout
            if 'Comment=checkpoint_ack:'+str(index) in description.split():break
            time.sleep(.2)
        else:raise TimeoutError('inventory metadata acknowledgment timed out')


def probe(target):
    import inspect
    code = 'import json\nfrom pathlib import Path\n' + inspect.getsource(scan_checkpoints)
    code += '\nprint(json.dumps(scan_checkpoints(' + repr(list(dict.fromkeys([target['checkpoint_base_dir'], *target.get('checkpoint_read_roots', [])]))) + ')))\n'
    host = target['submit_host']
    python = target.get('inventory_python') or target['openpi_python']
    if target.get('access_mode') != 'slurm_only':
        return json.loads(ssh(host,shlex.quote(python)+' -',input=code))
    try:
        from .cluster_read_guard import admit_read
    except ImportError:
        from cluster_read_guard import admit_read
    admit_read(target,'checkpoint_inventory_')
    # Return small metadata through Slurm's job comment; no shared NAS is
    # required for reading independent H200 filesystems.
    code = code[:code.rindex('print(json.dumps(scan_checkpoints(')]
    code += "\nimport base64, zlib, os, subprocess\n"
    code += "try:\n    payload = {'checkpoints': scan_checkpoints(" + repr(list(dict.fromkeys([target['checkpoint_base_dir'], *target.get('checkpoint_read_roots', [])]))) + ")}\n"
    code += "except Exception as exc:\n    payload = {'error': str(exc)}\n"
    code += '\n'+inspect.getsource(publish_inventory)+'\npublish_inventory(payload)\n'
    candidates = [python, *target.get('python_fallbacks', [])]
    choose = 'probe_python=; for candidate in ' + ' '.join(shlex.quote(p) for p in candidates) + '; do if [ -x "$candidate" ]; then probe_python=$candidate; break; fi; done'
    log='/tmp/checkpoint-inventory-'+uuid.uuid4().hex+'.log'
    script = '\n'.join(['#!/bin/bash', '#SBATCH --job-name=checkpoint_inventory_'+uuid.uuid4().hex[:8], '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=1',
                        '#SBATCH --mem=1G','#SBATCH --time=00:03:00',
                        '#SBATCH -p '+target['partition'], '#SBATCH -w '+target['node'],
                        '#SBATCH --output='+log,'#SBATCH --error='+log,
                        choose, "srun --ntasks=1 \"${probe_python:-"+python+"}\" - <<'CHECKPOINT_PROBE'",code,'CHECKPOINT_PROBE',
                        'rc=$?', 'if [ "$rc" -ne 0 ]; then',
                        '  msg=$(tail -c 500 '+shlex.quote(log)+' | base64 -w0)',
                        '  scontrol update JobId="$SLURM_JOB_ID" Comment="checkpoint_error:$msg"',
                        'fi', 'exit "$rc"'])
    job = ssh(host,'sbatch --parsable',input=script).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('invalid inventory job id: '+job)
    try:
        deadline=time.monotonic()+60
        import base64, re, zlib
        chunks=[]
        while time.monotonic()<deadline:
            description=ssh(host,'scontrol show job -o '+job)
            part=re.search(r'Comment=checkpoint_part:(\d+):(\d+):([A-Za-z0-9_=\-]+)',description)
            if part:
                index,total=int(part[1]),int(part[2])
                if total<1 or total>1000 or index>len(chunks):raise ValueError('invalid inventory metadata sequence')
                if index==len(chunks):chunks.append(part[3])
                ssh(host,'scontrol update JobId='+job+' Comment=checkpoint_ack:'+str(index))
                if len(chunks)==total:
                    payload=json.loads(zlib.decompress(base64.urlsafe_b64decode(''.join(chunks))))
                    if payload.get('error'):raise RuntimeError(payload['error'])
                    return payload['checkpoints']
                continue
            match=re.search(r'Comment=checkpoint_inventory:([A-Za-z0-9_=\-]+)',description)
            if match:
                payload=json.loads(zlib.decompress(base64.urlsafe_b64decode(match[1])))
                if payload.get('error'): raise RuntimeError(payload['error'])
                return payload['checkpoints']
            error=re.search(r'Comment=checkpoint_error:([A-Za-z0-9+/=]+)',description)
            if error: raise RuntimeError(base64.b64decode(error[1]).decode(errors='replace'))
            if any('JobState='+state in description for state in ('FAILED','CANCELLED','TIMEOUT','COMPLETED')):
                raise RuntimeError('Inventory job '+job+' ended without metadata; node log '+log)
            time.sleep(.5)
        raise TimeoutError('inventory job '+job+' exceeded 60s; retry refresh')
    finally:
        try:ssh(host,'scancel '+job,timeout=15)
        except Exception:pass



class RemoteCheckpointInventory:
    def __init__(self, targets):
        self.targets=targets
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.lock=threading.Lock()
        self.entries={}
        identity=json.dumps(targets,sort_keys=True,default=str).encode()
        self.cache_path=Path.home()/'.cache/bimanual-vla/checkpoints'/ (hashlib.sha256(identity).hexdigest()[:16]+'.json')
        try:
            cached=json.loads(self.cache_path.read_text())
            self.entries={name:row for name,row in cached.items() if name in targets}
            for row in self.entries.values():
                row['state']='ready'
                row['cached']=True
        except (OSError,ValueError,TypeError):pass

    def snapshot(self, force=False, source=None):
        with self.lock:
            for name,target in self.targets.items():
                if not target.get('checkpoint_base_dir'): continue
                row=self.entries.setdefault(name,dict(source=name,label=target.get('label',name),checkpoints=[],state='idle',updated_at=None,error=None))
                # A browser can issue several refreshes while opening a dialog.
                # Reuse the completed metadata for 30s; never submit duplicate jobs.
                fresh=time.time()-(row.get('updated_at') or 0)<30
                if force and name == source and row['state']!='loading' and not fresh:
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
            if result.get('state')=='ready':
                self.cache_path.parent.mkdir(parents=True,exist_ok=True)
                tmp=self.cache_path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
                tmp.write_text(json.dumps(self.entries))
                os.replace(tmp,self.cache_path)

    def selected(self,source,config,experiment,step):
        with self.lock:
            entry=self.entries.get(source,{})
            # Cached rows remain selectable during a background refresh. The
            # export worker checks completeness and pins the actual files.
            if not entry.get('updated_at'): raise ValueError('refresh remote checkpoint list before syncing')
            for row in entry['checkpoints']:
                if (row['config_name'],row['experiment'],row['step'])==(config,experiment,step):
                    return dict(row)
        raise ValueError('checkpoint is not in the complete remote inventory')
