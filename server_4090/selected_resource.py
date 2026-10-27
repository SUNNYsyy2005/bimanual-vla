"""Bounded, direct checks for the selected resource; no global inventory refresh."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def inspect_checkpoint(path, max_entries=4096, budget_seconds=.25):
    path=Path(path); started=time.monotonic()
    if not path.exists(): return dict(path=str(path),exists=False,available=False,checked_at=time.time(),reason='文件已不存在')
    resolved=path.resolve()
    orbax=(resolved/'params').is_dir() and (resolved/'_CHECKPOINT_METADATA').is_file()
    foundation=(resolved/'params').is_dir() and any((resolved/'params'/k).is_file() for k in ('_METADATA','manifest.ocdbt','_CHECKPOINT_METADATA'))
    weights=[resolved] if resolved.is_file() else [resolved/k for k in ('model.safetensors','full_weights.pt','model_state_dict/full_weights.pt','actor/model_state_dict/full_weights.pt')]
    weight=next((p for p in weights if p.is_file() and p.stat().st_size>0),None)
    total=0;entries=0;pending=[resolved];complete=True;links=0
    while pending:
        current=pending.pop()
        if entries>=max_entries or time.monotonic()-started>budget_seconds: complete=False;break
        if current.is_symlink(): links+=1;continue
        if current.is_file(): total+=current.stat().st_size;entries+=1;continue
        with os.scandir(current) as iterator:
            for entry in iterator:
                entries+=1
                if entries>max_entries or time.monotonic()-started>budget_seconds: complete=False;break
                if entry.is_symlink(): links+=1;continue
                if entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))
                else:total+=entry.stat(follow_symlinks=False).st_size
        if not complete:break
    return dict(path=str(path),resolved_path=str(resolved),exists=True,available=bool(orbax or foundation or weight),
                format='orbax' if orbax or foundation else 'pytorch' if weight else 'unknown',checked_at=time.time(),
                size_bytes=total if complete and not links else None,observed_bytes=total,size_complete=complete and not links,
                entries_checked=entries,links=links,mtime=resolved.stat().st_mtime,verification='关键文件与文件状态；未加载权重或校验全部内容')


def job_status(host, job_id):
    job_id=str(job_id)
    if not job_id.isdigit(): raise ValueError('Invalid Slurm job identity')
    command=shlex.join(['squeue','-h','-j',job_id,'-o','%i|%T'])
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=5',host,command],capture_output=True,text=True,timeout=8)
    if result.returncode and 'Invalid job id' not in result.stderr: raise RuntimeError('无法核对所选 Slurm 作业状态')
    if result.stdout.strip():state=result.stdout.strip().splitlines()[0].split('|')[-1]
    else:
        command=shlex.join(['sacct','-n','-X','-j',job_id,'--format=State','--parsable2'])
        result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=5',host,command],capture_output=True,text=True,timeout=8)
        if result.returncode: raise RuntimeError('无法核对所选 Slurm 作业终态')
        state=next((line.strip().split('|')[0].split()[0] for line in result.stdout.splitlines() if line.strip()),'UNKNOWN')
    mapping={'COMPLETED':'completed','FAILED':'failed','CANCELLED':'stopped','TIMEOUT':'failed','OUT_OF_MEMORY':'failed','RUNNING':'running','PENDING':'queued','COMPLETING':'running','PREEMPTED':'stopped','NODE_FAIL':'failed'}
    return dict(job_id=job_id,slurm_state=state,state=mapping.get(state,'unknown'),checked_at=time.time())


def evaluation_norm_asset(checkpoint, dataset_id):
    """Evaluate another dataset using the checkpoint's training normalization."""
    assets=Path(checkpoint)/'assets'
    if (assets/dataset_id/'norm_stats.json').is_file():return dataset_id
    candidates=[]
    if assets.is_dir():
        for path in assets.iterdir():
            if len(candidates)>32:raise ValueError('Too many checkpoint normalization assets; choose an explicit compatible checkpoint')
            if (path/'norm_stats.json').is_file():candidates.append(path.name)
    if len(candidates)==1:return candidates[0]
    raise ValueError('Cannot determine checkpoint training normalization for this dataset; no unique norm_stats asset')
