"""Discover owned training processes without walking datasets, weights or histories."""
import os
from pathlib import Path
import time


def local_training_processes(config, known_pids):
    root=Path(config['openpi_repo']).resolve()
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in known_pids:continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            command=[p.decode(errors='replace') for p in (proc/'cmdline').read_bytes().split(b'\0') if p]
            if len(command)<3:continue
            cwd=(proc/'cwd').resolve()
            entry=Path(command[1]);entry=(cwd/entry).resolve() if not entry.is_absolute() else entry.resolve()
            # Only configured OpenPI entrypoints, never arbitrary similarly named processes.
            is_train=(entry in {root/'scripts/train.py',root/'scripts/train_pytorch.py'} or entry==Path(__file__).resolve().with_name('openpi_single_arm.py') and 'train' in command[2:4])
            if not is_train:continue
            log=None
            for fd in ('1','2'):
                target=os.readlink(proc/'fd'/fd)
                if target.startswith('/') and Path(target).is_file() and Path(target).suffix in {'.log','.out','.err'}:
                    log=target;break
            start=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
            def arg(name):
                for i,value in enumerate(command):
                    if value==name and i+1<len(command):return command[i+1]
                    if value.startswith(name+'='):return value.split('=',1)[1]
            result.append(dict(id='train-external-'+proc.name+'-'+start,type='train',state='running',pid=int(proc.name),command=command,
                               created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),metadata=dict(external=True,external_training=True,runtime='external_4x4090',
                               execution_target='local_4090',external_log_path=log,exp_name=arg('--exp-name') or arg('--exp_name') or proc.name,
                               dataset_id=arg('--dataset-id') or arg('--dataset_id'),framework='openpi')))
        except (OSError,ValueError):continue
    return result


def cluster_training_processes(target_id, target):
    """One user queue plus small job descriptors; no storage inventory walk."""
    import re, shlex, subprocess
    def ssh(args):
        result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=5',target['submit_host'],shlex.join(args)],capture_output=True,text=True,timeout=10)
        if result.returncode:raise RuntimeError('读取训练作业失败：'+result.stderr[-500:])
        return result.stdout
    user=os.environ.get('USER','sunny')
    queue=ssh(['squeue','-h','-u',user,'-w',target['node'],'-o','%i|%T'])
    rows=[]
    for line in queue.splitlines()[:32]:
        fields=line.split('|');job=fields[0]
        if not job.isdigit():continue
        description=ssh(['scontrol','show','job','-o',job])
        def field(name):
            match=re.search(r'(?:^|\s)'+name+r'=(\S+)',description);return match[1] if match else None
        owner=field('UserId') or ''
        if not owner.startswith(user+'('):continue
        command=ssh(['scontrol','write','batch_script',job,'-'])[:65536]
        if not any(name in command for name in ('openpi_single_arm.py','scripts/train.py','scripts/train_pytorch.py','train_embodied_agent.py')):continue
        framework='RLinf' if 'train_embodied_agent.py' in command else 'openpi'
        if framework=='RLinf':continue  # RLinf run sources carry its richer identity and metric layout.
        rows.append(dict(id='train-external-'+target_id+'-'+job,type='train',state='queued' if fields[-1]=='PENDING' else 'running',
                         command=['external-slurm',job,field('Command') or ''],metadata=dict(external=True,external_training=True,runtime='slurm',
                         execution_target=target_id,cluster_target=target_id,slurm_target=target_id,slurm_submit_host=target['submit_host'],
                         slurm_job_id=job,remote_stdout=field('StdOut'),remote_stderr=field('StdErr'),framework=framework,
                         exp_name=field('JobName'),dataset_id=None)))
    return rows


def experiment_parent_sources(config):
    """Discover parent scopes using the reader host's interpreter, not a job's PATH."""
    import hashlib
    from .storage_catalog import locations
    scopes={}
    block=config.get('rlinf_dashboard',{})
    targets=block.get('targets',{})
    configured=block.get('sources',[])
    nodes=locations(config)
    def reader_python(target,host,preferred=None):
        if preferred and Path(preferred).is_absolute():return preferred
        if host:
            for source in configured:
                candidate=source.get('python','')
                if source.get('ssh_host')==host and Path(candidate).is_absolute():return candidate
            cluster=config.get('cluster_targets',{}).get(target,{})
            for key in ('inventory_python','openpi_python'):
                if Path(cluster.get(key,'')).is_absolute():return cluster[key]
        return config['openpi_python']
    def add(target,root,host=None,python=None,**flags):
        if not root:return
        root=str(root).rstrip('/')
        if root in {'/','/home','/DATA','/DATA/disk0','/DATA/disk0/sunny'}:return
        key=(target,root)
        # Preserve explicitly disabled storage roots, rather than re-enabling via a profile.
        if any(not x.get('enabled',True) and x.get('root')==root and str(x.get('target','')).lower()==target for x in configured):return
        scope=dict(id='live-'+hashlib.sha256((target+':'+root).encode()).hexdigest()[:20],target=target,root=root,
                   label='实验目录 · '+root,python=reader_python(target,host,python),max_depth=6,
                   query_slurm=bool(host and target in {'h100','h100-ksy-01'}),**({'ssh_host':host} if host else {}),**flags)
        # A source hosted on login-server under /home is an intentional metrics mirror.
        # Only H200 data-disk paths are known to require node-local reads here.
        if target.startswith('h200') and root.startswith('/DATA/disk0/'):
            if target not in nodes:return
            scope['storage_location']=nodes[target]
            scope['label']='节点实验目录（CPU Slurm） · '+root
        scopes.setdefault(key,scope)
    for source in configured:
        if not source.get('enabled',True):continue
        target=str(source.get('target','4x4090')).lower();root=source.get('root','')
        for component in ('/experiments/','/runs/','/outputs/'):
            if component in root:
                root=root.split(component,1)[0]+component.rstrip('/');break
        add(target,root,source.get('ssh_host'),source.get('python'),
            **({k:source[k] for k in ('status_from_process','evaluation_cohort') if k in source}))
    for name,target in targets.items():
        if not target.get('enabled'):continue
        host=target.get('submit_host') if target.get('kind')=='slurm' else None
        for profile in target.get('profiles',{}).values():add(name,profile.get('run_root'),host,target.get('python'))
        if target.get('kind')=='slurm':add(name,'/DATA/disk0/sunny/experiments',host,target.get('python'))
    for name,location in nodes.items():
        target='4x4090' if location['local'] else name
        host=None if location['local'] else location.get('submit_host')
        for root in location.get('artifact_roots',[]):
            if location['local'] and not Path(root).is_dir():continue
            add(target,root,host,canonical=True)
    result=[]
    for key,scope in sorted(scopes.items(),key=lambda pair:len(pair[0][1])):
        if any(scope['target']==old['target'] and scope.get('ssh_host')==old.get('ssh_host') and scope['root'].startswith(old['root']+'/') for old in result):continue
        result.append(scope)
    return result
