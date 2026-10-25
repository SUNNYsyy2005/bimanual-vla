"""Verified inference export with pinned source files and parallel bundle transfer."""
import argparse, base64, concurrent.futures, hashlib, inspect, io, json, os
from pathlib import Path
import shlex, shutil, subprocess, sys, tarfile, tempfile, time, uuid, threading
import checkpoint_export
from dataset_transfer_runner import SSH_OPTS, clean_transport_env, start_ssh_control_master, stop_ssh_control_master

def encoded(value):
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def validate_install(path):
    path=Path(path)
    for name in ['_CHECKPOINT_METADATA','params/_METADATA']:
        if not (path/name).is_file():raise ValueError(f'incomplete checkpoint: {path/name}')
    manifest=json.loads((path/'.dashboard-inference.json').read_text())
    for row in manifest['files']:
        name=Path(row['path'])
        if name.is_absolute() or '..' in name.parts:raise ValueError('invalid checkpoint path')
        f=path/name
        if f.is_symlink() or not f.is_file() or f.stat().st_size!=row['size']:
            raise ValueError(f'checkpoint manifest mismatch: {name}')
    return manifest

class PartsReader(io.RawIOBase):
    def __init__(self,paths):self.paths,self.current=iter(paths),None
    def read(self,size=-1):
        if self.current is None:
            path=next(self.paths,None)
            if path is None:return b''
            self.current=Path(path).open('rb')
        data=self.current.read(size)
        if data:return data
        self.current.close();self.current=None
        return self.read(size)
    def close(self):
        if self.current:self.current.close()
        super().close()

def install_bundle(bundle,destination):
    bundle,destination=Path(bundle),Path(destination)
    manifest=json.loads((bundle/'manifest.json').read_text())
    for part in manifest['parts']:
        name=part['name']
        if Path(name).name!=name or not name.startswith('part-'):raise ValueError('invalid bundle part')
        p=bundle/name
        if p.stat().st_size!=part['size'] or digest(p)!=part['sha256']:
            raise ValueError(f'corrupt checkpoint part: {p}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.install-',dir=destination.parent) as work:
        tmp=Path(work)/destination.name;tmp.mkdir()
        with PartsReader([bundle/p['name'] for p in manifest['parts']]) as reader:
            with tarfile.open(fileobj=reader,mode='r|') as archive:
                for member in archive:
                    name=Path(member.name)
                    if name.is_absolute() or '..' in name.parts or not member.isfile():
                        raise ValueError(f'unsafe archive member: {member.name}')
                    archive.extract(member,path=tmp,filter='data')
        (tmp/'.dashboard-inference.json').write_text(json.dumps(manifest))
        validate_install(tmp)
        if destination.exists():
            destination.rename(destination.with_name(f'.{destination.name}.incomplete-{uuid.uuid4().hex[:8]}'))
        tmp.rename(destination)
    return manifest

def sync_slurm(request):
    source,target=request['source'],request['target'];step=str(request['step'])
    dest=Path(target['dataset_root'])/step
    if (dest/'.dashboard-inference.json').is_file():return validate_install(dest),{'already_present':True}
    started=time.monotonic();source_path=str(Path(source['dataset_root'])/step)
    bundle_id=step+'-policy-'+uuid.uuid4().hex[:12]
    host=source['submit_host']
    # Resolve the existing SSH host alias on the Dashboard; compute nodes need
    # the public hostname rather than the Dashboard's private ~/.ssh alias.
    settings=subprocess.run(['ssh','-G',host],capture_output=True,text=True,check=True,env=clean_transport_env())
    options=dict(line.split(' ',1) for line in settings.stdout.splitlines() if ' ' in line)
    address=options['user']+'@'+options['hostname']
    staging=str(Path(source.get('checkpoint_ssh_staging_root') or str(Path(source['remote_job_dir']).parent/'checkpoint_exports'))/bundle_id)
    code=inspect.getsource(checkpoint_export)
    code+='\nexport_checkpoint('+repr(source_path)+','+repr(staging)+','+str(int(request['parallelism']))+','+repr(address)+')\n'
    python=source.get('openpi_python') or source.get('inventory_python')
    if not python:raise ValueError('checkpoint source must declare its Conda Python')
    target_config=dict(source,gpu_count=0,mem='8G',cpus_per_task=max(2,min(8,request['parallelism'])),
                       time='01:00:00',log_transport=source.get('log_transport','sattach'))
    command=[sys.executable,str(Path(__file__).with_name('slurm_job_runner.py')),
             '--target-json',encoded(target_config),'--commands-json',encoded([[python,'-c',code]]),
             '--command-labels-json',encoded(['snapshot_and_pack']),
             '--job-name','checkpoint_export_'+uuid.uuid4().hex[:12],'--poll-interval','5']
    command.append('--cancel-on-exit')
    export_process=subprocess.Popen(command)
    export_done=[]
    def watch_export():
        export_process.wait();export_done.append(time.monotonic())
    watcher=threading.Thread(target=watch_export,daemon=True);watcher.start()
    cache=Path(target['dataset_root'])/'.checkpoint-transfers'/bundle_id;cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='checkpoint-ssh-') as control:
        sock=str(Path(control)/'master');start_ssh_control_master(dict(host=host),sock)
        try:
            deadline=time.monotonic()+3600
            while True:
                result=subprocess.run([*SSH_OPTS,'-S',sock,host,'cat '+shlex.quote(staging+'/plan.json')],
                                      capture_output=True,env=clean_transport_env())
                if result.returncode==0:break
                if export_process.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('checkpoint export ended without a transfer plan; see Slurm log')
                time.sleep(2)
            plan_s=time.monotonic()-started
            manifest=json.loads(result.stdout);(cache/'manifest.json').write_bytes(result.stdout)
            actual=manifest['action_contract'];marker=Path(request['marker_path'])
            if request.get('contract') and actual!=request['contract']:raise ValueError('source action contract changed')
            if marker.exists() and json.loads(marker.read_text())!=actual:raise ValueError('destination action contract differs')
            def fetch(part):
                name=part['name']
                if Path(name).name!=name or not name.startswith('part-'):raise ValueError('invalid bundle part')
                path=cache/name
                attempt=0
                while attempt<3:
                    offset=path.stat().st_size if path.exists() else 0
                    if offset>part['size']:path.unlink();offset=0
                    remote=shlex.quote(staging+'/'+name)
                    cmd=[*SSH_OPTS,'-S',sock,host,'if test -f '+remote+'; then tail -c +'+str(offset+1)+' '+remote+'; else exit 75; fi']
                    with path.open('ab') as out:
                        p=subprocess.run(cmd,stdout=out,stderr=subprocess.PIPE,env=clean_transport_env())
                    if p.returncode==75:
                        if (export_process.poll() not in (None,0)) or time.monotonic()>deadline:
                            raise RuntimeError('checkpoint export failed before part publication: '+name)
                        time.sleep(2);continue
                    attempt+=1
                    if p.returncode==0 and path.stat().st_size==part['size'] and digest(path)==part['sha256']:
                        print(f"[dashboard] received {name} bytes={part['size']}",flush=True);return
                    if path.exists() and path.stat().st_size==part['size']:path.unlink()
                raise RuntimeError(f'failed checkpoint part after 3 attempts: {name}')
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(8,request['parallelism']))) as pool:
                list(pool.map(fetch,manifest['parts']))
        finally:
            stop_ssh_control_master(host,sock)
            if sys.exc_info()[0] is not None and export_process.poll() is None:export_process.terminate()
    watcher.join()
    if export_process.returncode:raise RuntimeError('checkpoint export failed; see Slurm log')
    staging_s=export_done[0]-started
    transfer_s=time.monotonic()-started-plan_s
    manifest=install_bundle(cache,dest)
    timings=dict(staging_s=staging_s,transfer_s=transfer_s,plan_s=plan_s,pipelined=True,elapsed_s=time.monotonic()-started,
                 bytes=manifest['bytes'],effective_mib_s=manifest['bytes']/2**20/(time.monotonic()-started))
    shutil.rmtree(cache)
    return manifest,timings

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request-json',required=True)
    request=json.loads(base64.urlsafe_b64decode(parser.parse_args().request_json))
    marker=Path(request['marker_path']);contract=request['contract'];marker.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    with marker.with_suffix('.sync.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if contract and marker.exists() and json.loads(marker.read_text())!=contract:raise ValueError('destination action contract differs')
        source=request['source']
        if source.get('kind')=='slurm_only':
            manifest,timings=sync_slurm(request)
            contract=manifest['action_contract']
        else:
            subprocess.run([sys.executable,str(Path(__file__).with_name('dataset_transfer_runner.py')),
                '--dataset-id',str(request['step']),'--source-json',encoded(source),
                '--target-json',encoded(request['target']),'--parallelism',str(request['parallelism']),
                '--skip-existing'],check=True)
            dest=Path(request['target']['dataset_root'])/str(request['step'])
            if not (dest/'_CHECKPOINT_METADATA').is_file() or not (dest/'params/_METADATA').is_file():raise ValueError('incomplete checkpoint')
            timings={}
        tmp=marker.with_suffix('.incoming.json');tmp.write_text(json.dumps(contract,indent=2)+'\n');tmp.replace(marker)
        print('[dashboard] checkpoint ready '+json.dumps(timings),flush=True)

if __name__=='__main__':main()
