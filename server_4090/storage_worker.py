"""Storage-side inventory and replica operations. Run cluster work under Slurm.

This file is self-contained so its source can be sent to an allocated worker.
No tensor/model imports. Inventory fingerprints are NOT content checksums.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import re
import zipfile
import struct
import errno
from contextlib import contextmanager


def read(path, default=None):
    try:
        if Path(path).stat().st_size > 8*1024*1024: return default
        return json.loads(Path(path).read_text())
    except (OSError, ValueError): return default


def atomic(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    for attempt in range(5):
        try:
            temp.write_text(json.dumps(value,ensure_ascii=False)); temp.replace(path)
            return
        except OSError as exc:
            if exc.errno not in {errno.EAGAIN, errno.EINTR, errno.ESTALE} or attempt == 4: raise
            time.sleep(2 ** attempt)



def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def tree(path, content=False):
    path=Path(path); result=[]
    paths=[path] if path.is_file() else sorted(path.rglob('*'))
    for p in paths:
        if p.is_symlink():
            result.append(dict(name=str(p.relative_to(path)) if p!=path else '.', link=True,target=str(p.resolve()))); continue
        if not p.is_file(): continue
        stat=p.stat(); row=dict(name=p.relative_to(path).as_posix() if p!=path else '.', size=stat.st_size)
        if content:
            checksum=hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''): checksum.update(block)
            row['sha256']=checksum.hexdigest()
        result.append(row)
    return result


def file_signatures(path):
    path=Path(path); result={}
    for row in tree(path):
        if row.get('link'):continue
        p=path if row['name']=='.' else path/row['name'];s=p.stat()
        result[row['name']]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    return result


def file_kind(name):
    suffix=Path(name).suffix.lower()
    if suffix in {'.pt','.pth','.safetensors'} or (suffix in {'.npz','.npy','.h5','.hdf5','.parquet'} and any(k in str(name).lower() for k in ('latent','feature','embedding','jepa'))):return 'feature'
    if suffix in {'.mp4','.webm','.avi','.mov'}:return 'video'
    if suffix in {'.npz','.npy','.h5','.hdf5','.parquet'}:return 'trajectory'
    if suffix in {'.png','.jpg','.jpeg','.exr','.webp','.tiff'}:return 'image'
    return 'metadata'


def structure_check(path):
    """Bounded file/container checks, without loading arrays or model weights."""
    path=Path(path);suffix=path.suffix.lower()
    try:
        if suffix in {'.npz','.pt'}:
            with zipfile.ZipFile(path) as archive:
                entries=archive.infolist()
                if not entries:return None
                size=path.stat().st_size
                if any(i.header_offset+i.compress_size>size for i in entries):return None
                for entry in entries[:32]:
                    with archive.open(entry) as stream:stream.read(8)
                return dict(evidence='archive_structure',fields=[i.filename[:-4] for i in entries[:80] if i.filename.endswith('.npy')])
        if suffix=='.mp4':
            size=path.stat().st_size;offset=0;boxes=set()
            with path.open('rb') as stream:
                for _ in range(10000):
                    if offset==size:break
                    stream.seek(offset);header=stream.read(8)
                    if len(header)!=8:return None
                    length,kind=struct.unpack('>I4s',header);minimum=8
                    if length==1:
                        extended=stream.read(8)
                        if len(extended)!=8:return None
                        length=struct.unpack('>Q',extended)[0];minimum=16
                    if length==0:length=size-offset
                    if length<minimum or offset+length>size:return None
                    boxes.add(kind);offset+=length
            if offset==size and {b'ftyp',b'moov',b'mdat'}<=boxes:return dict(evidence='mp4_container_structure')
        if suffix in {'.png','.jpg','.jpeg'}:
            with path.open('rb') as stream:
                head=stream.read(8);stream.seek(max(0,path.stat().st_size-12));tail=stream.read()
            if suffix=='.png' and head==b'\x89PNG\r\n\x1a\n' and tail[4:8]==b'IEND':return dict(evidence='image_container_structure')
            if suffix in {'.jpg','.jpeg'} and head[:2]==b'\xff\xd8' and tail.endswith(b'\xff\xd9'):return dict(evidence='image_container_structure')
    except (OSError,ValueError,zipfile.BadZipFile,EOFError,RuntimeError):pass
    return None


def unit(path, logical, kind, node, root, run=None, expected=None):
    path=Path(path)
    if path.is_symlink():
        entry=dict(name='.',link=True,target=str(path.resolve()),kind=kind,file_state='link',check=None)
        return dict(id=digest([node,str(path.absolute())]),logical_id=digest(logical),logical=logical,kind=kind,node=node,
                    version=str(path.lstat().st_mtime_ns),path=str(path.absolute()),root=str(Path(root).resolve()),
                    relative_path=path.relative_to(root).as_posix(),file_count=0,size_bytes=0,fingerprint=digest([entry]),files=[entry],
                    verification='reference_only',run=run,mutable=False,updated_at=time.time())
    if not path.exists(): return None
    entries=tree(path); files=[e for e in entries if not e.get('link')]
    if not entries: return None
    version=digest(file_signatures(path))
    check=read(Path(root)/'.storage-checks'/(digest(str(path.resolve()))+'.json'),{})
    signatures=file_signatures(path)
    states=[]
    checked=0
    for item in entries:
        temp=any(re.search(r'(^\.|\.)(incoming|partial|part|tmp|incomplete)([.-]|$)',part) for part in path.relative_to(root).parts+Path(item['name']).parts)
        verified=not item.get('link') and signatures.get(item['name'])==(check.get('signatures') or {}).get(item['name'])
        evidence=None
        if not item.get('link') and not temp and not verified and checked<8:
            candidate=path if item['name']=='.' else path/item['name']
            if candidate.suffix.lower() in {'.npz','.pt','.mp4','.png','.jpg','.jpeg'}:
                evidence=structure_check(candidate);checked+=1
        states.append(dict(item,kind=file_kind(item['name']),file_state='link' if item.get('link') else 'incomplete' if temp else 'verified' if verified else 'structure_checked' if evidence else 'present_unverified',check=evidence))
    return dict(id=digest([node,str(path.resolve())]),logical_id=digest(logical),logical=logical,kind=kind,node=node,version=version,
                path=str(path.resolve()),root=str(Path(root).resolve()),relative_path=path.relative_to(root).as_posix(),
                file_count=len(files),size_bytes=sum(e['size'] for e in files),fingerprint=digest(entries),
                files=states,verification='names_and_sizes_only',
                run=run,mutable=not any(e.get('link') or e['file_state']=='incomplete' for e in states),updated_at=time.time())


def inventory(request):
    rows=[];errors=[];seen=set();node=request['node']
    def observed(*args,**kwargs):
        try:return unit(*args,**kwargs)
        except (OSError,ValueError) as exc:
            errors.append(dict(path=str(args[0]),status='changed_or_unreadable',error=str(exc)));return None
    for raw in request.get('artifact_roots',[]):
        root=Path(raw).resolve()
        if root in seen: continue
        seen.add(root)
        if not root.exists(): errors.append(dict(path=str(root),status='unavailable'));continue
        for manifest in root.glob('*/*/*/*/run.json'):
            if manifest.is_symlink() or not manifest.resolve().is_relative_to(root):continue
            run=read(manifest,{})
            if run.get('schema')!='experiment_run.v1':continue
            directory=manifest.parent
            identity=[run.get(k,'unknown') for k in ('video_source','task','experiment','run_id')]
            units=[]
            for phase in ('train','eval'):
                for ep in (directory/phase).glob('*/*/*/episode.json'):
                    if ep.is_symlink():continue
                    detail=read(ep,{})
                    expected=[v for v in detail.get('artifacts',{}).values() if isinstance(v,str) and not Path(v).is_absolute() and '..' not in Path(v).parts]
                    units.append((ep.parent,'rollout',identity+[phase,ep.parent.relative_to(directory/phase).as_posix()],expected))
            index=read(directory/'checkpoints/index.json',{})
            for checkpoint in index.get('checkpoints',[]):
                rel=Path(str(checkpoint.get('path','')))
                if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0]!='checkpoints' or len(rel.parts)<2:continue
                units.append((directory/rel,'model',identity+['model',checkpoint.get('id')],[]))
            for kind in ('logs','metrics','metadata'):
                units.append((directory/kind,kind,identity+[kind],[]))
            for path,kind,logical,expected in units:
                if not path.resolve().is_relative_to(directory.resolve()):continue
                row=observed(path,logical,kind,node,root,run,expected)
                if row:
                    row['run_relative']=directory.relative_to(root).as_posix()
                    row['checkpoint_index']=index if kind=='model' else None
                    rows.append(row)
    for raw in request.get('dataset_roots',[]):
        root=Path(raw).resolve()
        if not root.is_dir():errors.append(dict(path=str(root),status='unavailable'));continue
        for info in root.glob('*/meta/info.json'):
            path=info.parent.parent
            row=observed(path,['dataset',path.name],'dataset',node,root)
            if row:
                row['dataset_info']=read(info,{})
                rows.append(row)
    # Existing nonstandard archives remain discoverable without moving them.
    # They are coarse, read-only directory summaries; do not guess safe delete scopes.
    for raw in request.get('legacy_roots',[]):
        root=Path(raw).resolve()
        if not root.is_dir():continue
        for path in sorted(root.iterdir()):
            if path.name.startswith('.') or path.is_symlink() or not path.is_dir():continue
            if (path/'run.json').is_file():continue
            row=observed(path,['legacy',str(path)],'unclassified',node,root)
            if row:
                row['mutable']=False
                row['file_types_summary']={kind:sum(f.get('kind')==kind for f in row['files']) for kind in ('video','trajectory','image','feature','metadata')}
                row['file_states_summary']={state:sum(f.get('file_state')==state for f in row['files']) for state in ('verified','structure_checked','present_unverified','incomplete','link')}
                row['sampled_file_list']=len(row['files'])>100
                row['files']=row['files'][:100]
                rows.append(row)
    unique={r['id']:r for r in rows}
    resources=[];devices=set()
    for raw in request.get('artifact_roots',[])+request.get('dataset_roots',[])+request.get('legacy_roots',[]):
        p=Path(raw)
        if not p.exists():continue
        dev=p.stat().st_dev
        if dev in devices:continue
        devices.add(dev);disk=shutil.disk_usage(p)
        resources.append(dict(path=str(p.resolve()),total_bytes=disk.total,free_bytes=disk.free))
    owned_usage=[]
    for raw in request.get('usage_roots',[]):
        try:
            result=subprocess.run(['du','-sx','--block-size=1',raw],capture_output=True,text=True,timeout=180,check=True)
            owned_usage.append(dict(path=raw,bytes=int(result.stdout.split()[0])))
        except (OSError,ValueError,subprocess.SubprocessError) as exc:owned_usage.append(dict(path=raw,error=type(exc).__name__))
    return dict(node=node,updated_at=time.time(),copies=list(unique.values()),errors=errors,
                resources=resources,owned_usage=owned_usage,
                scope='Configured canonical artifact and dataset roots; legacy unregistered files are not inferred complete.')


def check_idle(path, run=None, task_root=None):
    task_root=task_root or os.environ.get('BIMANUAL_STORAGE_TASK_ROOT')
    if task_root:
        for task_path in Path(task_root).glob('*/task.json'):
            task=read(task_path,{})
            if task.get('state') in {'completed','succeeded','failed','stopped','cancelled','canceled','lost'}:continue
            metadata=task.get('metadata') or {}
            if str(metadata.get('transfer_kind','')).startswith('artifact_'):continue
            for key in ('artifact_run_dir','checkpoint','checkpoint_dir','rlinf_output_dir','output_dir','base_checkpoint','resume_checkpoint','dataset_path'):
                value=metadata.get(key)
                if not value:continue
                ref=Path(str(value)).expanduser().resolve();current=Path(path).resolve()
                if current==ref or current in ref.parents or ref in current.parents:raise ValueError('Artifact referenced by active Dashboard task '+task.get('id','unknown'))
    if run and run.get('state','').lower() not in {'completed','complete','failed','stopped','cancelled','canceled','succeeded'}:
        raise ValueError('Run is not terminal')
    if os.environ.get('SLURM_JOB_ID'):
        queue=subprocess.run(['squeue','-h','-u',str(os.getuid()),'-o','%i'],capture_output=True,text=True,check=True)
        for job in queue.stdout.split():
            if job==os.environ['SLURM_JOB_ID']:continue
            script=subprocess.run(['scontrol','write','batch_script',job,'-'],capture_output=True,text=True)
            if script.returncode:raise ValueError('Cannot verify active job '+job)
            refs=[str(path)]+[str(v) for k,v in (run or {}).get('origin',{}).items() if v and ('path' in k or 'dir' in k)]
            if any(ref in script.stdout for ref in refs):raise ValueError('Artifact referenced by active Slurm job '+job)


def projected_bytes(copy):
    return copy['size_bytes']+copy.get('file_count',1)*16384+16*1024**2


def capacity(request,additional):
    """The per-node user limit includes temporary local copies, excludes NAS."""
    roots=[]
    for raw in request.get('usage_roots',[]):
        path=Path(raw).resolve()
        if not path.is_dir():raise ValueError('Cannot verify personal usage root: '+str(path))
        if path.exists() and not any(path==r or r in path.parents for r in roots):roots.append(path)
    total=0
    for path in roots:
        result=subprocess.run(['du','-sx','--block-size=1',str(path)],capture_output=True,text=True,check=True)
        total+=int(result.stdout.split()[0])
    if roots and total+additional>2_000_000_000_000:raise ValueError('Projected per-node usage exceeds 2 TB including staging')


def checkpoint_shape(path, backend):
    """Only a weight file or a single Orbax step, never a training run."""
    path=Path(path)
    if backend=='openpi':
        if not re.fullmatch(r'(?:step[-_])?\d+',path.name) or not (path/'params').is_dir() or not (path/'_CHECKPOINT_METADATA').is_file():
            raise ValueError('Select one complete SFT checkpoint step')
        allowed={'params','train_state','opt_state','assets','_CHECKPOINT_METADATA','_METADATA','_sharding','manifest.ocdbt','d','action_contract.json','metadata.json'}
        if any(p.name not in allowed for p in path.iterdir()):raise ValueError('Checkpoint contains unrelated files; refusing mixed directory')
    elif backend=='rlinf':
        if not path.is_file() or path.suffix not in {'.pt','.safetensors','.pth'}:raise ValueError('Select the RLinf weight file, not its parent directory')
    else:raise ValueError('Unknown checkpoint backend')


def prepare_checkpoint(request):
    path=Path(request['path']).absolute()
    roots=[Path(p).resolve() for p in request['roots']]
    # References are read-only; do not silently delete their targets.
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise ValueError('Linked checkpoint is read-only; select its actual location')
    path=path.resolve()
    root=next((p for p in sorted(roots,key=lambda p:len(p.parts),reverse=True) if path!=p and path.is_relative_to(p)),None)
    if root is None:raise ValueError('Checkpoint is outside configured storage roots')
    checkpoint_shape(path,request['backend'])
    run_root=next((p for p in path.parents if p.is_relative_to(root) and (p/'run.json').is_file()),None)
    run=read(run_root/'run.json',{}) if run_root else None
    if run and run.get('backend') not in {request['backend'],'sft' if request['backend']=='openpi' else 'rlinf'}:raise ValueError('Checkpoint backend differs from run')
    if run_root and not path.is_relative_to(run_root/'checkpoints'):raise ValueError('Not in the run checkpoint directory')
    copy=unit(path,dict(node=request['node'],path=str(path)),'model',request['node'],root,run)
    if not copy or not copy['mutable']:raise ValueError('Checkpoint is empty, incomplete, or contains links')
    copy['selected_checkpoint_backend']=request['backend']
    if run_root:
        copy['run_relative']=run_root.relative_to(root).as_posix()
        copy['checkpoint_index']=read(run_root/'checkpoints/index.json',{})
    else:
        # Give legacy weights a stable canonical destination without moving source files.
        identity=digest([request['node'],str(path.parent)])[:16]
        run_relative='imported/unknown/'+path.parent.name+'/'+identity
        copy['destination_relative']=run_relative+'/checkpoints/'+path.name
        copy['destination_run_relative']=run_relative
        copy['destination_run']=dict(schema='experiment_run.v1',video_source='imported',task='unknown',experiment=path.parent.name,run_id=identity,backend=request['backend'],state='completed',origin=dict(config_name=path.parent.parent.name,legacy_path=str(path.parent)))
    relative='checkpoints/'+path.name if not run_root else path.relative_to(run_root).as_posix()
    entries=(copy.get('checkpoint_index') or {}).get('checkpoints',[])
    entry=next((dict(c) for c in entries if c.get('path')==relative),None)
    if entry is None:
        match=re.search(r'(?:step[-_]|round)?(\d+)(?:\.pt)?$',path.name)
        contract={}
        for parent in (path if path.is_dir() else path.parent,path.parent,path.parent.parent):
            contract=read(parent/'action_contract.json',{}) or read(parent/'metadata/action_contract.json',{})
            if contract:break
        datasets=[p.name for p in (path/'assets').iterdir() if p.is_dir()] if (path/'assets').is_dir() else list((run or {}).get('dataset_ids',[]))
        if contract.get('dataset_id') and contract['dataset_id'] not in datasets:datasets.append(contract['dataset_id'])
        entry=dict(path=relative,format='orbax' if request['backend']=='openpi' else 'rlinf_safetensors' if path.suffix=='.safetensors' else 'rlinf_full_weights',
                   step=int(match[1]) if match else None,complete=True,dataset_ids=datasets,action_contract=contract or (run or {}).get('action_contract'))
    copy['checkpoint_index']=dict(schema='checkpoint_index.v1',checkpoints=[*filter(lambda c:c.get('path')!=relative,entries),entry])
    if copy.get('destination_run'):
        copy['destination_run'].update(dataset_ids=entry.get('dataset_ids',[]),action_contract=entry.get('action_contract'))
    validate_copy(copy)
    return copy


def validate_copy(copy, idle=True):
    path=Path(copy['path']);root=Path(copy['root']).resolve()
    if not root.is_dir() or path.is_symlink() or path.resolve()==root or not path.resolve().is_relative_to(root):raise ValueError('Copy escaped its root')
    relative=path.relative_to(root)
    if any(p.is_symlink() for p in path.parents if p == root or root in p.parents):raise ValueError('Linked ancestor is read-only')
    if copy.get('selected_checkpoint_backend'):
        checkpoint_shape(path,copy['selected_checkpoint_backend'])
    elif copy['kind']=='dataset':
        if len(relative.parts)!=1 or not (path/'meta/info.json').is_file():raise ValueError('Invalid dataset copy')
    else:
        if len(relative.parts)<5:raise ValueError('Cannot operate on a whole run')
        branch=relative.parts[4]
        expected={'model':'checkpoints','rollout':None,'logs':'logs','metrics':'metrics','metadata':'metadata'}[copy['kind']]
        if copy['kind']=='rollout':
            if branch not in {'train','eval'} or len(relative.parts)!=8 or not (path/'episode.json').is_file():raise ValueError('Invalid episode copy')
        elif branch!=expected or (copy['kind']=='model' and len(relative.parts)<6):raise ValueError('Wrong artifact category')
    current=tree(path)
    if digest(file_signatures(path))!=copy.get('version') or digest(current)!=copy['fingerprint']:raise ValueError('Copy changed since inventory; refresh first')
    if any(x.get('link') for x in current):raise ValueError('Linked copies require separate resolution')
    for p in [path,*path.rglob('*')] if path.is_dir() else [path]:
        if hasattr(os,'getuid') and p.lstat().st_uid!=os.getuid():raise ValueError('Copy contains another owner')
    if idle:
        run=read(root/copy.get('run_relative','')/'run.json',{}) if copy.get('run_relative') else None
        if copy.get('run_relative') and not run:raise ValueError('Cannot verify run state')
        check_idle(root/copy['run_relative'] if copy.get('run_relative') else path,run)
    return path


def export_copy(request):
    copy=request['copy'];path=validate_copy(copy)
    capacity(request,projected_bytes(copy))
    bundle=Path(request['bundle']);bundle.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(bundle).free < copy['size_bytes']+1024**3:raise ValueError('Insufficient staging space')
    payload=bundle/'payload'
    kinds=request.get('kinds')
    if kinds is not None and (not kinds or copy['kind']!='rollout'):raise ValueError('File type selection is only for rollouts')
    if kinds and (not isinstance(kinds,list) or not set(kinds)<={'video','trajectory','image','feature'}):raise ValueError('Invalid file types')
    if path.is_dir() and kinds:
        payload.mkdir()
        selected=[e for e in tree(path) if file_kind(e['name']) in set(kinds)|{'metadata'}]
        if not any(file_kind(e['name']) in kinds for e in selected):raise ValueError('No matching payload files')
        for item in selected:
            target=payload/item['name'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path/item['name'],target)
    elif path.is_dir():shutil.copytree(path,payload)
    else:shutil.copy2(path,payload)
    hashes=tree(payload,content=True)
    # Source must remain the same snapshot during export.
    names={f['name'] for f in hashes}
    if [f for f in tree(path,content=True) if f['name'] in names]!=hashes:raise ValueError('Source changed during export')
    exported_copy=dict(copy)
    if copy.get('destination_relative'):
        exported_copy.update(relative_path=copy['destination_relative'],run_relative=copy['destination_run_relative'],run=copy['destination_run'])
    manifest=dict(copy=exported_copy,files=hashes,content_digest=digest(hashes),is_directory=path.is_dir())
    atomic(bundle/'manifest.json',manifest)
    atomic(Path(copy['root'])/'.storage-checks'/(digest(str(path.resolve()))+'.json'),dict(files=hashes,signatures={k:v for k,v in file_signatures(path).items() if k in names},verified_at=time.time()))
    return dict(exported=True,content_digest=manifest['content_digest'])


def import_copy(request):
    bundle=Path(request['bundle']); manifest=read(bundle/'manifest.json',{})
    if not manifest or digest(tree(bundle/'payload',content=True))!=manifest.get('content_digest'):raise ValueError('Incoming bundle checksum mismatch')
    if any(f.get('link') for f in manifest['files']):raise ValueError('Incoming bundle contains links')
    copy=manifest['copy'];root=Path(request['destination_root']).resolve();rel=Path(copy['relative_path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe destination')
    destination=root/rel
    if not destination.resolve().is_relative_to(root) or destination==root:raise ValueError('Destination escapes root')
    if destination.is_symlink() or any(p.is_symlink() for p in destination.parents if root in p.parents):raise ValueError('Destination uses a symlink')
    existing_run=read(root/copy.get('run_relative','')/'run.json',{}) if copy.get('run_relative') else None
    check_idle(root/copy['run_relative'] if copy.get('run_relative') else destination,existing_run or None)
    if copy.get('run_relative') and (root/copy['run_relative']/'run.json').exists() and not existing_run:raise ValueError('Cannot verify destination run')
    if existing_run and any(existing_run.get(k)!=copy['run'].get(k) for k in ('video_source','task','experiment','run_id')):raise ValueError('Destination run identity differs')
    destination.parent.mkdir(parents=True,exist_ok=True)
    capacity(request,projected_bytes(copy))
    if shutil.disk_usage(destination.parent).free<copy['size_bytes']+1024**3:raise ValueError('Insufficient target space')
    lock=destination.with_name('.'+destination.name+'.replica-lock')
    lock.mkdir()  # Exclusive copy operation, including partial-replica repair.
    try:
        if destination.exists():
            # Fill missing files only. Any conflicting existing file aborts the plan.
            actual=tree(destination,content=True); incoming={f['name']:f for f in manifest['files']}
            if any(f.get('link') for f in actual):raise ValueError('Destination contains links')
            if any(f['name'] in incoming and incoming[f['name']]!=f for f in actual):raise ValueError('Existing files differ; refusing overwrite')
            if manifest['is_directory']:
                for item in manifest['files']:
                    target=destination/item['name']
                    if target.exists():continue
                    target.parent.mkdir(parents=True,exist_ok=True)
                    temp=target.with_name('.'+target.name+'.incoming-'+uuid.uuid4().hex)
                    shutil.copy2(bundle/'payload'/item['name'],temp);temp.replace(target)
            elif not actual:raise ValueError('Invalid existing file')
        else:
            temporary=destination.with_name('.'+destination.name+'.incoming-'+uuid.uuid4().hex)
            if manifest['is_directory']:shutil.copytree(bundle/'payload',temporary)
            else:shutil.copy2(bundle/'payload',temporary)
            if digest(tree(temporary,content=True))!=manifest['content_digest']:raise ValueError('Destination checksum mismatch')
            temporary.rename(destination)
        wanted={f['name'] for f in manifest['files']}
        if digest([f for f in tree(destination,content=True) if f['name'] in wanted])!=manifest['content_digest']:raise ValueError('Final file checksum mismatch')
        if copy.get('run_relative'):
            run_root=root/copy['run_relative'];run_path=run_root/'run.json'
            if not run_path.exists():atomic(run_path,dict(copy['run'],storage_origin=dict(node=copy['node'],path=copy['path'])))
            if copy['kind']=='model':
                index_path=run_root/'checkpoints/index.json';index=read(index_path,dict(schema='checkpoint_index.v1',checkpoints=[]))
                relative=destination.relative_to(run_root).as_posix()
                additions=[c for c in (copy.get('checkpoint_index') or {}).get('checkpoints',[]) if c.get('path')==relative]
                by_path={c['path']:c for c in index.get('checkpoints',[])};by_path.update({c['path']:c for c in additions})
                atomic(index_path,dict(schema='checkpoint_index.v1',checkpoints=list(by_path.values())))
        receipt=dict(path=str(destination),node=request['node'],content_digest=manifest['content_digest'],verified_at=time.time(),files=manifest['files'],signatures={k:v for k,v in file_signatures(destination).items() if k in wanted})
        atomic(root/'.storage-checks'/ (digest(str(destination))+'.json'),receipt)
        return receipt
    finally:lock.rmdir()


def delete_copy(request):
    copy=request['copy'];path=validate_copy(copy)
    kinds=request.get('kinds')
    if kinds is not None:
        if not kinds or copy['kind']!='rollout' or not isinstance(kinds,list) or not set(kinds)<={'video','trajectory','image','feature'}:raise ValueError('Invalid rollout file types')
        selected=[path/e['name'] for e in tree(path) if file_kind(e['name']) in kinds]
        for p in selected:
            if p.is_symlink() or not p.resolve().is_relative_to(path.resolve()):raise ValueError('Linked payload is read-only')
        deleted=[]
        for p in selected:p.unlink();deleted.append(str(p))
        return dict(deleted_files=deleted,path=str(path),node=copy['node'])
    # A category unit never includes the parent run or a sibling model/rollout.
    trash=path.with_name('.'+path.name+'.deleting-'+uuid.uuid4().hex)
    path.rename(trash)
    try:
        if trash.is_dir():shutil.rmtree(trash)
        else:trash.unlink()
    except BaseException:
        if trash.exists() and not path.exists():trash.rename(path)
        raise
    return dict(deleted=True,path=str(path),node=copy['node'])


def cleanup(request):
    bundle=Path(request['bundle'])
    if bundle.parent.name not in {'.replica-staging','replica-staging'} or len(bundle.name)!=32 or any(c not in '0123456789abcdef' for c in bundle.name):raise ValueError('Not a task staging bundle')
    if bundle.is_symlink() or not (bundle/'manifest.json').is_file():raise ValueError('Unverified staging bundle')
    shutil.rmtree(bundle)
    return dict(staging_removed=str(bundle))


@contextmanager
def mutation_lock(request):
    import fcntl
    if request['action'] == 'import':
        manifest=read(Path(request['bundle'])/'manifest.json',{})
        copy=manifest.get('copy',{})
        root=Path(request['destination_root']).resolve()
        relative=copy.get('run_relative') or copy.get('relative_path','')
    else:
        copy=request['copy'];root=Path(copy['root']).resolve()
        relative=copy.get('run_relative') or copy['relative_path']
    lock_root=root/'.storage-locks';lock_root.mkdir(parents=True,exist_ok=True)
    with (lock_root/(digest(relative)+'.lock')).open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another operation is modifying this run or dataset')
        yield


def preflight(request):
    capacity(request,request['additional_bytes'])
    path=Path(request['destination_root'])
    while not path.exists() and path != path.parent:path=path.parent
    if shutil.disk_usage(path).free < request['additional_bytes']+1024**3:raise ValueError('Insufficient destination space including staging')
    return dict(capacity_checked=True)


def history_cleanup(request):
    """Delete only run diagnostics. Never recurse into weights or rollout payloads."""
    roots = [Path(raw).resolve() for raw in request.get('run_paths', [])]
    files = set()
    protected = {'checkpoints', 'checkpoint', 'models', 'weights', 'params', 'train', 'eval', 'rollouts', 'rollout', 'video', 'videos', 'frames', 'images', 'latents', 'latent', 'features', 'trajectories', 'artifacts', 'assets', 'replay', 'replay_buffer'}
    names = {'training_curve.json','evaluation_curve.json','metrics.json','metrics.jsonl','metrics.csv','resource_samples.jsonl','metrics.log','eval_results.jsonl'}
    diagnostic_suffixes = {'.log', '.out', '.err'}
    for root in roots:
        if not root.is_dir(): continue
        run = read(root/'run.json', {})
        check_idle(root, run or request.get('run'), request.get('task_root'))
        for current, dirs, entries in os.walk(root, followlinks=False):
            parent = Path(current); relative = parent.relative_to(root)
            if len(relative.parts) > 8: raise ValueError('Diagnostic directory exceeds depth limit')
            dirs[:] = [d for d in dirs if d not in protected and not d.isdigit() and not d.startswith(('global_step_', 'step-', '.')) and not (parent/d).is_symlink()]
            if parent != root and (parent/'run.json').is_file():
                dirs[:] = []; continue
            diagnostic_dir = bool(set(relative.parts) & {'logs','metrics','tensorboard','wandb','curves'})
            for name in entries:
                path = parent/name
                # Tensor/array payloads are always retained, even if misplaced under logs.
                if file_kind(name) in {'trajectory','feature'} or path.suffix.lower() in {'.pt','.pth','.safetensors','.mp4','.webm','.avi','.mov'}: continue
                selected = name in names or path.suffix.lower() in diagnostic_suffixes or name.startswith('events.out.tfevents.')
                selected |= diagnostic_dir and path.suffix.lower() in {'.json','.jsonl','.csv','.txt','.png','.svg','.pdf'} and not any(k in name.lower() for k in ('latent','feature','frame','image'))
                if selected:
                    if path.is_symlink(): raise ValueError('Diagnostic symlink requires an explicit owning run: '+str(path))
                    if path.is_file(): files.add(path)
    for raw in request.get('log_files', []):
        path=Path(raw)
        if path.is_symlink(): raise ValueError('Linked external log is not deletable')
        if path.suffix.lower() not in diagnostic_suffixes: raise ValueError('External file is not a log')
        if path.is_file():
            check_idle(path, task_root=request.get('task_root'));files.add(path)
    signatures = {}
    for path in files:
        st=path.stat()
        if st.st_uid != os.getuid(): raise ValueError('Diagnostic file belongs to another user')
        signatures[path]=(st.st_dev,st.st_ino,st.st_mtime_ns,st.st_size)
    if request.get('dry_run'): return dict(paths=[str(p) for p in sorted(files)], bytes=sum(v[3] for v in signatures.values()))
    for root in roots:
        if root.exists(): check_idle(root, read(root/'run.json', {}) or request.get('run'), request.get('task_root'))
    for path, expected in signatures.items():
        st=path.stat()
        if path.is_symlink() or (st.st_dev,st.st_ino,st.st_mtime_ns,st.st_size)!=expected: raise ValueError('Diagnostic file changed during deletion')
    for path in sorted(files): path.unlink()
    return dict(deleted_paths=[str(p) for p in sorted(files)], deleted_count=len(files))


def discover_models(request):
    """Shallow checkpoint directory discovery: never descend into model payloads."""
    started=time.monotonic(); seen=set(); rows=[]; errors=[]; checked=0; truncated=False
    backend=request.get('backend','openpi'); experiment=request.get('experiment','')
    def children(path):
        nonlocal checked,truncated
        if not path.is_dir(): return []
        result=[]
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    checked+=1
                    if checked>10000 or time.monotonic()-started>2:
                        truncated=True;break
                    if not entry.name.startswith('.') and entry.is_dir():result.append(Path(entry.path))
        except OSError as exc:errors.append(str(exc))
        return result
    def add(path, config_name, exp, run=None):
        nonlocal checked
        if experiment and experiment.lower() not in exp.lower(): return
        key=str(path.resolve())
        if key in seen:return
        seen.add(key);run=run or {}
        if backend=='openpi':
            if not (path/'params').is_dir() or not (path/'_CHECKPOINT_METADATA').is_file():return
            match=re.fullmatch(r'(?:step[-_])?(\d+)',path.name)
            if not match:return
            step=int(match[1]);fmt='orbax'
            datasets=[p.name for p in children(path/'assets')]
            contract={}
            for parent in (path,path.parent,path.parent.parent):
                contract=read(parent/'action_contract.json',{}) or read(parent/'metadata/action_contract.json',{})
                if contract:break
            if contract.get('dataset_id') and contract['dataset_id'] not in datasets:datasets.append(contract['dataset_id'])
            variant=contract.get('model_variant') or ('pi05' if config_name.startswith('pi05') else 'pi0')
            arm=contract.get('arm_mode') or ('bimanual' if 'bimanual' in config_name else 'single')
            row=dict(path=key,backend='openpi',format=fmt,experiment=exp,step=step,config_name=config_name,
                     dataset_ids=datasets,model_variant=variant,arm_mode=arm,action_contract=contract,finetuning=contract.get('finetuning'),
                     full_state_available=(path/'train_state/_METADATA').is_file(),size_bytes=None,size_gib=None,
                     mtime=path.stat().st_mtime,updated_at=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(path.stat().st_mtime)))
        else:
            weight=path if path.is_file() and path.suffix in {'.pt','.safetensors'} else next((path/p for p in ('actor/model_state_dict/full_weights.pt','model_state_dict/full_weights.pt','model.safetensors') if (path/p).is_file()),None)
            if not weight:return
            match=re.search(r'(?:step[_-]|round)(\d+)',path.name)
            row=dict(path=str(weight.resolve()),backend='rlinf',format='rlinf_full_weights' if weight.name=='full_weights.pt' else 'rlinf_safetensors' if weight.name=='model.safetensors' else 'pytorch',
                     experiment=exp,run_id=run.get('run_id'),task=run.get('task'),step=int(match[1]) if match else None,size_bytes=weight.stat().st_size,dataset_ids=run.get('dataset_ids',[]))
        rows.append(dict(row,complete=True,discovered_at=time.time(),discovery='live_directory'))
    for item in request.get('roots',[]):
        root=Path(item['path'])
        if not root.is_dir():continue
        if item.get('layout')=='legacy':
            for cfg in children(root):
                if not re.fullmatch(r'pi(?:0|05)_piper_(?:single_arm|bimanual)_lora',cfg.name):continue
                for exp in children(cfg):
                    if experiment and experiment.lower() not in exp.name.lower():continue
                    for path in children(exp):add(path,cfg.name,exp.name)
        else:
            current=[root]
            for _ in range(4):current=[p for parent in current for p in children(parent)]
            for directory in current:
                run=read(directory/'run.json',{})
                if run.get('backend')!=backend:continue
                exp=run.get('experiment',directory.parent.name)
                if experiment and experiment.lower() not in exp.lower():continue
                config_name=run.get('origin',{}).get('config_name') or Path(run.get('origin',{}).get('legacy_path','')).parent.name
                cp=directory/'checkpoints'
                if not cp.is_dir():continue
                for path in cp.iterdir():
                    if path.name in {'index.json'} or path.name.startswith('.'):continue
                    add(path,config_name,exp,run)
    return dict(models=rows,checked_at=time.time(),entries_checked=checked,truncated=truncated,errors=errors,elapsed_seconds=time.monotonic()-started)


def run_summary(request):
    args=[sys.executable,'-c',request['scan_code'],request['root'],str(min(6,int(request.get('max_depth',6)))),
          '1' if request.get('parse_events') else '0','0']
    result=subprocess.run(args,capture_output=True,text=True,timeout=30,check=True)
    return json.loads(result.stdout)


def main(request):
    action=request['action']
    if request.get('attach_delay'):time.sleep(request['attach_delay'])
    operation={'inventory':inventory,'export':export_copy,'import':import_copy,'delete':delete_copy,'cleanup':cleanup,'preflight':preflight,'history_cleanup':history_cleanup,'discover_models':discover_models,'run_summary':run_summary,'prepare_checkpoint':prepare_checkpoint}[action]
    if action in {'export','import','delete'}:
        with mutation_lock(request):result=operation(request)
    else:result=operation(request)
    if request.get('output'):atomic(request['output'],result)
    if request.get('receipt_stdout'):
        import base64,zlib
        receipt='STORAGE_RECEIPT:'+base64.b64encode(zlib.compress(json.dumps(result).encode())).decode()
        for attempt in range(3 if request.get('attach_delay') else 1):
            print(receipt,flush=True)
            if request.get('attach_delay'):time.sleep(5)

    print(json.dumps({k:v for k,v in result.items() if k!='copies'},ensure_ascii=False))


if __name__=='__main__': main(json.loads(sys.argv[1]))
