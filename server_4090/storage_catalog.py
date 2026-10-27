"""Central replica catalog and Slurm-aware operations; no network on GET paths."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid
import tempfile
import zlib

from .artifact_inventory import Inventory
from . import storage_worker as worker


def encoded(value): return base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def locations(config):
    artifact_roots=Inventory(config).roots()
    main=Path(config.get('experiment_artifact_root') or Path(config['workspace_root'])/'experiment_artifacts').resolve()
    result={'local_4090':dict(name='local_4090',label='4090 主目录',local=True,artifact_root=str(main),
                            artifact_roots=[str(main)],dataset_roots=[config['dataset_root']],dataset_root=config['dataset_root'],
                            legacy_roots=list(dict.fromkeys(([config['checkpoint_base_dir']] if config.get('checkpoint_base_dir') else [])+
                                [str(p) for p in artifact_roots if p.name!='simulation-artifacts' and p!=main])))}
    local_artifact_roots={Path(v.get('artifact_root') or Path(v['dataset_root']).parent/'simulation-artifacts').resolve() for v in config.get('local_storage_locations',{}).values() if v.get('dataset_root')}
    for number,root in enumerate(artifact_roots):
        if root==main or root.name!='simulation-artifacts' or root in local_artifact_roots:continue
        name='local_disk_'+str(number)
        result[name]=dict(name=name,label='4090 · '+str(root),local=True,artifact_root=str(root),artifact_roots=[str(root)],dataset_roots=[])
    for name,value in config.get('local_storage_locations',{}).items():
        dataset=value.get('dataset_root')
        if dataset:
            artifact_root=str(Path(value.get('artifact_root') or Path(dataset).parent/'simulation-artifacts').resolve())
            result[name]=dict(name=name,label=value.get('label',name),local=True,artifact_root=artifact_root,artifact_roots=[artifact_root],dataset_roots=[dataset],dataset_root=dataset)
    for name,value in config.get('cluster_targets',{}).items():
        if not all(value.get(k) for k in ('submit_host','partition','node','openpi_python')):continue
        root=value.get('experiment_artifact_root','/DATA/disk0/sunny/simulation-artifacts')
        result[name]=dict(value,name=name,label=value.get('label',name),local=False,artifact_root=root,artifact_roots=[root],
                          dataset_roots=[value['dataset_root']] if value.get('dataset_root') else [],
                          legacy_roots=list(dict.fromkeys([p for p in [value.get('checkpoint_base_dir'),'/DATA/disk0/sunny/experiments',
                            '/DATA/disk0/sunny/models','/DATA/disk0/sunny/checkpoints','/DATA/disk0/sunny/.cache/openpi',
                            '/DATA/disk0/sunny/.cache/huggingface/hub'] if p])))
        if value.get('access_mode')=='login_shared_slurm':result[name]['artifact_roots'].append('/home/sunny/simulation-artifacts')
    return result


class Replicas:
    def __init__(self,config):
        self.config=config;self.inventory=Inventory(config)
        with self.inventory.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS replicas(id TEXT PRIMARY KEY,logical_id TEXT,node TEXT,path TEXT,status TEXT,revision INTEGER,data TEXT);
              CREATE INDEX IF NOT EXISTS replica_logical ON replicas(logical_id,status);
              CREATE TABLE IF NOT EXISTS replica_scans(node TEXT PRIMARY KEY,updated REAL,errors TEXT);
              CREATE TABLE IF NOT EXISTS replica_operations(id TEXT PRIMARY KEY,copy_id TEXT,action TEXT,state TEXT,updated REAL,error TEXT);''')

    def mark_dirty(self,nodes=None):
        nodes=list(nodes or locations(self.config))
        with self.inventory.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for node in nodes:
                db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('storage_dirty_'+node,str(self.inventory.bump(db))))

    def get(self,ident):
        with self.inventory.connect() as db:row=db.execute('SELECT * FROM replicas WHERE id=?',(ident,)).fetchone()
        if not row:raise ValueError('Unknown replica')
        return dict(json.loads(row['data']),status=row['status'],revision=row['revision'])

    def invalidate_local_paths(self,paths):
        targets=[Path(p).resolve() for p in paths];nodes=set()
        local={k for k,v in locations(self.config).items() if v['local']}
        with self.inventory.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute("SELECT id,node,path,status FROM replicas WHERE status='present'").fetchall():
                if row['node'] not in local:continue
                stored=Path(row['path'])
                if any(p==stored or stored in p.parents or p in stored.parents for p in targets):
                    db.execute("UPDATE replicas SET status='stale',revision=? WHERE id=?",(self.inventory.bump(db),row['id']));nodes.add(row['node'])
        return nodes

    def merge(self,snapshot,started_revision):
        node=snapshot['node'];seen=set()
        with self.inventory.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for copy in snapshot['copies']:
                seen.add(copy['id'])
                old=db.execute('SELECT * FROM replicas WHERE id=?',(copy['id'],)).fetchone()
                if old and (old['revision']>started_revision or old['status'] in {'deleting','transferring','operation_failed'}):continue
                if old and old['status']=='deleted' and json.loads(old['data']).get('version')==copy.get('version'):continue
                db.execute('INSERT OR REPLACE INTO replicas VALUES(?,?,?,?,?,?,?)',(copy['id'],copy['logical_id'],node,copy['path'],'present',self.inventory.bump(db),json.dumps(copy)))
            if not snapshot.get('errors'):
                for old in db.execute("SELECT id,revision,status,data FROM replicas WHERE node=? AND status IN ('present','stale')",(node,)).fetchall():
                    if snapshot.get('partial_scan') and json.loads(old['data']).get('kind') in ({'unclassified'} if snapshot.get('include_datasets') else {'dataset','unclassified'}):continue
                    if old['id'] not in seen and old['revision']<=started_revision:
                        db.execute("UPDATE replicas SET status='missing',revision=? WHERE id=?",(self.inventory.bump(db),old['id']))
            db.execute('INSERT OR REPLACE INTO replica_scans VALUES(?,?,?)',(node,snapshot['updated_at'],json.dumps(snapshot.get('errors',[]))))
            previous=db.execute('SELECT value FROM meta WHERE key=?',('storage_resources_'+node,)).fetchone()
            usage=[dict(item,measured_at=item.get('measured_at',snapshot['updated_at'])) for item in snapshot.get('owned_usage',[])] or (json.loads(previous[0]).get('owned_usage',[]) if previous else [])
            db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('storage_resources_'+node,json.dumps(dict(disks=snapshot.get('resources',[]),owned_usage=usage))))

    def page(self,offset=0,limit=25,kind='all',duplicates=False,q='',node='all',area='all',backend='all'):
        if not 0<=offset<=10000000 or not 1<=limit<=100:raise ValueError('Invalid storage page')
        where="status IN ('present','deleting','transferring','operation_failed','stale')";args=[]
        if kind!='all':where+=" AND json_extract(data,'$.kind')=?";args.append(kind)
        if node!='all':where+=' AND node=?';args.append(node)
        areas={'training':['model','logs','metrics','metadata','unclassified'],'rollout':['rollout','unclassified'],'datasets':['dataset']}
        if area in areas:
            where+=" AND json_extract(data,'$.kind') IN ("+','.join('?' for _ in areas[area])+')';args.extend(areas[area])
        if backend != 'all':
            if backend not in {'openpi', 'rlinf'}: raise ValueError('Unknown training backend')
            where+=" AND json_extract(data,'$.run.backend')=?";args.append(backend)
        if q:where+=' AND path LIKE ?';args.append('%'+q+'%')
        with self.inventory.connect() as db:
            total=db.execute('SELECT count(*) FROM replicas WHERE '+where,args).fetchone()[0]
            rows=db.execute('SELECT * FROM replicas WHERE '+where+" ORDER BY CASE WHEN node LIKE 'local_%' THEN 0 ELSE 1 END,node,path LIMIT ? OFFSET ?",args+[limit,offset]).fetchall()
            items=[]
            for row in rows:
                item=json.loads(row['data']);files=item.pop('files',[])
                item.pop('checkpoint_index',None);item.pop('logical_id',None);item.pop('logical',None)
                item['file_states']=item.get('file_states_summary') or {state:sum(f.get('file_state')==state for f in files) for state in ('verified','structure_checked','present_unverified','incomplete','link')}
                item['file_types']=item.get('file_types_summary') or {kind:sum(f.get('kind')==kind for f in files) for kind in ('video','trajectory','image','feature','metadata')}
                item['status']=row['status'];items.append(item)
            scans=[dict(row,errors=json.loads(row['errors'])) for row in db.execute('SELECT * FROM replica_scans')]
            resources={row['key'].removeprefix('storage_resources_'):json.loads(row['value']) for row in db.execute("SELECT key,value FROM meta WHERE key LIKE 'storage_resources_%'")}
        return dict(resources=resources,items=items,total=total,offset=offset,limit=limit,scans=scans,
                    locations=[dict(id=k,label=v['label'],artifact_root=v['artifact_root'],dataset_root=v.get('dataset_root')) for k,v in locations(self.config).items()])

    def claim(self,copy_id,action):
        copy=self.get(copy_id)
        if copy['status'] not in {'present','operation_failed'} or not copy['mutable']:raise ValueError('Replica is busy, missing, or contains links')
        with self.inventory.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT revision FROM replicas WHERE id=?',(copy_id,)).fetchone()
            if row['revision']!=copy['revision']:raise ValueError('Replica changed; refresh')
            operation=uuid.uuid4().hex
            db.execute('INSERT INTO replica_operations VALUES(?,?,?,?,?,?)',(operation,copy_id,action,'pending',time.time(),None))
            db.execute('UPDATE replicas SET status=?,revision=? WHERE id=?',('deleting' if action=='delete' else 'transferring',self.inventory.bump(db),copy_id))
        return operation,copy

    def finish(self,operation,error=None):
        with self.inventory.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM replica_operations WHERE id=?',(operation,)).fetchone()
            if not row:raise ValueError('Unknown operation')
            state='operation_failed' if error else 'deleted' if row['action']=='delete' else 'present'
            db.execute('UPDATE replicas SET status=?,revision=? WHERE id=?',(state,self.inventory.bump(db),row['copy_id']))
            db.execute('UPDATE replica_operations SET state=?,updated=?,error=? WHERE id=?',('failed' if error else 'completed',time.time(),error,operation))


def storage_worker(location,request):
    request=dict(request)
    if 'copy' in request:
        # The worker rechecks paths and fingerprints; file listings stay in the index.
        request['copy']={k:v for k,v in request['copy'].items() if k not in {'files','file_states_summary','file_types_summary'}}
    if location['local']:
        return {'history_cleanup':worker.history_cleanup,'discover_models':worker.discover_models,'run_summary':worker.run_summary,'prepare_checkpoint':worker.prepare_checkpoint}[request['action']](request) if request['action'] in {'history_cleanup','discover_models','run_summary','prepare_checkpoint'} else worker.main(request)
    read_only=request['action'] in {'inventory','discover_models','run_summary','prepare_checkpoint'}
    if read_only:
        from .cluster_read_guard import admit_read
        admit_read(location,'storage_'+request['action']+'_')
    code=Path(worker.__file__).read_text()
    # This is a storage job. Do not inherit model/environment preflight checks.
    target={k:location[k] for k in ('submit_host','partition','node','remote_job_dir','log_dir') if location.get(k)}
    target.update(gpu_count=0,ntasks=1,cpus_per_task=2,mem='8G',time='04:00:00',workdir='/DATA/disk0/sunny',
                  dataset_root=location.get('dataset_root','/DATA/disk0/sunny'),cache_root='/DATA/disk0/sunny/.cache',tmpdir='/tmp')
    request=dict(request)
    if location.get('access_mode') == 'slurm_only':
        target.update(log_dir='/tmp',log_transport='sattach')
        request['attach_delay']=15
    if request['action'] in {'inventory','history_cleanup','discover_models','run_summary','prepare_checkpoint'}:
        if location.get('access_mode') == 'login_shared_slurm':
            request['output']='/DATA/disk0/sunny/.storage-index/'+uuid.uuid4().hex+'.json'
        else:
            request.pop('output',None)
            request['receipt_stdout']=True
    command=[location.get('storage_python') or location['openpi_python'],'-c',code,json.dumps(request)]
    trace_root=Path(tempfile.gettempdir())/'bimanual-storage-traces';trace_root.mkdir(exist_ok=True)
    trace=trace_root/(uuid.uuid4().hex+'.log')
    print('Storage Slurm trace: '+str(trace),flush=True)
    with trace.open('w') as stream:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('slurm_job_runner.py')),'--target-json',encoded(target),
                    '--commands-json',encoded([command]),'--command-labels-json',encoded([request['action']]),
                    '--job-name','storage_'+request['action']+'_'+uuid.uuid4().hex[:8],'--poll-interval','5']+(['--cancel-on-exit','--max-pending-seconds','60'] if read_only else []),stdout=stream,stderr=subprocess.STDOUT)
    receipt=None
    with trace.open() as stream:
        for line in stream:
            if line.startswith('STORAGE_RECEIPT:'):
                try:receipt=json.loads(zlib.decompress(base64.b64decode(line.split(':',1)[1])))
                except (ValueError,zlib.error):pass
            else:print(line.rstrip(),flush=True)
    if result.returncode:raise RuntimeError(f"Storage worker failed on {location['name']} (exit {result.returncode}); trace {trace}")
    if request.get('receipt_stdout') and receipt is None:raise RuntimeError('Slurm finished without a complete inventory receipt; trace '+str(trace))
    if request.get('output'):
        receipt=read_remote(location,request['output'])
        from .slurm_job_runner import run_ssh
        run_ssh(location['submit_host'],shlex.join(['rm','--',request['output']]),timeout=30)
    return receipt



def read_remote(location,path):
    from .slurm_job_runner import run_ssh
    result=run_ssh(location['submit_host'],shlex.join(['cat',str(path)]),timeout=60)
    if result.returncode:raise RuntimeError('Could not read replica receipt')
    return json.loads(result.stdout)


def refresh_node(config,name,quick=False,include_datasets=False):
    registry=Replicas(config);loc=locations(config)[name];version=registry.inventory.revision()
    datasets=[] if quick and not include_datasets else loc['dataset_roots'];legacy=[] if quick else loc.get('legacy_roots',[])
    if loc['local']:
        snapshot=worker.inventory(dict(node=name,artifact_roots=loc['artifact_roots'],dataset_roots=datasets,legacy_roots=legacy))
    else:
        output_root=Path('/DATA/disk0/sunny/.storage-index') if loc.get('access_mode')=='login_shared_slurm' else Path(loc.get('nas_dataset_staging_root','/DATA/NAS/GPUServer/sunny/dashboard_dataset_sync'))/'replica-inventory'
        output=str(output_root/(name+'-'+uuid.uuid4().hex+'.json'))
        snapshot=storage_worker(loc,dict(action='inventory',node=name,artifact_roots=loc['artifact_roots'],dataset_roots=datasets,legacy_roots=legacy,usage_roots=[] if quick else ['/DATA/disk0/sunny'],output=output))
    snapshot['partial_scan']=quick
    snapshot['include_datasets']=include_datasets
    registry.merge(snapshot,version)
    return dict(node=name,copies=len(snapshot['copies']),errors=snapshot.get('errors',[]))


def transfer(config,copy,target_name,kinds=None):
    locs=locations(config);source=locs[copy['node']];target=locs[target_name]
    destroot=target.get('dataset_root') if copy['kind']=='dataset' else target['artifact_root']
    if not destroot:raise ValueError('Destination has no configured root for this category')
    destination=Path(destroot)/copy.get('destination_relative',copy['relative_path'])
    if source['name']==target_name and str(destination)==copy['path']:raise ValueError('Source and destination are identical')
    operation=uuid.uuid4().hex
    local_root=Path(config.get('local_storage_locations',{}).get('local_4090_hdd',{}).get('dataset_root',config['workspace_root'])).parent/'.replica-staging'
    remote=source if not source['local'] else target if not target['local'] else None
    nas_root=Path((remote or {}).get('nas_dataset_staging_root','/DATA/NAS/GPUServer/sunny/dashboard_dataset_sync'))/'replica-staging'
    bundle_root=lambda loc: local_root if loc['local'] else Path('/DATA/disk0/sunny/.replica-staging') if loc.get('access_mode')=='login_shared_slurm' else nas_root
    export_bundle=bundle_root(source)/operation
    usage_roots=lambda loc: [] if loc['local'] else ['/DATA/disk0/sunny']
    storage_worker(target,dict(action='preflight',destination_root=destroot,additional_bytes=worker.projected_bytes(copy)*2,usage_roots=usage_roots(target)))
    storage_worker(source,dict(action='export',copy=copy,kinds=kinds,bundle=str(export_bundle),usage_roots=usage_roots(source) if str(export_bundle).startswith('/DATA/disk0/') else []))
    import_bundle=bundle_root(target)/operation
    if export_bundle!=import_bundle:
        src=dict(name='replica_bundle_source',kind='local' if source['local'] else 'ssh',dataset_root=str(export_bundle.parent))
        dst=dict(name='replica_bundle_target',kind='local' if target['local'] else 'ssh',dataset_root=str(import_bundle.parent))
        if not source['local']:src.update(host=source['submit_host'],access_mode='login_shared_slurm')
        if not target['local']:dst.update(host=target['submit_host'],access_mode='login_shared_slurm')
        subprocess.run([sys.executable,str(Path(__file__).with_name('dataset_transfer_runner.py')),'--dataset-id',operation,
                        '--source-json',encoded(src),'--target-json',encoded(dst),'--parallelism','2'],check=True)
    storage_worker(target,dict(action='import',node=target_name,bundle=str(import_bundle),destination_root=destroot,usage_roots=usage_roots(target)))
    cleanup_errors=[]
    for loc,bundle in [(target,import_bundle)]+([(source,export_bundle)] if export_bundle!=import_bundle else []):
        try:storage_worker(loc,dict(action='cleanup',bundle=str(bundle)))
        except Exception as exc:cleanup_errors.append(str(exc))
    return dict(target=target_name,path=str(destination),verification='sha256_selected_unit',staging_cleanup_errors=cleanup_errors)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--action',choices=['refresh','copy','delete'],required=True)
    parser.add_argument('--local-only',action='store_true');parser.add_argument('--quick',action='store_true');parser.add_argument('--scheduled',action='store_true');parser.add_argument('--kinds');parser.add_argument('--copy-id');parser.add_argument('--target');parser.add_argument('--node');args=parser.parse_args()
    config=json.loads(os.path.expandvars(Path(args.config).read_text()));registry=Replicas(config)
    os.environ['BIMANUAL_STORAGE_TASK_ROOT']=str(Path(config['workspace_root'])/'tasks')
    if args.action=='refresh':
        import fcntl
        scan_lock=registry.inventory.path.with_suffix('.storage-refresh.lock').open('w')
        try:fcntl.flock(scan_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Storage refresh already running');return
        available=locations(config)
        names=[args.node] if args.node else [n for n in available if available[n]['local']]
        # Remote indexes are fixed until the user explicitly selects a source.
        if args.scheduled or args.local_only:names=[n for n in names if available[n]['local']]
        with registry.inventory.connect() as db:
            dirty=dict(db.execute("SELECT key,value FROM meta WHERE key LIKE 'storage_dirty_%'"))
            scans=dict(db.execute('SELECT node,updated FROM replica_scans'))
        if args.scheduled:
            names=[n for n in names if 'storage_dirty_'+n in dirty or time.time()-scans.get(n,0)>=900]
            args.quick=True
        def one(name):
            try:
                result=refresh_node(config,name,args.quick,include_datasets='storage_dirty_'+name in dirty)
                with registry.inventory.connect() as db:
                    db.execute('DELETE FROM meta WHERE key=? AND value=?',('storage_dirty_'+name,dirty.get('storage_dirty_'+name)))
                return result
            except Exception as exc:
                with registry.inventory.connect() as db:db.execute('INSERT OR REPLACE INTO replica_scans VALUES(?,?,?)',(name,time.time(),json.dumps([str(exc)])))
                return dict(node=name,error=str(exc))
        with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,names))
        if names:registry.inventory.reconcile()
        print(json.dumps(results))
        if any('error' in r for r in results):raise SystemExit(1)
        return
    kinds=json.loads(args.kinds) if args.kinds else None
    operation,copy=registry.claim(args.copy_id,args.action)
    try:
        loc=locations(config)[copy['node']]
        if args.action=='delete':storage_worker(loc,dict(action='delete',copy=copy,kinds=kinds))
        else:print(json.dumps(transfer(config,copy,args.target,kinds)))
    except BaseException as exc:
        registry.finish(operation,str(exc));raise
    registry.finish(operation)
    if kinds and args.action=='delete':
        with registry.inventory.connect() as db:
            db.execute("UPDATE replicas SET status='stale',revision=? WHERE id=?",(registry.inventory.bump(db),copy['id']))
    # A successful filesystem operation stays successful if reconciliation fails.
    for node in dict.fromkeys([copy['node']]+([args.target] if args.target else [])):
        try:refresh_node(config,node,quick=copy['kind']!='dataset')
        except Exception as exc:print('Replica operation completed; inventory refresh pending: '+str(exc),flush=True)
    registry.inventory.reconcile()


if __name__=='__main__':main()
