"""Single-checkpoint operations; discovery/index refresh is never a prerequisite."""
import argparse
import json
import os
from pathlib import Path

from .storage_catalog import Replicas, locations, storage_worker, transfer


def source_location(config, node, path):
    available=locations(config)
    if node not in available:raise ValueError('Unknown checkpoint source')
    requested=Path(path)
    if not requested.is_absolute() or '..' in requested.parts:raise ValueError('An absolute checkpoint path is required')
    location=available[node]
    candidates=[location]
    if location['local']:candidates=[v for v in available.values() if v['local']]
    matches=[]
    for candidate in candidates:
        roots=list(candidate.get('artifact_roots',[]))+list(candidate.get('legacy_roots',[]))
        if candidate['local']:
            roots+=config.get('checkpoint_allowed_roots',[])
        else:roots+=candidate.get('checkpoint_read_roots',[])
        for root in roots:
            if requested!=Path(root) and requested.is_relative_to(root):matches.append((len(Path(root).parts),candidate,roots))
    if not matches:raise ValueError('Checkpoint is outside configured storage roots')
    _,location,roots=max(matches,key=lambda item:item[0])
    return location,list(dict.fromkeys(roots))


def publish(registry, copy, revision):
    with registry.inventory.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old=db.execute('SELECT status,revision FROM replicas WHERE id=?',(copy['id'],)).fetchone()
        if old and (old['revision']>revision or old['status'] in {'deleting','transferring'}):raise ValueError('Checkpoint changed or another operation is running')
        db.execute('INSERT OR REPLACE INTO replicas VALUES(?,?,?,?,?,?,?)',(copy['id'],copy['logical_id'],copy['node'],copy['path'],'present',registry.inventory.bump(db),json.dumps(copy)))


def operate(config, node, path, backend, action, target=None):
    registry=Replicas(config);location,roots=source_location(config,node,path)
    revision=registry.inventory.revision()
    copy=storage_worker(location,dict(action='prepare_checkpoint',node=location['name'],path=path,backend=backend,roots=roots))
    publish(registry,copy,revision)
    operation,copy=registry.claim(copy['id'],action)
    try:
        if action=='delete':
            storage_worker(location,dict(action='delete',copy=copy))
            result=dict(deleted=True,path=path,node=location['name'])
        else:
            result=transfer(config,copy,target)
            # Register just the verified destination, not all experiments on the node.
            dest=locations(config)[target]
            try:
                revision=registry.inventory.revision()
                observed=storage_worker(dest,dict(action='prepare_checkpoint',node=target,path=result['path'],backend=backend,roots=dest['artifact_roots']))
                publish(registry,observed,revision)
            except Exception as exc:result['index_warning']=str(exc)
    except BaseException as exc:
        registry.finish(operation,str(exc));raise
    registry.finish(operation)
    if action=='delete' and location['local']:registry.inventory.invalidate_missing(path)
    registry.mark_dirty([location['name']]+([target] if target else []))
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','node','path','backend','action'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--target')
    args=parser.parse_args()
    config=json.loads(os.path.expandvars(Path(args.config).read_text()))
    os.environ['BIMANUAL_STORAGE_TASK_ROOT']=str(Path(config['workspace_root'])/'tasks')
    operate(config,args.node,args.path,args.backend,args.action,args.target)


if __name__=='__main__':main()
