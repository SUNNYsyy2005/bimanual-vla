"""Task-scene associations, independent of dataset and training-backend identity."""
import fcntl
import json
from pathlib import Path
import re


def normalize(value):
    value=str(value or '').strip().lower().replace('_','-')
    return '' if value in {'','unknown','未标注'} else value


def bindings(config):
    path=Path(config['workspace_root'])/'task-scene-bindings.json'
    try:return json.loads(path.read_text())
    except FileNotFoundError:return {}


def bind(config,kind,key,scene):
    if kind not in {'dataset','model'} or not isinstance(key,str) or not key or len(key)>4096:raise ValueError('Invalid scene association')
    if kind=='model' and key.split(':',1)[0].startswith('local_'):key='local_4090:'+key.split(':',1)[1]
    scene=normalize(scene)
    if len(scene)>120 or any(ord(c)<32 for c in scene):raise ValueError('Invalid task scene')
    path=Path(config['workspace_root'])/'task-scene-bindings.json';path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=bindings(config);data.setdefault(kind,{})[key]=scene
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False));temp.replace(path)
    return scene


def dataset_scene(dataset_id, metadata=None, associations=None):
    metadata=metadata or {};associations=associations or {}
    if dataset_id in associations.get('dataset',{}):return normalize(associations['dataset'][dataset_id])
    explicit=metadata.get('task_scene') or metadata.get('scenario') or metadata.get('task_name') or metadata.get('task')
    if isinstance(explicit,str) and normalize(explicit):return normalize(explicit)
    # Existing pipeline naming conventions only; no fuzzy matching of arbitrary names.
    name=normalize(dataset_id)
    for prefix,scene in [('handover-mic','handover-mic'),('put-bottles-dustbin','put-bottles-dustbin'),('put-bottle-dustbin','put-bottles-dustbin'),('tsa-basket','tsa-basket'),('tsa-microwave','tsa-microwave'),('tsa-moka','tsa-moka'),('fshoa-microwave','fshoa-microwave')]:
        if name==prefix or name.startswith(prefix+'-'):return scene
    return ''


def annotate(model, associations=None):
    associations=associations or {};result=dict(model)
    node=str(model.get('node') or 'local_4090')
    key=('local_4090' if node.startswith('local_') else node)+':'+str(model.get('path',''))
    override=associations.get('model',{}).get(key)
    scene=normalize(override if override is not None else model.get('task_scene') or model.get('task') or model.get('task_name'))
    scenes=[scene] if scene else []
    if override is None and not scenes:
        scenes=sorted({s for d in model.get('dataset_ids',[]) if (s:=dataset_scene(d,associations=associations))})
    result.update(task_scenes=scenes,task_scene=scenes[0] if len(scenes)==1 else None)
    return result
