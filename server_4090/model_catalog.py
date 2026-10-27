"""Read model identities and sizes from the storage index, independently of history."""
import json
from pathlib import Path
from .task_scenes import annotate, bindings, normalize


def is_foundation_path(value):
    """Known original OpenPI weights, never inferred from being the configured default."""
    return bool(value and Path(str(value).rstrip('/')).name in {'pi0_base','pi05_base','pi0_fast_base'})


def backend_name(value):
    return 'openpi' if value in {'openpi', 'sft'} else 'rlinf' if value in {'rlinf', 'grpo', 'ppo'} else value


def indexed_models(registry, backend=None, query=''):
    models = {}
    with registry.inventory.connect() as db:
        rows = db.execute("SELECT data,status FROM replicas WHERE status IN ('present','stale') AND json_extract(data,'$.kind')='model'").fetchall()
    for row in rows:
        item = json.loads(row['data']); run = item.get('run') or {}
        index = (item.get('checkpoint_index') or {}).get('checkpoints', [])
        checkpoint = next((c for c in index if str(Path(item.get('root', '')) / item.get('run_relative', '') / c.get('path', '')) == item['path']), {})
        data = dict(id=item['id'], path=item['path'], node=item['node'], backend=backend_name(run.get('backend')),
                    experiment=run.get('experiment'), run_id=run.get('run_id'), task=run.get('task'), task_scene=run.get('task_scene'),
                    dataset_ids=checkpoint.get('dataset_ids') or run.get('dataset_ids') or [], size_bytes=item.get('size_bytes'),
                    step=checkpoint.get('step'), format=checkpoint.get('format'), complete=checkpoint.get('complete'),
                    mutable=item.get('mutable', False), status=row['status'], action_contract=checkpoint.get('action_contract') or run.get('action_contract'))
        models[(data['node'],data['path'])] = data
    for item in registry.inventory.models():
        data = dict(item, node='local_4090', backend=backend_name(item.get('backend')))
        # Preserve the real disk label where the storage index has it.
        if not any(m['path'] == data['path'] for m in models.values()): models[('local_4090',data['path'])] = data
    associations=bindings(registry.config)
    models={k:annotate(m,associations) for k,m in models.items()}
    result = [m for m in models.values() if (not backend or backend == 'all' or m.get('backend') == backend_name(backend))
              and (not query or query.lower() in ' '.join(str(m.get(k,'')) for k in ('path','experiment','task','run_id','dataset_ids','step')).lower())]
    from .model_identity import enrich
    from .rlinf_identity import annotate_model
    result=[annotate_model(registry.config,m) if m.get("backend")=="rlinf" else m for m in result]
    result=enrich(result, registry.config)
    return sorted(result, key=lambda m: (str(m.get('experiment') or ''), str(m.get('path') or '')))


def compatibility(model, *, backend, dataset_ids=(), task=None, contract=None):
    """Provenance alone does not establish cross-format restore compatibility."""
    if model.get('complete') is not True: return '完整性尚未确认'
    formats={'openpi':{'orbax','openpi_orbax'},'rlinf':{'rlinf_full_weights','rlinf_safetensors'}}
    if model.get('format') not in formats[backend]: return '需要转换权重格式后才能由此训练后端加载'
    requested=normalize(task)
    scenes=model.get('task_scenes') or annotate(model)['task_scenes']
    if requested and scenes and requested not in scenes:
        return '任务场景不匹配'
    # Missing scene metadata is visible; dataset identity is never a load gate.
    expected=contract or {}; actual=model.get('action_contract') or {}
    from .model_identity import mismatch
    reason=mismatch(model, expected)
    if reason:return reason
    for key in ('model_variant','model_action_dim','model_action_convention','gripper_semantics'):
        if expected.get(key) is not None and expected[key]!=actual.get(key): return '模型或动作配置不兼容：'+key
    if model.get('backend')!=backend and not expected: return '跨后端动作配置尚未核实'
    return None


def rlinf_options(config, registry, target_id, profile_id, task_scene=None):
    import re
    target=config.get('rlinf_dashboard',{}).get('targets',{}).get(target_id,{})
    profile=target.get('profiles',{}).get(profile_id,{})
    if not target.get('enabled') or not profile or not profile.get('enabled',True): raise ValueError('Unknown RLinf launch profile')
    template=' '.join(profile.get('command_template',[]))
    match=re.search(r'actor\.model\.model_path=([^\s]+)',template)
    # The profile's default is loaded by its existing YAML if no override is present.
    base=profile.get('base_checkpoint') or (match[1] if match else None)
    foundation=is_foundation_path(base)
    category='foundation' if foundation else 'checkpoint' if base else 'profile_default'
    label='基础模型' if foundation else '配置默认 checkpoint' if base else '配置默认初始化权重（类型待确认）'
    models=[dict(id='profile-default',path=base,foundation=foundation,category=category,action_contract=profile.get('action_contract'),compatible=True,label=label+(' · '+base if base else ''))]
    node='local_4090' if target_id=='4x4090' else target_id
    for model in indexed_models(registry):
        local=model.get('node','').startswith('local_')
        if not (node=='local_4090' and local or model.get('node')==node or node=='h100' and model.get('node')=='h100-ksy-01'): continue
        reason=compatibility(model,backend='rlinf',dataset_ids=profile.get('dataset_ids',[]),task=task_scene or profile.get('task_scene') or profile.get('task'),contract=profile.get('action_contract'))
        models.append(dict(model,compatible=reason is None,reason=reason,label=f"[{model.get('task_scene') or '场景未标注'} · {model.get('backend') or '未知来源'}] {model.get('experiment') or model.get('run_id')} / {Path(model['path']).name}"))
    return models
