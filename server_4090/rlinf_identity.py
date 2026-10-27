"""Enrich RLinf identities from audited environment/config provenance, never names."""
from pathlib import PurePosixPath
from .model_identity import identity


def node_id(value):
    value=str(value or '').lower()
    return 'local_4090' if value=='4x4090' or value.startswith('local_') else 'h100' if value=='h100-ksy-01' else value


def contract_from_config(config, recorded):
    rules=config.get('rlinf_dashboard',{}).get('verified_action_environments',{})
    rule=rules.get(recorded.get('benchmark')) or {}
    if not rule or recorded.get('safety_cost_protocol')!=rule.get('safety_cost_protocol'):
        return {}
    # Overrides change the environment and invalidate the audited default.
    if any(recorded.get(k) is not None for k in ('robots','robot','robot_type','controller','controller_configs','env')):
        return {}
    return dict(rule['action_contract'])


def annotate_run(config, run):
    existing=run.get('action_contract') or {}
    paths={str(PurePosixPath(p)) for p in [run.get('path'),run.get('_physical_path'),*(run.get('_aliases') or [])] if p}
    node=node_id(run.get('execution_target') or run.get('target') or run.get('node'))
    for record in config.get('rlinf_dashboard',{}).get('verified_run_contracts',[]):
        if node_id(record['node'])!=node or not paths.intersection(record['paths']):continue
        actual=identity({'action_contract':existing});verified=identity({'action_contract':record['action_contract']})
        if any(k in actual and actual[k]!=v for k,v in verified.items()):
            return dict(run,identity_conflicts=['audited_run_contract'])
        return dict(run,action_contract={**record['action_contract'],**{k:v for k,v in existing.items() if str(v).lower() not in {'unknown','none',''}}},
                    identity_evidence=record['evidence'])
    return run


def annotate_model(config, model):
    node=node_id(model.get('node'))
    path=PurePosixPath(model.get('path') or '/')
    for record in config.get('rlinf_dashboard',{}).get('verified_run_contracts',[]):
        if node_id(record['node'])!=node:continue
        if any(path.is_relative_to(PurePosixPath(root)/'checkpoints') for root in record['paths']):
            return annotate_run(config,dict(model,target=node,path=record['paths'][0])) | {'path':str(path)}
    return model
