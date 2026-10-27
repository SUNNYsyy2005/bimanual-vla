"""Robot/action identities: explicit metadata first, exact dataset provenance second."""
from pathlib import Path
import json

FIELDS = {'robot_type':'机械臂型号', 'arm_mode':'单双臂', 'action_space':'动作输出类型'}
UNKNOWN = {'', 'unknown', 'custom', 'none'}


def normalize_robot(value):
    if value is None:return None
    text=str(value).strip().lower().replace('_','-').replace(' ','-')
    aliases={'panda':'franka-panda','franka':'franka-panda','aloha':'aloha-agilex','agilex':'aloha-agilex',
        'piper-single-arm':'piper','piper-bimanual':'piper','piper-single':'piper'}
    return aliases.get(text,text) or None


def identity(value):
    contract=value.get('action_contract') or {}
    data={**value, **contract}
    robot=normalize_robot(data.get('robot_model') or data.get('robot_type'))
    mode=str(data.get('arm_mode') or data.get('arm_layout') or '').lower()
    mode={'single_arm':'single','dual':'bimanual','dual_arm':'bimanual'}.get(mode,mode)
    space=str(data.get('action_space') or data.get('action_representation') or data.get('schema') or '').lower()
    space={'joint':'joint','joint_position':'joint','joint_angles':'joint','delivery':'eef_pose','eef':'eef_pose','ee_pose':'eef_pose','eef_pose':'eef_pose','end_effector_pose':'eef_pose'}.get(space,space)
    return {k:v for k,v in dict(robot_type=robot,arm_mode=mode,action_space=space,
        arm_side=data.get('arm_side'),model_action_dim=data.get('model_action_dim'),
        model_action_convention=data.get('model_action_convention'),gripper_semantics=data.get('gripper_semantics')).items()
        if v is not None and str(v).lower() not in UNKNOWN}


def enrich(models, config, datasets=None):
    known={d['id']:identity(d) for d in datasets or []}
    roots=list(dict.fromkeys([config.get('dataset_root',''),*config.get('dataset_read_roots',[]),
        *[v.get('dataset_root','') for v in config.get('local_storage_locations',{}).values()]]))
    def dataset(ident):
        if ident not in known:
            known[ident]={}
            if Path(ident).name != ident:return {}
            for root in roots:
                if not root:continue
                path=Path(root)/ident/'meta/info.json'
                try:info=json.loads(path.read_text())
                except (OSError,ValueError):continue
                from server_4090.app import describe_dataset_schema
                known[ident]=identity({**info,**describe_dataset_schema(info)})
                break
        return known[ident]
    result=[]
    for model in models:
        current=identity(model); conflicts=list(model.get("identity_conflicts") or [])
        ids=model.get('dataset_ids') or ([model['action_contract']['dataset_id']] if (model.get('action_contract') or {}).get('dataset_id') else [])
        sources=[dataset(i) for i in ids]
        for key in (*FIELDS,'arm_side','model_action_dim','model_action_convention','gripper_semantics'):
            values={source[key] for source in sources if source.get(key) is not None}
            # Robot and arm identity must agree with all known source datasets.
            # The model's output convention may intentionally differ from raw data.
            if len(values)>1 or (key in FIELDS and current.get(key) and values and current[key] not in values):
                conflicts.append(key)
            elif key not in current and len(values)==1:current[key]=next(iter(values))
        result.append(dict(model,model_identity=current,identity_conflicts=conflicts))
    return result


def mismatch(model, expected):
    actual=model.get('model_identity') or identity(model)
    target=expected.get('model_identity') or identity(expected)
    if model.get('identity_conflicts'):return '模型与来源数据集的机械臂 / 动作元数据冲突，待核实'
    for key,label in FIELDS.items():
        if not target.get(key):return '目标'+label+'未标注，待核实'
        if not actual.get(key):return '模型'+label+'未标注，待核实'
        if actual[key]!=target[key]:return label+'不匹配'
    for key,label in [('arm_side','机械臂侧'),('model_action_dim','动作维度'),('model_action_convention','动作约定'),('gripper_semantics','夹爪语义')]:
        if target.get(key) is not None:
            if actual.get(key) is None:return '模型'+label+'未标注，待核实'
            if actual[key]!=target[key]:return label+'不匹配'
    return None


def rlinf_run_contract(config, run):
    """Use recorded launch provenance only, never the evaluator's fallback profile."""
    if run.get('action_contract'):return run['action_contract']
    target=str(run.get('execution_target') or run.get('target') or '').lower()
    profile=run.get('launch_profile')
    return config.get('rlinf_dashboard',{}).get('targets',{}).get(target,{}).get('profiles',{}).get(profile,{}).get('action_contract',{})
