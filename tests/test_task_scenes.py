from server_4090.task_scenes import annotate,bind,bindings,dataset_scene
from server_4090.model_catalog import compatibility


def test_scene_shared_across_datasets_and_backends(tmp_path):
    cfg={'workspace_root':str(tmp_path)}
    bind(cfg,'dataset','sft-training-v1','tsa_basket')
    bind(cfg,'dataset','sft-finetune-v2','tsa-basket')
    mapping=bindings(cfg)
    sft=annotate(dict(dataset_ids=['sft-training-v1'],backend='openpi',format='orbax',complete=True,action_contract={'model_action_dim':7,'robot_type':'piper','arm_mode':'single','schema':'joint'}),mapping)
    assert sft['task_scene']==dataset_scene('sft-finetune-v2',associations=mapping)=='tsa-basket'
    assert compatibility(sft,backend='openpi',dataset_ids=['different-dataset'],task='tsa_basket',contract={'model_action_dim':7,'robot_type':'piper','arm_mode':'single','schema':'joint'}) is None
    converted_rl=annotate(dict(sft,backend='rlinf'),mapping)
    assert compatibility(converted_rl,backend='openpi',task='tsa-basket',contract={'model_action_dim':7,'robot_type':'piper','arm_mode':'single','schema':'joint'}) is None
    assert '转换' in compatibility(sft,backend='rlinf',task='tsa-basket',contract={'model_action_dim':7,'robot_type':'piper','arm_mode':'single','schema':'joint'})
    converted_sft=dict(sft,format='rlinf_full_weights')
    assert compatibility(converted_sft,backend='rlinf',task='tsa-basket',contract={'model_action_dim':7,'robot_type':'piper','arm_mode':'single','schema':'joint'}) is None
    assert compatibility(converted_sft,backend='rlinf',task='tsa-moka',contract={'model_action_dim':7,'robot_type':'piper','arm_mode':'single','schema':'joint'})


def test_model_scene_overrides_dataset_without_editing_weights(tmp_path):
    cfg={'workspace_root':str(tmp_path)}
    bind(cfg,'model','local_disk_6:/models/actor.pt','tsa_microwave')
    result=annotate(dict(node='local_4090',path='/models/actor.pt',task='unknown',dataset_ids=['tsa_basket_v1']),bindings(cfg))
    assert result['task_scene']=='tsa-microwave'
    assert not (tmp_path/'actor.pt').exists()
    assert dataset_scene('unrelated-dataset-123')==''


def test_unknown_scene_does_not_require_a_dataset():
    model=dict(backend='rlinf',complete=True,format='rlinf_full_weights',task='tsa_basket',action_contract={'robot_type':'piper','arm_mode':'single','schema':'joint'})
    assert compatibility(model,backend='rlinf',task='tsa-basket',contract=model['action_contract']) is None
    assert compatibility(dict(model,task='unknown'),backend='rlinf',contract=model['action_contract']) is None


def test_cross_dataset_eval_keeps_training_norm_stats(tmp_path):
    import pytest
    from server_4090.selected_resource import evaluation_norm_asset
    stats=tmp_path/'assets/training-v1/norm_stats.json';stats.parent.mkdir(parents=True);stats.write_text('{}')
    assert evaluation_norm_asset(tmp_path,'eval-v2')=='training-v1'
    other=tmp_path/'assets/another/norm_stats.json';other.parent.mkdir();other.write_text('{}')
    assert evaluation_norm_asset(tmp_path,'training-v1')=='training-v1'
    with pytest.raises(ValueError,match='unique'):evaluation_norm_asset(tmp_path,'eval-v2')


def test_remote_eval_snapshot_executes_reviewed_sources_without_overwriting(tmp_path):
    import json,subprocess,sys
    from server_4090.evaluation_snapshot import snapshot_command
    (tmp_path/'openpi_single_arm.py').write_text('value="reviewed"')
    (tmp_path/'selected_resource.py').write_text('norm="training-norm"')
    (tmp_path/'eval_heldout_loss.py').write_text('import json,sys,openpi_single_arm,selected_resource\nprint(json.dumps([openpi_single_arm.value,selected_resource.norm,sys.argv[1:]]))')
    result=subprocess.run(snapshot_command([sys.executable,'/nonexistent/remote/eval_heldout_loss.py','--checkpoint','chosen'],tmp_path),capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==['reviewed','training-norm',['--checkpoint','chosen']]


def test_scene_binding_endpoint_is_authenticated_and_metadata_only(tmp_path):
    from tests.test_dashboard_checkpoint_management import DashboardCheckpointManagementTest
    app,token,_,paths=DashboardCheckpointManagementTest()._make_app(tmp_path)
    client=app.test_client();body=dict(kind='model',key='local_4090:'+str(paths[0]),task_scene='tsa_basket')
    assert client.post('/api/task-scenes/bind',json=body).status_code in {401,403}
    response=client.post('/api/task-scenes/bind',json=body,headers={'Authorization':'Bearer '+token})
    assert response.status_code==200 and response.get_json()['task_scene']=='tsa-basket'
    assert all(p.exists() for p in paths)
    status=client.get('/api/status?compact=1',headers={'Authorization':'Bearer '+token}).get_json()
    selected=next(m for m in status['checkpoints'] if m['path']==str(paths[0]))
    assert selected['task_scene']=='tsa-basket'
