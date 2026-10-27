import json
from unittest.mock import patch
import pytest
from server_4090.model_identity import enrich, identity, mismatch, rlinf_run_contract
from server_4090.model_catalog import compatibility

CONTRACT=dict(robot_type='piper',arm_mode='bimanual',schema='joint',model_action_dim=14)

@pytest.mark.parametrize('field,value,label', [('robot_type','franka-panda','机械臂型号'),('arm_mode','single','单双臂'),('schema','delivery','动作输出类型'),('model_action_dim',16,'动作维度')])
def test_same_scene_is_not_enough(field,value,label):
    model=dict(complete=True,format='orbax',backend='openpi',task='same',action_contract={**CONTRACT,field:value})
    assert label in compatibility(model,backend='openpi',task='same',contract=CONTRACT)


def test_same_contract_different_dataset_and_backend_is_allowed():
    model=dict(complete=True,format='orbax',backend='rlinf',task='same',dataset_ids=['old'],action_contract=CONTRACT)
    assert compatibility(model,backend='openpi',task='same',dataset_ids=['new'],contract=CONTRACT) is None
    assert '待核实' in mismatch({},CONTRACT)
    assert '待核实' in mismatch({'action_contract':CONTRACT},{})


def test_exact_dataset_provenance_and_conflicts(tmp_path):
    datasets=[dict(id='a',**CONTRACT),dict(id='b',**{**CONTRACT,'robot_type':'franka-panda'})]
    model=enrich([dict(dataset_ids=['a'])],{},datasets)[0]
    assert mismatch(model,CONTRACT) is None
    conflict=enrich([dict(dataset_ids=['a','b'])],{},datasets)[0]
    assert '冲突' in mismatch(conflict,CONTRACT)
    assert identity({'robot_type':'aloha_agilex','arm_mode':'bimanual','schema':'joint'})['robot_type']=='aloha-agilex'


def test_rlinf_does_not_infer_contract_from_fallback_profile():
    cfg={'rlinf_dashboard':{'targets':{'h100':{'profiles':{'known':{'action_contract':CONTRACT}}}}}}
    assert rlinf_run_contract(cfg,{'target':'h100'})=={}
    assert rlinf_run_contract(cfg,{'target':'h100','launch_profile':'known'})==CONTRACT


@pytest.mark.parametrize('route', ['/api/tasks/eval','/api/tasks/train'])
def test_api_rejects_wrong_robot_before_scheduling(tmp_path,monkeypatch,route):
    from tests.test_dashboard_checkpoint_management import DashboardCheckpointManagementTest
    from server_4090.app import TaskManager
    app,token,_,paths=DashboardCheckpointManagementTest()._make_app(tmp_path)
    name='start_eval' if route.endswith('eval') else 'start_train'
    view=app.view_functions[name];cells=dict(zip(view.__code__.co_freevars,view.__closure__))
    target={**CONTRACT,'robot_type':'franka-panda','contract_fingerprint':{}}
    monkeypatch.setattr(cells['parse_dataset'],'cell_contents',lambda p:('real_ds','bimanual','both','joint',target))
    validation=cells['validate_weight_identity'].cell_contents
    vc=dict(zip(validation.__code__.co_freevars,validation.__closure__))
    monkeypatch.setitem(validation.__globals__,'checkpoint_action_contract',lambda p:CONTRACT)
    monkeypatch.setitem(view.__globals__,'action_contract_for_model',lambda c:c)
    with patch.object(TaskManager,'start') as start:
        response=app.test_client().post(route,headers={'Authorization':'Bearer '+token},json={
            'checkpoint':str(paths[0]),'base_checkpoint':str(paths[0]),'model_variant':'pi05','exp_name':'new_exp','mode':'new'})
        assert response.status_code==400 and '机械臂型号' in response.get_json()['error']
        start.assert_not_called()


def test_robot_generation_names_are_not_collapsed():
    assert identity({'robot_type':'Franka_FR3'})['robot_type']=='franka-fr3'
    assert identity({'robot_type':'ARX-X5'})['robot_type']=='arx-x5'
