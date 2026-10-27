from types import SimpleNamespace
from unittest import mock

import pytest
from tests.test_dashboard_checkpoint_management import DashboardCheckpointManagementTest
from server_4090.app import TaskManager


@pytest.mark.parametrize('cluster',[False,True])
def test_eval_uses_selected_checkpoint_without_foundation(tmp_path,monkeypatch,cluster):
    app,token,_,paths=DashboardCheckpointManagementTest()._make_app(tmp_path)
    view=app.view_functions['start_eval']
    cells=dict(zip(view.__code__.co_freevars,view.__closure__))
    contract=dict(contract_fingerprint={},raw_gripper_semantics='width',wire_gripper_semantics='width')
    replacements=dict(parse_dataset=lambda p:('fixture','single','right','fixture',{}),
        action_contract_for_model=lambda c:contract,
        training_episode_split=lambda *a,**k:(SimpleNamespace(seed=1),None),
        parse_gpus=lambda *a,**k:[0],dataset_root_for_id=lambda d:(tmp_path/'datasets/fixture',True),dataset_env_kwargs=lambda d:{})
    build=mock.Mock(return_value=['fixture-eval']);replacements['build_eval_command']=build
    if cluster:
        replacements.update(runtime_config_for_target=lambda target:dict(remote_job_dir='/remote/jobs',base_checkpoint='/missing/foundation'),
            translate_runtime_path=lambda path,target:'/remote/selected-checkpoint',slurm_runner_command=mock.Mock(return_value=['fixture-slurm']))
    # Keep real checkpoint validation; mock scheduling and unrelated dataset/GPU work.
    for name,value in replacements.items():
        if name in cells:monkeypatch.setattr(cells[name],'cell_contents',value)
        else:monkeypatch.setitem(view.__globals__,name,value)
    monkeypatch.setattr(cells['validate_weight_identity'],'cell_contents',lambda *a:None)
    with mock.patch.object(TaskManager,'start',return_value={'id':'eval-fixture'}) as start:
        response=app.test_client().post('/api/tasks/eval',headers={'Authorization':'Bearer '+token},json={
            'checkpoint':str(paths[0]),'model_variant':'pi05','execution_target':'h100' if cluster else 'local_4090',
            'base_checkpoint':'/obsolete/ignored/foundation'})
        assert response.status_code==201,response.get_json()
        args=build.call_args.kwargs
        assert str(args['base_checkpoint'])==str(args['checkpoint'])
        assert 'base_checkpoint' not in start.call_args.kwargs['metadata']
        if cluster:
            assert replacements['slurm_runner_command'].call_args.kwargs['target_config']['base_checkpoint']=='/remote/selected-checkpoint'
