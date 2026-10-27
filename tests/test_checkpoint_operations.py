import json
from pathlib import Path
from unittest import mock

import pytest
from server_4090 import storage_worker as w
from server_4090.checkpoint_operations import operate, source_location


@pytest.fixture
def fixture(tmp_path):
    root=tmp_path/'source';run=root/'simulation/task/experiment/run'
    (run/'checkpoints').mkdir(parents=True)
    w.atomic(run/'run.json',dict(schema='experiment_run.v1',backend='rlinf',state='completed',video_source='simulation',task='task',experiment='experiment',run_id='run'))
    for name in ('logs/train.log','metrics/curve.json','eval/rollout.mp4'):
        p=run/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b'keep')
    cp=run/'checkpoints/actor.pt';cp.write_bytes(b'fixture-weights')
    config=dict(workspace_root=str(tmp_path/'workspace'),dataset_root=str(tmp_path/'datasets'),experiment_artifact_root=str(root),
                local_storage_locations={'archive':dict(dataset_root=str(tmp_path/'target/datasets'),artifact_root=str(tmp_path/'target/artifacts'))})
    return config,run,cp


def test_selected_unindexed_rlinf_copy_delete_and_preservation(fixture):
    config,run,cp=fixture
    with mock.patch('server_4090.storage_catalog.refresh_node',side_effect=AssertionError('must not scan all experiments')):
        result=operate(config,'local_4090',str(cp),'rlinf','copy','archive')
        assert result['verification']=='sha256_selected_unit'
        assert Path(result['path']).read_bytes()==cp.read_bytes()
        operate(config,'local_4090',str(cp),'rlinf','delete')
    assert not cp.exists()
    assert Path(result['path']).is_file()
    assert all((run/p).read_bytes()==b'keep' for p in ('logs/train.log','metrics/curve.json','eval/rollout.mp4'))


def test_legacy_sft_copy_delete(fixture,tmp_path):
    config,_,_=fixture
    legacy=tmp_path/'legacy';cp=legacy/'pi05_piper_single_arm_lora/exp/100'
    (cp/'params').mkdir(parents=True);(cp/'params/weights').write_bytes(b'weights')
    (cp/'_CHECKPOINT_METADATA').write_text('{}')
    (cp.parent/'train.log').write_text('keep')
    config['checkpoint_base_dir']=str(legacy)
    result=operate(config,'local_4090',str(cp),'openpi','copy','archive')
    assert (Path(result['path'])/'params/weights').read_bytes()==b'weights'
    operate(config,'local_4090',str(cp),'openpi','delete')
    assert not cp.exists() and (cp.parent/'train.log').read_text()=='keep'


def test_reject_scope_links_running_and_changed(fixture,tmp_path):
    config,run,cp=fixture
    request=dict(path=str(cp),backend='rlinf',node='local_4090',roots=[config['experiment_artifact_root']])
    with pytest.raises(ValueError):source_location(config,'local_4090','/etc/passwd')
    with pytest.raises(ValueError):w.prepare_checkpoint(dict(request,path=str(run)))
    with pytest.raises(ValueError):w.prepare_checkpoint(dict(request,path=str(run/'logs/train.log')))
    link=cp.with_name('linked.pt');link.symlink_to(cp)
    with pytest.raises(ValueError):w.prepare_checkpoint(dict(request,path=str(link)))
    copy=w.prepare_checkpoint(request);cp.write_bytes(b'changed')
    with pytest.raises(ValueError):w.delete_copy(dict(copy=copy))
    run_data=json.loads((run/'run.json').read_text());run_data['state']='running';w.atomic(run/'run.json',run_data)
    with pytest.raises(ValueError):w.prepare_checkpoint(request)
    assert cp.exists()


def test_conflicting_target_is_not_overwritten(fixture):
    config,_,cp=fixture
    result=operate(config,'local_4090',str(cp),'rlinf','copy','archive')
    Path(result['path']).write_bytes(b'conflict')
    with pytest.raises(ValueError,match='refusing overwrite'):operate(config,'local_4090',str(cp),'rlinf','copy','archive')
    assert Path(result['path']).read_bytes()==b'conflict' and cp.exists()


def test_api_queues_exact_scope_and_checks_confirmation(tmp_path):
    from tests.test_dashboard_checkpoint_management import DashboardCheckpointManagementTest
    from server_4090.app import TaskManager
    app,token,_,paths=DashboardCheckpointManagementTest()._make_app(tmp_path)
    client=app.test_client();headers={'Authorization':'Bearer '+token}
    payload=dict(node='local_4090',path=str(paths[0]),backend='openpi',action='delete',confirm_path=str(paths[0]))
    assert client.post('/api/checkpoints/operate',json=payload).status_code in {401,403}
    with mock.patch.object(TaskManager,'start',return_value={'id':'fixture-task'}) as start:
        response=client.post('/api/checkpoints/operate',headers=headers,json=payload)
        assert response.status_code==202,response.get_json()
        command=start.call_args.args[1]
        assert command[command.index('--path')+1]==str(paths[0])
        assert start.call_args.kwargs['metadata']['transfer_kind']=='artifact_checkpoint_delete'
        start.reset_mock()
        for changes in ({'confirm_path':'wrong'},{'path':'/etc/passwd'},{'backend':'bad'},{'action':'copy','target':'unknown'}):
            assert client.post('/api/checkpoints/operate',headers=headers,json=dict(payload,**changes)).status_code==400
        start.assert_not_called()
    assert all(p.exists() for p in paths)


def test_unindexed_copy_becomes_discoverable_after_reindex(fixture):
    config,_,cp=fixture
    result=operate(config,'local_4090',str(cp),'rlinf','copy','archive')
    root=config['local_storage_locations']['archive']['artifact_root']
    indexed=w.inventory(dict(node='archive',artifact_roots=[root],dataset_roots=[]))
    models=[c for c in indexed['copies'] if c['kind']=='model']
    assert [c['path'] for c in models]==[result['path']]
    assert models[0]['checkpoint_index']['checkpoints'][0]['format']=='rlinf_full_weights'
