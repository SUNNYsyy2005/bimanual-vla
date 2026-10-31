from unittest.mock import Mock, patch
import json
import pytest
from server_4090.remote_checkpoints import RemoteCheckpointInventory, probe
from server_4090.cluster_read_guard import admit_read


def test_checkpoint_passive_reads_and_explicit_source():
    inventory=RemoteCheckpointInventory({n:{'checkpoint_base_dir':'/models'} for n in ['h100','h200']})
    with patch.object(inventory.pool,'submit') as submit:
        inventory.snapshot()
        inventory.snapshot(force=True)
        submit.assert_not_called()
        inventory.snapshot(force=True,source='h100')
        assert submit.call_count==1 and submit.call_args.args[1]=='h100'
        inventory.snapshot(force=True,source='h100')
        assert submit.call_count==1
    inventory.pool.shutdown()


def test_unavailable_and_duplicate_reads_rejected():
    target={'submit_host':'login','node':'h200-ali-01'}
    with patch('server_4090.cluster_read_guard.subprocess.run',return_value=Mock(returncode=0,stdout='draining')) as run:
        with pytest.raises(RuntimeError,match='未提交'):admit_read(target,'storage_inventory_')
        assert run.call_count==1
    with patch('server_4090.cluster_read_guard.subprocess.run',side_effect=[Mock(returncode=0,stdout='mixed'),Mock(returncode=0,stdout='storage_inventory_one|PENDING|h200-ali-01')]):
        with pytest.raises(RuntimeError,match='不重复'):admit_read(target,'storage_inventory_')


def test_checkpoint_probe_cancels_on_poll_failure():
    target=dict(submit_host='login',node='h200-ali-01',partition='h200',access_mode='slurm_only',checkpoint_base_dir='/models',openpi_python='/conda/python')
    with patch('server_4090.cluster_read_guard.admit_read'),patch('server_4090.remote_checkpoints.ssh',side_effect=['123',RuntimeError('connection lost'),'']) as ssh:
        with pytest.raises(RuntimeError,match='connection lost'):probe(target)
        assert ssh.call_args.args[1]=='scancel 123'
        assert '--job-name=checkpoint_inventory_' in ssh.call_args_list[0].kwargs['input']


def test_passive_api_never_contacts_cluster(tmp_path):
    from tests.test_dashboard_checkpoint_management import DashboardCheckpointManagementTest
    app,token,_,_=DashboardCheckpointManagementTest()._make_app(tmp_path)
    client=app.test_client();headers={'Authorization':'Bearer '+token}
    with patch('subprocess.run',side_effect=AssertionError('passive read must not SSH or submit')):
        for endpoint in ['/api/checkpoints/remote','/api/dataset-locations','/api/storage-items']:
            response=client.get(endpoint,headers=headers)
            assert response.status_code==200,(endpoint,response.get_json())


def test_scheduled_refresh_ignores_remote_dirty_markers(tmp_path, monkeypatch):
    from server_4090 import storage_catalog as sc
    cfg={'workspace_root':str(tmp_path/'workspace'),'experiment_artifact_root':str(tmp_path/'artifacts')}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    registry=sc.Replicas(cfg)
    with registry.inventory.connect() as db:
        db.execute("INSERT OR REPLACE INTO meta VALUES('storage_dirty_remote','1')")
    monkeypatch.setattr('sys.argv',['storage','--config',str(path),'--action','refresh','--scheduled'])
    with patch.object(sc,'locations',return_value={'local':{'local':True},'remote':{'local':False}}),patch.object(sc,'refresh_node',return_value={}) as refresh,patch.object(sc.Inventory,'reconcile'),patch('server_4090.artifact_index_repair.request_repair') as repair:
        sc.main()
        repair.assert_called_once_with(str(path),remote=False)
        assert refresh.call_count==1 and refresh.call_args.args[1]=='local'


def test_storage_refresh_api_scopes_to_selected_source(tmp_path):
    from tests.test_dashboard_checkpoint_management import DashboardCheckpointManagementTest
    from server_4090.app import TaskManager
    app,token,_,_=DashboardCheckpointManagementTest()._make_app(tmp_path)
    client=app.test_client();headers={'Authorization':'Bearer '+token}
    with patch.object(TaskManager,'list',return_value=[]),patch.object(TaskManager,'start',return_value={'state':'running'}) as start:
        assert client.post('/api/storage-items/refresh',json={},headers=headers).status_code==202
        assert start.call_args.args[1][-1]=='--local-only'
        assert client.post('/api/storage-items/refresh',json={'source':'local_4090'},headers=headers).status_code==202
        assert start.call_args.args[1][-2:]==['--node','local_4090']
        assert client.post('/api/storage-items/refresh',json={'source':'unknown'},headers=headers).status_code==400
        assert start.call_count==2
