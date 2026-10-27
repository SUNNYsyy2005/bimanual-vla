import json,time,threading
from unittest import mock
from tests.test_dashboard_checkpoint_management import DashboardCheckpointManagementTest
from server_4090.app import create_app


def test_discovery_singleflight_publishes_fast_source_before_slow(tmp_path):
    _,token,_,_=DashboardCheckpointManagementTest()._make_app(tmp_path)
    (tmp_path/'workspace/rlinf-runs-index.json').write_text(json.dumps(dict(data=[],updated_at=time.time())))
    sources=[dict(id=name,target='4x4090',root=str(tmp_path/name)) for name in ['fast','slow']]
    gate=threading.Event();calls=[]
    def scan(config,tasks):
        name=config['rlinf_dashboard']['sources'][0]['id'];calls.append(name)
        if name=='slow':assert gate.wait(5)
        return [dict(id=name,path=str(tmp_path/name),target='4x4090',state='running',name=name)]
    with mock.patch.dict('os.environ',{'BIMANUAL_VLA_SERVER_TOKEN':token}),mock.patch('server_4090.app.scan_runs',side_effect=scan),mock.patch('server_4090.training_discovery.experiment_parent_sources',return_value=sources):
        app=create_app(tmp_path/'config.json');client=app.test_client();headers={'Authorization':'Bearer '+token}
        try:
            start=time.monotonic()
            for _ in range(8):
                r=client.post('/api/rlinf/runs/discover',headers=headers,json={'source':'live-all'})
                assert r.status_code==200 and r.json['refreshing']
            assert time.monotonic()-start<1
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:
                rows=client.get('/api/rlinf/runs',headers=headers).json['runs']
                if any(r['id']=='fast' for r in rows):break
                time.sleep(.01)
            assert any(r['id']=='fast' for r in rows)
            assert not any(r['id']=='slow' for r in rows)
        finally:gate.set()
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            r=client.post('/api/rlinf/runs/discover',headers=headers,json={'source':'live-all'})
            if r.json['ready']:break
            time.sleep(.01)
        assert r.json['ready'] and sorted(calls)==['fast','slow']
        saved=json.loads((tmp_path/'workspace/rlinf-discovered-runs.json').read_text())
        assert set(saved)=={'fast','slow'}
