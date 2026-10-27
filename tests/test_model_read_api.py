import json
from unittest import mock
from tests import test_dashboard_checkpoint_management as fixtures
from server_4090.storage_catalog import Replicas


def test_catalog_reads_index_with_filter_without_discovery(tmp_path):
    app,token,_,_=fixtures.DashboardCheckpointManagementTest()._make_app(tmp_path)
    registry=Replicas(json.loads((tmp_path/'config.json').read_text()))
    row=dict(id='fixture-model',kind='model',node='local_4090',path=str(tmp_path/'model'),root=str(tmp_path),run_relative='run',run=dict(backend='rlinf',experiment='test'),size_bytes=42)
    with registry.inventory.connect() as db:
        db.execute('INSERT INTO replicas VALUES(?,?,?,?,?,?,?)',('fixture-model','logical','local_4090',row['path'],'present',1,json.dumps(row)))
    client=app.test_client();headers={'Authorization':'Bearer '+token}
    with mock.patch('server_4090.storage_worker.discover_models',side_effect=AssertionError('unexpected scan')):
        response=client.get('/api/model-catalog?backend=rlinf',headers=headers)
        assert response.status_code==200 and response.json['models'][0]['size_bytes']==42
        assert client.get('/api/model-catalog?backend=openpi',headers=headers).json['models']==[]
        assert app.test_client().get('/api/model-catalog').status_code==401


def test_local_discovery_and_inspection_routes(tmp_path):
    app,token,_,paths=fixtures.DashboardCheckpointManagementTest()._make_app(tmp_path)
    client=app.test_client();headers={'Authorization':'Bearer '+token}
    with mock.patch('server_4090.storage_catalog.storage_worker',side_effect=AssertionError('unexpected remote scan')):
        response=client.get('/api/checkpoints/discover?backend=openpi&node=local_4090',headers=headers)
        assert response.status_code==200
        assert {str(p) for p in paths}<={m['path'] for m in response.json['models']}
        assert client.post('/api/checkpoints/inspect',headers=headers,json={'path':str(paths[0])}).json['available']
        assert client.post('/api/checkpoints/inspect',headers=headers,json={'path':'/etc'}).status_code==400
        assert client.get('/api/checkpoints/discover?node=missing',headers=headers).status_code==400
