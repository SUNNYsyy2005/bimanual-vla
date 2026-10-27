import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_dashboard_eval_videos import DashboardEvalVideoManagementTest
from server_4090.artifact_inventory import Inventory
from server_4090.storage_catalog import Replicas
from server_4090.storage_worker import inventory, atomic


class DashboardStorageTest(unittest.TestCase):
    def test_authenticated_lazy_pages_and_cross_area_delete_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            app,token,_=DashboardEvalVideoManagementTest()._make_app(root)
            cfg=json.loads((root/'config.json').read_text());cfg['experiment_artifact_root']=str(root/'workspace/experiment_artifacts')
            run=Path(cfg['experiment_artifact_root'])/'libero-safety/tsa-moka/exp/run'
            ep=run/'eval/round-032/failed/episode1';ep.mkdir(parents=True)
            atomic(run/'run.json',dict(schema='experiment_run.v1',video_source='libero-safety',task='tsa-moka',experiment='exp',run_id='run',state='completed'))
            atomic(ep/'episode.json',dict(schema='rollout_episode.v1',episode_id='episode1'))
            (ep/'trajectory.npz').write_bytes(b'fixture');(ep/'review.mp4').write_bytes(b'fixture')
            inv=Inventory(cfg);inv.reconcile();registry=Replicas(cfg)
            snapshot=inventory(dict(node='local_4090',artifact_roots=[cfg['experiment_artifact_root']],dataset_roots=[]))
            registry.merge(snapshot,inv.revision())
            client=app.test_client();headers={'Authorization':'Bearer '+token}
            self.assertIn(client.get('/api/storage-items').status_code,{401,403})
            with mock.patch.object(Path,'glob',side_effect=AssertionError('query must not traverse directories')):
                response=client.get('/api/storage-items?area=rollout&limit=1',headers=headers)
                self.assertEqual(response.status_code,200)
                item=response.get_json()['items'][0]
                self.assertNotIn('files',item)
                files=client.get('/api/storage-items/'+item['id']+'/files?limit=1',headers=headers).get_json()
                self.assertEqual(len(files['files']),1)
                self.assertEqual(client.get('/api/rollout-artifacts',headers=headers).get_json()['episodes'],[])
            response=client.post('/api/storage-items/operate',headers=headers,json=dict(copy_id=item['id'],action='delete',area='training',confirm_path=item['path']))
            self.assertEqual(response.status_code,400)
            self.assertTrue((ep/'trajectory.npz').exists())


if __name__=='__main__':unittest.main()
