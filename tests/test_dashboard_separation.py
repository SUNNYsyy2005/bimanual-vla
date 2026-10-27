import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from server_4090.storage_worker import history_cleanup, file_kind, inventory, atomic
from server_4090.model_catalog import compatibility, indexed_models
from server_4090.storage_catalog import Replicas
from server_4090.selected_resource import inspect_checkpoint
from tests.test_dashboard_checkpoint_management import DashboardCheckpointManagementTest


class SeparationTest(unittest.TestCase):
    def test_history_removes_diagnostics_preserves_rollout_features_and_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';root.mkdir()
            remove=['logs/train.log','metrics/loss.json','metrics/loss.png','training_curve.json','tensorboard/events.out.tfevents.fixture']
            keep=['checkpoints/actor.pt','train/round/episode/video.mp4','eval/round/episode/frames/000.png','train/round/episode/vla_latent.pt','metrics/jepa_features.npz','logs/frame001.png','protocol.json','metadata/config.json']
            for name in remove+keep:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
            result=history_cleanup(dict(run_paths=[str(root)],run={'state':'completed'}))
            self.assertEqual(len(result['deleted_paths']),len(remove))
            for name in remove:self.assertFalse((root/name).exists(),name)
            for name in keep:self.assertEqual((root/name).read_bytes(),b'fixture',name)

    def test_history_active_run_and_links_refused_before_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'task.log').write_text('keep')
            with self.assertRaises(ValueError):history_cleanup(dict(run_paths=[str(root)],run={'state':'running'}))
            (root/'alias.log').symlink_to(root/'task.log')
            with self.assertRaises(ValueError):history_cleanup(dict(run_paths=[str(root)],run={'state':'completed'}))
            self.assertTrue((root/'task.log').exists())

    def test_history_checks_active_task_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';run.mkdir();(run/'task.log').write_text('keep')
            atomic(root/'tasks/active/task.json',dict(id='active',state='running',metadata={'checkpoint_dir':str(run)}))
            with self.assertRaises(ValueError):history_cleanup(dict(run_paths=[str(run)],run={'state':'completed'},task_root=str(root/'tasks')))
            self.assertTrue((run/'task.log').exists())

    def test_feature_classification(self):
        for name in ('vla_latent.pt','jepa_embeddings.npy','features.npz','episode/latent.safetensors'):
            self.assertEqual(file_kind(name),'feature')
        self.assertEqual(file_kind('episode/video.mp4'),'video')
        self.assertEqual(file_kind('episode/trajectory.npz'),'trajectory')

    def test_selected_model_is_live_without_inventory_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'params').mkdir();(root/'_CHECKPOINT_METADATA').write_text('{}')
            (root/'params/data').write_bytes(b'weights')
            self.assertTrue(inspect_checkpoint(root)['available'])
            self.assertEqual(inspect_checkpoint(root)['size_bytes'],9)
            self.assertFalse(inspect_checkpoint(root,max_entries=1)['size_complete'])
            (root/'_CHECKPOINT_METADATA').unlink()
            self.assertFalse(inspect_checkpoint(root)['available'])
            self.assertFalse(inspect_checkpoint(root/'missing')['exists'])

    def test_cross_backend_requires_format_task_and_action_contract(self):
        model=dict(backend='openpi',complete=True,format='orbax',dataset_ids=['a'])
        self.assertIsNotNone(compatibility(model,backend='rlinf',dataset_ids=['a']))
        model.update(format='rlinf_full_weights',action_contract={'model_action_dim':7,'robot_type':'piper','arm_mode':'single','schema':'joint'})
        self.assertIsNone(compatibility(model,backend='rlinf',dataset_ids=['a'],contract={'model_action_dim':7,'robot_type':'piper','arm_mode':'single','schema':'joint'}))
        self.assertIsNone(compatibility(model,backend='rlinf',dataset_ids=['b'],contract={'model_action_dim':7,'robot_type':'piper','arm_mode':'single','schema':'joint'}))
        self.assertIsNotNone(compatibility(dict(model,task='tsa-basket'),backend='rlinf',task='tsa-moka',contract={'model_action_dim':7,'robot_type':'piper','arm_mode':'single','schema':'joint'}))
        self.assertIsNotNone(compatibility(model,backend='rlinf',dataset_ids=['a'],contract={'model_action_dim':14}))

    def test_backend_storage_filter_does_not_read_filesystem(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=dict(workspace_root=str(root/'work'),dataset_root=str(root/'datasets'))
            registry=Replicas(config);copies=[]
            for backend in ('openpi','rlinf'):
                copies.append(dict(id=backend,logical_id=backend,path='/fixture/'+backend,kind='model',node='local_4090',run={'backend':backend},files=[],size_bytes=1))
            registry.merge(dict(node='local_4090',updated_at=1,copies=copies),registry.inventory.revision())
            with mock.patch.object(Path,'glob',side_effect=AssertionError('GET must not scan')):
                rows=registry.page(area='training',backend='openpi')['items']
                self.assertEqual([r['id'] for r in rows],['openpi'])
                self.assertEqual(len(indexed_models(registry,'rlinf')),1)

    def test_history_api_preserves_nested_payload_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);app,token,_,_=DashboardCheckpointManagementTest()._make_app(root)
            task=root/'workspace/tasks/train-fixture';task.mkdir(parents=True)
            atomic(task/'task.json',dict(id='train-fixture',type='train',state='completed',metadata={}))
            (task/'task.log').write_text('metric')
            (task/'vla_latent.pt').write_bytes(b'latent');(task/'model.safetensors').write_bytes(b'weight')
            response=app.test_client().delete('/api/tasks/train-fixture',headers={'Authorization':'Bearer '+token})
            self.assertEqual(response.status_code,200,response.get_json())
            self.assertFalse((task/'task.log').exists());self.assertFalse((task/'task.json').exists())
            self.assertTrue((task/'vla_latent.pt').exists());self.assertTrue((task/'model.safetensors').exists())
            self.assertTrue((root/'workspace/model-provenance/train-fixture.json').exists())

    def test_checkpoint_inspect_api_sees_deletion_and_rejects_arbitrary_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);app,token,_,paths=DashboardCheckpointManagementTest()._make_app(root)
            client=app.test_client();headers={'Authorization':'Bearer '+token}
            self.assertEqual(client.post('/api/checkpoints/inspect',headers=headers,json={'path':'/etc/passwd'}).status_code,400)
            self.assertTrue(client.post('/api/checkpoints/inspect',headers=headers,json={'path':str(paths[0])}).get_json()['available'])
            (paths[0]/'_CHECKPOINT_METADATA').unlink();(paths[0]/'params/_METADATA').unlink()
            self.assertFalse(client.post('/api/checkpoints/inspect',headers=headers,json={'path':str(paths[0])}).get_json()['available'])

    def test_unindexed_checkpoint_is_discovered_without_refreshing_any_index(self):
        from server_4090.storage_worker import discover_models
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);step=root/'pi05_piper_bimanual_lora/new-experiment/12000'
            (step/'params').mkdir(parents=True);(step/'_CHECKPOINT_METADATA').write_text('{}')
            (step/'assets/same_dataset').mkdir(parents=True)
            with mock.patch.object(Path,'rglob',side_effect=AssertionError('No recursive model scan')):
                result=discover_models(dict(roots=[dict(path=str(root),layout='legacy')],backend='openpi'))
            self.assertEqual(len(result['models']),1)
            self.assertEqual(result['models'][0]['dataset_ids'],['same_dataset'])
            self.assertLess(result['entries_checked'],10)
            self.assertFalse(result['truncated'])

    def test_unindexed_canonical_checkpoint_is_discovered(self):
        from server_4090.storage_worker import discover_models
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'source/task/exp/run';step=run/'checkpoints/step-12000'
            (step/'params').mkdir(parents=True);(step/'_CHECKPOINT_METADATA').write_text('{}')
            atomic(run/'run.json',dict(schema='experiment_run.v1',backend='openpi',experiment='exp',origin={'config_name':'pi05_piper_bimanual_lora'}))
            result=discover_models(dict(roots=[dict(path=str(root),layout='canonical')],backend='openpi'))
            self.assertEqual(result['models'][0]['step'],12000)
            self.assertFalse((run/'checkpoints/index.json').exists())

    def test_new_rlinf_checkpoint_is_not_hidden_by_old_index(self):
        from server_4090.rlinf_dashboard import checkpoints_for_run
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);atomic(root/'checkpoints/index.json',dict(schema='checkpoint_index.v1',checkpoints=[]))
            weight=root/'checkpoints/global_step_7/actor/model_state_dict/full_weights.pt';weight.parent.mkdir(parents=True)
            with zipfile.ZipFile(weight,'w') as z:z.writestr('data.pkl',b'fixture')
            rows=checkpoints_for_run(dict(path=str(root)))
            self.assertEqual(rows[0]['step'],7)

    def test_live_curve_reads_new_points_without_snapshot_refresh(self):
        from server_4090.rlinf_dashboard import live_metrics_for_run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=dict(id='fixture',path=str(root),_source={},state='running')
            atomic(root/'training_curve.json',{'points':[{'round':1,'loss':2.0}]})
            first=live_metrics_for_run(run)
            atomic(root/'training_curve.json',{'points':[{'round':1,'loss':2.0},{'round':2,'loss':1.0}]})
            second=live_metrics_for_run(run)
            self.assertEqual(len(first['training_points']),1)
            self.assertEqual(len(second['training_points']),2)

    def test_entire_new_experiment_group_loads_without_registration_or_index_update(self):
        """Regression for the user's actual requirement: a wholly new experiment group."""
        import time, hashlib
        from server_4090.app import create_app
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,token,_,_=DashboardCheckpointManagementTest()._make_app(root)
            old=root/'runs/previous-group';old.mkdir(parents=True)
            cfg=json.loads((root/'config.json').read_text())
            cfg['rlinf_dashboard']={'sources':[{'id':'registered-old-group','target':'4x4090','root':str(old)}],'targets':{}}
            (root/'config.json').write_text(json.dumps(cfg))
            snapshot=root/'workspace/rlinf-runs-index.json'
            snapshot.write_text(json.dumps({'data':[],'updated_at':time.time()}))
            with mock.patch.dict(os.environ,{'BIMANUAL_VLA_SERVER_TOKEN':token}):app=create_app(root/'config.json')
            before=hashlib.sha256(snapshot.read_bytes()).hexdigest()
            # This group is not registered in config, nor in any inventory.
            fresh=root/'runs/brand-new-experiment-group/seed-43/ppo'
            fresh.mkdir(parents=True)
            atomic(fresh/'training_curve.json',{'points':[{'round':1,'loss':.75}]})
            client=app.test_client();headers={'Authorization':'Bearer '+token}
            response=client.post('/api/rlinf/runs/discover',headers=headers,json={'source':'live-all'})
            self.assertEqual(response.status_code,200,response.get_json())
            # Discovery is now asynchronous: HTTP returns immediately while the
            # bounded worker publishes new records without rebuilding inventory.
            deadline=time.monotonic()+5
            while not response.get_json()['ready'] and time.monotonic()<deadline:
                time.sleep(.03)
                response=client.post('/api/rlinf/runs/discover',headers=headers,json={'source':'live-all'})
            self.assertTrue(response.get_json()['ready'],response.get_json())
            runs=response.get_json()['runs'];new=next(r for r in runs if r.get('path')==str(fresh))
            with mock.patch.dict(os.environ,{'BIMANUAL_VLA_SERVER_TOKEN':token}):
                restarted=create_app(root/'config.json')
            restored=restarted.test_client().get('/api/rlinf/runs',headers=headers).get_json()['runs']
            self.assertIn(new['id'],[r['id'] for r in restored])
            metrics=client.get('/api/rlinf/runs/'+new['id']+'/metrics',headers=headers).get_json()
            self.assertEqual(metrics['training_points'][0]['loss'],.75)
            self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(),before)
            self.assertEqual(json.loads((root/'config.json').read_text())['rlinf_dashboard']['sources'][0]['root'],str(old))

    def test_canonical_rlinf_run_appears_before_first_metric_and_reads_initial_curve(self):
        import sys
        from server_4090.rlinf_dashboard import scan_runs, live_metrics_for_run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'libero/tsa-basket/new-group/new-run'
            atomic(run/'run.json',dict(backend='rlinf',kind='rl',state='running',algorithm='grpo',task='tsa-basket',experiment='new-group',run_id='new-run',origin={'job_id':'47619'}))
            atomic(run/'metrics/training_curve.json',[])
            atomic(run/'metadata/protocol.json',dict(algorithm='grpo',status='RUNNING',job_id='47619'))
            source=dict(id='canonical',root=str(root),python=sys.executable,target='h100',query_slurm=False,max_depth=6)
            rows=scan_runs({'rlinf_dashboard':{'sources':[source]}})
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['job_id'],'47619')
            self.assertEqual(rows[0]['path'],str(run))
            self.assertEqual(rows[0]['training_points'],0)
            atomic(run/'metrics/training_curve.json',{'points':[{'round':1,'loss':.5}]})
            self.assertEqual(live_metrics_for_run(rows[0])['training_points'][0]['loss'],.5)

    def test_discovery_uses_reader_python_preserves_mirrors_and_disabled_sources(self):
        from server_4090.training_discovery import experiment_parent_sources
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,_,_=DashboardCheckpointManagementTest()._make_app(root)
            cfg=json.loads((root/'config.json').read_text())
            cfg['rlinf_dashboard']={'sources':[
                dict(id='old',target='H100',ssh_host='login-server',python='/conda/bin/python',root='/data/runs/old'),
                dict(id='mirror',target='h200-ali-01',ssh_host='login-server',python='/conda/bin/python',root='/home/sunny/metric-mirror'),
                dict(id='disabled',target='h100',ssh_host='login-server',root='/nas/unavailable',enabled=False)],
                'targets':{'h100':dict(enabled=True,kind='slurm',submit_host='login-server',python='python',profiles={'a':{'run_root':'/data/runs/old'},'b':{'run_root':'/nas/unavailable'}})}}
            rows=experiment_parent_sources(cfg)
            self.assertTrue(all(Path(s['python']).is_absolute() for s in rows))
            self.assertEqual(next(s for s in rows if s['root']=='/data/runs')['python'],'/conda/bin/python')
            self.assertNotIn('storage_location',next(s for s in rows if s['root']=='/home/sunny/metric-mirror'))
            self.assertFalse(any(s['root']=='/nas/unavailable' for s in rows))

    def test_parent_discovery_skips_rollout_payload_and_preserves_partial_results(self):
        import sys
        import server_4090.rlinf_dashboard as dashboard
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            atomic(root/'new-group/training_curve.json',[{'round':1,'loss':.5}])
            for n in range(50):atomic(root/'new-group/data'/str(n)/'frame.json',{})
            with mock.patch.object(dashboard,'REMOTE_SCAN',dashboard.REMOTE_SCAN.replace('entries>50000','entries>30')):
                result=dashboard._scan_root({'python':sys.executable},str(root),6)
            self.assertIn('new-group/training_curve.json',result['curves'])
            self.assertIsNone(result['scan_warning'])
            # An actual breadth limit still retains earlier curves instead of raising.
            for n in range(50):atomic(root/'old-tree'/str(n)/'config.json',{})
            os.utime(root/'new-group',(2000000000,2000000000))
            with mock.patch.object(dashboard,'REMOTE_SCAN',dashboard.REMOTE_SCAN.replace('entries>50000','entries>30')):
                result=dashboard._scan_root({'python':sys.executable},str(root),6)
            self.assertIn('new-group/training_curve.json',result['curves'])
            self.assertIsNotNone(result['scan_warning'])

    def test_rlinf_default_checkpoint_is_not_a_foundation_model(self):
        from server_4090.model_catalog import rlinf_options, is_foundation_path
        cfg={'rlinf_dashboard':{'targets':{'4x4090':{'enabled':True,'profiles':{'train':{'command_template':['actor.model.model_path=/models/pi05_libero_safety_rlinf']}}}}}}
        with mock.patch('server_4090.model_catalog.indexed_models',return_value=[]):
            default=rlinf_options(cfg,None,'4x4090','train')[0]
            self.assertFalse(default['foundation'])
            self.assertEqual(default['category'],'checkpoint')
            cfg['rlinf_dashboard']['targets']['4x4090']['profiles']['train']['command_template']=['actor.model.model_path=/models/pi05_base']
            self.assertTrue(rlinf_options(cfg,None,'4x4090','train')[0]['foundation'])
            cfg['rlinf_dashboard']['targets']['4x4090']['profiles']['train']['command_template']=[]
            self.assertEqual(rlinf_options(cfg,None,'4x4090','train')[0]['category'],'profile_default')
        self.assertFalse(is_foundation_path('/models/base-models/experiment/10000'))
        self.assertTrue(is_foundation_path('/models/pi05_base'))

    def test_api_replaces_nonfinite_metrics_without_mutating_results(self):
        import math
        from flask import jsonify
        with tempfile.TemporaryDirectory() as temp:
            app,token,_,_=DashboardCheckpointManagementTest()._make_app(Path(temp))
            raw={'baseline':{'loss':float('nan'),'other':[float('inf'),float('-inf'),.25]}}
            @app.get('/api/test-nonfinite-response')
            def nonfinite_response():return jsonify(raw)
            response=app.test_client().get('/api/test-nonfinite-response',headers={'Authorization':'Bearer '+token})
            self.assertEqual(response.status_code,200)
            # parse_constant rejects invalid JSON accepted by Python's default parser.
            result=json.loads(response.data,parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            self.assertEqual(result,{'baseline':{'loss':None,'other':[None,None,.25]}})
            self.assertTrue(math.isnan(raw['baseline']['loss']))
