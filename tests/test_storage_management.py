import json
import base64
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from server_4090 import storage_worker as w
from server_4090.storage_catalog import Replicas
from server_4090.storage_catalog import storage_worker
from server_4090.slurm_job_runner import build_script


class StorageManagementTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.destination=self.root/'destination'
        self.run=self.source/'libero-safety/tsa-moka/experiment/run1'
        self.episode=self.run/'eval/round-032/failed/episode1';self.episode.mkdir(parents=True)
        w.atomic(self.run/'run.json',dict(schema='experiment_run.v1',video_source='libero-safety',task='tsa-moka',experiment='experiment',run_id='run1',state='completed'))
        w.atomic(self.episode/'episode.json',dict(schema='rollout_episode.v1',artifacts=dict(trajectory='trajectory.npz',video='review.mp4')))
        (self.episode/'trajectory.npz').write_bytes(b'trajectory');(self.episode/'rgb.png').write_bytes(b'image');(self.episode/'review.mp4').write_bytes(b'video')
        self.config=dict(workspace_root=str(self.root/'workspace'),experiment_artifact_root=str(self.source),dataset_root=str(self.root/'datasets'))
        self.registry=Replicas(self.config)

    def inventory(self):return w.inventory(dict(node='local_4090',artifact_roots=[str(self.source)],dataset_roots=[]))

    def test_selected_trajectory_and_images_need_no_video(self):
        copy=self.inventory()['copies'][0];bundle=self.root/'bundle'
        w.export_copy(dict(copy=copy,bundle=str(bundle),kinds=['trajectory','image']))
        self.assertFalse((bundle/'payload/review.mp4').exists())
        w.import_copy(dict(bundle=str(bundle),destination_root=str(self.destination),node='target'))
        target=self.destination/copy['relative_path']
        self.assertTrue((target/'trajectory.npz').exists());self.assertTrue((target/'rgb.png').exists());self.assertFalse((target/'review.mp4').exists())
        observed=w.inventory(dict(node='target',artifact_roots=[str(self.destination)],dataset_roots=[]))['copies'][0]
        self.assertNotIn('coverage',observed)
        self.assertTrue(all(f['file_state']=='verified' for f in observed['files']))
        # A later same-size edit invalidates file-level verification.
        (target/'rgb.png').write_bytes(b'other')
        observed=w.inventory(dict(node='target',artifact_roots=[str(self.destination)],dataset_roots=[]))['copies'][0]
        image=next(f for f in observed['files'] if f['name']=='rgb.png')
        self.assertEqual(image['file_state'],'present_unverified')

    def test_missing_files_are_filled_but_conflicts_not_overwritten(self):
        copy=self.inventory()['copies'][0];bundle=self.root/'bundle'
        w.export_copy(dict(copy=copy,bundle=str(bundle)))
        target=self.destination/copy['relative_path'];target.mkdir(parents=True)
        (target/'rgb.png').write_bytes(b'image')
        request=dict(bundle=str(bundle),destination_root=str(self.destination),node='target')
        w.import_copy(request)
        self.assertTrue((target/'trajectory.npz').exists())
        (target/'rgb.png').write_bytes(b'conflict')
        with self.assertRaises(ValueError):w.import_copy(request)
        self.assertEqual((target/'rgb.png').read_bytes(),b'conflict')

    def test_selective_delete_keeps_models_logs_and_other_files(self):
        (self.run/'checkpoints').mkdir();(self.run/'checkpoints/actor.pt').write_bytes(b'weights')
        (self.run/'logs').mkdir();(self.run/'logs/train.log').write_text('log')
        copy=next(c for c in self.inventory()['copies'] if c['kind']=='rollout')
        w.delete_copy(dict(copy=copy,kinds=['video']))
        self.assertTrue((self.episode/'trajectory.npz').exists());self.assertTrue((self.episode/'episode.json').exists())
        self.assertTrue((self.run/'checkpoints/actor.pt').exists());self.assertTrue((self.run/'logs/train.log').exists())

    def test_index_reads_no_disk_scan_and_old_scan_cannot_undo_delete(self):
        snapshot=self.inventory();version=self.registry.inventory.revision();self.registry.merge(snapshot,version)
        with mock.patch.object(Path,'glob',side_effect=AssertionError('no filesystem scans')):
            page=self.registry.page(area='rollout');self.assertEqual(page['total'],1)
            self.assertNotIn('files',page['items'][0])
        before=self.registry.inventory.revision();op,copy=self.registry.claim(snapshot['copies'][0]['id'],'delete')
        self.registry.finish(op);self.registry.merge(snapshot,before)
        self.assertEqual(self.registry.get(copy['id'])['status'],'deleted')
        self.registry.merge(snapshot,self.registry.inventory.revision())
        self.assertEqual(self.registry.get(copy['id'])['status'],'deleted')

    def test_active_or_changed_source_refuses_delete(self):
        copy=self.inventory()['copies'][0]
        (self.episode/'trajectory.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError):w.delete_copy(dict(copy=copy))
        copy=self.inventory()['copies'][0]
        manifest=json.loads((self.run/'run.json').read_text());manifest['state']='running';w.atomic(self.run/'run.json',manifest)
        with self.assertRaises(ValueError):w.delete_copy(dict(copy=copy))

    def test_temporary_files_are_not_reported_complete(self):
        (self.episode/'capture.mp4.partial').write_bytes(b'partial')
        copy=self.inventory()['copies'][0]
        partial=next(f for f in copy['files'] if f['name']=='capture.mp4.partial')
        self.assertEqual(partial['file_state'],'incomplete')

    def test_storage_job_has_no_model_dependency_and_uses_cpu_slurm(self):
        loc=dict(local=False,name='h200',submit_host='login-server',partition='h200',node='h200-ali-01',
                 openpi_python='/conda/bin/python',base_checkpoint='/missing/model',dataset_root='/data/datasets')
        with mock.patch('server_4090.storage_catalog.subprocess.run') as run:
            run.return_value.returncode=0
            storage_worker(loc,dict(action='preflight',destination_root='/data/datasets',additional_bytes=1))
        command=run.call_args.args[0]
        target=json.loads(base64.urlsafe_b64decode(command[command.index('--target-json')+1]))
        self.assertNotIn('base_checkpoint',target);self.assertEqual(target['gpu_count'],0);self.assertEqual(target['ntasks'],1)
        script=build_script(target,'test_storage',[['/conda/bin/python','worker.py']],['inventory'])
        self.assertIn('srun --ntasks=1 ',script);self.assertIn('#SBATCH --ntasks=1',script);self.assertNotIn('--gres=',script)

    def test_same_size_source_edit_refuses_delete(self):
        copy=self.inventory()['copies'][0]
        (self.episode/'rgb.png').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError,'changed'):w.delete_copy(dict(copy=copy))
        self.assertTrue((self.episode/'review.mp4').exists())

    def test_empty_selection_does_not_delete_whole_episode(self):
        copy=self.inventory()['copies'][0]
        with self.assertRaises(ValueError):w.delete_copy(dict(copy=copy,kinds=[]))
        self.assertTrue((self.episode/'review.mp4').exists())

    def test_missing_quota_root_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'usage root'):
            w.capacity(dict(usage_roots=[str(self.root/'missing')]),1)

    def test_existing_destination_symlink_cannot_redirect_copy(self):
        (self.episode/'frames').mkdir();(self.episode/'frames/one.png').write_bytes(b'image')
        copy=self.inventory()['copies'][0];bundle=self.root/'bundle'
        w.export_copy(dict(copy=copy,bundle=str(bundle)))
        target=self.destination/copy['relative_path'];target.mkdir(parents=True)
        outside=self.root/'outside';outside.mkdir();(target/'frames').symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'links'):w.import_copy(dict(bundle=str(bundle),destination_root=str(self.destination),node='target'))
        self.assertFalse((outside/'one.png').exists())

    def test_mutation_lock_prevents_concurrent_delete(self):
        copy=self.inventory()['copies'][0];request=dict(action='delete',copy=copy)
        with w.mutation_lock(request):
            with self.assertRaisesRegex(ValueError,'Another operation'):w.main(request)
        self.assertTrue((self.episode/'review.mp4').exists())

    def test_new_external_episode_and_quick_scan_keep_dataset_and_usage(self):
        snap=self.inventory();snap['owned_usage']=[dict(path='/data',bytes=100)]
        self.registry.merge(snap,self.registry.inventory.revision())
        ep=self.run/'eval/round-032/success/new';ep.mkdir(parents=True)
        w.atomic(ep/'episode.json',dict(schema='rollout_episode.v1'));(ep/'rgb.png').write_bytes(b'new')
        snap=self.inventory();snap['partial_scan']=True
        self.registry.merge(snap,self.registry.inventory.revision())
        page=self.registry.page(area='rollout');self.assertEqual(page['total'],2)
        self.assertEqual(page['resources']['local_4090']['owned_usage'][0]['bytes'],100)

    def test_model_deletion_keeps_rollout_and_logs(self):
        model=self.run/'checkpoints/step-1';model.mkdir(parents=True);(model/'actor.pt').write_bytes(b'weights')
        w.atomic(self.run/'checkpoints/index.json',dict(schema='checkpoint_index.v1',checkpoints=[dict(id='step-1',path='checkpoints/step-1',complete=True)]))
        (self.run/'logs').mkdir();(self.run/'logs/train.log').write_text('log')
        copy=next(c for c in self.inventory()['copies'] if c['kind']=='model')
        w.delete_copy(dict(copy=copy))
        self.assertFalse(model.exists());self.assertTrue((self.episode/'trajectory.npz').exists());self.assertTrue((self.run/'logs/train.log').exists())

    def test_active_local_task_is_checked_at_execution(self):
        tasks=self.root/'tasks';tasks.mkdir()
        w.atomic(tasks/'active/task.json',dict(id='active',state='running',metadata=dict(checkpoint=str(self.run))))
        with mock.patch.dict('os.environ',BIMANUAL_STORAGE_TASK_ROOT=str(tasks)):
            with self.assertRaisesRegex(ValueError,'active Dashboard'):w.delete_copy(dict(copy=self.inventory()['copies'][0]))

    def test_dataset_storage_destination_is_also_discoverable_after_transfer(self):
        from server_4090.storage_catalog import locations, refresh_node, transfer
        self.config['local_storage_locations']={'local_4090_hdd':dict(dataset_root=str(self.root/'disk/datasets'))}
        copy=self.inventory()['copies'][0]
        transfer(self.config,copy,'local_4090_hdd',['trajectory','image'])
        refresh_node(self.config,'local_4090_hdd',quick=True)
        rows=self.registry.page(node='local_4090_hdd',area='rollout')['items']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['file_types']['video'],0)
        self.assertIn(Path(locations(self.config)['local_4090_hdd']['artifact_root']),self.registry.inventory.roots())

    def test_compatibility_delete_stale_row_disappears_on_refresh(self):
        import shutil
        snapshot=self.inventory();self.registry.merge(snapshot,self.registry.inventory.revision())
        self.registry.invalidate_local_paths([self.episode/'review.mp4'])
        shutil.rmtree(self.episode)
        self.registry.merge(self.inventory(),self.registry.inventory.revision())
        self.assertEqual(self.registry.page(area='rollout')['total'],0)

    def test_remote_mutation_does_not_put_large_file_listing_in_exec_argument(self):
        loc=dict(local=False,name='h100',submit_host='login-server',partition='h100',node='h100-ksy-01',openpi_python='/conda/bin/python')
        copy=self.inventory()['copies'][0];copy['files']=[dict(name='frame.png',padding='x'*1024)]*1000
        with mock.patch('server_4090.storage_catalog.subprocess.run') as run:
            run.return_value.returncode=0
            storage_worker(loc,dict(action='delete',copy=copy))
        argv=run.call_args.args[0]
        commands=json.loads(base64.urlsafe_b64decode(argv[argv.index('--commands-json')+1]))
        request=json.loads(commands[0][-1]);self.assertNotIn('files',request['copy']);self.assertEqual(request['copy']['fingerprint'],copy['fingerprint'])


if __name__=='__main__':unittest.main()
