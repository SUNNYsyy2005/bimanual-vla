import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from server_4090.artifact_inventory import Inventory, uid, signature


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inv = Inventory(dict(workspace_root=str(self.root/'workspace'), experiment_artifact_root=str(self.root/'runs')))
        self.run = self.root/'runs/libero-safety/tsa-moka/exp/run1'
        self.ep = self.run/'eval/round-032/failed/ep1'; self.ep.mkdir(parents=True)
        self.manifest = dict(schema='experiment_run.v1',video_source='libero-safety',task='tsa-moka',experiment='exp',run_id='run1',backend='rlinf',state='completed')
        self.write(self.run/'run.json',self.manifest)
        self.write(self.ep/'episode.json',dict(schema='rollout_episode.v1',episode_id='ep1',success='failed',trajectory_reference=dict(host='h100',path='/remote/trajectory.npz')))
        (self.ep/'review.mp4').write_bytes(b'video'); (self.ep/'rgb.png').write_bytes(b'image')
        (self.run/'logs').mkdir(); (self.run/'logs/train.log').write_text('log')
        (self.run/'checkpoints/round32').mkdir(parents=True); (self.run/'checkpoints/round32/actor.pt').write_bytes(b'weights')
        self.write(self.run/'checkpoints/index.json',dict(schema='checkpoint_index.v1',checkpoints=[dict(id='round32',path='checkpoints/round32',complete=True)]))
        self.inv.reconcile()

    def write(self,p,v): p.write_text(json.dumps(v))

    def test_index_query_without_filesystem_scan_and_no_video_requirement(self):
        (self.ep/'review.mp4').unlink(); self.inv.reconcile()
        with mock.patch.object(Path,'glob',side_effect=AssertionError('no request-time scans')):
            groups=self.inv.groups(); page=self.inv.page([groups['groups'][0]['id']])
            self.assertEqual(page['total'],1)
            self.assertEqual(page['episodes'][0]['trajectory_reference']['host'],'h100')
            self.assertEqual(self.inv.page([])['episodes'],[])

    def test_delete_payload_preserves_model_logs_and_evidence(self):
        ident=uid(self.ep/'rgb.png'); before=self.inv.get(ident); revision=self.inv.revision()
        result=self.inv.delete_files([ident],lambda path: None)
        self.assertEqual(result['deleted_count'],1)
        self.assertTrue((self.ep/'episode.json').exists()); self.assertTrue((self.run/'logs/train.log').exists())
        self.assertTrue((self.run/'checkpoints/round32/actor.pt').exists())
        self.assertFalse(self.inv.upsert(before,revision))
        # Even a later scan containing old stat data cannot resurrect a tombstone.
        self.assertFalse(self.inv.upsert(before,self.inv.revision()))
        self.assertEqual(self.inv.get(ident)['status'],'deleted')

    def test_model_delete_isolated_and_cross_area_ids_rejected(self):
        model=self.inv.models()[0]
        with self.assertRaises(ValueError): self.inv.delete_files([model['id']],lambda p: None)
        with self.assertRaises(ValueError): self.inv.delete_files([uid(self.ep/'review.mp4')],lambda p: None,models=True)
        self.inv.delete_files([model['id']],lambda p: None,models=True)
        self.assertTrue((self.ep/'review.mp4').exists()); self.assertTrue((self.run/'logs/train.log').exists())
        self.inv.reconcile(); self.assertEqual(self.inv.models(),[])

    def test_changed_file_and_active_run_rejected(self):
        path=self.ep/'rgb.png'; path.write_bytes(b'changed after inventory')
        with self.assertRaises(ValueError): self.inv.delete_files([uid(path)],lambda p: None)
        self.manifest['state']='running'; self.write(self.run/'run.json',self.manifest)
        with self.assertRaises(ValueError): self.inv.delete_files([uid(self.ep/'review.mp4')],lambda p: None)

    def test_failed_unlink_persists_and_is_not_reported_as_deleted(self):
        ident=uid(self.ep/'rgb.png')
        with mock.patch.object(Path,'unlink',side_effect=PermissionError('fixture')):
            result=self.inv.delete_files([ident],lambda p: None)
        self.assertEqual(result['deleted_count'],0); self.assertEqual(self.inv.get(ident)['status'],'delete_failed')
        self.inv.reconcile(); self.assertEqual(self.inv.get(ident)['status'],'delete_failed')

    def test_same_path_recreation_after_delete_has_new_version(self):
        path=self.ep/'rgb.png'; ident=uid(path)
        self.inv.delete_files([ident],lambda p: None)
        path.write_bytes(b'new version'); self.inv.reconcile()
        self.assertEqual(self.inv.get(ident)['status'],'present')

    def test_symlink_cannot_delete_checkpoint_through_episode(self):
        target=self.ep/'escape.npz'
        try: target.symlink_to(self.run/'checkpoints/round32/actor.pt')
        except OSError: self.skipTest('symlinks unavailable')
        self.inv.reconcile()
        with self.assertRaises(ValueError): self.inv.get(uid(target))


if __name__=='__main__': unittest.main()
