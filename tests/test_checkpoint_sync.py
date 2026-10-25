import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'server_4090'))
import checkpoint_export as export
import checkpoint_sync_runner as sync
import remote_checkpoints as inventory

class CheckpointSyncTests(unittest.TestCase):
    def test_inventory_payload_exceeding_slurm_comment_limit(self):
        import base64,os,random,string,types,zlib
        rng=random.Random(42)
        payload={'checkpoints':[{'path':''.join(rng.choices(string.ascii_letters,k=500))} for _ in range(10)]}
        chunks=[];comments=[]
        def run(args,**kwargs):
            if args[1]=='update':
                comment=args[-1].split('=',1)[1];comments.append(comment)
                prefix,index,total,chunk=comment.split(':',3);chunks.append(chunk)
                return types.SimpleNamespace(returncode=0)
            return types.SimpleNamespace(stdout='JobId=123 Comment=checkpoint_ack:'+str(len(chunks)-1)+'\n')
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch('subprocess.run',side_effect=run):inventory.publish_inventory(payload)
        self.assertGreater(len(chunks),1)
        self.assertLess(max(map(len,comments)),1024)
        self.assertEqual(json.loads(zlib.decompress(base64.urlsafe_b64decode(''.join(chunks)))),payload)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.source=self.root/'source'/'20000'
        marker=self.source.parent.parent/'.policy_action_conventions'/(self.source.parent.name+'.json')
        marker.parent.mkdir(parents=True);marker.write_text('{"dataset_id":"dataset"}')
        for name,data in {'_CHECKPOINT_METADATA':b'{}','params/_METADATA':b'{}',
                          'params/data':b'weights'*2000,'assets/dataset/norm_stats.json':b'{}',
                          'train_state/optimizer':b'optimizer'*2000}.items():
            f=self.source/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
    def tearDown(self):self.tmp.cleanup()
    def test_pinned_weights_survive_checkpoint_garbage_collection(self):
        snap=self.root/'snapshot';snap.mkdir()
        export.pin_checkpoint(self.source,snap)
        shutil.rmtree(self.source)
        self.assertEqual((snap/'params/data').read_bytes(),b'weights'*2000)
        self.assertFalse((snap/'train_state').exists())
    def test_bundle_roundtrip_and_incomplete_destination(self):
        bundle=self.root/'nas'/'bundle'
        manifest=export.export_checkpoint(self.source,bundle,2)
        self.assertFalse(any(r['path'].startswith('train_state') for r in manifest['files']))
        dest=self.root/'local'/'20000';dest.mkdir(parents=True);(dest/'partial').write_bytes(b'partial')
        sync.install_bundle(bundle,dest)
        self.assertEqual((dest/'params/data').read_bytes(),b'weights'*2000)
        self.assertFalse((dest/'train_state').exists())
        sync.validate_install(dest)
        self.assertEqual(len(list(dest.parent.glob('.20000.incomplete-*'))),1)
    def test_corrupt_bundle_cannot_publish_destination(self):
        bundle=self.root/'bundle';export.export_checkpoint(self.source,bundle,1)
        part=next(bundle.glob('part-*'));data=bytearray(part.read_bytes());data[100]^=1;part.write_bytes(data)
        dest=self.root/'local'/'20000'
        with self.assertRaisesRegex(ValueError,'corrupt'):sync.install_bundle(bundle,dest)
        self.assertFalse(dest.exists())
    def test_source_hardlinks_are_exported_as_regular_files(self):
        import os
        os.link(self.source/'params/data',self.source/'params/alias')
        bundle=self.root/'bundle';export.export_checkpoint(self.source,bundle,1)
        dest=self.root/'local'/'20000';sync.install_bundle(bundle,dest)
        self.assertEqual((dest/'params/alias').read_bytes(),(dest/'params/data').read_bytes())
    def test_split_writer_preserves_exact_stream(self):
        writer=export.PartWriter(self.root,13)
        writer.write(b'abc'*100);writer.close_part()
        with sync.PartsReader([self.root/p['name'] for p in writer.parts]) as reader:
            result=b''
            while True:
                part=reader.read(11)
                if not part:break
                result+=part
        self.assertEqual(result,b'abc'*100)
    def test_incomplete_source_fails_before_staging(self):
        (self.source/'_CHECKPOINT_METADATA').unlink()
        staging=self.root/'nas'/'bundle'
        with self.assertRaises(ValueError):export.export_checkpoint(self.source,staging,1)
        self.assertFalse(staging.exists())
    def test_inventory_cache_avoids_probe_and_survives_refresh(self):
        with patch.object(Path,'home',return_value=self.root):
            inv=inventory.RemoteCheckpointInventory({'target':{'checkpoint_base_dir':'/weights'}})
            inv.entries={'target':dict(state='ready',updated_at=__import__('time').time(),
                         checkpoints=[dict(config_name='config',experiment='exp',step=20000)])}
            with patch.object(inv.pool,'submit') as submit:
                inv.snapshot(force=True,source='target');submit.assert_not_called()
            inv.entries['target']['state']='loading'
            self.assertEqual(inv.selected('target','config','exp',20000)['step'],20000)
            inv.pool.shutdown()

if __name__=='__main__':unittest.main()
