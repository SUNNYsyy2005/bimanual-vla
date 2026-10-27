import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib
from server_4090.remote_checkpoints import scan_checkpoints, probe

class RemoteCheckpointsTest(unittest.TestCase):
    def test_scan_only_complete_numeric_steps_and_preserve_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'pi05_piper_bimanual_lora';exp=config/'test'
            for name in ('20000','19000','20001.orbax-checkpoint-tmp'):
                step=exp/name;(step/'params').mkdir(parents=True)
                (step/'_CHECKPOINT_METADATA').write_text('{}')
                if name!='19000': (step/'params/_METADATA').write_text('{}')
            marker=config/'.policy_action_conventions/test.json';marker.parent.mkdir()
            marker.write_text(json.dumps({'dataset_id':'fix','arm_mode':'bimanual','finetuning':{'rtc_training':True}}))
            rows=scan_checkpoints(root)
            self.assertEqual([r['step'] for r in rows],[20000])
            self.assertEqual(rows[0]['dataset_ids'],['fix'])
            self.assertTrue(rows[0]['finetuning']['rtc_training'])

    def test_h200_uses_compute_node_slurm_not_login_filesystem(self):
        payload=base64.urlsafe_b64encode(zlib.compress(json.dumps({'checkpoints':[{'step':20000}]}).encode())).decode()
        with patch('server_4090.cluster_read_guard.admit_read'), patch('server_4090.remote_checkpoints.ssh',side_effect=['123\n','JobState=RUNNING Comment=checkpoint_inventory:'+payload]) as ssh:
            rows=probe({'access_mode':'slurm_only','checkpoint_base_dir':'/independent/checkpoints','submit_host':'login-server',
                        'openpi_python':'/conda/bin/python','partition':'h200','node':'h200-ali-01'})
            self.assertEqual(rows,[{'step':20000}])
            script=ssh.call_args_list[0].kwargs['input']
            self.assertIn('#SBATCH -w h200-ali-01',script)
            self.assertIn('srun --ntasks=1',script)
            self.assertIn('/conda/bin/python',script)
            self.assertNotIn('/DATA/NAS',script)

    def test_missing_root_is_error_not_empty_success(self):
        with self.assertRaises(FileNotFoundError): scan_checkpoints('/nonexistent/checkpoint/root')
