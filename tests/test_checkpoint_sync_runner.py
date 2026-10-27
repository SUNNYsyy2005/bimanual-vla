import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from server_4090.checkpoint_sync_runner import main

class CheckpointSyncTest(unittest.TestCase):
    def test_step_sync_preserves_model_contract_after_validated_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); marker=root/'config/.policy_action_conventions/exp.json'
            target=root/'config/exp'
            contract={'finetuning':{'rtc_training':True,'image_resize_mode':'stretch'},'dataset_id':'fix'}
            request=dict(source={'kind':'ssh','dataset_root':'/source/exp'},target={'dataset_root':str(target)},
                         step=20000,marker_path=str(marker),contract=contract,parallelism=4)
            encoded=base64.urlsafe_b64encode(json.dumps(request).encode()).decode()
            def transfer(cmd,check):
                self.assertEqual(cmd[cmd.index('--dataset-id')+1],'20000')
                step=target/'20000';(step/'params').mkdir(parents=True)
                (step/'params/_METADATA').write_text('{}');(step/'_CHECKPOINT_METADATA').write_text('{}')
            with patch('sys.argv',['runner','--request-json',encoded]),patch('server_4090.checkpoint_sync_runner.subprocess.run',side_effect=transfer): main()
            self.assertEqual(json.loads(marker.read_text()),contract)
            marker.write_text('{}')
            with patch('sys.argv',['runner','--request-json',encoded]),patch('server_4090.checkpoint_sync_runner.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError,'contract differs'): main()
                run.assert_not_called()
