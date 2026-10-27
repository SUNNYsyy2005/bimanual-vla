import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from server_4090.extension_nodes import validate_node, resource_snapshot
from tests.test_dashboard_checkpoint_management import DashboardCheckpointManagementTest


def node():
    return dict(label='New GPU node',submit_host='new-login',partition='gpu',node='gpu-01',
                access_mode='slurm_only',gpu_type='a100',openpi_python='/remote/conda/envs/openpi/bin/python',
                workdir='/remote/openpi',dashboard_repo='/remote/vla',dataset_root='/remote/data',
                assets_base_dir='/remote/assets',checkpoint_base_dir='/remote/checkpoints',
                base_checkpoint='/remote/base',cache_root='/remote/cache',log_dir='/remote/logs',remote_job_dir='/remote/jobs')


class ExtensionNodesTest(unittest.TestCase):
    def test_remote_paths_do_not_need_to_exist_locally_and_optional_paths_fall_back(self):
        result=validate_node('new-node',{**node(),'openpi_src':'','checkpoint_read_roots':'/remote/old\n/remote/archive'})
        self.assertEqual(result['checkpoint_read_roots'],['/remote/old','/remote/archive'])
        self.assertNotIn('openpi_src',result)
        self.assertEqual(result['mem'],'64G')

    def test_invalid_scheduler_host_and_resource_values_are_rejected(self):
        for change in ({'node':'x\n#SBATCH --gres=gpu:8'},{'submit_host':'-oProxyCommand=bad'},
                       {'mem':'513G'},{'gpu_count':0},{'workdir':'relative'},{'time':'1;id'}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_node('new-node',{**node(),**change})

    def test_add_edit_reload_and_delete_are_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            app,token,_,_=DashboardCheckpointManagementTest()._make_app(root)
            client=app.test_client();headers={'Authorization':'Bearer '+token}
            url='/api/settings/extension-nodes/custom-a100'
            response=client.put(url,json=node(),headers=headers)
            self.assertEqual(response.status_code,200,response.get_json())
            self.assertFalse(response.get_json()['restart_required'])
            self.assertIn('custom-a100',json.loads((root/'config.json').read_text())['cluster_targets'])
            response=client.put(url,json={'label':'Edited','checkpoint_base_dir':'/remote/new'},headers=headers)
            self.assertEqual(response.status_code,200,response.get_json())
            target=client.get('/api/settings/extension-nodes',headers=headers).get_json()['nodes']['custom-a100']
            self.assertEqual(target['label'],'Edited');self.assertEqual(target['submit_host'],'new-login')
            self.assertEqual(client.delete(url,headers=headers).status_code,200)
            self.assertEqual(client.get('/api/settings/extension-nodes',headers=headers).get_json()['nodes'],{})

    def test_resource_query_uses_configured_host_and_node(self):
        class Result:
            returncode=0;stdout='resources';stderr=''
        with patch('server_4090.extension_nodes.subprocess.run',return_value=Result()) as run:
            result=resource_snapshot({'custom-a100':node()})
            # Non-protocol output must be reported as incomplete, not success.
            self.assertFalse(result['ok']);command=run.call_args.args[0]
            self.assertIn('new-login',command);self.assertIn('gpu-01',command[-1])
            self.assertNotIn('h100',command[-1]);self.assertNotIn('h200',command[-1])

    def test_active_task_prevents_edit_and_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, token, _, _ = DashboardCheckpointManagementTest()._make_app(root)
            client = app.test_client()
            headers = {'Authorization': 'Bearer ' + token}
            url = '/api/settings/extension-nodes/custom-a100'
            self.assertEqual(client.put(url, json=node(), headers=headers).status_code, 200)
            original = (root / 'config.json').read_text()
            for field in ('execution_target', 'cluster_target', 'slurm_target', 'source', 'target'):
                with self.subTest(field=field), patch('server_4090.app.TaskManager.list', return_value=[
                    {'id': 'active-test', 'state': 'running', 'metadata': {field: 'custom-a100'}}
                ]):
                    self.assertEqual(client.put(url, json={'label': 'changed'}, headers=headers).status_code, 400)
                    self.assertEqual(client.delete(url, headers=headers).status_code, 400)
                    self.assertEqual((root / 'config.json').read_text(), original)
