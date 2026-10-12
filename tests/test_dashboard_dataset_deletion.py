from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from server_4090.app import TaskManager
from server_4090.dataset_delete_runner import DELETE_SCRIPT, refresh_local_cache, slurm_deletion_command
from tests.test_rlinf_dashboard import RlinfDashboardTest


class DatasetCopyDeletionTest(RlinfDashboardTest):
    def setUp(self):
        super().setUp()
        self.archive = self.root / "archive"
        self.config["local_storage_locations"] = {"local_4090_hdd": {"kind":"local_archive", "dataset_root":str(self.archive)}}
        self.config["cluster_targets"] = {"h200": {
            "dataset_root":"/DATA/test/datasets", "submit_host":"login-server", "partition":"h200", "node":"h200-test",
            "access_mode":"slurm_only", "openpi_python":"/DATA/test/conda/envs/openpi/bin/python",
            "workdir":"/DATA/test", "remote_job_dir":"/DATA/test/jobs", "log_dir":"/DATA/test/logs",
        }}
        for folder in [self.root / "datasets" / "local_ds", self.archive / "archive_ds"]:
            (folder / "meta").mkdir(parents=True)
            (folder / "meta/info.json").write_text(json.dumps({"robot_type":"aloha", "total_episodes":5}))

    def test_archive_not_in_read_roots_is_restored_to_main_list(self):
        response=self.request(self.make_app(), "get", "/api/datasets")
        self.assertEqual(response.status_code,200,response.json)
        rows={row['id']:row for row in response.json['datasets']}
        self.assertEqual(set(rows),{'local_ds','archive_ds'})
        self.assertFalse(rows['archive_ds']['remote_only'])
        self.assertEqual(rows['archive_ds']['locations'][0]['target'],'local_4090_hdd')

    def test_copy_delete_confirmation_and_target_are_required(self):
        client=self.make_app()
        for payload in [{"target":"local_4090"}, {"target":"/etc","confirm_dataset_id":"local_ds"}]:
            response=self.request(client,"post","/api/datasets/local_ds/delete-copy",json=payload)
            self.assertEqual(response.status_code,400,response.json)
        self.assertTrue((self.root/'datasets/local_ds').exists())

    def test_local_copy_delete_does_not_touch_archive(self):
        response=self.request(self.make_app(),"post","/api/datasets/local_ds/delete-copy",json={"target":"local_4090","confirm_dataset_id":"local_ds"})
        self.assertEqual(response.status_code,200,response.json)
        self.assertFalse((self.root/'datasets/local_ds').exists())
        self.assertTrue((self.archive/'archive_ds').exists())

    def test_archive_and_remote_delete_are_tracked_tasks(self):
        client=self.make_app()
        for target,ds in [("local_4090_hdd","archive_ds"),("h200","remote_ds")]:
            with mock.patch.object(TaskManager,"start",return_value={"id":"delete-task"}) as start:
                response=self.request(client,"post",f"/api/datasets/{ds}/delete-copy",json={"target":target,"confirm_dataset_id":ds})
                self.assertEqual(response.status_code,202,response.json)
                self.assertEqual(start.call_args.kwargs['metadata']['transfer_kind'],'dataset_delete')
                location=json.loads(base64.urlsafe_b64decode(start.call_args.args[1][-1]))
                self.assertEqual(location['name'],target)
                if target=='h200':self.assertEqual(location['openpi_python'],'/DATA/test/conda/envs/openpi/bin/python')
        self.assertTrue((self.archive/'archive_ds').exists())

    def test_slurm_delete_uses_no_gpu_and_refreshes_node_inventory(self):
        location={"name":"h200","dataset_root":"/DATA/test/datasets","submit_host":"login-server","partition":"h200","node":"h200-test", "openpi_python":"/DATA/conda/bin/python", "gpu_count":4,"conda_env":"other"}
        command=slurm_deletion_command(location,'ds')
        target=json.loads(base64.urlsafe_b64decode(command[command.index('--target-json')+1]))
        self.assertEqual(target['gpu_count'],0)
        self.assertNotIn('conda_env',target)
        commands=json.loads(base64.urlsafe_b64decode(command[command.index('--commands-json')+1]))
        self.assertIn('/DATA/conda/bin/python',commands[0][2])
        self.assertIn('configured root',commands[0][2])

    def test_runner_rejects_traversal_and_deletes_only_valid_dataset(self):
        for ds in ['../archive','/etc','archive_ds/../other']:
            result=subprocess.run([sys.executable,'-c',DELETE_SCRIPT,str(self.archive),ds],capture_output=True)
            self.assertNotEqual(result.returncode,0)
        result=subprocess.run([sys.executable,'-c',DELETE_SCRIPT,str(self.archive),'archive_ds'],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse((self.archive/'archive_ds').exists())
        self.assertTrue((self.root/'datasets/local_ds').exists())

    def test_active_deletion_blocks_new_dataset_jobs(self):
        manager=TaskManager(self.config)
        self.addCleanup(manager.close)
        folder=manager.root/'delete-pending';folder.mkdir()
        (folder/'task.json').write_text(json.dumps({'id':'delete-pending','type':'transfer','state':'running','metadata':{'dataset_id':'local_ds','transfer_kind':'dataset_delete'}}))
        with self.assertRaisesRegex(ValueError,'in use'):
            manager._new_task('train',[],{'dataset_id':'local_ds'},state='starting')


if __name__=='__main__':unittest.main()
