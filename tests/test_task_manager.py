from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from server_4090.app import TaskManager


class _RunningProcess:
    def poll(self):
        return None


class TaskManagerDeleteTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.manager = TaskManager({
            "workspace_root": self.tempdir.name,
            "task_monitor_interval_s": 0,
            "openpi_python": "/opt/openpi/bin/python",
            "openpi_repo": "/opt/openpi",
            "dataset_root": "/datasets",
            "assets_base_dir": "/assets",
            "checkpoint_base_dir": "/checkpoints",
            "base_checkpoint": "/base/pi05",
            "allowed_gpu_ids": [0, 1, 2, 3],
            "evaluation_min_free_gpu_mib": 23000,
            "evaluation_xla_memory_fraction": 0.85,
        })

    def tearDown(self):
        self.manager.close()
        self.tempdir.cleanup()

    def write_task(
        self,
        task_id: str,
        *,
        task_type: str = "train",
        state: str = "completed",
        metadata: dict | None = None,
        dependency: dict | None = None,
        pid: int | None = None,
        command: list[str] | None = None,
        launch_command: list[str] | None = None,
    ) -> Path:
        task_dir = Path(self.tempdir.name) / "tasks" / task_id
        task_dir.mkdir(parents=True)
        task = {
            "id": task_id,
            "type": task_type,
            "state": state,
            "created_at": "2026-08-03T00:00:00+0800",
            "command": command or ["python", "/tmp/task.py", "--config"],
            "metadata": metadata or {},
            "log_path": str(task_dir / "task.log"),
        }
        if pid is not None:
            task["pid"] = pid
        if launch_command is not None:
            task["launch_command"] = launch_command
        if dependency is not None:
            task["dependency"] = dependency
        (task_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
        (task_dir / "task.log").write_text("history\n", encoding="utf-8")
        return task_dir

    def test_deletes_terminal_task_record_and_log(self):
        task_dir = self.write_task("train-history", state="completed")

        result = self.manager.delete("train-history")

        self.assertTrue(result["deleted"])
        self.assertEqual(result["task"]["id"], "train-history")
        self.assertFalse(task_dir.exists())

    def test_deletes_multiple_terminal_task_records_and_logs(self):
        first_dir = self.write_task("train-history-a", state="completed")
        second_dir = self.write_task("eval-history-b", task_type="eval", state="failed")

        result = self.manager.delete_many(["train-history-a", "eval-history-b", "train-history-a"])

        self.assertTrue(result["deleted"])
        self.assertEqual(result["deleted_count"], 2)
        self.assertEqual(result["task_ids"], ["train-history-a", "eval-history-b"])
        self.assertFalse(first_dir.exists())
        self.assertFalse(second_dir.exists())

    def test_batch_delete_validates_all_tasks_before_removing_anything(self):
        history_dir = self.write_task("train-history", state="completed")
        active_dir = self.write_task("train-active", state="running")
        self.manager.processes["train-active"] = _RunningProcess()

        with self.assertRaisesRegex(ValueError, "cannot delete active task train-active"):
            self.manager.delete_many(["train-history", "train-active"])

        self.assertTrue(history_dir.exists())
        self.assertTrue(active_dir.exists())

    def test_batch_delete_allows_terminal_dependency_in_same_batch(self):
        norm_dir = self.write_task("norm-history", task_type="norm", state="completed")
        train_dir = self.write_task(
            "train-history",
            state="failed",
            metadata={"depends_on": "norm-history"},
        )

        self.manager.delete_many(["norm-history", "train-history"])

        self.assertFalse(norm_dir.exists())
        self.assertFalse(train_dir.exists())

    def test_rejects_task_with_live_process(self):
        task_dir = self.write_task("train-running", state="running")
        self.manager.processes["train-running"] = _RunningProcess()

        with self.assertRaisesRegex(ValueError, "cannot delete active task"):
            self.manager.delete("train-running")

        self.assertTrue(task_dir.exists())

    def test_rejects_terminal_dependency_used_by_active_task(self):
        norm_dir = self.write_task("norm-history", task_type="norm", state="completed")
        self.write_task(
            "train-waiting",
            state="waiting_gpu",
            metadata={"depends_on": "norm-history"},
        )

        with self.assertRaisesRegex(ValueError, "active dependent task"):
            self.manager.delete("norm-history")

        self.assertTrue(norm_dir.exists())

    def make_auto_eval_train(self) -> dict:
        checkpoint_dir = Path(self.tempdir.name) / "checkpoints" / "experiment"
        step = checkpoint_dir / "5000"
        (step / "params").mkdir(parents=True)
        (step / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
        (step / "params" / "_METADATA").write_text("{}", encoding="utf-8")
        return {
            "id": "train-live",
            "type": "train",
            "state": "running",
            "created_at": "2026-08-04T00:00:00+0800",
            "command": ["python", "train.py"],
            "metadata": {
                "dataset_id": "dataset-v3",
                "arm_mode": "single",
                "arm_side": "right",
                "schema": "delivery",
                "model_variant": "pi05",
                "base_checkpoint": "/base/pi05",
                "checkpoint_dir": str(checkpoint_dir),
                "save_interval": 1000,
                "test_episodes": 1,
                "test_episode_indexes": [8],
                "test_ratio": 0.1,
                "split_seed": 42,
                "contract_version": 3,
                "raw_action_dim": 10,
                "model_action_dim": 7,
                "raw_action_semantics": "absolute_eef_target",
                "model_action_semantics": "eef_delta_chunk_origin_base_xyz_left_rotvec_gripper_opening_target",
                "raw_action_convention": "absolute_eef_target",
                "model_action_convention": "chunk_origin",
                "raw_gripper_semantics": "absolute_opening_fraction_0_closed_1_open",
                "gripper_semantics": "absolute_opening_fraction_0_closed_1_open",
                "action_offset": 1,
                "model_action_start_offset": 1,
                "auto_eval": {
                    "enabled": True,
                    "every_steps": 5000,
                    "batch_size": 1,
                    "num_workers": 0,
                    "max_batches": 2,
                    "seed": 42,
                    "minimum_free_gpu_mib": 23000,
                    "xla_memory_fraction": 0.85,
                },
            },
        }

    @mock.patch("server_4090.app.gpu_inventory", return_value=[])
    def test_auto_eval_records_no_gpu_skip_once(self, _inventory):
        train = self.make_auto_eval_train()

        self.manager._reconcile_auto_evals_locked([train])
        eval_tasks = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (Path(self.tempdir.name) / "tasks").glob("eval-*/task.json")
        ]
        self.assertEqual(len(eval_tasks), 1)
        self.assertEqual(eval_tasks[0]["state"], "skipped")
        self.assertEqual(eval_tasks[0]["skip_reason"], "no_idle_gpu")

        self.manager._reconcile_auto_evals_locked([train, *eval_tasks])
        self.assertEqual(len(list((Path(self.tempdir.name) / "tasks").glob("eval-*/task.json"))), 1)

    @mock.patch("server_4090.app.cuda_visible_devices", return_value="GPU-zero")
    @mock.patch("server_4090.app.gpu_inventory")
    def test_auto_eval_launches_independent_single_gpu_task(self, inventory, _visible):
        inventory.return_value = [{
            "index": 0,
            "uuid": "GPU-zero",
            "memory_total_mib": 24564,
            "memory_used_mib": 10,
            "processes": [],
            "compute_available": True,
        }]
        train = self.make_auto_eval_train()
        launched = []

        def fake_launch(task, *, env, raise_on_error):
            task["state"] = "running"
            task["captured_env"] = env
            launched.append(task)
            return task

        with mock.patch.object(self.manager, "_launch", side_effect=fake_launch):
            self.manager._reconcile_auto_evals_locked([train])

        self.assertEqual(len(launched), 1)
        task = launched[0]
        self.assertEqual(task["metadata"]["gpu_ids"], [0])
        self.assertEqual(task["metadata"]["checkpoint_step"], 5000)
        self.assertIn("eval_heldout_loss.py", " ".join(task["command"]))
        self.assertEqual(task["captured_env"]["CUDA_VISIBLE_DEVICES"], "GPU-zero")
        self.assertEqual(task["captured_env"]["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.85")

    def test_accepts_dependency_field_used_only_by_terminal_task(self):
        norm_dir = self.write_task("norm-old", task_type="norm", state="completed")
        self.write_task(
            "train-old",
            state="failed",
            dependency={"task_id": "norm-old", "type": "norm"},
        )

        self.manager.delete("norm-old")

        self.assertFalse(norm_dir.exists())

    def test_discover_external_policy_ignores_managed_runner_descendant(self):
        managed_dir = self.write_task(
            "policy-managed",
            task_type="policy",
            state="running",
            pid=111,
            command=["/opt/openpi/bin/python", "openpi_single_arm.py", "serve", "--port", "8000"],
            launch_command=[
                "/usr/bin/python3",
                "/repo/server_4090/task_runner.py",
                "--exit-json",
                "/tmp/exit.json",
            ],
            metadata={"port": 8000},
        )
        duplicate_dir = self.write_task(
            "policy-external-222",
            task_type="policy",
            state="running",
            pid=222,
            command=["/opt/openpi/bin/python", "openpi_single_arm.py", "serve", "--port", "8000"],
            metadata={"external": True, "adopted": True, "port": 8000},
        )

        with mock.patch("server_4090.app.pid_alive", return_value=True), \
            mock.patch("server_4090.app.process_matches_task", return_value=True), \
            mock.patch("server_4090.app.process_children_by_parent", return_value={111: {222}}), \
            mock.patch("server_4090.app.discover_external_policy_candidates", return_value=[]) as discover:
            adopted = self.manager.discover_external_policies()

        self.assertEqual(adopted, [])
        discover.assert_called_once()
        self.assertEqual(discover.call_args.kwargs["ignored_pids"], {111, 222})
        self.assertTrue(managed_dir.exists())
        self.assertFalse(duplicate_dir.exists())

    def test_training_metrics_probe_detects_curve_and_empty_train_logs(self):
        metric_dir = self.write_task("train-with-metrics", task_type="train", state="completed")
        (metric_dir / "task.log").write_text(
            "Step 10: loss=0.5\nStep 20: loss=0.25, grad_norm=0.1\n",
            encoding="utf-8",
        )
        metric_task = json.loads((metric_dir / "task.json").read_text(encoding="utf-8"))

        probe = self.manager.training_metrics_probe(metric_task)

        self.assertEqual(probe["status"], "has_metrics")
        self.assertTrue(probe["has_points"])
        self.assertEqual(probe["total_points"], 2)
        self.assertEqual(probe["latest_step"], 20)

        empty_dir = self.write_task("train-empty", task_type="train", state="completed")
        (empty_dir / "task.log").write_text("launcher exited before first step\n", encoding="utf-8")
        empty_task = json.loads((empty_dir / "task.json").read_text(encoding="utf-8"))

        empty_probe = self.manager.training_metrics_probe(empty_task)

        self.assertEqual(empty_probe["status"], "no_metrics")
        self.assertFalse(empty_probe["has_points"])

    def test_training_metrics_probe_marks_unstreamed_slurm_as_unknown(self):
        task_dir = self.write_task(
            "train-slurm",
            task_type="train",
            state="completed",
            metadata={"runtime": "slurm", "slurm_target": "h100"},
        )
        (task_dir / "task.log").write_text("[dashboard] slurm_job_id=123\n", encoding="utf-8")
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))

        probe = self.manager.training_metrics_probe(task)

        self.assertEqual(probe["status"], "unknown")
        self.assertIsNone(probe["has_points"])


if __name__ == "__main__":
    unittest.main()
