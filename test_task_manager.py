from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

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
    ) -> Path:
        task_dir = Path(self.tempdir.name) / "tasks" / task_id
        task_dir.mkdir(parents=True)
        task = {
            "id": task_id,
            "type": task_type,
            "state": state,
            "created_at": "2026-08-03T00:00:00+0800",
            "command": ["python", "/tmp/task.py", "--config"],
            "metadata": metadata or {},
            "log_path": str(task_dir / "task.log"),
        }
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

    def test_accepts_dependency_field_used_only_by_terminal_task(self):
        norm_dir = self.write_task("norm-old", task_type="norm", state="completed")
        self.write_task(
            "train-old",
            state="failed",
            dependency={"task_id": "norm-old", "type": "norm"},
        )

        self.manager.delete("norm-old")

        self.assertFalse(norm_dir.exists())


if __name__ == "__main__":
    unittest.main()
