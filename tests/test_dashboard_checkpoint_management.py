from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from server_4090.app import create_app, policy_config_name


class DashboardCheckpointManagementTest(unittest.TestCase):
    def _make_app(self, root: Path):
        dataset_root = root / "datasets"
        workspace_root = root / "workspace"
        assets_base_dir = root / "assets"
        checkpoint_base_dir = root / "checkpoints"
        openpi_repo = Path.cwd()
        for directory in (dataset_root, workspace_root, assets_base_dir, checkpoint_base_dir):
            directory.mkdir(parents=True, exist_ok=True)

        dataset_id = "real_ds"
        (dataset_root / dataset_id / "meta").mkdir(parents=True, exist_ok=True)
        (dataset_root / dataset_id / "meta" / "info.json").write_text(
            json.dumps({"robot_type": "piper"}), encoding="utf-8"
        )

        config_name = policy_config_name("single", "pi05")
        experiment_dir = checkpoint_base_dir / config_name / "exp_a"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_paths = []
        for step in ("5000", "10000"):
            step_dir = experiment_dir / step
            (step_dir / "params").mkdir(parents=True, exist_ok=True)
            (step_dir / "params" / "_METADATA").write_text("{}", encoding="utf-8")
            (step_dir / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
            (step_dir / "assets" / dataset_id).mkdir(parents=True, exist_ok=True)
            (step_dir / "assets" / dataset_id / "norm_stats.json").write_text("{}", encoding="utf-8")
            checkpoint_paths.append(step_dir)

        config = {
            "openpi_repo": str(openpi_repo),
            "openpi_python": sys.executable,
            "dataset_root": str(dataset_root),
            "workspace_root": str(workspace_root),
            "assets_base_dir": str(assets_base_dir),
            "checkpoint_base_dir": str(checkpoint_base_dir),
            "base_checkpoint": str(root / "base_checkpoint"),
            "checkpoint_allowed_roots": [str(checkpoint_base_dir)],
            "eval_video_roots": [],
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        env = {"BIMANUAL_VLA_SERVER_TOKEN": "x" * 32}
        with mock.patch.dict(os.environ, env, clear=False):
            app = create_app(config_path)
            app.config["TESTING"] = True
        return app, env["BIMANUAL_VLA_SERVER_TOKEN"], experiment_dir, checkpoint_paths

    def test_batch_delete_removes_selected_checkpoint_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, token, experiment_dir, checkpoint_paths = self._make_app(root)
            client = app.test_client()
            headers = {"Authorization": f"Bearer {token}"}

            status = client.get("/api/status", headers=headers)
            self.assertEqual(status.status_code, 200)
            self.assertEqual(len(status.get_json()["checkpoints"]), 2)

            response = client.post(
                "/api/checkpoints/batch-delete",
                headers=headers,
                json={"checkpoint_paths": [str(path) for path in checkpoint_paths]},
            )
            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            self.assertEqual(body["deleted_count"], 2)
            self.assertEqual(set(body["checkpoint_paths"]), {str(path) for path in checkpoint_paths})

            for path in checkpoint_paths:
                self.assertFalse(path.exists())
            self.assertFalse(experiment_dir.exists())

            status_after = client.get("/api/status", headers=headers)
            self.assertEqual(status_after.status_code, 200)
            self.assertEqual(status_after.get_json()["checkpoints"], [])

    def test_symlinked_simulation_checkpoint_is_available_as_training_base_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "datasets"
            workspace_root = root / "workspace"
            assets_base_dir = root / "assets"
            checkpoint_base_dir = root / "checkpoints"
            target_root = root / "dashboard-sim-checkpoints"
            for path in (dataset_root, workspace_root, assets_base_dir, checkpoint_base_dir, target_root):
                path.mkdir(parents=True, exist_ok=True)

            dataset_id = "sim_ds"
            (dataset_root / dataset_id / "meta").mkdir(parents=True)
            (dataset_root / dataset_id / "meta" / "info.json").write_text(
                json.dumps({"robot_type": "aloha"}), encoding="utf-8"
            )

            config_name = policy_config_name("bimanual", "pi05")
            experiment = "handover-mic-franka-camera-v7-bs8-10k-20260907"
            target_step = target_root / config_name / experiment / "10000"
            (target_step / "params").mkdir(parents=True)
            (target_step / "params" / "_METADATA").write_text("{}", encoding="utf-8")
            (target_step / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
            (target_step / "assets" / dataset_id).mkdir(parents=True)
            (target_step / "assets" / dataset_id / "norm_stats.json").write_text("{}", encoding="utf-8")

            logical_experiment = checkpoint_base_dir / config_name / experiment
            logical_experiment.parent.mkdir(parents=True, exist_ok=True)
            try:
                logical_experiment.symlink_to(target_root / config_name / experiment, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")

            base_checkpoint = root / "base-models" / "pi05_base"
            (base_checkpoint / "params").mkdir(parents=True)
            (base_checkpoint / "params" / "_METADATA").write_text("{}", encoding="utf-8")
            config = {
                "dashboard_profile": "simulation",
                "visible_dataset_origins": ["simulation"],
                "enable_policy": False,
                "openpi_repo": str(Path.cwd()),
                "openpi_python": sys.executable,
                "dataset_root": str(dataset_root),
                "workspace_root": str(workspace_root),
                "assets_base_dir": str(assets_base_dir),
                "checkpoint_base_dir": str(checkpoint_base_dir),
                "base_checkpoint": str(base_checkpoint),
                "checkpoint_allowed_roots": [str(checkpoint_base_dir), str(target_root), str(base_checkpoint.parent)],
                "eval_video_roots": [],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            token = "x" * 32
            with mock.patch.dict(os.environ, {"BIMANUAL_VLA_SERVER_TOKEN": token}, clear=False):
                app = create_app(config_path)
                app.config["TESTING"] = True

            response = app.test_client().get(
                "/api/status", headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 200)
            models = response.get_json()["base_models"]
            matching = [item for item in models if item.get("experiment") == experiment]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["checkpoint_step"], 10000)
            self.assertEqual(matching[0]["arm_mode"], "bimanual")


if __name__ == "__main__":
    unittest.main()
