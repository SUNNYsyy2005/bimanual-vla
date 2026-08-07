from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from server_4090.app import create_app


class DashboardEvalVideoManagementTest(unittest.TestCase):
    def _make_app(self, root: Path):
        dataset_root = root / "datasets"
        workspace_root = root / "workspace"
        assets_base_dir = root / "assets"
        checkpoint_base_dir = root / "checkpoints"
        eval_video_root = root / "eval_videos"
        openpi_repo = Path.cwd()
        for directory in (dataset_root, workspace_root, assets_base_dir, checkpoint_base_dir, eval_video_root):
            directory.mkdir(parents=True, exist_ok=True)

        config = {
            "openpi_repo": str(openpi_repo),
            "openpi_python": sys.executable,
            "dataset_root": str(dataset_root),
            "workspace_root": str(workspace_root),
            "assets_base_dir": str(assets_base_dir),
            "checkpoint_base_dir": str(checkpoint_base_dir),
            "base_checkpoint": str(root / "base_checkpoint"),
            "checkpoint_allowed_roots": [str(checkpoint_base_dir)],
            "eval_video_roots": [str(eval_video_root)],
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        token = "x" * 32
        with mock.patch.dict(os.environ, {"BIMANUAL_VLA_SERVER_TOKEN": token}, clear=False):
            app = create_app(config_path)
            app.config["TESTING"] = True
        return app, token, eval_video_root

    def test_batch_delete_removes_selected_eval_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, token, eval_video_root = self._make_app(root)
            local_run = eval_video_root / "exp_a"
            nested_run = local_run / "nested"
            nested_run.mkdir(parents=True, exist_ok=True)
            video_a = local_run / "episode0001.mp4"
            video_b = nested_run / "episode0002.webm"
            video_a.write_bytes(b"video-a")
            video_b.write_bytes(b"video-b")

            client = app.test_client()
            headers = {"Authorization": f"Bearer {token}"}

            status = client.get("/api/eval-videos?limit=20", headers=headers)
            self.assertEqual(status.status_code, 200)
            body = status.get_json()
            self.assertEqual(body["total"], 2)
            self.assertTrue(all(item["deletable"] for item in body["videos"]))

            response = client.post(
                "/api/eval-videos/batch-delete",
                headers=headers,
                json={"video_ids": [item["id"] for item in body["videos"]]},
            )
            self.assertEqual(response.status_code, 200)
            deleted = response.get_json()
            self.assertEqual(deleted["deleted_count"], 2)
            self.assertEqual(len(deleted["deleted_videos"]), 2)

            self.assertFalse(video_a.exists())
            self.assertFalse(video_b.exists())
            self.assertFalse(nested_run.exists())
            self.assertFalse(local_run.exists())
            self.assertTrue(eval_video_root.exists())

            status_after = client.get("/api/eval-videos?limit=20", headers=headers)
            self.assertEqual(status_after.status_code, 200)
            self.assertEqual(status_after.get_json()["total"], 0)

    def test_eval_video_dashboard_has_navigation_and_bulk_controls(self):
        template = (Path(__file__).parent / "server_4090/templates/index.html").read_text(encoding="utf-8")

        self.assertEqual(template.count('id="module-eval-videos"'), 1)
        self.assertIn('id="navEvalVideoCount"', template)
        self.assertIn('id="evalVideoSelectAll"', template)
        self.assertIn('batchDeleteSelectedEvalVideos()', template)
        self.assertIn('/api/eval-videos/batch-delete', template)


if __name__ == "__main__":
    unittest.main()
