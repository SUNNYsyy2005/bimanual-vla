from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from server_4090.app import create_app
from server_4090 import eval_video_catalog as catalog
from server_4090.eval_video_catalog_runner import organize_row


class DashboardEvalVideoManagementTest(unittest.TestCase):
    def test_canonical_catalog_keeps_event_edits_and_deletes_only_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, token, videos = self._make_app(root)
            run = videos / "handover_mic" / "exp"
            run.mkdir(parents=True)
            original = run / "episode0.mp4"
            original.write_bytes(b"video")
            (run / "_episode_results.jsonl").write_text(json.dumps({"episode_index": 0, "success": True, "event_timeline": [{"step": 4, "current_event": 2, "max_event_reached": 2}]}) + "\n")
            destination = root / "workspace/eval_video_catalog"
            registry = {}
            row = catalog.scan_source({"root": str(videos), "stable_seconds": 0})[0]
            organize_row(row, destination, registry)
            catalog.atomic_write(destination / "registry.json", registry)
            client = app.test_client()
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get('/api/eval-videos?video_source=robotwin&task=handover_mic&events=with', headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["total"], 1)
            item = response.json["videos"][0]
            self.assertEqual(item["event_track"]["max_event_reached"], 2)
            stream = client.get(item["url"], headers=headers)
            self.assertEqual(stream.data, b"video")
            stream.close()
            saved = client.post(f"/api/eval-videos/{item['id']}/event-overrides", headers=headers,
                                json={"step": 4, "current_event": 3, "max_event_reached": 3})
            self.assertEqual(saved.status_code, 200)
            after = client.get('/api/eval-videos', headers=headers).json["videos"][0]
            self.assertEqual(after["event_track"]["max_event_reached"], 3)
            deleted = client.post('/api/eval-videos/batch-delete', headers=headers, json={"video_ids": [item["id"]]})
            self.assertEqual(deleted.status_code, 200)
            self.assertTrue(original.is_file())
            self.assertEqual(client.get('/api/eval-videos', headers=headers).json["total"], 0)

    def test_remote_rollouts_use_cached_metadata_and_server_pagination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, token, _ = self._make_app(root)
            destination = root / "workspace/eval_video_catalog"
            rows = [{"video_source": "libero-safety", "task": "tsa-moka", "experiment": "exp1", "run_id": "run1", "phase": "train", "success": "success", "safe_success": "failed", "has_events": False,
                     "root": "/DATA/runs", "relative_path": f"episode{i}.mp4", "mtime": i + 1} for i in range(1005)]
            catalog.atomic_write(destination / "remote-inventory.json", {"sources": {"rlinf-h100": {"videos": rows}}})
            client = app.test_client()
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get('/api/eval-videos?video_source=libero-safety&task=tsa_moka&phase=train&safe_success=failed&offset=1000&limit=25', headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["total"], 1005)
            self.assertEqual(len(response.json["videos"]), 5)
            self.assertTrue(all(v["remote"] for v in response.json["videos"]))
            self.assertEqual(response.json["facets"]["video_sources"], ["libero-safety"])

    def test_generic_run_registration_accepts_new_backend_and_rejects_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, token, _ = self._make_app(root)
            client = app.test_client()
            headers = {"Authorization": f"Bearer {token}"}
            payload = {"video_source": "robotwin", "task": "handover-mic", "experiment": "sft", "run_id": "run1", "backend": "future-trainer", "kind": "distillation"}
            response = client.post('/api/artifact-runs', headers=headers, json=payload)
            self.assertEqual(response.status_code, 201)
            self.assertTrue((Path(response.json["path"]) / "run.json").is_file())
            history = client.get('/api/artifact-runs?backend=future-trainer', headers=headers)
            self.assertEqual(history.json["total"], 1)
            bad = client.post('/api/artifact-runs', headers=headers, json={**payload, "task": "../../outside"})
            self.assertEqual(bad.status_code, 400)

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
        template = (
            Path(__file__).resolve().parents[1] / "server_4090/templates/index.html"
        ).read_text(encoding="utf-8")

        self.assertEqual(template.count('id="module-eval-videos"'), 1)
        self.assertIn('id="navEvalVideoCount"', template)
        self.assertIn('id="evalVideoSelectAll"', template)
        self.assertIn('batchDeleteSelectedEvalVideos()', template)
        self.assertIn('/api/eval-videos/batch-delete', template)
        self.assertIn('const expandedEvalVideoIds = new Set();', template)
        self.assertIn('detachEvalVideoPlayers()', template)
        self.assertIn('restoreEvalVideoPlayers(detachedPlayers)', template)

    def test_eval_video_event_track_and_override_endpoint_are_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, token, eval_video_root = self._make_app(root)
            run = eval_video_root / "handover_mic_run"
            run.mkdir(parents=True, exist_ok=True)
            video = run / "episode0003.mp4"
            video.write_bytes(b"video")
            (run / "_episode_results.jsonl").write_text(
                json.dumps(
                    {
                        "episode_index": 3,
                        "success": False,
                        "stage_reward": 0.5,
                        "status": "failure",
                        "event_version": "event_v3",
                        "current_event": 0,
                        "max_event_reached": 2,
                        "event_timeline": [
                            {"step": 0, "current_event": 0, "max_event_reached": 0},
                            {"step": 4, "current_event": 2, "max_event_reached": 2},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            client = app.test_client()
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get("/api/eval-videos?limit=20", headers=headers)
            self.assertEqual(response.status_code, 200)
            item = response.get_json()["videos"][0]
            self.assertIn("event_track", item)
            self.assertEqual(item["event_track"]["max_event_reached"], 2)
            self.assertEqual(item["event_track"]["max_step"], 4)

            save = client.post(
                f"/api/eval-videos/{item['id']}/event-overrides",
                headers=headers,
                json={"step": 4, "current_event": 3, "max_event_reached": 3, "note": "checked"},
            )
            self.assertEqual(save.status_code, 200)
            saved = save.get_json()
            self.assertEqual(saved["override_count"], 1)
            self.assertEqual(saved["marker_semantics"], "start_frame_until_next_marker")
            self.assertNotIn("end_step", saved["edits"][0])
            self.assertTrue((run / "_event_overrides" / "episode0003.json").is_file())

            # Saving the same start frame replaces the marker rather than
            # creating an overlapping range annotation.
            corrected = client.post(
                f"/api/eval-videos/{item['id']}/event-overrides",
                headers=headers,
                json={"step": 4, "end_step": 999, "current_event": 2, "max_event_reached": 3},
            ).get_json()
            self.assertEqual(corrected["override_count"], 1)
            self.assertEqual(corrected["edits"][0]["start_step"], 4)
            self.assertNotIn("end_step", corrected["edits"][0])

            after = client.get("/api/eval-videos?limit=20", headers=headers).get_json()["videos"][0]
            self.assertEqual(after["event_track"]["override_count"], 1)
            self.assertEqual(after["event_track"]["overrides"]["edits"][0]["current_event"], 2)
            self.assertEqual(after["event_track"]["max_event_reached"], 3)


if __name__ == "__main__":
    unittest.main()
