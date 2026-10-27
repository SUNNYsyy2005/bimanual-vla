import json
import ast
from pathlib import Path
import tempfile
import unittest
import sys

from server_4090 import eval_video_catalog as catalog
from server_4090 import experiment_artifacts as artifacts
from server_4090.eval_video_catalog_runner import organize_row
from server_4090.rlinf_dashboard import scan_runs


class ExperimentArtifactsTest(unittest.TestCase):
    def test_submission_cannot_overwrite_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = artifacts.identity("robotwin", "handover-mic", "sft", "run1")
            directory = artifacts.run_directory(root / "experiment_artifacts", fields)
            directory.mkdir(parents=True)
            catalog.atomic_write(directory / "run.json", artifacts.manifest(fields, backend="openpi", kind="sft", state="created"))
            config = {"workspace_root": str(root)}
            task = {"id": "train1", "state": "starting", "log_path": str(root / "task.log"), "metadata": {"artifact_identity": fields}}
            artifacts.record_task(config, task)
            with self.assertRaisesRegex(ValueError, "already exists"):
                artifacts.record_task(config, {**task, "id": "train2"})
            self.assertEqual(catalog.read_object(directory / "run.json")["origin"]["task_id"], "train1")

    def test_tensorboard_only_history_survives_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old" / "run1"
            (old / "tensorboard").mkdir(parents=True)
            (old / "tensorboard/events.out.tfevents.test").write_bytes(b"event-file")
            config = {"rlinf_dashboard": {"sources": [{"id": "legacy", "root": str(root / "old"), "python": sys.executable}]}}
            before = scan_runs(config)
            entry = {"path": str(old), "artifact_root": str(root / "archive"), "backend": "rlinf", "state": "completed",
                     "video_source": "libero-safety", "task": "tsa-moka", "experiment": "ppo", "run_id": "run1"}
            artifacts.migrate(entry, apply=True)
            after = scan_runs(config)
            self.assertEqual(len(after), 1)
            self.assertEqual(before[0]["id"], after[0]["id"])

    def test_migration_moves_full_episode_and_keeps_old_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old" / "tsa_moka" / "train-job123"
            episode = old / "train-round003-parallel" / "worker0" / "data" / "seed123"
            episode.mkdir(parents=True)
            (episode / "review.mp4").write_bytes(b"video")
            (episode / "trajectory.npz").write_bytes(b"trajectory")
            (episode / "summary.json").write_text(json.dumps({"success": True, "safe_success": False, "split": "train"}))
            (old / "actor-round002.pt").write_bytes(b"checkpoint")
            (old / "training_curve.json").write_text("[]")
            entry = {"path": str(old), "artifact_root": str(root / "archive"), "backend": "rlinf", "state": "completed",
                     "video_source": "libero-safety", "task": "tsa-moka", "experiment": "ppo", "run_id": "job123"}
            preview = artifacts.migrate(entry, apply=False)
            self.assertEqual(preview["action"], "move")
            self.assertFalse((root / "archive").exists())
            result = artifacts.migrate(entry, apply=True)
            target = Path(result["target"])
            self.assertTrue(old.is_symlink())
            self.assertEqual((episode / "trajectory.npz").read_bytes(), b"trajectory")
            self.assertTrue(episode.is_symlink())
            self.assertTrue((old / "actor-round002.pt").is_file())
            self.assertEqual((target / "checkpoints/actor-round002.pt").read_bytes(), b"checkpoint")
            self.assertEqual(result["episodes"], 1)
            video = next((target / "train/unknown/success").glob("*/review.mp4"))
            metadata = catalog.read_object(catalog.metadata_path(video))
            self.assertEqual(metadata["safe_success"], "failed")
            self.assertEqual(metadata["rollout_round"], 3)
            self.assertEqual(artifacts.migrate(entry, apply=True)["action"], "already_migrated")
            # Discover migrated runs through the compatibility path exactly once.
            self.assertEqual(len(list(catalog.walk_videos(root / "old"))), 1)

    def test_openpi_history_retains_orbax_layout_and_marks_incomplete_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "pi05_piper_bimanual_lora" / "experiment"
            (old / "1000/params").mkdir(parents=True)
            (old / "1000/_CHECKPOINT_METADATA").write_text("{}")
            (old / "2000").mkdir()
            marker = old.parent / ".policy_action_conventions/experiment.json"
            marker.parent.mkdir()
            contract = {"contract_version": 2, "raw_action_dim": 14, "model_action_dim": 32, "model_action_convention": "joint_target"}
            marker.write_text(json.dumps(contract))
            entry = {"path": str(old), "artifact_root": str(root / "artifacts"), "backend": "openpi", "kind": "sft", "state": "completed",
                     "video_source": "robotwin", "task": "handover-mic", "experiment": "sft", "run_id": "run1"}
            result = artifacts.migrate(entry, apply=True)
            target = Path(result["target"])
            self.assertTrue((old / "1000/params").is_dir())
            index = catalog.read_object(target / "checkpoints/index.json")
            self.assertEqual({r["id"]: r["complete"] for r in index["checkpoints"]}, {"1000": True, "2000": False})
            from server_4090.app import checkpoint_action_contract, infer_model_variant, training_checkpoint_identity
            # Exercise marker lookup without importing the GPU training stack.
            helper = Path(__file__).resolve().parents[1] / "server_4090/openpi_single_arm.py"
            tree = ast.parse(helper.read_text())
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_checkpoint_marker_for_step"]
            namespace = {"Path": Path, "ACTION_CONVENTION_REGISTRY": ".policy_action_conventions"}
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(helper), "exec"), namespace)
            resolved = (old / "1000").resolve()
            self.assertEqual(infer_model_variant(resolved), "pi05")
            self.assertEqual(training_checkpoint_identity(resolved, old.parent.parent)["arm_mode"], "bimanual")
            self.assertEqual(checkpoint_action_contract(resolved), contract)
            self.assertEqual(catalog.read_object(namespace["_checkpoint_marker_for_step"](resolved)), contract)
            self.assertTrue(marker.is_file())
            # Upgrade a previously migrated run without moving its weights again.
            (target / "metadata/action-contract.json").unlink()
            manifest = catalog.read_object(target / "run.json")
            manifest.pop("migration_revision")
            catalog.atomic_write(target / "run.json", manifest)
            self.assertEqual(artifacts.migrate(entry, apply=True)["action"], "resumed")
            self.assertEqual(checkpoint_action_contract(resolved), contract)

    def test_active_run_is_never_moved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old"
            old.mkdir()
            entry = {"path": str(old), "artifact_root": str(root / "artifacts"), "backend": "rlinf", "state": "running",
                     "video_source": "libero-safety", "task": "tsa-moka", "experiment": "ppo", "run_id": "run1"}
            self.assertEqual(artifacts.migrate(entry, apply=True)["action"], "deferred_active")
            self.assertFalse(old.is_symlink())

    def test_task_identity_rejects_traversal(self):
        with self.assertRaises(ValueError):
            artifacts.identity("libero-safety", "../../outside", "exp", "run")

    def test_catalog_keeps_video_evidence_and_manual_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "handover_mic" / "exp"
            run.mkdir(parents=True)
            video = run / "episode0.mp4"
            video.write_bytes(b"video")
            (run / "_episode_results.jsonl").write_text(json.dumps({"episode_index": 0, "success": False, "max_event_reached": 3}) + "\n")
            overrides = run / "_event_overrides/episode0.json"
            overrides.parent.mkdir()
            overrides.write_text('{"edits": [{"start_step": 2, "current_event": 1}]}')
            row = catalog.scan_source({"root": str(root), "stable_seconds": 0})[0]
            registry = {}
            destination = root.parent / (root.name + "-catalog")
            try:
                organize_row(row, destination, registry)
                target = destination / row["catalog_relative_path"]
                self.assertTrue(target.is_symlink())
                self.assertEqual((target.parent / "_event_overrides" / (target.stem + ".json")).read_text(), overrides.read_text())
                self.assertEqual(catalog.classify(destination, row["catalog_relative_path"])["episode_record"]["max_event_reached"], 3)
                target.unlink()
                self.assertEqual(organize_row(row, destination, registry), "deleted")
                self.assertTrue(video.is_file())
            finally:
                import shutil
                shutil.rmtree(destination, ignore_errors=True)

    def test_no_aggregate_or_substring_success_and_string_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "success-experiment"
            run.mkdir()
            (run / "_result.txt").write_text("success rate 1.0")
            self.assertEqual(catalog.classify(root, "success-experiment/episode0.mp4")["success"], "unknown")
            (run / "_episode_results.jsonl").write_text(json.dumps({"episode_index": 0, "success": "false"}) + "\n")
            self.assertEqual(catalog.classify(root, "success-experiment/episode0.mp4")["success"], "failed")


if __name__ == "__main__":
    unittest.main()
