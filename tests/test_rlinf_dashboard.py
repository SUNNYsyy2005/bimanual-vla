from __future__ import annotations

import json
import base64
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

from server_4090.app import create_app, TaskManager
from server_4090.configure_rlinf_native import configure_native
from server_4090.rlinf_dashboard import build_launch_spec, checkpoints_for_run, scan_runs, upgrade_native_profiles
from server_4090.rlinf_eval_runner import evaluation_overrides, hydra_argv, write_results


class RlinfDashboardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "openpi_repo": str(self.root), "openpi_python": sys.executable,
            "dataset_root": str(self.root / "datasets"), "workspace_root": str(self.root / "workspace"),
            "assets_base_dir": str(self.root / "assets"), "checkpoint_base_dir": str(self.root / "checkpoints"),
            "base_checkpoint": str(self.root / "base"), "dashboard_profile": "simulation",
            "eval_video_roots": [str(self.root / "videos")], "task_monitor_interval_s": 0,
        }
        self.config = configure_native(self.config, project=self.root / "project", python=sys.executable,
            model=self.root / "model", libero_config=self.root / "libero_config", libero_source=self.root / "libero",
            transformer_root=self.root / "transformers", run_root=self.root / "runs")
        # Use the lightweight local curve reader in API tests.
        self.config["rlinf_dashboard"]["sources"][0].pop("python")
        self.output = self.root / "runs" / "run" / "grpo"
        self.output.mkdir(parents=True)
        (self.output / "training_curve.json").write_text(json.dumps([{"actor_updates": 1, "loss": .4}]))
        weight = self.output / "checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
        weight.parent.mkdir(parents=True)
        with zipfile.ZipFile(weight, "w") as archive:
            archive.writestr("archive/data.pkl", b"fake-weights")
        self.weight = weight

    def make_app(self):
        path = self.root / "config.json"
        path.write_text(json.dumps(self.config), encoding="utf-8")
        with mock.patch.dict(os.environ, {"BIMANUAL_VLA_SERVER_TOKEN": "test-token-" + "x" * 24}):
            return create_app(path).test_client()

    def request(self, client, method, endpoint, **kwargs):
        return getattr(client, method)(endpoint, headers={"Authorization": "Bearer test-token-" + "x" * 24}, **kwargs)

    def add_task(self, task_id, *, state="completed", task_type="train", **metadata):
        folder = self.root / "workspace/tasks" / task_id
        folder.mkdir(parents=True)
        (folder / "task.json").write_text(json.dumps({"id": task_id, "type": task_type, "state": state,
            "metadata": {"rlinf_output_dir": str(self.output).replace("\\", "/"), "framework": "RLinf", **metadata}}))
        (folder / "task.log").write_text("training log")
        return folder

    def test_native_launch_uses_real_parameters(self):
        spec = build_launch_spec(self.config, {"target": "4x4090", "profile": "libero_safety_pi05", "algorithm": "grpo-lag", "run_name": "new",
            "learning_rate": .00001, "save_interval": 8, "group_size": 4, "num_groups": 3})
        self.assertIn("train_embodied_agent.py", " ".join(spec["command"]))
        self.assertIn("actor.optim.lr=1e-05", spec["command"])
        self.assertIn("runner.save_interval=8", spec["command"])
        self.assertIn("env.train.total_num_envs=12", spec["command"])
        self.assertIn("algorithm.safety.enabled=true", spec["command"])
        for fields in [{"learning_rate": "nan"}, {"save_interval": 0}, {"micro_batch_size": 3}, {"eval_interval": 0}]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                build_launch_spec(self.config, {"target": "4x4090", "profile": "libero_safety_pi05", "algorithm": "grpo", "run_name": "new", **fields})

    def test_native_upgrade_is_idempotent(self):
        config = {"rlinf_dashboard": {"targets": {"h100": {"profiles": {"native": {"command_template": ["bash", "-lc", "python train_embodied_agent.py runner.max_steps={updates}"], "overrides": {}}}}}}}
        self.assertEqual(upgrade_native_profiles(config), upgrade_native_profiles(upgrade_native_profiles(config)))

    def test_observed_native_run_can_read_its_metrics_log(self):
        (self.output / "metrics.log").write_text("RLInf step=1 loss=0.4", encoding="utf-8")
        client = self.make_app()
        run = self.request(client, "get", "/api/rlinf/runs").json["runs"][0]
        response = self.request(client, "get", f"/api/rlinf/runs/{run['id']}/log")
        self.assertEqual(response.status_code, 200, response.json)
        self.assertIn("step=1", response.json["log"])

    def test_checkpoints_exclude_missing_empty_and_outside_weights(self):
        run = scan_runs(self.config)[0]
        rows = checkpoints_for_run(run)
        self.assertEqual([row["step"] for row in rows], [1])
        self.weight.write_bytes(b"")
        self.assertEqual(checkpoints_for_run(run), [])

    def test_delete_persists_and_preserves_model_files(self):
        client = self.make_app()
        run_id = self.request(client, "get", "/api/rlinf/runs").json["runs"][0]["id"]
        response = self.request(client, "delete", f"/api/rlinf/runs/{run_id}")
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(self.weight.exists())
        self.assertTrue((self.output / "training_curve.json").exists())
        self.assertEqual(self.request(self.make_app(), "get", "/api/rlinf/runs").json["runs"], [])

    def test_batch_delete_validates_whole_selection(self):
        client = self.make_app()
        run_id = self.request(client, "get", "/api/rlinf/runs").json["runs"][0]["id"]
        response = self.request(client, "post", "/api/rlinf/runs/batch-delete", json={"run_ids": [run_id, "unknown"]})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.request(client, "get", "/api/rlinf/runs").json["runs"]), 1)

    def test_delete_terminal_task_log_and_block_active_task(self):
        # Paths in source config and task metadata must use the same native separators.
        self.config["rlinf_dashboard"]["sources"][0]["root"] = str(self.root / "runs").replace("\\", "/")
        folder = self.add_task("terminal")
        client = self.make_app()
        run = self.request(client, "get", "/api/rlinf/runs").json["runs"][0]
        self.assertEqual(run["task_id"], "terminal")
        self.assertEqual(self.request(client, "delete", f"/api/rlinf/runs/{run['id']}").status_code, 200)
        self.assertFalse(folder.exists())

    def test_eval_resolves_checkpoint_and_video_toggle_without_launching_gpu_work(self):
        client = self.make_app()
        run = self.request(client, "get", "/api/rlinf/runs").json["runs"][0]
        rows = self.request(client, "get", f"/api/rlinf/runs/{run['id']}/checkpoints").json["checkpoints"]
        for record in [False, True]:
            with mock.patch("server_4090.app.gpu_inventory", return_value=[{"index": 0, "memory_free_mib": 24000, "memory_total_mib": 24000, "processes": []}]), mock.patch.object(TaskManager, "start", return_value={"id": "fake-eval"}) as start:
                response = self.request(client, "post", "/api/rlinf/evaluate", json={"run_id": run["id"], "checkpoint_id": rows[0]["id"], "gpu_ids": [0], "record_video": record})
                self.assertEqual(response.status_code, 201, response.json)
                self.assertEqual(start.call_args.kwargs["metadata"]["record_video"], record)
                self.assertIn(str(record).lower(), " ".join(start.call_args.args[1]))
        response = self.request(client, "post", "/api/rlinf/evaluate", json={"run_id": run["id"], "checkpoint_id": "../../bad", "record_video": False})
        self.assertEqual(response.status_code, 400)

    def test_eval_result_and_override_contract(self):
        args = mock.Mock(checkpoint="/models/full_weights.pt", output="/eval", episodes=5, seed=43, max_steps=512, record_video=False)
        overrides = evaluation_overrides(args)
        self.assertIn("rollout.model=${actor.model}", overrides)
        self.assertIn("runner.ckpt_path=/models/full_weights.pt", overrides)
        self.assertIn("env.eval.video_cfg.save_video=false", overrides)
        write_results(self.root / "result", {"success_once": .8, "bad": float("nan")}, step=50, checkpoint="weights.pt")
        curve = json.loads((self.root / "result/evaluation_curve.json").read_text())
        self.assertEqual(curve, [{"actor_updates": 50, "success_once": .8}])

    def test_slurm_eval_embeds_current_helper_in_batch_command(self):
        self.config['rlinf_dashboard']['targets']['4x4090'].update(kind='slurm',submit_host='login-server',partition='h100',node='h100-test')
        client = self.make_app()
        run = self.request(client,'get','/api/rlinf/runs').json['runs'][0]
        checkpoint = self.request(client,'get',f"/api/rlinf/runs/{run['id']}/checkpoints").json['checkpoints'][0]
        with mock.patch.object(TaskManager,'start',return_value={'id':'eval-slurm'}) as start:
            response = self.request(client,'post','/api/rlinf/evaluate',json={'run_id':run['id'],'checkpoint_id':checkpoint['id'],'record_video':False})
            self.assertEqual(response.status_code,201,response.json)
            command = start.call_args.args[1]
            slurm = json.loads(base64.urlsafe_b64decode(command[command.index('--command-json')+1]))
            batch = json.loads(base64.urlsafe_b64decode(slurm[slurm.index('--commands-json')+1]))
            self.assertIn('DashboardEvalRunner',batch[0][2])
            self.assertIn('rollout.model=${actor.model}',batch[0][2])

    def test_eval_wrapper_places_hydra_flags_before_overrides(self):
        args = mock.Mock(entrypoint="eval.py", config_path="/configs", config_name="eval",
                         checkpoint="weights.pt", output="/eval", episodes=5, seed=43, max_steps=512, record_video=False)
        argv = hydra_argv(args, ["actor.model.model_path=/models", "--cfg", "job", "--resolve"])
        self.assertLess(argv.index("--cfg"), argv.index("actor.model.model_path=/models"))
        self.assertLess(argv.index("--resolve"), argv.index("rollout.model=${actor.model}"))


if __name__ == "__main__":
    unittest.main()
