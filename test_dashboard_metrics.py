from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from server_4090.app import (
    action_contract_command_args,
    action_contract_for_model,
    build_environment,
    cuda_visible_devices,
    describe_dataset_schema,
    dataset_origin_info,
    gpu_memory_shortfalls,
    gpu_inventory,
    infer_model_variant,
    parse_training_metrics,
    complete_checkpoint_steps,
    merge_eval_metrics,
    select_idle_eval_gpu,
    policy_config_name,
    training_checkpoint_identity,
    training_experiment_catalog,
    MIN_POLICY_ACTION_HORIZON,
    PolicyTelemetryStore,
    UploadManager,
    policy_horizon_status,
    require_policy_execution_horizon,
    complete_action_contract_fingerprint,
    norm_extended_contract_matches,
    policy_time_contract_status,
    require_policy_execution_time_contract,
)
from server_4090.slurm_job_runner import slurm_output_paths


class DatasetOriginClassificationTest(unittest.TestCase):
    def test_marker_overrides_heuristics_and_legacy_shapes_are_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real_capture"
            sim = root / "simulation_capture"
            for path in (real, sim):
                (path / "meta").mkdir(parents=True)

            self.assertEqual(
                dataset_origin_info("real_capture", real, {"robot_type": "piper"})["dataset_origin"],
                "real",
            )
            self.assertEqual(
                dataset_origin_info("simulation_capture", sim, {"robot_type": "aloha"})["dataset_origin"],
                "simulation",
            )
            marker = sim / "meta" / "dashboard_dataset_origin.json"
            marker.write_text(json.dumps({"origin": "real", "source": "test"}), encoding="utf-8")
            marked = dataset_origin_info("simulation_capture", sim, {"robot_type": "aloha"})
            self.assertEqual(marked["dataset_origin"], "real")
            self.assertEqual(marked["dataset_origin_source"], "marker")


class UploadOriginDirectoryTest(unittest.TestCase):
    def test_upload_ids_and_staging_directories_are_separated_by_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = UploadManager(
                {
                    "workspace_root": str(root / "workspace"),
                    "dataset_root": str(root / "datasets"),
                    "max_upload_gib": 1,
                    "max_chunk_mib": 64,
                },
                mock.Mock(),
            )
            common = {
                "dataset_name": "sample",
                "size": 1024 * 1024,
                "sha256": "a" * 64,
                "chunk_size": 1024 * 1024,
                "overwrite": False,
                "merge": False,
            }

            real = manager.initialize({**common, "dataset_origin": "real"})
            simulation = manager.initialize(
                {**common, "dataset_origin": "simulation"}
            )

            self.assertTrue(real["id"].startswith("real-"))
            self.assertTrue(simulation["id"].startswith("simulation-"))
            self.assertNotEqual(real["id"], simulation["id"])
            self.assertTrue(
                (root / "workspace" / "uploads" / "real" / real["id"]).is_dir()
            )
            self.assertTrue(
                (
                    root
                    / "workspace"
                    / "uploads"
                    / "simulation"
                    / simulation["id"]
                ).is_dir()
            )


class TrainingExperimentCatalogTest(unittest.TestCase):
    def test_identifies_standard_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            checkpoint = (
                root
                / policy_config_name("single", "pi05")
                / "pick_cube_r1"
                / "5000"
            )
            identity = training_checkpoint_identity(checkpoint, root)

            self.assertEqual(identity, {
                "config_name": "pi05_piper_single_arm_lora",
                "experiment": "pick_cube_r1",
                "checkpoint_step": 5000,
                "model_variant": "pi05",
                "arm_mode": "single",
            })
            self.assertIsNone(
                training_checkpoint_identity(root / "custom" / "5000", root)
            )

    def test_catalog_includes_incomplete_experiment_and_merges_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            single = root / policy_config_name("single", "pi05") / "shared_exp"
            complete = single / "5000"
            (complete / "params").mkdir(parents=True)
            (complete / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
            (complete / "params" / "_METADATA").write_text("{}", encoding="utf-8")
            (root / policy_config_name("bimanual", "pi0") / "shared_exp").mkdir(
                parents=True
            )
            (root / policy_config_name("single", "pi05") / "new_no_step").mkdir(
                parents=True
            )

            catalog = {item["name"]: item for item in training_experiment_catalog(root)}

            self.assertEqual(set(catalog), {"shared_exp", "new_no_step"})
            self.assertEqual(catalog["shared_exp"]["model_variants"], ["pi0", "pi05"])
            self.assertEqual(catalog["shared_exp"]["arm_modes"], ["bimanual", "single"])
            self.assertEqual(catalog["shared_exp"]["checkpoint_count"], 1)
            self.assertEqual(catalog["shared_exp"]["latest_step"], 5000)
            self.assertEqual(catalog["new_no_step"]["checkpoint_count"], 0)
            self.assertIsNone(catalog["new_no_step"]["latest_step"])


class TrainingMetricsParserTest(unittest.TestCase):
    def test_parses_carriage_returns_ansi_and_metric_summary(self):
        log = (
            "\x1b[32mStep 100: grad_norm=0.1, loss=0.04, param_norm=1800\x1b[0m\r"
            "Step 200: grad_norm=8.0e-2, loss=0.03, "
            "loss_physical_14d=0.06, loss_padding_18d=0.0002\r"
        )

        result = parse_training_metrics(log)

        self.assertEqual([point["step"] for point in result["points"]], [100, 200])
        self.assertEqual(result["series"], [
            "grad_norm", "loss", "loss_padding_18d", "loss_physical_14d", "param_norm",
        ])
        self.assertEqual(result["summary"]["loss"], {"latest": 0.03, "min": 0.03, "max": 0.04})
        self.assertEqual(result["total_points"], 2)
        self.assertEqual(result["sampled_points"], 2)

    def test_later_duplicate_step_wins_and_sampling_keeps_endpoints(self):
        lines = [f"Step {step}: loss={step / 1000:.3f}" for step in range(100)]
        lines.extend(["Step 50: loss=9.5", "not a metric", "Step 101: loss=nan"])

        result = parse_training_metrics("\n".join(lines), max_points=10)

        self.assertEqual(result["total_points"], 100)
        self.assertEqual(result["sampled_points"], 10)
        self.assertEqual(result["points"][0]["step"], 0)
        self.assertEqual(result["points"][-1]["step"], 99)
        self.assertEqual(result["summary"]["loss"]["max"], 9.5)

    def test_slurm_log_stream_markers_do_not_block_metric_parsing(self):
        log = (
            "[dashboard] slurm_job_id=12345\n"
            "[dashboard] slurm log stream stdout: /logs/train_12345.out bytes=0:40/40\n"
            "Step 100: loss=0.42, grad_norm=1.5\n"
        )

        result = parse_training_metrics(log)

        self.assertEqual(result["points"], [{"step": 100, "loss": 0.42, "grad_norm": 1.5}])

    def test_slurm_runner_predicts_sbatch_output_paths(self):
        paths = slurm_output_paths(
            {"workdir": "/work/pi05", "log_dir": "/logs/dashboard"},
            "sim_train_pick_cube",
            "12345",
        )

        self.assertEqual(paths["stdout"], "/logs/dashboard/sim_train_pick_cube_12345.out")
        self.assertEqual(paths["stderr"], "/logs/dashboard/sim_train_pick_cube_12345.err")


class EvalMetricsTest(unittest.TestCase):
    def test_complete_checkpoint_scan_requires_orbax_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "5000"
            (complete / "params").mkdir(parents=True)
            (complete / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
            (complete / "params" / "_METADATA").write_text("{}", encoding="utf-8")
            incomplete = root / "10000"
            (incomplete / "params").mkdir(parents=True)
            (incomplete / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
            (root / "5000.orbax-checkpoint-tmp-1").mkdir()

            self.assertEqual(complete_checkpoint_steps(root), [(5000, complete.resolve())])

    def test_idle_eval_gpu_excludes_managed_external_and_unavailable(self):
        tasks = [{
            "type": "train",
            "state": "starting",
            "metadata": {"gpu_ids": [0]},
        }]
        inventory = [
            {"index": 0, "memory_total_mib": 24564, "memory_used_mib": 0, "processes": [], "compute_available": True},
            {"index": 1, "memory_total_mib": 24564, "memory_used_mib": 10, "processes": [{"pid": 1}], "compute_available": True},
            {"index": 2, "memory_total_mib": 24564, "memory_used_mib": 10, "processes": [], "compute_available": False},
            {"index": 3, "memory_total_mib": 24564, "memory_used_mib": 100, "processes": [], "compute_available": True},
        ]
        self.assertEqual(
            select_idle_eval_gpu(tasks, inventory, allowed_gpu_ids={0, 1, 2, 3}, minimum_free_mib=23000),
            3,
        )

    def test_merges_eval_result_at_checkpoint_step_and_summarizes_best(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "eval5000.json"
            second = Path(directory) / "eval10000.json"
            first.write_text(json.dumps({
                "checkpoint_step": 5000,
                "eval_loss_model": 0.3,
                "eval_loss_objective": 0.2,
            }), encoding="utf-8")
            second.write_text(json.dumps({
                "checkpoint_step": 10000,
                "eval_loss_model": 0.1,
                "eval_loss_objective": 0.15,
            }), encoding="utf-8")
            metrics = parse_training_metrics("Step 0: loss=1.0\nStep 10000: loss=0.2")
            tasks = [
                {"id": "eval-1", "type": "eval", "state": "completed", "metadata": {"checkpoint_step": 5000, "result_path": str(first), "gpu_ids": [0]}},
                {"id": "eval-2", "type": "eval", "state": "completed", "metadata": {"checkpoint_step": 10000, "result_path": str(second), "gpu_ids": [3]}},
                {"id": "eval-3", "type": "eval", "state": "skipped", "skip_reason": "no_idle_gpu", "metadata": {"checkpoint_step": 15000}},
            ]

            result = merge_eval_metrics(metrics, tasks)

            self.assertEqual([point["step"] for point in result["eval_points"]], [5000, 10000])
            self.assertEqual(result["eval_summary"]["best_step"], 10000)
            self.assertEqual(result["eval_summary"]["best_loss"], 0.1)
            self.assertEqual(result["eval_summary"]["counts"], {"completed": 2, "skipped": 1})
            self.assertIn("eval_loss_model", result["series"])
            self.assertEqual(result["summary"]["loss"]["min"], 0.2)


class GpuInventoryTest(unittest.TestCase):
    @mock.patch("server_4090.app.subprocess.check_output")
    def test_ignores_nvidia_smi_na_process_rows(self, check_output):
        check_output.side_effect = [
            "0, GPU-0, NVIDIA RTX 4090, 24564, 30\n",
            "GPU-0, [N/A], [N/A], [N/A]\n",
        ]

        self.assertEqual(
            gpu_inventory(),
            [{
                "index": 0,
                "uuid": "GPU-0",
                "name": "NVIDIA RTX 4090",
                "memory_total_mib": 24564,
                "memory_used_mib": 30,
                "processes": [],
                "compute_available": False,
                "health_issue": "nvidia-smi reports an unavailable compute context ([N/A])",
            }],
        )

    def test_cuda_visibility_uses_physical_gpu_uuids(self):
        inventory = [
            {"index": 0, "uuid": "GPU-zero"},
            {"index": 3, "uuid": "GPU-three"},
        ]
        self.assertEqual(cuda_visible_devices([0, 3], inventory), "GPU-zero,GPU-three")

    @mock.patch("server_4090.app.cuda_visible_devices", return_value="GPU-zero,GPU-two")
    def test_training_environment_uses_stable_gpu_order_and_memory_fraction(self, _visible):
        env = build_environment(
            {
                "openpi_python": "/opt/conda/envs/openpi/bin/python",
                "dataset_root": "/datasets",
                "xla_memory_fraction": 0.90,
            },
            [0, 2],
            xla_memory_fraction=0.88,
        )
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-zero,GPU-two")
        self.assertEqual(env["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.88")

    def test_gpu_memory_shortfall_uses_total_minus_used(self):
        inventory = {
            0: {"memory_total_mib": 24564, "memory_used_mib": 400},
            2: {"memory_total_mib": 24564, "memory_used_mib": 2200},
        }
        self.assertEqual(
            gpu_memory_shortfalls(inventory, [0, 2], 23_000),
            {2: {"free_mib": 22_364, "required_mib": 23_000}},
        )


class ModelVariantTest(unittest.TestCase):
    def test_infers_nearest_checkpoint_variant(self):
        # The checkout itself may be below a parent directory named ``pi05``;
        # the nearer checkpoint config must win.
        nested_pi0 = Path(
            "/srv/policy/pi05/checkpoints/pi0_piper_single_arm_lora/from_pi05_transfer/1000"
        )
        self.assertEqual(infer_model_variant(nested_pi0), "pi0")
        self.assertEqual(
            infer_model_variant(Path("/models/pi05_piper_bimanual_lora/demo/1000")),
            "pi05",
        )
        self.assertEqual(infer_model_variant(Path("/models/pi0.5_base")), "pi05")
        self.assertIsNone(infer_model_variant(Path("/models/custom_checkpoint")))

    def test_config_names_are_separated_by_model_and_arm_mode(self):
        self.assertEqual(policy_config_name("single", "pi05"), "pi05_piper_single_arm_lora")
        self.assertEqual(policy_config_name("bimanual", "pi0"), "pi0_piper_bimanual_lora")
        with self.assertRaisesRegex(ValueError, "unsupported model_variant"):
            policy_config_name("single", "pi0_fast")


class DatasetSchemaDescriptionTest(unittest.TestCase):
    @staticmethod
    def info(
        state_key,
        state_dim,
        action_key,
        action_dim,
        cameras,
        *,
        contract_version=None,
        gripper_semantics=None,
        raw_action_semantics=None,
        legacy_format=None,
        action_names=None,
        state_names=None,
    ):
        features = {
            state_key: {
                "dtype": "float32",
                "shape": [state_dim],
                **({"names": state_names} if state_names else {}),
            },
            action_key: {
                "dtype": "float32",
                "shape": [action_dim],
                **({"names": action_names} if action_names else {}),
            },
        }
        features.update({key: {"dtype": media_type} for key, media_type in cameras})
        info = {"features": features}
        if contract_version is not None:
            info["contract_version"] = contract_version
        if gripper_semantics is not None:
            info["gripper_semantics"] = gripper_semantics
        if raw_action_semantics is not None:
            info["raw_action_semantics"] = raw_action_semantics
        if legacy_format is not None:
            info["legacy_format"] = legacy_format
        return info

    def test_joint_v3_and_legacy_v2_units_are_visible(self):
        cases = [
            (
                self.info(
                    "observation.state", 7, "action", 7,
                    [("observation.images.cam_high", "video"), ("observation.images.cam_right_wrist", "video")],
                    contract_version=3,
                    gripper_semantics="absolute_opening_fraction_0_closed_1_open",
                    state_names=["joint_1_rad", "joint_2_rad", "joint_3_rad", "joint_4_rad", "joint_5_rad", "joint_6_rad", "gripper_opening_fraction"],
                    action_names=["joint_1_rad", "joint_2_rad", "joint_3_rad", "joint_4_rad", "joint_5_rad", "joint_6_rad", "gripper_opening_fraction"],
                ),
                "v3", "absolute_opening_fraction_0_closed_1_open",
            ),
            (
                self.info(
                    "observation.state", 7, "action", 7,
                    [("observation.images.cam_high", "video"), ("observation.images.cam_right_wrist", "video")],
                    contract_version=2,
                    gripper_semantics="absolute_opening_metres",
                ),
                "legacy v2", "absolute_opening_fraction_0_closed_1_open",
            ),
        ]
        for info, version_label, model_gripper in cases:
            with self.subTest(version_label=version_label):
                result = describe_dataset_schema(info)
                self.assertTrue(result["training_supported"])
                self.assertIn(version_label, result["schema_label"])
                self.assertEqual(result["raw_action_dim"], 7)
                self.assertEqual(result["model_action_dim"], 7)
                self.assertEqual(result["model_gripper_semantics"], model_gripper)
                self.assertEqual(result["wire_gripper_semantics"], model_gripper)

    def test_legacy_delivery_10d7d_is_step_delta_and_new_10d10d_is_absolute_raw(self):
        legacy = describe_dataset_schema(
            self.info(
                "state", 10, "actions", 7,
                [("image", "image"), ("wrist_image", "image")],
            )
        )
        self.assertTrue(legacy["training_supported"])
        self.assertTrue(legacy["model_contract_supported"])
        self.assertIsNone(legacy["training_error"])
        self.assertEqual(legacy["contract_version"], 2)
        self.assertTrue(legacy["legacy_delivery_v2"])
        self.assertEqual(legacy["raw_action_convention"], "step")
        self.assertEqual(legacy["raw_action_dim"], 7)
        self.assertEqual(legacy["model_action_dim"], 7)
        self.assertEqual(legacy["raw_gripper_semantics"], "absolute_closed_fraction_0_open_1_closed")
        self.assertEqual(legacy["action_offset"], 1)
        self.assertEqual(legacy["model_action_start_offset"], 1)

        absolute = describe_dataset_schema(
            self.info(
                "observation.state", 10, "action", 10,
                [
                    ("observation.images.cam_high", "video"),
                    ("observation.images.cam_right_wrist", "video"),
                ],
                contract_version=3,
                gripper_semantics="absolute_opening_fraction_0_closed_1_open",
            )
        )
        self.assertTrue(absolute["training_supported"])
        self.assertEqual(absolute["contract_version"], 3)
        self.assertFalse(absolute["legacy_delivery_v2"])
        self.assertEqual(absolute["raw_action_convention"], "absolute_eef_target")
        self.assertEqual(absolute["raw_action_dim"], 10)
        self.assertEqual(absolute["model_action_dim"], 7)
        self.assertEqual(absolute["model_action_semantics"], "eef_delta_chunk_origin_base_xyz_left_rotvec_gripper_opening_target")
        self.assertEqual(absolute["action_offset"], 0)
        self.assertEqual(absolute["model_action_start_offset"], 1)

    def test_canonical_10d7d_requires_explicit_legacy_metadata(self):
        result = describe_dataset_schema(
            self.info(
                "observation.state", 10, "action", 7,
                [("observation.images.cam_high", "video"), ("observation.images.cam_right_wrist", "video")],
            )
        )
        self.assertFalse(result["training_supported"])
        self.assertIn("explicit legacy_v2/step", result["contract_error"])

    def test_bimanual_raw_and_model_dimensions(self):
        result = describe_dataset_schema(
            self.info(
                "observation.state", 20, "action", 20,
                [
                    ("observation.images.cam_high", "video"),
                    ("observation.images.cam_left_wrist", "video"),
                    ("observation.images.cam_right_wrist", "video"),
                ],
                contract_version=3,
                gripper_semantics="absolute_opening_fraction_0_closed_1_open",
            )
        )
        self.assertTrue(result["training_supported"])
        self.assertEqual(result["arm_mode"], "bimanual")
        self.assertEqual(result["raw_action_dim"], 20)
        self.assertEqual(result["model_action_dim"], 14)

    def test_model_contract_switches_legacy_delivery_and_joint_explicitly(self):
        legacy_delivery = describe_dataset_schema(
            self.info("state", 10, "actions", 7, [("image", "image"), ("wrist_image", "image")])
        )
        step = action_contract_for_model(
            legacy_delivery, delivery_action_convention="step"
        )
        chunk = action_contract_for_model(
            legacy_delivery, delivery_action_convention="chunk_origin"
        )
        self.assertEqual(step["model_action_semantics"], step["raw_action_semantics"])
        self.assertNotEqual(chunk["model_action_semantics"], chunk["raw_action_semantics"])
        self.assertIn("--raw-action-dim", action_contract_command_args(chunk))
        command_args = action_contract_command_args(chunk)
        self.assertIn("--model-action-dim", command_args)
        self.assertEqual(command_args[command_args.index("--action-offset") + 1], "1")
        self.assertEqual(command_args[command_args.index("--model-action-start-offset") + 1], "1")

        legacy_joint = describe_dataset_schema(
            self.info(
                "observation.state", 7, "action", 7,
                [("observation.images.cam_high", "video"), ("observation.images.cam_right_wrist", "video")],
                contract_version=2,
                gripper_semantics="absolute_opening_metres",
            )
        )
        old_policy = action_contract_for_model(
            legacy_joint, model_gripper_semantics="absolute_opening_metres"
        )
        new_policy = action_contract_for_model(legacy_joint)
        self.assertEqual(old_policy["wire_gripper_semantics"], "absolute_opening_metres")
        self.assertEqual(
            new_policy["wire_gripper_semantics"],
            "absolute_opening_fraction_0_closed_1_open",
        )

    def test_unknown_dimensions_remain_visible_as_custom(self):
        result = describe_dataset_schema(
            self.info("observation.state", 12, "action", 8, [("custom_camera", "image")])
        )
        self.assertEqual(result["schema"], "custom")
        self.assertEqual(result["schema_label"], "通用格式 12D/8D")
        self.assertEqual(result["cameras"], ["custom_camera"])
        self.assertFalse(result["training_supported"])


    def test_simulation_joint_without_contract_defaults_to_v3_but_real_stays_closed(self):
        base = self.info(
            "observation.state", 14, "action", 14,
            [
                ("observation.images.cam_high", "video"),
                ("observation.images.cam_left_wrist", "video"),
                ("observation.images.cam_right_wrist", "video"),
            ],
        )
        real = describe_dataset_schema(base)
        self.assertFalse(real["training_supported"])
        self.assertIn("requires contract_version", real["contract_error"])

        sim = describe_dataset_schema({**base, "dataset_origin": "simulation"})
        self.assertTrue(sim["training_supported"])
        self.assertEqual(sim["contract_version"], 3)
        self.assertEqual(sim["raw_gripper_semantics"], "absolute_opening_fraction_0_closed_1_open")
        self.assertEqual(sim["model_action_dim"], 14)

class AsyncPolicyDashboardContractTest(unittest.TestCase):
    def test_horizon_gate_is_fail_closed(self):
        ready = policy_horizon_status({"action_horizon": 50, "client_action_horizon": 50})
        self.assertEqual(ready["minimum_horizon"], MIN_POLICY_ACTION_HORIZON)
        self.assertTrue(ready["horizon_execution_ready"])
        require_policy_execution_horizon(ready)

        short = policy_horizon_status({"action_horizon": 15})
        self.assertFalse(short["horizon_execution_ready"])
        self.assertIn("below", short["horizon_error"])
        with self.assertRaisesRegex(ValueError, "action_horizon >= 16"):
            require_policy_execution_horizon(short)

        missing = policy_horizon_status({})
        self.assertFalse(missing["horizon_execution_ready"])
        with self.assertRaisesRegex(ValueError, "missing"):
            require_policy_execution_horizon(missing)

        mismatch = policy_horizon_status({"action_horizon": 50, "client_action_horizon": 49})
        self.assertFalse(mismatch["horizon_execution_ready"])
        self.assertFalse(mismatch["horizon_contract_match"])

    def test_temporal_contract_and_norm_fingerprint_are_fail_closed(self):
        ready = {
            "action_hz": 20,
            "action_time_step_s": 0.05,
            "action_offset": 0,
            "model_action_start_offset": 1,
            "action_start_offset_steps": 1,
        }
        status = policy_time_contract_status(ready)
        self.assertTrue(status["time_contract_ready"])
        require_policy_execution_time_contract(ready)

        for patch in (
            {"action_time_step_s": 0.04},
            {"action_offset": 2},
            {"model_action_start_offset": 0},
            {"action_start_offset_steps": 0},
        ):
            invalid = {**ready, **patch}
            self.assertFalse(policy_time_contract_status(invalid)["time_contract_ready"])
            with self.assertRaises(ValueError):
                require_policy_execution_time_contract(invalid)

        representation = {
            "contract_version": 3,
            "raw_action_dim": 10,
            "model_action_dim": 7,
            "raw_action_semantics": "absolute_eef_target",
            "model_action_semantics": "eef_delta",
            "raw_action_convention": "absolute_eef_target",
            "model_action_convention": "chunk_origin",
            "gripper_semantics": "absolute_opening_fraction_0_closed_1_open",
            "raw_gripper_semantics": "absolute_opening_fraction_0_closed_1_open",
            "wire_gripper_semantics": "absolute_opening_fraction_0_closed_1_open",
            "action_offset": 0,
            "model_action_start_offset": 1,
        }
        fingerprint = complete_action_contract_fingerprint(representation)
        self.assertTrue(norm_extended_contract_matches({"version": 4, **fingerprint}, fingerprint))
        self.assertFalse(norm_extended_contract_matches({"version": 4, **fingerprint, "action_offset": 1}, fingerprint))

    def test_summary_dual_gate_requires_valid_action_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PolicyTelemetryStore({"workspace_root": tmp, "robot_observation_max_age_s": 3})
            session, directory = store.create_session()
            task = {
                "id": "policy-test",
                "type": "policy",
                "state": "running",
                "metadata": {"telemetry_session": session, "port": 8000},
            }
            store.set_control(task, mode="execute", expires_in_s=300)
            (directory / "connections.json").write_text(
                json.dumps({"client_connected": True, "active_clients": 1}), encoding="utf-8"
            )

            def summary(horizon):
                payload = {
                    "received_at": time.time(),
                    "client_allow_execution": True,
                    "client_execution_state": "execute",
                    "action_hz": 20.0,
                    "action_time_step_s": 0.05,
                    "action_offset": 0,
                    "model_action_start_offset": 1,
                    "action_start_offset_steps": 1,
                }
                if horizon is not None:
                    payload["action_horizon"] = horizon
                (directory / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
                return store.summary_for_task(task)

            (directory / "runtime.json").write_text(
                json.dumps({"in_flight": True, "active_inferences": 1}), encoding="utf-8"
            )
            active = summary(50)
            self.assertTrue(active["dual_gate_open"])
            self.assertTrue(active["client_in_flight"])
            self.assertTrue(active["policy_in_flight"])
            (directory / "runtime.json").write_text(
                json.dumps({"in_flight": False, "active_inferences": 0}), encoding="utf-8"
            )
            self.assertFalse(summary(15)["dual_gate_open"])
            self.assertFalse(summary(None)["dual_gate_open"])

    def test_summary_extrapolates_signed_target_error_and_falls_back_to_legacy_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PolicyTelemetryStore({"workspace_root": tmp, "robot_observation_max_age_s": 3})
            session, directory = store.create_session()
            task = {
                "id": "policy-time",
                "type": "policy",
                "state": "running",
                "metadata": {"telemetry_session": session, "port": 8000},
            }
            store.set_control(task, mode="shadow")
            (directory / "connections.json").write_text(
                json.dumps({"client_connected": True, "active_clients": 1}), encoding="utf-8"
            )
            (directory / "runtime.json").write_text(
                json.dumps({"in_flight": False, "active_inferences": 0}), encoding="utf-8"
            )
            payload = {
                "received_at": 100.0,
                "client_allow_execution": False,
                "client_execution_state": "shadow",
                "client_target_time_error_ms": -40.0,
                "client_timing_snapshot_at": 100.0,
                "action_horizon": 16,
                "action_hz": 20.0,
                "action_time_step_s": 0.05,
                "action_offset": 0,
                "model_action_start_offset": 1,
                "action_start_offset_steps": 1,
            }
            (directory / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch("server_4090.app.time.time", return_value=100.2):
                summary = store.summary_for_task(task)
            self.assertIsNotNone(summary)
            self.assertAlmostEqual(summary["client_current_target_age_s"], 0.16, places=3)
            self.assertAlmostEqual(summary["client_current_target_time_error_ms"], 160.0, places=1)

            # A negative target error must survive when no snapshot timestamp is
            # available; it means the active target is still in the future.
            payload.pop("client_target_time_error_ms")
            payload.pop("client_timing_snapshot_at")
            payload["client_timed_target"] = {"target_age_s": -0.04}
            (directory / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch("server_4090.app.time.time", return_value=100.2):
                summary = store.summary_for_task(task)
            self.assertAlmostEqual(summary["client_current_target_time_error_ms"], -40.0, places=1)

    def test_dashboard_template_and_readme_publish_full_async_contract(self):
        template = (Path(__file__).parent / "server_4090/templates/index.html").read_text(encoding="utf-8")
        readme = (Path(__file__).parent / "server_4090/README.md").read_text(encoding="utf-8")
        for marker in (
            "client_inference_launch_hz",
            "client_inference_result_hz",
            "client_configured_inference_hz",
            "client_inference_single_inflight_ceiling_hz",
            "client_control_hz",
            "client_chunk_rows",
            "minimum_horizon",
            "client_in_flight",
            "client_launch_at",
            "client_capture_at",
            "client_arrival_at",
            "client_latency_ms",
            "client_latency_steps",
            "client_skipped_prefix",
            "client_blend_steps",
            "client_queue_generation",
            "client_old_remaining",
            "client_new_remaining",
            "client_underrun",
            "client_rejected_result",
            "client_drop_reason",
            "client_last_wire_action",
            "client_last_decoded_target",
            "client_minimum_horizon",
            "client_result_generation",
            "client_safety_profile",
            "client_delivery_safety_limits",
            "client_actuator_delay_ms",
            "client_expired_prefix",
            "client_active_plan_generation",
            "client_plan_target_times",
            "client_hold_active",
            "client_blend_active",
            "client_gripper_filter",
            "client_timed_target",
            "client_last_safe_target",
            "client_target_monotonic",
            "client_observation_upload_ms",
            "client_model_inference_ms",
            "client_result_download_ms",
            "client_network_transport_total_ms",
            "client_round_trip_ms",
            "client_timing_source",
            "client_one_way_timing_clock",
            "one_way_timing_requires_clock_sync",
            "客户端实测",
            "服务端诊断",
            "client_current_target_time_error_ms",
            "result→first command",
        ):
            self.assertIn(marker, template)
        for marker in ("action_horizon", "action_hz", "action_time_step_s", "action_start_offset_steps", "--hz 4", "4 Hz", "200 ms", "旧 chunk", "动态", "2~4 步", "0.05", "0.18", "0.30", "x[-0.05,0.30]", "y[0.01,0.50]", "z[0.14,0.52]"):
            self.assertIn(marker, template)
        for marker in ("action_horizon", "action_hz", "action_time_step_s", "action_start_offset_steps", "action_offset", "model_action_start_offset", "--hz 4", "4 Hz", "200 ms", "旧 chunk", "动态", "2~4 步", "0.05", "0.18", "0.30"):
            self.assertIn(marker, readme)
        combined = template + "\n" + readme
        for stale in ("0.015", "0.15", "0.25", "--hz 5", "5 Hz", "选择第4行", "选择第 4 行", "只排队4步", "只排队 4 步", "每次排队", "典型约 4 步", "skip≈4"):
            self.assertNotIn(stale, combined)


if __name__ == "__main__":
    unittest.main()
