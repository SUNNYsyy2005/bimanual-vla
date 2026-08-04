from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation


def _load_openpi_helper():
    """Import the helper with lightweight OpenPI stubs for pure CPU unit tests."""
    openpi = types.ModuleType("openpi")
    transforms = types.ModuleType("openpi.transforms")
    transforms.DataTransformFn = object
    models = types.ModuleType("openpi.models")
    pi0_config = types.ModuleType("openpi.models.pi0_config")
    policies = types.ModuleType("openpi.policies")
    policy_config = types.ModuleType("openpi.policies.policy_config")
    serving = types.ModuleType("openpi.serving")
    websocket_policy_server = types.ModuleType("openpi.serving.websocket_policy_server")
    websocket_policy_server.WebsocketPolicyServer = object
    shared = types.ModuleType("openpi.shared")
    normalize = types.ModuleType("openpi.shared.normalize")
    training = types.ModuleType("openpi.training")
    training_config = types.ModuleType("openpi.training.config")
    training_config.DataConfigFactory = object
    data_loader = types.ModuleType("openpi.training.data_loader")
    weight_loaders = types.ModuleType("openpi.training.weight_loaders")

    openpi.transforms = transforms
    openpi.models = models
    openpi.policies = policies
    openpi.serving = serving
    openpi.shared = shared
    openpi.training = training
    models.pi0_config = pi0_config
    policies.policy_config = policy_config
    serving.websocket_policy_server = websocket_policy_server
    shared.normalize = normalize
    training.config = training_config
    training.data_loader = data_loader
    training.weight_loaders = weight_loaders

    stubs = {
        "openpi": openpi,
        "openpi.transforms": transforms,
        "openpi.models": models,
        "openpi.models.pi0_config": pi0_config,
        "openpi.policies": policies,
        "openpi.policies.policy_config": policy_config,
        "openpi.serving": serving,
        "openpi.serving.websocket_policy_server": websocket_policy_server,
        "openpi.shared": shared,
        "openpi.shared.normalize": normalize,
        "openpi.training": training,
        "openpi.training.config": training_config,
        "openpi.training.data_loader": data_loader,
        "openpi.training.weight_loaders": weight_loaders,
    }
    module_name = "server_4090._openpi_single_arm_contract_test_target"
    path = Path(__file__).with_name("openpi_single_arm.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {**stubs, module_name: module}):
        spec.loader.exec_module(module)
    return module


HELPER = _load_openpi_helper()


def _rotation6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)


class OpenPiActionTransformTest(unittest.TestCase):
    def test_absolute_eef_chunk_uses_one_current_anchor_for_all_rows(self):
        current = np.array(
            [0.4, -0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.25], dtype=np.float32
        )
        first_rotation = Rotation.from_euler("z", 0.1).as_matrix()
        second_rotation = Rotation.from_euler("z", 0.2).as_matrix()
        targets = np.stack(
            [
                np.concatenate(([0.5, -0.2, 0.3], _rotation6d(first_rotation), [0.4])),
                np.concatenate(([0.7, -0.1, 0.2], _rotation6d(second_rotation), [0.8])),
            ]
        ).astype(np.float32)

        result = HELPER.DeliveryAbsoluteEEFToChunkOrigin(arm_count=1)(
            {"state": current, "actions": targets}
        )["actions"]

        np.testing.assert_allclose(result[:, :3], [[0.1, 0.0, 0.0], [0.3, 0.1, -0.1]], atol=1e-6)
        np.testing.assert_allclose(result[:, 3:6], [[0, 0, 0.1], [0, 0, 0.2]], atol=1e-6)
        np.testing.assert_allclose(result[:, 6], [0.4, 0.8], atol=1e-7)

    def test_legacy_joint_metres_convert_before_norm(self):
        transform = HELPER.JointGripperMetersToFraction(arm_count=2)
        state = np.zeros(14, dtype=np.float32)
        state[[6, 13]] = [0.035, 0.07]
        actions = np.zeros((2, 14), dtype=np.float32)
        actions[:, 6] = [0.0, 0.07]
        actions[:, 13] = [0.014, 0.035]

        result = transform({"state": state, "actions": actions})

        np.testing.assert_allclose(result["state"][[6, 13]], [0.5, 1.0])
        np.testing.assert_allclose(result["actions"][:, 6], [0.0, 1.0])
        np.testing.assert_allclose(result["actions"][:, 13], [0.2, 0.5])


class AsyncTelemetrySanitizerTest(unittest.TestCase):
    def test_valid_async_fields_are_sanitized_and_preserved(self):
        telemetry = HELPER.sanitize_async_client_telemetry(
            {
                "inference_launch_hz": 4.02,
                "control_hz": 20,
                "chunk_rows": 50,
                "action_horizon": 50,
                "in_flight": True,
                "launch_at": 100.0,
                "capture_at": 100.01,
                "arrival_at": 100.21,
                "latency_ms": 200.0,
                "latency_steps": 4,
                "actuator_delay_s": 0.05,
                "expired_prefix_steps": 5,
                "skipped_prefix_steps": 5,
                "blend_steps": 3,
                "queue_generation": 9,
                "active_plan_generation": 9,
                "active_plan_index": 5,
                "active_plan_remaining": 42,
                "plan_target_times": [101.0, 101.05, 101.10],
                "hold_active": True,
                "hold_steps": 1,
                "hold_reason": "awaiting fresh plan",
                "blend_active": True,
                "blend_steps_remaining": 2,
                "gripper_filter_active": True,
                "gripper_filter": {"mode": "ema", "alpha": 0.25},
                "old_remaining": 18,
                "new_remaining": 43,
                "underrun": False,
                "underrun_count": 2,
                "rejected_result": True,
                "rejected_result_count": 3,
                "drop_reason": "stale generation",
                "last_wire_action": [0, 1, 2, 3, 4, 5, 0.6],
                "last_decoded_target": {"eef_xyz_m": [0.1, 0.2, 0.3], "opening_fraction": 0.6},
                "command_sequence": 13,
                "last_actuator_command": {
                    "command_sequence": 13,
                    "generation": 9,
                    "source_index": 5,
                    "queue_index": 5,
                    "sides": {"right": {"commanded_joints_rad": [0, 1, -1, 0, 0, 0]}},
                },
                "last_command_feedback": {
                    "command_sequence": 12,
                    "generation": 9,
                    "source_index": 4,
                    "queue_index": 4,
                    "command_to_feedback_ms": 50.0,
                    "max_joint_abs_error_rad": 0.012,
                    "max_gripper_abs_error_m": 0.001,
                    "max_eef_translation_error_m": 0.004,
                    "max_eef_rotation_error_rad": 0.02,
                },
            },
            action_dim=7,
            action_horizon=50,
        )

        self.assertAlmostEqual(telemetry["client_inference_launch_hz"], 4.02)
        self.assertEqual(telemetry["client_control_hz"], 20.0)
        self.assertEqual(telemetry["client_chunk_rows"], 50)
        self.assertEqual(telemetry["client_action_horizon"], 50)
        self.assertTrue(telemetry["client_horizon_matches_policy"])
        self.assertTrue(telemetry["client_in_flight"])
        self.assertEqual(telemetry["client_latency_steps"], 4.0)
        self.assertEqual(telemetry["client_actuator_delay_ms"], 50.0)
        self.assertEqual(telemetry["client_expired_prefix"], 5)
        self.assertEqual(telemetry["client_skipped_prefix"], 5)
        self.assertEqual(telemetry["client_blend_steps"], 3)
        self.assertEqual(telemetry["client_active_plan_generation"], 9)
        self.assertEqual(telemetry["client_active_plan_index"], 5)
        self.assertEqual(telemetry["client_plan_target_times"], [101.0, 101.05, 101.1])
        self.assertTrue(telemetry["client_hold_active"])
        self.assertTrue(telemetry["client_blend_active"])
        self.assertEqual(telemetry["client_gripper_filter"]["mode"], "ema")
        self.assertEqual((telemetry["client_old_remaining"], telemetry["client_new_remaining"]), (18, 43))
        self.assertFalse(telemetry["client_underrun"])
        self.assertTrue(telemetry["client_rejected_result"])
        self.assertEqual(telemetry["client_drop_reason"], "stale generation")
        self.assertEqual(len(telemetry["client_last_wire_action"]), 7)
        self.assertEqual(telemetry["client_last_decoded_target"]["opening_fraction"], 0.6)
        self.assertEqual(telemetry["client_command_sequence"], 13)
        self.assertEqual(telemetry["client_actuator_command_sequence"], 13)
        self.assertEqual(telemetry["client_actuator_command_generation"], 9)
        self.assertEqual(telemetry["client_actuator_command_source_index"], 5)
        self.assertEqual(telemetry["client_actuator_command_queue_index"], 5)
        self.assertEqual(telemetry["client_feedback_command_sequence"], 12)
        self.assertEqual(telemetry["client_feedback_command_generation"], 9)
        self.assertEqual(telemetry["client_feedback_command_source_index"], 4)
        self.assertEqual(telemetry["client_feedback_command_queue_index"], 4)
        self.assertEqual(telemetry["client_command_to_feedback_ms"], 50.0)
        self.assertEqual(telemetry["client_command_max_joint_abs_error_rad"], 0.012)

    def test_invalid_async_values_fail_closed_without_poisoning_json(self):
        telemetry = HELPER.sanitize_async_client_telemetry(
            {
                "inference_launch_hz": float("nan"),
                "control_hz": -20,
                "chunk_rows": 0,
                "action_horizon": 15,
                "latency_ms": float("inf"),
                "latency_steps": -1,
                "skipped_prefix": -2,
                "old_remaining": -1,
                "last_wire_action": [1, 2],
                "last_decoded_target": {"bad": float("nan")},
                "drop_reason": "x" * 700,
            },
            action_dim=7,
            action_horizon=50,
        )

        self.assertIsNone(telemetry["client_inference_launch_hz"])
        self.assertIsNone(telemetry["client_control_hz"])
        self.assertIsNone(telemetry["client_chunk_rows"])
        self.assertFalse(telemetry["client_horizon_matches_policy"])
        self.assertIsNone(telemetry["client_latency_ms"])
        self.assertIsNone(telemetry["client_latency_steps"])
        self.assertIsNone(telemetry["client_skipped_prefix"])
        self.assertIsNone(telemetry["client_old_remaining"])
        self.assertIsNone(telemetry["client_last_wire_action"])
        self.assertIsNone(telemetry["client_last_decoded_target"])
        self.assertEqual(len(telemetry["client_drop_reason"]), 500)


    def test_client_telemetry_alias_priority_is_explicit(self):
        telemetry = HELPER.sanitize_async_client_telemetry(
            {
                "round_trip_ms": 222.0,
                "client_observation_upload_ms": 12.0,
                "observation_upload_ms": 22.0,
                "client_result_download_ms": 13.0,
                "result_download_ms": 23.0,
                "client_network_transport_total_ms": 14.0,
                "network_transport_total_ms": 24.0,
                "client_hold_active": False,
                "holding": True,
                "client_timing_source": "canonical-client",
                "timing_source": "legacy-alias",
                "client_one_way_timing_clock": "client-wall",
                "one_way_timing_clock": "legacy-wall",
                "client_one_way_timing_requires_clock_sync": False,
                "one_way_timing_requires_clock_sync": True,
            },
            action_dim=7,
            action_horizon=50,
        )

        # Legacy aliases remain supported, but canonical client fields win.
        self.assertEqual(telemetry["client_round_trip_ms"], 222.0)
        self.assertTrue(telemetry["client_hold_active"])
        self.assertEqual(telemetry["client_observation_upload_ms"], 12.0)
        self.assertEqual(telemetry["client_result_download_ms"], 13.0)
        self.assertEqual(telemetry["client_network_transport_total_ms"], 14.0)
        self.assertEqual(telemetry["client_timing_source"], "canonical-client")
        self.assertEqual(telemetry["client_one_way_timing_clock"], "client-wall")
        self.assertFalse(telemetry["client_one_way_timing_requires_clock_sync"])

    def test_nested_transport_timing_supplies_missing_client_fields_and_generation(self):
        telemetry = HELPER.sanitize_async_client_telemetry(
            {
                "client_transport_timing": {
                    "client_observation_upload_ms": 31.0,
                    "client_result_download_ms": 32.0,
                    "client_network_transport_total_ms": 63.0,
                    "generation": 17,
                },
            },
            action_dim=7,
            action_horizon=50,
        )

        self.assertEqual(telemetry["client_observation_upload_ms"], 31.0)
        self.assertEqual(telemetry["client_result_download_ms"], 32.0)
        self.assertEqual(telemetry["client_network_transport_total_ms"], 63.0)
        self.assertEqual(telemetry["client_timing_generation"], 17)

    def test_clock_sync_metadata_is_preserved_and_non_boolean_values_fail_closed(self):
        telemetry = HELPER.sanitize_async_client_telemetry(
            {
                "timing_source": "client_wall_clock_echo",
                "one_way_timing_clock": "wall_clock",
                "one_way_timing_requires_clock_sync": True,
            },
            action_dim=7,
            action_horizon=50,
        )

        self.assertEqual(telemetry["client_timing_source"], "client_wall_clock_echo")
        self.assertEqual(telemetry["client_one_way_timing_clock"], "wall_clock")
        self.assertTrue(telemetry["client_one_way_timing_requires_clock_sync"])

        invalid = HELPER.sanitize_async_client_telemetry(
            {
                "timing_source": "client_wall_clock_echo",
                "one_way_timing_clock": "wall_clock",
                "one_way_timing_requires_clock_sync": "yes",
            },
            action_dim=7,
            action_horizon=50,
        )
        self.assertIsNone(invalid["client_one_way_timing_requires_clock_sync"])

    def test_current_bridge_queue_fields_map_without_semantic_regression(self):
        telemetry = HELPER.sanitize_async_client_telemetry(
            {
                "inference_hz": 4,
                "control_hz": 20,
                "policy_action_hz": 20,
                "expected_action_horizon": 50,
                "min_action_chunk_steps": 16,
                "inference_launch_at": 12.50,
                "inference_capture_at": 12.49,
                "inference_arrival_at": 12.70,
                "inference_latency_s": 0.20,
                "estimated_actuator_delay_s": 0.05,
                "inference_skip_steps": 4,
                "inference_blend_steps": 4,
                "inference_generation": 8,
                "action_generation": 7,
                "inference_old_remaining": 18,
                "queued_action_count": 46,
                "queue_underrun": False,
                "queue_underrun_count": 2,
                "hold_active": True,
                "hold_count": 3,
                "timed_target": {
                    "target_monotonic": 123.45,
                    "target_age_s": 0.01,
                    "source_generation": 7,
                    "source_index": 4,
                    "queue_index": 4,
                    "blended": True,
                    "blend_step": 2,
                    "hold": False,
                },
                "last_safe_target": {"target_monotonic": 123.40, "target_age_s": 0.06},
                "gripper_filter": {"lowpass_alpha": 0.35, "opening_fraction": {"right": 0.5}},
                "rejected_result": {"generation": 6, "reason": "stale generation"},
                "rejected_result_count": 3,
                "last_wire_action": [0, 1, 2, 3, 4, 5, 0.6],
                "last_decoded_absolute_target": {"eef_xyz_m": [0.1, 0.2, 0.3]},
                "safety_profile": "8_3_64eps_18034_frames_20hz",
                "delivery_safety_limits": {
                    "max_translation_step_m": 0.05,
                    "max_rotation_step_rad": 0.18,
                    "max_gripper_step": 0.30,
                    "workspace_x_m": [-0.05, 0.30],
                    "workspace_y_m": [0.01, 0.50],
                    "workspace_z_m": [0.14, 0.52],
                },
            },
            action_dim=7,
            action_horizon=50,
        )

        self.assertEqual(telemetry["client_inference_launch_hz"], 4.0)
        self.assertEqual(telemetry["client_control_hz"], 20.0)
        self.assertEqual(telemetry["client_chunk_rows"], 50)
        self.assertEqual(telemetry["client_minimum_horizon"], 16)
        self.assertEqual(telemetry["client_action_horizon"], 50)
        self.assertEqual(telemetry["client_capture_at"], 12.49)
        self.assertEqual(telemetry["client_latency_ms"], 200.0)
        self.assertEqual(telemetry["client_latency_steps"], 4.0)
        self.assertEqual(telemetry["client_actuator_delay_ms"], 50.0)
        self.assertEqual(telemetry["client_expired_prefix"], 4)
        self.assertEqual(telemetry["client_skipped_prefix"], 4)
        self.assertEqual(telemetry["client_blend_steps"], 4)
        self.assertEqual(telemetry["client_queue_generation"], 7)
        self.assertEqual(telemetry["client_result_generation"], 8)
        self.assertEqual((telemetry["client_old_remaining"], telemetry["client_new_remaining"]), (18, 46))
        self.assertFalse(telemetry["client_underrun"])
        self.assertTrue(telemetry["client_hold_active"])
        self.assertEqual(telemetry["client_hold_steps"], 3)
        self.assertEqual(telemetry["client_target_monotonic"], 123.45)
        self.assertTrue(telemetry["client_blend_active"])
        self.assertTrue(telemetry["client_gripper_filter_active"])
        self.assertEqual(telemetry["client_rejected_result"]["generation"], 6)
        self.assertTrue(telemetry["client_rejected_result_active"])
        self.assertEqual(telemetry["client_drop_reason"], "stale generation")
        self.assertEqual(telemetry["client_safety_profile"], "8_3_64eps_18034_frames_20hz")
        self.assertEqual(telemetry["client_delivery_safety_limits"]["max_translation_step_m"], 0.05)

    def test_policy_metadata_source_publishes_correct_async_defaults(self):
        source = Path(__file__).with_name("openpi_single_arm.py").read_text(encoding="utf-8")
        self.assertEqual(HELPER.DEFAULT_ASYNC_INFERENCE_LAUNCH_HZ, 4.0)
        self.assertIn('"action_hz": contract.action_hz', source)
        self.assertIn('"action_horizon": int(model.action_horizon)', source)
        self.assertIn('"recommended_inference_launch_hz": DEFAULT_ASYNC_INFERENCE_LAUNCH_HZ', source)
        self.assertIn('"action_time_step_s": (1.0 / contract.action_hz)', source)
        self.assertIn('"action_start_offset_steps": contract.model_action_start_offset', source)


class OpenPiActionTimingContractTest(unittest.TestCase):
    def test_raw_query_timestamps_align_both_offsets_to_model_plus_one(self):
        same_step = HELPER.action_delta_timestamps(4, 20, action_offset=0)
        next_measured = HELPER.action_delta_timestamps(4, 20, action_offset=1)
        np.testing.assert_allclose(same_step, [0.05, 0.10, 0.15, 0.20])
        np.testing.assert_allclose(next_measured, [0.00, 0.05, 0.10, 0.15])
        model_times = [(index + HELPER.MODEL_ACTION_START_OFFSET_STEPS) / 20 for index in range(4)]
        np.testing.assert_allclose(model_times, [0.05, 0.10, 0.15, 0.20])

    def test_complete_fingerprint_binds_temporal_offsets(self):
        base = {
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
        fingerprint = HELPER.complete_action_contract_fingerprint(base)
        self.assertEqual(fingerprint["action_offset"], 0)
        self.assertEqual(fingerprint["model_action_start_offset"], 1)
        with self.assertRaisesRegex(ValueError, "model_action_start_offset"):
            HELPER.complete_action_contract_fingerprint({**base, "model_action_start_offset": 0})
        with self.assertRaisesRegex(ValueError, "action_offset"):
            HELPER.complete_action_contract_fingerprint({**base, "action_offset": 2})


class OpenPiDatasetContractTest(unittest.TestCase):
    @staticmethod
    def args(dataset_id: str) -> argparse.Namespace:
        return argparse.Namespace(
            dataset_id=dataset_id,
            dataset_layout="auto",
            schema="auto",
            arm_mode="auto",
            arm_side="right",
        )

    @staticmethod
    def write_info(root: Path, dataset_id: str, info: dict) -> None:
        meta = root / dataset_id / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")

    def test_dimensions_distinguish_legacy_and_absolute_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_info(
                root,
                "legacy",
                {
                    "contract_version": 3,
                    "legacy_format": "legacy_v2",
                    "gripper_semantics": "absolute_closed_fraction_0_open_1_closed",
                    "features": {
                        "state": {"shape": [10]},
                        "actions": {"shape": [7]},
                        "image": {"dtype": "image"},
                        "wrist_image": {"dtype": "image"},
                    },
                },
            )
            self.write_info(
                root,
                "absolute",
                {
                    "contract_version": 3,
                    "gripper_semantics": "absolute_opening_fraction_0_closed_1_open",
                    "features": {
                        "observation.state": {"shape": [10]},
                        "action": {"shape": [10]},
                        "observation.images.cam_high": {"dtype": "video"},
                        "observation.images.cam_right_wrist": {"dtype": "video"},
                    },
                },
            )
            with mock.patch.dict(os.environ, {"HF_LEROBOT_HOME": str(root)}):
                legacy = HELPER.resolve_dataset_contract(self.args("legacy"))
                absolute = HELPER.resolve_dataset_contract(self.args("absolute"))

            self.assertEqual(legacy.contract_version, 2)
            self.assertTrue(legacy.legacy_delivery)
            self.assertEqual((legacy.raw_action_dim, legacy.model_action_dim), (7, 7))
            self.assertEqual(legacy.raw_action_convention, "step")
            self.assertEqual(legacy.action_offset, 1)
            self.assertEqual(legacy.model_action_start_offset, 1)
            self.assertEqual(absolute.contract_version, 3)
            self.assertFalse(absolute.legacy_delivery)
            self.assertEqual((absolute.raw_action_dim, absolute.model_action_dim), (10, 7))
            self.assertEqual(absolute.raw_action_convention, "absolute_eef_target")
            self.assertEqual(absolute.action_offset, 0)
            self.assertEqual(absolute.model_action_start_offset, 1)

    def test_extended_norm_contract_accepts_current_version(self):
        contract = {
            "action_offset": 1,
            "model_action_start_offset": 1,
            "model_action_dim": 7,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "norm_config.json"
            path.write_text(
                json.dumps({"version": HELPER.NORM_CONFIG_VERSION, **contract}),
                encoding="utf-8",
            )
            HELPER._require_extended_contract_fields(path, contract)

            path.write_text(
                json.dumps({"version": HELPER.NORM_CONFIG_VERSION - 1, **contract}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "normalization action/time contract mismatch"):
                HELPER._require_extended_contract_fields(path, contract)

    def test_joint_v2_metres_remain_explicit_in_raw_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_info(
                root,
                "joint_v2",
                {
                    "contract_version": 3,
                    "gripper_semantics": "absolute_opening_metres",
                    "features": {
                        "observation.state": {"shape": [7]},
                        "action": {"shape": [7]},
                        "observation.images.cam_high": {"dtype": "video"},
                        "observation.images.cam_right_wrist": {"dtype": "video"},
                    },
                },
            )
            with mock.patch.dict(os.environ, {"HF_LEROBOT_HOME": str(root)}):
                contract = HELPER.resolve_dataset_contract(self.args("joint_v2"))

            self.assertEqual(contract.contract_version, 2)
            self.assertEqual(contract.raw_gripper_semantics, "absolute_opening_metres")
            self.assertEqual(
                contract.model_gripper_semantics,
                "absolute_opening_fraction_0_closed_1_open",
            )


if __name__ == "__main__":
    unittest.main()
