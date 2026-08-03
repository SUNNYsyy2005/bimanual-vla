from __future__ import annotations

from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from collection_session import CollectionConfig, CollectionSession, SessionState
from collect_output_arm import (
    GRIPPER_FACTOR,
    RAD_FACTOR,
    PiperFeedbackStaleError,
    _require_fresh_feedback,
    require_can_interface_up,
    read_robot_state,
    read_output_gripper_command_target,
)
from piper_action_conventions import DELIVERY_RAW_ACTION_SEMANTICS, DELIVERY_STEP_ACTION_SEMANTICS
from piper_data_contract import (
    ACTION_NAMES,
    BIMANUAL,
    DELIVERY_SCHEMA,
    IMAGE_HW,
    JOINT_NAMES,
    JOINT_SCHEMA,
    LEROBOT_FEATURES,
    MODEL_ACTION_NAMES,
    EpisodeBuffer,
    EpisodeContract,
    build_delivery_actions_with_gripper_targets,
    build_delivery_state,
    build_legacy_delivery_step_actions,
    infer_episode_contract,
    next_observation_timestamps,
)
from teleop import _episode_contract as bimanual_episode_contract, _prepare_delivery_episode
from teleop_single import _episode_contract as single_episode_contract
from trajectory import TrajectoryRecorder
from validate_piper_data import EpisodeValidationError, validate_episode


class FakePiper:
    def __init__(self):
        self.disconnected = False

    def DisconnectPort(self):
        self.disconnected = True


class FakeCameras:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.counter = 0

    def open(self):
        return None

    def verify(self):
        return {key: {"ok": True, "fps": 20.0, "latency_ms": 1} for key in self.kwargs["cam_ids"]}

    def read(self):
        self.counter += 1
        now = 100.0 + self.counter * 0.05
        frames = {
            key: np.full((*IMAGE_HW, 3), 20 + index + self.counter, dtype=np.uint8)
            for index, key in enumerate(self.kwargs["cam_ids"])
        }
        return frames, {key: now for key in frames}

    def close(self):
        self.closed = True


class PiperDataContractTest(unittest.TestCase):
    @staticmethod
    def make_state(xyz, rotation, opening_m):
        return build_delivery_state(np.asarray(xyz), np.asarray(rotation), opening_m)

    @staticmethod
    def _images(keys):
        return {key: np.full((*IMAGE_HW, 3), 20 + index, dtype=np.uint8) for index, key in enumerate(keys)}

    def test_v3_state_uses_opening_fraction(self):
        state = self.make_state([0.1, -0.2, 0.3], np.eye(3), 0.07)
        np.testing.assert_allclose(state[-1], 1.0)
        closed = self.make_state([0, 0, 0], np.eye(3), 0.0)
        np.testing.assert_allclose(closed[-1], 0.0)
        self.assertEqual(EpisodeContract(schema=JOINT_SCHEMA).gripper_semantics, "absolute_opening_fraction_0_closed_1_open")

    def test_v3_delivery_raw_action_is_absolute_10d_and_timestamps_are_separate(self):
        keys = ("cam_high", "cam_right_wrist")
        buffer = EpisodeBuffer(fps=20, schema=DELIVERY_SCHEMA, arm_side="right", camera_keys=keys)
        state0 = self.make_state([0, 0, 0.2], np.eye(3), 0.07)
        state1 = self.make_state([0.01, -0.02, 0.2], Rotation.from_euler("z", 0.1).as_matrix(), 0.0)
        for index, state in enumerate((state0, state1)):
            ts = 100.0 + index * 0.05
            buffer.add(state, self._images(keys), {key: ts for key in keys}, qpos=np.zeros(7, np.float32), state_timestamp=ts)
        payload = buffer.build_payload("pick_cube", "pick up the cube", True)
        self.assertEqual(payload["state"].shape, (3, 10))
        self.assertEqual(payload["actions"].shape, (3, 10))
        np.testing.assert_allclose(payload["actions"][0], payload["state"][1])
        np.testing.assert_allclose(payload["actions"][1], payload["state"][2])
        self.assertEqual(payload["raw_action_dim"].item(), 10)
        self.assertEqual(payload["model_action_dim"].item(), 7)
        self.assertEqual(payload["action_semantics"].item(), DELIVERY_RAW_ACTION_SEMANTICS)
        self.assertEqual(payload["gripper_semantics"].item(), "absolute_opening_fraction_0_closed_1_open")
        self.assertIn("state_timestamp", payload)
        self.assertIn("action_timestamp", payload)
        self.assertNotEqual(payload["state_timestamp"].tolist(), payload["action_timestamp"].tolist())

    def test_v3_joint_is_7d_opening_fraction(self):
        keys = ("cam_high", "cam_right_wrist")
        buffer = EpisodeBuffer(fps=20, schema=JOINT_SCHEMA, arm_side="right", camera_keys=keys)
        state0 = np.asarray([0, 0, 0, 0, 0, 0, 0.25], dtype=np.float32)
        state1 = np.asarray([1, 2, 3, 4, 5, 6, 0.75], dtype=np.float32)
        buffer.add(state0, self._images(keys), {key: 100.0 for key in keys}, qpos=state0, state_timestamp=100.0)
        buffer.add(state1, self._images(keys), {key: 100.05 for key in keys}, qpos=state1, state_timestamp=100.05)
        payload = buffer.build_payload("joint", "move the arm", True)
        self.assertEqual(payload["state"].shape, (3, 7))
        self.assertEqual(payload["actions"].shape, (3, 7))
        np.testing.assert_allclose(payload["actions"][0], state1)
        self.assertEqual(payload["action_names"][-1].item(), "right_gripper_opening_fraction")
        self.assertEqual(payload["raw_action_dim"].item(), 7)

    def test_legacy_v2_delivery_is_explicitly_inferred(self):
        states = np.asarray([
            self.make_state([0, 0, 0], np.eye(3), 0.0),
            self.make_state([0.01, 0, 0], Rotation.from_euler("z", 0.1).as_matrix(), 0.035),
        ], dtype=np.float32)
        # v2 state gripper is closed fraction, so replace the v3 opening field.
        states[:, 9] = 1.0 - states[:, 9]
        actions = build_legacy_delivery_step_actions(states)
        data = {
            "state": states,
            "actions": actions,
            "schema": np.asarray("delivery"),
            "contract_version": np.asarray(2),
            "action_semantics": np.asarray(DELIVERY_STEP_ACTION_SEMANTICS),
            "action_dim": np.asarray(7),
            "camera_keys": np.asarray(["cam_high", "cam_wrist"]),
        }
        contract = infer_episode_contract(data)
        self.assertTrue(contract.legacy_delivery_v2)
        self.assertEqual(contract.raw_action_dim, 7)
        self.assertEqual(contract.gripper_semantics, "absolute_closed_fraction_0_open_1_closed")
        self.assertEqual(contract.action_semantics, DELIVERY_STEP_ACTION_SEMANTICS)
        self.assertIn("chunk_origin", contract.model_action_semantics)
        self.assertTrue(contract.model_action_names[-1].endswith("gripper_target_closed_fraction"))

    def test_legacy_v2_episode_validates_exact_one_step_layout(self):
        states = np.stack((
            self.make_state([0.00, 0, 0.2], np.eye(3), 0.07),
            self.make_state([0.01, 0, 0.2], Rotation.from_euler("z", 0.1).as_matrix(), 0.035),
            self.make_state([0.02, 0.01, 0.2], Rotation.from_euler("z", 0.2).as_matrix(), 0.0),
        )).astype(np.float32)
        states[:, 9] = 1.0 - states[:, 9]
        states = np.concatenate((states, states[-1:]), axis=0)
        actions = build_legacy_delivery_step_actions(states)
        timestamps = np.asarray([100.0, 100.05, 100.10, 100.15], dtype=np.float64)
        high = np.stack([np.full((*IMAGE_HW, 3), 20 + i, np.uint8) for i in range(4)])
        wrist = np.stack([np.full((*IMAGE_HW, 3), 80 + i, np.uint8) for i in range(4)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0000.npz"
            np.savez_compressed(
                path, state=states, actions=actions, timestamps=timestamps,
                image=high, wrist_image=wrist,
                image_timestamps_cam_high=timestamps, image_timestamps_cam_wrist=timestamps,
                instruction=np.asarray("move the object"), success=np.asarray(True, dtype=np.bool_),
                schema=np.asarray("delivery"), contract_version=np.asarray(2),
                arm_mode=np.asarray("single"), arm_side=np.asarray("right"),
                state_dim=np.asarray(10), action_dim=np.asarray(7),
                camera_keys=np.asarray(["cam_high", "cam_wrist"]),
                action_semantics=np.asarray(DELIVERY_STEP_ACTION_SEMANTICS),
                action_source=np.asarray("next_measured_eef"),
                action_alignment=np.asarray("next_observation"), action_offset=np.asarray(1),
                terminal_padding=np.asarray(True, dtype=np.bool_),
            )
            stats = validate_episode(path, target_fps=20)
            self.assertTrue(stats.legacy_layout)
            self.assertEqual(stats.action_dim, 7)

    def test_legacy_joint_v2_names_are_not_reinterpreted(self):
        data = {
            "qpos": np.zeros((3, 7), dtype=np.float32),
            "actions": np.zeros((3, 7), dtype=np.float32),
            "camera_keys": np.asarray(["cam_high", "cam_right_wrist"]),
            "schema": np.asarray("joint"),
            "contract_version": np.asarray(2),
            "action_semantics": np.asarray("absolute_next_joint_position"),
        }
        contract = infer_episode_contract(data)
        self.assertTrue(contract.legacy_joint_v2)
        self.assertEqual(contract.state_names[-1], "right_gripper_opening_m")
        self.assertEqual(contract.model_action_names[-1], "right_gripper_opening_m")

    def test_lerobot_features_and_model_dimensions(self):
        self.assertEqual(LEROBOT_FEATURES["state"]["shape"], (10,))
        self.assertEqual(LEROBOT_FEATURES["actions"]["shape"], (10,))
        self.assertEqual(len(MODEL_ACTION_NAMES), 7)
        self.assertEqual(len(ACTION_NAMES), 10)

    def test_moving_v3_fallback_episode_validates(self):
        keys = ("cam_high", "cam_right_wrist")
        buffer = EpisodeBuffer(fps=20, schema=DELIVERY_SCHEMA, arm_side="right", camera_keys=keys)
        for index in range(3):
            ts = 100.0 + index * 0.05
            state = self.make_state([0.01 * index, 0, 0.2], Rotation.from_euler("z", 0.05 * index).as_matrix(), 0.035)
            images = {key: np.full((*IMAGE_HW, 3), 20 + camera + index, dtype=np.uint8) for camera, key in enumerate(keys)}
            buffer.add(state, images, {key: ts for key in keys}, qpos=np.zeros(7, np.float32), state_timestamp=ts)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0000.npz"
            buffer.save(path, "move", "move the arm", True)
            stats = validate_episode(path, target_fps=20)
            self.assertFalse(stats.legacy_layout)
            self.assertEqual(stats.action_dim, 10)
            self.assertEqual(stats.model_action_dim, 7)

    def test_static_success_is_rejected(self):
        keys = ("cam_high", "cam_right_wrist")
        buffer = EpisodeBuffer(fps=20, schema=DELIVERY_SCHEMA, arm_side="right", camera_keys=keys)
        state = self.make_state([0.1, 0.2, 0.3], np.eye(3), 0.0)
        for index in range(4):
            ts = 100.0 + index * 0.05
            buffer.add(state, self._images(keys), {key: ts for key in keys}, qpos=np.zeros(7, np.float32), state_timestamp=ts)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0000.npz"
            buffer.save(path, "pick_cube", "pick up the cube", True)
            with self.assertRaisesRegex(EpisodeValidationError, "100% no-op"):
                validate_episode(path, target_fps=20)


class RobotStateReaderTest(unittest.TestCase):
    def test_stale_feedback_is_rejected(self):
        class Message:
            def __init__(self, timestamp):
                self.time_stamp = timestamp
                self.Hz = 100.0
        with self.assertRaises(PiperFeedbackStaleError):
            _require_fresh_feedback({"joint": Message(time.time() - 10)})

    def test_can_interface_must_exist_and_be_up(self):
        with tempfile.TemporaryDirectory() as directory:
            sysfs_root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                require_can_interface_up("can0", sysfs_root=sysfs_root)

            interface_dir = sysfs_root / "can0"
            interface_dir.mkdir()
            (interface_dir / "flags").write_text("0x0\n")
            with self.assertRaisesRegex(RuntimeError, "is DOWN"):
                require_can_interface_up("can0", sysfs_root=sysfs_root)

            (interface_dir / "flags").write_text("0x1\n")
            require_can_interface_up("can0", sysfs_root=sysfs_root)

    def test_bimanual_reader_returns_left_then_right_and_fraction(self):
        class Msg:
            time_stamp = time.time()
            Hz = 100.0
            def __init__(self, offset):
                self.joint_state = type("J", (), {f"joint_{i}": i * RAD_FACTOR for i in range(1, 7)})()
                self.gripper_state = type("G", (), {"grippers_angle": offset * GRIPPER_FACTOR})()
                self.end_pose = type("P", (), {"X_axis": 0, "Y_axis": 0, "Z_axis": 0, "RX_axis": 0, "RY_axis": 0, "RZ_axis": 0})()
        class Robot:
            def __init__(self, opening): self.opening = opening
            def GetArmJointMsgs(self): return Msg(self.opening)
            def GetArmGripperMsgs(self): return Msg(self.opening)
            def GetArmEndPoseMsgs(self): return Msg(self.opening)
        state, qpos = read_robot_state({"left": Robot(0.01), "right": Robot(0.02)}, schema=JOINT_SCHEMA, arm_mode=BIMANUAL)
        self.assertEqual(state.shape, (14,))
        np.testing.assert_allclose(qpos[[6, 13]], [0.01 / 0.07, 0.02 / 0.07], atol=1e-6)


class TeleopDeliveryContractTest(unittest.TestCase):
    def test_delivery_contract_does_not_require_eef_calibration(self):
        single = single_episode_contract(
            SimpleNamespace(schema=DELIVERY_SCHEMA, arm_side="right", fps=20, action_horizon=50)
        )
        bimanual = bimanual_episode_contract(
            SimpleNamespace(schema=DELIVERY_SCHEMA, fps=20, action_horizon=50)
        )
        for contract, raw_dim, model_dim in ((single, 10, 7), (bimanual, 20, 14)):
            self.assertEqual(contract.raw_action_dim, raw_dim)
            self.assertEqual(contract.model_action_dim, model_dim)
            self.assertEqual(contract.action_alignment, "next_observation_pose_same_step_gripper")
            self.assertEqual(contract.action_offset, 1)
            self.assertIn("master_gripper_feedback", contract.action_source)
            self.assertEqual(contract.coordinate_frame, "slave_base")

    def test_prepare_delivery_episode_keeps_master_gripper_and_shifts_slave_pose(self):
        contract = single_episode_contract(
            SimpleNamespace(schema=DELIVERY_SCHEMA, arm_side="right", fps=20, action_horizon=50)
        )
        states = np.zeros((3, 10), dtype=np.float32)
        states[:, :3] = [[0.0, 0.1, 0.2], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]]
        states[:, 3:9] = [1, 0, 0, 0, 1, 0]
        provisional = states.copy()
        provisional[:, 9] = [0.9, 0.6, 0.2]
        prepared = _prepare_delivery_episode(
            {
                "qpos": states,
                "actions": provisional,
                "state_timestamp": np.array([1.0, 1.05, 1.10]),
                "action_timestamp": np.array([1.01, 1.06, 1.11]),
            },
            contract,
        )
        np.testing.assert_allclose(prepared["actions"][:-1, :9], states[1:, :9])
        np.testing.assert_allclose(prepared["actions"][:, 9], [0.9, 0.6, 0.2])
        np.testing.assert_allclose(prepared["action_timestamp"], [1.05, 1.10, 1.15])
        np.testing.assert_allclose(prepared["gripper_command_timestamp"][:, 0], [1.01, 1.06, 1.11])
        self.assertTrue(prepared["gripper_command_present"].all())


class MixedDeliveryActionTest(unittest.TestCase):
    def test_next_slave_pose_uses_same_step_master_gripper(self):
        states = np.zeros((3, 20), dtype=np.float32)
        for arm in range(2):
            offset = arm * 10
            states[:, offset : offset + 3] = np.array(
                [[0.0, 0.1, 0.2], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]],
                dtype=np.float32,
            ) + arm
            states[:, offset + 3 : offset + 9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
            states[:, offset + 9] = [0.1, 0.2, 0.3]
        master_grippers = np.array([[0.8, 0.7], [0.6, 0.5], [0.4, 0.3]], dtype=np.float32)
        actions = build_delivery_actions_with_gripper_targets(states, master_grippers, arm_count=2)
        np.testing.assert_allclose(actions[:-1, :9], states[1:, :9])
        np.testing.assert_allclose(actions[:-1, 10:19], states[1:, 10:19])
        np.testing.assert_allclose(actions[:, [9, 19]], master_grippers)
        np.testing.assert_allclose(actions[-1, :9], states[-1, :9])
        np.testing.assert_allclose(actions[-1, 10:19], states[-1, 10:19])

    def test_next_observation_action_timestamps(self):
        values = next_observation_timestamps(np.array([1.0, 1.05, 1.10]), fps=20)
        np.testing.assert_allclose(values, [1.05, 1.10, 1.15])


class TimestampAndCommandTest(unittest.TestCase):
    def test_trajectory_records_independent_timestamps(self):
        recorder = TrajectoryRecorder()
        recorder.start()
        image = np.ones((*IMAGE_HW, 3), dtype=np.uint8)
        recorder.add(
            np.zeros(7, np.float32), np.ones(7, np.float32), {"cam_high": image},
            {"cam_high": 10.02}, state_timestamp=10.0, action_timestamp=10.01,
            joint_qpos=np.zeros(7, np.float32),
        )
        data = recorder.to_numpy_dict()
        self.assertEqual(data["state_timestamp"].tolist(), [10.0])
        self.assertEqual(data["action_timestamp"].tolist(), [10.01])
        self.assertEqual(data["image_timestamps_cam_high"].tolist(), [10.02])

    def test_commanded_gripper_support_uses_opening_fraction(self):
        message = type("Message", (), {
            "time_stamp": time.time(),
            "gripper_ctrl": type("Command", (), {"grippers_angle": 0.035 * GRIPPER_FACTOR})(),
        })()
        robot = type("Robot", (), {"GetArmGripperCtrl": lambda self: message})()
        self.assertAlmostEqual(read_output_gripper_command_target(robot), 0.5, places=6)


class CollectionSessionTest(unittest.TestCase):
    def test_collection_lifecycle_writes_v3_metadata(self):
        robots = {"can_left": FakePiper(), "can_right": FakePiper()}
        state = np.zeros(14, dtype=np.float32)
        state[6], state[13] = 0.2, 0.8
        with tempfile.TemporaryDirectory() as directory:
            session = CollectionSession(
                CollectionConfig(output_dir=Path(directory), schema=JOINT_SCHEMA, arm_mode=BIMANUAL, arm_side="both", left_can_name="can_left", right_can_name="can_right"),
                robot_connect=robots.__getitem__, camera_factory=FakeCameras,
                state_reader=lambda robot: (state, state.copy()),
                camera_verifier=lambda cameras, fps: {key: {"ok": True, "fps": fps} for key in cameras.kwargs["cam_ids"]},
            )
            session.connect(); session.start_episode("handover", "handover the object")
            session.capture_once(); session.stop_episode()
            path, _ = session.save_episode(True, validate=False)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["actions"].shape, (2, 14))
                self.assertEqual(data["action_source"].item(), "next_measured_joint_fallback")
                self.assertEqual(data["fps"].item(), 20)
                self.assertEqual(data["action_horizon"].item(), 50)
                self.assertIn("state_timestamp", data.files)
                self.assertIn("action_timestamp", data.files)
            session.disconnect()

    def test_delivery_collection_uses_10d_raw_fallback(self):
        robots = {"can_left": FakePiper(), "can_right": FakePiper()}
        left = build_delivery_state([0.1, 0.2, 0.3], np.eye(3), 0.07)
        right = build_delivery_state([0.4, 0.5, 0.6], np.eye(3), 0.02)
        state = np.concatenate((left, right))
        qpos = np.zeros(14, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            session = CollectionSession(
                CollectionConfig(output_dir=Path(directory), schema=DELIVERY_SCHEMA, arm_mode=BIMANUAL, arm_side="both", left_can_name="can_left", right_can_name="can_right"),
                robot_connect=robots.__getitem__, camera_factory=FakeCameras,
                state_reader=lambda robot: (state, qpos),
                camera_verifier=lambda cameras, fps: {key: {"ok": True, "fps": fps} for key in cameras.kwargs["cam_ids"]},
            )
            session.connect(); session.start_episode("handover", "handover the object")
            session.capture_once(); session.stop_episode()
            path, _ = session.save_episode(True, validate=False)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["state"].shape, (2, 20))
                self.assertEqual(data["actions"].shape, (2, 20))
                self.assertEqual(data["raw_action_dim"].item(), 20)
                self.assertEqual(data["model_action_dim"].item(), 14)
                self.assertEqual(data["action_source"].item(), "next_measured_eef_fallback")
            session.disconnect()

    def test_failed_validation_does_not_publish_episode(self):
        state = build_delivery_state(np.zeros(3), np.eye(3), 0.07)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            def fail_validation(path, target_fps):
                raise ValueError("synthetic validation failure")
            session = CollectionSession(
                CollectionConfig(output_dir=output_dir), robot_connect=lambda can_name: FakePiper(), camera_factory=FakeCameras,
                state_reader=lambda robot: (state, np.zeros(7, dtype=np.float32)), camera_verifier=lambda cameras, fps: {}, episode_validator=fail_validation,
            )
            session.connect(); session.start_episode("pick_cube", "pick up the cube"); session.capture_once(); session.stop_episode()
            with self.assertRaisesRegex(ValueError, "synthetic validation failure"):
                session.save_episode(True)
            self.assertFalse((output_dir / "ep_0000.npz").exists())
            self.assertIs(session.state, SessionState.REVIEW)


if __name__ == "__main__":
    unittest.main()
