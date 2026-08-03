from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from collection_session import CollectionConfig, CollectionSession, SessionState
from collect_gui import CollectorGUI, check_initial_pose
from collect_output_arm import (
    GRIPPER_FACTOR,
    RAD_FACTOR,
    PiperFeedbackStaleError,
    _require_fresh_feedback,
    read_robot_state,
)
from piper_data_contract import (
    ACTION_NAMES,
    BIMANUAL,
    DELIVERY_SCHEMA,
    IMAGE_HW,
    JOINT_SCHEMA,
    LEROBOT_FEATURES,
    REQUIRED_EPISODE_FIELDS,
    STATE_NAMES,
    EpisodeBuffer,
    EpisodeContract,
    build_delivery_state,
)
from validate_piper_data import EpisodeValidationError, validate_episode


class PiperDataContractTest(unittest.TestCase):
    def make_state(self, xyz, rotation, gripper_opening):
        return build_delivery_state(
            np.asarray(xyz, dtype=np.float64),
            np.asarray(rotation, dtype=np.float64),
            gripper_opening,
        )

    def test_state_layout_and_gripper_convention(self):
        state = self.make_state([0.1, -0.2, 0.3], np.eye(3), 0.07)
        np.testing.assert_allclose(
            state,
            [0.1, -0.2, 0.3, 1, 0, 0, 0, 1, 0, 0],
            atol=1e-7,
        )
        self.assertEqual(state.dtype, np.float32)

        closed = self.make_state([0, 0, 0], np.eye(3), 0.0)
        self.assertEqual(float(closed[-1]), 1.0)

    def test_payload_uses_base_frame_actions_and_terminal_observation(self):
        buffer = EpisodeBuffer(fps=20)
        rotation_0 = Rotation.from_euler("x", 0.2).as_matrix()
        rotation_1 = Rotation.from_rotvec([0.0, 0.0, 0.1]).as_matrix() @ rotation_0
        states = (
            self.make_state([0.0, 0.0, 0.2], rotation_0, 0.07),
            self.make_state([0.01, -0.02, 0.2], rotation_1, 0.0),
        )
        high = np.full((3, *IMAGE_HW), 20, dtype=np.uint8)
        wrist = np.full((3, *IMAGE_HW), 80, dtype=np.uint8)
        for index, state in enumerate(states):
            timestamp = 100.0 + index * 0.05
            buffer.add(
                state,
                {"cam_high": high + index, "cam_wrist": wrist + index},
                {"cam_high": timestamp, "cam_wrist": timestamp},
                qpos=np.arange(7, dtype=np.float32),
                state_timestamp=timestamp,
            )

        payload = buffer.build_payload("pick_cube", "pick up the cube", True)
        self.assertTrue(REQUIRED_EPISODE_FIELDS.issubset(payload))
        self.assertEqual(payload["state"].shape, (3, len(STATE_NAMES)))
        self.assertEqual(payload["actions"].shape, (3, len(ACTION_NAMES)))
        self.assertEqual(payload["image"].shape, (3, *IMAGE_HW, 3))
        self.assertEqual(payload["wrist_image"].shape, (3, *IMAGE_HW, 3))
        self.assertEqual(payload["joint_qpos"].shape, (3, 7))
        np.testing.assert_allclose(payload["actions"][0, :3], [0.01, -0.02, 0])
        np.testing.assert_allclose(payload["actions"][0, 3:6], [0, 0, 0.1], atol=1e-6)
        np.testing.assert_allclose(payload["actions"][-1, :6], 0)
        self.assertEqual(payload["actions"][-1, 6], payload["state"][-1, 9])
        self.assertEqual(payload["instruction"].item(), "pick up the cube")
        self.assertEqual(payload["success"].dtype, np.bool_)

    def test_lerobot_features_come_from_the_same_dimensions(self):
        self.assertEqual(LEROBOT_FEATURES["state"]["shape"], (len(STATE_NAMES),))
        self.assertEqual(LEROBOT_FEATURES["actions"]["shape"], (len(ACTION_NAMES),))
        self.assertEqual(LEROBOT_FEATURES["image"]["shape"], (*IMAGE_HW, 3))
        self.assertEqual(LEROBOT_FEATURES["wrist_image"]["shape"], (*IMAGE_HW, 3))


    def test_successful_static_episode_is_rejected(self):
        buffer = EpisodeBuffer(fps=20)
        state = self.make_state([0.1, 0.2, 0.3], np.eye(3), 0.0)
        high = np.full((3, *IMAGE_HW), 20, dtype=np.uint8)
        wrist = np.full((3, *IMAGE_HW), 80, dtype=np.uint8)
        for index in range(4):
            timestamp = 100.0 + index * 0.05
            buffer.add(
                state,
                {"cam_high": high + index, "cam_wrist": wrist + index},
                {"cam_high": timestamp, "cam_wrist": timestamp},
                qpos=np.zeros(7, dtype=np.float32),
                state_timestamp=timestamp,
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0000.npz"
            buffer.save(path, "pick_cube", "pick up the cube", True)
            with self.assertRaisesRegex(EpisodeValidationError, "100% no-op"):
                validate_episode(path, target_fps=20)

    def test_stale_piper_feedback_is_rejected(self):
        class Message:
            def __init__(self, timestamp):
                self.time_stamp = timestamp
                self.Hz = 100.0

        _require_fresh_feedback({"joint": Message(time.time())})
        with self.assertRaises(PiperFeedbackStaleError):
            _require_fresh_feedback({"joint": Message(time.time() - 1.0)})


    def test_bimanual_joint_payload_uses_left_then_right(self):
        contract = EpisodeContract(schema=JOINT_SCHEMA, arm_mode=BIMANUAL)
        self.assertEqual(contract.state_dim, 14)
        self.assertEqual(contract.action_dim, 14)
        self.assertEqual(
            contract.camera_keys,
            ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        )
        buffer = EpisodeBuffer(fps=20, schema=JOINT_SCHEMA, arm_mode=BIMANUAL)
        image = np.zeros((3, *IMAGE_HW), dtype=np.uint8)
        states = (
            np.arange(14, dtype=np.float32) / 100,
            np.arange(14, dtype=np.float32) / 100 + 0.1,
        )
        for index, state in enumerate(states):
            timestamp = 200.0 + index * 0.05
            images = {
                "cam_high": image + 10 + index,
                "cam_left_wrist": image + 20 + index,
                "cam_right_wrist": image + 30 + index,
            }
            buffer.add(
                state,
                images,
                {key: timestamp for key in images},
                qpos=state,
                state_timestamp=timestamp,
            )

        payload = buffer.build_payload("handover", "handover the object", True)
        self.assertEqual(payload["state"].shape, (3, 14))
        self.assertEqual(payload["actions"].shape, (3, 14))
        np.testing.assert_allclose(payload["state"][0, :7], states[0][:7])
        np.testing.assert_allclose(payload["state"][0, 7:], states[0][7:])
        np.testing.assert_allclose(payload["actions"][0], states[1])
        self.assertEqual(payload["arm_mode"].item(), BIMANUAL)
        self.assertEqual(payload["arm_side"].item(), "both")
        self.assertEqual(payload["schema"].item(), JOINT_SCHEMA)
        self.assertEqual(payload["action_offset"].item(), 1)
        for key in contract.camera_keys:
            self.assertEqual(payload[contract.image_field(key)].shape, (3, *IMAGE_HW, 3))

    def test_bimanual_delivery_actions_are_independent_per_arm(self):
        contract = EpisodeContract(schema=DELIVERY_SCHEMA, arm_mode=BIMANUAL)
        buffer = EpisodeBuffer(fps=20, schema=DELIVERY_SCHEMA, arm_mode=BIMANUAL)
        image = np.zeros((3, *IMAGE_HW), dtype=np.uint8)
        left_0 = self.make_state([0.0, 0.0, 0.2], np.eye(3), 0.07)
        right_0 = self.make_state([0.3, 0.0, 0.2], np.eye(3), 0.0)
        left_1 = self.make_state([0.01, 0.0, 0.2], np.eye(3), 0.0)
        right_1 = self.make_state([0.3, -0.02, 0.2], np.eye(3), 0.07)
        for index, state in enumerate(
            (np.concatenate((left_0, right_0)), np.concatenate((left_1, right_1)))
        ):
            timestamp = 300.0 + index * 0.05
            images = {
                "cam_high": image + 10 + index,
                "cam_left_wrist": image + 20 + index,
                "cam_right_wrist": image + 30 + index,
            }
            buffer.add(
                state,
                images,
                {key: timestamp for key in images},
                qpos=np.zeros(14, dtype=np.float32),
                state_timestamp=timestamp,
            )

        payload = buffer.build_payload("handover", "handover the object", True)
        self.assertEqual(payload["state"].shape, (3, 20))
        self.assertEqual(payload["actions"].shape, (3, 14))
        np.testing.assert_allclose(payload["actions"][0, :3], [0.01, 0.0, 0.0])
        np.testing.assert_allclose(payload["actions"][0, 7:10], [0.0, -0.02, 0.0])
        self.assertEqual(payload["actions"][0, 6], left_1[9])
        self.assertEqual(payload["actions"][0, 13], right_1[9])
        self.assertEqual(payload["schema"].item(), DELIVERY_SCHEMA)


class CollectionGuiStartPolicyTest(unittest.TestCase):
    def test_pose_warning_does_not_disable_collection_start(self):
        class Button:
            state = None

            def configure(self, *, state):
                self.state = state

        gui = CollectorGUI.__new__(CollectorGUI)
        gui.piper = object()
        gui.cameras = object()
        gui.recording = False
        gui.reset_thread = None
        gui.latest_pose_ok = False
        gui.start_button = Button()

        gui._update_start_button()

        self.assertEqual(gui.start_button.state, "normal")


class InitialPoseCheckTest(unittest.TestCase):
    def test_bimanual_pose_reports_side_specific_errors(self):
        qpos = np.zeros(14, dtype=np.float32)
        qpos[1] = np.deg2rad(7.0)
        qpos[13] = 0.008
        ok, reason, errors = check_initial_pose(
            qpos,
            np.zeros(14, dtype=np.float32),
            np.deg2rad(5.0),
            0.005,
        )
        self.assertFalse(ok)
        self.assertIn("L-J2", reason)
        self.assertIn("R-Gripper", reason)
        self.assertEqual(errors.shape, (14,))

    def test_bimanual_pose_accepts_both_arms_within_tolerance(self):
        qpos = np.zeros(14, dtype=np.float32)
        qpos[[0, 7]] = np.deg2rad([4.0, -4.0])
        qpos[[6, 13]] = [0.004, 0.004]
        ok, reason, errors = check_initial_pose(
            qpos,
            np.zeros(14, dtype=np.float32),
            np.deg2rad(5.0),
            0.005,
        )
        self.assertTrue(ok)
        self.assertIn("both robots", reason)
        self.assertEqual(errors.shape, (14,))



class RobotStateReaderTest(unittest.TestCase):
    class FeedbackArm:
        def __init__(self, joints, gripper_m, xyz_m):
            self.joints = np.asarray(joints, dtype=np.float64)
            self.gripper_m = float(gripper_m)
            self.xyz_m = np.asarray(xyz_m, dtype=np.float64)

        @staticmethod
        def _message(**kwargs):
            return SimpleNamespace(time_stamp=time.time(), Hz=100.0, **kwargs)

        def GetArmJointMsgs(self):
            values = np.rint(self.joints * RAD_FACTOR).astype(np.int64)
            joint_state = SimpleNamespace(
                **{f"joint_{index + 1}": int(value) for index, value in enumerate(values)}
            )
            return self._message(joint_state=joint_state)

        def GetArmGripperMsgs(self):
            gripper_state = SimpleNamespace(
                grippers_angle=int(round(self.gripper_m * GRIPPER_FACTOR))
            )
            return self._message(gripper_state=gripper_state)

        def GetArmEndPoseMsgs(self):
            xyz = np.rint(self.xyz_m * 1_000_000).astype(np.int64)
            end_pose = SimpleNamespace(
                X_axis=int(xyz[0]),
                Y_axis=int(xyz[1]),
                Z_axis=int(xyz[2]),
                RX_axis=0,
                RY_axis=0,
                RZ_axis=0,
            )
            return self._message(end_pose=end_pose)

    def test_bimanual_reader_returns_left_then_right_joint_and_delivery_states(self):
        left = self.FeedbackArm(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            0.01,
            [0.10, 0.20, 0.30],
        )
        right = self.FeedbackArm(
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
            0.02,
            [0.40, 0.50, 0.60],
        )
        robot = {"left": left, "right": right}

        joint_state, qpos = read_robot_state(
            robot, schema=JOINT_SCHEMA, arm_mode=BIMANUAL
        )
        self.assertEqual(joint_state.shape, (14,))
        np.testing.assert_allclose(joint_state, qpos)
        np.testing.assert_allclose(qpos[[6, 13]], [0.01, 0.02], atol=1e-7)
        self.assertGreater(float(qpos[0]), 0.0)
        self.assertLess(float(qpos[7]), 0.0)

        delivery_state, delivery_qpos = read_robot_state(
            robot, schema=DELIVERY_SCHEMA, arm_mode=BIMANUAL
        )
        self.assertEqual(delivery_state.shape, (20,))
        np.testing.assert_allclose(delivery_qpos, qpos, atol=2e-5)
        np.testing.assert_allclose(delivery_state[:3], [0.10, 0.20, 0.30])
        np.testing.assert_allclose(delivery_state[10:13], [0.40, 0.50, 0.60])
        np.testing.assert_allclose(delivery_state[3:9], [1, 0, 0, 0, 1, 0])
        np.testing.assert_allclose(delivery_state[13:19], [1, 0, 0, 0, 1, 0])


class FakePiper:
    def __init__(self):
        self.disconnected = False

    def DisconnectPort(self):
        self.disconnected = True


class FakeCameras:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    def open(self):
        return None

    def close(self):
        self.closed = True

    def read(self):
        camera_keys = tuple(self.kwargs["cam_ids"])
        images = {
            key: np.full((3, *IMAGE_HW), 20 + index * 30, dtype=np.uint8)
            for index, key in enumerate(camera_keys)
        }
        return images, {key: 100.0 for key in camera_keys}


class CollectionSessionTest(unittest.TestCase):
    def test_ui_neutral_collection_lifecycle(self):
        piper = FakePiper()
        state = build_delivery_state(np.zeros(3), np.eye(3), 0.07)
        qpos = np.zeros(7, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            session = CollectionSession(
                CollectionConfig(output_dir=Path(directory)),
                robot_connect=lambda can_name: piper,
                camera_factory=FakeCameras,
                state_reader=lambda robot: (state, qpos),
                camera_verifier=lambda cameras, fps: {
                    "cam_high": {"ok": True, "fps": fps},
                    "cam_wrist": {"ok": True, "fps": fps},
                },
            )
            session.connect()
            self.assertIs(session.state, SessionState.READY)
            session.start_episode("pick_cube", "pick up the cube")
            session.capture_once()
            self.assertEqual(session.stop_episode(), 1)
            path, stats = session.save_episode(True, validate=False)
            self.assertIsNone(stats)
            self.assertTrue(path.is_file())
            self.assertIs(session.state, SessionState.READY)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["state"].shape, (2, 10))
                self.assertEqual(data["instruction"].item(), "pick up the cube")
            session.disconnect()
            self.assertTrue(piper.disconnected)
            self.assertIs(session.state, SessionState.DISCONNECTED)

    def test_bimanual_joint_collection_lifecycle(self):
        robots = {"can_left": FakePiper(), "can_right": FakePiper()}
        qpos = np.arange(14, dtype=np.float32) / 100
        with tempfile.TemporaryDirectory() as directory:
            session = CollectionSession(
                CollectionConfig(
                    output_dir=Path(directory),
                    schema=JOINT_SCHEMA,
                    arm_mode=BIMANUAL,
                    arm_side="both",
                    left_can_name="can_left",
                    right_can_name="can_right",
                    cam_left_wrist_device="left-camera",
                    cam_right_wrist_device="right-camera",
                ),
                robot_connect=robots.__getitem__,
                camera_factory=FakeCameras,
                state_reader=lambda robot: (qpos, qpos),
                camera_verifier=lambda cameras, fps: {
                    key: {"ok": True, "fps": fps}
                    for key in cameras.kwargs["cam_ids"]
                },
            )
            checks = session.connect()
            self.assertEqual(set(session.piper), {"left", "right"})
            self.assertEqual(
                tuple(session.cameras.kwargs["cam_ids"]),
                ("cam_high", "cam_left_wrist", "cam_right_wrist"),
            )
            self.assertEqual(set(checks), set(session.cameras.kwargs["cam_ids"]))
            session.start_episode("handover", "handover the object")
            session.capture_once()
            session.stop_episode()
            path, _ = session.save_episode(True, validate=False)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["state"].shape, (2, 14))
                self.assertEqual(data["actions"].shape, (2, 14))
                self.assertEqual(data["arm_mode"].item(), BIMANUAL)
                self.assertEqual(data["action_source"].item(), "next_measured_qpos")
                self.assertEqual(data["action_alignment"].item(), "next_observation")
                self.assertEqual(data["action_offset"].item(), 1)
            session.disconnect()
            self.assertTrue(all(robot.disconnected for robot in robots.values()))

    def test_bimanual_delivery_collection_lifecycle(self):
        robots = {"can_left": FakePiper(), "can_right": FakePiper()}
        left_state = build_delivery_state([0.1, 0.2, 0.3], np.eye(3), 0.07)
        right_state = build_delivery_state([0.4, 0.5, 0.6], np.eye(3), 0.02)
        state = np.concatenate((left_state, right_state))
        qpos = np.arange(14, dtype=np.float32) / 100
        with tempfile.TemporaryDirectory() as directory:
            session = CollectionSession(
                CollectionConfig(
                    output_dir=Path(directory),
                    schema=DELIVERY_SCHEMA,
                    arm_mode=BIMANUAL,
                    arm_side="both",
                    left_can_name="can_left",
                    right_can_name="can_right",
                    cam_left_wrist_device="left-camera",
                    cam_right_wrist_device="right-camera",
                ),
                robot_connect=robots.__getitem__,
                camera_factory=FakeCameras,
                state_reader=lambda robot: (state, qpos),
                camera_verifier=lambda cameras, fps: {
                    key: {"ok": True, "fps": fps}
                    for key in cameras.kwargs["cam_ids"]
                },
            )
            session.connect()
            session.start_episode("handover", "handover the object")
            session.capture_once()
            session.stop_episode()
            path, _ = session.save_episode(True, validate=False)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["state"].shape, (2, 20))
                self.assertEqual(data["actions"].shape, (2, 14))
                self.assertEqual(data["joint_qpos"].shape, (2, 14))
                self.assertEqual(data["schema"].item(), DELIVERY_SCHEMA)
                self.assertEqual(data["arm_mode"].item(), BIMANUAL)
                self.assertEqual(data["action_source"].item(), "next_measured_eef")
                self.assertEqual(data["action_alignment"].item(), "next_observation")
                self.assertEqual(data["action_offset"].item(), 1)
                self.assertEqual(
                    data["camera_keys"].tolist(),
                    ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                )
            session.disconnect()
            self.assertTrue(all(robot.disconnected for robot in robots.values()))

    def test_failed_validation_does_not_publish_an_episode(self):
        state = build_delivery_state(np.zeros(3), np.eye(3), 0.07)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)

            def fail_validation(path, target_fps):
                raise ValueError("synthetic validation failure")

            session = CollectionSession(
                CollectionConfig(output_dir=output_dir),
                robot_connect=lambda can_name: FakePiper(),
                camera_factory=FakeCameras,
                state_reader=lambda robot: (state, np.zeros(7, dtype=np.float32)),
                camera_verifier=lambda cameras, fps: {},
                episode_validator=fail_validation,
            )
            session.connect()
            session.start_episode("pick_cube", "pick up the cube")
            session.capture_once()
            session.stop_episode()
            with self.assertRaisesRegex(ValueError, "synthetic validation failure"):
                session.save_episode(True)
            self.assertFalse((output_dir / "ep_0000.npz").exists())
            self.assertEqual(list(output_dir.glob(".*.npz")), [])
            self.assertIs(session.state, SessionState.REVIEW)
            self.assertEqual(session.frame_count, 1)


if __name__ == "__main__":
    unittest.main()
