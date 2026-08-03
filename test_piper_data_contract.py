from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from collection_session import CollectionConfig, CollectionSession, SessionState
from collect_output_arm import PiperFeedbackStaleError, _require_fresh_feedback
from piper_data_contract import (
    ACTION_NAMES,
    IMAGE_HW,
    LEROBOT_FEATURES,
    REQUIRED_EPISODE_FIELDS,
    STATE_NAMES,
    EpisodeBuffer,
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

        import time

        _require_fresh_feedback({"joint": Message(time.time())})
        with self.assertRaises(PiperFeedbackStaleError):
            _require_fresh_feedback({"joint": Message(time.time() - 1.0)})


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
        high = np.full((3, *IMAGE_HW), 20, dtype=np.uint8)
        wrist = np.full((3, *IMAGE_HW), 80, dtype=np.uint8)
        return {"cam_high": high, "cam_wrist": wrist}, {
            "cam_high": 100.0,
            "cam_wrist": 100.0,
        }


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
