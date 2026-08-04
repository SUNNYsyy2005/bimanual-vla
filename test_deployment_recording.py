from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from camera import CameraCapture
from deployment_recording import DeploymentRunRecorder


class DeploymentRunRecorderTest(unittest.TestCase):
    def test_saves_aligned_trajectory_model_chunk_and_video(self):
        with TemporaryDirectory() as tmp:
            recorder = DeploymentRunRecorder(tmp, video_fps=4.0)
            run_dir = recorder.start({"instruction": "test task"})
            assert run_dir is not None

            image = np.zeros((3, 32, 32), dtype=np.uint8)
            image[0, 4:12, 4:12] = 255
            recorder.record_camera_frames(
                {"cam_high": image},
                {"cam_high": 100.02},
                monotonic_timestamp=20.02,
                frame_group="generation:3",
            )
            recorder.record_control_tick(
                timestamp=100.0,
                monotonic_timestamp=20.0,
                delivery_state=np.arange(10, dtype=np.float32),
                qpos=np.arange(7, dtype=np.float32),
                command_sent=True,
                action_dim=7,
                absolute_dim=10,
                command_action=np.ones(7, dtype=np.float32),
                command_absolute_target=np.ones(10, dtype=np.float32) * 2,
                command_generation=3,
                command_queue_index=4,
                execution_state="executing",
            )
            launch = SimpleNamespace(
                generation=3,
                captured_at=100.0,
                captured_monotonic=20.0,
                launched_at=100.01,
                launched_monotonic=20.01,
                raw_delivery_state=np.arange(10, dtype=np.float32),
                qpos_m=np.arange(7, dtype=np.float32),
                image_timestamps={"cam_high": 100.02},
            )
            protocol = SimpleNamespace(
                schema="joint",
                arm_mode="single",
                arm_side="right",
                state_dim=7,
                action_dim=7,
                action_semantics="absolute_joint_position_opening_fraction",
                camera_keys=("cam_high", "cam_wrist"),
                action_hz=20.0,
                gripper_semantics="absolute_opening_fraction_0_closed_1_open",
                contract_version=3,
            )
            recorder.record_model_result(
                launch=launch,
                result={
                    "actions": np.arange(14, dtype=np.float32).reshape(2, 7),
                    "execution_control": {"mode": "shadow"},
                },
                arrived_at=100.2,
                arrived_monotonic=20.2,
                protocol=protocol,
                accepted=False,
                rejection={"reason": "shadow"},
            )
            recorder.stop(reason="test")

            trajectory = np.load(run_dir / "trajectory.npz")
            self.assertEqual(trajectory["qpos"].shape, (1, 7))
            self.assertEqual(trajectory["delivery_state"].shape, (1, 10))
            self.assertEqual(trajectory["command_action"].shape, (1, 7))
            self.assertTrue(bool(trajectory["command_sent"][0]))
            self.assertEqual(int(trajectory["command_generation"][0]), 3)

            command_records = [
                json.loads(line)
                for line in (run_dir / "model_commands.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(command_records), 1)
            self.assertFalse(command_records[0]["accepted"])
            command_file = run_dir / command_records[0]["action_file"]
            self.assertTrue(command_file.exists())
            command_data = np.load(command_file)
            self.assertEqual(command_data["raw_actions"].shape, (2, 7))

            video_index = (run_dir / "videos" / "timestamps.jsonl").read_text().splitlines()
            self.assertEqual(len(video_index), 1)
            frame_record = json.loads(video_index[0])
            self.assertEqual(frame_record["timestamp"], 100.02)
            video_path = run_dir / frame_record["storage"]
            self.assertTrue(video_path.exists())
            self.assertTrue((run_dir / "metadata.json").exists())

            capture = cv2.VideoCapture(str(video_path))
            try:
                ok, frame = capture.read()
            finally:
                capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (32, 32))

    def test_camera_background_stream_feeds_latest_model_frame(self):
        camera = CameraCapture(cam_ids={"cam_high": 0}, fps=30, image_hw=(16, 16))
        camera._caps = {"cam_high": object()}
        camera._read_direct = lambda: (
            {"cam_high": np.zeros((3, 16, 16), dtype=np.uint8)},
            {"cam_high": 123.0},
        )
        callback_count = []
        camera.start_background_capture(
            lambda images, timestamps, monotonic: callback_count.append(timestamps["cam_high"]),
            fps=100.0,
        )
        try:
            images, timestamps = camera.read()
        finally:
            camera.stop_background_capture()
        self.assertEqual(images["cam_high"].shape, (3, 16, 16))
        self.assertEqual(timestamps["cam_high"], 123.0)
        self.assertGreaterEqual(len(callback_count), 1)

    def test_disabled_recorder_is_noop(self):
        with TemporaryDirectory() as tmp:
            recorder = DeploymentRunRecorder(tmp, enabled=False)
            self.assertIsNone(recorder.start())
            recorder.record_camera_frames({}, {})
            self.assertIsNone(recorder.stop())
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
