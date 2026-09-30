from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from bimanual_vla.collection.camera import CameraCapture, CameraFrameSet
from bimanual_vla.deployment.recording import DeploymentRunRecorder
from bimanual_vla.data.analysis import compute_metrics, jitter_plot_series, load_analysis_data
from bimanual_vla.data.panel import DataProcessPanel


class DeploymentRunRecorderTest(unittest.TestCase):
    def test_shadow_run_records_model_smoothness_without_fabricating_sent_commands(self):
        with TemporaryDirectory() as tmp:
            recorder = DeploymentRunRecorder(tmp)
            run_dir = recorder.start({"control_hz": 20.0})
            assert run_dir is not None
            for tick in range(4):
                recorder.record_control_tick(
                    timestamp=100.0 + tick * 0.05,
                    monotonic_timestamp=20.0 + tick * 0.05,
                    delivery_state=np.zeros(7), qpos=np.zeros(7),
                    command_sent=False, action_dim=7, absolute_dim=7,
                )
            actions = np.zeros((3, 7), dtype=np.float32)
            actions[:, 0] = (0.0, 1.0, 3.0)
            launch = SimpleNamespace(
                generation=1, captured_at=100.05, captured_monotonic=20.05,
                launched_at=100.05, launched_monotonic=20.05,
                raw_delivery_state=np.zeros(7), qpos_m=np.zeros(7), image_timestamps={},
            )
            protocol = SimpleNamespace(
                schema="joint", arm_mode="single", arm_side="left", state_dim=7,
                action_dim=7, action_semantics="absolute_joint_position_opening_fraction",
                gripper_semantics="absolute_opening_fraction_0_closed_1_open",
                camera_keys=("cam_high", "cam_wrist"), action_hz=20.0,
                contract_version=3,
            )
            recorder.record_model_result(
                launch=launch, result={"actions": actions},
                arrived_at=100.15, arrived_monotonic=20.15,
                protocol=protocol, accepted=True,
            )
            recorder.stop(reason="shadow")
            jitter = compute_metrics(load_analysis_data(run_dir))["trajectory_jitter"]
            self.assertAlmostEqual(jitter["model_raw"]["intra_accel_mean_rad_per_step2"], 1.0)
            self.assertIsNone(jitter["command_sent"]["intra_accel_mean_rad_per_step2"])

    def test_model_and_sent_jitter_are_separate_and_legacy_runs_reconstruct(self):
        with TemporaryDirectory() as tmp:
            recorder = DeploymentRunRecorder(tmp)
            run_dir = recorder.start({"control_hz": 20.0})
            assert run_dir is not None
            protocol = SimpleNamespace(
                schema="joint", arm_mode="single", arm_side="left", state_dim=7,
                action_dim=7, action_semantics="absolute_joint_position_opening_fraction",
                gripper_semantics="absolute_opening_fraction_0_closed_1_open",
                camera_keys=("cam_high", "cam_wrist"), action_hz=20.0,
                contract_version=3,
            )
            for generation, values in ((1, (0.0, 1.0, 3.0)), (2, (4.0, 6.0, 9.0))):
                actions = np.zeros((3, 7), dtype=np.float32)
                actions[:, 0] = values
                actions[:, 6] = generation * 100.0  # Gripper must not enter joint L2 metrics.
                captured_at = 100.0 + (generation - 1) * 0.15
                launch = SimpleNamespace(
                    generation=generation, captured_at=captured_at,
                    captured_monotonic=10.0 + (generation - 1) * 0.15,
                    launched_at=captured_at, launched_monotonic=10.0,
                    raw_delivery_state=np.zeros(7), qpos_m=np.zeros(7),
                    image_timestamps={},
                )
                recorder.record_model_result(
                    launch=launch, result={"actions": actions},
                    arrived_at=captured_at + 0.1, arrived_monotonic=10.1,
                    protocol=protocol, accepted=True,
                )
            sent_values = (0.0, 0.5, 1.0, 1.1, 1.2, 1.3)
            for tick, value in enumerate(sent_values):
                generation = 1 if tick < 3 else 2
                recorder.record_control_tick(
                    timestamp=100.0 + tick * 0.05,
                    monotonic_timestamp=20.0 + tick * 0.05,
                    delivery_state=np.zeros(7), qpos=np.zeros(7),
                    command_sent=True, action_dim=7, absolute_dim=7,
                    command_joints_rad=np.array([value, 0, 0, 0, 0, 0]),
                    command_monotonic_timestamp=20.0 + tick * 0.05,
                    command_generation=generation,
                    command_queue_index=tick % 3,
                )
            recorder.stop(reason="test")
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertAlmostEqual(metadata["model_trajectory_jitter"]["boundary_jump_mean_rad_l2"], 1.0)
            self.assertAlmostEqual(metadata["trajectory_jitter"]["boundary_jump_mean_rad_l2"], 0.1, places=5)
            data = load_analysis_data(run_dir)
            metrics = compute_metrics(data)["trajectory_jitter"]
            self.assertAlmostEqual(metrics["model_raw"]["intra_accel_mean_rad_per_step2"], 1.0)
            self.assertAlmostEqual(metrics["model_raw"]["boundary_jump_mean_rad_l2"], 1.0)
            self.assertAlmostEqual(metrics["command_sent"]["boundary_jump_mean_rad_l2"], 0.1, places=5)
            self.assertAlmostEqual(metrics["model_raw"]["boundary_momentum_cosine_mean"], 1.0)
            x, model, sent = jitter_plot_series(data, 0, 5, "boundary_jump_mean_rad_l2")
            self.assertEqual(len(x), 2)
            self.assertEqual(int(np.isfinite(model).sum()), 1)
            self.assertEqual(int(np.isfinite(sent).sum()), 1)
            panel = SimpleNamespace(
                data=data,
                signal_var=SimpleNamespace(get=lambda: "All dimensions"),
                plot_var=SimpleNamespace(get=lambda: "Boundary position jump"),
            )
            chart_series, chart_x, units = DataProcessPanel._make_series(panel, 0, 5)
            self.assertEqual(len(chart_series), 2)
            self.assertEqual(len(chart_x), 2)
            self.assertEqual(units, "rad L2")

            # Older recordings have the arrays but no stream-tagged jitter events.
            (run_dir / "trajectory_jitter.jsonl").unlink()
            legacy = compute_metrics(load_analysis_data(run_dir))["trajectory_jitter"]
            self.assertAlmostEqual(legacy["model_raw"]["boundary_jump_mean_rad_l2"], 1.0)
            self.assertAlmostEqual(legacy["command_sent"]["boundary_jump_mean_rad_l2"], 0.1, places=5)

    def test_full_video_queue_cannot_block_shutdown_sentinel(self):
        recorder = DeploymentRunRecorder(queue_size=1)
        recorder._video_queue = Queue(maxsize=1)
        recorder._video_queue.put_nowait(("video", object()))
        sentinel = object()
        recorder._enqueue_video(sentinel, force=True)
        self.assertIs(recorder._video_queue.get_nowait(), sentinel)
        self.assertEqual(recorder._dropped_event_count, 1)

    def test_jitter_metrics_and_joint_commands_are_recorded_off_control_thread(self):
        with TemporaryDirectory() as tmp:
            recorder = DeploymentRunRecorder(tmp)
            run_dir = recorder.start({"control_hz": 20.0})
            assert run_dir is not None
            for tick, (generation, index, position) in enumerate(
                [(1, 0, 0.0), (1, 1, 1.0), (1, 2, 3.0),
                 (2, 0, 4.0), (2, 1, 6.0), (2, 2, 9.0)]
            ):
                recorder.record_control_tick(
                    timestamp=100.0 + tick * 0.05,
                    monotonic_timestamp=10.0 + tick * 0.05,
                    delivery_state=np.zeros(7, dtype=np.float32),
                    qpos=np.zeros(7, dtype=np.float32),
                    command_sent=True,
                    action_dim=7,
                    absolute_dim=7,
                    command_joints_rad=np.array([position, 0, 0, 0, 0, 0], dtype=np.float32),
                    command_monotonic_timestamp=10.0 + tick * 0.05,
                    command_generation=generation,
                    command_queue_index=index,
                )
            recorder.stop(reason="test")
            metadata = json.loads((run_dir / "metadata.json").read_text())
            jitter = metadata["trajectory_jitter"]
            self.assertAlmostEqual(jitter["intra_accel_mean_rad_per_step2"], 1.0)
            self.assertAlmostEqual(jitter["boundary_jump_mean_rad_l2"], 1.0)
            self.assertAlmostEqual(jitter["boundary_momentum_cosine_mean"], 1.0)
            self.assertEqual(jitter["boundary_jump_samples"], 1)
            with np.load(run_dir / "trajectory.npz") as trajectory:
                self.assertEqual(trajectory["command_joints_rad"].shape, (6, 6))
                self.assertAlmostEqual(float(trajectory["command_joints_rad"][-1, 0]), 9.0)
            events = [json.loads(line) for line in (run_dir / "trajectory_jitter.jsonl").read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], [
                "chunk_completed", "chunk_boundary", "chunk_boundary_momentum", "chunk_completed"
            ])

    def test_saves_aligned_trajectory_model_chunk_and_video(self):
        with TemporaryDirectory() as tmp:
            recorder = DeploymentRunRecorder(tmp, video_fps=4.0)
            run_dir = recorder.start({"instruction": "test task"})
            assert run_dir is not None
            self.assertRegex(run_dir.name, r"^\d{8}T\d{6}\.\d{6}\+0800_\d+$")

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

            with np.load(run_dir / "trajectory.npz") as trajectory:
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
            with np.load(command_file) as command_data:
                self.assertEqual(command_data["raw_actions"].shape, (2, 7))

            video_index = (run_dir / "videos" / "timestamps.jsonl").read_text().splitlines()
            self.assertEqual(len(video_index), 1)
            frame_record = json.loads(video_index[0])
            self.assertEqual(frame_record["timestamp"], 100.02)
            video_path = run_dir / frame_record["storage"]
            self.assertTrue(video_path.exists())
            self.assertTrue((run_dir / "metadata.json").exists())
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["directory_timezone"], "Asia/Shanghai")
            self.assertIn("+08:00", metadata["started_at_local"])

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

    def test_camera_nearest_frame_uses_monotonic_ring_buffer_and_returns_copy(self):
        camera = CameraCapture(cam_ids={"cam_high": 0}, fps=20, image_hw=(4, 4))
        camera._background_thread = object()
        older = CameraFrameSet(
            images={"cam_high": np.zeros((3, 4, 4), dtype=np.uint8)},
            timestamps={"cam_high": 100.0},
            monotonic_timestamps={"cam_high": 10.0},
            captured_monotonic=10.0,
        )
        nearer_image = np.ones((3, 4, 4), dtype=np.uint8)
        nearer = CameraFrameSet(
            images={"cam_high": nearer_image},
            timestamps={"cam_high": 100.05},
            monotonic_timestamps={"cam_high": 10.05},
            captured_monotonic=10.05,
        )
        camera._frame_history.extend((older, nearer))
        try:
            selected = camera.read_nearest(10.04)
        finally:
            camera._background_thread = None

        self.assertEqual(selected.captured_monotonic, 10.05)
        np.testing.assert_array_equal(selected.images["cam_high"], nearer_image)
        selected.images["cam_high"][0, 0, 0] = 9
        self.assertEqual(nearer.images["cam_high"][0, 0, 0], 1)

    def test_disabled_recorder_is_noop(self):
        with TemporaryDirectory() as tmp:
            recorder = DeploymentRunRecorder(tmp, enabled=False)
            self.assertIsNone(recorder.start())
            recorder.record_camera_frames({}, {})
            self.assertIsNone(recorder.stop())
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
