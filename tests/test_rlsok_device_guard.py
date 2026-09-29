"""RLSOK bridge checks use synthetic resolver replies; no hardware is opened."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from bimanual_vla.collection.gui import CollectorGUI
from bimanual_vla.collection.camera import CameraCapture
from bimanual_vla.collection.session import CollectionConfig, CollectionSession
from bimanual_vla.data.contract import DELIVERY_SCHEMA
from bimanual_vla.deployment import client as rtc_client
from bimanual_vla.deployment.client import PiperFeedbackStaleError, run_rtc_client
from bimanual_vla.device_guard import DeviceCheckRejected, ResolvedDevices, RuntimeDeviceGuard, check_devices


def _reply(**changes):
    value = {
        "schema_version": 1,
        "rlsok_version": "1.5.8",
        "review_decision": "UNCHANGED",
        "review_scope": "saved-configuration-only",
        "hardware_dispatch": False,
        "inventory_method": "linux-sysfs-udev",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "resolved": {
            "arms": {"right": "can1"},
            "cameras": {"overhead": "/dev/video4", "right_wrist": "/dev/v4l/by-path/example-video-index0"},
        },
    }
    value.update(changes)
    if value["review_decision"] != "UNCHANGED":
        value.pop("inventory_method")
        value.pop("observed_at")
        value.pop("resolved")
    return subprocess.CompletedProcess(["example-resolver"], 0, json.dumps(value), "")


def _reply_for_call(*_args, **kwargs):
    request = json.loads(kwargs["input"])
    return _reply(request_id=request["request_id"])


class DeviceGuardTest(unittest.TestCase):
    def test_live_camera_loss_exits_without_old_command_or_auto_return(self):
        args = SimpleNamespace(
            arm_mode="single", arm_side="right", can="can0", left_can="can1", right_can="can2",
            cam_high_device="auto", cam_wrist_device="auto",
            cam_left_wrist_device="auto", cam_right_wrist_device="auto",
            rlsok_resolver="", output_mode="auto", host="localhost", port=8000,
            source_name="synthetic", rtc_session_id="synthetic", instruction="test",
            monitoring_dir="/tmp/unused-monitoring", no_monitoring=True,
            monitoring_rate=1.0, monitoring_level="compact", camera_fps=20,
            camera_preview=False, camera_preview_fps=8.0,
            record_root="/tmp/unused-recording", record_video_fps=None,
            no_recording=True, max_feedback_age_s=0.5, control_hz=20.0,
            hz=4.0, allow_execution=True, min_action_chunk_steps=16,
            auto_return=True,
        )
        camera = MagicMock(spec=CameraCapture)
        camera.verify.return_value = {
            key: dict(ok=True, selected_device="/dev/video4", video_device="/dev/video4",
                      shape=(240, 424, 3), latency_ms=1.0)
            for key in ("cam_high", "cam_wrist")
        }
        camera.assert_background_healthy.side_effect = RuntimeError("synthetic unplug")
        execution = MagicMock()
        execution.inference_trigger_mode = "periodic"
        execution.inference_trigger_step = 10
        recorder = MagicMock()
        recorder.is_active = False
        preview = MagicMock()
        preview.enabled = False
        monitoring = MagicMock()
        monitoring.level = "compact"
        piper = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(patch("bimanual_vla.device_guard.check_devices", return_value=None))
            stack.enter_context(patch("bimanual_vla.deployment.client.connect_piper", return_value=piper))
            stack.enter_context(patch("bimanual_vla.deployment.client.read_output_qpos", return_value=np.zeros(7, dtype=np.float32)))
            stack.enter_context(patch("bimanual_vla.deployment.client.CameraCapture", return_value=camera))
            stack.enter_context(patch("bimanual_vla.deployment.client.CameraPreview", return_value=preview))
            stack.enter_context(patch("bimanual_vla.deployment.client.ExecutionController", return_value=execution))
            stack.enter_context(patch("bimanual_vla.deployment.client.DeploymentRunRecorder", return_value=recorder))
            stack.enter_context(patch("bimanual_vla.deployment.client.MonitoringRecorder", return_value=monitoring))
            worker = stack.enter_context(patch("bimanual_vla.deployment.client.AsyncPolicyInference"))
            policy = stack.enter_context(patch("bimanual_vla.deployment.client.connect_policy"))
            auto_return = stack.enter_context(patch("bimanual_vla.deployment.client.return_pipers_to_initial"))
            with self.assertRaisesRegex(DeviceCheckRejected, "camera stream unavailable"):
                run_rtc_client(args)
        execution.discard_pending_actions.assert_called_once()
        execution.execute_next.assert_not_called()
        policy.assert_not_called()
        auto_return.assert_not_called()
        worker.return_value.shutdown.assert_called_once()
        piper.DisconnectPort.assert_called_once()

    def test_live_can_feedback_loss_exits_without_old_command_or_auto_return(self):
        args = SimpleNamespace(
            arm_mode="single", arm_side="right", can="can0", left_can="can1", right_can="can2",
            cam_high_device="auto", cam_wrist_device="auto",
            cam_left_wrist_device="auto", cam_right_wrist_device="auto",
            rlsok_resolver="", output_mode="joint", host="localhost", port=8000,
            source_name="synthetic", rtc_session_id="synthetic", instruction="test",
            monitoring_dir="/tmp/unused-monitoring", no_monitoring=True,
            monitoring_rate=1.0, monitoring_level="compact", camera_fps=20,
            camera_preview=False, camera_preview_fps=8.0,
            record_root="/tmp/unused-recording", record_video_fps=None,
            no_recording=True, max_feedback_age_s=0.5, control_hz=20.0,
            hz=4.0, allow_execution=True, min_action_chunk_steps=16,
            auto_return=True, reconnect_delay=0.0,
        )
        camera = MagicMock(spec=CameraCapture)
        camera.verify.return_value = {
            key: dict(ok=True, selected_device="/dev/video4", video_device="/dev/video4",
                      shape=(240, 424, 3), latency_ms=1.0)
            for key in ("cam_high", "cam_wrist")
        }
        execution = MagicMock()
        execution.inference_trigger_mode = "periodic"
        execution.inference_trigger_step = 10
        execution.control_hz = 20.0
        execution.inference_hz = 4.0
        execution.pending_action_count = 0
        recorder = MagicMock()
        recorder.is_active = False
        preview = MagicMock()
        preview.enabled = False
        monitoring = MagicMock()
        monitoring.level = "compact"
        protocol = MagicMock()
        protocol.schema = "joint"
        protocol.camera_keys = ("cam_high", "cam_wrist")
        piper = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(patch("bimanual_vla.device_guard.check_devices", return_value=None))
            stack.enter_context(patch("bimanual_vla.deployment.client.connect_piper", return_value=piper))
            stack.enter_context(patch("bimanual_vla.deployment.client.read_output_qpos", side_effect=[
                np.zeros(7, dtype=np.float32), PiperFeedbackStaleError("synthetic CAN unplug"),
            ]))
            stack.enter_context(patch("bimanual_vla.deployment.client.CameraCapture", return_value=camera))
            stack.enter_context(patch("bimanual_vla.deployment.client.CameraPreview", return_value=preview))
            stack.enter_context(patch("bimanual_vla.deployment.client.ExecutionController", return_value=execution))
            stack.enter_context(patch("bimanual_vla.deployment.client.DeploymentRunRecorder", return_value=recorder))
            stack.enter_context(patch("bimanual_vla.deployment.client.MonitoringRecorder", return_value=monitoring))
            worker = stack.enter_context(patch("bimanual_vla.deployment.client.AsyncPolicyInference"))
            worker.return_value.in_flight = False
            policy = stack.enter_context(patch("bimanual_vla.deployment.client.connect_policy", return_value=(MagicMock(), protocol)))
            auto_return = stack.enter_context(patch("bimanual_vla.deployment.client.return_pipers_to_initial"))
            with self.assertRaisesRegex(DeviceCheckRejected, "synthetic CAN unplug"):
                run_rtc_client(args)
        execution.discard_pending_actions.assert_called_once()
        execution.execute_next.assert_not_called()
        policy.assert_called_once()
        auto_return.assert_not_called()
        piper.DisconnectPort.assert_called_once()

    @patch("bimanual_vla.device_guard.check_devices")
    def test_runtime_recheck_latches_role_change_off_control_thread(self, check):
        baseline = ResolvedDevices(
            arms={"right": "can1"},
            cameras={"overhead": "/dev/video4", "right_wrist": "/dev/video8"},
            observed_at="2026-09-29T08:00:00Z",
        )
        called = threading.Event()

        def changed_mapping(**_kwargs):
            self.assertEqual(threading.current_thread().name, "rlsok-runtime-check")
            called.set()
            return ResolvedDevices(
                arms={"right": "can1"},
                cameras={"overhead": "/dev/video9", "right_wrist": "/dev/video8"},
                observed_at="2026-09-29T08:00:05Z",
            )

        check.side_effect = changed_mapping
        guard = RuntimeDeviceGuard(
            baseline=baseline, purpose="deployment", arm_mode="single",
            arm_side="right", arms={"right": "auto"},
            cameras={"overhead": "auto", "right_wrist": "auto"},
            resolver="/tmp/example-resolver", interval_s=0.01,
        )
        guard.start()
        try:
            self.assertTrue(called.wait(1.0))
            deadline = time.monotonic() + 1.0
            while guard.fault is None and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIn("endpoint changed", guard.fault)
        finally:
            guard.close()

    @patch("bimanual_vla.device_guard.check_devices")
    def test_runtime_recheck_denial_latches_without_serial_details(self, check):
        check.side_effect = DeviceCheckRejected("RLSOK NEEDS_MATERIAL: MISSING_CAMERA")
        guard = RuntimeDeviceGuard(
            baseline=ResolvedDevices({"right": "can1"}, {"overhead": "/dev/video4"}, "now"),
            purpose="deployment", arm_mode="single", arm_side="right",
            arms={"right": "auto"}, cameras={"overhead": "auto"},
            resolver="/tmp/example-resolver",
        )
        guard.check_once()
        self.assertEqual(guard.fault, "RLSOK NEEDS_MATERIAL: MISSING_CAMERA")

    def _check(self):
        return check_devices(
            purpose="collection", arm_mode="single", arm_side="right",
            arms={"right": "auto"},
            cameras={"overhead": "auto", "right_wrist": "auto"},
            resolver="/tmp/example-resolver",
        )

    @patch("bimanual_vla.device_guard.subprocess.run")
    def test_resolved_locators_are_returned_from_fresh_inventory(self, run):
        run.side_effect = _reply_for_call
        result = self._check()
        self.assertEqual(result.arms, {"right": "can1"})
        self.assertEqual(result.cameras["overhead"], "/dev/video4")
        request = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(request["requested"]["arms"], {"right": "auto"})
        self.assertEqual(request["purpose"], "collection")

    @patch("bimanual_vla.device_guard.subprocess.run")
    def test_denial_and_stale_evidence_fail_closed(self, run):
        run.side_effect = lambda *_args, **kwargs: _reply(
            request_id=json.loads(kwargs["input"])["request_id"],
            review_decision="NEEDS_MATERIAL", reason_code="AMBIGUOUS_CAMERA",
        )
        with self.assertRaisesRegex(DeviceCheckRejected, "AMBIGUOUS_CAMERA"):
            self._check()
        run.side_effect = lambda *_args, **kwargs: _reply(
            request_id=json.loads(kwargs["input"])["request_id"],
            review_decision="REVIEW_REQUIRED", reason_code="ROLE_CHANGED",
        )
        with self.assertRaisesRegex(DeviceCheckRejected, "REVIEW_REQUIRED: ROLE_CHANGED"):
            self._check()
        run.side_effect = lambda *_args, **kwargs: _reply(
            request_id=json.loads(kwargs["input"])["request_id"],
            review_decision="allow",
        )
        with self.assertRaisesRegex(DeviceCheckRejected, "invalid or stale evidence"):
            self._check()
        run.side_effect = lambda *_args, **kwargs: _reply(
            request_id=json.loads(kwargs["input"])["request_id"],
            hardware_dispatch=True,
        )
        with self.assertRaisesRegex(DeviceCheckRejected, "invalid or stale evidence"):
            self._check()
        run.side_effect = lambda *_args, **kwargs: _reply(
            request_id=json.loads(kwargs["input"])["request_id"],
            observed_at=(datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(),
        )
        with self.assertRaisesRegex(DeviceCheckRejected, "stale evidence"):
            self._check()
        run.side_effect = None
        run.return_value = _reply(request_id="earlier-request")
        with self.assertRaisesRegex(DeviceCheckRejected, "invalid or stale evidence"):
            self._check()
        run.side_effect = _reply_for_call
        with self.assertRaisesRegex(DeviceCheckRejected, "invalid or stale evidence"):
            check_devices(
                purpose="collection", arm_mode="single", arm_side="right",
                arms={"right": "can0"},
                cameras={"overhead": "auto", "right_wrist": "auto"},
                resolver="/tmp/example-resolver",
            )
        run.side_effect = lambda *_args, **kwargs: _reply(
            request_id=json.loads(kwargs["input"])["request_id"],
            inventory={"serial": "must-not-leak"},
        )
        with self.assertRaisesRegex(DeviceCheckRejected, "invalid or stale evidence"):
            self._check()

    @patch("bimanual_vla.collection.session.check_devices")
    def test_collection_denial_precedes_can_and_camera_open(self, check):
        check.side_effect = DeviceCheckRejected("RLSOK rejected device check: MISSING_CAMERA")
        connect = Mock()
        camera_factory = Mock()
        with tempfile.TemporaryDirectory() as directory:
            session = CollectionSession(
                CollectionConfig(output_dir=Path(directory), can_name="auto", rlsok_resolver="/tmp/example-resolver"),
                robot_connect=connect, camera_factory=camera_factory,
            )
            with self.assertRaises(DeviceCheckRejected):
                session.connect()
        connect.assert_not_called()
        camera_factory.assert_not_called()

    @patch("bimanual_vla.collection.session.check_devices")
    def test_collection_uses_reviewed_locator_before_can_open(self, check):
        check.return_value = ResolvedDevices(
            arms={"right": "can1"},
            cameras={"overhead": "/dev/video4", "right_wrist": "/dev/video8"},
            observed_at=datetime.now(timezone.utc).isoformat(),
        )
        connect = Mock(side_effect=RuntimeError("synthetic stop before opening a device"))
        camera_factory = Mock()
        with tempfile.TemporaryDirectory() as directory:
            session = CollectionSession(
                CollectionConfig(output_dir=Path(directory), can_name="auto", rlsok_resolver="/tmp/example-resolver"),
                robot_connect=connect, camera_factory=camera_factory,
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                session.connect()
            self.assertEqual(session.config.cam_high_device, "auto")
        connect.assert_called_once_with("can1")
        camera_factory.assert_not_called()

    @patch("bimanual_vla.device_guard.check_devices")
    @patch("bimanual_vla.deployment.client.connect_piper")
    def test_deployment_denial_precedes_can_open(self, connect, check):
        check.side_effect = DeviceCheckRejected("RLSOK rejected device check: ROLE_CHANGED")
        args = SimpleNamespace(
            arm_mode="single", arm_side="right", can="can0", left_can="can0", right_can="can1",
            cam_high_device="auto", cam_wrist_device="auto",
            cam_left_wrist_device="auto", cam_right_wrist_device="auto",
            rlsok_resolver="/tmp/example-resolver",
        )
        with self.assertRaises(DeviceCheckRejected):
            run_rtc_client(args)
        connect.assert_not_called()

    @patch("bimanual_vla.collection.gui.subprocess.Popen")
    @patch("bimanual_vla.collection.gui.messagebox.showerror")
    @patch("bimanual_vla.collection.gui.check_devices")
    def test_gui_denial_returns_to_device_settings_without_child(self, check, showerror, popen):
        check.side_effect = DeviceCheckRejected("RLSOK rejected device check: ROLE_CHANGED")
        gui = CollectorGUI.__new__(CollectorGUI)
        gui._inference_running = Mock(return_value=False)
        gui._can_activation_running = Mock(return_value=False)
        gui._validate_inference_settings = Mock(return_value=(["python", "-m", "client"], "server"))
        gui.recording = False
        gui.piper = None
        gui.arm_mode_var = SimpleNamespace(get=lambda: "single")
        gui.arm_side_var = SimpleNamespace(get=lambda: "right")
        for name, value in (
            ("can_var", "can0"), ("left_can_var", "can0"), ("right_can_var", "can1"),
            ("high_var", "auto"), ("wrist_var", "auto"),
            ("left_wrist_var", "auto"), ("right_wrist_var", "auto"),
            ("rlsok_resolver_var", "/tmp/example-resolver"),
        ):
            setattr(gui, name, SimpleNamespace(get=lambda value=value: value))
        gui.inference_manual_trigger_path = None
        gui.inference_status_var = Mock()

        gui.start_inference()

        popen.assert_not_called()
        gui.inference_status_var.set.assert_called_once_with(
            "RLSOK rejected device check: ROLE_CHANGED"
        )
        self.assertIn("Open Device settings", showerror.call_args.args[1])

    @patch("bimanual_vla.collection.gui.messagebox.showerror")
    @patch("bimanual_vla.collection.gui.CollectionSession")
    def test_collection_gui_denial_stays_disconnected(self, session_type, showerror):
        session_type.return_value.connect.side_effect = DeviceCheckRejected(
            "RLSOK rejected device check: MISSING_CAMERA"
        )
        gui = CollectorGUI.__new__(CollectorGUI)
        gui._can_activation_running = Mock(return_value=False)
        gui.piper = None
        gui.root = Mock()
        gui.status_var = Mock()
        gui.activate_can_button = Mock()
        gui._cleanup_devices = Mock()
        gui._set_connection_config_enabled = Mock()
        for name, value in (
            ("fps_var", "20"), ("camera_fps_var", "20"),
            ("can_var", "can0"), ("left_can_var", "can0"), ("right_can_var", "can1"),
            ("high_var", "auto"), ("wrist_var", "auto"),
            ("left_wrist_var", "auto"), ("right_wrist_var", "auto"),
            ("rlsok_resolver_var", "/tmp/example-resolver"),
            ("arm_mode_var", "single"), ("arm_side_var", "right"),
            ("schema_var", DELIVERY_SCHEMA),
        ):
            setattr(gui, name, SimpleNamespace(get=lambda value=value: value))
        with tempfile.TemporaryDirectory() as directory:
            gui.out_var = SimpleNamespace(get=lambda: directory)
            gui.dataset_name_var = SimpleNamespace(get=lambda: "episodes")
            gui.toggle_connection()

        session_type.return_value.connect.assert_called_once()
        gui._cleanup_devices.assert_called_once()
        gui._set_connection_config_enabled.assert_called_once_with(True)
        self.assertEqual(showerror.call_args.args[0], "Device check rejected")
        self.assertIn("Open Device settings", showerror.call_args.args[1])

    @patch.object(sys, "argv", [
        "rtc-client", "--arm-mode", "bimanual", "--arm-side", "both",
        "--cam-high-device", "auto", "--cam-left-wrist-device", "auto",
        "--cam-right-wrist-device", "auto", "--instruction", "test",
        "--rlsok-resolver", "/tmp/example-resolver",
    ])
    @patch("bimanual_vla.deployment.client.run_rtc_client")
    def test_direct_client_accepts_auto_roles_for_resolver(self, run):
        rtc_client.main()
        args = run.call_args.args[0]
        self.assertEqual(args.arm_mode, "bimanual")
        self.assertEqual(args.rlsok_resolver, "/tmp/example-resolver")
        self.assertEqual(args.cam_left_wrist_device, "auto")


if __name__ == "__main__":
    unittest.main()
