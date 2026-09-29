"""RLSOK bridge checks use synthetic resolver replies; no hardware is opened."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bimanual_vla.collection.gui import CollectorGUI
from bimanual_vla.collection.session import CollectionConfig, CollectionSession
from bimanual_vla.data.contract import DELIVERY_SCHEMA
from bimanual_vla.deployment import client as rtc_client
from bimanual_vla.deployment.client import run_rtc_client
from bimanual_vla.device_guard import DeviceCheckRejected, ResolvedDevices, check_devices


def _reply(**changes):
    value = {
        "schema_version": 1,
        "rlsok_version": "1.5.8",
        "decision": "allow",
        "inventory_method": "linux-sysfs-udev",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "resolved": {
            "arms": {"right": "can1"},
            "cameras": {"overhead": "/dev/video4", "right_wrist": "/dev/v4l/by-path/example-video-index0"},
        },
    }
    value.update(changes)
    if value["decision"] == "deny":
        value.pop("inventory_method")
        value.pop("observed_at")
        value.pop("resolved")
    return subprocess.CompletedProcess(["example-resolver"], 0, json.dumps(value), "")


def _reply_for_call(*_args, **kwargs):
    request = json.loads(kwargs["input"])
    return _reply(request_id=request["request_id"])


class DeviceGuardTest(unittest.TestCase):
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
            decision="deny", reason_code="AMBIGUOUS_CAMERA",
        )
        with self.assertRaisesRegex(DeviceCheckRejected, "AMBIGUOUS_CAMERA"):
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
