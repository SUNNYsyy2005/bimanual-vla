"""Exercise the real resolver IPC and project pre-open paths without hardware."""

from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bimanual_vla.collection.gui import CollectorGUI
from bimanual_vla.collection.session import CollectionConfig, CollectionSession
from bimanual_vla.deployment.client import run_rtc_client
from bimanual_vla.device_guard import (
    DeviceCheckRejected,
    RuntimeDeviceGuard,
    check_devices,
)


_FIXTURE = Path(__file__).parent / "fixtures" / "fake_rlsok_resolver.py"
_MODE_ENV = "BIMANUAL_VLA_TEST_RLSOK_MODE"


class OfflineRLSOKIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory(prefix="bimanual-rlsok-offline-")
        cls.resolver = Path(cls._directory.name) / "fake-resolver"
        cls.resolver.write_text(
            f"#!{sys.executable}\n" + _FIXTURE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        cls.resolver.chmod(0o700)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _check(self, *, arm_mode: str = "single"):
        bimanual = arm_mode == "bimanual"
        return check_devices(
            purpose="deployment",
            arm_mode=arm_mode,
            arm_side="both" if bimanual else "right",
            arms={"left": "auto", "right": "auto"} if bimanual else {"right": "auto"},
            cameras=(
                {"overhead": "auto", "left_wrist": "auto", "right_wrist": "auto"}
                if bimanual else {"overhead": "auto", "right_wrist": "auto"}
            ),
            resolver=str(self.resolver),
        )

    def test_real_ipc_resolves_single_and_bimanual_roles(self):
        with patch.dict(os.environ, {_MODE_ENV: "UNCHANGED"}):
            single = self._check()
            bimanual = self._check(arm_mode="bimanual")
        self.assertEqual(single.arms, {"right": "can1"})
        self.assertEqual(single.cameras["right_wrist"], "/dev/video8")
        self.assertEqual(bimanual.arms, {"left": "can0", "right": "can1"})
        self.assertEqual(set(bimanual.cameras), {"overhead", "left_wrist", "right_wrist"})

    def test_review_outcomes_and_invalid_evidence_refuse(self):
        for mode, expected in (
            ("NEEDS_MATERIAL", "NEEDS_MATERIAL: MISSING_CAMERA"),
            ("REVIEW_REQUIRED", "REVIEW_REQUIRED: ROLE_CHANGED"),
            ("STALE", "invalid or stale evidence"),
            ("BAD_NONCE", "invalid or stale evidence"),
        ):
            with self.subTest(mode=mode), patch.dict(os.environ, {_MODE_ENV: mode}):
                with self.assertRaisesRegex(DeviceCheckRejected, expected):
                    self._check()

    def test_collection_refusal_precedes_hardware_factories(self):
        robot_connect = Mock(side_effect=AssertionError("CAN must stay closed"))
        camera_factory = Mock(side_effect=AssertionError("camera must stay closed"))
        with tempfile.TemporaryDirectory() as directory:
            session = CollectionSession(
                CollectionConfig(
                    output_dir=Path(directory),
                    can_name="auto",
                    rlsok_resolver=str(self.resolver),
                ),
                robot_connect=robot_connect,
                camera_factory=camera_factory,
            )
            with patch.dict(os.environ, {_MODE_ENV: "NEEDS_MATERIAL"}):
                with self.assertRaisesRegex(DeviceCheckRejected, "NEEDS_MATERIAL"):
                    session.connect()
        robot_connect.assert_not_called()
        camera_factory.assert_not_called()

    def test_collection_passes_reviewed_locator_to_mock_can_open(self):
        robot_connect = Mock(side_effect=RuntimeError("offline boundary reached"))
        camera_factory = Mock(side_effect=AssertionError("camera must stay closed"))
        with tempfile.TemporaryDirectory() as directory:
            session = CollectionSession(
                CollectionConfig(
                    output_dir=Path(directory),
                    can_name="auto",
                    rlsok_resolver=str(self.resolver),
                ),
                robot_connect=robot_connect,
                camera_factory=camera_factory,
            )
            with patch.dict(os.environ, {_MODE_ENV: "UNCHANGED"}):
                with self.assertRaisesRegex(RuntimeError, "offline boundary reached"):
                    session.connect()
        robot_connect.assert_called_once_with("can1")
        camera_factory.assert_not_called()

    def test_inference_refusal_precedes_can_open(self):
        args = SimpleNamespace(
            arm_mode="single", arm_side="right", can="auto",
            left_can="can0", right_can="can1",
            cam_high_device="auto", cam_wrist_device="auto",
            cam_left_wrist_device="auto", cam_right_wrist_device="auto",
            rlsok_resolver=str(self.resolver),
        )
        with patch.dict(os.environ, {_MODE_ENV: "REVIEW_REQUIRED"}):
            with patch("bimanual_vla.deployment.client.connect_piper") as connect:
                with self.assertRaisesRegex(DeviceCheckRejected, "REVIEW_REQUIRED"):
                    run_rtc_client(args)
                connect.assert_not_called()

    def test_gui_refusal_precedes_child_launch(self):
        gui = CollectorGUI.__new__(CollectorGUI)
        gui._inference_running = Mock(return_value=False)
        gui._can_activation_running = Mock(return_value=False)
        gui._validate_inference_settings = Mock(
            return_value=(["__offline_test_must_not_launch__"], "offline policy")
        )
        gui.recording = False
        gui.piper = None
        gui.arm_mode_var = SimpleNamespace(get=lambda: "single")
        gui.arm_side_var = SimpleNamespace(get=lambda: "right")
        for name, value in (
            ("can_var", "auto"), ("left_can_var", "can0"), ("right_can_var", "can1"),
            ("high_var", "auto"), ("wrist_var", "auto"),
            ("left_wrist_var", "auto"), ("right_wrist_var", "auto"),
            ("rlsok_resolver_var", str(self.resolver)),
        ):
            setattr(gui, name, SimpleNamespace(get=lambda value=value: value))
        gui.inference_manual_trigger_path = None
        gui.inference_status_var = Mock()
        with patch.dict(os.environ, {_MODE_ENV: "REVIEW_REQUIRED"}):
            with patch("bimanual_vla.collection.gui.messagebox.showerror") as showerror:
                gui.start_inference()
        gui.inference_status_var.set.assert_called_once_with(
            "RLSOK REVIEW_REQUIRED: ROLE_CHANGED"
        )
        self.assertEqual(showerror.call_args.args[0], "Device check rejected")

    def test_runtime_endpoint_change_latches_without_camera_or_can(self):
        with patch.dict(os.environ, {_MODE_ENV: "UNCHANGED"}):
            baseline = self._check()
        guard = RuntimeDeviceGuard(
            baseline=baseline, purpose="deployment",
            arm_mode="single", arm_side="right",
            arms={"right": "auto"},
            cameras={"overhead": "auto", "right_wrist": "auto"},
            resolver=str(self.resolver),
        )
        with patch.dict(os.environ, {_MODE_ENV: "ENDPOINT_CHANGED"}):
            guard.check_once()
        self.assertIn("endpoint changed", guard.fault)


if __name__ == "__main__":
    unittest.main()
