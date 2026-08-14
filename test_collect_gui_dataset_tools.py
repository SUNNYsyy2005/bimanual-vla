from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from camera import resolve_video_device, select_video_device
from collect_gui import (
    CollectorGUI,
    activate_can_interfaces,
    build_inference_bridge_command,
    build_can_activation_command,
    build_dataset_tool_command,
    discover_dataset_names,
    episode_list_values,
    format_arm_state_rows,
    letterbox_preview_frame,
    load_gui_preferences,
    move_episodes_to_trash,
    parse_can_link_status,
    save_gui_preferences,
    summarize_dataset_directory,
)
from collection_session import SessionState
from piper_data_contract import BIMANUAL, DELIVERY_SCHEMA, JOINT_SCHEMA, SINGLE_ARM


class GuiPreferencesTest(unittest.TestCase):
    def test_preferences_round_trip_with_user_only_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "preferences.json"
            values = {
                "remember_can_password": True,
                "can_admin_password": "secret",
                "remember_upload_token": True,
                "upload_token": "token-value",
            }

            save_gui_preferences(values, path)

            self.assertEqual(load_gui_preferences(path), values)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_invalid_preferences_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.json"
            path.write_text("not json", encoding="utf-8")

            self.assertEqual(load_gui_preferences(path), {})


class SpaceShortcutTest(unittest.TestCase):
    def test_episode_action_runs_once_on_space_release(self):
        gui = CollectorGUI.__new__(CollectorGUI)
        calls = []
        gui.root = SimpleNamespace(_w=".", after_idle=lambda callback: callback())
        gui._space_pressed = False
        gui._space_action_pending = False
        gui.recording = False
        gui.session = SimpleNamespace(state=SessionState.READY)
        gui.start_episode = lambda: calls.append("start")
        gui.stop_episode = lambda: calls.append("stop")
        gui.status_var = SimpleNamespace(set=lambda _value: None)
        widget = SimpleNamespace(
            winfo_toplevel=lambda: SimpleNamespace(_w="."),
            winfo_class=lambda: "TButton",
        )
        event = SimpleNamespace(widget=widget)

        self.assertEqual(gui._handle_space_key(event), "break")
        self.assertEqual(calls, [])
        self.assertEqual(gui._handle_space_release(event), "break")
        self.assertEqual(calls, ["start"])
        self.assertFalse(gui._space_pressed)


class CanActivationTest(unittest.TestCase):
    def test_activation_command_keeps_password_out_of_arguments(self):
        command = build_can_activation_command(
            "can0",
            "3-1.4:1.0",
            helper_path="/sdk/can_activate.sh",
        )
        self.assertEqual(
            command,
            [
                "sudo",
                "-S",
                "-p",
                "",
                "bash",
                "/sdk/can_activate.sh",
                "can0",
                "1000000",
                "3-1.4:1.0",
            ],
        )

    def test_activation_uses_current_usb_mapping_when_ports_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "can_activate.sh"
            helper.write_text("#!/bin/sh\n", encoding="utf-8")
            helper.chmod(0o755)
            bus_by_name = {"can0": "3-5.3:1.0", "can1": "3-5.4:1.0"}

            def runner(command, **kwargs):
                if command[:6] == ["ip", "-brief", "link", "show", "type", "can"]:
                    return subprocess.CompletedProcess(command, 0, "can0 DOWN\ncan1 DOWN\n", "")
                if command[:2] == ["ethtool", "-i"]:
                    address = bus_by_name[command[2]]
                    return subprocess.CompletedProcess(command, 0, f"bus-info: {address}\n", "")
                if command[0] == "sudo":
                    if command[4] == "bash":
                        target_name = command[-3]
                        address = command[-1]
                        current_name = next(
                            name for name, current_address in bus_by_name.items()
                            if current_address == address
                        )
                        bus_by_name[target_name] = bus_by_name.pop(current_name)
                    return subprocess.CompletedProcess(command, 0, "activated\n", "")
                if command[:4] == ["ip", "-details", "link", "show"]:
                    name = command[4]
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        f"17: {name}: <NOARP,UP,LOWER_UP,ECHO> mtu 16\n"
                        "    can state ERROR-ACTIVE\n"
                        "      bitrate 1000000 sample-point 0.750\n",
                        "",
                    )
                raise AssertionError(command)

            statuses = activate_can_interfaces(
                ("can0", "can1"),
                "secret",
                helper_path=helper,
                runner=runner,
            )

            self.assertEqual(statuses["can0"]["bus_info"], "3-5.3:1.0")
            self.assertEqual(statuses["can1"]["bus_info"], "3-5.4:1.0")

    def test_link_status_parses_up_and_bitrate(self):
        status = parse_can_link_status(
            "17: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16\n"
            "    can state ERROR-ACTIVE\n"
            "      bitrate 1000000 sample-point 0.750\n",
            "can0",
        )
        self.assertTrue(status["up"])
        self.assertEqual(status["bitrate"], 1_000_000)

    def test_two_interfaces_use_their_own_usb_addresses_and_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "can_activate.sh"
            helper.touch()
            calls = []
            bus_by_name = {"can0": "3-1.3:1.0", "can1": "3-1.4:1.0"}

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                if command == ["ip", "-brief", "link", "show", "type", "can"]:
                    output = "\n".join(f"{name} DOWN" for name in bus_by_name) + "\n"
                    return subprocess.CompletedProcess(command, 0, output, "")
                if command[:2] == ["ethtool", "-i"]:
                    address = bus_by_name[command[2]]
                    return subprocess.CompletedProcess(command, 0, f"bus-info: {address}\n", "")
                if command[0] == "sudo":
                    self.assertEqual(kwargs["input"], "secret\n")
                    self.assertNotIn("secret", command)
                    if command[4:7] == ["ip", "link", "set"] and "name" in command:
                        old_name = command[7]
                        new_name = command[9]
                        bus_by_name[new_name] = bus_by_name.pop(old_name)
                    elif command[4] == "bash":
                        target_name = command[-3]
                        address = command[-1]
                        current_name = next(
                            name for name, current_address in bus_by_name.items()
                            if current_address == address
                        )
                        bus_by_name[target_name] = bus_by_name.pop(current_name)
                    return subprocess.CompletedProcess(command, 0, "activated\n", "")
                if command[:4] == ["ip", "-details", "link", "show"]:
                    name = command[4]
                    output = (
                        f"17: {name}: <NOARP,UP,LOWER_UP,ECHO> mtu 16\n"
                        "    can state ERROR-ACTIVE\n"
                        "      bitrate 1000000 sample-point 0.750\n"
                    )
                    return subprocess.CompletedProcess(command, 0, output, "")
                raise AssertionError(command)

            statuses = activate_can_interfaces(
                ("can0", "can1"),
                "secret",
                expected_bus_info={"can0": "3-1.4:1.0", "can1": "3-1.3:1.0"},
                helper_path=helper,
                runner=runner,
            )

            self.assertEqual(statuses["can0"]["bus_info"], "3-1.4:1.0")
            self.assertEqual(statuses["can1"]["bus_info"], "3-1.3:1.0")
            sudo_commands = [command for command, _ in calls if command[0] == "sudo"]
            activation_commands = [command for command in sudo_commands if command[4] == "bash"]
            self.assertEqual(activation_commands[0][-1], "3-1.4:1.0")
            self.assertEqual(activation_commands[1][-1], "3-1.3:1.0")
            self.assertEqual(bus_by_name, {"can0": "3-1.4:1.0", "can1": "3-1.3:1.0"})


class InferenceCommandTest(unittest.TestCase):
    def _command(self, **overrides):
        values = dict(
            python_executable="/env/bin/python",
            script_path="/project/robot_observation_bridge.py",
            host="192.168.101.9",
            port=8099,
            hz=4,
            arm_mode=BIMANUAL,
            arm_side="both",
            can="can0",
            left_can="can0",
            right_can="can1",
            cam_high_device="/dev/video4",
            cam_wrist_device="auto",
            cam_left_wrist_device="/dev/video24",
            cam_right_wrist_device="/dev/video12",
            instruction="put bottles on the black site",
            allow_execution=True,
        )
        values.update(overrides)
        return build_inference_bridge_command(**values)

    def test_bimanual_command_defaults_to_execution_enabled(self):
        command = self._command()
        self.assertIn("--allow-execution", command)
        self.assertEqual(command[command.index("--left-can") + 1], "can0")
        self.assertEqual(command[command.index("--right-can") + 1], "can1")

    def test_execution_flag_can_be_disabled(self):
        self.assertNotIn("--allow-execution", self._command(allow_execution=False))

    def test_bimanual_rejects_duplicate_devices(self):
        with self.assertRaises(ValueError):
            self._command(right_can="can0")
        with self.assertRaises(ValueError):
            self._command(cam_right_wrist_device="/dev/video24")

    def test_auto_camera_selectors_are_resolved_by_camera_capture(self):
        command = self._command(
            cam_high_device="auto",
            cam_left_wrist_device="auto",
            cam_right_wrist_device="auto",
        )
        self.assertEqual(command[command.index("--cam-high-device") + 1], "auto")

    def test_single_arm_requires_left_or_right(self):
        with self.assertRaises(ValueError):
            self._command(arm_mode=SINGLE_ARM, arm_side="both")


class EpisodeTrashTest(unittest.TestCase):
    def test_selected_episodes_are_moved_to_recoverable_trash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episodes"
            output.mkdir()
            first = output / "ep_0001.npz"
            second = output / "ep_0002.npz"
            first.write_bytes(b"one")
            second.write_bytes(b"two")

            moved = move_episodes_to_trash(
                output,
                [first, second],
                timestamp="20260803-120000",
            )

            self.assertEqual(
                moved,
                [
                    output / ".trash" / "20260803-120000" / first.name,
                    output / ".trash" / "20260803-120000" / second.name,
                ],
            )
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertEqual(moved[0].read_bytes(), b"one")
            self.assertEqual(moved[1].read_bytes(), b"two")

    def test_path_outside_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "episodes"
            output.mkdir()
            outside = root / "ep_0001.npz"
            outside.write_bytes(b"unsafe")

            with self.assertRaisesRegex(ValueError, "unsafe episode path"):
                move_episodes_to_trash(output, [outside], timestamp="fixed")


class DatasetCommandTest(unittest.TestCase):
    def test_prepare_command_has_no_server_token_requirement(self):
        command = build_dataset_tool_command(
            python_executable="/env/bin/python",
            script_path="upload_dataset_4090.py",
            source_dir="episodes",
            dataset_name="pick_cube_v1",
            fps=20,
            action="prepare",
            rebuild=True,
        )

        self.assertIn("--prepare-only", command)
        self.assertIn("--rebuild", command)
        self.assertNotIn("--server", command)
        self.assertNotIn("--token", command)


class DatasetSummaryTest(unittest.TestCase):
    def test_summary_counts_real_frames_and_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, success in enumerate((True, False)):
                np.savez(
                    root / f"ep_{index:04d}.npz",
                    state=np.zeros((4, 7), dtype=np.float32),
                    success=np.asarray(success, dtype=np.bool_),
                )
            summary = summarize_dataset_directory(root)
            self.assertEqual(summary["episodes"], 2)
            self.assertEqual(summary["frames"], 6)
            self.assertEqual(summary["success"], 1)
            self.assertEqual(summary["failure"], 1)

    def test_upload_command_uses_merge_without_exposing_token(self):
        command = build_dataset_tool_command(
            python_executable="/env/bin/python",
            script_path="upload_dataset_4090.py",
            source_dir="episodes",
            dataset_name="pick_cube_v1",
            fps=20,
            action="upload",
            server="http://192.168.101.9:8090",
            workers=4,
            install_mode="merge",
        )

        self.assertIn("--server", command)
        self.assertIn("--workers", command)
        self.assertIn("--merge", command)
        self.assertNotIn("--token", command)


class DatasetDiscoveryTest(unittest.TestCase):
    def test_existing_root_and_child_datasets_are_listed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "episodes_piper_v21"
            root.mkdir()
            (root / "ep_0000.npz").touch()
            (root / "dual_arm").mkdir()
            (root / ".trash").mkdir()

            self.assertEqual(
                discover_dataset_names(root),
                ("episodes_piper_v21", "dual_arm"),
            )


class EpisodeListDisplayTest(unittest.TestCase):
    def test_episode_row_uses_dataset_and_metadata_without_exposing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0012.npz"
            np.savez(
                path,
                state=np.zeros((2, 7), dtype=np.float32),
                task=np.asarray("pick_cube"),
                success=np.asarray(True),
            )

            values = episode_list_values(path, "dual_arm")

            self.assertEqual(
                values,
                ("dual_arm", "Episode 0012"),
            )
            self.assertNotIn(str(path.parent), " ".join(values))


class ArmStateFormattingTest(unittest.TestCase):
    def test_joint_state_is_split_into_left_and_right_rows(self):
        state = np.arange(14, dtype=np.float32) / 10.0

        rows = format_arm_state_rows(
            state,
            schema=JOINT_SCHEMA,
            arm_mode=BIMANUAL,
            arm_side="both",
        )

        self.assertEqual(tuple(side for side, _values in rows), ("left", "right"))
        np.testing.assert_allclose(rows[0][1], state[:7])
        np.testing.assert_allclose(rows[1][1], state[7:])

    def test_delivery_state_displays_xyz_rpy_and_gripper(self):
        identity_rotation6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        state = np.concatenate(
            (
                np.array([0.1, -0.2, 0.3], dtype=np.float32),
                identity_rotation6d,
                np.array([0.75], dtype=np.float32),
            )
        )

        rows = format_arm_state_rows(
            state,
            schema=DELIVERY_SCHEMA,
            arm_mode=SINGLE_ARM,
            arm_side="left",
        )

        self.assertEqual(rows[0][0], "left")
        np.testing.assert_allclose(rows[0][1], [0.1, -0.2, 0.3, 0, 0, 0, 0.75])


class PreviewLayoutTest(unittest.TestCase):
    def test_preview_uses_requested_rectangular_canvas_without_stretching(self):
        frame = np.full((180, 320, 3), 255, dtype=np.uint8)

        preview = letterbox_preview_frame(
            frame,
            target_hw=(300, 500),
            source_aspect=16 / 9,
        )

        self.assertEqual(preview.shape, (300, 500, 3))
        self.assertTrue(np.all(preview[0] == 0))
        self.assertTrue(np.all(preview[150] == 255))


class VideoDeviceDisplayTest(unittest.TestCase):
    def test_integer_camera_id_is_displayed_as_video_device(self):
        self.assertEqual(resolve_video_device(8), "/dev/video8")

    def test_stable_symlink_is_resolved_to_numeric_video_device(self):
        with tempfile.TemporaryDirectory() as directory:
            stable = Path(directory) / "camera-by-path"
            stable.symlink_to("/dev/video17")
            self.assertEqual(resolve_video_device(stable), "/dev/video17")

    def test_existing_camera_selector_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            camera = Path(directory) / "video-test"
            camera.touch()
            self.assertEqual(select_video_device("cam_high", camera), str(camera))


if __name__ == "__main__":
    unittest.main()
