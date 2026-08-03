from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from camera import resolve_video_device
from collect_gui import build_dataset_tool_command, move_episodes_to_trash


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


class VideoDeviceDisplayTest(unittest.TestCase):
    def test_integer_camera_id_is_displayed_as_video_device(self):
        self.assertEqual(resolve_video_device(8), "/dev/video8")

    def test_stable_symlink_is_resolved_to_numeric_video_device(self):
        with tempfile.TemporaryDirectory() as directory:
            stable = Path(directory) / "camera-by-path"
            stable.symlink_to("/dev/video17")
            self.assertEqual(resolve_video_device(stable), "/dev/video17")


if __name__ == "__main__":
    unittest.main()
