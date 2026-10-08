from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from bimanual_vla.collection.camera import (
    VideoDeviceCandidate,
    _stable_video_selector,
    select_video_devices,
)


class JointCameraSelectionTest(unittest.TestCase):
    def _candidate(self, path: Path, model: str, score: int = 40) -> VideoDeviceCandidate:
        return VideoDeviceCandidate(
            device=path,
            index=int(path.name.removeprefix("video")),
            properties=model,
            format_score=score,
        )

    def test_two_identical_wrist_roles_without_topology_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "video12"
            right = root / "video22"
            left.touch()
            right.touch()
            candidates = [
                self._candidate(left, "depth_camera_405"),
                self._candidate(right, "depth_camera_405"),
            ]
            with mock.patch("bimanual_vla.collection.camera._enumerate_video_candidates", return_value=candidates):
                with self.assertRaisesRegex(RuntimeError, "Cannot safely distinguish"):
                    select_video_devices(
                        {"cam_left_wrist": "auto", "cam_right_wrist": "auto"},
                        device_root=root,
                    )

    def test_current_usb_topology_keeps_left_and_right_physical_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            port_52 = root / "video12"
            port_62 = root / "video22"
            port_52.touch()
            port_62.touch()
            candidates = [
                self._candidate(port_52, "depth_camera_405"),
                self._candidate(port_62, "depth_camera_405"),
            ]

            def selector(path: Path) -> str:
                return (
                    "usb-0:5.2:1.0-video-index4"
                    if path == port_52
                    else "usb-0:6.2:1.0-video-index4"
                )

            with mock.patch("bimanual_vla.collection.camera._enumerate_video_candidates", return_value=candidates), mock.patch(
                "bimanual_vla.collection.camera._stable_video_selector", side_effect=selector
            ):
                selected = select_video_devices(
                    {"cam_left_wrist": "auto", "cam_right_wrist": "auto"},
                    device_root=root,
                )

            self.assertEqual(selected["cam_left_wrist"], "usb-0:6.2:1.0-video-index4")
            self.assertEqual(selected["cam_right_wrist"], "usb-0:5.2:1.0-video-index4")

    def test_updated_workstation_topology_keeps_left_and_right_physical_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            port_22 = root / "video10"
            port_132 = root / "video22"
            port_22.touch()
            port_132.touch()
            candidates = [
                self._candidate(port_22, "depth_camera_405"),
                self._candidate(port_132, "depth_camera_405"),
            ]

            def selector(path: Path) -> str:
                return (
                    "usb-0:2.2:1.0-video-index4"
                    if path == port_22
                    else "usb-0:13.2:1.0-video-index4"
                )

            with mock.patch(
                "bimanual_vla.collection.camera._enumerate_video_candidates",
                return_value=candidates,
            ), mock.patch(
                "bimanual_vla.collection.camera._stable_video_selector",
                side_effect=selector,
            ):
                selected = select_video_devices(
                    {"cam_left_wrist": "auto", "cam_right_wrist": "auto"},
                    device_root=root,
                )

            self.assertEqual(selected["cam_left_wrist"], "usb-0:13.2:1.0-video-index4")
            self.assertEqual(selected["cam_right_wrist"], "usb-0:2.2:1.0-video-index4")

    def test_stale_paths_do_not_fall_back_to_ambiguous_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "video12"
            second = root / "video22"
            first.touch()
            second.touch()
            candidates = [
                self._candidate(first, "depth_camera_405"),
                self._candidate(second, "depth_camera_405"),
            ]
            with mock.patch("bimanual_vla.collection.camera._enumerate_video_candidates", return_value=candidates):
                with self.assertRaisesRegex(RuntimeError, "Cannot safely distinguish"):
                    select_video_devices(
                        {
                            "cam_left_wrist": root / "missing-left",
                            "cam_right_wrist": root / "missing-right",
                        },
                        device_root=root,
                    )

    def test_reviewed_selector_never_falls_back_after_disappearing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("bimanual_vla.collection.camera._enumerate_video_candidates") as discover:
                with self.assertRaisesRegex(RuntimeError, "reviewed selector is unavailable") as rejected:
                    select_video_devices(
                        {"cam_high": str(root / "missing-video")},
                        device_root=root,
                        strict_explicit=True,
                    )
            self.assertNotIn("missing-video", str(rejected.exception))
            discover.assert_not_called()

    def test_valid_explicit_device_is_reserved_before_auto_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "video12"
            remaining = root / "video22"
            explicit.touch()
            remaining.touch()
            candidates = [
                self._candidate(explicit, "depth_camera_405"),
                self._candidate(remaining, "depth_camera_405"),
            ]
            with mock.patch("bimanual_vla.collection.camera._enumerate_video_candidates", return_value=candidates):
                selected = select_video_devices(
                    {
                        "cam_left_wrist": str(explicit),
                        "cam_right_wrist": "auto",
                    },
                    device_root=root,
                )

            self.assertEqual(selected["cam_left_wrist"], str(explicit))
            self.assertEqual(selected["cam_right_wrist"], str(remaining))

    def test_duplicate_explicit_device_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            device = Path(directory) / "video12"
            device.touch()
            with self.assertRaisesRegex(RuntimeError, "both select"):
                select_video_devices(
                    {
                        "cam_left_wrist": str(device),
                        "cam_right_wrist": str(device),
                    },
                    device_root=Path(directory),
                )

    def test_stable_selector_prefers_by_path_over_by_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "video12"
            by_path = root / "by-path"
            by_id = root / "by-id"
            by_path.mkdir()
            by_id.mkdir()
            device.touch()
            path_link = by_path / "usb-port-video-index4"
            id_link = by_id / "generic-camera-video-index4"
            path_link.symlink_to(device)
            id_link.symlink_to(device)

            selected = _stable_video_selector(device, (by_path, by_id))

            self.assertEqual(selected, str(path_link))


if __name__ == "__main__":
    unittest.main()
