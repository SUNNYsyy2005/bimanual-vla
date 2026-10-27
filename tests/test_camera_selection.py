from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from bimanual_vla.collection.camera import (
    VideoDeviceCandidate,
    _stable_video_selector,
    select_video_devices,
    _serial_rgb_device,
)


class JointCameraSelectionTest(unittest.TestCase):
    def test_reviewed_selector_is_not_overridden_by_saved_serial_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = root / '.config/bimanual-vla'
            settings.mkdir(parents=True)
            (settings / 'camera_serials.json').write_text('{"cam_high": "STALE-SERIAL"}')
            reviewed = root / 'video4'
            reviewed.touch()
            with mock.patch('pathlib.Path.home', return_value=root), mock.patch(
                'bimanual_vla.collection.camera._serial_rgb_device'
            ) as serial_lookup:
                selected = select_video_devices({'cam_high': str(reviewed)}, strict_explicit=True)
            self.assertEqual(selected, {'cam_high': str(reviewed)})
            serial_lookup.assert_not_called()

    def test_serial_selects_rgb_not_auxiliary_or_other_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'usb_a', root / 'usb_b'
            a.mkdir(); b.mkdir()
            (a / 'serial').write_text('LEFT')
            (b / 'serial').write_text('RIGHT')
            nodes = [root / f'video{i}' for i in (6, 8, 22)]
            candidates = [self._candidate(n, 'depth_camera_405', score) for n, score in zip(nodes, (10, 35, 40))]
            with mock.patch('bimanual_vla.collection.camera._enumerate_video_candidates', return_value=candidates), mock.patch(
                'bimanual_vla.collection.camera._physical_video_device', side_effect=lambda n: a if n in nodes[:2] else b
            ):
                self.assertEqual(_serial_rgb_device('LEFT', device_root=root), str(nodes[1]))
                with self.assertRaisesRegex(RuntimeError, 'refusing camera substitution'):
                    _serial_rgb_device('MISSING', device_root=root)

    def test_different_nodes_of_one_camera_are_not_allocated_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = [root / f'video{i}' for i in (6, 8, 22)]
            for node in nodes:
                node.touch()
            candidates = [self._candidate(node, 'depth_camera_405') for node in nodes]
            identity = lambda node: root / ('usb_a' if node in nodes[:2] else 'usb_b')
            with mock.patch('bimanual_vla.collection.camera._physical_video_device', side_effect=identity), mock.patch(
                'bimanual_vla.collection.camera._enumerate_video_candidates', return_value=candidates
            ), mock.patch.dict('bimanual_vla.collection.camera.CAMERA_ROLE_PATH_HINTS',
                               {'cam_left_wrist': ('video6',), 'cam_right_wrist': ('video22',)}):
                selected = select_video_devices({'cam_left_wrist': 'auto', 'cam_right_wrist': 'auto'}, device_root=root)
                self.assertEqual(selected['cam_right_wrist'], str(nodes[2]))
                with self.assertRaisesRegex(RuntimeError, 'same physical camera'):
                    select_video_devices({'cam_left_wrist': str(nodes[0]), 'cam_right_wrist': str(nodes[1])}, device_root=root)

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

    def test_single_and_reordered_bimanual_keep_the_same_wrist_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "video8", root / "video22"]
            for path in paths:
                path.touch()
            candidates = [self._candidate(path, "depth_camera_405") for path in paths]
            with mock.patch("bimanual_vla.collection.camera._enumerate_video_candidates", return_value=candidates), mock.patch.dict(
                "bimanual_vla.collection.camera.CAMERA_ROLE_PATH_HINTS",
                {"cam_left_wrist": ("video8",), "cam_right_wrist": ("video22",)}
            ):
                pair = select_video_devices(
                    {"cam_right_wrist": "auto", "cam_left_wrist": "auto"}, device_root=root)
                for role in ("cam_left_wrist", "cam_right_wrist"):
                    self.assertEqual(select_video_devices({role: "auto"}, device_root=root),
                                     {role: pair[role]})
            self.assertEqual(pair["cam_left_wrist"], str(paths[0]))
            self.assertEqual(pair["cam_right_wrist"], str(paths[1]))

    def test_single_right_accepts_a_rig_with_only_one_wrist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "video8"
            path.touch()
            with mock.patch("bimanual_vla.collection.camera._enumerate_video_candidates",
                            return_value=[self._candidate(path, "depth_camera_405")]):
                self.assertEqual(select_video_devices({"cam_right_wrist": "auto"}, device_root=root),
                                 {"cam_right_wrist": str(path)})

    def test_hinted_right_is_reserved_before_unhinted_left(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "video8", root / "video22"]
            for path in paths:
                path.touch()
            candidates = [self._candidate(path, "depth_camera_405") for path in paths]
            with mock.patch("bimanual_vla.collection.camera._enumerate_video_candidates", return_value=candidates), \
                 mock.patch("bimanual_vla.collection.camera._stable_video_selector",
                            side_effect=lambda path: "usb-0:5.2:1.0-video-index4" if path == paths[0] else str(path)):
                pair = select_video_devices({"cam_left_wrist": "auto", "cam_right_wrist": "auto"}, device_root=root)
            self.assertEqual(pair["cam_left_wrist"], str(paths[1]))
            self.assertEqual(pair["cam_right_wrist"], "usb-0:5.2:1.0-video-index4")

    def test_unreviewed_workstation_topology_is_rejected(self):
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
                with self.assertRaisesRegex(RuntimeError, "Cannot safely distinguish"):
                    select_video_devices(
                        {"cam_left_wrist": "auto", "cam_right_wrist": "auto"},
                        device_root=root,
                    )

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
