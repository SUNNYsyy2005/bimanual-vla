from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from view_episode import discover_episode_camera_keys, load_episode, make_panel


class EpisodeCameraDiscoveryTest(unittest.TestCase):
    def test_legacy_camera_keys_are_supported(self):
        data = {"image": np.empty(0), "wrist_image": np.empty(0)}
        self.assertEqual(
            discover_episode_camera_keys(data),
            ("legacy_high", "legacy_wrist"),
        )

    def test_declared_bimanual_camera_order_is_preserved(self):
        data = {
            "camera_keys": np.asarray(
                ["cam_high", "cam_left_wrist", "cam_right_wrist"]
            ),
            "images_cam_high": np.empty(0),
            "images_cam_left_wrist": np.empty(0),
            "images_cam_right_wrist": np.empty(0),
        }
        self.assertEqual(
            discover_episode_camera_keys(data),
            ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        )


class EpisodeViewerLoadTest(unittest.TestCase):
    def test_bimanual_episode_loads_three_cameras_and_builds_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0000.npz"
            frames = np.zeros((2, 256, 256, 3), dtype=np.uint8)
            np.savez(
                path,
                state=np.zeros((2, 20), dtype=np.float32),
                joint_qpos=np.zeros((2, 14), dtype=np.float32),
                timestamps=np.asarray([0.0, 0.05]),
                task=np.asarray("pick_cube"),
                instruction=np.asarray("pick up the cube"),
                schema=np.asarray("delivery"),
                arm_mode=np.asarray("bimanual"),
                camera_keys=np.asarray(
                    ["cam_high", "cam_left_wrist", "cam_right_wrist"]
                ),
                images_cam_high=frames,
                images_cam_left_wrist=frames,
                images_cam_right_wrist=frames,
            )

            episode = load_episode(path)
            panel = make_panel(
                {key: value[0] for key, value in episode["cameras"].items()},
                episode["joint_qpos"][0],
                str(episode["task"]),
                str(episode["instruction"]),
                0,
                int(episode["frame_count"]),
            )

            self.assertEqual(
                tuple(episode["cameras"]),
                ("cam_high", "cam_left_wrist", "cam_right_wrist"),
            )
            self.assertEqual(panel.shape, (1020, 1280, 3))


if __name__ == "__main__":
    unittest.main()
