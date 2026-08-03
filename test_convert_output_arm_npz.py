from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from convert_output_arm_npz import load_episode, main
from piper_data_contract import EpisodeContract, build_legacy_delivery_step_actions


class ConvertOutputArmNpzTest(unittest.TestCase):
    def test_joint_measured_only_derives_next_absolute_action(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0000.npz"
            qpos = np.arange(21, dtype=np.float32).reshape(3, 7) / 100.0
            images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
            np.savez_compressed(
                path,
                qpos=qpos,
                timestamps=np.array([1.0, 1.05, 1.1]),
                images_cam_high=images,
                images_cam_right_wrist=images,
                instruction=np.asarray("joint task"),
                success=np.asarray(True),
            )
            contract = EpisodeContract(
                schema="joint",
                arm_mode="single",
                arm_side="right",
                action_source="next_measured_joint_fallback",
                action_alignment="next_observation",
                action_offset=1,
            )
            episode = load_episode(
                path,
                contract=contract,
                fps=20,
                action_offset=1,
                use_existing_actions=False,
            )
            np.testing.assert_array_equal(episode["actions"][0], qpos[1])
            np.testing.assert_array_equal(episode["actions"][-1], qpos[-1])
            self.assertEqual(set(episode["image_timestamps"]), set(contract.camera_keys))

    def test_check_only_accepts_metadata_free_legacy_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0000.npz"
            states = np.zeros((3, 10), dtype=np.float32)
            states[:, 3] = 1.0
            states[:, 7] = 1.0
            states[:, 9] = 0.5
            images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
            timestamps = np.array([1.0, 1.05, 1.1])
            np.savez_compressed(
                path,
                state=states,
                actions=build_legacy_delivery_step_actions(states),
                timestamps=timestamps,
                image=images,
                wrist_image=images,
                image_timestamps_cam_high=timestamps,
                image_timestamps_cam_wrist=timestamps,
                instruction=np.asarray("legacy task"),
                success=np.asarray(True),
            )
            with patch.object(sys, "argv", ["convert_output_arm_npz.py", str(path), "--check-only"]):
                self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
