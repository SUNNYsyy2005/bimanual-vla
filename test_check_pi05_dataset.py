from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from check_pi05_dataset import check_dataset, check_npz
from export_lerobot import inspect_npz_episode
from pi0_dataset import DELIVERY_LEGACY_ACTION_SEMANTICS, Pi0LeRobotDatasetWriter, default_eef_names
from piper_data_contract import EpisodeContract, build_legacy_delivery_step_actions


def valid_states(frames: int) -> np.ndarray:
    states = np.zeros((frames, 10), dtype=np.float32)
    states[:, 0] = np.arange(frames) * 0.01
    states[:, 3] = 1.0
    states[:, 7] = 1.0
    states[:, 9] = 0.5
    return states


class CheckPi05DatasetTest(unittest.TestCase):
    def test_npz_rejects_nan_and_non_monotonic_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
            states = valid_states(3)
            actions = states.copy()
            actions[1, 0] = np.nan
            path = root / "bad.npz"
            np.savez_compressed(
                path,
                state=states,
                actions=actions,
                timestamps=np.array([1.0, 1.1, 1.05]),
                images_cam_high=images,
                images_cam_right_wrist=images,
                instruction=np.asarray("bad"),
                success=np.asarray(True),
                schema=np.asarray("delivery"),
                action_source=np.asarray("same_step_slave_command"),
                action_alignment=np.asarray("same_step_command"),
                action_offset=np.asarray(0),
            )
            errors = check_npz(path)
            self.assertTrue(any("NaN/Inf" in error for error in errors))

    def test_metadata_free_legacy_npz_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
            states = valid_states(3)
            timestamps = np.array([1.0, 1.05, 1.1])
            path = root / "legacy.npz"
            np.savez_compressed(
                path,
                state=states,
                actions=build_legacy_delivery_step_actions(states),
                timestamps=timestamps,
                image=images,
                wrist_image=images,
                image_timestamps_cam_high=timestamps,
                image_timestamps_cam_wrist=timestamps,
                instruction=np.asarray("legacy"),
                success=np.asarray(True),
            )
            self.assertEqual(check_npz(path), [])
            self.assertEqual(inspect_npz_episode(path)["legacy_format"], "legacy_v2")

    def test_complete_canonical_and_legacy_lerobot_datasets_validate(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "dataset"
                if legacy:
                    state_names, action_names = default_eef_names(
                        arm_mode="single", arm_side="right", legacy=True
                    )
                    writer = Pi0LeRobotDatasetWriter(
                        root,
                        fps=20,
                        robot_type="piper_single_arm_right",
                        state_names=state_names,
                        action_names=action_names,
                        camera_keys=["cam_high", "cam_wrist"],
                        image_hw=(16, 16),
                        schema="delivery",
                        arm_mode="single",
                        arm_side="right",
                        action_semantics=DELIVERY_LEGACY_ACTION_SEMANTICS,
                        action_source="next_measured_eef",
                        action_alignment="next_observation",
                        action_offset=1,
                    )
                else:
                    contract = EpisodeContract(
                        schema="delivery",
                        arm_mode="single",
                        arm_side="right",
                        action_source="next_measured_eef_fallback",
                        action_alignment="next_observation",
                        action_offset=1,
                    )
                    writer = Pi0LeRobotDatasetWriter(
                        root,
                        fps=20,
                        robot_type=contract.robot_type,
                        state_names=list(contract.state_names),
                        action_names=list(contract.action_names),
                        camera_keys=list(contract.camera_keys),
                        image_hw=(16, 16),
                        schema=contract.schema,
                        arm_mode=contract.arm_mode,
                        arm_side=contract.arm_side,
                        action_source=contract.action_source,
                        action_alignment=contract.action_alignment,
                        action_offset=contract.action_offset,
                    )
                states = valid_states(3)
                actions = (
                    build_legacy_delivery_step_actions(states)
                    if legacy
                    else states[np.minimum(np.arange(3) + 1, 2)]
                )
                images = {
                    key: np.zeros((3, 16, 16, 3), dtype=np.uint8)
                    for key in writer.camera_keys
                }
                writer.append_episode(
                    states=states,
                    actions=actions,
                    timestamps=np.array([1.0, 1.05, 1.1]),
                    images=images,
                    task_name="test",
                    instruction="test",
                )
                self.assertEqual(check_dataset(root), [])


if __name__ == "__main__":
    unittest.main()
