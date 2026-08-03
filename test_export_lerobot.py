from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from export_lerobot import export_dataset, inspect_npz_episode
from piper_data_contract import EpisodeContract, build_legacy_delivery_step_actions


def eef_states(frames: int, arms: int = 1, *, closed_fraction: bool = False) -> np.ndarray:
    values = np.zeros((frames, 10 * arms), dtype=np.float32)
    for arm in range(arms):
        offset = arm * 10
        values[:, offset] = np.arange(frames, dtype=np.float32) * 0.01
        values[:, offset + 3] = 1.0
        values[:, offset + 7] = 1.0
        values[:, offset + 9] = np.linspace(0.2, 0.8, frames)
    return values


def save_delivery_npz(path: Path, *, legacy: bool) -> None:
    states = eef_states(3, closed_fraction=legacy)
    timestamps = np.array([1.0, 1.05, 1.10], dtype=np.float64)
    images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    if legacy:
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
        return
    contract = EpisodeContract(
        schema="delivery",
        arm_mode="single",
        arm_side="right",
        action_source="next_measured_eef_fallback",
        action_alignment="next_observation",
        action_offset=1,
    )
    actions = states[np.minimum(np.arange(len(states)) + 1, len(states) - 1)]
    np.savez_compressed(
        path,
        state=states,
        actions=actions,
        timestamps=timestamps,
        state_timestamp=timestamps,
        action_timestamp=timestamps + 0.001,
        images_cam_high=images,
        images_cam_right_wrist=images,
        image_timestamps_cam_high=timestamps + 0.002,
        image_timestamps_cam_right_wrist=timestamps + 0.003,
        instruction=np.asarray("canonical task"),
        success=np.asarray(True),
        **contract.metadata_payload(),
    )


class ExportLeRobotContractTest(unittest.TestCase):
    def test_metadata_free_10d_7d_is_legacy_v2_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0000.npz"
            save_delivery_npz(path, legacy=True)
            episode = inspect_npz_episode(path)
            self.assertEqual(episode["contract_format"], "legacy_v2")
            self.assertEqual((episode["state_dim"], episode["raw_action_dim"]), (10, 7))
            self.assertEqual(episode["gripper_semantics"], "absolute_closed_fraction_0_open_1_closed")
            self.assertTrue(episode["metadata"]["legacy_next_measured_verified"])

    def test_canonical_delivery_keeps_10d_absolute_action(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0000.npz"
            save_delivery_npz(path, legacy=False)
            episode = inspect_npz_episode(path)
            self.assertEqual(episode["contract_format"], "canonical")
            self.assertEqual((episode["state_dim"], episode["raw_action_dim"], episode["model_action_dim"]), (10, 10, 7))
            self.assertEqual(episode["action_semantics"], "absolute_eef_target")
            self.assertEqual(episode["gripper_semantics"], "absolute_opening_fraction_0_closed_1_open")

    def test_bimanual_metadata_free_20d_14d_is_legacy_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_0000.npz"
            states = eef_states(3, arms=2, closed_fraction=True)
            timestamps = np.array([1.0, 1.05, 1.10], dtype=np.float64)
            images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
            np.savez_compressed(
                path,
                state=states,
                actions=build_legacy_delivery_step_actions(states, arm_count=2),
                timestamps=timestamps,
                images_cam_high=images,
                images_cam_left_wrist=images,
                images_cam_right_wrist=images,
                image_timestamps_cam_high=timestamps,
                image_timestamps_cam_left_wrist=timestamps,
                image_timestamps_cam_right_wrist=timestamps,
                instruction=np.asarray("bimanual legacy task"),
                success=np.asarray(True),
            )
            episode = inspect_npz_episode(path)
            self.assertEqual(episode["arm_mode"], "bimanual")
            self.assertEqual((episode["state_dim"], episode["raw_action_dim"]), (20, 14))
            self.assertEqual(episode["contract_format"], "legacy_v2")
            self.assertEqual(len(episode["camera_keys"]), 3)

    def test_export_marks_legacy_and_canonical_contracts(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                save_delivery_npz(source / "ep_0000.npz", legacy=legacy)
                output = root / "output"
                with patch("pi0_dataset.Pi0LeRobotDatasetWriter._write_episode_videos"):
                    export_dataset(source, output, fps=20)
                info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
                self.assertEqual(info["raw_action_dim"], 7 if legacy else 10)
                self.assertEqual(info["model_action_dim"], 7)
                self.assertEqual(info["contract_format"], "legacy_v2" if legacy else "canonical")
                self.assertEqual(info["delivery_action_format"], "step_delta" if legacy else "absolute_eef_target")
                self.assertEqual(info["contract_version"], 2 if legacy else 3)


if __name__ == "__main__":
    unittest.main()
