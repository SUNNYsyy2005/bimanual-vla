from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pi0_dataset import (
    DELIVERY_LEGACY_ACTION_SEMANTICS,
    Pi0LeRobotDatasetWriter,
    default_eef_names,
)
from piper_data_contract import EpisodeContract


class Pi0LeRobotDatasetWriterContractTest(unittest.TestCase):
    def make_writer(
        self,
        root: Path,
        *,
        arm_mode: str,
        arm_side: str,
        schema: str = "joint",
    ) -> Pi0LeRobotDatasetWriter:
        contract = EpisodeContract(
            schema=schema,
            arm_mode=arm_mode,
            arm_side=arm_side,
            action_source="master_joint_feedback",
            action_alignment="same_step_command",
        )
        return Pi0LeRobotDatasetWriter(
            root,
            fps=20,
            robot_type=contract.robot_type,
            state_names=list(contract.state_names),
            action_names=list(contract.action_names),
            camera_keys=list(contract.camera_keys),
            image_hw=(8, 8),
            schema=contract.schema,
            arm_mode=contract.arm_mode,
            arm_side=contract.arm_side,
            action_source=contract.action_source,
            action_alignment=contract.action_alignment,
            save_raw_npz=True,
        )

    def append_synthetic(self, writer: Pi0LeRobotDatasetWriter) -> None:
        frames = 2
        states = np.zeros((frames, writer.contract.state_dim), dtype=np.float32)
        actions = np.zeros((frames, writer.contract.action_dim), dtype=np.float32)
        images = {
            key: np.full((frames, 8, 8, 3), index, dtype=np.uint8)
            for index, key in enumerate(writer.camera_keys)
        }
        with patch.object(writer, "_write_episode_videos"):
            writer.append_episode(
                states=states,
                actions=actions,
                timestamps=np.array([1.0, 1.05]),
                images=images,
                task_name="pick_cube",
                instruction="pick up the cube",
            )

    def test_single_arm_commanded_joint_contract_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "single"
            writer = self.make_writer(root, arm_mode="single", arm_side="right")
            self.append_synthetic(writer)
            info = json.loads((root / "meta" / "info.json").read_text())
            contract = json.loads((root / "meta" / "policy_contract.json").read_text())
            self.assertEqual(info["features"]["observation.state"]["shape"], [7])
            self.assertEqual(info["features"]["action"]["shape"], [7])
            self.assertEqual(contract["arm_mode"], "single")
            self.assertEqual(contract["action_source"], "master_joint_feedback")
            self.assertEqual(contract["action_alignment"], "same_step_command")
            self.assertEqual(contract["action_offset"], 0)
            with np.load(root / "raw" / "episode_000000.npz", allow_pickle=False) as raw:
                self.assertEqual(raw["arm_mode"].item(), "single")
                self.assertEqual(raw["action_source"].item(), "master_joint_feedback")

    def test_bimanual_joint_contract_uses_left_right_14d_and_three_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bimanual"
            writer = self.make_writer(root, arm_mode="bimanual", arm_side="both")
            self.append_synthetic(writer)
            info = json.loads((root / "meta" / "info.json").read_text())
            contract = json.loads((root / "meta" / "policy_contract.json").read_text())
            self.assertEqual(info["features"]["observation.state"]["shape"], [14])
            self.assertEqual(info["features"]["action"]["shape"], [14])
            self.assertEqual(
                contract["camera_keys"],
                ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            )
            self.assertEqual(contract["arm_mode"], "bimanual")
            self.assertEqual(contract["arm_side"], "both")
            names = info["features"]["observation.state"]["names"]
            self.assertTrue(names[0].startswith("left_"))
            self.assertTrue(names[7].startswith("right_"))

    def test_bimanual_delivery_contract_uses_20d_absolute_state_and_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bimanual_delivery"
            writer = self.make_writer(
                root,
                arm_mode="bimanual",
                arm_side="both",
                schema="delivery",
            )
            self.append_synthetic(writer)
            info = json.loads((root / "meta" / "info.json").read_text())
            contract = json.loads((root / "meta" / "policy_contract.json").read_text())
            self.assertEqual(info["features"]["observation.state"]["shape"], [20])
            self.assertEqual(info["features"]["action"]["shape"], [20])
            self.assertEqual(contract["schema"], "delivery")
            self.assertEqual(contract["arm_mode"], "bimanual")
            self.assertEqual(
                contract["camera_keys"],
                ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            )
            state_names = info["features"]["observation.state"]["names"]
            action_names = info["features"]["action"]["names"]
            self.assertTrue(state_names[0].startswith("left_"))
            self.assertTrue(state_names[10].startswith("right_"))
            self.assertTrue(action_names[0].startswith("left_"))
            self.assertTrue(action_names[10].startswith("right_"))

    def test_legacy_v2_delivery_is_10d_state_7d_step_delta_and_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "legacy"
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
                image_hw=(8, 8),
                schema="delivery",
                arm_mode="single",
                arm_side="right",
                action_semantics=DELIVERY_LEGACY_ACTION_SEMANTICS,
                action_source="next_measured_eef",
                action_alignment="next_observation",
                action_offset=1,
            )
            states = np.zeros((2, 10), dtype=np.float32)
            states[:, 3] = 1.0
            states[:, 7] = 1.0
            actions = np.zeros((2, 7), dtype=np.float32)
            images = {key: np.zeros((2, 8, 8, 3), dtype=np.uint8) for key in writer.camera_keys}
            with patch.object(writer, "_write_episode_videos"):
                writer.append_episode(
                    states=states,
                    actions=actions,
                    timestamps=np.array([1.0, 1.05]),
                    images=images,
                    task_name="legacy_task",
                    instruction="legacy task",
                )
            info = json.loads((root / "meta" / "info.json").read_text())
            self.assertEqual(info["contract_format"], "legacy_v2")
            self.assertTrue(info["legacy_delivery_v2"])
            self.assertEqual(info["raw_action_dim"], 7)
            self.assertEqual(info["model_action_dim"], 7)
            self.assertEqual(info["delivery_action_format"], "step_delta")
            self.assertEqual(info["gripper_semantics"], "absolute_closed_fraction_0_open_1_closed")

    def test_canonical_writer_persists_all_timestamp_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "canonical"
            writer = self.make_writer(root, arm_mode="single", arm_side="right", schema="delivery")
            states = np.zeros((2, 10), dtype=np.float32)
            states[:, 3] = 1.0
            states[:, 7] = 1.0
            actions = states.copy()
            images = {key: np.zeros((2, 8, 8, 3), dtype=np.uint8) for key in writer.camera_keys}
            image_ts = {key: np.array([1.01, 1.06]) for key in writer.camera_keys}
            with patch.object(writer, "_write_episode_videos"):
                writer.append_episode(
                    states=states,
                    actions=actions,
                    timestamps=np.array([1.0, 1.05]),
                    state_timestamps=np.array([1.0, 1.05]),
                    action_timestamps=np.array([1.002, 1.052]),
                    image_timestamps=image_ts,
                    images=images,
                    task_name="canonical_task",
                    instruction="canonical task",
                )
            info = json.loads((root / "meta" / "info.json").read_text())
            self.assertEqual(info["raw_action_dim"], 10)
            self.assertEqual(info["model_action_dim"], 7)
            self.assertEqual(info["action_horizon"], 50)
            table = __import__("pyarrow.parquet", fromlist=["read_table"]).read_table(
                root / "data/chunk-000/episode_000000.parquet"
            )
            self.assertTrue({"timestamp", "state_timestamp", "action_timestamp", "image_timestamp.cam_high"}.issubset(table.column_names))

    def test_incompatible_arm_mode_cannot_append_to_existing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            self.make_writer(root, arm_mode="single", arm_side="right")
            with self.assertRaisesRegex(ValueError, "existing dataset"):
                self.make_writer(root, arm_mode="bimanual", arm_side="both")


if __name__ == "__main__":
    unittest.main()
