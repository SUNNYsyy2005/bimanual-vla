"""LeRobot end-of-episode action padding is carried to the loss (2026-11-11)."""
from types import SimpleNamespace
import unittest

import numpy as np

from bimanual_vla.training.settings import FinetuningSettings
from tests.test_openpi_action_contract import HELPER


class ActionPadMaskSettingsTest(unittest.TestCase):
    def test_new_runs_mask_and_older_checkpoints_keep_their_objective(self):
        self.assertTrue(FinetuningSettings.resolve({}).action_pad_mask)
        self.assertFalse(FinetuningSettings.legacy().action_pad_mask)
        old_marker = FinetuningSettings().to_dict()
        old_marker.pop("action_pad_mask")
        self.assertFalse(FinetuningSettings.resolve({}, old_marker, resume=True).action_pad_mask)
        with self.assertRaisesRegex(ValueError, "resume"):
            FinetuningSettings.resolve({"action_pad_mask": True}, old_marker, resume=True)
        self.assertIn("--action-pad-mask", FinetuningSettings().command_args())
        self.assertIn("--no-action-pad-mask", FinetuningSettings(action_pad_mask=False).command_args())
        with self.assertRaises(ValueError):
            FinetuningSettings(action_pad_mask="true")


class ActionPadFlagTransformTest(unittest.TestCase):
    def test_repack_keeps_lerobot_flag(self):
        repack = lambda data: {"state": data["observation.state"], "actions": data["action"]}  # noqa: E731
        step = HELPER.KeepActionPadFlag(repack, "action_is_pad")
        item = {"observation.state": np.zeros(20), "action": np.zeros((5, 20)),
                "action_is_pad": np.array([0, 0, 0, 1, 1], dtype=bool)}
        out = step(item)
        np.testing.assert_array_equal(out[HELPER.ACTION_PAD_KEY], item["action_is_pad"])
        self.assertNotIn(HELPER.ACTION_PAD_KEY, step({k: v for k, v in item.items() if k != "action_is_pad"}))

    def test_piper_inputs_pass_flag_through(self):
        contract = SimpleNamespace(state_dim=7, camera_keys=("cam_high", "cam_wrist"), arm_mode="single",
                                   arm_side="right", raw_action_dim=7, schema="joint")
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        flag = np.array([False, True])
        out = HELPER.PiperInputs(contract=contract)({
            "state": np.zeros(7), "images": {"cam_high": image, "cam_wrist": image},
            "actions": np.zeros((2, 7)), HELPER.ACTION_PAD_KEY: flag})
        np.testing.assert_array_equal(out[HELPER.ACTION_PAD_KEY], flag)

    def test_flag_is_written_into_the_unused_last_channel(self):
        actions = np.zeros((5, 32), dtype=np.float32)
        actions[:, :14] = np.arange(14)
        flag = np.array([0, 0, 0, 1, 1], dtype=bool)
        out = HELPER.EncodeActionPadMask(channel=31)({"actions": actions, HELPER.ACTION_PAD_KEY: flag})
        np.testing.assert_array_equal(out["actions"][:, 31], flag.astype(np.float32))
        np.testing.assert_array_equal(out["actions"][:, :31], actions[:, :31])
        self.assertNotIn(HELPER.ACTION_PAD_KEY, out)
        self.assertEqual(actions[3, 31], 0)  # input not modified in place

    def test_inference_inputs_and_bad_channels(self):
        observation = {"state": np.zeros(32)}
        self.assertIs(HELPER.EncodeActionPadMask(channel=31)(observation), observation)
        busy = np.ones((5, 32), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "unused zero padding"):
            HELPER.EncodeActionPadMask(channel=31)({"actions": busy, HELPER.ACTION_PAD_KEY: np.zeros(5, bool)})
        with self.assertRaisesRegex(ValueError, "does not match"):
            HELPER.EncodeActionPadMask(channel=31)({"actions": np.zeros((5, 32)), HELPER.ACTION_PAD_KEY: np.zeros(4, bool)})


if __name__ == "__main__":
    unittest.main()
