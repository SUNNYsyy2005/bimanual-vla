import unittest

from bimanual_vla.training.settings import FinetuningSettings


class FinetuningSettingsTest(unittest.TestCase):
    def test_new_and_legacy_defaults_are_distinct(self):
        self.assertEqual(FinetuningSettings.resolve({}).action_expert_tuning, "full")
        self.assertFalse(FinetuningSettings.resolve({}).llm_lora)
        self.assertEqual(FinetuningSettings.resolve({}, legacy=True), FinetuningSettings.legacy())

    def test_independent_options_round_trip_and_resume(self):
        for expert in ("full", "lora"):
            for llm in (False, True):
                for rtc in (False, True):
                    settings = FinetuningSettings(expert, llm, rtc, 6)
                    self.assertEqual(FinetuningSettings.resolve({}, settings.to_dict(), resume=True), settings)
                    self.assertIn("--llm-lora" if llm else "--no-llm-lora", settings.command_args())
                    self.assertIn("--rtc-training" if rtc else "--no-rtc-training", settings.command_args())
                    with self.assertRaisesRegex(ValueError, "resume"):
                        FinetuningSettings.resolve({"llm_lora": not llm}, settings.to_dict(), resume=True)

    def test_invalid_types_and_unsupported_model(self):
        for delay in (True, 0, 50, 1.5, "8"):
            with self.assertRaises(ValueError):
                FinetuningSettings(rtc_training_max_delay=delay)
        with self.assertRaises(ValueError):
            FinetuningSettings(llm_lora="false")
        with self.assertRaises(ValueError):
            FinetuningSettings(rtc_training=True).validate_model("pi0")
        FinetuningSettings(rtc_training=True).validate_model("pi05")

    def test_full_backbone_settings_and_resume(self):
        settings = FinetuningSettings(train_vision=True, train_llm=True, rtc_training=True)
        self.assertEqual(FinetuningSettings.resolve({}, settings.to_dict(), resume=True), settings)
        self.assertIn("--train-vision", settings.command_args())
        self.assertIn("--train-llm", settings.command_args())
        for key in ("train_vision", "train_llm"):
            with self.assertRaisesRegex(ValueError, "resume"):
                FinetuningSettings.resolve({key: False}, settings.to_dict(), resume=True)
        with self.assertRaises(ValueError):
            FinetuningSettings(train_llm=True, llm_lora=True)
