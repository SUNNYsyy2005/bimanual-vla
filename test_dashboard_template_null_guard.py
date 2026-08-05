from pathlib import Path
import unittest


class DashboardTemplateNullGuardTest(unittest.TestCase):
    def test_training_experiment_picker_filters_initialization_models(self):
        template = (
            Path(__file__).parent / "server_4090/templates/index.html"
        ).read_text(encoding="utf-8")

        self.assertIn('id="trainExperimentName"', template)
        self.assertIn('list="trainExperimentOptions"', template)
        self.assertIn("fillTrainingExperiments(data.experiments || [])", template)
        self.assertIn("if (isFoundationModel(model)) return true;", template)
        self.assertIn("model.experiment !== experiment", template)
        self.assertIn("model.arm_mode === dataset.arm_mode", template)

    def test_dataset_origin_filter_and_upload_classification_are_visible(self):
        template = (
            Path(__file__).parent / "server_4090/templates/index.html"
        ).read_text(encoding="utf-8")

        self.assertIn('id="datasetOriginFilter"', template)
        self.assertIn('--dataset-origin real', template)
        self.assertIn('uploads/real', template)
        self.assertIn('uploads/simulation', template)
        self.assertIn('setDatasetOrigin', template)
        self.assertIn('datasetOriginBadge', template)

    def test_timed_target_helpers_guard_null_object_values(self):
        template = (
            Path(__file__).parent / "server_4090/templates/index.html"
        ).read_text(encoding="utf-8")
        guarded = (
            "t && t.client_timed_target && "
            "typeof t.client_timed_target === 'object'"
        )
        # Both helpers dereference target_time_error_s/target_at.  In JS,
        # typeof null is also "object", so the truthiness check is required.
        self.assertGreaterEqual(template.count(guarded), 2)
        self.assertNotIn(
            "const target = t && typeof t.client_timed_target === 'object' ? t.client_timed_target : {};",
            template,
        )


if __name__ == "__main__":
    unittest.main()
