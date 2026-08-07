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
        self.assertIn('--dataset-origin {{ upload_default_origin }}', template)
        self.assertIn('config.simulation.example.json', (Path(__file__).parent / 'deploy_4090_sim_dashboard.sh').read_text(encoding='utf-8'))
        self.assertIn('setDatasetOrigin', template)
        self.assertIn('datasetOriginBadge', template)
        self.assertIn('VISIBLE_DATASET_ORIGINS.has', template)
        self.assertIn('visible_dataset_origins', template)
        self.assertIn('仿真数据集已隐藏', template)

        app_source = (Path(__file__).parent / "server_4090/app.py").read_text(encoding="utf-8")
        self.assertIn('visible_dataset_origins', app_source)

    def test_batch_task_log_delete_controls_and_endpoint_are_present(self):
        template = (
            Path(__file__).parent / "server_4090/templates/index.html"
        ).read_text(encoding="utf-8")
        app_source = (Path(__file__).parent / "server_4090/app.py").read_text(encoding="utf-8")

        self.assertGreaterEqual(template.count("batchDeleteSelectedTasks("), 2)
        self.assertIn('class="task-select-all"', template)
        self.assertIn('class="task-select"', template)
        self.assertGreaterEqual(
            template.count('<td class="task-selection-cell">${checkbox}</td>'),
            2,
        )
        self.assertIn(
            '.training-jobs-table th:nth-child(1), .training-jobs-table td:nth-child(1) { width:42px; }',
            template,
        )
        self.assertIn("/api/tasks/batch-delete", template)
        self.assertIn('@app.post("/api/tasks/batch-delete")', app_source)
        self.assertIn("def delete_many", app_source)

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
