from __future__ import annotations

import unittest

from server_4090.app import parse_training_metrics


class TrainingMetricsParserTest(unittest.TestCase):
    def test_parses_carriage_returns_ansi_and_metric_summary(self):
        log = (
            "\x1b[32mStep 100: grad_norm=0.1, loss=0.04, param_norm=1800\x1b[0m\r"
            "Step 200: grad_norm=8.0e-2, loss=0.03, "
            "loss_physical_14d=0.06, loss_padding_18d=0.0002\r"
        )

        result = parse_training_metrics(log)

        self.assertEqual([point["step"] for point in result["points"]], [100, 200])
        self.assertEqual(result["series"], [
            "grad_norm", "loss", "loss_padding_18d", "loss_physical_14d", "param_norm",
        ])
        self.assertEqual(result["summary"]["loss"], {"latest": 0.03, "min": 0.03, "max": 0.04})
        self.assertEqual(result["total_points"], 2)
        self.assertEqual(result["sampled_points"], 2)

    def test_later_duplicate_step_wins_and_sampling_keeps_endpoints(self):
        lines = [f"Step {step}: loss={step / 1000:.3f}" for step in range(100)]
        lines.extend(["Step 50: loss=9.5", "not a metric", "Step 101: loss=nan"])

        result = parse_training_metrics("\n".join(lines), max_points=10)

        self.assertEqual(result["total_points"], 100)
        self.assertEqual(result["sampled_points"], 10)
        self.assertEqual(result["points"][0]["step"], 0)
        self.assertEqual(result["points"][-1]["step"], 99)
        self.assertEqual(result["summary"]["loss"]["max"], 9.5)


if __name__ == "__main__":
    unittest.main()
