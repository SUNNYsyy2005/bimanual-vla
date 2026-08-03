from __future__ import annotations

import unittest

from server_4090.app import describe_dataset_schema, parse_training_metrics


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


class DatasetSchemaDescriptionTest(unittest.TestCase):
    @staticmethod
    def info(state_key, state_dim, action_key, action_dim, cameras):
        features = {
            state_key: {"dtype": "float32", "shape": [state_dim]},
            action_key: {"dtype": "float32", "shape": [action_dim]},
        }
        features.update({key: {"dtype": media_type} for key, media_type in cameras})
        return {"features": features}

    def test_common_single_arm_and_bimanual_dimensions(self):
        cases = [
            (
                self.info(
                    "observation.state", 7, "action", 7,
                    [("observation.images.cam_high", "video"), ("observation.images.cam_right_wrist", "video")],
                ),
                "单臂 Joint 7D/7D", "joint", "single", "right", True,
            ),
            (
                self.info("state", 10, "actions", 7, [("image", "image"), ("wrist_image", "image")]),
                "单臂 Delivery 10D/7D", "delivery", "single", "right", True,
            ),
            (
                self.info(
                    "observation.state", 14, "action", 14,
                    [("observation.images.cam_high", "video"), ("observation.images.cam_left_wrist", "video"), ("observation.images.cam_right_wrist", "video")],
                ),
                "双臂 Joint 14D/14D", "joint", "bimanual", "both", True,
            ),
            (
                self.info("state", 20, "actions", 14, [("overhead", "image"), ("left", "image"), ("right", "image")]),
                "双臂 Delivery 20D/14D", "delivery", "bimanual", "both", False,
            ),
        ]
        cases.append(
            (
                self.info(
                    "observation.state", 20, "action", 14,
                    [("observation.images.cam_high", "video"), ("observation.images.cam_left_wrist", "video"), ("observation.images.cam_right_wrist", "video")],
                ),
                "双臂 Delivery 20D/14D", "delivery", "bimanual", "both", True,
            )
        )
        for info, label, schema, arm_mode, arm_side, trainable in cases:
            with self.subTest(label=label):
                result = describe_dataset_schema(info)
                self.assertEqual(result["schema_label"], label)
                self.assertEqual(result["schema"], schema)
                self.assertEqual(result["arm_mode"], arm_mode)
                self.assertEqual(result["arm_side"], arm_side)
                self.assertIs(result["training_supported"], trainable)
                self.assertEqual(
                    result["training_schema"], schema if trainable else None
                )
                self.assertEqual(len(result["media"]), len([
                    value for value in info["features"].values()
                    if value.get("dtype") in {"image", "video"}
                ]))

    def test_unknown_dimensions_remain_visible_as_custom(self):
        result = describe_dataset_schema(
            self.info("observation.state", 12, "action", 8, [("custom_camera", "image")])
        )
        self.assertEqual(result["schema"], "custom")
        self.assertEqual(result["schema_label"], "通用格式 12D/8D")
        self.assertEqual(result["cameras"], ["custom_camera"])
        self.assertFalse(result["training_supported"])


if __name__ == "__main__":
    unittest.main()
