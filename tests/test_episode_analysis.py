from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bimanual_vla.data.episode_analysis import (
    analysis_payload,
    analyze_episode,
    compute_joint_velocities,
    detect_motion_anomalies,
    detect_idle_frames,
    suggested_crop,
)


class EpisodeAnalysisTest(unittest.TestCase):
    def test_velocity_idle_runs_and_crop_suggestion(self):
        timestamps = np.arange(8, dtype=np.float64) / 10.0
        state = np.zeros((8, 7), dtype=np.float64)
        state[3:6, 0] = [0.0, 0.2, 0.4]
        action = state.copy()
        velocities = compute_joint_velocities(state, timestamps)
        self.assertGreater(float(velocities[4, 0]), 1.0)
        idle, _speed, _delta = detect_idle_frames(state, action=action, timestamps=timestamps, min_run=2)
        self.assertEqual(suggested_crop(idle)["start_frame"], 3)
        self.assertEqual(suggested_crop(idle)["end_frame"], 7)

    def test_payload_is_json_safe_and_includes_eef_trajectory(self):
        state = np.zeros((4, 7), dtype=np.float64)
        state[:, 0] = np.linspace(0.0, 0.4, 4)
        result = analysis_payload(
            analyze_episode(
                state,
                state,
                np.arange(4, dtype=np.float64) / 20.0,
                state_names=["right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper"],
                arm_side="right",
            )
        )
        json.dumps(result, allow_nan=False)
        self.assertIn("right", result["eef"])
        self.assertEqual(len(result["sampled_frames"]), 4)

    def test_franka_bimanual_16d_state_includes_panda_eef_trajectory(self):
        state = np.zeros((5, 16), dtype=np.float64)
        state[:, 1] = np.linspace(-0.2, 0.2, 5)
        state[:, 8] = np.linspace(0.1, -0.1, 5)
        result = analysis_payload(
            analyze_episode(
                state,
                state,
                np.arange(5, dtype=np.float64) / 20.0,
                state_names=[
                    f"{side}_{name}"
                    for side in ("left", "right")
                    for name in [*(f"joint_{index}" for index in range(1, 8)), "gripper"]
                ],
            )
        )
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["eef_method"], "franka_panda_fk")
        self.assertEqual(set(result["eef"]), {"left", "right"})
        self.assertEqual(len(result["eef"]["left"]["position"]), 5)
        self.assertEqual(len(result["eef"]["right"]["position"]), 5)

    def test_bimanual_payload_uses_non_mirrored_axes_without_offset(self):
        state = np.zeros((2, 16), dtype=np.float64)
        payload = analysis_payload(analyze_episode(state, state))
        self.assertEqual(payload["arm_axis_signs"], {"left": [1, 1, 1], "right": [1, 1, 1]})
        self.assertEqual(payload["arm_axis_convention"], "per_arm_base_frame_explicit_rotations")

    def test_bimanual_eef_trajectory_applies_right_base_offset(self):
        state = np.zeros((2, 16), dtype=np.float64)
        without_offset = analyze_episode(state, state, arm_base_offset=None)
        with_offset = analyze_episode(state, state, arm_base_offset=[0.8, 0.0, 0.0])
        np.testing.assert_allclose(
            with_offset.eef["left"]["position"],
            without_offset.eef["left"]["position"],
        )
        np.testing.assert_allclose(
            with_offset.eef["right"]["position"],
            np.column_stack((
                without_offset.eef["right"]["position"][:, 0] + 0.8,
                without_offset.eef["right"]["position"][:, 1],
                without_offset.eef["right"]["position"][:, 2],
            )),
        )
        payload = analysis_payload(with_offset)
        self.assertEqual(payload["arm_base_offset"], [0.8, 0.0, 0.0])
        self.assertEqual(payload["arm_origins"]["right"], [0.8, 0.0, 0.0])
        self.assertEqual(payload["arm_axis_signs"]["right"], [1, 1, 1])

    def test_robot_type_routes_to_matching_joint_model(self):
        state = np.zeros((2, 14), dtype=np.float64)
        for robot_type in ("piper", "aloha-agilex", "ARX-X5"):
            with self.subTest(robot_type=robot_type):
                analysis = analyze_episode(state, state, robot_type=robot_type)
                self.assertEqual(set(analysis.eef), {"left", "right"})
                self.assertIn(robot_type.lower().split("-")[0], analysis.eef_method)
                self.assertEqual(len(analysis.joint_names), 14)
        franka = analyze_episode(np.zeros((2, 16)), np.zeros((2, 16)), robot_type="franka-panda")
        self.assertEqual(franka.eef_method, "franka_panda_fk")
        self.assertEqual(analyze_episode(state, state, robot_type="franka-panda").eef_method, "unavailable")

    def test_aloha_uses_robotwin_embedded_base_geometry(self):
        analysis = analyze_episode(np.zeros((1, 14)), robot_type="aloha-agilex")
        payload = analysis_payload(analysis)
        self.assertAlmostEqual(payload["arm_base_offset"][0], 0.6033, places=3)
        self.assertAlmostEqual(payload["arm_base_offset"][1], 0.001, places=3)
        self.assertNotEqual(payload["arm_base_rotations"]["left"], np.eye(3).tolist())
        self.assertNotEqual(payload["arm_base_rotations"]["right"], np.eye(3).tolist())

    def test_robotwin_root_rotation_maps_local_x_to_common_y_for_simulation(self):
        state = np.zeros((1, 14), dtype=np.float64)
        local = analyze_episode(
            state,
            robot_type="piper",
        )
        simulation = analyze_episode(
            state,
            robot_type="piper",
            dataset_origin="simulation",
            arm_base_offset=[0.9, 0.0, 0.0],
        )
        for side in ("left", "right"):
            expected = np.column_stack((
                -local.eef[side]["position"][:, 1],
                local.eef[side]["position"][:, 0],
                local.eef[side]["position"][:, 2],
            ))
            if side == "right":
                expected[:, 0] += 0.9
            np.testing.assert_allclose(simulation.eef[side]["position"], expected)
        np.testing.assert_allclose(
            simulation.arm_base_rotations["left"],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-12,
        )

    def test_real_piper_applies_right_arm_xy_central_symmetry(self):
        state = np.zeros((1, 14), dtype=np.float64)
        local = analyze_episode(
            state,
            robot_type="piper",
        )
        real = analyze_episode(
            state,
            robot_type="piper",
            dataset_origin="real",
            arm_base_offset=[0.8, 0.0, 0.0],
        )
        np.testing.assert_allclose(real.eef["left"]["position"], local.eef["left"]["position"])
        expected_right = local.eef["right"]["position"].copy()
        expected_right[:, :2] *= -1.0
        expected_right[:, 0] += 0.8
        np.testing.assert_allclose(real.eef["right"]["position"], expected_right)
        np.testing.assert_allclose(
            real.arm_base_rotations["right"],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        )

    def test_explicit_arm_base_rotation_transforms_position_and_orientation(self):
        state = np.zeros((1, 16), dtype=np.float64)
        rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        baseline = analyze_episode(state, robot_type="franka-panda")
        analysis = analyze_episode(
            state,
            robot_type="franka-panda",
            arm_base_rotations={"right": rotation},
        )
        baseline_position = baseline.eef["right"]["position"][0]
        np.testing.assert_allclose(
            analysis.eef["right"]["position"][0],
            [
                -baseline_position[1],
                baseline_position[0],
                baseline_position[2],
            ],
        )
        self.assertFalse(
            np.allclose(
                analysis.eef["right"]["orientation"],
                baseline.eef["right"]["orientation"],
            )
        )
        self.assertEqual(analysis.arm_base_rotations["right"].tolist(), rotation)

    def test_motion_anomaly_detector_marks_reversal_jitter_and_spike(self):
        timestamps = np.arange(9, dtype=np.float64) / 20.0
        state = np.zeros((9, 7), dtype=np.float64)
        # Alternating increments create repeated velocity reversals (jitter).
        state[:, 0] = [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 2.0]
        result = detect_motion_anomalies(state, timestamps)
        self.assertTrue(bool(result["velocity_reversal"][2, 0]))
        self.assertTrue(bool(np.any(result["jitter"][:, 0])))
        self.assertTrue(bool(np.any(result["abrupt_change"])))
        payload = analysis_payload(analyze_episode(state, state, timestamps))
        self.assertGreater(payload["anomaly_frames"], 0)
        self.assertIn("anomaly", payload["timeline"])


if __name__ == "__main__":
    unittest.main()
