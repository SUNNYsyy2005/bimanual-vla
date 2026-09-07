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
