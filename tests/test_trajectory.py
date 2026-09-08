from __future__ import annotations

import unittest

import numpy as np

from bimanual_vla.deployment.client import ExecutionBlocked, check_external_control_streams
from bimanual_vla.deployment.trajectory import (
    JerkLimitedJointTrajectory,
    TrajectoryTrackingError,
    gripper_open_lookahead,
    rate_limit_grippers,
    smootherstep,
)


class TrajectoryShapingTest(unittest.TestCase):
    def test_smootherstep_has_zero_endpoint_slope(self):
        self.assertEqual(smootherstep(0.0), 0.0)
        self.assertEqual(smootherstep(1.0), 1.0)
        self.assertAlmostEqual(smootherstep(0.25), 0.103515625)
        self.assertGreater(smootherstep(0.5), 0.49)
        self.assertLess(smootherstep(0.5), 0.51)

    def test_joint_limits_apply_to_7d_and_14d_states(self):
        for dimension in (7, 14):
            initial = np.zeros(dimension, dtype=np.float32)
            lower = np.full(dimension, -2.0, dtype=np.float32)
            upper = np.full(dimension, 2.0, dtype=np.float32)
            shaper = JerkLimitedJointTrajectory(
                initial,
                lower,
                upper,
                max_speed_rad_s=0.3,
                max_acceleration_rad_s2=0.8,
                max_jerk_rad_s3=4.0,
                smoothing_cutoff_hz=3.0,
                tracking_time_constant_s=0.25,
                command_lookahead_rad=0.02,
                max_tracking_error_rad=2.0,
            )
            previous_acceleration = shaper.acceleration.copy()
            for _ in range(40):
                _, indices = shaper.update(
                    np.zeros(dimension, dtype=np.float32),
                    np.full(dimension, 1.5, dtype=np.float32),
                    0.05,
                )
                self.assertLessEqual(float(np.max(np.abs(shaper.velocity))), 0.3 + 1e-6)
                self.assertLessEqual(
                    float(np.max(np.abs(shaper.acceleration))), 0.8 + 1e-6
                )
                jerk = np.abs(shaper.acceleration - previous_acceleration) / 0.05
                self.assertLessEqual(float(np.max(jerk)), 4.0 + 1e-5)
                previous_acceleration = shaper.acceleration.copy()
                self.assertTrue(all(index % 7 != 6 for index in indices))
            self.assertTrue(np.all(shaper.position <= upper + 1e-6))
            self.assertTrue(np.all(shaper.position >= lower - 1e-6))

    def test_tracking_error_protection_fails_closed(self):
        initial = np.zeros(7, dtype=np.float32)
        shaper = JerkLimitedJointTrajectory(
            initial,
            np.full(7, -2.0, dtype=np.float32),
            np.full(7, 2.0, dtype=np.float32),
            max_tracking_error_rad=0.1,
        )
        feedback = np.zeros(7, dtype=np.float32)
        feedback[0] = 0.2
        with self.assertRaises(TrajectoryTrackingError):
            shaper.update(feedback, initial, 0.05)

    def test_gripper_lookahead_only_anticipates_opening(self):
        current = np.asarray([0.2, 0.8], dtype=np.float32)
        future = np.asarray(
            [[0.1, 0.7], [0.8, 0.1], [0.3, 0.2]], dtype=np.float32
        )
        anticipated = gripper_open_lookahead(current, future, lookahead_steps=1)
        np.testing.assert_allclose(anticipated, [0.8, 0.8])

        no_lookahead = gripper_open_lookahead(current, future, lookahead_steps=0)
        np.testing.assert_array_equal(no_lookahead, current)

    def test_gripper_rate_limit_bounds_step_and_feedback_lead(self):
        current = np.asarray([0.020, 0.040], dtype=np.float32)
        previous = np.asarray([0.020, 0.040], dtype=np.float32)
        proposed = np.asarray([0.060, 0.000], dtype=np.float32)
        result = rate_limit_grippers(
            current,
            proposed,
            max_speed_m_s=0.08,
            dt=0.05,
            previous_target_m=previous,
            max_command_lead_m=0.012,
        )
        np.testing.assert_allclose(result, [0.024, 0.036], atol=1e-6)
        self.assertTrue(np.all(np.abs(result - current) <= 0.012 + 1e-6))


class ExternalControlInterlockTest(unittest.TestCase):
    class _Control:
        def __init__(self, hz: float):
            self.Hz = hz

    class _Piper:
        def __init__(self, joint_hz: float = 0.0, gripper_hz: float = 0.0):
            self.joint_hz = joint_hz
            self.gripper_hz = gripper_hz

        def GetArmJointCtrl(self):
            return ExternalControlInterlockTest._Control(self.joint_hz)

        def GetArmGripperCtrl(self):
            return ExternalControlInterlockTest._Control(self.gripper_hz)

    def test_unknown_or_idle_stream_is_allowed(self):
        check_external_control_streams(self._Piper())

    def test_high_rate_stream_is_rejected(self):
        with self.assertRaisesRegex(ExecutionBlocked, "external"):
            check_external_control_streams(self._Piper(joint_hz=20.0))


if __name__ == "__main__":
    unittest.main()
