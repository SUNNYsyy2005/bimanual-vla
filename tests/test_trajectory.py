from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from bimanual_vla.deployment.client import (
    ExecutionBlocked, ExecutionController, PolicyProtocol, TimedTarget,
    check_external_control_streams,
)
from bimanual_vla.data.contract import GRIPPER_MAX_M
from bimanual_vla.deployment.trajectory import (
    JerkLimitedJointTrajectory,
    TrajectoryTrackingError,
    gripper_open_lookahead,
    rate_limit_grippers,
    smootherstep,
)


class TrajectoryShapingTest(unittest.TestCase):
    def test_controller_passes_independent_joint_switches_to_shaper(self):
        args = SimpleNamespace(
            trajectory_shaping=True, trajectory_lowpass=False,
            trajectory_tracking=False, trajectory_speed_limit=False,
            trajectory_acceleration_limit=False, trajectory_jerk_limit=False,
            trajectory_lookahead=False,
        )
        controller = ExecutionController(object(), args)
        feedback = np.zeros(7, dtype=np.float32)
        proposed = np.zeros(6, dtype=np.float32)
        proposed[0] = 0.3
        prepared = {"right": (proposed, 0.0)}
        pipeline = {"right": {}}

        controller._shape_prepared_targets(
            feedback, prepared, pipeline, ("right",), now_monotonic=1.0,
        )

        np.testing.assert_allclose(prepared["right"][0], proposed, atol=1e-5)
        self.assertTrue(pipeline["right"]["trajectory_shaped"])
        self.assertFalse(controller.joint_trajectory.jerk_limit_enabled)

    def test_gripper_and_ik_optional_filters(self):
        args = SimpleNamespace(
            gripper_lowpass=False, gripper_lowpass_alpha=0.5,
            gripper_endpoint_filter=False, gripper_confirm_steps=2,
            max_joint_gripper_step=1.0, ik_rate_limit=False,
            ik_max_joint_step_rad=0.02, ik_search_joint_radius_rad=0.30,
        )
        controller = ExecutionController(object(), args)
        protocol = PolicyProtocol(
            schema="joint", state_dim=7, action_dim=7, arm_side="right",
            action_semantics="absolute_joint_position", camera_keys=("cam_high",),
        )
        qpos = np.zeros(7, dtype=np.float32)
        qpos[6] = 0.5 * GRIPPER_MAX_M
        action = np.zeros(7, dtype=np.float32)
        action[6] = 1.0
        queued = TimedTarget(0, action, action, 1.0)
        filtered = controller._filter_gripper_target(queued, qpos, protocol)
        self.assertAlmostEqual(float(filtered.absolute_target[6]), 1.0)

        observed = {}
        def solve(_joints, _xyz, _rpy, **kwargs):
            observed.update(kwargs)
            return np.zeros(6)
        controller.ik_solver = SimpleNamespace(solve=solve)
        controller._solve_delivery_ik(np.zeros(6), np.zeros(3), np.zeros(3))
        self.assertAlmostEqual(observed["max_joint_step_rad"], 0.30)

    def test_independent_shaping_switches(self):
        initial = np.zeros(7, dtype=np.float32)
        lower = np.full(7, -2.0, dtype=np.float32)
        upper = np.full(7, 2.0, dtype=np.float32)
        target = initial.copy()
        target[0] = 0.3
        direct = JerkLimitedJointTrajectory(
            initial, lower, upper,
            lowpass_enabled=False, tracking_enabled=False,
            speed_limit_enabled=False, acceleration_limit_enabled=False,
            jerk_limit_enabled=False, lookahead_enabled=False,
        )
        direct_command, _ = direct.update(initial, target, 0.05)
        self.assertAlmostEqual(float(direct_command[0]), 0.3, places=5)

        limited = JerkLimitedJointTrajectory(
            initial, lower, upper,
            lowpass_enabled=False, tracking_enabled=False,
            speed_limit_enabled=True, acceleration_limit_enabled=True,
            jerk_limit_enabled=True, lookahead_enabled=False,
        )
        limited_command, _ = limited.update(initial, target, 0.05)
        self.assertLess(float(limited_command[0]), 0.01)
        self.assertAlmostEqual(float(limited.acceleration[0]), 0.2, places=5)

        ahead = JerkLimitedJointTrajectory(
            initial, lower, upper,
            lowpass_enabled=False, tracking_enabled=False,
            speed_limit_enabled=True, acceleration_limit_enabled=False,
            jerk_limit_enabled=False, lookahead_enabled=True,
        )
        ahead_command, _ = ahead.update(initial, target, 0.05)
        self.assertAlmostEqual(float(ahead_command[0]), 0.035, places=5)

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
