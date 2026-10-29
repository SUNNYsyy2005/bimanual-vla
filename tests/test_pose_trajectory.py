from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.deployment.pose_trajectory import PoseTrajectory, PoseTrajectoryConfig
from bimanual_vla.deployment.client import ExecutionController, ExecutionBlocked
from bimanual_vla.data.action_conventions import matrix_to_rotation6d


class PoseTrajectoryTest(unittest.TestCase):
    def test_nonfinite_candidate_is_rejected_without_mutating_reference(self):
        shaper = PoseTrajectory(np.zeros(3), np.eye(3))
        with patch('bimanual_vla.deployment.pose_trajectory._reference_velocity',
                   return_value=(np.full(3, np.nan), {})):
            with self.assertRaisesRegex(ValueError, 'candidate linear velocity'):
                shaper.update([.01, 0., 0.], np.eye(3), dt=1/30,
                              feedback_xyz=np.zeros(3), feedback_rotation=np.eye(3))
        np.testing.assert_array_equal(shaper.xyz, np.zeros(3))
        np.testing.assert_array_equal(shaper.linear_velocity, np.zeros(3))
        np.testing.assert_array_equal(shaper.rotation, np.eye(3))
        self.assertEqual(shaper.last_metrics, {})

    def test_reversals_and_changing_axes_preserve_vector_bounds(self):
        cfg = PoseTrajectoryConfig()
        shaper = PoseTrajectory(np.zeros(3), np.eye(3), cfg)
        previous_v, previous_w = np.zeros(3), np.zeros(3)
        rng = np.random.default_rng(33)
        for tick in range(500):
            target = np.array([0.2, -0.1, 0.1]) * (1 if tick % 60 < 30 else -1)
            rot = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
            dt = (0.03, 0.05, 0.08)[tick % 3]
            old = shaper.xyz.copy()
            xyz, rotation, metrics = shaper.update(target, rot, dt=dt,
                feedback_xyz=shaper.xyz, feedback_rotation=shaper.rotation)
            self.assertLessEqual(np.linalg.norm(shaper.linear_velocity), cfg.max_linear_speed_m_s + 1e-10)
            self.assertLessEqual(np.linalg.norm(shaper.angular_velocity), cfg.max_angular_speed_rad_s + 1e-10)
            self.assertLessEqual(np.linalg.norm(shaper.linear_velocity - previous_v) / dt,
                                 cfg.max_linear_acceleration_m_s2 + 1e-10)
            self.assertLessEqual(np.linalg.norm(shaper.angular_velocity - previous_w) / dt,
                                 cfg.max_angular_acceleration_rad_s2 + 1e-10)
            self.assertLessEqual(np.linalg.norm(xyz - old), cfg.max_linear_speed_m_s * dt + 1e-10)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            previous_v, previous_w = shaper.linear_velocity.copy(), shaper.angular_velocity.copy()

    def test_wrap_crossing_uses_shortest_rotation(self):
        initial = Rotation.from_euler("z", 179, degrees=True).as_matrix()
        target = Rotation.from_euler("z", -179, degrees=True).as_matrix()
        shaper = PoseTrajectory(np.zeros(3), initial)
        for _ in range(100):
            shaper.update(np.zeros(3), target, dt=0.05,
                          feedback_xyz=shaper.xyz, feedback_rotation=shaper.rotation)
            # Stay near the short arc instead of traversing the Euler wrap.
            self.assertLess(np.linalg.norm(Rotation.from_matrix(shaper.rotation @ initial.T).as_rotvec()),
                            np.deg2rad(2) + shaper.config.max_angular_speed_rad_s * .05)
        self.assertLess(np.linalg.norm(Rotation.from_matrix(shaper.rotation @ target.T).as_rotvec()), 1e-7)

    def test_near_pi_and_axis_change_stays_finite(self):
        shaper = PoseTrajectory(np.zeros(3), np.eye(3))
        for axis in np.eye(3):
            for sign in (1, -1):
                target = Rotation.from_rotvec(axis * sign * (np.pi - 1e-10)).as_matrix()
                for _ in range(12):
                    _, rotation, _ = shaper.update(np.zeros(3), target, dt=0.05,
                        feedback_xyz=shaper.xyz, feedback_rotation=shaper.rotation)
                    self.assertTrue(np.isfinite(rotation).all())
                    self.assertAlmostEqual(np.linalg.det(rotation), 1, places=11)

    def test_invalid_inputs_and_tracking_fault_do_not_mutate(self):
        shaper = PoseTrajectory(np.zeros(3), np.eye(3))
        for dt in (0, -1, float("nan"), 0.16):
            with self.assertRaises(ValueError):
                shaper.update(np.ones(3), np.eye(3), dt=dt,
                              feedback_xyz=np.zeros(3), feedback_rotation=np.eye(3))
        for feedback, rotation in ((np.ones(3), np.eye(3)),
                (np.zeros(3), Rotation.from_euler("x", 1).as_matrix())):
            with self.assertRaisesRegex(ValueError, "tracking"):
                shaper.update(np.ones(3), np.eye(3), dt=0.05,
                              feedback_xyz=feedback, feedback_rotation=rotation)
        for rotation in (np.zeros((3, 3)), np.diag([-1., 1., 1.])):
            with self.assertRaises(ValueError):
                shaper.update(np.ones(3), rotation, dt=0.05,
                              feedback_xyz=np.zeros(3), feedback_rotation=np.eye(3))
        np.testing.assert_array_equal(shaper.xyz, np.zeros(3))
        np.testing.assert_array_equal(shaper.linear_velocity, np.zeros(3))

    def test_stationary_target_settles(self):
        shaper = PoseTrajectory(np.zeros(3), np.eye(3))
        target = np.array([.03, -.02, .01])
        rotation = Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
        for _ in range(200):
            shaper.update(target, rotation, dt=.05, feedback_xyz=shaper.xyz,
                          feedback_rotation=shaper.rotation)
        np.testing.assert_allclose(shaper.xyz, target, atol=1e-8)
        np.testing.assert_allclose(shaper.rotation, rotation, atol=1e-8)
        self.assertLess(np.linalg.norm(shaper.linear_velocity), 1e-8)

    def test_candidate_tracking_guard_is_atomic_and_does_not_reject_far_raw_goal(self):
        config = replace(PoseTrajectoryConfig(), max_position_tracking_error_m=.0001)
        shaper = PoseTrajectory(np.zeros(3), np.eye(3), config)
        with self.assertRaisesRegex(ValueError, "candidate"):
            shaper.update(np.ones(3), np.eye(3), dt=.05,
                          feedback_xyz=np.zeros(3), feedback_rotation=np.eye(3))
        np.testing.assert_array_equal(shaper.xyz, np.zeros(3))
        np.testing.assert_array_equal(shaper.linear_velocity, np.zeros(3))
        self.assertEqual(shaper.last_metrics, {})
        shaper = PoseTrajectory(np.zeros(3), np.eye(3))
        _, _, metrics = shaper.update(np.ones(3), np.eye(3), dt=.05,
                                      feedback_xyz=np.zeros(3), feedback_rotation=np.eye(3))
        self.assertTrue(metrics["linear_speed_limited"])
        self.assertTrue(metrics["linear_acceleration_limited"])
        config = replace(PoseTrajectoryConfig(), max_rotation_tracking_error_rad=.0001)
        shaper = PoseTrajectory(np.zeros(3), np.eye(3), config)
        with self.assertRaisesRegex(ValueError, "candidate"):
            shaper.update(np.zeros(3), Rotation.from_euler("z", 1).as_matrix(), dt=.05,
                          feedback_xyz=np.zeros(3), feedback_rotation=np.eye(3))
        np.testing.assert_array_equal(shaper.rotation, np.eye(3))
        np.testing.assert_array_equal(shaper.angular_velocity, np.zeros(3))

    def test_real_sdk_fk_and_continuous_ik_accept_shaped_reachable_path(self):
        from bimanual_vla.deployment.client import PiperContinuousIK
        solver = PiperContinuousIK()  # Pure SDK forward kinematics; no CAN connection.
        initial = np.array([.2, 1., -1., .2, .2, .2])
        current = initial.copy()
        xyz, rotation = solver.pose(current)
        shaper = PoseTrajectory(xyz, rotation)
        previous_solution = None
        for tick in range(30):
            desired_joints = initial + .025 * np.sin(tick / 8 + np.arange(6))
            desired_xyz, desired_rotation = solver.pose(desired_joints)
            measured_xyz, measured_rotation = solver.pose(current)
            xyz, rotation, _ = shaper.update(desired_xyz, desired_rotation, dt=.05,
                feedback_xyz=measured_xyz, feedback_rotation=measured_rotation)
            result = solver.solve_with_diagnostics(current, xyz,
                Rotation.from_matrix(rotation).as_euler("xyz", degrees=True),
                initial_joints_rad=previous_solution, max_joint_step_rad=.02,
                search_joint_radius_rad=.30, position_tolerance_m=.0015,
                rotation_tolerance_rad=.02, max_nfev=100)
            self.assertLessEqual(result.solution_position_error_m, .0015)
            self.assertLessEqual(result.solution_rotation_error_rad, .02)
            previous_solution = result.solution_joints_rad.copy()
            current = result.command_joints_rad.copy()

    def test_selected_response_does_not_amplify_seeded_20hz_noise(self):
        rng = np.random.default_rng(20261020)
        noise_xyz = rng.normal(scale=.001, size=(600, 3))
        noise_rot = rng.normal(scale=.01, size=(600, 3))
        ratios = []
        for cfg in (PoseTrajectoryConfig(), replace(PoseTrajectoryConfig(), response_time_s=.05)):
            shaper = PoseTrajectory(np.zeros(3), np.eye(3), cfg)
            xyzs, rotations = [], []
            for xyz, rotvec in zip(noise_xyz, noise_rot):
                result, rotation, _ = shaper.update(xyz, Rotation.from_rotvec(rotvec).as_matrix(),
                    dt=.05, feedback_xyz=shaper.xyz, feedback_rotation=shaper.rotation)
                xyzs.append(result)
                rotations.append(Rotation.from_matrix(rotation).as_rotvec())
            ratios.append((np.sqrt(np.mean(np.asarray(xyzs)[100:]**2) / np.mean(noise_xyz[100:]**2)),
                np.sqrt(np.mean(np.asarray(rotations)[100:]**2) / np.mean(noise_rot[100:]**2))))
        self.assertLess(ratios[0][0], .97)
        self.assertLess(ratios[0][1], .97)
        # The old explicit velocity tracker amplified this faster setting.
        # Exact exponential response must remove that regression too.
        self.assertLess(ratios[1][0], .97)
        self.assertLess(ratios[1][1], .97)

    def test_fixed_goals_from_rest_do_not_overshoot_at_any_update_rate(self):
        for tau in (.02, .04, .06, .08):
            for periods in ((1/30,), (.05,), (.01, .08, .033, .15)):
                with self.subTest(tau=tau, periods=periods):
                    cfg = replace(PoseTrajectoryConfig(), max_linear_speed_m_s=.35,
                                  max_linear_acceleration_m_s2=2.4, max_angular_speed_rad_s=1.5,
                                  max_angular_acceleration_rad_s2=12., response_time_s=tau)
                    for distance, angle in ((.05, .18), (.1, .6), (.001, .01)):
                        shaper = PoseTrajectory(np.zeros(3), np.eye(3), cfg)
                        direction = np.array([1., -2., 3.]) / np.sqrt(14.)
                        goal = distance * direction
                        rotation = Rotation.from_rotvec(angle * direction).as_matrix()
                        last_distance, last_angle = distance, angle
                        for tick in range(200):
                            dt = periods[tick % len(periods)]
                            old = shaper.xyz.copy()
                            p, r, _ = shaper.update(goal, rotation, dt=dt,
                                feedback_xyz=shaper.xyz, feedback_rotation=shaper.rotation)
                            self.assertLessEqual(float(p @ direction), distance + 1e-12)
                            self.assertLessEqual(np.linalg.norm(p-goal), last_distance + 1e-12)
                            rotvec = Rotation.from_matrix(r).as_rotvec()
                            self.assertLessEqual(float(rotvec @ direction), angle + 1e-12)
                            angular_error = Rotation.from_matrix(rotation @ r.T).magnitude()
                            self.assertLessEqual(angular_error, last_angle + 1e-12)
                            np.testing.assert_allclose((p-old)/dt, shaper.linear_velocity, atol=1e-12)
                            last_distance, last_angle = np.linalg.norm(p-goal), angular_error
                        self.assertLess(last_distance, 1e-8)
                        self.assertLess(last_angle, 1e-8)

    def test_target_inside_stopping_distance_brakes_without_zeroing_velocity(self):
        shaper = PoseTrajectory(np.zeros(3), np.eye(3))
        for _ in range(10):
            shaper.update([.5, 0, 0], np.eye(3), dt=.02,
                          feedback_xyz=shaper.xyz, feedback_rotation=shaper.rotation)
        old_position, old_velocity = shaper.xyz.copy(), shaper.linear_velocity.copy()
        goal = old_position - [1e-4, 0, 0]
        p, _, metrics = shaper.update(goal, np.eye(3), dt=.02,
                                     feedback_xyz=shaper.xyz, feedback_rotation=shaper.rotation)
        self.assertTrue(metrics['linear_braking_for_target_change'])
        self.assertGreater(p[0], old_position[0])  # Finite deceleration cannot stop instantly.
        self.assertGreater(shaper.linear_velocity[0], 0)
        self.assertLessEqual(np.linalg.norm(shaper.linear_velocity-old_velocity),
                             shaper.config.max_linear_acceleration_m_s2 * .02 + 1e-12)
        for _ in range(200):
            shaper.update(goal, np.eye(3), dt=.02,
                          feedback_xyz=shaper.xyz, feedback_rotation=shaper.rotation)
        np.testing.assert_allclose(shaper.xyz, goal, atol=1e-8)

    def test_selected_limits_reverse_and_attenuate_a_pulse_without_persistent_motion(self):
        shaper = PoseTrajectory(np.zeros(3), np.eye(3))
        xs, angles = [], []
        for tick in range(240):
            x, angle = ((.05, .3) if tick < 60 else ((.15, 1.) if tick == 120 else (-.05, -.3)))
            xyz, rotation, _ = shaper.update([x, 0, 0], Rotation.from_rotvec([0, 0, angle]).as_matrix(),
                dt=.05, feedback_xyz=shaper.xyz, feedback_rotation=shaper.rotation)
            xs.append(xyz[0]); angles.append(Rotation.from_matrix(rotation).as_rotvec()[2])
        self.assertLess(max(0, -.05 - min(xs[60:120])), .011)
        self.assertLess(max(0, -.3 - min(angles[60:120])), np.deg2rad(1.2))
        self.assertLess(max(abs(x + .05) for x in xs[120:160]), .004)
        self.assertLess(max(abs(angle + .3) for angle in angles[120:160]), np.deg2rad(1.3))
        self.assertLess(abs(xs[-1] + .05), 1e-8)
        self.assertLess(abs(angles[-1] + .3), 1e-8)
        self.assertLess(np.linalg.norm(shaper.linear_velocity), 1e-7)
        self.assertLess(np.linalg.norm(shaper.angular_velocity), 1e-7)


class PoseExecutionTest(unittest.TestCase):
    def controller(self):
        return ExecutionController(object(), SimpleNamespace(pose_trajectory=True, servo_backend="ruckig"))

    def state(self):
        return np.r_[.2, .2, .2, matrix_to_rotation6d(np.eye(3)), .5]

    def test_pause_and_close_reset_and_restart_from_feedback(self):
        controller = self.controller()
        state = self.state()
        with patch("bimanual_vla.deployment.client.time.monotonic", return_value=100):
            xyz, _, _ = controller._shape_delivery_pose("right", state, state[:3] + .02, np.zeros(3))
        self.assertGreater(np.linalg.norm(xyz - state[:3]), 0)
        controller._ik_warm_starts["right"] = np.ones(6)
        controller._block("ready", "test hold")
        self.assertEqual(controller._pose_trajectories, {})
        self.assertNotIn("right", controller._ik_warm_starts)
        with patch("bimanual_vla.deployment.client.time.monotonic", return_value=200):
            restarted, _, _ = controller._shape_delivery_pose("right", state, state[:3] + .02, np.zeros(3))
        np.testing.assert_allclose(restarted, xyz)
        controller.close_servo()
        self.assertEqual(controller._pose_trajectories, {})

    def test_stale_update_is_blocked_and_disabled_mode_is_identity(self):
        controller = self.controller()
        state = self.state()
        with patch("bimanual_vla.deployment.client.time.monotonic", return_value=100):
            controller._shape_delivery_pose("right", state, state[:3] + .02, np.zeros(3))
        with patch("bimanual_vla.deployment.client.time.monotonic", return_value=101):
            with self.assertRaises(ExecutionBlocked):
                controller._shape_delivery_pose("right", state, state[:3] + .02, np.zeros(3))
        controller.args.pose_trajectory = False
        xyz, rpy, metrics = controller._shape_delivery_pose("right", state, state[:3] + .02, np.ones(3))
        np.testing.assert_allclose(xyz, state[:3] + .02)
        self.assertIsNone(metrics)

    def test_workspace_is_checked_after_reference_integration(self):
        controller = self.controller()
        controller.args.workspace_x = (.199, .2001)
        with self.assertRaisesRegex(ExecutionBlocked, "workspace"):
            controller._shape_delivery_pose("right", self.state(), np.array([.3, 0, .2]), np.zeros(3))

    def test_gui_command_roundtrip_and_backend_gate(self):
        from bimanual_vla.collection.gui import build_inference_bridge_command
        arguments = dict(python_executable="python", module_name="test", host="127.0.0.1",
            port=8000, hz=2, arm_mode="single", arm_side="right", can="can0", left_can="can1",
            right_can="can0", cam_high_device="0", cam_wrist_device="1", cam_left_wrist_device="2",
            cam_right_wrist_device="1", instruction="test", allow_execution=False,
            pose_trajectory_enabled=True, pose_trajectory_settings={"max_linear_speed_m_s": .08})
        command = build_inference_bridge_command(**arguments)
        self.assertIn("--pose-trajectory", command)
        self.assertEqual(command[command.index("--pose-max-linear-speed-m-s") + 1], "0.08")
        with self.assertRaisesRegex(ValueError, "Ruckig"):
            build_inference_bridge_command(**arguments, servo_backend="off")

    def test_nested_pose_diagnostics_persist_without_monitoring(self):
        import json
        from tempfile import TemporaryDirectory
        from bimanual_vla.deployment.recording import DeploymentRunRecorder
        with TemporaryDirectory() as root:
            recorder = DeploymentRunRecorder(root)
            directory = recorder.start({"control_hz": 20})
            diagnostics = {"right": {"pose_trajectory": {"position_lag_m": .01,
                "linear_acceleration_limited": True}, "ik_diagnostics": {"solution_position_error_m": .0001}}}
            recorder.record_control_tick(timestamp=100, monotonic_timestamp=20,
                delivery_state=self.state(), qpos=np.zeros(7), command_sent=True,
                action_dim=7, absolute_dim=10, pose_diagnostics=diagnostics)
            recorder.stop()
            row = json.loads((directory / "trajectory.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["pose_diagnostics"], diagnostics)
            with np.load(directory / "trajectory.npz", allow_pickle=False) as numeric:
                self.assertEqual(numeric["qpos"].shape, (1, 7))

    def test_two_arm_second_ik_failure_never_publishes_and_clears_both_states(self):
        import time
        from tests.test_robot_observation_bridge import (FakePiper, RecordingContinuousIK,
            BIMANUAL_DELIVERY_METADATA, execution_args, delivery_state, execution_result,
            validate_policy_metadata)
        protocol = validate_policy_metadata(BIMANUAL_DELIVERY_METADATA, "both", arm_mode="bimanual")
        pipers = {side: FakePiper() for side in ("left", "right")}
        controller = ExecutionController(pipers, execution_args(arm_mode="bimanual", arm_side="both",
            servo_backend="ruckig", pose_trajectory=True))
        controller.configure_protocol(protocol)
        controller.robot_enabled = set(pipers)

        class FailSecondIK(RecordingContinuousIK):
            def solve(self, *args, **kwargs):
                if self.targets:
                    raise ExecutionBlocked("synthetic second-arm IK failure")
                return super().solve(*args, **kwargs)

        controller.ik_solver = FailSecondIK()
        raw = np.tile(delivery_state(), 2)
        qpos = np.tile([0., 1., -1., 0., 0., 0., .035], 2)
        actions = np.zeros((1, 14)); actions[0, [0, 7]] = .004; actions[0, [6, 13]] = .5
        now = time.time()
        controller.queue_result(execution_result(actions), raw, qpos, protocol,
            {key: now for key in protocol.camera_keys}, .01)
        with patch.object(controller, "_servo_publish_target") as publish:
            self.assertFalse(controller.execute_next(raw, qpos, protocol, feedback_captured_at=time.time()))
            publish.assert_not_called()
        self.assertIn("second-arm", controller.blocked_reason)
        self.assertEqual(controller._pose_trajectories, {})
        self.assertEqual(controller._pose_reference_at, {})
        self.assertEqual(controller._ik_warm_starts, {})
        for piper in pipers.values():
            self.assertFalse(any(call[0] in ("JointCtrl", "EndPoseCtrl") for call in piper.calls))

    def test_delivery_trace_hold_and_protocol_change(self):
        import time
        from tests.test_robot_observation_bridge import (FakePiper, RecordingContinuousIK, DELIVERY_METADATA,
            JOINT_METADATA, execution_args, delivery_state, execution_result, validate_policy_metadata)
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        controller = ExecutionController(FakePiper(), execution_args(servo_backend="ruckig", pose_trajectory=True))
        controller.configure_protocol(protocol)
        controller.ik_solver = RecordingContinuousIK()
        controller.robot_enabled = {"right"}
        raw = delivery_state()
        qpos = np.array([0., 1., -1., 0., 0., 0., .035])
        actions = np.array([[.004, 0, 0, 0, 0, 0, .5]])
        now = time.time()
        controller.queue_result(execution_result(actions), raw, qpos, protocol,
            {key: now for key in protocol.camera_keys}, .01)
        sample = dict(qpos=qpos.tolist(), sides={"right": {}}, sequence=1)
        with patch.object(controller, "_servo_publish_target", return_value=sample) as publish:
            self.assertTrue(controller.execute_next(raw, qpos, protocol, feedback_captured_at=time.time()))
            side = controller.last_actuator_command["sides"]["right"]
            self.assertGreater(side["pose_trajectory"]["position_lag_m"], 0)
            self.assertLess(side["pre_ik_eef_target"]["xyz_m"][0], .204)
            self.assertFalse(controller.planner_prefix_enabled)
            self.assertTrue(controller.execute_next(raw, qpos, protocol, feedback_captured_at=time.time()))
            self.assertTrue(publish.call_args.kwargs["hold"])
            self.assertEqual(controller._pose_trajectories, {})
        controller._shape_delivery_pose("right", raw, raw[:3] + .004, np.zeros(3))
        controller.configure_protocol(validate_policy_metadata(JOINT_METADATA, "right"))
        self.assertEqual(controller._pose_trajectories, {})
        self.assertEqual(controller._pose_reference_at, {})


if __name__ == "__main__":
    unittest.main()
