import importlib.util
import threading
import time
import unittest

import numpy as np

from bimanual_vla.deployment.servo import (
    RuckigJointPlanner, RuckigJointServo, ServoConfig, ServoError, ServoLoop,
)


@unittest.skipUnless(importlib.util.find_spec("ruckig"), "requires ruckig==0.12.2")
class JointServoTest(unittest.TestCase):
    def test_sub_quantum_extrema_do_not_block_a_continuous_trajectory(self):
        # Today's failures were 1e-9..1e-8 rad excursions. Exercise actual
        # Ruckig extrema at both stops, not just the comparison helper.
        for direction in (1, -1):
            with self.subTest(direction=direction):
                q = np.array([0. if direction == 1 else 1.])
                config = ServoConfig(max_jerk=60)
                planner = RuckigJointPlanner(q, np.zeros(1), np.ones(1), config)
                velocity, acceleration = np.array([-direction * .0001]), np.array([direction * .5])
                planner.set_state(q, velocity, acceleration)
                planner.retarget(np.array([.1 if direction == 1 else .9]))
                np.testing.assert_array_equal(planner.velocity, velocity)
                np.testing.assert_array_equal(planner.acceleration, acceleration)
                for _ in range(100):
                    old_a = planner.acceleration.copy()
                    command = planner.advance()
                    self.assertGreaterEqual(command[0], -1e-7)
                    self.assertLessEqual(command[0], 1+1e-7)
                    self.assertLessEqual(abs(planner.acceleration[0]-old_a[0])*config.hz,
                                         config.max_jerk+1e-6)

    def test_extrema_outside_numerical_slack_are_still_rejected(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                q = np.array([0. if direction == 1 else 1.])
                planner = RuckigJointPlanner(q, np.zeros(1), np.ones(1), ServoConfig(max_jerk=60))
                planner.set_state(q, np.array([-direction*.001]), np.array([direction*.5]))
                # Even ~9e-7 rad is outside the numerical allowance; the
                # larger physical reversal regression below must also fail.
                with self.assertRaisesRegex(ServoError, 'planned .*physical position bounds'):
                    planner.retarget(np.array([.1 if direction == 1 else .9]))

    def test_position_roundoff_is_retained_without_resetting_derivatives(self):
        for residual in (1.1e-9, 4e-9, 1e-8):
            for q in (-residual, 1+residual):
                with self.subTest(q=q):
                    planner = RuckigJointPlanner(np.array([q]), np.zeros(1), np.ones(1), ServoConfig())
                    planner.set_state(np.array([q]), np.array([.01]), np.array([.02]))
                    np.testing.assert_array_equal(planner.position, [q])
                    np.testing.assert_array_equal(planner.input.current_position, [q])
                    np.testing.assert_array_equal(planner.velocity, [.01])
                    np.testing.assert_array_equal(planner.acceleration, [.02])
        for q in (-1e-6, 1+1e-6):
            with self.assertRaisesRegex(ServoError, 'initial state.*physical position bounds'):
                RuckigJointPlanner(np.array([q]), np.zeros(1), np.ones(1), ServoConfig())

    def test_constant_acceleration_interior_minimum_is_checked(self):
        # Ruckig 0.12.2 reports minimum .00338 for this trajectory, but
        # q crosses below zero while v reverses during its j=0 segment.
        planner = RuckigJointPlanner(np.array([.003382413]), np.zeros(1), np.ones(1),
                                    ServoConfig(max_velocity=.6, max_acceleration=3, max_jerk=60))
        planner.set_state(np.array([.003382413]), np.array([-.235277]), np.array([3.]))
        with self.assertRaisesRegex(ServoError, 'planned minimum.*physical position bounds'):
            planner.retarget(np.array([.0744672]))

    def test_reference_planning_does_not_follow_feedback_noise(self):
        config = ServoConfig()
        planner = RuckigJointPlanner(np.zeros(6), -np.ones(6), np.ones(6), config)
        guarded = RuckigJointServo(np.zeros(6), -np.ones(6), np.ones(6), config)
        for tick in range(100):
            if tick in (0, 35):
                target = np.full(6, .2 if tick == 0 else -.1)
                planner.retarget(target)
                guarded.retarget(target)
            feedback = guarded.position + .01 * (-1 if tick % 2 else 1)
            np.testing.assert_allclose(planner.advance(), guarded.update(feedback), atol=1e-12)

    def test_terminal_velocity_near_lower_stop_falls_back_before_publication(self):
        # Recorded right joint 2, 2026-10-20: the endpoint is legal, but
        # its estimated terminal velocity makes Ruckig initially move below 0.
        config = ServoConfig(max_jerk=60)
        servo = RuckigJointServo(np.zeros(1), np.zeros(1), np.ones(1), config)
        servo.retarget(np.array([0.01268662]), np.array([0.2537324]))
        self.assertEqual(servo.input.target_velocity, [0.0])
        for _ in range(100):
            old_acceleration = servo.acceleration.copy()
            position = servo.update(servo.position)
            self.assertGreaterEqual(position[0], -1e-9)
            self.assertLessEqual(abs(servo.velocity[0]), config.max_velocity + 1e-9)
            self.assertLessEqual(abs(servo.acceleration[0]), config.max_acceleration + 1e-9)
            self.assertLessEqual(abs(servo.acceleration[0] - old_acceleration[0]) * config.hz,
                                 config.max_jerk + 1e-8)
        np.testing.assert_allclose(servo.position, [0.01268662], atol=1e-9)

    def test_bounds_error_identifies_stage_and_joint(self):
        servo = RuckigJointServo(np.zeros(12), -np.ones(12), np.ones(12), ServoConfig())
        target = np.zeros(12)
        target[7] = 2
        with self.assertRaisesRegex(ServoError, r"target.*arm_index=1 joint=2"):
            servo.retarget(target)

    def test_reversals_and_braking_preserve_motion_bounds(self):
        for joints in (6, 12):
            with self.subTest(joints=joints):
                config = ServoConfig()
                servo = RuckigJointServo(np.zeros(joints), -np.ones(joints), np.ones(joints), config)
                for tick in range(450):
                    if tick in (0, 40, 90, 130):
                        servo.retarget(np.full(joints, 0.5 if tick % 3 else -0.5))
                    if tick == 180:
                        servo.brake()
                    old_q, old_a = servo.position.copy(), servo.acceleration.copy()
                    q = servo.update(old_q)
                    self.assertLessEqual(np.max(np.abs(q - old_q)), config.max_velocity / config.hz + 1e-9)
                    self.assertLessEqual(np.max(np.abs(servo.velocity)), config.max_velocity + 1e-9)
                    self.assertLessEqual(np.max(np.abs(servo.acceleration)), config.max_acceleration + 1e-9)
                    self.assertLessEqual(np.max(np.abs(servo.acceleration - old_a)) * config.hz, config.max_jerk + 1e-8)
                    np.testing.assert_allclose(q, q[0], atol=1e-12)
                np.testing.assert_allclose(servo.velocity, 0, atol=1e-9)
                np.testing.assert_allclose(servo.acceleration, 0, atol=1e-9)

    def test_invalid_target_and_feedback_never_advance_state(self):
        servo = RuckigJointServo(np.zeros(6), -np.ones(6), np.ones(6), ServoConfig())
        for target in (np.full(6, np.nan), np.full(6, 2), np.zeros(5)):
            with self.assertRaises(ServoError):
                servo.retarget(target)
        with self.assertRaises(ServoError):
            servo.update(np.ones(6))
        np.testing.assert_array_equal(servo.position, 0)


@unittest.skipUnless(importlib.util.find_spec("ruckig"), "requires ruckig==0.12.2")
class ServoLoopTest(unittest.TestCase):
    def test_records_measured_feedback_separately_from_planned_command(self):
        feedback = np.zeros(7)
        published = []
        loop = ServoLoop(feedback.copy(), -np.ones(7), np.ones(7), ServoConfig(),
                         lambda: feedback, lambda command: published.append(command.copy()) or {})
        self.addCleanup(loop.close)
        sample = loop.submit(np.full(7, .1))
        loop.pause()
        np.testing.assert_array_equal(sample['feedback_qpos'], np.zeros(7))
        np.testing.assert_array_equal(sample['command_qpos'], sample['qpos'])
        self.assertGreater(np.linalg.norm(sample['command_qpos']), 0)
        self.assertLessEqual(sample['feedback_read_monotonic'], sample['monotonic_timestamp'])
        self.assertEqual(sample['role'], 'joint_reference_planner')

    def make_loop(self, dimension=7):
        feedback = np.zeros(dimension)
        published = []

        def publish(command):
            feedback[:] = command
            published.append((time.monotonic(), threading.get_ident(), command.copy()))
            return {}

        loop = ServoLoop(feedback.copy(), -np.ones(dimension), np.ones(dimension),
                         ServoConfig(target_timeout_s=0.04), lambda: feedback.copy(), publish)
        self.addCleanup(loop.close)
        return loop, published

    def test_timeout_brakes_and_close_stops_the_only_writer(self):
        loop, published = self.make_loop(14)
        sample = loop.submit(np.full(14, 0.1))
        self.assertGreaterEqual(sample["sequence"], 1)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if loop.metrics()["state"]["braking"]:
                break
            time.sleep(0.01)
        self.assertTrue(loop.metrics()["state"]["braking"])
        self.assertTrue(loop.close())
        count = len(published)
        time.sleep(0.025)
        self.assertEqual(len(published), count)
        self.assertEqual(len({item[1] for item in published}), 1)
        self.assertNotEqual(published[0][1], threading.get_ident())
        with self.assertRaises(ServoError):
            loop.submit(np.zeros(14))

    def test_pause_waits_for_writer_and_moving_state_cannot_resume(self):
        loop, published = self.make_loop()
        loop.submit(np.full(7, 0.2))
        loop.pause()
        count = len(published)
        time.sleep(0.025)
        self.assertEqual(count, len(published))
        with self.assertRaisesRegex(ServoError, "safety-paused"):
            loop.submit(np.zeros(7))
        self.assertFalse(loop.close())

    def test_feedback_fault_prevents_publication(self):
        loop, published = self.make_loop()
        loop.read_feedback = lambda: np.full(7, np.nan)
        with self.assertRaisesRegex(ServoError, "feedback"):
            loop.submit(np.full(7, 0.1))
        self.assertEqual(published, [])
        self.assertFalse(loop.close())

    def test_slow_publication_does_not_replay_missed_ticks(self):
        loop, published = self.make_loop()
        original_publish = loop.publish
        completed = threading.Event()
        finished_at = []

        def slow_publish(command):
            result = original_publish(command)
            if len(published) == 1:
                time.sleep(0.03)
                finished_at.append(time.monotonic())
            if len(published) >= 3:
                completed.set()
            return result

        loop.publish = slow_publish
        loop.submit(np.full(7, 0.1))
        self.assertTrue(completed.wait(timeout=1))
        loop.pause()
        self.assertGreaterEqual(published[1][0] - finished_at[0], 0.009)
        self.assertGreaterEqual(loop.metrics()["overruns"], 1)


if __name__ == "__main__":
    unittest.main()
