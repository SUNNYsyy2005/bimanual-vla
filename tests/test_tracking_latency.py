"""IK target lead and latency-aware tracking guard (2026-11-04 run review)."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_robot_observation_bridge as bridge

DT = 1 / 30


class IKTargetLeadTest(unittest.TestCase):
    def run_first_row(self, lead_s):
        helper = bridge.ExecutionQueueTest()
        helper.setUp()
        execution, protocol, state, qpos = helper.recovery_execution(full_solve=True)
        execution.ik_target_lead_s = lead_s
        rows = list(execution.pending_actions)
        for index, row in enumerate(rows):
            goal = row.absolute_target.copy()
            goal[[0, 10]] += .002 * index  # distinct arm poses along the chunk
            execution.pending_actions[index] = replace(row, absolute_target=goal)
        rows = list(execution.pending_actions)
        with patch('bimanual_vla.deployment.client.time.monotonic', return_value=rows[0].target_monotonic):
            self.assertTrue(execution.execute_next(state, qpos, protocol))
        requests = execution.ik_workers.solve.call_args.args[0]
        return execution, protocol, rows, requests

    def test_ik_solves_the_lead_pose_without_changing_queue_rtc_or_gripper(self):
        plain, protocol, rows, plain_requests = self.run_first_row(0.)
        led, _, _, led_requests = self.run_first_row(.05)
        expected = next(row for row in rows if row.target_monotonic >= rows[0].target_monotonic + .05)
        self.assertNotEqual(expected.queue_index, rows[0].queue_index)
        for index, side in enumerate(('left', 'right')):
            np.testing.assert_allclose(plain_requests[side][0][1], rows[0].absolute_target[index*10:index*10+3])
            np.testing.assert_allclose(led_requests[side][0][1], expected.absolute_target[index*10:index*10+3])
            plain_trace = plain.last_actuator_command['sides'][side]
            led_trace = led.last_actuator_command['sides'][side]
            self.assertIsNone(plain_trace['ik_target_lead'])
            self.assertEqual(led_trace['ik_target_lead']['queue_index'], expected.queue_index)
            # The guard still measures against the current row.
            self.assertEqual(led_trace['policy_eef_target'], rows[0].absolute_target[index*10:index*10+10].tolist())
            self.assertEqual(led_trace['pre_ik_eef_target']['gripper_opening_m'],
                             plain_trace['pre_ik_eef_target']['gripper_opening_m'])
        for execution in (plain, led):
            self.assertEqual(execution.last_safe_target.queue_index, rows[0].queue_index)
            self.assertEqual(len(execution.pending_actions), len(rows) - 1)
        self.assertEqual(plain.rtc_request_metadata(protocol)['previous_chunk_offset_steps'],
                         led.rtc_request_metadata(protocol)['previous_chunk_offset_steps'])

    def test_lead_beyond_the_queue_uses_the_last_row(self):
        led, _, rows, requests = self.run_first_row(10.)
        np.testing.assert_allclose(requests['right'][0][1], rows[-1].absolute_target[10:13])

    def test_streaming_projection_has_its_own_wider_bound(self):
        # The 5 cm per-step check bound capped EEF tracking near 18 cm/s.
        for bound, projected in ((None, False), (.05, True)):
            with self.subTest(bound=bound):
                helper = bridge.ExecutionQueueTest()
                helper.setUp()
                execution, protocol, state, qpos = helper.recovery_execution(full_solve=True)
                execution.ik_target_lead_s = 0.
                execution.args.workspace_x = execution.args.workspace_y = execution.args.workspace_z = None
                if bound is not None:
                    execution.args.ik_projection_translation_m = bound
                self.assertEqual(execution.args.max_translation_step_m, .05)
                for index, row in enumerate(execution.pending_actions):
                    goal = row.absolute_target.copy()
                    goal[[0, 10]] += .07
                    execution.pending_actions[index] = replace(row, absolute_target=goal)
                head = execution.pending_actions[0]
                with patch('bimanual_vla.deployment.client.time.monotonic', return_value=head.target_monotonic):
                    self.assertTrue(execution.execute_next(state, qpos, protocol))
                for side in ('left', 'right'):
                    trace = execution.last_actuator_command['sides'][side]['pre_ik_eef_target']
                    self.assertEqual(trace['projected_for_local_ik'], projected)


class WindowedTrackingGuardTest(unittest.TestCase):
    def setUp(self):
        helper = bridge.ExecutionQueueTest()
        helper.setUp()
        self.execution, self.protocol, self.state, self.qpos = helper.recovery_execution(full_solve=True)
        self.assertTrue(self.execution.execute_next(self.state, self.qpos, self.protocol))
        self.base = copy.deepcopy(self.execution.last_actuator_command)
        self.execution.servo_loop.hold.reset_mock()

    def feed(self, tick, *, commanded, measured, targets=None, measured_x=None, selection=None):
        """One command at ``tick`` and its next-cycle feedback."""
        trace = copy.deepcopy(self.base)
        trace['command_monotonic'] = 1000. + tick * DT
        qpos = self.qpos.copy()
        state = self.state.copy()
        for index, side in enumerate(('left', 'right')):
            issued = trace['sides'][side]
            joints = self.qpos[:6].copy()
            joints[0] += commanded
            issued['commanded_joints_rad'] = joints.tolist()
            issued['full_ik_solution_joints_rad'] = issued['commanded_joints_rad']
            target = self.state[index*10:index*10+10].copy()
            if targets is not None:
                target[0] += targets
            issued['policy_eef_target'] = target.tolist()
            issued['ik_diagnostics'] = {'selection_method': selection or 'bounded_lm'}
            qpos[index*7] += measured
            if measured_x is not None:
                state[index*10] += measured_x
        self.execution._pending_feedback_command = trace
        self.execution._complete_pending_command_feedback(
            state, qpos, self.protocol, feedback_at=1000. + (tick + 1) * DT)
        return self.execution.tracking_lag_active

    def test_normal_actuation_delay_is_not_lag(self):
        # 1.2 rad/s commands with feedback three cycles (100 ms) behind: the
        # newest command is 0.12 rad away, the one issued 100 ms ago matches.
        for tick in range(30):
            self.assertFalse(self.feed(tick, commanded=.04*tick, measured=.04*(tick-3)))
        self.assertLess(self.execution.last_command_feedback['tracking_guard_joint_error_rad'], 1e-9)
        self.assertGreater(self.execution.last_command_feedback['max_joint_abs_error_rad'], .1)
        self.execution.servo_loop.hold.assert_not_called()

    def test_without_window_the_same_delay_freezes(self):
        self.execution.tracking_joint_lag_window_s = 0.
        tripped = [self.feed(tick, commanded=.04*tick, measured=.04*(tick-3)) for tick in range(8)]
        self.assertTrue(tripped[-1])
        self.assertIn('source=sent_joint', self.execution.blocked_reason)

    def test_blocked_arm_still_freezes_after_the_window(self):
        tripped = [self.feed(tick, commanded=.04*tick, measured=0.) for tick in range(12)]
        first = tripped.index(True)
        self.assertLessEqual(first, 8)  # window (3 cycles) + confirm cycles + margin
        self.assertIn('source=sent_joint', self.execution.blocked_reason)
        self.execution.servo_loop.hold.assert_called_once()

    def test_policy_eef_delay_within_window_is_not_lag(self):
        # 30 cm/s policy motion with the EEF four cycles (133 ms, 4 cm) behind.
        for tick in range(30):
            self.assertFalse(self.feed(tick, commanded=0., measured=0., targets=.01*tick,
                                       measured_x=.01*(tick-4)))
        self.assertLess(self.execution.last_command_feedback['tracking_guard_position_error_m'], 1e-9)

    def test_policy_eef_falling_behind_still_freezes(self):
        tripped = [self.feed(tick, commanded=0., measured=0., targets=.01*tick, measured_x=0.)
                   for tick in range(40)]
        self.assertTrue(any(tripped))
        self.assertIn('source=policy_reference', self.execution.blocked_reason)

    def test_unreachable_target_at_closest_pose_does_not_freeze_both_arms(self):
        for tick in range(40):
            self.assertFalse(self.feed(tick, commanded=0., measured=0., targets=.06,
                                       selection='best_effort_hold'))
        feedback = self.execution.last_command_feedback
        self.assertEqual(feedback['tracking_guard_exempt_sides'], ['left', 'right'])
        self.assertAlmostEqual(feedback['policy_position_error_m'], .06)  # still reported
        self.execution.servo_loop.hold.assert_not_called()

    def test_unreachable_target_does_not_exempt_a_blocked_joint(self):
        tripped = [self.feed(tick, commanded=.2, measured=0., targets=.06, selection='best_effort_hold')
                   for tick in range(4)]
        self.assertTrue(tripped[-1])
        self.assertIn('source=sent_joint', self.execution.blocked_reason)


if __name__ == "__main__":
    unittest.main()
