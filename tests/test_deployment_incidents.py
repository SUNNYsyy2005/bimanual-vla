"""Hardware-free regressions for the five complete 2026-11-02 sessions."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bimanual_vla.deployment import client
from bimanual_vla.deployment.ik_workers import solve_ik_request, IKUnreachableError
from tests import test_robot_observation_bridge as bridge


class RecordedIKTest(unittest.TestCase):
    def cases(self):
        return json.loads((Path(__file__).parent/'fixtures/ik_recovery_20261102.json').read_text())

    def assert_progress_in_bounds(self, case, result):
        progress = result['goal_progress']
        self.assertTrue(progress['progress'])
        self.assertLess(progress['goal_score_after'], .99*progress['goal_score_before'])
        q = result['solution_joints_rad']
        current = np.asarray(case['args'][0])
        bounds = np.asarray(case['kwargs']['joint_limits'])
        self.assertTrue(np.all(q >= bounds[:, 0]) and np.all(q <= bounds[:, 1]))
        self.assertLessEqual(np.max(np.abs(q-current)), .3+1e-9)

    def test_recorded_recovery_uses_component_caps_in_candidate_search(self):
        case = self.cases()[0]
        result = solve_ik_request(client.PiperContinuousIK(), case['args'], case['kwargs'])
        self.assert_progress_in_bounds(case, result)
        progress = result['goal_progress']
        # J2 is pinned at its limit: position may not grow beyond 0.1 tolerance.
        self.assertLessEqual(progress['goal_position_after_m'],
                             progress['goal_position_before_m']+.1*case['kwargs']['position_tolerance_m'])
        self.assertLessEqual(progress['goal_rotation_after_rad'], .02)

    def test_recorded_moving_branch_still_rejected_then_safe_after_settling(self):
        case = self.cases()[1]
        # Keep the unsafe moving-anchor solution rejected by the final gate.
        with self.assertRaises(IKUnreachableError):
            solve_ik_request(client.PiperContinuousIK(), case['args'],
                             dict(case['kwargs'], recovery_fraction=.25))
        kwargs = dict(case['kwargs'], initial_joints_rad=None, reference_joint_radius_rad=.3,
                      minimum_motion=False, progress_goal=case['args'][1:])
        result = solve_ik_request(client.PiperContinuousIK(), case['args'], kwargs)
        self.assert_progress_in_bounds(case, result)


class RecoveryStateTest(unittest.TestCase):
    def recovery(self, **kwargs):
        helper = bridge.ExecutionQueueTest()
        helper.setUp()
        return helper.recovery_execution(**kwargs)

    def test_exhausted_moving_branch_brakes_without_sending_failed_candidate(self):
        execution, protocol, raw, qpos = self.recovery(exhausted=True)
        execution._ik_warm_starts = {s: qpos[i*7:i*7+6].copy()+.1
                                     for i, s in enumerate(('left', 'right'))}
        self.assertFalse(execution.execute_next(raw, qpos, protocol))
        execution.servo_loop.hold.assert_called_once()
        execution._servo_publish_target.assert_not_called()
        self.assertTrue(execution.ik_recovery_braking)
        self.assertIsNone(execution.ik_recovery_fault)
        self.assertEqual(execution.pending_action_count, 0)
        np.testing.assert_array_equal(execution.eef_goal_held_grippers, qpos[6::7])
        # Settled-goal failure still stops, rather than repeatedly restarting.
        execution.ik_recovery_braking = False
        execution._ik_warm_starts.clear()
        target = execution.ik_recovery_target.copy()
        indices = [i for i in range(14) if i % 7 != 6]
        execution.servo_loop.metrics.return_value = dict(fault=None, paused=False, stopping=False,
            state=dict(position=target[indices], velocity=np.zeros(12), acceleration=np.zeros(12)))
        execution._advance_ik_recovery(target, execution.ik_recovery_started+.1,
                                     raw_delivery_state=raw+np.tile([.02]+[0.]*9, 2), protocol=protocol)
        self.assertIsNotNone(execution.ik_recovery_fault)
        self.assertIsNone(execution.ik_recovery_target)

    def test_gripper_release_gets_one_separate_bounded_deadline(self):
        execution, protocol, raw, qpos = self.recovery()
        execution.execute_next(raw, qpos, protocol)
        target = execution.ik_recovery_target.copy()
        indices = [i for i in range(14) if i % 7 != 6]
        execution.servo_loop.metrics.return_value = dict(fault=None, paused=False, stopping=False,
            state=dict(position=target[indices], velocity=np.zeros(12), acceleration=np.zeros(12)))
        execution.servo_loop.submit.return_value = {'qpos': qpos.copy()}
        start = execution.ik_recovery_started
        execution._advance_ik_recovery(target, start+1.85, raw_delivery_state=raw, protocol=protocol)
        phase_started = execution.ik_recovery_phase_started
        self.assertTrue(execution.eef_goal_releasing)
        execution._advance_ik_recovery(target, start+2.1, raw_delivery_state=raw, protocol=protocol)
        self.assertIsNone(execution.ik_recovery_fault)
        self.assertEqual(execution.ik_recovery_phase_started, phase_started)
        execution._advance_ik_recovery(target, phase_started+2.01, raw_delivery_state=raw, protocol=protocol)
        self.assertIn('phase=gripper release', execution.ik_recovery_fault)
        self.assertIsNone(execution.ik_recovery_target)


class DriverFaultTest(unittest.TestCase):
    def test_faulted_driver_is_never_reenabled(self):
        piper = bridge.FakePiper()
        execution = client.ExecutionController(piper, bridge.execution_args())
        with patch.object(client, 'driver_enable_status_dict', return_value={'faulted': True}):
            with self.assertRaisesRegex(client.RuntimeHardwareFault, 'before enable'):
                execution._enable_robot('right', piper, np.array([0., 1., -1., 0., 0., 0., .035]))
        self.assertEqual(piper.calls, [])

    def test_runtime_driver_fault_propagates_before_reenable(self):
        helper = bridge.ExecutionQueueTest()
        helper.setUp()
        execution, protocol, raw, qpos = helper.recovery_execution()
        with patch.object(client, 'driver_enable_status_dict', return_value={'faulted': True, 'ready': False}):
            with self.assertRaisesRegex(client.RuntimeHardwareFault, 'driver fault'):
                execution.execute_next(raw, qpos, protocol)
        execution._servo_publish_target.assert_not_called()
        for piper in execution.pipers.values():
            self.assertEqual(piper.calls, [])

    def test_return_checks_driver_even_if_already_at_home(self):
        piper = SimpleNamespace(ModeCtrl=Mock(), JointCtrl=Mock(), GripperCtrl=Mock())
        home = np.array([0., 1., -1., 0., 0., 0., .035])
        with patch.object(client, 'read_output_qpos', return_value=home), \
             patch.object(client, 'arm_status_dict', return_value={'arm_status': 0, 'err_code': 0}), \
             patch.object(client, 'driver_enable_status_dict', return_value={'faulted': True}):
            with self.assertRaises(client.RuntimeHardwareFault):
                client.return_pipers_to_initial({'right': piper}, home, arm_mode='single', speed_pct=10,
                    hz=20., timeout_s=3., max_feedback_age_s=.5)
        piper.ModeCtrl.assert_not_called()
        piper.JointCtrl.assert_not_called()
        piper.GripperCtrl.assert_not_called()

    def _return_with_stalled_gripper(self, opening, home_opening):
        clock = self.clock = [0.]
        current = np.array([0., 1., -1., 0., 0., 0., opening])
        home = current.copy(); home[-1] = home_opening
        piper = SimpleNamespace(ModeCtrl=Mock(), JointCtrl=Mock(), GripperCtrl=Mock())
        def sleep(dt):
            clock[0] += dt
        with patch.object(client.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(client.time, 'sleep', side_effect=sleep), \
             patch.object(client, 'read_output_qpos', return_value=current), \
             patch.object(client, 'arm_status_dict', return_value={'arm_status': 0, 'err_code': 0}):
            result = client.return_pipers_to_initial({'right': piper}, home, arm_mode='single', speed_pct=10,
                hz=20., timeout_s=30., max_feedback_age_s=.5)
        return result, clock[0]

    def test_reached_arms_with_stalled_gripper_report_contact_early(self):
        # A gripper still holding an object must not fail the return.
        result, elapsed = self._return_with_stalled_gripper(.009, .0025)
        self.assertTrue(result['returned'])
        self.assertTrue(result['gripper_contact'])
        self.assertLess(elapsed, 3.)

    def test_closed_gripper_a_few_millimetres_from_startup_counts_as_returned(self):
        # 2026-11-02 12:22 run: closed right gripper stalled at 5.0 mm vs 2.5 mm.
        result, elapsed = self._return_with_stalled_gripper(.005, .0025)
        self.assertTrue(result['returned'])
        self.assertAlmostEqual(result['gripper_error_m'], .0025)
        self.assertFalse(result['gripper_contact'])
        self.assertLess(elapsed, 3.)
