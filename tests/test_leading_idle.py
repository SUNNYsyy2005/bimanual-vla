import unittest
import numpy as np
from bimanual_vla.data.leading_idle import leading_hold_cut, boundary_hold_range


class LeadingIdleTest(unittest.TestCase):
    def test_right_arm_motion_and_action_lead_are_preserved(self):
        q = np.zeros((30, 14)); q[12:, 9] = .1
        a = np.concatenate([q[1:], q[-1:]])
        self.assertEqual(leading_hold_cut(q, a)["start_frame"], 9)

    def test_gripper_only_motion_is_not_idle(self):
        q = np.zeros((20, 14)); q[8:, 13] = .1
        self.assertEqual(leading_hold_cut(q, q)["start_frame"], 6)

    def test_slow_ramp_uses_cumulative_displacement(self):
        q = np.zeros((40, 7)); q[:, 1] = np.arange(40) * .0002
        self.assertLessEqual(leading_hold_cut(q, q)["start_frame"], 4)

    def test_initial_command_and_all_idle_are_kept(self):
        q = np.zeros((20, 14)); a = q.copy(); a[:, 0] = .1
        self.assertEqual(leading_hold_cut(q, a)["start_frame"], 0)
        self.assertEqual(leading_hold_cut(q, q), dict(start_frame=0, motion_onset_frame=None, all_idle=True))

    def test_invalid_data_is_not_silently_trimmed(self):
        q = np.zeros((20, 14)); q[5, 2] = np.nan
        with self.assertRaises(ValueError):
            leading_hold_cut(q, q)

    def test_both_boundaries_keep_context_and_internal_stop(self):
        q = np.zeros((40, 7))
        q[10:20, 0] = .1
        q[20:30, 0] = .2
        q[30:, 0] = .3
        crop = boundary_hold_range(q, q)
        self.assertEqual((crop["start_frame"], crop["end_frame"]), (8, 31))
        self.assertTrue(np.array_equal(q[crop["start_frame"]:crop["end_frame"]+1][12:22, 0], q[20:30, 0]))

    def test_boundary_crop_preserves_gripper_only_and_slow_motion(self):
        q = np.zeros((40, 14)); q[10:30, 13] = np.linspace(0, .1, 20); q[30:, 13] = .1
        crop = boundary_hold_range(q, q)
        self.assertEqual((crop["start_frame"], crop["end_frame"]), (9, 30))
        ramp = np.zeros((40, 7)); ramp[:, 0] = np.arange(40)*.0002
        self.assertGreater(boundary_hold_range(ramp, ramp)["end_frame"], 30)

    def test_idle_and_endpoint_commands_are_preserved(self):
        q = np.zeros((20, 7)); a = q.copy()
        self.assertEqual(boundary_hold_range(q, a), dict(start_frame=0, end_frame=19,
            removed_leading=0, removed_trailing=0, all_idle=True))
        a[0, 0] = .1; a[-1, 0] = .2
        self.assertEqual(boundary_hold_range(q, a)["end_frame"], 19)
        self.assertEqual(boundary_hold_range(q, a)["start_frame"], 0)
