import unittest
import numpy as np
from bimanual_vla.data.leading_idle import leading_hold_cut


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
