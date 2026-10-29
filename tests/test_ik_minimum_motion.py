import unittest
import numpy as np

from bimanual_vla.deployment.ik_continuity import minimum_motion_candidate


class MinimumMotionTest(unittest.TestCase):
    def solve(self, goal, anchor=None, **kwargs):
        anchor = np.zeros(6) if anchor is None else anchor
        return minimum_motion_candidate(lambda q: (q-goal, np.eye(6)),
            anchor, -np.ones(6), np.ones(6), .003, .02, budget_s=1., **kwargs)

    def test_uses_pose_tolerance_to_avoid_unnecessary_joint_motion(self):
        goal = np.array([.01, 0., 0., .04, 0., 0.])
        result = self.solve(goal)
        self.assertIsNotNone(result.joints)
        # Nearest point on the intersection of two independent norm balls.
        self.assertAlmostEqual(result.joints[0], .007, places=5)
        self.assertAlmostEqual(result.joints[3], .02, places=5)
        self.assertLessEqual(np.linalg.norm((result.joints-goal)[:3]), .003)
        self.assertLessEqual(np.linalg.norm((result.joints-goal)[3:]), .02)

    def test_chained_tracking_does_not_accumulate_position_error(self):
        previous = np.zeros(6)
        for x in np.linspace(0, .03, 61):
            goal = np.array([x, 0., 0., 0., 0., 0.])
            result = self.solve(goal, previous)
            self.assertIsNotNone(result.joints)
            self.assertLessEqual(np.linalg.norm((result.joints-goal)[:3]), .003)
            self.assertLessEqual(np.max(abs(result.joints-previous)), .00051)
            previous = result.joints
        self.assertGreater(previous[0], .0269)

    def test_stationary_jitter_does_not_wander(self):
        anchor = np.zeros(6)
        for sign in (-1, 1, -1, 1):
            result = self.solve(np.array([sign*.002, 0., 0., 0., 0., 0.]), anchor)
            np.testing.assert_array_equal(result.joints, anchor)

    def test_infeasible_pose_does_not_return_clipped_solution(self):
        result = self.solve(np.full(6, 2.), max_iterations=5)
        self.assertIsNone(result.joints)

    def test_budget_expiry_is_not_reported_as_a_solution(self):
        result = minimum_motion_candidate(lambda q: (q, np.eye(6)), np.zeros(6),
            -np.ones(6), np.ones(6), .003, .02, budget_s=1e-12)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.joints)


if __name__ == '__main__':
    unittest.main()
