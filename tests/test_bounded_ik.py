"""Bounded Levenberg-Marquardt IK: no large joint swing for a tiny pose change."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.deployment.client import PiperContinuousIK
from bimanual_vla.deployment.hardware_safety import NOMINAL_JOINT_LIMITS_RAD

KWARGS = dict(max_joint_step_rad=.4, search_joint_radius_rad=.4, reference_joint_radius_rad=.2,
              position_tolerance_m=.003, rotation_tolerance_rad=.02, max_nfev=100,
              joint_limits=NOMINAL_JOINT_LIMITS_RAD)


class BoundedIKTest(unittest.TestCase):
    def setUp(self):
        self.solver = PiperContinuousIK()

    def test_tiny_rotation_along_wrist_null_direction_does_not_swing_j4_j6(self):
        # Dataset replay ep_0045: J5 = 0.21 rad, where 0.02 rad of orientation
        # along the weakest direction needs ~0.5 rad of opposite J4/J6 motion.
        q = np.array([1.495, .01, -.112, .274, .214, -.369])
        _, _, geometric = self.solver._kinematics.evaluate(q, jacobian=True)
        scale = np.array([.003]*3 + [.02]*3)
        u, _, vt = np.linalg.svd(geometric / scale[:, None])
        self.assertGreater(abs(vt[-1, 3]), .6)  # weakest direction is J4/J6
        xyz, rotation = self.solver.pose(q)
        rotvec = u[3:, -1] / np.linalg.norm(u[3:, -1]) * .05
        target = Rotation.from_rotvec(rotvec).as_matrix() @ rotation
        result = self.solver.solve_with_diagnostics(
            q, xyz, Rotation.from_matrix(target).as_euler('xyz', degrees=True),
            initial_joints_rad=q, **KWARGS)
        # Joint motion stays proportional to the 0.05 rad request.
        self.assertLess(np.max(np.abs(result.solution_joints_rad - q)), .06)
        self.assertLessEqual(result.solution_rotation_error_rad, .02)

    def test_joint_pinned_at_limit_makes_bounded_progress_without_blocking(self):
        # 2026-11-02 start pose: J2 at its lower driver limit with the arm
        # nearly folded. Some 10 mm moves are only partly reachable per tick;
        # the solver must stay in bounds and make progress, not raise.
        q = np.array([1.677, 0., -.138, .007, .334, .025])
        xyz, rotation = self.solver.pose(q)
        rpy = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
        for offset in ([.01, 0, 0], [-.01, 0, 0], [0, .01, 0], [0, -.01, 0], [0, 0, .01], [0, 0, -.01]):
            with self.subTest(offset=offset):
                result = self.solver.solve_with_diagnostics(
                    q, xyz + offset, rpy, initial_joints_rad=q, **KWARGS)
                solved = result.solution_joints_rad
                self.assertGreaterEqual(solved[1], NOMINAL_JOINT_LIMITS_RAD[1, 0])
                self.assertLess(result.solution_position_error_m, .01)
                self.assertLessEqual(np.max(np.abs(solved - q)), KWARGS['reference_joint_radius_rad'] + 1e-9)


if __name__ == "__main__":
    unittest.main()
