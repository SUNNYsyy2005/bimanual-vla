"""Numerical IK regression from the 09:46 left-arm reference reversal."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.deployment.client import PiperContinuousIK, rotation_from_state
from bimanual_vla.deployment.hardware_safety import NOMINAL_JOINT_LIMITS_RAD
from bimanual_vla.deployment.ik_workers import solve_ik_request


class IKContinuityTest(unittest.TestCase):
    def test_small_recorded_pose_change_does_not_flip_joint_reference(self):
        rows = json.loads((Path(__file__).parent/'fixtures/ik_reference_reversal.json').read_text())['rows']
        solver = PiperContinuousIK()
        previous = None
        solutions = []
        for row in rows:
            goal = np.asarray(row['goal'])
            result = solve_ik_request(solver, (np.asarray(row['joints']), goal[:3],
                Rotation.from_matrix(rotation_from_state(goal.copy())).as_euler('xyz', degrees=True)),
                dict(initial_joints_rad=previous, max_joint_step_rad=.3, search_joint_radius_rad=.3,
                     position_tolerance_m=.0015, rotation_tolerance_rad=.02, max_nfev=100))
            previous = result['solution_joints_rad']
            solutions.append(previous)
            self.assertTrue(np.all(previous >= NOMINAL_JOINT_LIMITS_RAD[:, 0]))
            self.assertTrue(np.all(previous <= NOMINAL_JOINT_LIMITS_RAD[:, 1]))
            self.assertLessEqual(result['solution_position_error_m'], .01)
            self.assertLessEqual(result['solution_rotation_error_rad'], .12)
        # Previously source 6 -> 7 changed J1 from 37.67 to 71.83 degrees
        # despite only 5.4 mm / 0.42 degrees of Cartesian target movement.
        self.assertLess(np.max(np.abs(solutions[-1]-solutions[-2])), .04)

    def test_shrink_keeps_previous_branch_anchor(self):
        from unittest.mock import Mock
        solver = Mock()
        solver.pose.return_value = (np.zeros(3), np.eye(3))
        # Stop at solver entry: inspect the retry's branch anchor independently
        # of any fabricated geometric success or progress result.
        solver.solve_with_diagnostics.side_effect = RuntimeError('capture')
        previous = np.ones(6)*.04
        with self.assertRaisesRegex(RuntimeError, 'capture'):
            solve_ik_request(solver, (np.zeros(6), np.array([.01, 0., 0.]), np.zeros(3)),
                             dict(recovery_fraction=.25, initial_joints_rad=previous))
        np.testing.assert_array_equal(solver.solve_with_diagnostics.call_args.kwargs['initial_joints_rad'], previous)
