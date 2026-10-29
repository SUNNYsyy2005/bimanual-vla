"""Verify optimized kinematics against SDK geometry and numerical derivatives."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.deployment.client import (
    ExecutionBlocked, JOINT_LIMITS_RAD, PiperContinuousIK,
)
from bimanual_vla.deployment.piper_kinematics import (
    PiperKinematics, rotation_error_jacobian,
)


class PiperKinematicsTest(unittest.TestCase):
    def test_matches_sdk_geometry_for_both_joint_offset_modes(self):
        from piper_sdk import C_PiperForwardKinematics
        rng = np.random.default_rng(81)
        for mode in (0, 1):
            sdk = C_PiperForwardKinematics(mode)
            matrix = PiperKinematics(sdk)
            for _ in range(100):
                q = rng.uniform(JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
                xyz, rotation = matrix.evaluate(q)
                sdk_pose = np.asarray(sdk.CalFK(q.tolist())[-1])
                np.testing.assert_allclose(xyz, sdk_pose[:3]/1000, atol=1e-12)
                # SDK snaps pitch to +/-90 deg in this narrow Euler region.
                # The matrix path deliberately preserves the exact rotation.
                if abs(rotation[2, 0]) < 1 - .0001:
                    np.testing.assert_allclose(rotation,
                        Rotation.from_euler('xyz', sdk_pose[3:], degrees=True).as_matrix(), atol=1e-12)

    def test_task_jacobian_matches_central_difference_near_limits_and_large_rotation(self):
        solver = PiperContinuousIK()
        rng = np.random.default_rng(82)
        for _ in range(50):
            q = rng.uniform(JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
            xyz, rotation, geometric = solver._kinematics.evaluate(q, jacobian=True)
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            target = Rotation.from_rotvec(axis * rng.choice([1e-8, .3, 2.9])).as_matrix() @ rotation
            error = Rotation.from_matrix(target @ rotation.T).as_rotvec()
            analytical = np.concatenate((geometric[:3],
                rotation_error_jacobian(error, geometric[3:])), axis=0)
            numerical = np.empty((6, 6))
            def residual(joints):
                p, r = solver.pose(joints)
                return np.concatenate((p-xyz, Rotation.from_matrix(target @ r.T).as_rotvec()))
            for j in range(6):
                delta = np.zeros(6); delta[j] = 1e-6
                numerical[:, j] = (residual(q+delta)-residual(q-delta))/(2e-6)
            np.testing.assert_allclose(analytical, numerical, atol=2e-8, rtol=1e-6)

    def test_reachable_targets_remain_bounded_and_accurate(self):
        solver = PiperContinuousIK()
        rng = np.random.default_rng(83)
        for _ in range(60):
            q = rng.uniform(JOINT_LIMITS_RAD[:, 0]+.1, JOINT_LIMITS_RAD[:, 1]-.1)
            goal = q + rng.uniform(-.04, .04, 6)
            xyz, rotation = solver.pose(goal)
            result = solver.solve_with_diagnostics(q, xyz,
                Rotation.from_matrix(rotation).as_euler('xyz', degrees=True),
                max_joint_step_rad=.3, search_joint_radius_rad=.3,
                position_tolerance_m=.0015, rotation_tolerance_rad=.02, max_nfev=100)
            self.assertLessEqual(result.solution_position_error_m, .0015)
            self.assertLessEqual(result.solution_rotation_error_rad, .02)
            self.assertLessEqual(np.max(abs(result.solution_joints_rad-q)), .3+1e-9)
            self.assertTrue(np.all(result.solution_joints_rad >= JOINT_LIMITS_RAD[:, 0]))
            self.assertTrue(np.all(result.solution_joints_rad <= JOINT_LIMITS_RAD[:, 1]))

    def test_unreachable_target_still_rejected(self):
        solver = PiperContinuousIK()
        q = np.array([.2, 1., -1., .1, .2, -.1])
        xyz, rotation = solver.pose(q)
        with self.assertRaisesRegex(ExecutionBlocked, 'could not reach'):
            solver.solve_with_diagnostics(q, xyz + [2., 0., 0.],
                Rotation.from_matrix(rotation).as_euler('xyz', degrees=True),
                max_joint_step_rad=.3, search_joint_radius_rad=.3,
                position_tolerance_m=.0015, rotation_tolerance_rad=.02, max_nfev=100)
