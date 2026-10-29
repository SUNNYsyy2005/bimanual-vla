import os
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.deployment.client import PiperContinuousIK
from bimanual_vla.deployment.ik_workers import IKWorkerError, PiperIKWorkers


def delayed_worker(connection):
    connection.send(("ready", None))
    while True:
        request = connection.recv()
        if request is None:
            return
        batch, args, kwargs = request
        time.sleep(kwargs.get('delay', 0.))
        connection.send((batch, {'value': args[0], 'pid': os.getpid()}, None))


class IKWorkersTest(unittest.TestCase):
    def test_real_workers_match_serial_and_recover_after_rejected_batch(self):
        solver = PiperContinuousIK()
        q = np.array([.2, 1., -1., .1, .2, -.1])
        xyz, rot = solver.pose(q + .01)
        rpy = Rotation.from_matrix(rot).as_euler('xyz', degrees=True)
        kwargs = dict(max_joint_step_rad=.3, search_joint_radius_rad=.3,
                      position_tolerance_m=.0015, rotation_tolerance_rad=.02, max_nfev=100)
        requests = {side: ((q, xyz, rpy), kwargs) for side in ('left', 'right')}
        pool = PiperIKWorkers(requests)
        try:
            self.assertEqual(len({p.pid for p in pool.processes.values()}), 2)
            self.assertNotIn(os.getpid(), [p.pid for p in pool.processes.values()])
            expected = solver.solve_with_diagnostics(q, xyz, rpy, **kwargs)
            result = pool.solve(requests, timeout_s=2.)
            for arm in result.values():
                np.testing.assert_allclose(arm['solution_joints_rad'], expected.solution_joints_rad, atol=1e-12)
            bad = dict(requests)
            bad['right'] = ((q, xyz + [2., 0., 0.], rpy), kwargs)
            with self.assertRaisesRegex(IKWorkerError, 'right:.*could not reach'):
                pool.solve(bad, timeout_s=2.)
            self.assertEqual(set(pool.solve(requests, timeout_s=2.)), set(requests))
        finally:
            pool.close()
        self.assertTrue(all(not p.is_alive() for p in pool.processes.values()))

    def test_timeout_never_reuses_a_late_result_or_queues_more_work(self):
        pool = PiperIKWorkers(('right',), worker_target=delayed_worker)
        try:
            with self.assertRaisesRegex(IKWorkerError, 'deadline'):
                pool.solve({'right': ((1,), {'delay': .1})}, timeout_s=.005)
            with self.assertRaisesRegex(IKWorkerError, 'still running'):
                pool.solve({'right': ((2,), {})}, timeout_s=.005)
            self.assertEqual(pool.batch, 1)
            # Wait for the deliberately delayed test response, not hardware.
            self.assertTrue(pool.connections['right'].poll(2.))
            result = pool.solve({'right': ((3,), {})}, timeout_s=2.)
            self.assertEqual(result['right']['value'], 3)
        finally:
            pool.close()

    def test_exited_worker_is_detected_without_publishing_partial_result(self):
        pool = PiperIKWorkers(('left', 'right'), worker_target=delayed_worker)
        try:
            pool.processes['left'].terminate()
            pool.processes['left'].join(timeout=2.)
            with self.assertRaisesRegex(IKWorkerError, 'exited'):
                pool.solve({side: ((1,), {}) for side in ('left', 'right')})
            self.assertEqual(pool.batch, 0)
        finally:
            pool.close()


class ShrinkRecoveryTest(unittest.TestCase):
    def requests(self):
        kwargs = dict(max_joint_step_rad=.3, search_joint_radius_rad=.3,
                      position_tolerance_m=.0015, rotation_tolerance_rad=.02, max_nfev=100,
                      joint_limits=np.tile([-1., 1.], (6, 1)))
        return {side: ((np.zeros(6), np.array([.08, 0., 0.]), np.zeros(3)), kwargs)
                for side in ('left', 'right')}

    def test_common_fraction_retries_both_arms_and_retains_limits(self):
        from bimanual_vla.deployment.ik_workers import solve_recovering, IKUnreachableError
        batches = []
        def solve(requests, *, timeout_s):
            fractions = {s: kw['recovery_fraction'] for s, (_, kw) in requests.items()}
            batches.append(fractions)
            for _, kw in requests.values():
                np.testing.assert_array_equal(kw['joint_limits'], self.requests()['left'][1]['joint_limits'])
            if fractions['right'] != .25:
                raise IKUnreachableError('right cannot solve')
            return {s: {'recovery': {'fraction': .25, 'progress': True}} for s in requests}
        result = solve_recovering(solve, self.requests())
        self.assertEqual(batches, [{'left': f, 'right': f} for f in (1., .5, .25)])
        self.assertEqual(set(result), {'left', 'right'})

    def test_transport_or_input_error_is_never_retried(self):
        from bimanual_vla.deployment.ik_workers import solve_recovering
        from unittest.mock import Mock
        for error in (IKWorkerError('deadline exceeded'), ValueError('bad input'), EOFError()):
            solve = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                solve_recovering(solve, self.requests())
            self.assertEqual(solve.call_count, 1)

    def test_no_progress_exhausts_without_a_solution(self):
        from bimanual_vla.deployment.ik_workers import solve_recovering, IKUnreachableError, IKRecoveryExhausted
        from unittest.mock import Mock
        solve = Mock(side_effect=IKUnreachableError('no progress'))
        with self.assertRaises(IKRecoveryExhausted):
            solve_recovering(solve, self.requests())
        self.assertEqual(solve.call_count, 3)

    def test_cartesian_shrink_uses_measured_fk_and_checks_actual_progress(self):
        from bimanual_vla.deployment.client import PiperIKSolveResult
        from bimanual_vla.deployment.ik_workers import solve_ik_request, IKUnreachableError
        class Solver:
            stuck = False
            def pose(self, q):
                return np.asarray(q[:3]), Rotation.from_rotvec(q[3:]).as_matrix()
            def solve_with_diagnostics(self, current, xyz, rpy, **kwargs):
                q = np.concatenate((xyz, Rotation.from_euler('xyz', rpy, degrees=True).as_rotvec()))
                if self.stuck:
                    q = current.copy()
                return PiperIKSolveResult(solution_joints_rad=q, command_joints_rad=q, rate_limited=False)
        solver = Solver()
        args, kwargs = self.requests()['right']
        args = (args[0], args[1], np.array([0., 0., 20.]))
        result = solve_ik_request(solver, args, dict(kwargs, recovery_fraction=.25))
        np.testing.assert_allclose(result['solution_joints_rad'][:3], [.02, 0., 0.])
        self.assertAlmostEqual(result['solution_joints_rad'][5], np.deg2rad(5))
        self.assertTrue(result['recovery']['progress'])
        solver.stuck = True
        with self.assertRaisesRegex(IKUnreachableError, 'no safe progress'):
            solve_ik_request(solver, args, dict(kwargs, recovery_fraction=.25))
        solver.stuck = False
        kwargs['joint_limits'][0] = [-.01, .01]
        with self.assertRaisesRegex(IKUnreachableError, 'driver joint'):
            solve_ik_request(solver, args, dict(kwargs, recovery_fraction=.25))

    def test_real_worker_bad_input_not_classified_as_geometry(self):
        from bimanual_vla.deployment.ik_workers import solve_recovering, IKUnreachableError
        pool = PiperIKWorkers(('right',))
        try:
            args, kwargs = self.requests()['right']
            with self.assertRaises(IKWorkerError) as caught:
                solve_recovering(pool.solve, {'right': ((np.zeros(5), args[1], args[2]), kwargs)})
            self.assertNotIsInstance(caught.exception, IKUnreachableError)
            self.assertEqual(pool.batch, 1)
        finally:
            pool.close()

    def test_progress_allows_rotation_tradeoff_only_inside_existing_tolerance(self):
        from bimanual_vla.deployment.client import PiperIKSolveResult
        from bimanual_vla.deployment.ik_workers import solve_ik_request, IKUnreachableError
        from unittest.mock import Mock
        # Reconstructed 12:08 failure: translation improves substantially while
        # rotation remains within the unchanged 0.02 rad solver tolerance.
        for after_pos, after_rot, accepted in ((.004720, .01769, True),
                                              (.004720, .0201, False),
                                              (.007, .01769, False)):
            with self.subTest(after_pos=after_pos, after_rot=after_rot):
                solver = Mock()
                q = np.zeros(6)
                solved = np.full(6, .01)
                solver.pose.side_effect = [
                    (np.array([.007092, 0, 0]), Rotation.from_rotvec([0, 0, .00514]).as_matrix()),
                    (np.array([after_pos, 0, 0]), Rotation.from_rotvec([0, 0, after_rot]).as_matrix())]
                solver.solve_with_diagnostics.return_value = PiperIKSolveResult(
                    solution_joints_rad=solved, command_joints_rad=solved, rate_limited=False)
                _, kwargs = self.requests()['left']
                kwargs['progress_goal'] = (np.zeros(3), np.zeros(3))
                if accepted:
                    result = solve_ik_request(solver, (q, np.zeros(3), np.zeros(3)), kwargs)
                    self.assertTrue(result['goal_progress']['progress'])
                else:
                    with self.assertRaisesRegex(IKUnreachableError, 'no safe progress'):
                        solve_ik_request(solver, (q, np.zeros(3), np.zeros(3)), kwargs)

    def test_arrived_arm_holds_measured_joints_without_loosening_limits(self):
        from unittest.mock import Mock
        from bimanual_vla.deployment.ik_workers import solve_ik_request, IKUnreachableError
        solver = Mock()
        solver.pose.return_value = (np.zeros(3), np.eye(3))
        args, kwargs = self.requests()['left']
        kwargs.update(hold_position=True, recovery_fraction=.5)
        result = solve_ik_request(solver, args, kwargs)
        np.testing.assert_array_equal(result['solution_joints_rad'], args[0])
        self.assertFalse(result['recovery']['progress'])
        solver.solve_with_diagnostics.assert_not_called()
        bad = args[0].copy()
        bad[0] = 1.00001
        with self.assertRaisesRegex(IKUnreachableError, 'driver limits'):
            solve_ik_request(solver, (bad, *args[1:]), kwargs)
