import time
import unittest
from unittest.mock import patch

import numpy as np

from bimanual_vla.data.contract import GRIPPER_MAX_M
from bimanual_vla.deployment.chunk_planner import compile_chunk, splice_ready_chunk
from bimanual_vla.deployment.servo import ServoConfig, ServoLoop


class ChunkPlannerTest(unittest.TestCase):
    def test_deferred_install_cannot_reserve_another_old_prefix_window(self):
        from bimanual_vla.deployment.servo import ChunkPlanDeferred
        loop, times, targets, old_prefix = self.prefix_fixture()
        with patch('bimanual_vla.deployment.servo.time.monotonic', return_value=100.5):
            with patch('bimanual_vla.deployment.chunk_planner.compile_chunk',
                       side_effect=ChunkPlanDeferred('concurrent writer change')):
                with self.assertRaises(ChunkPlanDeferred):
                    loop.install_chunk(targets, times, np.arange(len(times)), generation=2)
            commitment = loop._committed_until
            for _ in range(2):
                with self.assertRaisesRegex(ChunkPlanDeferred, 'deferred chunk 2'):
                    loop.rtc_prefix(100.5, 7, 20)
            self.assertEqual(loop._committed_until, commitment)
            self.assertTrue(loop.prefix_is_valid(old_prefix['plan_id']))
        with patch('bimanual_vla.deployment.servo.time.monotonic', return_value=100.55):
            loop.install_chunk(targets, times, np.arange(len(times)), generation=2)
            self.assertLessEqual(loop.metrics()['plan_timing']['splice_monotonic'], 100.57)
            prefix = loop.rtc_prefix(100.55, 7, 20)
            self.assertTrue(loop.prefix_is_valid(prefix['plan_id']))
            self.assertIsNone(loop.metrics()['deferred_install_generation'])
            for offset in prefix['sample_offsets_s']:
                self.assertEqual(loop._plan.context_at(100.55+offset)['generation'], 2)

    def test_deferred_prefix_gate_releases_on_expiry_hold_and_rejection(self):
        from bimanual_vla.deployment.servo import ChunkPlanDeferred, ChunkPlanRejected
        for release in ('expiry', 'hold', 'rejection'):
            with self.subTest(release=release):
                loop, times, targets, _ = self.prefix_fixture()
                with patch('bimanual_vla.deployment.chunk_planner.compile_chunk',
                           side_effect=ChunkPlanDeferred('retry')):
                    with self.assertRaises(ChunkPlanDeferred):
                        loop.install_chunk(targets, times, np.arange(len(times)), generation=2)
                if release == 'hold':
                    loop.hold()
                    self.assertIsNone(loop.rtc_prefix(100.5, 7, 20))
                elif release == 'expiry':
                    with patch('bimanual_vla.deployment.servo.time.monotonic', return_value=times[-1]+1):
                        loop.rtc_prefix(times[-1]+1, 7, 20)
                else:
                    bad = targets.copy(); bad[:, 0] = 100
                    with self.assertRaises(ChunkPlanRejected):
                        loop.install_chunk(bad, times, np.arange(len(times)), generation=2)
                    self.assertIsNotNone(loop.rtc_prefix(100.5, 7, 20))
                self.assertIsNone(loop.metrics()['deferred_install_generation'])

    def test_install_before_next_request_avoids_an_extra_rtc_delay_window(self):
        waits = []
        for reserve_first in (True, False):
            loop, times, targets, _ = self.prefix_fixture()
            with patch('bimanual_vla.deployment.servo.time.monotonic', return_value=100.5):
                if reserve_first:
                    prefix = loop.rtc_prefix(100.5, 7, 20)
                loop.install_chunk(targets, times, np.arange(len(times)), generation=2)
                timing = loop.metrics()['plan_timing']
                waits.append(timing['splice_monotonic'] - 100.5)
                if not reserve_first:
                    prefix = loop.rtc_prefix(100.5, 7, 20)
                    # The next request is now conditioned on the new trajectory
                    # actually scheduled for publication, not its predecessor.
                    for offset, position in zip(prefix['sample_offsets_s'], prefix['positions']):
                        at = 100.5 + offset
                        self.assertEqual(loop._plan.context_at(at)['generation'], 2)
                        np.testing.assert_array_equal(position, loop._plan.positions[loop._plan.index_at(at)])
                self.assertTrue(loop.prefix_is_valid(prefix['plan_id']))
        self.assertGreaterEqual(waits[0], .35 - 1e-9)
        self.assertLess(waits[1], .02 + 1e-9)

    def test_ready_splice_preserves_state_limits_and_original_suffix_clock(self):
        q, lower, upper, times, targets = self.inputs()
        config = ServoConfig(max_jerk=60)
        prepared = compile_chunk(q, np.zeros(12), np.zeros(12), lower, upper, config,
                                 targets, times, np.arange(len(times)), start=0, generation=7)
        initial = q.copy(); initial[:6] = .03; initial[6::7] = .001
        velocity = np.full(12, .1); acceleration = np.full(12, -.2)
        plan = splice_ready_chunk(prepared, initial, velocity, acceleration,
                                  lower, upper, config, start=.15)
        np.testing.assert_array_equal(plan.positions[0], initial)
        np.testing.assert_array_equal(plan.velocities[0], velocity)
        np.testing.assert_array_equal(plan.accelerations[0], acceleration)
        np.testing.assert_array_equal(plan.source_times, prepared.source_times)
        np.testing.assert_array_equal(plan.source_indices, prepared.source_indices)
        joints = np.array([i for i in range(14) if i % 7 != 6])
        np.testing.assert_array_equal(plan.positions[-1, joints], prepared.positions[-1, joints])
        self.assertEqual(plan.times[-1], prepared.times[-1])
        self.assertTrue((np.diff(plan.times) > 0).all())
        self.assertTrue((plan.positions >= lower - 1e-7).all())
        self.assertTrue((plan.positions <= upper + 1e-7).all())
        self.assertLessEqual(abs(plan.velocities).max(), config.max_velocity + 1e-7)
        self.assertLessEqual(abs(plan.accelerations).max(), config.max_acceleration + 1e-7)
        dt = np.diff(plan.times)[:, None]
        self.assertLessEqual((abs(np.diff(plan.accelerations, axis=0)) / dt).max(),
                             config.max_jerk + 1e-5)
        self.assertLessEqual((abs(np.diff(plan.positions[:, 6::7], axis=0)) / dt).max(),
                             config.max_gripper_velocity + 1e-8)

    def test_late_short_splice_retries_without_recompiling_full_chunk(self):
        loop, times, targets, prefix = self.prefix_fixture()
        calls = []
        with patch('bimanual_vla.deployment.servo.time.monotonic', return_value=100.5) as clock:
            def delayed_once(*args, **kwargs):
                result = splice_ready_chunk(*args, **kwargs)
                calls.append(kwargs['start'])
                if len(calls) == 1:
                    clock.return_value += .03
                return result
            with patch('bimanual_vla.deployment.chunk_planner.compile_chunk', wraps=compile_chunk) as full:
                with patch('bimanual_vla.deployment.chunk_planner.splice_ready_chunk', side_effect=delayed_once):
                    loop.install_chunk(targets, times, np.arange(len(times)), generation=2)
        self.assertEqual(full.call_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[1], calls[0])
        self.assertEqual(loop.metrics()['plan_timing']['splice_retry_count'], 1)
        self.assertTrue(loop.prefix_is_valid(prefix['plan_id']))

    def test_hold_during_full_compilation_cannot_install_stale_candidate(self):
        from bimanual_vla.deployment.servo import ChunkPlanDeferred
        loop, times, targets, prefix = self.prefix_fixture()
        def concurrent_hold(*args, **kwargs):
            loop.hold()
            return compile_chunk(*args, **kwargs)
        with patch('bimanual_vla.deployment.chunk_planner.compile_chunk', side_effect=concurrent_hold):
            with self.assertRaises(ChunkPlanDeferred):
                loop.install_chunk(targets, times, np.arange(len(times)), generation=2)
        self.assertFalse(loop.prefix_is_valid(prefix['plan_id']))
        self.assertIsNone(loop._plan)
        self.assertTrue(loop._hold)

    def prefix_fixture(self):
        # Exercise real compilation/merge with a deterministic clock and no
        # actuator thread. Physical publication is covered by the tests below.
        q, lower, upper, times, targets = self.inputs(7)
        with patch('bimanual_vla.deployment.servo.threading.Thread.start'):
            loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: q.copy(), lambda _: {})
        clock = patch('bimanual_vla.deployment.servo.time.monotonic', return_value=100.0)
        clock.start()
        self.addCleanup(clock.stop)
        times = 100.0 + times
        loop.install_chunk(targets, times, np.arange(len(times)), generation=1)
        prefix = loop.rtc_prefix(100.1, 5, 20)
        self.assertIsNotNone(prefix)
        return loop, times, targets, prefix

    def test_reserved_prefix_survives_multiple_revisions_and_history_pruning(self):
        loop, times, targets, prefix = self.prefix_fixture()
        old = loop._plan
        changed = targets.copy()
        changed[:, :6] *= -1
        for generation in (2, 3):
            loop.install_chunk(changed, times, np.arange(len(times)), generation=generation)
            self.assertNotEqual(prefix['plan_id'], loop.metrics()['plan_id'])
            self.assertTrue(loop.prefix_is_valid(prefix['plan_id']))
            for offset in prefix['sample_offsets_s']:
                at = 100.1 + offset
                for field in ('positions', 'velocities', 'accelerations'):
                    np.testing.assert_array_equal(
                        getattr(old, field)[old.index_at(at)],
                        getattr(loop._plan, field)[loop._plan.index_at(at)])
        # Simulate the writer advancing: past samples are legitimately pruned.
        loop._plan_last_index = loop._plan.index_at(100.3)
        loop.install_chunk(targets, times, np.arange(len(times)), generation=4)
        self.assertGreater(loop._plan.times[0], 100.15)
        self.assertTrue(loop.prefix_is_valid(prefix['plan_id']))
        # A revision without an issued reservation must not be accepted.
        self.assertFalse(loop.prefix_is_valid(loop.metrics()['plan_id']))
        self.assertFalse(loop.prefix_is_valid('0:99999'))

    def test_revoked_prefix_cannot_be_revived_by_recovery(self):
        for action in ('hold', 'pause', 'clock_gap'):
            with self.subTest(action=action):
                loop, times, targets, prefix = self.prefix_fixture()
                if action == 'clock_gap':
                    with loop._condition:
                        loop._invalidate_prefixes()
                else:
                    getattr(loop, action)()
                self.assertFalse(loop.prefix_is_valid(prefix['plan_id']))
                loop.install_chunk(targets, times, np.arange(len(times)), generation=2)
                fresh = loop.rtc_prefix(100.1, 5, 20)
                self.assertTrue(loop.prefix_is_valid(fresh['plan_id']))
                self.assertFalse(loop.prefix_is_valid(prefix['plan_id']))

    def test_rejected_chunk_can_fall_back_to_single_target_without_old_plan(self):
        q, lower, upper, times, targets = self.inputs(7)
        feedback = q.copy()
        samples = []
        def publish(command):
            feedback[:] = command
            return {}
        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: feedback.copy(), publish, samples.append)
        self.addCleanup(loop.close)
        loop.install_chunk(targets, time.monotonic()+times, np.arange(len(times)), generation=1)
        prefix = loop.rtc_prefix(time.monotonic(), 5, 20)
        self.assertIsNotNone(prefix)
        # This is a single-target fallback, not a successful timed-plan splice.
        # Its first publication must carry the replacement context, revoke all
        # old prefix promises, and continue bounded q/v/a from the writer state.
        target = feedback.copy()
        target[0] += .02
        result = loop.submit(target, context={"generation": 2}, discard_chunk=True)
        self.assertIsNone(loop._plan)
        self.assertFalse(loop.prefix_is_valid(prefix['plan_id']))
        self.assertFalse(loop._hold)
        self.assertEqual(result["context"]["generation"], 2)
        self.assertIsNone(result["plan_id"])
        self.assertLessEqual(np.max(np.abs(result["velocity"])), loop.config.max_velocity+1e-7)
        self.assertLessEqual(np.max(np.abs(result["acceleration"])), loop.config.max_acceleration+1e-7)

    def test_single_target_fallback_cannot_unpause_moving_safety_paused_writer(self):
        from bimanual_vla.deployment.servo import ServoError
        loop, _, _, _ = self.prefix_fixture()
        loop._sample_count = 1
        loop._paused = True
        loop.planner.velocity[:] = .1
        with self.assertRaisesRegex(ServoError, "moving servo was safety-paused"):
            loop.submit(loop._target, discard_chunk=True)
        self.assertTrue(loop._paused)

    def test_prefix_still_rejected_when_writer_unavailable(self):
        loop, _, _, prefix = self.prefix_fixture()
        for field, value in (('_paused', True), ('_hold', True), ('_fault', 'driver fault'),
                             ('_stopping', True), ('_closed', True)):
            with self.subTest(field=field):
                original = getattr(loop, field)
                setattr(loop, field, value)
                self.assertFalse(loop.prefix_is_valid(prefix['plan_id']))
                setattr(loop, field, original)
        self.assertTrue(loop.prefix_is_valid(prefix['plan_id']))

    def test_reservation_during_compile_moves_actual_splice_without_revoking_prefix(self):
        from bimanual_vla.deployment.servo import ChunkPlanDeferred
        loop, times, targets, prefix = self.prefix_fixture()
        old = loop._plan
        reservations = []
        def reserve_during_compile(*args, **kwargs):
            reservations.append(loop.rtc_prefix(100.1, 10, 20))
            return compile_chunk(*args, **kwargs)
        with patch('bimanual_vla.deployment.chunk_planner.compile_chunk',
                   side_effect=reserve_during_compile):
            loop.install_chunk(targets, times, np.arange(len(times)), generation=2)
        self.assertIsNot(loop._plan, old)
        self.assertTrue(loop.prefix_is_valid(prefix['plan_id']))
        self.assertTrue(loop.prefix_is_valid(reservations[0]['plan_id']))
        for offset in reservations[0]['sample_offsets_s']:
            at = 100.1 + offset
            np.testing.assert_array_equal(loop._plan.positions[loop._plan.index_at(at)],
                                          old.positions[old.index_at(at)])
        self.assertTrue(loop.prefix_is_valid(prefix['plan_id']))
        self.assertGreaterEqual(loop.metrics()['plan_timing']['splice_monotonic'], 100.6)

    def test_slow_preparation_has_no_forecast_wait_or_future_budget(self):
        q, lower, upper, times, targets = self.inputs(7)
        feedback = q.copy()
        def publish(command):
            feedback[:] = command
            return {}
        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: feedback.copy(), publish)
        self.addCleanup(loop.close)
        def slow_compile(*args, **kwargs):
            time.sleep(.10)
            return compile_chunk(*args, **kwargs)
        with patch('bimanual_vla.deployment.chunk_planner.compile_chunk', side_effect=slow_compile):
            loop.install_chunk(targets, time.monotonic()+times, np.arange(len(times)), generation=1)
        timing = loop.metrics()['plan_timing']
        self.assertGreater(timing['compile_ms'], 100)
        self.assertLess(timing['splice_monotonic'] - timing['prepared_monotonic'], .02)
        for generation in range(2, 5):
            loop.install_chunk(targets, time.monotonic()+times, np.arange(len(times)), generation=generation)
            timing = loop.metrics()['plan_timing']
            self.assertLess(timing['splice_monotonic'] - timing['prepared_monotonic'], .02)
        self.assertEqual(loop.metrics()['handoff_mode'], 'ready_then_splice')
        self.assertNotIn('preparation_budget_ms', loop.metrics())
        self.assertIsNone(loop.metrics()['fault'])

    def test_retained_prefix_keeps_old_source_and_records_actual_handoff(self):
        q, lower, upper, times, targets = self.inputs(7)
        feedback = q.copy()
        samples = []
        def publish(command):
            feedback[:] = command
            return {}
        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: feedback.copy(), publish, samples.append)
        self.addCleanup(loop.close)
        now = time.monotonic()
        loop.install_chunk(targets, now+times, np.arange(len(times)), generation=1)
        time.sleep(.08)
        capture = time.monotonic()
        prefix = loop.rtc_prefix(capture, 5, 20)
        self.assertIsNotNone(prefix)
        old = loop._plan
        changed = targets.copy(); changed[:, :6] *= -1
        # Simulate result arrival after 100 ms. Remaining preparation and the
        # reserved prefix must also be reflected in the selected source index.
        time.sleep(.10)
        loop.install_chunk(changed, capture+times, np.arange(len(times)), generation=2)
        new = loop._plan
        self.assertTrue(loop.prefix_is_valid(prefix['plan_id']))
        timing = loop.metrics()['plan_timing']
        self.assertGreaterEqual(timing['splice_monotonic'], capture+.25)
        expected = int(np.searchsorted(capture+times, timing['splice_monotonic'], side='left'))
        self.assertEqual(timing['source_index_at_splice'], expected)
        for offset in prefix['sample_offsets_s']:
            if capture+offset < new.times[0]:
                continue  # Already published samples have been pruned.
            self.assertEqual(new.context_at(capture+offset), old.context_at(capture+offset))
        deadline = time.monotonic()+1
        while loop.metrics()['plan_timing']['first_published_monotonic'] is None and time.monotonic()<deadline:
            time.sleep(.01)
        timing = loop.metrics()['plan_timing']
        self.assertIsNotNone(timing['first_published_monotonic'])
        self.assertGreaterEqual(timing['first_published_monotonic'], timing['splice_monotonic'])
        pending = [s for s in samples if s['plan_timing'].get('generation') == 2
                   and s['context']['generation'] == 1]
        self.assertTrue(pending)
        self.assertTrue(all(s['plan_timing']['first_published_monotonic'] is None for s in pending))
        applied = [s for s in samples if s['context']['generation'] == 2]
        self.assertTrue(applied)
        self.assertEqual(applied[0]['context']['source_index'], timing['first_published_source_index'])
        self.assertGreaterEqual(timing['first_published_source_index'], timing['source_index_at_splice'])

    def test_bad_candidate_can_brake_then_resume_with_fresh_chunk(self):
        from bimanual_vla.deployment.servo import ChunkPlanRejected
        q, lower, upper, times, targets = self.inputs(7)
        feedback = q.copy()
        commands = []
        def publish(command):
            feedback[:] = command
            commands.append(command.copy())
            return {}
        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: feedback.copy(), publish)
        self.addCleanup(loop.close)
        loop.install_chunk(targets, time.monotonic()+times, np.arange(len(times)), generation=1)
        deadline = time.monotonic()+1
        while loop._stationary() and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertFalse(loop._stationary())
        prefix = loop.rtc_prefix(time.monotonic(), 5, 20)
        old = loop._plan
        bad = targets.copy(); bad[31, 2] = upper[2]+.10799
        with self.assertRaises(ChunkPlanRejected):
            loop.install_chunk(bad, time.monotonic()+times, np.arange(len(times)), generation=2)
        self.assertIs(loop._plan, old)
        self.assertFalse(loop.metrics()['paused'])
        self.assertIsNone(loop.metrics()['fault'])
        # Controller rejection uses healthy braking, invalidating outstanding
        # prefix promises. It does not freeze the moving q/v/a state.
        loop.hold()
        self.assertFalse(loop.prefix_is_valid(prefix['plan_id']))
        deadline = time.monotonic()+2
        while not loop._stationary() and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertTrue(loop._stationary())
        loop.install_chunk(targets, time.monotonic()+times, np.arange(len(times)), generation=3)
        self.assertTrue(loop.has_chunk(3))
        self.assertFalse(loop.metrics()['paused'])
        self.assertIsNone(loop.metrics()['fault'])
        self.assertTrue(np.all(np.array(commands) <= upper+1e-7))

    def test_concurrent_writer_fault_is_not_downgraded_to_candidate_rejection(self):
        from bimanual_vla.deployment.servo import ChunkPlanRejected, ServoError
        q, lower, upper, times, targets = self.inputs(7)
        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: q.copy(), lambda _: {})
        self.addCleanup(loop.close)
        def failed_compile(*args, **kwargs):
            with loop._condition:
                loop._fault = 'driver failed'
                loop._paused = True
            raise ServoError('bad candidate')
        with patch('bimanual_vla.deployment.chunk_planner.compile_chunk', side_effect=failed_compile):
            with self.assertRaisesRegex(ServoError, 'driver failed') as error:
                loop.install_chunk(targets, time.monotonic()+times, np.arange(len(times)), generation=1)
        self.assertNotIsInstance(error.exception, ChunkPlanRejected)

    def test_splice_accepts_joint_roundoff_but_not_gripper_or_real_overrun(self):
        from bimanual_vla.deployment.servo import ServoError
        q, lower, upper, times, targets = self.inputs(7)
        lower[:6] = 0
        q[0] = -1e-8
        plan = compile_chunk(q, np.zeros(6), np.zeros(6), lower, upper, ServoConfig(),
                             targets, times, np.arange(len(times)), start=0, generation=1)
        self.assertEqual(plan.positions[0, 0], q[0])
        self.assertGreaterEqual(plan.positions[:, 0].min(), -1e-7)
        for axis, value in [(0, -1e-6), (6, -1e-8)]:
            invalid = q.copy(); invalid[axis] = value
            with self.assertRaisesRegex(ServoError, 'initial chunk state exceeds'):
                compile_chunk(invalid, np.zeros(6), np.zeros(6), lower, upper, ServoConfig(),
                              targets, times, np.arange(len(times)), start=0, generation=1)

    def test_healthy_braking_defers_then_accepts_after_stopping(self):
        from bimanual_vla.deployment.servo import ChunkPlanDeferred
        q, lower, upper, times, targets = self.inputs(7)
        feedback = q.copy()
        def publish(command):
            feedback[:] = command
            return {}
        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: feedback.copy(), publish)
        self.addCleanup(loop.close)
        loop.submit(targets[-1])
        with loop._condition:
            self.assertFalse(loop._stationary())
            loop.hold()
            with self.assertRaisesRegex(ChunkPlanDeferred, 'braking'):
                loop.install_chunk(targets, time.monotonic()+times, np.arange(len(times)), generation=1)
        self.assertFalse(loop.metrics()['paused'])
        deadline = time.monotonic()+2
        while not loop._stationary() and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertTrue(loop._stationary())
        loop.install_chunk(targets, time.monotonic()+times, np.arange(len(times)), generation=2)
        self.assertTrue(loop.has_chunk(2))
        self.assertIsNone(loop.metrics()['fault'])

    def test_missed_splice_retains_old_plan_and_hard_pause_cannot_restart(self):
        from bimanual_vla.deployment.servo import ChunkPlanDeferred, ServoError
        q, lower, upper, times, targets = self.inputs(7)
        feedback = q.copy()
        def publish(command):
            feedback[:] = command
            return {}
        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: feedback.copy(), publish)
        self.addCleanup(loop.close)
        now = time.monotonic()
        loop.install_chunk(targets, now+times, np.arange(len(times)), generation=1)
        old = loop._plan
        def slow_splice(*args, **kwargs):
            time.sleep(.12)
            return splice_ready_chunk(*args, **kwargs)
        with patch('bimanual_vla.deployment.chunk_planner.splice_ready_chunk', side_effect=slow_splice):
            with self.assertRaises(ChunkPlanDeferred):
                loop.install_chunk(targets, now+times, np.arange(len(times)), generation=2)
        self.assertIs(loop._plan, old)
        self.assertFalse(loop.metrics()['paused'])
        self.assertGreater(loop.metrics()['sample_count'], 0)
        # Pause during real planned movement must stay a hard safety boundary.
        deadline = time.monotonic()+1
        while loop._stationary() and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertFalse(loop._stationary())
        loop.pause()
        with self.assertRaisesRegex(ServoError, 'safety-paused'):
            loop.install_chunk(targets, time.monotonic()+times, np.arange(len(times)), generation=3)
        self.assertTrue(loop.metrics()['paused'])

    def test_slow_compile_can_install_without_a_deadline_retry(self):
        from bimanual_vla.deployment.servo import ChunkPlanDeferred
        q, lower, upper, times, targets = self.inputs(7)
        feedback = q.copy()
        def publish(command):
            feedback[:] = command
            return {}
        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: feedback.copy(), publish)
        self.addCleanup(loop.close)
        def slow_compile(*args, **kwargs):
            time.sleep(.08)
            return compile_chunk(*args, **kwargs)
        now = time.monotonic()
        with patch('bimanual_vla.deployment.chunk_planner.compile_chunk', side_effect=slow_compile):
            loop.install_chunk(targets, now+times, np.arange(len(times)), generation=1)
            self.assertTrue(loop.has_chunk(1))
            self.assertIsNone(loop.metrics()['fault'])
            loop.install_chunk(targets, now+times, np.arange(len(times)), generation=2)
        self.assertTrue(loop.has_chunk(2))
        self.assertFalse(loop.metrics()['paused'])

    def test_chunk_can_restart_after_previous_terminal_stop(self):
        q, lower, upper, times, targets = self.inputs(7)
        feedback = q.copy()
        def publish(command):
            feedback[:] = command
            return {}
        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: feedback.copy(), publish)
        self.addCleanup(loop.close)
        # An already exhausted, stationary old plan must not block all new chunks.
        old = compile_chunk(q, np.zeros(6), np.zeros(6), lower, upper, ServoConfig(),
                            np.zeros_like(targets), times-10, np.arange(len(times)),
                            start=-10, generation=1)
        with loop._condition:
            loop._plan = old
        now = time.monotonic()
        loop.install_chunk(targets, now+times, np.arange(len(times)), generation=2)
        self.assertTrue(loop.has_chunk(2))
        self.assertTrue(np.all(np.diff(loop._plan.times) > 0))

    def inputs(self, dimension=14):
        initial = np.zeros(dimension)
        lower, upper = np.full(dimension, -1.), np.ones(dimension)
        lower[6::7] = 0; upper[6::7] = GRIPPER_MAX_M
        times = np.arange(1, 41) * .05
        targets = np.tile((.2 * times)[:, None], (1, dimension))
        targets[:, 6::7] = GRIPPER_MAX_M
        return initial, lower, upper, times, targets

    def test_materialized_curve_has_bounded_derivatives_and_terminal_stop(self):
        q, lower, upper, times, targets = self.inputs()
        targets[:, 0] += .02 * (-1.) ** np.arange(len(times))
        config = ServoConfig(max_jerk=60)
        plan = compile_chunk(q, np.zeros(12), np.zeros(12), lower, upper, config,
                             targets, times, np.arange(len(times)), start=0, generation=1)
        self.assertTrue((plan.positions >= lower - 1e-9).all())
        self.assertTrue((plan.positions <= upper + 1e-9).all())
        self.assertLessEqual(abs(plan.velocities).max(), config.max_velocity + 1e-7)
        self.assertLessEqual(abs(plan.accelerations).max(), config.max_acceleration + 1e-7)
        self.assertLessEqual(abs(np.diff(plan.accelerations, axis=0)).max() * config.hz,
                             config.max_jerk + 1e-6)
        self.assertLessEqual(abs(np.diff(plan.positions[:, 6::7], axis=0)).max() * config.hz,
                             config.max_gripper_velocity + 1e-9)
        np.testing.assert_allclose(plan.velocities[-1], 0, atol=1e-8)
        np.testing.assert_allclose(plan.accelerations[-1], 0, atol=1e-8)
        self.assertFalse(plan.positions.flags.writeable)

    def test_future_points_change_plan_before_current_waypoint_changes(self):
        q, lower, upper, times, targets = self.inputs(7)
        alternative = targets.copy()
        alternative[2:, :6] = -.3
        a = compile_chunk(q, np.zeros(6), np.zeros(6), lower, upper, ServoConfig(),
                          targets, times, np.arange(len(times)), start=0, generation=1)
        b = compile_chunk(q, np.zeros(6), np.zeros(6), lower, upper, ServoConfig(),
                          alternative, times, np.arange(len(times)), start=0, generation=1)
        self.assertFalse(np.allclose(a.positions[1:10, :6], b.positions[1:10, :6]))

    def test_prefix_is_the_published_plan_and_splice_preserves_it(self):
        q, lower, upper, times, targets = self.inputs(7)
        feedback = q.copy()
        published = []

        def publish(command):
            feedback[:] = command
            published.append(command.copy())
            return {}

        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: feedback.copy(), publish)
        self.addCleanup(loop.close)
        now = time.monotonic()
        loop.install_chunk(targets, now+times, np.arange(len(times)), generation=1)
        loop.submit(targets[0], context={"generation": 1})
        # Give the initial held state time to reach the planned clock.
        time.sleep(.07)
        capture = time.monotonic()
        prefix = loop.rtc_prefix(capture, 5, 20)
        self.assertIsNotNone(prefix)
        old = loop._plan
        for i, offset in enumerate(prefix["sample_offsets_s"]):
            np.testing.assert_array_equal(prefix["positions"][i], old.positions[old.index_at(capture+offset)])
        self.assertTrue(loop.prefix_is_valid(prefix["plan_id"]))
        # A dramatically different new suffix must not change the commitment.
        changed = targets.copy(); changed[:, :6] *= -1
        loop.install_chunk(changed, capture+times, np.arange(len(times)), generation=2)
        new = loop._plan
        for offset in prefix["sample_offsets_s"]:
            np.testing.assert_array_equal(new.positions[new.index_at(capture+offset)],
                                          old.positions[old.index_at(capture+offset)])
        time.sleep(.27)
        self.assertIsNone(loop.metrics()["fault"])
        # Every actual command is a precomputed sample, not a second smoothing pass.
        for command in published:
            self.assertTrue(any(np.any(np.max(abs(p.positions-command), axis=1) < 1e-10)
                                for p in (old, new)))
        active = loop.rtc_prefix(time.monotonic(), 3, 20)
        self.assertIsNotNone(active)
        loop.hold()
        self.assertFalse(loop.prefix_is_valid(active["plan_id"]))
        self.assertIsNone(loop.rtc_prefix(time.monotonic(), 3, 20))

    def test_slow_publication_does_not_accumulate_lag_or_latch_a_clock_fault(self):
        q, lower, upper, times, targets = self.inputs(7)
        feedback = q.copy()
        samples = []

        def publish(command):
            # Simulate CAN/publication taking longer than one 100 Hz period.
            time.sleep(.025)
            feedback[:] = command
            return {}

        loop = ServoLoop(q, lower, upper, ServoConfig(), lambda: feedback.copy(), publish, samples.append)
        self.addCleanup(loop.close)
        now = time.monotonic()
        loop.install_chunk(targets, now+times, np.arange(len(times)), generation=1)
        loop.submit(targets[0])
        prefix = loop.rtc_prefix(time.monotonic(), 6, 20)
        time.sleep(.2)
        metrics = loop.metrics()
        self.assertIsNone(metrics["fault"])
        self.assertGreater(metrics["timing_invalidations"], 0)
        self.assertFalse(loop.prefix_is_valid(prefix["plan_id"]))
        self.assertGreater(len(samples), 3)
        self.assertLess(samples[-1]["feedback_read_monotonic"] - samples[-1]["planned_monotonic"], .02)


if __name__ == "__main__":
    unittest.main()
