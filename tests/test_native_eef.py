"""Native execution tests with in-memory transports only; never open CAN."""
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.deployment.native_eef import EEFReference, NativeEEFLoop, NativeEEFError
from bimanual_vla.deployment.client import ExecutionController, RuntimeHardwareFault, validate_policy_metadata
from bimanual_vla.deployment.hardware_safety import NOMINAL_JOINT_LIMITS_RAD
from tests.test_robot_observation_bridge import (execution_args, delivery_state, execution_result,
                                                inference_launch, BIMANUAL_DELIVERY_METADATA, JOINT_METADATA, FakePiper)


class FakeWorld:
    def __init__(self, *, acknowledge=True):
        self.calls = []
        self.acknowledge = acknowledge
        self.mode = {s: 1 for s in ('left', 'right')}
        self.reference = {s: EEFReference(np.array([.2, .2, .3]), np.eye(3), .035) for s in self.mode}
        self.stale = self.failed = self.moving = False
        self.follow = True
        self.pipers = {}
        for side in self.mode:
            piper = Mock()
            def mode(*args, side=side):
                self.calls.append((side, 'mode', args))
                if self.acknowledge:
                    self.mode[side] = args[1]
            def pose(*args, side=side):
                self.calls.append((side, 'pose', args))
                if self.follow:
                    old = self.reference[side]
                    self.reference[side] = EEFReference(np.array(args[:3])/1e6,
                        Rotation.from_euler('xyz', np.array(args[3:])/1000, degrees=True).as_matrix(), old.gripper_m)
            def grip(*args, side=side):
                self.calls.append((side, 'grip', args))
                old = self.reference[side]
                self.reference[side] = EEFReference(old.xyz, old.rotation, args[0]/1e6)
            piper.MotionCtrl_2.side_effect = mode
            piper.EndPoseCtrl.side_effect = pose
            piper.GripperCtrl.side_effect = grip
            self.pipers[side] = piper

    def feedback(self):
        if self.failed:
            raise NativeEEFError('CAN send failed')
        stamp = time.time() - (2 if self.stale else 0)
        return {side: dict(reference=reference, qpos=np.array([0., 1., -1., 0., 0., 0., reference.gripper_m]),
            joint_limits_rad=NOMINAL_JOINT_LIMITS_RAD.tolist(),
            stamp=stamp, status=dict(ctrl_mode=1, mode_feed=self.mode[side], arm_status=0, err_code=0,
                feedback_fresh=not self.stale, feedback_timestamp=stamp, motion_status=int(self.moving)))
            for side, reference in self.reference.items()}


def wait_for(predicate, timeout=1.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.002)
    raise AssertionError('fake native publisher did not reach expected state')


class NativeLoopTest(unittest.TestCase):
    def start(self, **kwargs):
        world = FakeWorld()
        loop = NativeEEFLoop(world.pipers, world.feedback, hz=100, settle_s=.02, **kwargs)
        self.addCleanup(loop.close, stop_timeout_s=.1)
        loop.start()
        wait_for(lambda: loop.metrics()['settled'])
        return world, loop

    def test_units_rotation_and_invalid_values(self):
        value = EEFReference(np.array([.123456, -.1, .2]), Rotation.from_euler('xyz', [10, -20, 30], degrees=True).as_matrix(), .0919)
        self.assertEqual(value.wire(), (123456, -100000, 200000, 10000, -20000, 30000))
        for bad in (np.array([np.nan, 0, 0]), np.array([1e9, 0, 0])):
            with self.assertRaises((NativeEEFError, ValueError)):
                EEFReference(bad, np.eye(3), .03).wire()

    def test_mode_ack_precedes_commands_and_close_joins_owner(self):
        world, loop = self.start()
        target = EEFReference(np.array([.21, .2, .3]), np.eye(3), .04)
        sample = loop.submit({s: target for s in world.pipers})
        self.assertGreater(sample['sequence'], 0)
        self.assertEqual(world.calls[0][1:], ('mode', (1, 0, 10, 0)))
        for piper in world.pipers.values():
            piper.JointCtrl.assert_not_called()
            piper.DisableArm.assert_not_called()
        self.assertTrue(loop.close())
        count = len(world.calls)
        time.sleep(.02)
        self.assertEqual(len(world.calls), count)
        self.assertFalse(loop._thread.is_alive())
        with self.assertRaises(NativeEEFError):
            loop.submit({s: target for s in world.pipers})

    def test_paired_validation_does_not_publish_partial_target(self):
        def check(side, target):
            if target.xyz[0] > .3:
                raise NativeEEFError('workspace')
        world, loop = self.start(check_target=check)
        with self.assertRaisesRegex(NativeEEFError, 'workspace'):
            loop.submit(dict(left=EEFReference(np.array([.21, .2, .3]), np.eye(3), .03),
                             right=EEFReference(np.array([.4, .2, .3]), np.eye(3), .03)))
        self.assertFalse(any(name == 'pose' and args[0] == 210000 for _, name, args in world.calls))

    def test_mode_ack_timeout_cannot_authorize_return(self):
        world = FakeWorld(acknowledge=False)
        loop = NativeEEFLoop(world.pipers, world.feedback, mode_timeout_s=.03)
        with self.assertRaisesRegex(NativeEEFError, 'acknowledgement'):
            loop.start()
        self.assertFalse(loop.close(stop_timeout_s=.01))
        self.assertIsNone(loop._thread)

    def test_firmware_limit_rejection_preserves_handoff_evidence(self):
        world = FakeWorld()
        def feedback():
            frames = world.feedback()
            if any(name == 'pose' for _, name, _ in world.calls):
                frames['left']['status']['arm_status'] = 4
            return frames
        loop = NativeEEFLoop(world.pipers, feedback)
        with self.assertRaisesRegex(NativeEEFError, 'target joint limit exceeded') as caught:
            loop.start()
        self.assertIn('MOVE_P_handoff', str(caught.exception))
        evidence = loop.metrics()['fault_feedback']
        self.assertEqual(evidence['status']['arm_status'], 4)
        self.assertEqual(evidence['last_sent_pose_units'], world.reference['left'].wire())
        self.assertEqual(evidence['joint_limits_rad'], NOMINAL_JOINT_LIMITS_RAD.tolist())
        count = len(world.calls)
        self.assertFalse(loop.close(stop_timeout_s=.01))
        self.assertEqual(len(world.calls), count)
        for piper in world.pipers.values():
            piper.DisableArm.assert_not_called()
            piper.JointCtrl.assert_not_called()

    def test_watchdog_holds_and_invalidates_execution_epoch(self):
        world, loop = self.start(target_timeout_s=.03)
        loop.submit({s: EEFReference(np.array([.21, .2, .3]), np.eye(3), .04) for s in world.pipers})
        wait_for(lambda: loop.metrics()['epoch'] == 1)
        self.assertTrue(loop.metrics()['holding'])
        self.assertEqual(loop.metrics()['hold_reason'], 'reference_timeout')
        wait_for(lambda: loop.metrics()['settled'])
        self.assertTrue(loop.close())

    def test_fault_or_unsettled_stop_never_reports_safe_return(self):
        for fault in ('stale', 'failed', 'mode', 'moving'):
            with self.subTest(fault=fault):
                world, loop = self.start()
                loop.submit({s: EEFReference(np.array([.21, .2, .3]), np.eye(3), .04) for s in world.pipers})
                if fault == 'mode':
                    world.mode['right'] = 1
                else:
                    setattr(world, fault, True)
                if fault != 'moving':
                    wait_for(lambda: loop.metrics()['fault'])
                self.assertFalse(loop.close(stop_timeout_s=.04))
                for piper in world.pipers.values():
                    piper.JointCtrl.assert_not_called()
                    piper.DisableArm.assert_not_called()

    def test_frozen_timestamps_cannot_establish_settling(self):
        world, loop = self.start()
        loop.submit({s: EEFReference(np.array([.21, .2, .3]), np.eye(3), .04) for s in world.pipers})
        frame = world.feedback()
        loop.read_feedback = lambda: frame
        self.assertFalse(loop.close(stop_timeout_s=.08))

    def test_startup_hold_timeout_latches_fault(self):
        world = FakeWorld()
        world.moving = True
        loop = NativeEEFLoop(world.pipers, world.feedback, hz=100, settle_s=.01, hold_timeout_s=.04)
        self.addCleanup(loop.close, stop_timeout_s=.01)
        with self.assertRaisesRegex(NativeEEFError, 'did not settle'):
            loop.start()
        self.assertIn('did not settle', loop.metrics()['fault'])
        self.assertIsNone(loop._thread)
        self.assertFalse(loop.close(stop_timeout_s=.01))

    def test_boundary_on_either_arm_blocks_both_before_mode_switch(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                world = FakeWorld()
                def feedback():
                    frames = world.feedback()
                    frames[side]['qpos'][1] = 0.
                    return frames
                loop = NativeEEFLoop(world.pipers, feedback)
                with self.assertRaisesRegex(NativeEEFError, 'startup joint clearance'):
                    loop.start()
                self.assertEqual(world.calls, [])
                self.assertFalse(loop.close(stop_timeout_s=.01))

    def test_delayed_firmware_rejection_after_mode_ack_is_startup_failure(self):
        world = FakeWorld()
        def feedback():
            frames = world.feedback()
            if sum(name == 'pose' for _, name, _ in world.calls) >= 4:
                frames['left']['status']['arm_status'] = 4
            return frames
        loop = NativeEEFLoop(world.pipers, feedback, hz=100, settle_s=.03)
        with self.assertRaisesRegex(NativeEEFError, 'target joint limit exceeded'):
            loop.start()
        self.assertEqual(loop.metrics()['fault_feedback']['phase'], 'startup_hold_confirmation')
        self.assertIsNone(loop._thread)
        self.assertFalse(loop.close(stop_timeout_s=.01))


class NativeClientTest(unittest.TestCase):
    def controller(self, **args):
        protocol = validate_policy_metadata(BIMANUAL_DELIVERY_METADATA, 'both', 'bimanual')
        controller = ExecutionController({s: FakePiper() for s in ('left', 'right')}, execution_args(
            arm_mode='bimanual', arm_side='both', servo_backend='ruckig', eef_control_mode='piper_eef', **args))
        controller.configure_protocol(protocol)
        controller.robot_enabled = {'left', 'right'}
        state = np.tile(delivery_state(), 2)
        qpos = np.tile([0., 1., -1., 0., 0., 0., .035], 2)
        return controller, protocol, state, qpos

    def test_native_dispatch_skips_ik_and_joint_writer(self):
        controller, protocol, state, qpos = self.controller(pose_trajectory=True)
        controller.rtc_enabled = True
        controller.native_loop = Mock()
        controller.native_loop.metrics.return_value = dict(fault=None, stopping=False, epoch=0, holding=False, settled=False)
        controller.native_loop.submit.return_value = dict(sequence=3, grippers={'left': .035, 'right': .035})
        actions = np.zeros((4, 14)); actions[:, [0, 7]] = .01; actions[:, [6, 13]] = .5
        controller.queue_result(execution_result(actions), state, qpos, protocol,
            {key: time.time() for key in protocol.camera_keys}, .01)
        controller._solve_ik_batch = Mock(side_effect=AssertionError('local IK used'))
        controller._servo_publish_target = Mock(side_effect=AssertionError('joint owner used'))
        self.assertTrue(controller.execute_next(state, qpos, protocol))
        controller.native_loop.submit.assert_called_once()
        self.assertFalse(controller._pose_trajectories)
        self.assertIsNone(controller.ik_recovery_target)
        for piper in controller.pipers.values():
            self.assertEqual(piper.calls, [])
        trace = controller.last_actuator_command['sides']['left']
        self.assertIsNone(trace['commanded_joints_rad'])
        self.assertEqual(trace['control_path'], 'piper_native_eef')
        self.assertEqual(trace['pose_trajectory_bypass_reason'], 'rtc_continuous_tracking')

    def test_native_mode_rejects_joint_policy_and_does_not_start_ik_workers(self):
        controller, protocol, _, _ = self.controller()
        with patch('bimanual_vla.deployment.ik_workers.PiperIKWorkers') as workers:
            controller.configure_ik_workers(protocol)
            workers.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'delivery'):
            controller.configure_protocol(validate_policy_metadata(JOINT_METADATA, 'right', 'single'))

    def test_native_nonrtc_host_smoothing_can_be_enabled_or_disabled(self):
        for smoothing in (False, True):
            with self.subTest(smoothing=smoothing):
                controller, protocol, state, qpos = self.controller(pose_trajectory=smoothing)
                controller.rtc_enabled = False
                controller.native_loop = Mock()
                controller.native_loop.metrics.return_value = dict(fault=None, stopping=False, epoch=0, holding=False, settled=False)
                controller.native_loop.submit.return_value = dict(sequence=1, grippers={'left': .035, 'right': .035})
                actions = np.zeros((4, 14)); actions[:, [0, 7]] = .01; actions[:, [6, 13]] = .5
                controller.queue_result(execution_result(actions), state, qpos, protocol,
                    {key: time.time() for key in protocol.camera_keys}, .01)
                self.assertTrue(controller.execute_next(state, qpos, protocol))
                target = controller.native_loop.submit.call_args.args[0]['left']
                if smoothing:
                    self.assertGreater(target.xyz[0], .2)
                    self.assertLess(target.xyz[0], .21)
                else:
                    self.assertAlmostEqual(target.xyz[0], .21)
                self.assertIsNone(controller.servo_loop)
                self.assertEqual(bool(controller._pose_trajectories), smoothing)

    def test_native_start_cannot_steal_live_joint_owner(self):
        controller, _, _, _ = self.controller()
        controller.servo_loop = Mock()
        with self.assertRaisesRegex(RuntimeHardwareFault, 'joint owner'):
            controller._start_native()
        self.assertIsNone(controller.native_loop)

    def test_native_hold_clears_prefix_and_requires_post_settle_observation(self):
        controller, protocol, state, qpos = self.controller()
        controller.rtc_enabled = True
        controller.native_loop = Mock()
        metrics = dict(fault=None, stopping=False, epoch=0, holding=False, settled=False)
        controller.native_loop.metrics.side_effect = lambda: metrics.copy()
        actions = np.zeros((4, 14)); actions[:, [6, 13]] = .5
        result = execution_result(actions)
        controller.queue_result(result, state, qpos, protocol,
            {key: time.time() for key in protocol.camera_keys}, .01)
        metrics.update(epoch=1, holding=True)
        self.assertFalse(controller.native_inference_ready())
        self.assertFalse(controller.execute_next(state, qpos, protocol))
        self.assertEqual(controller.pending_action_count, 0)
        self.assertIsNone(controller.rtc_request_metadata(protocol)['previous_chunk_generation'])
        self.assertIsNone(controller.tracking_lag_required_after_monotonic)
        metrics['settled'] = True
        self.assertFalse(controller.execute_next(state, qpos, protocol))
        barrier = controller.tracking_lag_required_after_monotonic
        self.assertIsNotNone(barrier)
        for offset, accepted in ((-.01, False), (.01, True)):
            launch, arrived = inference_launch(state, qpos, generation=51, captured_monotonic=barrier+offset)
            self.assertEqual(controller.accept_inference_result(result, launch, protocol,
                arrived_at=arrived, arrived_monotonic=barrier+.02,
                min_steps_override=1, skip_steps_override=0, blend_steps_override=0), accepted)
        self.assertFalse(controller.tracking_lag_active)
        controller.native_loop.submit.assert_not_called()

    def test_close_failure_blocks_joint_return_handoff(self):
        controller, _, _, _ = self.controller()
        controller.native_loop = Mock()
        controller.native_loop.close.return_value = False
        self.assertFalse(controller.close_servo())
        controller._enable_robot = Mock()
        with self.assertRaises(RuntimeHardwareFault):
            controller.prepare_native_return_handoff()
        controller._enable_robot.assert_not_called()

    def test_successful_close_requires_acknowledged_joint_handoff(self):
        controller, _, _, qpos = self.controller()
        controller.native_loop = Mock()
        controller.native_loop.close.return_value = True
        self.assertTrue(controller.close_servo())
        controller._enable_robot = Mock()
        with patch('bimanual_vla.deployment.client.read_output_qpos', return_value=qpos[:7]):
            controller.prepare_native_return_handoff()
        self.assertEqual(controller._enable_robot.call_count, 2)
