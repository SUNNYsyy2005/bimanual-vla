"""Holds, braking and recovery: keep the grasp, never hang, never let the gripper block the arm."""
from types import SimpleNamespace
import unittest

import numpy as np

from bimanual_vla.deployment.client import ExecutionController


class HeldGripperOpeningsTest(unittest.TestCase):
    def setUp(self):
        # 2026-10-29 09:24: left gripper held an object at 18.3 mm while its
        # command was 0.4 mm; recovery then commanded the measured 18.3 mm.
        self.measured = np.array([0, 1, -1, 0, 0, 0, .0183, 0, 1, -1, 0, 0, 0, .0175])
        self.published = np.array([0, 1, -1, 0, 0, 0, .0004, 0, 1, -1, 0, 0, 0, .0083])

    def held(self, servo_loop):
        controller = SimpleNamespace(servo_loop=servo_loop)
        return ExecutionController._held_gripper_openings(controller, self.measured, ("left", "right"))

    def test_keeps_last_published_command_not_contact_opening(self):
        servo = SimpleNamespace(last_published_command=lambda: self.published.copy())
        np.testing.assert_allclose(self.held(servo), [.0004, .0083])

    def test_falls_back_to_measured_without_a_published_command(self):
        for servo in (None, SimpleNamespace(last_published_command=lambda: None),
                      SimpleNamespace(last_published_command=lambda: self.published[:7])):
            np.testing.assert_allclose(self.held(servo), [.0183, .0175])



class GripperNeverBlocksArmTest(unittest.TestCase):
    """Contact or slightly out-of-range feedback must not reject arm targets."""

    def setUp(self):
        from bimanual_vla.deployment import client
        self.client = client
        self.controller = client.ExecutionController(
            object(), SimpleNamespace(pose_trajectory=True, servo_backend="ruckig"))
        self.protocol = SimpleNamespace(arm_mode="bimanual", arm_side="both", schema="delivery",
                                        gripper_semantics=client.GRIPPER_OPENING_FRACTION)
        self.state = np.zeros(10); self.state[3] = self.state[7] = 1.

    def run_ticks(self, measured_m, policy_fraction, ticks=8):
        c = self.client
        commands = []
        for _ in range(ticks):
            target = np.concatenate([self.state, self.state]); target[[9, 19]] = policy_fraction
            queued = c.TimedTarget(0, target.copy(), target.copy(), 0., generation=1, source_index=0)
            qpos = np.zeros(14, dtype=np.float32); qpos[[6, 13]] = measured_m
            filtered = self.controller._filter_gripper_target(queued, qpos, self.protocol)
            # Same arguments as the delivery execution path.
            c._check_delivery_absolute_target(
                self.state, float(qpos[6]), filtered.absolute_target[:10],
                gripper_semantics=self.protocol.gripper_semantics,
                max_translation_step_m=float("inf"), max_rotation_step_rad=float("inf"),
                max_gripper_step=float("inf"), gripper_range_tolerance=c.DEFAULT_GRIPPER_RANGE_TOLERANCE,
                workspace_x=None, workspace_y=None, workspace_z=None)
            commands.append(filtered.absolute_target[9])
        return np.array(commands)

    def test_object_in_gripper_still_commands_full_close(self):
        commands = self.run_ticks(.025, 0.)
        self.assertLess(commands[-1], .001 / .07)  # within 1 mm of fully closed
        self.assertLessEqual(np.max(np.abs(np.diff(commands))), self.client.DEFAULT_MAX_GRIPPER_STEP + 1e-12)

    def test_feedback_outside_range_does_not_freeze_gripper(self):
        for measured, policy in ((-.00007, 1.), (.0704, 0.)):
            commands = self.run_ticks(measured, policy)
            self.assertAlmostEqual(commands[-1], policy, delta=.001 / .07)



class TrackingLagHoldSettleTest(unittest.TestCase):
    """The tracking-lag hold must neither hang on a stuck offset nor hide a driver fault."""

    def controller(self, *, offset=.02, low_speed=None):
        from unittest.mock import Mock
        from bimanual_vla.deployment import client
        self.client = client
        position = np.zeros(12); position[2] = offset
        servo = Mock()
        servo.metrics.return_value = dict(fault=None, paused=False, stopping=False, state=dict(
            position=position, velocity=np.zeros(12), acceleration=np.zeros(12)))
        servo.last_published_command.return_value = np.zeros(14)
        piper = SimpleNamespace() if low_speed is None else SimpleNamespace(GetArmLowSpdInfoMsgs=low_speed)
        return SimpleNamespace(tracking_lag_required_after_monotonic=None, servo_loop=servo,
            tracking_lag_settle_feedback_at=None, control_hz=30., tracking_lag_settle_cycles=0,
            tracking_lag_settle_joints=None, tracking_lag_offset_since=None, tracking_lag_confirm_cycles=3,
            blocked_reason="", pipers={"left": piper, "right": piper})

    def settle(self, controller, ticks):
        for k in range(ticks):
            at = k / 30.
            self.client.ExecutionController._update_tracking_lag_settle(controller, np.zeros(14), at, feedback_at=at)

    def test_stationary_offset_reanchors_at_measured_joints(self):
        controller = self.controller()
        self.settle(controller, 40)
        command = controller.servo_loop.submit.call_args.args[0]
        np.testing.assert_array_equal(command, np.zeros(14))
        self.assertTrue(controller.servo_loop.submit.call_args.kwargs["hold"])

    def test_disabled_joint_stops_instead_of_waiting(self):
        import time
        motor = lambda enabled: SimpleNamespace(can_id=1, foc_status=SimpleNamespace(
            driver_enable_status=enabled, voltage_too_low=False, motor_overheating=False,
            driver_overcurrent=False, driver_overheating=False, collision_status=False,
            driver_error_status=False, stall_status=False))
        message = SimpleNamespace(time_stamp=time.time(), Hz=40., **{
            f"motor_{i}": motor(i != 3) for i in range(1, 7)})
        controller = self.controller(low_speed=lambda: message)
        with self.assertRaisesRegex(self.client.RuntimeHardwareFault, "driver fault during tracking-lag hold"):
            self.settle(controller, 2)


if __name__ == "__main__":
    unittest.main()
