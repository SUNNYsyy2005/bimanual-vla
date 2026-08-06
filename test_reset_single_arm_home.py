from __future__ import annotations

import math
import socket
import struct
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from reset_single_arm_home import (
    HOME_GRIPPER_M,
    HOME_JOINTS_DEG,
    RAD_FACTOR,
    REAL_EXECUTION_DISABLED_REASON,
    enable_all_motors_while_holding,
    home_qpos,
    plan_home_trajectory,
    preload_movej_while_disabled,
    run,
    stationary_movej_handshake,
    smoothstep,
    wait_for_piper_can_feedback,
)


class FakePiper:
    def __init__(self, enable_results: list[bool] | None = None):
        self.enable_results = list(enable_results or [True])
        self.last_enable_result = False
        self.events: list[tuple] = []
        self.ctrl_mode = 1
        self.mode_feed = 1
        self.arm_status_code = 0

    def GetArmStatus(self):
        return SimpleNamespace(
            arm_status=SimpleNamespace(
                ctrl_mode=self.ctrl_mode,
                arm_status=self.arm_status_code,
                mode_feed=self.mode_feed,
                motion_status=0,
                err_code=0,
            )
        )

    def EnablePiper(self):
        self.last_enable_result = self.enable_results.pop(0)
        self.events.append(("EnablePiper", self.last_enable_result))
        return self.last_enable_result

    def GetArmEnableStatus(self):
        self.events.append(("GetArmEnableStatus", self.last_enable_result))
        return [self.last_enable_result] * 6

    def MotionCtrl_2(self, *args):
        self.events.append(("MotionCtrl_2", *args))
        self.ctrl_mode = args[0]
        self.mode_feed = args[1]

    def JointCtrl(self, *args):
        self.events.append(("JointCtrl", *args))

    def GripperCtrl(self, *args):
        self.events.append(("GripperCtrl", *args))

    def EmergencyStop(self, value):
        self.events.append(("EmergencyStop", value))
        self.arm_status_code = 0 if value == 0x02 else 1


class FakeCanSocket:
    def __init__(self, frames: list[bytes]):
        self.frames = list(frames)
        self.closed = False

    def bind(self, address):
        self.address = address

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self, _size):
        if not self.frames:
            raise socket.timeout
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class ResetSingleArmHomeTest(unittest.TestCase):
    @staticmethod
    def can_frame(can_id: int) -> bytes:
        return struct.pack("=IB3x8s", can_id, 8, bytes(8))

    def test_home_target_is_j1_90_degrees_with_closed_gripper(self):
        target = home_qpos()
        np.testing.assert_allclose(np.degrees(target[:6]), HOME_JOINTS_DEG)
        self.assertEqual(target[6], HOME_GRIPPER_M)

    def test_smoothstep_has_zero_and_one_endpoints(self):
        values = smoothstep(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_planner_reaches_home_and_respects_peak_speed(self):
        start = np.array([1.0, 1.6, -1.5, 0.4, 1.0, -0.8, 0.04])
        hz = 20.0
        max_speed = math.radians(10.0)
        trajectory, duration_s = plan_home_trajectory(
            start,
            control_hz=hz,
            min_duration_s=2.0,
            max_joint_speed_rad_s=max_speed,
        )
        np.testing.assert_allclose(trajectory[-1], home_qpos())
        commands = np.vstack((start, trajectory))
        discrete_speed = np.abs(np.diff(commands[:, :6], axis=0)) * hz
        self.assertLessEqual(float(np.max(discrete_speed)), max_speed + 1e-4)
        self.assertAlmostEqual(duration_s, len(trajectory) / hz)

    def test_minimum_duration_is_preserved_for_nearby_start(self):
        start = home_qpos().copy()
        start[0] += 0.01
        trajectory, duration_s = plan_home_trajectory(
            start,
            control_hz=20.0,
            min_duration_s=5.0,
            max_joint_speed_rad_s=math.radians(10.0),
        )
        self.assertEqual(len(trajectory), 100)
        self.assertAlmostEqual(duration_s, 5.0)

    def test_passive_can_preflight_requires_complete_joint_feedback_trio(self):
        fake_socket = FakeCanSocket(
            [self.can_frame(0x2A5), self.can_frame(0x2A6), self.can_frame(0x2A7)]
        )
        with (
            mock.patch("reset_single_arm_home.require_can_interface_up"),
            mock.patch("reset_single_arm_home.socket.socket", return_value=fake_socket),
        ):
            received = wait_for_piper_can_feedback("can0", timeout_s=1.0)

        self.assertEqual(received, {0x2A5, 0x2A6, 0x2A7})
        self.assertEqual(fake_socket.address, ("can0",))
        self.assertTrue(fake_socket.closed)

    def test_passive_can_preflight_rejects_silent_robot(self):
        fake_socket = FakeCanSocket([])
        with (
            mock.patch("reset_single_arm_home.require_can_interface_up"),
            mock.patch("reset_single_arm_home.socket.socket", return_value=fake_socket),
        ):
            with self.assertRaisesRegex(RuntimeError, "robot-side CAN controller"):
                wait_for_piper_can_feedback("can0", timeout_s=0.001)

        self.assertTrue(fake_socket.closed)

    def test_current_target_and_movej_are_preloaded_while_all_motors_disabled(self):
        piper = FakePiper([False, False, True])
        piper.ctrl_mode = 0
        piper.mode_feed = 0
        anchor = np.array([1.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.025])
        with (
            mock.patch(
                "reset_single_arm_home.read_qpos",
                return_value=anchor.copy(),
            ),
            mock.patch("reset_single_arm_home.time.sleep"),
        ):
            preload_movej_while_disabled(
                piper,
                anchor,
                speed_pct=1,
                gripper_effort=1000,
                duration_s=0.001,
                preload_hz=100.0,
                max_drift_rad=math.radians(1.0),
                max_feedback_age_s=0.5,
            )

        names = [event[0] for event in piper.events]
        first_joint = names.index("JointCtrl")
        first_mode = names.index("MotionCtrl_2")
        self.assertLess(first_joint, first_mode)
        self.assertNotIn("EnablePiper", names)
        self.assertEqual(piper.ctrl_mode, 1)
        self.assertEqual(piper.mode_feed, 1)

    def test_real_execution_is_hard_disabled_before_hardware_access(self):
        args = SimpleNamespace(execute=True)
        with (
            mock.patch("reset_single_arm_home.competing_controller_processes") as processes,
            mock.patch("reset_single_arm_home.wait_for_piper_can_feedback") as can_read,
            mock.patch("reset_single_arm_home.connect_piper") as connect,
        ):
            with self.assertRaisesRegex(RuntimeError, "safety-locked"):
                run(args)

        processes.assert_not_called()
        can_read.assert_not_called()
        connect.assert_not_called()
        self.assertIn("EmergencyStop(0x02)", REAL_EXECUTION_DISABLED_REASON)

    def test_enable_is_bracketed_by_preloaded_hold_commands(self):
        piper = FakePiper([False, False, True])
        anchor = np.array([1.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.025])
        with (
            mock.patch(
                "reset_single_arm_home.read_qpos",
                return_value=anchor.copy(),
            ),
            mock.patch("reset_single_arm_home.time.sleep"),
        ):
            _, enable_status, _ = enable_all_motors_while_holding(
                piper,
                anchor,
                speed_pct=1,
                gripper_effort=1000,
                timeout_s=1.0,
                max_drift_rad=math.radians(1.0),
                max_feedback_age_s=0.5,
            )

        self.assertEqual(enable_status, [True] * 6)
        self.assertEqual(
            [event for event in piper.events if event[0] == "EnablePiper"],
            [
                ("EnablePiper", False),
                ("EnablePiper", False),
                ("EnablePiper", True),
            ],
        )
        for index, event in enumerate(piper.events):
            if event[0] == "EnablePiper":
                self.assertEqual(piper.events[index - 1][0], "GripperCtrl")
                self.assertEqual(piper.events[index + 1][0], "MotionCtrl_2")

    def test_handshake_first_movej_target_is_latest_feedback_in_sdk_order(self):
        piper = FakePiper()
        anchor = np.array([1.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.025])
        with (
            mock.patch(
                "reset_single_arm_home.read_qpos",
                side_effect=[anchor.copy(), anchor.copy()],
            ),
            mock.patch("reset_single_arm_home.time.sleep"),
        ):
            stationary_movej_handshake(
                piper,
                anchor,
                speed_pct=1,
                gripper_effort=1000,
                duration_s=0.001,
                handshake_hz=100.0,
                max_drift_rad=math.radians(1.0),
                max_feedback_age_s=0.5,
            )

        control_events = [
            event
            for event in piper.events
            if event[0] in {"MotionCtrl_2", "JointCtrl", "GripperCtrl"}
        ]
        self.assertEqual(
            [event[0] for event in control_events],
            ["MotionCtrl_2", "JointCtrl", "GripperCtrl"],
        )
        self.assertEqual(control_events[0][1:], (0x01, 0x01, 1, 0x00))
        expected_raw_joints = tuple(
            np.rint(anchor[:6] * RAD_FACTOR).astype(np.int64).tolist()
        )
        self.assertEqual(control_events[1][1:], expected_raw_joints)

    def test_handshake_drift_latches_estop_without_another_hold_command(self):
        piper = FakePiper()
        anchor = np.zeros(7)
        drifted = anchor.copy()
        drifted[3] = math.radians(2.0)
        with (
            mock.patch(
                "reset_single_arm_home.read_qpos",
                side_effect=[anchor.copy(), drifted],
            ),
            mock.patch("reset_single_arm_home.time.sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "emergency stop latched"):
                stationary_movej_handshake(
                    piper,
                    anchor,
                    speed_pct=1,
                    gripper_effort=1000,
                    duration_s=0.02,
                    handshake_hz=100.0,
                    max_drift_rad=math.radians(1.0),
                    max_feedback_age_s=0.5,
                )

        self.assertEqual(
            len([event for event in piper.events if event[0] == "JointCtrl"]),
            1,
        )
        self.assertEqual(piper.events[-1], ("EmergencyStop", 0x01))


if __name__ == "__main__":
    unittest.main()
