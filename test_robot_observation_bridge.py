from __future__ import annotations

from types import SimpleNamespace
import time
import unittest

import numpy as np

from robot_observation_bridge import (
    ExecutionBlocked,
    ExecutionController,
    PolicyProtocol,
    build_checked_joint_target,
    build_checked_target,
    build_observation,
    validate_policy_metadata,
)


DELIVERY_METADATA = {
    "transport": "openpi_websocket_v1",
    "schema": "delivery",
    "state_dim": 10,
    "action_dim": 7,
    "arm_side": "right",
    "action_semantics": "eef_delta_base_xyz_left_rotvec_gripper_target",
    "camera_keys": ["cam_high", "cam_wrist"],
}

JOINT_METADATA = {
    "transport": "openpi_websocket_v1",
    "schema": "joint",
    "state_dim": 7,
    "action_dim": 7,
    "arm_side": "right",
    "action_semantics": "absolute_joint_position",
    "camera_keys": ["cam_high", "cam_right_wrist"],
}

BIMANUAL_JOINT_METADATA = {
    "transport": "openpi_websocket_v1",
    "arm_mode": "bimanual",
    "schema": "joint",
    "state_dim": 14,
    "action_dim": 14,
    "arm_side": "both",
    "action_semantics": "absolute_joint_position",
    "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
}

BIMANUAL_DELIVERY_METADATA = {
    "transport": "openpi_websocket_v1",
    "arm_mode": "bimanual",
    "schema": "delivery",
    "state_dim": 20,
    "action_dim": 14,
    "arm_side": "both",
    "action_semantics": "eef_delta_base_xyz_left_rotvec_gripper_target",
    "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
}


class FakeExecution:
    def metadata(self):
        return {"allow_execution": False, "execution_state": "client_disabled"}


class FakePiper:
    def __init__(self):
        self.calls = []

    def GetArmStatus(self):
        status = SimpleNamespace(
            ctrl_mode=1,
            arm_status=0,
            mode_feed=1,
            motion_status=0,
            err_code=0,
        )
        return SimpleNamespace(arm_status=status)

    def EnablePiper(self):
        self.calls.append(("EnablePiper",))
        return True

    def ModeCtrl(self, *args):
        self.calls.append(("ModeCtrl", *args))

    def JointCtrl(self, *args):
        self.calls.append(("JointCtrl", *args))

    def EndPoseCtrl(self, *args):
        self.calls.append(("EndPoseCtrl", *args))

    def MotionCtrl_2(self, *args):
        self.calls.append(("MotionCtrl_2", *args))

    def GripperCtrl(self, *args):
        self.calls.append(("GripperCtrl", *args))


class ProtocolCompatibilityTest(unittest.TestCase):
    def test_accepts_delivery_and_joint_metadata(self):
        delivery = validate_policy_metadata(DELIVERY_METADATA, "right")
        joint = validate_policy_metadata(JOINT_METADATA, "right")
        self.assertEqual(delivery.schema, "delivery")
        self.assertEqual(joint.schema, "joint")
        self.assertEqual(joint.camera_keys, ("cam_high", "cam_right_wrist"))

    def test_accepts_bimanual_joint_and_delivery_metadata(self):
        joint = validate_policy_metadata(BIMANUAL_JOINT_METADATA, "both", "bimanual")
        delivery = validate_policy_metadata(BIMANUAL_DELIVERY_METADATA, "both", "bimanual")
        self.assertEqual((joint.state_dim, joint.action_dim), (14, 14))
        self.assertEqual((delivery.state_dim, delivery.action_dim), (20, 14))
        self.assertEqual(joint.arm_side, "both")
        self.assertEqual(
            set(delivery.camera_keys),
            {"cam_high", "cam_left_wrist", "cam_right_wrist"},
        )

    def test_rejects_joint_metadata_with_delivery_camera_key(self):
        metadata = dict(JOINT_METADATA, camera_keys=["cam_high", "cam_wrist"])
        with self.assertRaisesRegex(RuntimeError, "camera_keys"):
            validate_policy_metadata(metadata, "right")

    def test_builds_schema_specific_observations(self):
        args = SimpleNamespace(
            can="can0",
            cam_high_device="/dev/video8",
            cam_wrist_device="/dev/video16",
            arm_side="right",
        )
        delivery_state = np.arange(10, dtype=np.float32)
        qpos = np.arange(7, dtype=np.float32) / 100.0
        images = {
            "cam_high": np.zeros((4, 4, 3), dtype=np.uint8),
            "cam_wrist": np.ones((4, 4, 3), dtype=np.uint8),
        }
        timestamps = {"cam_high": 1.0, "cam_wrist": 2.0}

        delivery = build_observation(
            delivery_state=delivery_state,
            qpos=qpos,
            protocol=validate_policy_metadata(DELIVERY_METADATA, "right"),
            images=images,
            image_timestamps=timestamps,
            instruction="test",
            source_name="robot",
            args=args,
            execution=FakeExecution(),
        )
        joint = build_observation(
            delivery_state=delivery_state,
            qpos=qpos,
            protocol=validate_policy_metadata(JOINT_METADATA, "right"),
            images=images,
            image_timestamps=timestamps,
            instruction="test",
            source_name="robot",
            args=args,
            execution=FakeExecution(),
        )

        np.testing.assert_array_equal(delivery["state"], delivery_state)
        self.assertEqual(set(delivery["images"]), {"cam_high", "cam_wrist"})
        np.testing.assert_array_equal(joint["state"], qpos)
        self.assertEqual(set(joint["images"]), {"cam_high", "cam_right_wrist"})

    def test_builds_bimanual_observation_in_left_right_order(self):
        args = SimpleNamespace(
            left_can="can0",
            right_can="can1",
            cam_high_device="/dev/video8",
            cam_left_wrist_device="/dev/video14",
            cam_right_wrist_device="/dev/video16",
            cam_wrist_device="/dev/video16",
        )
        left_qpos = np.arange(7, dtype=np.float32)
        right_qpos = np.arange(7, dtype=np.float32) + 10
        images = {
            "cam_high": np.zeros((4, 4, 3), dtype=np.uint8),
            "cam_left_wrist": np.ones((4, 4, 3), dtype=np.uint8),
            "cam_right_wrist": np.full((4, 4, 3), 2, dtype=np.uint8),
        }
        timestamps = {key: time.time() for key in images}
        observation = build_observation(
            delivery_state=np.arange(20, dtype=np.float32),
            qpos=np.concatenate([left_qpos, right_qpos]),
            protocol=validate_policy_metadata(BIMANUAL_JOINT_METADATA, "both", "bimanual"),
            images=images,
            image_timestamps=timestamps,
            instruction="test",
            source_name="robot",
            args=args,
            execution=FakeExecution(),
        )
        np.testing.assert_array_equal(observation["state"][:7], left_qpos)
        np.testing.assert_array_equal(observation["state"][7:], right_qpos)
        self.assertEqual(set(observation["images"]), set(images))
        self.assertEqual(observation["client_metadata"]["can_names"], {"left": "can0", "right": "can1"})


class DeliveryExecutionSafetyTest(unittest.TestCase):
    def setUp(self):
        # Identity rotation6d and a real-capture-like EEF pose.
        self.state = np.array([0.10, 0.20, 0.25, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.50])
        self.kwargs = {
            "max_translation_step_m": 0.015,
            "max_rotation_step_rad": 0.15,
            "max_gripper_step": 0.25,
            "gripper_range_tolerance": 0.02,
            "workspace_x": (-0.04, 0.30),
            "workspace_y": (0.02, 0.52),
            "workspace_z": (0.12, 0.50),
        }

    def test_small_gripper_overshoot_is_clipped(self):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.008])
        state = self.state.copy()
        state[-1] = 0.80
        target_xyz, _, target_gripper_m = build_checked_target(state, action, **self.kwargs)
        np.testing.assert_allclose(target_xyz, state[:3])
        self.assertAlmostEqual(target_gripper_m, 0.0)

        action[-1] = -0.008
        state[-1] = 0.10
        _, _, target_gripper_m = build_checked_target(state, action, **self.kwargs)
        self.assertGreater(target_gripper_m, 0.0)

    def test_severe_gripper_overshoot_is_blocked(self):
        for value in (1.03, -0.03):
            action = np.zeros(7)
            action[-1] = value
            with self.assertRaisesRegex(ExecutionBlocked, "gripper target"):
                build_checked_target(self.state, action, **self.kwargs)

    def test_gripper_step_is_checked_after_clipping(self):
        action = np.zeros(7)
        action[-1] = 1.008
        state = self.state.copy()
        state[-1] = 0.70
        with self.assertRaisesRegex(ExecutionBlocked, "gripper step"):
            build_checked_target(state, action, **self.kwargs)

    def test_translation_rotation_and_workspace_gates_remain_active(self):
        action = np.zeros(7)
        action[0] = 0.0151
        with self.assertRaisesRegex(ExecutionBlocked, "translation step"):
            build_checked_target(self.state, action, **self.kwargs)

        action = np.zeros(7)
        action[3] = 0.1501
        with self.assertRaisesRegex(ExecutionBlocked, "rotation step"):
            build_checked_target(self.state, action, **self.kwargs)

        state = self.state.copy()
        state[0] = -0.035
        action = np.zeros(7)
        action[0] = -0.006
        action[-1] = state[-1]
        with self.assertRaisesRegex(ExecutionBlocked, "outside workspace"):
            build_checked_target(state, action, **self.kwargs)


class JointExecutionSafetyTest(unittest.TestCase):
    def setUp(self):
        self.qpos = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.03])

    def test_checked_joint_target_accepts_small_absolute_step(self):
        target = self.qpos + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01])
        joints, gripper = build_checked_joint_target(
            self.qpos,
            target,
            max_joint_step_rad=0.3,
            max_gripper_step_m=0.02,
        )
        np.testing.assert_allclose(joints, target[:6])
        self.assertAlmostEqual(gripper, target[6])

    def test_checked_joint_target_rejects_large_step_and_limit_violation(self):
        large_step = self.qpos.copy()
        large_step[0] += 0.31
        with self.assertRaisesRegex(ExecutionBlocked, "joint1 step"):
            build_checked_joint_target(
                self.qpos,
                large_step,
                max_joint_step_rad=0.3,
                max_gripper_step_m=0.02,
            )

        outside_limit = self.qpos.copy()
        outside_limit[1] = -0.01
        with self.assertRaisesRegex(ExecutionBlocked, "joint2 target"):
            build_checked_joint_target(
                self.qpos,
                outside_limit,
                max_joint_step_rad=2.0,
                max_gripper_step_m=0.02,
            )

    def test_execution_controller_uses_joint_control_for_joint_schema(self):
        args = SimpleNamespace(
            allow_execution=True,
            enable_timeout_s=0.1,
            max_action_age_s=2.0,
            max_translation_step_m=0.015,
            max_rotation_step_rad=0.15,
            max_gripper_step=0.25,
            gripper_range_tolerance=0.02,
            max_joint_step_rad=0.3,
            max_joint_gripper_step_m=0.02,
            workspace_x=(0.05, 0.60),
            workspace_y=(-0.45, 0.45),
            workspace_z=(0.02, 0.60),
            speed_pct=10,
            gripper_effort=1000,
        )
        piper = FakePiper()
        execution = ExecutionController(piper, args)
        protocol = PolicyProtocol(
            schema="joint",
            state_dim=7,
            action_dim=7,
            arm_side="right",
            action_semantics="absolute_joint_position",
            camera_keys=("cam_high", "cam_right_wrist"),
        )
        target = self.qpos + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01])
        now = time.time()
        result = {
            "actions": target[None, :],
            "execution_control": {
                "mode": "execute",
                "task_id": "policy-test",
                "session_id": "session-test",
                "server_time": now,
                "expires_at": now + 30.0,
                "revision": 1,
            },
        }
        image_timestamps = {"cam_high": now, "cam_wrist": now}
        delivery_state = np.zeros(10, dtype=np.float32)

        self.assertFalse(
            execution.process(
                result,
                delivery_state,
                self.qpos,
                protocol,
                image_timestamps,
                0.01,
            )
        )
        self.assertTrue(
            execution.process(
                result,
                delivery_state,
                self.qpos,
                protocol,
                image_timestamps,
                0.01,
            )
        )
        names = [call[0] for call in piper.calls]
        self.assertIn("ModeCtrl", names)
        self.assertIn("JointCtrl", names)
        self.assertIn("GripperCtrl", names)
        self.assertNotIn("EndPoseCtrl", names)

    def test_bimanual_joint_execution_commands_both_arms(self):
        args = SimpleNamespace(
            arm_mode="bimanual",
            arm_side="both",
            allow_execution=True,
            enable_timeout_s=0.1,
            max_action_age_s=2.0,
            max_translation_step_m=0.015,
            max_rotation_step_rad=0.15,
            max_gripper_step=0.25,
            gripper_range_tolerance=0.02,
            max_joint_step_rad=0.3,
            max_joint_gripper_step_m=0.02,
            workspace_x=(0.05, 0.60),
            workspace_y=(-0.45, 0.45),
            workspace_z=(0.02, 0.60),
            speed_pct=10,
            gripper_effort=1000,
        )
        pipers = {"left": FakePiper(), "right": FakePiper()}
        execution = ExecutionController(pipers, args)
        protocol = validate_policy_metadata(BIMANUAL_JOINT_METADATA, "both", "bimanual")
        qpos = np.concatenate([self.qpos, self.qpos])
        target = qpos + np.array([0.1, 0, 0, 0, 0, 0, 0.01] * 2)
        now = time.time()
        result = {
            "actions": target[None, :],
            "execution_control": {
                "mode": "execute",
                "task_id": "policy-test",
                "session_id": "session-test",
                "server_time": now,
                "expires_at": now + 30.0,
                "revision": 1,
            },
        }
        image_timestamps = {
            "cam_high": now,
            "cam_left_wrist": now,
            "cam_right_wrist": now,
        }
        self.assertFalse(execution.process(result, np.zeros(20), qpos, protocol, image_timestamps, 0.01))
        self.assertTrue(execution.process(result, np.zeros(20), qpos, protocol, image_timestamps, 0.01))
        for piper in pipers.values():
            names = [call[0] for call in piper.calls]
            self.assertIn("JointCtrl", names)
            self.assertIn("GripperCtrl", names)
            self.assertNotIn("EndPoseCtrl", names)

    def test_bimanual_validation_is_fail_closed_before_either_arm_moves(self):
        args = SimpleNamespace(
            arm_mode="bimanual",
            arm_side="both",
            allow_execution=True,
            enable_timeout_s=0.1,
            max_action_age_s=2.0,
            max_translation_step_m=0.015,
            max_rotation_step_rad=0.15,
            max_gripper_step=0.25,
            gripper_range_tolerance=0.02,
            max_joint_step_rad=0.3,
            max_joint_gripper_step_m=0.02,
            workspace_x=(0.05, 0.60),
            workspace_y=(-0.45, 0.45),
            workspace_z=(0.02, 0.60),
            speed_pct=10,
            gripper_effort=1000,
        )
        pipers = {"left": FakePiper(), "right": FakePiper()}
        execution = ExecutionController(pipers, args)
        execution.robot_enabled = {"left", "right"}
        protocol = validate_policy_metadata(BIMANUAL_JOINT_METADATA, "both", "bimanual")
        qpos = np.concatenate([self.qpos, self.qpos])
        target = qpos.copy()
        target[7] += 0.31
        now = time.time()
        result = {
            "actions": target[None, :],
            "execution_control": {
                "mode": "execute",
                "task_id": "policy-test",
                "session_id": "session-test",
                "server_time": now,
                "expires_at": now + 30.0,
                "revision": 1,
            },
        }
        image_timestamps = {
            "cam_high": now,
            "cam_left_wrist": now,
            "cam_right_wrist": now,
        }
        self.assertFalse(execution.process(result, np.zeros(20), qpos, protocol, image_timestamps, 0.01))
        self.assertIn("joint1 step", execution.blocked_reason)
        for piper in pipers.values():
            names = [call[0] for call in piper.calls]
            self.assertNotIn("JointCtrl", names)
            self.assertNotIn("EndPoseCtrl", names)


if __name__ == "__main__":
    unittest.main()
