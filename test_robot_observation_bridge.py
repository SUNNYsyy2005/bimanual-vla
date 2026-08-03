from __future__ import annotations

from types import SimpleNamespace
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robot_observation_bridge import (
    ExecutionBlocked,
    ExecutionController,
    PiperContinuousIK,
    PiperFeedbackStaleError,
    PolicyProtocol,
    RAD_FACTOR,
    _require_fresh_feedback,
    aggregate_action_chunk,
    build_checked_joint_target,
    build_checked_target,
    build_observation,
    resolve_action_chunk_steps,
    validate_policy_metadata,
)
from piper_data_contract import build_delivery_state


DELIVERY_METADATA = {
    "transport": "openpi_websocket_v1",
    "schema": "delivery",
    "state_dim": 10,
    "action_dim": 7,
    "arm_side": "right",
    "action_semantics": "eef_delta_base_xyz_left_rotvec_gripper_target",
    "camera_keys": ["cam_high", "cam_wrist"],
    "action_hz": 20,
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
        return SimpleNamespace(
            arm_status=status,
            time_stamp=time.time(),
            Hz=200.0,
        )

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


class ActionChunkTimingTest(unittest.TestCase):
    def test_stale_piper_feedback_is_rejected(self):
        message = SimpleNamespace(time_stamp=time.time(), Hz=200.0)
        _require_fresh_feedback({"joint": message})
        message.time_stamp -= 1.0
        with self.assertRaises(PiperFeedbackStaleError):
            _require_fresh_feedback({"joint": message})

    def test_resolves_model_steps_per_command_interval(self):
        self.assertEqual(resolve_action_chunk_steps(action_hz=20, command_hz=5), 4)
        self.assertEqual(resolve_action_chunk_steps(action_hz=20, command_hz=10), 2)
        self.assertEqual(resolve_action_chunk_steps(action_hz=20, command_hz=30), 1)
        self.assertEqual(
            resolve_action_chunk_steps(action_hz=20, command_hz=5, override=3),
            3,
        )

    def test_aggregates_single_arm_delivery_prefix(self):
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        actions = np.zeros((4, 7), dtype=np.float64)
        actions[:, :3] = [
            [0.001, 0.002, 0.0],
            [0.002, 0.0, 0.001],
            [0.0, -0.001, 0.002],
            [0.003, 0.001, -0.001],
        ]
        actions[0, 3:6] = [0.1, 0.0, 0.0]
        actions[1, 3:6] = [0.0, 0.2, 0.0]
        actions[2, 3:6] = [0.0, 0.0, -0.1]
        actions[:, 6] = [0.1, 0.2, 0.3, 0.4]

        command, used_steps = aggregate_action_chunk(actions, protocol, 4)

        self.assertEqual(used_steps, 4)
        np.testing.assert_allclose(command[:3], actions[:, :3].sum(axis=0), atol=1e-12)
        expected_rotation = np.eye(3)
        for rotvec in actions[:, 3:6]:
            from scipy.spatial.transform import Rotation

            expected_rotation = Rotation.from_rotvec(rotvec).as_matrix() @ expected_rotation
        from scipy.spatial.transform import Rotation

        np.testing.assert_allclose(
            Rotation.from_rotvec(command[3:6]).as_matrix(),
            expected_rotation,
            atol=1e-12,
        )
        self.assertAlmostEqual(command[6], 0.4)

    def test_read_only_websocket_action_can_reach_rotation_conversion(self):
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        actions = np.zeros((2, 7), dtype=np.float64)
        actions[0, 3] = 0.01
        actions[0, 6] = 0.5
        actions.setflags(write=False)
        command, used_steps = aggregate_action_chunk(actions, protocol, 2)
        self.assertEqual(used_steps, 2)
        self.assertTrue(command.flags.writeable)
        self.assertTrue(np.all(np.isfinite(command)))

    def test_aggregates_each_bimanual_delivery_arm_independently(self):
        protocol = validate_policy_metadata(BIMANUAL_DELIVERY_METADATA, "both", "bimanual")
        actions = np.zeros((2, 14), dtype=np.float64)
        actions[:, 0] = [0.001, 0.002]
        actions[:, 8] = [-0.003, -0.004]
        actions[:, 6] = [0.1, 0.2]
        actions[:, 13] = [0.8, 0.7]

        command, used_steps = aggregate_action_chunk(actions, protocol, 4)

        self.assertEqual(used_steps, 2)
        self.assertAlmostEqual(command[0], 0.003)
        self.assertAlmostEqual(command[8], -0.007)
        self.assertAlmostEqual(command[6], 0.2)
        self.assertAlmostEqual(command[13], 0.7)

    def test_joint_chunk_selects_future_absolute_target(self):
        protocol = validate_policy_metadata(JOINT_METADATA, "right")
        actions = np.arange(28, dtype=np.float64).reshape(4, 7) / 100.0

        command, used_steps = aggregate_action_chunk(actions, protocol, 4)

        self.assertEqual(used_steps, 4)
        np.testing.assert_allclose(command, actions[3])

    def test_rejects_malformed_or_nonfinite_chunk(self):
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        with self.assertRaisesRegex(ExecutionBlocked, "shape"):
            aggregate_action_chunk(np.zeros((2, 6)), protocol, 2)
        actions = np.zeros((2, 7))
        actions[1, 0] = np.nan
        with self.assertRaisesRegex(ExecutionBlocked, "non-finite"):
            aggregate_action_chunk(actions, protocol, 2)

    def test_shadow_client_still_reports_composed_action_preview(self):
        args = SimpleNamespace(
            arm_mode="single",
            arm_side="right",
            allow_execution=False,
            hz=5.0,
            action_hz=None,
            action_chunk_steps=None,
        )
        execution = ExecutionController(FakePiper(), args)
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        execution.configure_protocol(protocol)
        actions = np.zeros((4, 7), dtype=np.float64)
        actions[:, 0] = 0.002
        actions[:, 6] = [0.1, 0.2, 0.3, 0.4]

        self.assertFalse(
            execution.process(
                {"actions": actions},
                np.zeros(10, dtype=np.float32),
                np.zeros(7, dtype=np.float32),
                protocol,
                {},
                0.01,
            )
        )
        metadata = execution.metadata()
        self.assertEqual(metadata["last_action_chunk_steps"], 4)
        np.testing.assert_allclose(
            metadata["last_composed_action"],
            [0.008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
        )
        self.assertIsNotNone(metadata["last_composed_action_at"])
        self.assertEqual(metadata["execution_state"], "client_disabled")


class ProtocolCompatibilityTest(unittest.TestCase):
    def test_accepts_delivery_and_joint_metadata(self):
        delivery = validate_policy_metadata(DELIVERY_METADATA, "right")
        joint = validate_policy_metadata(JOINT_METADATA, "right")
        self.assertEqual(delivery.schema, "delivery")
        self.assertEqual(delivery.action_hz, 20.0)
        self.assertEqual(joint.schema, "joint")
        self.assertIsNone(joint.action_hz)
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


class ContinuousIKTest(unittest.TestCase):
    def setUp(self):
        self.solver = PiperContinuousIK()
        self.joints = np.array(
            [1.7377021, 0.00226893, -0.0148353, -0.21961477, 0.29255208, 0.22399555],
            dtype=np.float64,
        )

    def test_solves_nearby_pose_without_switching_joint_branch(self):
        xyz, rotation = self.solver.pose(self.joints)
        target_xyz = xyz + np.array([0.0002, 0.0025, -0.0003])
        target_rotation = (
            Rotation.from_rotvec([-0.015, 0.001, -0.001]).as_matrix() @ rotation
        )
        target_rpy = Rotation.from_matrix(target_rotation).as_euler("xyz", degrees=True)

        solved = self.solver.solve(
            self.joints,
            target_xyz,
            target_rpy,
            max_joint_step_rad=0.08,
            position_tolerance_m=0.0015,
            rotation_tolerance_rad=0.02,
            max_nfev=100,
        )

        solved_xyz, solved_rotation = self.solver.pose(solved)
        self.assertLessEqual(float(np.max(np.abs(solved - self.joints))), 0.080001)
        self.assertLessEqual(float(np.linalg.norm(solved_xyz - target_xyz)), 0.0015)
        self.assertLessEqual(
            float(
                np.linalg.norm(
                    Rotation.from_matrix(target_rotation @ solved_rotation.T).as_rotvec()
                )
            ),
            0.02,
        )

    def test_rejects_pose_without_a_nearby_joint_solution(self):
        xyz, rotation = self.solver.pose(self.joints)
        target_rpy = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        with self.assertRaisesRegex(ExecutionBlocked, "could not reach a nearby solution"):
            self.solver.solve(
                self.joints,
                xyz + np.array([0.0, 0.05, 0.0]),
                target_rpy,
                max_joint_step_rad=0.01,
                position_tolerance_m=0.0005,
                rotation_tolerance_rad=0.005,
                max_nfev=50,
            )

    def test_delivery_execution_uses_joint_control_not_firmware_cartesian_ik(self):
        xyz, rotation = self.solver.pose(self.joints)
        qpos = np.append(self.joints, 0.035).astype(np.float32)
        state = build_delivery_state(xyz, rotation, float(qpos[6]))
        action = np.zeros(7, dtype=np.float64)
        action[6] = state[9]
        now = time.time()
        result = {
            "actions": action[None, :],
            "execution_control": {
                "mode": "execute",
                "task_id": "ik-policy-test",
                "session_id": "ik-session-test",
                "server_time": now,
                "expires_at": now + 30.0,
                "revision": 1,
            },
        }
        args = SimpleNamespace(
            arm_mode="single",
            arm_side="right",
            allow_execution=True,
            enable_timeout_s=0.1,
            max_action_age_s=2.0,
            max_translation_step_m=0.005,
            max_rotation_step_rad=0.05,
            max_gripper_step=0.1,
            gripper_range_tolerance=0.02,
            max_joint_step_rad=0.3,
            max_joint_gripper_step_m=0.02,
            workspace_x=(-0.04, 0.30),
            workspace_y=(0.02, 0.52),
            workspace_z=(0.12, 0.50),
            speed_pct=5,
            gripper_effort=1000,
        )
        piper = FakePiper()
        execution = ExecutionController(piper, args)
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        image_timestamps = {"cam_high": now, "cam_wrist": now}

        self.assertFalse(
            execution.process(result, state, qpos, protocol, image_timestamps, 0.01)
        )
        self.assertTrue(
            execution.process(result, state, qpos, protocol, image_timestamps, 0.01)
        )
        names = [call[0] for call in piper.calls]
        self.assertIn("JointCtrl", names)
        self.assertIn("ModeCtrl", names)
        self.assertNotIn("EndPoseCtrl", names)
        self.assertNotIn("MotionCtrl_2", names)


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

    def test_execution_controller_uses_configured_future_joint_target(self):
        args = SimpleNamespace(
            arm_mode="single",
            arm_side="right",
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
            hz=5.0,
            action_hz=None,
            action_chunk_steps=None,
        )
        piper = FakePiper()
        execution = ExecutionController(piper, args)
        protocol = validate_policy_metadata(dict(JOINT_METADATA, action_hz=20), "right")
        execution.configure_protocol(protocol)
        targets = np.stack(
            [self.qpos + np.array([step, 0, 0, 0, 0, 0, 0]) for step in (0.02, 0.04, 0.06, 0.08)]
        )
        now = time.time()
        result = {
            "actions": targets,
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
            execution.process(result, delivery_state, self.qpos, protocol, image_timestamps, 0.01)
        )
        self.assertTrue(
            execution.process(result, delivery_state, self.qpos, protocol, image_timestamps, 0.01)
        )
        self.assertEqual(execution.action_chunk_steps, 4)
        self.assertEqual(execution.last_action_chunk_steps, 4)
        np.testing.assert_allclose(execution.last_composed_action, targets[3])
        self.assertIsNotNone(execution.last_composed_action_at)
        joint_call = next(call for call in reversed(piper.calls) if call[0] == "JointCtrl")
        self.assertEqual(joint_call[1], int(np.rint(targets[3, 0] * RAD_FACTOR)))

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
