from __future__ import annotations

import json
from threading import Event
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from piper_action_conventions import (
    DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS,
    DELIVERY_MODEL_ACTION_SEMANTICS,
    DELIVERY_STEP_ACTION_SEMANTICS,
    JOINT_ACTION_SEMANTICS,
    LEGACY_GRIPPER_SEMANTICS,
    NEW_GRIPPER_SEMANTICS,
    matrix_to_rotation6d,
)
from piper_data_contract import (
    CONTRACT_VERSION,
    GRIPPER_MAX_M,
    LEGACY_GRIPPER_OPENING_METRES_SEMANTICS,
)
from robot_observation_bridge import (
    AsyncPolicyInference,
    DEFAULT_BLEND_STEPS,
    DEFAULT_INFERENCE_HZ,
    DEFAULT_MAX_GRIPPER_STEP,
    DEFAULT_MAX_ROTATION_STEP_RAD,
    DEFAULT_MAX_TRANSLATION_STEP_M,
    DEFAULT_MIN_ACTION_CHUNK_STEPS,
    DEFAULT_WORKSPACE_X_M,
    DEFAULT_WORKSPACE_Y_M,
    DEFAULT_WORKSPACE_Z_M,
    ExecutionBlocked,
    ExecutionController,
    IK_FEEDBACK_LIMIT_TOLERANCE_RAD,
    PiperContinuousIK,
    PiperFeedbackStaleError,
    GRIPPER_CLOSED_FRACTION,
    GRIPPER_OPENING_FRACTION,
    GRIPPER_OPENING_METRES,
    InferenceLaunch,
    InferenceWorkerResult,
    MonitoringRecorder,
    PeriodicSchedule,
    PolicyProtocol,
    RAD_FACTOR,
    _require_fresh_feedback,
    aggregate_action_chunk,
    build_checked_joint_target,
    build_checked_target,
    build_client_transport_timing,
    build_observation,
    decode_action_queue,
    policy_observation_state,
    resolve_action_chunk_steps,
    rotation_from_state,
    validate_policy_metadata,
)


class MonitoringRecorderTest(unittest.TestCase):
    def test_jsonl_session_preserves_finite_safe_events_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = MonitoringRecorder(directory, SimpleNamespace(arm_mode="single"))
            recorder.record(
                "control_tick",
                qpos_m=np.array([1.0, np.nan], dtype=np.float64),
                nested={"vector": np.array([2, 3], dtype=np.int64)},
            )
            events_path = recorder.events_path
            manifest_path = recorder.manifest_path
            recorder.close(reason="test")

            self.assertTrue(events_path.exists())
            self.assertTrue(manifest_path.exists())
            rows = [json.loads(line) for line in events_path.read_text().splitlines()]
            self.assertEqual([row["event_type"] for row in rows], [
                "session_started", "control_tick", "session_finished"
            ])
            self.assertIsNone(rows[1]["qpos_m"][1])
            self.assertEqual(rows[1]["nested"]["vector"], [2, 3])
            self.assertEqual(json.loads(manifest_path.read_text())["format"], "bimanual-vla-monitoring-v1")


def delivery_state(
    xyz=(0.20, 0.20, 0.30),
    rotation: np.ndarray | None = None,
    opening_fraction: float = 0.5,
) -> np.ndarray:
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    return np.concatenate(
        (
            np.asarray(xyz, dtype=np.float64),
            matrix_to_rotation6d(rotation),
            [opening_fraction],
        )
    ).astype(np.float32)


def execution_args(**overrides):
    values = dict(
        arm_mode="single",
        arm_side="right",
        allow_execution=True,
        enable_timeout_s=0.1,
        max_action_age_s=2.0,
        max_feedback_age_s=0.5,
        max_translation_step_m=DEFAULT_MAX_TRANSLATION_STEP_M,
        max_rotation_step_rad=DEFAULT_MAX_ROTATION_STEP_RAD,
        max_gripper_step=DEFAULT_MAX_GRIPPER_STEP,
        gripper_range_tolerance=0.02,
        max_joint_step_rad=0.3,
        max_joint_gripper_step=0.25,
        max_joint_gripper_step_m=None,
        workspace_x=DEFAULT_WORKSPACE_X_M,
        workspace_y=DEFAULT_WORKSPACE_Y_M,
        workspace_z=DEFAULT_WORKSPACE_Z_M,
        speed_pct=10,
        gripper_effort=1000,
        hz=DEFAULT_INFERENCE_HZ,
        control_hz=20.0,
        action_hz=None,
        action_chunk_steps=None,
        min_action_chunk_steps=DEFAULT_MIN_ACTION_CHUNK_STEPS,
        blend_steps=DEFAULT_BLEND_STEPS,
        latency_skip_compensation_steps=0,
        actuator_delay_s=0.0,
        gripper_lowpass_alpha=0.5,
        gripper_hysteresis=0.05,
        gripper_confirm_steps=2,
        ik_max_joint_step_rad=0.02,
        ik_search_joint_radius_rad=0.30,
        ik_joint_regularization_weight=1.0,
        ik_position_tolerance_m=0.0015,
        ik_rotation_tolerance_rad=0.02,
        ik_max_nfev=100,
        arm_settle_s=0.0,
        arm_hold_tolerance_rad=0.02,
        max_commands=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def execution_result(actions: np.ndarray) -> dict:
    now = time.time()
    return {
        "actions": np.asarray(actions, dtype=np.float64),
        "execution_control": {
            "mode": "execute",
            "task_id": "policy-test",
            "session_id": "session-test",
            "server_time": now,
            "expires_at": now + 30.0,
            "revision": 7,
        },
    }


def inference_launch(
    raw_state: np.ndarray,
    qpos: np.ndarray,
    *,
    generation: int = 1,
    latency_s: float = 0.0,
    captured_monotonic: float | None = None,
) -> tuple[InferenceLaunch, float]:
    arrived_at = time.time()
    captured_at = arrived_at - latency_s
    if captured_monotonic is None:
        captured_monotonic = time.monotonic() - latency_s
    return (
        InferenceLaunch(
            generation=generation,
            captured_at=captured_at,
            captured_monotonic=captured_monotonic,
            launched_at=captured_at,
            launched_monotonic=captured_monotonic,
            raw_delivery_state=np.asarray(raw_state, dtype=np.float64).copy(),
            qpos_m=np.asarray(qpos, dtype=np.float64).copy(),
            image_timestamps={"cam_high": captured_at, "cam_wrist": captured_at},
        ),
        arrived_at,
    )


DELIVERY_METADATA = {
    "transport": "openpi_websocket_v1",
    "schema": "delivery",
    "state_dim": 10,
    "action_dim": 7,
    "arm_side": "right",
    "action_semantics": DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS,
    "delivery_action_convention": "chunk_origin",
    "gripper_semantics": NEW_GRIPPER_SEMANTICS,
    "contract_version": CONTRACT_VERSION,
    "camera_keys": ["cam_high", "cam_wrist"],
    "action_hz": 20,
}

LEGACY_DELIVERY_METADATA = dict(
    DELIVERY_METADATA,
    action_semantics=DELIVERY_STEP_ACTION_SEMANTICS,
    delivery_action_convention="step",
    gripper_semantics=LEGACY_GRIPPER_SEMANTICS,
    contract_version=2,
)

JOINT_METADATA = {
    "transport": "openpi_websocket_v1",
    "schema": "joint",
    "state_dim": 7,
    "action_dim": 7,
    "arm_side": "right",
    "action_semantics": JOINT_ACTION_SEMANTICS,
    "gripper_semantics": NEW_GRIPPER_SEMANTICS,
    "contract_version": CONTRACT_VERSION,
    "camera_keys": ["cam_high", "cam_right_wrist"],
    "action_hz": 20,
}

LEGACY_JOINT_METADATA = dict(
    JOINT_METADATA,
    action_semantics="absolute_joint_position",
    gripper_semantics=LEGACY_GRIPPER_OPENING_METRES_SEMANTICS,
    contract_version=2,
)

BIMANUAL_JOINT_METADATA = dict(
    JOINT_METADATA,
    arm_mode="bimanual",
    state_dim=14,
    action_dim=14,
    arm_side="both",
    camera_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
)

BIMANUAL_DELIVERY_METADATA = dict(
    DELIVERY_METADATA,
    arm_mode="bimanual",
    state_dim=20,
    action_dim=14,
    arm_side="both",
    camera_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
)


class FakeExecution:
    def metadata(self):
        return {"allow_execution": False, "execution_state": "client_disabled"}


class FakePiper:
    def __init__(self, *, arm_status=0, err_code=0):
        self.calls = []
        self.arm_status = arm_status
        self.err_code = err_code

    def GetArmStatus(self):
        status = SimpleNamespace(
            ctrl_mode=1,
            arm_status=self.arm_status,
            mode_feed=1,
            motion_status=0,
            err_code=self.err_code,
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


class RecordingContinuousIK:
    def __init__(self):
        self.targets: list[tuple[np.ndarray, np.ndarray]] = []

    def solve(self, current_joints_rad, target_xyz_m, target_rpy_deg, **_kwargs):
        self.targets.append(
            (
                np.asarray(target_xyz_m, dtype=np.float64).copy(),
                np.asarray(target_rpy_deg, dtype=np.float64).copy(),
            )
        )
        return np.asarray(current_joints_rad, dtype=np.float64).copy()


class RejectFirstContinuousIK(RecordingContinuousIK):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def solve(self, current_joints_rad, target_xyz_m, target_rpy_deg, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise ExecutionBlocked("continuous IK synthetic target rejection")
        return super().solve(
            current_joints_rad, target_xyz_m, target_rpy_deg, **kwargs
        )


class LinearForwardKinematics:
    """Small deterministic FK used to test the bounded numerical IK itself."""

    def CalFK(self, joints):
        joints = np.asarray(joints, dtype=np.float64)
        pose = np.concatenate((joints[:3] * 1000.0, np.rad2deg(joints[3:6])))
        return [pose.copy() for _ in range(6)]


class FeedbackFreshnessTest(unittest.TestCase):
    def test_stale_piper_feedback_is_rejected(self):
        message = SimpleNamespace(time_stamp=time.time(), Hz=200.0)
        _require_fresh_feedback({"joint": message})
        message.time_stamp -= 1.0
        with self.assertRaises(PiperFeedbackStaleError):
            _require_fresh_feedback({"joint": message})


class ContinuousIKTest(unittest.TestCase):
    def setUp(self):
        self.solver = PiperContinuousIK(fk=LinearForwardKinematics())
        self.joints = np.array([0.20, 0.40, -0.40, 0.10, -0.10, 0.05])

    def test_solves_nearby_pose_without_switching_joint_branch(self):
        xyz, rotation = self.solver.pose(self.joints)
        target_xyz = xyz + np.array([0.002, 0.003, -0.001])
        target_rotation = Rotation.from_rotvec([0.01, -0.005, 0.004]).as_matrix() @ rotation
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
            float(np.linalg.norm(Rotation.from_matrix(target_rotation @ solved_rotation.T).as_rotvec())),
            0.02,
        )

    def test_rejects_pose_without_a_nearby_joint_solution(self):
        xyz, rotation = self.solver.pose(self.joints)
        target_rpy = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        with self.assertRaisesRegex(ExecutionBlocked, "could not reach a nearby solution"):
            self.solver.solve(
                self.joints,
                xyz + np.array([0.05, 0.0, 0.0]),
                target_rpy,
                max_joint_step_rad=0.01,
                position_tolerance_m=0.0005,
                rotation_tolerance_rad=0.005,
                max_nfev=50,
            )

    def test_rate_limits_a_reachable_future_pose_and_still_makes_progress(self):
        xyz, rotation = self.solver.pose(self.joints)
        target_xyz = xyz + np.array([0.05, 0.0, 0.0])
        target_rpy = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        solved = self.solver.solve(
            self.joints,
            target_xyz,
            target_rpy,
            max_joint_step_rad=0.01,
            search_joint_radius_rad=0.08,
            joint_regularization_weight=0.0,
            position_tolerance_m=0.0015,
            rotation_tolerance_rad=0.02,
            max_nfev=100,
        )
        solved_xyz, _ = self.solver.pose(solved)
        self.assertAlmostEqual(float(np.max(np.abs(solved - self.joints))), 0.01)
        self.assertLess(
            float(np.linalg.norm(solved_xyz - target_xyz)),
            float(np.linalg.norm(xyz - target_xyz)),
        )

    def test_accepts_small_zero_offset_and_does_not_move_farther_outward(self):
        joints = self.joints.copy()
        joints[1] = -0.0345
        joints[2] = 0.0491
        xyz, rotation = self.solver.pose(joints)
        target_rpy = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        solved = self.solver.solve(
            joints,
            xyz,
            target_rpy,
            max_joint_step_rad=0.08,
            position_tolerance_m=0.0015,
            rotation_tolerance_rad=0.02,
            max_nfev=100,
        )
        np.testing.assert_allclose(solved, joints, atol=2e-6)
        self.assertGreaterEqual(solved[1], joints[1] - 1e-8)
        self.assertLessEqual(solved[2], joints[2] + 1e-8)

    def test_rejects_feedback_beyond_zero_offset_tolerance(self):
        joints = self.joints.copy()
        joints[2] = IK_FEEDBACK_LIMIT_TOLERANCE_RAD + 0.01
        xyz, rotation = self.solver.pose(joints)
        target_rpy = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        with self.assertRaisesRegex(ExecutionBlocked, "too far outside IK limits"):
            self.solver.solve(
                joints,
                xyz,
                target_rpy,
                max_joint_step_rad=0.08,
                position_tolerance_m=0.0015,
                rotation_tolerance_rad=0.02,
                max_nfev=100,
            )


class EnableHoldTest(unittest.TestCase):
    def test_enable_holds_measured_zero_offset_without_clipping(self):
        piper = FakePiper()
        execution = ExecutionController(piper, execution_args())
        measured = np.array([1.0, -0.0345, 0.0491, 0.1, 0.2, -0.3, 0.035])
        execution._enable_robot("right", piper, measured)
        joint_call = next(call for call in piper.calls if call[0] == "JointCtrl")
        np.testing.assert_array_equal(
            np.asarray(joint_call[1:]),
            np.rint(measured[:6] * RAD_FACTOR).astype(np.int64),
        )
        np.testing.assert_allclose(execution.arm_hold_targets["right"], measured[:6])


class MetadataCompatibilityTest(unittest.TestCase):
    def test_v3_delivery_accepts_side_specific_wrist_key(self):
        metadata = dict(
            DELIVERY_METADATA,
            camera_keys=["cam_high", "cam_right_wrist"],
        )
        protocol = validate_policy_metadata(metadata, "right")
        self.assertEqual(protocol.camera_keys, ("cam_high", "cam_right_wrist"))

        wrong_side = dict(metadata, camera_keys=["cam_high", "cam_left_wrist"])
        with self.assertRaisesRegex(RuntimeError, "camera_keys"):
            validate_policy_metadata(wrong_side, "right")

    def test_explicit_delivery_metadata_selects_old_and_new_conventions(self):
        new = validate_policy_metadata(DELIVERY_METADATA, "right")
        old = validate_policy_metadata(LEGACY_DELIVERY_METADATA, "right")
        self.assertEqual(new.action_semantics, DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS)
        self.assertEqual(new.gripper_semantics, GRIPPER_OPENING_FRACTION)
        self.assertEqual(old.action_semantics, DELIVERY_STEP_ACTION_SEMANTICS)
        self.assertEqual(old.gripper_semantics, GRIPPER_CLOSED_FRACTION)

    def test_delivery_metadata_rejects_mixed_semantics(self):
        bad = dict(
            DELIVERY_METADATA,
            action_semantics=DELIVERY_MODEL_ACTION_SEMANTICS,
            gripper_semantics=LEGACY_GRIPPER_SEMANTICS,
        )
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            validate_policy_metadata(bad, "right")

    def test_legacy_checkpoint_may_advertise_preconverted_chunk_with_closed_fraction(self):
        metadata = dict(
            LEGACY_DELIVERY_METADATA,
            action_semantics=DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS,
            delivery_action_convention="chunk_origin",
        )
        protocol = validate_policy_metadata(metadata, "right")
        self.assertEqual(protocol.action_semantics, DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS)
        self.assertEqual(protocol.gripper_semantics, GRIPPER_CLOSED_FRACTION)

    def test_joint_metadata_distinguishes_fraction_and_legacy_metres(self):
        new = validate_policy_metadata(JOINT_METADATA, "right")
        old = validate_policy_metadata(LEGACY_JOINT_METADATA, "right")
        self.assertEqual(new.gripper_semantics, GRIPPER_OPENING_FRACTION)
        self.assertEqual(old.gripper_semantics, GRIPPER_OPENING_METRES)

    def test_legacy_joint_can_have_metre_state_but_fraction_wire(self):
        metadata = dict(
            LEGACY_JOINT_METADATA,
            gripper_semantics=NEW_GRIPPER_SEMANTICS,
            wire_gripper_semantics=NEW_GRIPPER_SEMANTICS,
            state_gripper_semantics=LEGACY_GRIPPER_OPENING_METRES_SEMANTICS,
        )
        protocol = validate_policy_metadata(metadata, "right")
        self.assertEqual(protocol.gripper_semantics, GRIPPER_OPENING_FRACTION)
        self.assertEqual(protocol.state_gripper_semantics, GRIPPER_OPENING_METRES)
        qpos = np.array([0, 1, -1, 0, 0, 0, 0.035], dtype=np.float32)
        state = policy_observation_state(delivery_state(), qpos, protocol)
        self.assertAlmostEqual(state[6], 0.035)

    def test_legacy_joint_missing_metadata_uses_names_or_fails_closed(self):
        inferred = dict(LEGACY_JOINT_METADATA)
        inferred.pop("gripper_semantics")
        inferred.pop("contract_version")
        inferred["action_names"] = [f"joint_{i}_rad" for i in range(1, 7)] + [
            "gripper_opening_m"
        ]
        self.assertEqual(
            validate_policy_metadata(inferred, "right").gripper_semantics,
            GRIPPER_OPENING_METRES,
        )

        ambiguous = dict(inferred)
        ambiguous.pop("action_names")
        with self.assertRaisesRegex(RuntimeError, "refusing to guess"):
            validate_policy_metadata(ambiguous, "right")

    def test_wire_action_fields_override_raw_delivery_dimension(self):
        metadata = dict(
            DELIVERY_METADATA,
            action_dim=10,
            model_action_dim=7,
            model_action_semantics=DELIVERY_METADATA["action_semantics"],
        )
        self.assertEqual(validate_policy_metadata(metadata, "right").action_dim, 7)


class ActionQueueDecodeTest(unittest.TestCase):
    def test_resolves_five_20hz_actions_per_4hz_compatibility_interval(self):
        self.assertEqual(resolve_action_chunk_steps(action_hz=20, command_hz=4), 5)
        self.assertEqual(resolve_action_chunk_steps(action_hz=20, command_hz=10), 2)
        self.assertEqual(resolve_action_chunk_steps(action_hz=20, command_hz=30), 1)
        self.assertEqual(
            resolve_action_chunk_steps(action_hz=20, command_hz=4, override=3), 3
        )

    def test_v3_chunk_rows_decode_independently_from_one_anchor(self):
        anchor_rotation = Rotation.from_euler("z", 0.4).as_matrix()
        raw_state = delivery_state(rotation=anchor_rotation, opening_fraction=0.4)
        qpos = np.array([0, 1, -1, 0, 0, 0, 0.4 * GRIPPER_MAX_M])
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        actions = np.zeros((4, 7), dtype=np.float64)
        actions[:, :3] = [
            [0.002, 0.0, 0.0],
            [0.004, 0.001, 0.0],
            [0.006, 0.001, 0.002],
            [0.008, -0.001, 0.002],
        ]
        actions[:, 3:6] = [
            [0.02, 0.00, 0.00],
            [0.00, 0.03, 0.00],
            [0.00, 0.00, -0.04],
            [0.01, 0.02, -0.03],
        ]
        actions[:, 6] = [0.4, 0.5, 0.6, 0.7]

        anchor, decoded = decode_action_queue(
            actions, protocol, raw_state, qpos, steps=4
        )

        np.testing.assert_allclose(anchor, raw_state)
        self.assertEqual([item.queue_index for item in decoded], [0, 1, 2, 3])
        for index, item in enumerate(decoded):
            np.testing.assert_allclose(
                item.absolute_target[:3], raw_state[:3] + actions[index, :3], atol=1e-7
            )
            expected_rotation = (
                Rotation.from_rotvec(actions[index, 3:6]).as_matrix() @ anchor_rotation
            )
            np.testing.assert_allclose(
                rotation_from_state(item.absolute_target), expected_rotation, atol=1e-6
            )
            self.assertAlmostEqual(item.absolute_target[9], actions[index, 6])
        self.assertFalse(
            np.allclose(
                decoded[1].absolute_target[:3],
                decoded[0].absolute_target[:3] + actions[1, :3],
            )
        )

    def test_every_decoded_row_has_capture_relative_monotonic_time(self):
        protocol = validate_policy_metadata(JOINT_METADATA, "right")
        qpos = np.array([0, 1, -1, 0, 0, 0, 0.035], dtype=np.float64)
        actions = np.stack(
            [np.array([0.01 * index, 1, -1, 0, 0, 0, 0.5]) for index in range(4)]
        )
        _, decoded = decode_action_queue(
            actions,
            protocol,
            delivery_state(),
            qpos,
            observation_capture_monotonic=50.0,
            action_hz=20.0,
        )
        np.testing.assert_allclose(
            [target.target_monotonic for target in decoded],
            [50.05, 50.10, 50.15, 50.20],
        )

    def test_legacy_rows_accumulate_and_closed_fraction_is_preserved_on_wire(self):
        raw_state = delivery_state(opening_fraction=0.8)
        qpos = np.array([0, 1, -1, 0, 0, 0, 0.8 * GRIPPER_MAX_M])
        protocol = validate_policy_metadata(LEGACY_DELIVERY_METADATA, "right")
        actions = np.zeros((3, 7), dtype=np.float64)
        actions[:, :3] = [[0.002, 0, 0], [0.003, 0.001, 0], [0, 0, 0.004]]
        actions[:, 3:6] = [[0.02, 0, 0], [0, 0.03, 0], [0, 0, -0.04]]
        actions[:, 6] = [0.2, 0.5, 0.75]

        anchor, decoded = decode_action_queue(
            actions, protocol, raw_state, qpos, steps=3
        )

        self.assertAlmostEqual(anchor[9], 0.2)  # opening 0.8 -> legacy closed 0.2
        np.testing.assert_allclose(
            decoded[-1].absolute_target[:3],
            raw_state[:3] + actions[:, :3].sum(axis=0),
            atol=1e-7,
        )
        expected_rotation = np.eye(3)
        for rotvec in actions[:, 3:6]:
            expected_rotation = Rotation.from_rotvec(rotvec).as_matrix() @ expected_rotation
        np.testing.assert_allclose(
            rotation_from_state(decoded[-1].absolute_target), expected_rotation, atol=1e-6
        )
        self.assertAlmostEqual(decoded[-1].absolute_target[9], 0.75)

    def test_aggregate_preview_keeps_current_compatibility(self):
        legacy = validate_policy_metadata(LEGACY_DELIVERY_METADATA, "right")
        actions = np.zeros((3, 7))
        actions[:, 0] = [0.001, 0.002, 0.003]
        actions[:, 6] = [0.1, 0.2, 0.3]
        preview, used = aggregate_action_chunk(actions, legacy, 3)
        self.assertEqual(used, 3)
        self.assertAlmostEqual(preview[0], 0.006)
        self.assertAlmostEqual(preview[6], 0.3)


class ObservationConventionTest(unittest.TestCase):
    def test_policy_state_selects_delivery_closed_or_opening(self):
        raw = delivery_state(opening_fraction=0.8)
        qpos = np.array([0, 1, -1, 0, 0, 0, 0.8 * GRIPPER_MAX_M])
        new = policy_observation_state(
            raw, qpos, validate_policy_metadata(DELIVERY_METADATA, "right")
        )
        old = policy_observation_state(
            raw, qpos, validate_policy_metadata(LEGACY_DELIVERY_METADATA, "right")
        )
        self.assertAlmostEqual(new[9], 0.8)
        self.assertAlmostEqual(old[9], 0.2)

    def test_policy_state_selects_joint_metres_or_fraction(self):
        raw = delivery_state()
        qpos = np.array([0, 1, -1, 0, 0, 0, 0.035], dtype=np.float32)
        new = policy_observation_state(
            raw, qpos, validate_policy_metadata(JOINT_METADATA, "right")
        )
        old = policy_observation_state(
            raw, qpos, validate_policy_metadata(LEGACY_JOINT_METADATA, "right")
        )
        self.assertAlmostEqual(new[6], 0.5)
        self.assertAlmostEqual(old[6], 0.035)

    def test_build_observation_reports_protocol_and_queue_telemetry(self):
        args = SimpleNamespace(
            arm_mode="single",
            arm_side="right",
            can="can0",
            cam_high_device="/dev/video8",
            cam_wrist_device="/dev/video16",
        )
        now = time.time()
        observation = build_observation(
            delivery_state=delivery_state(opening_fraction=0.75),
            qpos=np.array([0, 1, -1, 0, 0, 0, 0.0525]),
            protocol=validate_policy_metadata(DELIVERY_METADATA, "right"),
            images={
                "cam_high": np.zeros((4, 4, 3), dtype=np.uint8),
                "cam_wrist": np.zeros((4, 4, 3), dtype=np.uint8),
            },
            image_timestamps={"cam_high": now, "cam_wrist": now},
            instruction="pick",
            source_name="robot",
            args=args,
            execution=FakeExecution(),
        )
        self.assertAlmostEqual(observation["state"][9], 0.75)
        self.assertEqual(
            observation["client_metadata"]["policy_gripper_semantics"],
            NEW_GRIPPER_SEMANTICS,
        )


class TargetSafetyTest(unittest.TestCase):
    def setUp(self):
        self.qpos = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.035])
        self.delivery_kwargs = dict(
            max_translation_step_m=0.015,
            max_rotation_step_rad=0.15,
            max_gripper_step=0.25,
            gripper_range_tolerance=0.02,
            workspace_x=(0.05, 0.60),
            workspace_y=(-0.45, 0.45),
            workspace_z=(0.02, 0.60),
            gripper_semantics=GRIPPER_OPENING_FRACTION,
        )

    def test_delivery_workspace_translation_rotation_and_gripper_gates(self):
        state = delivery_state(opening_fraction=0.5)
        action = np.array([0.01, 0, 0, 0.05, 0, 0, 0.6])
        xyz, _, gripper_m = build_checked_target(state, action, **self.delivery_kwargs)
        np.testing.assert_allclose(xyz, [0.21, 0.20, 0.30])
        self.assertAlmostEqual(gripper_m, 0.6 * GRIPPER_MAX_M)

        for index, value, message in ((0, 0.016, "translation step"), (3, 0.16, "rotation step")):
            bad = action.copy()
            bad[index] = value
            with self.assertRaisesRegex(ExecutionBlocked, message):
                build_checked_target(state, bad, **self.delivery_kwargs)
        bad = action.copy()
        bad[6] = 1.5
        with self.assertRaisesRegex(ExecutionBlocked, "gripper target"):
            build_checked_target(state, bad, **self.delivery_kwargs)

    def test_8_3_64eps_default_envelope_accepts_observed_maxima(self):
        kwargs = dict(
            max_translation_step_m=DEFAULT_MAX_TRANSLATION_STEP_M,
            max_rotation_step_rad=DEFAULT_MAX_ROTATION_STEP_RAD,
            max_gripper_step=DEFAULT_MAX_GRIPPER_STEP,
            gripper_range_tolerance=0.02,
            workspace_x=DEFAULT_WORKSPACE_X_M,
            workspace_y=DEFAULT_WORKSPACE_Y_M,
            workspace_z=DEFAULT_WORKSPACE_Z_M,
            gripper_semantics=GRIPPER_OPENING_FRACTION,
        )
        state = delivery_state(xyz=(0.20, 0.20, 0.30), opening_fraction=0.5)
        observed_max_step = np.array([0.04830, 0, 0, 0, 0, 0.15766, 0.761])
        xyz, _, gripper_m = build_checked_target(state, observed_max_step, **kwargs)
        np.testing.assert_allclose(xyz, [0.24830, 0.20, 0.30], atol=1e-8)
        self.assertAlmostEqual(gripper_m, 0.761 * GRIPPER_MAX_M)

        workspace_state = delivery_state(xyz=(0.25, 0.45, 0.48))
        workspace_max = np.array(
            [0.02987, 0.02802, 0.02322, 0, 0, 0, 0.5], dtype=np.float64
        )
        xyz, _, _ = build_checked_target(workspace_state, workspace_max, **kwargs)
        np.testing.assert_allclose(xyz, [0.27987, 0.47802, 0.50322], atol=1e-8)

    def test_8_3_64eps_default_envelope_still_blocks_clear_violations(self):
        kwargs = dict(
            max_translation_step_m=DEFAULT_MAX_TRANSLATION_STEP_M,
            max_rotation_step_rad=DEFAULT_MAX_ROTATION_STEP_RAD,
            max_gripper_step=DEFAULT_MAX_GRIPPER_STEP,
            gripper_range_tolerance=0.02,
            workspace_x=DEFAULT_WORKSPACE_X_M,
            workspace_y=DEFAULT_WORKSPACE_Y_M,
            workspace_z=DEFAULT_WORKSPACE_Z_M,
            gripper_semantics=GRIPPER_OPENING_FRACTION,
        )
        state = delivery_state(opening_fraction=0.5)
        for action, message in (
            (np.array([0.051, 0, 0, 0, 0, 0, 0.5]), "translation step"),
            (np.array([0, 0, 0, 0, 0, 0.181, 0.5]), "rotation step"),
            (np.array([0, 0, 0, 0, 0, 0, 0.801]), "gripper step"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ExecutionBlocked, message):
                    build_checked_target(state, action, **kwargs)

        near_x_max = delivery_state(xyz=(0.28, 0.20, 0.30))
        with self.assertRaisesRegex(ExecutionBlocked, "outside workspace"):
            build_checked_target(
                near_x_max,
                np.array([0.021, 0, 0, 0, 0, 0, 0.5]),
                **kwargs,
            )

    def test_joint_fraction_and_legacy_metres_never_mix(self):
        new_target = np.array([0.1, 1, -1, 0, 0, 0, 0.6])
        joints, gripper_m = build_checked_joint_target(
            self.qpos,
            new_target,
            max_joint_step_rad=0.3,
            max_gripper_step=0.25,
            gripper_semantics=GRIPPER_OPENING_FRACTION,
        )
        np.testing.assert_allclose(joints, new_target[:6])
        self.assertAlmostEqual(gripper_m, 0.6 * GRIPPER_MAX_M)

        legacy_target = new_target.copy()
        legacy_target[6] = 0.042
        _, legacy_gripper_m = build_checked_joint_target(
            self.qpos,
            legacy_target,
            max_joint_step_rad=0.3,
            max_gripper_step=0.25,
            gripper_semantics=GRIPPER_OPENING_METRES,
        )
        self.assertAlmostEqual(legacy_gripper_m, 0.042)


class ClientTransportTimingTest(unittest.TestCase):
    def test_build_client_transport_timing_uses_fixed_boundaries_and_generation(self):
        timing = build_client_transport_timing(
            request_sent_at=100.000,
            request_sent_monotonic=10.000,
            response_received_at=100.250,
            response_received_monotonic=10.200,
            server_timing={
                "server_request_received_at": 100.050,
                "server_model_completed_at": 100.190,
                "server_response_ready_at": 100.210,
                "model_inference_ms": 180.0,
            },
            camera_capture_ms=12.5,
            inference_generation=42,
        )

        self.assertAlmostEqual(timing["camera_capture_ms"], 12.5)
        self.assertAlmostEqual(timing["observation_upload_ms"], 50.0)
        self.assertAlmostEqual(timing["client_observation_upload_ms"], 50.0)
        self.assertAlmostEqual(timing["result_download_ms"], 40.0)
        self.assertAlmostEqual(timing["client_result_download_ms"], 40.0)
        self.assertAlmostEqual(timing["network_transport_total_ms"], 90.0)
        self.assertAlmostEqual(timing["client_network_transport_total_ms"], 90.0)
        self.assertAlmostEqual(timing["round_trip_ms"], 200.0)
        self.assertAlmostEqual(timing["non_model_rtt_ms"], 20.0)
        self.assertEqual(timing["inference_generation"], 42)
        self.assertEqual(timing["timing_source"], "client_wall_clock_echo")
        self.assertEqual(timing["one_way_timing_clock"], "wall_clock")
        self.assertTrue(timing["one_way_timing_requires_clock_sync"])

    def test_build_client_transport_timing_fails_closed_for_missing_or_reversed_wall_timestamps(self):
        timing = build_client_transport_timing(
            request_sent_at=100.100,
            request_sent_monotonic=10.200,
            response_received_at=100.200,
            response_received_monotonic=10.100,
            server_timing={
                # Server clock is earlier than the client send timestamp.
                "server_request_received_at": 100.000,
                # Server response-ready time is after the client receive time.
                "server_response_ready_at": 100.300,
                "model_inference_ms": -1.0,
            },
            camera_capture_ms=float("nan"),
            inference_generation=7,
        )

        self.assertIsNone(timing["camera_capture_ms"])
        self.assertIsNone(timing["observation_upload_ms"])
        self.assertIsNone(timing["client_observation_upload_ms"])
        self.assertIsNone(timing["result_download_ms"])
        self.assertIsNone(timing["client_result_download_ms"])
        self.assertIsNone(timing["network_transport_total_ms"])
        self.assertIsNone(timing["client_network_transport_total_ms"])
        # A reversed local monotonic interval is clamped, never reported negative.
        self.assertEqual(timing["round_trip_ms"], 0.0)
        self.assertIsNone(timing["non_model_rtt_ms"])
        self.assertEqual(timing["inference_generation"], 7)
        self.assertEqual(timing["request_sent_at"], 100.1)
        self.assertEqual(timing["response_received_at"], 100.2)

    def test_build_client_transport_timing_accepts_missing_server_timing_without_throwing(self):
        timing = build_client_transport_timing(
            request_sent_at=200.0,
            request_sent_monotonic=20.0,
            response_received_at=200.3,
            response_received_monotonic=20.3,
            server_timing=None,
            camera_capture_ms=5.0,
            inference_generation=8,
        )

        self.assertAlmostEqual(timing["round_trip_ms"], 300.0)
        self.assertIsNone(timing["observation_upload_ms"])
        self.assertIsNone(timing["result_download_ms"])
        self.assertIsNone(timing["model_inference_ms"])
        self.assertIsNone(timing["network_transport_total_ms"])
        self.assertEqual(timing["inference_generation"], 8)


class AsyncInferencePipelineTest(unittest.TestCase):
    def setUp(self):
        self.raw_state = delivery_state(opening_fraction=0.5)
        self.qpos = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.035])
        self.joint_protocol = validate_policy_metadata(JOINT_METADATA, "right")
        self.delivery_protocol = validate_policy_metadata(DELIVERY_METADATA, "right")

    @staticmethod
    def joint_chunk(count: int, first_joint: float | np.ndarray = 0.1) -> np.ndarray:
        values = np.asarray(first_joint, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(count, float(values), dtype=np.float64)
        if values.shape != (count,):
            raise ValueError(f"expected {count} first-joint values, got {values.shape}")
        return np.stack(
            [np.array([value, 1, -1, 0, 0, 0, 0.5]) for value in values]
        )

    def configured_execution(self, protocol=None, **overrides):
        protocol = self.joint_protocol if protocol is None else protocol
        piper = FakePiper()
        execution = ExecutionController(piper, execution_args(**overrides))
        execution.configure_protocol(protocol)
        execution.ik_solver = RecordingContinuousIK()
        execution.robot_enabled = {"right"}
        return execution, piper

    def test_slow_async_inference_does_not_stop_old_20hz_queue(self):
        class BlockingPolicy:
            def __init__(self):
                self.started = Event()
                self.release = Event()

            def infer(self, _observation):
                self.started.set()
                self.release.wait(timeout=2.0)
                return execution_result(AsyncInferencePipelineTest.joint_chunk(16, 0.2))

        execution, piper = self.configured_execution()
        old_targets = self.joint_chunk(8, np.linspace(0.01, 0.08, 8))
        now = time.time()
        execution.queue_result(
            execution_result(old_targets),
            self.raw_state,
            self.qpos,
            self.joint_protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        launch, _ = inference_launch(self.raw_state, self.qpos, generation=2)
        policy = BlockingPolicy()
        worker = AsyncPolicyInference()
        try:
            self.assertTrue(worker.launch(policy, {}, launch))
            self.assertTrue(policy.started.wait(timeout=1.0))
            self.assertTrue(worker.in_flight)
            self.assertIsNone(worker.poll())

            qpos = self.qpos.copy()
            for target in old_targets[:3]:
                self.assertTrue(
                    execution.execute_next(
                        self.raw_state,
                        qpos,
                        self.joint_protocol,
                        feedback_captured_at=time.time(),
                    )
                )
                qpos[:6] = target[:6]
                qpos[6] = target[6] * GRIPPER_MAX_M
            self.assertEqual(execution.pending_action_count, 5)
            self.assertEqual(
                len([call for call in piper.calls if call[0] == "JointCtrl"]), 3
            )
            self.assertTrue(worker.in_flight)
        finally:
            policy.release.set()
            deadline = time.monotonic() + 1.0
            while worker.in_flight and time.monotonic() < deadline:
                worker.poll()
                time.sleep(0.001)
            worker.shutdown()

    def test_worker_callable_returns_camera_timestamps_without_control_wait(self):
        worker = AsyncPolicyInference()
        launch, _ = inference_launch(self.raw_state, self.qpos, generation=22)
        release = Event()

        def capture_and_infer():
            release.wait(timeout=1.0)
            return InferenceWorkerResult(
                execution_result(self.joint_chunk(16, 0.2)),
                {"cam_high": 123.0, "cam_wrist": 124.0},
            )

        try:
            started_at = time.monotonic()
            self.assertTrue(worker.launch_callable(capture_and_infer, launch))
            self.assertLess(time.monotonic() - started_at, 0.05)
            self.assertTrue(worker.in_flight)
            self.assertIsNone(worker.poll())
            release.set()
            completion = None
            deadline = time.monotonic() + 1.0
            while completion is None and time.monotonic() < deadline:
                completion = worker.poll()
                time.sleep(0.001)
            self.assertIsNotNone(completion)
            self.assertIsNone(completion.error)
            self.assertEqual(
                completion.launch.image_timestamps,
                {"cam_high": 123.0, "cam_wrist": 124.0},
            )
        finally:
            release.set()
            worker.shutdown()

    def test_180_200_240ms_latency_selects_monotonic_future_target(self):
        actions = self.joint_chunk(20, np.linspace(0.01, 0.20, 20))
        cases = ((0.18, 0.0, 3), (0.20, 0.0, 3), (0.24, 0.0, 4), (0.20, 0.02, 4))
        for latency_s, actuator_delay_s, expected_source_index in cases:
            with self.subTest(
                latency_s=latency_s, actuator_delay_s=actuator_delay_s
            ):
                execution, _ = self.configured_execution(
                    control_hz=20.0, actuator_delay_s=actuator_delay_s
                )
                launch, _ = inference_launch(
                    self.raw_state,
                    self.qpos,
                    generation=3,
                    captured_monotonic=100.0,
                )
                self.assertTrue(
                    execution.accept_inference_result(
                        execution_result(actions),
                        launch,
                        self.joint_protocol,
                        arrived_at=launch.captured_at + latency_s,
                        arrived_monotonic=100.0 + latency_s,
                    )
                )
                self.assertEqual(
                    execution.inference_skip_steps, expected_source_index
                )
                selected = execution.pending_actions[0]
                self.assertEqual(selected.source_index, expected_source_index)
                self.assertAlmostEqual(
                    selected.target_monotonic,
                    100.0 + (expected_source_index + 1) / 20.0,
                )
                self.assertAlmostEqual(
                    selected.absolute_target[0], actions[expected_source_index, 0]
                )

    def test_control_tick_reselects_closest_future_target_as_time_advances(self):
        execution, piper = self.configured_execution(control_hz=20.0)
        actions = self.joint_chunk(20, np.linspace(0.01, 0.20, 20))
        launch, _ = inference_launch(
            self.raw_state,
            self.qpos,
            generation=31,
            captured_monotonic=100.0,
        )
        self.assertTrue(
            execution.accept_inference_result(
                execution_result(actions),
                launch,
                self.joint_protocol,
                arrived_at=launch.captured_at + 0.01,
                arrived_monotonic=100.01,
            )
        )
        self.assertEqual(execution.pending_actions[0].source_index, 0)

        with patch("robot_observation_bridge.time.monotonic", return_value=100.18):
            self.assertTrue(
                execution.execute_next(
                    self.raw_state,
                    self.qpos,
                    self.joint_protocol,
                    feedback_captured_at=time.time(),
                )
            )
        self.assertEqual(execution.last_queued_action_index, 3)
        self.assertEqual(execution.pending_actions[0].source_index, 4)
        self.assertEqual(execution.dropped_action_count, 3)
        joint_call = [call for call in piper.calls if call[0] == "JointCtrl"][-1]
        self.assertEqual(joint_call[1], int(np.rint(actions[3, 0] * RAD_FACTOR)))

    def test_result_shorter_than_16_rows_is_rejected_without_losing_old_queue(self):
        execution, _ = self.configured_execution()
        old = self.joint_chunk(6, np.linspace(0.01, 0.06, 6))
        now = time.time()
        execution.queue_result(
            execution_result(old),
            self.raw_state,
            self.qpos,
            self.joint_protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        old_targets = [item.absolute_target.copy() for item in execution.pending_actions]
        launch, arrived_at = inference_launch(self.raw_state, self.qpos, generation=4)
        self.assertFalse(
            execution.accept_inference_result(
                execution_result(self.joint_chunk(15, 0.2)),
                launch,
                self.joint_protocol,
                arrived_at=arrived_at,
                arrived_monotonic=time.monotonic(),
            )
        )
        self.assertEqual(execution.pending_action_count, len(old_targets))
        for queued, expected in zip(execution.pending_actions, old_targets):
            np.testing.assert_allclose(queued.absolute_target, expected)
        self.assertIn("requires at least 16", execution.rejected_result["reason"])

    def test_joint_trajectory_switch_blends_four_rows_then_uses_new_rows(self):
        execution, _ = self.configured_execution(blend_steps=4)
        old_first = np.array([0.00, 0.04, 0.08, 0.12, 0.16, 0.20])
        old = self.joint_chunk(len(old_first), old_first)
        now = time.time()
        execution.queue_result(
            execution_result(old),
            self.raw_state,
            self.qpos,
            self.joint_protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        new = self.joint_chunk(20, 0.20)
        launch, arrived_at = inference_launch(self.raw_state, self.qpos, generation=5)
        self.assertTrue(
            execution.accept_inference_result(
                execution_result(new),
                launch,
                self.joint_protocol,
                arrived_at=arrived_at,
                arrived_monotonic=time.monotonic(),
            )
        )
        expected = [
            old_first[index] + ((index + 1) / 4.0) * (0.20 - old_first[index])
            for index in range(4)
        ]
        np.testing.assert_allclose(
            [item.absolute_target[0] for item in execution.pending_actions[:4]],
            expected,
        )
        self.assertTrue(all(item.blended for item in execution.pending_actions[:4]))
        self.assertEqual([item.blend_step for item in execution.pending_actions[:4]], [1, 2, 3, 4])
        self.assertFalse(execution.pending_actions[4].blended)
        self.assertAlmostEqual(execution.pending_actions[4].absolute_target[0], 0.20)

    def test_default_three_step_blend_keeps_old_gripper_then_switches(self):
        self.assertEqual(DEFAULT_BLEND_STEPS, 3)
        execution, _ = self.configured_execution(blend_steps=DEFAULT_BLEND_STEPS)
        old = self.joint_chunk(6, 0.0)
        old[:, 6] = 0.2
        now = time.time()
        execution.queue_result(
            execution_result(old),
            self.raw_state,
            self.qpos,
            self.joint_protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        new = self.joint_chunk(20, 0.2)
        new[:, 6] = 0.8
        launch, arrived_at = inference_launch(self.raw_state, self.qpos, generation=51)
        self.assertTrue(
            execution.accept_inference_result(
                execution_result(new),
                launch,
                self.joint_protocol,
                arrived_at=arrived_at,
                arrived_monotonic=launch.captured_monotonic,
            )
        )
        np.testing.assert_allclose(
            [target.absolute_target[6] for target in execution.pending_actions[:3]],
            [0.2, 0.2, 0.2],
        )
        self.assertAlmostEqual(execution.pending_actions[3].absolute_target[6], 0.8)

    def test_bimanual_blend_uses_the_same_alpha_for_both_arms(self):
        protocol = validate_policy_metadata(BIMANUAL_JOINT_METADATA, "both", "bimanual")
        pipers = {"left": FakePiper(), "right": FakePiper()}
        execution = ExecutionController(
            pipers,
            execution_args(arm_mode="bimanual", arm_side="both", blend_steps=3),
        )
        execution.configure_protocol(protocol)
        qpos = np.concatenate([self.qpos, self.qpos])

        old_arm = np.array([0.0, 1, -1, 0, 0, 0, 0.5])
        old = np.tile(np.concatenate([old_arm, old_arm]), (6, 1))
        now = time.time()
        execution.queue_result(
            execution_result(old),
            np.concatenate([self.raw_state, self.raw_state]),
            qpos,
            protocol,
            {"cam_high": now, "cam_left_wrist": now, "cam_right_wrist": now},
            0.01,
        )
        new_left = old_arm.copy()
        new_right = old_arm.copy()
        new_left[0] = 0.3
        new_right[0] = -0.3
        new = np.tile(np.concatenate([new_left, new_right]), (20, 1))
        launch, arrived_at = inference_launch(
            np.concatenate([self.raw_state, self.raw_state]),
            qpos,
            generation=52,
        )
        self.assertTrue(
            execution.accept_inference_result(
                execution_result(new),
                launch,
                protocol,
                arrived_at=arrived_at,
                arrived_monotonic=launch.captured_monotonic,
            )
        )
        np.testing.assert_allclose(
            [target.absolute_target[0] for target in execution.pending_actions[:3]],
            [0.1, 0.2, 0.3],
        )
        np.testing.assert_allclose(
            [target.absolute_target[7] for target in execution.pending_actions[:3]],
            [-0.1, -0.2, -0.3],
        )

    def test_delivery_rotation_blend_uses_so3_interpolation(self):
        execution, _ = self.configured_execution(self.delivery_protocol, blend_steps=4)
        old = np.zeros((8, 7), dtype=np.float64)
        old[:, 6] = 0.5
        now = time.time()
        execution.queue_result(
            execution_result(old),
            self.raw_state,
            self.qpos,
            self.delivery_protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        new = np.zeros((20, 7), dtype=np.float64)
        new[:, 5] = np.pi / 2
        new[:, 6] = 0.5
        launch, arrived_at = inference_launch(self.raw_state, self.qpos, generation=6)
        self.assertTrue(
            execution.accept_inference_result(
                execution_result(new),
                launch,
                self.delivery_protocol,
                arrived_at=arrived_at,
                arrived_monotonic=time.monotonic(),
            )
        )
        angles = [
            Rotation.from_matrix(rotation_from_state(item.absolute_target)).as_euler("zxy")[0]
            for item in execution.pending_actions[:4]
        ]
        np.testing.assert_allclose(angles, np.deg2rad([22.5, 45.0, 67.5, 90.0]), atol=1e-6)

    def test_fully_stale_result_is_rejected_and_old_queue_survives(self):
        execution, _ = self.configured_execution()
        old = self.joint_chunk(6, np.linspace(0.01, 0.06, 6))
        now = time.time()
        execution.queue_result(
            execution_result(old),
            self.raw_state,
            self.qpos,
            self.joint_protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        old_targets = [item.absolute_target.copy() for item in execution.pending_actions]
        launch, arrived_at = inference_launch(
            self.raw_state, self.qpos, generation=7, latency_s=1.0
        )
        self.assertFalse(
            execution.accept_inference_result(
                execution_result(self.joint_chunk(16, 0.2)),
                launch,
                self.joint_protocol,
                arrived_at=arrived_at,
                arrived_monotonic=time.monotonic(),
            )
        )
        self.assertEqual(execution.inference_skip_steps, 16)
        self.assertIn("fully stale", execution.rejected_result["reason"])
        for queued, expected in zip(execution.pending_actions, old_targets):
            np.testing.assert_allclose(queued.absolute_target, expected)

    def test_queue_underrun_holds_last_safe_target_without_zero_command(self):
        execution, piper = self.configured_execution()
        target = self.joint_chunk(1, 0.02)
        now = time.time()
        execution.queue_result(
            execution_result(target),
            self.raw_state,
            self.qpos,
            self.joint_protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        self.assertTrue(
            execution.execute_next(
                self.raw_state,
                self.qpos,
                self.joint_protocol,
                feedback_captured_at=time.time(),
            )
        )
        sent = len([call for call in piper.calls if call[0] == "JointCtrl"])
        qpos_after_command = target[0].copy()
        qpos_after_command[6] *= GRIPPER_MAX_M
        self.assertTrue(
            execution.execute_next(
                self.raw_state,
                qpos_after_command,
                self.joint_protocol,
                feedback_captured_at=time.time(),
            )
        )
        joint_calls = [call for call in piper.calls if call[0] == "JointCtrl"]
        self.assertEqual(len(joint_calls), sent + 1)
        self.assertEqual(joint_calls[-1][1:], joint_calls[-2][1:])
        self.assertTrue(any(value != 0 for value in joint_calls[-1][1:]))
        self.assertEqual(execution.state, "holding")
        self.assertTrue(execution.queue_underrun)
        self.assertEqual(execution.queue_underrun_count, 1)
        self.assertIn("queue underrun", execution.blocked_reason)
        telemetry = execution.metadata()["timed_target"]
        self.assertTrue(telemetry["hold"])
        self.assertEqual(telemetry["source_generation"], execution.active_generation)
        self.assertIn("target_monotonic", telemetry)
        self.assertIn("target_age_s", telemetry)

        recovery = self.joint_chunk(16, 0.20)
        launch, arrived_at = inference_launch(
            self.raw_state, qpos_after_command, generation=53
        )
        self.assertTrue(
            execution.accept_inference_result(
                execution_result(recovery),
                launch,
                self.joint_protocol,
                arrived_at=arrived_at,
                arrived_monotonic=launch.captured_monotonic,
            )
        )
        self.assertEqual(execution.inference_blend_steps, 3)
        self.assertTrue(execution.pending_actions[0].blended)
        self.assertFalse(execution.hold_active)

    def test_gripper_endpoint_requires_two_confirmed_steps_and_is_step_limited(self):
        execution, piper = self.configured_execution(
            gripper_lowpass_alpha=1.0,
            gripper_confirm_steps=2,
            max_joint_gripper_step=0.25,
        )
        actions = self.joint_chunk(2, 0.0)
        actions[:, 6] = 1.0
        now = time.time()
        execution.queue_result(
            execution_result(actions),
            self.raw_state,
            self.qpos,
            self.joint_protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )

        qpos = self.qpos.copy()
        self.assertTrue(
            execution.execute_next(
                self.raw_state,
                qpos,
                self.joint_protocol,
                feedback_captured_at=time.time(),
            )
        )
        first_gripper = [call for call in piper.calls if call[0] == "GripperCtrl"][-1][1]
        self.assertEqual(first_gripper, round(0.5 * GRIPPER_MAX_M * 1_000_000.0))

        self.assertTrue(
            execution.execute_next(
                self.raw_state,
                qpos,
                self.joint_protocol,
                feedback_captured_at=time.time(),
            )
        )
        second_gripper = [call for call in piper.calls if call[0] == "GripperCtrl"][-1][1]
        self.assertEqual(second_gripper, round(0.75 * GRIPPER_MAX_M * 1_000_000.0))
        self.assertGreater(second_gripper, first_gripper)

    def test_legacy_closed_fraction_is_converted_before_gripper_confirmation(self):
        protocol = validate_policy_metadata(LEGACY_DELIVERY_METADATA, "right")
        execution, piper = self.configured_execution(
            protocol,
            gripper_lowpass_alpha=1.0,
            gripper_confirm_steps=2,
            max_gripper_step=0.30,
        )
        actions = np.zeros((2, 7), dtype=np.float64)
        actions[:, 6] = 0.0  # legacy closed fraction 0 means fully open
        now = time.time()
        execution.queue_result(
            execution_result(actions),
            self.raw_state,
            self.qpos,
            protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        for _ in range(2):
            self.assertTrue(
                execution.execute_next(
                    self.raw_state,
                    self.qpos,
                    protocol,
                    feedback_captured_at=time.time(),
                )
            )
        gripper_calls = [call for call in piper.calls if call[0] == "GripperCtrl"]
        self.assertEqual(gripper_calls[0][1], round(0.5 * GRIPPER_MAX_M * 1_000_000.0))
        self.assertEqual(gripper_calls[1][1], round(0.8 * GRIPPER_MAX_M * 1_000_000.0))

    def test_default_schedule_is_4hz_while_control_remains_20hz(self):
        self.assertEqual(DEFAULT_INFERENCE_HZ, 4.0)
        execution, _ = self.configured_execution(hz=4.0, control_hz=20.0)
        self.assertEqual(execution.inference_hz, 4.0)
        self.assertEqual(execution.control_hz, 20.0)
        self.assertEqual(execution.policy_action_hz, 20.0)

        schedule = PeriodicSchedule(DEFAULT_INFERENCE_HZ, next_at=0.0)
        due_at = [tick * 0.05 for tick in range(20) if schedule.due(tick * 0.05)]
        np.testing.assert_allclose(due_at, [0.0, 0.25, 0.50, 0.75], atol=1e-12)
        self.assertAlmostEqual(schedule.period_s, 0.25)

    def test_blended_target_cannot_bypass_per_tick_delivery_safety(self):
        execution, piper = self.configured_execution(self.delivery_protocol, blend_steps=4)
        old = np.zeros((8, 7), dtype=np.float64)
        old[:, 6] = 0.5
        now = time.time()
        execution.queue_result(
            execution_result(old),
            self.raw_state,
            self.qpos,
            self.delivery_protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        new = np.zeros((20, 7), dtype=np.float64)
        new[:, 0] = 0.24  # first 1/4 blend row is a 0.06 m live-state step
        new[:, 6] = 0.5
        launch, arrived_at = inference_launch(self.raw_state, self.qpos, generation=8)
        self.assertTrue(
            execution.accept_inference_result(
                execution_result(new),
                launch,
                self.delivery_protocol,
                arrived_at=arrived_at,
                arrived_monotonic=time.monotonic(),
            )
        )
        self.assertTrue(execution.pending_actions[0].blended)
        self.assertFalse(
            execution.execute_next(
                self.raw_state,
                self.qpos,
                self.delivery_protocol,
                feedback_captured_at=time.time(),
            )
        )
        self.assertIn("translation step", execution.blocked_reason)
        self.assertNotIn("EndPoseCtrl", [call[0] for call in piper.calls])


class ExecutionQueueTest(unittest.TestCase):
    def setUp(self):
        self.raw_state = delivery_state(opening_fraction=0.5)
        self.qpos = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.035])

    def test_joint_queue_executes_first_four_at_individual_indices(self):
        protocol = validate_policy_metadata(JOINT_METADATA, "right")
        piper = FakePiper()
        execution = ExecutionController(piper, execution_args())
        execution.configure_protocol(protocol)
        execution.robot_enabled = {"right"}
        targets = np.stack(
            [
                np.array([step, 1, -1, 0, 0, 0, 0.5 + step])
                for step in (0.02, 0.04, 0.06, 0.08)
            ]
        )
        now = time.time()
        queued = execution.queue_result(
            execution_result(targets),
            self.raw_state,
            self.qpos,
            protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        self.assertEqual(queued, 4)
        np.testing.assert_allclose(execution.queue_anchor_state, self.qpos / np.array([1, 1, 1, 1, 1, 1, GRIPPER_MAX_M]))

        qpos = self.qpos.copy()
        for index, target in enumerate(targets):
            self.assertEqual(execution.queued_action_index, index)
            self.assertTrue(
                execution.execute_next(
                    self.raw_state, qpos, protocol, feedback_captured_at=time.time()
                )
            )
            qpos[:6] = target[:6]
            qpos[6] = target[6] * GRIPPER_MAX_M
        self.assertEqual(execution.pending_action_count, 0)
        self.assertEqual(execution.last_queued_action_index, 3)
        joint_calls = [call for call in piper.calls if call[0] == "JointCtrl"]
        self.assertEqual(len(joint_calls), 4)
        self.assertEqual(joint_calls[-1][1], int(np.rint(targets[-1, 0] * RAD_FACTOR)))
        self.assertEqual(
            execution.metadata()["last_decoded_absolute_target"]["right"][
                "gripper_opening_fraction"
            ],
            execution.metadata()["gripper_filter"]["opening_fraction"]["right"],
        )

    def test_double_gate_blocks_motion_but_keeps_anchor_and_decode_telemetry(self):
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        piper = FakePiper()
        execution = ExecutionController(piper, execution_args(allow_execution=False))
        execution.configure_protocol(protocol)
        action = np.array([[0.004, 0, 0, 0, 0, 0, 0.5]])
        now = time.time()
        execution.queue_result(
            execution_result(action),
            self.raw_state,
            self.qpos,
            protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        self.assertFalse(
            execution.execute_next(
                self.raw_state, self.qpos, protocol, feedback_captured_at=time.time()
            )
        )
        telemetry = execution.metadata()
        self.assertEqual(telemetry["execution_state"], "client_disabled")
        self.assertIsNotNone(telemetry["queue_anchor_state"])
        self.assertIsNotNone(telemetry["last_decoded_absolute_target"])
        self.assertNotIn("EndPoseCtrl", [call[0] for call in piper.calls])

    def test_delivery_queue_commands_absolute_targets_without_accumulating_v3_rows(self):
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        piper = FakePiper()
        execution = ExecutionController(piper, execution_args())
        execution.configure_protocol(protocol)
        ik_solver = RecordingContinuousIK()
        execution.ik_solver = ik_solver
        execution.robot_enabled = {"right"}
        actions = np.zeros((4, 7))
        actions[:, 0] = [0.004, 0.008, 0.012, 0.016]
        actions[:, 6] = 0.5
        now = time.time()
        execution.queue_result(
            execution_result(actions),
            self.raw_state,
            self.qpos,
            protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )

        current_state = self.raw_state.copy()
        for expected_index in range(4):
            self.assertTrue(
                execution.execute_next(
                    current_state,
                    self.qpos,
                    protocol,
                    feedback_captured_at=time.time(),
                )
            )
            current_state[0] = self.raw_state[0] + actions[expected_index, 0]
        joint_calls = [call for call in piper.calls if call[0] == "JointCtrl"]
        self.assertEqual(len(joint_calls), 4)
        self.assertNotIn("EndPoseCtrl", [call[0] for call in piper.calls])
        target_x_m = [target_xyz[0] for target_xyz, _ in ik_solver.targets]
        np.testing.assert_allclose(
            target_x_m, self.raw_state[0] + actions[:, 0], atol=1e-6
        )
        self.assertNotAlmostEqual(target_x_m[-1], self.raw_state[0] + actions[:, 0].sum())

    def test_delivery_execution_uses_continuous_ik_joint_control(self):
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        piper = FakePiper()
        execution = ExecutionController(piper, execution_args())
        execution.configure_protocol(protocol)
        ik_solver = RecordingContinuousIK()
        execution.ik_solver = ik_solver
        execution.robot_enabled = {"right"}
        actions = np.zeros((1, 7), dtype=np.float64)
        actions[0, 0] = 0.004
        actions[0, 6] = 0.5
        now = time.time()
        execution.queue_result(
            execution_result(actions),
            self.raw_state,
            self.qpos,
            protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        self.assertTrue(
            execution.execute_next(
                self.raw_state, self.qpos, protocol, feedback_captured_at=time.time()
            )
        )
        names = [call[0] for call in piper.calls]
        self.assertIn("JointCtrl", names)
        self.assertNotIn("EndPoseCtrl", names)
        self.assertEqual(len(ik_solver.targets), 1)

    def test_command_trace_correlates_ik_piper_units_and_next_cycle_feedback(self):
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        piper = FakePiper()
        execution = ExecutionController(piper, execution_args())
        execution.configure_protocol(protocol)
        execution.ik_solver = RecordingContinuousIK()
        execution.robot_enabled = {"right"}
        actions = np.zeros((2, 7), dtype=np.float64)
        actions[:, 0] = [0.004, 0.008]
        actions[:, 6] = 0.5
        now = time.time()
        execution.queue_result(
            execution_result(actions),
            self.raw_state,
            self.qpos,
            protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )

        self.assertTrue(
            execution.execute_next(
                self.raw_state, self.qpos, protocol, feedback_captured_at=time.time()
            )
        )
        first_trace = execution.metadata()["last_actuator_command"]
        self.assertEqual(first_trace["generation"], 1)
        self.assertEqual(first_trace["source_index"], 0)
        self.assertEqual(first_trace["queue_index"], 0)
        side = first_trace["sides"]["right"]
        self.assertEqual(side["control_path"], "delivery_continuous_ik_joint")
        self.assertEqual(len(side["pre_ik_eef_target"]["absolute_target"]), 10)
        self.assertEqual(len(side["full_ik_solution_joints_rad"]), 6)
        self.assertEqual(len(side["commanded_joints_rad"]), 6)
        self.assertEqual(len(side["piper_jointctrl_units"]), 6)
        np.testing.assert_array_equal(
            side["piper_jointctrl_units"],
            np.rint(np.asarray(side["commanded_joints_rad"]) * RAD_FACTOR).astype(np.int64),
        )

        next_qpos = self.qpos.copy()
        next_qpos[:6] = np.asarray(side["commanded_joints_rad"])
        next_qpos[6] = float(side["commanded_gripper_m"])
        self.assertTrue(
            execution.execute_next(
                self.raw_state, next_qpos, protocol, feedback_captured_at=time.time()
            )
        )
        feedback = execution.metadata()["last_command_feedback"]
        self.assertEqual(feedback["command_sequence"], first_trace["command_sequence"])
        self.assertEqual(feedback["generation"], first_trace["generation"])
        self.assertEqual(feedback["source_index"], first_trace["source_index"])
        self.assertEqual(feedback["feedback_cycle_offset"], 1)
        self.assertAlmostEqual(feedback["max_joint_abs_error_rad"], 0.0)
        self.assertAlmostEqual(feedback["max_gripper_abs_error_m"], 0.0)
        self.assertEqual(
            feedback["sides"]["right"]["command_feedback_error"][
                "joint_error_definition"
            ],
            "feedback_minus_command",
        )

    def test_unreachable_delivery_row_is_dropped_without_destroying_chunk(self):
        protocol = validate_policy_metadata(DELIVERY_METADATA, "right")
        piper = FakePiper()
        execution = ExecutionController(piper, execution_args())
        execution.configure_protocol(protocol)
        execution.ik_solver = RejectFirstContinuousIK()
        execution.robot_enabled = {"right"}
        actions = np.zeros((2, 7), dtype=np.float64)
        actions[:, 6] = 0.5
        now = time.time()
        execution.queue_result(
            execution_result(actions),
            self.raw_state,
            self.qpos,
            protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        self.assertFalse(
            execution.execute_next(
                self.raw_state, self.qpos, protocol, feedback_captured_at=time.time()
            )
        )
        self.assertEqual(execution.state, "ready")
        self.assertEqual(execution.pending_action_count, 1)
        self.assertIn("dropped unsafe queued target", execution.blocked_reason)
        self.assertTrue(
            execution.execute_next(
                self.raw_state, self.qpos, protocol, feedback_captured_at=time.time()
            )
        )
        self.assertEqual(execution.pending_action_count, 0)
        self.assertIn("JointCtrl", [call[0] for call in piper.calls])

    def test_feedback_freshness_and_status_block_before_motion(self):
        protocol = validate_policy_metadata(JOINT_METADATA, "right")
        target = np.array([[0.02, 1, -1, 0, 0, 0, 0.5]])
        now = time.time()

        stale_piper = FakePiper()
        stale = ExecutionController(stale_piper, execution_args())
        stale.robot_enabled = {"right"}
        stale.queue_result(
            execution_result(target),
            self.raw_state,
            self.qpos,
            protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        self.assertFalse(
            stale.execute_next(
                self.raw_state,
                self.qpos,
                protocol,
                feedback_captured_at=now - 1.0,
            )
        )
        self.assertIn("feedback age", stale.blocked_reason)
        self.assertNotIn("JointCtrl", [call[0] for call in stale_piper.calls])

        bad_piper = FakePiper(err_code=1)
        bad = ExecutionController(bad_piper, execution_args())
        bad.robot_enabled = {"right"}
        bad.queue_result(
            execution_result(target),
            self.raw_state,
            self.qpos,
            protocol,
            {"cam_high": now, "cam_wrist": now},
            0.01,
        )
        self.assertFalse(
            bad.execute_next(
                self.raw_state, self.qpos, protocol, feedback_captured_at=time.time()
            )
        )
        self.assertIn("status is not normal", bad.blocked_reason)
        self.assertNotIn("JointCtrl", [call[0] for call in bad_piper.calls])

    def test_bimanual_prevalidation_is_atomic(self):
        protocol = validate_policy_metadata(BIMANUAL_JOINT_METADATA, "both", "bimanual")
        pipers = {"left": FakePiper(), "right": FakePiper()}
        execution = ExecutionController(
            pipers, execution_args(arm_mode="bimanual", arm_side="both")
        )
        execution.robot_enabled = {"left", "right"}
        qpos = np.concatenate([self.qpos, self.qpos])
        target = np.concatenate(
            [
                np.array([0.1, 1, -1, 0, 0, 0, 0.5]),
                np.array([0.31, 1, -1, 0, 0, 0, 0.5]),
            ]
        )[None, :]
        now = time.time()
        execution.queue_result(
            execution_result(target),
            np.concatenate([self.raw_state, self.raw_state]),
            qpos,
            protocol,
            {"cam_high": now, "cam_left_wrist": now, "cam_right_wrist": now},
            0.01,
        )
        self.assertFalse(
            execution.execute_next(
                np.concatenate([self.raw_state, self.raw_state]),
                qpos,
                protocol,
                feedback_captured_at=time.time(),
            )
        )
        self.assertIn("joint1 step", execution.blocked_reason)
        for piper in pipers.values():
            self.assertNotIn("JointCtrl", [call[0] for call in piper.calls])


if __name__ == "__main__":
    unittest.main()
