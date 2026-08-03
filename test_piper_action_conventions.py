from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from piper_action_conventions import (
    DELIVERY_RAW_ACTION_SEMANTICS,
    DELIVERY_STEP_ACTION_SEMANTICS,
    LEGACY_GRIPPER_SEMANTICS,
    NEW_GRIPPER_SEMANTICS,
    absolute_eef_targets_to_chunk_origin,
    chunk_origin_deltas_to_absolute_eef_targets,
    load_eef_command_mappings,
    matrix_to_rotation6d,
    step_deltas_to_chunk_origin,
)


def eef(xyz, rotation, gripper):
    return np.concatenate((xyz, matrix_to_rotation6d(rotation), [gripper])).astype(np.float32)


class PiperActionConventionTest(unittest.TestCase):
    def test_v3_absolute_eef_round_trip_anchors_every_row_to_current(self):
        current_rotation = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
        current = eef([0.1, 0.2, 0.3], current_rotation, 0.4)
        target0 = eef([0.11, 0.18, 0.35], Rotation.from_rotvec([0.1, 0, 0]).as_matrix() @ current_rotation, 0.8)
        target1 = eef([0.15, 0.21, 0.31], Rotation.from_rotvec([0, -0.2, 0]).as_matrix() @ current_rotation, 0.2)
        targets = np.stack((target0, target1))

        actions = absolute_eef_targets_to_chunk_origin(current, targets)
        np.testing.assert_allclose(actions[:, :3], targets[:, :3] - current[:3], atol=1e-7)
        np.testing.assert_allclose(actions[:, 6], [0.8, 0.2])
        restored = chunk_origin_deltas_to_absolute_eef_targets(current, actions)
        np.testing.assert_allclose(restored[:, :3], targets[:, :3], atol=1e-6)
        np.testing.assert_allclose(restored[:, 9], targets[:, 9], atol=1e-6)
        for actual, expected in zip(restored, targets):
            np.testing.assert_allclose(
                Rotation.from_matrix(np.column_stack((actual[3:6], actual[6:9], np.cross(actual[3:6], actual[6:9])))).as_matrix(),
                Rotation.from_matrix(np.column_stack((expected[3:6], expected[6:9], np.cross(expected[3:6], expected[6:9])))).as_matrix(),
                atol=1e-6,
            )

    def test_bimanual_v3_arms_are_independent(self):
        current = np.concatenate((eef([0, 0, 0], np.eye(3), 0.1), eef([1, 2, 3], np.eye(3), 0.9)))
        targets = np.stack((
            np.concatenate((eef([0.1, 0, 0], np.eye(3), 0.2), eef([1, 2.2, 3], np.eye(3), 0.8))),
            np.concatenate((eef([0.2, 0, 0], np.eye(3), 0.3), eef([1, 2.4, 3], np.eye(3), 0.7))),
        ))
        actions = absolute_eef_targets_to_chunk_origin(current, targets, arm_count=2)
        np.testing.assert_allclose(actions[:, 0], [0.1, 0.2])
        np.testing.assert_allclose(actions[:, 8], [0.2, 0.4], atol=1e-6)
        np.testing.assert_allclose(actions[:, [6, 13]], [[0.2, 0.8], [0.3, 0.7]])

    def test_legacy_step_conversion_is_unchanged(self):
        actions = np.zeros((4, 7), dtype=np.float32)
        actions[:, :3] = [[0.01, 0, 0], [0, 0.02, 0], [0, 0, -0.03], [0.04, 0, 0]]
        actions[:, 6] = [0.0, 0.25, 0.75, 1.0]
        converted = step_deltas_to_chunk_origin(actions, arm_count=1)
        np.testing.assert_allclose(converted[:, :3], [[0.01, 0, 0], [0.01, 0.02, 0], [0.01, 0.02, -0.03], [0.05, 0.02, -0.03]])
        np.testing.assert_array_equal(converted[:, 6], actions[:, 6])

    def test_noncommuting_legacy_rotations_are_left_composed(self):
        actions = np.zeros((3, 7), dtype=np.float64)
        actions[:, 3:6] = [[0.3, 0, 0], [0, -0.25, 0], [0, 0, 0.2]]
        converted = step_deltas_to_chunk_origin(actions, arm_count=1)
        total = np.eye(3)
        for index, delta in enumerate(actions[:, 3:6]):
            total = Rotation.from_rotvec(delta).as_matrix() @ total
            np.testing.assert_allclose(Rotation.from_rotvec(converted[index, 3:6]).as_matrix(), total, atol=1e-12)

    def test_delivery_mapping_requires_explicit_valid_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                load_eef_command_mappings(missing, arm_sides=("right",))
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps({
                "rotation_matrix": np.eye(3).tolist(),
                "translation_m": [0.1, -0.2, 0.3],
                "position_scale": 2.0,
                "gripper_fraction_scale": 0.5,
                "gripper_fraction_offset": 0.1,
                "source_frame": "master_base",
                "target_frame": "slave_base",
            }))
            mapping = load_eef_command_mappings(path, arm_sides=("right",))["right"]
            result = mapping.map_target(eef([0.2, 0.1, -0.1], np.eye(3), 0.8))
            np.testing.assert_allclose(result[:3], [0.5, 0.0, 0.1], atol=1e-7)
            self.assertAlmostEqual(float(result[9]), 0.5)

    def test_public_semantics_are_explicit(self):
        self.assertEqual(DELIVERY_RAW_ACTION_SEMANTICS, "absolute_eef_target")
        self.assertEqual(DELIVERY_STEP_ACTION_SEMANTICS, "eef_delta_base_xyz_left_rotvec_gripper_target")
        self.assertIn("0_closed_1_open", NEW_GRIPPER_SEMANTICS)
        self.assertIn("0_open_1_closed", LEGACY_GRIPPER_SEMANTICS)

    def test_rejects_invalid_shapes(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            step_deltas_to_chunk_origin(np.zeros((2, 6)), arm_count=1)
        with self.assertRaisesRegex(ValueError, "arm_count"):
            absolute_eef_targets_to_chunk_origin(np.zeros(30), np.zeros((2, 30)), arm_count=3)


if __name__ == "__main__":
    unittest.main()
