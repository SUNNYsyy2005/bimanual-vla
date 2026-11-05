"""Zero-phase EEF chunk smoothing (2026-11-05 end-effector jitter review)."""
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_vla.data.action_conventions import matrix_to_rotation6d
from bimanual_vla.deployment.client import (
    DEFAULT_EEF_CHUNK_SMOOTHING_STEPS, decode_action_queue, rotation_from_state, smooth_delivery_targets)
from tests import test_robot_observation_bridge as bridge


def chunk(positions, rotvecs, grippers):
    return np.array([np.r_[p, matrix_to_rotation6d(Rotation.from_rotvec(r).as_matrix()), g]
                     for p, r, g in zip(positions, rotvecs, grippers)])


class SmoothDeliveryTargetsTest(unittest.TestCase):
    def test_quadratic_motion_and_gripper_switch_are_unchanged(self):
        t = np.arange(50) / 30
        pos = np.stack([.1 + .2 * t + .3 * t**2, -.2 * t, .25 + .05 * t**2], 1)
        rotation = np.tile([.1, -.2, .5], (50, 1))
        grip = (t > .8).astype(float)
        absolute = np.c_[chunk(pos, rotation, grip), chunk(pos[::-1], rotation, 1 - grip)]
        np.testing.assert_allclose(smooth_delivery_targets(absolute, 2, 7), absolute, atol=1e-9)

    def test_step_noise_is_removed_and_rotations_stay_orthonormal(self):
        rng = np.random.default_rng(0)
        centre = np.array([.1, .3, .25])
        absolute = chunk(centre + rng.normal(0, .004, (50, 3)),
                         rng.normal(0, .01, (50, 3)) + [0, 0, .5], np.full(50, .5))
        smoothed = smooth_delivery_targets(absolute, 1, DEFAULT_EEF_CHUNK_SMOOTHING_STEPS)
        before = np.linalg.norm(absolute[10:, :3] - centre, axis=1)
        after = np.linalg.norm(smoothed[10:, :3] - centre, axis=1)
        self.assertLess(np.sqrt(np.mean(after**2)), .6 * np.sqrt(np.mean(before**2)))
        for row in smoothed:
            np.testing.assert_allclose(row[3:9], matrix_to_rotation6d(rotation_from_state(row.copy())), atol=1e-12)
        np.testing.assert_array_equal(smoothed[:, 9], absolute[:, 9])

    def test_disabled_or_short_chunks_are_returned_unchanged(self):
        absolute = chunk(np.zeros((5, 3)), np.zeros((5, 3)), np.zeros(5))
        self.assertIs(smooth_delivery_targets(absolute, 1, 0), absolute)
        self.assertIs(smooth_delivery_targets(absolute, 1, 7), absolute)


class ControllerSmoothingTest(unittest.TestCase):
    def queued_targets(self, **overrides):
        helper = bridge.AsyncInferencePipelineTest()
        helper.setUp()
        execution, _ = helper.configured_execution(helper.delivery_protocol, **overrides)
        rng = np.random.default_rng(1)
        actions = np.zeros((50, 7))
        actions[:, :3] = rng.normal(0, .004, (50, 3))
        actions[:, 6] = .5
        now = time.time()
        execution.queue_result(bridge.execution_result(actions), helper.raw_state, helper.qpos,
                               helper.delivery_protocol, {key: now for key in helper.delivery_protocol.camera_keys}, .01)
        return {row.queue_index: row.absolute_target.copy() for row in execution.pending_actions}, helper

    def test_controller_smooths_decoded_chunks_unless_disabled(self):
        raw, helper = self.queued_targets(eef_chunk_smoothing_steps=0)
        smoothed, _ = self.queued_targets()
        common = sorted(set(raw) & set(smoothed))
        self.assertGreater(len(common), 20)
        centre = helper.raw_state[:3]
        spread = {name: np.sqrt(np.mean([np.sum((rows[i][:3] - centre)**2) for i in common]))
                  for name, rows in (('raw', raw), ('smoothed', smoothed))}
        self.assertLess(spread['smoothed'], .6 * spread['raw'])
        for i in common:
            self.assertEqual(raw[i][9], smoothed[i][9])

    def test_decode_default_keeps_raw_targets(self):
        helper = bridge.AsyncInferencePipelineTest()
        helper.setUp()
        actions = np.zeros((50, 7))
        actions[::2, 0] = .004
        actions[:, 6] = .5
        _, rows = decode_action_queue(actions, helper.delivery_protocol, helper.raw_state, helper.qpos,
                                      observation_capture_monotonic=100.)
        self.assertAlmostEqual(rows[0].absolute_target[0] - rows[1].absolute_target[0], .004)


if __name__ == "__main__":
    unittest.main()
