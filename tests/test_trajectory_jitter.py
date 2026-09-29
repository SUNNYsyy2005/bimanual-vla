from __future__ import annotations

import unittest

import numpy as np

from bimanual_vla.deployment.jitter import TrajectoryJitterMonitor


class TrajectoryJitterMonitorTest(unittest.TestCase):
    def test_three_metrics_use_executed_chunk_rows(self):
        monitor = TrajectoryJitterMonitor(20.0)
        events = []
        for tick, (generation, index, position) in enumerate(
            [
                (1, 0, 0.0), (1, 1, 1.0), (1, 2, 3.0), (1, 3, 6.0),
                (2, 0, 7.0), (2, 1, 9.0), (2, 2, 12.0),
            ]
        ):
            events.extend(monitor.observe(
                joints_rad=np.array([position, 0.0]),
                generation=generation,
                queue_index=index,
                command_at=10.0 + tick * 0.05,
            ))
        summary = monitor.summary()
        self.assertAlmostEqual(summary["intra_accel_mean_rad_per_step2"], 1.0)
        self.assertEqual(summary["intra_accel_samples"], 3)
        self.assertAlmostEqual(summary["boundary_jump_mean_rad_l2"], 1.0)
        self.assertEqual(summary["boundary_jump_samples"], 1)
        self.assertAlmostEqual(summary["boundary_momentum_cosine_mean"], 1.0)
        self.assertEqual(summary["boundary_momentum_samples"], 1)
        self.assertEqual([event["event"] for event in events], [
            "chunk_completed", "chunk_boundary", "chunk_boundary_momentum"
        ])

    def test_hold_gap_and_stationary_velocity_are_excluded(self):
        monitor = TrajectoryJitterMonitor(20.0)
        samples = [
            (1, 0, 0.0, 0.00, False),
            (1, 1, 1.0, 0.05, False),
            (1, 2, 2.0, 0.10, False),
            (1, 2, 2.0, 0.15, True),
            (2, 0, 5.0, 0.20, False),
            (2, 1, 5.0, 0.25, False),
            (2, 3, 8.0, 0.30, False),
        ]
        for generation, index, value, timestamp, hold in samples:
            monitor.observe(
                joints_rad=np.array([value]),
                generation=generation,
                queue_index=index,
                command_at=timestamp,
                hold=hold,
            )
        summary = monitor.summary()
        self.assertAlmostEqual(summary["intra_accel_mean_rad_per_step2"], 0.0)
        self.assertEqual(summary["intra_accel_samples"], 1)
        self.assertAlmostEqual(summary["boundary_jump_mean_rad_l2"], 3.0)
        self.assertEqual(summary["boundary_momentum_samples"], 0)


if __name__ == "__main__":
    unittest.main()
