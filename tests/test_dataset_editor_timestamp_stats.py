import unittest

import pyarrow as pa

from server_4090.dataset_editor import _recompute_episode_stats


class TimestampStatsTest(unittest.TestCase):
    def test_image_acquisition_timestamps_are_preserved_without_rgb_stats(self):
        table = pa.table({"observation.state": [[0., 1.], [1., 2.]],
                          "image_timestamp.cam_high": [100., 100.05],
                          "image_timestamp.cam_right_wrist": [100.01, 100.06],
                          "state_timestamp": [100., 100.05]})
        original = table.to_pydict()
        rgb_stats = {"mean": [[[.3]], [[.4]], [[.5]]], "count": [2]}
        existing = {"stats": {"observation.images.cam_high": rgb_stats,
                    "image_timestamp.cam_high": {"mean": [100.025], "count": [2]}}}
        stats = _recompute_episode_stats(table, existing)["stats"]
        self.assertNotIn("image_timestamp.cam_high", stats)
        self.assertNotIn("image_timestamp.cam_right_wrist", stats)
        self.assertEqual(stats["observation.images.cam_high"], rgb_stats)
        self.assertAlmostEqual(stats["state_timestamp"]["mean"][0], 100.025, places=4)
        self.assertEqual(stats["observation.state"]["mean"], [.5, 1.5])
        self.assertEqual(table.to_pydict(), original)
        self.assertIn("image_timestamp.cam_high", existing["stats"])


if __name__ == "__main__":
    unittest.main()
