from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from export_lerobot import _export_legacy_single_delivery


class LegacySingleDeliveryExportTest(unittest.TestCase):
    def test_server_compatible_feature_names_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ep_0000.npz"
            states = np.zeros((2, 10), dtype=np.float32)
            actions = np.zeros((2, 7), dtype=np.float32)
            images = np.zeros((2, 256, 256, 3), dtype=np.uint8)
            np.savez(
                source,
                state=states,
                actions=actions,
                image=images,
                wrist_image=images,
            )

            output = root / "legacy_delivery"
            episodes, frames = _export_legacy_single_delivery(
                [SimpleNamespace(path=source, instruction="pick up the cube")],
                output,
                fps=20,
            )

            info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual((episodes, frames), (1, 2))
            self.assertTrue({"state", "actions", "image", "wrist_image"}.issubset(info["features"]))
            self.assertNotIn("observation.state", info["features"])
            self.assertNotIn("action", info["features"])
            self.assertEqual(info["total_videos"], 0)


if __name__ == "__main__":
    unittest.main()
