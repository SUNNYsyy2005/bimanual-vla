"""Regression checks for real RECAP parquet return computation."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

path = Path(os.environ["REPO_PATH"]) / "examples/offline_rl/advantage_labeling/recap/process/compute_returns.py"
spec = importlib.util.spec_from_file_location("recap_compute_returns_under_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ReturnContractTest(unittest.TestCase):
    def test_failure_terminal_penalty_survives_lerobot_vector_boolean(self):
        for vector in [False, True]:
            with self.subTest(vector=vector), tempfile.TemporaryDirectory() as temp:
                target = Path(temp) / "episode.parquet"
                flags = [False, False, False, True, True]
                if vector:
                    flags = [[flag] for flag in flags]
                pq.write_table(pa.table({"episode_index": [0, 0, 0, 1, 1],
                                         "frame_index": [0, 1, 2, 0, 1], "is_success": flags}), target)
                table = module._process_single_parquet(str(target), "rollout", 1.0, -20.0, {})
                np.testing.assert_array_equal(table["return"].to_numpy(), [-22, -21, -20, -1, 0])
    def test_invalid_success_flags_fail_explicitly(self):
        for value in [[], [False, True], "false", None, float("nan"), 2]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.coerce_episode_success(value)

if __name__ == "__main__":
    unittest.main()
