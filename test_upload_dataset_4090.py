from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from upload_dataset_4090 import (
    classify_dataset_source,
    classify_lerobot_contract,
    main,
    prepare_dataset_directory,
    prepare_lerobot_dataset,
)


class DatasetUploadInputTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.cache = self.root / "cache"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_lerobot_directory_is_used_without_export(self):
        dataset = self.root / "lerobot"
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta" / "info.json").write_text("{}", encoding="utf-8")

        prepared, kind = prepare_dataset_directory(
            dataset,
            "pick_cube",
            self.cache,
            fps=20,
            allow_incomplete_gripper_coverage=False,
            rebuild=False,
        )

        self.assertEqual(kind, "lerobot")
        self.assertEqual(prepared, dataset)

    def test_gui_npz_directory_is_exported_and_cached(self):
        dataset = self.root / "episodes"
        dataset.mkdir()
        (dataset / "ep_0000.npz").write_bytes(b"synthetic-npz")
        calls: list[tuple[Path, Path, int, bool]] = []

        def fake_export(input_dir, output_root, *, fps, allow_incomplete_gripper_coverage):
            input_dir = Path(input_dir)
            output_root = Path(output_root)
            calls.append((input_dir, output_root, fps, allow_incomplete_gripper_coverage))
            (output_root / "meta").mkdir(parents=True)
            (output_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
            return output_root

        with patch("export_lerobot.export_dataset", side_effect=fake_export):
            first, first_kind = prepare_dataset_directory(
                dataset,
                "pick.cube-v1",
                self.cache,
                fps=20,
                allow_incomplete_gripper_coverage=True,
                rebuild=False,
            )
            second, second_kind = prepare_dataset_directory(
                dataset,
                "pick.cube-v1",
                self.cache,
                fps=20,
                allow_incomplete_gripper_coverage=True,
                rebuild=False,
            )

        self.assertEqual(first_kind, "raw_npz")
        self.assertEqual(second_kind, "raw_npz")
        self.assertEqual(first, second)
        self.assertTrue((first / "meta" / "info.json").is_file())
        self.assertTrue(first.with_name(first.name + ".json").is_file())
        self.assertEqual(calls, [(dataset, first.with_name(first.name + ".building"), 20, True)])

    def test_rebuild_forces_raw_export_again(self):
        dataset = self.root / "episodes"
        dataset.mkdir()
        (dataset / "ep_0000.npz").write_bytes(b"synthetic-npz")
        call_count = 0

        def fake_export(input_dir, output_root, *, fps, allow_incomplete_gripper_coverage):
            nonlocal call_count
            call_count += 1
            output_root = Path(output_root)
            (output_root / "meta").mkdir(parents=True)
            (output_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
            return output_root

        with patch("export_lerobot.export_dataset", side_effect=fake_export):
            for rebuild in (False, True):
                prepare_dataset_directory(
                    dataset,
                    "pick_cube",
                    self.cache,
                    fps=20,
                    allow_incomplete_gripper_coverage=False,
                    rebuild=rebuild,
                )

        self.assertEqual(call_count, 2)

    def test_unsupported_directory_is_rejected(self):
        dataset = self.root / "empty"
        dataset.mkdir()

        with self.assertRaisesRegex(ValueError, "LeRobot directory.*GUI collection"):
            classify_dataset_source(dataset)

    def test_prepare_only_does_not_require_server_token(self):
        dataset = self.root / "episodes"
        dataset.mkdir()
        (dataset / "ep_0000.npz").write_bytes(b"synthetic-npz")
        prepared = self.root / "prepared"
        (prepared / "meta").mkdir(parents=True)
        (prepared / "meta" / "info.json").write_text("{}", encoding="utf-8")
        output = io.StringIO()

        with (
            patch.dict(os.environ, {"BIMANUAL_VLA_SERVER_TOKEN": ""}),
            patch.object(
                sys,
                "argv",
                [
                    "upload_dataset_4090.py",
                    str(dataset),
                    "--name",
                    "pick_cube",
                    "--prepare-only",
                    "--cache-dir",
                    str(self.cache),
                ],
            ),
            patch(
                "upload_dataset_4090.prepare_dataset_directory",
                return_value=(prepared, "raw_npz"),
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(main(), 0)

        self.assertIn(f"PREPARED_LEROBOT_PATH={prepared}", output.getvalue())

    def test_canonical_v3_single_and_bimanual_contracts(self):
        cases = (
            ("joint_single_7d", "joint", 7, 7, 1, 7, None),
            ("delivery_single_10d", "delivery", 10, 10, 1, 7, "absolute_eef_target"),
            ("delivery_bimanual_20d", "delivery", 20, 20, 2, 14, "absolute_eef_target"),
        )
        for name, schema, state_dim, raw_action_dim, arm_count, model_action_dim, action_format in cases:
            with self.subTest(name=name):
                contract = classify_lerobot_contract(
                    {
                        "contract_version": 3,
                        "contract_format": "canonical",
                        "schema": schema,
                        "features": {
                            "observation.state": {"dtype": "float32", "shape": [state_dim]},
                            "action": {"dtype": "float32", "shape": [raw_action_dim]},
                        },
                    }
                )
                self.assertEqual(contract["contract_format"], "canonical")
                self.assertFalse(contract["legacy"])
                self.assertEqual(contract["schema"], schema)
                self.assertEqual(contract["arm_count"], arm_count)
                self.assertEqual(contract["arm_mode"], "single" if arm_count == 1 else "bimanual")
                self.assertEqual(contract["state_dim"], state_dim)
                self.assertEqual(contract["raw_action_dim"], raw_action_dim)
                self.assertEqual(contract["model_action_dim"], model_action_dim)
                self.assertEqual(contract["delivery_action_format"], action_format)

    def test_canonical_10d_delivery_is_not_rewritten_as_legacy(self):
        dataset = self.root / "canonical"
        (dataset / "meta").mkdir(parents=True)
        info = {
            "contract_version": 3,
            "contract_format": "canonical",
            "schema": "delivery",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [10]},
                "action": {"dtype": "float32", "shape": [10]},
            },
        }
        (dataset / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
        self.assertEqual(
            prepare_lerobot_dataset(dataset, "canonical", self.cache, rebuild=False),
            dataset,
        )

    def test_metadata_free_delivery_is_copied_and_marked_legacy_v2(self):
        dataset = self.root / "legacy"
        (dataset / "meta").mkdir(parents=True)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "piper",
            "fps": 20,
            "features": {
                "state": {"dtype": "float32", "shape": [10]},
                "actions": {"dtype": "float32", "shape": [7]},
                "image": {"dtype": "video", "shape": [3, 256, 256]},
                "wrist_image": {"dtype": "video", "shape": [3, 256, 256]},
            },
        }
        (dataset / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

        prepared = prepare_lerobot_dataset(
            dataset, "8_3_64eps", self.cache, rebuild=False
        )

        self.assertNotEqual(prepared, dataset)
        normalized = json.loads((prepared / "meta" / "info.json").read_text(encoding="utf-8"))
        self.assertEqual(normalized["contract_format"], "legacy_v2")
        self.assertEqual(normalized["legacy_format"], "legacy_v2")
        self.assertEqual(normalized["raw_action_dim"], 7)
        self.assertEqual(normalized["model_action_dim"], 7)
        self.assertEqual(normalized["delivery_action_format"], "step_delta")
        self.assertEqual(normalized["gripper_semantics"], "absolute_closed_fraction_0_open_1_closed")
        self.assertEqual(classify_lerobot_contract(normalized)["contract_format"], "legacy_v2")
        original = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
        self.assertNotIn("contract_format", original)

    def test_metadata_free_bimanual_delivery_is_classified_as_legacy_v2(self):
        contract = classify_lerobot_contract(
            {
                "features": {
                    "state": {"dtype": "float32", "shape": [20]},
                    "actions": {"dtype": "float32", "shape": [14]},
                }
            }
        )
        self.assertTrue(contract["legacy"])
        self.assertEqual(contract["contract_format"], "legacy_v2")
        self.assertEqual(contract["arm_mode"], "bimanual")
        self.assertEqual(contract["raw_action_dim"], 14)
        self.assertEqual(contract["model_action_dim"], 14)

    def test_legacy_normalization_repairs_stale_numeric_episode_stats(self):
        dataset = self.root / "legacy_stale"
        (dataset / "meta").mkdir(parents=True)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "piper",
            "fps": 20,
            "features": {
                "state": {"dtype": "float32", "shape": [10]},
                "actions": {"dtype": "float32", "shape": [7]},
            },
        }
        (dataset / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
        parquet = dataset / "data/chunk-000/episode_000000.parquet"
        parquet.parent.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "state": [np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32)],
                    "actions": [np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32)],
                    "episode_index": np.zeros(2, dtype=np.int64),
                    "index": np.arange(2, dtype=np.int64),
                }
            ),
            parquet,
        )
        (dataset / "meta" / "episodes_stats.jsonl").write_text(
            json.dumps(
                {
                    "episode_index": 0,
                    "stats": {
                        "index": {"min": [0], "max": [1], "mean": [999], "std": [0], "count": [2]}
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        prepared = prepare_lerobot_dataset(dataset, "legacy_stale", self.cache, rebuild=False)
        repaired = json.loads((prepared / "meta" / "episodes_stats.jsonl").read_text().strip())
        self.assertEqual(repaired["stats"]["index"]["mean"], [0.5])
        original = json.loads((dataset / "meta" / "episodes_stats.jsonl").read_text().strip())
        self.assertEqual(original["stats"]["index"]["mean"], [999])


if __name__ == "__main__":
    unittest.main()
