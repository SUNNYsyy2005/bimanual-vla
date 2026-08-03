from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from upload_dataset_4090 import classify_dataset_source, prepare_dataset_directory


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


if __name__ == "__main__":
    unittest.main()
