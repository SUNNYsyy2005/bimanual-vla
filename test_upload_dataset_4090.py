from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from upload_dataset_4090 import classify_dataset_source, complete_upload, main, prepare_dataset_directory


class UploadCompletionDiagnosticsTest(unittest.TestCase):
    def test_failed_completion_prints_server_validation_details(self):
        class FakeClient:
            def json(self, method, path, payload=None):
                raise RuntimeError("HTTP 400: dataset structural validation failed")

            def request(self, method, path):
                return {
                    "state": "failed",
                    "error": "dataset structural validation failed",
                    "structural_validation": "FAILED: missing video",
                }

        error_output = io.StringIO()
        with redirect_stderr(error_output):
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                complete_upload(FakeClient(), "upload-id")

        output = error_output.getvalue()
        self.assertIn("Server upload diagnostics", output)
        self.assertIn("dataset structural validation failed", output)
        self.assertIn("FAILED: missing video", output)


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


if __name__ == "__main__":
    unittest.main()
