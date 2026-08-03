from __future__ import annotations

import json
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from server_4090.dataset_editor import DatasetEditor


DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_dataset(root: Path, name: str, episode_lengths: list[int], *, robot_type: str = "piper") -> Path:
    dataset = root / name
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "fps": 20,
        "chunks_size": 1000,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7]},
            "action": {"dtype": "float32", "shape": [7]},
        },
        "action_semantics": "absolute_joint_position",
        "action_offset": 1,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "total_episodes": len(episode_lengths),
        "total_frames": sum(episode_lengths),
        "total_tasks": len(episode_lengths),
        "total_videos": 0,
        "total_chunks": 1,
        "splits": {"train": f"0:{len(episode_lengths)}"},
    }
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    write_jsonl(
        dataset / "meta" / "tasks.jsonl",
        [{"task_index": index, "task": f"instruction {name} {index}"} for index in range(len(episode_lengths))],
    )
    write_jsonl(
        dataset / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": index,
                "tasks": [f"instruction {name} {index}"],
                "length": length,
                "task_name": f"task_{index}",
                "success": True,
                "operator": name,
            }
            for index, length in enumerate(episode_lengths)
        ],
    )
    write_jsonl(
        dataset / "meta" / "episodes_stats.jsonl",
        [{"episode_index": index, "stats": {}} for index in range(len(episode_lengths))],
    )

    global_index = 0
    for episode_index, length in enumerate(episode_lengths):
        path = dataset / DATA_PATH.format(episode_chunk=0, episode_index=episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "observation.state": [np.full(7, episode_index, dtype=np.float32) for _ in range(length)],
                "action": [np.full(7, episode_index + 0.5, dtype=np.float32) for _ in range(length)],
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(global_index, global_index + length, dtype=np.int64),
                "task_index": np.full(length, episode_index, dtype=np.int64),
            }
        )
        pq.write_table(table, path)
        raw = dataset / "raw" / f"episode_{episode_index:06d}.npz"
        raw.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw,
            state=np.full((length, 7), episode_index, dtype=np.float32),
            actions=np.full((length, 7), episode_index + 0.5, dtype=np.float32),
            frame_index=np.arange(length, dtype=np.int64),
            episode_index=np.full(length, episode_index, dtype=np.int64),
            index=np.arange(global_index, global_index + length, dtype=np.int64),
            task_index=np.full(length, episode_index, dtype=np.int64),
            instruction=np.asarray(f"instruction {name} {episode_index}"),
        )
        global_index += length
    return dataset


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def basic_validate(path: Path) -> str:
    info = json.loads((path / "meta" / "info.json").read_text(encoding="utf-8"))
    parquets = sorted((path / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquets) != info["total_episodes"]:
        raise ValueError("episode count mismatch")
    expected_global = 0
    for episode_index, parquet in enumerate(parquets):
        table = pq.read_table(parquet)
        length = table.num_rows
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), episode_index)
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(length))
        np.testing.assert_array_equal(
            table["index"].to_numpy(), np.arange(expected_global, expected_global + length)
        )
        expected_global += length
    if expected_global != info["total_frames"]:
        raise ValueError("frame count mismatch")
    return "ok"


class DatasetEditorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.datasets = self.root / "datasets"
        self.assets = self.root / "assets"
        self.datasets.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def editor(self, *, staging=basic_validate, installed=None, assert_idle=None) -> DatasetEditor:
        if installed is None:
            installed = lambda dataset_id: basic_validate(self.datasets / dataset_id)
        return DatasetEditor(
            dataset_root=self.datasets,
            assets_base_dir=self.assets,
            validate_staging=staging,
            validate_installed=installed,
            assert_idle=assert_idle,
        )

    def test_merge_and_delete_reindex_without_changing_frame_payloads(self):
        target = make_dataset(self.datasets, "target", [2, 3])
        make_dataset(self.datasets, "source", [1, 2])
        original_state = pq.read_table(
            target / DATA_PATH.format(episode_chunk=0, episode_index=0),
            columns=["observation.state", "action"],
        )

        result = self.editor().merge_existing("target", "source")
        self.assertEqual(result["episodes"], 4)
        merged_info = json.loads((target / "meta" / "info.json").read_text())
        self.assertEqual((merged_info["total_episodes"], merged_info["total_frames"]), (4, 8))
        merged_first = pq.read_table(
            target / DATA_PATH.format(episode_chunk=0, episode_index=0),
            columns=["observation.state", "action"],
        )
        self.assertTrue(original_state.equals(merged_first))

        self.editor().delete_episodes("target", [1, 2])
        basic_validate(target)
        remaining = json.loads((target / "meta" / "info.json").read_text())
        self.assertEqual((remaining["total_episodes"], remaining["total_frames"]), (2, 4))

    def test_uploaded_dataset_can_merge_into_existing_dataset(self):
        target = make_dataset(self.datasets, "target", [2])
        staging_root = self.root / "staging"
        staging_root.mkdir()
        extracted = make_dataset(staging_root, "incoming", [3])

        result = self.editor().install_upload(
            "target", extracted, overwrite=False, merge=True
        )
        self.assertEqual(result["operation"], "merge")
        self.assertEqual((result["episodes"], result["frames"]), (2, 5))
        self.assertFalse(extracted.exists())
        basic_validate(target)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.editor().install_upload(
                "target", target, overwrite=True, merge=True
            )

    def test_image_media_details_and_frame_lookup(self):
        target = make_dataset(self.datasets, "target", [3])
        info_path = target / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["features"]["image"] = {
            "dtype": "image",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
        }
        info_path.write_text(json.dumps(info), encoding="utf-8")

        parquet_path = target / DATA_PATH.format(episode_chunk=0, episode_index=0)
        table = pq.read_table(parquet_path)
        image_values = pa.array(
            [
                {
                    "bytes": None,
                    "path": "custom/frame-one.jpg" if frame_index == 1 else f"frame_{frame_index:06d}.png",
                }
                for frame_index in range(table.num_rows)
            ],
            type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        )
        pq.write_table(table.append_column("image", image_values), parquet_path)

        for frame_index in range(table.num_rows):
            if frame_index == 1:
                continue
            frame_path = target / "images" / "image" / "episode_000000" / f"frame_{frame_index:06d}.png"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame_path.write_bytes(b"synthetic-png")
        custom_frame = target / "images" / "custom" / "frame-one.jpg"
        custom_frame.parent.mkdir(parents=True, exist_ok=True)
        custom_frame.write_bytes(b"synthetic-jpeg")

        editor = self.editor()
        details = editor.details("target")
        episode = details["episodes"][0]
        self.assertEqual(episode["image_keys"], ["image"])
        self.assertEqual(episode["video_keys"], [])
        self.assertEqual(
            episode["media"],
            [{"key": "image", "type": "image", "frames": 3, "fps": 20}],
        )
        self.assertEqual(
            editor.image_path("target", 0, "image", 1),
            custom_frame,
        )
        with self.assertRaisesRegex(ValueError, "unknown image key"):
            editor.image_path("target", 0, "wrist_image", 0)
        with self.assertRaisesRegex(ValueError, "frame index"):
            editor.image_path("target", 0, "image", 3)

        editor.update_episode("target", 0, {"metadata": {"reviewed": True}})
        for frame_index in range(3):
            rebuilt = editor.image_path("target", 0, "image", frame_index)
            self.assertTrue(rebuilt.is_file())
            self.assertEqual(
                rebuilt.read_bytes(),
                b"synthetic-jpeg" if frame_index == 1 else b"synthetic-png",
            )

    def test_embedded_image_bytes_are_served_without_external_image_directory(self):
        target = make_dataset(self.datasets, "target", [1])
        info_path = target / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["features"]["image"] = {"dtype": "image", "shape": [1, 1, 3]}
        info_path.write_text(json.dumps(info), encoding="utf-8")
        parquet_path = target / DATA_PATH.format(episode_chunk=0, episode_index=0)
        table = pq.read_table(parquet_path)
        image = pa.array(
            [{"bytes": b"\x89PNG\r\n\x1a\nsynthetic", "path": "frame_000000.png"}],
            type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        )
        pq.write_table(table.append_column("image", image), parquet_path)

        editor = self.editor()
        source, mimetype = editor.image_source("target", 0, "image", 0)
        self.assertIsInstance(source, io.BytesIO)
        self.assertEqual(source.getvalue(), b"\x89PNG\r\n\x1a\nsynthetic")
        self.assertEqual(mimetype, "image/png")

        editor.update_episode("target", 0, {"metadata": {"embedded": True}})
        source, mimetype = editor.image_source("target", 0, "image", 0)
        self.assertIsInstance(source, io.BytesIO)
        self.assertEqual(source.getvalue(), b"\x89PNG\r\n\x1a\nsynthetic")
        self.assertEqual(mimetype, "image/png")

    def test_episode_metadata_supports_nested_json_and_invalidates_norm(self):
        target = make_dataset(self.datasets, "target", [2])
        norm = self.assets / "pi05_piper_single_arm_lora" / "target" / "norm_stats.json"
        norm.parent.mkdir(parents=True)
        norm.write_text("{}", encoding="utf-8")

        result = self.editor().update_episode(
            "target",
            0,
            {
                "instruction": "new instruction",
                "task_name": "new_task",
                "success": False,
                "metadata": {"operator": "sunny", "nested": {"attempt": 2}, "scores": [1, 2, 3]},
            },
        )
        self.assertFalse(norm.exists())
        self.assertTrue(result["norm_stats_invalidated"])
        self.assertTrue(Path(result["norm_stats_invalidated"]).is_file())
        details = self.editor().details("target")
        episode = details["episodes"][0]
        self.assertEqual(episode["instruction"], "new instruction")
        self.assertEqual(episode["task_name"], "new_task")
        self.assertIs(episode["success"], False)
        self.assertEqual(episode["parameters"]["nested"], {"attempt": 2})
        with np.load(target / "raw" / "episode_000000.npz", allow_pickle=False) as raw:
            self.assertEqual(raw["meta.nested"].item(), '{"attempt":2}')
            np.testing.assert_array_equal(raw["meta.scores"], [1, 2, 3])

        self.editor().update_episode(
            "target", 0, {"instruction": "new instruction", "task_name": None, "success": None}
        )
        episode_row = json.loads((target / "meta" / "episodes.jsonl").read_text().splitlines()[0])
        self.assertNotIn("task_name", episode_row)
        self.assertNotIn("success", episode_row)
        details = self.editor().details("target")["episodes"][0]
        self.assertIsNone(details["task_name"])
        self.assertIsNone(details["success"])

    def test_rename_moves_norm_stats_and_delete_removes_dataset_only(self):
        make_dataset(self.datasets, "target", [2])
        norm_dir = self.assets / "pi05_piper_single_arm_lora" / "target"
        norm_dir.mkdir(parents=True)
        (norm_dir / "norm_stats.json").write_text("{}", encoding="utf-8")

        result = self.editor().rename_dataset("target", "renamed")
        self.assertEqual(result["dataset_id"], "renamed")
        self.assertFalse((self.datasets / "target").exists())
        self.assertTrue((self.datasets / "renamed").is_dir())
        self.assertTrue(
            (self.assets / "pi05_piper_single_arm_lora" / "renamed" / "norm_stats.json").is_file()
        )

        result = self.editor().delete_dataset("renamed")
        self.assertTrue(result["dataset_deleted"])
        self.assertTrue(result["norm_stats_deleted"])
        self.assertFalse((self.datasets / "renamed").exists())
        self.assertFalse((self.assets / "pi05_piper_single_arm_lora" / "renamed").exists())

    def test_rename_conflict_and_loader_failure_leave_original_untouched(self):
        target = make_dataset(self.datasets, "target", [2])
        make_dataset(self.datasets, "existing", [1])
        before = snapshot(target)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.editor().rename_dataset("target", "existing")
        self.assertEqual(snapshot(target), before)

        def reject_renamed(dataset_id: str) -> str:
            if dataset_id == "renamed":
                raise RuntimeError("synthetic rename validation failure")
            return basic_validate(self.datasets / dataset_id)

        with self.assertRaisesRegex(RuntimeError, "synthetic rename validation failure"):
            self.editor(installed=reject_renamed).rename_dataset("target", "renamed")
        self.assertEqual(snapshot(target), before)
        self.assertFalse((self.datasets / "renamed").exists())

    def test_active_task_blocks_dataset_rename_and_delete(self):
        target = make_dataset(self.datasets, "target", [2])
        before = snapshot(target)

        def busy(_dataset_id: str) -> None:
            raise RuntimeError("dataset is busy")

        editor = self.editor(assert_idle=busy)
        with self.assertRaisesRegex(RuntimeError, "dataset is busy"):
            editor.rename_dataset("target", "renamed")
        with self.assertRaisesRegex(RuntimeError, "dataset is busy"):
            editor.delete_dataset("target")
        self.assertEqual(snapshot(target), before)

    def test_incompatible_merge_is_rejected_without_modifying_target(self):
        target = make_dataset(self.datasets, "target", [2])
        make_dataset(self.datasets, "source", [2], robot_type="other")
        before = snapshot(target)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.editor().merge_existing("target", "source")
        self.assertEqual(snapshot(target), before)
        self.assertEqual(list(self.datasets.glob(".target.editing-*")), [])

    def test_structural_validation_failure_removes_candidate(self):
        target = make_dataset(self.datasets, "target", [2])
        before = snapshot(target)

        def reject_candidate(path: Path) -> str:
            if ".editing-" in path.name:
                raise ValueError("synthetic structural failure")
            return basic_validate(path)

        with self.assertRaisesRegex(ValueError, "synthetic structural failure"):
            self.editor(staging=reject_candidate).update_episode(
                "target", 0, {"metadata": {"note": "should not commit"}}
            )
        self.assertEqual(snapshot(target), before)
        self.assertEqual(list(self.datasets.glob(".target.editing-*")), [])

    def test_loader_failure_restores_original_dataset(self):
        target = make_dataset(self.datasets, "target", [2])
        before = snapshot(target)
        norm = self.assets / "pi05_piper_single_arm_lora" / "target" / "norm_stats.json"
        norm.parent.mkdir(parents=True)
        norm.write_text("{}", encoding="utf-8")

        def reject_installed(_dataset_id: str) -> str:
            raise RuntimeError("synthetic loader failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic loader failure"):
            self.editor(installed=reject_installed).update_episode(
                "target", 0, {"metadata": {"note": "rollback"}}
            )
        self.assertEqual(snapshot(target), before)
        self.assertTrue(norm.is_file())
        self.assertEqual(list(self.datasets.glob(".target.editing-*")), [])
        self.assertEqual(len(list(self.datasets.glob(".target.failed-*"))), 1)

    def test_active_task_blocks_episode_edit_before_rebuild(self):
        target = make_dataset(self.datasets, "target", [2])
        before = snapshot(target)

        def busy(_dataset_id: str) -> None:
            raise RuntimeError("dataset is busy")

        with self.assertRaisesRegex(RuntimeError, "dataset is busy"):
            self.editor(assert_idle=busy).update_episode(
                "target", 0, {"metadata": {"note": "blocked"}}
            )
        self.assertEqual(snapshot(target), before)
        self.assertEqual(list(self.datasets.glob(".target.editing-*")), [])


if __name__ == "__main__":
    unittest.main()
