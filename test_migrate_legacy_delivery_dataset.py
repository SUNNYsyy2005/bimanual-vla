from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from check_pi05_dataset import check_dataset
from migrate_legacy_delivery_dataset import (
    MigrationError,
    decode_legacy_delivery_episode,
    main,
    migrate_dataset,
)
from piper_action_conventions import matrix_to_rotation6d, rotation6d_to_matrix
from piper_data_contract import build_legacy_delivery_step_actions


DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def legacy_states(frames: int, episode_index: int = 0) -> np.ndarray:
    values = np.zeros((frames, 10), dtype=np.float32)
    for frame in range(frames):
        angle = 0.05 * (episode_index + frame)
        rotation = Rotation.from_euler("z", angle).as_matrix()
        values[frame, :3] = [0.1 * episode_index + 0.01 * frame, -0.02 * frame, 0.2]
        values[frame, 3:9] = matrix_to_rotation6d(rotation)
        values[frame, 9] = 0.2 + 0.1 * frame  # legacy closed fraction
    return values


def make_legacy_dataset(
    root: Path,
    name: str = "legacy",
    *,
    episode_lengths: tuple[int, ...] = (3, 2),
    media_dtype: str = "image",
    corrupt_action: bool = False,
) -> Path:
    dataset = root / name
    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper",
        "fps": 20,
        "chunks_size": 1000,
        "features": {
            "image": {
                "dtype": media_dtype,
                "shape": [8, 8, 3] if media_dtype == "image" else [3, 8, 8],
                "names": ["height", "width", "channel"] if media_dtype == "image" else ["channels", "height", "width"],
            },
            "wrist_image": {
                "dtype": media_dtype,
                "shape": [8, 8, 3] if media_dtype == "image" else [3, 8, 8],
                "names": ["height", "width", "channel"] if media_dtype == "image" else ["channels", "height", "width"],
            },
            "state": {
                "dtype": "float32",
                "shape": [10],
                "names": [
                    "eef_x_base_m", "eef_y_base_m", "eef_z_base_m",
                    "rotation6d_col0_x", "rotation6d_col0_y", "rotation6d_col0_z",
                    "rotation6d_col1_x", "rotation6d_col1_y", "rotation6d_col1_z",
                    "gripper_closed_fraction",
                ],
            },
            "actions": {
                "dtype": "float32",
                "shape": [7],
                "names": [
                    "delta_x_base_m", "delta_y_base_m", "delta_z_base_m",
                    "delta_rx_base_rad", "delta_ry_base_rad", "delta_rz_base_rad",
                    "gripper_target_closed_fraction",
                ],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH if media_dtype == "video" else None,
        "total_episodes": len(episode_lengths),
        "total_frames": sum(episode_lengths),
        "total_tasks": 1,
        "total_videos": len(episode_lengths) * 2 if media_dtype == "video" else 0,
        "total_chunks": 1,
        "splits": {"train": f"0:{len(episode_lengths)}"},
    }
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    write_jsonl(dataset / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "pick cube"}])
    write_jsonl(
        dataset / "meta" / "episodes.jsonl",
        [
            {"episode_index": index, "tasks": ["pick cube"], "length": length}
            for index, length in enumerate(episode_lengths)
        ],
    )

    stats_rows = []
    global_index = 0
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    for episode_index, length in enumerate(episode_lengths):
        states = legacy_states(length, episode_index)
        actions = build_legacy_delivery_step_actions(states)
        if corrupt_action and episode_index == 0:
            actions[0, 0] += 0.2
        columns: dict[str, object] = {
            "state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 10)),
            "actions": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 7)),
            "timestamp": np.arange(length, dtype=np.float32) / 20.0,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(global_index, global_index + length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
        }
        if media_dtype == "image":
            columns["image"] = pa.array(
                [{"bytes": b"high-" + bytes([frame]), "path": None} for frame in range(length)],
                type=image_type,
            )
            columns["wrist_image"] = pa.array(
                [{"bytes": b"wrist-" + bytes([frame]), "path": None} for frame in range(length)],
                type=image_type,
            )
        parquet = dataset / DATA_PATH.format(episode_chunk=0, episode_index=episode_index)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), parquet)
        stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "index": {
                        "min": [global_index],
                        "max": [global_index + length - 1],
                        "mean": [999.0],
                        "std": [0.0],
                        "count": [length],
                    },
                    "image": {"count": [min(100, length)]},
                    "wrist_image": {"count": [min(100, length)]},
                },
            }
        )
        if media_dtype == "video":
            for key in ("image", "wrist_image"):
                video = dataset / VIDEO_PATH.format(
                    episode_chunk=0, video_key=key, episode_index=episode_index
                )
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(f"{key}-{episode_index}".encode())
        global_index += length
    write_jsonl(dataset / "meta" / "episodes_stats.jsonl", stats_rows)
    raw = dataset / "raw/episode_000000.npz"
    raw.parent.mkdir(parents=True)
    np.savez_compressed(raw, state=legacy_states(3), actions=np.zeros((3, 7), np.float32))
    return dataset


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class LegacyDeliveryMigrationTest(unittest.TestCase):
    def test_decode_math_uses_current_state_left_rotation_and_opening_fraction(self):
        states = np.zeros((2, 10), dtype=np.float32)
        current_rotation = Rotation.from_euler("x", 0.2).as_matrix()
        states[:, 3:9] = matrix_to_rotation6d(current_rotation)
        states[:, 9] = [0.8, 0.7]
        actions = np.zeros((2, 7), dtype=np.float32)
        actions[0, :3] = [0.1, -0.2, 0.3]
        actions[0, 3:6] = [0.0, 0.0, 0.4]
        actions[0, 6] = 0.25
        actions[1, 6] = 0.7

        converted_states, targets = decode_legacy_delivery_episode(
            states, actions, arm_count=1, verify_next_measured=False
        )

        np.testing.assert_allclose(converted_states[:, 9], [0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(targets[0, :3], [0.1, -0.2, 0.3], atol=1e-6)
        expected_rotation = Rotation.from_rotvec([0.0, 0.0, 0.4]).as_matrix() @ current_rotation
        np.testing.assert_allclose(
            rotation6d_to_matrix(targets[0, 3:9]), expected_rotation, atol=1e-6
        )
        self.assertAlmostEqual(float(targets[0, 9]), 0.75)
        np.testing.assert_allclose(targets[-1], converted_states[-1], atol=1e-6)

    def test_bimanual_20d_14d_decode_keeps_arm_order_and_dimensions(self):
        left = legacy_states(3, 0)
        right = legacy_states(3, 1)
        states = np.concatenate([left, right], axis=1)
        actions = build_legacy_delivery_step_actions(states, arm_count=2)
        converted_states, targets = decode_legacy_delivery_episode(
            states, actions, arm_count=2
        )
        self.assertEqual(converted_states.shape, (3, 20))
        self.assertEqual(targets.shape, (3, 20))
        np.testing.assert_allclose(targets[:-1], converted_states[1:], atol=1e-5)
        np.testing.assert_allclose(targets[-1], converted_states[-1], atol=1e-6)
        np.testing.assert_allclose(converted_states[:, 9], 1.0 - left[:, 9])
        np.testing.assert_allclose(converted_states[:, 19], 1.0 - right[:, 9])

    def test_migration_preserves_source_images_tasks_and_recomputes_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_legacy_dataset(root)
            before = snapshot(source)
            target = root / "canonical"

            migrated = migrate_dataset(source, target, verify=True)

            self.assertEqual(migrated, target)
            self.assertEqual(snapshot(source), before)
            self.assertFalse((target / "raw").exists())
            info = json.loads((target / "meta/info.json").read_text())
            self.assertEqual(info["contract_version"], 3)
            self.assertEqual(info["contract_format"], "canonical")
            self.assertEqual((info["state_dim"], info["raw_action_dim"], info["model_action_dim"]), (10, 10, 7))
            self.assertEqual(info["action_source"], "next_measured_eef_fallback")
            self.assertEqual(info["action_alignment"], "next_observation")
            self.assertEqual(info["action_offset"], 1)
            self.assertEqual(info["gripper_semantics"], "absolute_opening_fraction_0_closed_1_open")
            self.assertEqual(info["source_dataset"], source.name)
            self.assertEqual(info["source_contract"]["legacy_format"], "legacy_v2")
            self.assertIn("not actual commanded actions", info["migration_note"])

            parquet_path = target / DATA_PATH.format(episode_chunk=0, episode_index=0)
            table = pq.read_table(parquet_path)
            hf_metadata = json.loads(
                pq.ParquetFile(parquet_path).schema_arrow.metadata[b"huggingface"]
            )
            hf_features = hf_metadata["info"]["features"]
            self.assertEqual(hf_features["observation.images.cam_high"], {"_type": "Image"})
            self.assertEqual(hf_features["action"]["length"], 10)
            self.assertEqual(hf_features["state_timestamp"]["dtype"], "double")
            self.assertIn("observation.state", table.column_names)
            self.assertIn("action", table.column_names)
            self.assertIn("observation.images.cam_high", table.column_names)
            self.assertIn("observation.images.cam_right_wrist", table.column_names)
            self.assertNotIn("state", table.column_names)
            self.assertNotIn("actions", table.column_names)
            states = np.asarray(table["observation.state"].to_pylist())
            actions = np.asarray(table["action"].to_pylist())
            np.testing.assert_allclose(actions[:-1], states[1:], atol=1e-5)
            np.testing.assert_allclose(actions[-1], states[-1], atol=1e-6)
            np.testing.assert_allclose(states[:, 9], [0.8, 0.7, 0.6], atol=1e-6)
            self.assertEqual(
                table["observation.images.cam_high"][0].as_py()["bytes"], b"high-\x00"
            )
            np.testing.assert_allclose(
                np.asarray(table["action_timestamp"]),
                np.array([0.05, 0.10, 0.15]),
            )

            tasks = (target / "meta/tasks.jsonl").read_text()
            self.assertEqual(tasks, (source / "meta/tasks.jsonl").read_text())
            episodes = [json.loads(line) for line in (target / "meta/episodes.jsonl").read_text().splitlines()]
            self.assertEqual(episodes[0]["tasks"], ["pick cube"])
            self.assertEqual(episodes[0]["raw_action_dim"], 10)
            stats = [json.loads(line) for line in (target / "meta/episodes_stats.jsonl").read_text().splitlines()]
            self.assertEqual(stats[0]["stats"]["index"]["mean"], [1.0])
            self.assertIn("observation.images.cam_high", stats[0]["stats"])
            self.assertNotIn("image_timestamp.cam_high", stats[0]["stats"])
            self.assertNotIn("image_timestamp.cam_right_wrist", stats[0]["stats"])
            self.assertEqual(check_dataset(target), [])

    def test_video_files_are_relocated_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_legacy_dataset(root, episode_lengths=(2,), media_dtype="video")
            before = snapshot(source)
            target = root / "canonical_video"
            migrate_dataset(source, target, verify=False)
            self.assertEqual(snapshot(source), before)
            high = target / VIDEO_PATH.format(
                episode_chunk=0,
                video_key="observation.images.cam_high",
                episode_index=0,
            )
            wrist = target / VIDEO_PATH.format(
                episode_chunk=0,
                video_key="observation.images.cam_right_wrist",
                episode_index=0,
            )
            self.assertEqual(high.read_bytes(), b"image-0")
            self.assertEqual(wrist.read_bytes(), b"wrist_image-0")

    def test_existing_or_nested_target_is_rejected_and_source_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_legacy_dataset(root)
            before = snapshot(source)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                migrate_dataset(source, existing)
            with self.assertRaises(MigrationError):
                migrate_dataset(source, source / "child")
            self.assertEqual(snapshot(source), before)

    def test_mismatched_delta_is_rejected_instead_of_claiming_next_measured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_legacy_dataset(root, corrupt_action=True)
            target = root / "canonical"
            with self.assertRaisesRegex(MigrationError, "not next-measured"):
                migrate_dataset(source, target)
            self.assertFalse(target.exists())

    def test_cli_source_target_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_legacy_dataset(root, episode_lengths=(2,))
            target = root / "canonical_cli"
            with patch.object(
                sys,
                "argv",
                ["migrate_legacy_delivery_dataset.py", str(source), str(target), "--verify"],
            ):
                self.assertEqual(main(), 0)
            self.assertTrue((target / "meta/info.json").is_file())


if __name__ == "__main__":
    unittest.main()
