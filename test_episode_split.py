import json
from pathlib import Path
import tempfile
import unittest

from server_4090.episode_split import norm_split_matches, resolve_episode_split, write_norm_split


class EpisodeSplitTest(unittest.TestCase):
    def make_dataset(self, root: Path, count: int, dataset_id: str = "demo") -> Path:
        dataset = root / dataset_id
        meta = dataset / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps({"total_episodes": count}), encoding="utf-8")
        rows = "\n".join(json.dumps({"episode_index": index}) for index in range(count)) + "\n"
        (meta / "episodes.jsonl").write_text(rows, encoding="utf-8")
        return dataset

    def test_deterministic_episode_level_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.make_dataset(root, 20)
            split = resolve_episode_split(root, "demo", test_ratio=0.1, seed=42)
            self.assertEqual(len(split.train_episodes), 18)
            self.assertEqual(len(split.test_episodes), 2)
            self.assertTrue(set(split.train_episodes).isdisjoint(split.test_episodes))
            self.assertEqual(set(split.all_episodes), set(range(20)))
            persisted = resolve_episode_split(root, "demo", test_ratio=0.1, seed=42)
            self.assertEqual(split, persisted)
            self.assertTrue((dataset / "meta" / "train_test_split.json").is_file())

    def test_changed_seed_or_episode_count_regenerates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.make_dataset(root, 12)
            first = resolve_episode_split(root, "demo", test_ratio=0.25, seed=1)
            second = resolve_episode_split(root, "demo", test_ratio=0.25, seed=2)
            self.assertNotEqual(first.test_episodes, second.test_episodes)
            with (dataset / "meta" / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"episode_index": 12}) + "\n")
            third = resolve_episode_split(root, "demo", test_ratio=0.25, seed=2)
            self.assertEqual(third.all_episodes, tuple(range(13)))
            self.assertEqual(len(third.train_episodes) + len(third.test_episodes), 13)

    def test_single_episode_remains_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root, 1)
            split = resolve_episode_split(root, "demo", test_ratio=0.5, seed=42)
            self.assertEqual(split.train_episodes, (0,))
            self.assertEqual(split.test_episodes, ())

    def test_norm_manifest_must_match_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root, 10)
            split = resolve_episode_split(root, "demo", test_ratio=0.2, seed=42)
            norm_dir = root / "assets" / "demo"
            norm_dir.mkdir(parents=True)
            (norm_dir / "norm_stats.json").write_text("{}", encoding="utf-8")
            self.assertFalse(norm_split_matches(norm_dir, split))
            write_norm_split(norm_dir, split)
            self.assertTrue(norm_split_matches(norm_dir, split))

    def test_invalid_ratio_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root, 2)
            with self.assertRaisesRegex(ValueError, "test_ratio"):
                resolve_episode_split(root, "demo", test_ratio=1.0, seed=42)


if __name__ == "__main__":
    unittest.main()
