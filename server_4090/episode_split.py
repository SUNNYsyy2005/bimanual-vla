"""Deterministic, persisted train/test splits for LeRobot episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Any

SPLIT_VERSION = 1
DATASET_SPLIT_FILENAME = "train_test_split.json"
NORM_SPLIT_FILENAME = "episode_split.json"


@dataclass(frozen=True)
class EpisodeSplit:
    dataset_id: str
    test_ratio: float
    seed: int
    all_episodes: tuple[int, ...]
    train_episodes: tuple[int, ...]
    test_episodes: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": SPLIT_VERSION,
            "dataset_id": self.dataset_id,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "all_episodes": list(self.all_episodes),
            "train_episodes": list(self.train_episodes),
            "test_episodes": list(self.test_episodes),
            "num_episodes": len(self.all_episodes),
            "num_train_episodes": len(self.train_episodes),
            "num_test_episodes": len(self.test_episodes),
        }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _episode_indices(dataset_path: Path) -> tuple[int, ...]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    indexes: list[int] = []
    if episodes_path.is_file():
        for line_number, line in enumerate(episodes_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                indexes.append(int(row["episode_index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid episode metadata at {episodes_path}:{line_number}") from exc
    else:
        info_path = dataset_path / "meta" / "info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            indexes = list(range(int(info["total_episodes"])))
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot determine episodes for dataset {dataset_path.name!r}") from exc

    if not indexes:
        raise ValueError(f"dataset {dataset_path.name!r} has no episodes")
    if any(index < 0 for index in indexes) or len(set(indexes)) != len(indexes):
        raise ValueError(f"dataset {dataset_path.name!r} has invalid or duplicate episode indexes")
    return tuple(sorted(indexes))


def _from_payload(payload: Any) -> EpisodeSplit | None:
    if not isinstance(payload, dict) or payload.get("version") != SPLIT_VERSION:
        return None
    try:
        return EpisodeSplit(
            dataset_id=str(payload["dataset_id"]),
            test_ratio=float(payload["test_ratio"]),
            seed=int(payload["seed"]),
            all_episodes=tuple(map(int, payload["all_episodes"])),
            train_episodes=tuple(map(int, payload["train_episodes"])),
            test_episodes=tuple(map(int, payload["test_episodes"])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _same_definition(split: EpisodeSplit, *, dataset_id: str, episodes: tuple[int, ...], test_ratio: float, seed: int) -> bool:
    return (
        split.dataset_id == dataset_id
        and split.all_episodes == episodes
        and math.isclose(split.test_ratio, test_ratio, rel_tol=0.0, abs_tol=1e-12)
        and split.seed == seed
        and set(split.train_episodes).isdisjoint(split.test_episodes)
        and tuple(sorted((*split.train_episodes, *split.test_episodes))) == episodes
    )


def resolve_episode_split(
    dataset_root: Path,
    dataset_id: str,
    *,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> EpisodeSplit:
    """Load or create a deterministic episode-level split for a local dataset."""
    test_ratio = float(test_ratio)
    seed = int(seed)
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1)")

    dataset_path = Path(dataset_root).expanduser().resolve() / dataset_id
    episodes = _episode_indices(dataset_path)
    split_path = dataset_path / "meta" / DATASET_SPLIT_FILENAME
    try:
        existing = _from_payload(json.loads(split_path.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError):
        existing = None
    if existing is not None and _same_definition(
        existing,
        dataset_id=dataset_id,
        episodes=episodes,
        test_ratio=test_ratio,
        seed=seed,
    ):
        return existing

    if len(episodes) < 2 or test_ratio <= 0:
        test_count = 0
    else:
        test_count = min(len(episodes) - 1, max(1, int(round(len(episodes) * test_ratio))))
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    test_set = set(shuffled[:test_count])
    split = EpisodeSplit(
        dataset_id=dataset_id,
        test_ratio=test_ratio,
        seed=seed,
        all_episodes=episodes,
        train_episodes=tuple(index for index in episodes if index not in test_set),
        test_episodes=tuple(index for index in episodes if index in test_set),
    )
    _atomic_json(split_path, split.as_dict())
    return split


def write_norm_split(norm_stats_dir: Path, split: EpisodeSplit) -> Path:
    path = Path(norm_stats_dir) / NORM_SPLIT_FILENAME
    _atomic_json(path, split.as_dict())
    return path


def norm_split_matches(norm_stats_dir: Path, split: EpisodeSplit) -> bool:
    norm_stats_dir = Path(norm_stats_dir)
    if not (norm_stats_dir / "norm_stats.json").is_file():
        return False
    try:
        saved = _from_payload(json.loads((norm_stats_dir / NORM_SPLIT_FILENAME).read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return saved is not None and saved.as_dict() == split.as_dict()
