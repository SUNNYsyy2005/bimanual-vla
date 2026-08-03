"""Deterministic, persisted train/test splits for LeRobot episodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from typing import Any

SPLIT_VERSION = 1
NORM_CONFIG_VERSION = 1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42
DATASET_SPLIT_FILENAME = "train_test_split.json"
NORM_SPLIT_FILENAME = "episode_split.json"
NORM_CONFIG_FILENAME = "norm_config.json"


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


def _valid_for_dataset(split: EpisodeSplit, *, dataset_id: str, episodes: tuple[int, ...]) -> bool:
    return (
        split.dataset_id == dataset_id
        and split.all_episodes == episodes
        and 0.0 <= split.test_ratio < 1.0
        and split.seed >= 0
        and bool(split.train_episodes)
        and set(split.train_episodes).isdisjoint(split.test_episodes)
        and tuple(sorted((*split.train_episodes, *split.test_episodes))) == episodes
    )


def _same_definition(split: EpisodeSplit, *, dataset_id: str, episodes: tuple[int, ...], test_ratio: float, seed: int) -> bool:
    return (
        _valid_for_dataset(split, dataset_id=dataset_id, episodes=episodes)
        and math.isclose(split.test_ratio, test_ratio, rel_tol=0.0, abs_tol=1e-12)
        and split.seed == seed
    )


def load_episode_split(dataset_root: Path, dataset_id: str) -> EpisodeSplit | None:
    """Load the persisted split when it still matches the current dataset."""
    dataset_path = Path(dataset_root).expanduser().resolve() / dataset_id
    episodes = _episode_indices(dataset_path)
    split_path = dataset_path / "meta" / DATASET_SPLIT_FILENAME
    try:
        split = _from_payload(json.loads(split_path.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if split is None or not _valid_for_dataset(split, dataset_id=dataset_id, episodes=episodes):
        return None
    return split


def resolve_episode_split(
    dataset_root: Path,
    dataset_id: str,
    *,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
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


def write_norm_config(
    norm_stats_dir: Path,
    split: EpisodeSplit,
    *,
    model_variant: str,
    base_checkpoint: str,
    arm_mode: str,
    arm_side: str,
    schema: str,
    requested_batch_size: int,
    effective_batch_size: int,
    num_workers: int,
    max_frames: int | None,
    available_train_frames: int,
    processed_batches: int,
) -> Path:
    """Persist norm provenance separately from the split compatibility manifest."""
    path = Path(norm_stats_dir) / NORM_CONFIG_FILENAME
    payload = {
        "version": NORM_CONFIG_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": split.dataset_id,
        "model_variant": model_variant,
        "base_checkpoint": base_checkpoint,
        "arm_mode": arm_mode,
        "arm_side": arm_side,
        "schema": schema,
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": effective_batch_size,
        "num_workers": num_workers,
        "max_frames": max_frames,
        "available_train_frames": available_train_frames,
        "processed_batches": processed_batches,
        "test_ratio": split.test_ratio,
        "split_seed": split.seed,
        "train_episodes": list(split.train_episodes),
        "test_episodes": list(split.test_episodes),
    }
    _atomic_json(path, payload)
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
