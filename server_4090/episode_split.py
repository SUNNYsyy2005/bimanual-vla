"""Deterministic, persisted train/test splits for LeRobot episodes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from typing import Any

SPLIT_VERSION = 3
NORM_CONFIG_VERSION = 4
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42
DATASET_SPLIT_FILENAME = "train_test_split.json"
NORM_SPLIT_FILENAME = "episode_split.json"
NORM_CONFIG_FILENAME = "norm_config.json"

CONTRACT_FIELDS = (
    "contract_version",
    "raw_action_dim",
    "model_action_dim",
    "raw_action_semantics",
    "model_action_semantics",
    "raw_action_convention",
    "model_action_convention",
    "gripper_semantics",
    "raw_gripper_semantics",
    "wire_gripper_semantics",
    "action_offset",
    "model_action_start_offset",
)
_INTEGER_CONTRACT_FIELDS = frozenset(
    {"contract_version", "raw_action_dim", "model_action_dim", "action_offset", "model_action_start_offset"}
)


def normalize_contract_fingerprint(
    contract: Mapping[str, Any] | None,
) -> dict[str, int | str]:
    """Return the stable action-contract fields persisted with splits/norm stats.

    Passing ``None`` keeps old split-only callers working.  A non-empty
    fingerprint is deliberately all-or-nothing so partially-described datasets
    cannot accidentally reuse normalization statistics from another action
    representation.
    """
    if contract is None or not contract:
        return {}
    values: dict[str, int | str] = {}
    missing: list[str] = []
    for field in CONTRACT_FIELDS:
        value = contract.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
            continue
        if field in _INTEGER_CONTRACT_FIELDS:
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be a positive integer") from exc
            if field == "action_offset":
                if parsed not in {0, 1}:
                    raise ValueError("action_offset must be 0 or 1")
            elif field == "model_action_start_offset":
                if parsed != 1:
                    raise ValueError("model_action_start_offset must be 1")
            elif parsed <= 0:
                raise ValueError(f"{field} must be a positive integer")
            values[field] = parsed
        else:
            values[field] = str(value).strip()
    if missing:
        raise ValueError(
            "action contract fingerprint is incomplete; missing " + ", ".join(missing)
        )
    return values


@dataclass(frozen=True)
class EpisodeSplit:
    dataset_id: str
    test_ratio: float
    seed: int
    all_episodes: tuple[int, ...]
    train_episodes: tuple[int, ...]
    test_episodes: tuple[int, ...]
    contract_version: int | None = None
    raw_action_dim: int | None = None
    model_action_dim: int | None = None
    raw_action_semantics: str | None = None
    model_action_semantics: str | None = None
    raw_action_convention: str | None = None
    model_action_convention: str | None = None
    gripper_semantics: str | None = None
    raw_gripper_semantics: str | None = None
    wire_gripper_semantics: str | None = None
    action_offset: int | None = None
    model_action_start_offset: int | None = None

    def contract_dict(self) -> dict[str, int | str]:
        values = {field: getattr(self, field) for field in CONTRACT_FIELDS}
        present = {field: value for field, value in values.items() if value is not None}
        if not present:
            return {}
        return normalize_contract_fingerprint(present)

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
            **self.contract_dict(),
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


def _payload_contract(payload: Mapping[str, Any]) -> dict[str, int | str]:
    present = {field: payload.get(field) for field in CONTRACT_FIELDS if payload.get(field) is not None}
    if not present:
        return {}
    try:
        return normalize_contract_fingerprint(present)
    except ValueError:
        return {}


def _from_payload(payload: Any) -> EpisodeSplit | None:
    # v1 split manifests had no action contract.  They remain readable when a
    # caller doesn't request contract validation, but a contract-aware caller
    # will regenerate them as v2 before norm/train.
    if not isinstance(payload, dict) or payload.get("version") not in {1, 2, SPLIT_VERSION}:
        return None
    contract = _payload_contract(payload)
    try:
        return EpisodeSplit(
            dataset_id=str(payload["dataset_id"]),
            test_ratio=float(payload["test_ratio"]),
            seed=int(payload["seed"]),
            all_episodes=tuple(map(int, payload["all_episodes"])),
            train_episodes=tuple(map(int, payload["train_episodes"])),
            test_episodes=tuple(map(int, payload["test_episodes"])),
            **contract,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _valid_for_dataset(
    split: EpisodeSplit,
    *,
    dataset_id: str,
    episodes: tuple[int, ...],
    contract: Mapping[str, Any] | None = None,
) -> bool:
    expected_contract = normalize_contract_fingerprint(contract)
    return (
        split.dataset_id == dataset_id
        and split.all_episodes == episodes
        and 0.0 <= split.test_ratio < 1.0
        and split.seed >= 0
        and bool(split.train_episodes)
        and set(split.train_episodes).isdisjoint(split.test_episodes)
        and tuple(sorted((*split.train_episodes, *split.test_episodes))) == episodes
        and (not expected_contract or split.contract_dict() == expected_contract)
    )


def _same_definition(
    split: EpisodeSplit,
    *,
    dataset_id: str,
    episodes: tuple[int, ...],
    test_ratio: float,
    seed: int,
    contract: Mapping[str, Any] | None = None,
) -> bool:
    return (
        _valid_for_dataset(
            split,
            dataset_id=dataset_id,
            episodes=episodes,
            contract=contract,
        )
        and math.isclose(split.test_ratio, test_ratio, rel_tol=0.0, abs_tol=1e-12)
        and split.seed == seed
    )


def load_episode_split(
    dataset_root: Path,
    dataset_id: str,
    *,
    contract: Mapping[str, Any] | None = None,
) -> EpisodeSplit | None:
    """Load the persisted split when it still matches dataset and action contract."""
    dataset_path = Path(dataset_root).expanduser().resolve() / dataset_id
    episodes = _episode_indices(dataset_path)
    split_path = dataset_path / "meta" / DATASET_SPLIT_FILENAME
    try:
        split = _from_payload(json.loads(split_path.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if split is None or not _valid_for_dataset(
        split,
        dataset_id=dataset_id,
        episodes=episodes,
        contract=contract,
    ):
        return None
    return split


def resolve_episode_split(
    dataset_root: Path,
    dataset_id: str,
    *,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
    contract: Mapping[str, Any] | None = None,
) -> EpisodeSplit:
    """Load or create a deterministic episode-level split for a local dataset."""
    test_ratio = float(test_ratio)
    seed = int(seed)
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1)")
    contract_fingerprint = normalize_contract_fingerprint(contract)

    dataset_path = Path(dataset_root).expanduser().resolve() / dataset_id
    episodes = _episode_indices(dataset_path)
    split_path = dataset_path / "meta" / DATASET_SPLIT_FILENAME
    try:
        existing = _from_payload(json.loads(split_path.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        existing = None
    if existing is not None and _same_definition(
        existing,
        dataset_id=dataset_id,
        episodes=episodes,
        test_ratio=test_ratio,
        seed=seed,
        contract=contract_fingerprint,
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
        **contract_fingerprint,
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
    contract: Mapping[str, Any] | None = None,
    delivery_action_convention: str | None = None,
) -> Path:
    """Persist norm provenance and the exact model/raw action contract."""
    fingerprint = normalize_contract_fingerprint(contract) if contract is not None else split.contract_dict()
    split_contract = split.contract_dict()
    if split_contract and fingerprint != split_contract:
        raise ValueError("norm contract does not match the persisted episode split contract")
    if not fingerprint and delivery_action_convention is not None:
        # Compatibility with the intermediate convention-only implementation.
        # Such configs are intentionally not considered contract-complete.
        fingerprint = {}
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
        **fingerprint,
        # Keep the old key for tools/checkpoints created during the migration.
        "delivery_action_convention": (
            fingerprint.get("model_action_convention", delivery_action_convention)
            if schema == "delivery"
            else None
        ),
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


def norm_split_matches(
    norm_stats_dir: Path,
    split: EpisodeSplit,
    *,
    contract: Mapping[str, Any] | None = None,
    delivery_action_convention: str | None = None,
) -> bool:
    """Return whether norm stats match split and the complete action contract."""
    norm_stats_dir = Path(norm_stats_dir)
    if not (norm_stats_dir / "norm_stats.json").is_file():
        return False
    try:
        saved = _from_payload(
            json.loads((norm_stats_dir / NORM_SPLIT_FILENAME).read_text(encoding="utf-8"))
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if saved is None or saved.as_dict() != split.as_dict():
        return False

    expected = normalize_contract_fingerprint(contract) if contract is not None else split.contract_dict()
    if not expected and delivery_action_convention is None:
        return True
    try:
        norm_config = json.loads(
            (norm_stats_dir / NORM_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(norm_config, dict) or norm_config.get("version") != NORM_CONFIG_VERSION:
        return False
    if expected:
        return all(norm_config.get(field) == value for field, value in expected.items())
    return norm_config.get("delivery_action_convention") == delivery_action_convention
