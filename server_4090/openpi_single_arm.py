#!/usr/bin/env python3
"""Single-arm and bimanual Piper π0/π0.5 training, norm-stat, and serving entrypoint.

Run this file with the Python environment of an OpenPI checkout and use the
checkout as the current working directory. It intentionally builds the config
in Python so an uploaded LeRobot repo_id can be selected without editing the
upstream OpenPI config registry.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
from websockets.exceptions import ConnectionClosedError, InvalidMessage

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config as training_config
from openpi.training import data_loader
from openpi.training import weight_loaders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import piper_action_conventions as _piper_action_conventions
except ImportError:  # Keep the OpenPI helper usable while collection code is staged separately.
    _piper_action_conventions = None


def _convention_value(name: str, default: str) -> str:
    return str(getattr(_piper_action_conventions, name, default))


DELIVERY_STEP_ACTION_CONVENTION = _convention_value(
    "DELIVERY_STEP_ACTION_CONVENTION", "step"
)
DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION = _convention_value(
    "DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION", "chunk_origin"
)
DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION = _convention_value(
    "DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION", "absolute_eef_target"
)
DELIVERY_ACTION_CONVENTIONS = frozenset(
    getattr(
        _piper_action_conventions,
        "DELIVERY_ACTION_CONVENTIONS",
        {DELIVERY_STEP_ACTION_CONVENTION, DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION},
    )
)
DELIVERY_LEGACY_STEP_ACTION_SEMANTICS = _convention_value(
    "DELIVERY_STEP_ACTION_SEMANTICS",
    "eef_delta_base_xyz_left_rotvec_gripper_target",
)
DELIVERY_ABSOLUTE_EEF_ACTION_SEMANTICS = _convention_value(
    "DELIVERY_RAW_ACTION_SEMANTICS", "absolute_eef_target"
)
DELIVERY_LEGACY_CHUNK_ORIGIN_ACTION_SEMANTICS = _convention_value(
    "DELIVERY_CHUNK_ORIGIN_ACTION_SEMANTICS",
    "eef_delta_chunk_origin_base_xyz_left_rotvec_gripper_target",
)
DELIVERY_MODEL_ACTION_SEMANTICS = _convention_value(
    "DELIVERY_MODEL_ACTION_SEMANTICS",
    "eef_delta_chunk_origin_base_xyz_left_rotvec_gripper_opening_target",
)
JOINT_RAW_ACTION_SEMANTICS = _convention_value(
    "JOINT_ACTION_SEMANTICS", "absolute_joint_position_opening_fraction"
)
JOINT_MODEL_ACTION_SEMANTICS = _convention_value(
    "JOINT_MODEL_ACTION_SEMANTICS",
    "joint_delta_chunk_origin_first_6_absolute_gripper_target",
)
JOINT_RAW_ACTION_CONVENTION = _convention_value(
    "JOINT_RAW_ACTION_CONVENTION", "absolute_joint_target"
)
JOINT_MODEL_ACTION_CONVENTION = _convention_value(
    "JOINT_MODEL_ACTION_CONVENTION", "chunk_origin"
)
GRIPPER_OPENING_FRACTION = _convention_value(
    "NEW_GRIPPER_SEMANTICS", "absolute_opening_fraction_0_closed_1_open"
)
GRIPPER_CLOSED_FRACTION = _convention_value(
    "LEGACY_GRIPPER_SEMANTICS", "absolute_closed_fraction_0_open_1_closed"
)
GRIPPER_OPENING_METERS = _convention_value(
    "LEGACY_GRIPPER_OPENING_METRES_SEMANTICS", "absolute_opening_metres"
)
PIPER_GRIPPER_MAX_M = float(
    getattr(_piper_action_conventions, "PIPER_GRIPPER_MAX_M", 0.07)
)
CURRENT_CONTRACT_VERSION = int(
    getattr(_piper_action_conventions, "CONTRACT_VERSION", 3)
)
LEGACY_DELIVERY_CONTRACT_VERSION = 2
# The asynchronous client keeps consuming the old chunk while inference runs.
# A returned horizon shorter than this cannot cover launch-to-arrival jitter and
# is therefore rejected by the Dashboard execution gate.
MIN_EXECUTION_ACTION_HORIZON = 16
DEFAULT_ASYNC_INFERENCE_LAUNCH_HZ = 4.0
MODEL_ACTION_START_OFFSET_STEPS = 1
ABSOLUTE_EEF_TARGETS_TO_CHUNK_ORIGIN = getattr(
    _piper_action_conventions, "absolute_eef_targets_to_chunk_origin", None
)
STEP_DELTAS_TO_CHUNK_ORIGIN = getattr(
    _piper_action_conventions, "step_deltas_to_chunk_origin", None
)

try:
    from .episode_split import (
        DEFAULT_SPLIT_SEED,
        DEFAULT_TEST_RATIO,
        NORM_CONFIG_FILENAME,
        NORM_CONFIG_VERSION,
        EpisodeSplit,
        load_episode_split,
        resolve_episode_split,
        normalize_contract_fingerprint,
        write_norm_config,
        write_norm_split,
    )
except ImportError:  # openpi_single_arm.py is normally executed directly
    from episode_split import (
        DEFAULT_SPLIT_SEED,
        DEFAULT_TEST_RATIO,
        NORM_CONFIG_FILENAME,
        NORM_CONFIG_VERSION,
        EpisodeSplit,
        load_episode_split,
        resolve_episode_split,
        normalize_contract_fingerprint,
        write_norm_config,
        write_norm_split,
    )


class _ExpectedWebsocketProbeFilter(logging.Filter):
    """Drop expected health-check and bare-TCP probe noise from websockets."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        # OpenPI serves HTTP /healthz from process_request. websockets reports
        # every successful health response as a rejected WebSocket connection.
        if message == "connection rejected (200 OK)":
            return False
        # Port scanners and TCP readiness probes may connect and close without
        # sending an HTTP Upgrade request. Keep all other handshake errors.
        exception = record.exc_info[1] if record.exc_info else None
        if message == "opening handshake failed":
            if isinstance(exception, ConnectionClosedError):
                return False
            if isinstance(exception, InvalidMessage):
                cause: BaseException | None = exception
                while cause is not None:
                    if isinstance(cause, EOFError):
                        return False
                    cause = cause.__cause__
        return True


def _install_websocket_probe_filter() -> None:
    logger = logging.getLogger("websockets.server")
    if not any(isinstance(item, _ExpectedWebsocketProbeFilter) for item in logger.filters):
        logger.addFilter(_ExpectedWebsocketProbeFilter())


CONFIG_NAMES = {
    ("pi05", "single"): "pi05_piper_single_arm_lora",
    ("pi05", "bimanual"): "pi05_piper_bimanual_lora",
    ("pi0", "single"): "pi0_piper_single_arm_lora",
    ("pi0", "bimanual"): "pi0_piper_bimanual_lora",
}


def config_name(model_variant: str, arm_mode: str) -> str:
    try:
        return CONFIG_NAMES[(model_variant, arm_mode)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported model/arm combination: model_variant={model_variant!r}, arm_mode={arm_mode!r}"
        ) from exc


ACTION_CONVENTION_REGISTRY = ".policy_action_conventions"
ACTION_CONVENTION_MARKER_VERSION = 3


def _action_convention_marker(
    checkpoint_base_dir: Path, model_variant: str, arm_mode: str, exp_name: str
) -> Path:
    return (
        checkpoint_base_dir
        / config_name(model_variant, arm_mode)
        / ACTION_CONVENTION_REGISTRY
        / f"{exp_name}.json"
    )


def _read_action_contract_marker(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    # Version-1 markers only recorded the delivery model convention.  Keep
    # them recognizable so an explicitly-requested legacy resume can migrate
    # the marker, but never treat them as a complete contract.
    convention = payload.get("model_action_convention") or payload.get(
        "delivery_action_convention"
    )
    if convention is not None:
        payload["model_action_convention"] = str(convention)
    return payload


def _checkpoint_has_saved_steps(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        child.is_dir() and child.name.isdigit() and (child / "params").is_dir()
        for child in path.iterdir()
    )


def _checkpoint_marker_for_step(checkpoint: Path) -> Path:
    experiment_dir = checkpoint.parent
    return (
        experiment_dir.parent
        / ACTION_CONVENTION_REGISTRY
        / f"{experiment_dir.name}.json"
    )


def _resolve_delivery_action_convention(
    args: argparse.Namespace, *, contract: "DatasetContract"
) -> str | None:
    if contract.schema != "delivery":
        return None
    cached = getattr(args, "_resolved_delivery_action_convention", None)
    if cached in DELIVERY_ACTION_CONVENTIONS:
        return cached

    requested_delivery = str(getattr(args, "delivery_action_convention", "auto"))
    requested_model = getattr(args, "model_action_convention", None)
    if requested_model not in (None, "", "auto"):
        requested_model = str(requested_model)
        if requested_delivery not in {"auto", requested_model}:
            raise ValueError(
                "--delivery-action-convention and --model-action-convention disagree"
            )
        requested_delivery = requested_model

    if requested_delivery in DELIVERY_ACTION_CONVENTIONS:
        resolved = requested_delivery
    elif contract.raw_action_convention == DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION:
        resolved = DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
    elif args.command == "norm":
        resolved = DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
    elif args.command == "train":
        checkpoint_base = Path(args.checkpoint_base_dir).expanduser().resolve()
        experiment_dir = (
            checkpoint_base
            / config_name(args.model_variant, contract.arm_mode)
            / args.exp_name
        )
        marker = _action_convention_marker(
            checkpoint_base, args.model_variant, contract.arm_mode, args.exp_name
        )
        recorded = _read_action_contract_marker(marker)
        if args.resume and _checkpoint_has_saved_steps(experiment_dir):
            if recorded is None:
                raise ValueError(
                    "existing delivery checkpoint has no action-contract marker; "
                    "explicitly pass --delivery-action-convention step for a legacy "
                    "step-delta checkpoint, or start a new experiment"
                )
            resolved = str(recorded.get("model_action_convention", ""))
        else:
            resolved = DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
    elif args.command == "serve":
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        recorded = _read_action_contract_marker(_checkpoint_marker_for_step(checkpoint))
        if recorded is None:
            raise ValueError(
                "checkpoint has no action-contract marker; explicitly pass "
                "--delivery-action-convention step only for a verified legacy "
                "step-delta checkpoint"
            )
        resolved = str(recorded.get("model_action_convention", ""))
    else:
        raise ValueError(f"cannot resolve delivery action convention for command {args.command!r}")

    if resolved not in DELIVERY_ACTION_CONVENTIONS:
        raise ValueError(
            "delivery action convention must be one of "
            f"{sorted(DELIVERY_ACTION_CONVENTIONS)}, got {resolved!r}"
        )
    if (
        contract.raw_action_convention == DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION
        and resolved != DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
    ):
        raise ValueError(
            "absolute-EEF delivery datasets can only train/serve chunk_origin model actions"
        )
    setattr(args, "_resolved_delivery_action_convention", resolved)
    return resolved


def _resolve_model_gripper_semantics(
    args: argparse.Namespace, *, contract: "DatasetContract"
) -> str:
    if contract.schema == "delivery":
        return contract.raw_gripper_semantics
    requested = getattr(args, "model_gripper_semantics", None)
    requested_alias = getattr(args, "gripper_semantics", None)
    if requested not in (None, "", "auto") and requested_alias not in (
        None,
        "",
        "auto",
        requested,
    ):
        raise ValueError("--model-gripper-semantics and --gripper-semantics disagree")
    requested = requested if requested not in (None, "", "auto") else requested_alias
    if requested not in (None, "", "auto"):
        resolved = _canonical_gripper_semantics(
            requested, default=GRIPPER_OPENING_FRACTION
        )
    elif args.command == "norm":
        resolved = GRIPPER_OPENING_FRACTION
    elif args.command == "train":
        experiment_dir = (
            Path(args.checkpoint_base_dir).expanduser().resolve()
            / config_name(args.model_variant, contract.arm_mode)
            / args.exp_name
        )
        marker = _read_action_contract_marker(
            _action_convention_marker(
                Path(args.checkpoint_base_dir).expanduser().resolve(),
                args.model_variant,
                contract.arm_mode,
                args.exp_name,
            )
        )
        if args.resume and _checkpoint_has_saved_steps(experiment_dir):
            if marker is None or marker.get("gripper_semantics") in (None, ""):
                raise ValueError(
                    "existing joint checkpoint has no gripper-semantics marker; "
                    "explicitly pass --model-gripper-semantics absolute_opening_metres "
                    "for a verified legacy-v2 checkpoint"
                )
            resolved = _canonical_gripper_semantics(
                marker["gripper_semantics"], default=GRIPPER_OPENING_FRACTION
            )
        else:
            resolved = GRIPPER_OPENING_FRACTION
    elif args.command == "serve":
        marker = _read_action_contract_marker(
            _checkpoint_marker_for_step(Path(args.checkpoint).expanduser().resolve())
        )
        if marker is None or marker.get("gripper_semantics") in (None, ""):
            raise ValueError(
                "joint checkpoint has no gripper-semantics marker; explicitly pass "
                "--model-gripper-semantics absolute_opening_metres only for a verified "
                "legacy-v2 checkpoint"
            )
        resolved = _canonical_gripper_semantics(
            marker["gripper_semantics"], default=GRIPPER_OPENING_FRACTION
        )
    else:
        resolved = GRIPPER_OPENING_FRACTION

    if contract.raw_gripper_semantics == GRIPPER_OPENING_FRACTION:
        if resolved != GRIPPER_OPENING_FRACTION:
            raise ValueError("joint v3 opening-fraction data cannot serve/train meter grippers")
    elif contract.raw_gripper_semantics == GRIPPER_OPENING_METERS:
        if resolved not in {GRIPPER_OPENING_METERS, GRIPPER_OPENING_FRACTION}:
            raise ValueError("legacy joint v2 supports meter or converted opening-fraction models")
    return resolved


def _validate_checkpoint_contract(
    args: argparse.Namespace, contract: "DatasetContract"
) -> None:
    if args.command not in {"train", "serve"}:
        return
    if args.command == "train":
        experiment_dir = (
            Path(args.checkpoint_base_dir).expanduser().resolve()
            / config_name(args.model_variant, contract.arm_mode)
            / args.exp_name
        )
        if not (args.resume and _checkpoint_has_saved_steps(experiment_dir)):
            return
        marker_path = _action_convention_marker(
            Path(args.checkpoint_base_dir).expanduser().resolve(),
            args.model_variant,
            contract.arm_mode,
            args.exp_name,
        )
    else:
        marker_path = _checkpoint_marker_for_step(
            Path(args.checkpoint).expanduser().resolve()
        )

    marker = _read_action_contract_marker(marker_path)
    if marker is None:
        explicit_legacy_delivery = (
            contract.legacy_delivery
            and contract.model_action_convention == DELIVERY_STEP_ACTION_CONVENTION
            and str(getattr(args, "delivery_action_convention", "auto"))
            == DELIVERY_STEP_ACTION_CONVENTION
        )
        explicit_legacy_joint = (
            contract.schema == "joint"
            and contract.raw_gripper_semantics == GRIPPER_OPENING_METERS
            and contract.model_gripper_semantics == GRIPPER_OPENING_METERS
            and str(getattr(args, "model_gripper_semantics", "auto"))
            == GRIPPER_OPENING_METERS
        )
        if explicit_legacy_delivery or explicit_legacy_joint:
            logging.warning(
                "using explicitly-selected legacy checkpoint without marker: %s",
                marker_path,
            )
            return
        raise ValueError(f"checkpoint action-contract marker is missing: {marker_path}")

    if marker.get("dataset_id") not in (None, args.dataset_id):
        raise ValueError(
            f"checkpoint marker dataset_id={marker.get('dataset_id')!r} does not match "
            f"{args.dataset_id!r}"
        )
    expected = contract.fingerprint()
    if int(marker.get("version", 1)) < ACTION_CONVENTION_MARKER_VERSION:
        marker_convention = marker.get("model_action_convention")
        if (
            contract.legacy_delivery
            and marker_convention == contract.model_action_convention
            and str(getattr(args, "delivery_action_convention", "auto"))
            == contract.model_action_convention
        ):
            logging.warning("migrating incomplete legacy action marker: %s", marker_path)
            return
        raise ValueError(
            f"checkpoint marker {marker_path} predates the complete raw/model action contract; "
            "select the legacy convention explicitly only after verifying the checkpoint"
        )
    mismatches = {
        key: {"checkpoint": marker.get(key), "dataset": value}
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "checkpoint action contract does not match dataset/training contract: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def complete_action_contract_fingerprint(
    contract: dict[str, Any] | DatasetContract | Any,
) -> dict[str, int | str]:
    """Return the representation and temporal fields that bind norm/checkpoints."""
    if dataclasses.is_dataclass(contract):
        values = {field.name: getattr(contract, field.name) for field in dataclasses.fields(contract)}
        values["gripper_semantics"] = values.get("model_gripper_semantics")
    elif isinstance(contract, dict):
        values = contract
    else:
        raise ValueError("action contract must be a mapping or dataclass")
    fingerprint = normalize_contract_fingerprint(values)
    try:
        action_offset = int(values.get("action_offset"))
        model_start = int(values.get("model_action_start_offset", values.get("model_action_start_offset_steps")))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "action contract requires integer action_offset and model_action_start_offset"
        ) from exc
    if action_offset not in {0, 1}:
        raise ValueError("action_offset must be 0 (same-step command) or 1 (next-measured fallback)")
    if model_start != MODEL_ACTION_START_OFFSET_STEPS:
        raise ValueError(
            f"model_action_start_offset must be {MODEL_ACTION_START_OFFSET_STEPS}, got {model_start}"
        )
    return {
        **fingerprint,
        "action_offset": action_offset,
        "model_action_start_offset": model_start,
    }


def _write_action_convention_marker(config: training_config.TrainConfig) -> Path:
    metadata = config.policy_metadata or {}
    path = (
        config.checkpoint_dir.parent
        / ACTION_CONVENTION_REGISTRY
        / f"{config.checkpoint_dir.name}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": ACTION_CONVENTION_MARKER_VERSION,
        "dataset_id": metadata.get("dataset_id"),
        "schema": metadata.get("schema"),
        "arm_mode": metadata.get("arm_mode"),
        "arm_side": metadata.get("arm_side"),
        "state_dim": metadata.get("state_dim"),
        "raw_action_dim": metadata.get("raw_action_dim"),
        "model_action_dim": metadata.get("model_action_dim"),
        "contract_version": metadata.get("contract_version"),
        "raw_action_semantics": metadata.get("raw_action_semantics"),
        "model_action_semantics": metadata.get("model_action_semantics"),
        "raw_action_convention": metadata.get("raw_action_convention"),
        "model_action_convention": metadata.get("model_action_convention"),
        "gripper_semantics": metadata.get("model_gripper_semantics"),
        "raw_gripper_semantics": metadata.get("raw_gripper_semantics"),
        "wire_gripper_semantics": metadata.get("wire_gripper_semantics"),
        "delivery_action_convention": metadata.get("delivery_action_convention"),
        "wire_action_semantics": metadata.get("wire_action_semantics"),
        "wire_action_convention": metadata.get("wire_action_convention"),
        "state_gripper_semantics": metadata.get("state_gripper_semantics"),
        "action_offset": metadata.get("action_offset"),
        "model_action_start_offset": metadata.get("model_action_start_offset"),
    }
    complete_action_contract_fingerprint(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


@dataclasses.dataclass(frozen=True)
class DatasetContract:
    schema: str
    arm_mode: str
    arm_side: str
    layout: str
    contract_version: int
    state_dim: int
    raw_action_dim: int
    model_action_dim: int
    camera_keys: tuple[str, ...]
    action_hz: float | None
    raw_action_semantics: str
    model_action_semantics: str
    wire_action_semantics: str
    raw_action_convention: str
    model_action_convention: str
    raw_gripper_semantics: str
    model_gripper_semantics: str
    wire_gripper_semantics: str
    action_source: str
    action_alignment: str
    action_offset: int
    model_action_start_offset: int = MODEL_ACTION_START_OFFSET_STEPS
    legacy_delivery: bool = False

    @property
    def action_dim(self) -> int:
        """Backward-compatible alias for the Policy/wire action dimension."""
        return self.model_action_dim

    def fingerprint(self) -> dict[str, int | str]:
        return complete_action_contract_fingerprint(self)

    def with_model_action_convention(self, convention: str | None) -> "DatasetContract":
        if self.schema != "delivery":
            return self
        if convention == DELIVERY_STEP_ACTION_CONVENTION:
            if not self.legacy_delivery:
                raise ValueError(
                    "step model actions are only valid for explicit legacy 7D delivery data"
                )
            semantics = self.raw_action_semantics
        elif convention == DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION:
            semantics = (
                DELIVERY_LEGACY_CHUNK_ORIGIN_ACTION_SEMANTICS
                if self.legacy_delivery
                else DELIVERY_MODEL_ACTION_SEMANTICS
            )
        else:
            raise ValueError(f"unsupported delivery model action convention: {convention!r}")
        return dataclasses.replace(
            self,
            model_action_convention=convention,
            model_action_semantics=semantics,
            wire_action_semantics=semantics,
        )

    def with_model_gripper_semantics(self, semantics: str) -> "DatasetContract":
        if semantics not in {
            GRIPPER_OPENING_FRACTION,
            GRIPPER_OPENING_METERS,
            GRIPPER_CLOSED_FRACTION,
        }:
            raise ValueError(f"unsupported model gripper semantics: {semantics!r}")
        if self.schema == "delivery" and semantics != self.raw_gripper_semantics:
            raise ValueError("delivery gripper semantics cannot be changed during training")
        if self.schema == "joint" and semantics == GRIPPER_CLOSED_FRACTION:
            raise ValueError("joint schema cannot use closed-fraction gripper semantics")
        wire_semantics = self.wire_action_semantics
        if self.schema == "joint":
            wire_semantics = (
                JOINT_RAW_ACTION_SEMANTICS
                if semantics == GRIPPER_OPENING_FRACTION
                else self.raw_action_semantics
            )
        return dataclasses.replace(
            self,
            model_gripper_semantics=semantics,
            wire_gripper_semantics=semantics,
            wire_action_semantics=wire_semantics,
        )


def _dataset_root() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))


def _dataset_info(dataset_id: str) -> dict[str, Any]:
    path = _dataset_root() / dataset_id / "meta" / "info.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"invalid LeRobot info.json: {path}")
    return value


def _dataset_contract_metadata(info: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key in ("data_contract", "contract", "piper_contract"):
        value = info.get(key)
        if isinstance(value, dict):
            nested.update(value)
    # LeRobot exporters currently write these fields at top-level.  Top-level
    # wins so metadata can be patched without rewriting a nested object.
    nested.update(info)
    return nested


def _positive_metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dataset {key} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"dataset {key} must be a positive integer")
    return result


def _resolve_temporal_offsets(
    metadata: dict[str, Any],
    *,
    legacy_delivery: bool,
) -> tuple[int, int]:
    alignment = str(metadata.get("action_alignment") or "").strip().lower()
    source = str(metadata.get("action_source") or "").strip().lower()
    expected_raw_offset: int | None = None
    if alignment.startswith("same_step_command"):
        expected_raw_offset = 0
    elif alignment in {"next_observation", "next_measured", "next_measured_fallback"}:
        expected_raw_offset = 1
    elif "next_measured" in source:
        expected_raw_offset = 1

    raw_value = metadata.get("action_offset")
    if raw_value is None:
        action_offset = (
            expected_raw_offset
            if expected_raw_offset is not None
            else 1 if legacy_delivery else 0
        )
    else:
        try:
            action_offset = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset action_offset must be integer 0 or 1") from exc
    if action_offset not in {0, 1}:
        raise ValueError("dataset action_offset must be 0 or 1")
    if expected_raw_offset is not None and action_offset != expected_raw_offset:
        raise ValueError(
            f"action_offset={action_offset} conflicts with action_alignment={alignment!r}; "
            f"expected {expected_raw_offset}"
        )

    raw_model_start = metadata.get(
        "model_action_start_offset",
        metadata.get("model_action_start_offset_steps", MODEL_ACTION_START_OFFSET_STEPS),
    )
    try:
        model_start = int(raw_model_start)
    except (TypeError, ValueError) as exc:
        raise ValueError("model_action_start_offset must be integer 1") from exc
    if model_start != MODEL_ACTION_START_OFFSET_STEPS:
        raise ValueError(
            f"model_action_start_offset must be {MODEL_ACTION_START_OFFSET_STEPS}, got {model_start}"
        )
    return action_offset, model_start


def _canonical_gripper_semantics(value: Any, *, default: str) -> str:
    if value in (None, ""):
        return default
    normalized = str(value).strip().lower()
    aliases = {
        GRIPPER_CLOSED_FRACTION.lower(): GRIPPER_CLOSED_FRACTION,
        "closed_fraction": GRIPPER_CLOSED_FRACTION,
        "absolute_closed_fraction": GRIPPER_CLOSED_FRACTION,
        "gripper_closed_fraction": GRIPPER_CLOSED_FRACTION,
        GRIPPER_OPENING_FRACTION.lower(): GRIPPER_OPENING_FRACTION,
        "opening_fraction": GRIPPER_OPENING_FRACTION,
        "absolute_opening_fraction": GRIPPER_OPENING_FRACTION,
        "gripper_opening_fraction": GRIPPER_OPENING_FRACTION,
        GRIPPER_OPENING_METERS.lower(): GRIPPER_OPENING_METERS,
        "opening_m": GRIPPER_OPENING_METERS,
        "gripper_opening_m": GRIPPER_OPENING_METERS,
        "absolute_opening_m": GRIPPER_OPENING_METERS,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported gripper semantics: {value!r}") from exc


def _canonical_raw_delivery_convention(value: Any, *, legacy_delivery: bool) -> str:
    if value in (None, ""):
        return (
            DELIVERY_STEP_ACTION_CONVENTION
            if legacy_delivery
            else DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION
        )
    normalized = str(value).strip().lower()
    step_aliases = {
        DELIVERY_STEP_ACTION_CONVENTION.lower(),
        "one_step",
        "one_step_delta",
        "step_delta",
    }
    absolute_aliases = {
        DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION.lower(),
        "absolute",
        "absolute_eef",
        "absolute_eef_target",
    }
    if normalized in step_aliases:
        result = DELIVERY_STEP_ACTION_CONVENTION
    elif normalized in absolute_aliases:
        result = DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION
    else:
        raise ValueError(f"unsupported delivery raw action convention: {value!r}")
    expected = (
        DELIVERY_STEP_ACTION_CONVENTION
        if legacy_delivery
        else DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION
    )
    if result != expected:
        raise ValueError(
            f"delivery raw action dimension requires convention {expected!r}, got {result!r}"
        )
    return result


def _validate_requested_contract(args: argparse.Namespace, contract: DatasetContract) -> None:
    for field in (
        "contract_version",
        "raw_action_dim",
        "model_action_dim",
        "raw_action_semantics",
        "model_action_semantics",
        "raw_action_convention",
        "model_action_convention",
        "action_offset",
        "model_action_start_offset",
    ):
        requested = getattr(args, field, None)
        if requested in (None, "", "auto"):
            continue
        actual = getattr(contract, field)
        if field in {"contract_version", "raw_action_dim", "model_action_dim", "action_offset", "model_action_start_offset"}:
            requested = int(requested)
        else:
            requested = str(requested)
        if requested != actual:
            raise ValueError(
                f"--{field.replace('_', '-')}={requested!r} conflicts with dataset value {actual!r}"
            )


def resolve_dataset_contract(args: argparse.Namespace) -> DatasetContract:
    info = _dataset_info(args.dataset_id)
    metadata = _dataset_contract_metadata(info)
    features = info.get("features", {}) if isinstance(info.get("features", {}), dict) else {}
    if "observation.state" in features and "action" in features:
        layout = "canonical"
        state_key, action_key = "observation.state", "action"
    elif "state" in features and "actions" in features:
        layout = "legacy"
        state_key, action_key = "state", "actions"
    else:
        layout = str(args.dataset_layout)
        if layout == "auto":
            raise ValueError(
                f"cannot infer dataset layout for {args.dataset_id!r}; expected canonical "
                "observation.state/action or legacy state/actions features"
            )
        state_key, action_key = (
            ("observation.state", "action")
            if layout == "canonical"
            else ("state", "actions")
        )
    if args.dataset_layout != "auto" and layout != args.dataset_layout:
        raise ValueError(
            f"--dataset-layout={args.dataset_layout} conflicts with dataset layout={layout}"
        )

    def last_dim(key: str) -> int | None:
        feature = features.get(key, {})
        shape = feature.get("shape") if isinstance(feature, dict) else None
        return int(shape[-1]) if isinstance(shape, (list, tuple)) and shape else None

    state_dim = last_dim(state_key)
    raw_action_dim = last_dim(action_key)
    by_dims = {
        (7, 7): ("joint", "single", False),
        (14, 14): ("joint", "bimanual", False),
        (10, 7): ("delivery", "single", True),
        (20, 14): ("delivery", "bimanual", True),
        (10, 10): ("delivery", "single", False),
        (20, 20): ("delivery", "bimanual", False),
    }
    inferred = by_dims.get((state_dim, raw_action_dim))
    if inferred is None:
        raise ValueError(
            f"unsupported Piper state/raw-action dimensions {(state_dim, raw_action_dim)}; "
            "expected joint (7,7)/(14,14), legacy delivery (10,7)/(20,14), "
            "or absolute-EEF delivery (10,10)/(20,20)"
        )
    inferred_schema, inferred_arm_mode, legacy_delivery = inferred
    schema = str(metadata.get("schema") or inferred_schema).lower()
    arm_mode = str(metadata.get("arm_mode") or inferred_arm_mode).lower()
    if schema != inferred_schema or arm_mode != inferred_arm_mode:
        raise ValueError(
            "dataset schema/arm metadata conflicts with physical feature dimensions: "
            f"metadata={schema}/{arm_mode} dims={(state_dim, raw_action_dim)} "
            f"imply={inferred_schema}/{inferred_arm_mode}"
        )
    if args.schema != "auto" and schema != args.schema:
        raise ValueError(f"--schema={args.schema} conflicts with dataset schema={schema}")
    if args.arm_mode != "auto" and arm_mode != args.arm_mode:
        raise ValueError(f"--arm-mode={args.arm_mode} conflicts with dataset arm_mode={arm_mode}")

    # The measured 8_3_64eps contract is the legacy state/actions layout.  A
    # canonical 10D/7D dataset is accepted only when metadata explicitly marks
    # the v2 one-step convention; this avoids confusing it with new 10D raw
    # absolute EEF targets.
    raw_convention_metadata = (
        metadata.get("raw_action_convention")
        or metadata.get("dataset_action_convention")
        or metadata.get("action_convention")
    )
    contract_version_metadata = _positive_metadata_int(metadata, "contract_version")
    if legacy_delivery and layout == "canonical" and not (
        str(metadata.get("legacy_format") or "").lower() == "legacy_v2"
        or raw_convention_metadata not in (None, "")
    ):
        raise ValueError(
            "canonical 10D/7D delivery data is ambiguous; explicitly set "
            "contract_version=2 and raw_action_convention=step"
        )
    if layout == "legacy" and not (legacy_delivery and arm_mode == "single"):
        raise ValueError(
            "legacy state/actions image/wrist_image layout only supports single-arm "
            "10D state + 7D delivery-v2 step actions"
        )

    arm_count = 2 if arm_mode == "bimanual" else 1
    model_action_dim = 7 * arm_count
    if schema == "joint":
        declared_gripper = metadata.get("raw_gripper_semantics") or metadata.get(
            "gripper_semantics"
        )
        state_names = features.get(state_key, {}).get("names", [])
        action_names = features.get(action_key, {}).get("names", [])
        joined_names = " ".join(map(str, [*state_names, *action_names])).lower()
        if declared_gripper in (None, ""):
            if "gripper_opening_m" in joined_names:
                declared_gripper = GRIPPER_OPENING_METERS
            elif "gripper_opening_fraction" in joined_names:
                declared_gripper = GRIPPER_OPENING_FRACTION
            elif contract_version_metadata is None:
                raise ValueError(
                    "7D/14D joint data is ambiguous without contract_version or explicit "
                    "gripper_semantics (legacy v2 uses gripper_opening_m; v3 uses opening_fraction)"
                )
        default_gripper = (
            GRIPPER_OPENING_METERS
            if contract_version_metadata is not None
            and contract_version_metadata <= LEGACY_DELIVERY_CONTRACT_VERSION
            else GRIPPER_OPENING_FRACTION
        )
        raw_gripper_semantics = _canonical_gripper_semantics(
            declared_gripper, default=default_gripper
        )
        # Some v3 LeRobot exporters wrap legacy-v2 rows while preserving an
        # explicit meter gripper marker. The effective training contract stays
        # v2 so split/norm/checkpoint fingerprints cannot mix the units.
        contract_version = (
            LEGACY_DELIVERY_CONTRACT_VERSION
            if raw_gripper_semantics == GRIPPER_OPENING_METERS
            else max(contract_version_metadata or CURRENT_CONTRACT_VERSION, 3)
        )
        raw_action_convention = str(
            metadata.get("raw_action_convention") or JOINT_RAW_ACTION_CONVENTION
        )
        if raw_action_convention != JOINT_RAW_ACTION_CONVENTION:
            raise ValueError(
                f"joint raw actions must use {JOINT_RAW_ACTION_CONVENTION!r}"
            )
        raw_action_semantics = str(
            metadata.get("raw_action_semantics")
            or metadata.get("action_semantics")
            or JOINT_RAW_ACTION_SEMANTICS
        )
        model_action_convention = JOINT_MODEL_ACTION_CONVENTION
        model_action_semantics = JOINT_MODEL_ACTION_SEMANTICS
        wire_action_semantics = raw_action_semantics
        model_gripper_semantics = GRIPPER_OPENING_FRACTION
        wire_gripper_semantics = model_gripper_semantics
    else:
        contract_version = (
            LEGACY_DELIVERY_CONTRACT_VERSION
            if legacy_delivery
            else contract_version_metadata or max(CURRENT_CONTRACT_VERSION, 3)
        )
        if not legacy_delivery and contract_version < 3:
            raise ValueError(
                "10D absolute-EEF raw delivery data requires contract_version>=3"
            )
        raw_action_convention = _canonical_raw_delivery_convention(
            raw_convention_metadata, legacy_delivery=legacy_delivery
        )
        raw_action_semantics = str(
            metadata.get("raw_action_semantics")
            or metadata.get("dataset_action_semantics")
            or metadata.get("action_semantics")
            or (
                DELIVERY_LEGACY_STEP_ACTION_SEMANTICS
                if legacy_delivery
                else DELIVERY_ABSOLUTE_EEF_ACTION_SEMANTICS
            )
        )
        model_action_convention = DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION
        model_action_semantics = (
            DELIVERY_LEGACY_CHUNK_ORIGIN_ACTION_SEMANTICS
            if legacy_delivery
            else DELIVERY_MODEL_ACTION_SEMANTICS
        )
        wire_action_semantics = model_action_semantics
        raw_gripper_semantics = _canonical_gripper_semantics(
            metadata.get("raw_gripper_semantics") or metadata.get("gripper_semantics"),
            default=(
                GRIPPER_CLOSED_FRACTION
                if legacy_delivery
                else GRIPPER_OPENING_FRACTION
            ),
        )
        expected_delivery_gripper = (
            GRIPPER_CLOSED_FRACTION
            if legacy_delivery
            else GRIPPER_OPENING_FRACTION
        )
        if raw_gripper_semantics != expected_delivery_gripper:
            raise ValueError(
                f"delivery contract requires gripper semantics {expected_delivery_gripper!r}, "
                f"got {raw_gripper_semantics!r}"
            )
        model_gripper_semantics = raw_gripper_semantics
        wire_gripper_semantics = raw_gripper_semantics

    declared_raw_dim = _positive_metadata_int(metadata, "raw_action_dim")
    if declared_raw_dim is not None and declared_raw_dim != raw_action_dim:
        raise ValueError(
            f"dataset raw_action_dim={declared_raw_dim} conflicts with feature dim {raw_action_dim}"
        )
    declared_model_dim = _positive_metadata_int(metadata, "model_action_dim")
    if declared_model_dim is not None and declared_model_dim != model_action_dim:
        raise ValueError(
            f"dataset model_action_dim={declared_model_dim} conflicts with expected {model_action_dim}"
        )

    arm_side = "both" if arm_mode == "bimanual" else str(metadata.get("arm_side") or args.arm_side).lower()
    if arm_mode == "single" and arm_side not in {"left", "right"}:
        raise ValueError("single-arm dataset requires arm_side left or right")
    if arm_mode == "bimanual" and args.arm_side not in {"both", "right"}:
        raise ValueError("bimanual dataset requires --arm-side both")

    media = [
        key.removeprefix("observation.images.")
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") in {"image", "video"}
    ]
    if layout == "legacy":
        camera_keys = ("cam_high", "cam_wrist")
    elif arm_mode == "bimanual":
        camera_keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    else:
        expected = f"cam_{arm_side}_wrist"
        wrist = expected if expected in media else "cam_wrist" if "cam_wrist" in media else expected
        camera_keys = ("cam_high", wrist)
    missing_media = [
        f"observation.images.{key}"
        for key in camera_keys
        if layout == "canonical" and f"observation.images.{key}" not in features
    ]
    if missing_media:
        raise ValueError(f"dataset is missing required camera features: {missing_media}")

    raw_action_hz = info.get("fps")
    action_hz: float | None = None
    if raw_action_hz is not None:
        try:
            action_hz = float(raw_action_hz)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dataset fps must be a positive number, got {raw_action_hz!r}") from exc
        if not np.isfinite(action_hz) or action_hz <= 0:
            raise ValueError(f"dataset fps must be a positive number, got {raw_action_hz!r}")

    action_offset, model_action_start_offset = _resolve_temporal_offsets(
        metadata, legacy_delivery=legacy_delivery
    )
    contract = DatasetContract(
        schema=schema,
        arm_mode=arm_mode,
        arm_side=arm_side,
        layout=layout,
        contract_version=contract_version,
        state_dim=int(state_dim),
        raw_action_dim=int(raw_action_dim),
        model_action_dim=model_action_dim,
        camera_keys=camera_keys,
        action_hz=action_hz,
        raw_action_semantics=raw_action_semantics,
        model_action_semantics=model_action_semantics,
        wire_action_semantics=wire_action_semantics,
        raw_action_convention=raw_action_convention,
        model_action_convention=model_action_convention,
        raw_gripper_semantics=raw_gripper_semantics,
        model_gripper_semantics=model_gripper_semantics,
        wire_gripper_semantics=wire_gripper_semantics,
        action_source=str(metadata.get("action_source") or "unknown"),
        action_alignment=str(metadata.get("action_alignment") or "unknown"),
        action_offset=action_offset,
        model_action_start_offset=model_action_start_offset,
        legacy_delivery=legacy_delivery,
    )
    return contract

def _as_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"image must be rank 3, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"image must have three RGB channels, got {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Map single-arm or bimanual Piper observations into OpenPI inputs."""

    contract: DatasetContract

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != self.contract.state_dim:
            raise ValueError(
                f"Piper {self.contract.arm_mode} {self.contract.schema} state must be "
                f"{self.contract.state_dim}D, got {state.shape}"
            )
        images = data["images"]
        expected = set(self.contract.camera_keys)
        if set(images) != expected:
            raise ValueError(f"camera keys must be {sorted(expected)}, got {sorted(images)}")
        high = _as_hwc_uint8(images["cam_high"])
        if self.contract.arm_mode == "bimanual":
            left = _as_hwc_uint8(images["cam_left_wrist"])
            right = _as_hwc_uint8(images["cam_right_wrist"])
            mapped_images = {
                "base_0_rgb": high,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            }
            image_mask = {key: np.True_ for key in mapped_images}
        else:
            wrist_key = next(key for key in self.contract.camera_keys if "wrist" in key)
            wrist = _as_hwc_uint8(images[wrist_key])
            mapped_images = {
                "base_0_rgb": high,
                "left_wrist_0_rgb": wrist if self.contract.arm_side == "left" else np.zeros_like(wrist),
                "right_wrist_0_rgb": wrist if self.contract.arm_side == "right" else np.zeros_like(wrist),
            }
            image_mask = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.bool_(self.contract.arm_side == "left"),
                "right_wrist_0_rgb": np.bool_(self.contract.arm_side == "right"),
            }
        output = {"image": mapped_images, "image_mask": image_mask, "state": state}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != self.contract.raw_action_dim:
                raise ValueError(
                    f"Piper {self.contract.arm_mode} raw actions must be "
                    f"{self.contract.raw_action_dim}D, got {actions.shape}"
                )
            output["actions"] = actions
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


def _fallback_absolute_eef_targets_to_chunk_origin(
    current_state: np.ndarray, targets: np.ndarray, *, arm_count: int
) -> np.ndarray:
    """Fallback matching piper_action_conventions when the shared helper is absent."""
    from scipy.spatial.transform import Rotation

    state = np.asarray(current_state, dtype=np.float32)
    values = np.asarray(targets, dtype=np.float32)
    expected_state = 10 * arm_count
    if state.shape != (expected_state,):
        raise ValueError(f"current EEF state must have shape ({expected_state},), got {state.shape}")
    if values.ndim != 2 or values.shape[-1] != expected_state:
        raise ValueError(f"absolute EEF targets must have shape (T,{expected_state}), got {values.shape}")
    if not np.isfinite(state).all() or not np.isfinite(values).all():
        raise ValueError("EEF state/targets contain NaN or Inf")

    def rotation6d_to_matrix(rotation6d: np.ndarray) -> np.ndarray:
        col0 = np.asarray(rotation6d, dtype=np.float64)[..., :3]
        col1 = np.asarray(rotation6d, dtype=np.float64)[..., 3:]
        norm0 = np.linalg.norm(col0, axis=-1, keepdims=True)
        if np.any(norm0 < 1e-12):
            raise ValueError("rotation6d first column has zero norm")
        col0 = col0 / norm0
        col1 = col1 - col0 * np.sum(col0 * col1, axis=-1, keepdims=True)
        norm1 = np.linalg.norm(col1, axis=-1, keepdims=True)
        if np.any(norm1 < 1e-12):
            raise ValueError("rotation6d second column is degenerate")
        col1 = col1 / norm1
        return np.stack((col0, col1, np.cross(col0, col1)), axis=-1)

    output = np.empty((len(values), 7 * arm_count), dtype=np.float32)
    for arm in range(arm_count):
        ss = arm * 10
        aa = arm * 7
        current_rotation = rotation6d_to_matrix(state[ss + 3 : ss + 9])
        target_rotation = rotation6d_to_matrix(values[:, ss + 3 : ss + 9])
        output[:, aa : aa + 3] = values[:, ss : ss + 3] - state[ss : ss + 3]
        output[:, aa + 3 : aa + 6] = Rotation.from_matrix(
            target_rotation @ current_rotation.T
        ).as_rotvec().astype(np.float32)
        output[:, aa + 6] = values[:, ss + 9]
    return output


def _absolute_eef_targets_to_chunk_origin(
    state: np.ndarray, targets: np.ndarray, *, arm_count: int
) -> np.ndarray:
    if callable(ABSOLUTE_EEF_TARGETS_TO_CHUNK_ORIGIN):
        return np.asarray(
            ABSOLUTE_EEF_TARGETS_TO_CHUNK_ORIGIN(
                state, targets, arm_count=arm_count
            ),
            dtype=np.float32,
        )
    return _fallback_absolute_eef_targets_to_chunk_origin(
        state, targets, arm_count=arm_count
    )


def _fallback_step_deltas_to_chunk_origin(
    actions: np.ndarray, *, arm_count: int
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    values = np.asarray(actions, dtype=np.float32)
    expected = 7 * arm_count
    if values.ndim != 2 or values.shape[-1] != expected:
        raise ValueError(f"step actions must have shape (T,{expected}), got {values.shape}")
    output = values.copy()
    for arm in range(arm_count):
        offset = arm * 7
        output[:, offset : offset + 3] = np.cumsum(
            values[:, offset : offset + 3], axis=0
        )
        rotation = np.eye(3)
        for index, rotvec in enumerate(values[:, offset + 3 : offset + 6]):
            rotation = Rotation.from_rotvec(rotvec).as_matrix() @ rotation
            output[index, offset + 3 : offset + 6] = Rotation.from_matrix(
                rotation
            ).as_rotvec()
    return output


@dataclasses.dataclass(frozen=True)
class DeliveryAbsoluteEEFToChunkOrigin(transforms.DataTransformFn):
    """Map 10D absolute EEF targets to 7D actions anchored at current state."""

    arm_count: int

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            data["actions"] = _absolute_eef_targets_to_chunk_origin(
                np.asarray(data["state"]),
                np.asarray(data["actions"]),
                arm_count=self.arm_count,
            )
        return data


@dataclasses.dataclass(frozen=True)
class DeliveryStepDeltasToChunkOrigin(transforms.DataTransformFn):
    """Convert explicit legacy one-step EEF deltas to chunk-origin deltas."""

    arm_count: int

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            if callable(STEP_DELTAS_TO_CHUNK_ORIGIN):
                converted = STEP_DELTAS_TO_CHUNK_ORIGIN(
                    data["actions"], arm_count=self.arm_count
                )
            else:
                converted = _fallback_step_deltas_to_chunk_origin(
                    data["actions"], arm_count=self.arm_count
                )
            data["actions"] = np.asarray(converted, dtype=np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class JointGripperMetersToFraction(transforms.DataTransformFn):
    """Convert legacy-v2 joint state/action gripper metres before norm."""

    arm_count: int
    gripper_max_m: float = PIPER_GRIPPER_MAX_M

    def __call__(self, data: dict) -> dict:
        if self.gripper_max_m <= 0:
            raise ValueError("gripper_max_m must be positive")
        state = np.asarray(data["state"], dtype=np.float32).copy()
        for arm in range(self.arm_count):
            state[arm * 7 + 6] = np.clip(
                state[arm * 7 + 6] / self.gripper_max_m, 0.0, 1.0
            )
        data["state"] = state
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32).copy()
            for arm in range(self.arm_count):
                actions[..., arm * 7 + 6] = np.clip(
                    actions[..., arm * 7 + 6] / self.gripper_max_m, 0.0, 1.0
                )
            data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}


@dataclasses.dataclass(frozen=True)
class RemoveStrings(transforms.DataTransformFn):
    """Drop prompt/string fields before numeric norm-stat accumulation."""

    def __call__(self, sample: dict) -> dict:
        return {
            key: value
            for key, value in sample.items()
            if not np.issubdtype(np.asarray(value).dtype, np.str_)
        }


@dataclasses.dataclass(frozen=True)
class PiperDataConfig(training_config.DataConfigFactory):
    contract: DatasetContract | None = None
    default_prompt: str | None = None

    def create(self, assets_dirs: Path, model_config) -> training_config.DataConfig:
        if self.contract is None:
            raise ValueError("dataset contract is required")
        if self.contract.layout == "legacy":
            repack_mapping = {
                "images": {"cam_high": "image", "cam_wrist": "wrist_image"},
                "state": "state",
                "actions": "actions",
                "prompt": "prompt",
            }
            action_sequence_keys = ("actions",)
        else:
            repack_mapping = {
                "images": {
                    key: f"observation.images.{key}" for key in self.contract.camera_keys
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            }
            action_sequence_keys = ("action",)

        repack = transforms.Group(inputs=[transforms.RepackTransform(repack_mapping)])
        robot_transforms = transforms.Group(
            inputs=[PiperInputs(contract=self.contract)],
            outputs=[PiperOutputs(action_dim=self.contract.model_action_dim)],
        )
        arm_count = 2 if self.contract.arm_mode == "bimanual" else 1
        if self.contract.schema == "joint":
            if (
                self.contract.raw_gripper_semantics == GRIPPER_OPENING_METERS
                and self.contract.model_gripper_semantics == GRIPPER_OPENING_FRACTION
            ):
                robot_transforms = robot_transforms.push(
                    inputs=[JointGripperMetersToFraction(arm_count=arm_count)]
                )
            mask = (
                transforms.make_bool_mask(6, -1, 6, -1)
                if self.contract.arm_mode == "bimanual"
                else transforms.make_bool_mask(6, -1)
            )
            robot_transforms = robot_transforms.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
        elif self.contract.raw_action_convention == DELIVERY_ABSOLUTE_EEF_ACTION_CONVENTION:
            robot_transforms = robot_transforms.push(
                inputs=[DeliveryAbsoluteEEFToChunkOrigin(arm_count=arm_count)]
            )
        elif self.contract.model_action_convention == DELIVERY_CHUNK_ORIGIN_ACTION_CONVENTION:
            robot_transforms = robot_transforms.push(
                inputs=[DeliveryStepDeltasToChunkOrigin(arm_count=arm_count)]
            )
        elif self.contract.model_action_convention != DELIVERY_STEP_ACTION_CONVENTION:
            raise ValueError(
                "legacy delivery model convention must resolve to step or chunk_origin, got "
                f"{self.contract.model_action_convention!r}"
            )
        model_transforms = training_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=robot_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=action_sequence_keys,
        )


def build_config(args: argparse.Namespace) -> training_config.TrainConfig:
    contract = resolve_dataset_contract(args)
    model_variant = str(getattr(args, "model_variant", "pi05"))
    delivery_action_convention = _resolve_delivery_action_convention(
        args, contract=contract
    )
    contract = contract.with_model_action_convention(delivery_action_convention)
    contract = contract.with_model_gripper_semantics(
        _resolve_model_gripper_semantics(args, contract=contract)
    )
    _validate_requested_contract(args, contract)
    requested_gripper = getattr(args, "gripper_semantics", None)
    if requested_gripper not in (None, "", "auto") and str(requested_gripper) != contract.model_gripper_semantics:
        raise ValueError(
            f"--gripper-semantics={requested_gripper!r} conflicts with model value "
            f"{contract.model_gripper_semantics!r}"
        )
    requested_raw_gripper = getattr(args, "raw_gripper_semantics", None)
    if requested_raw_gripper not in (None, "", "auto") and str(requested_raw_gripper) != contract.raw_gripper_semantics:
        raise ValueError(
            f"--raw-gripper-semantics={requested_raw_gripper!r} conflicts with dataset value "
            f"{contract.raw_gripper_semantics!r}"
        )
    _validate_checkpoint_contract(args, contract)

    model = pi0_config.Pi0Config(
        pi05=model_variant == "pi05",
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    params_path = base_checkpoint / "params"
    if args.command == "train" and not params_path.exists():
        raise FileNotFoundError(
            f"base checkpoint params not found: {params_path}. "
            f"Install a compatible {model_variant} checkpoint or choose another base model."
        )
    data_factory = PiperDataConfig(
        repo_id=args.dataset_id,
        contract=contract,
        base_config=training_config.DataConfig(prompt_from_task=True),
    )
    wire_action_convention = (
        JOINT_RAW_ACTION_CONVENTION
        if contract.schema == "joint"
        else contract.model_action_convention
    )
    return training_config.TrainConfig(
        name=config_name(model_variant, contract.arm_mode),
        exp_name=getattr(args, "exp_name", "runtime"),
        model=model,
        data=data_factory,
        freeze_filter=model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(params_path)),
        assets_base_dir=str(Path(args.assets_base_dir).expanduser().resolve()),
        checkpoint_base_dir=str(Path(args.checkpoint_base_dir).expanduser().resolve()),
        batch_size=getattr(args, "batch_size", 8),
        num_workers=getattr(args, "num_workers", 2),
        num_train_steps=getattr(args, "num_train_steps", 30_000),
        save_interval=getattr(args, "save_interval", 1_000),
        log_interval=getattr(args, "log_interval", 100),
        fsdp_devices=getattr(args, "fsdp_devices", 1),
        resume=getattr(args, "resume", False),
        overwrite=getattr(args, "overwrite", False),
        wandb_enabled=getattr(args, "wandb_enabled", False),
        policy_metadata={
            "robot_type": "piper_bimanual" if contract.arm_mode == "bimanual" else "piper_single_arm",
            "model_variant": model_variant,
            "base_checkpoint": str(base_checkpoint),
            "dataset_id": args.dataset_id,
            "arm_mode": contract.arm_mode,
            "arm_side": contract.arm_side,
            "schema": contract.schema,
            "dataset_layout": contract.layout,
            "contract_version": contract.contract_version,
            "state_dim": contract.state_dim,
            "action_dim": contract.model_action_dim,
            "raw_action_dim": contract.raw_action_dim,
            "model_action_dim": contract.model_action_dim,
            "camera_keys": list(contract.camera_keys),
            "action_hz": contract.action_hz,
            "action_horizon": int(model.action_horizon),
            "action_time_step_s": (1.0 / contract.action_hz) if contract.action_hz else None,
            "action_start_offset_steps": contract.model_action_start_offset,
            "action_offset": contract.action_offset,
            "model_action_start_offset": contract.model_action_start_offset,
            "minimum_horizon": MIN_EXECUTION_ACTION_HORIZON,
            "recommended_inference_launch_hz": DEFAULT_ASYNC_INFERENCE_LAUNCH_HZ,
            "action_semantics": contract.wire_action_semantics,
            "action_convention": wire_action_convention,
            "wire_action_semantics": contract.wire_action_semantics,
            "wire_action_convention": wire_action_convention,
            "raw_action_semantics": contract.raw_action_semantics,
            "model_action_semantics": contract.model_action_semantics,
            "dataset_action_semantics": contract.raw_action_semantics,
            "raw_action_convention": contract.raw_action_convention,
            "model_action_convention": contract.model_action_convention,
            "delivery_action_convention": delivery_action_convention,
            "raw_gripper_semantics": contract.raw_gripper_semantics,
            "model_gripper_semantics": contract.model_gripper_semantics,
            "wire_gripper_semantics": contract.wire_gripper_semantics,
            "state_gripper_semantics": contract.raw_gripper_semantics,
            "gripper_semantics": contract.wire_gripper_semantics,
            "legacy_delivery_v2": contract.legacy_delivery,
            "legacy_joint_v2": (
                contract.schema == "joint"
                and contract.raw_gripper_semantics == GRIPPER_OPENING_METERS
            ),
            "action_source": contract.action_source,
            "action_alignment": contract.action_alignment,
            "transport": "openpi_websocket_v1",
        },
    )


def _resolve_training_split(
    args: argparse.Namespace, contract: dict[str, int | str]
) -> EpisodeSplit:
    requested_ratio = getattr(args, "test_ratio", None)
    requested_seed = getattr(args, "split_seed", None)
    persisted = load_episode_split(
        _dataset_root(), args.dataset_id, contract=contract
    )
    if requested_ratio is None and requested_seed is None and persisted is not None:
        split = persisted
        source = "persisted dataset split"
    else:
        split = resolve_episode_split(
            _dataset_root(),
            args.dataset_id,
            test_ratio=float(
                requested_ratio
                if requested_ratio is not None
                else persisted.test_ratio if persisted is not None else DEFAULT_TEST_RATIO
            ),
            seed=int(
                requested_seed
                if requested_seed is not None
                else persisted.seed if persisted is not None else DEFAULT_SPLIT_SEED
            ),
            contract=contract,
        )
        source = "requested split" if requested_ratio is not None or requested_seed is not None else "default split"
    print(
        "Episode split: "
        f"train={len(split.train_episodes)} test={len(split.test_episodes)} "
        f"ratio={split.test_ratio:g} seed={split.seed} "
        f"source={source} test_episodes={list(split.test_episodes)}",
        flush=True,
    )
    if not split.test_episodes:
        logging.warning("dataset has fewer than two episodes or test_ratio=0; no held-out test episodes")
    return split


def action_delta_timestamps(
    action_horizon: int,
    fps: float,
    action_offset: int,
) -> list[float]:
    action_horizon = int(action_horizon)
    fps = float(fps)
    action_offset = int(action_offset)
    if action_horizon <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("action_horizon and fps must be positive")
    if action_offset not in {0, 1}:
        raise ValueError("action_offset must be 0 or 1")
    return [(step + 1 - action_offset) / fps for step in range(action_horizon)]


def _create_torch_dataset_for_episodes(
    data_config: training_config.DataConfig,
    action_horizon: int,
    model_config: Any,
    episodes: tuple[int, ...],
    *,
    action_offset: int,
):
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return data_loader.FakeDataset(model_config, num_samples=1024)
    dataset_meta = data_loader.lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = data_loader.lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: action_delta_timestamps(action_horizon, dataset_meta.fps, action_offset)
            for key in data_config.action_sequence_keys
        },
    )
    # This pinned LeRobot version accepts non-contiguous episode lists but its
    # delta-query indexing still assumes contiguous episode ids. Keep the full
    # dataset's boundary table and select frame indices at the PyTorch layer.
    selected_episodes = set(episodes)
    sample_indices = [
        index
        for index, episode_index in enumerate(dataset.hf_dataset["episode_index"])
        if int(episode_index) in selected_episodes
    ]
    if not sample_indices:
        raise ValueError(f"episode subset for {repo_id!r} contains no frames")
    dataset = data_loader.torch.utils.data.Subset(dataset, sample_indices)
    if data_config.prompt_from_task:
        dataset = data_loader.TransformedDataset(
            dataset,
            [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )
    return dataset


def _install_training_episode_subset(
    dataset_id: str, episodes: tuple[int, ...], *, action_offset: int
) -> None:
    """Make upstream OpenPI's training loader consume only selected episodes."""
    original = data_loader.create_torch_dataset

    @functools.wraps(original)
    def create_subset(data_config, action_horizon, model_config):
        if data_config.repo_id != dataset_id:
            return original(data_config, action_horizon, model_config)
        return _create_torch_dataset_for_episodes(
            data_config, action_horizon, model_config, episodes, action_offset=action_offset
        )

    data_loader.create_torch_dataset = create_subset


def _load_upstream_train_main(openpi_root: Path):
    train_path = openpi_root / "scripts" / "train.py"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    spec = importlib.util.spec_from_file_location("openpi_upstream_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


def _write_extended_contract_fields(
    path: Path, contract: dict[str, int | str]
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot update action contract artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"action contract artifact must be a JSON object: {path}")
    payload.update(contract)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _require_extended_contract_fields(
    path: Path, contract: dict[str, int | str]
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"normalization contract is missing or invalid: {path}; recompute norm stats"
        ) from exc
    mismatches = {
        key: {"saved": payload.get(key), "expected": value}
        for key, value in contract.items()
        if payload.get(key) != value
    }
    if payload.get("version") != NORM_CONFIG_VERSION:
        mismatches["version"] = {
            "saved": payload.get("version"),
            "expected": NORM_CONFIG_VERSION,
        }
    if mismatches:
        raise ValueError(
            "normalization action/time contract mismatch; recompute norm stats: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def run_train(args: argparse.Namespace) -> None:
    config = build_config(args)
    contract_fingerprint = complete_action_contract_fingerprint(config.policy_metadata)
    split = _resolve_training_split(args, contract_fingerprint)
    _require_extended_contract_fields(
        config.assets_dirs / args.dataset_id / NORM_CONFIG_FILENAME,
        contract_fingerprint,
    )
    _install_training_episode_subset(
        args.dataset_id,
        split.train_episodes,
        action_offset=int(config.policy_metadata["action_offset"]),
    )
    if config.checkpoint_dir.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"checkpoint directory already exists: {config.checkpoint_dir}; use --resume or --overwrite"
        )
    marker = _write_action_convention_marker(config)
    print(f"Writing action contract marker to: {marker}", flush=True)
    train_main = _load_upstream_train_main(Path.cwd())
    train_main(config)


def run_norm(args: argparse.Namespace) -> None:
    config = build_config(args)
    contract_fingerprint = complete_action_contract_fingerprint(config.policy_metadata)
    split = _resolve_training_split(args, contract_fingerprint)
    concrete = config.data.create(config.assets_dirs, config.model)
    dataset = _create_torch_dataset_for_episodes(
        concrete,
        config.model.action_horizon,
        config.model,
        split.train_episodes,
        action_offset=int(config.policy_metadata["action_offset"]),
    )

    dataset = data_loader.TransformedDataset(
        dataset,
        [*concrete.repack_transforms.inputs, *concrete.data_transforms.inputs, RemoveStrings()],
    )
    if len(dataset) <= 0:
        raise ValueError(f"dataset {args.dataset_id!r} is empty")
    available_train_frames = len(dataset)
    batch_size = min(args.batch_size, len(dataset))
    if args.max_frames is not None:
        effective_frames = min(args.max_frames, len(dataset))
        num_batches = max(1, effective_frames // batch_size)
        shuffle = effective_frames < len(dataset)
    else:
        num_batches = max(1, len(dataset) // batch_size)
        shuffle = False
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for index, batch in enumerate(loader, start=1):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
        if index % 20 == 0 or index == num_batches:
            print(f"norm stats: {index}/{num_batches} batches", flush=True)
    output = {key: value.get_statistics() for key, value in stats.items()}
    output_path = config.assets_dirs / args.dataset_id
    print(f"Writing stats to: {output_path}", flush=True)
    normalize.save(output_path, output)
    split_path = write_norm_split(output_path, split)
    print(f"Writing episode split manifest to: {split_path}", flush=True)
    config_path = write_norm_config(
        output_path,
        split,
        model_variant=args.model_variant,
        base_checkpoint=str(Path(args.base_checkpoint).expanduser().resolve()),
        arm_mode=str(config.policy_metadata["arm_mode"]),
        arm_side=str(config.policy_metadata["arm_side"]),
        schema=str(config.policy_metadata["schema"]),
        contract=contract_fingerprint,
        delivery_action_convention=config.policy_metadata.get("delivery_action_convention"),
        requested_batch_size=args.batch_size,
        effective_batch_size=batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        available_train_frames=available_train_frames,
        processed_batches=num_batches,
    )
    _write_extended_contract_fields(config_path, contract_fingerprint)
    print(f"Writing norm configuration to: {config_path}", flush=True)


def _first_client_value(client: dict[str, Any], *keys: str) -> Any:
    """Return the first explicitly supplied value, preserving false/zero values."""
    for key in keys:
        if key in client and client[key] is not None:
            return client[key]
    return None


def _telemetry_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _telemetry_nonnegative_float(value: Any) -> float | None:
    result = _telemetry_finite_float(value)
    return result if result is not None and result >= 0 else None


def _telemetry_positive_float(value: Any) -> float | None:
    result = _telemetry_nonnegative_float(value)
    return result if result is not None and result > 0 else None


def _telemetry_nonnegative_int(value: Any) -> int | None:
    # bool is deliberately accepted for flags elsewhere, not as a counter.
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _telemetry_positive_int(value: Any) -> int | None:
    result = _telemetry_nonnegative_int(value)
    return result if result is not None and result > 0 else None


def _telemetry_bool_or_count(value: Any) -> tuple[bool | None, int | None]:
    if isinstance(value, bool):
        return value, int(value)
    count = _telemetry_nonnegative_int(value)
    return (count > 0, count) if count is not None else (None, None)


def _telemetry_finite_number_list(
    value: Any, *, max_length: int = 512
) -> list[float] | None:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        result.ndim != 1
        or len(result) > max_length
        or not np.all(np.isfinite(result))
        or np.any(result < 0)
    ):
        return None
    return result.tolist()


def _telemetry_json_value(value: Any, *, action_dim: int, max_chars: int = 16_000) -> Any:
    """Keep JSON telemetry finite and bounded without changing action contracts."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        vector = PolicyTelemetry._finite_vector(value, action_dim)
        return vector
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded) > max_chars:
            return None
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError):
        return None


def sanitize_async_client_telemetry(
    client: dict[str, Any],
    *,
    action_dim: int,
    action_horizon: int,
) -> dict[str, Any]:
    """Sanitize the async policy-client telemetry contract.

    Input names intentionally accept the current bridge names and a few stable
    aliases so a Dashboard can be upgraded before/after the robot client. The
    returned names are the Dashboard-facing ``client_*`` names. Missing fields
    remain ``None`` rather than being guessed from an old synchronous queue.
    """
    if not isinstance(client, dict):
        client = {}
    try:
        action_dim = int(action_dim)
    except (TypeError, ValueError):
        action_dim = 0
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")

    result: dict[str, Any] = {}
    positive_float_fields = {
        "client_inference_launch_hz": ("inference_launch_hz", "launch_hz", "client_inference_launch_hz", "inference_hz"),
        "client_control_hz": ("control_hz", "action_hz", "client_control_hz", "command_hz"),
    }
    for output, keys in positive_float_fields.items():
        result[output] = _telemetry_positive_float(_first_client_value(client, *keys))

    nonnegative_float_fields = {
        "client_launch_at": ("launch_at", "inference_launched_at", "inference_launch_at", "client_launch_at"),
        "client_capture_at": ("capture_at", "captured_at", "observation_captured_at", "inference_capture_at", "client_capture_at"),
        "client_arrival_at": ("arrival_at", "result_arrived_at", "inference_arrival_at", "client_arrival_at"),
        "client_latency_ms": ("latency_ms", "inference_latency_ms", "client_latency_ms"),
        "client_latency_steps": ("latency_steps", "inference_latency_steps", "client_latency_steps"),
        "client_actuator_delay_ms": ("actuator_delay_ms", "command_actuator_delay_ms", "client_actuator_delay_ms"),
        "client_actuator_delay_steps": ("actuator_delay_steps", "command_actuator_delay_steps", "client_actuator_delay_steps"),
        "client_last_target_time_at": ("last_target_time_at", "last_action_target_at", "last_command_target_at", "client_last_target_time_at"),
        "client_next_target_time_at": ("next_target_time_at", "next_action_target_at", "client_next_target_time_at"),
        "client_active_plan_started_at": ("active_plan_started_at", "plan_started_at", "client_active_plan_started_at"),
        "client_timing_snapshot_at": ("timing_snapshot_at", "client_timing_snapshot_at"),
        "client_request_sent_at": ("request_sent_at", "client_request_sent_at"),
        "client_server_request_received_at": ("server_request_received_at", "client_server_request_received_at"),
        "client_server_model_completed_at": ("server_model_completed_at", "client_server_model_completed_at"),
        "client_server_response_ready_at": ("server_response_ready_at", "client_server_response_ready_at"),
        "client_response_received_at": ("response_received_at", "client_response_received_at"),
        "client_camera_capture_ms": ("camera_capture_ms", "client_camera_capture_ms"),
        "client_observation_upload_ms": ("observation_upload_ms", "client_observation_upload_ms"),
        "client_model_inference_ms": ("model_inference_ms", "client_model_inference_ms"),
        "client_result_download_ms": ("result_download_ms", "client_result_download_ms"),
        "client_network_transport_total_ms": ("network_transport_total_ms", "client_network_transport_total_ms"),
        "client_round_trip_ms": ("round_trip_ms", "client_round_trip_ms"),
        "client_result_to_first_command_ms": ("result_to_first_command_ms", "client_result_to_first_command_ms"),
        "client_observation_to_first_command_ms": ("observation_to_first_command_ms", "client_observation_to_first_command_ms"),
    }
    for output, keys in nonnegative_float_fields.items():
        result[output] = _telemetry_nonnegative_float(_first_client_value(client, *keys))
    latency_s = _telemetry_nonnegative_float(client.get("inference_latency_s"))
    if result["client_latency_ms"] is None and latency_s is not None:
        result["client_latency_ms"] = latency_s * 1000.0
    if result["client_latency_steps"] is None and latency_s is not None:
        control_hz = result["client_control_hz"]
        if control_hz is not None:
            result["client_latency_steps"] = latency_s * control_hz
    actuator_delay_s = _telemetry_nonnegative_float(
        _first_client_value(client, "actuator_delay_s", "estimated_actuator_delay_s")
    )
    if result["client_actuator_delay_ms"] is None and actuator_delay_s is not None:
        result["client_actuator_delay_ms"] = actuator_delay_s * 1000.0
    if result["client_actuator_delay_steps"] is None and actuator_delay_s is not None:
        control_hz = result["client_control_hz"]
        if control_hz is not None:
            result["client_actuator_delay_steps"] = actuator_delay_s * control_hz

    positive_int_fields = {
        "client_chunk_rows": ("chunk_rows", "returned_chunk_rows", "expected_action_horizon", "action_chunk_rows", "action_chunk_steps", "client_chunk_rows"),
        "client_minimum_horizon": ("minimum_horizon", "min_horizon", "min_action_chunk_steps", "client_minimum_horizon"),
    }
    for output, keys in positive_int_fields.items():
        result[output] = _telemetry_positive_int(_first_client_value(client, *keys))

    nonnegative_int_fields = {
        "client_skipped_prefix": ("skipped_prefix", "skipped_prefix_steps", "skip_prefix_steps", "inference_skip_steps", "client_skipped_prefix"),
        "client_blend_steps": ("blend_steps", "chunk_blend_steps", "inference_blend_steps", "client_blend_steps"),
        "client_queue_generation": ("queue_generation", "action_generation", "generation", "client_queue_generation"),
        "client_result_generation": ("result_generation", "inference_generation", "client_result_generation"),
        "client_timing_generation": ("timing_generation", "transport_timing_generation", "client_timing_generation"),
        "client_old_remaining": ("old_remaining", "old_chunk_remaining", "inference_old_remaining", "client_old_remaining"),
        "client_new_remaining": ("new_remaining", "new_chunk_remaining", "queued_action_count", "client_new_remaining"),
        "client_rejected_result_count": ("rejected_result_count", "client_rejected_result_count"),
        "client_underrun_count": ("underrun_count", "queue_underrun_count", "client_underrun_count"),
        "client_inference_launch_count": ("inference_launch_count", "client_inference_launch_count"),
        "client_inference_launch_deferred_count": ("inference_launch_deferred_count", "client_inference_launch_deferred_count"),
        "client_control_tick_count": ("control_tick_count", "client_control_tick_count"),
        "client_control_overrun_count": ("control_overrun_count", "client_control_overrun_count"),
        "client_expired_prefix": ("expired_prefix", "expired_prefix_steps", "dynamic_expired_prefix_steps", "inference_skip_steps", "client_expired_prefix"),
        "client_active_plan_generation": ("active_plan_generation", "plan_generation", "action_generation", "client_active_plan_generation"),
        "client_active_plan_index": ("active_plan_index", "plan_index", "queued_action_index", "client_active_plan_index"),
        "client_active_plan_remaining": ("active_plan_remaining", "plan_remaining", "queued_action_count", "client_active_plan_remaining"),
        "client_hold_steps": ("hold_steps", "hold_count", "client_hold_steps"),
        "client_blend_remaining": ("blend_remaining", "blend_steps_remaining", "client_blend_remaining"),
    }
    for output, keys in nonnegative_int_fields.items():
        result[output] = _telemetry_nonnegative_int(_first_client_value(client, *keys))

    result["client_action_horizon"] = _telemetry_positive_int(
        _first_client_value(client, "action_horizon", "expected_action_horizon", "horizon", "client_action_horizon")
    )
    policy_horizon = _telemetry_positive_int(action_horizon)
    result["client_horizon_matches_policy"] = (
        None
        if result["client_action_horizon"] is None or policy_horizon is None
        else result["client_action_horizon"] == policy_horizon
    )

    in_flight = _first_client_value(client, "in_flight", "inference_in_flight", "client_in_flight")
    if isinstance(in_flight, bool):
        result["client_in_flight"] = in_flight
    else:
        result["client_in_flight"] = _telemetry_nonnegative_int(in_flight)

    for output, keys in {
        "client_hold_active": ("hold_active", "holding", "client_hold_active"),
        "client_blend_active": ("blend_active", "blending", "client_blend_active"),
        "client_gripper_filter_active": ("gripper_filter_active", "gripper_filtering", "client_gripper_filter_active"),
    }.items():
        raw = _first_client_value(client, *keys)
        result[output] = raw if isinstance(raw, bool) else None
    result["client_hold_reason"] = str(
        _first_client_value(client, "hold_reason", "client_hold_reason") or ""
    )[:500]
    result["client_plan_target_times"] = _telemetry_finite_number_list(
        _first_client_value(
            client, "plan_target_times", "active_plan_target_times", "action_target_times",
            "client_plan_target_times"
        )
    )
    for output, keys in {
        "client_active_plan": ("active_plan", "plan_state", "client_active_plan"),
        "client_hold": ("hold", "hold_state", "client_hold"),
        "client_blend": ("blend", "blend_state", "client_blend"),
        "client_gripper_filter": ("gripper_filter", "gripper_filter_state", "client_gripper_filter"),
        "client_timed_target": ("timed_target", "current_timed_target", "client_timed_target"),
        "client_last_safe_target": ("last_safe_target", "client_last_safe_target"),
        "client_transport_timing": ("client_transport_timing", "transport_timing"),
    }.items():
        result[output] = _telemetry_json_value(
            _first_client_value(client, *keys), action_dim=action_dim
        )
    if result.get("client_timing_generation") is None:
        timing_payload = result.get("client_transport_timing")
        if isinstance(timing_payload, dict):
            result["client_timing_generation"] = _telemetry_nonnegative_int(
                timing_payload.get("generation")
            )

    timed_target = result.get("client_timed_target")
    if isinstance(timed_target, dict):
        result["client_target_monotonic"] = _telemetry_nonnegative_float(
            timed_target.get("target_monotonic")
        )
        result["client_target_at"] = _telemetry_nonnegative_float(
            timed_target.get("target_at")
        )
        result["client_target_age_s"] = _telemetry_finite_float(
            timed_target.get("target_time_error_s", timed_target.get("target_age_s"))
        )
        result["client_target_time_error_ms"] = (
            result["client_target_age_s"] * 1000.0
            if result["client_target_age_s"] is not None
            else None
        )
        if result.get("client_blend_active") is None:
            result["client_blend_active"] = bool(timed_target.get("blended", False))
        if result.get("client_hold_active") is None:
            result["client_hold_active"] = bool(timed_target.get("hold", False))
    else:
        result["client_target_monotonic"] = None
        result["client_target_at"] = None
        result["client_target_age_s"] = None
        result["client_target_time_error_ms"] = None
    if result.get("client_gripper_filter") is not None and result.get("client_gripper_filter_active") is None:
        result["client_gripper_filter_active"] = True

    result["client_underrun"] = None
    for source in ("underrun", "queue_underrun"):
        flag, count = _telemetry_bool_or_count(client.get(source))
        if flag is not None:
            result["client_underrun"] = flag
            if count is not None and result.get("client_underrun_count") is None:
                result["client_underrun_count"] = count
            break

    rejected_raw = _first_client_value(client, "rejected_result", "client_rejected_result")
    rejected_json = _telemetry_json_value(rejected_raw, action_dim=action_dim)
    if isinstance(rejected_raw, (dict, list, tuple, np.ndarray)):
        result["client_rejected_result"] = rejected_json
        result["client_rejected_result_active"] = rejected_json is not None
    else:
        rejected_flag, rejected_count = _telemetry_bool_or_count(rejected_raw)
        result["client_rejected_result"] = rejected_flag
        result["client_rejected_result_active"] = rejected_flag
        if rejected_count is not None and result.get("client_rejected_result_count") is None:
            result["client_rejected_result_count"] = rejected_count

    drop_reason = _first_client_value(
        client, "drop_reason", "queue_drop_reason", "last_queue_drop_reason", "client_drop_reason"
    )
    if not drop_reason and isinstance(rejected_json, dict):
        drop_reason = rejected_json.get("reason")
    result["client_drop_reason"] = str(drop_reason or "")[:500]
    result["client_last_wire_action"] = _telemetry_json_value(
        _first_client_value(client, "last_wire_action", "wire_action", "client_last_wire_action"),
        action_dim=action_dim,
    )
    result["client_last_decoded_target"] = _telemetry_json_value(
        _first_client_value(client, "last_decoded_target", "last_decoded_absolute_target", "decoded_target", "client_last_decoded_target"),
        action_dim=action_dim,
    )
    result["client_safety_profile"] = str(client.get("safety_profile", ""))[:256]
    result["client_delivery_safety_limits"] = _telemetry_json_value(
        client.get("delivery_safety_limits"), action_dim=action_dim
    )
    result["client_async_telemetry_present"] = any(value is not None for value in result.values())
    return result


class PolicyTelemetry:
    """Mirror official WebSocket requests/results for the local dashboard."""

    def __init__(self, root: Path, metadata: dict[str, Any]):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.lock = threading.Lock()
        self.sequence = 0
        self.active_clients = 0
        self.client_addresses: set[str] = set()
        self.active_inferences = 0
        self.last_inference_started_at: float | None = None
        self.last_inference_finished_at: float | None = None

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)

    @staticmethod
    def _atomic_image(path: Path, image: np.ndarray) -> list[int]:
        from PIL import Image

        image = _as_hwc_uint8(image)
        temp = path.with_suffix(path.suffix + ".tmp")
        Image.fromarray(image).save(temp, format="JPEG", quality=90)
        os.replace(temp, path)
        return list(image.shape)

    @staticmethod
    def _client_address(remote_address: Any) -> str:
        if isinstance(remote_address, tuple):
            return ":".join(map(str, remote_address))
        return str(remote_address or "unknown")

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result) else None

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            result = int(value)
        except (TypeError, ValueError):
            return None
        return result if result > 0 else None

    @staticmethod
    def _nonnegative_int(value: Any) -> int | None:
        try:
            result = int(value)
        except (TypeError, ValueError):
            return None
        return result if result >= 0 else None

    @staticmethod
    def _json_object(value: Any, *, max_chars: int = 16_000) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return None
        if len(encoded) > max_chars:
            return None
        decoded = json.loads(encoded)
        return decoded if isinstance(decoded, dict) else None

    @staticmethod
    def _finite_vector(value: Any, expected_dim: int) -> list[float] | None:
        try:
            result = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if result.shape != (expected_dim,) or not np.all(np.isfinite(result)):
            return None
        return result.tolist()

    def _publish_connections(self, *, event: str, address: str) -> None:
        now = time.time()
        payload = {
            "event": event,
            "address": address,
            "updated_at": now,
            "active_clients": self.active_clients,
            "client_connected": self.active_clients > 0,
            "client_addresses": sorted(self.client_addresses),
        }
        self._atomic_json(self.root / "connections.json", payload)

    def _publish_runtime(self) -> None:
        self._atomic_json(
            self.root / "runtime.json",
            {
                "active_inferences": self.active_inferences,
                "in_flight": self.active_inferences > 0,
                "last_inference_started_at": self.last_inference_started_at,
                "last_inference_finished_at": self.last_inference_finished_at,
                "updated_at": time.time(),
            },
        )

    def inference_started(self) -> None:
        with self.lock:
            self.active_inferences += 1
            self.last_inference_started_at = time.time()
            self._publish_runtime()

    def inference_finished(self) -> None:
        with self.lock:
            self.active_inferences = max(0, self.active_inferences - 1)
            self.last_inference_finished_at = time.time()
            self._publish_runtime()

    def client_opened(self, remote_address: Any) -> None:
        address = self._client_address(remote_address)
        with self.lock:
            self.active_clients += 1
            self.client_addresses.add(address)
            self._publish_connections(event="connected", address=address)

    def client_closed(self, remote_address: Any) -> None:
        address = self._client_address(remote_address)
        with self.lock:
            self.active_clients = max(0, self.active_clients - 1)
            self.client_addresses.discard(address)
            self._publish_connections(event="disconnected", address=address)

    def execution_control(self) -> dict[str, Any]:
        """Read the Dashboard gate; missing, malformed, or expired means shadow."""
        path = self.root / "execution_control.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            value = {}
        if not isinstance(value, dict):
            value = {}
        now = time.time()
        requested_mode = value.get("mode") if value.get("mode") in {"shadow", "execute"} else "shadow"
        expires_at = value.get("expires_at")
        try:
            expires_at = float(expires_at) if expires_at is not None else None
        except (TypeError, ValueError):
            expires_at = None
        expired = requested_mode == "execute" and (expires_at is None or expires_at <= now)
        return {
            "mode": "shadow" if expired else requested_mode,
            "requested_mode": requested_mode,
            "revision": int(value.get("revision", 0)),
            "updated_at": value.get("updated_at"),
            "expires_at": expires_at,
            "expired": expired,
            "task_id": value.get("task_id"),
            "session_id": value.get("session_id", self.root.name),
            "server_time": now,
        }

    def publish(self, observation: dict, result: dict, elapsed_s: float) -> int:
        with self.lock:
            self.sequence += 1
            client = observation.get("client_metadata")
            if not isinstance(client, dict):
                client = {}
            images = observation.get("images", {})
            camera_shapes: dict[str, list[int]] = {}
            for camera_key in self.metadata["camera_keys"]:
                camera_shapes[camera_key] = self._atomic_image(
                    self.root / f"{camera_key}.jpg", images[camera_key]
                )
            if self.metadata["arm_mode"] == "single":
                wrist_key = next(key for key in self.metadata["camera_keys"] if "wrist" in key)
                if wrist_key != "cam_wrist":
                    self._atomic_image(self.root / "cam_wrist.jpg", images[wrist_key])
            actions = np.asarray(result.get("actions"), dtype=np.float32)
            state = np.asarray(observation.get("state"), dtype=np.float32)
            action_horizon = self._positive_int(self.metadata.get("action_horizon"))
            if action_horizon is None:
                action_horizon = 0
            async_client = sanitize_async_client_telemetry(
                client,
                action_dim=int(self.metadata["action_dim"]),
                action_horizon=action_horizon,
            )
            if actions.ndim >= 2 and int(actions.shape[0]) > 0:
                # The returned tensor is authoritative for displayed chunk rows;
                # expected_action_horizon remains the negotiated client contract.
                async_client["client_chunk_rows"] = int(actions.shape[0])
            client_policy_action_hz = _telemetry_positive_float(client.get("policy_action_hz"))
            client_command_hz = async_client["client_control_hz"]
            client_inference_hz = async_client["client_inference_launch_hz"]
            client_action_chunk_steps = async_client["client_chunk_rows"]
            client_last_action_chunk_steps = self._positive_int(
                client.get("last_action_chunk_steps")
            )
            client_last_composed_action = self._finite_vector(
                client.get("last_composed_action"), int(self.metadata["action_dim"])
            )
            client_last_composed_action_at = self._finite_float(
                client.get("last_composed_action_at")
            )
            client_queue_anchor_state = self._finite_vector(
                client.get("queue_anchor_state"), int(self.metadata["state_dim"])
            )
            expected_qpos_dim = 14 if self.metadata["arm_mode"] == "bimanual" else 7
            client_queue_anchor_qpos = self._finite_vector(
                client.get("queue_anchor_qpos_m"), expected_qpos_dim
            )
            client_last_wire_action = async_client["client_last_wire_action"]
            now = time.time()
            captured_at = _telemetry_nonnegative_float(client.get("captured_at"))
            transport_timing = result.get("transport_timing")
            transport_timing = transport_timing if isinstance(transport_timing, dict) else {}
            server_model_inference_ms = _telemetry_nonnegative_float(
                transport_timing.get("model_inference_ms")
            )
            server_observation_upload_ms = _telemetry_nonnegative_float(
                transport_timing.get("observation_upload_ms")
            )
            payload = {
                "sequence": self.sequence,
                "received_at": now,
                "captured_at": captured_at if captured_at is not None else now,
                "source_name": str(client.get("source_name", "official-openpi-client"))[:256],
                "can_name": str(client.get("can_name", ""))[:256],
                "cam_high_device": str(client.get("cam_high_device", ""))[:256],
                "cam_wrist_device": str(client.get("cam_wrist_device", ""))[:256],
                "client_allow_execution": bool(client.get("allow_execution", False)),
                "client_execution_state": str(client.get("execution_state", "unknown"))[:64],
                "client_blocked_reason": str(client.get("blocked_reason", ""))[:500],
                "client_last_command_at": _telemetry_nonnegative_float(client.get("last_command_at")),
                "client_control_revision": self._nonnegative_int(client.get("control_revision")),
                "action_hz": self.metadata.get("action_hz"),
                "action_horizon": action_horizon or None,
                "action_offset": self.metadata.get("action_offset"),
                "model_action_start_offset": self.metadata.get("model_action_start_offset"),
                "action_time_step_s": self.metadata.get("action_time_step_s"),
                "action_start_offset_steps": self.metadata.get("action_start_offset_steps"),
                "minimum_horizon": MIN_EXECUTION_ACTION_HORIZON,
                "recommended_inference_launch_hz": self.metadata.get(
                    "recommended_inference_launch_hz", DEFAULT_ASYNC_INFERENCE_LAUNCH_HZ
                ),
                "client_policy_action_hz": client_policy_action_hz,
                "client_command_hz": client_command_hz,
                "client_inference_hz": client_inference_hz,
                "client_action_chunk_steps": client_action_chunk_steps,
                "client_last_action_chunk_steps": client_last_action_chunk_steps,
                "client_last_composed_action": client_last_composed_action,
                "client_last_composed_action_at": client_last_composed_action_at,
                "client_queue_anchor_state": client_queue_anchor_state,
                "client_queue_anchor_qpos_m": client_queue_anchor_qpos,
                "client_queue_anchor_at": self._finite_float(client.get("queue_anchor_at")),
                "client_queue_loaded_at": self._finite_float(client.get("queue_loaded_at")),
                "client_queued_action_count": self._nonnegative_int(
                    client.get("queued_action_count")
                ),
                "client_queued_action_index": self._nonnegative_int(
                    client.get("queued_action_index")
                ),
                "client_last_queued_action_index": self._nonnegative_int(
                    client.get("last_queued_action_index")
                ),
                "client_last_wire_action": client_last_wire_action,
                "client_last_decoded_absolute_target": async_client[
                    "client_last_decoded_target"
                ],
                "client_last_feedback_at": self._finite_float(client.get("last_feedback_at")),
                "client_dropped_action_count": self._nonnegative_int(
                    client.get("dropped_action_count")
                ),
                "client_unqueued_action_count": self._nonnegative_int(
                    client.get("unqueued_action_count")
                ),
                "client_last_queue_drop_reason": async_client["client_drop_reason"],
                **async_client,
                "robot_arm_status": client.get("robot_arm_status"),
                "schema": self.metadata["schema"],
                "arm_mode": self.metadata["arm_mode"],
                "arm_side": self.metadata["arm_side"],
                "contract_version": self.metadata.get("contract_version"),
                "raw_action_dim": self.metadata.get("raw_action_dim"),
                "model_action_dim": self.metadata.get("model_action_dim"),
                "raw_action_semantics": self.metadata.get("raw_action_semantics"),
                "model_action_semantics": self.metadata.get("model_action_semantics"),
                "wire_action_semantics": self.metadata.get("wire_action_semantics"),
                "raw_action_convention": self.metadata.get("raw_action_convention"),
                "model_action_convention": self.metadata.get("model_action_convention"),
                "wire_action_convention": self.metadata.get("wire_action_convention"),
                "raw_gripper_semantics": self.metadata.get("raw_gripper_semantics"),
                "model_gripper_semantics": self.metadata.get("model_gripper_semantics"),
                "wire_gripper_semantics": self.metadata.get("wire_gripper_semantics"),
                "state_gripper_semantics": self.metadata.get("state_gripper_semantics"),
                "transport": "openpi_websocket_v1",
                "state": state.tolist(),
                "state_dim": int(state.shape[-1]),
                "prompt": str(observation.get("prompt", ""))[:500],
                "camera_shapes": camera_shapes,
                "cam_high_shape": camera_shapes.get("cam_high"),
                "cam_wrist_shape": next(
                    (shape for key, shape in camera_shapes.items() if "wrist" in key), None
                ) if self.metadata["arm_mode"] == "single" else None,
                "actions_shape": list(actions.shape),
                "first_action": actions[0].tolist() if actions.ndim > 1 and len(actions) else actions.tolist(),
                "action_min": float(actions.min()) if actions.size else None,
                "action_max": float(actions.max()) if actions.size else None,
                "policy_elapsed_s": elapsed_s,
                "server_model_inference_ms": server_model_inference_ms,
                "server_timing_generation": _telemetry_nonnegative_int(
                    transport_timing.get("inference_generation")
                ),
                "server_observation_upload_ms": server_observation_upload_ms,
                "server_request_received_at": _telemetry_nonnegative_float(
                    transport_timing.get("server_request_received_at")
                ),
                "server_model_completed_at": _telemetry_nonnegative_float(
                    transport_timing.get("server_model_completed_at")
                ),
                "transport_timing": _telemetry_json_value(
                    transport_timing, action_dim=int(self.metadata["action_dim"])
                ),
                "server_timing": result.get("server_timing"),
                "policy_timing": result.get("policy_timing"),
                "execution_control": result.get("execution_control"),
            }
            self._atomic_json(self.root / "latest.json", payload)
            return self.sequence

    def mark_response_ready(self, sequence: int, response_ready_at: float) -> None:
        """Attach the post-publish response boundary to the same telemetry row."""
        with self.lock:
            path = self.root / "latest.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return
            if not isinstance(payload, dict) or int(payload.get("sequence", -1)) != int(sequence):
                return
            transport_timing = payload.get("transport_timing")
            if not isinstance(transport_timing, dict):
                transport_timing = {}
            transport_timing["server_response_ready_at"] = float(response_ready_at)
            payload["transport_timing"] = transport_timing
            payload["server_response_ready_at"] = float(response_ready_at)
            self._atomic_json(path, payload)


class TelemetryWebsocketPolicyServer(websocket_policy_server.WebsocketPolicyServer):
    """Keep the official wire protocol while mirroring client lifecycle events."""

    def __init__(self, *args: Any, telemetry: PolicyTelemetry, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.telemetry = telemetry

    async def _handler(self, websocket: Any) -> None:
        self.telemetry.client_opened(websocket.remote_address)
        try:
            await super()._handler(websocket)
        finally:
            self.telemetry.client_closed(websocket.remote_address)


class TelemetryPolicy:
    def __init__(self, policy: Any, telemetry: PolicyTelemetry):
        self.policy = policy
        self.telemetry = telemetry

    def infer(self, observation: dict) -> dict:
        server_request_received_at = time.time()
        started = time.monotonic()
        self.telemetry.inference_started()
        try:
            result = dict(self.policy.infer(observation))
            model_inference_s = time.monotonic() - started
            server_model_completed_at = time.time()
            client = observation.get("client_metadata")
            client = client if isinstance(client, dict) else {}
            request_sent_at = _telemetry_nonnegative_float(client.get("request_sent_at"))
            request_generation = _telemetry_nonnegative_int(client.get("inference_generation"))
            observation_upload_ms = (
                max(0.0, (server_request_received_at - request_sent_at) * 1000.0)
                if request_sent_at is not None
                and server_request_received_at >= request_sent_at
                else None
            )
            transport_timing = {
                "client_request_sent_at": request_sent_at,
                "inference_generation": request_generation,
                "server_request_received_at": server_request_received_at,
                "server_model_completed_at": server_model_completed_at,
                "model_inference_ms": model_inference_s * 1000.0,
                "observation_upload_ms": observation_upload_ms,
            }
            result["transport_timing"] = transport_timing
            result["execution_control"] = self.telemetry.execution_control()
            try:
                sequence = self.telemetry.publish(observation, result, model_inference_s)
                response_ready_at = time.time()
                transport_timing["server_response_ready_at"] = response_ready_at
                self.telemetry.mark_response_ready(sequence, response_ready_at)
            except Exception:
                logging.exception("failed to publish policy telemetry")
                transport_timing["server_response_ready_at"] = time.time()
            return result
        finally:
            self.telemetry.inference_finished()

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            reset()


def run_serve(args: argparse.Namespace) -> None:
    _install_websocket_probe_filter()
    config = build_config(args)
    policy = policy_config.create_trained_policy(
        config,
        Path(args.checkpoint).expanduser().resolve(),
        default_prompt=args.default_prompt,
    )
    if args.telemetry_dir:
        telemetry = PolicyTelemetry(Path(args.telemetry_dir).expanduser().resolve(), config.policy_metadata)
        policy = TelemetryPolicy(policy, telemetry)
        server = TelemetryWebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=config.policy_metadata,
            telemetry=telemetry,
        )
    else:
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=config.policy_metadata,
        )
    server.serve_forever()


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--arm-mode", choices=("auto", "single", "bimanual"), default="auto")
    parser.add_argument("--arm-side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--schema", choices=("auto", "delivery", "joint"), default="auto")
    parser.add_argument("--dataset-layout", choices=("auto", "legacy", "canonical"), default="auto")
    parser.add_argument("--model-variant", choices=("pi05", "pi0"), default="pi05")
    parser.add_argument(
        "--delivery-action-convention",
        choices=("auto", *sorted(DELIVERY_ACTION_CONVENTIONS)),
        default="auto",
        help=(
            "delivery model/wire convention; auto uses chunk_origin for new training and "
            "requires markers or an explicit legacy selection for checkpoints"
        ),
    )
    parser.add_argument("--contract-version", type=int, default=None)
    parser.add_argument("--raw-action-dim", type=int, default=None)
    parser.add_argument("--model-action-dim", type=int, default=None)
    parser.add_argument("--raw-action-semantics", default=None)
    parser.add_argument("--model-action-semantics", default=None)
    parser.add_argument("--raw-action-convention", default=None)
    parser.add_argument("--model-action-convention", default=None)
    parser.add_argument("--action-offset", type=int, choices=(0, 1), default=None)
    parser.add_argument("--model-action-start-offset", type=int, default=None)
    parser.add_argument("--raw-gripper-semantics", default=None)
    parser.add_argument("--gripper-semantics", default=None)
    parser.add_argument(
        "--model-gripper-semantics",
        choices=(
            "auto",
            GRIPPER_OPENING_FRACTION,
            GRIPPER_OPENING_METERS,
            GRIPPER_CLOSED_FRACTION,
        ),
        default="auto",
    )
    parser.add_argument("--assets-base-dir", default="./assets")
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument(
        "--base-checkpoint",
        default=str(Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    norm = subparsers.add_parser("norm")
    add_common(norm)
    norm.add_argument("--batch-size", type=int, default=16)
    norm.add_argument("--num-workers", type=int, default=2)
    norm.add_argument("--max-frames", type=int, default=None)
    norm.add_argument("--test-ratio", type=float, default=0.1)
    norm.add_argument("--split-seed", type=int, default=42)

    train = subparsers.add_parser("train")
    add_common(train)
    train.add_argument("--exp-name", required=True)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--num-train-steps", type=int, default=30_000)
    train.add_argument("--save-interval", type=int, default=1_000)
    train.add_argument("--log-interval", type=int, default=100)
    train.add_argument("--fsdp-devices", type=int, default=1)
    train.add_argument("--test-ratio", type=float, default=None)
    train.add_argument("--split-seed", type=int, default=None)
    mode = train.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    train.add_argument("--wandb-enabled", action="store_true")

    serve = subparsers.add_parser("serve")
    add_common(serve)
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--default-prompt", default=None)
    serve.add_argument("--telemetry-dir", default=None)

    args = parser.parse_args()
    if not args.dataset_id or "/" in args.dataset_id or ".." in args.dataset_id:
        parser.error("--dataset-id must be a single safe LeRobot repository directory name")
    for name in ("batch_size", "num_workers", "num_train_steps", "save_interval", "log_interval", "fsdp_devices"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "test_ratio") and args.test_ratio is not None and not 0.0 <= args.test_ratio < 1.0:
        parser.error("--test-ratio must be in [0, 1)")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    {"norm": run_norm, "train": run_train, "serve": run_serve}[args.command](args)


if __name__ == "__main__":
    main()
