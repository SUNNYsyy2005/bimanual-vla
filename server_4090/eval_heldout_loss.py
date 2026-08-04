#!/usr/bin/env python3
"""Evaluate an OpenPI checkpoint on the persisted held-out episode split.

This process intentionally restores only model parameters (EMA weights saved in
``checkpoint/params``), not optimizer state, so it can run independently from
training on one otherwise-idle GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import data_loader

import openpi_single_arm as piper


RESULT_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _build_config_args(args: argparse.Namespace) -> argparse.Namespace:
    # Reuse serve-mode checkpoint marker validation. Eval consumes the same
    # model/data contract as inference but builds a held-out transformed loader.
    return argparse.Namespace(
        command="serve",
        dataset_id=args.dataset_id,
        arm_mode=args.arm_mode,
        arm_side=args.arm_side,
        schema=args.schema,
        dataset_layout=args.dataset_layout,
        model_variant=args.model_variant,
        delivery_action_convention=args.delivery_action_convention,
        contract_version=args.contract_version,
        raw_action_dim=args.raw_action_dim,
        model_action_dim=args.model_action_dim,
        raw_action_semantics=args.raw_action_semantics,
        model_action_semantics=args.model_action_semantics,
        raw_action_convention=args.raw_action_convention,
        model_action_convention=args.model_action_convention,
        action_offset=args.action_offset,
        model_action_start_offset=args.model_action_start_offset,
        raw_gripper_semantics=args.raw_gripper_semantics,
        gripper_semantics=args.gripper_semantics,
        model_gripper_semantics=args.model_gripper_semantics,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        base_checkpoint=args.base_checkpoint,
        checkpoint=str(args.checkpoint),
    )


def _create_test_loader(
    config: Any,
    checkpoint: Path,
    episodes: tuple[int, ...],
    *,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    eval_seed: int,
    data_sharding: jax.sharding.Sharding,
) -> tuple[Any, int, int]:
    concrete = config.data.create(config.assets_dirs, config.model)
    if concrete.asset_id is None:
        raise ValueError("data config has no asset_id")

    # Use the exact training norm stats copied into this checkpoint. The test
    # split must never compute or substitute its own normalization statistics.
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", concrete.asset_id)
    concrete = dataclasses.replace(concrete, norm_stats=norm_stats)
    raw_dataset = piper._create_torch_dataset_for_episodes(
        concrete,
        config.model.action_horizon,
        config.model,
        episodes,
        action_offset=int(config.policy_metadata["action_offset"]),
    )
    dataset = data_loader.transform_dataset(raw_dataset, concrete, skip_norm_stats=False)
    available_batches = len(dataset) // batch_size
    if available_batches <= 0:
        raise ValueError(
            f"held-out dataset has {len(dataset)} frames, smaller than batch_size={batch_size}"
        )
    effective_batches = min(max_batches, available_batches)
    torch_loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sharding=data_sharding,
        shuffle=True,
        sampler=None,
        num_batches=effective_batches,
        num_workers=num_workers,
        seed=eval_seed,
        framework="jax",
    )
    return data_loader.DataLoaderImpl(concrete, torch_loader), len(dataset), effective_batches


def _component_metrics(
    mean_per_dim: np.ndarray, *, model_action_dim: int, schema: str
) -> dict[str, float | int]:
    """Return semantic diagnostics for 7D-per-arm Piper model actions."""
    if model_action_dim <= 0 or model_action_dim % 7:
        return {}
    arms = model_action_dim // 7
    translation_indices = [base + offset for base in range(0, model_action_dim, 7) for offset in range(3)]
    rotation_indices = [base + offset for base in range(0, model_action_dim, 7) for offset in range(3, 6)]
    gripper_indices = [base + 6 for base in range(0, model_action_dim, 7)]
    first_name, second_name = (
        ("eval_loss_translation", "eval_loss_rotation")
        if schema == "delivery"
        else ("eval_loss_joint_first3", "eval_loss_joint_last3")
    )
    return {
        first_name: float(np.mean(mean_per_dim[translation_indices])),
        second_name: float(np.mean(mean_per_dim[rotation_indices])),
        "eval_loss_gripper": float(np.mean(mean_per_dim[gripper_indices])),
        "evaluated_arms": arms,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "_CHECKPOINT_METADATA").is_file():
        raise ValueError(f"incomplete Orbax checkpoint: {checkpoint}")
    if not (checkpoint / "params" / "_METADATA").is_file():
        raise ValueError(f"checkpoint params are incomplete: {checkpoint / 'params'}")
    try:
        checkpoint_step = int(checkpoint.name)
    except ValueError as exc:
        raise ValueError(f"checkpoint directory must be a numeric step: {checkpoint}") from exc

    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(
            f"held-out evaluation requires exactly one visible JAX device, got {len(devices)}: {devices}"
        )
    mesh = jax.sharding.Mesh(np.asarray(devices), ("eval",))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("eval"))

    config_args = _build_config_args(args)
    config = piper.build_config(config_args)
    contract = piper.complete_action_contract_fingerprint(config.policy_metadata)
    split = piper._resolve_training_split(config_args, contract)
    if not split.test_episodes:
        raise ValueError("persisted episode split contains no held-out test episodes")

    loader, available_frames, effective_batches = _create_test_loader(
        config,
        checkpoint,
        split.test_episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        eval_seed=args.eval_seed,
        data_sharding=data_sharding,
    )

    params = model_lib.restore_params(checkpoint / "params", sharding=replicated_sharding)
    model = config.model.load(params)
    eval_loss = nnx_utils.module_jit(model.compute_loss_per_dim)

    rng = jax.random.key(args.eval_seed)
    total_per_dim: np.ndarray | None = None
    total_action_rows = 0
    evaluated_batches = 0
    evaluated_samples = 0
    for observation, actions in loader:
        batch_rng = jax.random.fold_in(rng, evaluated_batches)
        per_dim = eval_loss(batch_rng, observation, actions, train=False)
        per_dim_sum = np.asarray(
            jax.device_get(jnp.sum(per_dim, axis=(0, 1))), dtype=np.float64
        )
        if total_per_dim is None:
            total_per_dim = np.zeros_like(per_dim_sum)
        total_per_dim += per_dim_sum
        batch_samples = int(actions.shape[0])
        evaluated_samples += batch_samples
        total_action_rows += batch_samples * int(actions.shape[1])
        evaluated_batches += 1
        print(
            f"Eval progress: {evaluated_batches}/{effective_batches} batches",
            flush=True,
        )

    if total_per_dim is None or evaluated_batches == 0 or total_action_rows == 0:
        raise RuntimeError("evaluation loader produced no batches")
    mean_per_dim = total_per_dim / total_action_rows
    if not np.all(np.isfinite(mean_per_dim)):
        raise RuntimeError("evaluation produced non-finite loss values")

    network_action_dim = int(mean_per_dim.shape[0])
    model_action_dim = int(config.policy_metadata["model_action_dim"])
    if not 0 < model_action_dim <= network_action_dim:
        raise ValueError(
            f"invalid model action dimension {model_action_dim} for network dimension {network_action_dim}"
        )
    configured_objective_dim = getattr(config.model, "loss_action_dim", None)
    objective_dim = int(configured_objective_dim or network_action_dim)
    objective_dim = min(max(1, objective_dim), network_action_dim)

    result: dict[str, Any] = {
        "version": RESULT_VERSION,
        "status": "completed",
        "dataset_id": args.dataset_id,
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "weights": "ema",
        "device": str(devices[0]),
        "eval_seed": args.eval_seed,
        "test_ratio": split.test_ratio,
        "split_seed": split.seed,
        "test_episode_indexes": list(split.test_episodes),
        "num_test_episodes": len(split.test_episodes),
        "available_test_frames": available_frames,
        "batch_size": args.batch_size,
        "requested_max_batches": args.max_batches,
        "evaluated_batches": evaluated_batches,
        "evaluated_samples": evaluated_samples,
        "action_horizon": int(config.model.action_horizon),
        "network_action_dim": network_action_dim,
        "model_action_dim": model_action_dim,
        "training_objective_dim": objective_dim,
        "eval_loss_model": float(np.mean(mean_per_dim[:model_action_dim])),
        "eval_loss_objective": float(np.mean(mean_per_dim[:objective_dim])),
        "eval_loss_all": float(np.mean(mean_per_dim)),
        "eval_loss_padding": (
            float(np.mean(mean_per_dim[model_action_dim:]))
            if network_action_dim > model_action_dim
            else None
        ),
        "eval_loss_per_dim": mean_per_dim.tolist(),
        "duration_s": time.monotonic() - started,
    }
    result.update(
        _component_metrics(
            mean_per_dim,
            model_action_dim=model_action_dim,
            schema=str(config.policy_metadata.get("schema", args.schema)),
        )
    )
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite evaluation result: {key}={value}")

    _atomic_json(args.result_json.expanduser().resolve(), result)
    metric_text = ", ".join(
        f"{key}={result[key]:.6f}"
        for key in (
            "eval_loss_model",
            "eval_loss_objective",
            "eval_loss_all",
            "eval_loss_padding",
        )
        if result.get(key) is not None
    )
    print(f"Eval Step {checkpoint_step}: {metric_text}", flush=True)
    print("EVAL_RESULT_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--arm-mode", choices=("auto", "single", "bimanual"), default="auto")
    parser.add_argument("--arm-side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--schema", choices=("auto", "delivery", "joint"), default="auto")
    parser.add_argument("--dataset-layout", choices=("auto", "legacy", "canonical"), default="auto")
    parser.add_argument("--model-variant", choices=("pi05", "pi0"), default="pi05")
    parser.add_argument("--delivery-action-convention", default="auto")
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
    parser.add_argument("--model-gripper-semantics", default="auto")
    parser.add_argument("--assets-base-dir", required=True)
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=0)
    args = parser.parse_args()
    for name in ("batch_size", "max_batches"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
