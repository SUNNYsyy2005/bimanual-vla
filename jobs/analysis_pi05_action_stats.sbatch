#!/bin/bash
#SBATCH --job-name=pi05_act_stat
#SBATCH -p h100
#SBATCH -w h100-ksy-01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

PROJECT="${PROJECT:-/DATA/sync/$USER/projects/bimanual-vla}"
OPENPI_ROOT="${OPENPI_ROOT:-/DATA/sync/$USER/projects/RoboTwin/policy/pi05}"
OPENPI_ENV="${OPENPI_ENV:-/DATA/sync/$USER/miniconda3/envs/openpi}"
WORK_ROOT="${WORK_ROOT:-/DATA/sync/$USER/openpi_h100}"
export PROJECT OPENPI_ROOT WORK_ROOT

export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="/DATA/sync/$USER/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip/cache"
export TMPDIR="$XDG_CACHE_HOME/pip/tmp"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT/server_4090:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$PROJECT:${PYTHONPATH:-}"

cd "$PROJECT"

"$OPENPI_ENV/bin/python" -u <<'PY'
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(os.environ["PROJECT"])
OPENPI_ROOT = Path(os.environ["OPENPI_ROOT"])
WORK_ROOT = Path(os.environ["WORK_ROOT"])
sys.path[:0] = [
    str(PROJECT / "server_4090"),
    str(OPENPI_ROOT / "src"),
    str(OPENPI_ROOT / "packages/openpi-client/src"),
    str(PROJECT),
]

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import nnx_utils


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


piper = _load_local_module("openpi_single_arm", PROJECT / "server_4090" / "openpi_single_arm.py")
ehl = _load_local_module("eval_heldout_loss", PROJECT / "server_4090" / "eval_heldout_loss.py")

DATASET_ID = "put_bottles_dustbin_piper_100_25hz_realqpos_v4_rgbsync"
CHECKPOINTS = [
    (
        "small_b1_19000",
        Path(
            f"{WORK_ROOT}/checkpoints/pi05_piper_bimanual_lora/"
            "pi05-put-bottles-v4-rgbsync-h100/19000"
        ),
    ),
    (
        "large_b32_10000",
        Path(
            f"{WORK_ROOT}/checkpoints/pi05_piper_bimanual_lora/"
            "pi05-put-bottles-v4-rgbsync-official-b32-250k-h100-20260808/10000"
        ),
    ),
    (
        "large_b32_15000",
        Path(
            f"{WORK_ROOT}/checkpoints/pi05_piper_bimanual_lora/"
            "pi05-put-bottles-v4-rgbsync-official-b32-250k-h100-20260808/15000"
        ),
    ),
]

base_args = argparse.Namespace(
    command="train",
    dataset_id=DATASET_ID,
    arm_mode="bimanual",
    arm_side="both",
    schema="joint",
    dataset_layout="auto",
    model_variant="pi05",
    delivery_action_convention="auto",
    contract_version=3,
    raw_action_dim=14,
    model_action_dim=14,
    raw_action_semantics="absolute_joint_position_opening_fraction",
    model_action_semantics="joint_delta_chunk_origin_first_6_absolute_gripper_target",
    raw_action_convention="absolute_joint_target",
    model_action_convention="chunk_origin",
    action_offset=1,
    model_action_start_offset=1,
    raw_gripper_semantics="absolute_opening_fraction_0_closed_1_open",
    gripper_semantics="absolute_opening_fraction_0_closed_1_open",
    model_gripper_semantics="absolute_opening_fraction_0_closed_1_open",
    assets_base_dir=str(OPENPI_ROOT / "assets"),
    checkpoint_base_dir=str(WORK_ROOT / "checkpoints/pi05_piper_bimanual_lora"),
    base_checkpoint=str(Path(os.environ["HF_HOME"]) / "openpi/openpi-assets/checkpoints/pi05_base"),
    exp_name="analysis",
    batch_size=4,
    num_workers=2,
    max_batches=8,
    eval_seed=0,
    test_ratio=None,
    split_seed=None,
)


def _safe_cosine(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_flat = pred.reshape(pred.shape[0], -1)
    gt_flat = gt.reshape(gt.shape[0], -1)
    pred_norm = np.linalg.norm(pred_flat, axis=1)
    gt_norm = np.linalg.norm(gt_flat, axis=1)
    return np.sum(pred_flat * gt_flat, axis=1) / (pred_norm * gt_norm + 1e-8)


def _stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")

    batch, horizon, action_dim = pred.shape
    pred_flat = pred.reshape(batch, -1)
    gt_flat = gt.reshape(batch, -1)
    pred_abs = np.abs(pred)
    gt_abs = np.abs(gt)

    cosine = _safe_cosine(pred, gt)
    mse = np.mean((pred_flat - gt_flat) ** 2, axis=1)

    gripper_idx = [i for i in (6, 13) if i < action_dim]
    joint_idx = [i for i in range(action_dim) if i not in gripper_idx]

    pred_step_abs = pred_abs.mean(axis=(0, 2))
    gt_step_abs = gt_abs.mean(axis=(0, 2))
    pred_step_l2 = np.linalg.norm(pred, axis=-1).mean(axis=0)
    gt_step_l2 = np.linalg.norm(gt, axis=-1).mean(axis=0)

    pred_dim_min = pred.min(axis=(0, 1))
    pred_dim_max = pred.max(axis=(0, 1))
    gt_dim_min = gt.min(axis=(0, 1))
    gt_dim_max = gt.max(axis=(0, 1))

    def _round_list(x: np.ndarray | list[float], digits: int = 6) -> list[float]:
        return np.round(np.asarray(x, dtype=np.float64), digits).tolist()

    out: dict[str, Any] = {
        "num_samples": int(batch),
        "action_horizon": int(horizon),
        "action_dim": int(action_dim),
        "pred_abs_mean": float(pred_abs.mean()),
        "gt_abs_mean": float(gt_abs.mean()),
        "pred_abs_std": float(pred.std()),
        "gt_abs_std": float(gt.std()),
        "pred_norm_mean": float(np.linalg.norm(pred_flat, axis=1).mean()),
        "gt_norm_mean": float(np.linalg.norm(gt_flat, axis=1).mean()),
        "pred_norm_std": float(np.linalg.norm(pred_flat, axis=1).std()),
        "gt_norm_std": float(np.linalg.norm(gt_flat, axis=1).std()),
        "pred_l2_to_gt_l2_ratio": float(
            np.linalg.norm(pred_flat, axis=1).mean() / (np.linalg.norm(gt_flat, axis=1).mean() + 1e-9)
        ),
        "pred_joint_abs_mean": float(np.abs(pred[..., joint_idx]).mean()) if joint_idx else None,
        "gt_joint_abs_mean": float(np.abs(gt[..., joint_idx]).mean()) if joint_idx else None,
        "pred_gripper_abs_mean": float(np.abs(pred[..., gripper_idx]).mean()) if gripper_idx else None,
        "gt_gripper_abs_mean": float(np.abs(gt[..., gripper_idx]).mean()) if gripper_idx else None,
        "pred_cosine_mean": float(cosine.mean()),
        "pred_cosine_std": float(cosine.std()),
        "pred_mse_mean": float(mse.mean()),
        "pred_mse_std": float(mse.std()),
        "pred_step_abs_mean": _round_list(pred_step_abs),
        "gt_step_abs_mean": _round_list(gt_step_abs),
        "pred_step_l2_mean": _round_list(pred_step_l2),
        "gt_step_l2_mean": _round_list(gt_step_l2),
        "pred_dim_min": _round_list(pred_dim_min),
        "pred_dim_max": _round_list(pred_dim_max),
        "pred_dim_range": _round_list(pred_dim_max - pred_dim_min),
        "gt_dim_min": _round_list(gt_dim_min),
        "gt_dim_max": _round_list(gt_dim_max),
        "gt_dim_range": _round_list(gt_dim_max - gt_dim_min),
        "pred_near_zero_frac_0p05": float((np.abs(pred) < 0.05).mean()),
        "gt_near_zero_frac_0p05": float((np.abs(gt) < 0.05).mean()),
        "first_example": {
            "pred_first3_steps": _round_list(pred[0, : min(3, horizon), :]),
            "gt_first3_steps": _round_list(gt[0, : min(3, horizon), :]),
        },
    }
    return out


config = piper.build_config(base_args)
contract = piper.complete_action_contract_fingerprint(config.policy_metadata)
split = piper._resolve_training_split(base_args, contract)
print("[analysis] test episodes:", list(split.test_episodes), flush=True)
print("[analysis] jax devices:", jax.devices(), flush=True)

mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("eval",))
replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("eval"))

# Build a deterministic, shared held-out slice once so every checkpoint sees
# the exact same validation observations and expert actions.
loader, available_frames, effective_batches = ehl._create_test_loader(
    config,
    CHECKPOINTS[0][1],
    split.test_episodes,
    batch_size=base_args.batch_size,
    num_workers=base_args.num_workers,
    max_batches=base_args.max_batches,
    eval_seed=base_args.eval_seed,
    data_sharding=data_sharding,
)
print(
    f"[analysis] heldout_frames={available_frames} effective_batches={effective_batches}",
    flush=True,
)
shared_batches: list[tuple[np.ndarray, np.ndarray]] = []
for batch_idx, batch in enumerate(loader):
    observation, actions = batch
    shared_batches.append(
        (
            np.asarray(jax.device_get(observation)),
            np.asarray(jax.device_get(actions)),
        )
    )
    if batch_idx + 1 >= base_args.max_batches:
        break
print(f"[analysis] collected_batches={len(shared_batches)}", flush=True)

if not shared_batches:
    raise RuntimeError("no held-out batches collected")

results: dict[str, Any] = {
    "dataset_id": DATASET_ID,
    "test_episode_indexes": list(split.test_episodes),
    "batch_size": base_args.batch_size,
    "num_batches": len(shared_batches),
    "action_horizon": int(config.model.action_horizon),
    "action_dim": int(config.policy_metadata["model_action_dim"]),
    "checkpoints": {},
}

for label, ckpt in CHECKPOINTS:
    ckpt = ckpt.expanduser().resolve()
    if not (ckpt / "params" / "_METADATA").is_file():
        raise FileNotFoundError(f"checkpoint params missing: {ckpt}")
    print(f"[analysis] loading {label}: {ckpt}", flush=True)
    params = model_lib.restore_params(ckpt / "params", sharding=replicated_sharding)
    model = config.model.load(params)
    sample_fn = nnx_utils.module_jit(model.sample_actions)

    pred_batches: list[np.ndarray] = []
    gt_batches: list[np.ndarray] = []
    for batch_idx, (observation, actions) in enumerate(shared_batches):
        rng = jax.random.fold_in(jax.random.key(0), batch_idx)
        pred = sample_fn(rng, observation, num_steps=int(config.model.action_horizon))
        pred = np.asarray(jax.device_get(pred), dtype=np.float32)
        gt = np.asarray(actions, dtype=np.float32)
        pred_batches.append(pred)
        gt_batches.append(gt)
        print(
            f"[analysis] {label} batch={batch_idx} pred_abs={np.abs(pred).mean():.6f} gt_abs={np.abs(gt).mean():.6f} pred_norm={np.linalg.norm(pred.reshape(pred.shape[0], -1), axis=1).mean():.6f}",
            flush=True,
        )

    pred_all = np.concatenate(pred_batches, axis=0)
    gt_all = np.concatenate(gt_batches, axis=0)
    summary = _stats(pred_all, gt_all)
    summary["checkpoint"] = str(ckpt)
    summary["checkpoint_step"] = int(ckpt.name)
    results["checkpoints"][label] = summary

print("[analysis] summary_json:")
print(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True))

out_dir = WORK_ROOT / "analysis/pi05_action_stats"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "compare_batch1_vs_b32_action_stats.json"
out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
print(f"[analysis] wrote {out_path}", flush=True)
PY
