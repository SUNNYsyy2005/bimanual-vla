#!/usr/bin/env python3
"""Compare π0.5 action statistics for two checkpoints on the 4×4090 host."""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import importlib.util

REPO_ROOT = Path(os.environ.get('BIMANUAL_VLA_REPO', Path(__file__).resolve().parents[1])).expanduser().resolve()
PROJECT = Path(os.environ.get('ROBOTWIN_PROJECT', '/home/sunny/robotwin_ws/RoboTwin')).expanduser().resolve()
HELPER_DIR = REPO_ROOT / 'server_4090'
sys.path[:0] = [
    str(HELPER_DIR),
    str(PROJECT / 'policy/pi05/src'),
    str(PROJECT / 'policy/pi05/packages/openpi-client/src'),
]

import jax
import jax.numpy as jnp
import numpy as np

import eval_heldout_loss as ehl
import openpi_single_arm as piper
from openpi.models import model as model_lib
from openpi.shared import nnx_utils

DATASET_ID = os.environ.get('PI05_STATS_DATASET_ID', 'put_bottles_dustbin_piper_100_25hz_realqpos_v4_rgbsync')
CHECKPOINTS = [
    (
        'small_b1_19000',
        PROJECT / 'policy/pi05/checkpoints/pi05_piper_bimanual_lora/pi05-put-bottles-v4-rgbsync-h100/19000',
    ),
    (
        'large_b32_10000',
        PROJECT / 'policy/pi05/checkpoints/pi05_piper_bimanual_lora/pi05-put-bottles-v4-rgbsync-official-b32-250k-h100-20260808/10000',
    ),
]

base_args = type('Args', (), {})()
base_args.command = 'train'
base_args.dataset_id = DATASET_ID
base_args.arm_mode = 'bimanual'
base_args.arm_side = 'both'
base_args.schema = 'joint'
base_args.dataset_layout = 'auto'
base_args.model_variant = 'pi05'
base_args.delivery_action_convention = 'auto'
base_args.contract_version = 3
base_args.raw_action_dim = 14
base_args.model_action_dim = 14
base_args.raw_action_semantics = 'absolute_joint_position_opening_fraction'
base_args.model_action_semantics = 'joint_delta_chunk_origin_first_6_absolute_gripper_target'
base_args.raw_action_convention = 'absolute_joint_target'
base_args.model_action_convention = 'chunk_origin'
base_args.action_offset = 1
base_args.model_action_start_offset = 1
base_args.raw_gripper_semantics = 'absolute_opening_fraction_0_closed_1_open'
base_args.gripper_semantics = 'absolute_opening_fraction_0_closed_1_open'
base_args.model_gripper_semantics = 'absolute_opening_fraction_0_closed_1_open'
base_args.assets_base_dir = str(PROJECT / 'policy/pi05/assets')
base_args.checkpoint_base_dir = str(PROJECT / 'policy/pi05/checkpoints/pi05_piper_bimanual_lora')
base_args.base_checkpoint = os.environ.get('PI05_BASE_CHECKPOINT', str(Path.home() / '.cache/openpi/openpi-assets/checkpoints/pi05_base'))
base_args.exp_name = 'analysis'
base_args.num_train_steps = 1
base_args.save_interval = 1
base_args.log_interval = 1
base_args.fsdp_devices = 1
base_args.resume = False
base_args.overwrite = False
base_args.resume_checkpoint = None
base_args.wandb_enabled = False
base_args.batch_size = 4
base_args.num_workers = 0
base_args.max_batches = 8
base_args.eval_seed = 0
base_args.test_ratio = None
base_args.split_seed = None


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
        raise ValueError(f'shape mismatch: pred={pred.shape} gt={gt.shape}')

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

    return {
        'num_samples': int(batch),
        'action_horizon': int(horizon),
        'action_dim': int(action_dim),
        'pred_abs_mean': float(pred_abs.mean()),
        'gt_abs_mean': float(gt_abs.mean()),
        'pred_abs_std': float(pred.std()),
        'gt_abs_std': float(gt.std()),
        'pred_norm_mean': float(np.linalg.norm(pred_flat, axis=1).mean()),
        'gt_norm_mean': float(np.linalg.norm(gt_flat, axis=1).mean()),
        'pred_norm_std': float(np.linalg.norm(pred_flat, axis=1).std()),
        'gt_norm_std': float(np.linalg.norm(gt_flat, axis=1).std()),
        'pred_l2_to_gt_l2_ratio': float(np.linalg.norm(pred_flat, axis=1).mean() / (np.linalg.norm(gt_flat, axis=1).mean() + 1e-9)),
        'pred_joint_abs_mean': float(np.abs(pred[..., joint_idx]).mean()) if joint_idx else None,
        'gt_joint_abs_mean': float(np.abs(gt[..., joint_idx]).mean()) if joint_idx else None,
        'pred_gripper_abs_mean': float(np.abs(pred[..., gripper_idx]).mean()) if gripper_idx else None,
        'gt_gripper_abs_mean': float(np.abs(gt[..., gripper_idx]).mean()) if gripper_idx else None,
        'pred_cosine_mean': float(cosine.mean()),
        'pred_cosine_std': float(cosine.std()),
        'pred_mse_mean': float(mse.mean()),
        'pred_mse_std': float(mse.std()),
        'pred_step_abs_mean': _round_list(pred_step_abs),
        'gt_step_abs_mean': _round_list(gt_step_abs),
        'pred_step_l2_mean': _round_list(pred_step_l2),
        'gt_step_l2_mean': _round_list(gt_step_l2),
        'pred_dim_min': _round_list(pred_dim_min),
        'pred_dim_max': _round_list(pred_dim_max),
        'pred_dim_range': _round_list(pred_dim_max - pred_dim_min),
        'gt_dim_min': _round_list(gt_dim_min),
        'gt_dim_max': _round_list(gt_dim_max),
        'gt_dim_range': _round_list(gt_dim_max - gt_dim_min),
        'pred_near_zero_frac_0p05': float((np.abs(pred) < 0.05).mean()),
        'gt_near_zero_frac_0p05': float((np.abs(gt) < 0.05).mean()),
        'first_example': {
            'pred_first3_steps': _round_list(pred[0, : min(3, horizon), :]),
            'gt_first3_steps': _round_list(gt[0, : min(3, horizon), :]),
        },
    }


config = piper.build_config(base_args)
contract = piper.complete_action_contract_fingerprint(config.policy_metadata)
split = piper._resolve_training_split(base_args, contract)
print('[analysis] test episodes:', list(split.test_episodes), flush=True)
print('[analysis] jax devices:', jax.devices(), flush=True)

mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('eval',))
replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('eval'))

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
print(f'[analysis] heldout_frames={available_frames} effective_batches={effective_batches}', flush=True)
shared_batches: list[tuple[np.ndarray, np.ndarray]] = []
for batch_idx, batch in enumerate(loader):
    observation, actions = batch
    shared_batches.append(
        (
            jax.device_get(observation),
            np.asarray(jax.device_get(actions)),
        )
    )
    if batch_idx + 1 >= base_args.max_batches:
        break
print(f'[analysis] collected_batches={len(shared_batches)}', flush=True)

results: dict[str, Any] = {
    'dataset_id': DATASET_ID,
    'test_episode_indexes': list(split.test_episodes),
    'batch_size': base_args.batch_size,
    'num_batches': len(shared_batches),
    'action_horizon': int(config.model.action_horizon),
    'action_dim': int(config.policy_metadata['model_action_dim']),
    'checkpoints': {},
}

for label, ckpt in CHECKPOINTS:
    ckpt = ckpt.expanduser().resolve()
    if not (ckpt / 'params' / '_METADATA').is_file():
        raise FileNotFoundError(f'checkpoint params missing: {ckpt}')
    print(f'[analysis] loading {label}: {ckpt}', flush=True)
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass
    params = model_lib.restore_params(ckpt / 'params', sharding=replicated_sharding)
    model = config.model.load(params)
    sample_fn = nnx_utils.module_jit(model.sample_actions)

    pred_batches: list[np.ndarray] = []
    gt_batches: list[np.ndarray] = []
    for batch_idx, (observation, actions) in enumerate(shared_batches):
        rng = jax.random.fold_in(jax.random.key(0), batch_idx)
        pred = sample_fn(rng, jax.device_put(observation), num_steps=int(config.model.action_horizon))
        pred = np.asarray(jax.device_get(pred), dtype=np.float32)
        gt = np.asarray(actions, dtype=np.float32)
        pred_batches.append(pred)
        gt_batches.append(gt)
        print(
            f'[analysis] {label} batch={batch_idx} pred_abs={np.abs(pred).mean():.6f} gt_abs={np.abs(gt).mean():.6f} pred_norm={np.linalg.norm(pred.reshape(pred.shape[0], -1), axis=1).mean():.6f}',
            flush=True,
        )

    pred_all = np.concatenate(pred_batches, axis=0)
    gt_all = np.concatenate(gt_batches, axis=0)
    summary = _stats(pred_all, gt_all)
    summary['checkpoint'] = str(ckpt)
    summary['checkpoint_step'] = int(ckpt.name)
    results['checkpoints'][label] = summary

    del sample_fn, model, params, pred_batches, gt_batches, pred_all, gt_all, summary
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass

print('[analysis] summary_json:')
print(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True))

out_dir = Path(os.environ.get('PI05_STATS_OUTPUT_DIR', REPO_ROOT / 'artifacts/pi05_stats')).expanduser().resolve()
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'compare_batch1_vs_b32_action_stats.json'
out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')
print(f'[analysis] wrote {out_path}', flush=True)
