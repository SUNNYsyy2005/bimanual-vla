"""EMA for trainable parameters; frozen parameters always follow Raw exactly.

Keep the full NNX parameter tree so existing OpenPI checkpoint split/merge and
inference loaders remain compatible. Trainable EMA accumulators use float32,
even when model parameters use bfloat16. No inference-time layer is added.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp

POLICY = "trainable_fp32_frozen_raw.v1"


def _float32(variable):
    value = variable.value
    if isinstance(value, jax.ShapeDtypeStruct):
        value = jax.ShapeDtypeStruct(value.shape, jnp.float32, sharding=value.sharding)
    else:
        value = value.astype(jnp.float32)
    return variable.replace(value)


def normalize_ema(params, ema_params, trainable_filter):
    """Initialize or migrate EMA, discarding historical frozen-param drift."""
    source = params if ema_params is None else ema_params
    trainable = source.filter(trainable_filter).map(lambda _, v: _float32(v))
    return nnx.State.merge(params.filter(nnx.Not(trainable_filter)), trainable)


def update_ema(params, ema_params, trainable_filter, decay):
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    if ema_params is None:
        raise ValueError("EMA must be initialized before the first update")
    trainable = jax.tree.map(
        lambda old, new: old.astype(jnp.float32) * decay
        + new.astype(jnp.float32) * (1.0 - decay),
        ema_params.filter(trainable_filter), params.filter(trainable_filter),
    )
    return nnx.State.merge(params.filter(nnx.Not(trainable_filter)), trainable)


def install_training_ema(module, config):
    """Adapt a freshly loaded upstream trainer without editing vendor files.

    The upstream optimizer/loss/gradient step runs with its old EMA disabled.
    Only this module owns EMA initialization, resume migration and updates.
    Checkpoint calls are proxied per trainer, not patched process-wide.
    """
    if getattr(module, "_bimanual_ema_policy", None) == POLICY:
        return
    original_init = module.init_train_state
    original_step = module.train_step
    original_checkpoints = module._checkpoints

    def normalize(state):
        if state.ema_decay is None:
            return state
        return dataclasses.replace(state, ema_params=normalize_ema(
            state.params, state.ema_params, config.trainable_filter))

    @functools.wraps(original_init)
    def init(*args, **kwargs):
        state, sharding = original_init(*args, **kwargs)
        # Both real arrays and the abstract resume template retain structure.
        return normalize(state), sharding

    @functools.wraps(original_step)
    def step(current_config, rng, state, batch):
        if state.ema_decay is None:
            return original_step(current_config, rng, state, batch)
        base = dataclasses.replace(state, ema_decay=None, ema_params=None)
        result, info = original_step(current_config, rng, base, batch)
        return dataclasses.replace(result, ema_decay=state.ema_decay,
            ema_params=update_ema(result.params, state.ema_params,
                current_config.trainable_filter, state.ema_decay)), info

    @functools.wraps(original_checkpoints.restore_state)
    def restore(*args, **kwargs):
        state = normalize(original_checkpoints.restore_state(*args, **kwargs))
        logging.info("EMA resume policy: %s; frozen params restored from Raw", POLICY)
        return state

    module.init_train_state = init
    module.train_step = step
    module._checkpoints = SimpleNamespace(**{
        **vars(original_checkpoints), "restore_state": restore})
    module._bimanual_ema_policy = POLICY
    logging.info("Installed EMA policy: %s", POLICY)
