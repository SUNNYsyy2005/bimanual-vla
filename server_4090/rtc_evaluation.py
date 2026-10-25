"""Evaluation-only clean-prefix flow loss; never changes training or serving state.

Inputs use the checkpoint's training transforms (chunk-origin deltas, normalized
physical dimensions, then zero padding). This measures denoising loss against a
demonstration suffix, not closed-loop task success or sampled trajectory MAE.
"""
from __future__ import annotations

from numbers import Integral


def evaluation_delays(maximum: int, horizon: int, requested=None) -> tuple[int, ...]:
    """Keep d=0 comparable to historical metrics and cover supported RTC delays."""
    if not 0 <= maximum < horizon:
        raise ValueError("RTC maximum must leave a generated suffix")
    values = ((0, 4, 5, 6, 7, 8, 10, maximum) if requested is None and maximum
              else (0,) if requested is None else (0, *requested))
    if requested is None:
        values = tuple(d for d in values if d <= maximum)
    if any(isinstance(d, bool) or not isinstance(d, Integral) or not 0 <= d <= maximum
           for d in values):
        raise ValueError(f"evaluation delays must be integers in [0, {maximum}]")
    return tuple(sorted(set(values)))


def prefix_loss_per_dim(model, rng, observation, actions, *, delay: int, planned_prefix=None):
    """BxTxD loss, prefix masked and suffix scaled by H/(H-d).

    ``delay`` is a static JIT argument. By default the clean prefix is the
    demonstration. Optional ``planned_prefix`` is BxPxD in the SAME normalized
    new-chunk coordinates as actions, with P >= delay. The demonstration suffix
    remains the label; a planner prefix must not replace its ground truth.
    Callers must pair planner inputs with the exact observation used to plan.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib, pi0

    horizon = model.action_horizon
    maximum = getattr(model, "rtc_training_max_delay", 0)
    evaluation_delays(maximum, horizon, (delay,))
    if actions.ndim != 3 or actions.shape[1:] != (horizon, model.action_dim):
        raise ValueError("actions must have shape B x action_horizon x action_dim")
    if planned_prefix is not None:
        if delay == 0:
            raise ValueError("a planner prefix requires a nonzero delay")
        if (planned_prefix.ndim != 3 or planned_prefix.shape[0] != actions.shape[0]
                or planned_prefix.shape[2] != actions.shape[2]
                or not delay <= planned_prefix.shape[1] <= horizon):
            raise ValueError("planner prefix must match B, D and cover delay rows")
    # Match the production Piper loss RNG split for paired d=0 comparisons.
    preprocess_rng, noise_rng, time_rng, _ = jax.random.split(rng, 4)
    observation = model_lib.preprocess_observation(preprocess_rng, observation, train=False)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    prefix = jnp.arange(horizon)[None, :] < delay
    model_time = jnp.where(prefix, 0.0, time[:, None])
    noisy = model_time[..., None] * noise + (1 - model_time[..., None]) * actions
    if planned_prefix is not None:
        # The serving sampler also clears unused action dimensions in its prefix.
        physical_dim = getattr(model, "loss_action_dim", model.action_dim)
        clean = jnp.where(jnp.arange(model.action_dim) < physical_dim,
                          planned_prefix[:, :delay], 0.0)
        noisy = noisy.at[:, :delay].set(clean)
    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar, cond = model.embed_suffix(observation, noisy, model_time)
    mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    attention = pi0.make_attn_mask(mask, jnp.concatenate([prefix_ar, suffix_ar]))
    (_, output), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attention, positions=jnp.cumsum(mask, axis=1) - 1,
        adarms_cond=[None, cond])
    velocity = model.action_out_proj(output[:, -horizon:])
    loss = jnp.square(velocity - (noise - actions))
    return jnp.where(prefix[..., None], 0.0, loss) * (horizon / (horizon - delay))
