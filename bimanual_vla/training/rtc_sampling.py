"""Training RTC delay sampling, with optional empirical categorical weights."""
from __future__ import annotations
import math


def validate_delay_weights(weights, maximum):
    if weights is None:
        return None
    values = tuple(float(value) for value in weights)
    if len(values) != maximum + 1:
        raise ValueError("RTC delay weights must contain max_delay + 1 entries (including zero)")
    if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("RTC delay weights must be finite, nonnegative, and have positive sum")
    return values


def sample_rtc_delays(rng, shape, maximum, weights=None):
    import jax
    import jax.numpy as jnp
    if weights is None:
        return jax.random.randint(rng, shape, 0, maximum + 1)
    # Zero-weight categories remain impossible; normalization is implicit in softmax.
    return jax.random.categorical(rng, jnp.log(jnp.asarray(weights, dtype=jnp.float32)),
                                  shape=shape).astype(jnp.int32)
