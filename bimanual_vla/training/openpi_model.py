"""OpenPI JAX adapter for trainable partitions and clean-prefix RTC.

Training-time action conditioning follows LeRobot Pi05 (Apache-2.0):
https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/policies/pi05/modeling_pi05.py
The flow loss, sampler, and adaptive norm follow OpenPI (Apache-2.0).
See docs/deployment/OPEN_SOURCE_CONTROL_REFERENCES.md for the source contract.
"""

from __future__ import annotations

import dataclasses
from bimanual_vla.training.rtc_sampling import sample_rtc_delays, validate_delay_weights

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import gemma, model as model_lib, pi0, pi0_config
from openpi.shared import nnx_utils


def install_per_action_adarms() -> None:
    """Extend the existing norm without changing its parameter names/shapes.

    Stock OpenPI broadcasts a single timestep per chunk. Clean-prefix RTC
    supplies BxTxD conditioning: prefix actions use time zero, postfix actions
    use the sampled flow time. Other model instances retain the original path.
    """
    if getattr(gemma.RMSNorm, "_bimanual_per_action", False):
        return

    # Subclasses let Linen install its own method/scope wrappers. Assigning a
    # compact function to an existing class bypasses that registration.
    class PerActionRMSNorm(gemma.RMSNorm):
        _bimanual_per_action = True

        @nn.compact
        def __call__(self, x, cond):
            dtype = x.dtype
            variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
            normalized = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(variance + 1e-6)))
            if cond is None:
                scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1],))
                return (normalized * (1 + scale)).astype(dtype), None
            modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros,
                                  dtype=dtype, name="Dense_0")(cond)
            if cond.ndim == 2:
                modulation = modulation[:, None, :]
            elif cond.ndim != 3 or cond.shape[:2] != x.shape[:2]:
                raise ValueError("adaptive norm conditioning must be BxD or matching BxTxD")
            scale, shift, gate = jnp.split(modulation, 3, axis=-1)
            return (normalized * (1 + scale) + shift).astype(dtype), gate

    class PerActionGemma(gemma.Module):
        # Extend stock BxD admission to token-specific conditioning. The layer
        # scan, residual gates, cache, and all parameter scopes are unchanged.
        def __call__(self, embedded, positions, mask, adarms_cond=None, *,
                     kv_cache=None, deterministic=True):
            embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
            mask = jnp.asarray(mask)[:, None, :, :]
            if adarms_cond is None:
                adarms_cond = [None] * len(self.configs)
            embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask,
                                            adarms_cond, deterministic)
            return [f(e, a)[0] if e is not None else e
                    for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)], kv_cache

    gemma.RMSNorm = PerActionRMSNorm
    gemma.Module = PerActionGemma


@dataclasses.dataclass(frozen=True)
class PiperPi0Config(pi0_config.Pi0Config):
    image_resize_mode: str = "pad"
    loss_action_dim: int | None = None
    llm_lora_enabled: bool = False
    train_vision: bool = False
    train_llm: bool = False
    rtc_training_max_delay: int = 0
    rtc_training_delay_weights: tuple[float, ...] | None = None

    def create(self, rng):
        validate_delay_weights(self.rtc_training_delay_weights, self.rtc_training_max_delay)
        if self.loss_action_dim is not None and not 1 <= self.loss_action_dim <= self.action_dim:
            raise ValueError("physical action loss dimension must fit the model head")
        if not 0 <= self.rtc_training_max_delay < self.action_horizon:
            raise ValueError("training RTC delay must leave at least one predicted action")
        if self.rtc_training_max_delay:
            if not self.pi05:
                raise ValueError("clean-prefix training RTC requires pi05")
            install_per_action_adarms()
        return PiperPi0(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter(self):
        language = nnx_utils.PathRegex(".*PaliGemma.*llm.*")
        expert = nnx_utils.PathRegex(".*llm.*_1.*")
        lora = nnx_utils.PathRegex(".*lora.*")
        language_freeze = nnx.All(language, nnx.Not(expert))
        if self.llm_lora_enabled:
            language_freeze = nnx.All(language_freeze, nnx.Not(lora))
        if self.train_llm and self.llm_lora_enabled:
            raise ValueError("full LLM training and LLM LoRA are mutually exclusive")
        filters = []
        if not self.train_vision:
            filters.append(nnx_utils.PathRegex(".*PaliGemma.*img.*"))
        if not self.train_llm:
            filters.append(language_freeze)
        if "lora" in self.action_expert_variant:
            filters.append(nnx.All(expert, nnx.Not(lora)))
        return nnx.Any(*filters)


class PiperPi0(pi0.Pi0):
    def __init__(self, config: PiperPi0Config, rngs):
        super().__init__(config, rngs)
        self.image_resize_mode = config.image_resize_mode
        self.loss_action_dim = config.loss_action_dim or config.action_dim
        self.rtc_training_max_delay = config.rtc_training_max_delay
        self.rtc_training_delay_weights = config.rtc_training_delay_weights

    def embed_suffix(self, obs, noisy_actions, timestep):
        if timestep.ndim == 1:
            return super().embed_suffix(obs, noisy_actions, timestep)
        if not self.pi05 or timestep.shape != noisy_actions.shape[:2]:
            raise ValueError("per-action flow time requires Pi05 and matching BxT timesteps")
        tokens = self.action_in_proj(noisy_actions)
        times = pi0.posemb_sincos(timestep.reshape(-1), self.action_in_proj.out_features,
                                  min_period=4e-3, max_period=4.0)
        times = nnx.swish(self.time_mlp_in(times))
        times = nnx.swish(self.time_mlp_out(times)).reshape(*timestep.shape, -1)
        mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        ar = jnp.arange(self.action_horizon) == 0
        return tokens, mask, ar, times

    def compute_loss_per_dim(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        # Stretch checkpoints keep geometry identical to serving. Stock train-time
        # rotation/crop can reintroduce border fill; retain only color augmentation.
        observation = model_lib.preprocess_observation(
            preprocess_rng, observation, train=train and self.image_resize_mode != "stretch")
        if train and self.image_resize_mode == "stretch":
            import augmax
            images = {}
            for key, image in observation.images.items():
                keys = jax.random.split(jax.random.fold_in(preprocess_rng, len(images)), image.shape[0])
                jitter = augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
                images[key] = 2 * jax.vmap(jitter)(keys, image / 2 + 0.5) - 1
            observation = dataclasses.replace(observation, images=images)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        delays = (sample_rtc_delays(delay_rng, actions.shape[:-2], self.rtc_training_max_delay,
                                    self.rtc_training_delay_weights)
                  if train and self.rtc_training_max_delay else jnp.zeros(actions.shape[:-2], jnp.int32))
        prefix = jnp.arange(self.action_horizon)[None, :] < delays[:, None]
        if self.rtc_training_max_delay:
            model_time = jnp.where(prefix, 0.0, time[:, None])
            time_expanded = model_time[..., None]
        else:
            model_time = time
            time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar, cond = self.embed_suffix(observation, x_t, model_time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        attn_mask = pi0.make_attn_mask(input_mask, jnp.concatenate([prefix_ar, suffix_ar]))
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
            adarms_cond=[None, cond])
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        losses = jnp.square(velocity - u_t)
        # Keep the BxTxD evaluation API, with a mean equivalent to averaging
        # over only the predicted postfix for each sample.
        weights = (~prefix).astype(losses.dtype) * (self.action_horizon / (self.action_horizon - delays))[:, None]
        return losses * weights[..., None]

    def compute_loss(self, rng, observation, actions, *, train=False):
        return self.compute_loss_per_dim(rng, observation, actions, train=train)[..., :self.loss_action_dim].mean(axis=-1)

    def sample_actions(self, rng, observation, *, num_steps=5, noise=None,
                       prev_chunk_left_over=None, inference_delay=None,
                       previous_left_over_steps=None, **kwargs):
        if not self.rtc_training_max_delay:
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        if num_steps <= 0:
            raise ValueError("denoising steps must be positive")
        observation = model_lib.preprocess_observation(None, observation, train=False)
        batch = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))
        if noise.shape != (batch, self.action_horizon, self.action_dim):
            raise ValueError("noise must match the batched action chunk shape")
        delay = jnp.asarray(0 if inference_delay is None else inference_delay, dtype=jnp.int32)
        previous = jnp.zeros_like(noise)
        if prev_chunk_left_over is not None:
            previous = jnp.asarray(prev_chunk_left_over, dtype=noise.dtype)
            if previous.ndim == 2:
                previous = previous[None, ...]
            previous = jnp.broadcast_to(previous, noise.shape)
        prefix = jnp.arange(self.action_horizon)[None, :] < delay
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, cache = self.PaliGemma.llm([prefix_tokens, None],
                                    mask=pi0.make_attn_mask(prefix_mask, prefix_ar), positions=positions)
        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            x_t = jnp.where(prefix[..., None], previous, x_t)
            if self.loss_action_dim < self.action_dim:
                # PadStatesAndActions supplies a known zero target here, and
                # these output channels have no velocity loss. Keep their
                # inputs on the training interpolation instead of integrating
                # arbitrary, untrained velocities into the physical channels.
                padding = jnp.where(prefix[..., None], 0.0,
                                    time * noise[:, :, self.loss_action_dim:])
                x_t = x_t.at[:, :, self.loss_action_dim:].set(padding)
            model_time = jnp.broadcast_to(jnp.where(prefix, 0.0, time), (batch, self.action_horizon))
            tokens, mask, ar, cond = self.embed_suffix(observation, x_t, model_time)
            full_mask = jnp.concatenate([
                jnp.broadcast_to(prefix_mask[:, None, :], (batch, self.action_horizon, prefix_mask.shape[1])),
                pi0.make_attn_mask(mask, ar)], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
            (_, output), _ = self.PaliGemma.llm([None, tokens], mask=full_mask,
                positions=positions, kv_cache=cache, adarms_cond=[None, cond])
            velocity = self.action_out_proj(output[:, -self.action_horizon:])
            return jnp.where(prefix[..., None], previous, x_t + dt * velocity), time + dt

        result, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, step,
                                     (noise, jnp.asarray(1.0, noise.dtype)))
        if self.loss_action_dim < self.action_dim:
            result = result.at[:, :, self.loss_action_dim:].set(0.0)
        return result
