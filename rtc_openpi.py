#!/usr/bin/env python3
"""Model-side Real-Time Chunking (RTC) for the OpenPI flow-matching policy.

RTC is not a client-side action interpolation.  During flow-matching denoising,
this module guides the new chunk toward the still-unexecuted prefix of the
previous normalized chunk.  The guide is applied through the Jacobian of the
predicted clean action, so the model itself resolves inference latency before
the action chunk is unnormalized and sent back to the robot.

The repository's OpenPI checkout is kept external by deployment policy.  The
runtime adapter below patches its PyTorch ``PI0Pytorch`` or JAX/NNX ``Pi0``
instance without modifying that checkout, and wraps the upstream ``Policy`` to
pass per-request RTC state over the existing WebSocket observation payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import types
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RTCConfig:
    """Runtime RTC parameters shared by the server and client handshake."""

    enabled: bool = True
    execution_horizon: int = 8
    max_guidance_weight: float = 5.0
    prefix_attention_schedule: str = "linear"

    def __post_init__(self) -> None:
        if self.execution_horizon <= 0:
            raise ValueError("RTC execution_horizon must be positive")
        if self.max_guidance_weight <= 0:
            raise ValueError("RTC max_guidance_weight must be positive")
        if self.prefix_attention_schedule not in {"zeros", "ones", "linear", "exp"}:
            raise ValueError(
                "RTC prefix_attention_schedule must be one of zeros, ones, linear, exp"
            )


def _prefix_weights(
    start: int,
    end: int,
    total: int,
    *,
    schedule: str = "linear",
    dtype: Any = None,
    device: Any = None,
):
    """Return the RTC prefix attention schedule as a 1-D Torch tensor."""
    import torch

    if total <= 0:
        raise ValueError("total must be positive")
    start = max(0, min(int(start), int(end)))
    end = max(0, min(int(end), int(total)))
    if schedule == "zeros":
        weights = torch.zeros(total, dtype=dtype, device=device)
        weights[:start] = 1.0
        return weights
    if schedule == "ones":
        weights = torch.ones(total, dtype=dtype, device=device)
        weights[end:] = 0.0
        return weights
    span = total - max(total - end, 0) - start
    if end <= start or span <= 0:
        middle = torch.empty(0, dtype=dtype, device=device)
    else:
        middle = torch.linspace(1.0, 0.0, span + 2, dtype=dtype, device=device)[1:-1]
        if schedule == "exp":
            middle = middle * torch.expm1(middle) / (np.e - 1.0)
    leading = torch.ones(start, dtype=dtype, device=device)
    trailing = torch.zeros(max(total - end, 0), dtype=dtype, device=device)
    return torch.cat((leading, middle, trailing), dim=0)


class RTCProcessor:
    """Apply Physical-Intelligence-style RTC guidance to a flow denoiser."""

    def __init__(self, config: RTCConfig):
        self.config = config

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay: int | None,
        time,
        original_denoise_step,
        execution_horizon: int | None = None,
    ):
        """Return a guided velocity for one reverse-flow denoising step.

        ``prev_chunk_left_over`` is already in the model's normalized action
        space.  ``inference_delay`` is the number of action-rate steps expected
        to execute while this request is in flight.  The first ``delay`` rows
        receive full prefix attention and the following rows fade according to
        the configured schedule.
        """
        import torch

        if prev_chunk_left_over is None or not self.config.enabled:
            return original_denoise_step(x_t)

        if not isinstance(prev_chunk_left_over, torch.Tensor):
            prev_chunk_left_over = torch.as_tensor(
                prev_chunk_left_over, dtype=x_t.dtype, device=x_t.device
            )
        else:
            prev_chunk_left_over = prev_chunk_left_over.to(device=x_t.device, dtype=x_t.dtype)

        squeezed = x_t.ndim == 2
        if squeezed:
            x_t = x_t.unsqueeze(0)
        if prev_chunk_left_over.ndim == 2:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)
        if prev_chunk_left_over.ndim != 3 or x_t.ndim != 3:
            raise ValueError("RTC action chunks must have shape (T,A) or (B,T,A)")
        if prev_chunk_left_over.shape[0] not in {1, x_t.shape[0]}:
            raise ValueError("RTC previous chunk batch dimension does not match current batch")
        if prev_chunk_left_over.shape[0] == 1 and x_t.shape[0] != 1:
            prev_chunk_left_over = prev_chunk_left_over.expand(x_t.shape[0], -1, -1)

        action_horizon = x_t.shape[1]
        action_dim = x_t.shape[2]
        if prev_chunk_left_over.shape[2] != action_dim:
            raise ValueError(
                f"RTC action dimension mismatch: previous={prev_chunk_left_over.shape[2]} "
                f"current={action_dim}"
            )
        previous_length = int(prev_chunk_left_over.shape[1])
        if prev_chunk_left_over.shape[1] > action_horizon:
            prev_chunk_left_over = prev_chunk_left_over[:, :action_horizon]
            previous_length = action_horizon
        if prev_chunk_left_over.shape[1] < action_horizon:
            padded = torch.zeros(
                x_t.shape[0], action_horizon, action_dim,
                dtype=x_t.dtype,
                device=x_t.device,
            )
            padded[:, : prev_chunk_left_over.shape[1]] = prev_chunk_left_over
            prev_chunk_left_over = padded

        delay = max(0, int(inference_delay or 0))
        horizon = int(execution_horizon or self.config.execution_horizon)
        horizon = max(1, min(horizon, action_horizon, previous_length))
        weights = _prefix_weights(
            delay,
            horizon,
            action_horizon,
            schedule=self.config.prefix_attention_schedule,
            dtype=x_t.dtype,
            device=x_t.device,
        ).view(1, action_horizon, 1)

        # This is the key RTC operation.  We need the Jacobian of the predicted
        # clean action x_1(t) with respect to the current noisy sample x_t.
        x_t = x_t.detach().requires_grad_(True)
        with torch.enable_grad():
            velocity = original_denoise_step(x_t)
            x1_t = x_t - time * velocity
            error = (prev_chunk_left_over - x1_t) * weights
            correction = torch.autograd.grad(
                outputs=x1_t,
                inputs=x_t,
                grad_outputs=error.detach(),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

        tau = 1.0 - time
        tau_tensor = torch.as_tensor(tau, dtype=x_t.dtype, device=x_t.device)
        eps = torch.finfo(x_t.dtype).eps
        tau_safe = tau_tensor.clamp_min(eps)
        one_minus_tau_sq = (1.0 - tau_tensor) ** 2
        inv_r2 = (one_minus_tau_sq + tau_tensor**2) / one_minus_tau_sq.clamp_min(eps)
        guidance_weight = ((1.0 - tau_tensor) / tau_safe) * inv_r2
        guidance_weight = torch.nan_to_num(
            guidance_weight,
            nan=self.config.max_guidance_weight,
            posinf=self.config.max_guidance_weight,
            neginf=0.0,
        ).clamp(min=0.0, max=self.config.max_guidance_weight)
        guided = velocity - guidance_weight * correction
        return guided.squeeze(0) if squeezed else guided


def _rtc_sample_actions_pytorch(
    self,
    device,
    observation,
    noise=None,
    num_steps=10,
    **kwargs,
):
    """Patched ``PI0Pytorch.sample_actions`` with the RTC denoising hook."""
    import torch

    # Keep the adapter independent from the external checkout at import time,
    # but use its exact attention-mask helper once a live OpenPI model is running.
    try:
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    except ImportError as exc:  # pragma: no cover - exercised only in a broken deployment env
        raise RuntimeError("RTC PyTorch adapter requires OpenPI's make_att_2d_masks") from exc

    rtc_processor: RTCProcessor | None = getattr(self, "_rtc_processor", None)
    prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
    inference_delay = kwargs.get("inference_delay")
    execution_horizon = kwargs.get("execution_horizon")
    bsize = observation.state.shape[0]
    if noise is None:
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        noise = self.sample_noise(actions_shape, device)
    if not isinstance(noise, torch.Tensor):
        noise = torch.as_tensor(noise, device=device)
    noise = noise.to(device=device)

    images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
        observation, train=False
    )
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

    dt = torch.tensor(-1.0 / int(num_steps), dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    for _ in range(int(num_steps)):
        expanded_time = time.expand(bsize)

        def denoise(input_x_t):
            return self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                input_x_t,
                expanded_time,
            )

        if rtc_processor is not None and prev_chunk_left_over is not None:
            v_t = rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time,
                original_denoise_step=denoise,
                execution_horizon=execution_horizon,
            )
        else:
            with torch.no_grad():
                v_t = denoise(x_t)
        x_t = (x_t + dt * v_t).detach()
        time = time + dt

    # Policy.infer applies the unnormalize transform afterwards.  Keep the
    # normalized model output for the next request's RTC prefix.
    self._rtc_last_normalized_actions = x_t.detach().float().cpu().numpy()
    return x_t


def patch_pytorch_model(model: Any, config: RTCConfig) -> None:
    """Install RTC sampling on one OpenPI PyTorch model instance."""
    if not hasattr(model, "denoise_step") or not hasattr(model, "_preprocess_observation"):
        raise TypeError("RTC requires an OpenPI PyTorch flow-matching model")
    model._rtc_processor = RTCProcessor(config)
    model._rtc_original_sample_actions = getattr(model, "sample_actions", None)
    # Assigning a bound function on the instance avoids changing unrelated
    # policies in a process that may host more than one model.
    model.sample_actions = types.MethodType(_rtc_sample_actions_pytorch, model)


def _jax_prefix_weights(start: int, end: int, total: int, schedule: str):
    """Build RTC prefix weights without introducing a data-dependent shape."""
    import jax.numpy as jnp

    start = max(0, min(int(start), int(end), int(total)))
    end = max(start, min(int(end), int(total)))
    positions = jnp.arange(total)
    if schedule == "zeros":
        return (positions < start).astype(jnp.float32)
    if schedule == "ones":
        return (positions < end).astype(jnp.float32)
    span = max(1, end - start)
    # Match the PyTorch implementation's open interval linspace(1, 0).
    middle = (end - positions) / float(span + 1)
    middle = jnp.clip(middle, 0.0, 1.0)
    if schedule == "exp":
        middle = middle * jnp.expm1(middle) / (np.e - 1.0)
    return jnp.where(positions < start, 1.0, jnp.where(positions < end, middle, 0.0))


def _rtc_sample_actions_jax(
    self,
    rng,
    observation,
    *,
    num_steps=10,
    noise=None,
    prev_chunk_left_over=None,
    inference_delay=None,
    execution_horizon=None,
):
    """JAX counterpart of the RTC flow-matching denoiser hook.

    The deployed OpenPI checkout is commonly JAX/Orbax based.  Keeping this
    implementation here means the same model-side RTC path works for both
    checkpoint formats instead of silently falling back to client blending.
    """
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / int(num_steps)
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
    )

    rtc_active = prev_chunk_left_over is not None
    if rtc_active:
        previous = jnp.asarray(prev_chunk_left_over, dtype=noise.dtype)
        if previous.ndim == 2:
            previous = previous[None, ...]
        if previous.ndim != 3:
            raise ValueError("RTC previous chunk must have shape (T,A) or (B,T,A)")
        if previous.shape[-1] != self.action_dim:
            raise ValueError(
                f"RTC action dimension mismatch: previous={previous.shape[-1]} "
                f"current={self.action_dim}"
            )
        if previous.shape[0] == 1 and batch_size != 1:
            previous = jnp.broadcast_to(previous, (batch_size, previous.shape[1], previous.shape[2]))
        elif previous.shape[0] != batch_size:
            raise ValueError("RTC previous chunk batch dimension does not match current batch")
        previous_length = int(previous.shape[1])
        if previous.shape[1] > self.action_horizon:
            previous = previous[:, : self.action_horizon]
            previous_length = self.action_horizon
        elif previous.shape[1] < self.action_horizon:
            previous = jnp.pad(
                previous,
                ((0, 0), (0, self.action_horizon - previous.shape[1]), (0, 0)),
            )
        delay = max(0, int(inference_delay or 0))
        horizon = max(
            1,
            min(
                int(execution_horizon or self.action_horizon),
                self.action_horizon,
                previous_length,
            ),
        )
        weights = _jax_prefix_weights(
            delay,
            horizon,
            self.action_horizon,
            getattr(self, "_rtc_prefix_attention_schedule", "linear"),
        ).reshape(1, self.action_horizon, 1).astype(noise.dtype)
    else:
        previous = None
        weights = None
        delay = 0
        horizon = int(execution_horizon or self.action_horizon)

    def denoise(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_len = suffix_tokens.shape[1]
        prefix_len = prefix_tokens.shape[1]
        prefix_attn = jnp.broadcast_to(prefix_mask[:, None, :], (batch_size, suffix_len, prefix_len))
        suffix_attn = make_attn_mask(suffix_mask, suffix_ar_mask)
        full_attn = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def step(carry):
        x_t, time = carry
        v_t = denoise(x_t, time)
        if rtc_active:
            x1_t = x_t - time * v_t
            error = (previous - x1_t) * weights

            def clean_action(x):
                return x - time * denoise(x, time)

            _, pullback = jax.vjp(clean_action, x_t)
            correction = pullback(error)[0]
            tau = 1.0 - time
            eps = jnp.finfo(x_t.dtype).eps
            tau_safe = jnp.maximum(tau, eps)
            denominator = jnp.maximum((1.0 - tau) ** 2, eps)
            guidance_weight = ((1.0 - tau) / tau_safe) * (
                ((1.0 - tau) ** 2 + tau**2) / denominator
            )
            guidance_weight = jnp.nan_to_num(
                guidance_weight,
                nan=self._rtc_max_guidance_weight,
                posinf=self._rtc_max_guidance_weight,
                neginf=0.0,
            )
            guidance_weight = jnp.clip(guidance_weight, 0.0, self._rtc_max_guidance_weight)
            v_t = v_t - guidance_weight * correction
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, jnp.asarray(1.0, dtype=noise.dtype)))
    return x_0


def patch_jax_model(model: Any, config: RTCConfig) -> None:
    """Install the RTC sampler on an OpenPI JAX/NNX model instance."""
    required = ("embed_prefix", "embed_suffix", "PaliGemma", "action_out_proj")
    if not all(hasattr(model, name) for name in required):
        raise TypeError("RTC requires an OpenPI Pi0 JAX/NNX flow-matching model")
    model._rtc_max_guidance_weight = float(config.max_guidance_weight)
    model._rtc_prefix_attention_schedule = str(config.prefix_attention_schedule)
    model._rtc_original_sample_actions = getattr(model, "sample_actions", None)
    model.sample_actions = types.MethodType(_rtc_sample_actions_jax, model)


@dataclass
class _RTCSession:
    normalized_actions: np.ndarray | None = None
    generation: int | None = None


class RTCAwarePolicy:
    """Adapter that feeds per-client RTC state into an upstream OpenPI Policy."""

    def __init__(self, policy: Any, config: RTCConfig):
        self.policy = policy
        self.config = config
        self._sessions: dict[str, _RTCSession] = {}
        self._lock = threading.Lock()
        self._last_normalized_actions: np.ndarray | None = None
        model = getattr(policy, "_model", None)
        if model is None:
            raise TypeError("OpenPI Policy does not expose its model")
        if bool(getattr(policy, "_is_pytorch_model", False)):
            patch_pytorch_model(model, config)
            sample_actions = model.sample_actions
        else:
            patch_jax_model(model, config)
            # JAX policies cache nnx_utils.module_jit(model.sample_actions) in
            # their constructor. Rebuild that wrapper after patching the
            # bound method, then keep a Python-side normalized-action copy for
            # the next request's per-session RTC prefix.
            try:
                from openpi.shared import nnx_utils

                sample_actions = nnx_utils.module_jit(model.sample_actions)
            except ImportError as exc:  # pragma: no cover - deployment-only path
                raise RuntimeError("RTC JAX adapter requires OpenPI's nnx_utils") from exc

        def capture_normalized_actions(*args: Any, **kwargs: Any):
            self._last_normalized_actions = None
            output = sample_actions(*args, **kwargs)
            raw_output = output
            if hasattr(raw_output, "detach"):
                raw_output = raw_output.detach().float().cpu().numpy()
            self._last_normalized_actions = np.asarray(raw_output, dtype=np.float32)
            return output

        # Policy.__init__ cached the old bound method in _sample_actions.
        policy._sample_actions = capture_normalized_actions

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self.policy, "metadata", {})

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        client = observation.get("client_metadata")
        client = client if isinstance(client, dict) else {}
        rtc = client.get("rtc")
        rtc = rtc if isinstance(rtc, dict) else {}
        session_id = str(rtc.get("session_id") or client.get("source_name") or "default")
        enabled = bool(rtc.get("enabled", False))
        delay = max(0, int(rtc.get("inference_delay_steps", 0) or 0))
        offset = max(0, int(rtc.get("previous_chunk_offset_steps", 0) or 0))
        requested_horizon = rtc.get("execution_horizon")
        execution_horizon = (
            self.config.execution_horizon
            if requested_horizon is None
            else max(1, min(int(requested_horizon), self.config.execution_horizon))
        )
        client_previous_generation = rtc.get("previous_chunk_generation")
        try:
            client_previous_generation = (
                None
                if client_previous_generation is None
                else int(client_previous_generation)
            )
        except (TypeError, ValueError):
            client_previous_generation = None

        with self._lock:
            session = self._sessions.setdefault(session_id, _RTCSession())
            previous = session.normalized_actions
            previous_left_over = None
            generation_matches = (
                previous is not None
                and session.generation is not None
                and client_previous_generation == session.generation
            )
            if enabled and generation_matches:
                offset = min(offset, len(previous))
                previous_left_over = previous[offset:].copy()
            previous_steps = 0 if previous_left_over is None else len(previous_left_over)

            sample_kwargs = dict(getattr(self.policy, "_sample_kwargs", {}) or {})
            if enabled and previous_left_over is not None and previous_steps:
                sample_kwargs.update(
                    {
                        "prev_chunk_left_over": previous_left_over,
                        "inference_delay": delay,
                        "execution_horizon": execution_horizon,
                    }
                )
            else:
                sample_kwargs.pop("prev_chunk_left_over", None)
                sample_kwargs.pop("inference_delay", None)
                sample_kwargs.pop("execution_horizon", None)
            old_kwargs = getattr(self.policy, "_sample_kwargs", {})
            self.policy._sample_kwargs = sample_kwargs
            try:
                result = dict(self.policy.infer(observation))
            finally:
                self.policy._sample_kwargs = old_kwargs

            normalized = self._last_normalized_actions
            if normalized is None:
                raise RuntimeError("RTC model did not expose its normalized action chunk")
            normalized = np.asarray(normalized, dtype=np.float32)
            if normalized.ndim == 3 and normalized.shape[0] == 1:
                normalized = normalized[0]
            if normalized.ndim != 2 or not np.all(np.isfinite(normalized)):
                raise RuntimeError(f"RTC normalized action chunk is invalid: {normalized.shape}")
            session.normalized_actions = normalized.copy()
            generation = rtc.get("inference_generation")
            session.generation = int(generation) if generation is not None else None

        result["rtc"] = {
            "enabled": bool(enabled and previous_left_over is not None and previous_steps),
            "algorithm": "real_time_chunking_prefix_guidance",
            "inference_delay_steps": delay,
            "previous_chunk_offset_steps": offset,
            "previous_chunk_left_over_steps": previous_steps,
            "execution_horizon": execution_horizon,
            "prefix_attention_schedule": self.config.prefix_attention_schedule,
            "max_guidance_weight": self.config.max_guidance_weight,
        }
        return result

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            reset()


def build_rtc_policy(policy: Any, config: RTCConfig) -> RTCAwarePolicy:
    """Build the RTC adapter and fail loudly for unsupported model backends."""
    return RTCAwarePolicy(policy, config)
