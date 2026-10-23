"""Clean-prefix RTC sampling for the local converted Piper Pi05 model."""
import types

from rtc_evaluation import install_per_token_time


def install_trained_sampler(model, max_delay):
    """Reuse upstream KV caching and Euler integration, with token-time prefixes."""
    import torch

    if not model.pi05 or not 0 < max_delay < model.config.action_horizon:
        raise ValueError('Invalid Pi05 RTC training configuration')
    install_per_token_time(model)
    original_sample = model.sample_actions
    original_denoise = model.denoise_step
    active = {}

    def denoise(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        prefix, previous = active['prefix'], active['previous']
        clamped = torch.where(prefix[..., None], previous, x_t)
        times = torch.where(prefix, 0.0, timestep[:, None])
        velocity = original_denoise(state, prefix_pad_masks, past_key_values, clamped, times)
        return torch.where(prefix[..., None], 0.0, velocity)

    @torch.inference_mode()
    def sample(self, device, observation, noise=None, num_steps=5,
               prev_chunk_left_over=None, inference_delay=None,
               previous_left_over_steps=None, **kwargs):
        delay = int(inference_delay or 0)
        if not 0 <= delay <= max_delay:
            raise ValueError(f'RTC delay must be between 0 and {max_delay}')
        if not 1 <= int(num_steps) <= 50:
            raise ValueError('Denoising steps must be between 1 and 50')
        shape = (observation.state.shape[0], self.config.action_horizon, self.config.action_dim)
        previous = torch.zeros(shape, dtype=torch.float32, device=device)
        if delay:
            if prev_chunk_left_over is None or int(previous_left_over_steps or 0) < delay:
                raise ValueError('RTC requires enough valid previous actions for its prefix')
            previous = torch.as_tensor(prev_chunk_left_over, dtype=torch.float32, device=device)
            if previous.ndim == 2:
                previous = previous[None]
            if previous.shape != shape or not torch.isfinite(previous).all():
                raise ValueError('RTC previous chunk must be finite and match the model chunk shape')
        if noise is None:
            noise = self.sample_noise(shape, device)
        noise = torch.as_tensor(noise, device=device, dtype=torch.float32)
        if noise.shape != shape or not torch.isfinite(noise).all():
            raise ValueError('Noise must be finite and match the model chunk shape')
        prefix = (torch.arange(shape[1], device=device)[None] < delay).expand(shape[:2])
        active.update(prefix=prefix, previous=previous)
        try:
            result = original_sample(device, observation,
                noise=torch.where(prefix[..., None], previous, noise), num_steps=int(num_steps))
            return torch.where(prefix[..., None], previous, result)
        finally:
            active.clear()

    model.denoise_step = types.MethodType(denoise, model)
    model.sample_actions = types.MethodType(sample, model)
    model.action_horizon = model.config.action_horizon
    model.rtc_training_max_delay = max_delay
    model._rtc_trained_sampling = True
