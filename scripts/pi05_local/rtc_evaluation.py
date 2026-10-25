"""Evaluation-only PyTorch equivalent of PiperPi0's RTC training objective.

Uses ground-truth prefix actions, as training does. It does not simulate a
previous policy chunk or measure asynchronous real-time control performance.
"""
import types


def install_per_token_time(model):
    """Extend only this model instance; preserve ordinary scalar-time behavior."""
    import torch
    from torch.nn import functional as F
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding

    original_suffix = model.embed_suffix

    def suffix(self, state, actions, timestep):
        if timestep.ndim == 1:
            return original_suffix(state, actions, timestep)
        if not self.pi05 or timestep.shape != actions.shape[:2]:
            raise ValueError('RTC time must have shape BxT for Pi05')
        tokens = self.action_in_proj(actions)
        times = create_sinusoidal_pos_embedding(
            timestep.reshape(-1), self.action_in_proj.out_features,
            min_period=4e-3, max_period=4.0, device=timestep.device).to(timestep.dtype)
        cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(times))))
        cond = cond.reshape(*timestep.shape, -1)
        mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        attention = torch.zeros(actions.shape[:2], dtype=tokens.dtype, device=actions.device)
        attention[:, 0] = 1
        return tokens, mask, attention, cond

    model.embed_suffix = types.MethodType(suffix, model)
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != 'GemmaRMSNorm' or module.dense is None:
            continue
        original_norm = module.forward

        def norm(self, x, cond=None, _original=original_norm):
            if cond is None or cond.ndim != 3:
                return _original(x, cond=cond)
            if cond.shape[:2] != x.shape[:2] or cond.shape[-1] != self.cond_dim:
                raise ValueError('Per-token adaptive norm conditioning shape mismatch')
            scale, shift, gate = self.dense(cond).chunk(3, dim=-1)
            normalized = self._norm(x) * (1 + scale.float()) + shift.float()
            return normalized.to(x.dtype), gate.to(x.dtype)

        module.forward = types.MethodType(norm, module)
        patched += 1
    if not patched:
        raise ValueError('No adaptive RMSNorm modules found')
    return patched


def flow_loss(model, observation, actions, noise, time, delay):
    """Return BxTxD loss, masked/scaled to mean over the predicted postfix."""
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    horizon = actions.shape[1]
    if not 0 <= delay < horizon:
        raise ValueError('RTC prefix must leave at least one predicted action')
    images, img_masks, language, lang_masks, state = model._preprocess_observation(observation, train=False)
    prefix = torch.arange(horizon, device=actions.device)[None] < delay
    token_time = torch.where(prefix, 0.0, time[:, None])
    noisy = token_time[..., None] * noise + (1 - token_time[..., None]) * actions
    pe, pm, pa = model.embed_prefix(images, img_masks, language, lang_masks)
    se, sm, sa, cond = model.embed_suffix(state, noisy, token_time)
    if model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        pe, se = pe.bfloat16(), se.bfloat16()
    masks = torch.cat([pm, sm], dim=1)
    attention = make_att_2d_masks(masks, torch.cat([pa, sa], dim=1))
    (_, suffix), _ = model.paligemma_with_expert.forward(
        attention_mask=model._prepare_attention_masks_4d(attention),
        position_ids=masks.cumsum(dim=1) - 1, past_key_values=None,
        inputs_embeds=[pe, se], use_cache=False, adarms_cond=[None, cond])
    velocity = model.action_out_proj(suffix[:, -horizon:].float())
    loss = (velocity - (noise - actions)).square()
    return loss * (~prefix)[..., None] * (horizon / (horizon - delay))
