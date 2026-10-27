"""CPU-sized tests; run in the OpenPI training environment, without checkpoints."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

HAS_OPENPI = all(importlib.util.find_spec(name) for name in ("jax", "flax", "openpi"))
if HAS_OPENPI:
    import flax.linen as nn
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import gemma, model as model_lib, pi0
    from bimanual_vla.training.openpi_model import PiperPi0Config, install_per_action_adarms


@unittest.skipUnless(HAS_OPENPI, "requires the OpenPI JAX training environment")
class OpenPITrainingTest(unittest.TestCase):
    def setUp(self):
        self.original_norm, self.original_module = gemma.RMSNorm, gemma.Module
        self.addCleanup(setattr, gemma, "RMSNorm", self.original_norm)
        self.addCleanup(setattr, gemma, "Module", self.original_module)

    def test_adaptive_norm_preserves_parameters_and_broadcast_behavior(self):
        x = jax.random.normal(jax.random.key(1), (2, 4, 8))
        cond = jnp.ones((2, 8))
        original = self.original_norm()
        params = original.init(jax.random.key(0), x, cond)
        # Nonzero modulation exercises scale, shift and gate broadcasting.
        params["params"]["Dense_0"]["kernel"] = jnp.full((8, 24), 0.02)
        expected = original.apply(params, x, cond)
        install_per_action_adarms()
        adapted = gemma.RMSNorm()
        actual = adapted.apply(params, x, cond)
        token_actual = adapted.apply(params, x, jnp.broadcast_to(cond[:, None], (2, 4, 8)))
        for index in (0, 1):
            np.testing.assert_allclose(actual[index], expected[index], atol=1e-6)
            np.testing.assert_allclose(token_actual[index], jnp.broadcast_to(expected[index], token_actual[index].shape), atol=1e-6)
        plain_params = original.init(jax.random.key(0), x, None)
        np.testing.assert_allclose(adapted.apply(plain_params, x, None)[0], original.apply(plain_params, x, None)[0])

    def tiny_model(self, *, rtc=2, llm_lora=False, expert="full", resize="pad", weights=None, train_vision=False, train_llm=False):
        class TinyVision(nn.Module):
            @nn.compact
            def __call__(self, images, train=False):
                return nn.Dense(8)(images.mean(axis=(1, 2))[:, None, :]), None

        def tiny_config(variant):
            lora = {"attn": gemma.lora.LoRAConfig(rank=2, alpha=2),
                    "ffn": gemma.lora.LoRAConfig(rank=2, alpha=2)} if "lora" in variant else {}
            return gemma.Config(width=8, depth=1, mlp_dim=16, num_heads=2,
                                num_kv_heads=1, head_dim=4, lora_configs=lora)

        config = PiperPi0Config(pi05=True, dtype="float32", action_dim=4, action_horizon=5,
                                max_token_len=2, loss_action_dim=2,
                                paligemma_variant="gemma_2b_lora" if llm_lora else "gemma_2b",
                                action_expert_variant="gemma_300m_lora" if expert == "lora" else "gemma_300m",
                                llm_lora_enabled=llm_lora, rtc_training_max_delay=rtc,
                                image_resize_mode=resize, rtc_training_delay_weights=weights,
                                train_vision=train_vision, train_llm=train_llm)
        with patch.object(gemma, "get_config", tiny_config), patch.object(gemma, "PALIGEMMA_VOCAB_SIZE", 32), patch.object(pi0._siglip, "Module", lambda **kw: TinyVision()):
            model = config.create(jax.random.key(1))
        obs = model_lib.Observation(images={"base_0_rgb": jnp.zeros((2, 4, 4, 3))},
                                    image_masks={"base_0_rgb": jnp.ones(2, dtype=bool)},
                                    state=jnp.zeros((2, 4)), tokenized_prompt=None,
                                    tokenized_prompt_mask=None)
        return config, model, obs

    def test_stretch_training_with_weighted_prefix(self):
        _, model, obs = self.tiny_model(resize="stretch", weights=(0,0,1))
        with patch.object(model_lib, "preprocess_observation", lambda rng, obs, **kw: obs):
            losses=model.compute_loss_per_dim(jax.random.key(9),obs,jnp.ones((2,5,4)),train=True)
        np.testing.assert_array_equal(np.asarray(losses)[:,:2],0)
        self.assertTrue(np.isfinite(losses).all())

    def test_clean_prefix_loss_gradients_and_jitted_sampling(self):
        _, model, obs = self.tiny_model()
        actions = jnp.ones((2, 5, 4))
        with patch.object(model_lib, "preprocess_observation", lambda rng, obs, **kw: obs):
            rng = jax.random.key(7)
            delays = jax.random.randint(jax.random.split(rng, 4)[3], (2,), 0, 3)
            prefix = np.arange(5)[None, :] < np.asarray(delays)[:, None]
            losses = model.compute_loss_per_dim(rng, obs, actions, train=True)
            self.assertTrue(prefix.any())
            np.testing.assert_array_equal(np.asarray(losses)[prefix], 0)
            self.assertTrue(np.isfinite(losses).all())
            gradients = nnx.grad(lambda m: m.compute_loss(rng, obs, actions, train=True).mean())(model)
            head_gradient = np.asarray(gradients.action_out_proj.kernel.value)
            self.assertGreater(np.max(np.abs(head_gradient[:, :2])), 0)
            np.testing.assert_array_equal(head_gradient[:, 2:], 0)
            previous = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)
            sample = jax.jit(lambda key: model.sample_actions(key, obs, num_steps=2,
                             prev_chunk_left_over=previous, inference_delay=2))(rng)
            np.testing.assert_array_equal(sample[:, :2, :2], np.broadcast_to(previous[:2, :2], (2, 2, 2)))
            np.testing.assert_array_equal(sample[:, :, 2:], 0)
            self.assertTrue(np.isfinite(sample).all())

    def test_untrained_padding_head_cannot_change_physical_sampling(self):
        _, model, obs = self.tiny_model()
        rng = jax.random.key(11)
        noise = jax.random.normal(rng, (2, 5, 4))
        previous = jnp.ones((5, 4))
        with patch.object(model_lib, "preprocess_observation", lambda rng, obs, **kw: obs):
            for delay in (0, 2):
                before = model.sample_actions(rng, obs, num_steps=4, noise=noise,
                                               prev_chunk_left_over=previous, inference_delay=delay)
                kernel = model.action_out_proj.kernel.value
                bias = model.action_out_proj.bias.value
                model.action_out_proj.kernel.value = kernel.at[:, 2:].set(1000)
                model.action_out_proj.bias.value = bias.at[2:].set(-1000)
                after = model.sample_actions(rng, obs, num_steps=4, noise=noise,
                                              prev_chunk_left_over=previous, inference_delay=delay)
                model.action_out_proj.kernel.value = kernel
                model.action_out_proj.bias.value = bias
                np.testing.assert_allclose(after[:, :, :2], before[:, :, :2], atol=1e-6)
                np.testing.assert_array_equal(after[:, :, 2:], 0)
                if delay:
                    np.testing.assert_array_equal(after[:, :delay, :2], 1)

    def test_freeze_filter_matches_actual_lora_state_paths(self):
        for llm_lora, expert in ((False, "full"), (True, "full"), (False, "lora"), (True, "lora")):
            config, model, _ = self.tiny_model(llm_lora=llm_lora, expert=expert)
            frozen = config.get_freeze_filter()
            counts = {"vision": 0, "language": 0, "expert": 0, "head": 0}
            for path, value in nnx.state(model, nnx.Param).flat_state().items():
                name = "/".join(map(str, path))
                if "PaliGemma/img" in name:
                    kind, expected = "vision", True
                elif "PaliGemma/llm" in name:
                    if "_1" in name:
                        kind, expected = "expert", expert == "lora" and "lora" not in name
                    else:
                        kind, expected = "language", not (llm_lora and "lora" in name)
                else:
                    kind, expected = "head", False
                counts[kind] += 1
                self.assertEqual(frozen(path, value), expected, name)
            self.assertTrue(all(counts.values()), counts)

    def test_full_backbone_freezes_no_parameters(self):
        config, model, _ = self.tiny_model(train_vision=True, train_llm=True)
        frozen = config.get_freeze_filter()
        params = nnx.state(model, nnx.Param).flat_state()
        self.assertTrue(params)
        for path, value in params.items():
            self.assertFalse(frozen(path, value), str(path))
