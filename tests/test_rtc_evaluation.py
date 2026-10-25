"""CPU regression tests for prefix-conditioned evaluation and metric aggregation."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import argparse
import json
import types
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server_4090"))
from rtc_evaluation import evaluation_delays, prefix_loss_per_dim

HAS_OPENPI = all(importlib.util.find_spec(name) for name in ("jax", "flax", "openpi"))


class DelayTests(unittest.TestCase):
    def test_defaults_and_bounds(self):
        self.assertEqual(evaluation_delays(10, 50), (0, 4, 5, 6, 7, 8, 10))
        self.assertEqual(evaluation_delays(3, 50), (0, 3))
        self.assertEqual(evaluation_delays(9, 50), (0, 4, 5, 6, 7, 8, 9))
        self.assertEqual(evaluation_delays(0, 50), (0,))
        self.assertEqual(evaluation_delays(10, 50, (8, 6, 8)), (0, 6, 8))
        for maximum, horizon, requested in ((10, 10, None), (0, 50, (1,)),
                                             (10, 50, (-1,)), (10, 50, (1.5,))):
            with self.assertRaises(ValueError):
                evaluation_delays(maximum, horizon, requested)


@unittest.skipUnless(HAS_OPENPI, "requires OpenPI JAX environment")
class PrefixLossTests(unittest.TestCase):
    def setUp(self):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_lib
        self.jax, self.jnp = jax, jnp

        class TinyModel(nnx.Module):
            action_horizon = 5
            action_dim = 4
            loss_action_dim = 2
            rtc_training_max_delay = 2

            def __init__(self):
                self.PaliGemma = nnx.Dict(llm=self.llm)

            def embed_prefix(self, observation):
                return jnp.zeros((2, 1, 4)), jnp.ones((2, 1), bool), jnp.zeros(1, bool)

            def embed_suffix(self, observation, noisy, times):
                return noisy, jnp.ones((2, 5), bool), jnp.zeros(5, bool), times

            @staticmethod
            def llm(tokens, **kwargs):
                # Prefix conditions actually influence generated rows.
                suffix = tokens[1]
                return (None, suffix + suffix.mean(axis=1, keepdims=True)), None

            def compute_loss_per_dim(self, rng, observation, actions, *, train=False):
                assert not train
                return prefix_loss_per_dim(self, rng, observation, actions, delay=0)

            @staticmethod
            def action_out_proj(output):
                return output

        self.model = TinyModel()
        self.rng = jax.random.key(9)
        self.actions = jnp.ones((2, 5, 4))
        self.preprocess = self.enterContext(patch.object(
            model_lib, "preprocess_observation", side_effect=lambda rng, obs, **kw: obs))

    def test_suffix_only_denominator_and_no_augmentation(self):
        jax, jnp = self.jax, self.jnp
        loss = prefix_loss_per_dim(self.model, self.rng, None, self.actions, delay=2)
        np.testing.assert_array_equal(loss[:, :2], 0)
        _, noise_rng, time_rng, _ = jax.random.split(self.rng, 4)
        noise = jax.random.normal(noise_rng, self.actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (2,)) * .999 + .001
        times = jnp.broadcast_to(time[:, None], (2, 5)).at[:, :2].set(0)
        x = times[..., None] * noise + (1 - times[..., None]) * self.actions
        expected = (x + x.mean(axis=1, keepdims=True) - noise + self.actions) ** 2
        np.testing.assert_allclose(loss.mean(), expected[:, 2:].mean(), rtol=1e-6)
        self.assertFalse(self.preprocess.call_args.kwargs["train"])

    def test_planner_prefix_padding_and_suffix_labels(self):
        jnp = self.jnp
        prefix = jnp.full((2, 2, 4), 3.)
        loss = prefix_loss_per_dim(self.model, self.rng, None, self.actions, delay=2,
                                   planned_prefix=prefix)
        clean = prefix.at[:, :, 2:].set(0)
        reference = prefix_loss_per_dim(self.model, self.rng, None, self.actions, delay=2,
                                        planned_prefix=clean)
        np.testing.assert_array_equal(loss, reference)
        demo = prefix_loss_per_dim(self.model, self.rng, None, self.actions, delay=2)
        self.assertGreater(float(jnp.max(jnp.abs(loss[:, 2:] - demo[:, 2:]))), 0)
        for bad in (jnp.zeros((2, 1, 4)), jnp.zeros((1, 2, 4)), jnp.zeros((2, 2, 3))):
            with self.assertRaises(ValueError):
                prefix_loss_per_dim(self.model, self.rng, None, self.actions, delay=2,
                                    planned_prefix=bad)

    def test_runner_publishes_paired_conditions_and_preserves_baseline(self):
        runner_path = Path(__file__).resolve().parents[1] / "server_4090/eval_heldout_loss.py"
        spec = importlib.util.spec_from_file_location("rtc_eval_runner_test", runner_path)
        runner = importlib.util.module_from_spec(spec)
        piper = MagicMock()
        with patch.dict(sys.modules, {"openpi_single_arm": piper}):
            spec.loader.exec_module(runner)
        config = types.SimpleNamespace(
            model=types.SimpleNamespace(load=lambda params: self.model, action_horizon=5,
                                        loss_action_dim=2),
            policy_metadata={"model_action_dim": 2, "schema": "joint",
                             "finetuning": {"rtc_training": True}})
        piper.build_config.return_value = config
        piper._resolve_training_split.return_value = types.SimpleNamespace(
            test_episodes=(1, 2), test_ratio=.1, seed=42)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "20000"
            (checkpoint / "params").mkdir(parents=True)
            (checkpoint / "_CHECKPOINT_METADATA").touch()
            (checkpoint / "params/_METADATA").touch()
            args = argparse.Namespace(checkpoint=checkpoint, result_json=Path(folder) / "result.json",
                                      dataset_id="test", batch_size=2, max_batches=2,
                                      num_workers=0, eval_seed=9, schema="joint", rtc_eval_delays=None)
            loader = [(None, self.actions), (None, self.actions * 2)]
            with patch.object(runner, "_build_config_args", return_value=args), \
                 patch.object(runner, "_create_test_loader", return_value=(loader, 4, 2)), \
                 patch.object(runner.model_lib, "restore_params", return_value={}):
                result = runner.run(args)
            self.assertEqual(result["rtc_evaluation"]["delays"], [0, 2])
            self.assertFalse(result["rtc_evaluation"]["planner_prefix_evaluated"])
            self.assertEqual(result["eval_loss_model"], result["eval_rtc_demo_d0_loss_model"])
            self.assertEqual(result["rtc_evaluation"]["by_delay"]["2"]["generated_rows_per_sample"], 3)
            expected = []
            for i, (_, actions) in enumerate(loader):
                key = self.jax.random.fold_in(self.rng, i)
                expected.append(np.asarray(prefix_loss_per_dim(self.model, key, None, actions, delay=2)))
            np.testing.assert_allclose(result["eval_rtc_demo_d2_loss_model"],
                                       np.concatenate(expected)[..., :2].mean(), rtol=2e-6)
            self.assertEqual(json.loads(args.result_json.read_text())["version"], 2)

    def test_jitted_bound_method(self):
        from openpi.shared import nnx_utils
        fn = nnx_utils.module_jit(types.MethodType(prefix_loss_per_dim, self.model),
                                  static_argnames=("delay",))
        actual = fn(self.rng, None, self.actions, delay=2)
        expected = prefix_loss_per_dim(self.model, self.rng, None, self.actions, delay=2)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
