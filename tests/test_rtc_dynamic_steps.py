"""Run the trained sampler's actual control flow on a tiny CPU identity model.

Extract the method to avoid importing OpenPI's weight/training dependencies;
the function body is compiled directly from the production source.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    raise unittest.SkipTest('requires a Conda environment with JAX') from exc
import numpy as np


class DynamicStepsTest(unittest.TestCase):
    def test_dynamic_counts_share_trace_and_keep_clean_prefix_and_math(self):
        source = Path(__file__).parents[1]/'bimanual_vla/training/openpi_model.py'
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PiperPi0')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'sample_actions')
        namespace = dict(jax=jax, jnp=jnp,
            model_lib=SimpleNamespace(preprocess_observation=lambda rng, obs, train: obs),
            pi0=SimpleNamespace(make_attn_mask=lambda mask, ar: jnp.ones(
                (mask.shape[0], mask.shape[1], mask.shape[1]), dtype=bool)))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        sampler = namespace['sample_actions']
        model = SimpleNamespace(
            rtc_training_max_delay=10, action_horizon=8, action_dim=32, loss_action_dim=14,
            embed_prefix=lambda obs: (jnp.zeros((1, 1, 32)), jnp.ones((1, 1), dtype=bool), jnp.zeros(1, dtype=bool)),
            embed_suffix=lambda obs, actions, time: (actions, jnp.ones((1, 8), dtype=bool), jnp.arange(8) == 0, None),
            PaliGemma=SimpleNamespace(llm=lambda embeddings, **kwargs: (embeddings, {})),
            action_out_proj=lambda values: values)
        for guided in (False, True):
            traces = []
            def sample(steps):
                traces.append(True)
                return sampler(model, jax.random.key(0), SimpleNamespace(state=jnp.zeros((1, 32))),
                    num_steps=steps, noise=jnp.ones((1, 8, 32)),
                    prev_chunk_left_over=jnp.full((8, 32), .3) if guided else None,
                    inference_delay=3 if guided else 0)
            compiled = jax.jit(sample)
            for steps in (5, 2, 7, 2):
                result = np.asarray(compiled(np.asarray(steps, dtype=np.int32)))
                start = 3 if guided else 0
                np.testing.assert_allclose(result[:, start:, :14], (1.-1./steps)**steps, atol=1e-6)
                np.testing.assert_array_equal(result[..., 14:], 0.)
                if guided:
                    np.testing.assert_allclose(result[:, :3, :14], .3)
            self.assertEqual(len(traces), 1)
        with self.assertRaisesRegex(ValueError, 'positive'):
            sampler(model, None, None, num_steps=0)


if __name__ == '__main__':
    unittest.main()
