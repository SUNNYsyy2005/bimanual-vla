"""Exercise the real trained sampler control flow with a tiny identity network."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytest

jax = pytest.importorskip("jax", reason="requires the OpenPI JAX training environment")
pytest.importorskip("openpi", reason="requires the OpenPI training environment")
import jax.numpy as jnp
import numpy as np

from bimanual_vla.training.openpi_model import PiperPi0


class TrainedPrefixPaddingTest(unittest.TestCase):
    @staticmethod
    def sample(previous, delay, physical_dim=14):
        model = SimpleNamespace(
            rtc_training_max_delay=10, action_horizon=8, action_dim=32,
            loss_action_dim=physical_dim,
            embed_prefix=lambda obs: (jnp.zeros((1, 1, 32)),
                jnp.ones((1, 1), dtype=bool), jnp.zeros((1,), dtype=bool)),
            embed_suffix=lambda obs, actions, time: (actions,
                jnp.ones((1, 8), dtype=bool), jnp.arange(8) == 0, None),
            PaliGemma=SimpleNamespace(llm=lambda embeddings, **kwargs: (embeddings, {})),
            action_out_proj=lambda values: values,
        )
        with patch("bimanual_vla.training.openpi_model.model_lib.preprocess_observation",
                   side_effect=lambda rng, obs, train: obs):
            return np.asarray(PiperPi0.sample_actions(model, jax.random.key(0),
                SimpleNamespace(state=jnp.zeros((1, 32))), num_steps=5,
                noise=jnp.ones((1, 8, 32)), prev_chunk_left_over=previous,
                inference_delay=delay))

    def test_nonzero_padding_cannot_contaminate_clean_prefix(self):
        previous = np.full((8, 32), 0.3, dtype=np.float32)
        previous[:, 14:] = 7.0
        result = self.sample(previous, 3)
        np.testing.assert_array_equal(result[0, :3, :14], previous[:3, :14])
        np.testing.assert_array_equal(result[0, :3, 14:], np.zeros((3, 18)))
        # The predicted postfix must keep its original noise/flow distribution.
        np.testing.assert_allclose(result[0, 3:, 14:], 0.8 ** 5, atol=1e-6)
        np.testing.assert_array_equal(previous[:, 14:], np.full((8, 18), 7.0))

    def test_no_prefix_output_is_unchanged(self):
        dirty = np.full((8, 32), 9.0, dtype=np.float32)
        np.testing.assert_array_equal(self.sample(dirty, 0), self.sample(None, 0))

    def test_full_width_physical_actions_are_preserved(self):
        previous = np.full((8, 32), 0.7, dtype=np.float32)
        result = self.sample(previous, 3, physical_dim=32)
        np.testing.assert_array_equal(result[0, :3], previous[:3])


if __name__ == "__main__":
    unittest.main()
