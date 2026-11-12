"""Regression tests for the frozen-bf16 EMA drift found in the 20k audit."""
import dataclasses
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from flax import nnx, struct
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import optax

from bimanual_vla.training.ema import install_training_ema, normalize_ema, update_ema


def trainable(path, variable):
    return path[0] == "head"


def params(head=1.0, frozen=0.26953125):
    return nnx.State({
        "head": nnx.VariableState(nnx.Param, jnp.array([head], jnp.bfloat16)),
        "frozen": nnx.VariableState(nnx.Param, jnp.array([frozen], jnp.bfloat16)),
    })


@struct.dataclass
class State:
    params: nnx.State
    ema_params: nnx.State | None
    step: int = 0
    ema_decay: float | None = struct.field(pytree_node=False, default=0.99)


class TrainingEMATest(unittest.TestCase):
    def test_original_norm_value_reproduces_old_drift_and_stays_fixed(self):
        # Original final_norm/scale[1]: Raw .87109375, EMA .83203125.
        raw = params(frozen=.87109375)
        value = raw["frozen"].value
        old = jax.jit(lambda x: jax.lax.fori_loop(
            0, 100, lambda _, y: .99 * y + (1 - .99) * x, x))(value)
        np.testing.assert_array_equal(old, jnp.array([.83203125], jnp.bfloat16))
        repaired = jax.jit(lambda e: jax.lax.fori_loop(
            0, 100, lambda _, y: update_ema(raw, y, trainable, .99), e))(
                normalize_ema(raw, None, trainable))
        np.testing.assert_array_equal(repaired["frozen"].value, value)

    def test_constant_frozen_params_survive_20000_compiled_updates(self):
        raw = params()
        ema = normalize_ema(raw, None, trainable)
        result = jax.jit(lambda value: jax.lax.fori_loop(
            0, 20000, lambda _, e: update_ema(raw, e, trainable, 0.99), value))(ema)
        np.testing.assert_array_equal(result["frozen"].value, raw["frozen"].value)
        self.assertEqual(result["frozen"].value.dtype, jnp.bfloat16)
        self.assertEqual(result["head"].value.dtype, jnp.float32)

    def test_trainable_ema_tracks_known_sequence_in_fp32(self):
        raw = params()
        ema = normalize_ema(raw, None, trainable)
        expected = 1.0
        for value in [2.0, 4.0, -1.0, 3.0]:
            raw = params(head=value)
            ema = update_ema(raw, ema, trainable, 0.9)
            expected = 0.9 * expected + 0.1 * value
        np.testing.assert_allclose(ema["head"].value, [expected], atol=1e-6)

    def test_resume_repairs_only_frozen_values(self):
        raw, old = params(head=2), params(head=1.5, frozen=0.5)
        migrated = normalize_ema(raw, old, trainable)
        np.testing.assert_array_equal(migrated["head"].value, old["head"].value)
        np.testing.assert_array_equal(migrated["frozen"].value, raw["frozen"].value)
        abstract = jax.eval_shape(lambda p: p, State(raw, old))
        migrated_shape = normalize_ema(abstract.params, abstract.ema_params, trainable)
        self.assertEqual(migrated_shape["head"].value.dtype, jnp.float32)

    def test_orbax_roundtrip_preserves_accumulator_and_continuation(self):
        raw = params()
        ema = update_ema(params(head=2), normalize_ema(raw, None, trainable), trainable, .99)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params"
            with ocp.PyTreeCheckpointer() as checkpoint:
                checkpoint.save(path, {"params": ema.to_pure_dict()})
                restored = checkpoint.restore(path)["params"]
            reloaded = nnx.State({k: nnx.VariableState(nnx.Param, v) for k, v in restored.items()})
            resumed = update_ema(params(head=3), reloaded, trainable, .99)
            direct = update_ema(params(head=3), ema, trainable, .99)
            for key in ("head", "frozen"):
                np.testing.assert_array_equal(resumed[key].value, direct[key].value)

    def test_adapter_keeps_optimizer_step_and_scopes_restore_hook(self):
        raw, old = params(head=2), params(head=1, frozen=.5)
        original = State(raw, old)
        def optimizer_step(config, rng, state, batch):
            self.assertIsNone(state.ema_decay)
            self.assertIsNone(state.ema_params)
            return dataclasses.replace(state, params=params(head=3), step=state.step + 1), {"loss": 7}
        checkpoints = SimpleNamespace(restore_state=lambda *a, **k: original)
        module = SimpleNamespace(init_train_state=lambda *a, **k: (original, "sharding"),
            train_step=optimizer_step, _checkpoints=checkpoints)
        config = SimpleNamespace(trainable_filter=trainable)
        install_training_ema(module, config)
        state, sharding = module.init_train_state(config)
        self.assertEqual(sharding, "sharding")
        new, info = module.train_step(config, None, state, None)
        self.assertEqual(new.step, 1)
        self.assertEqual(info["loss"], 7)
        np.testing.assert_allclose(new.ema_params["head"].value, [1.02], atol=1e-6)
        np.testing.assert_array_equal(new.ema_params["frozen"].value, raw["frozen"].value)
        restored = module._checkpoints.restore_state()
        np.testing.assert_array_equal(restored.ema_params["frozen"].value, raw["frozen"].value)
        self.assertIs(checkpoints.restore_state(), original)
        wrapper = module.train_step
        install_training_ema(module, config)
        self.assertIs(module.train_step, wrapper)

    def test_ema_disabled_is_unchanged(self):
        state = State(params(), None, ema_decay=None)
        config = SimpleNamespace(trainable_filter=trainable)
        module = SimpleNamespace(init_train_state=lambda *a, **k: (state, None),
            train_step=lambda *a, **k: (state, {}),
            _checkpoints=SimpleNamespace(restore_state=lambda *a, **k: state))
        install_training_ema(module, config)
        self.assertIs(module.init_train_state(config)[0], state)
        self.assertIs(module.train_step(config, None, state, None)[0], state)

    def test_real_upstream_gradient_step_is_unchanged(self):
        import openpi
        from openpi.models import model as model_lib
        from openpi.shared import array_typing as at
        from openpi.training.utils import TrainState
        script = Path(openpi.__file__).parents[2] / "scripts/train.py"
        spec = importlib.util.spec_from_file_location("ema_test_upstream", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class TinyModel(nnx.Module):
            def __init__(self):
                self.head = nnx.Param(jnp.array([[1.]], jnp.float32))
                self.frozen = nnx.Param(jnp.array([.26953125], jnp.bfloat16))

            def compute_loss(self, rng, observation, actions, *, train):
                return jnp.square(observation.state @ self.head.value + self.frozen.value - actions)

        model = TinyModel()
        graphdef, raw = nnx.split(model)
        tx = optax.sgd(.1)
        config = SimpleNamespace(trainable_filter=trainable)
        batch = (model_lib.Observation(images={}, image_masks={}, state=jnp.ones((2, 1))),
                 jnp.zeros((2, 1)))
        with at.disable_typechecking():
            baseline = TrainState(step=0, params=raw, model_def=graphdef,
                tx=tx, opt_state=tx.init(raw.filter(trainable)), ema_decay=None, ema_params=None)
            expected, expected_info = module.train_step(config, jax.random.key(1), baseline, batch)
            state = dataclasses.replace(baseline, ema_decay=.99,
                ema_params=normalize_ema(raw, None, trainable))
            install_training_ema(module, config)
            actual, info = jax.jit(lambda s: module.train_step(config, jax.random.key(1), s, batch))(state)
        for name in ("head", "frozen"):
            np.testing.assert_array_equal(actual.params[name].value, expected.params[name].value)
        np.testing.assert_allclose(info["loss"], expected_info["loss"])
        np.testing.assert_array_equal(actual.ema_params["frozen"].value, raw["frozen"].value)
        np.testing.assert_allclose(actual.ema_params["head"].value,
            .99 * raw["head"].value + .01 * expected.params["head"].value, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
