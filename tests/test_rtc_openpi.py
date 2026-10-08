from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from bimanual_vla.deployment import rtc_policy as rtc_openpi

from bimanual_vla.deployment.rtc_policy import (
    RTCConfig,
    RTCProcessor,
    _prefix_weights,
    _reanchor_normalized_actions,
)


class RTCProcessorTest(unittest.TestCase):
    def test_prefix_schedule_has_full_attention_then_fade(self):
        weights = _prefix_weights(2, 6, 10, schedule="linear", dtype=torch.float32)
        self.assertEqual(weights.shape, (10,))
        self.assertTrue(torch.allclose(weights[:2], torch.ones(2)))
        self.assertTrue(torch.all(weights[2:6] < 1.0))
        self.assertTrue(torch.all(weights[2:6] > 0.0))
        self.assertTrue(torch.equal(weights[6:], torch.zeros(4)))

    def test_guidance_changes_velocity_toward_previous_chunk(self):
        processor = RTCProcessor(
            RTCConfig(
                execution_horizon=4,
                max_guidance_weight=5.0,
                prefix_attention_schedule="ones",
            )
        )
        x_t = torch.tensor([[[0.4, -0.2], [0.2, 0.3], [0.1, -0.1], [0.0, 0.2]]])
        previous = torch.zeros_like(x_t)

        def denoise(x):
            return 0.25 * x

        base = denoise(x_t)
        guided = processor.denoise_step(
            x_t,
            previous,
            inference_delay=0,
            time=torch.tensor(0.5),
            original_denoise_step=denoise,
            execution_horizon=4,
        )
        self.assertEqual(guided.shape, base.shape)
        self.assertTrue(torch.isfinite(guided).all())
        self.assertFalse(torch.allclose(guided, base))

    def test_no_previous_chunk_is_exact_base_velocity(self):
        processor = RTCProcessor(RTCConfig())
        x_t = torch.randn(1, 4, 2)

        def denoise(x):
            return x * 0.5

        self.assertTrue(
            torch.allclose(
                processor.denoise_step(
                    x_t,
                    None,
                    inference_delay=0,
                    time=torch.tensor(0.5),
                    original_denoise_step=denoise,
                ),
                denoise(x_t),
            )
        )

    def test_unused_model_dimensions_do_not_affect_guidance(self):
        processor = RTCProcessor(
            RTCConfig(
                execution_horizon=2,
                prefix_attention_schedule="ones",
                physical_action_dim=2,
            )
        )
        x_t = torch.tensor([[[0.4, -0.2, 0.3, -0.1], [0.2, 0.3, -0.4, 0.6]]])
        previous_a = torch.zeros_like(x_t)
        previous_b = previous_a.clone()
        previous_b[..., 2:] = 1_000.0

        def denoise(x):
            return 0.25 * x

        guided_a = processor.denoise_step(
            x_t,
            previous_a,
            inference_delay=0,
            time=torch.tensor(0.5),
            original_denoise_step=denoise,
            execution_horizon=2,
        )
        guided_b = processor.denoise_step(
            x_t,
            previous_b,
            inference_delay=0,
            time=torch.tensor(0.5),
            original_denoise_step=denoise,
            execution_horizon=2,
        )
        self.assertTrue(torch.allclose(guided_a, guided_b))


class RTCReanchorTest(unittest.TestCase):
    @staticmethod
    def _encoder(mask, mean, std, model_action_dim):
        mask = np.asarray(mask, dtype=bool)
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)

        def transform(data):
            state = np.asarray(data["state"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32).copy()
            actions[..., mask] -= state[: len(mask)][mask]
            normalized = (actions - mean) / std
            padded = np.zeros((len(normalized), model_action_dim), dtype=np.float32)
            padded[:, : len(mask)] = normalized
            return {"actions": padded}

        return transform

    @staticmethod
    def _decode(encoded, state, mask, mean, std):
        mask = np.asarray(mask, dtype=bool)
        decoded = np.asarray(encoded, dtype=np.float32)[..., : len(mask)] * std + mean
        decoded = decoded.copy()
        decoded[..., mask] += np.asarray(state, dtype=np.float32)[: len(mask)][mask]
        return decoded

    def test_bimanual_multistep_reanchor_preserves_physical_targets(self):
        mask = (True,) * 6 + (False,) + (True,) * 6 + (False,)
        old_state = np.array(
            [0.50, 0.40, 0.30, 0.20, 0.10, 0.00, 0.25,
             -0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.75],
            dtype=np.float32,
        )
        new_state = np.array(
            [0.56, 0.43, 0.28, 0.24, 0.08, 0.03, 0.90,
             -0.46, -0.44, -0.25, -0.18, -0.14, 0.02, 0.10],
            dtype=np.float32,
        )
        targets = np.stack(
            [old_state + 0.02, old_state + 0.06, old_state + 0.10], axis=0
        )
        targets[:, 6] = [0.2, 0.4, 0.8]
        targets[:, 13] = [0.9, 0.6, 0.3]
        mean = np.linspace(-0.2, 0.2, 14, dtype=np.float32)
        std = np.linspace(0.5, 1.5, 14, dtype=np.float32)
        policy = type("Policy", (), {})()
        policy._input_transform = self._encoder(mask, mean, std, model_action_dim=32)

        old_encoded = policy._input_transform(
            {"state": old_state, "actions": targets.copy()}
        )["actions"]
        old_encoded[:, 14:] = np.linspace(3.0, 4.0, 18, dtype=np.float32)
        reanchored = _reanchor_normalized_actions(
            policy,
            {"state": new_state},
            old_encoded,
            targets,
            mask,
        )
        decoded = self._decode(reanchored, new_state, mask, mean, std)

        np.testing.assert_allclose(decoded, targets, atol=1e-6)
        np.testing.assert_array_equal(reanchored[:, 6], old_encoded[:, 6])
        np.testing.assert_array_equal(reanchored[:, 13], old_encoded[:, 13])
        np.testing.assert_array_equal(reanchored[:, 14:], old_encoded[:, 14:])

    def test_action_only_reanchor_skips_image_transform_after_parity_check(self):
        resize_calls = []

        class RepackTransform:
            def __call__(self, data):
                return data

        class PiperInputs:
            def __call__(self, data):
                return {"state": data["state"], "actions": data["actions"], "image": data["images"]}

        class DeltaActions:
            def __call__(self, data):
                data["actions"][:, 0] -= data["state"][0]
                return data

        class Normalize:
            def __call__(self, data):
                data["actions"] = data["actions"] / 2.0
                return data

        class ResizeImages:
            def __call__(self, data):
                resize_calls.append(True)
                return data

        class TokenizePrompt:
            def __call__(self, data):
                return data

        class PadStatesAndActions:
            def __call__(self, data):
                padded = np.zeros((len(data["actions"]), 4), dtype=np.float32)
                padded[:, :2] = data["actions"]
                data["actions"] = padded
                return data

        class Pipeline:
            def __init__(self):
                self.transforms = (
                    RepackTransform(), PiperInputs(), DeltaActions(), Normalize(),
                    ResizeImages(), TokenizePrompt(), PadStatesAndActions(),
                )

            def __call__(self, data):
                for transform in self.transforms:
                    data = transform(data)
                return data

        policy = type("Policy", (), {"_input_transform": Pipeline()})()
        targets = np.array([[0.6, 0.2], [0.7, 0.3]], dtype=np.float32)
        old_encoded = np.zeros((2, 4), dtype=np.float32)
        old_encoded[:, 0] = (targets[:, 0] - 0.5) / 2.0
        old_encoded[:, 1] = targets[:, 1] / 2.0
        observation = {
            "state": np.array([0.55, 0.1], dtype=np.float32),
            "images": {"cam_high": np.zeros((4, 4, 3), dtype=np.uint8)},
        }
        first = _reanchor_normalized_actions(
            policy, observation, old_encoded, targets, (True, False)
        )
        second = _reanchor_normalized_actions(
            policy, observation, old_encoded, targets, (True, False)
        )
        self.assertEqual(len(resize_calls), 1)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(second[:, 0], (targets[:, 0] - 0.55) / 2.0)


class _FakePyTorchModel:
    def denoise_step(self, *args, **kwargs):  # pragma: no cover - constructor contract only
        raise AssertionError("fake denoiser must not be called by session tests")

    def _preprocess_observation(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("fake preprocessor must not be called by session tests")

    def sample_actions(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("fake sampler must not be called by session tests")


class _FakePolicy:
    _is_pytorch_model = True

    def __init__(self):
        self._model = _FakePyTorchModel()
        self._sample_kwargs = {}
        self.seen_sample_kwargs = []
        self.next_normalized = None
        self.next_absolute = None
        self.input_transform = None
        self.adapter = None
        self.call_sampler = False

    def _input_transform(self, data):
        if self.input_transform is None:
            raise AssertionError("fake input transform was not configured")
        return self.input_transform(data)

    def infer(self, observation):
        self.seen_sample_kwargs.append(dict(self._sample_kwargs))
        if self.call_sampler:
            self.sampled_output = self._sample_actions("cpu", None, **self._sample_kwargs)
        else:
            self.adapter._last_normalized_actions = np.asarray(
                self.next_normalized, dtype=np.float32
            )
        if self.next_absolute is not None:
            actions = np.asarray(self.next_absolute, dtype=np.float32)
        else:
            actions = np.zeros_like(np.asarray(self.next_normalized, dtype=np.float32))
        return {"actions": actions}


class RTCAwarePolicyTest(unittest.TestCase):
    def _make(self, config=None):
        from bimanual_vla.deployment.rtc_policy import RTCAwarePolicy

        policy = _FakePolicy()
        adapter = RTCAwarePolicy(
            policy,
            config or RTCConfig(execution_horizon=3),
        )
        policy.adapter = adapter
        return adapter, policy

    @staticmethod
    def _observation(
        *, generation, previous_generation=None, offset=0, delay=0, state=None
    ):
        return {
            **({"state": np.asarray(state, dtype=np.float32)} if state is not None else {}),
            "client_metadata": {
                "rtc": {
                    "enabled": True,
                    "session_id": "test-session",
                    "inference_generation": generation,
                    "previous_chunk_generation": previous_generation,
                    "previous_chunk_offset_steps": offset,
                    "inference_delay_steps": delay,
                    "execution_horizon": 3,
                }
            }
        }

    def test_first_request_is_fail_safe_without_previous_chunk(self):
        adapter, policy = self._make()
        policy.next_normalized = np.arange(8, dtype=np.float32).reshape(4, 2)

        result = adapter.infer(self._observation(generation=1))

        self.assertNotIn("prev_chunk_left_over", policy.seen_sample_kwargs[0])
        self.assertFalse(result["rtc"]["enabled"])
        self.assertEqual(result["rtc"]["previous_chunk_left_over_steps"], 0)

    def test_model_preprocessing_receives_observation_without_client_telemetry(self):
        adapter, policy = self._make()
        policy.next_normalized = np.zeros((4, 2), dtype=np.float32)
        received = []
        original_infer = policy.infer

        def capture(observation):
            received.append(observation)
            return original_infer(observation)

        policy.infer = capture
        observation = self._observation(generation=1, state=[0.1, 0.2])
        observation["images"] = {"cam_high": np.zeros((2, 2, 3), dtype=np.uint8)}
        observation["prompt"] = "pick up"
        observation["client_metadata"]["large_telemetry"] = {"values": list(range(100))}
        adapter.infer(observation)

        self.assertNotIn("client_metadata", received[0])
        self.assertIs(received[0]["images"], observation["images"])
        self.assertEqual(received[0]["prompt"], "pick up")
        self.assertIn("client_metadata", observation)

    def test_input_transform_timer_preserves_reanchor_pipeline(self):
        class Pipeline:
            def __init__(self):
                self.transforms = (object(),)

            def __call__(self, data):
                return {**data, "state": data["state"] + 1}

        policy = _FakePolicy()
        pipeline = Pipeline()
        policy._input_transform = pipeline
        adapter = rtc_openpi.RTCAwarePolicy(policy, RTCConfig())

        self.assertIs(policy._input_transform.transforms, pipeline.transforms)
        result = policy._input_transform({"state": np.array([1], dtype=np.float32)})
        np.testing.assert_array_equal(result["state"], [2])
        self.assertIsNotNone(adapter._last_input_transform_ms)
        self.assertGreaterEqual(adapter._last_input_transform_ms, 0.0)

        policy.adapter = adapter
        policy.next_normalized = np.zeros((4, 2), dtype=np.float32)
        original_infer = policy.infer

        def infer_with_transform(observation):
            policy._input_transform({"state": np.array([1], dtype=np.float32)})
            return original_infer(observation)

        policy.infer = infer_with_transform
        measured = adapter.infer(self._observation(generation=1))
        self.assertGreaterEqual(
            measured["model_phase_timing"]["openpi_input_transform_ms"], 0.0
        )

    def test_result_generation_is_preserved_for_legacy_policy_wrappers(self):
        adapter, policy = self._make()
        policy.next_normalized = np.arange(8, dtype=np.float32).reshape(4, 2)

        original_infer = policy.infer

        def infer_with_result_generation(observation):
            result = original_infer(observation)
            result["inference_generation"] = 11
            return result

        policy.infer = infer_with_result_generation
        adapter.infer(self._observation(generation=None))

        policy.next_normalized = np.ones((4, 2), dtype=np.float32)
        result = adapter.infer(
            self._observation(
                generation=12,
                previous_generation=11,
                offset=1,
                delay=1,
            )
        )

        self.assertIn("prev_chunk_left_over", policy.seen_sample_kwargs[1])
        self.assertTrue(result["rtc"]["enabled"])

    def test_session_reanchors_joint_prefix_to_current_observation(self):
        mask = (True, False)
        config = RTCConfig(
            execution_horizon=3,
            physical_action_dim=2,
            reanchor_action_mask=mask,
        )
        adapter, policy = self._make(config)
        policy.input_transform = RTCReanchorTest._encoder(
            mask,
            mean=np.array([0.0, 0.0], dtype=np.float32),
            std=np.array([1.0, 1.0], dtype=np.float32),
            model_action_dim=2,
        )
        old_state = np.array([0.50, 0.10], dtype=np.float32)
        old_absolute = np.array(
            [[0.52, 0.2], [0.54, 0.3], [0.58, 0.4], [0.60, 0.5]],
            dtype=np.float32,
        )
        policy.next_absolute = old_absolute
        policy.next_normalized = policy.input_transform(
            {"state": old_state, "actions": old_absolute}
        )["actions"]
        adapter.infer(self._observation(generation=1, state=old_state))

        new_state = np.array([0.56, 0.90], dtype=np.float32)
        policy.next_absolute = np.zeros((4, 2), dtype=np.float32)
        policy.next_normalized = np.zeros((4, 2), dtype=np.float32)
        result = adapter.infer(
            self._observation(
                generation=2,
                previous_generation=1,
                offset=2,
                delay=1,
                state=new_state,
            )
        )

        prefix = policy.seen_sample_kwargs[1]["prev_chunk_left_over"]
        np.testing.assert_allclose(prefix[:2, 0], [0.02, 0.04], atol=1e-6)
        np.testing.assert_allclose(prefix[:2, 1], [0.4, 0.5], atol=1e-6)
        np.testing.assert_array_equal(prefix[2:], np.zeros((2, 2), dtype=np.float32))
        self.assertTrue(result["rtc"]["enabled"])
        self.assertTrue(result["rtc"]["reanchored"])
        self.assertAlmostEqual(result["rtc"]["origin_shift_l2"], 0.06, places=6)

    def test_reanchor_failure_disables_rtc_instead_of_reusing_old_origin(self):
        config = RTCConfig(
            execution_horizon=3,
            physical_action_dim=2,
            reanchor_action_mask=(True, False),
        )
        adapter, policy = self._make(config)
        policy.next_absolute = np.zeros((4, 2), dtype=np.float32)
        policy.next_normalized = np.zeros((4, 2), dtype=np.float32)
        adapter.infer(self._observation(generation=1, state=[0.0, 0.0]))
        policy.input_transform = lambda data: (_ for _ in ()).throw(RuntimeError("boom"))

        result = adapter.infer(
            self._observation(
                generation=2,
                previous_generation=1,
                offset=1,
                state=[0.1, 0.0],
            )
        )

        self.assertNotIn("prev_chunk_left_over", policy.seen_sample_kwargs[1])
        self.assertFalse(result["rtc"]["enabled"])
        self.assertFalse(result["rtc"]["reanchored"])
        self.assertIn("RuntimeError: boom", result["rtc"]["reanchor_error"])

    def test_matching_generation_uses_offset_and_latency_metadata(self):
        adapter, policy = self._make()
        first = np.arange(8, dtype=np.float32).reshape(4, 2)
        second = np.full((4, 2), 9.0, dtype=np.float32)
        policy.next_normalized = first
        adapter.infer(self._observation(generation=1))
        policy.next_normalized = second

        result = adapter.infer(
            self._observation(
                generation=2,
                previous_generation=1,
                offset=2,
                delay=3,
            )
        )

        kwargs = policy.seen_sample_kwargs[1]
        np.testing.assert_array_equal(kwargs["prev_chunk_left_over"][:2], first[2:])
        np.testing.assert_array_equal(
            kwargs["prev_chunk_left_over"][2:],
            np.zeros((2, 2), dtype=np.float32),
        )
        self.assertEqual(kwargs["previous_left_over_steps"], 2)
        self.assertEqual(kwargs["inference_delay"], 3)
        self.assertEqual(kwargs["execution_horizon"], 3)
        self.assertTrue(result["rtc"]["enabled"])
        self.assertEqual(result["rtc"]["previous_chunk_left_over_steps"], 2)

    def test_generation_mismatch_does_not_reuse_stale_chunk(self):
        adapter, policy = self._make()
        policy.next_normalized = np.zeros((4, 2), dtype=np.float32)
        adapter.infer(self._observation(generation=1))
        policy.next_normalized = np.ones((4, 2), dtype=np.float32)

        result = adapter.infer(
            self._observation(
                generation=2,
                previous_generation=99,
                offset=1,
                delay=2,
            )
        )

        self.assertNotIn("prev_chunk_left_over", policy.seen_sample_kwargs[1])
        self.assertFalse(result["rtc"]["enabled"])
        self.assertEqual(result["rtc"]["previous_chunk_left_over_steps"], 0)

    def test_pytorch_capture_wrapper_preserves_tensor_output(self):
        original_sampler = rtc_openpi._rtc_sample_actions_pytorch

        def fake_sampler(self, device, observation, noise=None, num_steps=5, **kwargs):
            return torch.ones((1, 4, 2), dtype=torch.float32)

        rtc_openpi._rtc_sample_actions_pytorch = fake_sampler
        try:
            policy = _FakePolicy()
            policy.call_sampler = True
            adapter = rtc_openpi.RTCAwarePolicy(policy, RTCConfig())
            policy.adapter = adapter
            result = adapter.infer(self._observation(generation=1))
            policy._sample_kwargs = {"num_steps": 7}
            overridden = adapter.infer(self._observation(generation=2))
            disabled_policy = _FakePolicy()
            disabled_policy.call_sampler = True
            disabled_adapter = rtc_openpi.RTCAwarePolicy(
                disabled_policy, RTCConfig(inference_diagnostics_enabled=False)
            )
            disabled_policy.adapter = disabled_adapter
            disabled_result = disabled_adapter.infer(self._observation(generation=1))
        finally:
            rtc_openpi._rtc_sample_actions_pytorch = original_sampler

        self.assertIsInstance(policy.sampled_output, torch.Tensor)
        np.testing.assert_array_equal(
            adapter._last_normalized_actions,
            np.ones((1, 4, 2), dtype=np.float32),
        )
        self.assertFalse(result["rtc"]["enabled"])
        self.assertEqual(result["rtc"]["denoising_steps"], 5)
        self.assertEqual(result["rtc"]["denoising_steps_source"], "sampler_default")
        self.assertGreaterEqual(result["rtc"]["sampler_wall_ms"], 0.0)
        phases = result["model_phase_timing"]
        self.assertGreaterEqual(phases["rtc_prepare_ms"], 0.0)
        self.assertGreaterEqual(phases["openpi_pre_sampler_ms"], 0.0)
        self.assertGreaterEqual(phases["openpi_post_sampler_ms"], 0.0)
        self.assertAlmostEqual(phases["sampler_wall_ms"], result["rtc"]["sampler_wall_ms"])
        self.assertAlmostEqual(
            phases["openpi_infer_ms"],
            phases["openpi_pre_sampler_ms"]
            + phases["sampler_wall_ms"]
            + phases["openpi_post_sampler_ms"],
            places=3,
        )
        self.assertAlmostEqual(
            phases["rtc_total_ms"],
            sum(
                value for key, value in phases.items()
                if key not in {"rtc_total_ms", "openpi_infer_ms"}
            ),
            places=3,
        )
        self.assertEqual(overridden["rtc"]["denoising_steps"], 7)
        self.assertEqual(overridden["rtc"]["denoising_steps_source"], "explicit")
        self.assertNotIn("denoising_steps", disabled_result["rtc"])
        self.assertNotIn("model_phase_timing", disabled_result)

    def test_jax_capture_reports_default_and_explicit_steps(self):
        def fake_sampler(self, rng, observation, *, num_steps=5, **kwargs):
            return np.ones((1, 4, 2), dtype=np.float32)

        openpi = types.ModuleType("openpi")
        shared = types.ModuleType("openpi.shared")
        nnx_utils = types.ModuleType("openpi.shared.nnx_utils")
        nnx_utils.module_jit = lambda sampler: sampler
        shared.nnx_utils = nnx_utils
        openpi.shared = shared
        modules = {
            "openpi": openpi,
            "openpi.shared": shared,
            "openpi.shared.nnx_utils": nnx_utils,
        }
        with patch.dict(sys.modules, modules), patch.object(
            rtc_openpi, "_rtc_sample_actions_jax", fake_sampler
        ):
            policy = _FakePolicy()
            policy._is_pytorch_model = False
            policy.call_sampler = True
            for name in ("embed_prefix", "embed_suffix", "PaliGemma", "action_out_proj"):
                setattr(policy._model, name, object())
            adapter = rtc_openpi.RTCAwarePolicy(
                policy, RTCConfig(temporal_consistency=False)
            )
            policy.adapter = adapter
            default = adapter.infer(self._observation(generation=1))
            policy._sample_kwargs = {"num_steps": 7}
            overridden = adapter.infer(self._observation(generation=2))

        self.assertEqual(default["rtc"]["denoising_steps"], 5)
        self.assertEqual(default["rtc"]["denoising_steps_source"], "sampler_default")
        self.assertEqual(overridden["rtc"]["denoising_steps"], 7)
        self.assertEqual(overridden["rtc"]["denoising_steps_source"], "explicit")




if __name__ == "__main__":
    unittest.main()
