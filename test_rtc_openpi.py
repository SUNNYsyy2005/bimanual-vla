from __future__ import annotations

import unittest

import numpy as np
import torch
import rtc_openpi

from rtc_openpi import (
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
            self.sampled_output = self._sample_actions("cpu", None)
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
        from rtc_openpi import RTCAwarePolicy

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

        def fake_sampler(self, *args, **kwargs):
            return torch.ones((1, 4, 2), dtype=torch.float32)

        rtc_openpi._rtc_sample_actions_pytorch = fake_sampler
        try:
            policy = _FakePolicy()
            policy.call_sampler = True
            adapter = rtc_openpi.RTCAwarePolicy(policy, RTCConfig())
            policy.adapter = adapter
            result = adapter.infer(self._observation(generation=1))
        finally:
            rtc_openpi._rtc_sample_actions_pytorch = original_sampler

        self.assertIsInstance(policy.sampled_output, torch.Tensor)
        np.testing.assert_array_equal(
            adapter._last_normalized_actions,
            np.ones((1, 4, 2), dtype=np.float32),
        )
        self.assertFalse(result["rtc"]["enabled"])




if __name__ == "__main__":
    unittest.main()
