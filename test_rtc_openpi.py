from __future__ import annotations

import unittest

import numpy as np
import torch
import rtc_openpi

from rtc_openpi import RTCConfig, RTCProcessor, _prefix_weights


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
        self.adapter = None
        self.call_sampler = False

    def infer(self, observation):
        self.seen_sample_kwargs.append(dict(self._sample_kwargs))
        if self.call_sampler:
            self.sampled_output = self._sample_actions("cpu", None)
        else:
            self.adapter._last_normalized_actions = np.asarray(
                self.next_normalized, dtype=np.float32
            )
        return {"actions": np.zeros((4, 2), dtype=np.float32)}


class RTCAwarePolicyTest(unittest.TestCase):
    def _make(self):
        from rtc_openpi import RTCAwarePolicy

        policy = _FakePolicy()
        adapter = RTCAwarePolicy(policy, RTCConfig(execution_horizon=3))
        policy.adapter = adapter
        return adapter, policy

    @staticmethod
    def _observation(*, generation, previous_generation=None, offset=0, delay=0):
        return {
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
        np.testing.assert_array_equal(kwargs["prev_chunk_left_over"], first[2:])
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
