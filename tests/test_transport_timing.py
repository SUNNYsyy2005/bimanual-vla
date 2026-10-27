import asyncio
import unittest
from unittest.mock import patch

import numpy as np
from openpi_client import msgpack_numpy
from websockets.exceptions import ConnectionClosedOK

from bimanual_vla.deployment.transport import (
    TimedWebsocketClientPolicy,
    serve_policy_requests,
    _decode_jpeg_images,
    _encode_jpeg_images,
)


class TransportTimingTest(unittest.TestCase):
    def test_jpeg_chw_transport_round_trips_and_reduces_payload(self):
        image = np.random.default_rng(4).integers(
            0, 256, (3, 224, 224), dtype=np.uint8
        )
        observation = {
            "images": {"cam_high": image},
            "_transport": {"image_codec": "jpeg_chw_v1", "quality": 90},
        }
        wire = _encode_jpeg_images(observation)
        self.assertIsInstance(wire["images"]["cam_high"], bytes)
        packed_raw = msgpack_numpy.packb({"images": observation["images"]})
        packed_jpeg = msgpack_numpy.packb(wire)
        self.assertLess(len(packed_jpeg), len(packed_raw))
        decoded = _decode_jpeg_images(msgpack_numpy.unpackb(packed_jpeg))
        self.assertEqual(decoded["images"]["cam_high"].shape, image.shape)
        self.assertEqual(decoded["images"]["cam_high"].dtype, np.uint8)

    def test_wire_arrays_and_client_phase_boundaries(self):
        actions = np.arange(350, dtype=np.float32).reshape(50, 7)
        response = msgpack_numpy.packb({"actions": actions})
        sent = []
        class Socket:
            send = staticmethod(sent.append)
            recv = staticmethod(lambda: response)
        client = TimedWebsocketClientPolicy.__new__(TimedWebsocketClientPolicy)
        client._packer, client._ws = msgpack_numpy.Packer(), Socket()
        observation = {"images": {"cam_high": np.ones((224, 224, 3), dtype=np.uint8)}}
        with patch("bimanual_vla.deployment.transport.time.monotonic", side_effect=[0, .002, .005, .125, .126]):
            result = client.infer(observation)
        np.testing.assert_array_equal(result["actions"], actions)
        np.testing.assert_array_equal(msgpack_numpy.unpackb(sent[0])["images"]["cam_high"], observation["images"]["cam_high"])
        timing = result["client_transport_timing"]
        for key, value in {"pack_ms": 2, "send_ms": 3, "wait_response_ms": 120, "unpack_ms": 1, "total_ms": 126}.items():
            self.assertAlmostEqual(timing[key], value)
        self.assertEqual(timing["request_bytes"], len(sent[0]))
        self.assertEqual(timing["request_image_shapes"], {"cam_high": [224, 224, 3]})

    def test_server_codec_metrics_preserve_generation_and_do_not_count_idle_recv(self):
        responses = []
        requests = [msgpack_numpy.packb({"client_metadata": {"inference_generation": g}}) for g in (4, 5)]
        class Socket:
            async def recv(self):
                if requests:
                    return requests.pop(0)
                raise ConnectionClosedOK(None, None)
            async def send(self, data):
                responses.append(msgpack_numpy.unpackb(data))
        class Policy:
            def infer(self, obs):
                return {"actions": np.zeros((50, 7)), "transport_timing": {"model_inference_ms": 109}}
        asyncio.run(serve_policy_requests(Policy(), {"action_hz": 20}, Socket()))
        self.assertEqual(responses[0], {"action_hz": 20})
        first, second = [r["transport_timing"] for r in responses[1:]]
        self.assertIsNone(first["previous_response_timing"])
        self.assertEqual(second["previous_response_timing"]["inference_generation"], 4)
        self.assertEqual(first["model_inference_ms"], 109)
        self.assertGreaterEqual(first["server_unpack_ms"], 0)
        self.assertGreaterEqual(second["previous_response_timing"]["server_pack_ms"], 0)

    def test_server_error_retains_text_error_protocol(self):
        sent = []
        closed = []
        class Socket:
            async def recv(self):
                return msgpack_numpy.packb({})
            async def send(self, data):
                sent.append(data)
            async def close(self, **kwargs):
                closed.append(kwargs)
        class Policy:
            def infer(self, obs):
                raise ValueError("synthetic server error")
        with self.assertRaises(ValueError):
            asyncio.run(serve_policy_requests(Policy(), {}, Socket()))
        self.assertIsInstance(sent[-1], str)
        self.assertIn("synthetic server error", sent[-1])
        self.assertEqual(closed[0]["code"], 1011)
