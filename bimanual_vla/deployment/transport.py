"""Measured OpenPI WebSocket transport with optional compact image payloads."""

import time
import traceback

import numpy as np
import websockets

from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def _encode_jpeg_images(observation):
    """Encode CHW uint8 policy images while leaving the public observation intact."""
    transport = observation.get("_transport")
    if not isinstance(transport, dict) or transport.get("image_codec") != "jpeg_chw_v1":
        return observation
    quality = int(transport.get("quality", 90))
    if not 1 <= quality <= 100:
        raise ValueError("JPEG image quality must be in [1, 100]")
    import cv2

    wire = dict(observation)
    encoded = {}
    for key, value in (observation.get("images") or {}).items():
        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[0] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"JPEG image {key!r} must be uint8 CHW with 3 channels")
        ok, payload = cv2.imencode(
            ".jpg", np.transpose(frame, (1, 2, 0)),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for {key!r}")
        encoded[key] = payload.tobytes()
    wire["images"] = encoded
    return wire


def _decode_jpeg_images(observation):
    """Decode the compact client representation before policy preprocessing."""
    transport = observation.get("_transport")
    if not isinstance(transport, dict) or transport.get("image_codec") != "jpeg_chw_v1":
        return observation
    import cv2

    decoded = {}
    for key, payload in (observation.get("images") or {}).items():
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError(f"JPEG image {key!r} is not a byte payload")
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"JPEG decode failed for {key!r}")
        decoded[key] = np.transpose(frame, (2, 0, 1))
    observation = dict(observation)
    observation["images"] = decoded
    observation.pop("_transport", None)
    return observation


class TimedWebsocketClientPolicy(WebsocketClientPolicy):
    """Keep official connection/handshake behavior and measure each RPC phase.

    send/recv include socket buffering, scheduling and server work: they are
    deliberately not labelled pure network latency. Image compression is
    negotiated per observation and remains disabled for raw-compatible peers.
    """

    def infer(self, obs):
        started = time.monotonic()
        wire_observation = _encode_jpeg_images(obs)
        data = self._packer.pack(wire_observation)
        packed = time.monotonic()
        self._ws.send(data)
        sent = time.monotonic()
        response = self._ws.recv()
        received = time.monotonic()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        result = msgpack_numpy.unpackb(response)
        unpacked = time.monotonic()
        if not isinstance(result, dict):
            raise RuntimeError("OpenPI response must be a mapping")
        images = obs.get("images", {})
        result["client_transport_timing"] = {
            "pack_ms": (packed - started) * 1000,
            "send_ms": (sent - packed) * 1000,
            "wait_response_ms": (received - sent) * 1000,
            "unpack_ms": (unpacked - received) * 1000,
            "total_ms": (unpacked - started) * 1000,
            "request_bytes": len(data),
            "response_bytes": len(response),
            "request_image_shapes": {
                key: list(value.shape) for key, value in images.items()
                if hasattr(value, "shape")
            },
        }
        return result


async def serve_policy_requests(policy, metadata, websocket):
    """Official message exchange with server codec timing in the response.

    Response packing/sending can only be reported on the next response. Keep
    its generation attached rather than attributing it to the current request.
    Socket recv waiting includes client idle time, so it isn't a latency metric.
    """
    packer = msgpack_numpy.Packer()
    await websocket.send(packer.pack(metadata))
    previous_response_timing = None
    previous_total_ms = None
    while True:
        try:
            cycle_started = time.monotonic()
            request = await websocket.recv()
            received = time.monotonic()
            observation = msgpack_numpy.unpackb(request)
            unpacked = time.monotonic()
            observation = _decode_jpeg_images(observation)
            result = dict(policy.infer(observation))
            completed = time.monotonic()
            timing = dict(result.get("transport_timing") or {})
            timing.update(server_unpack_ms=(unpacked - received) * 1000,
                          server_policy_ms=(completed - unpacked) * 1000,
                          request_bytes=len(request),
                          previous_response_timing=previous_response_timing)
            result["transport_timing"] = timing
            result["server_timing"] = {"infer_ms": (completed - unpacked) * 1000}
            if previous_total_ms is not None:
                result["server_timing"]["prev_total_ms"] = previous_total_ms
            pack_started = time.monotonic()
            response = packer.pack(result)
            packed = time.monotonic()
            await websocket.send(response)
            sent = time.monotonic()
            previous_total_ms = (sent - cycle_started) * 1000
            client = observation.get("client_metadata") or {}
            previous_response_timing = {
                "inference_generation": client.get("inference_generation"),
                "server_pack_ms": (packed - pack_started) * 1000,
                "server_response_prepare_ms": (pack_started - completed) * 1000,
                "server_send_ms": (sent - packed) * 1000,
                "response_bytes": len(response),
            }
        except websockets.ConnectionClosed:
            return
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(code=1011, reason="Internal server error. Traceback included in previous frame.")
            raise
