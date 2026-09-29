from __future__ import annotations

import sys
import threading
import unittest
from unittest.mock import patch
import logging

from bimanual_vla.deployment import client as rtc_client
from bimanual_vla.deployment.client import estimate_event_rate_hz, estimate_single_inflight_ceiling_hz
from bimanual_vla.deployment.legacy_bridge import run as legacy_run
from bimanual_vla.deployment.legacy_bridge import run_rtc_client


class RTCClientEntrypointTest(unittest.TestCase):
    def test_console_logging_drops_when_sink_is_backed_up(self):
        handler = rtc_client.NonBlockingConsoleHandler(queue_size=1)
        entered = threading.Event()
        release = threading.Event()
        def slow_handle(_record):
            entered.set()
            release.wait(timeout=1.0)
        handler.sink.handle = slow_handle
        try:
            for index in range(3):
                if index == 1:
                    self.assertTrue(entered.wait(timeout=1.0))
                handler.emit(logging.LogRecord(
                    name="test", level=logging.INFO, pathname="test", lineno=1,
                    msg="message %d", args=(index,), exc_info=None,
                ))
            self.assertEqual(handler.dropped_count, 1)
        finally:
            release.set()
            handler.close()

    def test_canonical_entrypoint_uses_the_real_control_loop(self):
        self.assertIs(rtc_client.run, run_rtc_client)
        self.assertIs(legacy_run, run_rtc_client)

    @patch.object(sys, "argv", ["bimanual_vla.deployment.client.py", "--instruction", "test"])
    @patch("bimanual_vla.deployment.client.run_rtc_client")
    def test_main_runs_the_canonical_control_loop(self, run_client):
        rtc_client.main()
        run_client.assert_called_once()
        args = run_client.call_args.args[0]
        self.assertEqual(args.instruction, "test")

    def test_estimate_event_rate_uses_observed_timestamps(self):
        self.assertAlmostEqual(
            estimate_event_rate_hz([0.0, 0.25, 0.5, 0.75]),
            4.0,
        )
        self.assertAlmostEqual(
            estimate_event_rate_hz([0.0, 0.55, 1.10]),
            1.8181818,
            places=5,
        )
        self.assertIsNone(estimate_event_rate_hz([1.0]))
        self.assertAlmostEqual(estimate_single_inflight_ceiling_hz(0.55), 1.8181818, places=5)
        self.assertIsNone(estimate_single_inflight_ceiling_hz(0.0))


if __name__ == "__main__":
    unittest.main()
