from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import rtc_client
from robot_observation_bridge import run as legacy_run
from robot_observation_bridge import run_rtc_client


class RTCClientEntrypointTest(unittest.TestCase):
    def test_canonical_entrypoint_uses_the_real_control_loop(self):
        self.assertIs(rtc_client.run, run_rtc_client)
        self.assertIs(legacy_run, run_rtc_client)

    @patch.object(sys, "argv", ["rtc_client.py", "--instruction", "test"])
    @patch("rtc_client.run_rtc_client")
    def test_main_runs_the_canonical_control_loop(self, run_client):
        rtc_client.main()
        run_client.assert_called_once()
        args = run_client.call_args.args[0]
        self.assertEqual(args.instruction, "test")


if __name__ == "__main__":
    unittest.main()
