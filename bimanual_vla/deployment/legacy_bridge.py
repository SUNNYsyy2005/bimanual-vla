#!/usr/bin/env python3
"""Compatibility launcher for the canonical robot-side RTC client.

New integrations should import or execute :mod:`rtc_client` directly. This
module intentionally contains no control logic so the physical robot has one
implementation of inference, safety checks, and command scheduling.
"""

from bimanual_vla.deployment.client import main, run_rtc_client


# Preserve the public entry-point used by older launch scripts.
run = run_rtc_client


if __name__ == "__main__":
    main()
