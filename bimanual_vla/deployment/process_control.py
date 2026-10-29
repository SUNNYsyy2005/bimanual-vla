"""Explicit stop signals and bounded cleanup for owned client processes."""
from contextlib import contextmanager
import os
import signal
import subprocess


@contextmanager
def graceful_stop_signals():
    """Override inherited SIG_IGN; repeated requests must not interrupt cleanup."""
    received = False

    def stop(signum, frame):
        nonlocal received
        if not received:
            received = True
            raise KeyboardInterrupt

    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for s in previous:
            signal.signal(s, stop)
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def close_owned_process_group(process, *, grace_s=10., terminate_s=3., kill_s=2.):
    """Reap a Popen(start_new_session=True) group, including orphan workers.

    Caller sends the initial graceful stop. Return True if forced termination
    of the main client was necessary. This is not a robot safety confirmation.
    """
    group = process.pid  # Session/group leader assigned by start_new_session.
    if group == os.getpgrp():
        raise RuntimeError("refusing to terminate the GUI's own process group")

    def send(sig):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass

    forced = False
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        forced = True
        send(signal.SIGTERM)
        try:
            process.wait(timeout=terminate_s)
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL)
            process.wait(timeout=kill_s)
    # The leader may already have exited while compute workers remain alive.
    send(signal.SIGKILL)
    return forced
