import os
import signal
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from bimanual_vla.deployment.process_control import close_owned_process_group


class ProcessControlTest(unittest.TestCase):
    def test_client_overrides_ignored_stop_and_preserves_cleanup_on_repeat(self):
        code = '''
import signal, time
from bimanual_vla.deployment.process_control import graceful_stop_signals
signal.signal(signal.SIGINT, signal.SIG_IGN)
with graceful_stop_signals():
    try:
        print("ready", flush=True)
        time.sleep(20)
    except KeyboardInterrupt:
        print("braking", flush=True)
        time.sleep(.2)
        print("cleaned", flush=True)
'''
        process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), 'ready')
            process.send_signal(signal.SIGINT)
            self.assertEqual(process.stdout.readline().strip(), 'braking')
            process.send_signal(signal.SIGTERM)
            output, error = process.communicate(timeout=3.)
            self.assertEqual(process.returncode, 0, error)
            self.assertEqual(output.strip(), 'cleaned')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2.)

    def test_shutdown_escalates_and_reaps_owned_group(self):
        process = Mock(pid=987654)
        process.wait.side_effect = [subprocess.TimeoutExpired('client', 10),
                                   subprocess.TimeoutExpired('client', 3), 0]
        with patch('bimanual_vla.deployment.process_control.os.killpg') as killpg:
            self.assertTrue(close_owned_process_group(process))
        self.assertEqual([x.args for x in killpg.call_args_list], [
            (process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL),
            (process.pid, signal.SIGKILL)])
        self.assertEqual(process.wait.call_count, 3)

    def test_parent_exited_still_cleans_up_orphan_workers(self):
        process = Mock(pid=987654)
        with patch('bimanual_vla.deployment.process_control.os.killpg') as killpg:
            self.assertFalse(close_owned_process_group(process))
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)

    def test_never_kills_own_group(self):
        with self.assertRaises(RuntimeError):
            close_owned_process_group(Mock(pid=os.getpgrp()))
