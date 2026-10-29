"""GUI process lifecycle regressions; no Tk display or hardware required."""
import io
import queue
import signal
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bimanual_vla.collection.gui import CollectorGUI


class InferenceProcessTest(unittest.TestCase):
    def gui(self):
        return SimpleNamespace(
            inference_log_widget=Mock(), _append_inference_log=Mock(),
            inference_stop_requested=False, inference_process=None,
            inference_status_var=Mock(), inference_pid_var=Mock(),
            inference_stop_button=Mock(), messages=queue.Queue(),
            _set_connection_config_enabled=Mock(), _update_mode_controls=Mock(),
        )

    def test_launch_starts_log_reader_and_updates_controls_for_both_entry_points(self):
        for checked in (False, True):
            with self.subTest(device_checked=checked):
                gui = self.gui()
                process = Mock(pid=123, stdout=io.StringIO("client ready\nwarning\n"))
                process.wait.return_value = 0
                command = ["python", "-m", "client"]
                kwargs = {"device_check": object()} if checked else {}
                with patch("bimanual_vla.collection.gui.subprocess.Popen", return_value=process):
                    CollectorGUI._launch_inference_process(gui, command, "localhost:8098", **kwargs)
                    gui.inference_process_thread.join(timeout=2.)
                self.assertFalse(gui.inference_process_thread.is_alive())
                self.assertIs(gui.inference_process, process)
                self.assertEqual(gui.messages.get_nowait(), ("inference_log", "client ready"))
                self.assertEqual(gui.messages.get_nowait(), ("inference_log", "warning"))
                self.assertEqual(gui.messages.get_nowait(), ("inference_done", 0, False))
                gui._set_connection_config_enabled.assert_called_once_with(False)
                gui._update_mode_controls.assert_called_once()
                self.assertEqual("RLSOK" in gui.inference_status_var.set.call_args.args[0], checked)
                gui.inference_pid_var.set.assert_called_once_with("PID 123")
                gui._append_inference_log.assert_called_once_with("$ python -m client")

    def test_stop_signals_actuator_owner_once_without_interrupting_workers(self):
        gui = self.gui()
        gui.inference_process = Mock(pid=123)
        gui.inference_process.poll.return_value = None
        with patch("bimanual_vla.collection.gui.os.killpg") as killpg:
            CollectorGUI.stop_inference(gui)
            CollectorGUI.stop_inference(gui)
        killpg.assert_not_called()
        gui.inference_process.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertTrue(gui.inference_stop_requested)
        gui.inference_stop_button.configure.assert_called_once_with(state="disabled")

    def test_failed_stop_signal_leaves_retry_available(self):
        gui = self.gui()
        gui.inference_process = Mock(pid=123)
        gui.inference_process.poll.return_value = None
        gui.inference_process.send_signal.side_effect = OSError("denied")
        CollectorGUI.stop_inference(gui)
        self.assertFalse(gui.inference_stop_requested)
        gui.inference_stop_button.configure.assert_called_with(state="normal")
        self.assertIn("failed", gui.inference_status_var.set.call_args.args[0])

    def test_launch_failure_does_not_mark_gui_running(self):
        gui = self.gui()
        with patch("bimanual_vla.collection.gui.subprocess.Popen", side_effect=OSError("failed")), \
             patch("bimanual_vla.collection.gui.messagebox.showerror") as error:
            CollectorGUI._launch_inference_process(gui, ["missing"], "localhost:8098")
        self.assertIsNone(gui.inference_process)
        error.assert_called_once_with("Cannot start inference", "failed")
        gui._set_connection_config_enabled.assert_not_called()
        gui._update_mode_controls.assert_not_called()

    def test_failed_stop_never_reports_idle_or_schedules_restart_or_return(self):
        gui = self.gui()
        gui.inference_restart_requested = True
        gui.inference_return_after_stop = True
        gui.inference_return_only = False
        gui.inference_manual_trigger_path = None
        gui.root = Mock()
        gui.return_inference_to_initial = Mock()
        gui.start_inference = Mock()
        CollectorGUI._finish_inference(gui, 2, True)
        gui.inference_status_var.set.assert_called_once_with('Inference exited with code 2')
        gui.root.after.assert_not_called()
