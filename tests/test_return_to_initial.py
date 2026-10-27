import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from bimanual_vla.deployment import client
from bimanual_vla.collection.gui import CollectorGUI


class ReturnToInitialTest(unittest.TestCase):
    def test_startup_pose_is_clipped_to_effective_driver_limits(self):
        robot = SimpleNamespace()
        robot._vla_joint_limits = np.deg2rad(
            [[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-180, 180]]
        )
        pose = np.array([0.0, -0.034, -0.2, 0.0, 2.0, 0.0, 0.01])
        clipped, changes = client.clip_startup_pose_to_limits(
            {"right": robot}, pose, arm_mode="single", arm_side="right"
        )
        self.assertEqual(clipped[1], 0.0)
        self.assertAlmostEqual(clipped[4], client.JOINT_LIMITS_RAD[4, 1], places=6)
        self.assertIn("right", changes)
        np.testing.assert_allclose(clipped[[0, 2, 3, 5, 6]], pose[[0, 2, 3, 5, 6]])

    def simulate(self, speed, timeout):
        clock = [0.0]
        current = np.array([0., .8, -.5, 0., 0., 0., 0.])
        target = current.copy()
        home = current.copy()
        home[1] = 0.
        lead = []

        def sleep(dt):
            current[:6] += np.clip(target[:6] - current[:6], -speed * dt, speed * dt)
            clock[0] += dt

        def publish(*raw):
            target[:6] = np.asarray(raw) / client.RAD_FACTOR
            lead.append(float(np.max(np.abs(target[:6] - current[:6]))))

        robot = SimpleNamespace(ModeCtrl=Mock(), JointCtrl=publish, GripperCtrl=Mock())
        with patch.object(client.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(client.time, 'sleep', side_effect=sleep), \
             patch.object(client, 'read_output_qpos', side_effect=lambda *a, **k: current.copy()), \
             patch.object(client, 'arm_status_dict', return_value={'arm_status': 0, 'err_code': 0}):
            try:
                result = client.return_pipers_to_initial(
                    {'right': robot}, home, arm_mode='single', speed_pct=10, hz=20,
                    timeout_s=timeout, max_feedback_age_s=.5, servo_backend='ruckig',
                    max_joint_speed_rad_s=.6, max_joint_acceleration_rad_s2=3,
                    max_joint_jerk_rad_s3=60, max_tracking_error_rad=.35)
                return result, current, max(lead)
            except TimeoutError:
                self.assertLess(max(lead), .11)
                raise

    def test_slow_firmware_arrives_without_accumulating_tracking_error(self):
        result, current, lead = self.simulate(.08, 25)
        self.assertTrue(result['returned'])
        self.assertLess(abs(current[1]), .03)
        self.assertLess(lead, .12)

    def test_stalled_arm_times_out_without_advancing_entire_home_path(self):
        with self.assertRaises(TimeoutError):
            self.simulate(0., 2.)


class ReturnButtonTest(unittest.TestCase):
    def gui(self, running):
        return SimpleNamespace(
            inference_stop_requested=False, inference_return_only=False,
            recording=False, piper=None, _can_activation_running=lambda: False,
            _replay_running=lambda: False,
            _inference_running=lambda: running, inference_restart_requested=True,
            stop_inference=Mock(), inference_status_var=Mock(),
            inference_session_command=['python', '-m', 'client', '--allow-execution',
                                       '--initial-pose-file', '/tmp/session.json'],
            inference_return_pose_path='/tmp/session.json', _launch_inference_process=Mock())

    def test_running_button_stops_current_owner_and_does_not_launch_second_writer(self):
        gui = self.gui(True)
        CollectorGUI.return_inference_to_initial(gui)
        gui.stop_inference.assert_called_once()
        gui._launch_inference_process.assert_not_called()

        self.assertFalse(gui.inference_restart_requested)

    def test_idle_button_reuses_session_command_in_return_only_mode(self):
        gui = self.gui(False)
        CollectorGUI.return_inference_to_initial(gui)
        command = gui._launch_inference_process.call_args.args[0]
        self.assertEqual(command[-1], '--return-only')
        self.assertIn('/tmp/session.json', command)
        self.assertTrue(gui.inference_return_only)
        CollectorGUI.return_inference_to_initial(gui)
        gui._launch_inference_process.assert_called_once()

    def test_manual_return_authorizes_only_return_for_preview_session(self):
        gui = self.gui(False)
        gui.inference_session_command.remove('--allow-execution')
        CollectorGUI.return_inference_to_initial(gui)
        command = gui._launch_inference_process.call_args.args[0]
        self.assertIn('--allow-execution', command)
        self.assertIn('--return-only', command)
        self.assertNotIn('--allow-execution', gui.inference_session_command)

    def test_running_preview_defers_manual_return_until_bridge_exits(self):
        gui = self.gui(True)
        gui.inference_session_command.remove('--allow-execution')
        CollectorGUI.return_inference_to_initial(gui)
        self.assertTrue(gui.inference_return_after_stop)
        gui.stop_inference.assert_called_once()
        gui._launch_inference_process.assert_not_called()


class ManualReturnEnableTest(unittest.TestCase):
    def test_enable_failure_aborts_return_and_disconnects_both_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pose.json'
            path.write_text(json.dumps(dict(arm_mode='bimanual', left_can='can0',
                                           right_can='can1', qpos=[0.] * 14, safe_to_return=False)))
            args = SimpleNamespace(allow_execution=True, initial_pose_file=str(path),
                                   arm_mode='bimanual', left_can='can0', right_can='can1',
                                   max_feedback_age_s=.5)
            left, right = Mock(), Mock()
            controller = Mock()
            controller._enable_robot.side_effect = [None, client.ExecutionBlocked('enable timed out')]
            with patch.object(client, 'connect_piper', side_effect=[left, right]), \
                 patch.object(client, 'ExecutionController', return_value=controller), \
                 patch.object(client, 'read_output_qpos', return_value=np.zeros(7)), \
                 patch.object(client, 'recover_return_faults'), \
                 patch.object(client, 'return_pipers_to_initial') as move:
                with self.assertRaisesRegex(client.ExecutionBlocked, 'enable timed out'):
                    client.run_return_only(args)
                self.assertEqual(controller._enable_robot.call_count, 2)
                move.assert_not_called()
                left.DisconnectPort.assert_called_once()
                right.DisconnectPort.assert_called_once()


if __name__ == '__main__':
    unittest.main()
