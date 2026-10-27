"""Inference/replay configuration parity without Tk windows or hardware."""
import inspect
import tkinter as tk
from unittest.mock import Mock

import pytest

from bimanual_vla.collection.gui import CollectorGUI, build_inference_bridge_command
from bimanual_vla.collection.replay import ReplayConfig
from bimanual_vla.collection.replay_gui import ReplayDialog
from bimanual_vla.deployment import client
from bimanual_vla.deployment.execution_defaults import DEFAULT_SPEED_PCT
from bimanual_vla.deployment.pose_trajectory import PoseTrajectoryConfig
from bimanual_vla.deployment.servo import ServoConfig


def panel(*, shaping=True, backend='ruckig'):
    gui = CollectorGUI.__new__(CollectorGUI)
    values = {
        'inference_trajectory_shaping_var': shaping,
        'inference_servo_backend_var': backend,
        'inference_control_hz_var': '30',
        'inference_servo_hz_var': '90',
        'inference_joint_max_speed_var': '.42',
        'inference_joint_max_acceleration_var': '1.8',
        'inference_joint_max_jerk_var': '11',
        'inference_pose_trajectory_var': True,
    }
    for name, value in values.items():
        setattr(gui, name, Mock(get=Mock(return_value=value)))
    gui.inference_pose_settings = {k: Mock(get=Mock(return_value=str(v)))
                                   for k, v in vars(PoseTrajectoryConfig()).items()}
    gui.inference_pose_settings['max_linear_speed_m_s'].get.return_value = '.12'
    gui.inference_pose_settings['max_position_tracking_error_m'].get.return_value = '.04'
    return gui


def test_shared_execution_defaults_match_inference_and_servo():
    replay, servo = ReplayConfig(), ServoConfig()
    assert replay.control_hz == client.DEFAULT_ACTION_HZ
    assert replay.hz == servo.hz
    assert replay.max_joint_speed == servo.max_velocity == client.DEFAULT_TRAJECTORY_MAX_SPEED_RAD_S
    assert replay.max_joint_acceleration == servo.max_acceleration == client.DEFAULT_TRAJECTORY_MAX_ACCELERATION_RAD_S2
    assert replay.max_joint_jerk == servo.max_jerk == client.DEFAULT_TRAJECTORY_MAX_JERK_RAD_S3
    assert replay.max_tracking_error == servo.max_tracking_error == client.DEFAULT_TRAJECTORY_MAX_TRACKING_ERROR_RAD
    assert replay.max_gripper_speed == servo.max_gripper_velocity
    assert replay.feedback_max_age == client.DEFAULT_FEEDBACK_MAX_AGE_S
    assert replay.speed_pct == DEFAULT_SPEED_PCT
    fields = inspect.signature(build_inference_bridge_command).parameters
    assert fields['control_hz'].default == replay.control_hz
    assert fields['servo_hz'].default == replay.hz
    assert fields['joint_max_speed_rad_s'].default == replay.max_joint_speed
    assert fields['joint_max_acceleration_rad_s2'].default == replay.max_joint_acceleration
    assert fields['joint_max_jerk_rad_s3'].default == replay.max_joint_jerk


def test_current_panel_settings_are_snapshotted_and_can_be_refreshed():
    gui = panel()
    config = gui._replay_inference_defaults()
    assert (config.control_hz, config.hz) == (30, 90)
    assert config.speed_pct == 100
    assert (config.max_joint_speed, config.max_joint_acceleration, config.max_joint_jerk) == (.42, 1.8, 11)
    assert config.joint_smoothing and config.eef_smoothing
    assert config.pose.max_linear_speed_m_s == .12
    assert config.pose.max_position_tracking_error_m == .04
    gui.inference_joint_max_speed_var.get.return_value = '.5'
    assert config.max_joint_speed == .42
    assert gui._replay_inference_defaults().max_joint_speed == .5


def test_thirty_hz_chunk_steps_and_hundred_hz_smoothed_publication_remain_distinct():
    gui = panel()
    gui.inference_servo_hz_var.get.return_value = '100'
    config = gui._replay_inference_defaults()
    assert config.control_hz == 30
    assert config.hz == 100
    assert config.joint_smoothing


def test_enabling_joint_smoothing_restores_inference_servo_rate():
    root = tk.Tcl()
    dialog = ReplayDialog.__new__(ReplayDialog)
    dialog.worker = None
    dialog.smoothed_hz = 100.
    dialog.joint_smoothing = tk.BooleanVar(root, value=True)
    dialog.settings = {'control_hz': tk.StringVar(root, value='30'),
                       'hz': tk.StringVar(root, value='30')}
    dialog.update_command_rate()
    assert float(dialog.settings['hz'].get()) == 100
    dialog.joint_smoothing.set(False)
    dialog.update_command_rate()
    assert float(dialog.settings['hz'].get()) == 30


@pytest.mark.parametrize('shaping,backend', [(False, 'ruckig'), (True, 'off')])
def test_disabled_shaping_uses_inference_control_rate_and_ignores_inactive_fields(shaping, backend):
    gui = panel(shaping=shaping, backend=backend)
    gui.inference_servo_hz_var.get.return_value = 'inactive'
    gui.inference_joint_max_jerk_var.get.return_value = 'inactive'
    config = gui._replay_inference_defaults()
    assert config.hz == config.control_hz == 30
    assert not config.joint_smoothing and not config.eef_smoothing


def test_unsupported_legacy_backend_is_not_silently_treated_as_ruckig():
    with pytest.raises(ValueError, match='legacy'):
        panel(backend='legacy')._replay_inference_defaults()


@pytest.mark.parametrize('updates', [{'control_hz': 101}, {'control_hz': 0},
    {'control_hz': float('nan')}, {'hz': 20, 'control_hz': 30}])
def test_reference_and_command_rates_must_be_consistent(updates):
    with pytest.raises(ValueError):
        ReplayConfig(**updates)
