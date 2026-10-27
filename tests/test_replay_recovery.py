"""Exercise the real replay recovery boundary with simulated feedback, no CAN."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from pathlib import Path
import runpy

import numpy as np
import pytest

from bimanual_vla.collection.replay import (
    PiperReplayHardware, ReplayBlocked, ReplayCancelled, ReplayConfig, ReplayFeedbackPending,
)


class Clock:
    def __init__(self):
        self.now = 100.
        self.stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.now += seconds
        return self.stopped


@contextmanager
def device(*, raw=0x22, temperature=35, persists=False, recurrence=False):
    clock = Clock()
    hardware = PiperReplayHardware(('left', 'right'), {}, ReplayConfig())
    hardware.check_ownership = Mock()
    motors, status = {}, {'arm_status': 0, 'err_code': 0}
    positions = {s: np.array([0., 1., -1., 0., 0., 0., .01]) for s in hardware.sides}
    cleared_at = {}
    for side in hardware.sides:
        motors[side] = [NS(can_id=i, foc_status_code=0, motor_temp=temperature,
                           foc_temp=35, foc_status=NS(driver_enable_status=False)) for i in range(1, 7)]
        motors[side][2].foc_status_code = raw if side == 'left' else 0
        arm = Mock(side=side)

        def low_speed(s=side):
            if recurrence and s in cleared_at and clock.now > cleared_at[s] + .2:
                motors[s][2].foc_status_code = raw
            for motor in motors[s]:
                for bit, name in enumerate(('voltage_too_low', 'motor_overheating', 'driver_overcurrent',
                        'driver_overheating', 'collision_status', 'driver_error_status', 'stall_status')):
                    setattr(motor.foc_status, name, bool(motor.foc_status_code & (1 << bit)))
            return NS(time_stamp=clock.now, Hz=50.,
                      **{f'motor_{i}': m for i, m in enumerate(motors[s], 1)})

        def clear(*, joint_num, s=side, **kwargs):
            if not persists:
                motors[s][joint_num-1].foc_status_code = 0
                cleared_at[s] = clock.now

        arm.GetArmLowSpdInfoMsgs.side_effect = low_speed
        arm.JointConfig.side_effect = clear
        for name in ('GetArmJointMsgs', 'GetArmGripperMsgs', 'GetArmEndPoseMsgs'):
            getattr(arm, name).side_effect = lambda: NS(time_stamp=clock.now)
        hardware.arms[side] = arm
    def arm_status(arm):
        return {'feedback_fresh': True, 'ctrl_mode': 0, 'mode_feed': 0,
                'feedback_timestamp': clock.now, **status}
    def feedback(arm, **kwargs):
        return np.zeros(10), positions[arm.side].copy()
    with patch('bimanual_vla.collection.replay_recovery.time.time', lambda: clock.now), \
         patch('bimanual_vla.collection.replay_recovery.time.monotonic', lambda: clock.now), \
         patch('bimanual_vla.deployment.client.arm_status_dict', side_effect=arm_status), \
         patch('bimanual_vla.deployment.client.read_output_state', side_effect=feedback):
        yield hardware, clock, motors, status, positions


def assert_no_motion(hardware):
    for arm in hardware.arms.values():
        for name in ('EnableArm', 'EnablePiper', 'JointCtrl', 'GripperCtrl', 'ModeCtrl',
                     'ResetPiper', 'EmergencyStop', 'ReqMasterArmMoveToHome'):
            getattr(arm, name).assert_not_called()


@pytest.mark.parametrize('raw', [0x02, 0x22])
def test_cool_disabled_thermal_fault_cleared_once_and_health_verified(raw):
    with device(raw=raw) as (hardware, clock, *_):
        events = []
        hardware.recover_startup(clock, events.append)
        hardware.arms['left'].JointConfig.assert_called_once_with(joint_num=3,
            set_zero=0, acc_param_is_effective=0, max_joint_acc=500, clear_err=0xAE)
        hardware.arms['right'].JointConfig.assert_not_called()
        assert events[-1]['state'] == 'recovery_completed'
        assert clock.now >= 102. - 1e-10
        assert hardware.recovery_attempted == {('left', 3)}
        assert_no_motion(hardware)


def test_healthy_enabled_arms_need_no_recovery():
    with device(raw=0) as (hardware, clock, motors, *_):
        for side in motors:
            for m in motors[side]:
                m.foc_status.driver_enable_status = True
        events = []
        hardware.recover_startup(clock, events.append)
        assert events[-1]['state'] == 'recovery_not_needed'
        assert not hardware.recovery_attempted
        assert_no_motion(hardware)


@pytest.mark.parametrize('raw', [0x01, 0x04, 0x08, 0x10, 0x20, 0x24, 0x40, 0x62])
def test_other_faults_are_never_cleared(raw):
    with device(raw=raw) as (hardware, clock, *_):
        with pytest.raises(ReplayBlocked, match='not eligible'):
            hardware.recover_startup(clock, lambda e: None)
        for arm in hardware.arms.values():
            arm.JointConfig.assert_not_called()
        assert_no_motion(hardware)


@pytest.mark.parametrize('field,value', [('arm_status', 1), ('arm_status', 2), ('err_code', 1)])
def test_arm_level_faults_and_emergency_stops_never_reset(field, value):
    with device() as (hardware, clock, _, status, _):
        status[field] = value
        with pytest.raises(ReplayBlocked, match='arm fault'):
            hardware.recover_startup(clock, lambda e: None)
        hardware.arms['left'].JointConfig.assert_not_called()
        assert_no_motion(hardware)


@pytest.mark.parametrize('temperature', [46, -1])
def test_hot_or_invalid_temperature_never_cleared(temperature):
    with device(temperature=temperature) as (hardware, clock, *_):
        with pytest.raises(ReplayBlocked, match='not eligible'):
            hardware.recover_startup(clock, lambda e: None)
        hardware.arms['left'].JointConfig.assert_not_called()


@pytest.mark.parametrize('side', ['left', 'right'])
def test_either_arm_enabled_blocks_clear(side):
    with device() as (hardware, clock, motors, *_):
        motors[side][0].foc_status.driver_enable_status = True
        with pytest.raises(ReplayBlocked, match='disabled'):
            hardware.recover_startup(clock, lambda e: None)
        hardware.arms['left'].JointConfig.assert_not_called()


def test_movement_before_clear_blocks_recovery():
    with device() as (hardware, clock, _, _, positions):
        def move(event):
            if event['state'] == 'recovery_waiting_stationary':
                positions['right'][0] += .01
        with pytest.raises(ReplayBlocked, match='moved'):
            hardware.recover_startup(clock, move)
        hardware.arms['left'].JointConfig.assert_not_called()


@pytest.mark.parametrize('stage', ['recovery_checking', 'recovery_waiting_stationary', 'recovering_driver_fault'])
def test_user_stop_prevents_clear_even_at_publication_boundary(stage):
    with device() as (hardware, clock, *_):
        def cancel(event):
            if event['state'] == stage:
                clock.stopped = True
        with pytest.raises(ReplayCancelled):
            hardware.recover_startup(clock, cancel)
        hardware.arms['left'].JointConfig.assert_not_called()
        assert_no_motion(hardware)


@pytest.mark.parametrize('persists,recurrence,message', [(True, False, 'persisted'), (False, True, 'recurred')])
def test_persistent_or_recurrent_fault_never_repeatedly_cleared(persists, recurrence, message):
    with device(persists=persists, recurrence=recurrence) as (hardware, clock, *_):
        with pytest.raises(ReplayBlocked, match=message):
            hardware.recover_startup(clock, lambda e: None)
        assert hardware.arms['left'].JointConfig.call_count == 1
        assert_no_motion(hardware)


def test_missing_motor_reply_blocks_recovery():
    with device() as (hardware, clock, motors, *_):
        motors['left'][5].can_id = 0
        with pytest.raises(ReplayBlocked, match='six-motor'):
            hardware.recover_startup(clock, lambda e: None)
        hardware.arms['left'].JointConfig.assert_not_called()


def test_stale_feedback_blocks_recovery():
    with device() as (hardware, clock, *_):
        arm = hardware.arms['left']
        old_message = arm.GetArmLowSpdInfoMsgs()
        old_message.time_stamp -= .3
        arm.GetArmLowSpdInfoMsgs.side_effect = lambda: old_message
        with pytest.raises(ReplayBlocked, match='fresh six-motor'):
            hardware.recover_startup(clock, lambda e: None)
        arm.JointConfig.assert_not_called()


def test_new_fault_before_publication_prevents_clear():
    with device() as (hardware, clock, motors, *_):
        def change(event):
            if event['state'] == 'recovering_driver_fault':
                motors['right'][0].foc_status_code = 0x10
        with pytest.raises(ReplayBlocked, match='not eligible'):
            hardware.recover_startup(clock, change)
        hardware.arms['left'].JointConfig.assert_not_called()


def test_auto_recovery_config_is_strict_boolean():
    assert replace(ReplayConfig(), auto_recover=False).auto_recover is False
    with pytest.raises(ValueError, match='auto_recover must be a boolean'):
        ReplayConfig(auto_recover='true')


def test_worker_runtime_pending_feedback_is_caught_across_module_identities():
    # -m runs replay under __main__; importing its exceptions through replay
    # used to create distinct types. Load a second module identity to reproduce.
    runtime = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               'bimanual_vla/collection/replay.py'))
    assert runtime['ReplayFeedbackPending'] is ReplayFeedbackPending
    hardware = runtime['PiperReplayHardware'](('right',), {}, ReplayConfig())
    hardware.arms = {'right': Mock()}
    hardware.check_ownership = Mock()
    hardware.read = Mock(side_effect=[runtime['ReplayFeedbackPending']('first packets pending'),
                                     {'right': {'joint': np.zeros(7)}}])
    clock, events = Clock(), []
    with patch('bimanual_vla.deployment.client.driver_enable_status_dict',
               return_value={'faulted': False}), \
         patch('bimanual_vla.collection.replay_recovery.time.monotonic', lambda: clock.now):
        hardware.recover_startup(clock, events.append)
    assert hardware.read.call_count == 2
    assert events[-1]['state'] == 'recovery_not_needed'
    assert clock.now >= 100.02
    assert_no_motion(hardware)


def test_pending_startup_feedback_only_fails_after_configured_deadline():
    hardware = PiperReplayHardware(('right',), {}, ReplayConfig(feedback_ready_timeout=.1))
    hardware.check_ownership = Mock()
    hardware.read = Mock(side_effect=ReplayFeedbackPending('missing motor replies'))
    clock = Clock()
    with patch('bimanual_vla.collection.replay_recovery.time.monotonic', lambda: clock.now):
        with pytest.raises(ReplayBlocked, match='feedback timeout.*missing motor replies'):
            hardware.recover_startup(clock, lambda event: None)
    assert .1 <= clock.now - 100. < .13
    assert hardware.read.call_count > 1


def test_worker_runtime_treats_recovery_cancellation_as_stop():
    runtime = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               'bimanual_vla/collection/replay.py'))
    hardware = Mock()
    hardware.recover_startup.side_effect = ReplayCancelled()
    hardware.stop_holding.return_value = []
    episode = NS(formats=('joint',), sides=('right',))
    events = []
    result = runtime['run_replay'](episode, ReplayConfig(), {}, exclusive_control=True,
        hardware_factory=lambda *a: hardware, progress=events.append)
    assert result == 'stopped'
    assert events[-1]['state'] == 'stopped'
    hardware.close.assert_called_once()
