"""Hardware replay boundaries; tests never create a real CAN socket."""
from dataclasses import replace
from contextlib import contextmanager
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import socket
import struct
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from bimanual_vla.collection.replay import (
    PiperReplayHardware, RemoteCommandMonitor, ReplayBlocked, ReplayCancelled, ReplayConfig,
    ReplayFeedbackPending, ReplayPlanner, load_replay_episode, main, run_replay,
    validate_start, wait_for_replay_feedback, prepare_replay_start, validate_eef_position,
)
from bimanual_vla.data.contract import EpisodeContract, matrix_to_rotation6d
from bimanual_vla.deployment.client import JOINT_LIMITS_RAD


def episode_file(tmp_path, *, schema='delivery', mode='single', side='right', both=True):
    count = 2 if mode == 'bimanual' else 1
    joints = np.tile([.05, 1., -1., 0., 0., 0., .2], (4, count))
    for arm in range(count):
        joints[:, arm * 7] += np.arange(4) * .001
    eef = np.zeros((4, 10 * count))
    for arm in range(count):
        eef[:, arm * 10:arm * 10 + 3] = np.c_[.2 + joints[:, arm * 7], np.full(4, .2), np.full(4, .3)]
        eef[:, arm * 10 + 3:arm * 10 + 9] = matrix_to_rotation6d(np.eye(3))
        eef[:, arm * 10 + 9] = .2
    padded = lambda a: np.concatenate((a, a[-1:]))
    timestamps = 100. + np.arange(5) * .1
    metadata = EpisodeContract(schema=schema, arm_mode=mode, arm_side=side).metadata_payload()
    data = {**metadata, 'state': padded(joints if schema == 'joint' else eef),
            'actions': padded(joints if schema == 'joint' else eef),
            'state_timestamp': timestamps, 'timestamps': timestamps,
            'joint_qpos': padded(joints)}
    if both:
        data.update(joint_state=padded(joints), joint_state_timestamp=timestamps,
                    eef_state=padded(eef), eef_state_timestamp=timestamps)
    path = tmp_path / 'ep_0000.npz'
    np.savez(path, **data)
    return path


def mutate(path, **updates):
    with np.load(path, allow_pickle=False) as file:
        data = dict(file)
    data.update(updates)
    np.savez(path, **data)


@pytest.mark.parametrize('schema', ['joint', 'delivery'])
@pytest.mark.parametrize('side', ['left', 'right', 'both'])
def test_current_collector_modes_units_and_terminal_hold(tmp_path, schema, side):
    episode = load_replay_episode(episode_file(tmp_path, schema=schema,
        mode='bimanual' if side == 'both' else 'single', side=side, both=schema == 'delivery'))
    assert episode.sides == (('left', 'right') if side == 'both' else (side,))
    for mode in episode.formats:
        width = 7 if mode == 'joint' else 10
        assert len(episode.targets[mode]) == 4
        assert episode.targets[mode][0, width - 1] == pytest.approx(.014)
        assert episode.timestamps[mode] == pytest.approx([0, .1, .2, .3])


def test_joint_only_does_not_invent_end_effector_representation(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path, schema='joint', both=False))
    assert episode.formats == ('joint',)


@pytest.mark.parametrize('updates,match', [
    ({'contract_version': 2}, 'current'),
    ({'gripper_semantics': 'absolute_opening_metres'}, 'Legacy|fraction'),
    ({'coordinate_frame': 'world'}, 'slave_base'),
    ({'joint_state_timestamp': [100, 100, 100.2, 100.3, 100.4]}, 'timestamps'),
])
def test_malformed_or_legacy_files_rejected(tmp_path, updates, match):
    path = episode_file(tmp_path)
    mutate(path, **updates)
    with pytest.raises((ValueError, ReplayBlocked), match=match):
        load_replay_episode(path)


def test_invalid_rotation_rejected_before_device_access(tmp_path):
    path = episode_file(tmp_path)
    with np.load(path) as file:
        state = file['eef_state']
    state[:, 3:9] = 0
    mutate(path, eef_state=state)
    with pytest.raises(ReplayBlocked, match='rotation6d'):
        load_replay_episode(path)


def test_gripper_range_rejected_without_silent_clipping(tmp_path):
    path = episode_file(tmp_path)
    with np.load(path) as file:
        state = file['joint_state']
    state[:, 6] = 1.1
    mutate(path, joint_state=state)
    with pytest.raises(ReplayBlocked, match='fractions'):
        load_replay_episode(path)


def test_inspect_cli_never_connects(tmp_path, capsys):
    path = episode_file(tmp_path)
    with patch.object(PiperReplayHardware, 'open', side_effect=AssertionError('CAN accessed')):
        main([str(path)])
    assert json.loads(capsys.readouterr().out)['execution'] is False


class ClockStop:
    def __init__(self):
        self.time = 100.
        self.stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.time += seconds
        return self.stopped


class FakeIK:
    def solve(self, current, xyz, rotation, **kwargs):
        result = current.copy()
        result[0] = xyz[0] - .2
        return result


class FakeHardware:
    def __init__(self, episode, offset=0):
        self.sides = episode.sides
        self.q = {s: episode.targets['joint'][0, i * 7:(i + 1) * 7].copy()
                  for i, s in enumerate(self.sides)}
        for q in self.q.values():
            q[0] += offset
        self.sent, self.stops = [], []
        self.opened, self.closed = False, False
        self.external = False
        self.enable_calls = 0

    def open(self):
        self.opened = True

    def recover_startup(self, stop, progress):
        self.check_ownership()
        progress({'state': 'recovery_not_needed'})

    def check_ownership(self):
        if self.external:
            raise ReplayBlocked('external master control')

    def limits(self):
        return {s: JOINT_LIMITS_RAD.copy() for s in self.sides}

    def joint_speed_limit(self, requested):
        return requested

    def read(self, *, require_ready=True):
        return {s: {'joint': q.copy(), 'eef': np.r_[.2 + q[0], .2, .3,
                 matrix_to_rotation6d(np.eye(3)), q[6]]} for s, q in self.q.items()}

    def enable_holding(self, initial, stop, progress, *, limits=None):
        self.enable_calls += 1
        progress({'state': 'enabled_hold_ready'})

    def send(self, commands):
        self.sent.append({s: q.copy() for s, q in commands.items()})
        self.q = {s: q.copy() for s, q in commands.items()}

    def stop_holding(self):
        if self.sent:
            self.stops.extend(self.sides)
        return []

    def close(self):
        self.closed = True


def simulate(episode, cfg, *, offset=.02, progress=lambda *args: None, configure=None):
    device, clock = FakeHardware(episode, offset=offset), ClockStop()
    if configure:
        configure(device, clock)
    with patch('bimanual_vla.collection.replay.time.monotonic', lambda: clock.time), \
         patch('bimanual_vla.deployment.client.PiperContinuousIK', FakeIK):
        result = run_replay(episode, cfg, {s: s for s in episode.sides}, exclusive_control=True,
            hardware_factory=lambda *a: device, stop=clock,
            progress=lambda e: progress(e, device, clock))
    return result, device, clock


@pytest.mark.parametrize('mode', ['joint', 'eef'])
@pytest.mark.parametrize('joint_smoothing', [False, True])
@pytest.mark.parametrize('eef_smoothing', [False, True])
def test_auto_approach_then_replay_all_smoothing_combinations(tmp_path, mode, joint_smoothing, eef_smoothing):
    episode = load_replay_episode(episode_file(tmp_path))
    events = []
    def progress(event, device, clock):
        events.append(event)
        if event['state'] == 'first_pose_reached':
            assert abs(device.q['right'][0] - episode.targets['joint'][0, 0]) < .01
    result, device, clock = simulate(episode, ReplayConfig(mode=mode,
        joint_smoothing=joint_smoothing, eef_smoothing=eef_smoothing), progress=progress)
    assert result == 'completed'
    states = [e['state'] for e in events]
    assert states.index('first_pose_reached') < states.index('running')
    assert device.sent and device.closed and not device.stops
    assert device.q['right'][0] == pytest.approx(episode.targets['joint'][-1, 0], abs=.003)


def test_bimanual_approach_and_synchronized_commands(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path, mode='bimanual', side='both'))
    _, device, _ = simulate(episode, ReplayConfig(), progress=lambda *a: None)
    assert all(set(batch) == {'left', 'right'} for batch in device.sent)


def test_recorded_positions_outside_legacy_task_box_are_valid_eef_targets():
    validate_eef_position([.002822, -.054516, .186954])
    validate_eef_position([-.363714, .356546, .409141])


@pytest.mark.parametrize('xyz', [[float('nan'), 0., 0.], [0., float('inf'), 0.], [0., 0.]])
def test_nonfinite_or_malformed_eef_targets_are_still_rejected(xyz):
    with pytest.raises(ReplayBlocked, match='finite XYZ'):
        validate_eef_position(xyz)


def test_eef_replay_outside_old_task_box_reaches_ik_and_joint_checks(tmp_path):
    path = episode_file(tmp_path)
    with np.load(path, allow_pickle=False) as source:
        updates = {key: source[key].copy() for key in ('state', 'eef_state')}
    for values in updates.values():
        values[:, 0] = -.3
        values[:, 1] = -.4
    mutate(path, **updates)
    episode = load_replay_episode(path)
    feedback = FakeHardware(episode).read()
    feedback['right']['eef'] = episode.targets['eef'][0].copy()
    ik = Mock(wraps=FakeIK())
    # This synthetic IK solution remains within actual joint bounds; no XYZ
    # workspace bypass may bypass the existing downstream joint checks.
    planner = ReplayPlanner(episode, ReplayConfig(mode='eef'),
        FakeHardware(episode).limits(), feedback, ik=ik)
    commands = planner.prepare(0, feedback)
    ik.solve.assert_called_once()
    assert np.isfinite(commands['right']).all()


@pytest.mark.parametrize('mode', ['joint', 'eef'])
def test_reference_and_ik_update_at_control_rate_while_joint_planner_keeps_advancing(tmp_path, mode):
    episode = load_replay_episode(episode_file(tmp_path))
    config = ReplayConfig(mode=mode, hz=100, control_hz=20)
    device = FakeHardware(episode)
    feedback = device.read()
    ik = Mock(wraps=FakeIK())
    planner = ReplayPlanner(episode, config, device.limits(), feedback, ik=ik)
    planner.prepare(0, feedback)
    joint = planner.joint['right']
    with patch.object(joint, 'retarget', wraps=joint.retarget) as retarget, \
         patch.object(joint, 'update', wraps=joint.update) as advance:
        for _ in range(4):
            planner.prepare(2, feedback, reference_update=False)
        assert retarget.call_count == 0
        assert advance.call_count == 4
        np.testing.assert_equal(planner.reference_raw['right'], episode.targets[mode][0])
        assert ik.solve.call_count == (1 if mode == 'eef' else 0)
        planner.prepare(2, feedback, reference_update=True)
        assert retarget.call_count == 1
        assert ik.solve.call_count == (2 if mode == 'eef' else 0)


@pytest.mark.parametrize('control_hz,command_hz', [(20, 100), (30, 100), (30, 30)])
def test_replay_runs_reference_and_command_clocks_at_distinct_rates(tmp_path, control_hz, command_hz):
    episode = load_replay_episode(episode_file(tmp_path))
    updates = []
    original = ReplayPlanner.prepare
    def track(self, index, feedback, **kwargs):
        updates.append(kwargs['reference_update'])
        return original(self, index, feedback, **kwargs)
    with patch.object(ReplayPlanner, 'prepare', track):
        result, _, _ = simulate(episode, ReplayConfig(control_hz=control_hz, hz=command_hz))
    assert result == 'completed'
    assert abs(len(updates) / sum(updates) - command_hz / control_hz) < .2


def test_preflight_external_master_never_commands_or_stops_any_arm(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path))
    device = FakeHardware(episode)
    device.external = True
    with pytest.raises(ReplayBlocked, match='external'):
        run_replay(episode, ReplayConfig(), {'right': 'can1'}, exclusive_control=True,
                   hardware_factory=lambda *a: device)
    assert device.closed and not device.sent and not device.stops


def test_operator_must_establish_exclusive_control_before_any_connection(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path))
    factory = Mock()
    with pytest.raises(ReplayBlocked, match='exclusive'):
        run_replay(episode, ReplayConfig(), {}, hardware_factory=factory)
    factory.assert_not_called()


@pytest.mark.parametrize('phase', ['approaching', 'running'])
def test_stop_in_approach_or_replay_only_stops_commanded_side(tmp_path, phase):
    episode = load_replay_episode(episode_file(tmp_path, side='left'))
    def progress(event, device, clock):
        if event['state'] == phase and device.sent:
            clock.stopped = True
    result, device, _ = simulate(episode, ReplayConfig(), progress=progress)
    assert result == 'stopped'
    assert device.stops == ['left']
    assert device.closed


def test_out_of_range_later_frame_blocks_before_approach(tmp_path):
    path = episode_file(tmp_path)
    with np.load(path) as data:
        joints = data['joint_state']
    joints[2, 1] = -1
    mutate(path, joint_state=joints)
    episode = load_replay_episode(path)
    device, clock = FakeHardware(episode), ClockStop()
    with patch('bimanual_vla.collection.replay.time.monotonic', lambda: clock.time), \
         pytest.raises(ReplayBlocked, match='recorded joint path'):
        run_replay(episode, ReplayConfig(), {}, exclusive_control=True, stop=clock,
                   hardware_factory=lambda *a: device)
    assert not device.sent and not device.stops and device.closed


def test_second_arm_ik_failure_publishes_neither_arm(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path, mode='bimanual', side='both'))
    device = FakeHardware(episode)
    class FailSecond(FakeIK):
        calls = 0
        def solve(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise ReplayBlocked('second arm IK failure')
            return super().solve(*args, **kwargs)
    cfg = ReplayConfig(mode='eef', joint_smoothing=False, eef_smoothing=False)
    planner = ReplayPlanner(episode, cfg, device.limits(), device.read(), ik=FailSecond())
    with pytest.raises(ReplayBlocked, match='second arm'):
        planner.prepare(0, device.read())
    assert not device.sent


def test_unsmoothed_jump_is_rejected_not_silently_clipped(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path))
    device = FakeHardware(episode)
    cfg = ReplayConfig(joint_smoothing=False, max_joint_speed=.01)
    planner = ReplayPlanner(episode, cfg, device.limits(), device.read())
    with pytest.raises(ReplayBlocked, match='unsmoothed'):
        planner.prepare(3, device.read())


@pytest.mark.parametrize('eef_smoothing', [False, True])
def test_eef_uses_inference_ik_step_settings(tmp_path, eef_smoothing):
    episode = load_replay_episode(episode_file(tmp_path))
    device = FakeHardware(episode)
    initial = device.read()
    desired = initial['right']['joint'][:6] + np.array([.006, .003, -.0015, 0., 0., 0.])
    class FixedIK:
        def solve(self, *args, **kwargs):
            return desired.copy()
    cfg = ReplayConfig(mode='eef', joint_smoothing=False, eef_smoothing=eef_smoothing,
                       hz=30, control_hz=30, ik_max_joint_step_rad=.02)
    planner = ReplayPlanner(episode, cfg, device.limits(), initial, ik=FixedIK())
    commands = planner.prepare(0, device.read())
    assert commands['right'][:6] == pytest.approx(desired)
    metrics = planner.ik_rate_limit['right']
    assert not planner.joint


def test_eef_step_limiting_does_not_bypass_feedback_tracking_guard(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path))
    device = FakeHardware(episode)
    initial = device.read()
    class FixedIK:
        def solve(self, current, *args, **kwargs):
            return current + [.2, 0., 0., 0., 0., 0.]
    cfg = ReplayConfig(mode='eef', joint_smoothing=False, hz=30, control_hz=30,
                       max_joint_speed=.6, max_tracking_error=.03)
    planner = ReplayPlanner(episode, cfg, device.limits(), initial, ik=FixedIK())
    with pytest.raises(ReplayBlocked, match='tracking error exceeded'):
        planner.prepare(0, initial)
    assert not device.sent


def test_unsmoothed_eef_run_uses_reference_rate_and_reports_limiting(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path))
    events = []
    result, device, _ = simulate(episode, ReplayConfig(mode='eef', joint_smoothing=False,
        control_hz=30, hz=100), progress=lambda event, *_: events.append(event))
    assert result == 'completed'
    running = [event for event in events if event['state'] == 'running']
    assert running[0]['command_hz'] == running[0]['control_hz'] == 30
    assert any('right' in event.get('ik_rate_limit', {}) for event in running)


@pytest.mark.parametrize('eef_smoothing', [False, True])
def test_eef_limiting_does_not_retime_shared_clock(tmp_path, eef_smoothing):
    path = episode_file(tmp_path, mode='bimanual', side='both')
    with np.load(path) as data:
        eef = data['eef_state'].copy()
    eef[1:, 10] += .06  # Only right-arm IK needs to slow down.
    eef[2:, 9] += .01  # Left gripper must wait on the same source clock.
    mutate(path, eef_state=eef)
    episode = load_replay_episode(path)
    events = []
    cfg = ReplayConfig(mode='eef', speed=1., joint_smoothing=False,
        eef_smoothing=eef_smoothing, control_hz=30, hz=30,
        max_joint_speed=.025, settle_timeout=1.5)
    result, device, _ = simulate(episode, cfg,
        progress=lambda event, *_: events.append(deepcopy(event)))
    assert result == 'completed'
    running = [e for e in events if 'commands_rad_m' in e and e['state'] == 'running']
    assert running[-1]['elapsed_s'] >= .3
    assert running[-1]['playback_elapsed_s'] == pytest.approx(running[-1]['elapsed_s'])
    for side in episode.sides:
        assert abs(device.q[side][0] - (episode.targets['eef'][-1,
            0 if side == 'left' else 10] - .2)) < .003


def test_raw_eef_gripper_also_limits_speed_and_holds_the_source_clock(tmp_path):
    path = episode_file(tmp_path)
    with np.load(path) as data:
        eef = data['eef_state'].copy()
    eef[1:, 9] = .8
    mutate(path, eef_state=eef)
    episode = load_replay_episode(path)
    events = []
    cfg = ReplayConfig(mode='eef', speed=1., joint_smoothing=False,
                       eef_smoothing=False, control_hz=30, hz=30)
    result, device, _ = simulate(episode, cfg,
        progress=lambda event, *_: events.append(deepcopy(event)))
    assert result == 'completed'
    openings = np.array([batch['right'][6] for batch in device.sent])
    assert np.max(abs(np.diff(openings))) <= cfg.max_gripper_speed / cfg.hz + 1e-12
    assert device.q['right'][6] == pytest.approx(.8 * .07)
    running = [event for event in events if 'ik_rate_limit' in event]
    assert any(event['ik_rate_limit']['right']['gripper_limited'] for event in running)


def test_external_control_appearing_during_motion_stops_replay(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path, side='left'))
    captured = []
    def configure(device, clock):
        captured.append(device)
    def progress(event, device, clock):
        if event['state'] == 'running':
            device.external = True
    with pytest.raises(ReplayBlocked, match='external'):
        simulate(episode, ReplayConfig(), progress=progress, configure=configure)
    assert captured[0].closed and captured[0].stops == ['left']


def test_approach_timeout_does_not_start_episode_clock(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path))
    events, devices = [], []
    with pytest.raises(ReplayBlocked, match='approach timed out'):
        simulate(episode, ReplayConfig(approach_timeout=.01), offset=.1,
                 configure=lambda device, clock: devices.append(device),
                 progress=lambda e, *args: events.append(e))
    assert 'running' not in [e['state'] for e in events]
    assert devices[0].closed and devices[0].stops == ['right']


def test_feedback_expiring_after_planning_never_publishes_commands():
    hardware = PiperReplayHardware(('right',), {}, ReplayConfig())
    hardware.check_ownership = Mock()
    hardware.arms = {'right': Mock()}
    with pytest.raises(ReplayBlocked, match='expired'):
        hardware.send({'right': np.zeros(7)})
    assert not hardware.commanded
    hardware.arms['right'].ModeCtrl.assert_not_called()


def test_driver_speed_limit_is_applied_to_approach_and_replay(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path))
    events = []
    def configure(device, clock):
        device.joint_speed_limit = lambda requested: min(requested, .1)
    simulate(episode, ReplayConfig(), configure=configure,
             progress=lambda event, *a: events.append(event))
    event = next(e for e in events if e['state'] == 'limits_verified')
    assert event['effective_max_joint_speed_rad_s'] == .1


def test_raw_monitor_excludes_local_echo_and_rejects_remote_commands():
    monitor = RemoteCommandMonitor.__new__(RemoteCommandMonitor)
    monitor.interface = 'can_test'
    packet = struct.pack('=IB3x8s', 0x155, 8, bytes(8))
    monitor.sock = Mock()
    monitor.sock.recvmsg.side_effect = [(packet, [], socket.MSG_DONTROUTE, None), BlockingIOError()]
    monitor.check()
    monitor.sock.recvmsg.side_effect = [(packet, [], 0, None)]
    with pytest.raises(ReplayBlocked, match='external control frame'):
        monitor.check()


def test_partial_send_failure_never_disables_or_takes_control_of_an_uncommanded_arm():
    hardware = PiperReplayHardware(('left', 'right'), {}, ReplayConfig())
    hardware.check_ownership = Mock()
    hardware.feedback_deadline = float('inf')
    hardware.arms = {'left': Mock(), 'right': Mock()}
    hardware.arms['left'].ModeCtrl.side_effect = RuntimeError('transport failure')
    with pytest.raises(RuntimeError, match='transport'):
        hardware.send({'left': np.zeros(7), 'right': np.zeros(7)})
    with patch('bimanual_vla.deployment.client.arm_status_dict', return_value={
        'feedback_fresh': False}):
        errors = hardware.stop_holding()
    assert len(errors) == 1 and errors[0].startswith('left:')
    hardware.arms['left'].EmergencyStop.assert_not_called()
    hardware.arms['right'].EmergencyStop.assert_not_called()
    hardware.arms['right'].ModeCtrl.assert_not_called()


@pytest.mark.parametrize('unsafe', [None, 'stale', 'fault', 'disabled', 'external', 'bounds', 'transport'])
def test_stop_holds_fresh_measured_joints_without_disabling_or_reenabling(unsafe):
    hardware = PiperReplayHardware(('left', 'right'), {}, ReplayConfig())
    hardware.arms = {'left': Mock(), 'right': Mock()}
    for arm in hardware.arms.values():
        arm._vla_joint_limits = JOINT_LIMITS_RAD.copy()
    hardware.commanded = {'left'}
    hardware.check_ownership = Mock()
    q = np.array([.2, 1., -1., .1, -.1, .2, .014])
    status = {'feedback_fresh': True, 'arm_status': 0, 'err_code': 0,
              'ctrl_mode': 1, 'mode_feed': 1}
    drivers = {'healthy': True, 'ready': True, 'faulted': False}
    if unsafe == 'fault':
        status['arm_status'] = 1
    if unsafe == 'disabled':
        drivers['ready'] = False
    if unsafe == 'external':
        hardware.check_ownership.side_effect = ReplayBlocked('external control')
    if unsafe == 'bounds':
        q[1] = -1.
    if unsafe == 'transport':
        hardware.arms['left'].JointCtrl.side_effect = RuntimeError('send failed')
    with patch('bimanual_vla.deployment.client.arm_status_dict', return_value=status), \
         patch('bimanual_vla.deployment.client.driver_enable_status_dict', return_value=drivers), \
         patch('bimanual_vla.deployment.client.read_output_qpos', return_value=q) as read:
        if unsafe == 'stale':
            read.side_effect = ReplayFeedbackPending('stale joint feedback')
        errors = hardware.stop_holding()
    if unsafe is None:
        assert errors == []
        hardware.arms['left'].JointCtrl.assert_called_once_with(
            *[round(v * 180000 / np.pi) for v in q[:6]])
        assert read.call_args.kwargs['max_feedback_age_s'] == .1
    else:
        assert len(errors) == 1 and 'position hold not requested' in errors[0]
        if unsafe != 'transport':
            hardware.arms['left'].JointCtrl.assert_not_called()
    hardware.arms['right'].assert_not_called()
    assert not hardware.arms['right'].mock_calls
    for arm in hardware.arms.values():
        for name in ('EmergencyStop', 'DisableArm', 'DisablePiper', 'ResetPiper', 'ModeCtrl',
                     'EnableArm', 'EnablePiper', 'GripperCtrl', 'MasterSlaveConfig'):
            getattr(arm, name).assert_not_called()


@pytest.mark.parametrize('updates', [{'speed': 0}, {'hz': float('nan')},
    {'approach_timeout': -1}, {'speed_pct': 3.5}, {'max_joint_speed': float('inf')}])
def test_invalid_settings_are_rejected(updates):
    with pytest.raises(ValueError):
        replace(ReplayConfig(), **updates)


def test_gui_collection_and_inference_interlocks_without_tk_display():
    from bimanual_vla.collection.gui import CollectorGUI
    gui = CollectorGUI.__new__(CollectorGUI)
    gui.replay_dialog = SimpleNamespace(running=True)
    with patch('bimanual_vla.collection.gui.messagebox.showwarning') as warning:
        gui.start_episode()
        gui.start_inference()
        gui.activate_can()
        gui.return_inference_to_initial()
        gui.toggle_connection()
        gui.disconnect()
        gui.open_device_settings()
    assert warning.call_count == 7


def test_initial_feedback_waits_for_all_first_motor_replies():
    hardware, clock = Mock(), ClockStop()
    hardware.read.side_effect = [ReplayFeedbackPending('partial'), ReplayFeedbackPending('partial'), {'fresh': True}]
    with patch('bimanual_vla.collection.replay.time.monotonic', lambda: clock.time):
        result = wait_for_replay_feedback(hardware, clock, .2)
    assert result == {'fresh': True}
    assert hardware.read.call_count == 3
    assert clock.time == pytest.approx(100.04)


def test_feedback_wait_has_timeout_and_does_not_enable_anything():
    hardware, clock = Mock(), ClockStop()
    hardware.read.side_effect = ReplayFeedbackPending('SDK cache stale')
    with patch('bimanual_vla.collection.replay.time.monotonic', lambda: clock.time), \
         pytest.raises(ReplayBlocked, match='within 0.1s.*cache stale'):
        wait_for_replay_feedback(hardware, clock, .1)
    hardware.send.assert_not_called()


def test_disabled_or_faulted_arm_is_not_retried_as_missing_feedback():
    hardware, clock = Mock(), ClockStop()
    hardware.read.side_effect = ReplayBlocked('motors disabled at joints [1, 2, 3, 4, 5, 6]')
    with pytest.raises(ReplayBlocked, match='motors disabled'):
        wait_for_replay_feedback(hardware, clock, 3.)
    assert hardware.read.call_count == 1
    hardware.send.assert_not_called()


def test_real_hardware_read_reports_disabled_joints_and_mode():
    hardware = PiperReplayHardware(('left',), {}, ReplayConfig())
    hardware.arms = {'left': Mock()}
    status = {'feedback_fresh': True, 'arm_status': 0, 'err_code': 0,
              'ctrl_mode': 0, 'mode_feed': 0}
    drivers = {'faulted': False, 'healthy': True, 'ready': False, 'enabled': [False] * 6}
    with patch('bimanual_vla.deployment.client.arm_status_dict', return_value=status), \
         patch('bimanual_vla.deployment.client.driver_enable_status_dict', return_value=drivers), \
         patch('bimanual_vla.deployment.client.read_output_state') as read:
        with pytest.raises(ReplayBlocked, match=r'left: motors disabled.*ctrl_mode=0'):
            hardware.read()
        read.assert_not_called()
    assert not hardware.commanded


def test_gui_retries_start_fresh_processes_and_preserve_all_settings(tmp_path):
    from bimanual_vla.collection.replay_gui import run_replay_worker
    config = ReplayConfig(mode='eef', speed=.4, joint_smoothing=False,
                          approach_joint_speed=.08, feedback_ready_timeout=4.)
    processes = [Mock(pid=123), Mock(pid=124)]
    for process in processes:
        process.stdout = io.StringIO('{"state":"completed","index":0,"frames":1}\n')
        process.wait.return_value = 0
        process.poll.return_value = 0
    with patch('bimanual_vla.collection.replay_gui.subprocess.Popen', side_effect=processes) as popen:
        for i in range(2):
            config_path = tmp_path / f'retry_{i}.json'
            run_replay_worker(tmp_path / 'ep.npz', config, {'left': 'serial:L', 'right': 'serial:R'},
                              config_path, threading.Event(), lambda event: None)
            data = json.loads(config_path.read_text())
            assert data['speed'] == .4 and data['joint_smoothing'] is False
            assert data['approach_joint_speed'] == .08
            assert data['feedback_ready_timeout'] == 4.
    assert popen.call_count == 2
    for call in popen.call_args_list:
        command = call.args[0]
        assert command[1:3] == ['-m', 'bimanual_vla.collection.replay']
        assert command[command.index('--parent-pid') + 1] == str(os.getpid())
        assert call.kwargs['start_new_session'] is True


def test_worker_propagates_hardware_error_without_losing_reason(tmp_path):
    from bimanual_vla.collection.replay_gui import run_replay_worker
    process = Mock(pid=123)
    process.stdout = io.StringIO('{"state":"failed","error":"left: motors disabled","traceback":"test trace"}\n')
    process.wait.return_value = process.poll.return_value = 1
    events = []
    with patch('bimanual_vla.collection.replay_gui.subprocess.Popen', return_value=process), \
         pytest.raises(ReplayBlocked, match='left: motors disabled'):
        run_replay_worker(tmp_path / 'ep.npz', ReplayConfig(), {}, tmp_path / 'config.json',
                          threading.Event(), events.append)
    assert any(e.get('traceback') == 'test trace' for e in events)


def test_stop_watcher_reaches_child_while_stdout_is_waiting(tmp_path):
    from bimanual_vla.collection.replay_gui import run_replay_worker
    stop, child_stopped = threading.Event(), threading.Event()
    process = Mock(pid=123)
    process.poll.side_effect = lambda: 0 if child_stopped.is_set() else None
    process.wait.return_value = 0
    process.send_signal.side_effect = lambda signal: child_stopped.set()
    class WaitingOutput:
        def __iter__(self):
            stop.set()
            assert child_stopped.wait(1), 'stop must interrupt a pending stdout read'
            yield '{"state":"stopped","stop_errors":[]}\n'
        def close(self):
            pass
    process.stdout = WaitingOutput()
    with patch('bimanual_vla.collection.replay_gui.subprocess.Popen', return_value=process):
        run_replay_worker(tmp_path / 'ep.npz', ReplayConfig(), {}, tmp_path / 'config.json', stop, lambda event: None)
    assert process.send_signal.call_count == 1


def test_cli_reports_structured_runtime_failure(tmp_path, capsys):
    path = episode_file(tmp_path)
    with patch('bimanual_vla.collection.replay.run_replay', side_effect=ReplayBlocked('motor disabled')), \
         pytest.raises(SystemExit) as error:
        main([str(path), '--allow-execution', '--exclusive-control'])
    event = json.loads(capsys.readouterr().out)
    assert event['state'] == 'failed' and event['error'] == 'motor disabled'
    assert event['traceback']
    assert error.value.code == 1


def test_cli_receives_gui_config_and_restores_signal_handlers(tmp_path):
    from dataclasses import asdict
    import signal
    config = ReplayConfig(mode='eef', joint_smoothing=False, speed=.4, approach_joint_speed=.07)
    config_path = tmp_path / 'worker.config.json'
    config_path.write_text(json.dumps(asdict(config)))
    previous = signal.getsignal(signal.SIGTERM)
    with patch('bimanual_vla.collection.replay.run_replay') as run:
        main([str(episode_file(tmp_path)), '--allow-execution', '--exclusive-control',
              '--config-file', str(config_path)])
    assert run.call_args.args[1] == config
    assert signal.getsignal(signal.SIGTERM) == previous


def test_worker_with_lost_parent_is_cancelled_before_device_connection(tmp_path):
    with patch('bimanual_vla.collection.replay.run_replay') as run:
        main([str(episode_file(tmp_path)), '--allow-execution', '--exclusive-control', '--parent-pid', '0'])
    assert run.call_args.kwargs['stop'].is_set()


@contextmanager
def enable_device(*, initially_enabled=(False,) * 6, stuck=False):
    """Exercise the real hardware boundary with simulated SDK feedback only."""
    clock = ClockStop()
    hardware = PiperReplayHardware(('left', 'right'), {}, ReplayConfig(enable_before_replay=True))
    initial = {s: {'joint': np.array([.05, 1., -1., 0., 0., 0., .014])}
               for s in hardware.sides}
    enabled = {s: list(initially_enabled) for s in hardware.sides}
    calls, faulted = [], set()
    for side in hardware.sides:
        arm = Mock(side=side)
        for name in ('GetArmJointMsgs', 'GetArmGripperMsgs', 'GetArmEndPoseMsgs'):
            getattr(arm, name).side_effect = lambda: SimpleNamespace(time_stamp=clock.time)
        for name in ('ModeCtrl', 'JointCtrl', 'GripperCtrl', 'EmergencyStop'):
            getattr(arm, name).side_effect = lambda *args, s=side, n=name: calls.append((s, n, args))
        def enable(joint, s=side):
            calls.append((s, 'EnableArm', (joint,)))
            if not stuck:
                enabled[s][joint - 1] = True
        arm.EnableArm.side_effect = enable
        hardware.arms[side] = arm
    hardware.check_ownership = Mock()
    def status(arm):
        return {'feedback_fresh': True, 'arm_status': 0, 'err_code': 0,
                'ctrl_mode': int(all(enabled[arm.side])), 'mode_feed': int(all(enabled[arm.side])),
                'feedback_timestamp': clock.time}
    def drivers(arm):
        return {'healthy': True, 'faulted': arm.side in faulted, 'faults': ['driver_error'],
                'ready': all(enabled[arm.side]), 'enabled': enabled[arm.side],
                'feedback_timestamp': clock.time}
    def feedback(arm, **kwargs):
        return np.r_[.2, .2, .3, matrix_to_rotation6d(np.eye(3)), .014], initial[arm.side]['joint'].copy()
    with patch('bimanual_vla.collection.replay.time.monotonic', lambda: clock.time), \
         patch('bimanual_vla.collection.replay.time.time', lambda: clock.time), \
         patch('bimanual_vla.deployment.client.arm_status_dict', side_effect=status), \
         patch('bimanual_vla.deployment.client.driver_enable_status_dict', side_effect=drivers), \
         patch('bimanual_vla.deployment.client.read_output_state', side_effect=feedback):
        yield hardware, initial, clock, calls, enabled, faulted


def test_enable_preloads_exact_measured_targets_then_confirms_all_motors():
    with enable_device() as (hardware, initial, clock, calls, enabled, _):
        events = []
        hardware.enable_holding(initial, clock, events.append)
        first_enable = next(i for i, c in enumerate(calls) if c[1] == 'EnableArm')
        assert clock.time >= 100.34
        assert [e['state'] for e in events] == ['enabling_hold', 'enabled_hold_ready']
        for side in hardware.sides:
            side_calls = [c for c in calls[:first_enable] if c[0] == side]
            assert side_calls[0][1] == 'JointCtrl'  # Before mode selection or enable.
            assert side_calls[0][2] == tuple(round(v * 180000 / np.pi) for v in initial[side]['joint'][:6])
            assert all(enabled[side])
            hardware.arms[side].EmergencyStop.assert_not_called()
            hardware.arms[side].ResetPiper.assert_not_called()
            hardware.arms[side].MasterSlaveConfig.assert_not_called()
            hardware.arms[side].EnablePiper.assert_not_called()
            hardware.arms[side].GripperCtrl.assert_not_called()
        assert hardware.commanded == {'left', 'right'}


def test_enable_timeout_is_bounded_without_disabling_or_resetting():
    with enable_device(stuck=True) as (hardware, initial, clock, calls, *_):
        with pytest.raises(ReplayBlocked, match='did not stabilize'):
            hardware.enable_holding(initial, clock, lambda e: None)
        assert clock.time < 103.1
        errors = hardware.stop_holding()
        assert len(errors) == 2  # Not enabled: no new hold command is safe.
        assert not [c for c in calls if c[1] == 'EmergencyStop']
        for arm in hardware.arms.values():
            arm.DisableArm.assert_not_called()
            arm.ResetPiper.assert_not_called()


def test_partial_motor_enable_rejected_before_any_writes():
    with enable_device(initially_enabled=(True, False, False, False, False, False)) as state:
        hardware, initial, clock, calls, *_ = state
        with pytest.raises(ReplayBlocked, match='partially enabled'):
            hardware.enable_holding(initial, clock, lambda e: None)
        assert not calls and not hardware.commanded


def test_enable_cancelled_or_external_control_never_writes():
    for external in (False, True):
        with enable_device() as (hardware, initial, clock, calls, *_):
            if external:
                hardware.check_ownership.side_effect = ReplayBlocked('external')
            else:
                clock.stopped = True
            with pytest.raises(ReplayBlocked if external else ReplayCancelled):
                hardware.enable_holding(initial, clock, lambda e: None)
            assert not calls and not hardware.commanded


def test_driver_fault_during_enable_stops_further_enable_requests():
    with enable_device() as (hardware, initial, clock, calls, enabled, faulted):
        original = hardware.arms['left'].EnableArm.side_effect
        def fault_on_enable(joint):
            original(joint)
            faulted.add('left')
        hardware.arms['left'].EnableArm.side_effect = fault_on_enable
        with pytest.raises(ReplayBlocked, match='driver faults'):
            hardware.enable_holding(initial, clock, lambda e: None)
        assert hardware.arms['left'].EnableArm.call_count <= 6


def test_pose_drift_during_preload_blocks_before_enabling():
    with enable_device() as (hardware, initial, clock, calls, *_):
        anchor = {s: {'joint': v['joint'].copy()} for s, v in initial.items()}
        original = hardware.arms['left'].JointCtrl.side_effect
        def drift(*args):
            original(*args)
            initial['left']['joint'][0] += .02
        hardware.arms['left'].JointCtrl.side_effect = drift
        with pytest.raises(ReplayBlocked, match='pose drifted'):
            hardware.enable_holding(anchor, clock, lambda e: None)
        assert not any(name == 'EnableArm' for _, name, _ in calls)


def test_stale_feedback_during_enable_fails_without_retrying():
    with enable_device() as (hardware, initial, clock, calls, *_):
        original = hardware.read
        def read(**kwargs):
            if clock.time >= 100.1:
                raise ReplayFeedbackPending('stale during enable')
            return original(**kwargs)
        hardware.read = read
        with pytest.raises(ReplayFeedbackPending, match='stale during enable'):
            hardware.enable_holding(initial, clock, lambda e: None)
        assert clock.time < 100.2
        assert not any(name == 'EnableArm' for _, name, _ in calls)


def test_optional_enable_precedes_first_frame_approach(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path))
    events = []
    result, device, _ = simulate(episode, ReplayConfig(enable_before_replay=True),
                                progress=lambda e, *a: events.append(e['state']))
    assert result == 'completed' and device.enable_calls == 1
    assert events.index('enabled_hold_ready') < events.index('approaching') < events.index('running')


def test_invalid_start_blocks_before_enable_and_reports_joint_value(tmp_path):
    episode = load_replay_episode(episode_file(tmp_path))
    devices = []
    def bad_start(device, clock):
        device.q['right'][1] = -.07
        devices.append(device)
    with pytest.raises(ReplayBlocked, match=r'J2=-4.011deg.*no enabling'):
        simulate(episode, ReplayConfig(enable_before_replay=True), configure=bad_start)
    assert devices[0].enable_calls == 0 and not devices[0].sent


def test_cli_enable_is_explicit_and_preserved(tmp_path):
    with patch('bimanual_vla.collection.replay.run_replay') as run:
        main([str(episode_file(tmp_path)), '--allow-execution', '--exclusive-control', '--enable-before-replay'])
    assert run.call_args.args[1].enable_before_replay is True


@pytest.mark.parametrize('enabled', [False, True])
def test_cli_auto_recovery_switch_is_preserved(tmp_path, enabled):
    with patch('bimanual_vla.collection.replay.run_replay') as run:
        main([str(episode_file(tmp_path)), '--allow-execution', '--exclusive-control',
              '--auto-recover' if enabled else '--no-auto-recover'])
    assert run.call_args.args[1].auto_recover is enabled


@pytest.mark.parametrize('enabled', [False, True])
def test_startup_recovery_switch_controls_hardware_call(tmp_path, enabled):
    episode = load_replay_episode(episode_file(tmp_path))
    def configure(device, clock):
        device.recover_startup = Mock()
    result, device, _ = simulate(episode, ReplayConfig(auto_recover=enabled), configure=configure)
    assert result == 'completed'
    assert device.recover_startup.call_count == int(enabled)


def test_small_negative_j2_gets_explicit_valid_boundary_target_without_changing_feedback():
    measured = {'right': {'joint': np.array([0., -.034, -1., 0., 0., 0., .01])}}
    original = measured['right']['joint'].copy()
    bounds = {'right': JOINT_LIMITS_RAD.copy()}
    result = prepare_replay_start(measured, bounds, allow_boundary_alignment=True)
    assert result['right'][1] == bounds['right'][1, 0]
    assert np.array_equal(measured['right']['joint'], original)
    assert np.array_equal(result['right'][[0, 2, 3, 4, 5, 6]], original[[0, 2, 3, 4, 5, 6]])
    with pytest.raises(ReplayBlocked, match='outside limits'):
        prepare_replay_start(measured, bounds)


@pytest.mark.parametrize('joint,value', [(1, -.061), (1, 4.), (2, .01), (0, 3.)])
def test_boundary_alignment_does_not_relax_other_command_limits(joint, value):
    q = np.array([0., .1, -1., 0., 0., 0., .01])
    q[joint] = value
    with pytest.raises(ReplayBlocked, match='outside limits'):
        prepare_replay_start({'left': {'joint': q}}, {'left': JOINT_LIMITS_RAD},
                             allow_boundary_alignment=True)


def test_real_enable_boundary_alignment_never_publishes_negative_j2():
    with enable_device() as (hardware, live, clock, calls, enabled, _):
        for side in hardware.sides:
            live[side]['joint'][1] = -.034
        snapshot = {s: {'joint': d['joint'].copy()} for s, d in live.items()}
        original_read = hardware.read
        def moving_read(**kwargs):
            data = original_read(**kwargs)
            for side in hardware.sides:
                if all(enabled[side]):
                    live[side]['joint'][1] = min(0., live[side]['joint'][1] + .004)
            return data
        hardware.read = moving_read
        events = []
        hardware.enable_holding(snapshot, clock, events.append,
                                limits={s: JOINT_LIMITS_RAD.copy() for s in hardware.sides})
        assert all(c[2][1] == 0 for c in calls if c[1] == 'JointCtrl')
        assert all(snapshot[s]['joint'][1] == -.034 for s in hardware.sides)
        states = [e['state'] for e in events]
        assert states.index('aligning_start_boundary') < states.index('enabled_hold_ready')
        assert clock.time >= 100.8
