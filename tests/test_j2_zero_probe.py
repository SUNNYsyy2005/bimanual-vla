"""Offline verification of the single-side J2-zero diagnostic."""
import json
import math
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from analysis.collection_check_20261025 import probe_j2_zero as probe
from bimanual_vla.collection.replay import ReplayBlocked
from bimanual_vla.deployment.client import JOINT_LIMITS_RAD


def anchor():
    return np.r_[np.deg2rad([90., -2., -8., 3., 22., -9.]), .004]


def test_zero_target_only_changes_j2_and_preserves_input():
    q = anchor()
    target = probe.zero_target(q)
    assert target[1] == 0 and q[1] == math.radians(-2.)
    assert np.array_equal(target[[0, 2, 3, 4, 5, 6]], q[[0, 2, 3, 4, 5, 6]])
    probe.check_feedback(target, q)


@pytest.mark.parametrize('joint,delta', [(0, .01), (1, -.01), (5, .01), (6, .002)])
def test_j2_zero_envelope_rejects_unplanned_movement(joint, delta):
    q = anchor()
    measured = q.copy()
    measured[joint] += delta
    with pytest.raises(ReplayBlocked, match='envelope'):
        probe.check_feedback(measured, q)


def test_default_is_passive_and_execution_writes_only_left_joint_commands(tmp_path, monkeypatch):
    clock = {'now': 100.}
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock['now'])
    monkeypatch.setattr(probe.time, 'time', lambda: clock['now'])
    monkeypatch.setattr(probe, '__file__', str(tmp_path / 'probe_j2_zero.py'))
    def wait(seconds):
        clock['now'] += seconds
        return False
    stop = Mock()
    stop.is_set.return_value = False
    stop.wait.side_effect = wait
    monkeypatch.setattr(probe.threading, 'Event', lambda: stop)
    monkeypatch.setattr(probe.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(probe, 'resolve_can_selector', lambda s: 'can0' if s == probe.SELECTOR else 'can1')
    original_read = Path.read_text
    def read_text(path, *args, **kwargs):
        if str(path).startswith('/sys/class/net/'):
            return '0'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read_text)
    arm = Mock()
    enabled = {'value': False, 'mode': False}
    requested = set()
    def enable_joint(joint):
        requested.add(joint)
        enabled['value'] = requested == set(range(1, 7))
    arm.EnableArm.side_effect = enable_joint
    arm.ModeCtrl.side_effect = lambda *a: enabled.update(mode=True)
    hardware = Mock()
    hardware.commanded = set()
    hardware.arms = {'left': arm}
    hardware.limits.return_value = {'left': JOINT_LIMITS_RAD.copy()}
    hardware.emergency_stop.return_value = []
    q = anchor()
    def feedback(**kwargs):
        if enabled['value'] and enabled['mode']:
            q[1] = min(0., q[1] + math.radians(.04))
        hardware.feedback_deadline = clock['now'] + .15
        return {'left': {'joint': q.copy()}}
    hardware.read.side_effect = feedback
    monkeypatch.setattr(probe, 'PiperReplayHardware', lambda *a: hardware)
    monkeypatch.setattr(probe, 'wait_for_replay_feedback', lambda *a, **k: feedback())
    monkeypatch.setattr(probe, 'driver_enable_status_dict', lambda arm: {
        'enabled': [enabled['value']] * 6, 'ready': enabled['value']})
    monkeypatch.setattr(probe, 'arm_status_dict', lambda arm: {
        'ctrl_mode': int(enabled['mode']), 'mode_feed': int(enabled['mode']),
        'feedback_timestamp': clock['now']})
    plan = tmp_path / 'plan.json'
    probe.main(['--plan', str(plan)])
    arm.JointCtrl.assert_not_called()
    arm.EnablePiper.assert_not_called()
    arm.EnableArm.assert_not_called()
    probe.main(['--plan', str(plan), '--execute'])
    assert hardware.commanded == {'left'} and q[1] == 0
    assert all(call.args[1] == 0 for call in arm.JointCtrl.call_args_list)
    arm.GripperCtrl.assert_not_called()
    arm.EnablePiper.assert_not_called()
    assert {call.args[0] for call in arm.EnableArm.call_args_list} == set(range(1, 7))
    arm.ResetPiper.assert_not_called()
    arm.EmergencyStop.assert_not_called()
    hardware.emergency_stop.assert_not_called()
    assert json.loads(plan.read_text())['execution_attempted_at'] == pytest.approx(102.)
    writes = arm.JointCtrl.call_count
    with pytest.raises(ReplayBlocked, match='already consumed'):
        probe.main(['--plan', str(plan), '--execute'])
    assert arm.JointCtrl.call_count == writes
