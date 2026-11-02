import json
import time
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np

from bimanual_vla.collection.gui import CollectorGUI
from bimanual_vla.collection.session import SessionState
from bimanual_vla.collection.quest.input import QuestReader


def gui():
    controller = SimpleNamespace(resetting=False, clutches={'left': object(), 'right': object()})
    session = SimpleNamespace(quest_controller=controller, state=SessionState.READY,
                              request_quest_reset=Mock())
    ui = SimpleNamespace(session=session, recording=False, arm_mode='bimanual', arm_side='both',
        can_var=Mock(get=lambda: 'unused'), left_can_var=Mock(get=lambda: 'left-serial'),
        right_can_var=Mock(get=lambda: 'right-serial'), status_var=Mock(),
        _replay_running=lambda: False, _inference_running=lambda: False,
        _pending_collection_reset=None, _update_start_button=Mock(), stop_episode=Mock())
    ui._execute_pending_collection_reset = MethodType(CollectorGUI._execute_pending_collection_reset, ui)
    return ui


def test_hand_reset_uses_connection_pose_and_gui_reset_uses_saved_pose(tmp_path, monkeypatch):
    ui = gui()
    path = tmp_path/'return_pose.json'
    path.write_text(json.dumps(dict(arm_mode='bimanual', left_can='left-serial',
        right_can='right-serial', qpos=[.1]*6+[.07]+[.2]*6+[0.])))
    monkeypatch.setattr('bimanual_vla.collection.gui.RETURN_POSE_PATH', path)
    CollectorGUI._request_collection_reset(ui, 'initial')
    ui.session.request_quest_reset.assert_called_once_with(None)
    CollectorGUI._request_collection_reset(ui, 'saved')
    target = ui.session.request_quest_reset.call_args.args[0]
    np.testing.assert_allclose(target['left'], .1)
    np.testing.assert_allclose(target['right'], .2)


def test_reset_waits_for_episode_review_and_never_starts_while_recording():
    ui = gui()
    ui.recording = True
    ui.session.state = SessionState.RECORDING
    CollectorGUI._request_collection_reset(ui, 'initial')
    ui.stop_episode.assert_called_once()
    ui.session.request_quest_reset.assert_not_called()
    ui.session.state = SessionState.REVIEW
    ui._execute_pending_collection_reset()
    ui.session.request_quest_reset.assert_not_called()
    ui.session.state = SessionState.READY
    ui._execute_pending_collection_reset()
    ui.session.request_quest_reset.assert_called_once_with(None)


def test_wrong_can_binding_in_saved_pose_never_moves(tmp_path, monkeypatch):
    ui = gui()
    path=tmp_path/'return_pose.json'
    path.write_text(json.dumps(dict(arm_mode='bimanual', left_can='wrong',
        right_can='right-serial', qpos=[0.]*14)))
    monkeypatch.setattr('bimanual_vla.collection.gui.RETURN_POSE_PATH', path)
    CollectorGUI._request_collection_reset(ui, 'saved')
    ui.session.request_quest_reset.assert_not_called()
    assert ui._pending_collection_reset is None


def input_ui(reader):
    return SimpleNamespace(
        session=SimpleNamespace(quest_controller=SimpleNamespace(reader=reader)),
        quest_input_status_var=Mock(), quest_hand_vars={'left': Mock(), 'right': Mock()})


def test_input_panel_shows_received_buttons_triggers_and_clears_stale_values(monkeypatch):
    reader = QuestReader(max_age_s=.2)
    matrix = ' '.join(str(v) for v in np.eye(4).ravel())
    assert reader.ingest(f'l:{matrix}|r:{matrix}&L,Y,R,A,leftTrig 0.125,rightTrig 1', time.time())
    ui = input_ui(reader)
    CollectorGUI._update_quest_input(ui)
    assert 'Receiving' in ui.quest_input_status_var.set.call_args.args[0]
    left = ui.quest_hand_vars['left'].set.call_args.args[0]
    right = ui.quest_hand_vars['right'].set.call_args.args[0]
    assert 'OK' in left and 'X:up' in left and 'Y:DOWN' in left and '0.125' in left
    assert 'OK' in right and 'A:DOWN' in right and 'B:up' in right and '1.000' in right
    # Use the real reader timeout so a stale pressed button is never shown as live.
    received_at = reader.snapshot().received_monotonic
    monkeypatch.setattr(time, 'monotonic', lambda: received_at + .21)
    CollectorGUI._update_quest_input(ui)
    assert 'No fresh data' in ui.quest_input_status_var.set.call_args.args[0]
    for var in ui.quest_hand_vars.values():
        text = var.set.call_args.args[0]
        assert 'DOWN' not in text and 'trigger --' in text


def test_input_panel_distinguishes_missing_hand_stream_error_and_disconnect():
    reader = QuestReader()
    matrix = ' '.join(str(v) for v in np.eye(4).ravel())
    assert reader.ingest(f'l:{matrix}&L,X,leftTrig 0', time.time())
    ui = input_ui(reader)
    CollectorGUI._update_quest_input(ui)
    assert 'OK' in ui.quest_hand_vars['left'].set.call_args.args[0]
    right = ui.quest_hand_vars['right'].set.call_args.args[0]
    assert 'LOST' in right and 'A:--' in right and 'trigger --' in right
    reader._error = RuntimeError('Quest ADB stream disconnected')
    CollectorGUI._update_quest_input(ui)
    assert 'ADB stream disconnected' in ui.quest_input_status_var.set.call_args.args[0]
    assert 'buttons --' in ui.quest_hand_vars['left'].set.call_args.args[0]
    ui.session = None
    CollectorGUI._update_quest_input(ui)
    assert 'disconnected' in ui.quest_input_status_var.set.call_args.args[0]


def test_explicit_save_home_keeps_backup_without_confirmation(tmp_path, monkeypatch):
    ui = gui()
    controller = ui.session.quest_controller
    controller.clutches = {s: SimpleNamespace(active=False) for s in ('left', 'right')}
    controller.snapshot_feedback = Mock(return_value={
        'left': np.r_[np.ones(6) * .1, .5], 'right': np.r_[np.ones(6) * .2, .7]})
    path = tmp_path / 'return_pose.json'
    original = {'qpos': [.3] * 14, 'arm_mode': 'bimanual'}
    path.write_text(json.dumps(original))
    monkeypatch.setattr('bimanual_vla.collection.gui.RETURN_POSE_PATH', path)
    confirm = Mock(side_effect=AssertionError('Unexpected confirmation'))
    monkeypatch.setattr('bimanual_vla.collection.gui.messagebox.askyesno', confirm)
    CollectorGUI._save_collection_reset_pose(ui)
    saved = json.loads(path.read_text())
    np.testing.assert_allclose(saved['qpos'][:6], .1)
    np.testing.assert_allclose(saved['qpos'][7:13], .2)
    assert saved['qpos'][6] == .035
    backup, = tmp_path.glob('return_pose.before_save_*.json')
    assert json.loads(backup.read_text()) == original
    confirm.assert_not_called()
