"""Idle inference-page teleoperation, using no ADB, CAN or motor commands."""
import io
import queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from bimanual_vla.collection.gui import CollectorGUI
from bimanual_vla.deployment.idle_quest import IdleButtons, IdleQuestHardware, IdleQuestSession


def sample(sequence, **buttons):
    return SimpleNamespace(source_time=sequence, poses={"l": np.eye(4), "r": np.eye(4)},
                           buttons={"A": False, "B": False, "Y": False, **buttons})


def test_idle_a_requires_release_and_never_uses_collection_x_shortcut():
    keys = IdleButtons()
    assert keys.update(sample(1, A=True)) == []  # held during connection
    assert keys.update(sample(2, X=True)) == []
    assert keys.update(sample(3, A=True)) == ["return_saved_home"]
    assert keys.update(sample(3, A=True)) == []  # same snapshot
    assert keys.update(sample(4, A=True)) == []  # long press
    keys.update(None)
    assert keys.update(sample(5, A=True)) == []  # tracking loss disarms
    assert keys.update(sample(6)) == []
    assert keys.update(sample(7, A=True, B=True)) == ["return_saved_home"]  # same as GUI home button
    assert keys.update(sample(8, A=True)) == []  # release A again
    keys.update(sample(9))
    assert keys.update(sample(10, A=True)) == ["return_saved_home"]


def test_enable_only_adopts_the_arm_whose_clutch_was_pressed():
    hw = IdleQuestHardware({"left": Mock(), "right": Mock()})
    measured = {"left": np.zeros(7), "right": np.zeros(7)}
    hw.read = Mock(return_value=measured)
    hw.prepare_for_reset = Mock(return_value=measured)
    hw.send = Mock()
    hw.enable("right", measured["right"])
    hw.prepare_for_reset.assert_called_once_with(measured, speed_pct=80, sides=("right",))
    hw.send.assert_called_once_with("right", measured["right"])
    assert not hw.arms["left"].mock_calls


def make_session():
    return IdleQuestSession(arms={"left": "can0", "right": "can1"}, cameras={},
                            arm_mode="bimanual", arm_side="both", model_dir="model", serial="quest")


def test_session_starts_without_collection_recording_and_releases_writer_before_ports():
    events = []
    arms = [Mock(), Mock()]
    for index, arm in enumerate(arms):
        arm.DisconnectPort.side_effect = lambda index=index: events.append(f"disconnect{index}")
    controller = Mock()
    controller.close.side_effect = lambda: events.append("writer_closed")
    session = make_session()
    with patch("bimanual_vla.deployment.idle_quest.check_devices", return_value=None), \
            patch("bimanual_vla.deployment.client.connect_piper", side_effect=arms), \
            patch("bimanual_vla.deployment.idle_quest.QuestController", return_value=controller):
        session.start()
    assert isinstance(controller.collection_buttons, IdleButtons)
    controller.enable_collection_actions.assert_not_called()
    controller.begin_episode.assert_not_called()
    for arm in arms:
        arm.EnablePiper.assert_not_called()
        arm.JointCtrl.assert_not_called()
    session.close()
    assert events == ["writer_closed", "disconnect0", "disconnect1"]
    assert session.arms == {}


def test_partial_start_failure_is_cleaned_and_duplicate_can_is_rejected_before_open():
    session = make_session()
    arm = Mock()
    with patch("bimanual_vla.deployment.idle_quest.check_devices", return_value=None), \
            patch("bimanual_vla.deployment.client.connect_piper",
                  side_effect=[arm, RuntimeError("right feedback missing")]):
        with pytest.raises(RuntimeError, match="feedback missing"):
            session.start()
    session.close()
    arm.DisconnectPort.assert_called_once()
    session = make_session()
    session.request["arms"] = {"left": "can0", "right": "can0"}
    with patch("bimanual_vla.deployment.client.connect_piper") as connect:
        with pytest.raises(ValueError, match="same interface"):
            session.start()
        connect.assert_not_called()


def test_writer_close_failure_keeps_can_handles_and_prevents_inference_launch():
    session = make_session()
    arm = Mock()
    session.arms = {"left": arm}
    session.controller = Mock()
    session.controller.close.side_effect = RuntimeError("writer did not stop")
    ui = SimpleNamespace(_idle_quest=session, _idle_quest_connecting=False,
                         _idle_quest_wanted=True, recap_status_var=Mock(),
                         inference_return_only=True)
    ui._stop_idle_quest = lambda: CollectorGUI._stop_idle_quest(ui)
    with patch("bimanual_vla.collection.gui.messagebox.showerror") as error, \
            patch("bimanual_vla.collection.gui.subprocess.Popen") as launch:
        CollectorGUI._launch_inference_process(ui, ["client"], "policy")
    launch.assert_not_called()
    arm.DisconnectPort.assert_not_called()
    assert ui._idle_quest is session
    assert not ui._idle_quest_wanted
    assert not ui.inference_return_only
    error.assert_called_once()


def test_inference_spawn_happens_only_after_idle_session_is_closed():
    events = []
    ui = SimpleNamespace(
        _idle_quest=Mock(), _idle_quest_connecting=False,
        inference_log_widget=None, _append_inference_log=Mock(),
        inference_status_var=Mock(), inference_pid_var=Mock(), messages=queue.Queue(),
        _set_connection_config_enabled=Mock(), _update_mode_controls=Mock())
    ui._stop_idle_quest = lambda: events.append("idle_closed") or True
    process = Mock(pid=7, stdout=io.StringIO(""))
    process.wait.return_value = 0
    def launch(*args, **kwargs):
        events.append("spawn")
        return process
    with patch("bimanual_vla.collection.gui.subprocess.Popen", side_effect=launch):
        CollectorGUI._launch_inference_process(ui, ["client"], "policy")
        ui.inference_process_thread.join(timeout=2)
    assert events == ["idle_closed", "spawn"]


def test_idle_a_calls_the_existing_home_button_action_once():
    controller = Mock(ui_actions=queue.Queue())
    controller.ui_actions.put("return_saved_home")
    ui = SimpleNamespace(_idle_quest=Mock(controller=controller),
                         _idle_quest_connecting=False, _idle_quest_fault=None,
                         recap_buttons={}, _set_recap_device_status=Mock(),
                         recap_status_var=Mock(), return_inference_to_initial=Mock())
    CollectorGUI._poll_idle_quest(ui)
    CollectorGUI._poll_idle_quest(ui)
    ui.return_inference_to_initial.assert_called_once_with()


@pytest.mark.parametrize("code,restore", [(0, True), (2, False)])
def test_restore_idle_vr_only_after_normal_inference_or_home_completion(code, restore):
    ui = SimpleNamespace(
        inference_process=None, inference_restart_requested=False,
        inference_return_after_stop=False, inference_return_only=True,
        inference_pid_var=Mock(), inference_manual_trigger_path=None,
        inference_status_var=Mock(), _set_connection_config_enabled=Mock(),
        _update_mode_controls=Mock(), root=Mock(), _idle_quest_wanted=True,
        _start_idle_quest=Mock())
    CollectorGUI._finish_inference(ui, code, True)
    if restore:
        ui.root.after.assert_called_once_with(250, ui._start_idle_quest)
    else:
        ui.root.after.assert_not_called()
        assert not ui._idle_quest_wanted


def test_return_then_restart_does_not_restore_an_idle_writer_between_processes():
    ui = SimpleNamespace(
        inference_process=None, inference_restart_requested=True,
        inference_return_after_stop=True, inference_return_only=False,
        inference_pid_var=Mock(), inference_manual_trigger_path=None,
        inference_status_var=Mock(), _set_connection_config_enabled=Mock(),
        _update_mode_controls=Mock(), root=Mock(), _idle_quest_wanted=True,
        _start_idle_quest=Mock(), return_inference_to_initial=Mock(), start_inference=Mock())
    CollectorGUI._finish_inference(ui, 0, True)
    ui.root.after.assert_called_once_with(250, ui.return_inference_to_initial)
