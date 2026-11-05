import io
import json
import queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bimanual_vla.collection.recap_gui import labelled_runs
from bimanual_vla.collection.gui import CollectorGUI

import pytest


@pytest.mark.parametrize("outcome", ["success", "failure", "aborted"])
def test_labelling_finalized_a_stopped_rollout_preserves_recorded_data(tmp_path, outcome):
    from bimanual_vla.collection.recap_gui import label_completed_rollout
    metadata = {"finished_at": 123, "other": {"preserved": True},
                "recap": {"finished": True, "owner": "hold", "epoch": 4, "error": None,
                          "outcome": "unlabelled", "label_requested": True}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "trajectory.npz").write_bytes(b"unchanged trajectory")
    label_completed_rollout(tmp_path, outcome)
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["recap"]["outcome"] == outcome
    assert saved["recap"]["epoch"] == 4
    assert not saved["recap"]["label_requested"]
    assert saved["other"] == metadata["other"]
    assert (tmp_path / "trajectory.npz").read_bytes() == b"unchanged trajectory"
    with pytest.raises(ValueError, match="not waiting"):
        label_completed_rollout(tmp_path, "failure")


def test_running_or_faulted_rollout_cannot_be_relabelled_as_success(tmp_path):
    from bimanual_vla.collection.recap_gui import label_completed_rollout
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"recap": {"label_requested": True}}))
    with pytest.raises(ValueError, match="not finished"):
        label_completed_rollout(tmp_path, "success")
    path.write_text(json.dumps({"finished_at": 123, "recap": {
        "finished": True, "label_requested": True, "error": "unsafe stop", "outcome": "aborted"}}))
    (tmp_path / "trajectory.npz").touch()
    with pytest.raises(ValueError, match="not waiting"):
        label_completed_rollout(tmp_path, "success")


def test_gui_uses_final_status_after_process_exit_even_when_poll_missed_a_press(tmp_path):
    from bimanual_vla.deployment.recap import atomic_json
    status = dict(session_id="current", finished=True, label_requested=True,
                  recording=str(tmp_path / "run"), error=None)
    atomic_json(tmp_path / "status.json", status)
    ui = SimpleNamespace(recap_directory=SimpleNamespace(name=str(tmp_path)),
                         recap_session_id="current", root=Mock(), recap_status_var=Mock())
    with patch("bimanual_vla.collection.recap_gui.choose_rollout_outcome", return_value="success") as choose:
        CollectorGUI._label_stopped_recap(ui)
        choose.assert_called_once()
        assert choose.call_args.args[1] == status["recording"]
        ui.recap_status_var.set.assert_called_once_with("Rollout saved: success")
        choose.reset_mock()
        ui.recap_session_id = "newer-session"
        CollectorGUI._label_stopped_recap(ui)
        choose.assert_not_called()


def test_inference_finish_labels_before_restoring_idle_vr():
    order = []
    ui = SimpleNamespace(
        inference_process=None, inference_restart_requested=False,
        inference_return_after_stop=False, inference_return_only=False,
        inference_pid_var=Mock(), inference_manual_trigger_path=None,
        inference_status_var=Mock(), _set_connection_config_enabled=Mock(),
        _update_mode_controls=Mock(), root=Mock(), _idle_quest_wanted=True,
        _start_idle_quest=Mock())
    ui.root.after.side_effect = lambda *args: order.append("restore_scheduled")
    with patch.object(CollectorGUI, "_label_stopped_recap", side_effect=lambda self: order.append("dialog_closed")):
        CollectorGUI._finish_inference(ui, 0, False)
    assert order == ["dialog_closed", "restore_scheduled"]


@pytest.mark.parametrize("code,return_only", [(2, False), (0, True)])
def test_failed_exit_and_home_only_process_never_open_outcome_dialog(code, return_only):
    ui = SimpleNamespace(
        inference_process=None, inference_restart_requested=False,
        inference_return_after_stop=False, inference_return_only=return_only,
        inference_pid_var=Mock(), inference_manual_trigger_path=None,
        inference_status_var=Mock(), _set_connection_config_enabled=Mock(),
        _update_mode_controls=Mock(), root=Mock(), _idle_quest_wanted=False)
    with patch.object(CollectorGUI, "_label_stopped_recap") as label:
        CollectorGUI._finish_inference(ui, code, False)
    label.assert_not_called()


def test_only_labelled_finalized_runs_are_offered(tmp_path):
    for name, outcome in (("ok", "success"), ("bad", "failure"), ("abort", "aborted"), ("live", "success")):
        path = tmp_path / name
        path.mkdir()
        (path / "metadata.json").write_text(json.dumps({"recap": {"outcome": outcome}}))
        if name != "live":
            (path / "trajectory.npz").touch()
    assert {path.name for path, _ in labelled_runs(tmp_path)} == {"ok", "bad"}


def test_export_worker_reports_prepared_path_only_after_success(tmp_path):
    for code in (0, 2):
        gui = SimpleNamespace(status_var=Mock(), messages=queue.Queue(),
                              _update_start_button=Mock(), _update_dataset_action_buttons=Mock())
        process = Mock(stdout=io.StringIO("conversion log\n"))
        process.wait.return_value = code
        with patch("bimanual_vla.collection.gui.subprocess.Popen", return_value=process):
            CollectorGUI._launch_dataset_command(gui, ["conda-python", "export"], "RECAP export", {},
                                                  prepared_path=tmp_path)
            gui.dataset_task_thread.join(timeout=2)
        assert not gui.dataset_task_thread.is_alive()
        events = list(gui.messages.queue)
        assert any("PREPARED_LEROBOT_PATH=" in str(event) for event in events) == (code == 0)
        assert events[-1] == ("dataset_done", "RECAP export", code, None)


def test_idle_recap_uses_adb_scan_without_deployment_status():
    import time
    from bimanual_vla.collection.quest.devices import ADBDevice

    ui = SimpleNamespace(
        _quest_scan_busy=False, quest_adb_serial_var=Mock(get=lambda: "quest"),
        quest_device_status_var=Mock(), piper=None,
        _inference_running=lambda: False, _replay_running=lambda: False,
        recap_enabled_var=Mock(get=lambda: True), _refresh_quest_devices=Mock(),
        _set_recap_device_status=Mock(), _recap_adb_scan_at=time.monotonic(),
        # A failed/old inference can leave a directory with no status.json.
        recap_directory=SimpleNamespace(name="/nonexistent/old-recap"),
    )
    CollectorGUI._finish_quest_devices(ui, [ADBDevice("quest", "device", "Quest 3")], None)
    CollectorGUI._poll_recap(ui)
    status = ui._set_recap_device_status.call_args.args[0]
    assert status["vr_connection"] == "connected (ADB)"
    assert "not monitored" in status["controller_tracking"]["left"]

    CollectorGUI._finish_quest_devices(ui, [], None)
    CollectorGUI._poll_recap(ui)
    assert ui._set_recap_device_status.call_args.args[0]["vr_connection"] == "disconnected"


def test_idle_recap_refreshes_expired_adb_status():
    ui = SimpleNamespace(
        _inference_running=lambda: False, quest_adb_serial_var=Mock(get=lambda: "quest"),
        recap_enabled_var=Mock(get=lambda: True), _quest_scan_busy=False,
        _recap_adb_status=("quest", 0, {"vr_connection": "connected (ADB)"}),
        _set_recap_device_status=Mock(), _refresh_quest_devices=Mock(),
    )
    CollectorGUI._poll_recap(ui)
    assert ui._set_recap_device_status.call_args.args[0]["vr_connection"] == "checking ADB"
    ui._refresh_quest_devices.assert_called_once()
