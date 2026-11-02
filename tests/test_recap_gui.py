import io
import json
import queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bimanual_vla.collection.recap_gui import labelled_runs
from bimanual_vla.collection.gui import CollectorGUI


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
