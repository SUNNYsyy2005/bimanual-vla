import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from server_4090.app import TaskManager
from tests import test_dashboard_checkpoint_management as fixtures
from tests.test_recap_export import make_run
from bimanual_vla.data.recap_export import export_runs


@pytest.mark.parametrize("enabled,annotated,expected", [
    (False, True, "select RECAP"),
    (True, False, "missing RECAP"),
    (True, True, "backend is not staged"),
])
def test_recap_api_rejects_incomplete_requests_before_scheduling(tmp_path, monkeypatch, enabled, annotated, expected):
    app, token, _, _ = fixtures.DashboardCheckpointManagementTest()._make_app(tmp_path)
    root = export_runs([make_run(tmp_path / "run")], tmp_path / "datasets/recap_ds")
    if not annotated:
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        del info["recap"]
        info_path.write_text(json.dumps(info))
    view = app.view_functions["start_train"]
    cells = dict(zip(view.__code__.co_freevars, view.__closure__))
    monkeypatch.setattr(cells["resolve_base_model"], "cell_contents", lambda payload: (tmp_path / "pi05_base", "pi05"))
    monkeypatch.setattr(cells["validate_weight_identity"], "cell_contents", lambda *a: None)
    split = SimpleNamespace(train_episodes=(0,), test_episodes=(1,))
    monkeypatch.setattr(cells["training_episode_split"], "cell_contents", lambda *a, **kw: (split, "fixture"))
    monkeypatch.setitem(view.__globals__, "norm_split_matches", lambda *a, **kw: False)
    with patch.object(TaskManager, "start") as start:
        response = app.test_client().post("/api/tasks/train", headers={"Authorization": "Bearer " + token},
            json=dict(dataset_id="recap_ds", exp_name="recap_test", mode="new", recap_enabled=enabled))
        assert response.status_code == 400, response.get_json()
        assert expected in response.get_json()["error"]
        start.assert_not_called()
