import time
from unittest.mock import Mock

import pytest

from bimanual_vla.deployment.recap import RecapMailbox, RecapSession, atomic_json


def test_stop_ack_precedes_new_writer_and_old_inference_is_rejected():
    calls = []
    session = RecapSession(**{name: lambda name=name: calls.append((name, session.owner))
                            for name in ("stop_policy", "start_human", "stop_human", "start_policy")})
    old_capture = time.monotonic()
    assert session.accepts_policy(old_capture)
    session.request("takeover")
    assert calls == [("stop_policy", "transition"), ("start_human", "transition")]
    assert not session.accepts_policy(time.monotonic())
    session.request("resume")
    assert calls[-2:] == [("stop_human", "transition"), ("start_policy", "transition")]
    assert not session.accepts_policy(old_capture)
    assert session.accepts_policy(time.monotonic())
    assert session.epoch == 4


def test_failed_stop_never_grants_human_ownership():
    start = Mock()
    session = RecapSession(stop_policy=Mock(side_effect=RuntimeError("did not settle")),
                           start_human=start, stop_human=Mock(), start_policy=Mock())
    with pytest.raises(RuntimeError, match="settle"):
        session.request("takeover")
    start.assert_not_called()
    assert session.owner == "hold" and session.outcome == "aborted"
    assert not session.accepts_policy(time.monotonic())


def test_finishing_stops_current_owner_and_cannot_resume():
    stop = Mock()
    session = RecapSession(stop_policy=stop, start_human=Mock(), stop_human=Mock(), start_policy=Mock())
    session.request("success")
    stop.assert_called_once()
    assert session.finished and session.outcome == "success"
    with pytest.raises(RuntimeError, match="closed"):
        session.request("resume")


def test_mailbox_rejects_other_session_and_duplicate(tmp_path):
    box = RecapMailbox(tmp_path, "current")
    atomic_json(tmp_path / "request.json", dict(session_id="old", id="1", command="takeover"))
    assert box.poll() is None
    atomic_json(tmp_path / "request.json", dict(session_id="current", id="1", command="takeover"))
    assert box.poll() == "takeover"
    assert box.poll() is None
