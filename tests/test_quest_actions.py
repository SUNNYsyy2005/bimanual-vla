from types import SimpleNamespace

import numpy as np
import pytest

from bimanual_vla.collection.quest.actions import CollectionButtons, JointResetPlan
from bimanual_vla.collection.session import CollectionSession, SessionState


def sample(t, **buttons):
    return SimpleNamespace(source_time=t, received_monotonic=t, buttons=buttons)


def test_record_edge_and_reset_long_press_require_release_after_disconnect():
    keys = CollectionButtons()
    assert keys.update(sample(0, X=True, A=True)) == []
    assert keys.update(sample(1)) == []
    assert keys.update(sample(2, X=True)) == ['toggle_record']
    assert keys.update(sample(2.1, X=True)) == []
    assert keys.update(sample(3, A=True)) == []
    assert keys.update(sample(3.9, A=True)) == []
    assert keys.update(sample(4.1, A=True)) == ['reset_initial']
    assert keys.update(sample(4.2, A=True)) == []
    assert keys.update(None) == []
    assert keys.update(sample(6, X=True)) == []
    assert keys.update(sample(7)) == []
    assert keys.update(sample(8, X=True)) == ['toggle_record']


def test_reset_cannot_be_armed_while_pose_clutch_is_held_or_by_replayed_frame():
    keys = CollectionButtons()
    keys.update(sample(0))
    keys.update(sample(1, A=True))
    assert keys.update(sample(1, A=True)) == []
    assert keys.update(sample(2, A=True, Y=True)) == []
    assert keys.update(sample(3, A=True)) == []
    assert keys.update(sample(3.5, A=True)) == []
    assert keys.update(sample(4.01, A=True)) == ['reset_initial']


def test_reset_quintic_is_bounded_and_starts_and_ends_at_rest():
    start, end = np.zeros(6), np.array([.2, -.4, .8, 0, -.1, 1.])
    plan = JointResetPlan(start, end)
    dt = .001
    path = np.array([plan.at(t) for t in np.arange(0, plan.duration, dt)])
    assert np.max(abs(np.diff(path, axis=0)/dt)) <= .10001
    assert np.max(abs(np.diff(path, n=2, axis=0)/dt**2)) <= .30001
    assert np.all(path >= np.minimum(start, end)-1e-10)
    assert np.all(path <= np.maximum(start, end)+1e-10)
    np.testing.assert_allclose(plan.at(0), start)
    np.testing.assert_allclose(plan.at(plan.duration), end)
    assert np.max(abs(plan.at(dt)-start)/dt) < .001
    assert np.max(abs(end-plan.at(plan.duration-dt))/dt) < .001


@pytest.mark.parametrize('state', [SessionState.RECORDING, SessionState.REVIEW])
def test_reset_cannot_enter_a_recorded_episode(state):
    session = CollectionSession.__new__(CollectionSession)
    session.state = state
    session.quest_controller = SimpleNamespace(request_reset=lambda _: pytest.fail('must not move'))
    with pytest.raises(RuntimeError, match='stop and save'):
        session.request_quest_reset()
