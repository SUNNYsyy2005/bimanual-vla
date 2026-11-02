import numpy as np
import pytest

from bimanual_vla.data.recap import RecapEpisode


def episode(outcome="success"):
    return RecapEpisode(np.array(["policy"] * 3 + ["transition"] * 2 + ["human"] * 5 + ["policy"] * 3),
                        np.array([0] * 3 + [1] * 2 + [2] * 5 + [3] * 3),
                        np.array([True] * 3 + [False] * 2 + [True, True, False, True, True] + [True] * 3),
                        outcome)


def test_chunks_never_cross_handoff_or_invalid_sample():
    data = episode()
    assert list(data.policy_spans()) == [(0, 3), (5, 7), (8, 10), (10, 13)]
    assert data.chunk_starts(2).tolist() == [0, 1, 5, 8, 10, 11]
    assert data.chunk_starts(2, human_only=True).tolist() == [5, 8]
    assert data.chunk_starts(4).tolist() == []


def test_roundtrip_and_intervention_is_actual_human_control():
    data = RecapEpisode.from_payload(episode().to_payload())
    assert np.flatnonzero(data.intervention).tolist() == [5, 6, 8, 9]
    assert not data.owner.flags.writeable
    payload = data.to_payload()
    payload["intervention_mask"] = np.ones(13, dtype=bool)
    with pytest.raises(ValueError, match="conflicts"):
        RecapEpisode.from_payload(payload)


@pytest.mark.parametrize("outcome,eligible", [("success", True), ("failure", True),
                                               ("aborted", False), ("unlabelled", False)])
def test_aborted_or_unlabelled_is_not_failure(outcome, eligible):
    assert episode(outcome).value_eligible is eligible


def test_bad_annotations_fail_before_training():
    with pytest.raises(ValueError, match="increment"):
        RecapEpisode(np.array(["human", "policy"]), np.array([0, 0]), np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match="boolean"):
        RecapEpisode(np.array(["human"]), np.array([0]), np.array([1]))
    with pytest.raises(ValueError, match="monotonic"):
        RecapEpisode(np.array(["human", "human"]), np.array([1, 0]), np.ones(2, dtype=bool))
