"""Offline checks of the bounded diagnostic; no hardware or CAN is opened."""
import math

import numpy as np
import pytest

from analysis.collection_check_20261025.probe_negative_j2 import (
    check_anchor, target_at, validate_feedback,
)
from bimanual_vla.collection.replay import ReplayBlocked
from bimanual_vla.deployment.client import JOINT_LIMITS_RAD


def anchor():
    return np.r_[np.deg2rad([90., -1.834, -8., 3., 22., -9.]), .004]


def test_negative_probe_stays_between_observed_pose_and_point_three_degrees_inward():
    q = anchor()
    times = np.linspace(0, 2, 201)
    outgoing = np.stack([target_at(q, t) for t in times])
    returning = np.stack([target_at(q, t, True) for t in times])
    assert np.allclose(outgoing[-1], returning[0])
    assert np.allclose(returning[-1], q)
    assert np.all(np.diff(outgoing[:, 1]) >= 0)
    assert outgoing[-1, 1] - q[1] == pytest.approx(math.radians(.3))
    assert np.all(outgoing[:, 1] < 0)
    assert np.allclose(outgoing[:, [0, 2, 3, 4, 5, 6]], q[[0, 2, 3, 4, 5, 6]])
    assert np.max(abs(np.diff(outgoing[:, 1]) / .01)) <= math.radians(.282)


@pytest.mark.parametrize('j2', [-10., -.4, 0., 10.])
def test_diagnostic_refuses_other_negative_or_positive_targets(j2):
    q = anchor()
    q[1] = math.radians(j2)
    with pytest.raises(ReplayBlocked, match='only allows'):
        check_anchor(q, JOINT_LIMITS_RAD)


def test_only_j2_can_be_below_nominal_range_and_other_limits_are_unchanged():
    q = anchor()
    check_anchor(q, JOINT_LIMITS_RAD)
    q[2] = .2
    with pytest.raises(ReplayBlocked, match='Another joint'):
        check_anchor(q, JOINT_LIMITS_RAD)


@pytest.mark.parametrize('index,delta', [(0, math.radians(1)), (1, math.radians(-.2)),
                                        (1, math.radians(2)), (6, .01)])
def test_departure_from_small_envelope_aborts(index, delta):
    q = anchor()
    measured = q.copy()
    measured[index] += delta
    with pytest.raises(ReplayBlocked, match='envelope'):
        validate_feedback(measured, q, q)


def test_small_negative_tracking_is_accepted():
    q = anchor()
    target = target_at(q, 1.)
    validate_feedback(target, q, target)
