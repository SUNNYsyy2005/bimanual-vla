import numpy as np
import pytest

from bimanual_vla.deployment.client import (
    ExecutionBlocked, ExecutionController, GRIPPER_OPENING_METRES,
    _opening_fraction, _gripper_value_from_opening_fraction,
    build_checked_joint_target,
)


@pytest.mark.parametrize('opening', [0.07285, 0.08034, 0.0919])
def test_metre_target_is_preserved_through_conversion_and_safety(opening):
    ratio = _opening_fraction(opening, semantics=GRIPPER_OPENING_METRES, tolerance=0.02)
    assert _gripper_value_from_opening_fraction(ratio, semantics=GRIPPER_OPENING_METRES) == pytest.approx(opening)
    assert ExecutionController._opening_to_wire(ratio, GRIPPER_OPENING_METRES) == pytest.approx(opening)
    pose = np.array([0, 1, -1, 0, 0, 0, opening])
    _, metres = build_checked_joint_target(pose, pose, max_joint_step_rad=0.1,
        gripper_semantics=GRIPPER_OPENING_METRES)
    assert metres == pytest.approx(opening)


def test_metre_target_above_execution_limit_is_rejected():
    with pytest.raises(ExecutionBlocked, match='gripper target'):
        _opening_fraction(0.095, semantics=GRIPPER_OPENING_METRES, tolerance=0.02)
