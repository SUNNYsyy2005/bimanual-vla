import numpy as np
import pytest
import json
from pathlib import Path
from scipy.spatial.transform import Rotation

from bimanual_vla.data.action_conventions import (
    chunk_origin_deltas_to_absolute_eef_targets, matrix_to_rotation6d,
)
from bimanual_vla.deployment.client import validate_policy_metadata
from bimanual_vla.deployment.full_eef_compat import (
    EEFBridge, FullEEFPolicy, compatibility_metadata, execution_ready_from_config,
)


def state():
    return np.r_[
        [.2, .3, .4], matrix_to_rotation6d(Rotation.from_euler('xyz', [.3, -.4, 3.13]).as_matrix()), [.5],
        [.1, -.3, .2], matrix_to_rotation6d(Rotation.from_euler('xyz', [-.5, .2, -3.13]).as_matrix()), [1.],
    ]


def test_bridge_roundtrip_with_arm_permutation_base_and_tool_calibration():
    base = np.tile(np.eye(4), (2, 1, 1))
    tool = base.copy()
    base[0, :3, :3] = Rotation.from_euler('z', .6).as_matrix()
    base[1, :3, 3] = [.04, -.05, .06]
    tool[0, :3, 3] = [.03, 0, 0]
    tool[1, :3, :3] = Rotation.from_euler('y', -.4).as_matrix()
    bridge = EEFBridge(base, tool, ('right', 'left'))
    model = bridge.to_model_state(state())
    assert model[6] == pytest.approx(.07)
    assert model[13] == pytest.approx(.035)
    restored = bridge.to_client_targets(model[None])[0]
    np.testing.assert_allclose(restored, state(), atol=2e-7)


def test_policy_converts_absolute_targets_without_euler_wrap_jump_and_keeps_prompt():
    bridge = EEFBridge()
    model = bridge.to_model_state(state())
    targets = np.tile(model, (50, 1))
    targets[:, 0] += .01
    targets[:, 5] -= 2 * np.pi  # same physical rotation across the Euler branch
    seen = []
    class Policy:
        def infer(self, observation):
            seen.append(observation)
            return {'actions': targets}
    policy = FullEEFPolicy(Policy())
    obs = dict(state=state(), images={k: np.zeros((3, 224, 224), np.uint8)
                                    for k in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
               prompt='custom task', client_metadata={'allow_execution': False, 'rtc': {'session_id': 'one'}})
    result = policy.infer(obs)
    restored = chunk_origin_deltas_to_absolute_eef_targets(state(), result['actions'], arm_count=2)
    expected = bridge.to_client_targets(targets)
    np.testing.assert_allclose(restored, expected, atol=3e-7)
    np.testing.assert_allclose(result['actions'][:, 3:6], 0, atol=3e-7)
    policy.infer({k: v for k, v in obs.items() if k != 'prompt'})
    assert seen[-1]['prompt'] == 'custom task'
    with pytest.raises(ValueError, match='shadow-only'):
        policy.infer(dict(obs, client_metadata={'allow_execution': True}))
    ready_policy = FullEEFPolicy(Policy(), execution_ready=True)
    ready_result = ready_policy.infer(dict(obs, client_metadata={'allow_execution': True}))
    np.testing.assert_allclose(ready_result['actions'], result['actions'])


def test_bridge_rejects_invalid_state_calibration_and_model_gripper():
    bridge = EEFBridge()
    with pytest.raises(ValueError, match='finite'):
        bridge.to_model_state(np.full(20, np.nan))
    bad = np.tile(np.eye(4), (2, 1, 1))
    bad[0, 0, 0] = 2
    with pytest.raises(ValueError, match='rigid transforms'):
        EEFBridge(bad)
    actions = np.tile(bridge.to_model_state(state()), (50, 1))
    actions[0, 6] = .071
    with pytest.raises(ValueError, match='gripper metres'):
        bridge.to_client_targets(actions)


def test_shadow_handshake_still_blocks_execution_and_checks_contract():
    metadata = compatibility_metadata('test-checkpoint')
    protocol = validate_policy_metadata(metadata, 'both', 'bimanual', allow_execution=False)
    assert protocol.state_dim == 20 and protocol.action_dim == 14 and not protocol.rtc_supported
    with pytest.raises(RuntimeError, match='deployment_ready=false'):
        validate_policy_metadata(metadata, 'both', 'bimanual', allow_execution=True)
    with pytest.raises(RuntimeError, match='state_dim'):
        validate_policy_metadata(dict(metadata, state_dim=14), 'both', 'bimanual', allow_execution=False)


def test_confirmed_sdk_config_admits_execution_but_incomplete_or_changed_frames_do_not():
    path = Path(__file__).resolve().parents[1] / 'scripts/models/full_eef_bridge.json'
    settings = json.loads(path.read_text())
    assert execution_ready_from_config(settings)
    metadata = compatibility_metadata('source', execution_ready=True)
    assert validate_policy_metadata(metadata, 'both', 'bimanual', allow_execution=True).state_dim == 20
    assert not execution_ready_from_config(dict(settings, validated=False))
    with pytest.raises(ValueError, match='confirmed SDK'):
        execution_ready_from_config(dict(settings, confirmed_sdk_contract={}))
    changed = json.loads(path.read_text())
    changed['transforms']['client_tool_from_model_tool'][0][0][3] = .13
    with pytest.raises(ValueError, match='identity'):
        execution_ready_from_config(changed)
    with pytest.raises(ValueError, match='feedback/FK'):
        execution_ready_from_config(dict(settings, kinematics_validation={}))
