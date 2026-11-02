import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from bimanual_vla.deployment.recap_runtime import RecapRuntime
from bimanual_vla.deployment.servo import ServoLoop


def test_full_open_float32_servo_command_is_valid_at_handoff(tmp_path):
    result, quest, calls, fresh = runtime(tmp_path)
    command = np.tile(np.r_[np.zeros(6), np.float32(.07)], 2)
    result.execution.servo_loop.last_published_command.return_value = command
    result.session.request("takeover")
    assert all(target[6] == 1. for target in result.targets.values())


@pytest.mark.parametrize("opening", [float('nan'), -.001, .071])
def test_invalid_servo_gripper_commands_block_handoff(tmp_path, opening):
    result, quest, calls, fresh = runtime(tmp_path)
    command = np.tile(np.r_[np.zeros(6), opening], 2)
    result.execution.servo_loop.last_published_command.return_value = command
    with pytest.raises(RuntimeError, match="invalid commanded gripper"):
        result.session.request("takeover")
    quest.engage.assert_not_called()


def runtime(tmp_path, *, safe=True):
    calls = []
    command = np.tile(np.r_[np.zeros(6), 0.], 2)
    servo = Mock(last_published_command=Mock(return_value=command))
    controller = SimpleNamespace(pipers={"left": object(), "right": object()},
        servo_loop=servo, robot_enabled={"left", "right"},
        discard_pending_actions=Mock(), close_ik_workers=Mock(),
        close_servo=Mock(side_effect=lambda: calls.append("policy_stopped") or safe))
    quest = Mock()
    quest.controller.status = "ready"
    quest.settle.side_effect = lambda targets: calls.append("measured_settled")
    quest.engage.side_effect = lambda targets: calls.append("human_engaged")
    args = SimpleNamespace(recap_quest_model_dir="model", recap_adb_serial="",
                           gripper_effort=1000, recap_control_dir=tmp_path, recap_session_id="s")
    fresh = SimpleNamespace(_filtered_gripper_opening={}, servo_loop=None)
    with patch("bimanual_vla.deployment.recap_runtime.RecapQuest", return_value=quest):
        result = RecapRuntime(controller, args, Mock(run_dir=tmp_path), Mock(), lambda: fresh)
    return result, quest, calls, fresh


def test_handoff_orders_stop_settle_engage_and_keeps_commanded_grip(tmp_path):
    result, quest, calls, fresh = runtime(tmp_path)
    result.session.request("takeover")
    assert calls == ["policy_stopped", "measured_settled", "human_engaged"]
    assert result.targets["left"][6] == 0
    quest.stop.return_value = {"left": np.r_[np.zeros(6), 0.], "right": np.r_[np.zeros(6), .7]}
    previous = result.session.barrier
    result.session.request("resume")
    assert fresh.resume_gripper_targets == {"left": 0., "right": .7}
    assert fresh._filtered_gripper_opening == fresh.resume_gripper_targets
    assert not result.session.accepts_policy(previous)
    result.close()
    assert result.session.finished and result.session.outcome == "aborted"


def test_stop_failure_never_grants_human_writer(tmp_path):
    result, quest, calls, _ = runtime(tmp_path, safe=False)
    with pytest.raises(RuntimeError, match="stop safely"):
        result.session.request("takeover")
    quest.engage.assert_not_called()
    quest.settle.assert_not_called()
    assert result.session.owner == "hold" and result.session.error


def test_published_command_is_a_copy_not_mutable_servo_state():
    servo = ServoLoop.__new__(ServoLoop)
    servo._condition = threading.Condition()
    servo._last_sample = None
    assert servo.last_published_command() is None
    servo._last_sample = {"command_qpos": np.zeros(14)}
    servo.last_published_command()[:] = 7
    assert not servo._last_sample["command_qpos"].any()


def test_handoff_pose_is_not_resent_after_the_resumed_policy_moved(tmp_path):
    result, quest, calls, fresh = runtime(tmp_path)
    result.targets = {s: np.r_[np.zeros(6), 0.] for s in result.sides}
    result.execution = SimpleNamespace(servo_loop=None)
    result.hold_until_policy()  # bridge until the resumed policy publishes
    assert quest.hold.call_count == 1
    result.execution.servo_loop = Mock()
    result.hold_until_policy()  # policy servo running: handoff pose is stale
    result.execution.servo_loop = None  # e.g. reconnect closed the servo
    result.hold_until_policy()
    assert quest.hold.call_count == 1
    assert result.targets is None
