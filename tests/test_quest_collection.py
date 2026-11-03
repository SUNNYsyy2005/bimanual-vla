"""VR behavior and motion admission tests; never connect ADB or CAN."""
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bimanual_vla.collection.quest.control import (
    Clutch, GripPickup, InvalidCommand, QuestController, QuestHardware, validate_command,
)
from bimanual_vla.collection.quest.ik import IKRejected, QuestIK
from bimanual_vla.collection.quest.input import QuestReader, Snapshot, adjusted_pose, parse_payload


def payload(matrix=None, buttons="L,Y,leftTrig 0.25"):
    matrix = np.eye(4) if matrix is None else matrix
    return "l:" + " ".join(str(v) for v in matrix.ravel()) + "&" + buttons


def test_official_protocol_requires_valid_tracking_and_matching_buttons():
    poses, buttons = parse_payload(payload())
    np.testing.assert_array_equal(poses["l"], np.eye(4))
    assert buttons["Y"] and not buttons["X"] and buttons["leftTrig"] == (.25,)
    for bad in (payload(np.full((4, 4), np.nan)), "l:1 2&L,leftTrig 0",
                payload(buttons="L,Y"), payload(buttons="L,leftTrig 1.5"),
                payload(np.diag([1, 1, -1, 1]))):
        with pytest.raises(ValueError):
            parse_payload(bad)


def test_reader_uses_real_reception_time_rejects_backlog_and_duplicate():
    reader = QuestReader(max_age_s=.2)
    now = time.time()
    assert reader.ingest(payload(), now, now=now)
    first = reader.snapshot()
    assert not reader.ingest(payload(), now, now=now)
    assert not reader.ingest(payload(), now - 1, now=now)
    assert not reader.ingest(payload(), now + 1, now=now)
    assert reader.snapshot().sequence == first.sequence
    first.poses["l"][0, 3] = 50
    assert reader.snapshot().poses["l"][0, 3] == 0
    with patch("bimanual_vla.collection.quest.input.time.monotonic", return_value=first.received_monotonic + .21):
        assert reader.snapshot() is None
    reader._error = RuntimeError("disconnected")
    with pytest.raises(RuntimeError, match="disconnected"):
        reader.snapshot()


def test_basis_matches_official_rpy_transform():
    basis = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.]])
    r = np.eye(4)
    r[:3, :3] = Rotation.from_euler("xyz", [-np.pi, 0, -np.pi/2]).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [.3, .1, -.2]).as_matrix()
    np.testing.assert_allclose(adjusted_pose(pose), basis @ pose @ r, atol=1e-12)


@pytest.mark.parametrize("opening", [0, .17, .5, .8, 1])
def test_gripper_pickup_then_full_range_strictly_linear(opening):
    grip = GripPickup(opening, 0.)
    # Cross the opening before following the whole trigger range.
    assert grip.update(0) == 1
    assert grip.picked
    for trigger in np.linspace(0, 1, 101):
        assert grip.update(trigger) == pytest.approx(1 - trigger)


def test_gripper_does_not_jump_before_pickup():
    grip = GripPickup(.7, .1)
    assert grip.update(.8) == .7
    assert grip.update(.5) == .7
    assert not grip.picked
    assert grip.update(.2) == pytest.approx(.8)
    assert grip.update(.1) == pytest.approx(.9)


class Solver:
    def __init__(self):
        self.seed = None
        self.output = np.zeros(6)
        self.on_solve = None
    def reset(self, q):
        self.seed = q.copy()
    def fk(self, q):
        result = np.eye(4)
        result[:3, 3] = [.24, -.05, .31]
        result[:3, :3] = Rotation.from_euler("xyz", [.1, .2, .3]).as_matrix()
        return result
    def solve(self, target, opening):
        if self.on_solve:
            self.on_solve()
        if isinstance(self.output, Exception):
            raise self.output
        return self.output.copy()
    def accept(self, q):
        self.seed = q.copy()


def test_clutch_first_target_exactly_measured_and_reclutch_reanchors():
    clutch, solver = Clutch(), Solver()
    measured = np.array([.1, .2, -.3, .1, .1, .2, .73])
    hand = np.eye(4)
    assert clutch.target(True, hand, measured, solver, .9) is None
    assert clutch.target(False, hand, measured, solver, .9) is None
    pose, command, first = clutch.target(True, hand, measured, solver, .9)
    np.testing.assert_array_equal(command, measured)
    np.testing.assert_array_equal(pose, solver.fk(measured[:6]))
    assert first
    clutch.target(False, hand, measured, solver, .9)
    hand[0, 3] = 5
    pose2, command2, first2 = clutch.target(True, hand, measured, solver, .9)
    np.testing.assert_array_equal(pose2, pose)
    np.testing.assert_array_equal(command2, measured)
    assert first2


def test_relative_target_uses_anchored_se3_and_rejects_recenter():
    clutch, solver = Clutch(), Solver()
    measured = np.zeros(7)
    hand = np.eye(4)
    clutch.target(False, hand, measured, solver, 1)
    clutch.target(True, hand, measured, solver, 1)
    hand[0, 3] = .01
    pose, _, first = clutch.target(True, hand, measured, solver, 1)
    np.testing.assert_allclose(pose, solver.fk(measured[:6]) @ hand)
    assert not first
    hand[0, 3] = 2
    with pytest.raises(IKRejected, match="recenter"):
        clutch.target(True, hand, measured, solver, 1)


class Hardware:
    def __init__(self, sides):
        self.measured = {side: np.r_[np.zeros(6), .7] for side in sides}
        self.limits = {side: np.tile([-1., 1.], (6, 1)) for side in sides}
        self.enabled = set()
        self.sent = []
    def read(self):
        return {k: v.copy() for k, v in self.measured.items()}
    def require_ready(self, sides=None):
        return self.read()
    def enable(self, side, target):
        self.enabled.add(side)
    def send(self, side, target):
        validate_command(side, target, self.limits[side])
        self.sent.append((side, target.copy()))
    def set_speed(self, side, speed_pct):
        self.speeds = getattr(self, "speeds", []) + [(side, speed_pct, len(self.sent))]


class Reader:
    max_age_s = .2
    def __init__(self):
        self.sample = None
    def snapshot(self):
        return self.sample
    def update(self, pressed=False):
        self.sample = Snapshot(1, time.monotonic(), time.time(),
                               {"l": np.eye(4), "r": np.eye(4)},
                               {"Y": pressed, "B": pressed, "leftTrig": (.3,), "rightTrig": (.3,)})


def controller(sides=("left", "right")):
    reader, hardware = Reader(), Hardware(sides)
    runtime = QuestController({s: object() for s in sides}, "unused", reader=reader, hardware=hardware)
    runtime.solvers = {side: Solver() for side in sides}
    reader.update(False)
    runtime.step()
    assert not hardware.sent and not hardware.enabled
    reader.update(True)
    runtime.step()
    hardware.sent.clear()
    return runtime, reader, hardware


def test_empty_vr_frame_holds_and_reports_missing_hands_then_recovers_status():
    runtime, reader, hw = controller()
    reader.update(False)
    reader.sample.poses.clear()
    reader.sample.buttons.clear()
    runtime.step()
    assert 'Controller tracking lost' in runtime.status
    assert all(not c.ready and not c.active for c in runtime.clutches.values())
    assert all(np.array_equal(q, hw.measured[s]) for s, q in hw.sent)
    reader.update(True)
    runtime.step()
    assert 'Release Y/B first' in runtime.status
    assert all(not c.active for c in runtime.clutches.values())
    reader.update(False)
    runtime.step()
    assert 'VR ready' in runtime.status
    assert all(c.ready for c in runtime.clutches.values())


@pytest.mark.parametrize("bad", [np.full(6, np.nan), np.ones(6) * .5,
                                IKRejected("self collision"), IKRejected("no convergence")])
def test_reject_both_arms_before_any_motion_when_one_ik_fails(bad):
    runtime, reader, hw = controller()
    runtime.solvers["left"].output = np.ones(6) * .05
    runtime.solvers["right"].output = bad
    runtime.begin_episode()
    reader.update(True)
    runtime.step()
    assert hw.sent
    assert all(np.array_equal(target, hw.measured[side]) for side, target in hw.sent)
    assert all(not c.active for c in runtime.clutches.values())
    assert "motion_rejected" in runtime.end_episode()
    hw.sent.clear()
    reader.update(True)
    runtime.step()
    assert all(np.array_equal(target, hw.measured[side]) for side, target in hw.sent)


def test_driver_limit_admission_is_atomic_across_arms():
    runtime, reader, hw = controller()
    runtime.solvers["left"].output[:] = .05
    runtime.solvers["right"].output[:] = .05
    hw.limits["right"][4, 1] = .01
    reader.update(True)
    runtime.step()
    assert all(np.array_equal(target, hw.measured[side]) for side, target in hw.sent)


def test_collection_reset_uses_one_writer_preserves_grip_and_requires_reanchor(monkeypatch):
    runtime, reader, hw = controller()
    runtime.enable_collection_actions()
    runtime.initial_joints = {s: np.ones(6)*.05 for s in hw.measured}
    for solver in runtime.solvers.values():
        solver.collides = lambda q, opening: False
    with pytest.raises(RuntimeError, match='release Y/B'):
        runtime.request_reset()
    reader.update(False)
    clock = [100.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    runtime.request_reset()
    for _ in range(180):
        runtime.step()
        for s, target in hw.sent:
            hw.measured[s] = target.copy()
        hw.sent.clear()
        clock[0] += .02
        if not runtime.resetting:
            break
    assert not runtime.resetting
    assert runtime.ui_actions.get_nowait() == 'reset_done'
    for side, measured in hw.measured.items():
        np.testing.assert_allclose(measured[:6], .05)
        assert measured[6] == .7
        assert not runtime.clutches[side].ready
        assert not runtime.clutches[side].active


def test_gui_reset_target_is_independent_of_connection_pose():
    runtime, reader, hw = controller()
    runtime.enable_collection_actions()
    runtime.initial_joints = {s: np.zeros(6) for s in hw.measured}
    reader.update(False)
    saved = {s: np.ones(6)*.1 for s in hw.measured}
    runtime.request_reset(saved)
    for s, target in runtime._requests.get_nowait().items():
        np.testing.assert_allclose(target, .1)
        np.testing.assert_allclose(runtime.initial_joints[s], 0)


def test_reset_clips_out_of_bounds_target_like_inference_without_mutating_home():
    runtime, reader, hw = controller()
    runtime.enable_collection_actions()
    runtime.initial_joints = {s: np.ones(6)*2 for s in hw.measured}
    reader.update(False)
    runtime.request_reset()
    assert runtime.resetting and not hw.sent
    targets = runtime._requests.get_nowait()
    for s, q in targets.items():
        np.testing.assert_array_equal(q, hw.limits[s][:, 1])
        np.testing.assert_array_equal(runtime.initial_joints[s], np.ones(6)*2)


@pytest.mark.parametrize('bad', [np.full(6, np.nan), np.full(6, np.inf), np.zeros(5)])
def test_reset_still_rejects_malformed_target_before_any_send(bad):
    runtime, reader, hw = controller()
    runtime.enable_collection_actions()
    runtime.initial_joints = {s: bad.copy() for s in hw.measured}
    reader.update(False)
    with pytest.raises(ValueError, match='six finite joints'):
        runtime.request_reset()
    assert not runtime.resetting and not hw.sent


def test_saved_home_negative_j2_reaches_clipped_boundary_with_same_writer(monkeypatch):
    runtime, reader, hw = controller()
    runtime.enable_collection_actions()
    runtime.initial_joints = {s: np.zeros(6) for s in hw.measured}
    for side in hw.measured:
        hw.limits[side][1, 0] = 0
        hw.measured[side][1] = .1
        runtime.solvers[side].collides = lambda q, opening: False
    saved = {s: np.array([0., -.034, 0., 0., 0., 0.]) for s in hw.measured}
    reader.update(False)
    clock = [100.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    runtime.request_reset(saved)
    for _ in range(180):
        runtime.step()
        for side, q in hw.sent:
            assert q[1] >= 0
            hw.measured[side] = q.copy()
        hw.sent.clear()
        clock[0] += .02
        if not runtime.resetting:
            break
    assert not runtime.resetting
    for side in hw.measured:
        np.testing.assert_allclose(hw.measured[side][:6], 0, atol=1e-12)
        assert hw.measured[side][6] == .7
        assert saved[side][1] == -.034


@pytest.mark.parametrize('status_change,driver_change', [
    ({'ctrl_mode': 0}, {}), ({'mode_feed': 0}, {}),
    ({'feedback_fresh': False}, {}), ({}, {'ready': False}),
])
def test_adoption_of_unready_arm_never_enables_or_writes(status_change, driver_change):
    arm = Mock()
    hw = QuestHardware({'left': arm})
    hw.read = Mock(return_value={'left': np.zeros(7)})
    status = dict(feedback_fresh=True, arm_status=0, err_code=0, ctrl_mode=1, mode_feed=1)
    status.update(status_change)
    driver = dict(ready=True)
    driver.update(driver_change)
    with patch('bimanual_vla.deployment.client.arm_status_dict', return_value=status), \
            patch('bimanual_vla.deployment.client.driver_enable_status_dict', return_value=driver):
        with pytest.raises(RuntimeError, match='already be enabled'):
            hw.enable('left', np.ones(7))
    assert not arm.mock_calls
    assert not hw.enabled


def reset_runtime(monkeypatch):
    runtime, reader, hw = controller()
    runtime.enable_collection_actions()
    runtime.initial_joints = {s: np.ones(6)*.2 for s in hw.measured}
    for solver in runtime.solvers.values():
        solver.collides = lambda q, opening: False
    reader.update(False)
    clock = [100.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    runtime.request_reset()
    return runtime, hw, clock


def test_reset_checks_both_arms_before_first_write(monkeypatch):
    runtime, hw, _ = reset_runtime(monkeypatch)
    hw.require_ready = Mock(side_effect=RuntimeError('right not ready'))
    with pytest.raises(RuntimeError, match='right not ready'):
        runtime.step()
    hw.require_ready.assert_called_once_with()
    assert not hw.sent


def moving_reset(monkeypatch):
    runtime, hw, clock = reset_runtime(monkeypatch)
    runtime.step()
    for _ in range(14):
        clock[0] += .02
        runtime.step()
    assert runtime._reset['phase'] == 'moving'
    hw.sent.clear()
    return runtime, hw, clock


@pytest.mark.parametrize('failure,expected', [
    ('speed', 'measured joint speed'), ('stall', 'writer stalled'),
    ('lag', 'tracking error'), ('step', 'per-cycle speed'),
])
def test_reset_aborts_before_next_target_on_motion_fault(monkeypatch, failure, expected):
    runtime, hw, clock = moving_reset(monkeypatch)
    clock[0] += .02
    if failure == 'speed':
        hw.measured['right'][0] += .01
    elif failure == 'stall':
        clock[0] += .2
    elif failure == 'lag':
        hw.measured['right'][0] += .06
        runtime._reset['previous']['right'][0] += .06
    else:
        runtime._reset['plans']['right'].at = lambda _: np.ones(6)*.1
    before = {s: q.copy() for s, q in runtime._holds.items()}
    with pytest.raises(RuntimeError, match=expected):
        runtime.step()
    assert not hw.sent
    for side in before:
        np.testing.assert_array_equal(runtime._holds[side], before[side])


@pytest.mark.parametrize('signal', ['_stop', '_estop'])
def test_reset_cancellation_during_collision_check_prevents_write(monkeypatch, signal):
    runtime, hw, clock = moving_reset(monkeypatch)
    clock[0] += .02
    def cancel(q, opening):
        getattr(runtime, signal).set()
        return False
    runtime.solvers['right'].collides = cancel
    runtime.step()
    assert not hw.sent


@pytest.mark.parametrize("reason", ["stale", "released", "stopped", "estop"])
def test_solve_result_is_rejected_if_control_lost_during_solve(reason):
    runtime, reader, hw = controller(("right",))
    runtime.solvers["right"].output[:] = .05
    def during_solve():
        if reason == "stale":
            reader.sample = None
        elif reason == "released":
            reader.update(False)
        elif reason == "stopped":
            runtime._stop.set()
        else:
            runtime._estop.set()
    runtime.solvers["right"].on_solve = during_solve
    reader.update(True)
    runtime.step()
    assert all(np.array_equal(target, hw.measured[side]) for side, target in hw.sent)


def test_stale_holds_measured_and_requires_release_before_reengage():
    runtime, reader, hw = controller(("right",))
    hw.measured["right"][0] = .03
    reader.sample = None
    runtime.step()
    np.testing.assert_array_equal(hw.sent[-1][1], hw.measured["right"])
    reader.update(True)
    runtime.step()
    assert not runtime.clutches["right"].active
    reader.update(False)
    runtime.step()
    reader.update(True)
    runtime.step()
    assert runtime.clutches["right"].active
    np.testing.assert_array_equal(hw.sent[-1][1], hw.measured["right"])


def test_commands_keep_independent_timestamps_and_linear_gripper():
    runtime, reader, hw = controller(("right",))
    runtime.begin_episode()
    reader.update(True)
    reader.sample.buttons["rightTrig"] = (.6,)
    runtime.step()
    np.testing.assert_allclose(hw.sent[-1][1][6], .4)
    events = json.loads(runtime.end_episode())
    assert events[-1]["event"] == "command"
    assert events[-1]["time"] >= events[-1]["vr_time"]
    assert events[-1]["joint_target"][6] == pytest.approx(.4)


def test_hardware_rejects_invalid_output_before_sdk_writes():
    arm = Mock()
    hw = QuestHardware({"left": arm})
    hw.limits["left"] = np.tile([-1., 1.], (6, 1))
    with pytest.raises(RuntimeError, match="invalid"):
        hw.send("left", np.r_[np.ones(6) * 2, .5])
    arm.JointCtrl.assert_not_called()
    arm.GripperCtrl.assert_not_called()


def test_outside_measured_pose_rejects_first_engagement_without_creating_writer():
    reader, hw = Reader(), Hardware(('left', 'right'))
    hw.limits['right'][1, 0] = 0
    hw.measured['right'][1] = np.deg2rad(-1.928)
    runtime = QuestController({s: object() for s in hw.measured}, 'unused', reader=reader, hardware=hw)
    runtime.solvers = {s: Solver() for s in hw.measured}
    reader.update(False)
    runtime.step()
    reader.update(True)
    runtime.step()
    assert not hw.sent and not hw.enabled
    assert runtime.error is None
    assert 'right' in runtime.status and 'J2=-1.928000000' in runtime.status
    assert 'Cannot hold' in runtime.status
    assert all(not c.active and not c.ready for c in runtime.clutches.values())
    # Holding the button must not re-engage even after feedback is back in bounds.
    hw.measured['right'][1] = .01
    reader.update(True)
    runtime.step()
    assert not hw.sent
    reader.update(False)
    runtime.step()
    reader.update(True)
    runtime.step()
    assert all(c.active for c in runtime.clutches.values())
    assert not runtime._hold_errors


@pytest.mark.parametrize('reason', ['stale', 'released', 'ik_rejected'])
def test_outside_measured_hold_never_clips_or_replays_previous_target(reason):
    runtime, reader, hw = controller()
    # A previous valid hold must also be discarded when measured joints leave bounds.
    runtime._holds['right'] = hw.measured['right'].copy()
    hw.limits['right'][1, 0] = 0
    hw.measured['right'][1] = np.deg2rad(-1.928)
    if reason == 'stale':
        reader.sample = None
    else:
        reader.update(reason != 'released')
        if reason == 'ik_rejected':
            runtime.solvers['right'].output = IKRejected('no convergence')
    runtime.step()
    assert hw.sent and all(s == 'left' for s, _ in hw.sent)
    assert 'right' not in runtime._holds
    assert 'J2=-1.928000000' in runtime.status
    assert runtime.error is None
    hw.sent.clear()
    runtime.step()
    assert all(s == 'left' for s, _ in hw.sent)
    assert 'J2=-1.928000000' in runtime.status


def test_invalid_feedback_during_adoption_never_writes_or_marks_enabled():
    arm = Mock()
    hw = QuestHardware({'right': arm})
    hw.limits['right'] = np.tile([-1., 1.], (6, 1))
    hw.require_ready = Mock(return_value={'right': np.r_[2., np.zeros(5), .7]})
    with pytest.raises(InvalidCommand, match='right:.*J1='):
        hw.enable('right', np.zeros(7))
    assert not hw.enabled and not arm.mock_calls


def test_adoption_feedback_race_is_reported_without_stopping_controller():
    reader, hw = Reader(), Hardware(('right',))
    runtime = QuestController({'right': object()}, 'unused', reader=reader, hardware=hw)
    runtime.solvers = {'right': Solver()}
    def adopt(side, target):
        hw.measured[side][1] = -2
        validate_command(side, hw.measured[side], hw.limits[side])
    hw.enable = adopt
    reader.update(False)
    runtime.step()
    reader.update(True)
    runtime.step()
    assert not hw.sent and not hw.enabled
    assert not runtime.clutches['right'].active
    assert runtime.error is None and 'J2=' in runtime.status


def test_final_hold_skips_invalid_arm_and_still_holds_other_arm():
    runtime, _, hw = controller()
    hw.measured['right'][1] = -2
    runtime._stop.set()
    runtime._run()
    assert hw.sent and all(s == 'left' for s, _ in hw.sent)
    assert runtime.error is None


def test_reset_rejects_invalid_measured_pose_before_any_hold(monkeypatch):
    runtime, hw, _ = reset_runtime(monkeypatch)
    hw.measured['right'][1] = -2
    with pytest.raises(InvalidCommand, match='right:.*J2='):
        runtime.step()
    assert not hw.sent


@pytest.mark.parametrize('excess', [5e-8, 5e-7, 2e-6])
def test_ik_bound_tolerance_is_projected_before_collision_check(excess):
    # Exercise the production solve postprocessing without loading native IK.
    solver = QuestIK.__new__(QuestIK)
    solver.seed, solver.q, solver.reference_q, solver.target = np.zeros(6), object(), object(), object()
    solver.limits = np.tile([-1., 1.], (6, 1))
    q = np.zeros(6)
    q[1], q[2] = -1 - excess, 1 + excess
    solver.opti = Mock()
    solver.opti.solve.return_value.value.return_value = q
    solver.collides = Mock(return_value=False)
    if excess > 1e-6:
        with pytest.raises(IKRejected, match='outside joint limits'):
            solver.solve(np.eye(4), .035)
        solver.collides.assert_not_called()
    else:
        result = solver.solve(np.eye(4), .035)
        assert result[1] == -1 and result[2] == 1
        np.testing.assert_array_equal(solver.collides.call_args.args[0], result)
        validate_command('right', np.r_[result, .5], solver.limits)
        # Collision checking still rejects the projected command.
        solver.collides.return_value = True
        with pytest.raises(IKRejected, match='self collision'):
            solver.solve(np.eye(4), .035)


def test_first_target_reanchors_after_arm_enable_has_changed_feedback():
    reader, hw = Reader(), Hardware(("right",))
    runtime = QuestController({"right": object()}, "unused", reader=reader, hardware=hw)
    runtime.solvers = {"right": Solver()}
    def enable(side, target):
        hw.enabled.add(side)
        hw.measured[side][0] = .1
        hw.measured[side][6] = .43
    hw.enable = enable
    reader.update(False)
    runtime.step()
    reader.update(True)
    runtime.step()
    np.testing.assert_array_equal(hw.sent[-1][1], hw.measured["right"])
    np.testing.assert_array_equal(runtime.solvers["right"].seed, hw.measured["right"][:6])


def test_session_vr_lifecycle_records_auxiliary_data_without_replacing_actions(tmp_path):
    from bimanual_vla.collection.session import CollectionConfig, CollectionSession
    from tests.test_piper_data_contract import FakeCameras, FakePiper
    robot, vr = FakePiper(), Mock()
    vr.resetting = False
    vr.end_episode.return_value = '[{"event":"command","time":123}]'
    factory = Mock(return_value=vr)
    state = np.r_[np.zeros(6), .7]
    session = CollectionSession(
        CollectionConfig(output_dir=tmp_path, schema="joint", control_source="quest_vr"),
        robot_connect=lambda _: robot, camera_factory=FakeCameras,
        state_reader=lambda _: (state, state), camera_verifier=lambda *_: {},
        quest_controller_factory=factory)
    session.connect()
    vr.start.assert_called_once()
    session.start_episode("test", "move")
    session.capture_once()
    session.stop_episode()
    path, _ = session.save_episode(True, validate=False)
    with np.load(path, allow_pickle=False) as data:
        assert data["collection_control_source"].item() == "quest_vr"
        assert json.loads(data["quest_telemetry_json"].item())[0]["time"] == 123
        assert data["action_source"].item() == "next_measured_joint_fallback"
        np.testing.assert_array_equal(data["actions"][:-1], data["state"][1:])
    session.disconnect()
    vr.close.assert_called_once()
    assert robot.disconnected


def test_failed_vr_start_closes_cameras_and_robot(tmp_path):
    from bimanual_vla.collection.session import CollectionConfig, CollectionSession
    from tests.test_piper_data_contract import FakeCameras, FakePiper
    robot, vr = FakePiper(), Mock()
    vr.start.side_effect = RuntimeError("IK dependencies absent")
    session = CollectionSession(CollectionConfig(output_dir=tmp_path, control_source="quest_vr"),
        robot_connect=lambda _: robot, camera_factory=FakeCameras,
        state_reader=lambda _: (np.zeros(10), np.zeros(7)), camera_verifier=lambda *_: {},
        quest_controller_factory=Mock(return_value=vr))
    with pytest.raises(RuntimeError, match="IK dependencies"):
        session.connect()
    assert robot.disconnected
    assert session.piper is None
    vr.close.assert_called_once()


def test_engagement_holds_measured_pose_before_setting_teleop_speed():
    reader, hardware = Reader(), Hardware(("right",))
    runtime = QuestController({"right": object()}, "unused", reader=reader, hardware=hardware)
    runtime.solvers = {"right": Solver()}
    reader.update(False)
    runtime.step()
    reader.update(True)
    runtime.step()
    assert hardware.speeds == [("right", 80, 1)]  # speed set only after the hold was sent
    np.testing.assert_array_equal(hardware.sent[0][1], hardware.measured["right"])


def test_lagging_arm_is_tracked_with_bounded_lead_instead_of_rejected(monkeypatch):
    runtime, reader, hw = controller(("right",))
    clock = [100.]
    monkeypatch.setattr("bimanual_vla.collection.quest.control.time.monotonic", lambda: clock[0])
    runtime._command_at["right"] = clock[0]
    runtime.solvers["right"].output = np.full(6, .19)
    previous = hw.measured["right"].copy()
    for _ in range(60):
        clock[0] += 1 / 60
        reader.update(True)
        runtime.step()
        runtime.solvers["right"].output = np.minimum(runtime.solvers["right"].output + .19, .9)
        sent = hw.sent[-1][1]
        # The arm never moves (measured stays 0); motion continues without a re-engage.
        assert runtime.clutches["right"].active
        assert np.max(np.abs(sent[:6] - hw.measured["right"][:6])) <= .35 + 1e-9
        assert np.max(np.abs(sent[:6] - previous[:6])) <= 1.2 / 60 + 1e-9
        previous = sent
    assert np.max(np.abs(hw.sent[-1][1][:6])) == pytest.approx(.35)
