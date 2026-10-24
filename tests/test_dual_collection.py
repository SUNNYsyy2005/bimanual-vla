"""One recording -> two actual LeRobot exports, without hardware or GPU work."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest

from bimanual_vla.collection.camera import CameraCapture
from bimanual_vla.collection.gui import build_dataset_tool_command, prepare_preview_frame
from bimanual_vla.collection.output import read_robot_sample, PiperFeedbackStaleError
from bimanual_vla.data.contract import EpisodeBuffer, EpisodeContract, build_delivery_state
from bimanual_vla.data.check import check_dataset
from bimanual_vla.data.export import export_dataset, inspect_npz_episode
from bimanual_vla.data.image_resize import resize_stretch
from bimanual_vla.data.upload import prepare_raw_npz_dataset
from bimanual_vla.data.validate import validate_episode, EpisodeValidationError


def recorded_episode(path, mode="single", side="left", *, commanded_gripper=False, fps=30):
    buffer = EpisodeBuffer(fps=fps, schema="delivery", arm_mode=mode, arm_side=side,
                           record_both=True, image_resize_mode="stretch")
    sides = ("left", "right") if mode == "bimanual" else (side,)
    for index in range(5):
        timestamp = 100.0 + index / fps
        qpos, eef = [], []
        feedback = {}
        for arm_index, arm in enumerate(sides):
            qpos.append(np.asarray([index * 0.01 + arm_index] * 6 + [index / 4], np.float32))
            eef.append(build_delivery_state([index * 0.01, arm_index * 0.2, 0.3], np.eye(3), index / 4 * 0.07))
            feedback.update({f"{arm}_joint": timestamp - 0.002,
                             f"{arm}_eef": timestamp, f"{arm}_gripper": timestamp - 0.001})
        images = {key: np.full((224, 224, 3), 20 + index + camera * 40, np.uint8)
                  for camera, key in enumerate(buffer.contract.camera_keys)}
        buffer.add(np.concatenate(eef), images, {key: timestamp + .003 for key in images},
                   qpos=np.concatenate(qpos), state_timestamp=timestamp, feedback_timestamps=feedback,
                   gripper_targets=np.full(len(sides), .37, np.float32) if commanded_gripper else None,
                   gripper_command_timestamps=np.full(len(sides), timestamp - .004) if commanded_gripper else None)
    buffer.save(path, "pick_cube", "pick up the cube", True)
    return buffer


@pytest.mark.parametrize("mode,side,arms", [("single", "left", 1), ("single", "right", 1), ("bimanual", "both", 2)])
def test_one_recording_exports_both_training_contracts(tmp_path, mode, side, arms):
    source = tmp_path / "raw"
    source.mkdir()
    path = source / "ep_0000.npz"
    recorded_episode(path, mode, side)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    validate_episode(path)
    specs = {}
    for schema, dimension in (("joint", arms * 7), ("delivery", arms * 10)):
        specs[schema] = spec = inspect_npz_episode(path, target_schema=schema)
        assert spec["state_dim"] == spec["raw_action_dim"] == dimension
        assert spec["model_action_dim"] == arms * 7
        assert spec["arm_side"] == side
        assert all(name.startswith(("left_", "right_")) for name in spec["state_names"])
        expected_cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"] if arms == 2 else ["cam_high", f"cam_{side}_wrist"]
        assert spec["camera_keys"] == expected_cameras
        np.testing.assert_array_equal(spec["actions"][:-1], spec["states"][1:])
        np.testing.assert_array_equal(spec["action_timestamps"][:-1], spec["state_timestamps"][1:])
        output = tmp_path / schema
        export_dataset(source, output, target_schema=schema)
        assert check_dataset(output) == []
        info = json.loads((output / "meta/info.json").read_text())
        contract = json.loads((output / "meta/policy_contract.json").read_text())
        assert info["image_resize_mode"] == contract["image_resize_mode"] == "stretch"
        assert info["schema"] == schema and info["action_offset"] == 1
        assert contract["arm_side"] == side
        assert info["gripper_semantics"] == "absolute_opening_fraction_0_closed_1_open"
        table = pq.read_table(next((output / "data").glob("chunk-*/*.parquet")))
        np.testing.assert_allclose(table["observation.state"].to_pylist(), spec["states"])
        np.testing.assert_allclose(table["action"].to_pylist(), spec["actions"])
        np.testing.assert_allclose(table["state_timestamp"].to_numpy(), spec["state_timestamps"], rtol=0)
        np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(len(spec["states"])) / 30)
    for key in specs["joint"]["images"]:
        np.testing.assert_array_equal(specs["joint"]["images"][key], specs["delivery"]["images"][key])
        np.testing.assert_array_equal(specs["joint"]["image_timestamps"][key], specs["delivery"]["image_timestamps"][key])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_commanded_gripper_is_not_copied_to_measured_joint_targets(tmp_path):
    path = tmp_path / "ep_0000.npz"
    recorded_episode(path, commanded_gripper=True)
    delivery = inspect_npz_episode(path, target_schema="delivery")
    joint = inspect_npz_episode(path, target_schema="joint")
    np.testing.assert_allclose(delivery["actions"][:, -1], .37)
    np.testing.assert_allclose(joint["actions"][:-1, -1], joint["states"][1:, -1])
    assert "same_step_command" in delivery["metadata"]["gripper_action_source"]
    assert joint["action_source"] == "next_measured_joint_fallback"


def test_caches_and_gui_names_are_distinct_for_each_training_schema(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    recorded_episode(source / "ep_0000.npz")
    paths = []
    for schema in ("joint", "delivery"):
        kwargs = dict(fps=None, allow_incomplete_gripper_coverage=False, rebuild=False, target_schema=schema)
        path = prepare_raw_npz_dataset(source, "cube", tmp_path / "cache", **kwargs)
        assert prepare_raw_npz_dataset(source, "cube", tmp_path / "cache", **kwargs) == path
        paths.append(path)
        command = build_dataset_tool_command(python_executable="python", module_name="bimanual_vla.data.upload",
                    source_dir=source, dataset_name="cube", fps=20, action="prepare", target_schema=schema)
        assert command[command.index("--target-schema") + 1] == schema
        assert command[command.index("--name") + 1] == f"cube_{schema}"
    assert paths[0] != paths[1]
    assert all(path.is_dir() for path in paths)


def test_new_recording_rejects_corrupted_sync_and_wrong_fps(tmp_path):
    path = tmp_path / "ep_0000.npz"
    recorded_episode(path)
    with pytest.raises(ValueError, match="FPS|fps"):
        inspect_npz_episode(path, fps=20)
    with np.load(path) as source:
        payload = dict(source)
    payload["image_timestamps_cam_left_wrist"] += 0.2
    np.savez_compressed(path, **payload)
    with pytest.raises(EpisodeValidationError, match="sync exceeds"):
        inspect_npz_episode(path, target_schema="joint")


def test_single_representation_cannot_fake_missing_eef(tmp_path):
    path = tmp_path / "ep_0000.npz"
    buffer = EpisodeBuffer(schema="joint", arm_side="left")
    images = {key: np.full((224, 224, 3), 20, np.uint8) for key in buffer.contract.camera_keys}
    buffer.add(np.zeros(7, np.float32), images, {key: 1.0 for key in images}, state_timestamp=1.0)
    buffer.save(path, "cube", "pick up the cube", True)
    with pytest.raises(ValueError, match="not recorded"):
        inspect_npz_episode(path, target_schema="delivery")


def test_dual_buffer_rejects_missing_joint_and_timing_gaps(tmp_path):
    buffer = recorded_episode(tmp_path / "ep_0000.npz")
    images = {key: np.full((224, 224, 3), 30, np.uint8) for key in buffer.contract.camera_keys}
    kwargs = dict(state=buffer.states[-1], images=images, image_ts={key: 100.5 for key in images}, state_timestamp=100.5)
    with pytest.raises(ValueError, match="requires joint_qpos"):
        buffer.add(**kwargs)
    with pytest.raises(ValueError, match="capture interval"):
        buffer.add(**kwargs, qpos=buffer.joint_qpos[-1], feedback_timestamps={key: 100.5 for key in buffer.feedback_timestamps[-1]})
    assert len(buffer) == 5


def test_sdk_timestamps_survive_snapshot_and_stale_values_fail():
    def arm(stamp, joint=1000):
        return SimpleNamespace(
            GetArmJointMsgs=lambda: SimpleNamespace(time_stamp=stamp-.002, joint_state=SimpleNamespace(**{f"joint_{i}": joint for i in range(1, 7)})),
            GetArmGripperMsgs=lambda: SimpleNamespace(time_stamp=stamp-.001, gripper_state=SimpleNamespace(grippers_angle=35000)),
            GetArmEndPoseMsgs=lambda: SimpleNamespace(time_stamp=stamp, end_pose=SimpleNamespace(X_axis=100000,Y_axis=0,Z_axis=200000,RX_axis=0,RY_axis=0,RZ_axis=0)),
        )
    with patch("bimanual_vla.collection.output.time.time", return_value=100.01):
        sample = read_robot_sample({"left": arm(100, 1000), "right": arm(100, 2000)}, schema="delivery", arm_mode="bimanual", arm_side="both")
        assert sample.state_timestamp == 100
        assert sample.feedback_timestamps["left_joint"] == 99.998
        assert sample.joint_qpos[7] == pytest.approx(2 * sample.joint_qpos[0])
        assert sample.state.shape == (20,) and sample.joint_qpos.shape == (14,)
        with pytest.raises(PiperFeedbackStaleError):
            read_robot_sample(arm(99), schema="delivery", arm_mode="single", arm_side="left")
        with pytest.raises(PiperFeedbackStaleError):
            read_robot_sample(arm(float("nan")), schema="delivery", arm_mode="single", arm_side="left")


def test_stretch_camera_keeps_edges_without_black_padding():
    native = np.full((60, 100, 3), 80, np.uint8)
    native[0:10] = [10, 90, 200]
    native[-10:] = [200, 90, 10]
    camera = CameraCapture(cam_ids={"cam_high": "fake"}, image_hw=(224, 224), resize_mode="stretch")
    camera._caps = {"cam_high": SimpleNamespace(read=lambda: (True, native))}
    images, _ = camera.read()
    rgb = images["cam_high"].transpose(1, 2, 0)
    np.testing.assert_array_equal(rgb, resize_stretch(native[:, :, ::-1], 224, 224))
    assert np.all(rgb > 0)
    preview = prepare_preview_frame(rgb, target_aspect=5/3, resize_mode="stretch")
    np.testing.assert_array_equal(preview[0, 0], rgb[0, 0])
    np.testing.assert_array_equal(preview[-1, 0], rgb[-1, 0])


def test_physical_arm_label_must_agree_with_wrist():
    with pytest.raises(ValueError, match="physical arm_side"):
        EpisodeContract(schema="joint", arm_side="left", camera_keys=("cam_high", "cam_right_wrist"))


def test_old_20hz_and_new_30hz_keep_their_own_rate(tmp_path):
    from bimanual_vla.collection.session import CollectionConfig
    assert CollectionConfig().capture_fps == CollectionConfig().camera_fps == 30
    for fps in (20, 30):
        source = tmp_path / f"raw_{fps}"
        source.mkdir()
        recorded_episode(source / "ep_0000.npz", fps=fps)
        output = tmp_path / f"joint_{fps}"
        export_dataset(source, output, target_schema="joint")
        assert json.loads((output / "meta/info.json").read_text())["fps"] == fps
    command = build_dataset_tool_command(python_executable="python", module_name="bimanual_vla.data.upload",
        source_dir=tmp_path, dataset_name="cube", fps=None, action="prepare", target_schema="joint")
    assert "--fps" not in command


def test_actual_15fps_is_rejected_even_when_driver_reports_30():
    from bimanual_vla.collection.output import verify_camera_streams
    cameras = SimpleNamespace(verify=lambda: {"cam_high": {"ok": True, "fps": 30}},
                              measure_fps=lambda: {"cam_high": 15})
    with pytest.raises(RuntimeError, match="measured 15"):
        verify_camera_streams(cameras, 30)


def test_read_next_waits_for_new_camera_timestamp():
    from bimanual_vla.collection.camera import CameraFrameSet
    camera = CameraCapture(cam_ids={"cam_high": "fake"})
    camera._background_thread = object()
    frame = CameraFrameSet(images={"cam_high": np.ones((3, 224, 224), np.uint8)},
        timestamps={"cam_high": 100.0}, monotonic_timestamps={"cam_high": 1.0}, captured_monotonic=1.0)
    camera._frame_history.append(frame)
    assert camera.read_next({"cam_high": 99.9}).timestamps["cam_high"] == 100.0
    with pytest.raises(RuntimeError, match="fresh camera"):
        camera.read_next({"cam_high": 100.0}, timeout_s=.001)


def test_actual_session_dual_capture_uses_camera_paced_30hz_and_sdk_stamps(tmp_path):
    from bimanual_vla.collection.session import CollectionConfig, CollectionSession
    from bimanual_vla.collection.output import RobotStateSample
    clock = {"index": 0}
    class Cameras:
        def __init__(self, **kwargs): self.keys = kwargs["cam_ids"]
        def open(self): pass
        def close(self): pass
        def start_background_capture(self): pass
        def read_next(self, previous):
            clock["index"] += 1
            timestamp = 100 + clock["index"] / 30
            return SimpleNamespace(images={key: np.full((224, 224, 3), 10 + clock["index"] + i*30, np.uint8) for i,key in enumerate(self.keys)},
                timestamps={key: timestamp for key in self.keys})
    def read_state(robot):
        step = clock["index"]
        timestamp = 100 + step / 30
        qpos = np.asarray([step*.01]*6 + [step*.1], np.float32)
        state = build_delivery_state([step*.01, 0, .2], np.eye(3), step*.1*.07)
        return RobotStateSample(state, qpos, timestamp, {"left_joint":timestamp-.002,"left_eef":timestamp,"left_gripper":timestamp-.001})
    robot = SimpleNamespace(DisconnectPort=lambda: None)
    session = CollectionSession(CollectionConfig(output_dir=tmp_path, arm_side="left", record_both=True),
        robot_connect=lambda _:robot, camera_factory=Cameras, state_reader=read_state,
        camera_verifier=lambda cameras,fps:{}, gripper_command_reader=lambda robot:(None,None))
    session.connect()
    session.start_episode("cube", "pick up the cube")
    for _ in range(5): session.capture_once()
    session.stop_episode()
    path, stats = session.save_episode(True)
    assert stats.actual_fps == pytest.approx(30)
    with np.load(path) as data:
        assert data["fps"] == 30 and set(data["recorded_state_schemas"]) == {"joint", "delivery"}
        assert data["feedback_timestamp_left_joint"][0] == pytest.approx(100+1/30-.002)
        assert data["joint_state"].shape == (6,7)
    session.disconnect()
