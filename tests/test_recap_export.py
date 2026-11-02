import json

import cv2
import numpy as np
import pytest

from bimanual_vla.data.recap import RecapEpisode
from bimanual_vla.data.recap_export import export_runs, load_run


def make_run(root, outcome="success"):
    root.mkdir()
    (root / "videos").mkdir()
    keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    metadata = dict(recap=dict(outcome=outcome), instruction="Hand over the container", control_hz=30.,
                    policy_protocol=dict(schema="delivery", contract_version=3, arm_mode="bimanual",
                                         arm_side="both", image_resize_mode="stretch", camera_keys=keys))
    (root / "metadata.json").write_text(json.dumps(metadata))
    frames = 70
    timestamps = 1000 + np.arange(frames) / 30
    states = np.zeros((frames, 20), dtype=np.float32)
    states[:, [3, 7, 13, 17]] = 1
    states[:, [9, 19]] = .4  # contact opening differs from closing command
    states[:, 0] = np.arange(frames) * .001
    owner = np.array(["policy"] * 10 + ["human"] * 60)
    epoch = np.array([0] * 10 + [2] * 60)
    np.savez(root / "trajectory.npz", timestamp=timestamps, delivery_state=states,
             command_gripper_fraction=np.zeros((frames, 2)), recap_owner=owner,
             recap_epoch=epoch, recap_valid=np.ones(frames, dtype=bool))
    rows = []
    for key in keys:
        video = root / f"videos/{key}.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 24))
        assert writer.isOpened()
        for i, stamp in enumerate(timestamps):
            writer.write(np.full((24, 32, 3), i, dtype=np.uint8))
            rows.append(dict(camera=key, timestamp=float(stamp), frame_index=i, storage=f"videos/{key}.mp4"))
        writer.release()
    (root / "videos/timestamps.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return root


def test_export_keeps_contact_pose_and_command_gripper_separate(tmp_path):
    run = make_run(tmp_path / "run")
    _, annotation, values = load_run(run)
    np.testing.assert_allclose(values["states"][:, [9, 19]], .4)
    np.testing.assert_allclose(values["actions"][:, [9, 19]], 0)
    assert not annotation.valid[9]  # next measured pose would cross the handoff
    output = export_runs([run], tmp_path / "dataset")
    info = json.loads((output / "meta/info.json").read_text())
    assert info["recap"]["version"] == 1
    with np.load(output / "meta/recap/episode_000000.npz") as data:
        annotation = RecapEpisode.from_payload(data)
    assert len(annotation.chunk_starts(50, human_only=True)) > 0
    with pytest.raises(FileExistsError):
        export_runs([run], output)


def test_abort_is_not_silently_treated_as_failure(tmp_path):
    with pytest.raises(ValueError, match="explicit success/failure"):
        load_run(make_run(tmp_path / "run", "aborted"))


def test_export_retains_explicit_base_calibration(tmp_path):
    output = export_runs([make_run(tmp_path / "run")], tmp_path / "dataset", arm_base_offset=(0, .5, 0))
    info = json.loads((output / "meta/info.json").read_text())
    assert info["arm_base_offset"]["right_relative_to_left_m"] == [0, .5, 0]


def test_default_rate_follows_recording_and_upsampling_is_refused(tmp_path):
    run = make_run(tmp_path / "run")
    assert len(load_run(run)[2]["timestamps"]) == len(load_run(run, fps=30)[2]["timestamps"])
    metadata = json.loads((run / "metadata.json").read_text())
    metadata["control_hz"] = 20.
    (run / "metadata.json").write_text(json.dumps(metadata))
    # A 20 Hz run exported at 30 fps repeats rows that are then masked invalid.
    with pytest.raises(ValueError, match="exceeds the recorded control rate"):
        load_run(run, fps=30)
