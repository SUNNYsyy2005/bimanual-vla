import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest
import subprocess
import yaml

from bimanual_vla.data.recap import RecapEpisode
from bimanual_vla.data.recap_export import export_runs
from bimanual_vla.training.recap_adapter import (
    ConditionedDataset, PiperValueInputs, PositivePolicy, policy_admission,
)
from bimanual_vla.training.recap_settings import RecapSettings
from bimanual_vla.training.recap_settings import RLINF_COMMIT
from server_4090.recap_pipeline import stage_split, input_fingerprint, complete_value_checkpoints, run_pipeline
from tests.test_recap_export import make_run


def test_value_views_match_training_and_scoring_inputs():
    sample = {"prompt": "place it", "observation.state": np.zeros(20)}
    for i, key in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist")):
        sample["observation.images." + key] = np.full((3, 8, 9), i / 2, dtype=np.float32)
    result = PiperValueInputs()(sample)
    assert list(result["image"]) == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    assert result["image"]["left_wrist_0_rgb"].shape == (8, 9, 3)
    assert result["image"]["right_wrist_0_rgb"].max() == 255
    assert all(result["image_mask"].values())


def test_chunks_cannot_cross_handoff_or_action_offset(tmp_path):
    meta = tmp_path / "meta/recap"
    meta.mkdir(parents=True)
    annotation = RecapEpisode(np.array(["policy"] * 6 + ["human"] * 6),
                              np.array([0] * 6 + [2] * 6), np.ones(12, dtype=bool), "success")
    np.savez(meta / "episode_000000.npz", **annotation.to_payload())
    assert policy_admission(tmp_path, [0], 4, 1)[0] == {0, 1, 2, 6, 7, 8}
    assert policy_admission(tmp_path, [0], 4, 0)[0] == {0, 1, 6, 7}


def test_all_samples_retained_and_positive_condition_matches_serving(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"train_episodes": [5]}))
    np.savez(tmp_path / "episode_000005.npz", positive=[False, True])
    source = [dict(episode_index=5, frame_index=i, prompt="place") for i in range(2)]
    dataset = ConditionedDataset(source, tmp_path, .1)
    assert len(dataset) == 2
    with patch("numpy.random.random", return_value=.5):
        assert dataset[0]["prompt"] == "place"
        assert dataset[1]["prompt"] == "place\nAdvantage: positive"
    with patch("numpy.random.random", return_value=.01):
        assert dataset[1]["prompt"] == "place"
    class Echo:
        def infer(self, obs):
            return obs
    policy = PositivePolicy(Echo())
    obs = {"prompt": "place", "rtc": {"session": "s"}}
    assert policy.infer(obs)["prompt"] == "place\nAdvantage: positive"
    assert policy.infer(policy.infer(obs))["prompt"].count("Advantage:") == 1
    assert obs["prompt"] == "place"
    assert policy.infer({"rtc": {}}) == {"rtc": {}}  # server prompt cache


def test_private_splits_reindex_without_mutating_source(tmp_path):
    root = export_runs([make_run(tmp_path / "r0"), make_run(tmp_path / "r1", "failure")], tmp_path / "ds")
    before = input_fingerprint(root, [0, 1])
    assert stage_split(root, tmp_path / "split", [1]) == {0: 1}
    table = pq.read_table(tmp_path / "split/data/chunk-000/episode_000000.parquet")
    assert set(table["episode_index"].to_pylist()) == {0}
    assert not any(table["is_success"].to_pylist())
    assert table["index"].to_pylist() == list(range(len(table)))
    assert before == input_fingerprint(root, [0, 1])
    for path in (tmp_path / "split/videos").rglob("*.mp4"):
        assert path.is_symlink() and path.resolve().is_file()


def test_checkpoint_resume_ignores_interrupted_saves(tmp_path):
    for step in (10, 20):
        p = tmp_path / f"value/piper/checkpoints/global_step_{step}"
        (p / "actor/model_state_dict").mkdir(parents=True)
        if step == 10:
            (p / "bimanual_complete").touch()
    assert [p.name for p in complete_value_checkpoints(tmp_path)] == ["global_step_10"]


@pytest.mark.parametrize("settings", [dict(positive_fraction=0), dict(positive_fraction=float("nan")),
                                      dict(n_step=True), dict(value_steps=0), dict(enabled="true")])
def test_invalid_training_settings_rejected(settings):
    with pytest.raises(ValueError):
        RecapSettings(**settings)


def test_export_failure_does_not_leave_trainable_partial_dataset(tmp_path):
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="success/failure"):
        export_runs([make_run(tmp_path / "ok"), make_run(tmp_path / "bad", "aborted")], output)
    assert not output.exists()


def test_pipeline_resumes_value_and_reuses_labels_without_training_again(tmp_path):
    import pyarrow as pa
    root = export_runs([make_run(tmp_path / "r0"), make_run(tmp_path / "r1", "failure")], tmp_path / "ds")
    upstream = tmp_path / "upstream"
    config_dir = upstream / "examples/offline_rl/config"
    config_dir.mkdir(parents=True)
    (upstream / "bimanual_recap_adapter.json").write_text(json.dumps(
        dict(upstream_commit=RLINF_COMMIT, adapter_version=2)))
    configs = {
        "recap_compute_returns": {"data": {}},
        "recap_value_model_sft": {"data": {}, "runner": {"logger": {}}, "actor": {"model": {}, "optim": {}}},
        "recap_compute_advantages": {"data": {}, "distributed": {}, "advantage": {"model": {}}},
    }
    for key, value in configs.items():
        (config_dir / (key + ".yaml")).write_text(yaml.safe_dump(value))
    backend = dict(root=str(upstream), python="/conda/bin/python", siglip_path=str(tmp_path), gemma3_path=str(tmp_path))
    config = dict(settings=RecapSettings(enabled=True, value_steps=2).to_dict(), backend=backend)
    output = tmp_path / "artifacts"
    stages = []

    def execute(command, **kwargs):
        name = command[-1]
        cfg = yaml.safe_load((output / "configs" / (name + ".yaml")).read_text())
        stages.append(name)
        if name == "recap_value_model_sft":
            checkpoint = output / "value/piper/checkpoints/global_step_1"
            if stages.count(name) == 1:
                (checkpoint / "actor/model_state_dict").mkdir(parents=True)
                (checkpoint / "bimanual_complete").touch()
                raise subprocess.CalledProcessError(1, command)
            assert cfg["runner"]["resume_dir"] == str(checkpoint)
            assert cfg["actor"]["model"]["critic_expert_variant"] == "gemma_1m"
        if name == "recap_compute_advantages":
            assert cfg["data"]["train_data_paths"][0]["dataset_path"] == str(output / "train")
            count = len(pq.read_table(output / "train/data/chunk-000/episode_000000.parquet"))
            pq.write_table(pa.table(dict(episode_index=[0]*count, frame_index=list(range(count)),
                                         advantage=[False]*count)), output / "train/meta/advantages_piper.parquet")

    with patch("server_4090.recap_pipeline.subprocess.check_output", return_value=RLINF_COMMIT), \
         patch("server_4090.recap_pipeline.subprocess.run", side_effect=execute):
        with pytest.raises(subprocess.CalledProcessError):
            run_pipeline(root, output, [0], [1], config)
        run_pipeline(root, output, [0], [1], config)
        run_pipeline(root, output, [0], [1], config)
        assert stages.count("recap_compute_returns") == 1
        assert stages.count("recap_value_model_sft") == 2
        assert stages.count("recap_compute_advantages") == 1
        with np.load(output / "episode_000000.npz") as labels:
            assert labels["positive"].any()  # human corrections, even when critic labels are negative
        config["settings"]["n_step"] = 10
        with pytest.raises(ValueError, match="mismatch"):
            run_pipeline(root, output, [0], [1], config)
