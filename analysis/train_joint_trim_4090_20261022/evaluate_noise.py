"""Evaluate the final single-arm checkpoint with explicit noise, no RTC/filter."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from server_4090 import openpi_single_arm as app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-config", action="store_true")
    options = parser.parse_args()
    root = Path(__file__).parent
    request = json.loads((root / "train_request.json").read_text())
    response = json.loads((root / "train_final_response.json").read_text())
    dataset = Path(response["metadata"]["dataset_root_path"]) / request["dataset_id"]
    command = ["serve", "--dataset-id", request["dataset_id"], "--arm-mode", "single",
        "--arm-side", "right", "--schema", "joint", "--model-variant", "pi05",
        "--checkpoint", str(options.checkpoint), "--base-checkpoint", request["base_checkpoint"],
        "--assets-base-dir", "/home/sunny/robotwin_ws/RoboTwin/policy/pi05/assets",
        "--checkpoint-base-dir", str(Path(response["metadata"]["checkpoint_dir"]).parent.parent)]
    sys.argv = [sys.argv[0], *command]
    args = app.parse_args()
    config = app.build_config(args)
    assert config.model.rtc_training_max_delay == 0
    assert config.model.loss_action_dim == 7
    assert config.policy_metadata["action_offset"] == 1
    assert config.policy_metadata["arm_side"] == "right"
    if options.check_config:
        print("CONFIG_OK", config.model, config.policy_metadata["finetuning"])
        return
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu", jax.devices()
    base = app.policy_config.create_trained_policy(config, options.checkpoint)
    sample = nnx_utils.module_jit(base._model.sample_actions, static_argnames=("num_steps",))
    split = json.loads((dataset / "meta/train_test_split.json").read_text())
    tasks = {r["task_index"]: r["task"] for r in
             map(json.loads, (dataset / "meta/tasks.jsonl").read_text().splitlines())}
    episodes = split["train_episodes"][:2]+split["test_episodes"]
    options.output.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = dict(checkpoint=str(options.checkpoint), dataset=str(dataset),
        train_task_id=response["id"], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inference_prefix=False, output_filter=False, initial_noise="white Gaussian, explicit saved seeds 0/1",
        training_episodes=split["train_episodes"][:2], test_episodes=split["test_episodes"], rows=rows)
    cv2.setNumThreads(1)
    for episode in episodes:
        table = pq.read_table(dataset / f"data/chunk-000/episode_{episode:06d}.parquet")
        states = np.asarray(table["observation.state"].to_pylist(), np.float32)
        labels = np.asarray(table["action"].to_pylist(), np.float32)
        indices = table["task_index"].to_pylist()
        frames = sorted(set(min(frame, len(states)-50) for frame in (0, 40, 80, 120, 180, 240)))
        assert frames[0] >= 0
        for frame in frames:
            images = {}
            for camera in ("cam_high", "cam_right_wrist"):
                capture = cv2.VideoCapture(str(dataset /
                    f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                capture.release()
                assert ok
                images[camera] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = base._input_transform(copy.deepcopy(dict(state=states[frame], images=images,
                                                               prompt=tasks[int(indices[frame])])))
            obs = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], inputs))
            truth = labels[frame:frame+50]
            for seed in (0, 1):
                key = jax.random.split(jax.random.key(seed))[1]
                noise = jax.random.normal(key, (1, 50, 32))
                for steps in (5, 10):
                    normalized = np.asarray(sample(key, obs, num_steps=steps, noise=noise))[0]
                    pred = base._output_transform(dict(state=inputs["state"].copy(),
                                                       actions=normalized.copy()))["actions"]
                    assert pred.shape == truth.shape == (50, 7) and np.isfinite(pred).all()
                    y = np.diff(pred[:, :6], n=2, axis=0)
                    x = np.diff(np.asarray(noise)[0, :, :6], n=2, axis=0)
                    corr = np.mean([np.corrcoef(y[:, j], x[:, j])[0, 1] for j in range(6)])
                    error = abs(pred-truth)
                    rows.append(dict(episode=episode, frame=frame, seed=seed, steps=steps,
                        split="heldout" if episode in split["test_episodes"] else "train",
                        noise_d2_correlation=float(corr), mean_abs_d2_rad=float(abs(y).mean()),
                        demo_mean_abs_d2_rad=float(abs(np.diff(truth[:, :6], n=2, axis=0)).mean()),
                        joint_mae_rad=float(error[:, :6].mean()),
                        first5_joint_mae_rad=float(error[:5, :6].mean()),
                        steps6_10_joint_mae_rad=float(error[5:10, :6].mean()),
                        gripper_mae_fraction=float(error[:, 6].mean())))
                    np.savez_compressed(options.output/f"ep{episode}_f{frame}_s{seed}_n{steps}.npz",
                        prediction=pred, truth=truth, noise=np.asarray(noise)[0], normalized=normalized)
            (options.output / "results.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
            print("PROGRESS", episode, frame, len(rows), flush=True)
    groups = []
    for split_name in ("train", "heldout"):
        for steps in (5, 10):
            selected = [r for r in rows if r["split"] == split_name and r["steps"] == steps]
            keys = [k for k in selected[0] if k.endswith(("rad", "correlation", "fraction"))]
            groups.append(dict(split=split_name, steps=steps, predictions=len(selected),
                observations=len(set((r["episode"], r["frame"]) for r in selected)),
                means={k: float(np.mean([r[k] for r in selected])) for k in keys}))
    (options.output / "summary.json").write_text(json.dumps(dict(checkpoint=str(options.checkpoint),
        script_sha256=manifest["script_sha256"], groups=groups), indent=2, allow_nan=False))
    print("COMPLETED", json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
