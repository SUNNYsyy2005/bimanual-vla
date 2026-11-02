"""Run pinned RLinf returns/value/advantage stages in the allocated job.

Original datasets remain read-only. Per-run snapshots separate critic training
and heldout episodes; advantage labels map back by explicit episode/frame IDs.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import fcntl

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from bimanual_vla.data.recap import RecapEpisode
from bimanual_vla.training.recap_settings import RecapSettings, RLINF_COMMIT


def input_fingerprint(root, episodes):
    digest = hashlib.sha256()
    for path in [root / "meta/info.json", root / "meta/tasks.jsonl"] + [p for ep in sorted(episodes) for p in (
            root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet",
            root / f"meta/recap/episode_{ep:06d}.npz")]:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    for ep in sorted(episodes):
        for path in sorted(root.glob(f"videos/chunk-{ep // 1000:03d}/*/episode_{ep:06d}.mp4")):
            stat = path.stat()
            digest.update(str((path.relative_to(root), stat.st_size, stat.st_mtime_ns)).encode())
    return digest.hexdigest()


def stage_split(source, destination, episodes):
    """Reindex numeric tables, link immutable videos, retain original IDs."""
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("chunks_size", 1000) != 1000:
        raise ValueError("RECAP adapter currently requires LeRobot chunk size 1000")
    if info.get("recap", {}).get("version") != 1:
        raise ValueError("dataset has no supported RECAP annotations")
    destination.mkdir(parents=True)
    (destination / "meta").mkdir()
    shutil.copy2(source / "meta/tasks.jsonl", destination / "meta/tasks.jsonl")
    episode_metadata = {int(row["episode_index"]): row for row in
                        (json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines())}
    stats_path = source / "meta/episodes_stats.jsonl"
    episode_stats = ({int(row["episode_index"]): row for row in
                      (json.loads(line) for line in stats_path.read_text().splitlines())} if stats_path.exists() else {})
    frame_offset, rows, stats, mapping = 0, [], [], {}
    for new, old in enumerate(episodes):
        with np.load(source / f"meta/recap/episode_{old:06d}.npz", allow_pickle=False) as data:
            annotation = RecapEpisode.from_payload(data)
        if not annotation.value_eligible:
            raise ValueError("unlabelled/aborted episodes cannot enter value learning")
        table = pq.read_table(source / f"data/chunk-{old // 1000:03d}/episode_{old:06d}.parquet")
        count = len(table)
        if count != len(annotation.owner):
            raise ValueError("RECAP annotations do not align with dataset rows")
        for name, values in (("episode_index", np.full(count, new, dtype=np.int64)),
                             ("index", np.arange(frame_offset, frame_offset + count, dtype=np.int64))):
            table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
        success = pa.array(np.full(count, annotation.outcome == "success"))
        if "is_success" in table.column_names:
            table = table.set_column(table.schema.get_field_index("is_success"), "is_success", success)
        else:
            table = table.append_column("is_success", success)
        data_path = destination / f"data/chunk-{new // 1000:03d}/episode_{new:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)
        for key, feature in info["features"].items():
            if feature.get("dtype") != "video":
                continue
            target = source / info["video_path"].format(episode_chunk=old // 1000, video_key=key, episode_index=old)
            link = destination / info["video_path"].format(episode_chunk=new // 1000, video_key=key, episode_index=new)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target.resolve(strict=True))
        rows.append(dict(episode_metadata[old], episode_index=new))
        if old in episode_stats:
            stats.append(dict(episode_stats[old], episode_index=new))
        mapping[new] = old
        frame_offset += count
    info.update(total_episodes=len(episodes), total_frames=frame_offset,
                total_chunks=math.ceil(len(episodes) / 1000), splits={"train": f"0:{len(episodes)}"})
    info["total_videos"] = len(episodes) * sum(v.get("dtype") == "video" for v in info["features"].values())
    info["features"]["is_success"] = {"dtype": "bool", "shape": [1], "names": None}
    info.pop("recap", None)  # This private critic view uses RLinf's native columns.
    (destination / "meta/info.json").write_text(json.dumps(info))
    for name, values in (("episodes", rows), ("episodes_stats", stats)):
        (destination / f"meta/{name}.jsonl").write_text("\n".join(json.dumps(row) for row in values) + "\n")
    return mapping


def complete_value_checkpoints(output):
    return sorted((p.parent for p in (output / "value/piper/checkpoints").glob(
        "global_step_*/bimanual_complete") if (p.parent / "actor/model_state_dict").is_dir()),
        key=lambda p: int(p.name.split("_")[-1]))


def run_pipeline(root, output, train, test, config):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (output.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_pipeline(root, output, train, test, config)


def _run_pipeline(root, output, train, test, config):
    settings = RecapSettings(**config["settings"])
    backend = config["backend"]
    upstream = Path(backend["root"]).expanduser().resolve()
    python = str(Path(backend["python"]).expanduser().resolve())
    marker = json.loads((upstream / "bimanual_recap_adapter.json").read_text())
    if marker != {"upstream_commit": RLINF_COMMIT, "adapter_version": 2}:
        raise ValueError("RLinf backend must be installed with setup_recap_backend.py at the pinned revision")
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != RLINF_COMMIT:
        raise ValueError("RLinf backend revision drift")
    for name in ("siglip_path", "gemma3_path"):
        if not Path(backend[name]).is_dir():
            raise FileNotFoundError(f"RECAP pretrained model is not staged: {name}")
    if (not train or not test or set(train) & set(test)
            or len(set(train)) != len(train) or len(set(test)) != len(test)):
        raise ValueError("RECAP requires nonempty disjoint train/heldout episode sets")
    identity = dict(train_episodes=list(train), test_episodes=list(test), settings=settings.to_dict(),
                    upstream_commit=RLINF_COMMIT, adapter_version=2,
                    fingerprint=input_fingerprint(root, list(train) + list(test)))
    manifest = output / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError("RECAP artifacts mismatch dataset/split/settings; use a new experiment")
        return
    if output.exists():
        saved = json.loads((output / "identity.json").read_text())
        if saved != identity:
            raise ValueError("incomplete RECAP run differs from dataset/split/settings")
    else:
        # A killed staging attempt cannot masquerade as a complete split.
        with tempfile.TemporaryDirectory(prefix=".recap-stage-", dir=output.parent) as temporary:
            staged = Path(temporary) / "artifacts"
            staged.mkdir()
            stage_split(root, staged / "train", train)
            stage_split(root, staged / "heldout", test)
            (staged / "identity.json").write_text(json.dumps(identity))
            staged.rename(output)
    mapping = dict(enumerate(train))
    configs = output / "configs"
    configs.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(REPO_PATH=str(upstream), TOKENIZERS_PARALLELISM="false",
                       PYTHONPATH=os.pathsep.join([str(upstream), str(Path(__file__).resolve().parents[1])]))
    # Do not inject OpenPI's Torch/NCCL libraries into the separate RLinf env.
    environment.pop("LD_PRELOAD", None)
    environment["LD_LIBRARY_PATH"] = str(Path(python).parent.parent / "lib")
    base = upstream / "examples/offline_rl/config"

    def execute(name, script, modify):
        done = output / (name + ".done")
        if done.exists():
            return
        cfg = yaml.safe_load((base / (name + ".yaml")).read_text())
        modify(cfg)
        (configs / (name + ".yaml")).write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"RECAP stage: {name}", flush=True)
        subprocess.run([python, str(upstream / script), "--config-path", str(configs), "--config-name", name],
                       cwd=upstream, env=environment, check=True)
        done.touch()

    def paths(names):
        return [dict(dataset_path=str(output / name), type="rollout", robot_type="piper_bimanual", weight=1.)
                for name in names]

    process = "examples/offline_rl/advantage_labeling/recap/"
    def returns(cfg):
        cfg["data"].update(train_data_paths=paths(["train", "heldout"]), failure_reward=settings.failure_reward,
                           gamma=1., tag="piper", num_workers=4)
    execute("recap_compute_returns", process + "process/compute_returns.py", returns)

    def value(cfg):
        cfg["data"].update(train_data_paths=paths(["train"]), eval_data_paths=paths(["heldout"]),
                           tag="piper", action_dim=14, action_horizon=settings.n_step, robot_type="piper_bimanual",
                           train_num_workers=2, eval_num_workers=2)
        cfg["runner"].update(max_steps=settings.value_steps, max_epochs=settings.value_steps,
                             save_interval=math.gcd(settings.value_steps, 1000))
        cfg["runner"]["logger"].update(log_path=str(output / "value"), experiment_name="piper")
        checkpoints = complete_value_checkpoints(output)
        if checkpoints:
            cfg["runner"]["resume_dir"] = str(checkpoints[-1])
        cfg["actor"].update(micro_batch_size=1, global_batch_size=32)
        cfg["actor"]["model"].update(siglip_path=backend["siglip_path"], gemma3_path=backend["gemma3_path"],
                                    tokenizer_path=backend["gemma3_path"], action_dim=14, action_horizon=settings.n_step,
                                    critic_expert_variant="gemma_1m")
        cfg["actor"]["optim"]["total_training_steps"] = settings.value_steps
    execute("recap_value_model_sft", process + "train_value.py", value)
    checkpoints = complete_value_checkpoints(output)
    if not checkpoints:
        raise FileNotFoundError("RLinf did not produce a complete value checkpoint")
    checkpoint = checkpoints[-1] / "actor/model_state_dict"

    def advantage(cfg):
        cfg["data"].update(train_data_paths=paths(["train"]), advantage_lookahead_step=settings.n_step)
        cfg["distributed"]["enabled"] = False
        cfg["advantage"].update(value_checkpoint=str(checkpoint), positive_quantile=settings.positive_fraction,
                                tag="piper", returns_tag="piper", batch_size=8, num_dataloader_workers_per_gpu=2)
        cfg["advantage"]["model"].update(siglip_path=backend["siglip_path"], gemma3_path=backend["gemma3_path"],
                                        tokenizer_path=backend["gemma3_path"], critic_expert_variant="gemma_1m",
                                        action_dim=14, action_horizon=settings.n_step)
    execute("recap_compute_advantages", process + "process/compute_advantages.py", advantage)
    labels = pq.read_table(output / "train/meta/advantages_piper.parquet").to_pydict()
    for new, old in mapping.items():
        with np.load(root / f"meta/recap/episode_{old:06d}.npz", allow_pickle=False) as data:
            annotation = RecapEpisode.from_payload(data)
        selected = np.flatnonzero(np.asarray(labels["episode_index"]) == new)
        frames = np.asarray(labels["frame_index"])[selected]
        order = np.argsort(frames)
        if not np.array_equal(frames[order], np.arange(len(annotation.owner))):
            raise ValueError("RLinf advantage output has missing/duplicate/misaligned frames")
        positive = np.asarray(labels["advantage"], dtype=bool)[selected][order] | annotation.intervention
        np.savez_compressed(output / f"episode_{old:06d}.npz", positive=positive,
                            intervention=annotation.intervention)
    from bimanual_vla.deployment.recap import atomic_json
    atomic_json(manifest, dict(identity, value_checkpoint=str(checkpoint)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=json.loads, required=True)
    parser.add_argument("--test-episodes", type=json.loads, required=True)
    parser.add_argument("--config", type=json.loads, required=True)
    args = parser.parse_args()
    run_pipeline(args.dataset.resolve(), args.output.resolve(), args.train_episodes, args.test_episodes, args.config)
