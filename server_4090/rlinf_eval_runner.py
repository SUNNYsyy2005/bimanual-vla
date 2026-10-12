#!/usr/bin/env python3
"""Run native RLInf checkpoint rollouts and emit dashboard-readable results."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
import sys


def evaluation_overrides(args) -> list[str]:
    return [
        # Rollout workers use actor.model during training but rollout.model in
        # eval-only mode. Keep the architecture (including any auxiliary heads)
        # identical so exported actor state_dict weights load strictly.
        "rollout.model=${actor.model}",
        "runner.only_eval=true", f"runner.ckpt_path={args.checkpoint}",
        f"runner.logger.log_path={args.output}",
        "env.eval.total_num_envs=1", f"env.eval.rollout_epoch={args.episodes}",
        "env.eval.auto_reset=false", "env.eval.ignore_terminations=false",
        f"env.eval.seed={args.seed}", f"env.eval.max_episode_steps={args.max_steps}",
        f"env.eval.max_steps_per_rollout_epoch={args.max_steps}",
        f"env.eval.video_cfg.save_video={str(args.record_video).lower()}",
        f"env.eval.video_cfg.video_base_dir={Path(args.output) / 'video'}",
    ]


def hydra_argv(args, extra: list[str]) -> list[str]:
    options, overrides = [], []
    tokens = iter(extra)
    for token in tokens:
        if token.startswith("--"):
            options.append(token)
            if token in {"--cfg", "--info", "--package", "--config-dir"}:
                options.append(next(tokens))
        else:
            overrides.append(token)
    # Hydra's positional override parser requires its CLI flags first.
    return [args.entrypoint, "--config-path", args.config_path, "--config-name", args.config_name,
            *options, *overrides, *evaluation_overrides(args)]


def write_results(output: Path, metrics: dict, *, step: int, checkpoint: str) -> None:
    values = {str(key): float(value) for key, value in metrics.items()
              if isinstance(value, (float, int)) and math.isfinite(float(value))}
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "completed", "checkpoint": checkpoint, "checkpoint_step": step, "metrics": values}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "evaluation_curve.json").write_text(json.dumps([{"actor_updates": step, **values}]), encoding="utf-8")
    print("RLINF_EVAL_RESULT " + json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--record-video", choices=["true", "false"], default="false")
    args, extra = parser.parse_known_args()
    args.record_video = args.record_video == "true"
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"complete checkpoint weights not found: {checkpoint}")
    if not 1 <= args.episodes <= 1000 or not 1 <= args.max_steps <= 2000:
        raise ValueError("invalid rollout episode count or horizon")

    from rlinf.runners import embodied_eval_runner
    base = embodied_eval_runner.EmbodiedEvalRunner

    class DashboardEvalRunner(base):
        def evaluate(self):
            metrics = super().evaluate()
            write_results(Path(args.output), metrics, step=args.step, checkpoint=str(checkpoint))
            return metrics

    embodied_eval_runner.EmbodiedEvalRunner = DashboardEvalRunner
    sys.argv = hydra_argv(args, extra)
    runpy.run_path(args.entrypoint, run_name="__main__")


if __name__ == "__main__":
    main()
