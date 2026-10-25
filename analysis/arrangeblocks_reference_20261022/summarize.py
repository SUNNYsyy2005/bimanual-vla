"""Summarize paired reference predictions and optionally plot a representative case."""
import argparse
import json
from pathlib import Path

import numpy as np

JOINTS = [i for i in range(14) if i % 7 != 6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.predictions / "results.json").read_text())
    groups = []
    for resize in ("pad", "stretch"):
        for steps in (5, 10):
            for offset in (0, 1):
                rows = [x for x in manifest["rows"] if (x["resize"], x["steps"], x["offset"]) == (resize, steps, offset)]
                frames = sorted(set(x["frame"] for x in rows))
                assert len(rows) == 48 and len(frames) == 24
                numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and
                           k not in {"steps", "seed", "frame", "offset", "sampler_ms"}]
                mean = {k: float(np.mean([r[k] for r in rows])) for k in numeric}
                mean["per_joint_mae"] = np.mean([r["per_joint_mae"] for r in rows], axis=0).tolist()
                active = [r for r in rows if r["hold_mae_1_5"] > .01]
                seed_difference = []
                noise_gain = []
                example_errors = []
                for frame in frames:
                    with np.load(args.predictions / f"{resize}_f{frame}_s0_n{steps}.npz") as a, np.load(
                            args.predictions / f"{resize}_f{frame}_s1_n{steps}.npz") as b:
                        delta = (a["prediction"] - b["prediction"])[:, JOINTS]
                        seed_difference.append(float(abs(delta).mean()))
                        for case in (a, b):
                            p2 = np.diff(case["normalized"][:, JOINTS], n=2, axis=0)
                            n2 = np.diff(case["noise"][:, JOINTS], n=2, axis=0)
                            # Remove means to estimate per-joint noise gain, in normalized space.
                            p2 -= p2.mean(axis=0)
                            n2 -= n2.mean(axis=0)
                            noise_gain.append(np.sum(p2*n2, axis=0) / np.sum(n2*n2, axis=0))
                        example_errors.append(float(abs(a["prediction"][:, JOINTS] -
                            a[f"truth_offset{offset}"][:, JOINTS]).mean()))
                groups.append(dict(resize=resize, steps=steps, offset=offset, observations=24,
                    seeds=[0, 1], predictions=48, mean=mean,
                    roughness_ratio=mean["pred_d2"]/mean["truth_d2"],
                    seed_pair_joint_mae=float(np.mean(seed_difference)),
                    normalized_noise_gain=np.mean(noise_gain, axis=0).tolist(),
                    active_definition="hold MAE for steps 1-5 > 0.01 rad across 12 joints",
                    active_observations=len(active)//2,
                    active_mean={k: float(np.mean([r[k] for r in active])) for k in numeric},
                    observation_errors_seed0={str(f): e for f, e in zip(frames, example_errors)}))
    summary = {k: v for k, v in manifest.items() if k != "rows"}
    summary["groups"] = groups
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Choose median-error observation for the stock pad/n5 setting.
        errors = groups[0]["observation_errors_seed0"]
        frame = sorted(errors, key=lambda f: errors[f])[len(errors)//2]
        a = np.load(args.predictions / f"pad_f{frame}_s0_n5.npz")
        b = np.load(args.predictions / f"pad_f{frame}_s1_n5.npz")
        truth = a["truth_offset0"]
        selected = [int(np.argmax(abs(a["prediction"][:, JOINTS] - truth[:, JOINTS]).mean(axis=0)))]
        joint = JOINTS[selected[0]]
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
        t = np.arange(50) / 20
        for data, label, color in ((truth, "Demonstration", "black"),
                                  (a["prediction"], "Prediction, seed 0", "tab:blue"),
                                  (b["prediction"], "Prediction, seed 1", "tab:orange")):
            axes[0].plot(t, data[:, joint], label=label, color=color)
            axes[1].plot(t, abs(data[:, JOINTS]-truth[:, JOINTS]).mean(axis=1), color=color)
            axes[2].plot(t[1:-1], np.diff(data[:, joint], n=2), color=color)
        axes[0].axhline(a["state"][joint], color="gray", ls="--", label="Hold current position")
        axes[0].set_ylabel(f"Joint {joint+1} position (rad)")
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel("12-joint MAE (rad)")
        axes[2].set_ylabel("Second difference (rad)")
        axes[2].set_xlabel("Future trajectory time (s); action offset 0 assumed")
        fig.suptitle(f"Non-RTC model, frame {frame}, standard pad, 5 denoising steps\n"
                     "Median-error observation; displayed joint has its largest mean error")
        fig.savefig(args.output.with_suffix(".png"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
