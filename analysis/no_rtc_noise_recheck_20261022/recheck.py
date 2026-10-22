"""Independently recompute paired noise statistics from archived model outputs.

CPU only. Does not load checkpoints, run inference, or filter any trajectories.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == "cpu", "Use JAX_PLATFORMS=cpu"
    sources = [("rtc", "rtc_padding_fix_20261021", "20000.json", "20000_predictions"),
               ("no_rtc", "no_rtc_comparison_20261021", "metrics.json", "predictions")]
    provenance_path = args.analysis_root / "no_rtc_comparison_20261021/provenance.json"
    contracts_path = args.analysis_root / "no_rtc_comparison_20261021/contracts.json"
    provenance = json.loads(provenance_path.read_text())
    assert provenance["model"]["rtc_training_max_delay"] == 0
    contracts = json.loads(contracts_path.read_text())
    norm_hashes = {v["norm_stats_sha256"] for v in contracts.values()}
    assert len(norm_hashes) == 1
    rows = []
    paired = {}
    evidence = {}
    for model, folder, metrics_name, predictions in sources:
        metrics_path = args.analysis_root / folder / metrics_name
        original = json.loads(metrics_path.read_text())
        original = original["rows"] if isinstance(original, dict) else original
        evidence[str(metrics_path)] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        for split in ("train", "heldout"):
            for steps in (5, 10, 20, 50):
                selected = [r for r in original if r["split"] == split and
                            r["mode"] == "no_prefix" and r["steps"] == steps]
                identities = sorted((r["id"], r["seed"]) for r in selected)
                assert len(selected) == 24 and len(set(identities)) == 24
                pairing = paired.setdefault((split, steps), identities)
                assert identities == pairing, "Both models must use identical observation/seed pairs"
                values = []
                saved_hash = hashlib.sha256()
                for r in sorted(selected, key=lambda r: (r["id"], r["seed"])):
                    path = args.analysis_root / folder / predictions / f'{r["id"]}_n{steps}_s{r["seed"]}_no_prefix.npz'
                    saved_hash.update(path.name.encode())
                    saved_hash.update(hashlib.sha256(path.read_bytes()).digest())
                    with np.load(path, allow_pickle=False) as arrays:
                        prediction = arrays["prediction"]
                    assert prediction.shape == (50, 14) and np.isfinite(prediction).all()
                    # The historical Policy.infer resets this key, then splits once.
                    key = jax.random.split(jax.random.key(r["seed"]))[1]
                    noise = np.asarray(jax.random.normal(key, (1, 50, 32)))[0][:, JOINTS]
                    y = np.diff(prediction[:, JOINTS], n=2, axis=0).astype(np.float64)
                    x = np.diff(noise, n=2, axis=0).astype(np.float64)
                    corr = np.array([np.corrcoef(y[:, j], x[:, j])[0, 1] for j in range(12)])
                    yc, xc = y-y.mean(axis=0), x-x.mean(axis=0)
                    gain = np.sum(yc*xc, axis=0) / np.sum(xc*xc, axis=0)
                    # This projection is a diagnostic statistic, never a deployed correction.
                    residual = yc-xc*gain
                    mean_corr = float(corr.mean())
                    assert abs(mean_corr-r["noise_curvature_corr_mean"]) < 1e-5
                    values.append(dict(correlation=mean_corr, gain_rad=float(gain.mean()),
                        mean_abs_d2_rad=float(abs(y).mean()),
                        fitted_residual_d2_rad=float(abs(residual).mean())))
                rows.append(dict(model=model, split=split, denoising_steps=steps,
                    observation_seed_pairs=len(selected), observations=12, independent_episodes=2,
                    prediction_files_sha256=saved_hash.hexdigest(),
                    observation_ids=sorted(set(r["id"] for r in selected)),
                    means={k: float(np.mean([v[k] for v in values])) for k in values[0]},
                    correlation_min=min(v["correlation"] for v in values),
                    correlation_max=max(v["correlation"] for v in values)))
    result = dict(method="Per-observation, per-joint second-difference correlations, then mean",
        evaluation="Recomputed from archived predictions and initial RNG seeds; no new inference",
        both_inference_modes="no_prefix", normalization_sha256=next(iter(norm_hashes)),
        no_rtc_checkpoint=provenance["checkpoint"], no_rtc_model=provenance["model"],
        source_files_sha256=evidence,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    for row in rows:
        print(row["model"], row["split"], row["denoising_steps"], row["means"])


if __name__ == "__main__":
    main()
