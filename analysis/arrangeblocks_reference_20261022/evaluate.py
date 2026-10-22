"""Offline stock Pi05 evaluation; no RTC, filtering, robot, or service operations.

Until the original recipe is supplied, camera naming/RGB, delta joints, metric
grippers, quantile norm and image resizing are explicit reconstruction assumptions.
Evaluate both pad/stretch and action offsets 0/1, without selecting by error.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib, pi0_config
from openpi.policies import aloha_policy, policy
from openpi.shared import normalize, nnx_utils
from openpi.training.config import ModelTransformFactory

JOINTS = [i for i in range(14) if i % 7 != 6]
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def correlation(a, b):
    return float(np.mean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(12)]))


def metrics(pred, truth, state, noise):
    error = abs(pred - truth)
    hold = abs(state[None] - truth)
    p2, t2, n2 = (np.diff(a[:, JOINTS], n=2, axis=0) for a in (pred, truth, noise))
    result = {"joint_mae": float(error[:, JOINTS].mean()),
              "joint_rmse": float(np.sqrt(np.square(error[:, JOINTS]).mean())),
              "joint_abs_error_p95": float(np.quantile(error[:, JOINTS], .95)),
              "gripper_mae_mm": float(error[:, [6, 13]].mean()*1000),
              "pred_d2": float(abs(p2).mean()), "truth_d2": float(abs(t2).mean()),
              "noise_d2_correlation": correlation(p2, n2),
              "error_noise_d2_correlation": correlation(p2-t2, n2)}
    for label, start, stop in (("1_5", 0, 5), ("6_10", 5, 10),
                               ("11_20", 10, 20), ("21_50", 20, 50), ("all", 0, 50)):
        result[f"joint_mae_{label}"] = float(error[start:stop, JOINTS].mean())
        result[f"hold_mae_{label}"] = float(hold[start:stop, JOINTS].mean())
    result["per_joint_mae"] = error[:, JOINTS].mean(axis=0).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu", jax.devices()
    print("DEVICE", jax.devices(), flush=True)
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora",
                                 action_expert_variant="gemma_300m_lora")
    params = model_lib.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params, remove_extra_params=False)
    del params
    stats = normalize.load(args.checkpoint / "assets")
    model_transforms = ModelTransformFactory()(config)
    mask = transforms.make_bool_mask(6, -1, 6, -1)
    base = policy.Policy(model, transforms=[aloha_policy.AlohaInputs(adapt_to_pi=False),
        transforms.DeltaActions(mask), transforms.Normalize(stats, use_quantiles=True),
        *model_transforms.inputs], output_transforms=[*model_transforms.outputs,
        transforms.Unnormalize(stats, use_quantiles=True), transforms.AbsoluteActions(mask),
        aloha_policy.AlohaOutputs(adapt_to_pi=False)])
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
    bundle = np.load(args.bundle, allow_pickle=False)
    rows = []
    manifest = {"checkpoint": str(args.checkpoint), "checkpoint_step": 29999,
                "model_config": str(config), "bundle": str(args.bundle),
                "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_metadata": json.loads(args.bundle.with_suffix(".json").read_text()),
                "assumptions": {"image_rgb": True, "camera_mapping": "named cameras to named model streams",
                    "joint_semantics": "delta from observation state, not inter-step deltas",
                    "gripper_semantics": "absolute metres", "normalization": "checkpoint q01/q99",
                    "resize": ["pad", "stretch"], "action_offset": [0, 1],
                    "original_training_recipe_available": False},
                "rtc": False, "output_filter": False, "rows": rows}
    for resize in ("pad", "stretch"):
        for idx, frame in enumerate(bundle["frames"]):
            state = bundle["state"][idx]
            images = {}
            for cam in CAMS:
                image = bundle[f"images_{cam}"][idx]
                if resize == "stretch":
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
                images[cam] = image.transpose(2, 0, 1)
            inputs = base._input_transform(copy.deepcopy(dict(state=state, images=images,
                                                              prompt=str(bundle["instruction"]))))
            observation = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], inputs))
            for seed in (0, 1):
                key = jax.random.split(jax.random.key(seed))[1]
                noise = jax.random.normal(key, (1, 50, 32))
                for steps in (5, 10):
                    start = time.perf_counter()
                    normalized = np.asarray(sample(key, observation, num_steps=steps, noise=noise))[0]
                    elapsed = (time.perf_counter()-start)*1000
                    prediction = base._output_transform(dict(state=inputs["state"].copy(),
                                                             actions=normalized.copy()))["actions"]
                    assert prediction.shape == (50, 14) and np.isfinite(prediction).all()
                    name = f"{resize}_f{frame}_s{seed}_n{steps}"
                    np.savez_compressed(args.output / f"{name}.npz", prediction=prediction,
                        truth_offset0=bundle["truth_offset0"][idx], truth_offset1=bundle["truth_offset1"][idx],
                        noise=np.asarray(noise)[0, :, :14], normalized=normalized, state=state)
                    for offset in (0, 1):
                        truth = bundle[f"truth_offset{offset}"][idx]
                        rows.append(dict(frame=int(frame), resize=resize, seed=seed, steps=steps,
                            offset=offset, sampler_ms=elapsed,
                            **metrics(prediction, truth, state, np.asarray(noise)[0])))
            (args.output / "results.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
            print("PROGRESS", resize, int(frame), len(rows), flush=True)


if __name__ == "__main__":
    main()
