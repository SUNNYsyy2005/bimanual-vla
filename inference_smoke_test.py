"""Offline inference smoke test for axiboai/pi05-piper-bimanual-v1.

Runs on the 4090 server (no simulation, no real robot needed).
Verifies: model loads, shapes correct, action values in range, timing.

Usage:
    ssh 4x4090
    conda activate openpi
    export LD_LIBRARY_PATH=/home/sunny/miniconda3/envs/openpi/lib:$LD_LIBRARY_PATH
    cd /home/sunny/robotwin_ws/RoboTwin/policy/pi05
    python /home/sunny/bimanual-vla/inference_smoke_test.py
"""

import sys, time, pathlib
import numpy as np

OPENPI = "/home/sunny/robotwin_ws/RoboTwin/policy/pi05/src"
CKPT   = "/home/sunny/checkpoints/pi05-piper-bimanual-v1"
sys.path.insert(0, OPENPI)

# ── 1. Verify norm stats conversion (same as serve_piper.py) ─────────────────
from openpi.shared import normalize as _normalize
from openpi.shared.normalize import NormStats
from safetensors import safe_open

def load_or_convert_norm_stats():
    stat_dir = pathlib.Path(CKPT) / "assets" / "piper-bimanual-v1"
    if (stat_dir / "norm_stats.json").exists():
        print("  [norm_stats] loading cached")
        return _normalize.load(stat_dir)
    print("  [norm_stats] converting from LeRobot safetensors...")
    src = pathlib.Path(CKPT) / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    raw = {}
    with safe_open(str(src), framework="pt") as f:
        for k in f.keys(): raw[k] = f.get_tensor(k).numpy()
    stats = {
        "state":   NormStats(mean=raw["observation.state.mean"],
                             std=raw["observation.state.std"],
                             q01=raw["observation.state.q01"],
                             q99=raw["observation.state.q99"]),
        "actions": NormStats(mean=raw["action.mean"],
                             std=raw["action.std"],
                             q01=raw["action.q01"],
                             q99=raw["action.q99"]),
    }
    stat_dir.mkdir(parents=True, exist_ok=True)
    _normalize.save(stat_dir, stats)
    print(f"  [norm_stats] saved to {stat_dir}")
    return stats

# ── 2. Load model ─────────────────────────────────────────────────────────────
import safetensors.torch
from openpi.models import pi0_config as _pi0_cfg
from openpi.models_pytorch import pi0_pytorch
from openpi.policies import policy as _policy
from openpi.policies.aloha_policy import AlohaInputs, AlohaOutputs
import openpi.transforms as transforms
from openpi.training.config import ModelTransformFactory

def load_policy(norm_stats):
    print("  [model] loading weights (~8.8 GB, takes ~30s)...")
    t0 = time.time()
    model_cfg = _pi0_cfg.Pi0Config(pi05=True)
    model = pi0_pytorch.PI0Pytorch(config=model_cfg)

    # LeRobot saves with "model." prefix; openpi PI0Pytorch expects no prefix.
    # Vision tower: LeRobot uses vision_tower.* but openpi expects vision_tower.vision_model.*
    import torch
    raw = safetensors.torch.load_file(str(pathlib.Path(CKPT) / "model.safetensors"))
    remapped = {}
    for k, v in raw.items():
        k = k.removeprefix("model.")
        k = k.replace("vision_tower.embeddings.",    "vision_tower.vision_model.embeddings.")
        k = k.replace("vision_tower.encoder.",       "vision_tower.vision_model.encoder.")
        k = k.replace("vision_tower.post_layernorm.","vision_tower.vision_model.post_layernorm.")
        remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (first 3: {missing[:3]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")
    if not missing and not unexpected:
        print("  [ok] all keys matched")

    print(f"  [model] loaded in {time.time()-t0:.1f}s")

    model_transform = ModelTransformFactory()(model_cfg)
    policy = _policy.Policy(
        model,
        transforms=[
            AlohaInputs(adapt_to_pi=False),
            transforms.Normalize(norm_stats, use_quantiles=True),
            *model_transform.inputs,
        ],
        output_transforms=[
            *model_transform.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=True),
            AlohaOutputs(adapt_to_pi=False),
        ],
        is_pytorch=True,
        pytorch_device="cuda",
    )
    return policy

# ── 3. Build fake observation ─────────────────────────────────────────────────
def fake_obs(norm_stats):
    """Generate a plausible observation using q50 (median) state."""
    # Use median state values so normalization is near zero (not clipped)
    from openpi.shared import normalize as N
    s = norm_stats["state"]
    state_median = (s.q01 + s.q99) / 2.0   # rough median

    return {
        "state": state_median.astype(np.float32),
        "images": {
            "cam_high":        np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8),
            "cam_left_wrist":  np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8),
        },
        "prompt": "pick up the red cube and place it in the box",
    }

# ── 4. Run test ───────────────────────────────────────────────────────────────
JOINT_LIMITS = np.array([
    [-2.6179,  2.6179],
    [ 0.0000,  3.1400],
    [-2.9670,  0.0000],
    [-1.7450,  1.7450],
    [-1.2200,  1.2200],
    [-2.0944,  2.0944],
])

def check_action(actions_14d, label=""):
    ok = True
    for side, offset in [("left", 0), ("right", 7)]:
        joints = actions_14d[offset:offset+6]
        gripper = actions_14d[offset+6]
        for i, (lo, hi) in enumerate(JOINT_LIMITS):
            if not (lo <= joints[i] <= hi):
                print(f"  [WARN]{label} {side} j{i+1}={joints[i]:.3f} out of [{lo},{hi}]")
                ok = False
        if not (0.0 <= gripper <= 0.07):
            print(f"  [WARN]{label} {side} gripper={gripper:.4f} out of [0, 0.07]")
            ok = False
    return ok

def main():
    print("\n=== π0.5 Piper Bimanual Smoke Test ===\n")

    print("[1/3] Loading norm stats...")
    norm_stats = load_or_convert_norm_stats()
    print(f"  state  q01={norm_stats['state'].q01[:4].round(3)}...")
    print(f"  action q01={norm_stats['actions'].q01[:4].round(3)}...")

    print("\n[2/3] Loading policy...")
    policy = load_policy(norm_stats)

    print("\n[3/3] Running inference (10 rounds)...")
    obs = fake_obs(norm_stats)

    # warmup
    print("  warmup...")
    result = policy.infer(obs)
    actions_all = result["actions"]   # (50, 14) after AlohaOutputs

    # timing
    times = []
    for i in range(10):
        t0 = time.time()
        result = policy.infer(obs)
        times.append((time.time() - t0) * 1000)

    print(f"\n  inference time: avg={np.mean(times):.0f}ms  min={np.min(times):.0f}ms  max={np.max(times):.0f}ms")
    print(f"  action chunk shape: {result['actions'].shape}")

    # Check first, middle, last step
    actions_all = result["actions"]
    print("\n  Action value check (first 3 steps):")
    all_ok = True
    for step in range(3):
        a = actions_all[step]
        ok = check_action(a, f" step{step}")
        if ok:
            print(f"  step {step}: left_joints={a[:6].round(3)}  l_gripper={a[6]:.4f}m"
                  f"  right_joints={a[7:13].round(3)}  r_gripper={a[13]:.4f}m  [OK]")
        else:
            all_ok = False

    print(f"\n  Action range across full chunk (50 steps):")
    print(f"  left  joints min={actions_all[:,:6].min():.3f}  max={actions_all[:,:6].max():.3f}")
    print(f"  right joints min={actions_all[:,7:13].min():.3f}  max={actions_all[:,7:13].max():.3f}")
    print(f"  left  gripper min={actions_all[:,6].min():.4f}m  max={actions_all[:,6].max():.4f}m")
    print(f"  right gripper min={actions_all[:,13].min():.4f}m  max={actions_all[:,13].max():.4f}m")

    print(f"\n{'✅ All action values in valid range.' if all_ok else '⚠ Some action values out of range — check above.'}")
    print("\n=== Done ===\n")

if __name__ == "__main__":
    main()
