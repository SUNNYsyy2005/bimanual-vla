#!/usr/bin/env python3
"""Legacy Policy server for axiboai/pi05-piper-bimanual-v1.

This standalone entrypoint does not install model-side Real-Time Chunking.
For real-robot deployment with RTC, use Dashboard's
``server_4090/openpi_single_arm.py serve --rtc-enabled`` instead.

Run on 192.168.101.9 (4x4090 server):
    ssh 4x4090
    export LD_LIBRARY_PATH=/home/sunny/miniconda3/envs/openpi/lib:$LD_LIBRARY_PATH
    conda activate openpi
    cd /home/sunny/robotwin_ws/RoboTwin/policy/pi05
    python /home/sunny/bimanual-vla/serve_piper.py

First run converts LeRobot norm stats → openpi format and saves them.
Subsequent runs load them directly.
"""

import logging
import pathlib
import sys

import numpy as np
import safetensors.torch
from safetensors import safe_open

OPENPI_DIR = "/home/sunny/robotwin_ws/RoboTwin/policy/pi05/src"
CHECKPOINT  = "/home/sunny/checkpoints/pi05-piper-bimanual-v1"
ASSET_ID    = "piper-bimanual-v1"
PORT        = 8000

sys.path.insert(0, OPENPI_DIR)

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_cfg
from openpi.models_pytorch import pi0_pytorch
from openpi.policies import policy as _policy
from openpi.policies.aloha_policy import AlohaInputs, AlohaOutputs
from openpi.serving import websocket_policy_server
import openpi.transforms as transforms
from openpi.shared import normalize as _normalize
from openpi.shared.normalize import NormStats
from openpi.training.config import ModelTransformFactory


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("serve_piper")


# ── Norm stats ──────────────────────────────────────────────────────────────

def _load_lerobot_norm_stats(safetensors_path: str) -> dict[str, NormStats]:
    """Convert LeRobot preprocessor safetensors → openpi NormStats dict.

    openpi Normalize uses keys "state" and "actions"
    (after AlohaInputs has renamed observation.state → state).
    """
    raw: dict[str, np.ndarray] = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k).numpy()

    def extract(prefix: str) -> NormStats:
        return NormStats(
            mean=raw[f"{prefix}.mean"],
            std=raw[f"{prefix}.std"],
            q01=raw[f"{prefix}.q01"],
            q99=raw[f"{prefix}.q99"],
        )

    return {
        "state":   extract("observation.state"),
        "actions": extract("action"),
    }


def ensure_openpi_norm_stats(checkpoint_dir: str, asset_id: str) -> dict[str, NormStats]:
    """Load from cache or convert from LeRobot format on first run."""
    assets_dir = pathlib.Path(checkpoint_dir) / "assets"
    stat_dir   = assets_dir / asset_id

    if (stat_dir / "norm_stats.json").exists():
        logger.info("Loading cached openpi norm stats from %s", stat_dir)
        return _normalize.load(stat_dir)

    logger.info("Converting LeRobot norm stats → openpi format …")
    src = pathlib.Path(checkpoint_dir) / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    if not src.exists():
        raise FileNotFoundError(f"Norm stats not found: {src}")

    stats = _load_lerobot_norm_stats(str(src))
    stat_dir.mkdir(parents=True, exist_ok=True)
    _normalize.save(stat_dir, stats)
    logger.info("Saved openpi norm stats to %s", stat_dir)
    return stats


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(checkpoint_dir: str):
    weight_path = pathlib.Path(checkpoint_dir) / "model.safetensors"
    logger.info("Loading PyTorch model from %s …", weight_path)
    model_cfg   = _pi0_cfg.Pi0Config(pi05=True)
    model       = pi0_pytorch.PI0Pytorch(config=model_cfg)
    safetensors.torch.load_model(model, str(weight_path))
    logger.info("Model loaded.")
    return model, model_cfg


# ── Policy ───────────────────────────────────────────────────────────────────

def create_policy(model, model_cfg, norm_stats: dict[str, NormStats]) -> _policy.Policy:
    # model_transforms: tokenize prompt, pad state to 32D, format images for PaliGemma
    model_transform = ModelTransformFactory()(model_cfg)

    return _policy.Policy(
        model,
        transforms=[
            # Client sends {"state":(14,), "images":{"cam_high":…, "cam_left_wrist":…, "cam_right_wrist":…}, "prompt":…}
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


# ── Server ───────────────────────────────────────────────────────────────────

def main():
    norm_stats = ensure_openpi_norm_stats(CHECKPOINT, ASSET_ID)
    model, model_cfg = load_model(CHECKPOINT)
    policy = create_policy(model, model_cfg, norm_stats)

    logger.info("Starting WebSocket policy server on port %d …", PORT)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=PORT,
        metadata={},
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
