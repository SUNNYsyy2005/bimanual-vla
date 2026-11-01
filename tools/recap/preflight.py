"""Check actual CUDA and all model/data entrypoint imports in the allocated job."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import transformers
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
from transformers import SiglipVisionModel

root = Path(os.environ["TASK_ROOT"])
modules = ["openpi.models_pytorch.pi0_pytorch", "rlinf.workers.sft.fsdp_value_sft_worker",
           "rlinf.workers.sft.fsdp_cfg_worker", "rlinf.data.datasets.recap.value_dataset"]
for name in modules:
    module = importlib.import_module(name)
    print(name, module.__file__, flush=True)
cpu_only = os.environ.get("RECAP_CPU_PREFLIGHT") == "1"
if not cpu_only:
    assert torch.cuda.is_available(), "CUDA unavailable in Slurm GPU allocation"
    assert torch.cuda.device_count() == 1, "Expected exactly the allocated GPU"
    sample = torch.randn(64, 64, device="cuda")
    assert torch.isfinite(sample @ sample).all()
versions = {}
for name in ["torch", "torchvision", "transformers", "rlinf-transformer-openpi", "rlinf-openpi", "lerobot", "ray"]:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
result = {"versions": versions, "transformers_runtime": transformers.__version__,
          "cuda": torch.version.cuda, "gpu": None if cpu_only else torch.cuda.get_device_name(0), "imports": modules,
          "model_classes": [Gemma3ForCausalLM.__name__, SiglipVisionModel.__name__]}
filename = "preflight-cpu.json" if cpu_only else "preflight.json"
(root / "records" / filename).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
