"""Distinguish an installed runtime from a ready-to-train model directory."""
import json
import os
from pathlib import Path

root = Path(os.environ["TASK_ROOT"])
required = [
    "models/siglip2-so400m-patch14-224/config.json",
    "models/siglip2-so400m-patch14-224/model.safetensors",
    "models/gemma-3-270m/config.json",
    "models/gemma-3-270m/model.safetensors",
    "models/gemma-3-270m/tokenizer_config.json",
    "models/pi05_libero/model.safetensors",
]
missing = [name for name in required if not (root / name).is_file()]
result = {"status": "incomplete" if missing else "ready", "missing": missing}
(root / "records/model-assets.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
if missing:
    print("Complete the listed local model assets before running the training test.")
    raise SystemExit(2)
