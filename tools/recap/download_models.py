"""Fetch official value backbones; authentication stays in the HF runtime."""
import json
import os
from pathlib import Path

for token_path in [Path.home() / ".cache/huggingface/token",
                   Path("/DATA/disk0/sunny/.cache/huggingface/token")]:
    if token_path.is_file():
        os.environ.setdefault("HF_TOKEN_PATH", str(token_path))
        break

from huggingface_hub import snapshot_download

root = Path(os.environ["TASK_ROOT"])
models = ["google/siglip2-so400m-patch14-224", "google/gemma-3-270m"]
model_filter = os.environ.get("RECAP_MODEL_FILTER")
if model_filter:
    models = [repo for repo in models if model_filter in repo]
record_path = root / "records" / ("models.json" if not model_filter else "model-gemma.json")
records = []
for repo in models:
    target = root / "models" / repo.split("/")[-1]
    try:
        snapshot_download(repo, local_dir=target, max_workers=4,
                          allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.txt"])
        records.append({"repo": repo, "path": str(target), "status": "downloaded"})
    except Exception as exc:
        # Do not echo request headers, auth tokens or redirect URLs.
        chain = []
        cause = exc
        while cause is not None and len(chain) < 5:
            response = getattr(cause, "response", None)
            chain.append({"type": type(cause).__name__, "http_status": getattr(response, "status_code", None),
                          "error_code": response.headers.get("X-Error-Code") if response is not None else None})
            cause = cause.__cause__ or cause.__context__
        records.append({"repo": repo, "status": "failed", "error_type": type(exc).__name__, "causes": chain})
    record_path.write_text(json.dumps(records, indent=2))
    print(json.dumps(records[-1]), flush=True)
if any(r["status"] != "downloaded" for r in records):
    raise SystemExit(1)
