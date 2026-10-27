#!/usr/bin/env bash
set -euo pipefail
PI05_REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PI05_REPO"
PI05_PYTHON=${PI05_PYTHON:-/home/user/miniconda3/envs/pi05_5060/bin/python}
exec "$PI05_PYTHON" scripts/pi05_local/benchmark.py \
    --source /home/user/src/openpi-5060 \
    --checkpoint /home/user/models/pi05_5060/piper_15000_pt \
    --tokenizer /home/user/models/pi05_5060/paligemma_tokenizer.model \
    --recording deployment_runs/20261021T113041.846378+0800_109455 \
    --mode bf16 --steps 5 --repeats 4 \
    --output analysis/pi05_5060/latest.json "$@"
