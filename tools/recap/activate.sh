#!/bin/bash
# Source this file from either installed standalone or H100 runtime.
export TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$TASK_ROOT/source"
export PATH="$TASK_ROOT/env/bin:$PATH"
export CONDA_PREFIX="$TASK_ROOT/env"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$TASK_ROOT/cache"
export HF_HOME="$TASK_ROOT/cache/huggingface"
export OPENPI_DATA_HOME="$TASK_ROOT/cache/openpi"
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PIP_REQUIRE_VIRTUALENV=false
if [ -d "$TASK_ROOT/models/pi05_pretrained" ]; then
  export RECAP_PI05_BASE="$TASK_ROOT/models/pi05_pretrained"
fi
