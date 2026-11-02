#!/bin/bash
set -euo pipefail
export TASK_ROOT=/mnt/nvme2n1p1_2T/sunny/experiments/recap-environment-20261101
export REPO_PATH="$TASK_ROOT/source"
export RECAP_PI05_BASE="$TASK_ROOT/models/pi05_pretrained"
export RECAP_LOW_MEMORY=1
export RECAP_GPU_MEMORY_FRACTION=0.58
export PATH="$TASK_ROOT/env/bin:$PATH"
export CONDA_PREFIX="$TASK_ROOT/env"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_PATH"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$TASK_ROOT/cache"
export HF_HOME="$TASK_ROOT/cache/huggingface"
export OPENPI_DATA_HOME="$TASK_ROOT/cache/openpi"
export TMPDIR="$TASK_ROOT/tmp"
export RAY_TMPDIR="/tmp/recap-ray-4090-$(date +%s)"
export RAY_ADDRESS=local
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=268435456
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export no_proxy=localhost,127.0.0.1,192.168.101.9,10.147.17.9
export NO_PROXY="$no_proxy"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$OPENPI_DATA_HOME/big_vision" "$TASK_ROOT/records"
test -f "$TASK_ROOT/records/staging-base-complete"
python "$TASK_ROOT/prepare_policy_assets.py"
python "$TASK_ROOT/check_model_assets.py"
touch "$TASK_ROOT/records/staging-complete"
cp -n /home/sunny/.cache/openpi/big_vision/paligemma_tokenizer.model "$OPENPI_DATA_HOME/big_vision/"
# This is the standalone workstation, outside the Slurm cluster. Select only
# a card with adequate free memory and low current load; do not stop any job.
GPU_INDEX=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
  | awk -F, '$2 >= 16500 && $3 <= 10 {print $1, $2}' | sort -k2,2nr | head -1 | awk '{print $1}')
test -n "$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
printf 'Standalone validation uses GPU %s with a per-worker memory cap.\n' "$GPU_INDEX"
exec nice -n 10 ionice -c 2 -n 7 python "$TASK_ROOT/run_pipeline.py"
