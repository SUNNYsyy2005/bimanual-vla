#!/bin/bash
#SBATCH --job-name=recap-synthetic-e2e
#SBATCH --partition=h100
#SBATCH --nodelist=h100-ksy-01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/DATA/disk0/sunny/experiments/recap-environment-20261101/records/train-%j.log
set -euo pipefail
export TASK_ROOT=/DATA/disk0/sunny/experiments/recap-environment-20261101
export REPO_PATH="$TASK_ROOT/source"
export PATH="$TASK_ROOT/env/bin:$PATH"
export CONDA_PREFIX="$TASK_ROOT/env"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_PATH"
export XDG_CACHE_HOME="$TASK_ROOT/cache"
export HF_HOME="$TASK_ROOT/cache/huggingface"
export TMPDIR="$TASK_ROOT/tmp"
export RAY_TMPDIR="$TASK_ROOT/tmp/ray-${SLURM_JOB_ID}"
export RAY_DEDUP_LOGS=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export http_proxy=http://10.0.0.32:8888
export https_proxy=http://10.0.0.32:8888
export no_proxy=localhost,127.0.0.1,10.0.0.0/8
export RAY_ADDRESS=local
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=268435456
mkdir -p "$RAY_TMPDIR"
if [ ! -d "$REPO_PATH" ]; then
  cp -a /DATA/disk0/sunny/rlinf_stack/src/RLinf-20260820 "$REPO_PATH"
fi
srun python "$TASK_ROOT/run_pipeline.py"
