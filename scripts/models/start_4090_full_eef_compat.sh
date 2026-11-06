#!/usr/bin/env bash
set -euo pipefail
case "$(hostname)" in login-server|h100-*|h200-*) echo 'Use sbatch on cluster nodes' >&2; exit 1;; esac
package=${FULL_EEF_PACKAGE:-/home/sunny/Desktop/sunny_openpi_foldbox_full_transfer}
repo=${FULL_EEF_REPO:-/home/sunny/bimanual-vla}
case "${1:?expected abc20k or pi05_base}" in
  abc20k)
    gpu=${FULL_EEF_GPU:-2}; port=${FULL_EEF_PORT:-8220}
    checkpoint="$package/pi05/from_abc20k/full/piper_foldbox_eef_full_from_abc20k_1h200_b16_w4_20261103210353/step_10000" ;;
  pi05_base)
    gpu=${FULL_EEF_GPU:-1}; port=${FULL_EEF_PORT:-8221}
    checkpoint="$package/pi05/from_pi05_base/full/piper_foldbox_eef_full_from_pi05_base_1h200_b16_w4_20261103223908/step_10000" ;;
  *) echo "unknown variant: $1" >&2; exit 2 ;;
esac
free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
if (( free_mib < 13000 )); then
  echo "GPU $gpu has only $free_mib MiB free; choose another GPU with FULL_EEF_GPU" >&2
  exit 1
fi
source /home/sunny/miniconda3/etc/profile.d/conda.sh
conda activate openpi
export CUDA_VISIBLE_DEVICES="$gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="/home/sunny/openpi-pytorch-official/src:$package/training_configs/common/piperdem:$repo"
cd /home/sunny/openpi-pytorch-official
exec python -u "$repo/scripts/models/serve_full_eef_compat.py" \
  --checkpoint "$checkpoint" --port "$port" \
  --bridge-config "$repo/scripts/models/full_eef_bridge.json"
