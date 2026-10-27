#!/usr/bin/env bash
# Standalone 4x4090 workstation only. Cluster serving must use sbatch.
set -euo pipefail
case "$(hostname)" in login-server|h100-*|h200-*) echo 'Use sbatch on the cluster' >&2; exit 1;; esac
workspace=${LINGBOT_WORKSPACE:-/home/sunny/lingbotvla}
checkpoint=${PIPER_CHECKPOINT:-$workspace/piper-sync/model/checkpoints/global_step_10700/hf_ckpt}
gpu=${PIPER_GPU_INDEX:-1}
free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
if (( free_mib < 22000 )); then
  echo "GPU $gpu has only $free_mib MiB free; choose an idle GPU with PIPER_GPU_INDEX" >&2
  exit 1
fi
source "$workspace/scripts/env.sh"
export QWEN3VL_PATH="$workspace/models/Qwen3-VL-4B-Instruct"
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4
exec "$workspace/envs/model/bin/python" -u "$(dirname "$0")/serve_piper.py" \
  --lingbot-root "$workspace/code" --checkpoint "$checkpoint" --port 8101 "$@"
