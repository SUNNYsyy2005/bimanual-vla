#!/usr/bin/env bash
# Run from this repository. Uses an independent Conda environment.
set -euo pipefail
PI05_CONDA_ROOT=${PI05_CONDA_ROOT:-/home/user/miniconda3}
PI05_SOURCE=${PI05_SOURCE:-/home/user/src/openpi-5060}
PI05_ENV=${PI05_ENV:-pi05_5060}
source "$PI05_CONDA_ROOT/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "$PI05_ENV"; then
    conda create -n "$PI05_ENV" python=3.11 pip -y
fi
conda activate "$PI05_ENV"
python -m pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128
python -m pip install -r scripts/pi05_local/requirements.txt
python - "$PI05_SOURCE" <<'PY'
from pathlib import Path
import shutil
import sys
import transformers
source = Path(sys.argv[1]) / 'src/openpi/models_pytorch/transformers_replace'
assert transformers.__version__ == '4.53.2'
assert source.is_dir(), source
shutil.copytree(source, Path(transformers.__file__).parent, dirs_exist_ok=True)
import torch
print('torch', torch.__version__, 'GPU', torch.cuda.get_device_name())
x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
print('Blackwell BF16 CUDA matmul passed')
PY
