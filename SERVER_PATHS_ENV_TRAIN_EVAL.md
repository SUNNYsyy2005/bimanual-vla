# 4×4090 / H100 / H200 数据、环境、训练与评测运行手册

最后核验：**2026-08-05**

本文档记录当前 π0.5 + RoboTwin 项目在三类服务器上的实际目录、连接方式、Conda/CUDA 环境、常见训练命令、cuRobo-only 评测命令、日志与视频位置。

> 约束：只使用 π0.5；正式 RoboTwin 评测要求 `cuRobo-only + real_qpos + expert_order + video`。H100/H200 上的 GPU/重 CPU 任务必须使用 Slurm；`login-server` 只用于检查、编辑、传输和提交作业。

---

## 1. 快速索引

| 目标 | 主要用途 | 项目根目录 | 数据集根目录 | checkpoint 根目录 | 评测/日志根目录 |
|---|---|---|---|---|---|
| 本机工作区 | 文档、代码、下载后视频 | `/home/sunny/bimanual-vla` | 不保存完整训练集 | 不保存正式 checkpoint | `/home/sunny/bimanual-vla/eval_videos` |
| 4×4090 | RoboTwin 数据采集、cuRobo 正式评测、视频、训练 fallback | `/home/sunny/robotwin_ws/RoboTwin` | `/home/sunny/robotwin_datasets` | `/home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints` | `/home/sunny/robotwin_ws/RoboTwin/eval_result` |
| H100 | 正式训练优先节点 | `/DATA/disk0/sunny/openpi_h100/src/pi05` | `/DATA/disk0/sunny/.cache/huggingface/lerobot` | `/DATA/disk0/sunny/openpi_h100/checkpoints` | `/DATA/disk0/sunny/openpi_h100/slurm_logs` |
| H200 | H100 不可用时的正式训练节点 | `/DATA/disk0/sunny/openpi_h100/src/pi05`，但每个 H200 节点是独立副本 | `/DATA/disk0/sunny/.cache/huggingface/lerobot` | `/DATA/disk0/sunny/openpi_h100/checkpoints` | `/DATA/NAS/GPUServer/sunny/pi05_contract_v3_jobs` |
| NAS | 节点间传输，不直接训练 | `/DATA/NAS/GPUServer/sunny` | 仅 staging/export | checkpoint tar/export | job 日志和传输结果 |

### 当前 Put Bottles v3 数据集 ID

```text
put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned
```

数据属性：

```text
100 episodes
59798 frames
25 FPS
14D state/action
real_qpos
prompt 与 expert 执行顺序对齐
```

---

## 2. 连接方式

### 2.1 4×4090

公司内网：

```bash
ssh 4x4090
# 当前 alias 解析为：sunny@192.168.101.9
```

公网环境先启动本机 WireGuard。配置文件当前位于：

```text
/home/sunny/Downloads/6.conf
```

首次准备配置需要在本机执行；不要把 sudo 密码写入脚本或文档：

```bash
sudo cp /home/sunny/Downloads/6.conf /etc/wireguard/wg0.conf
sudo chmod 400 /etc/wireguard/wg0.conf
sudo chown root:root /etc/wireguard/wg0.conf
```

启动和验证：

```bash
sudo systemctl restart wg-quick@wg0
sleep 2
ping -c 4 10.0.200.100
ip addr show wg0
ssh sunny@10.0.200.100
```

关闭 VPN：

```bash
sudo systemctl stop wg-quick@wg0
```

建议在 `~/.ssh/config` 增加单独的 VPN alias，避免覆盖内网 alias：

```sshconfig
Host 4x4090-vpn
    HostName 10.0.200.100
    User sunny
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

然后：

```bash
ssh 4x4090-vpn
```

### 2.2 H100

H100 不能直接从本机访问，使用 `login-server` 跳转：

```bash
ssh login-server
ssh h100-ksy-01
```

本机 `~/.ssh/config` 应包含：

```sshconfig
Host login-server
    HostName 36.103.167.186
    User sunny
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host h100-ksy-01
    HostName h100-ksy-01
    User sunny
    ProxyJump login-server
```

日常应在 `login-server` 上检查资源并提交 Slurm，而不是 SSH 到 H100 后直接启动训练：

```bash
ssh login-server
resources
myquota
squeue -u sunny
sbatch <job.sbatch>
```

### 2.3 H200

```bash
ssh h200-ali-01   # 47.116.14.100
ssh h200-ali-02   # 120.55.15.209
```

H200 当前使用密码认证，不要把密码保存到 Git、脚本或 Markdown。正式任务仍通过 `login-server` 使用 Slurm 提交：

```bash
ssh login-server
sbatch -p h200 -w h200-ali-01 <job.sbatch>
```

### 2.4 文件传输

```bash
# 本机 -> 4×4090
scp <file> 4x4090:/home/sunny/<path>/

# 本机 -> H100，共享路径通过 ProxyJump
scp <file> h100-ksy-01:/DATA/disk0/sunny/<path>/

# 本机 -> H200，目标节点需要单独传一份
scp <file> h200-ali-01:/DATA/disk0/sunny/<path>/
```

大文件跨集群节点使用：

```text
/DATA/NAS/GPUServer/sunny
```

NAS 只用于传输/发布，不要直接从 NAS 训练。

### 2.5 `/DATA/disk0` 与 `/DATA/sync` 的说明

仓库通用 `AGENTS.md` 中存在以 `/DATA/sync/$USER` 为根目录的推荐模板；但 **2026-08-05 当前已经运行并验证的 Put Bottles v3 H100/H200 作业实际使用 `/DATA/disk0/sunny`**，包括项目、数据缓存和 checkpoint。

因此：

1. 维护当前 Job `7630` 或复现本批实验时，沿用本文列出的 `/DATA/disk0/sunny/...` 实际路径；
2. 新建通用项目时，可考虑 `/DATA/sync/$USER`，但必须先在目标节点执行 `ls -ld`、`df -h` 验证挂载、配额和节点间共享关系；
3. 不要在训练中途把现有实验目录从 `/DATA/disk0` 迁移到 `/DATA/sync`；
4. H200 节点无论使用哪一种根目录，都必须逐节点确认文件确实存在。

---

## 3. 4×4090：目录和环境

### 3.1 主要目录

```text
RoboTwin 项目：
/home/sunny/robotwin_ws/RoboTwin

π0.5 源码：
/home/sunny/robotwin_ws/RoboTwin/policy/pi05

LeRobot 数据集根目录：
/home/sunny/robotwin_datasets/lerobot_contract_v2

Put Bottles v3 数据集：
/home/sunny/robotwin_datasets/lerobot_contract_v2/
  put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned

OpenPI/LeRobot 默认缓存入口（当前是指向上面目录的软链接）：
/home/sunny/.cache/huggingface/lerobot/
  put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned

Dashboard 仓库：
/home/sunny/bimanual-vla

Dashboard workspace：
/home/sunny/.local/share/bimanual-vla-sim-dashboard

专家/采集输出：
/home/sunny/robotwin_data_logs

专家基线：
/home/sunny/robotwin_data_logs/expert_baseline_pb3_v3

π0.5 checkpoint 根目录：
/home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints

RoboTwin 评测根目录：
/home/sunny/robotwin_ws/RoboTwin/eval_result

Put Bottles 视频评测目录：
/home/sunny/robotwin_ws/RoboTwin/eval_result/put_bottles_dustbin/pi05/
  put_bottles_dustbin_piper_eval1_video
```

当前 H200 cp10000 已解包到：

```text
/home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints/
  pi05_put_bottles_dustbin_piper_lora_100_25hz_realqpos_v3_order_aligned/
  pi05-put-bottles-original3-v3-aligned-h200-7630/10000
```

### 3.2 Conda 环境

初始化：

```bash
source /home/sunny/miniconda3/etc/profile.d/conda.sh
conda env list
```

| 环境 | Python / Torch | 用途 |
|---|---|---|
| `RoboTwin2` | Python 3.10.20；Torch `2.4.1+cu121` | RoboTwin 仿真、CUDA 12.1 工具链、cuRobo 已编译 |
| `openpi_eval_cu121` | Python 3.11.15；Torch `2.4.1+cu121` | **首选 π0.5 + cuRobo-only 评测环境** |
| `openpi` | Python 3.11.15；Torch `2.13.0+cu130` | OpenPI 训练环境；无 cuRobo，不用于 cuRobo-only 评测 |
| `tsq-pilot` | legacy | 除非明确需要，否则不要使用 |

验证环境：

```bash
source /home/sunny/miniconda3/etc/profile.d/conda.sh
conda activate openpi_eval_cu121
python -V
python -c 'import torch, sapien, mplib, curobo, openpi; print(torch.__version__, torch.version.cuda)'
```

CUDA 12.1 编译器来自 `RoboTwin2`，不是系统 CUDA 13.3：

```text
/home/sunny/miniconda3/envs/RoboTwin2/bin/nvcc
```

验证：

```bash
/home/sunny/miniconda3/envs/RoboTwin2/bin/nvcc --version
# CUDA 12.1 / V12.1.66
```

### 3.3 cuRobo-only 评测环境变量

```bash
PROJECT=/home/sunny/robotwin_ws/RoboTwin
ENV=/home/sunny/miniconda3/envs/openpi_eval_cu121
CUDA121=/home/sunny/miniconda3/envs/RoboTwin2

export CUDA_HOME="$CUDA121"
export PATH="$ENV/bin:$CUDA121/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:$ENV/lib/python3.11/site-packages/torch/lib:$CUDA121/lib:$ENV/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT:$PROJECT/policy/pi05/src:$PROJECT/policy/pi05/packages/openpi-client/src:$PROJECT/envs/curobo/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME=/home/sunny/.cache
export HF_HOME=/home/sunny/.cache/huggingface
export TORCH_EXTENSIONS_DIR=/home/sunny/.cache/torch_extensions/openpi_eval_py311_cu121
export TORCH_CUDA_ARCH_LIST=8.9
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ROBOTWIN_REQUIRE_CUROBO=1
export ROBOTWIN_OBSERVATION_STATE_SOURCE=real_qpos
export ROBOTWIN_EVAL_PROMPT_MODE=expert_order
export ROBOTWIN_ACTION_FILTER=1
export ROBOTWIN_CLAMP_JOINT_LIMITS=1
export ROBOTWIN_MULTI_WAYPOINT_TOPP=1
```

不要改回系统 CUDA 13.3，也不要随意更换 `TORCH_EXTENSIONS_DIR`。

---

## 4. H100：目录和环境

### 4.1 实际使用目录

当前 H100 与 `login-server` 可见的正式训练目录：

```text
项目 staging 根目录：
/DATA/disk0/sunny/openpi_h100

π0.5 项目：
/DATA/disk0/sunny/openpi_h100/src/pi05

LeRobot 源码：
/DATA/disk0/sunny/openpi_h100/src/lerobot_src/a445d9c

HuggingFace / LeRobot 数据：
/DATA/disk0/sunny/.cache/huggingface/lerobot

OpenPI base checkpoint：
/DATA/disk0/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base

训练 checkpoint：
/DATA/disk0/sunny/openpi_h100/checkpoints

Slurm 脚本：
/DATA/disk0/sunny/openpi_h100/slurm

Slurm 日志：
/DATA/disk0/sunny/openpi_h100/slurm_logs
```

Put Bottles v3：

```text
数据集：
/DATA/disk0/sunny/.cache/huggingface/lerobot/
  put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned

norm stats：
/DATA/disk0/sunny/openpi_h100/src/pi05/assets/
  pi05_put_bottles_dustbin_piper_lora_100_25hz_realqpos_v3_order_aligned/
  put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned/norm_stats.json

checkpoint 根目录：
/DATA/disk0/sunny/openpi_h100/checkpoints/
  pi05_put_bottles_dustbin_piper_lora_100_25hz_realqpos_v3_order_aligned
```

Lift Pot 旧任务：

```text
数据集：
/DATA/disk0/sunny/.cache/huggingface/lerobot/lift_pot_piper

checkpoint：
/DATA/disk0/sunny/openpi_h100/checkpoints/pi05_lift_pot_piper_lora/
  pi05-lift-pot-piper150-h100-6415
```

H100 上曾建立的 RoboTwin 评测输出根目录：

```text
/DATA/disk0/sunny/robotwin_eval/RoboTwin/eval_result
```

RoboTwin 正式仿真评测目前仍优先放在 4×4090；H100 主要用于训练。

### 4.2 环境

当前训练脚本实际使用：

```text
Conda env：/home/sunny/miniconda3/envs/openpi
系统：Ubuntu 22.04
节点 CUDA：12.8
项目：/DATA/disk0/sunny/openpi_h100/src/pi05
```

常用环境变量：

```bash
ROOT=/DATA/disk0/sunny/openpi_h100
PROJECT=$ROOT/src/pi05
ENV=/home/sunny/miniconda3/envs/openpi
LEROBOT_SRC=$ROOT/src/lerobot_src/a445d9c

export HOME=/DATA/disk0/sunny
export XDG_CACHE_HOME=/DATA/disk0/sunny/.cache
export HF_HOME=/DATA/disk0/sunny/.cache/huggingface
export HF_DATASETS_CACHE=/DATA/disk0/sunny/.cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=/DATA/disk0/sunny/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/DATA/disk0/sunny/.cache/huggingface/transformers
export PIP_CACHE_DIR=/DATA/disk0/sunny/.cache/pip/cache
export TMPDIR=/DATA/disk0/sunny/.cache/tmp
export TORCH_HOME=/DATA/disk0/sunny/.cache/torch
export OPENPI_DATA_HOME=/DATA/disk0/sunny/.cache/openpi
export MPLCONFIGDIR=/DATA/disk0/sunny/.cache/matplotlib
export PYTHONNOUSERSITE=1
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$LEROBOT_SRC:$PROJECT/src:$PROJECT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
```

---

## 5. H200：目录和环境

### 5.1 独立文件系统

`h200-ali-01` 和 `h200-ali-02` 各自有独立的 `/home` 与 `/DATA/disk0`。即使绝对路径相同，也必须在目标 H200 节点单独准备环境、项目、数据和 base checkpoint。

当前 H200 正式训练 Job `7630` 使用：

```text
节点：h200-ali-01
GPU：2×H200
项目：/DATA/disk0/sunny/openpi_h100/src/pi05
环境：/home/sunny/miniconda3/envs/openpi
数据：/DATA/disk0/sunny/.cache/huggingface/lerobot/
      put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned
base：/DATA/disk0/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base
checkpoint：/DATA/disk0/sunny/openpi_h100/checkpoints/
  pi05_put_bottles_dustbin_piper_lora_100_25hz_realqpos_v3_order_aligned/
  pi05-put-bottles-original3-v3-aligned-h200-7630
```

训练日志发布到 NAS：

```text
/DATA/NAS/GPUServer/sunny/pi05_contract_v3_jobs
```

checkpoint staging/export：

```text
/DATA/NAS/GPUServer/sunny/pi05_contract_v3_checkpoints
```

已发布的 cp10000 tar：

```text
/DATA/NAS/GPUServer/sunny/pi05_contract_v3_checkpoints/
  pi05-put-bottles-original3-v3-aligned-h200-7630_cp10000_eval_ready_20260805_1508.tar
```

### 5.2 环境

```text
系统：Alibaba Cloud Linux 3
节点 CUDA：13.0
Conda env：/home/sunny/miniconda3/envs/openpi
训练环境变量：与 H100 训练脚本相同，但路径必须存在于对应 H200 节点自己的磁盘
```

不要假设 `h200-ali-01` 上准备好的环境会自动出现在 `h200-ali-02`。

---

## 6. 常见训练命令

### 6.1 训练前检查

在本机：

```bash
ssh login-server
```

在 `login-server`：

```bash
hostname
pwd
resources
myquota
squeue -u sunny
```

正式训练优先级：

```text
H100 > H200 > 4×4090 fallback
```

### 6.2 用 Slurm 验证 H100/H200 环境

不要在 `login-server` 运行 GPU 检查。可提交 1 CPU、1G 内存、5 分钟的轻量诊断作业：

```bash
cat > /tmp/check_openpi_env.sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=check_openpi_env
#SBATCH -p h100
#SBATCH -w h100-ksy-01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/DATA/disk0/sunny/openpi_h100/slurm_logs/%x_%j.out
#SBATCH --error=/DATA/disk0/sunny/openpi_h100/slurm_logs/%x_%j.err
set -euo pipefail
ENV=/home/sunny/miniconda3/envs/openpi
srun "$ENV/bin/python" - <<'PY'
import sys, jax
print('python', sys.version)
print('jax', jax.__version__)
print('devices', jax.devices())
PY
SBATCH

sbatch /tmp/check_openpi_env.sbatch
```

检查 H200 时一致地修改：

```bash
#SBATCH -p h200
#SBATCH -w h200-ali-01
#SBATCH --gres=gpu:h200:1
```

### 6.3 H100：Put Bottles v3

已存在脚本：

```text
准备数据/环境：
/DATA/disk0/sunny/openpi_h100/slurm/prepare_put_bottles_contract_v3_h100.sbatch

正式训练：
/DATA/disk0/sunny/openpi_h100/slurm/train_put_bottles_contract_v3_h100.sbatch
```

推荐带依赖提交：

```bash
cd /DATA/disk0/sunny/openpi_h100
prep=$(sbatch --parsable slurm/prepare_put_bottles_contract_v3_h100.sbatch)
sbatch --dependency=afterok:${prep} slurm/train_put_bottles_contract_v3_h100.sbatch
```

训练脚本参数：

```text
2×H100
batch_size=32
steps=30000
save_interval=5000
keep_period=10000
fsdp_devices=2
```

### 6.4 H200：Put Bottles v3

已存在脚本：

```text
准备 H200 独立文件系统：
/DATA/disk0/sunny/openpi_h100/slurm/prepare_put_bottles_contract_v3_h200.sbatch

正式训练：
/DATA/disk0/sunny/openpi_h100/slurm/train_put_bottles_contract_v3_h200.sbatch
```

提交：

```bash
cd /DATA/disk0/sunny/openpi_h100
prep=$(sbatch --parsable slurm/prepare_put_bottles_contract_v3_h200.sbatch)
sbatch --dependency=afterok:${prep} slurm/train_put_bottles_contract_v3_h200.sbatch
```

训练主体等价于：

```bash
$ENV/bin/python -u scripts/train.py \
  pi05_put_bottles_dustbin_piper_lora_100_25hz_realqpos_v3_order_aligned \
  --exp-name "pi05-put-bottles-original3-v3-aligned-h200-${SLURM_JOB_ID}" \
  --checkpoint-base-dir /DATA/disk0/sunny/openpi_h100/checkpoints \
  --assets-base-dir /DATA/disk0/sunny/openpi_h100/src/pi05/assets \
  --seed 0 \
  --batch-size 32 \
  --num-workers 8 \
  --num-train-steps 30000 \
  --log-interval 10 \
  --save-interval 5000 \
  --keep-period 10000 \
  --fsdp-devices 2 \
  --overwrite \
  --no-wandb-enabled
```

不要在 H100/H200 上手工设置 `CUDA_VISIBLE_DEVICES` 绕过 Slurm。

### 6.5 4×4090 fallback 训练

仅在 H100/H200 无法使用时采用。当前只允许使用 GPU0/1，启动前检查 GPU2/3 上的其他用户进程，不得停止或影响它们：

```bash
ssh 4x4090
nvidia-smi
```

当前 v3 fallback 的恢复训练命令：

```bash
ROOT=/home/sunny/robotwin_ws/RoboTwin/policy/pi05
PY=/home/sunny/miniconda3/envs/openpi/bin/python
ENV_LIB=/home/sunny/miniconda3/envs/openpi/lib
EXP=pb3_realqpos_v3_aligned_seed0_2x4090_20260805_112717

cd "$ROOT"
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  HF_HOME=/home/sunny/.cache/huggingface \
  XDG_CACHE_HOME=/home/sunny/.cache \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  LD_LIBRARY_PATH="$ENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$PY" -u scripts/train.py \
    pi05_put_bottles_dustbin_piper_lora_100_25hz_realqpos_v3_order_aligned \
    --exp-name "$EXP" \
    --resume \
    --seed 0 \
    --batch-size 16 \
    --num-workers 4 \
    --num-train-steps 60000 \
    --log-interval 10 \
    --save-interval 5000 \
    --keep-period 10000 \
    --fsdp-devices 2 \
    --no-wandb-enabled \
  >"/home/sunny/pi05_train_logs_${EXP}_resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

如果实验目录不存在，不要使用 `--resume`；首次训练改用新的 `--exp-name`，并根据需要使用 `--overwrite`。

---

## 7. 常见评测命令

### 7.1 区分两类“评测”

| 类型 | 入口 | 结果 | 是否是任务成功率 |
|---|---|---|---|
| Dataset held-out loss | Dashboard `/api/tasks/eval`、`server_4090/eval_heldout_loss.py` | loss JSON | 否 |
| RoboTwin rollout | `/home/sunny/robotwin_ws/RoboTwin/script/eval_policy.py` | Success/Fail、`_episode_results.jsonl`、MP4 | 是 |

Dashboard held-out loss 不会启动 SAPIEN、cuRobo 或任务 rollout，也不会验证 planner fallback。正式成功率必须使用本文后面的 RoboTwin rollout 命令。

Dashboard 常用位置：

```text
服务代码：/home/sunny/bimanual-vla/server_4090
任务记录：/home/sunny/.local/share/bimanual-vla-sim-dashboard/tasks/<TASK_ID>
任务日志：/home/sunny/.local/share/bimanual-vla-sim-dashboard/tasks/<TASK_ID>/task.log
held-out 结果：/home/sunny/.local/share/bimanual-vla-sim-dashboard/tasks/<TASK_ID>/result.json
视频索引目录：/home/sunny/.local/share/bimanual-vla-sim-dashboard/eval_videos
```

内网 Dashboard 地址：

```text
仿真：http://192.168.101.9:8091
实机：http://192.168.101.9:8090
```

### 7.2 正式 rollout 评测位置

RoboTwin + cuRobo 正式仿真评测优先在 4×4090 运行，原因是该机器已验证：

- `openpi_eval_cu121`；
- CUDA 12.1；
- cuRobo 编译扩展；
- SAPIEN/RoboTwin 渲染；
- 视频录制。

H100/H200 当前主要用于训练和 checkpoint 产出，不作为默认 RoboTwin 正式视频评测机器。

### 7.3 Put Bottles v3 评测 wrapper

4×4090 当前脚本：

```text
/home/sunny/run_pb3_v3_h200_eval_variant.sh
```

参数：

```text
GPU SEED CHECKPOINT_ID MAX_ARM_DELTA BLEND_STEPS PI0_STEP
```

例：cp10000、seed100013、GPU1、`max_delta=0.08`、不做 chunk blend、执行 50 actions/chunk：

```bash
ssh 4x4090
/home/sunny/run_pb3_v3_h200_eval_variant.sh \
  1 100013 10000 0.08 0 50
```

后台运行并保存日志：

```bash
seed=100013
gpu=1
log=/home/sunny/pb3_v3_eval_seed${seed}_$(date +%Y%m%d_%H%M%S).log
nohup /home/sunny/run_pb3_v3_h200_eval_variant.sh \
  "$gpu" "$seed" 10000 0.08 0 50 \
  >"$log" 2>&1 < /dev/null &
echo $! > "${log}.pid"
```

正式约束由脚本设置：

```text
ROBOTWIN_REQUIRE_CUROBO=1
ROBOTWIN_OBSERVATION_STATE_SOURCE=real_qpos
ROBOTWIN_EVAL_PROMPT_MODE=expert_order
ROBOTWIN_EVAL_START_SEED=<seed>
ROBOTWIN_PI05_FIXED_NOISE_SEED=<seed>
ROBOTWIN_ACTION_FILTER=1
ROBOTWIN_CLAMP_JOINT_LIMITS=1
ROBOTWIN_MULTI_WAYPOINT_TOPP=1
video task config=put_bottles_dustbin_piper_eval1_video
```

### 7.4 直接调用 `eval_policy.py`

完成第 3.3 节环境变量后：

```bash
cd /home/sunny/robotwin_ws/RoboTwin

$ENV/bin/python -u script/eval_policy.py \
  --config policy/pi05/deploy_policy.yml \
  --overrides \
  --task_name put_bottles_dustbin \
  --task_config put_bottles_dustbin_piper_eval1_video \
  --train_config_name pi05_put_bottles_dustbin_piper_lora_100_25hz_realqpos_v3_order_aligned \
  --model_name pi05-put-bottles-original3-v3-aligned-h200-7630 \
  --checkpoint_id 10000 \
  --ckpt_setting <unique-eval-name> \
  --seed 0 \
  --policy_name pi05 \
  --instruction_type seen \
  --pi0_step 50
```

`ckpt_setting` 必须唯一，否则不同实验的视频/结果可能混在一起。

### 7.5 查看结果

```bash
# 日志中的 episode 结果
grep -aE 'Success!|Fail!|\[EVAL_META\]|Success rate|Traceback|ERROR' "$log" \
  | tr '\r' '\n' | tail -30

# 找到最近视频
find /home/sunny/robotwin_ws/RoboTwin/eval_result \
  -type f -name '*.mp4' -printf '%T@ %s %p\n' \
  | sort -nr | head -20
```

标准结果目录结构：

```text
/home/sunny/robotwin_ws/RoboTwin/eval_result/<task>/pi05/<task_config>/
  <ckpt_setting>/<timestamp>/
    episode0.mp4
    _result.txt
    _episode_results.jsonl
```

### 7.6 把视频下载到本机

本机汇总目录：

```text
/home/sunny/bimanual-vla/eval_videos
```

无空格路径可直接：

```bash
scp '4x4090:/remote/path/episode0.mp4' \
  /home/sunny/bimanual-vla/eval_videos/output.mp4
```

远端目录含时间空格时，推荐用 `cat`：

```bash
ssh 4x4090 "cat '/remote/path/2026-08-05 15:53:05/episode0.mp4'" \
  > /home/sunny/bimanual-vla/eval_videos/output.mp4
```

---

## 8. 当前评测结果与视频目录

以下都是小样本诊断，不应当作正式总体成功率。

| checkpoint / 配置 | Seed 100002 | Seed 100013 | 小样本结果 |
|---|---:|---:|---:|
| 4×4090 cp5000，`pi0_step=10` | Fail | Fail | 0/2 |
| 4×4090 cp5000，`pi0_step=50` | Success | Fail | 1/2 |
| H200 cp10000，`max_delta=0.12, blend=0` | Fail | Fail | 0/2 |
| H200 cp10000，`max_delta=0.08, blend=0` | Fail | Success | 1/2 |
| H200 cp10000，`max_delta=0.10, blend=0` | Fail | Fail | 0/2 |
| H200 cp10000，`max_delta=0.08, blend=5` | Fail | Fail | 0/2 |

本机视频：

```text
cp5000：
/home/sunny/bimanual-vla/eval_videos/pb3_v3_cp5000_20260805_1440

cp10000：
/home/sunny/bimanual-vla/eval_videos/pb3_v3_cp10000_h200_20260805

专家基线：
/home/sunny/bimanual-vla/eval_videos/pb3_expert_baseline_seeds_20260805

旧 v2 cp16000：
/home/sunny/bimanual-vla/eval_videos/pb3_cp16000_4x4090_20260805_090955
```

cp10000 成功视频：

```text
/home/sunny/bimanual-vla/eval_videos/pb3_v3_cp10000_h200_20260805/
  success_seed100013_pi0step50_delta0.08_blend0.mp4
```

cp10000 机器可读汇总：

```text
/home/sunny/bimanual-vla/eval_videos/pb3_v3_cp10000_h200_20260805/results.json
```

---

## 9. checkpoint 从 H200 送到 4×4090

### 9.1 H200 -> NAS

H200 checkpoint 通常只打包评测所需的：

```text
params
assets
_CHECKPOINT_METADATA
```

H200 作业中：

```bash
SRC=/DATA/disk0/sunny/openpi_h100/checkpoints/<config>/<experiment>/<step>
DST=/DATA/NAS/GPUServer/sunny/pi05_contract_v3_checkpoints/<name>.tar

tar -C "$SRC" -cf /DATA/disk0/sunny/openpi_h100/staging/<name>.tar \
  params assets _CHECKPOINT_METADATA
cp /DATA/disk0/sunny/openpi_h100/staging/<name>.tar "$DST.partial"
sha256sum "$DST.partial"
mv "$DST.partial" "$DST"
touch "$DST.TRANSFER_COMPLETE"
```

该过程属于大文件处理，必须放在 H200 `sbatch` 中执行，不能在 `login-server` 直接打包。

### 9.2 NAS/login-server -> 本机 -> 4×4090

使用 `rsync --partial --append-verify` 可断点续传：

```bash
rsync -ah --info=progress2 --partial --append-verify \
  login-server:/DATA/NAS/GPUServer/sunny/pi05_contract_v3_checkpoints/<name>.tar \
  /home/sunny/Downloads/<name>.tar

rsync -ah --info=progress2 --partial --append-verify \
  /home/sunny/Downloads/<name>.tar \
  4x4090:/home/sunny/<name>.tar
```

4×4090 解包到对应模型目录：

```bash
mkdir -p /home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints/<config>/<experiment>/<step>
tar -C /home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints/<config>/<experiment>/<step> \
  -xf /home/sunny/<name>.tar
```

验证：

```bash
test -s <step>/params/_METADATA
test -f <step>/_CHECKPOINT_METADATA
find <step>/assets -name norm_stats.json -type f -size +0c
```

---

## 10. 监控与故障排查

### 10.1 Slurm

```bash
ssh login-server
squeue -u sunny
scontrol show job <jobid>
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,AllocTRES%80
scancel <jobid>  # 只能取消自己的作业
```

日志：

```bash
tail -f /DATA/disk0/sunny/openpi_h100/slurm_logs/<job>.out
tail -f /DATA/NAS/GPUServer/sunny/pi05_contract_v3_jobs/<job>.out
```

### 10.2 4×4090

```bash
ssh 4x4090
nvidia-smi
ps -fu sunny
```

当前约束：只使用 GPU0/1；不要操作 GPU2/3 上的其他用户进程。

### 10.3 数据/checkpoint 完整性

```bash
# 数据集
python - <<'PY'
import json
from pathlib import Path
p = Path('<dataset>/meta/info.json')
print(json.loads(p.read_text()))
PY

test -f <dataset>/TRANSFER_COMPLETE

# checkpoint
test -s <checkpoint>/params/_METADATA
test -f <checkpoint>/_CHECKPOINT_METADATA
find <checkpoint>/assets -name norm_stats.json -type f -size +0c
```

### 10.4 正式评测必要检查

日志中至少确认：

```text
ROBOTWIN_REQUIRE_CUROBO=1
real_qpos
expert_order
Model loaded OK
cuRobo 没有 planner fallback
存在 episode0.mp4
存在 [EVAL_META]
```

---

## 11. 不要做的事

1. 不要在 `login-server` 训练、推理、编译 cuRobo、解压大 checkpoint 或处理大数据集。
2. 不要在 H100/H200 绕过 Slurm 手工占 GPU。
3. 不要从 `/DATA/NAS/GPUServer` 直接训练。
4. 不要假设两个 H200 节点共享 `/home` 或 `/DATA/disk0`。
5. 不要把 4×4090 的 `openpi` 环境用于 cuRobo-only 评测；它没有 cuRobo。
6. 不要改用系统 CUDA 13.3 编译当前 cuRobo；使用 `RoboTwin2` 的 CUDA 12.1。
7. 不要删除、停止或影响其他用户的 GPU2/3 任务。
8. 不要在任何仓库、脚本、日志或 Markdown 中保存 SSH/sudo 明文密码。
