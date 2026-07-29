# AGENTS.md

Operational rules for AI agents on this Slurm GPU cluster with quota-aware scheduling.

Last updated: 2026-07-29

## Hard Rules

1. Use `login-server` only for lightweight work: editing files, reading logs, Git operations, checking resources/quotas, and submitting Slurm jobs.
2. Do not train, run inference, compile at scale, decompress large archives, or process large datasets on `login-server`.
3. GPU, heavy CPU, and high-memory jobs must be submitted with `sbatch`.
4. Use `srun` only inside `sbatch` scripts to launch the actual workload.
5. Do not bypass Slurm with manual `CUDA_VISIBLE_DEVICES=...`.
6. Do not use system Python or system pip. Use Conda / Miniconda / Mambaforge.
7. Do not use `sudo`.
8. Do not modify other users' files or cancel other users' jobs.
9. Do not run proxy software on GPU nodes. Do not use the server as a Proxy / VPN / jump host.

## Nodes

| Node | Role | GPU | Slurm options |
|---|---|---|---|
| `login-server` | login / submit | none | lightweight work only |
| `h100-ksy-01` | compute | H100 | `-p h100 -w h100-ksy-01` |
| `h200-ali-01` | compute | H200 | `-p h200 -w h200-ali-01` |
| `h200-ali-02` | compute | H200 | `-p h200 -w h200-ali-02` |

Match GPU type and node:

```bash
# H100
#SBATCH -p h100
#SBATCH -w h100-ksy-01
#SBATCH --gres=gpu:h100:<count>

# H200
#SBATCH -p h200
#SBATCH -w h200-ali-01   # or h200-ali-02
#SBATCH --gres=gpu:h200:<count>
```

## Quotas And Scheduling

For student accounts, check quota before submitting nontrivial jobs:

```bash
myquota
```

Each job can only apply for a maximum of 512G of memory.

GPU multipliers:

| GPU | Multiplier |
|---|---|
| H100 | 1.0 |
| H200 | 1.2 |

Recent high usage lowers scheduling priority. Company/business jobs have priority under resource pressure. Student jobs may be requeued, so long training jobs must support checkpoints and resume.

## Pre-Run Checks

Before heavy jobs, check:

```bash
hostname
pwd
resources
myquota
```

- `resources` / `check_resources.sh`: available cluster resources and queue status.
- `myquota`: remaining quota for student accounts.
- If running on `login-server`, do only lightweight work. Heavy jobs must be submitted with `sbatch`.

## Storage

Use `/DATA/sync/$USER` for projects, datasets, environments, caches, models, checkpoints, and outputs. Avoid large files under `/home/$USER`.

`login-server` mounts `h100-ksy-01`'s `/home` and `/DATA/sync` through NFS. Files prepared on `login-server` under these paths are immediately visible to H100 jobs.

`h200-ali-01` and `h200-ali-02` have independent `/home` and `/DATA/sync` filesystems. Before running on either H200 node, make sure the environment, project files, data, models, and caches exist on that exact node. You may submit CPU/memory-only `sbatch` jobs on H200 nodes for setup.

Use NAS for sharing and transfer, not direct training:

```bash
/DATA/NAS/GPUServer
```

## Environment

Recommended cache variables:

```bash
export XDG_CACHE_HOME=/DATA/sync/$USER/.cache
export PIP_CACHE_DIR=/DATA/sync/$USER/.cache/pip/cache
export TMPDIR=/DATA/sync/$USER/.cache/pip/tmp
export HF_HOME=/DATA/sync/$USER/.cache/huggingface
```

Recommended conda path:

```bash
/DATA/sync/$USER/miniconda3
```

H100:

- Ubuntu 22.04
- CUDA 12.8
- Network goes through `login-server`; proxy environment variables already exist.
- Do not start proxy software on H100.

H200 nodes:

- Alibaba Cloud Linux 3
- CUDA 13.0
- Separate environment setup is required per H200 node.

## sbatch Template

```bash
#!/bin/bash
#SBATCH --job-name=my_train
#SBATCH -p h100
#SBATCH -w h100-ksy-01
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err

set -euo pipefail

cd /DATA/sync/$USER/projects/<project_name>
mkdir -p logs outputs checkpoints

export XDG_CACHE_HOME=/DATA/sync/$USER/.cache
export PIP_CACHE_DIR=/DATA/sync/$USER/.cache/pip/cache
export TMPDIR=/DATA/sync/$USER/.cache/pip/tmp
export HF_HOME=/DATA/sync/$USER/.cache/huggingface

source /DATA/sync/$USER/miniconda3/bin/activate
conda activate myenv

srun python train.py
```

For H200, change `-p`, `-w`, and `--gres` consistently:

```bash
#SBATCH -p h200
#SBATCH -w h200-ali-01   # or h200-ali-02
#SBATCH --gres=gpu:h200:1
```

For CPU/memory-only setup jobs, omit `--gres`.

## SSH Connection

Connect from your local workstation. Host aliases are defined in `~/.ssh/config`.

| Alias | Address | Auth | Notes |
|---|---|---|---|
| `4x4090` | 192.168.101.9 | SSH key (`id_ed25519`) | standalone 4×RTX 4090 box; hostname observed as `GPUServer` |
| `4090-user` | 100.124.93.40 | password / Tailscale SSH check | standalone RTX 4090 box; login user `user`; do not store plaintext password in files |
| `login-server` | 36.103.167.186 | SSH key (`id_ed25519`) | login / submit node; jump host for H100 |
| `h100-ksy-01` | via `login-server` | SSH key (`id_ed25519`) | reached through `ProxyJump login-server` |
| `h200-ali-01` | 47.116.14.100 | password (no key) | direct connection |
| `h200-ali-02` | 120.55.15.209 | password (no key) | direct connection |

```bash
# Standalone 4×4090 box (key auth)
ssh 4x4090

# Standalone RTX 4090 box (user/password or Tailscale SSH browser check)
ssh 4090-user
# Equivalent without alias:
ssh user@100.124.93.40

# Login / submit node (key auth)
ssh login-server

# H100 compute node — jumps through login-server automatically
ssh h100-ksy-01

# H200 compute nodes — direct, password authentication
ssh h200-ali-01   # 47.116.14.100
ssh h200-ali-02   # 120.55.15.209
```

Required `~/.ssh/config` entries:

```sshconfig
Host 4x4090
    HostName 192.168.101.9
    User sunny
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host 4090-user
    HostName 100.124.93.40
    User user
    IdentitiesOnly no
    PreferredAuthentications password

Host login-server
    HostName 36.103.167.186
    User sunny
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host h100-ksy-01
    HostName h100-ksy-01
    User sunny
    ProxyJump login-server

Host h200-ali-01
    HostName 47.116.14.100
    User sunny
    IdentitiesOnly no
    PreferredAuthentications password

Host h200-ali-02
    HostName 120.55.15.209
    User sunny
    IdentitiesOnly no
    PreferredAuthentications password
```

- `4x4090` and `4090-user` are standalone GPU workstations, not Slurm cluster nodes; Slurm-only rules do not apply there, but avoid disrupting existing processes.
- For `4090-user`, use the password provided out-of-band. Do **not** save plaintext passwords in `AGENTS.md`, shell scripts, Git, or logs. If Tailscale SSH displays a browser verification URL, complete that check first and retry SSH.
- H100 is not directly reachable; connect only by jumping through `login-server` (`ProxyJump`). Its key lives on `login-server`.
- H200 nodes use password authentication only — no SSH key is installed. Enter the account password when prompted.
- File transfer examples: `scp <local> 4x4090:/home/sunny/...`, `scp <local> 4090-user:/home/user/...`, `scp <local> h100-ksy-01:/DATA/sync/$USER/...` (routed through the jump host), or `scp <local> h200-ali-01:/DATA/sync/$USER/...` (direct).

## Data Storage Per Node

| Node | Home | Data / caches | Shared? |
|---|---|---|---|
| `login-server` | `/home/$USER` | `/DATA/sync/$USER` | shares H100's `/home` + `/DATA/sync` over NFS |
| `h100-ksy-01` | `/home/$USER` | `/DATA/sync/$USER` | same NFS mount as `login-server` |
| `h200-ali-01` | `/home/$USER` | `/DATA/sync/$USER` | independent — nothing shared |
| `h200-ali-02` | `/home/$USER` | `/DATA/sync/$USER` | independent — nothing shared |

- **H100 + login-server share one filesystem** (NFS). Files staged on `login-server` under `/home` or `/DATA/sync` are immediately visible to H100 jobs — prepare data/envs on `login-server`, run on H100.
- **Each H200 node is fully independent.** `h200-ali-01` and `h200-ali-02` do not share storage with each other, with H100, or with `login-server`. Environment, project files, datasets, models, and caches must be staged separately on each H200 node before running.
- Use `/DATA/NAS/GPUServer` only as a transfer/sharing area between nodes, not as a training data path.

## Commands

```bash
resources
myquota
sbatch job.sbatch
squeue -u $USER
scontrol show job <jobid>
scancel <jobid>
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,AllocTRES%80
```

Common quota-related failures:

| Reason | Meaning |
|---|---|
| `AssocGrpCPUMinsLimit` | CPU hours exhausted |
| `AssocGrpBillingMinutes` / `AssocGrpTRESMinutes` | GPU card-hours exhausted |
| `QOSMaxMemoryPerJob` | Requested memory exceeds per-job limit |
| `Resources` | Currently waiting for free resources |
| `Priority` | Lower scheduling priority |

## Mirrors

```text
GitHub: https://gh-proxy.org
HuggingFace: https://hf-mirror.com
Miniconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
PyPI: https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
PyTorch wheels: https://mirrors.aliyun.com/pytorch-wheels
```

## Token 节约规范（AI Agent 执行准则）

与 AI 协作时遵守，以降低 Token 消耗：

1. 不要把整个项目或数千行的文件一次性丢给模型阅读；只读取任务实际需要的文件/片段。
2. 善用忽略规则：跳过构建目录（如 `build/`）、权重文件、大型数据集、与当前任务无关的业务代码，避免工具自动读取消耗输入 Token。
3. 修改已有文件时优先输出增量 diff / 具体需要替换的行，而不是重写整份文件（除非用户要求全量重写或新建文件）。
4. 用户要求精简输出时（如 "只要代码" "不要解释"），严格收敛回答，不加多余说明。
5. 定位报错时，优先看 `grep`/`tail` 截取出的关键堆栈（如 `ERROR`、`Traceback`、`Segmentation fault` 附近的 20-50 行），不要求整份日志。
6. 当前话题与历史上下文无关时，建议用户开新会话，而不是在长上下文里继续堆叠不相关内容。
7. 编写代码遵循单一职责、高内聚低耦合，便于后续以更小的上下文提问和修改。
8. 长任务/长会话中，适时对已完成部分做总结压缩，而不是让完整历史一直累积。
9. 只使用任务需要的工具，避免不必要的工具定义占用上下文。
10. 回复不加寒暄客套话。
11. 较大任务先列出执行计划（明确涉及的文件和范围），经确认后再动手执行，避免走弯路浪费 Token。
