# π0.5 / LIBERO / RoboTwin / MiniCPM-RobotManip 交接文档

更新时间: 2026-07-21 16:40 CST（最新）

## 0. ⚠️ 当前阻塞：pi 集群网络不可达（2026-07-21 16:35起）

- `ssh picluster`、`ping 36.103.167.186` 均超时无响应（ICMP层面就不通，不是SSH服务问题）
- 已排除本地网络问题：同一时刻 `ping 100.124.93.40`（4090服务器，Tailscale）正常
- 结论：故障在 pi 集群这一侧（服务器本身/其网络），非我方可控，只能等待恢复
- **受影响的未完成工作**：
  - H200 上 `pi05_base` 权重下载作业 `2971` 已因触发1小时上限被 **TIMEOUT** 强制终止（不是完成！），下载到约一半左右，恢复连接后需要续跑（复用已下载部分，脚本自带断点跳过逻辑，见第4节）
  - 无法确认阶段二4090侧RoboTwin安装是否需要与H200协同的后续步骤

## 1. 现状摘要

- 阶段一（π0.5 + LIBERO 官方复现）**已完成**：冒烟测试通过，`libero_spatial` 成功率95%（19/20），接近官方98.8%基准。全量4任务集评测已按用户要求取消（判断验证已充分）。
- 阶段二（微调+可视化）进行中，详见第4节。

## 2. 访问与规则

- 集群别名：`ssh picluster`
- 免密 key：`~/.ssh/id_ed25519_picluster`
- 原始凭据备份：`E:\test\server.env`
- 规则文件：`E:\test\AGENTS.md`

关键规则：
- `login-server` 只做轻量操作
- 重活必须走 `sbatch`
- `h200-ali-01/02` 的 `/home` 和 `/DATA/disk0` 都是独立的，不和 login-server 共享
- 禁止 `sudo`、系统 Python、手动 `CUDA_VISIBLE_DEVICES`
- **2026-07-24：h200-ali-01/02 两台节点均处于 Slurm `draining` 状态**（原因：其他用户作业 Kill task failed，非我方导致），新提交的作业会一直排队，已在跑的作业不受影响；需等节点恢复或联系管理员

## 2.5 新增服务器：4x4090（2026-07-24 接入）

- 别名：`ssh 4x4090`
- IP：`192.168.101.9`（局域网，非公网/Tailscale，延迟2-6ms）
- 免密 key：`~/.ssh/id_ed25519_4x4090`
- 用户：`sunny`，**完全无 sudo 权限**（不在sudo组，连输密码的sudo都没有——比之前的4090服务器权限更受限，比pi集群稍宽松，但没有apt install能力）
- 系统：Ubuntu 22.04（GPUServer主机名），与openpi官方测试系统一致
- GPU：**4张 RTX 4090，全部空闲**（显存/利用率几乎为0）
- 用途：用户计划用这台机器搭建仿真环境（取代/补充之前那台4090服务器，因为那台目前离线中）
- **限制**：所有系统级依赖（Vulkan runtime、libX11等）必须走用户空间方案（conda-forge、pip、手动装到`~/.local`等），不能`apt install`，参考阶段一/阶段二在pi集群上积累的"无sudo环境搭建"经验

## 3. 阶段一当前状态

### 作业历史

- `2631`：超时，卡在 `uv pip install -e packages/openpi-client`
- `2697`：失败，`uv` 默认 30s 超时
- `2700`：失败，`jax-cuda12-pjrt` 下载时网络抖动
- `2759`：当前运行中

### 当前作业 2759

- 节点：`h200-ali-02`
- 状态：`RUNNING`（2026-07-21 09:42 复查）
- 当前阶段：主环境 `uv sync`
- 现象：一直在下载/解包大依赖（`scipy`、`torchvision`、`mujoco`、`rerun-sdk`、`transformers`、`numcodecs` 等）
- `/DATA/disk0/sunny/.cache/uv` 已经涨到约 `9.0G`
- Slurm 日志路径仍是：
  - stdout：`/DATA/NAS/GPUServer/sunny/setup_jobs/logs/env_resume_2759.out`
  - stderr：`/DATA/NAS/GPUServer/sunny/setup_jobs/logs/env_resume_2759.err`
- 但截至 09:38 左右，stdout/stderr 文件仍为 `0` 字节，可能是 Slurm/NAS 缓冲或 `uv sync` 阶段无输出；不要仅凭日志为空判断失败。
- 轻量 `ps` 诊断作业 `2770` 显示：
  - `/bin/bash /var/spool/slurmd/job02759/slurm_script`
  - `uv sync`
  - `uv sync` 已运行约 38 分钟，CPU 低但进程存活。

### H200 文件系统诊断

已通过同节点轻量 Slurm 诊断确认，不能在 `login-server` 上直接判断 H200 的 `/DATA/disk0/sunny` 文件是否存在。

- 诊断作业 `2765`：失败于本地 PowerShell 提前展开 `$PROJECT_ROOT`，但确认：
  - `/DATA/disk0/sunny/env.sh` 存在
  - `/DATA/disk0/sunny/projects/pi05_libero/openpi` 存在
  - `/DATA/disk0/sunny/.cache/uv` 约 `9.0G`
- 诊断作业 `2766`：`COMPLETED`
  - `env.sh` 内容确认：
    - `PROJECT_ROOT=/DATA/disk0/sunny/projects/pi05_libero`
    - `XDG_CACHE_HOME=/DATA/disk0/sunny/.cache`
    - `PIP_CACHE_DIR=/DATA/disk0/sunny/.cache/pip/cache`
    - `TMPDIR=/DATA/disk0/sunny/.cache/pip/tmp`
    - `HF_HOME=/DATA/disk0/sunny/.cache/huggingface`
    - `OPENPI_DATA_HOME=/DATA/disk0/sunny/.cache/openpi`
    - `UV_CACHE_DIR=/DATA/disk0/sunny/.cache/uv`
    - `PATH=$HOME/.local/bin:$PATH`
  - `PROJECT_ROOT` 和 `PROJECT_ROOT/openpi` 目录均存在。

### 续跑脚本

远程脚本：`~/setup_jobs/env_resume.sbatch`

当前版本特征：
- `UV_HTTP_TIMEOUT=600`
- 每个 `uv` 命令都包了 `retry()`
- 最终成功后会输出：`=== SETUP COMPLETE ===`

## 4. 后续阶段准备情况

### 阶段一：已完成（COMPLETED，见第1-3节历史）

- 冒烟测试通过：EGL渲染、策略server、LIBERO仿真联调全部正常
- `libero_spatial` 成功率 95%（19/20），接近官方 98.8% 基准
- 用户判断"验证环境足够"，取消了全量4任务集评测（作业2952），节省GPU配额
- **重要决策变更（2026-07-21）：阶段二不用 RoboTwin 2.0，改用 4090 已有的 Isaac Sim 双 Franka 工程**

### 阶段二：改为 Isaac Sim（双 Franka），已重新规划分工

**背景**：原计划用 RoboTwin 2.0（默认绑定 Aloha-AgileX 机械臂），已经开始装 conda 环境（作业2967，装了7分钟后按用户要求取消）。用户改主意，要求仿真环境与4090一致，即改用 Isaac Sim。

**已确认的关键事实**：
- 4090 服务器（`100.124.93.40`，Tailscale SSH，见 `E:\test\4090_pi05_visualization_tutorial.md`）已有现成的双 Franka Panda Isaac Sim 工程：`/home/user/sjj_ws/pi05_dual_arm_sim`，含专家数据生成脚本、18维action（左右臂各9维）、三路相机、可视化脚本
- pi 集群和 4090 网络不通（ping不通，pi集群不能装tailscale——用户明确拒绝），排除了"仿真在4090、推理在pi集群、跨机通信"的方案（openpi的policy server+client是websocket通信，跨机延迟/连通性都是问题）

**最终分工决策（用户确认）**：
- **4090 服务器**：负责 Isaac Sim 仿真、专家数据生成、π0.5 **推理+评测**（sim和policy server同机走localhost，不受pi集群网络限制；RTX 4090 24GB显存，推理只需>8GB，足够）
- **pi 集群 H200**：只负责 π0.5 **微调训练**（不需要实时通信，训练完导出checkpoint文件，用"本机中转"方案传给4090去跑评测）

**下一步待做**：
1. 到4090服务器实地确认当前 `pi05_dual_arm_sim` 工程能否直接跑通专家数据生成（可能需要SSH过去检查，而不是用pi集群sbatch）
2. 确认π0.5微调训练所需的数据格式（openpi微调需要什么格式的demonstration数据，与Isaac Sim输出的.npz格式如何转换/对齐）
3. 在pi集群H200上准备π0.5微调训练环境（复用阶段一已装好的openpi主环境，可能只需要装微调相关的训练脚本依赖，不需要重新装LIBERO那一套）
4. 规划"本机中转"的具体传输脚本（checkpoint从pi集群→本机→4090）

### 阶段四：MiniCPM-RobotManip

**调研已完成（2026-07-21，委托其他AI调研，非直接验证，后续动手前建议关键点二次核实）**：

- 真实仓库：`OpenBMB/MiniCPM-RobotManip`（不是独立顶层仓库，实为 `OpenBMB/MiniCPM-Robot` 下的子目录），Public，Apache-2.0
- 基础模型：MiniCPM-V 4.6（视觉token 256→64压缩）+ VLA action head，约1.5B~2B参数
- 安装：**Conda**（不是uv），Python 3.10 + PyTorch 2.6.0 + CUDA 12.4，`conda env create -f environment.yml`
- 输入：RGB图像(可多视角)+文本指令+可选80维state(未传则全0)+embodiment_id
- 输出：`(30, 80)` action chunk（30步时间horizon × 80维动作）
- **关键结论：动作格式与 π0.5 不兼容**，80维各维度语义、归一化统计、env adapter 官方均未公开，需要自己写映射
- 预训练权重：HuggingFace `openbmb/MiniCPM-RobotManip`，约3.6GB，ModelScope也有镜像
- 双臂支持：官方报了 RoboTwin2 基准成绩（clean 91.3 / random 91.6），理论支持双臂，但**没有公开的Aloha-AgileX专用适配层**
- 官方benchmark成绩：LIBERO 97.5、CALVIN ABC→D 4.1、RoboTwin2 91.3/91.6、RMBench 53.3（但评测脚本/adapter细节未公开，GitHub有相关issue在问，仍open）
- **可行性判断**：当前只能做"推理wrapper + smoke test"（写个脚本把观测转成模型输入格式，跑通推理），**做不到严格同条件微调对比**——官方没有发布训练pipeline，只有推理脚本(`vla_infer.py`)和权重

**下一步（本地可做，不依赖pi集群网络）**：
1. 写MiniCPM-RobotManip推理wrapper脚本框架（输入适配：图像/指令→模型格式；输出适配：80维action chunk→需要先确定具体环境的动作空间再解码）
2. conda环境配置脚本草稿（复用RoboTwin那次积累的"conda + 官方environment.yml"经验）
3. 待权重下载测试、待与π0.5的LIBERO动作空间做维度对比分析

## 5. 实用命令

### 查作业状态

```bash
ssh picluster "sacct -j 2759 --format=JobID,State,ExitCode,Elapsed -P"
ssh picluster "squeue -j 2759"
```

### 看日志

```bash
ssh picluster "tail -n 20 /DATA/NAS/GPUServer/sunny/setup_jobs/logs/env_resume_2759.out"
ssh picluster "tail -n 20 /DATA/NAS/GPUServer/sunny/setup_jobs/logs/env_resume_2759.err"
```

### 看 h200-ali-02 的真实进程

```bash
ssh picluster "sbatch -p h200 -w h200-ali-02 --job-name=pi05_ps --cpus-per-task=1 --mem=1G --time=00:05:00 --output=/DATA/NAS/GPUServer/sunny/setup_jobs/logs/ps_%j.out --error=/DATA/NAS/GPUServer/sunny/setup_jobs/logs/ps_%j.err --wrap 'hostname; date; ps -u sunny -o pid,ppid,stat,etime,pcpu,pmem,args'"
```

说明：按集群规则，不要在登录节点上用 `srun` 做交互式检查；更稳妥方式是提交 1CPU/1G/5min 的轻量 `sbatch` 诊断作业。

### H200 环境路径诊断

```bash
ssh picluster "sbatch -p h200 -w h200-ali-02 --job-name=pi05_diag --cpus-per-task=1 --mem=1G --time=00:05:00 --output=/DATA/NAS/GPUServer/sunny/setup_jobs/logs/diag_%j.out --error=/DATA/NAS/GPUServer/sunny/setup_jobs/logs/diag_%j.err --wrap 'hostname; sed -n \"1,120p\" /DATA/disk0/sunny/env.sh; source /DATA/disk0/sunny/env.sh; printf \"PROJECT_ROOT=%s\n\" \"\$PROJECT_ROOT\"; ls -ld \"\$PROJECT_ROOT\" \"\$PROJECT_ROOT/openpi\"; du -sh /DATA/disk0/sunny/.cache/uv 2>/dev/null || true'"
```

## 6. 下一步建议

1. 继续等 `2759` 结束；当前不建议中断，因为 `uv sync` 进程存活且 cache 已增长。
2. 如果 `2759` 成功，立刻提交 GPU 冒烟评测作业跑 LIBERO。
3. 如果 `2759` 失败，先读 `env_resume_2759.err/out`，再基于 `~/setup_jobs/env_resume.sbatch` 修复续跑；不要清空 `/DATA/disk0/sunny/.cache/uv`。
4. 同时在 `login-server` 上继续做后续阶段的轻量准备：
   - RoboTwin 文档整理
   - Vulkan 可用性确认
   - MiniCPM-RobotManip 仓库确认
   - 后续 sbatch 模板整理

## 7. 备注

- 之前出现过一次 `squeue` 瞬时误报，别只信一次查询，最终以 `sacct` 为准。
- `h200-ali-02` 是当前最合适的阶段一工作节点。
- `login-server` 看不到 H200 本地 `/DATA/disk0/sunny/env.sh` 和项目目录是正常现象；H200 节点文件系统独立。
- 本地 Windows 当前没有 `picluster` SSH alias，可用 `server.env` 里的 host/user/key 信息直连；不要在交接文档里记录或传播备用密码。

## 7.5 4090 可视化接入计划（暂缓，阶段二再接）

- 4090 服务器（`100.124.93.40`，Tailscale SSH）已有一套现成的 Isaac Sim 双臂可视化工程，见 `E:\test\4090_pi05_visualization_tutorial.md`：双 Franka 仿真 + `visualize_pi05_output.py`，输入是 18 维 state/action 的 `.npz`。
- pi 集群（login-server/h200）**不能装 Tailscale**（用户明确拒绝），且当前和 4090 完全不在同一网络（ping 不通）。
- 决定：阶段一的官方 LIBERO 单臂评测（action 空间和任务都与 4090 那套双臂配置不同）**不接可视化**，直接跳过。
- 等到阶段二（RoboTwin/Aloha-AgileX 双臂）时，π0.5 的输出才会是双臂格式，届时用**本机中转**方案接可视化：
  1. 从 pi 集群把 π0.5 双臂输出（action/state 序列）拉到本机 `E:\test`
  2. 本机通过已验证可用的 `tailscale ssh user@100.124.93.40` 把文件传到 4090 的 `/home/user/sjj_ws/pi05_dual_arm_sim/runs/...`
  3. 在 4090 上跑 `visualize_pi05_output.py` 生成 HTML 报告
- 不需要改任何服务器网络配置，pi 集群侧无需装任何东西。

## 8. 分工协定（Claude ⇄ Codex）

背景：本任务由两个 AI 协作——Claude（当前会话，负责集群实操）与 Codex（gpt-5.5，高推理强度，能力弱于 Claude）。为了让 Codex 能稳定产出可用结果，分工原则如下：

### 8.1 谁做什么

**Claude 独占（Codex 不碰）：**
- 一切需要 SSH/`ssh picluster` 的操作：提交/监控/诊断 Slurm 作业、读远程日志、判断作业是否卡死
- 节点选择、资源分配、超时/重试参数等运维判断
- 任何会修改集群状态或产生真实费用（GPU 小时/存储）的动作
- `HANDOFF.md` 第 1~7 节（集群状态）的更新

**Codex 负责：**
- **只在本地 `E:\test` 目录下写代码/脚本/文档草稿**，不连集群、不联网搜索
- 根据 Claude 提供的**已核实事实**（不是让 Codex 自己去查资料，避免臆造），写：
  - sbatch 脚本模板草稿（`.sbatch.draft` 后缀，Claude 审核后再提交，不由 Codex 直接 `sbatch`）
  - 数据处理/结果解析脚本（如 LIBERO 成功率解析、结果对比表生成）
  - 阶段三"统一数据/动作/评测协议"的骨架代码和 schema 定义
  - 阶段二/四相关的本地工具脚本（数据格式转换、配置文件生成等），基于 Claude 给的规格文档，而不是自行调研
- 每个任务必须有：明确输入、明确输出文件路径、明确验收标准（如"能在无 GPU 环境下用假数据跑通"）

### 8.2 协作规则

1. **不并发改同一文件**：Claude 和 Codex 各自负责的文件路径不重叠（见下方任务清单里的"输出路径"）。
2. **Codex 产出先审后用**：任何 Codex 写的 sbatch 草稿，Claude 必须先读一遍、按 AGENTS.md 硬规则核对（分区/节点/`--gres`/日志路径/不碰 sudo 等）后才提交。
3. **单一事实来源**：`HANDOFF.md` 是双方共享状态文档。Claude 更新第 1~7 节（集群实况）；Codex 只在本文件底部"8.3 Codex 任务清单"里勾选/更新任务状态，不改其他章节。
4. **Codex 任务必须自包含**：交给 Codex 的任务描述要给全上下文（已确认的事实、约束、示例），不要让它自己去猜集群拓扑或搜索网络信息——它能力较弱，臆造细节的风险更高。
5. **避免大范围重写**：给 Codex 的任务尽量是新增文件或小范围、职责单一的模块，不要求它一次性改动大量已有代码。

### 8.3 Codex 任务清单（当前）

| # | 任务 | 输出路径 | 依赖/输入 | 验收标准 | 状态 |
|---|---|---|---|---|---|
| 1 | 写 LIBERO 评测结果解析脚本：读取评测输出（约定格式待 Claude 补充示例日志后给出），计算 4 个任务集成功率 + 均值，并与官方基准(98.8/98.2/98.0/92.4，均值96.85)对比，输出 markdown 报告 | `E:\test\scripts\parse_libero_results.py` | 需 Claude 提供样例日志/结果格式（阶段一 GPU 评测作业跑完后补充） | 用构造的假数据跑通，输出正确的均值和对比结论 | 待 Claude 提供样例后再派发 |
| 2 | 起草阶段二 RoboTwin 2.0 sbatch 模板草稿：conda 环境（python 3.10）、按 `E:\test\AGENTS.md` 里的 sbatch 模板风格、日志写到 `/DATA/NAS/GPUServer/sunny/...`，Vulkan 依赖处理先留 `# TODO(Claude): 确认节点 Vulkan 可用性` 占位注释 | `E:\test\jobs\robotwin_env_setup.sbatch.draft` | 已确认事实：官方仓库 `RoboTwin-Platform/RoboTwin`，用 conda 非 uv，`bash script/_install.sh`，需要 Vulkan | Claude 审核后可直接改路径提交，无需大改结构 | 已完成：草稿已写入，待 Claude 审核节点/Vulkan/路径后提交 |
| 3 | 起草阶段三"统一数据/动作/评测协议"的 schema 骨架（纯 Python dataclass 或 JSON Schema，暂不含具体任务列表，先搭结构：观测格式、动作空间维度、评测指标字段） | `E:\test\protocols\unified_task_schema.py` | 无需集群信息，纯设计任务，Claude 会给出字段约束 | 能被 Python 直接 import，字段命名清晰、有类型注解 | 已完成：schema 骨架可 import，含轻量校验函数 |

> Claude 会在每次进展后更新第 1~7 节；Codex 完成任务后在本表"状态"列标注"已完成 + 简要说明"，不要改动其他章节内容。
