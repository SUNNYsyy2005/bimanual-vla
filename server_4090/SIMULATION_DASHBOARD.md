# 仿真 Dashboard 使用手册（AI 提交训练 / 评测流程）

本文档给 AI agent / 操作者使用，目标是：以后提交仿真数据上传、4×4090 训练、H100/H200 Slurm 训练、4×4090 评测、H100/H200 Slurm 评测、评估视频查看时，优先走 Dashboard API / 页面，不再临时重复写 sbatch、norm、train、eval 脚本。

## 1. 服务与隔离关系

| 用途 | 地址 | systemd user service | 配置 | workspace | 数据集可见范围 |
|---|---|---|---|---|---|
| 实机 Dashboard | `http://192.168.101.9:8090` | `bimanual-vla-dashboard.service` | `server_4090/config.json` | `~/.local/share/bimanual-vla-server` | `real` / `unknown` |
| 仿真 Dashboard | `http://192.168.101.9:8091` | `bimanual-vla-sim-dashboard.service` | `server_4090/config.simulation.json` | `~/.local/share/bimanual-vla-sim-dashboard` | `simulation` |

仿真 Dashboard 与实机 Dashboard 共用底层 LeRobot 兼容数据根目录、OpenPI 代码、checkpoint 根目录，但页面/API 会按数据集来源过滤：

- 仿真 Dashboard 只显示 `dataset_origin=simulation` 的数据集。
- 仿真 Dashboard 的训练初始化权重 / checkpoint 列表只显示：
  - foundation/base model；
  - 由仿真数据集训练出的 checkpoint。
- 实机数据集训练出的 checkpoint 不会出现在仿真 Dashboard 的权重选择器里；即使手动传入隐藏 checkpoint 路径，服务端也会拒绝。
- 实机 Dashboard 仍只显示实机/未分类数据集，不显示仿真数据集。

## 2. Token 获取

仿真 Dashboard token 在 `4x4090`：

```bash
ssh 4x4090 'source ~/.config/bimanual-vla-sim-dashboard/server.env && printf "%s\n" "$BIMANUAL_VLA_SERVER_TOKEN"'
```

登录账号密码也在同一文件：

```bash
ssh 4x4090 'source ~/.config/bimanual-vla-sim-dashboard/server.env && printf "user=%s\npassword=%s\n" "$BIMANUAL_VLA_LOGIN_USER" "$BIMANUAL_VLA_LOGIN_PASSWORD"'
```

命令行示例中统一使用：

```bash
export SIM_DASHBOARD=http://192.168.101.9:8091
export SIM_TOKEN='替换为上面查到的 token'
```

不要把 token 写进 Git、文档提交或公开日志。

## 3. 服务管理

在 `4x4090` 上检查状态：

```bash
ssh 4x4090 'systemctl --user status bimanual-vla-sim-dashboard.service --no-pager'
ssh 4x4090 'journalctl --user -u bimanual-vla-sim-dashboard.service -n 100 --no-pager'
ssh 4x4090 'tail -n 100 ~/.local/share/bimanual-vla-sim-dashboard/dashboard.log'
```

重启仿真 Dashboard：

```bash
ssh 4x4090 'systemctl --user restart bimanual-vla-sim-dashboard.service'
```

重新部署：

```bash
./deploy_4090_sim_dashboard.sh
```

部署脚本会：

1. 同步 Dashboard 运行需要的文件到 `4x4090:/home/sunny/bimanual-vla`。
2. 如果不存在 `server_4090/config.simulation.json`，从 `config.simulation.example.json` 复制。
3. 安装并启用 `bimanual-vla-sim-dashboard.service`。
4. 重启 8091 服务。

## 4. 推荐 AI 工作流总览

AI agent 处理仿真训练 / 评测任务时，按下面顺序执行：

1. 确认用户需求：数据集 ID、模型系列 `pi05/pi0`、实验名、训练目标机器、GPU 数量、训练步数、batch、评测设置。
2. 刷新 Dashboard 状态：
   ```bash
   curl -sS -H "Authorization: Bearer $SIM_TOKEN" "$SIM_DASHBOARD/api/status" | jq .
   ```
3. 确认数据集在 8091 可见，且 `dataset_origin=simulation`。
4. 如果数据集不存在，先用上传脚本上传，必须带：
   ```bash
   --dataset-origin simulation
   ```
5. 从 `/api/status` 选择可用 foundation/base model 或仿真 checkpoint。
6. 提交训练：
   - 本地 4×4090：`execution_target=local_4090`，`gpu_ids` 填物理 GPU ID 列表，例如 `0,1`。
   - H100/H200：`execution_target=h100/h200-ali-01/h200-ali-02`，`cluster_gpus` 填 Slurm 申请卡数；Dashboard 会提交 sbatch。
7. 通过 `/api/tasks/<task_id>/log` 和 `/api/tasks/<task_id>/metrics` 监控。
8. 训练间隔 checkpoint 的自动 held-out eval 会在本地 4×4090 上按空闲卡触发；没有空闲卡会记录 skipped，不阻塞训练。
9. 需要额外评测时，走 `POST /api/tasks/eval` 手动提交。
10. 有视频评测输出时，把视频保存或软链到 `eval_video_roots` 中，再在页面“评估视频”模块直接看。

## 5. 数据上传

### 5.1 上传仿真 LeRobot / GUI NPZ 数据集

在能访问 `4x4090:8091` 的机器执行：

```bash
python upload_dataset_4090.py /path/to/LEROBOT_OR_GUI_NPZ_DIR \
  --name DATASET_ID \
  --dataset-origin simulation \
  --server http://192.168.101.9:8091 \
  --token "$SIM_TOKEN" \
  --workers 4 \
  --merge
```

说明：

- `--dataset-origin simulation` 是关键；否则数据可能被归到实机 Dashboard。
- 上传暂存目录在仿真 Dashboard workspace 下按来源隔离：
  ```text
  ~/.local/share/bimanual-vla-sim-dashboard/uploads/simulation
  ```
- 最终 LeRobot 安装目录仍保持兼容 OpenPI 的扁平结构：
  ```text
  ~/.cache/huggingface/lerobot/<DATASET_ID>
  ```
- 来源 marker 写在：
  ```text
  ~/.cache/huggingface/lerobot/<DATASET_ID>/meta/dashboard_dataset_origin.json
  ```

### 5.2 检查数据集是否可见

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/status" \
  | jq '.datasets[] | {id, dataset_origin, episodes, frames, schema_label, arm_mode, training_supported}'
```

仿真 Dashboard 正常情况下所有返回项都应是：

```json
{"dataset_origin":"simulation"}
```

如果数据集不显示，常见原因：

- 上传时忘了 `--dataset-origin simulation`。
- 数据集没有有效 `meta/info.json`。
- marker 被手动改成了 `real` 或 `unknown`。
- 数据集格式不兼容，但即使不兼容通常仍会显示为“仅管理 / 预览”；完全不显示优先查来源 marker。

## 6. 训练提交

训练 API：

```http
POST /api/tasks/train
Authorization: Bearer <SIM_TOKEN>
Content-Type: application/json
```

### 6.1 本地 4×4090 训练示例

```bash
curl -sS -X POST "$SIM_DASHBOARD/api/tasks/train" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "pick_cube_piper_v1",
    "exp_name": "pick_cube_piper_v1_pi05_sim_001",
    "model_variant": "pi05",
    "base_checkpoint": "/home/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base",
    "execution_target": "local_4090",
    "gpu_ids": ["0", "1"],
    "fsdp_devices": 2,
    "batch_size": 2,
    "num_workers": 2,
    "num_train_steps": 30000,
    "save_interval": 1000,
    "mode": "auto",
    "eval_enabled": true,
    "eval_interval_steps": 5000,
    "eval_batch_size": 1,
    "eval_num_workers": 2,
    "eval_max_batches": 50,
    "eval_xla_memory_fraction": 0.85,
    "xla_memory_fraction": 0.90
  }' | jq .
```

返回 `task.id` 后，记录它，例如：

```bash
export TRAIN_TASK_ID=train-YYYYMMDD-HHMMSS-xxxxxxxx
```

### 6.2 H100 Slurm 训练示例

> 重要：H100/H200 训练必须走 Slurm。Dashboard 只在 4×4090 上做轻量 SSH/sbatch 提交和轮询，真实训练在 Slurm 节点执行。

```bash
curl -sS -X POST "$SIM_DASHBOARD/api/tasks/train" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "pick_cube_piper_v1",
    "exp_name": "pick_cube_piper_v1_pi05_h100_001",
    "model_variant": "pi05",
    "base_checkpoint": "/home/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base",
    "execution_target": "h100",
    "cluster_gpus": 1,
    "fsdp_devices": 1,
    "batch_size": 4,
    "num_workers": 4,
    "num_train_steps": 30000,
    "save_interval": 1000,
    "mode": "auto",
    "eval_enabled": true,
    "eval_interval_steps": 5000,
    "eval_batch_size": 1,
    "eval_num_workers": 2,
    "eval_max_batches": 50,
    "xla_memory_fraction": 0.90
  }' | jq .
```

Dashboard 会生成一个本地 `train` 任务，但它的 command 实际是：

```text
server_4090/slurm_job_runner.py --target-json ... --commands-json ...
```

`slurm_job_runner.py` 会：

1. SSH 到 `login-server`。
2. 在 H100 共享文件系统中的远端工作目录写入 sbatch 文件。
3. 执行 `sbatch`。
4. 用 `squeue` / `sacct` 轮询 job 状态。
5. 把 Slurm job id、状态和最终 exit code 写入 Dashboard 任务日志。

### 6.3 H200 Slurm 训练示例

```bash
curl -sS -X POST "$SIM_DASHBOARD/api/tasks/train" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "pick_cube_piper_v1",
    "exp_name": "pick_cube_piper_v1_pi05_h200_001",
    "model_variant": "pi05",
    "base_checkpoint": "/home/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base",
    "execution_target": "h200-ali-01",
    "cluster_gpus": 1,
    "fsdp_devices": 1,
    "batch_size": 4,
    "num_workers": 4,
    "num_train_steps": 30000,
    "save_interval": 1000,
    "mode": "auto"
  }' | jq .
```

注意：H200 不作为 Dashboard 可直连登录节点使用。8091 自动提交 H200 任务时只 SSH 到 `login-server`，再由 `login-server` 调用 Slurm 申请 `h200` 分区和指定 `node=h200-ali-01/02`。因此 H200 配置应满足：

- `submit_host=login-server`；
- `partition=h200`；
- `node=h200-ali-01` 或 `h200-ali-02`；
- H200 本地 `/DATA/sync/$USER` 下已提前准备好项目、环境、数据集和模型，或通过 staging/setup Slurm 作业准备。

否则任务会失败并在日志里显示 SSH 认证错误。

### 6.4 训练参数解释

| 字段 | 含义 | 本地 4×4090 | H100/H200 |
|---|---|---|---|
| `execution_target` | 执行位置 | `local_4090` | `h100` / `h200-ali-01` / `h200-ali-02` |
| `gpu_ids` | GPU 选择 | 物理 GPU ID 列表，如 `["0","1"]` | 可忽略，建议用 `cluster_gpus` |
| `cluster_gpus` | Slurm 申请卡数 | 不用 | `1/2/4...` |
| `fsdp_devices` | FSDP 分片数 | 必须整除 GPU 数 | 必须整除 `cluster_gpus` |
| `batch_size` | 全局 batch | 必须整除 GPU 数 | 必须整除 `cluster_gpus` |
| `base_checkpoint` | 初始化权重 | foundation 或仿真 checkpoint | 同左，路径会按配置映射到远端 |
| `mode` | 启动方式 | `auto/new/resume/overwrite` | 同左 |
| `eval_enabled` | 自动 held-out eval | 本地训练会自动找空闲卡 | Slurm 训练内仍写入配置；推荐另行手动提交评测 |

### 6.5 norm / split 规则

训练提交时会自动处理 norm：

- 如果对应数据集、模型系列、action contract、episode split 的 `norm_stats.json` 已存在，则直接训练。
- 如果不存在，则先启动 norm，再进入 `waiting_norm`。
- norm 完成后自动启动训练。
- split 会写入数据集：
  ```text
  meta/train_test_split.json
  ```
- norm 目录还会保存：
  ```text
  episode_split.json
  norm_config.json
  norm_stats.json
  ```
- `norm_batch_size` 不需要和训练 `batch_size` 一致；它只影响统计计算吞吐。

## 7. 监控训练

### 7.1 列出任务

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/status" \
  | jq '.tasks[] | {id,type,state,pid,metadata,waiting_reason,skip_reason,returncode}'
```

### 7.2 看日志

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks/$TRAIN_TASK_ID/log?max_bytes=65536" \
  | jq -r .log
```

Slurm 任务日志里应能看到：

```text
[dashboard] submitting Slurm job on login-server: ...
Submitted batch job <JOB_ID>
[dashboard] squeue <JOB_ID>: RUNNING|...
[dashboard] final_state=COMPLETED exit_code=0:0
```

### 7.3 看训练曲线

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks/$TRAIN_TASK_ID/metrics?max_points=1200" \
  | jq '.summary, .eval_summary'
```

页面“训练指标曲线”会把训练 loss 和自动 eval loss 合并到同一张曲线里。

### 7.4 停止任务

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"force": false}' \
  "$SIM_DASHBOARD/api/tasks/$TRAIN_TASK_ID/stop" | jq .
```

强杀：

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"force": true}' \
  "$SIM_DASHBOARD/api/tasks/$TRAIN_TASK_ID/stop" | jq .
```

删除终态任务记录和日志，不删除 checkpoint：

```bash
curl -sS -X DELETE \
  -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks/$TRAIN_TASK_ID" | jq .
```

## 8. 手动评测

评测 API：

```http
POST /api/tasks/eval
Authorization: Bearer <SIM_TOKEN>
Content-Type: application/json
```

### 8.1 选择 checkpoint

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/status" \
  | jq '.checkpoints[] | {path, experiment, step, dataset_ids, dataset_origins, model_variant, arm_mode}'
```

仿真 Dashboard 已过滤实机 checkpoint；这里列出的训练 checkpoint 都应来自仿真数据集，foundation/base model 不在 `checkpoints` 列表而在 `base_models` 中。

### 8.2 4×4090 本地评测

```bash
curl -sS -X POST "$SIM_DASHBOARD/api/tasks/eval" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "execution_target": "local_4090",
    "dataset_id": "pick_cube_piper_v1",
    "checkpoint": "/home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints/pi05_piper_single_arm_lora/pick_cube_piper_v1_pi05_sim_001/5000",
    "model_variant": "pi05",
    "base_checkpoint": "/home/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base",
    "gpu_ids": ["2"],
    "batch_size": 1,
    "num_workers": 2,
    "max_batches": 50,
    "xla_memory_fraction": 0.85
  }' | jq .
```

### 8.3 H100 Slurm 评测

```bash
curl -sS -X POST "$SIM_DASHBOARD/api/tasks/eval" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "execution_target": "h100",
    "cluster_gpus": 1,
    "dataset_id": "pick_cube_piper_v1",
    "checkpoint": "/home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints/pi05_piper_single_arm_lora/pick_cube_piper_v1_pi05_sim_001/5000",
    "model_variant": "pi05",
    "base_checkpoint": "/home/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base",
    "batch_size": 1,
    "num_workers": 2,
    "max_batches": 50,
    "xla_memory_fraction": 0.85
  }' | jq .
```

H100 评测同样通过 `slurm_job_runner.py` 提交 `sbatch`。路径会按 `config.simulation.json` 中的 target 配置从 4×4090 路径映射到远端路径，例如：

```text
/home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints
→ /DATA/sync/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints
```

因此提交前要确保远端节点对应路径已经同步了代码、数据集、base checkpoint 和目标 checkpoint。

## 9. 评估视频可视化

页面“评估视频”模块扫描：

```text
~/.local/share/bimanual-vla-sim-dashboard/eval_videos
/home/sunny/robotwin_ws/RoboTwin/policy/pi05/outputs
```

支持：

```text
.mp4 .webm .mov .mkv .avi .gif
```

如果仿真评测脚本产生视频，推荐保存到：

```text
/home/sunny/.local/share/bimanual-vla-sim-dashboard/eval_videos/<experiment>/<run>.mp4
```

或软链进去：

```bash
ssh 4x4090 'mkdir -p ~/.local/share/bimanual-vla-sim-dashboard/eval_videos/my_eval && ln -sf /path/to/video.mp4 ~/.local/share/bimanual-vla-sim-dashboard/eval_videos/my_eval/video.mp4'
```

刷新页面模块即可播放，不需要把视频同步到本地。

命令行列出视频：

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/eval-videos" | jq '.videos[] | {name, relative_path, size_mib, updated_at, url}'
```

## 10. H100/H200 远端准备要求

Dashboard 可以封装提交，但不会自动同步大数据和环境。提交 H100/H200 前检查：

### 10.1 H100

H100 通过 `login-server` 提交，`login-server` 和 `h100-ksy-01` 共享 `/DATA/sync/$USER`。建议远端路径：

```text
/DATA/sync/sunny/bimanual-vla
/DATA/sync/sunny/robotwin_ws/RoboTwin/policy/pi05
/DATA/sync/sunny/.cache/huggingface/lerobot
/DATA/sync/sunny/.cache/openpi/openpi-assets/checkpoints
/DATA/sync/sunny/miniconda3/envs/openpi
```

提交前：

```bash
ssh login-server 'hostname; pwd; resources; myquota; ls /DATA/sync/sunny/bimanual-vla/server_4090/openpi_single_arm.py'
```

### 10.2 H200

H200 节点存储互相独立，也不和 H100/login-server 共享。每个 H200 节点都要单独准备：

```text
/DATA/sync/sunny/bimanual-vla
/DATA/sync/sunny/robotwin_ws/RoboTwin/policy/pi05
/DATA/sync/sunny/.cache/huggingface/lerobot
/DATA/sync/sunny/.cache/openpi/openpi-assets/checkpoints
/DATA/sync/sunny/miniconda3/envs/openpi
```

并且需要解决非交互 SSH 认证；否则 Dashboard 后台无法提交 sbatch。

### 10.3 配置目标

目标配置在 `4x4090`：

```text
/home/sunny/bimanual-vla/server_4090/config.simulation.json
```

关键字段：

```json
{
  "cluster_targets": {
    "h100": {
      "submit_host": "login-server",
      "partition": "h100",
      "node": "h100-ksy-01",
      "gpu_type": "h100",
      "workdir": "/DATA/sync/sunny/bimanual-vla",
      "openpi_python": "/DATA/sync/sunny/miniconda3/envs/openpi/bin/python",
      "dataset_root": "/DATA/sync/sunny/.cache/huggingface/lerobot",
      "assets_base_dir": "/DATA/sync/sunny/robotwin_ws/RoboTwin/policy/pi05/assets",
      "checkpoint_base_dir": "/DATA/sync/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints"
    }
  }
}
```

修改配置后重启：

```bash
ssh 4x4090 'systemctl --user restart bimanual-vla-sim-dashboard.service'
```

## 11. 常见故障

### 11.1 仿真 Dashboard 里看不到某个 checkpoint

原因通常是 checkpoint provenance 不属于当前 Dashboard 可见来源。检查：

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" "$SIM_DASHBOARD/api/status" \
  | jq '.checkpoints[] | {path,dataset_ids,dataset_origins}'
```

仿真 Dashboard 只允许：

```text
foundation/base model
或 dataset_origins 中至少一个是 simulation 的训练 checkpoint
```

实机 checkpoint 会被隐藏；手动传入也会被服务端拒绝。

### 11.2 H100/H200 任务提交失败

看 Dashboard 任务日志：

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks/$TASK_ID/log?max_bytes=131072" | jq -r .log
```

常见问题：

- SSH alias 不存在或不能非交互登录。
- H200 被当成 Slurm-only 节点，不要要求 Dashboard 直接 SSH H200；检查 `submit_host=login-server`、`partition=h200`、`node=h200-ali-01/02`。
- 远端没有 `/DATA/sync/sunny/bimanual-vla` 或代码版本不一致。
- 远端没有数据集、base checkpoint、训练 checkpoint。
- 远端 conda env 路径不对。
- Slurm quota 不足：`myquota`。
- 资源不足或优先级低：`resources` / `squeue -u sunny`。

### 11.3 本地 4×4090 训练 OOM

4090 训练优先从保守参数开始：

```text
2 张 4090
batch_size=2
fsdp_devices=2
xla_memory_fraction=0.90
```

Dashboard 默认要求训练 GPU 每张至少 `23000 MiB` 空闲显存。若卡上有其他进程，任务会拒绝或进入等待。

### 11.4 视频页面能列出但不能播放

确认：

- 视频后缀在支持列表内。
- 浏览器访问的是 8091 页面，不是 8090。
- token 未过期或未输错。
- 文件在 `eval_video_roots` 配置的目录内。

## 12. 给 AI agent 的最小操作模板

```bash
# 1. 取 token
SIM_TOKEN=$(ssh 4x4090 'source ~/.config/bimanual-vla-sim-dashboard/server.env && printf "%s" "$BIMANUAL_VLA_SERVER_TOKEN"')
SIM_DASHBOARD=http://192.168.101.9:8091

# 2. 查数据、权重、GPU/target
curl -sS -H "Authorization: Bearer $SIM_TOKEN" "$SIM_DASHBOARD/api/status" > /tmp/sim_status.json
jq '.datasets[] | {id,dataset_origin,episodes,training_supported}' /tmp/sim_status.json
jq '.base_models[] | {path,foundation,experiment,checkpoint_step,dataset_origins}' /tmp/sim_status.json
jq '.checkpoints[] | {path,experiment,step,dataset_ids,dataset_origins}' /tmp/sim_status.json
jq '.config.cluster_targets | keys' /tmp/sim_status.json

# 3. 提交本地训练或 Slurm 训练
curl -sS -X POST "$SIM_DASHBOARD/api/tasks/train" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d @train_payload.json | tee /tmp/train_task.json

TASK_ID=$(jq -r .id /tmp/train_task.json)

# 4. 监控
curl -sS -H "Authorization: Bearer $SIM_TOKEN" "$SIM_DASHBOARD/api/tasks/$TASK_ID/log?max_bytes=65536" | jq -r .log
curl -sS -H "Authorization: Bearer $SIM_TOKEN" "$SIM_DASHBOARD/api/tasks/$TASK_ID/metrics?max_points=1200" | jq '.summary, .eval_summary'

# 5. 提交手动评测
curl -sS -X POST "$SIM_DASHBOARD/api/tasks/eval" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d @eval_payload.json | jq .

# 6. 看视频
curl -sS -H "Authorization: Bearer $SIM_TOKEN" "$SIM_DASHBOARD/api/eval-videos" | jq '.videos[] | {relative_path,url}'
```

## 13. AI 专用任务状态接口

除完整 `/api/status` 外，8091 提供更轻量的任务查询接口，适合 AI agent 轮询：

### 13.1 列出任务摘要

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks?limit=50" | jq .
```

常用过滤：

```bash
# 只看活动任务：running / starting / stopping / waiting_norm / waiting_gpu
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks?active=true" | jq .

# 只看某个数据集的训练任务
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks?type=train&dataset_id=pick_cube_piper_v1" | jq .

# 只看某个实验
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks?exp_name=pick_cube_piper_v1_pi05_sim_001" | jq .

# train 任务附带简要 metrics/progress
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks?type=train&include_metrics=true" | jq .
```

返回字段包括：

```text
id, type, state, active, terminal, pid,
created_at, queued_at, started_at, finished_at,
returncode, waiting_reason, skip_reason,
dependency, dependency_state,
metadata.dataset_id / exp_name / execution_target / runtime / gpu_ids / checkpoint_dir,
result(eval), metrics(train 可选)
```

### 13.2 查询单个任务状态

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks/$TASK_ID/status?include_metrics=true" | jq .
```

这个接口比 `/api/tasks/<task_id>` 更适合 AI 使用，因为它去掉了冗长 command，只保留状态判断所需字段。

## 14. 数据集多服务器位置、去重、同步

仿真数据集可能同时存在于：

```text
local_4090
h100
h200-ali-01
h200-ali-02
```

这些位置来自 `config.simulation.json` 的 `dataset_root` 和 `cluster_targets.*.dataset_root`。H200 若配置为 `access_mode=slurm_only`，Dashboard 不会直接 SSH 扫描/传输该节点；需要先通过 login-server 提交 staging/setup Slurm 作业，或经 NAS 中转到 H200 本地 `/DATA/sync/$USER`。

### 14.1 查询所有位置并去重

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/dataset-locations?origin=simulation" | jq .
```

返回结构按 `dataset_id` 去重：

```json
{
  "datasets": [
    {
      "id": "pick_cube_piper_v1",
      "duplicate_count": 2,
      "origins": ["simulation"],
      "targets": ["local_4090", "h100"],
      "locations": [
        {
          "target": "local_4090",
          "path": "/home/sunny/.cache/huggingface/lerobot/pick_cube_piper_v1",
          "origin": "simulation",
          "episodes": 100,
          "frames": 20000
        }
      ]
    }
  ],
  "errors": {}
}
```

说明：

- local_4090 直接扫描本地文件系统。
- H100 可通过 `login-server` 共享文件系统做只读扫描。
- H200 配置为 `access_mode=slurm_only` 时不做直接 SSH 扫描；数据位置接口会在 `errors` 中提示需要 staging/setup Slurm 作业或 NAS 中转。
- 不在 H100/H200 开 HTTP 服务，也不监听端口。

### 14.2 跨服务器同步数据集

同步 API：

```http
POST /api/datasets/<dataset_id>/sync
```

例：从 4×4090 同步到 H100：

```bash
curl -sS -X POST "$SIM_DASHBOARD/api/datasets/pick_cube_piper_v1/sync" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "source": "local_4090",
    "target": "h100",
    "overwrite": false
  }' | jq .
```

例：从 H100 拉回 4×4090：

```bash
curl -sS -X POST "$SIM_DASHBOARD/api/datasets/pick_cube_piper_v1/sync" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "source": "h100",
    "target": "local_4090",
    "overwrite": false
  }' | jq .
```

同步任务类型是：

```text
transfer
```

查看日志：

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/tasks/$TRANSFER_TASK_ID/log?max_bytes=131072" | jq -r .log
```

实现方式：

```text
source tar -cf - DATASET_ID | target tar -xf -
```

跨机器时使用 SSH 管道，不需要在目标服务器开启端口。

注意：

- `overwrite=false` 时，如果目标已经存在同名数据集，任务会失败，避免误覆盖。
- `overwrite=true` 会删除目标同名目录再解包，慎用。
- 同步只处理 LeRobot 数据目录，不自动同步 checkpoint / conda env / OpenPI 代码。

## 15. 数据采集接口

仿真 Dashboard 提供轻量采集会话接口，用来让 AI 统一登记“将要采什么数据、采到哪里、上传命令是什么”。它不在 H100/H200 开服务，不直接控制仿真器；真正采集脚本仍由对应机器运行，采集结束后用 upload 或 sync 接口进入 Dashboard 管理。

### 15.1 创建采集会话

```bash
curl -sS -X POST "$SIM_DASHBOARD/api/collection-sessions" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "pick_cube_piper_v1_new",
    "dataset_origin": "simulation",
    "target": "local_4090",
    "metadata": {
      "task": "pick_cube",
      "episodes": 100,
      "simulator": "RoboTwin"
    }
  }' | jq .
```

返回中会包含推荐上传命令：

```text
upload_command
```

### 15.2 列出采集会话

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/collection-sessions" | jq .
```

### 15.3 更新采集状态

```bash
curl -sS -X PATCH "$SIM_DASHBOARD/api/collection-sessions/$SESSION_ID" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "status": "uploaded",
    "notes": "100 episodes collected and uploaded",
    "metadata": {"upload_task_id": "simulation-..."}
  }' | jq .
```

建议状态：

```text
created -> collecting -> collected -> uploading -> uploaded -> validated
failed
```

## 16. H100/H200 资源查询接口

8091 总览会显示 H100/H200 Slurm 资源，后端调用：

```text
scripts/query_h100_h200_resources.sh --compact
```

API：

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/cluster-resources" | jq -r .output
```

可选显示全队列：

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/cluster-resources?all_jobs=true" | jq -r .output
```

注意：

- 该接口只读。
- 默认从 4×4090 通过 SSH 查询 `login-server` 的 Slurm 状态。
- 不会在 H100/H200 上运行 Dashboard，也不会打开任何监听端口。
- 资源查询失败通常是 SSH alias / key / login-server 连接问题。

## 17. H100/H200 评估视频同步到 4×4090 播放

评估视频可能生成在：

```text
4×4090 本地 eval_video_roots
H100/H200 的远端 eval_video_roots
```

Dashboard 不能也不应该在 H100/H200 上开端口提供视频服务。H100 可通过 `login-server` 共享文件系统扫描/同步；H200 为 Slurm-only 时需先由 staging/setup Slurm 作业或 NAS 把视频同步回 4×4090，然后由 8091 播放。

### 17.1 扫描本地 + 远端视频

```bash
curl -sS -H "Authorization: Bearer $SIM_TOKEN" \
  "$SIM_DASHBOARD/api/eval-videos?include_remote=true" | jq .
```

返回中：

- `remote=false`：已经在 4×4090，本地可直接播放，字段里有 `url`。
- `remote=true`：视频在 H100/H200，需要先同步，字段里有 `source/root/relative_path`。

页面“评估视频”里：

- “刷新本地视频”只扫 4×4090。
- “扫描 H100/H200 并同步”会显示可直接扫描的远端视频条目；H200 Slurm-only 节点会显示提示，需要先用 staging/NAS 回传视频。

### 17.2 同步远端视频到本地

```bash
curl -sS -X POST "$SIM_DASHBOARD/api/eval-videos/sync" \
  -H "Authorization: Bearer $SIM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "source": "h100",
    "root": "/DATA/sync/sunny/robotwin_ws/RoboTwin/policy/pi05/outputs",
    "relative_path": "my_eval/video.mp4",
    "overwrite": false
  }' | jq .
```

该接口会创建一个 `transfer` 任务，执行：

```text
ssh <source> 'tar -C <remote_root> -cf - <relative_path>' | tar -C <local_eval_video_root>/<source> -xf -
```

同步完成后视频位置类似：

```text
~/.local/share/bimanual-vla-sim-dashboard/eval_videos/h100/my_eval/video.mp4
```

然后刷新“评估视频”即可播放。

注意：

- H100 通过 login-server 读共享文件；H200 Slurm-only 不直接 SSH 读文件；任何节点都不开 HTTP/WebSocket 端口。
- `root` 必须在 `config.simulation.json` 的 `cluster_targets.<target>.eval_video_roots` 中。H200 Slurm-only 节点上的视频需要先通过 Slurm staging/NAS 同步回 4×4090，Dashboard 不直接 SSH 读取 H200 计算节点。
- `relative_path` 不能是绝对路径，不能包含 `..`。
- `overwrite=false` 时，本地已存在同名文件会失败，避免误覆盖。

## 14. 传输性能与并行策略

所有 Dashboard 托管的跨服务器传输默认采用并行流式传输：

- 数据集同步 `/api/datasets/<dataset_id>/sync`：对可 SSH 访问的位置，先在源端按文件大小做均衡分片，然后同时启动多个 `tar` 流，经由 4×4090 控制端并行抽取/写入目标位置；H200 Slurm-only 不走该直传接口，需要 staging/setup Slurm 作业或 NAS 中转。
- 远端评估视频同步 `/api/eval-videos/sync`：按 64MiB 分片并行从远端读取，全部分片校验大小后再在 4×4090 合并为可播放文件。
- 默认并行度来自 `config.simulation.json` 的 `transfer_parallelism`，建议 4；接口也可在 JSON body 里临时覆盖：

```bash
curl -X POST "$SIM_DASHBOARD/api/datasets/my_sim_dataset/sync" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"source":"local_4090","target":"h100","parallelism":8}'

curl -X POST "$SIM_DASHBOARD/api/eval-videos/sync" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"source":"h100","root":"/DATA/sync/sunny/robotwin_ws/RoboTwin/policy/pi05/outputs","relative_path":"eval/run.mp4","parallelism":8}'
```

并行度上限为 16。若源/目标是 H200 Slurm-only，接口会直接拒绝直传并提示使用 staging/setup Slurm 作业或 NAS；不会在 H200 上启动任何监听服务。

## 15. 仿真 joint 数据集训练兼容

旧实机 Dashboard 对没有 `contract_version`/`gripper_semantics` 的 7D/14D joint 数据集默认 fail-closed，避免把实机 legacy v2 的夹爪米制开度误当作 v3 opening fraction。仿真 Dashboard 现在做了 profile/source 级兼容：

- `dataset_origin=simulation` 或带仿真来源 marker 的 canonical joint 7D/14D 数据集，默认按当前 v3 合同处理：`absolute_joint_position_opening_fraction`，夹爪为 `absolute_opening_fraction_0_closed_1_open`。
- `dataset_origin=real/unknown` 的同类模糊 joint 数据集仍保持不兼容，需要显式写入合同元数据后才能训练。

这保证仿真数据可直接走训练/评测链路，同时不放松实机数据的安全约束。

## 18. 当前已探测的数据集位置（2026-08-05）

按 `SERVER_PATHS_ENV_TRAIN_EVAL.md` 重新核验后，仿真 Dashboard 的远端路径应使用 `/DATA/disk0/sunny`，不是旧模板里的 `/DATA/sync/sunny`。

### 18.1 H100 / login-server 共享路径

Dashboard 可在 `login-server` 轻量扫描：

```text
/DATA/disk0/sunny/.cache/huggingface/lerobot
```

当前探测到：

| dataset_id | episodes | frames | fps | root |
|---|---:|---:|---:|---|
| `lift_pot_piper` | 150 | 12385 | 50 | `/DATA/disk0/sunny/.cache/huggingface/lerobot` |

文档中的 Put Bottles v3 norm_stats 已在 H100 assets 下存在，但 H100 的 LeRobot dataset root 当前未探测到该 dataset 目录；训练前需按准备脚本重新 staging 数据。

### 18.2 H200-ali-01 Slurm 探测结果

H200 不由 Dashboard 直接 SSH 扫描；通过 login-server 提交 CPU-only Slurm probe，并把 inventory 写入：

```text
/DATA/NAS/GPUServer/sunny/dashboard_probe/h200-ali-01_inventory.json
```

当前探测到：

| dataset_id | episodes | frames | fps | root |
|---|---:|---:|---:|---|
| `lift_pot_piper` | 150 | 12385 | 50 | `/DATA/disk0/sunny/.cache/huggingface/lerobot` |
| `put_bottles_dustbin_piper_100_25hz_realqpos_v2` | 100 | 59798 | 25 | `/DATA/disk0/sunny/.cache/huggingface/lerobot` |
| `put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned` | 100 | 59798 | 25 | `/DATA/disk0/sunny/.cache/huggingface/lerobot` |
| `put_single_bottle_dustbin_piper_200` | 200 | 42489 | 50 | `/DATA/disk0/sunny/.cache/huggingface/lerobot` |

### 18.3 H200-ali-02 Slurm 探测结果

Inventory cache：

```text
/DATA/NAS/GPUServer/sunny/dashboard_probe/h200-ali-02_inventory.json
```

当前探测结果：`/DATA/disk0/sunny/.cache/huggingface/lerobot` 不存在或无 LeRobot 数据集。若要在 h200-ali-02 训练，需要先单独 staging 数据、项目、环境和 checkpoint。
