# Bimanual-VLA Dashboard API 使用文档

最后更新：2026-08-05  
适用服务：

- 实机 Dashboard：`http://192.168.101.9:8090`
- 仿真 Dashboard：`http://192.168.101.9:8091`

本文档面向 AI agent / 自动化脚本调用 Dashboard。H100/H200 相关操作只通过 4×4090 Dashboard + `login-server` + Slurm 编排；不要在 H100/H200 上启动 Dashboard 或监听端口。

---

## 1. 通用约定

### 1.1 认证

除 `/`、`/healthz`、`/api/auth/token` 外，所有 API 都需要 Token：

```bash
export DASH=http://192.168.101.9:8091
export TOKEN='<dashboard token>'
curl -H "Authorization: Bearer $TOKEN" "$DASH/api/status"
```

等价 Header：

```http
Authorization: Bearer <token>
X-API-Token: <token>
```

媒体 URL（`<video>` 无法带 Header）支持 query token：

```text
/api/eval-videos/<video_id>?token=<token>
```

### 1.1.1 获取 Token

Token 不写入 Git；由 4×4090 上的 user systemd 启动脚本首次启动时生成，保存在 `sunny` 用户自己的 `server.env` 中。

仿真 Dashboard（8091）：

```bash
ssh 4x4090 'source ~/.config/bimanual-vla-sim-dashboard/server.env && printf "%s\n" "$BIMANUAL_VLA_SERVER_TOKEN"'
```

实机 Dashboard（8090）：

```bash
ssh 4x4090 'source ~/.config/bimanual-vla/server.env && printf "%s\n" "$BIMANUAL_VLA_SERVER_TOKEN"'
```

也可以直接在 4×4090 本机执行：

```bash
# 仿真 8091
source ~/.config/bimanual-vla-sim-dashboard/server.env
printf "%s\n" "$BIMANUAL_VLA_SERVER_TOKEN"

# 实机 8090
source ~/.config/bimanual-vla/server.env
printf "%s\n" "$BIMANUAL_VLA_SERVER_TOKEN"
```

如果你需要通过 Dashboard 登录账号密码换取 API token，可先查看对应服务的登录用户名。密码属于敏感信息，不要写入脚本、日志、Git 或聊天记录；只在交互式终端临时读取/复制：

```bash
# 仿真 8091 登录凭据位置
ssh 4x4090 'source ~/.config/bimanual-vla-sim-dashboard/server.env && printf "user=%s\n" "$BIMANUAL_VLA_LOGIN_USER"'

# 实机 8090 登录凭据位置
ssh 4x4090 'source ~/.config/bimanual-vla/server.env && printf "user=%s\n" "$BIMANUAL_VLA_LOGIN_USER"'
```

账号密码换 token：

```bash
curl -sS -X POST "$DASH/api/auth/token" \
  -H 'Content-Type: application/json' \
  -d '{"username":"<user>","password":"<password>"}'
```

返回：

```json
{"token":"...","token_type":"Bearer","username":"..."}
```

AI agent 自动化调用时推荐优先通过 SSH 读取 `BIMANUAL_VLA_SERVER_TOKEN` 到当前 shell 变量，而不是处理登录密码：

```bash
export DASH=http://192.168.101.9:8091
export TOKEN=$(ssh 4x4090 'source ~/.config/bimanual-vla-sim-dashboard/server.env && printf "%s" "$BIMANUAL_VLA_SERVER_TOKEN"')
curl -H "Authorization: Bearer $TOKEN" "$DASH/api/status" | jq .
```

### 1.2 错误格式

错误统一返回 JSON：

```json
{"error":"message","type":"ValueError"}
```

常见状态码：

- `401`：token 错误或缺失；
- `400`：参数、数据集、checkpoint、资源状态不满足要求；
- `404`：资源不存在；
- `500`：服务端异常。

### 1.3 常用 ID 与安全字符

`dataset_id`、`task_id`、`session_id` 等必须是安全名称，通常允许：

```text
A-Z a-z 0-9 . _ -
```

不要传空格、斜杠、中文或 shell 特殊字符。

### 1.4 任务状态

训练、评测、同步、policy、norm 都会返回 task：

```json
{
  "id":"train-20260805-...",
  "type":"train",
  "state":"running",
  "created_at":"...",
  "command":["..."],
  "metadata":{},
  "log_path":"..."
}
```

常见状态：

- 进行中：`starting`、`running`、`waiting_norm`、`waiting_gpu`、`stopping`
- 终态：`completed`、`failed`、`lost`、`stopped`、`skipped`

---

## 2. 服务与总览

### 2.1 Web 页面

```http
GET /
```

返回 HTML Dashboard，无需认证。

### 2.2 健康检查

```http
GET /healthz
```

返回：

```json
{"ok":true,"time":"2026-08-05T..."}
```

### 2.3 总状态

```http
GET /api/status
```

返回核心对象：

```json
{
  "datasets": [],
  "checkpoints": [],
  "experiments": [],
  "base_models": [],
  "robot_observation": {},
  "tasks": [],
  "gpus": [],
  "config": {
    "dashboard_profile":"simulation",
    "upload_default_origin":"simulation",
    "visible_dataset_origins":["simulation"],
    "cluster_targets": {},
    "dataset_root":"...",
    "upload_roots":{"real":"...","simulation":"..."},
    "checkpoint_base_dir":"...",
    "base_checkpoint":"...",
    "transfer_parallelism":4,
    "nas_dataset_staging_root":"/DATA/NAS/GPUServer/sunny/dashboard_dataset_sync"
  }
}
```

推荐 AI 首次调用顺序：

```bash
curl -H "Authorization: Bearer $TOKEN" "$DASH/api/status" | jq '{datasets:(.datasets|length),tasks:(.tasks|length),targets:(.config.cluster_targets|keys)}'
curl -H "Authorization: Bearer $TOKEN" "$DASH/api/dataset-locations?origin=simulation" | jq .
curl -H "Authorization: Bearer $TOKEN" "$DASH/api/tasks" | jq .
```

---

## 3. 数据集上传

上传采用分片协议。更推荐直接使用仓库脚本：

```bash
bin/bimanual-vla data-upload LEROBOT_OR_GUI_NPZ_DIR \
  --name my_dataset \
  --dataset-origin simulation \
  --server "$DASH" \
  --token "$TOKEN" \
  --workers 4 \
  --merge
```

仿真 Dashboard 默认上传到 4×4090：

```text
/home/sunny/.cache/huggingface/lerobot
```

实机 Dashboard 同理默认上传到 4×4090 的配置 `dataset_root`。

### 3.1 初始化上传

```http
POST /api/uploads/init
Content-Type: application/json
```

请求：

```json
{
  "dataset_name":"my_dataset",
  "dataset_origin":"simulation",
  "size":123456789,
  "sha256":"64位小写sha256",
  "chunk_size":67108864,
  "overwrite":false,
  "merge":true
}
```

字段：

- `dataset_name`：最终安装的数据集 ID；
- `dataset_origin`：`real` / `simulation` / `unknown`，上传时通常只能用 `real` 或 `simulation`；
- `size`：tar 包总字节数；
- `sha256`：tar 包整体 sha256；
- `chunk_size`：每片大小，不能超过配置 `max_chunk_mib`；
- `overwrite`：目标已存在时覆盖；
- `merge`：目标已存在时增量合并；`overwrite` 与 `merge` 互斥。

返回含 `upload_id`：

```json
{
  "id":"simulation-...",
  "chunk_count":10,
  "received":[0,1]
}
```

### 3.2 查询上传状态

```http
GET /api/uploads/<upload_id>
```

返回上传状态、已收到分片、安装结果或错误。

### 3.3 上传分片

```http
PUT /api/uploads/<upload_id>/chunks/<index>
X-Chunk-SHA256: <该分片sha256>
Content-Length: <bytes>
```

Body 是原始二进制分片。

### 3.4 完成上传并安装

```http
POST /api/uploads/<upload_id>/complete
```

服务端会：

1. 拼接分片；
2. 校验整体 sha256；
3. 解包到 staging；
4. 校验数据集结构与 LeRobot/OpenPI loader；
5. 原子安装到 `dataset_root/<dataset_name>`；
6. 写入来源 marker。

---

## 4. 数据集管理

### 4.1 获取数据集详情 / episode 列表

```http
GET /api/datasets/<dataset_id>?offset=0&limit=200
```

返回：

- 数据集 meta；
- episode 列表；
- media/image/video keys；
- 可编辑参数；
- train/test split 信息等。

`limit` 最大 500。

### 4.2 重命名数据集

```http
PATCH /api/datasets/<dataset_id>
Content-Type: application/json
```

请求：

```json
{"new_dataset_id":"new_name"}
```

注意：只重命名数据集目录和相关 norm 资产；历史 checkpoint / task 记录不会自动改名。

### 4.3 修改数据集来源

```http
PATCH /api/datasets/<dataset_id>/origin
Content-Type: application/json
```

请求：

```json
{"dataset_origin":"simulation"}
```

取值：`real` / `simulation` / `unknown`。

### 4.4 删除整个数据集

```http
DELETE /api/datasets/<dataset_id>
Content-Type: application/json
```

请求必须二次确认：

```json
{"confirm_dataset_id":"<dataset_id>"}
```

不会删除历史 checkpoint / task 记录。

### 4.5 获取 episode 视频

```http
GET /api/datasets/<dataset_id>/episodes/<episode_index>/video/<video_key>
```

返回 `video/mp4`。可用 query token：

```text
...?token=<token>
```

### 4.6 获取 episode 图像帧

```http
GET /api/datasets/<dataset_id>/episodes/<episode_index>/image/<image_key>/<frame_index>
```

返回图片二进制。

### 4.7 修改 episode 参数

```http
PATCH /api/datasets/<dataset_id>/episodes/<episode_index>
Content-Type: application/json
```

请求 body 是需要更新的 episode 字段。典型字段取决于数据集 metadata，例如：

```json
{
  "instruction":"pick up the cube",
  "task":"pick_cube",
  "tags":["eval"]
}
```

服务端会重新校验数据集，并使相关 norm stats 失效。

### 4.8 批量删除 episode

```http
POST /api/datasets/<dataset_id>/episodes/delete
Content-Type: application/json
```

请求：

```json
{"episode_indexes":[0,3,5]}
```

### 4.9 合并已安装数据集

```http
POST /api/datasets/<dataset_id>/merge
Content-Type: application/json
```

请求：

```json
{"source_dataset_id":"source_dataset"}
```

约束：`real` 与 `simulation` 一般不允许互相合并，除非其中一方是 `unknown`。

---

### 4.10 时间对齐（检查 → 新数据集）

数据集面板的“时间对齐”入口支持本机可读的 v3 joint / delivery 绝对动作数据。先检查，再生成新名称的数据集；原始数据不改动。远端数据须先同步到 4×4090。接口沿用 Dashboard Bearer token 鉴权。

```http
POST /api/datasets/<dataset_id>/time-alignment/inspect
Content-Type: application/json

{"fps":30,"max_skew_ms":50,"camera_offsets_ms":{"cam_high":0,"cam_left_wrist":0}}
```

返回 `202` 和 TaskManager 任务（含 `id`）。`fps` 可省略以沿用源频率，支持 15/20/25/30 Hz；偏差上限大于 0 且不超过 50 ms。相机修正单位 ms，**正值加到原时间戳上**，未知延迟保持 0。

```http
GET /api/datasets/<dataset_id>/time-alignment/jobs/<task_id>
```

返回 `task` 与 `report`。成功检查的 `report.status=inspected`，`can_align=true` 才可应用；报告含每 episode 有效帧率、原相机相对状态的有符号中位偏差、对齐后的最大偏差、预计输出大小/内存。缺时间戳、源变化、缺帧、低帧率升频等问题会阻止生成。

```http
POST /api/datasets/<dataset_id>/time-alignment/apply
Content-Type: application/json

{"preview_task_id":"<检查任务 id>","target_dataset_id":"cube_aligned"}
```

返回 `202`，用新任务 id 查询。完成须同时看到 `task.state=completed` 和 `report.status=completed`、`validation=passed`。目标名必须未占用；预览过期须重新检查。

- 最近真实样本重采样；不插值姿态、不复制图像补帧、不做滤波。15 FPS 相机不能生成真实 30 Hz 数据。
- 下一观测动作按新频率重建，并去掉缺少后继测量的末帧；同拍命令按动作时间戳选择。末端混合动作保留当前状态对应的夹爪命令及下一状态姿态。
- 保留原臂侧、相机名、关节单位、坐标系和夹爪语义；训练/验证 episode 划分保持。生成新的 metadata/statistics 与数据集 norm，训练前仍须计算对应模型的 norm 资产。
- `meta/time_alignment.json` 保存检查/生成报告；`meta/time_alignment/episode_*.npz` 保存精确源行映射，旁边保存原 info 与 episode metadata。原始接收时间对齐不能证明硬件曝光同步。
- 目前不重排 Event 标注与未知附加列；遇到这些数据会明确阻止，避免静默丢标签。没有原始时间戳的旧数据不能用帧号代替实测同步。
- CPU worker 在独立 transient user scope 中运行，检查自身 cgroup，GPU 不可见。request/result/runtime 文件保存到 `<workspace_root>/dataset_alignment/<request_id>/`；任务列表支持恢复查看、日志和停止。需正常的 user systemd 环境，失败不会回退到服务内处理。

## 5. 跨服务器数据集位置与同步

### 5.1 查询数据集在哪些服务器

```http
GET /api/dataset-locations
GET /api/dataset-locations?origin=simulation
GET /api/dataset-locations?origin=real
```

返回：

```json
{
  "datasets":[
    {
      "id":"put_bottles...",
      "duplicate_count":2,
      "origins":["simulation"],
      "targets":["local_4090","h200-ali-01"],
      "locations":[
        {
          "target":"local_4090",
          "label":"4×4090",
          "kind":"local",
          "path":"/home/sunny/.cache/huggingface/lerobot/...",
          "root":"/home/sunny/.cache/huggingface/lerobot",
          "episodes":100,
          "frames":59798,
          "fps":25,
          "updated_at":"2026-08-05 16:00:00"
        }
      ]
    }
  ],
  "locations":{
    "local_4090":{"label":"4×4090","kind":"local"},
    "h100":{"label":"H100 h100-ksy-01","kind":"ssh"},
    "h200-ali-01":{"label":"H200 ali-01","kind":"slurm_only"}
  },
  "errors":{}
}
```

说明：

- `local_4090`：4×4090 本地数据集；
- `h100`：通过 `login-server` 扫描 `/DATA/disk0/sunny/.cache/huggingface/lerobot`；
- `h200-*`：Slurm-only，读取 4×4090 本地镜像 inventory，不直接 SSH 到计算节点。

### 5.2 同步数据集到某个服务器

```http
POST /api/datasets/<dataset_id>/sync
Content-Type: application/json
```

请求：

```json
{
  "source":"local_4090",
  "target":"h100",
  "parallelism":8,
  "overwrite":false,
  "skip_existing":true
}
```

支持的 `source` / `target` 来自 `/api/dataset-locations` 的 `locations` key，常见：

```text
local_4090
h100
h200-ali-01
h200-ali-02
```

行为：

- local/H100 等 SSH 可访问位置：并行 tar 流；
- 涉及 H200 Slurm-only：自动使用 NAS staging + CPU-only Slurm copy job；
- 4×4090→H100 的多路 rsync 使用单个 SSH multiplex master 复用连接，避免 login-server 的多次 SSH 握手触发 `kex_exchange_identification`；默认/推荐 `parallelism=4`，中断后可安全重试并通过 `--partial/--append-verify` 续传；
- `skip_existing=true` 时完整且 manifest 一致的目标会直接跳过；若检测到中断留下的部分数据，则原地续传并在结束后重新校验 manifest，适合训练前幂等同步；
- 返回 `transfer` task，可用 `/api/tasks/<task_id>/status` 和 `/api/tasks/<task_id>/log` 查询；Dashboard 的“数据传输任务”表会自动刷新进度。
- 新版 dataset transfer runner 会在任务目录写入 `progress.json`，字段包括 `progress`、`completed_bytes` / `total_bytes`、`completed_files` / `total_files`、`completed_shards` / `total_shards`、`speed_bytes_per_sec` 和 `eta_seconds`。
- 旧版 transfer 若没有 sidecar，Dashboard 会根据传输日志和目标目录 manifest 给出节流后的保守估算；这类估算可能按完整文件跳变，不应当解释为精确的 rsync 当前块进度。

示例：

```bash
curl -X POST "$DASH/api/datasets/put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned/sync" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"source":"local_4090","target":"h200-ali-01","parallelism":8,"overwrite":false,"skip_existing":true}'
```


### 5.3 同步 H100/H200 checkpoint 回 4×4090

Policy 推理只能在 4×4090 本地启动。H100/H200 Slurm 训练完成后，可以把远端实验目录同步回 4×4090 checkpoint root：

```http
POST /api/checkpoints/sync
Content-Type: application/json
```

请求：

```json
{
  "source":"h100",
  "target":"local_4090",
  "config_name":"pi05_piper_single_arm_lora",
  "exp_name":"my_real_exp",
  "parallelism":8,
  "overwrite":false,
  "skip_existing":true
}
```

也可以用 `arm_mode` + `model_variant` 代替 `config_name`。涉及 H200 Slurm-only 时同样通过 NAS staging + CPU-only Slurm copy job，不在 H100/H200 上开启服务端口。返回 `transfer` task，日志里会显示源/目标 checkpoint 路径。

示例：

```bash
curl -X POST "$DASH/api/checkpoints/sync" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"source":"h100","target":"local_4090","config_name":"pi05_piper_single_arm_lora","exp_name":"my_real_exp","parallelism":8,"overwrite":false,"skip_existing":true}'
```

### 5.4 批量删除 checkpoint

Dashboard 训练页里的 checkpoint 表支持按实验筛选后的多选批量删除；后端接口仅删除完整训练 checkpoint 目录，不会删除任务记录、日志、数据集或训练输出。

```http
POST /api/checkpoints/batch-delete
Content-Type: application/json
```

请求：

```json
{
  "checkpoint_paths": [
    "/.../pi05_piper_single_arm_lora/exp_a/5000",
    "/.../pi05_piper_single_arm_lora/exp_a/10000"
  ]
}
```

返回：

```json
{
  "deleted": true,
  "deleted_count": 2,
  "checkpoint_paths": [...],
  "deleted_checkpoints": [...]
}
```

---

## 6. 采集会话接口

用于 AI/采集程序登记“准备采集/上传某个数据集”的状态，不直接采集重负载。

### 6.1 列出采集会话

```http
GET /api/collection-sessions
```

返回：

```json
{"sessions":[...]}
```

### 6.2 新建采集会话

```http
POST /api/collection-sessions
Content-Type: application/json
```

请求：

```json
{
  "dataset_id":"my_dataset",
  "dataset_origin":"simulation",
  "target":"local_4090",
  "status":"created",
  "metadata":{"task":"pick_cube"}
}
```

返回中含 `upload_command`，可直接给采集端执行。

### 6.3 查询单个采集会话

```http
GET /api/collection-sessions/<session_id>
```

### 6.4 更新采集会话

```http
PATCH /api/collection-sessions/<session_id>
Content-Type: application/json
```

可更新字段：

```json
{
  "status":"uploaded",
  "upload_task_id":"transfer-...",
  "notes":"...",
  "dataset_id":"new_dataset_id",
  "metadata":{"frames":12345}
}
```

---

## 7. 训练 norm / train / eval

### 7.1 单独计算 norm stats

```http
POST /api/tasks/norm
Content-Type: application/json
```

请求：

```json
{
  "dataset_id":"put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned",
  "arm_side":"both",
  "model_variant":"pi05",
  "base_checkpoint":"/home/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base",
  "batch_size":16,
  "num_workers":2,
  "test_ratio":0.1,
  "split_seed":42,
  "max_frames":null
}
```

说明：

- `dataset_id` 必填；
- `arm_side`：单臂用 `left`/`right`，双臂自动为 `both`；
- `model_variant`：`pi05` 或 `pi0`；
- `base_checkpoint` 可省略，使用默认基础模型；
- `test_ratio` / `split_seed` 会写入并复用 train/test split。

返回 `norm` task。

### 7.2 启动训练

```http
POST /api/tasks/train
Content-Type: application/json
```

本地 4×4090 示例：

```json
{
  "execution_target":"local_4090",
  "dataset_id":"put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned",
  "exp_name":"pi05-put-bottles-v3-test",
  "mode":"auto",
  "model_variant":"pi05",
  "base_checkpoint":"/home/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base",
  "gpu_ids":"0,1",
  "batch_size":2,
  "fsdp_devices":2,
  "num_workers":2,
  "num_train_steps":30000,
  "save_interval":1000,
  "keep_period":5000,
  "xla_memory_fraction":0.9,
  "test_ratio":0.1,
  "split_seed":42,
  "norm_batch_size":16,
  "norm_num_workers":2,
  "eval_enabled":true,
  "eval_interval_steps":5000,
  "eval_batch_size":1,
  "eval_num_workers":2,
  "eval_max_batches":50,
  "eval_seed":42,
  "eval_xla_memory_fraction":0.85,
  "wandb_enabled":false
}
```

H100/H200 Slurm 示例：

```json
{
  "execution_target":"h100",
  "cluster_gpus":1,
  "dataset_id":"put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned",
  "exp_name":"pi05-put-bottles-v3-h100",
  "mode":"auto",
  "model_variant":"pi05",
  "batch_size":1,
  "fsdp_devices":1,
  "num_workers":2,
  "num_train_steps":30000,
  "save_interval":1000,
  "keep_period":5000,
  "test_ratio":0.1,
  "split_seed":42,
  "eval_enabled":true,
  "eval_interval_steps":5000
}
```

字段说明：

- `execution_target`：`local_4090` / `h100` / `h200-ali-01` / `h200-ali-02`；
- `action_expert_tuning`：`full`（新实验默认）或 `lora`，独立选择 action expert 全量微调或 LoRA；
- `llm_lora`：布尔值，新实验默认 `false`（冻结 LLM），设为 `true` 时仅训练 LLM 的 LoRA；视觉编码器默认冻结；
- `train_vision`：布尔值，默认 `false`；设为 `true` 全量训练视觉编码器；
- `train_llm`：布尔值，默认 `false`；设为 `true` 全量训练 LLM，与 `llm_lora=true` 互斥。三部分全量训练使用 `train_vision=true`、`train_llm=true`、`llm_lora=false`、`action_expert_tuning="full"`，可同时开启 `rtc_training`；
- `rtc_training`：布尔值，默认 `false`；设为 `true` 启用 Pi05 干净动作前缀训练 RTC，仅支持 JAX/Orbax 路径；
- `rtc_training_max_delay`：1–49 个 action 步，默认 8。训练随机采样 0 到该上限的干净前缀，前缀不计入 loss；20 Hz 下 8 步为 400 ms；
- `rtc_training_delay_weights`：可选数组，长度必须为 `rtc_training_max_delay + 1`，依次对应 0 到上限的采样权重；必须非负、有限且总和大于零。省略时均匀采样，零权重的步数不会被采样。例如上限 10 时可传 `[39,0,0,0,0,574,873,275,101,32,22]`；
- `image_resize_mode`：`stretch` 直接缩放、保留全画面且不补黑边；`pad` 保留旧版等比例补边。API 兼容默认值为 `pad`，新实验 GUI 默认选 `stretch`。该设置随 checkpoint 保存。新采集/导出在生成时完成 224×224 图像准备；stretch 模型输入仅校验尺寸，不在训练时补边、裁剪或缩放。推理客户端在采集时直接生成 224×224；已有视频内的黑边需要预先处理。`stretch` 训练只保留颜色增强，不做几何裁剪或旋转；
- `gpu_ids`：本地 4×4090 用，如 `"0,1"`；
- `cluster_gpus` / `gpu_count`：Slurm 目标用，表示申请卡数；
- `mode`：`auto` / `new` / `resume` / `overwrite`；
- `resume_checkpoint`：可选的、已完成的数字 Orbax checkpoint 路径；仅用于 `mode=resume` 或 `mode=auto`，语义是 **full-state resume**（恢复 params、optimizer、EMA 和 step），不是基础模型权重；
- 当目标实验只有 `*.orbax-checkpoint-tmp-*` 临时目录时，必须显式传 `resume_checkpoint`；Dashboard 不会把 norm 配置里的 `base_checkpoint` 或 `pi05_base` 静默当成恢复源；
- `resume_checkpoint` 必须包含 `_CHECKPOINT_METADATA`、`params/_METADATA`、`train_state/_METADATA`，并且必须有与目标实机数据集完全匹配的 action-contract marker；不同 schema/动作语义（例如 EEF 7D 与 joint 7D）会被拒绝；
- `batch_size` 必须能被 GPU 数整除；
- `fsdp_devices` 必须整除 GPU 数；
- `keep_period`：传给 Orbax checkpoint manager，额外保护 step 能被该值整除的 checkpoint；例如 `save_interval=2000` 且 `keep_period=2000` 会保留 `4000/6000/8000/10000` 等 2k 倍数 checkpoint。传 `0`/空值可禁用周期保护（只保留最新 checkpoint）。
- `eval_interval_steps` 必须能被 `save_interval` 整除，且当前要求是 5000 的倍数；
- 若 norm 不存在，本地训练会自动先创建 norm task，并在 norm 完成后训练；Slurm 训练会在一个 Slurm job 内先 norm 后 train。
- H100/H200 重试可显式传 `reuse_norm_stats=true` 跳过 norm，前提是目标节点已有当前数据集的统计结果；训练启动仍检查 split manifest 和动作契约。数据内容变更后应重新计算，不应复用。
- 独立存储节点（`access_mode=slurm_only`）使用节点配置中的基础模型路径，并在 Slurm 作业内检查 `params`；不要求 4090 同时保存基础模型。NAS 不可用时，可设置节点的 `log_transport=sattach`，通过 Slurm 回传后续训练输出；`log_dir` 仍需是计算节点上已存在的目录。此方式不补传连接前的输出，也不替代数据集或 checkpoint 同步。
- 节点可设置 `checkpoint_chunk_pause_s`（0–60 秒），覆盖分块保存的等待时间。H200 可设为 `0` 以保留分块保存并取消额外等待；此配置仅影响新启动的 Slurm 作业。
- `execution_target` 为 H100/H200 时，Dashboard 会先根据远端 inventory 判断数据集是否存在；不存在或 inventory 不可用时，默认先执行一次幂等数据集同步（`auto_sync_dataset=true`，目标已存在则跳过），然后再提交 Slurm norm/train。
- 本地 Dashboard 训练默认使用 `training_task_launch_backend=nohup`：训练 runner 通过 `nohup` 和独立会话脱离 Dashboard/终端生命周期；任务状态仍由 `task_runner.py` 的 `exit.json` 和日志维护。该设置不等价于绕过内核/显存 OOM 或磁盘配额限制。
- Policy 推理只能在 4×4090 启动；H100/H200 训练出的 checkpoint 需要通过 `POST /api/checkpoints/sync` 同步回 4×4090 的本地 checkpoint root 后才能在实机 Policy 中选择。

返回 `train` task。若自动 norm，返回的 train task 可能处于 `waiting_norm`。

例如，在上述训练请求中加入以下字段可启用 action expert 全量微调、冻结
LLM，并训练 RTC：

```json
{
  "action_expert_tuning": "full",
  "llm_lora": false,
  "rtc_training": true,
  "rtc_training_max_delay": 8
}
```

对应训练 CLI 参数为 `--action-expert-tuning full --no-llm-lora
--rtc-training --rtc-training-max-delay 8`。参数写入 checkpoint 的 action-contract
marker，完整状态恢复和 serving 自动继承，不能更改这些设置。无该元数据的
旧 checkpoint 按旧配置（expert LoRA、LLM LoRA、无训练 RTC）加载。
从已微调 checkpoint 初始化新实验也必须保持 LoRA 网络结构；切换结构应选择
基础模型或先显式转换 adapter，不能静默丢弃权重。现有配置/目录名称中的
`_lora` 为兼容历史路径保留，实际训练方式以 `finetuning` 元数据为准。

Serving 的 `--rtc-mode auto` 自动为训练 RTC checkpoint 选择 `trained`，为旧
checkpoint 选择 `guided`（推理时 prefix guidance）。可显式选择两种模式；
`trained` 拒绝未经 RTC 训练的 checkpoint、超过训练上限的延迟和不足以覆盖
延迟的旧动作前缀。伺服后端与训练 RTC 开关相互独立。

### 7.3 启动 heldout loss 评测

```http
POST /api/tasks/eval
Content-Type: application/json
```

请求：

```json
{
  "execution_target":"local_4090",
  "dataset_id":"put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned",
  "checkpoint":"/home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints/.../10000",
  "model_variant":"pi05",
  "base_checkpoint":"/home/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base",
  "arm_side":"both",
  "gpu_ids":"2",
  "batch_size":1,
  "num_workers":2,
  "max_batches":50,
  "xla_memory_fraction":0.85,
  "cluster_gpus":1
}
```

说明：

- `checkpoint` 必须在允许的 checkpoint root 下；
- `execution_target` 为 Slurm 目标时用 `cluster_gpus`；
- 评测默认使用 test split。

---

## 8. Policy / 实机推理接口

仿真 Dashboard 通常 `enable_policy=false`，实机 Dashboard 使用以下接口。Policy 服务端只允许在 4×4090 本地启动，不支持在 H100/H200 上开端口或做实机推理。

### 8.1 启动或切换 policy

```http
POST /api/tasks/policy
Content-Type: application/json
```

请求：

```json
{
  "dataset_id":"8_3_64eps_v3_eef_absolute",
  "checkpoint":"/home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints/.../10000",
  "model_variant":"pi05",
  "arm_side":"right",
  "gpu_ids":"0",
  "port":8000,
  "default_prompt":"pick up the cube",
  "replace_task_id":""
}
```

返回 `policy` task，metadata 中包含：

```json
{"ws_url":"ws://192.168.101.9:8000","telemetry_session":"..."}
```

### 8.2 获取执行安全门状态

```http
GET /api/tasks/<task_id>/execution-control
```

### 8.3 执行权限

```http
POST /api/tasks/<task_id>/execution-control
Content-Type: application/json
```

此旧接口返回 HTTP 410。执行权限仅由机械臂客户端的 `--allow-execution` 设置；每条命令仍经过客户端本地安全检查。

### 8.4 机器人实时遥测

```http
GET /api/robot/observation
```

返回最近 policy/client 上报的 observation / latency / active plan / execution gate 摘要。

### 8.5 获取 policy 遥测图片

```http
GET /api/policy-telemetry/<session>/image/<view>
```

返回 JPEG。`session` 来自 policy task metadata，`view` 如 `image`、`wrist_image`、`cam_high` 等。

---

## 9. 评估视频接口

### 9.1 列出评估视频

```http
GET /api/eval-videos
```

查询参数：

| 参数 | 说明 |
|---|---|
| `limit` | 返回条数，默认 200，最大 1000 |
| `include_remote=true` | 额外扫描可 SSH 访问的远端视频；H200 Slurm-only 不直接扫 |
| `task` | 按任务/目录模糊筛选 |
| `experiment` | 按实验/checkpoint 模糊筛选 |
| `success` | `success` / `failed` / `unknown` / `all` |
| `q` | 路径/任务/实验/source 关键词 |

示例：

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "$DASH/api/eval-videos?limit=20&task=put_bottles&success=failed" | jq .
```

返回：

```json
{
  "videos":[
    {
      "id":"base64id",
      "name":"episode0.mp4",
      "display_name":"episode0.mp4",
      "relative_path":"put_bottles/.../episode0.mp4",
      "root":"/home/sunny/robotwin_ws/RoboTwin/eval_result",
      "source":"local_4090",
      "task":"put_bottles_dustbin",
      "experiment":"put_bottles_dustbin_piper_eval1_video",
      "success":"failed",
      "score":0.0,
      "size_mib":12.3,
      "updated_at":"2026-08-05 16:00:40",
      "url":"/api/eval-videos/<id>",
      "remote":false,
      "playable":true,
      "syncable":false
    }
  ],
  "total":22,
  "facets":{"tasks":[],"experiments":[],"success":{"success":0,"failed":22,"unknown":0}},
  "roots":["..."],
  "remote_errors":{}
}
```

注意：页面现在只加载列表；真正播放时再请求 `url`。

### 9.2 获取视频流

```http
GET /api/eval-videos/<video_id>?token=<token>
```

返回视频文件，支持 Range/conditional。

### 9.3 同步远端视频到 4×4090

```http
POST /api/eval-videos/sync
Content-Type: application/json
```

请求：

```json
{
  "source":"h100",
  "root":"/DATA/disk0/sunny/robotwin_eval/RoboTwin/eval_result",
  "relative_path":"put_bottles/.../episode0.mp4",
  "parallelism":4,
  "overwrite":false
}
```

返回 `transfer` task。H200 Slurm-only 视频不能直接通过该接口扫描/拉取，需要先通过 Slurm staging/NAS 发布到 4×4090 可访问位置。

### 9.4 批量删除评估视频

训练/评估页面的评估视频模块支持勾选多个**本地**视频并批量删除；远端视频只允许同步，不支持直接删除。

```http
POST /api/eval-videos/batch-delete
Content-Type: application/json
```

请求：

```json
{
  "video_ids": ["base64id1", "base64id2"]
}
```

返回：

```json
{
  "deleted": true,
  "deleted_count": 2,
  "video_ids": [...],
  "deleted_videos": [...]
}
```

---

## 10. 资源查询

### 10.1 H100/H200 资源快照

```http
GET /api/cluster-resources
GET /api/cluster-resources?all_jobs=true
GET /api/cluster-resources?native=true
```

返回：

```json
{
  "ok":true,
  "returncode":0,
  "elapsed_s":3.2,
  "output":"...",
  "note":"H100/H200 are queried via SSH/Slurm only; no remote Dashboard port is opened."
}
```

按配置的 `cluster_targets` 提交主机分组查询。默认输出节点 GPU 已分配/总数、
未分配数量、CPU/内存分配，以及配置分区中所有用户的运行作业、GPU 物理索引、
当前 SSH 用户的完整队列（包括尚未分配节点的排队任务）。`all_jobs=true` 额外显示
所有用户的完整队列。作业名称和索引不截断；任何查询段失败会令 `ok=false` 并保留
已取得的结果和错误说明。GPU 数量与索引来自 Slurm，不代表实时利用率、显存或 OS 进程。

---

## 11. 任务查询、日志、指标与删除

### 11.1 列出任务

```http
GET /api/tasks
```

返回：

```json
{"tasks":[...]}
```

### 11.2 查询任务状态

```http
GET /api/tasks/<task_id>/status
```

轻量状态接口，适合 AI 轮询。transfer 任务会额外返回：

```json
{
  "task": {
    "id": "transfer-...",
    "type": "transfer",
    "state": "running",
    "progress": {
      "progress": 0.526663,
      "completed_bytes": 11266863540,
      "total_bytes": 21392916007,
      "completed_files": 52,
      "total_files": 105,
      "completed_shards": 0,
      "total_shards": 4,
      "parallelism": 4,
      "speed_bytes_per_sec": 460303700.26,
      "eta_seconds": 22,
      "source": "local_4090",
      "target": "h100",
      "updated_at": "2026-08-07T11:22:25+0800"
    }
  }
}
```

`progress` 是 0–1 的比例；文件数和字节数以目标端已经可见的完整文件为准。由于 rsync 的临时文件可能尚未改名，传输中的进度是保守值，完成某个大文件时可能一次跳升。

### 11.3 查询任务完整信息

```http
GET /api/tasks/<task_id>
```

### 11.4 查询任务日志

```http
GET /api/tasks/<task_id>/log?tail=4000
```

`tail` 是返回末尾字符数，默认/上限以服务端实现为准。用于训练、norm、eval、transfer、policy 的 stdout/stderr 汇总日志。

### 11.5 查询训练指标曲线

```http
GET /api/tasks/<task_id>/metrics
```

返回从日志解析出的 loss/eval_loss 等序列，供 Dashboard 画曲线。

### 11.6 停止任务

```http
POST /api/tasks/<task_id>/stop
Content-Type: application/json
```

请求：

```json
{"force":false}
```

`force=true` 会强制结束本地受管进程。Slurm 任务的停止依赖 runner/Slurm 状态。

### 11.7 删除任务记录

```http
DELETE /api/tasks/<task_id>
```

只能删除终态任务记录；不会删除 checkpoint 或数据集。

### 11.8 批量删除任务记录和日志

```http
POST /api/tasks/batch-delete
Content-Type: application/json
```

请求体：

```json
{"task_ids":["train-20260801-120000-ab12cd34","eval-20260801-120500-ef56gh78"]}
```

一次最多删除 200 条任务。所有任务必须处于终态；如果某个任务仍在运行、仍被活动依赖，或任一任务校验失败，则整个批次不会删除任何记录。批量删除允许同时选择已结束的依赖任务和被依赖任务。删除范围仅限 Dashboard 任务目录中的任务记录和日志，不会删除 checkpoint、模型、数据集、训练输出或外部日志路径。

---

## 12. 推荐 AI 工作流

### 12.1 上传并同步仿真数据到 H200

```bash
# 1. 上传到 4×4090
bin/bimanual-vla data-upload ./my_lerobot_dataset \
  --name my_sim_dataset --dataset-origin simulation \
  --server "$DASH" --token "$TOKEN" --workers 4 --merge

# 2. 确认 4×4090 已安装
curl -H "Authorization: Bearer $TOKEN" "$DASH/api/dataset-locations?origin=simulation" | jq '.datasets[]|select(.id=="my_sim_dataset")'

# 3. 同步到 h200-ali-01
curl -X POST "$DASH/api/datasets/my_sim_dataset/sync" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"source":"local_4090","target":"h200-ali-01","parallelism":8,"overwrite":false}'

# 4. 轮询 task
curl -H "Authorization: Bearer $TOKEN" "$DASH/api/tasks/<task_id>/status" | jq .
curl -H "Authorization: Bearer $TOKEN" "$DASH/api/tasks/<task_id>/log?tail=8000"
```

### 12.2 提交 H100 训练

```bash
curl -X POST "$DASH/api/tasks/train" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{
    "execution_target":"h100",
    "cluster_gpus":1,
    "dataset_id":"put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned",
    "exp_name":"pi05-put-bottles-v3-h100-ai",
    "mode":"auto",
    "model_variant":"pi05",
    "batch_size":1,
    "fsdp_devices":1,
    "num_train_steps":30000,
    "save_interval":1000,
    "keep_period":5000,
    "test_ratio":0.1,
    "split_seed":42,
    "eval_enabled":true,
    "eval_interval_steps":5000
  }'
```

### 12.3 查评估失败视频列表并播放某个视频

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "$DASH/api/eval-videos?task=put_bottles&success=failed&limit=5" | jq '.videos[]|{id,relative_path,success}'

# 浏览器打开：
# http://192.168.101.9:8091/api/eval-videos/<id>?token=<TOKEN>
```

---

## 13. 端点总表

| Method | Path | 说明 |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/healthz` | 健康检查 |
| POST | `/api/auth/token` | 账号密码换 token |
| GET | `/api/status` | 总状态：数据集、任务、GPU、checkpoint、配置 |
| GET | `/api/robot/observation` | 最新机器人/policy 遥测 |
| GET | `/api/policy-telemetry/<session>/image/<view>` | policy 遥测图片 |
| POST | `/api/uploads/init` | 初始化分片上传 |
| GET | `/api/uploads/<upload_id>` | 上传状态 |
| PUT | `/api/uploads/<upload_id>/chunks/<index>` | 上传分片 |
| POST | `/api/uploads/<upload_id>/complete` | 完成上传并安装 |
| GET | `/api/datasets/<dataset_id>` | 数据集详情和 episodes |
| PATCH | `/api/datasets/<dataset_id>` | 重命名数据集 |
| PATCH | `/api/datasets/<dataset_id>/origin` | 修改数据集来源 |
| DELETE | `/api/datasets/<dataset_id>` | 删除数据集 |
| GET | `/api/datasets/<dataset_id>/episodes/<episode_index>/video/<video_key>` | episode 视频 |
| GET | `/api/datasets/<dataset_id>/episodes/<episode_index>/image/<image_key>/<frame_index>` | episode 图像帧 |
| GET | `/api/datasets/<dataset_id>/episodes/<episode_index>/analysis` | episode 逐帧分析、空闲帧区间、末端轨迹，以及 jitter/速度反向/剧烈变化异常标记（`max_points` 可调） |
| GET | `/api/datasets/<dataset_id>/episodes/<episode_index>/frames/<frame_index>` | 当前帧关节角、速度、加速度、目标值、异常类型和末端位姿 |
| POST | `/api/datasets/<dataset_id>/episodes/<episode_index>/crop` | 按 `start_frame` / `end_frame` 原子裁剪 episode（`/trim` 为别名） |
| PATCH | `/api/datasets/<dataset_id>/episodes/<episode_index>` | 修改 episode 参数 |
| POST | `/api/datasets/<dataset_id>/episodes/delete` | 批量删除 episodes |
| POST | `/api/datasets/<dataset_id>/merge` | 合并数据集 |
| GET | `/api/dataset-locations` | 查询 4×4090/H100/H200 数据集位置 |
| POST | `/api/datasets/<dataset_id>/sync` | 跨服务器同步数据集 |
| GET | `/api/collection-sessions` | 列采集会话 |
| POST | `/api/collection-sessions` | 新建采集会话 |
| GET | `/api/collection-sessions/<session_id>` | 查采集会话 |
| PATCH | `/api/collection-sessions/<session_id>` | 更新采集会话 |
| POST | `/api/tasks/norm` | 启动 norm stats 计算 |
| POST | `/api/tasks/train` | 启动训练 |
| POST | `/api/tasks/eval` | 启动 heldout loss 评测 |
| POST | `/api/tasks/policy` | 启动/切换 policy |
| GET | `/api/tasks/<task_id>/execution-control` | 查 policy 执行安全门 |
| POST | `/api/tasks/<task_id>/execution-control` | 设置 policy 执行安全门 |
| POST | `/api/tasks/<task_id>/stop` | 停止任务 |
| DELETE | `/api/tasks/<task_id>` | 删除终态任务记录 |
| POST | `/api/tasks/batch-delete` | 批量删除终态任务记录和日志 |
| GET | `/api/tasks` | 列任务 |
| GET | `/api/tasks/<task_id>/status` | 查任务轻量状态 |
| GET | `/api/tasks/<task_id>` | 查任务完整信息 |
| GET | `/api/tasks/<task_id>/log` | 查任务日志 |
| GET | `/api/tasks/<task_id>/metrics` | 查训练/评测指标 |
| GET | `/api/cluster-resources` | 查 H100/H200 Slurm 资源 |
| GET | `/api/eval-videos` | 列评估视频，支持筛选 |
| GET | `/api/eval-videos/<video_id>` | 获取视频流 |
| POST | `/api/eval-videos/sync` | 同步远端视频到 4×4090 |

### 拓展节点配置

在 Dashboard「设置 → 拓展节点」中编辑已有 H100/H200，或新增 Slurm 节点。SSH 别名及认证需事先在 Dashboard 服务器配置；页面不保存密码或私钥。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/settings/extension-nodes` | 返回 `nodes`，以节点 ID 为键 |
| PUT | `/api/settings/extension-nodes/<id>` | 新增节点或合并更新现有配置 |
| DELETE | `/api/settings/extension-nodes/<id>` | 仅移除节点配置，不删除远端文件 |

保存会持久化到 `cluster_targets` 并立即更新运行时配置，无需重启。训练、数据同步、Checkpoint 列表和资源查询使用该配置。有活跃任务引用的节点禁止修改或删除，需待任务结束。

接入方式支持 `login_shared_slurm`（提交主机可读取共享存储）和 `slurm_only`（通过 Slurm 访问节点独立存储）；暂不支持新增普通 SSH 独立工作站作为训练节点。

必填字段：`submit_host`、`partition`、`node`、`access_mode`、`openpi_python`、`workdir`、`dashboard_repo`、`dataset_root`、`assets_base_dir`、`checkpoint_base_dir`、`base_checkpoint`、`cache_root`、`log_dir`、`remote_job_dir`。路径均为远端绝对路径，保存时不以 4090 本地目录是否存在作为依据，也不自动安装环境或搬迁数据。

可选配置包括 GPU 类型/数量、CPU、内存（最多 512G）、时限、Conda、源码路径、目录扫描 Python，以及历史读取目录 `checkpoint_read_roots` / `dataset_read_roots`、NAS 中转目录。列表字段接受数组或每行一个路径。`GET /api/cluster-resources` 按配置的提交主机和节点查询，不再固定为三个原始节点。

### 远端 Checkpoint 列表与单步同步

`GET /api/checkpoints/remote` 返回 `locations`，每个位置包含 `source`、`label`、`state`（loading/ready/error）、`updated_at`、`error` 和 `checkpoints`。追加 `?refresh=1` 强制刷新；普通请求缓存10分钟，查询异步执行，不阻塞本地状态。
H100 在共享提交节点读取目录；H200 通过 CPU-only Slurm 作业读取节点独立磁盘，将小型目录清单通过 Slurm job comment 返回，不依赖 NAS。配置中的 `checkpoint_read_roots` 可补充历史目录；缺失目录或访问失败显式报错。

`POST /api/checkpoints/sync` 传入 `step` 时只同步选定的完整 checkpoint：

```json
{"source":"h100","target":"local_4090","config_name":"pi05_piper_bimanual_lora","exp_name":"fix_nopad_rtc_empirical_d10_20261019","step":20000,"parallelism":4,"skip_existing":true}
```

先读取远端列表，确认该 step 已完成。同步包含该步的参数、训练状态、assets，以及实验的动作契约/finetuning marker。不同契约拒绝覆盖；重复请求返回已有活跃任务，失败后可以续传。H200 checkpoint 数据传输仍使用既有 NAS staging（与不依赖NAS的列表查询不同）。完成后本地 checkpoint 列表/Policy 选择器自动可见。

### 统一训练历史与 rollout 产物（2026-10-23）

完整约束见 [EXPERIMENT_ARTIFACT_CONTRACT.md](../docs/EXPERIMENT_ARTIFACT_CONTRACT.md)。适用于 π0.5/OpenPI SFT、RLinf 和后续后端。物理根目录随节点/磁盘变化，根目录内使用固定层级：

```text
<root>/<环境来源>/<任务>/<实验>/<运行>/
  run.json
  metadata/ logs/ metrics/ checkpoints/
  train/<checkpoint>/<success|failed|unknown>/<episode>/
  eval/<checkpoint>/<success|failed|unknown>/<episode>/
```

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/artifact-runs` | 通用训练历史，返回 `experiment_run.v1`，支持 source/task/backend/kind/state/experiment 筛选 |
| POST | `/api/artifact-runs` | 分配本地运行目录和清单；不会提交训练或创建远端目录 |
| POST | `/api/eval-videos/catalog/refresh` | 提交视频目录整理与索引刷新任务，响应为含 `id` 的任务对象 |

分配请求示例：

```json
{"video_source":"libero-safety","task":"tsa-moka","experiment":"ppo-safety-v2","run_id":"train-20261023-180000-a1b2c3d4","backend":"rlinf","kind":"rl"}
```

SFT 与 RLinf 的训练提交接口均可附加 `artifact_identity`，内容为 `video_source`、`task`、`experiment`、`run_id`。旧客户端可以省略，无法确认的来源/任务保留 unknown。提交钩子登记通用运行清单及原生输出引用；第三方 writer 的物理输出路径仍须按约束配置，并在结束后规范化。不能将清单登记等同于全部产物已经重定向。

`GET /api/eval-videos` 追加筛选：`video_source`（环境）、`task`、`experiment`、`phase`（train/eval/unknown）、`success`（success/failed/unknown）、`safe_success`；事件使用 `events=with|without|all`，保留原有 `source`（存储来源）、`q` 等。响应包含 `facets`、`total`、`offset`、`limit`；使用 offset 分页，limit 最大 1000。RLinf 远端视频清单由后台定时刷新，远端条目需同步后播放。

视频条目含 `evaluation_video.v1` 分类字段、`catalog_relative_path` 和 provenance。任务成功与安全成功分开；训练曲线中的总体成功率不能当作某个视频成功的证据。RoboTwin 旧事件记录和人工修订继续保留。

`POST /api/eval-videos/sync` 接受列表返回的 `source`、`root`、`relative_path`，支持已配置的 RLinf 视频源，写入规范目录并保存元数据 sidecar。同步后媒体接口支持 HTTP Range。视频聚合目录主要使用链接，删除索引链接保留原始 rollout。
