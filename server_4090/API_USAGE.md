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
python upload_dataset_4090.py LEROBOT_OR_GUI_NPZ_DIR \
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
  "overwrite":false
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
- 返回 `transfer` task，可用 `/api/tasks/<task_id>/status` 和 `/api/tasks/<task_id>/log` 查询。

示例：

```bash
curl -X POST "$DASH/api/datasets/put_bottles_dustbin_piper_100_25hz_realqpos_v3_order_aligned/sync" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"source":"local_4090","target":"h200-ali-01","parallelism":8,"overwrite":false}'
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
  "test_ratio":0.1,
  "split_seed":42,
  "eval_enabled":true,
  "eval_interval_steps":5000
}
```

字段说明：

- `execution_target`：`local_4090` / `h100` / `h200-ali-01` / `h200-ali-02`；
- `gpu_ids`：本地 4×4090 用，如 `"0,1"`；
- `cluster_gpus` / `gpu_count`：Slurm 目标用，表示申请卡数；
- `mode`：`auto` / `new` / `resume` / `overwrite`；
- `batch_size` 必须能被 GPU 数整除；
- `fsdp_devices` 必须整除 GPU 数；
- `eval_interval_steps` 必须能被 `save_interval` 整除，且当前要求是 5000 的倍数；
- 若 norm 不存在，本地训练会自动先创建 norm task，并在 norm 完成后训练；Slurm 训练会在一个 Slurm job 内先 norm 后 train。

返回 `train` task。若自动 norm，返回的 train task 可能处于 `waiting_norm`。

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

仿真 Dashboard 通常 `enable_policy=false`，实机 Dashboard 使用以下接口。

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

### 8.3 设置执行安全门

```http
POST /api/tasks/<task_id>/execution-control
Content-Type: application/json
```

请求：

```json
{
  "mode":"shadow",
  "ttl_s":30,
  "reason":"manual dashboard toggle"
}
```

`mode` 常见值：

- `shadow`：只推理/观测，不允许下发执行；
- `execute`：短时授权执行，仍需客户端允许和本地安全检查。

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
  "command":["scripts/query_h100_h200_resources.sh","--compact"],
  "output":"...",
  "note":"H100/H200 are queried via SSH/Slurm only; no remote Dashboard port is opened."
}
```

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

轻量状态接口，适合 AI 轮询。

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

---

## 12. 推荐 AI 工作流

### 12.1 上传并同步仿真数据到 H200

```bash
# 1. 上传到 4×4090
python upload_dataset_4090.py ./my_lerobot_dataset \
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
| GET | `/api/tasks` | 列任务 |
| GET | `/api/tasks/<task_id>/status` | 查任务轻量状态 |
| GET | `/api/tasks/<task_id>` | 查任务完整信息 |
| GET | `/api/tasks/<task_id>/log` | 查任务日志 |
| GET | `/api/tasks/<task_id>/metrics` | 查训练/评测指标 |
| GET | `/api/cluster-resources` | 查 H100/H200 Slurm 资源 |
| GET | `/api/eval-videos` | 列评估视频，支持筛选 |
| GET | `/api/eval-videos/<video_id>` | 获取视频流 |
| POST | `/api/eval-videos/sync` | 同步远端视频到 4×4090 |
