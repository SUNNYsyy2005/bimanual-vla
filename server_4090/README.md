# 4×4090 数据 / π0.5 微调 / Policy 管理服务

## 架构

管理面和真实推理数据面彼此分离：

```text
机械臂控制电脑
  └─ openpi_client.WebsocketClientPolicy（OpenPI 官方协议）
       └─ 4×4090 OpenPI WebsocketPolicyServer
            ├─ π0.5 模型推理
            └─ 镜像 state / 单臂两路或双臂三路图像 / prompt / action telemetry
                 └─ Dashboard 只读可视化
```

- 真实观测由机械臂电脑直接发送到 Policy WebSocket 端口：单臂为 `state + cam_high + 单腕部相机 + prompt`，双臂为 `state + cam_high + cam_left_wrist + cam_right_wrist + prompt`。
- Dashboard 不接收机械臂观测上传，也不代发 inference。
- Dashboard 负责数据集、norm、训练、GPU、checkpoint 和 Policy 进程管理。
- 页面顶部按“总览 / 数据集 / 训练 / Policy / 实时遥测”分模块导航；总览集中显示 GPU、数据量和活动任务。
- Dashboard 可以新建、健康检测、停止、强制结束 Policy，并用新 checkpoint 替换运行中的 Policy。
- 已完成、失败、丢失或停止的训练 / Policy 历史任务可从对应模块删除任务记录和日志；checkpoint、模型与训练输出不会被删除。
- 机械臂客户端默认是 shadow-only；只有显式添加 `--allow-execution`、Dashboard 对同一 Policy 给出未过期的 EXECUTE 授权、telemetry 新鲜且本地安全检查全部通过时，才会发布第一步 action。

## 部署并启动 Dashboard

在本仓库执行：

```bash
bash deploy_4090_server.sh
```

脚本只同步本服务需要的文件到 `4x4090-wg:/home/sunny/bimanual-vla`。它会重启 Dashboard 本身，但不会停止页面管理的 Policy、训练任务或服务器上已有的其他 GPU 进程。首次启动会生成随机 Token，并打印：

```text
URL: http://192.168.101.9:8090
Token: ...
```

Token 保存在服务器：

```text
~/.config/bimanual-vla/server.env
```

随时读取现有 Token：

```bash
ssh 4x4090-wg 'source ~/.config/bimanual-vla/server.env && printf "%s\n" "$BIMANUAL_VLA_SERVER_TOKEN"'
```

Dashboard 也支持用账号密码换取 Token。`start_server.sh` 会在首次启动时生成一组独立的 Dashboard 登录凭据，并与 Token 一起保存在权限为 `0600` 的 `server.env` 中：

```bash
ssh 4x4090-wg 'source ~/.config/bimanual-vla/server.env && printf "user=%s\npassword=%s\n" "$BIMANUAL_VLA_LOGIN_USER" "$BIMANUAL_VLA_LOGIN_PASSWORD"'
```

网页顶部的“账号密码获取 Token”按钮会调用 `POST /api/auth/token`，验证成功后自动把返回的 Bearer Token 保存到当前浏览器。命令行也可以这样获取（不要把密码放在 URL 中）：

```bash
curl -sS -X POST http://192.168.101.9:8090/api/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"LOGIN_USER","password":"LOGIN_PASSWORD"}'
```

验证流程：

1. `start_server.sh` 首次启动时用 `secrets.token_urlsafe(36)` 生成随机 Token，文件权限为 `0600`。
2. 浏览器将 Token 保存在当前浏览器的 `localStorage`，并在每个管理请求中发送 `Authorization: Bearer <token>`。
3. 上传脚本使用同一个 Bearer Token。
4. 服务端用恒定时间比较验证 Token；除 `/` 和 `/healthz` 外，所有 Dashboard API 都必须验证。

Token 只用于 Dashboard 管理 API；OpenPI Policy WebSocket 使用官方通信协议，机械臂客户端不需要这个 Token。不要把 Token 提交到 Git、放进 URL，或保存在不可信浏览器。需要轮换时：

```bash
ssh 4x4090-wg 'bash /home/sunny/bimanual-vla/server_4090/stop_server.sh && rm -f ~/.config/bimanual-vla/server.env && bash /home/sunny/bimanual-vla/server_4090/start_server.sh'
```

自定义路径、端口或 JAX 显存比例时修改服务器上的：

```text
/home/sunny/bimanual-vla/server_4090/config.json
```

重启 Dashboard：

```bash
ssh 4x4090-wg 'bash /home/sunny/bimanual-vla/server_4090/stop_server.sh'
ssh 4x4090-wg 'bash /home/sunny/bimanual-vla/server_4090/start_server.sh'
```

## 上传数据集

上传器自动支持两种输入目录：

- GUI 原始采集目录：顶层包含 `ep_*.npz`，客户端自动校验并导出 LeRobot；
- 已导出的 LeRobot v2.1 目录：包含 `meta/info.json`，直接打包上传。

直接上传 GUI 原始采集目录并增量追加：

```bash
python upload_dataset_4090.py /path/to/gui_episodes \
  --name pick_cube_piper_r1 \
  --server http://192.168.101.9:8090 \
  --token "$BIMANUAL_VLA_SERVER_TOKEN" \
  --workers 4 \
  --fps 20 \
  --merge
```

上传已导出的 LeRobot 目录：

```bash
python upload_dataset_4090.py /path/to/pi0_dataset_single \
  --name pick_cube_piper_r1 \
  --server http://192.168.101.9:8090 \
  --token "$BIMANUAL_VLA_SERVER_TOKEN" \
  --workers 4 \
  --chunk-mib 32 \
  --merge
```

原始 NPZ 自动导出缓存在 `~/.cache/bimanual-vla/uploads/exports`。同一命令重跑会复用未变化的导出和 tar，并查询服务端已有分块后断点续传；`--rebuild` 强制重建本地缓存。服务端完成 SHA256、tar 安全、LeRobot v2.1 结构、视频帧数和 OpenPI loader 校验后才安装数据集。

- `--merge` 和 `--overwrite` 互斥；目标不存在时按首次安装处理。
- 合并要求版本、robot type、FPS、chunk size、features、action semantics 和 action offset 完全兼容。
- 新 episodes 会重新生成连续的 `episode_index`、全局 `index` 和 `task_index`；视频只复制/硬链接，不重新编码。
- 合并在隐藏临时目录执行，最终数据集通过结构和实际 loader 校验后才原子替换旧目录。

支持四种 canonical Piper 数据合同，双臂向量顺序固定为 `left + right`：

| 模式 | schema | state | action | 相机 |
|---|---|---:|---:|---|
| 单臂 | joint | 7D 关节角 + 夹爪开度 | 7D 绝对关节/夹爪目标 | `cam_high` + 单腕部 |
| 单臂 | delivery | 10D EEF xyz + rotation6d + 夹爪闭合比例 | 7D Δxyz + Δrotvec + 夹爪目标 | `cam_high` + 单腕部 |
| 双臂 | joint | 14D = left 7D + right 7D | 14D = left 7D + right 7D | `cam_high` + 左右腕部 |
| 双臂 | delivery | 20D = left 10D + right 10D | 14D = left 7D + right 7D | `cam_high` + 左右腕部 |

canonical LeRobot 相机字段为：

```text
single:   observation.images.cam_high + observation.images.cam_<side>_wrist
bimanual: observation.images.cam_high + observation.images.cam_left_wrist + observation.images.cam_right_wrist
```

同一 Policy 进程只运行与其 checkpoint 一致的一种合同；机械臂 bridge 会读取握手 metadata，并自动选择匹配的 7D/10D/14D/20D state、相机 key 和动作执行方式，不能把 EEF 与 joint 数值直接互相解释。

## 网页 Episode 编辑器

Dashboard 的“Episode 级数据集编辑”区域支持：

1. 识别并显示单臂 Joint 7D/7D、单臂 Delivery 10D/7D、双臂 Joint 14D/14D、双臂 Delivery 20D/14D，以及其他自定义 state/action 维度；四种 canonical Piper 格式均可进入对应训练配置，自定义格式只开放管理和预览。
2. 分页查看 episode 的帧数、instruction、可选 task name、可选 success 和附加 metadata；未设置的可选字段保持为空，不会被自动写成 `success=true`。
3. 数据集级重命名和整库删除；重命名同步移动当前 dataset-level norm stats，删除整库不删除历史 checkpoint、模型或任务记录。
4. 独立预览该 episode 的各路摄像头媒体：
   - `dtype: video` 直接播放 MP4；
   - `dtype: image` 按数据集 FPS 播放逐帧图片，并支持暂停和拖动帧索引；既支持 `images/` 外部文件，也支持 Parquet 内嵌 image bytes。
5. 修改 instruction、task name、success 和 JSON metadata。
6. 批量删除错误 episode；剩余 episode、parquet、video、image、可选 raw NPZ 和 metadata 会连续重新编号。
7. 把服务器另一个兼容数据集的全部 episodes 合并到当前目标数据集；源数据集保持不变。

编辑器不提供帧级裁剪，也不会修改 state、action 或图像帧。写操作有以下保护：

- 数据集被正在运行或等待中的 norm/train/Policy 任务使用时拒绝修改。
- 所有修改都在临时目录完成，并通过结构校验和 LeRobot/OpenPI loader 校验后原子替换。
- 原目录保留为隐藏的 `.DATASET.backup-*`；失败自动回滚。
- 对应训练配置下的 `assets/pi05_piper_single_arm_lora/DATASET/norm_stats.json` 或 `assets/pi05_piper_bimanual_lora/DATASET/norm_stats.json` 会被重命名为 `norm_stats.invalidated-*`，防止继续使用旧统计；下一次提交训练会自动重新计算 norm stats。
- 所有 API 都要求 Dashboard Bearer Token。

相关管理 API：

```text
GET    /api/datasets/<dataset_id>?offset=0&limit=100
PATCH  /api/datasets/<dataset_id>                         # 重命名
DELETE /api/datasets/<dataset_id>                         # 删除整库，需 confirm_dataset_id
PATCH  /api/datasets/<dataset_id>/episodes/<episode_index>
POST   /api/datasets/<dataset_id>/episodes/delete
POST   /api/datasets/<dataset_id>/merge
GET    /api/datasets/<dataset_id>/episodes/<episode_index>/video/<video_key>
GET    /api/datasets/<dataset_id>/episodes/<episode_index>/image/<image_key>/<frame_index>
```

LeRobot 数据集的 `meta/info.json` 中 `total_videos: 0` 只表示没有编码后的 MP4，**不代表没有摄像头画面**。例如 `my_dataset` 的 `image` 和 `wrist_image` feature 均为 `dtype: image`，画面可以位于 `images/<camera>/episode_x/frame_x.png`，也可以内嵌在 Parquet image bytes 中；Dashboard 会通过上述 image API 逐帧读取，不需要破坏性地转换原数据集。

## 下载训练基座权重

单臂和双臂 Piper LoRA 微调都使用 `pi05_base`，不是 DROID 8D 的 `pi05_droid`：

```bash
cd /home/sunny/bimanual-vla
/home/sunny/miniconda3/envs/openpi/bin/python download_openpi_checkpoint.py \
  --checkpoint gs://openpi-assets/checkpoints/pi05_base \
  --source auto \
  --workers 16 \
  --chunks-per-file 16
```

默认保存到：

```text
/home/sunny/.cache/openpi/openpi-assets/checkpoints/pi05_base
```

## 页面工作流

1. 输入 Dashboard Token，或在页面顶部用 Dashboard 账号密码获取 Token。
2. 通过顶部导航进入各模块，查看数据集结构、GPU 占用和活动任务。
3. 选择 RTX 4090，提交 FSDP LoRA 微调：
   - `norm_stats.json` 已存在时直接启动训练；
   - 缺失时自动启动完整 norm 任务，训练进入持久化 `waiting_norm`；
   - norm 成功后自动启动训练；GPU 暂忙时进入 `waiting_gpu` 并自动重试；
   - norm 失败、丢失或未生成统计文件时，训练任务标记失败并显示依赖原因；
   - 同一数据集已有运行中的 norm 时复用该任务，Dashboard 重启后依赖仍可恢复；
   - 启动方式默认使用 `auto`：实验目录存在时等价于 `--resume`，不存在时创建新训练；只有明确选择 `overwrite` 才会删除原 checkpoint。
4. “计算归一化统计”表单保留为手动重算或限制帧数调试入口，正常训练无需预先手动点击。
5. 训练模块集中展示 Norm / Train 进程管理、任务日志和指标曲线；从日志提取 `Step N: key=value`，绘制 `loss`、`loss_physical_14d`、`loss_padding_18d` 等曲线，并显示 step 进度、latest/min/max；图例按钮可切换 `grad_norm`、`param_norm` 等其他指标。
6. 页面按数据集臂模式扫描 `pi05_piper_single_arm_lora/<experiment>/<step>` 或 `pi05_piper_bimanual_lora/<experiment>/<step>`，过滤完整 checkpoint，并在训练模块列出 checkpoint 表。
7. 在“新建 / 切换 Policy 进程”中选择 GPU、端口和 checkpoint：
   - 留空“操作对象”：新建独立 Policy；
   - 选择运行中的 Policy：先停止旧进程，再从新 checkpoint 启动替代进程。
8. 在“Policy 进程管理”中查看：
   - PID 和进程状态；
   - WebSocket `/healthz`；
   - GPU、端口和 schema；
   - dataset 和 checkpoint；
   - 最近 telemetry / 客户端推理时间；
   - 独立 Policy 日志、正常停止、强制结束，以及终态历史记录删除。
9. 在机械臂控制电脑启动官方 WebSocket 客户端。
10. Dashboard 按 schema 显示 Policy 实际收到的单臂 7D/10D 或双臂 14D/20D state、Policy 要求的两路/三路图像、prompt、7D/14D 预测 action，以及服务端授权、客户端本地执行许可、双重门结果和实际执行/阻断原因。

训练指标 API（Bearer Token 必需）：

```text
GET    /api/tasks/<train_task_id>/metrics?max_points=1200
DELETE /api/tasks/<task_id>
```

指标接口最多读取任务日志尾部 16 MiB，同一步的后出现记录覆盖前记录，并对返回曲线降采样；latest/min/max 汇总仍基于读取到的全部指标点。删除接口只接受终态任务，并拒绝删除仍被活动训练依赖的 norm；删除范围仅限 Dashboard 任务目录中的记录和日志。

服务端只列出同时包含 `params/` 和 `_CHECKPOINT_METADATA` 的完整 checkpoint，并通过 `assets/<dataset_id>/norm_stats.json` 判断 checkpoint 所属数据集。启动 Policy 时会再次校验，防止 checkpoint 与数据集错配。

## 机械臂电脑：官方 Policy 客户端

脚本必须运行在物理连接 Piper CAN 和相机的电脑，而不是 4×4090；单臂使用一个 CAN 和两路相机：

```bash
python robot_observation_bridge.py \
  --host 192.168.101.9 \
  --port 8000 \
  --can can0 \
  --cam-high-device /dev/video8 \
  --cam-wrist-device /dev/video16 \
  --arm-side right \
  --instruction "pick up the cube" \
  --hz 5
```

双臂 shadow-only 示例使用两个 CAN 和三路相机：

```bash
python robot_observation_bridge.py \
  --host 192.168.101.9 \
  --port 8000 \
  --arm-mode bimanual \
  --arm-side both \
  --left-can can1 \
  --right-can can3 \
  --cam-high-device /dev/video8 \
  --cam-left-wrist-device /dev/video12 \
  --cam-right-wrist-device /dev/video16 \
  --instruction "pick up the cube" \
  --hz 5
```

单次验证可在任一示例末尾追加 `--once`。

客户端行为：

1. 从每个活动 Piper 读取 `GetArmJointMsgs`、`GetArmGripperMsgs` 和 `GetArmEndPoseMsgs`；双臂固定按 `left + right` 拼接。
2. 单臂同时构造 10D delivery state 和 7D joint qpos；双臂同时构造 20D delivery state 和 14D joint qpos。
3. 单臂并行读取顶部与单腕部两路相机，双臂并行读取顶部与左右腕部三路相机，并转换为 256×256 RGB。
4. 使用 `openpi_client.websocket_client_policy.WebsocketClientPolicy` 直连所选 Policy 端口。
5. 校验服务端 metadata：`transport`、`schema`、`arm_mode`、`state_dim`、`action_dim`、`arm_side`、`action_semantics` 和 `camera_keys`；不一致时 fail closed。
6. 根据 metadata 自动发送匹配的 7D/10D/14D/20D state 和相机字段，并只接受匹配的 7D/14D action。
7. 打印 action chunk；切换模型造成断线时自动重连并重新协商合同。

该脚本不需要 Dashboard URL 或 Token。上述默认命令不会调用动作控制 API。

需要让客户端具备执行能力时，必须在机械臂电脑本地显式追加：

```bash
  --allow-execution
```

这只打开客户端本地安全门，并不立即执行。网页还必须为同一个运行中 Policy 输入 task id，授权最多 5 分钟的 EXECUTE；任一门关闭、授权过期、连接断开、telemetry 过期、delivery 动作/工作空间超限、joint 目标/单步变化超限或 Piper 状态异常都会阻断下发。网页可随时点击“只推理 / SHADOW”立即撤销服务端授权。joint 默认限制为每关节 `0.3 rad/step`、夹爪 `0.02 m/step`，可分别用 `--max-joint-step-rad` 和 `--max-joint-gripper-step-m` 收紧。

## Policy 进程管理与模型切换

### 新建

网页向 `POST /api/tasks/policy` 提交白名单参数，Dashboard 启动独立进程并记录 PID、日志、GPU、端口、checkpoint 和 telemetry session。

### 停止

- “停止”：向整个 Policy 进程组发送 `SIGTERM`。
- “强制结束”：仅在正常停止无响应时发送 `SIGKILL`。
- Dashboard 不会根据 GPU PID 杀死不属于自身任务管理器的进程。

### 切换 checkpoint

网页提交 `replace_task_id` 后，服务端执行：

1. 停止选中的旧 Policy；
2. 等待端口和 GPU 释放；
3. 必要时对卡死的旧 Policy 强制结束；
4. 在所选 checkpoint 上创建新 Policy 任务；
5. 机械臂客户端自动重连。

切换会先把旧 Policy 强制切回 SHADOW，再中断已有 WebSocket 连接；替代 Policy 默认也是 SHADOW，必须重新满足双重门条件后才能执行。

## 安全边界

- Dashboard 管理接口需要 Token，并只接受白名单参数，不接受任意 shell。
- 真实观测不经过 Dashboard HTTP API。
- Dashboard telemetry 是 Policy 收到数据后的只读镜像。
- 服务端 EXECUTE 授权最长 1 小时，网页默认 5 分钟；Dashboard 重启、Policy 停止或模型切换都会回到 SHADOW。
- 客户端没有 `--allow-execution` 时永远不会发布动作；即使双重门打开，动作新鲜度、单步位移/旋转/夹爪变化、工作空间和 Piper 状态仍会在本地逐次检查。
- 默认拒绝在已有计算进程的 GPU 上启动任务；如修改 `allow_busy_gpus`，应明确确认不会干扰其他任务。
