# 4×4090 数据 / π0.5 微调 / Policy 管理服务

## 架构

管理面和真实推理数据面彼此分离：

```text
机械臂控制电脑
  └─ openpi_client.WebsocketClientPolicy（OpenPI 官方协议）
       └─ 4×4090 OpenPI WebsocketPolicyServer
            ├─ π0.5 模型推理
            └─ 镜像 state / 两路图像 / prompt / action telemetry
                 └─ Dashboard 只读可视化
```

- 真实 `state + cam_high + cam_wrist + prompt` 由机械臂电脑直接发送到 Policy WebSocket 端口。
- Dashboard 不接收机械臂观测上传，也不代发 inference。
- Dashboard 负责数据集、norm、训练、GPU、checkpoint 和 Policy 进程管理。
- Dashboard 可以新建、健康检测、停止、强制结束 Policy，并用新 checkpoint 替换运行中的 Policy。
- 机械臂客户端默认是 shadow-only；只有显式添加 `--allow-execution`、Dashboard 对同一 Policy 给出未过期的 EXECUTE 授权、telemetry 新鲜且本地安全检查全部通过时，才会发布第一步 action。

## 部署并启动 Dashboard

在本仓库执行：

```bash
bash deploy_4090_server.sh
```

脚本只同步本服务需要的文件到 `4x4090:/home/sunny/bimanual-vla`。它会重启 Dashboard 本身，但不会停止页面管理的 Policy、训练任务或服务器上已有的其他 GPU 进程。首次启动会生成随机 Token，并打印：

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
ssh 4x4090 'source ~/.config/bimanual-vla/server.env && printf "%s\n" "$BIMANUAL_VLA_SERVER_TOKEN"'
```

验证流程：

1. `start_server.sh` 首次启动时用 `secrets.token_urlsafe(36)` 生成随机 Token，文件权限为 `0600`。
2. 浏览器将 Token 保存在当前浏览器的 `localStorage`，并在每个管理请求中发送 `Authorization: Bearer <token>`。
3. 上传脚本使用同一个 Bearer Token。
4. 服务端用恒定时间比较验证 Token；除 `/` 和 `/healthz` 外，所有 Dashboard API 都必须验证。

Token 只用于 Dashboard 管理 API；OpenPI Policy WebSocket 使用官方通信协议，机械臂客户端不需要这个 Token。不要把 Token 提交到 Git、放进 URL，或保存在不可信浏览器。需要轮换时：

```bash
ssh 4x4090 'bash /home/sunny/bimanual-vla/server_4090/stop_server.sh && rm -f ~/.config/bimanual-vla/server.env && bash /home/sunny/bimanual-vla/server_4090/start_server.sh'
```

自定义路径、端口或 JAX 显存比例时修改服务器上的：

```text
/home/sunny/bimanual-vla/server_4090/config.json
```

重启 Dashboard：

```bash
ssh 4x4090 'bash /home/sunny/bimanual-vla/server_4090/stop_server.sh'
ssh 4x4090 'bash /home/sunny/bimanual-vla/server_4090/start_server.sh'
```

## 上传数据集

```bash
python upload_dataset_4090.py /path/to/pi0_dataset_single \
  --name pick_cube_piper_r1 \
  --server http://192.168.101.9:8090 \
  --token "$BIMANUAL_VLA_SERVER_TOKEN" \
  --workers 4 \
  --chunk-mib 32
```

同一命令重跑会查询已存在分块并续传。服务端完成 SHA256、tar 安全、LeRobot v2.1 结构、视频帧数和 OpenPI loader 校验后才安装数据集。

当前 delivery schema：

```text
state:        10D = EEF xyz + rotation6d + gripper_closed_fraction
actions:       7D = base-frame Δxyz + left-multiplied Δrotvec + gripper target
image:         cam_high, RGB 256×256
wrist_image:   cam_wrist, RGB 256×256
```

## 下载训练基座权重

单臂 Piper LoRA 微调使用 `pi05_base`，不是 DROID 8D 的 `pi05_droid`：

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

1. 输入 Dashboard Token。
2. 查看数据集结构及 GPU 占用。
3. 选择 RTX 4090，提交 FSDP LoRA 微调：
   - `norm_stats.json` 已存在时直接启动训练；
   - 缺失时自动启动完整 norm 任务，训练进入持久化 `waiting_norm`；
   - norm 成功后自动启动训练；GPU 暂忙时进入 `waiting_gpu` 并自动重试；
   - norm 失败、丢失或未生成统计文件时，训练任务标记失败并显示依赖原因；
   - 同一数据集已有运行中的 norm 时复用该任务，Dashboard 重启后依赖仍可恢复。
4. “计算归一化统计”表单保留为手动重算或限制帧数调试入口，正常训练无需预先手动点击。
5. 页面扫描 `pi05_piper_single_arm_lora/<experiment>/<step>`，按数据集过滤完整 checkpoint。
6. 在“新建 / 切换 Policy 进程”中选择 GPU、端口和 checkpoint：
   - 留空“操作对象”：新建独立 Policy；
   - 选择运行中的 Policy：先停止旧进程，再从新 checkpoint 启动替代进程。
7. 在“Policy 进程管理”中查看：
   - PID 和进程状态；
   - WebSocket `/healthz`；
   - GPU、端口和 schema；
   - dataset 和 checkpoint；
   - 最近 telemetry / 客户端推理时间；
   - 日志、正常停止和强制结束。
8. 在机械臂控制电脑启动官方 WebSocket 客户端。
9. Dashboard 显示 Policy 实际收到的 10D state、两路图像、prompt、预测 action，以及服务端授权、客户端本地执行许可、双重门结果和实际执行/阻断原因。

服务端只列出同时包含 `params/` 和 `_CHECKPOINT_METADATA` 的完整 checkpoint，并通过 `assets/<dataset_id>/norm_stats.json` 判断 checkpoint 所属数据集。启动 Policy 时会再次校验，防止 checkpoint 与数据集错配。

## 机械臂电脑：官方 Policy 客户端

脚本必须运行在物理连接 Piper CAN 和两路 USB 相机的电脑，而不是 4×4090：

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

单次验证：

```bash
python robot_observation_bridge.py \
  --host 192.168.101.9 \
  --port 8000 \
  --can can0 \
  --cam-high-device /dev/video8 \
  --cam-wrist-device /dev/video16 \
  --instruction "pick up the cube" \
  --once
```

客户端行为：

1. 读取 `GetArmJointMsgs`、`GetArmGripperMsgs` 和 `GetArmEndPoseMsgs`。
2. 构造与 `collect_output_arm.py` 完全一致的 10D delivery state。
3. 以 30 FPS 打开 `/dev/video8` 和 `/dev/video16`，并行读取后转换为 256×256 RGB。
4. 使用 `openpi_client.websocket_client_policy.WebsocketClientPolicy` 直连所选 Policy 端口。
5. 校验服务端 metadata：`transport`、`schema`、`state_dim`、`action_dim` 和 `arm_side`。
6. 打印 action chunk；切换模型造成断线时自动重连。

该脚本不需要 Dashboard URL 或 Token。上述默认命令不会调用动作控制 API。

需要让客户端具备执行能力时，必须在机械臂电脑本地显式追加：

```bash
  --allow-execution
```

这只打开客户端本地安全门，并不立即执行。网页还必须为同一个运行中 Policy 输入 task id，授权最多 5 分钟的 EXECUTE；任一门关闭、授权过期、连接断开、telemetry 过期、动作超限或 Piper 状态异常都会阻断下发。网页可随时点击“只推理 / SHADOW”立即撤销服务端授权。

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
