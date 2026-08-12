# 4×4090 数据 / π0.5 微调 / Policy 管理服务


## 仿真 Dashboard

仿真训练 / 评测 / 视频查看的独立 8091 服务和 AI API 使用流程见：

```text
server_4090/SIMULATION_DASHBOARD.md
```

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
- 机械臂客户端默认是 shadow-only；只有显式添加 `--allow-execution`、Dashboard 对同一 Policy 给出未过期的 EXECUTE 授权、telemetry 新鲜、`action_horizon >= 16` 且本地安全检查全部通过时，才会发布异步 chunk 命令。
- 机械臂客户端默认把完整本地监测轨迹追加保存到 `./monitoring_data/<session>/events.jsonl`；可用 `--monitoring-dir` 指定其他目录。记录器在后台线程写盘，不阻塞 20 Hz 控制循环；原始图像不写入 JSONL，只保留相机设备和时间戳。

## 部署并启动 Dashboard

在本仓库执行：

```bash
bash deploy_4090_server.sh
```

脚本只同步本服务需要的文件到 `4x4090:/home/sunny/bimanual-vla`，安装并启用用户级 systemd 服务 `bimanual-vla-dashboard.service`。Dashboard 会随 4×4090 开机自动启动，并在异常退出后自动重启；重启 Dashboard 本身不会停止页面管理的 Policy、训练任务或服务器上已有的其他 GPU 进程。首次启动会生成随机 Token，Dashboard 地址为：

```text
http://192.168.101.9:8090
```

Token 保存在服务器：

```text
~/.config/bimanual-vla/server.env
```

随时读取现有 Token：

```bash
ssh 4x4090 'source ~/.config/bimanual-vla/server.env && printf "%s\n" "$BIMANUAL_VLA_SERVER_TOKEN"'
```

Dashboard 也支持用账号密码换取 Token。启动脚本会在首次启动时生成一组独立的 Dashboard 登录凭据，并与 Token 一起保存在权限为 `0600` 的 `server.env` 中：

```bash
ssh 4x4090 'source ~/.config/bimanual-vla/server.env && printf "user=%s\npassword=%s\n" "$BIMANUAL_VLA_LOGIN_USER" "$BIMANUAL_VLA_LOGIN_PASSWORD"'
```

网页顶部的“账号密码获取 Token”按钮会调用 `POST /api/auth/token`，验证成功后自动把返回的 Bearer Token 保存到当前浏览器。命令行也可以这样获取（不要把密码放在 URL 中）：

```bash
curl -sS -X POST http://192.168.101.9:8090/api/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"LOGIN_USER","password":"LOGIN_PASSWORD"}'
```

验证流程：

1. Dashboard 启动脚本首次运行时用 `secrets.token_urlsafe(36)` 生成随机 Token，文件权限为 `0600`。
2. 浏览器将 Token 保存在当前浏览器的 `localStorage`，并在每个管理请求中发送 `Authorization: Bearer <token>`。
3. 上传脚本使用同一个 Bearer Token。
4. 服务端用恒定时间比较验证 Token；除 `/` 和 `/healthz` 外，所有 Dashboard API 都必须验证。

Token 只用于 Dashboard 管理 API；OpenPI Policy WebSocket 使用官方通信协议，机械臂客户端不需要这个 Token。不要把 Token 提交到 Git、放进 URL，或保存在不可信浏览器。需要轮换时：

```bash
ssh 4x4090 'systemctl --user stop bimanual-vla-dashboard.service && rm -f ~/.config/bimanual-vla/server.env && systemctl --user start bimanual-vla-dashboard.service'
```

自定义路径、端口或 JAX 显存比例时修改服务器上的：

```text
/home/sunny/bimanual-vla/server_4090/config.json
```

管理 Dashboard 自启动服务：

```bash
ssh 4x4090 'systemctl --user status bimanual-vla-dashboard.service'
ssh 4x4090 'systemctl --user restart bimanual-vla-dashboard.service'
ssh 4x4090 'journalctl --user -u bimanual-vla-dashboard.service -n 100 --no-pager'
ssh 4x4090 'tail -n 100 ~/.local/share/bimanual-vla-server/dashboard.log'
```

部署脚本会尝试为当前用户开启 systemd linger，使用户尚未登录时服务也能随系统启动。可用 `loginctl show-user sunny -p Linger` 验证；如果服务器策略拒绝无管理员授权开启 linger，用户服务仍会在 `sunny` 登录后自动启动，但需要管理员执行 `loginctl enable-linger sunny` 才能实现完全无人登录的开机自启动。

## 上传数据集

上传器自动支持两种输入目录：

- GUI 原始采集目录：顶层包含 `ep_*.npz`，客户端自动校验并导出 LeRobot；
- 已导出的 LeRobot v2.1 目录：包含 `meta/info.json`，直接打包上传。

直接上传 GUI 原始采集目录并增量追加：

```bash
python upload_dataset_4090.py /path/to/gui_episodes \
  --name pick_cube_piper_r1 \
  --dataset-origin real \
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
  --dataset-origin real \
  --server http://192.168.101.9:8090 \
  --token "$BIMANUAL_VLA_SERVER_TOKEN" \
  --workers 4 \
  --chunk-mib 32 \
  --merge
```

真机上传默认使用 `--dataset-origin real`，仿真上传使用 `--dataset-origin simulation`。服务端分开暂存到 `uploads/real` 与 `uploads/simulation`，安装后通过 `meta/dashboard_dataset_origin.json` 保留来源分类，同时继续使用 OpenPI 兼容的统一 LeRobot 根目录。

原始 NPZ 自动导出缓存在 `~/.cache/bimanual-vla/uploads/exports`。同一命令重跑会复用未变化的导出和 tar，并查询服务端已有分块后断点续传；`--rebuild` 强制重建本地缓存。服务端完成 SHA256、tar 安全、LeRobot v2.1 结构、视频帧数和 OpenPI loader 校验后才安装数据集。

- `--merge` 和 `--overwrite` 互斥；目标不存在时按首次安装处理。
- 合并要求版本、robot type、FPS、chunk size、features、action semantics 和 action offset 完全兼容。
- 新 episodes 会重新生成连续的 `episode_index`、全局 `index` 和 `task_index`；视频只复制/硬链接，不重新编码。
- 合并在隐藏临时目录执行，最终数据集通过结构和实际 loader 校验后才原子替换旧目录。

Piper 合同区分 **raw action（数据集存储）**、**model action（norm/训练）** 和 **wire action（Policy 输出）**，双臂向量顺序固定为 `left + right`：

| 模式 | schema / 版本 | state | raw action | model / wire action |
|---|---|---:|---:|---:|
| 单臂 | joint v3 | 7D 关节 + opening fraction | 7D 绝对关节目标 + opening fraction | 训练为前 6D current-anchored delta；wire 经 `AbsoluteActions` 恢复 7D 绝对目标 |
| 双臂 | joint v3 | 14D | 14D | 14D，逐臂前 6D delta，夹爪保持绝对 opening fraction |
| 单/双臂 | delivery v3 | 每臂 10D EEF xyz + rotation6d + opening fraction | 每臂 10D 绝对 EEF target | 每臂 7D `[delta_xyz, log(Rt*Rs.T), absolute opening fraction]`，chunk 所有行共用同一当前 state |
| 单臂 | delivery legacy v2（如 `8_3_64eps`） | legacy `state` 10D + closed fraction | legacy `actions` 7D one-step delta + closed fraction | 仅通过显式 `step` 或转换后的 `chunk_origin` convention 使用 |
| 单/双臂 | joint legacy v2 | 每臂 7D，夹爪为 opening metres | 每臂 7D 绝对目标，夹爪为 metres | 新训练默认先把 metres 转为 fraction 再 norm；旧 checkpoint 必须显式保留 metres wire 语义 |

新 delivery 必须是 canonical `observation.state/action` 的 `(10,10)` 或 `(20,20)`；`(10,7)` / `(20,14)` 一律识别为 legacy delivery step-delta，不能仅依赖 `schema=delivery` 混用。

所有 model/wire 行采用统一目标时间：第 `i` 行对应 `t_obs + (i + 1) / fps`。LeRobot raw action 的 delta timestamp 从 `(step + 1 - action_offset) / fps` 查询：`same_step_command` 的 `action_offset=0` 从下一帧 action 开始，`next_measured_*_fallback` 的 `action_offset=1` 从当前 action 行开始；两种 raw 对齐最终都转换成从 `+1 step` 开始的 model/wire chunk。`action_offset` 只能是 0/1，`model_action_start_offset=1` 固定且不可由 CLI 改成其他值。

canonical LeRobot 相机字段为：

```text
single:   observation.images.cam_high + observation.images.cam_<side>_wrist
bimanual: observation.images.cam_high + observation.images.cam_left_wrist + observation.images.cam_right_wrist
```

同一 Policy 进程只运行与其 checkpoint 一致的一种合同；机械臂 bridge 会读取握手 metadata，并自动选择匹配的 7D/10D/14D/20D state、相机 key 和动作执行方式，不能把 EEF 与 joint 数值直接互相解释。

## 网页 Episode 编辑器

Dashboard 的“Episode 级数据集编辑”区域支持：

1. 同时显示 `contract_version`、state/raw/model action 维度、raw/model convention、动作语义和 raw/wire 夹爪语义；明确区分 legacy delivery `(10,7)/(20,14)` 与 canonical absolute-EEF `(10,10)/(20,20)`，不完整或冲突的合同只开放管理和预览。
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

## 下载和选择训练基座权重

Dashboard 训练表单支持动态选择 `π0.5` 或 `π0` 模型系列，并会扫描 `checkpoint_allowed_roots` 下所有包含完整 `params/` 的预训练权重和训练 checkpoint。首次使用时至少准备一个与所选模型系列匹配的基座；例如下载 `pi05_base`：

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
3. 选择模型系列、基础模型和 RTX 4090，提交 FSDP LoRA 微调：
   - 可选择 `π0.5` / `π0`，也可把已有完整 checkpoint 作为初始化权重；
   - 不同模型系列使用独立 config、norm stats 和 checkpoint 目录，避免互相覆盖；
   - norm 时确定的测试集比例、划分种子和 episode 清单持久化到数据集 `meta/train_test_split.json`，训练自动加载，不需要重复填写；
   - 每次成功 norm 还会保存 `episode_split.json` 和 `norm_config.json`，除模型/批量/处理规模外，强制记录并校验 `contract_version`、raw/model action 维度、raw/model semantics/convention、模型夹爪语义、`action_offset` 和 `model_action_start_offset=1`；任一字段变化都会使旧 norm 失效；
   - norm batch size 只影响统计阶段的吞吐与资源占用，不需要和训练 batch size 一致；
   - 2×24 GiB RTX 4090 的 π0.5 LoRA 已验证安全起点为全局 batch `2`、`fsdp_devices=2`、`xla_memory_fraction=0.90`；batch `4` 可能在首个训练步的 NCCL 通信阶段 OOM；
   - Dashboard 按 GPU UUID 设置 `CUDA_VISIBLE_DEVICES`，避免某张故障卡从 CUDA 枚举中消失后数字序号错位；`nvidia-smi` 出现 `[N/A]` compute context 的卡会标记为不可训练；
   - 正式训练启动前要求每张 4090 至少有 23000 MiB 空闲显存，避免数据采集、仿真或其他进程与 JAX 预分配/NCCL 抢占显存；
   - `norm_stats.json` 已存在且 episode 划分一致时直接启动训练；
   - 缺失时自动启动完整 norm 任务，训练进入持久化 `waiting_norm`；
   - norm 成功后自动启动训练；GPU 暂忙时进入 `waiting_gpu` 并自动重试；
   - norm 失败、丢失或未生成统计文件时，训练任务标记失败并显示依赖原因；
   - 同一数据集、模型系列、基础权重和划分参数已有运行中的 norm 时复用该任务，Dashboard 重启后依赖仍可恢复；
   - 启动方式默认使用 `auto`：实验目录存在时等价于 `--resume`，不存在时创建新训练；只有明确选择 `overwrite` 才会删除原 checkpoint。
4. “计算归一化统计”表单用于首次确定或主动修改 episode 划分，也可手动重算或限制帧数调试；训练提交时会复用已保存划分，缺少 norm 时自动补算。
5. 训练模块集中展示 Norm / Train 进程管理、任务日志和指标曲线；从日志提取 `Step N: key=value`，绘制 `loss`、`loss_physical_14d`、`loss_padding_18d` 等曲线，并显示 step 进度、latest/min/max；图例按钮可切换 `grad_norm`、`param_norm` 等其他指标。
6. 页面按模型系列和数据集臂模式扫描 `pi05_piper_*_lora/<experiment>/<step>` 与 `pi0_piper_*_lora/<experiment>/<step>`，过滤完整 checkpoint，并在训练模块列出 checkpoint 表；可以按实验筛选，并批量删除选中的完整 checkpoint 目录（不会删除任务记录、日志或数据集）。
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
10. Dashboard 按 schema 显示 Policy 实际收到的单臂 7D/10D 或双臂 14D/20D state、Policy 要求的两路/三路图像、prompt、7D/14D 预测 action，以及 4 Hz inference launch、20 Hz action/control、horizon、每步目标时间、actuator delay、动态 expired prefix、active plan/hold/blend/gripper filter、queue generation/remaining、underrun/rejected/drop、最后 wire/decoded target 和实际执行/阻断原因。每条真实命令还按 `command_sequence + generation + source_index + queue_index` 对齐显示 IK 前末端目标、完整 IK 解、限速后关节目标、Piper `JointCtrl`/`GripperCtrl` 整数、下一控制周期反馈及指令跟随误差。
    客户端本地 `monitoring_data/<session>/events.jsonl` 同时保留这些字段的逐控制周期历史，以及推理结果、策略连接、Piper/相机状态和阻断/错误事件，适合离线分析。

训练指标 API（Bearer Token 必需）：

```text
GET    /api/tasks/<train_task_id>/metrics?max_points=1200
DELETE /api/tasks/<task_id>
POST   /api/tasks/batch-delete
```

指标接口最多读取任务日志尾部 16 MiB，同一步的后出现记录覆盖前记录，并对返回曲线降采样；latest/min/max 汇总仍基于读取到的全部指标点。删除接口只接受终态任务，并拒绝删除仍被活动训练依赖的 norm；删除范围仅限 Dashboard 任务目录中的记录和日志。Dashboard 训练/Policy 任务表支持勾选多个终态任务，通过 `POST /api/tasks/batch-delete` 一次删除最多 200 条记录和日志；批次校验失败时不会部分删除。

服务端只列出同时包含 `params/` 和 `_CHECKPOINT_METADATA` 的完整 checkpoint，并通过 `assets/<dataset_id>/norm_stats.json` 判断 checkpoint 所属数据集。启动 Policy 时会再次校验，防止 checkpoint 与数据集错配。

## 机械臂电脑：官方 Policy 客户端

脚本必须运行在物理连接 Piper CAN 和相机的电脑，而不是 4×4090；单臂使用一个 CAN 和两路相机：

```bash
python robot_observation_bridge.py \
  --host 192.168.101.9 \
  --port 8000 \
  --can can0 \
  --cam-high-device auto \
  --cam-wrist-device auto \
  --arm-side right \
  --instruction "pick up the cube" \
  --hz 4
```

`robot_observation_bridge.py` 默认使用 `--output-mode auto`，按 Policy 握手
metadata 自动选择 `delivery` 或 `joint`。部署 joint 模型时可以显式锁定合同：

```bash
python robot_observation_bridge.py \
  --host 192.168.101.9 \
  --port 8099 \
  --can can0 \
  --cam-high-device /dev/video4 \
  --cam-wrist-device /dev/video16 \
  --arm-side right \
  --output-mode joint \
  --instruction "pick up the cube"
```

显式模式只做合同校验，不会把服务端输出强行重解释：若服务端 metadata
声明的 `schema` 不是所选模式，客户端会在握手阶段拒绝连接，不会发送机器人
指令。`--policy-schema` 是同一参数的别名；需要兼容旧 delivery 模型时使用
`--output-mode delivery`。

双臂 shadow-only 示例使用两个 CAN 和三路相机：

```bash
python robot_observation_bridge.py \
  --host 192.168.101.9 \
  --port 8000 \
  --arm-mode bimanual \
  --arm-side both \
  --left-can can1 \
  --right-can can3 \
  --cam-high-device auto \
  --cam-left-wrist-device auto \
  --cam-right-wrist-device auto \
  --instruction "pick up the cube" \
  --hz 4
```

单次验证可在任一示例末尾追加 `--once`。

客户端行为：

1. 从每个活动 Piper 读取 `GetArmJointMsgs`、`GetArmGripperMsgs` 和 `GetArmEndPoseMsgs`；双臂固定按 `left + right` 拼接。
2. 单臂同时构造 10D delivery state 和 7D joint qpos；双臂同时构造 20D delivery state 和 14D joint qpos。
3. 单臂并行读取顶部与单腕部两路相机，双臂并行读取顶部与左右腕部三路相机，并转换为 256×256 RGB。
4. 使用 `openpi_client.websocket_client_policy.WebsocketClientPolicy` 直连所选 Policy 端口。
5. 校验服务端 metadata：除 `transport/schema/arm/state/action/camera/action_hz/action_horizon` 外，还校验 `action_time_step_s=1/fps`、`action_start_offset_steps=1`、`action_offset`、`model_action_start_offset=1`、`contract_version`、raw/model action dim、raw/model/wire semantics/convention 及 state/wire gripper semantics；`action_horizon < 16` 或任一合同不一致时 fail closed。
6. 根据 metadata 自动发送匹配的 state 和相机字段，并只接受 `model_action_dim` 指定的 7D/14D wire action；legacy joint checkpoint 继续使用 metres，新 checkpoint 使用 opening fraction。
7. 每次推理把当前观测保存为唯一 anchor；Dashboard telemetry 同时显示 raw/model/wire 合同、chunk 生命周期和最后 wire/decoded target。切换模型造成断线时自动重连并重新协商合同。

Policy metadata 会继续发布：`action_hz` 等于数据集 `meta/info.json` 的 `fps`（实测通常为
20 Hz），`action_horizon` 等于模型输出 horizon（通常为 50），`action_time_step_s=1/fps`，
`action_start_offset_steps=1`。执行客户端和 Dashboard 都要求 `action_horizon >= 16`，
并校验 raw `action_offset` 与固定的 model/wire 起点；缺失或不一致时保持 SHADOW / fail closed。

异步执行采用长 chunk 流水切换：控制循环持续以 20 Hz 消费旧 chunk，
而推理默认以 4 Hz 启动，推理与传输通常约 200 ms；旧 chunk 在 inference launch 到
result arrival 期间继续执行。新 chunk 到达后，根据每行绝对目标时间、
launch/capture/arrival、actuator delay 和当前 20 Hz control tick 动态计算已过期 prefix；
过期行数不是固定常量。随后用 2~4 步做 old/new blend，融合完成后才切换到新的 queue
`generation`。`chunk rows` 是返回 chunk 的实际行数，`active plan`、`hold`、`blend`、
`gripper filter`、`old remaining` / `new remaining`、`underrun`、`rejected result` 和
`drop reason` 必须在 telemetry 中保留。新 delivery current-anchored 行共享同一当前 state，客户端把每一行
相对于该 immutable anchor 独立解码为 absolute EEF target。legacy delivery `step`
checkpoint 只有在显式 metadata/convention 下才可累计到同一
anchor，不能与 canonical absolute-EEF 静默混用。`joint` wire action 已是绝对关节目标，
按 20 Hz 逐行执行。

该脚本不需要 Dashboard URL 或 Token。上述默认命令不会调用动作控制 API。

需要让客户端具备执行能力时，必须在机械臂电脑本地显式追加：

```bash
  --allow-execution
```

这只打开客户端本地安全门，并不立即执行。网页还必须为同一个运行中 Policy 输入 task id，授权最多 5 分钟的 EXECUTE；任一门关闭、授权过期、连接断开、telemetry 过期、`action_horizon < 16`、delivery 命令/工作空间超限、joint 目标/命令变化超限或 Piper 状态异常都会阻断下发。网页可随时点击“只推理 / SHADOW”立即撤销服务端授权。每一条实际的 20 Hz command 都独立执行安全检查，不会因为 chunk、skip 或 blend 而放宽限制。

`8_3_64eps` 全量 20 Hz 统计对应的 delivery 默认阈值为：translation `0.05 m/step`、rotation `0.18 rad/step`、gripper `0.30 fraction/step`；workspace 为 x `[-0.05, 0.30] m`、y `[0.01, 0.50] m`、z `[0.14, 0.52] m`。这些是默认拒绝边界，CLI 参数只能进一步收紧；请在改变阈值前重新确认数据分布和机械臂工作空间。

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
- 客户端没有 `--allow-execution` 时永远不会发布动作；即使双重门打开，动作新鲜度、每条 20 Hz command 的位移/旋转/夹爪变化、workspace、`action_horizon` 和 Piper 状态仍会在本地逐次检查。
- 默认拒绝在已有计算进程的 GPU 上启动任务；如修改 `allow_busy_gpus`，应明确确认不会干扰其他任务。
