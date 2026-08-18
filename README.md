# bimanual-vla

双臂 / 单臂 Piper 遥操作采集、pi0.5/openpi 推理接入与训练数据导出工具集。

现场快速操作请参阅 [Piper GUI 操作手册](GUI_OPERATION_GUIDE.md)。数据协议、命令行批次导出和完整排障说明请参阅 [Piper 数据采集操作指南](DATA_COLLECTION_GUIDE.md)。

## 当前可用能力

- 双臂主从遥操作采集：`teleop.py`
- 单臂主从遥操作采集：`teleop_single.py`
- 3 路相机采集：`camera.py`
- 单臂/双臂执行输出臂反馈采集（2/3 路 RGB）：`collect_output_arm.py`
- episode 多视角 + 关节角回放：`view_episode.py`
- 图形化采集界面：`collect_gui.py` / `start_gui.sh`
- LeRobot v2.1 导出：`export_lerobot.py`
- 数据验收：`validate_piper_data.py`、`check_pi05_dataset.py`
- 旧关节空间 NPZ 转换：`convert_output_arm_npz.py`
- Legacy Delivery 数据集升级到 v3：`migrate_legacy_delivery_dataset.py`
- OpenPI 权重并行断点续传：`download_openpi_checkpoint.py`
- 上传至 4×4090：`upload_dataset_4090.py`
- 4×4090 数据 / 微调 / Policy / shadow inference 网页：`server_4090/`

### 图形化采集界面

启动：

```bash
cd /home/sunny/bimanual-vla
bash start_gui.sh
```

界面可选择：

- 单臂 / 双臂；
- `joint` / `delivery` schema；
- 单臂 left / right；
- 单臂 CAN + 两路相机，或左右 CAN + 顶部/左腕/右腕三路相机；
- 采集 FPS、相机源 FPS、任务、instruction 和输出目录。

连接成功后会实时显示所有活动相机、单臂 7 个或双臂 14 个关节值、
schema 对应的 7D/10D/14D/20D state，以及相对 Home 的位姿误差。任一
机械臂超出默认 ±5° / ±5 mm 阈值时，界面以红色警告但仍允许开始 episode；无反馈或反馈过期时才会拒绝采集；
双臂 Reset 会并行复位左右输出臂。连接期间会锁定 arm mode、schema、CAN、
相机、FPS 和输出目录，避免 UI 配置与当前设备会话不一致。

保存 episode 列表下方的“Convert / upload dataset”可在后台执行 NPZ→LeRobot
转换或上传。上传 token 只放在 GUI 内存和子进程环境变量中，不会出现在命令行；
原始 NPZ 的转换结果复用 `~/.cache/bimanual-vla/uploads/exports` 缓存。
“Delete selected data”不会直接抹除文件，而是将选中的 `ep_*.npz` 移到当前
目录下的 `.trash/`，需要时可以手动恢复。连接成功后，相机标题和状态栏会同时
显示稳定 by-path 链接解析出的实际 `/dev/videoN` 编号，便于确认顶部/腕部配对。

### 单双臂 π0.5 数据合同

| 模式 | schema | state | raw action | model/wire action | 相机 |
|---|---|---:|---:|---:|---|
| 单臂 | joint | 7D | 7D absolute joint target | 7D | `cam_high` + 单腕部 |
| 双臂 | joint | 14D | 14D absolute joint target | 14D | `cam_high` + 左右腕部 |
| 单臂 | delivery | 10D absolute EEF | 10D absolute EEF target | 7D current-anchored delta | `cam_high` + 单腕部 |
| 双臂 | delivery | 20D absolute EEF | 20D absolute EEF target | 14D current-anchored delta | `cam_high` + 左右腕部 |

双臂向量永远按 `left + right` 拼接。v3 `joint` 每臂为 6 个关节角加
`gripper_opening_fraction`；v3 `delivery` 每臂 state/raw action 均使用
base-frame xyz、rotation-6D 和绝对夹爪开口比例。夹爪统一为 `0=闭合、1=张开`。
训练边界才把 10D absolute EEF target 转为 7D current-anchored
`Δxyz + Δrotvec + absolute gripper`，同一 action chunk 的所有行共享同一个当前观测 anchor。

动作来源必须区分：

- `collect_output_arm.py` / `collect_gui.py` 只读输出臂反馈，无法看到遥操命令；
  joint action 标记为下一帧实测绝对关节目标，delivery raw action 保存下一帧
  实测 absolute EEF target；两者均明确标为 fallback，并使用
  `action_alignment=next_observation`、`action_offset=1`。
- `teleop_single.py` / `teleop.py` 的 Joint 模式仍记录主臂映射后的同周期
  joint target；Delivery 模式不读取主臂 EEF pose，而是使用“下一帧从臂实测
  EEF pose + 当前周期主臂 gripper opening fraction”。对应
  `action_alignment=next_observation_pose_same_step_gripper`、`action_offset=1`，
  因此不需要主从 EEF 空间标定。

默认时序固定为：采集/模型动作/机器人控制 `20 Hz`，模型异步推理启动约
`4 Hz`。每个预测目标的时间为 `t_obs + (i+1)×0.05 s`。推理和传输约
200 ms 时，控制线程继续消费上一条 chunk；新结果到达后按当前时间和执行器
延迟动态丢弃过时目标，再用默认 3 步位姿融合切入新 chunk。夹爪独立做低通、
迟滞和连续预测确认，不参与 old/new 位姿线性融合。
OpenPI 默认输出 50 步，执行客户端要求至少 16 步，推理期间不会停下控制等待结果。

导出前验收：

```bash
python validate_piper_data.py --input-dir episodes_piper_v21 --target-fps 20
```

验收会按 episode metadata 检查单双臂维度、活动相机、同步误差、shape/dtype、
动作语义、terminal padding、空帧/冻结帧和动作统计。

导出 LeRobot v2.1：

```bash
python export_lerobot.py \
  --input-dir episodes_piper_v21 \
  --repo-id piper/piper_v1 \
  --root piper/piper_v1 \
  --fps 20
```

导出使用 `instruction` 写入 LeRobot `meta/tasks.jsonl`。完整字段和动作语义见
`PIPER_DATA_CONTRACT.md` 和 `PI05_PIPER_7D_10D_DATA_ACTION_DESIGN.md`。

### Collection UI backend

新的采集 UI 不应自行构造 `state`、`actions` 或 NPZ 字典。统一调用：

- `piper_data_contract.py`：唯一的数据协议定义与 episode 序列化实现；
- `collection_session.py`：连接、开始、采样、停止、保存、丢弃状态机；
- `validate_piper_data.py`：保存后的强制协议验收；
- `PIPER_DATA_CONTRACT.md`：UI 事件到后端方法的映射。

运行协议回归测试：

```bash
python -m unittest -v test_piper_data_contract.py
```

`collect_output_arm.py`、GUI、单双臂遥操、NPZ 转换器、LeRobot writer 和
Dashboard 均使用同一份 `EpisodeContract` 元数据。输出臂反馈采集与遥操采集
的 action source 不同，不能在转换时静默互换。

- 轨迹保存 / 回放：`trajectory.py`
- 导出可直接用于 pi0.5 / openpi 训练的 LeRobot 风格数据集：`pi0_dataset.py`
- openpi / pi0.5 实机推理桥接：Dashboard 的 `server_4090/openpi_single_arm.py` 与 `rtc_client.py`；`robot_observation_bridge.py` 是兼容别名，`serve_piper.py`、`run.py` 为 legacy 入口

---

## 1. 采集数据格式

### 1.0 当前硬件的推荐采集方式

单臂输出反馈采集：

```bash
python collect_output_arm.py \
  --arm-mode single \
  --arm-side right \
  --schema joint \
  --can can0 \
  --cam-high-device auto \
  --cam-wrist-device auto \
  --fps 20 --camera-fps 30 \
  --task-name pick_cube \
  --instruction "pick up the cube"
```

双臂输出反馈采集：

```bash
python collect_output_arm.py \
  --arm-mode bimanual \
  --schema joint \
  --left-can can1 \
  --right-can can3 \
  --cam-high-device auto \
  --cam-left-wrist-device auto \
  --cam-right-wrist-device auto \
  --fps 20 --camera-fps 30 \
  --task-name handover \
  --instruction "handover the object"
```

该脚本只读取执行输出臂，因此 joint action 是下一帧实测 qpos，不是 master
命令。若采集遥操训练数据，优先使用 `teleop_single.py` / `teleop.py`，它们会
直接保存同一步 master joint command。按键：`SPACE` 结束 episode，`S` 保存
成功，`F` 保存失败，`D` 丢弃，`Q` 退出。

### 1.1 遥操 joint 原始 episode

默认目录：

```bash
episodes/
episodes_single/
```

每条 episode 为一个 `.npz`，包含：

- `qpos`
- `actions`
- `timestamps`
- `images_cam_high`
- `images_cam_left_wrist` / `images_cam_right_wrist`
- `instruction`
- `task_name`
- `success`

### 1.2 遥操 joint pi0.5 / openpi 数据集

双臂默认输出：

```bash
pi0_dataset_bimanual/
```

单臂默认输出：

```bash
pi0_dataset_single/
```

目录结构示例：

```bash
pi0_dataset_bimanual/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.cam_high/episode_000000.mp4
  videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
  videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4
  raw/episode_000000.npz
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/openpi_norm_stats.json
```

其中已经补齐训练常用字段：

- `observation.state`
- `action`
- `instruction`
- `task`
- `success`
- 多路视频
- 归一化统计：`meta/openpi_norm_stats.json`

---

## 2. 双臂采集：`teleop.py`

### 2.1 硬件映射

默认 4 臂 / 4 CAN：

```text
can0 -> left master
can1 -> left slave
can2 -> right master
can3 -> right slave
```

默认 3 路相机：

```text
cam_high        -> 0
cam_left_wrist  -> 2
cam_right_wrist -> 4
```

### 2.2 第一次采集：先记录起始位

```bash
python teleop.py \
  --record \
  --capture-start \
  --task-name lift_pot \
  --instruction "lift the pot with both arms"
```

### 2.3 后续正式采集

```bash
python teleop.py \
  --record \
  --task-name lift_pot \
  --instruction "lift the pot with both arms"
```

### 2.4 常用参数

```bash
--left-master can0
--left-slave can1
--right-master can2
--right-slave can3
--start-pose start_pose.npy
--out-dir episodes
--dataset-root pi0_dataset_bimanual
--robot-type piper_bimanual
--cam-high-id 0
--cam-left-wrist-id 2
--cam-right-wrist-id 4
--no-pi0-export
```

### 2.5 录制按键

- `SPACE`：结束当前 episode
- `S`：保存为成功轨迹
- `F`：保存为失败轨迹
- `D`：丢弃当前轨迹
- `E`：急停
- `Q`：退出

---

## 3. 单臂采集：`teleop_single.py`

单臂脚本适用于 1 主臂 + 1 从臂。

### 3.1 右臂示例

```bash
python teleop_single.py \
  --record \
  --arm-side right \
  --capture-start \
  --task-name pick_cube \
  --instruction "pick up the cube"
```

后续继续采：

```bash
python teleop_single.py \
  --record \
  --arm-side right \
  --task-name pick_cube \
  --instruction "pick up the cube"
```

### 3.2 左臂示例

```bash
python teleop_single.py \
  --record \
  --arm-side left \
  --task-name open_drawer \
  --instruction "open the drawer"
```

### 3.3 常用参数

```bash
--master can0
--slave can1
--arm-side left|right
--start-pose start_pose_single.npy
--out-dir episodes_single
--dataset-root pi0_dataset_single
--cam-high-id 0
--cam-wrist-id 2
--no-pi0-export
```

单臂默认导出两路相机：

- `cam_high`
- `cam_left_wrist` 或 `cam_right_wrist`（由 `--arm-side` 决定）

---

## 4. 数据维度约定

### 双臂

状态 / 动作均为 14 维：

```text
[left_j1..j6, left_gripper, right_j1..j6, right_gripper]
```

### 单臂

状态 / 动作均为 7 维：

```text
[arm_j1..j6, arm_gripper]
```

单位：

- 关节：`rad`
- 夹爪：`m`

---

## 5. 训练建议

为了让 pi0.5 / openpi 微调更稳定，建议：

- 同一任务采集足够多的成功轨迹
- 保留少量失败轨迹并显式标注 `success=false`
- 让物体初始位置、朝向存在一定随机性
- 相机固定安装，避免频繁移动
- 每条轨迹尽量完整覆盖接近、抓取、搬运、放置全过程

---

## 6. 测试脚本

### 6.1 机械臂 / 相机 smoke test

最安全的只读测试：

```bash
python robot_smoke_test.py
```

如果还想验证控制链路可用，但尽量不让机械臂产生明显运动，可以重发当前位姿：

```bash
python robot_smoke_test.py --send-current
```

可选参数：

```bash
--left-can can0
--right-can can1
--cam-high-id 0
--cam-left-wrist-id 2
--cam-right-wrist-id 4
--skip-cameras
--skip-arms
```

### 6.2 与远程 policy server 联合推理 smoke test

只做相机 + server 推理，不发动作到机械臂：

```bash
python policy_server_smoke_test.py \
  --server 192.168.101.9 \
  --port 8000 \
  --shadow \
  --steps 10
```

shadow 模式下也读取真实机械臂状态：

```bash
python policy_server_smoke_test.py \
  --server 192.168.101.9 \
  --port 8000 \
  --shadow \
  --read-arms-in-shadow \
  --steps 10
```

确认 shadow 没问题后，才建议做真实动作联调：

```bash
python policy_server_smoke_test.py \
  --server 192.168.101.9 \
  --port 8000 \
  --steps 5
```

该脚本会检查：

- 相机可读
- policy server 连通性
- 原始 action chunk 形状是否正常
- broker 循环推理是否正常
- 非 shadow 模式下是否能通过 SafetyChecker 再下发动作

## 7. 下载 π0.5 基座权重

Piper 单臂 LoRA 使用 `pi05_base`。脚本支持 HuggingFace 镜像优先、GCS 回退、多文件并行、单文件 Range 分片和断点续传：

```bash
/home/sunny/miniconda3/envs/openpi/bin/python download_openpi_checkpoint.py \
  --checkpoint gs://openpi-assets/checkpoints/pi05_base \
  --source auto \
  --workers 16 \
  --chunks-per-file 16
```

默认保存到 `~/.cache/openpi/openpi-assets/checkpoints/pi05_base`。重复执行会校验并续传未完成分块。

## 8. 4×4090 数据上传、训练与 Policy 管理网页

部署并启动：

```bash
bash deploy_4090_server.sh
```

上传器支持两种目录，自动识别输入格式：

1. GUI 直接采集的原始目录（包含 `ep_*.npz`），上传前自动校验并导出 LeRobot；
2. 已经导出的 LeRobot v2.1 目录（包含 `meta/info.json`），直接打包上传。

直接上传 GUI 采集批次并追加到服务器同名数据集：

```bash
python upload_dataset_4090.py episodes_batches/20260803_pick_cube_01 \
  --name piper_v1 \
  --dataset-origin real \
  --server http://192.168.101.9:8090 \
  --token "$BIMANUAL_VLA_SERVER_TOKEN" \
  --workers 4 \
  --fps 20 \
  --merge
```

上传已导出的 LeRobot 数据集：

```bash
python upload_dataset_4090.py piper/piper_v1_increment \
  --name piper_v1 \
  --dataset-origin real \
  --server http://192.168.101.9:8090 \
  --token "$BIMANUAL_VLA_SERVER_TOKEN" \
  --workers 4 \
  --merge
```

真机数据使用 `--dataset-origin real`，仿真数据使用 `--dataset-origin simulation`。服务端上传暂存目录按来源隔离，最终数据通过来源 marker 在 Dashboard 中分组，同时保持统一 LeRobot 根目录以兼容训练加载。

原始 NPZ 的自动导出结果保存在 `~/.cache/bimanual-vla/uploads/exports`，源目录未变化时会复用；`--rebuild` 可强制重新导出并重建 tar。若一个原始目录混合了不同 arm mode/schema，导出会拒绝。`--merge` 与 `--overwrite` 互斥；目标不存在时 `--merge` 按首次安装处理。服务端会检查两个数据集的版本、robot type、FPS、chunk size、features、action semantics 和 action offset，随后重新编号新增 episode/global/task index，并在临时目录完成结构与 loader 校验后原子替换。

只转换、不上传时可使用：

```bash
python upload_dataset_4090.py episodes_piper_v21 \
  --name piper_v1 \
  --fps 20 \
  --prepare-only
```

网页地址：`http://192.168.101.9:8090`。页面作为管理面，可执行：

1. 查看数据集、GPU、训练任务和 Policy 进程。
2. 计算 norm stats，提交白名单参数的 π0.5 LoRA 微调。
3. 按数据集筛选完整 checkpoint。
4. 新建 Policy 进程，显示 PID、WebSocket `/healthz`、GPU、端口、schema、checkpoint 和最近 telemetry。
5. 正常停止或强制结束指定 Policy。
6. 选择运行中的 Policy，以新 checkpoint 执行“停止旧进程 → 启动替代进程”的模型切换。
7. 按 Policy schema 只读显示单臂 7D/10D 或双臂 14D/20D state、单臂两路或双臂三路图像、prompt 和预测 action。
8. 显示并管理短时服务端 EXECUTE 授权，同时显示机械臂客户端本地 `--allow-execution`、双重门结果和实际执行/阻断原因。
9. 在网页执行 episode 级数据集编辑：修改 instruction、task name、success 和附加 metadata，批量删除 episode，或把另一个兼容数据集的全部 episodes 增量合并进目标数据集。不会修改 episode 内的 state/action/图像帧。

真实推理数据面不经过 Dashboard。机械臂控制电脑使用官方 `openpi_client.WebsocketClientPolicy` 直接连接 Policy 端口：

这里的 RTC 是 **Real-Time Chunking**：服务端在 flow-matching denoising 内使用上一
action chunk 尚未执行的 normalized prefix 做 guidance，补偿相机、传输和推理延迟；
不是只在客户端对 old/new action 做插值。Dashboard 创建 Policy 时默认启用
`--rtc-enabled`，Policy metadata 会声明 `rtc_algorithm=real_time_chunking_prefix_guidance`
及 `rtc_backend=jax|pytorch`。客户端只发送 session、generation、offset 和 latency
估计，默认 `--rtc-client-blend-steps 0`，避免重复引入轨迹切换延迟。

```bash
python rtc_client.py \
  --host 192.168.101.9 \
  --port 8000 \
  --arm-mode single \
  --arm-side right \
  --can can0 \
  --cam-high-device auto \
  --cam-wrist-device auto \
  --camera-preview \
  --output-mode auto \
  --instruction "pick up the cube"
```

双臂客户端示例：

```bash
python rtc_client.py \
  --host 192.168.101.9 \
  --port 8000 \
  --arm-mode bimanual \
  --arm-side both \
  --left-can can1 \
  --right-can can3 \
  --cam-high-device auto \
  --cam-left-wrist-device auto \
  --cam-right-wrist-device auto \
  --instruction "handover the object"
```

RTC 客户端会同时读取每条机械臂的 10D EEF delivery state 与 7D 实测 joint qpos，
根据服务端 metadata 自动选择单臂 7D/10D 或双臂 14D/20D state、相机 wire key
和 7D/14D action 执行方式，并严格校验 `arm_mode`、维度、左右顺序和相机集合。
也可以用 `--output-mode joint` 或 `--output-mode delivery` 显式锁定输出合同；
显式模式与服务端 metadata 不一致时会在握手阶段 fail-closed，不会把 EEF 输出
误解释为关节角（`--policy-schema` 为同义别名）。
追加 `--camera-preview`（或 `--show-cameras`）会在推理期间打开一个放大的实时窗口，
并排显示保持原始宽高比的相机画面；默认每路约 600 像素宽、8 FPS 刷新，
可用 `--camera-preview-fps` 调整。按 `q` 或 `Esc` 只关闭预览，`Ctrl-C` 才停止 bridge。
默认只打印预测 action；显式追加 `--allow-execution` 后仍需网页对同一 Policy
短时授权 EXECUTE，并通过 schema 对应的新鲜度、动作幅度/关节限位和 Piper
状态检查才会下发。joint 模式默认用 `--joint-limit-tolerance-rad 0.05` 吸收
模型在 Piper 硬限位附近的数值 overshoot，并夹回真实限位；超过该容差仍会阻断。
模型切换导致连接断开后客户端会自动重连、重新协商
schema，并回到 SHADOW。

每次启动 `rtc_client.py` 默认都会在 `deployment_runs/` 下创建一个独立
运行目录，异步保存：

- `trajectory.npz`：每个控制 tick 的实际 Piper 反馈、delivery state、是否真正下发
  command、原始 command row、解码后的绝对目标、chunk generation 和队列索引；
- `trajectory.jsonl`：同一轨迹的可流式读取版本，包含 Unix/monotonic 时间戳和阻断原因；
- `model_commands/command_*.npz` + `model_commands.jsonl`：每次收到的完整模型 action
  chunk，包括被拒绝/过期的 chunk、推理锚点、到达时间和执行授权信息；
- `videos/<camera>.mp4` + `videos/timestamps.jsonl`：与模型观测帧对应的顶部/腕部视频。
  MP4 使用 nominal FPS 播放，精确同步应使用旁边的 timestamp index；若本机 MP4 编码器
  不可用，则自动保留 `video_frames/<camera>/*.jpg`。视频由独立相机采集线程连续记录，
  模型请求的观测时间戳可直接在 `model_commands.jsonl` 中与视频时间索引对齐。

如果只想临时关闭写盘，可追加 `--no-recording`；也可以通过
`--record-root /path/to/deployment_runs` 修改保存位置。录制视频的 nominal FPS 默认等于
`--camera-fps`，可用 `--record-video-fps` 覆盖。

Dashboard Token 只保护管理 API，机械臂客户端不需要 Dashboard URL 或 Token。首次启动自动生成，保存在 4×4090 的 `~/.config/bimanual-vla/server.env`；可用 `ssh 4x4090-wg 'source ~/.config/bimanual-vla/server.env && printf "%s\n" "$BIMANUAL_VLA_SERVER_TOKEN"'` 读取。上传采用并行分块、SHA256 和断点续传。任何合并、删除或 episode 参数修改都会保留隐藏备份、重新校验并让旧 norm stats 失效；下一次训练会自动重新执行 norm→train。详细架构和 Policy 生命周期说明见 `server_4090/README.md`。

## 9. 相关脚本

- `serve_piper.py`：旧版独立 Policy server；不包含模型侧 RTC。需要实机 RTC 时使用 Dashboard 启动的 `server_4090/openpi_single_arm.py serve --rtc-enabled`
- `rtc_client.py`：正式实机客户端，记录机械臂轨迹、模型 action chunk 和同步视频
- `robot_observation_bridge.py`：兼容旧启动命令的别名，直接转发到 `rtc_client.py`，不维护第二套控制实现
- `run.py`：旧版简化实机循环，读取观测并请求 policy 动作
- `robot_smoke_test.py`：机械臂 / 相机 smoke 测试
- `policy_server_smoke_test.py`：与远程 policy server 联合推理 smoke 测试
- `inference_smoke_test.py`：本地推理冒烟测试
- `dl_pi05_base.py` / `dl_hf_chunks.py` / `dl_chunks.py`：模型下载辅助脚本

---

## 10. 注意事项

- 实机采集前先确认 CAN 口和相机编号
- 起始位错误时请重新执行 `--capture-start`
- 默认会同步导出训练集；如果只想录原始 `.npz`，加 `--no-pi0-export`
- 数据集导出是追加式的，同一 `--dataset-root` 下会持续新增 episode
