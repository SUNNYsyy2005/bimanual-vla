# bimanual-vla

双臂 / 单臂 Piper 遥操作采集、pi0.5/openpi 推理接入与训练数据导出工具集。

## 当前可用能力

- 双臂主从遥操作采集：`teleop.py`
- 单臂主从遥操作采集：`teleop_single.py`
- 3 路相机采集：`camera.py`
- 单 CAN 执行输出臂 + 双 RGB 相机反馈采集：`collect_output_arm.py`
- episode 双视角 + 关节角回放：`view_episode.py`
- 图形化采集界面：`collect_gui.py` / `start_gui.sh`
- LeRobot v2.1 导出：`export_lerobot.py`

### 图形化采集界面

启动：

```bash
cd /home/user/dual_ARM_project/bimanual-vla
bash start_gui.sh
```

界面支持：

- 配置 `can0`、`/dev/video8`、`/dev/video16` 和任务信息
- 放大字体、窗口和按钮，方便现场操作
- 连接/断开机械臂与相机
- 连接成功后弹出两个实时预览窗口：第三视角和腕部第一视角
- 开始/停止 episode
- 停止后标记成功、失败或丢弃
- 自动从已有编号继续保存，避免覆盖旧数据
- 选择已保存的 episode 并启动双视角回放

采集频率默认是交付规范要求的 `20 Hz`，可在 GUI 的 `Capture FPS` 中切换。最终导出到 Piper 训练集时必须使用 `20 Hz`。

### Piper delivery schema

新采集的 NPZ episode 包含：

```text
image:       (T, 256, 256, 3) uint8 RGB HWC
wrist_image: (T, 256, 256, 3) uint8 RGB HWC
state:       (T, 10) float32
actions:     (T, 7) float32
task:        UTF-8 string
```

`state` 是 EEF base-frame position、rotation 6D 和 gripper closed fraction；`actions[t]` 是从当前帧到下一帧的 base-frame position/rotation delta 和下一帧夹爪目标。每个 episode 最后自动增加 terminal no-op frame。

导出 LeRobot v2.1：

```bash
python export_lerobot.py \
  --input-dir episodes_output_arm \
  --repo-id piper/piper_v1 \
  --root piper/piper_v1 \
  --fps 20
```
- 轨迹保存 / 回放：`trajectory.py`
- 导出可直接用于 pi0.5 / openpi 训练的 LeRobot 风格数据集：`pi0_dataset.py`
- openpi / pi0.5 实机推理桥接：`serve_piper.py`、`run.py`

---

## 1. 采集数据格式

### 1.0 当前硬件的推荐采集方式

如果电脑只通过 `can0` 连接执行输出臂，不读取示教臂和控制指令，使用：

```bash
python collect_output_arm.py \
  --can can0 \
  --cam-high-device /dev/video12 \
  --cam-wrist-device /dev/video4 \
  --task-name pick_cube \
  --instruction "pick up the cube"
```

该脚本只保存执行输出臂的 7 维反馈：

```text
[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
```

以及 `cam_high`（第三视角）和 `cam_wrist`（腕部第一视角）两路 RGB 图像。它不读取示教臂、不读取关节控制指令，也不执行自动复位。

按键：`SPACE` 结束 episode，`S` 保存成功，`F` 保存失败，`D` 丢弃，`Q` 退出。

如果 `/dev/video12` 或 `/dev/video4` 不是 RGB 节点，先用 `v4l2-ctl --list-formats-ext -d /dev/videoN` 确认后替换参数。

当前采集脚本在录制时会同时产出两类数据：

### 1.1 原始 episode

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

### 1.2 可直接训练的 pi0.5 / openpi 数据集

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

## 8. 相关脚本

- `serve_piper.py`：加载 pi0.5 / openpi policy 并启动服务
- `run.py`：实机循环，读取观测并请求 policy 动作
- `robot_smoke_test.py`：机械臂 / 相机 smoke 测试
- `policy_server_smoke_test.py`：与远程 policy server 联合推理 smoke 测试
- `inference_smoke_test.py`：本地推理冒烟测试
- `dl_pi05_base.py` / `dl_hf_chunks.py` / `dl_chunks.py`：模型下载辅助脚本

---

## 9. 注意事项

- 实机采集前先确认 CAN 口和相机编号
- 起始位错误时请重新执行 `--capture-start`
- 默认会同步导出训练集；如果只想录原始 `.npz`，加 `--no-pi0-export`
- 数据集导出是追加式的，同一 `--dataset-root` 下会持续新增 episode
