# Piper 数据采集操作指南

本文用于现场操作人员完成 Piper 执行臂、第三视角相机和腕部相机的数据采集、检查、回放、导出与上传。

当前采集链路只读取执行输出臂反馈，不读取示教臂控制指令。训练数据使用 10D 末端状态和 7D 末端增量动作协议。

## 1. 系统组成

- 机械臂：Piper 执行输出臂，通过 SocketCAN `can0` 通讯，波特率 `1,000,000`。
- 第三视角相机：Intel RealSense D435i 的 RGB 节点。
- 腕部相机：Intel RealSense D405 的 RGB 节点。
- 采集频率：固定使用 `20 Hz`。
- 相机源频率：推荐使用 `30 FPS`。
- 图像尺寸：保存为 `256 × 256` RGB。

`/dev/videoX` 编号可能在重新插拔 USB 后变化。不要只依赖上一次使用的编号，连接设备后应检查 GUI 中两路预览是否正确。

## 2. 采集前检查

进入项目目录并激活环境：

```bash
cd ~/dual_ARM_project/arm_collect/bimanual-vla
conda activate dual_arm
```

### 2.1 检查 CAN

查看 CAN 接口：

```bash
ip -br link show type can
```

正常情况下应看到：

```text
can0  UP
```

如果接口存在但未激活：

```bash
cd ~/dual_ARM_project/piper_sdk/piper_sdk
bash can_activate.sh can0 1000000
```

检查状态：

```bash
ip -s -details link show can0
timeout 2 candump -L can0
```

正常要求：

- 波特率是 `1000000`。
- CAN 状态通常为 `ERROR-ACTIVE`。
- `candump` 能持续看到机械臂反馈帧。

如果系统中没有 `can0`，检查 USB-CAN 是否被 `gs_usb` 驱动识别：

```bash
lsusb -t
```

官方 USB-CAN 应显示 `Driver=gs_usb`。`/dev/ttyUSB0` 或 `Driver=ch341` 是串口设备，不能直接作为本项目的 `can0` 使用。

### 2.2 检查相机

列出设备：

```bash
v4l2-ctl --list-devices
```

检查各节点格式：

```bash
for d in /dev/video*; do
  echo "$d"
  v4l2-ctl -d "$d" --get-fmt-video 2>/dev/null | grep -E "Width/Height|Pixel Format"
done
```

选择能输出 `424 × 240`、`YUYV` 或 RGB 图像的节点。深度 `Z16` 和灰度 `GREY` 节点不能作为本项目的 RGB 输入。

本机曾使用过以下映射，但 USB 重新枚举后可能改变：

```text
Third-Person Camera: /dev/video8 或 /dev/video9
Wrist Camera:        /dev/video16 或 /dev/video15
```

最终以 GUI 中显示的实际画面为准：

- `Third-Person Camera` 应显示外部全局视角。
- `Wrist Camera` 应显示机械臂末端附近视角。

## 3. 启动 GUI

```bash
cd ~/dual_ARM_project/arm_collect/bimanual-vla
bash start_gui.sh
```

GUI 使用英文界面，主要区域如下：

### Devices and Task

| 字段 | 建议值 | 说明 |
|---|---|---|
| `CAN Interface` | `can0` | 执行输出臂 CAN 接口 |
| `Third-Person Camera` | 实际 RGB 节点 | 第三视角相机 |
| `Wrist Camera` | 实际 RGB 节点 | 腕部相机 |
| `Capture Rate (Hz)` | `20` | 数据采样频率 |
| `Camera Rate (Hz)` | `30` | RealSense 原始流频率 |
| `Output Directory` | 每批次独立目录 | 原始 NPZ 保存位置 |
| `Task Name` | 如 `pick_cube` | 内部任务标识 |
| `Instruction` | 如 `pick up the cube` | 模型训练使用的自然语言指令 |

推荐每次采集使用独立批次目录，例如：

```text
episodes_batches/20260801_pick_cube_01
episodes_batches/20260801_pick_cube_02
```

不要长期把所有新旧数据都放在同一个目录后再整体执行 `--merge`，否则容易重复上传旧 episode。

### Home Pose Safety Check

默认参考位姿为：

- J1-J6 全部为 `0°`。
- 夹爪闭合。
- 默认关节误差阈值为 `±5°`。
- 默认夹爪误差阈值为 `±5 mm`。

机械臂不在参考位姿时，界面会显示红色警告，但 `Start Episode` 仍可使用；初始位姿检查只用于提示，不再强制阻塞有效遥操作数据采集。无机械臂反馈或反馈过期时仍会拒绝采集。只有在确认当前 CAN 确实连接到需要控制的机械臂后，才使用 `Reset Home`。

## 4. 采集一个 episode

### 4.1 连接设备

1. 填写 CAN、两路相机、输出目录、任务名和指令。
2. 点击 `Connect`。
3. 检查右侧两路实时画面是否正确。
4. 检查 `Live Robot Pose` 是否持续更新。
5. 查看 `Home pose` 状态；若为红色，确认该偏差是否符合当前任务的真实起始位姿。

如果两路画面颠倒，断开设备后交换两个相机节点再连接。

### 4.2 开始采集

1. 将场景恢复到任务起始状态。
2. 确认机械臂位于当前任务要求的起始位姿；该位姿不必是全零 Home。
3. 确认 `Instruction` 与实际任务一致。
4. 点击 `Start Episode`。
5. 完成一次完整、自然的遥操作任务。
6. 点击 `Stop Episode`。

建议：

- 每个 episode 只包含一次完整任务尝试。
- 开始前保留短暂稳定画面，但不要长时间静止。
- 失败尝试也可以保留，但必须标记为 Failure。
- 不要在 episode 中途修改 instruction。
- 避免相机被手、线缆或其他物体遮挡。

### 4.3 停止后标注

停止后会出现三个选项：

- `Save as Success`：任务成功完成并保存。
- `Save as Failure`：任务失败但数据仍有分析价值。
- `Discard`：误操作、空数据、严重遮挡或设备异常，不保存。

保存前程序会自动进行数据协议校验。校验失败时不会发布正式 `ep_XXXX.npz` 文件。

## 5. 回放和现场质量检查

1. 在 `Saved Episodes` 中选择一个文件。
2. 点击 `Replay Selected`。
3. 检查第三视角、腕部视角和关节数据是否同步变化。

至少抽查以下内容：

- 两路画面对应关系正确。
- 图像没有长时间冻结或明显跳帧。
- 机械臂运动时关节角和 EEF 状态同步变化。
- 夹爪开合状态有记录。
- instruction 与画面中的任务一致。
- 成功/失败标签正确。

## 6. 原始 NPZ 数据格式

每个 episode 保存为一个 `ep_XXXX.npz`：

```text
state                         float32 (T,10)
actions                       float32 (T,7)
timestamps                    float64 (T,)
image                         uint8   (T,256,256,3), RGB HWC
wrist_image                   uint8   (T,256,256,3), RGB HWC
task                          Unicode scalar
instruction                   Unicode scalar
success                       bool scalar
joint_qpos                    float32 (T,7)
image_timestamps_cam_high     float64 (T,)
image_timestamps_cam_wrist    float64 (T,)
```

10D `state`：

```text
EEF xyz in base frame (3)
+ rotation 6D (6)
+ gripper closed fraction (1)
```

7D `actions`：

```text
base-frame delta xyz (3)
+ left-multiplied delta rotvec (3)
+ next gripper target (1)
```

每个 episode 包含 terminal observation，最后一个动作保持夹爪状态且机械臂运动增量为零。

## 7. 批次验证

假设本次采集目录为：

```bash
BATCH=20260801_pick_cube_01
```

执行：

```bash
python validate_piper_data.py \
  --input-dir "episodes_batches/$BATCH" \
  --target-fps 20
```

正常结果应以 `Dataset PASS` 结束。验证会检查：

- shape 和 dtype。
- 实际 FPS。
- 两路图像时间同步。
- rotation 6D 合法性。
- action 是否可由相邻 state 正确重算。
- terminal action。
- 图像冻结和 no-op 比例。
- 夹爪开合覆盖。

高 no-op episode 虽然可能通过格式验证，但不建议直接用于训练，应在 Dashboard 中删除或禁用。

## 8. 导出为 LeRobot v2.1

每个批次使用新的导出目录：

```bash
python export_lerobot.py \
  --input-dir "episodes_batches/$BATCH" \
  --repo-id "piper/$BATCH" \
  --root "lerobot_batches/$BATCH" \
  --fps 20
```

检查导出结果：

```bash
test -f "lerobot_batches/$BATCH/meta/info.json" \
  && echo "LeRobot dataset ready"
```

导出器默认只把标记为 Success 的 episode 写入训练集。Failure episode 保留在原始 NPZ 目录中，但不会进入当前训练导出。

不要把旧批次和新批次一起重新导出后再使用 `--merge`。

## 9. 上传并合并到服务器

安全读取 Dashboard Token：

```bash
read -rsp "Dashboard token: " BIMANUAL_VLA_SERVER_TOKEN
echo
```

上传器支持两种输入方式。

方式一，直接提交 GUI 采集的原始 NPZ 批次；脚本会自动校验并导出为 LeRobot：

```bash
python upload_dataset_4090.py \
  "episodes_batches/$BATCH" \
  --name my_dataset \
  --server http://192.168.101.9:8090 \
  --token "$BIMANUAL_VLA_SERVER_TOKEN" \
  --workers 4 \
  --fps 20 \
  --merge
```

方式二，提交已经手动导出的 LeRobot 目录：

```bash
python upload_dataset_4090.py \
  "lerobot_batches/$BATCH" \
  --name my_dataset \
  --server http://192.168.101.9:8090 \
  --token "$BIMANUAL_VLA_SERVER_TOKEN" \
  --workers 4 \
  --merge
```

直接上传原始 NPZ 时，自动导出的 LeRobot 数据缓存在 `~/.cache/bimanual-vla/uploads/exports`。重复执行且源目录未变化时会复用导出结果和 tar；使用 `--rebuild` 可强制重建。一个原始目录中不要混合不同 arm mode 或 schema。

完成后清除环境变量：

```bash
unset BIMANUAL_VLA_SERVER_TOKEN
```

上传器会自动：

1. 构建未压缩 tar 包。
2. 计算 SHA256。
3. 分块并行上传。
4. 在服务器上进行结构和 LeRobot loader 校验。
5. 对新增 episode、frame、task index 连续重新编号。
6. 在临时目录完成合并后原子替换服务器数据集。

网络中断后可以重新执行同一条上传命令，程序会复用缓存并进行断点续传。

上传成功后打开：

```text
http://192.168.101.9:8090
```

检查：

- `my_dataset` episode 和 frame 数量是否正确增加。
- schema 是否为 `delivery`。
- state/action shape 是否为 `[10]` / `[7]`。
- 相机字段是否为 `image` / `wrist_image`。
- 是否存在 no-op、错误 instruction 或错误标签 episode。

任何合并、删除或 episode 元数据修改都会让旧 norm stats 失效。下一次训练应重新执行 norm；Dashboard 的自动训练流程会自动处理。

## 10. 常见问题

### `No such device: can0`

系统中不存在 SocketCAN 接口。检查：

```bash
ip -br link show type can
lsusb -t
```

重新连接官方 USB-CAN，直到出现 `Driver=gs_usb` 和 `can0`。

### `SEND_MESSAGE_FAILED (100017)` 或 `No buffer space available`

通常表示 CAN 总线上没有机械臂节点进行 ACK。检查机械臂供电、CAN 线缆、转接头和总线连接。硬件恢复后重新激活接口：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
```

随后确认 `candump can0` 能看到反馈。

### `Cannot open camera /dev/videoX`

USB 重新枚举导致节点变化，或节点不是 RGB 流。重新执行：

```bash
v4l2-ctl --list-devices
```

在 GUI 中更新相机路径，并通过实时画面确认。

### 上传时报输入目录既不是 LeRobot 也不是 GUI NPZ

输入目录必须满足其中一种格式：包含 `meta/info.json` 的 LeRobot 数据集，或顶层包含 `ep_*.npz` 的 GUI 采集目录。不要传入这两类目录的上一级目录。

### 上传时报 Token 长度不足

命令中应使用：

```bash
--token "$BIMANUAL_VLA_SERVER_TOKEN"
```

不要把 Token 本身写成 `$具体Token`。不要在聊天、Git 或截图中公开 Token。

### Policy 提示 `schema='joint', expected 'delivery'`

这是推理服务器上运行了旧 joint Policy，与数据采集或上传无关。需要在 Dashboard 中停止旧 Policy，并使用 delivery 数据集和对应 checkpoint 重建。

## 11. 每批次推荐检查清单

采集前：

- [ ] `can0` 已激活且能收到反馈。
- [ ] 两路 RGB 相机画面正确。
- [ ] Capture Rate 为 20 Hz，Camera Rate 为 30 Hz。
- [ ] 使用新的批次输出目录。
- [ ] task 和 instruction 正确。
- [ ] 机械臂位于当前任务要求的起始位姿，并已检查界面上的位姿告警。

采集后：

- [ ] 每个 episode 已正确标记 Success、Failure 或 Discard。
- [ ] 已抽查回放。
- [ ] `validate_piper_data.py` 显示 `Dataset PASS`。
- [ ] 只导出本批次的新 episode。
- [ ] 使用 `--merge` 上传到 `my_dataset`。
- [ ] Dashboard 中 episode 数量正确增加。
- [ ] 已删除或禁用 no-op 和错误 episode。

## 12. 数据目录管理

推荐目录结构：

```text
episodes_batches/
  20260801_pick_cube_01/
  20260801_pick_cube_02/

lerobot_batches/
  20260801_pick_cube_01/
  20260801_pick_cube_02/
```

这些目录可能非常大，不要执行：

```bash
git add episodes_batches lerobot_batches lerobot_datasets
```

确认服务器安装成功后，可以将原始批次复制到独立存储进行归档。删除本地数据前，应确认服务器数据集和备份均可正常读取。
