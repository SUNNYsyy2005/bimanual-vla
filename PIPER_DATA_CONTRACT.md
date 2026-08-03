# Piper 采集端与原始数据合同

本文件定义采集端、原始 NPZ、数据转换和 π0.5 训练之间的边界。
`schema` 名继续使用 `joint` / `delivery` 以兼容现有系统；其中
`delivery` 在新合同中明确表示 **10D absolute EEF**。

## 1. 合同版本

### v3（所有新采集）

| arm mode | schema | measured state | raw action | model action |
|---|---|---:|---:|---:|
| single | `joint` | 7D | 7D | 7D |
| bimanual | `joint` | 14D | 14D | 14D |
| single | `delivery` | 10D | 10D | 7D |
| bimanual | `delivery` | 20D | 20D | 14D |

双臂向量固定为 `left + right`。

v3 默认：

```text
fps=20
action_horizon=50
contract_version=3
gripper_semantics=absolute_opening_fraction_0_closed_1_open
coordinate_frame=slave_base
```

CLI 显式覆盖 `fps` / `action_horizon` 时，写入元数据必须使用实际值。

### legacy v2（只读兼容，不改写语义）

`8_3_64eps` 等旧 delivery 数据明确识别为：

```text
state:   10D absolute EEF, gripper_closed_fraction (0=open, 1=closed)
actions: 7D one-step EEF delta, gripper=next frame closed_fraction
rotation delta: R[t+1] @ R[t].T
contract_version=2 / legacy_delivery_v2=true
```

旧 joint v2 的第 7 维为 `gripper_opening_m`，而不是 opening fraction。
其语义通过版本、字段名和 `gripper_semantics=absolute_opening_metres`
显式保留。任何 legacy 数据都不能静默解释成 v3。

## 2. v3 Joint Schema

单臂 measured state：

```text
[
  joint_1_rad, ..., joint_6_rad,
  gripper_opening_fraction
]
```

其中夹爪统一为：

```text
0 = 完全闭合
1 = 完全张开
```

推荐使用 Piper 最大开口 `0.07 m` 归一化。前六维均为 rad。

采集标签：

```text
state  = slave measured joint state
action = same-step actual slave command，或经过明确映射的 master absolute target
action_alignment = same_step_command
action_offset = 0
```

输出臂-only 采集无法读取命令时允许：

```text
action_source = next_measured_joint_fallback
action_alignment = next_observation
action_offset = 1
```

该 fallback 是观测近似，不能标记成操作者命令。

## 3. v3 Delivery Schema

单臂 state 和 raw action 都使用 10D 绝对 EEF：

```text
[
  eef_x_base_m, eef_y_base_m, eef_z_base_m,
  rotation6d_col0_x, rotation6d_col0_y, rotation6d_col0_z,
  rotation6d_col1_x, rotation6d_col1_y, rotation6d_col1_z,
  gripper_opening_fraction
]
```

`state` 是 slave measured EEF；`actions` 是 slave base 坐标系中的
`absolute_eef_target`。delivery episode 同时保存：

```text
joint_qpos: first 6 joint radians + gripper opening fraction
```

### 3.1 训练边界

raw action 为 10D，但模型 action 为 7D：

```text
[delta_xyz_from_current, left_delta_rotvec_from_current, absolute_gripper_target]
```

使用稳定纯函数：

```python
absolute_eef_targets_to_chunk_origin(state, targets, arm_count=1)
chunk_origin_deltas_to_absolute_eef_targets(state, actions, arm_count=1)
```

一个 action chunk 的所有 target 都相对于同一个当前 `state`；不能对未来
动作逐步累计。旧数据转换继续使用：

```python
step_deltas_to_chunk_origin(actions, arm_count=1)
```

### 3.2 Teleop mixed-source action（无需 EEF 标定）

Teleop Delivery 默认不直接使用主臂 EEF pose，因此不需要假设主从臂坐标系、
零位和工作空间比例一致。每个 10D raw action 使用：

```text
pose = 下一帧从臂实测 absolute EEF（slave_base）
gripper = 当前周期主臂 gripper opening fraction
```

对应元数据：

```text
action_source = next_measured_eef_pose_with_same_step_master_gripper_feedback
action_alignment = next_observation_pose_same_step_gripper
action_offset = 1
pose_action_source = next_measured_eef_fallback
pose_action_alignment = next_observation
gripper_action_source = master_gripper_feedback
gripper_action_alignment = same_step_command
```

不能使用同一帧从臂 pose 作为 pose action，否则模型只会看到接近零的 EEF
位移。只有未来重新启用“主臂 EEF 直接映射为从臂 absolute target”时，才需要
单独提供并验证主从 EEF 标定。

### 3.3 Output-only fallback

如果输出臂采集无法读取完整命令：

```text
pose action = next measured absolute EEF
pose_action_source = next_measured_eef_fallback
pose_action_alignment = next_observation
action_offset = 1
```

若 SDK 能读取实际夹爪命令，则 raw target 的夹爪维度可使用 same-step
command；同时保存 `gripper_command_target/present/timestamp`，并通过
`gripper_action_source` / `gripper_action_alignment` 明确标记。姿态来源仍是
next-measured fallback，不能将整个 action 伪装成 command。

## 4. 时间戳

v3 每帧分别保存：

```text
state_timestamp   float64 (T,)
action_timestamp  float64 (T,)
image_timestamps_<camera_key> float64 (T,)
```

`timestamps` 仅作为兼容 alias，值等于 `state_timestamp`。采集循环应在同一
控制周期读取 state、command 和 images，但不能用一个 host timestamp
冒充所有设备时间。终端 padding 的时间戳继续递增。

## 5. v3 必需元数据

```text
contract_version
schema, arm_mode, arm_side, robot_type
state_dim, raw_action_dim, model_action_dim
state_names, action_names, model_action_names
camera_keys
action_semantics, model_action_semantics
action_source, action_alignment, action_offset
gripper_semantics, rotation_semantics
coordinate_frame, source_frame
fps, action_horizon
terminal_padding
```

相机字段为 RGB HWC `uint8`：

```text
images_cam_high
images_cam_<side>_wrist
images_cam_left_wrist / images_cam_right_wrist
```

legacy single-arm delivery 仍可读取 `image` / `wrist_image`，但 v3 新文件不再
依赖该模糊命名。

## 6. 采集入口

### Joint P0

```bash
python teleop_single.py --record --schema joint --fps 20
python teleop.py --record --schema joint --fps 20
```

state 使用 slave measured，action 使用 same-step master mapped absolute target。

### Delivery P1

```bash
python teleop_single.py --record --schema delivery
python teleop.py --record --schema delivery
```

这两条链路不会读取主臂 EEF pose；从臂下一帧 pose 和主臂当前夹爪来源会分别
写入元数据。旧的 `--eef-calibration` 参数仅为命令行兼容保留，不再参与采集。

### Output-only

```bash
python collect_output_arm.py --schema joint --fps 20 --action-horizon 50
python collect_output_arm.py --schema delivery --fps 20 --action-horizon 50
```

输出臂-only 数据始终以 fallback 来源明确标记。

## 7. 验收

`validate_piper_data.py` 检查：

1. v2/v3 layout 与 metadata 一致；
2. state/raw action/model action 维度；
3. opening/closed/metres 三种夹爪语义不混用；
4. state/action/image 时间戳和采样率；
5. rotation-6D 正交性；
6. v3 fallback absolute target 与下一 measured state 的关系；
7. legacy v2 one-step xyz/rotation/gripper 精确重构；
8. NaN/Inf、黑帧、冻结帧和无动作 episode。

原始 episode 保存后应先通过：

```bash
python validate_piper_data.py --input-dir <episode_dir> --target-fps 20
```
