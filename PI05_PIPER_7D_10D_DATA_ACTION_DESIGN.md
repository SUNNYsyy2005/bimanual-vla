# π0.5 Piper 单臂 7D/10D 数据采集与动作输出方案

## 1. 目标与适用范围

本文档定义 Piper 单臂在 π0.5/OpenPI 微调与实机部署中的两套独立数据合同：

1. **7D Joint Schema**：从臂关节状态输入，关节目标输出；
2. **10D EEF Schema**：从臂末端状态输入，末端位姿增量输出。

两套数据合同不得混合使用，应分别保存数据集、计算归一化统计并训练 checkpoint。

本文档采用 OpenPI 的 current-anchored action convention：一个 action chunk 中的所有未来动作都以同一次推理的当前状态为锚点，而不是定义成相邻未来状态之间的增量。

---

## 2. 总体原则

### 2.1 State 与 Action 的来源

统一采用：

```text
state  = 从臂/输出臂当前实测状态
action = 操作者希望发送给从臂的实际控制目标
```

动作标签的推荐来源优先级：

1. 主从控制系统实际发送给从臂的控制命令；
2. Teleop 默认使用下一帧从臂实测 EEF pose，并使用同周期主臂夹爪 opening fraction；
3. 下一时刻从臂实测状态推导出的动作，仅作为无法读取命令时的备选。

不应把“从臂最终运动了多少”静默标记为“操作者发出了什么命令”。

### 2.2 时间对齐

主从遥操作数据推荐：

```text
action_source    = same_step_slave_command 或 master_command_after_mapping
action_alignment = same_step_command
action_offset    = 0
```

每条样本应分别保存：

```text
state_timestamp
action_timestamp
image_timestamps
```

采集循环应在同一控制周期读取从臂状态、主臂命令和相机图像。若设备时间戳不同，应按时间戳对齐，而不是只依赖数组下标。

### 2.3 夹爪表示

模型侧推荐使用连续的绝对开口目标：

$$
g = \frac{w}{w_{max}} \in [0,1]
$$

其中：

```text
0 = 完全闭合
1 = 完全张开
```

Piper 可取 $w_{max}=0.07\,\text{m}$。原始开口米数也应额外保存用于验收和回放。

夹爪 action 使用绝对目标，不使用增量：

```text
gripper_action = future_gripper_target
```

---

## 3. 方案 A：7D Joint Schema

### 3.1 模型输入

从臂实测状态：

$$
s_t=[q^s_{1,t},\ldots,q^s_{6,t},g^s_t]\in\mathbb{R}^7
$$

字段顺序：

```text
[
  slave_joint_1_rad,
  slave_joint_2_rad,
  slave_joint_3_rad,
  slave_joint_4_rad,
  slave_joint_5_rad,
  slave_joint_6_rad,
  slave_gripper_opening_norm
]
```

视觉和语言输入同时包括：

```text
observation.images.cam_high
observation.images.cam_<side>_wrist
prompt / task instruction
```

### 3.2 原始 Action 采集

保存实际发送给从臂的绝对关节目标：

$$
u_t=[q^{cmd}_{1,t},\ldots,q^{cmd}_{6,t},g^{cmd}_t]
$$

当前主从臂结构下，可以使用：

```python
state_t = read_7d(slave)
command_t = read_7d(master)
```

但如果主从 SDK 内部还包含比例、零位补偿或限幅，应优先记录处理后的最终从臂命令，而不是未经映射的主臂反馈。

### 3.3 数据集存储格式

```text
observation.state: float32 [7]
action:            float32 [7]   # 绝对关节/夹爪命令
```

建议字段名：

```text
observation.state = slave measured qpos
action            = absolute slave joint command
```

### 3.4 OpenPI 训练变换

形成未来 action chunk 后，对前六个关节维度执行 `DeltaActions`：

$$
a_k=
\left[
q^{cmd}_{t+k}-q^s_t,\;
g^{cmd}_{t+k}
\right]
$$

代码语义：

```python
mask = transforms.make_bool_mask(6, -1)

inputs  = [transforms.DeltaActions(mask)]
outputs = [transforms.AbsoluteActions(mask)]
```

注意：整个未来 action chunk 都减去同一个当前从臂状态 $q^s_t$，不是计算相邻未来关节目标之间的差值。

### 3.5 模型输出

π0.5 内部动作输出：

```text
shape = (50, 7)
```

模型空间中每一行为：

```text
[
  delta_joint_1_from_current,
  ...,
  delta_joint_6_from_current,
  absolute_gripper_target
]
```

经过 `AbsoluteActions` 后，Policy 客户端接收：

```text
[
  joint_1_target_rad,
  ...,
  joint_6_target_rad,
  gripper_target_norm
]
```

### 3.6 实机执行

```python
piper.JointCtrl(*joint_targets)
piper.GripperCtrl(gripper_target)
```

执行前必须检查：

- 关节位置限位；
- 单周期最大关节变化；
- 夹爪范围和最大变化；
- Piper 状态码和控制模式；
- 命令和观测新鲜度。

---

## 4. 方案 B：10D EEF Schema

### 4.1 模型输入

从臂实测末端状态：

$$
s_t=[p^s_t,R^s_{6D,t},g^s_t]\in\mathbb{R}^{10}
$$

字段顺序：

```text
[
  slave_eef_x_base_m,
  slave_eef_y_base_m,
  slave_eef_z_base_m,
  rotation6d_col0_x,
  rotation6d_col0_y,
  rotation6d_col0_z,
  rotation6d_col1_x,
  rotation6d_col1_y,
  rotation6d_col1_z,
  slave_gripper_opening_norm
]
```

其中 rotation-6D 为末端旋转矩阵的前两列。

即使关节角不作为模型输入，也建议同时保存：

```text
joint_qpos: float32 [7]
```

这可用于关节限位、奇异位形、IK 分支和回放检查。后续也可升级为“关节状态 + EEF 状态”的 16D observation。

### 4.2 原始 Action 采集

最规范的 action 是主臂映射到从臂坐标系后的绝对 EEF 控制目标：

```text
absolute_eef_target = [target_xyz, target_rotation6d, target_gripper]
```

推荐生成流程：

```text
主臂关节反馈/控制命令
        ↓ 主臂正运动学 FK
主臂末端目标
        ↓ 主从零位、坐标系和比例映射
从臂基坐标系下的绝对 EEF 控制目标
```

必须验证：

- 主从臂零位是否一致；
- 两个基坐标系轴方向是否一致；
- 主从运动比例是否为 1:1；
- 夹爪开口范围是否一致；
- 主臂命令与从臂状态是否时间同步。

如果能截取主从系统最终发送给从臂的笛卡尔目标，应直接使用该目标，避免重复实现映射。

### 4.3 数据集存储格式

推荐保存绝对 EEF 目标，而不是提前保存相邻帧 delta：

```text
observation.state:         float32 [10]  # 从臂实测 EEF 状态
action.absolute_eef_target float32 [10]  # 映射后的绝对 EEF 命令
joint_qpos:                float32 [7]   # 诊断字段
```

在进入模型前，再通过自定义 transform 将 10D 绝对目标转换为 7D current-anchored delta。

### 4.4 OpenPI 训练目标

对于同一次推理时刻 $t$，未来第 $k$ 个动作定义为：

$$
a_k=
\left[
p^{cmd}_{t+k}-p^s_t,\;
\log\left(R^{cmd}_{t+k}R^{s\top}_t\right),\;
g^{cmd}_{t+k}
\right]
$$

因此所有未来动作都相对于同一个当前从臂状态：

```text
a0 = target(t)   relative to state(t)
a1 = target(t+1) relative to state(t)
a2 = target(t+2) relative to state(t)
...
```

不能构造成：

```text
target(t+2) - target(t+1)
target(t+3) - target(t+2)
```

10D state 使用 rotation-6D，而 7D action 使用 rotvec，因此不能直接套用 OpenPI 的普通向量减法，必须实现 EEF 专用 transform。

训练变换伪代码：

```python
def absolute_eef_targets_to_delta(state_10d, targets_10d):
    current_xyz = state_10d[:3]
    current_R = rotation6d_to_matrix(state_10d[3:9])

    result = []
    for target in targets_10d:
        target_xyz = target[:3]
        target_R = rotation6d_to_matrix(target[3:9])

        delta_xyz = target_xyz - current_xyz
        delta_rotvec = log_so3(target_R @ current_R.T)
        gripper_target = target[9]

        result.append(concat(delta_xyz, delta_rotvec, gripper_target))

    return stack(result)  # (H, 7)
```

### 4.5 模型输出

π0.5 模型空间输出：

```text
shape = (50, 7)
```

每一行为：

```text
[
  delta_x_from_current_m,
  delta_y_from_current_m,
  delta_z_from_current_m,
  delta_rx_from_current_rad,
  delta_ry_from_current_rad,
  delta_rz_from_current_rad,
  absolute_gripper_target
]
```

### 4.6 执行前解码

对同一个推理观测 $s_t$，将每个未来动作独立解码为绝对目标：

$$
p^{target}_k=p^s_t+\Delta p_k
$$

$$
R^{target}_k=\exp(\Delta r_k)R^s_t
$$

```python
def decode_eef_action(current_state, action):
    current_xyz = current_state[:3]
    current_R = rotation6d_to_matrix(current_state[3:9])

    target_xyz = current_xyz + action[:3]
    target_R = exp_so3(action[3:6]) @ current_R
    target_gripper = action[6]

    return target_xyz, target_R, target_gripper
```

所有 action 必须基于同一个推理状态解码。不能执行完 `actions[0]` 后，再把 `actions[1]` 加到新的机械臂状态上，否则会重复累计。

### 4.7 实机执行与 IK

将解码后的绝对 EEF 目标转换为 Piper 所需的 xyz + RPY：

```python
piper.EndPoseCtrl(x, y, z, rx, ry, rz)
piper.GripperCtrl(gripper_target)
```

π0.5 不负责计算 IK。逆运动学、轨迹插值和关节控制应由经过验证的底层控制器完成。

执行端必须额外检查：

- EEF 工作空间；
- 单周期最大平移和旋转；
- 当前关节限位；
- IK 解连续性；
- 奇异位形；
- 自碰撞和环境碰撞；
- Piper 状态和命令新鲜度。

IK 应优先以当前从臂关节角作为初值，避免切换到另一个关节解分支。

---

## 5. 20 Hz 数据、4 Hz 异步推理与连续动作执行

推荐将模型推理频率与机械臂控制频率解耦：

```text
数据采集频率：20 Hz
模型动作频率：20 Hz
模型推理启动频率：约 4 Hz
推理 + 传输延迟：约 200 ms（约 4 个动作周期）
机器人命令频率：20 Hz
```

每次推理返回：

```python
actions.shape == (50, 7)
```

每个预测必须带明确时间语义。若观测单调时间戳为 `t_obs`、动作频率为
20 Hz，则第 `i` 行（从 0 开始）对应：

```python
target_time = t_obs + (i + 1) * 0.05
```

即 50 行覆盖 `t_obs + 0.05 s` 到 `t_obs + 2.50 s`。训练 loader 也必须按
raw `action_offset` 对齐，使 model/wire chunk 的第一行始终代表观测后的下一个
20 Hz 控制周期。

不能采用“发起推理后停下控制等待结果”的同步循环。执行端维护长动作队列，
推理线程与 20 Hz 控制线程并行：

```text
持续执行旧 chunk
      ↓
约每 250 ms 异步启动一次新推理
      ↓（推理期间旧 chunk 继续以 20 Hz 消费）
新 chunk 返回
      ↓
按 target timestamp 和 execution_time 动态丢弃过时前缀（约 4 步）
      ↓
旧轨迹与仍有效的新轨迹用默认 3 步平滑融合
      ↓
原子切换到新轨迹并继续 20 Hz 执行
```

OpenPI 默认 50 步 horizon；执行客户端要求 chunk 至少 16 步。新结果过短、
过时或合同不匹配时应拒绝它并继续旧 chunk。只有旧 chunk 也耗尽时才 fail closed。

伪代码：

```python
while control_loop_20hz:
    execute_next_from_active_chunk()

    if inference_schedule_4hz_due() and not inference_in_flight:
        launch_async_inference(latest_observation)

    if new_result_ready():
        execution_time = monotonic() + estimated_actuator_delay
        fresh = [x for x in decode_timed_chunk(new_actions)
                 if x.timestamp >= execution_time]
        active_chunk = blend(old_remaining, fresh, steps=3)
```

其中：

- Joint Schema：`decode` 使用 `AbsoluteActions` 恢复绝对关节目标；
- EEF Schema：`decode` 使用同一个推理状态恢复绝对末端目标；
- 新 EEF chunk 的每一行都基于 launch 时的同一个 anchor 独立解码，不能逐行累加；
- legacy one-step delta chunk 先累计为 launch-anchor 目标，再进入延迟补偿和融合；
- 推理期间控制线程持续按 target timestamp 消费旧 active plan，不能被推理线程阻塞；
- active plan 意外耗尽时，在授权和反馈仍有效的前提下保持最后一个安全 absolute target，不能发送零动作或突然退出位置控制；
- 轨迹融合只插值机械臂位姿；双臂使用相同 alpha，旋转使用 SLERP/SO(3) 插值；
- 夹爪独立使用低通、step limit、迟滞和至少 2 步确认，不对 old/new gripper 直接线性平均；
- 融合后的每个 20 Hz absolute target 仍必须逐步通过 workspace、动作幅度、反馈新鲜度和 Piper 状态安全门。

---

## 6. 推荐的数据字段

### 6.1 公共字段

```text
observation.state
observation.images.cam_high
observation.images.cam_<side>_wrist
action / action.absolute_target
joint_qpos
timestamp
state_timestamp
action_timestamp
image_timestamps
instruction
task_name
success
episode_index
frame_index
```

### 6.2 数据合同元数据

```text
schema
arm_mode
arm_side
state_dim
raw_action_dim
model_action_dim
state_names
action_names
camera_keys
fps
action_horizon
action_source
action_alignment
action_offset
gripper_semantics
rotation_semantics
coordinate_frame
contract_version
```

推荐值示例：

```text
fps              = 20
action_horizon   = 50
arm_mode         = single
action_alignment = same_step_command
action_offset    = 0
coordinate_frame = slave_base
gripper_semantics = absolute_opening_fraction
```

---

## 7. 数据验收要求

每个 episode 至少检查：

1. state/action/image 数量一致；
2. state、action 不含 NaN/Inf；
3. 时间戳严格递增；
4. 实际采样频率接近目标频率；
5. 主从命令时间差在允许范围内；
6. 相机延迟和冻结帧比例可接受；
7. 关节角、EEF 位置、旋转和夹爪范围合法；
8. action 来源和 alignment 与元数据一致；
9. 训练 transform 后 action shape 正确；
10. 模型输出经逆变换后能恢复为可执行的绝对目标。

对于 EEF Schema，还应离线验证：

```text
主臂映射后的绝对 EEF 目标
          与
经过合理延迟后的从臂实测 EEF
```

二者误差若长期较大，说明主从坐标映射、时间对齐或底层控制存在问题，不应直接用于训练。

---

## 8. 当前项目推荐实施顺序

### P0：先完成 7D Joint Schema

```text
输入：从臂实测 6 关节角 + 夹爪
输出：主臂/实际从臂命令的 6 关节绝对目标 + 夹爪目标
```

理由：

- 当前仓库已经能读取主臂与从臂的 7D 数据；
- 不需要重新实现 FK、坐标标定和 IK；
- action source 清晰；
- 最容易验证数据对齐和实机执行闭环。

### P1：独立实现 10D EEF Schema

```text
输入：从臂实测 EEF 10D 状态
输出：主臂映射后的 current-anchored 7D EEF 动作
```

需要新增：

1. 主臂 FK；
2. 验证下一帧从臂 EEF pose 与同周期主臂夹爪的混合来源和时间对齐；
3. 绝对 EEF action 存储；
4. 10D absolute target → 7D current-anchored delta transform；
5. 7D delta → 绝对 EEF target 输出变换；
6. IK 连续性、奇异位形和关节限位检查；
7. 5 Hz 推理、20 Hz action queue。

---

## 9. 最终建议

短期实机演示优先采用：

```text
7D 从臂关节状态
        ↓ π0.5
7D 主臂/从臂关节控制目标
        ↓
Piper JointCtrl
```

完成稳定基线后，再验证：

```text
10D 从臂末端状态
        ↓ π0.5
7D current-anchored 末端位姿增量
        ↓ 绝对位姿解码
Piper EndPoseCtrl / 笛卡尔控制器
```

两种方案的核心不变：从臂状态作为模型观测，操作者实际控制意图作为 action；action chunk 中所有未来动作统一以当前观测状态为锚点。
