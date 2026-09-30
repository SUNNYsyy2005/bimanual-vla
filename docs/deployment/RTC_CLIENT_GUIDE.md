# 实机 RTC（Real-Time Chunking）部署指南

这里的 RTC 指 **Real-Time Chunking**，不是单纯的实时控制客户端。
它用于补偿相机采集、网络传输和模型推理造成的 action chunk 延迟：

- 服务端在 **flow-matching denoising 内部**读取上一 chunk 尚未执行的 normalized prefix；
- 用 prefix guidance 让新 chunk 的前缀连续地贴近上一 chunk；
- 客户端只上传 session、generation、已执行 offset 和 latency 估计，不上传或插值 normalized action；
- 客户端仍负责 20 Hz 安全执行、队列消费和必要的 fail-closed 保护。

服务端实现位于 `bimanual_vla/deployment/rtc_policy.py`，同时支持 OpenPI 的 PyTorch `model.safetensors`
和 JAX/Orbax `params` checkpoint。Dashboard 创建 Policy 时默认带上
`--rtc-enabled`；也可以在命令行显式关闭。

## Shadow 模式

默认只读反馈、采集相机并请求 Policy，不向机械臂发送动作：

```bash
cd /home/user/dual_ARM_project/arm_collect/bimanual-vla
bin/bimanual-vla rtc-client \
  --host 192.168.101.9 \
  --port 8000 \
  --arm-mode bimanual \
  --arm-side both \
  --left-can can0 \
  --right-can can1 \
  --cam-high-device auto \
  --cam-left-wrist-device auto \
  --cam-right-wrist-device auto \
  --instruction "pick up the cube" \
  --hz 4 \
  --control-hz 20 \
  --rtc-enabled
```

客户端会根据 Policy metadata 自动协商 RTC 参数。常用参数：

- `--rtc-execution-horizon 8`：RTC 前缀融合范围；必须不大于 action horizon；
- `--rtc-max-guidance-weight 5.0`：denoising guidance 上限；
- `--rtc-prefix-attention-schedule linear`：可选 `zeros`、`ones`、`linear`、`exp`；
- `--rtc-client-blend-steps 0`：默认关闭额外的客户端 old/new blend，避免把延迟重新加回来；
- `--no-rtc-enabled`：显式关闭客户端 RTC 协议。服务端也必须发布 `rtc_enabled=false` 才会完全关闭。

## 推理图像预缩放

GUI 的 **Devices → Pre-resize policy images to 224 px** 对应
`--preresize-policy-images`，默认关闭。启用后，相机后台线程仍先生成并记录
256×256 图像，再按服务端 OpenPI 使用的 PIL 双线性算法生成独立的 224×224
策略图像；服务端收到 224×224 后跳过重复缩放。录像与本地预览继续使用原图。
这个选项需要重新启动推理客户端才能生效。比较性能时同时检查模型耗时、
端到端延迟和相机帧龄；服务端模型耗时下降不等于整体观察到动作的延迟必然下降。

## 实际执行模式

必须同时满足以下条件，客户端才会发送一条真实 Piper 命令：

1. 启动参数包含 `--allow-execution`；
2. Dashboard 已对同一个运行中的 Policy task 授予未过期的 `EXECUTE`；
3. Policy WebSocket、Piper CAN 反馈和相机数据均新鲜；
4. schema/action/camera/time contract 握手一致；
5. action horizon、workspace、关节/夹爪变化、IK 和 Piper 驱动状态检查全部通过。

```bash
bin/bimanual-vla rtc-client \
  --host 192.168.101.9 \
  --port 8000 \
  --arm-mode bimanual \
  --arm-side both \
  --left-can can0 \
  --right-can can1 \
  --cam-high-device auto \
  --cam-left-wrist-device auto \
  --cam-right-wrist-device auto \
  --instruction "pick up the cube" \
  --hz 4 \
  --control-hz 20 \
  --rtc-enabled \
  --allow-execution
```

`--allow-execution` 不是绕过安全门的开关。没有 Dashboard 授权、授权过期、
telemetry 断开或任一逐周期安全检查失败时，客户端只会保持安全目标或阻断发送。

## 控制与 RTC 时序

- 相机和 Piper 反馈持续运行；
- 异步推理默认使用 `--inference-trigger-mode periodic`，以 `--hz 4` **尝试发起**请求，单次只允许一个在途请求；
- 可改用 `--inference-trigger-mode chunk_step --inference-trigger-step 10`：每个已接受 chunk 的原始第 10 步成功下发后发起下一次请求，旧 chunk 在推理期间继续执行。若新 chunk 跳过了前缀，仍按其原始步号判断触发点；每个 chunk 只触发一次；
- `chunk_step` 模式下 `--hz` 是首次请求、失败及队列耗尽时的重试频率，不限制正常的步号触发。触发步号必须小于服务端公布的 action horizon；
- `--control-hz` 控制机器人下发频率，必须与服务端公布的 action rate 一致；GUI 左侧 Policy and task 的 Inference timing 区域可设置触发模式、步号、请求频率和控制频率；
- 周期模式的 4 Hz 是调度目标，不是实际吞吐保证；若 capture-to-result 为 550 ms，实际频率上限约为 `1/0.55=1.82 Hz`；
- 客户端 telemetry 分开上报 `configured_inference_hz`、`inference_launch_hz`、`inference_result_hz`，并上报单在途上限；
- 客户端根据上一轮 capture-to-result latency 估计本次 `inference_delay_steps`；
- 客户端根据 active chunk 的 `source_index` 发送 `previous_chunk_offset_steps`；
- 服务端按 WebSocket session 保存上一轮 normalized chunk，并在 denoising 时应用 RTC guidance；
- 新 action chunk 到达后，客户端继续用旧 chunk 消费，按实际 capture/launch/arrival 时间丢弃过期前缀；
- RTC 模式默认不做额外客户端轨迹插值；只有显式设置 `--rtc-client-blend-steps` 才启用安全 fallback；
- 推理失败、generation 不匹配、连接断开或队列耗尽时 fail closed 并保持最后安全目标；
- `monitoring_data/<session>/events.jsonl` 和模型结果记录中包含 RTC telemetry。

运行中的 Trajectory jitter 显示在 Dashboard 的 Policy 实时观测区，并保存在
`monitoring_data/<session>/events.jsonl` 的周期 `control_tick.trajectory_jitter`
和结束事件中。完整运行记录还包含 `deployment_runs/<run>/trajectory_jitter.jsonl`
（逐 chunk 与边界事件）、`metadata.json` 的 `trajectory_jitter` 汇总，以及
`trajectory.npz` 的 `command_joints_rad` 和 `command_monotonic_timestamp`。
使用 `--no-recording` 时不生成这些抖动统计。

客户端 **Data Process** 页按 **Trajectory / Chunk boundaries / Timing / End effector**
分组选择图表。Position 同时显示单个关节的测量值和录制目标；Velocity 和 Acceleration
分别叠加模型输出与实际下发曲线；Tracking error 显示目标与测量值之差。
Chunk boundaries 保留位置跳变和速度方向两张边界图；chunk 内平均加速度仍在指标表中，
无需再用每个 chunk 的平均值曲线重复展示。Timing 收纳推理延迟和
控制间隔。End effector 的 **3D trajectory** 参考 Dashboard 的数据集末端位姿视图：
显示所选时间范围内的左右臂空间轨迹、末端最新点、基座和世界坐标轴；拖拽旋转、滚轮缩放，
可选单臂或双臂并重置视角。10D/20D 录制末端状态直接使用 XYZ，关节状态通过 FK 计算。
旧部署记录若省略机器人类型，会按实机 Piper 解释右臂坐标轴；缺少双臂基座偏移时，
左右轨迹分屏显示在各自对齐的坐标系里，不用虚构的共同基座位置叠画。
Episode 来源只提供有记录数据的视图。

两套 chunk 指标分别是：

- **Policy output**：统计客户端收到且已接受的完整、等间隔绝对关节目标 chunk（即服务端输出变换后的 `actions`）；边界使用上一预测的最后一行与下一预测的第一行。它可在关闭动作执行的影子推理中记录，但完整 horizon 的边界不代表机器人实际切换点。
- **Command sent**：统计通过安全检查、轨迹整形后实际下发的关节命令；边界使用真实发送顺序。关闭动作执行时该列为空。两列均排除单位不同的夹爪维度；非 joint schema 不计算这些关节指标。

两套逐 chunk/边界事件写入同一个 `trajectory_jitter.jsonl`，通过 `stream` 区分；`metadata.json` 分别保存 `model_trajectory_jitter` 和 `trajectory_jitter` 汇总。Data Process 也可从旧运行记录保存的模型 NPZ 和命令轨迹补算两套指标。

Data Process 的 **Velocity** 和 **Acceleration** 两张图均叠加 **Policy output**
与 **Command sent** 曲线，展示逐关节或关节向量 L2 范数，单位分别为 `rad/s`、`rad/s²`；
`Signal` 可切换具体关节。蓝色竖向虚线表示新预测到达，橙色竖向虚线表示实际下发命令
切换 generation。Policy 曲线以预测到达时刻为起点，按该预测的 `action_hz`
展开完整 horizon，因此不同预测的曲线可能在时间上重叠；它不是机器人的执行轨迹。
Sent 曲线按实际命令时间戳求导，hold、漏发/跳步、不规则控制间隔以及 chunk 边界均断线，
不会把这些间隔的变化误算为 chunk 内加速度。两张图均只使用弧度制关节目标，
排除夹爪；影子推理仅显示 Policy 曲线。逐行 Action delta 不具备上述边界和时间戳语义，
因此不再作为独立曲线入口。

- **Mean intra-chunk acceleration magnitude**：同一 chunk 内，连续且接近固定控制周期的已下发关节命令，其二阶差分 L2 范数的均值，单位 `rad/step²`。跳步、未下发命令和保持命令不参与计算；此指标只在固定采样频率下有意义。
- **Position jump at chunk boundary**：上一 chunk 最后一条实际下发关节命令与下一 chunk 第一条之间的位置 L2 距离均值，单位 `rad`。
- **Cosine similarity of velocity direction at chunk boundary**：用上一 chunk 最后两个连续命令及下一 chunk 最前两个连续命令计算方向余弦均值。存在保持、漏周期或零速度时，该边界不纳入余弦均值；Dashboard 同时显示有效样本数。

监控 JSONL 写盘、运行记录与视频编码均在后台线程执行；GUI 控制台日志通过有界队列输出，队列满时丢弃控制台日志以保护控制周期。日志不通过动作 WebSocket 传输。Dashboard 图像预览最多每秒更新一次，浏览器只在新图像序号出现时重新下载。

GUI 左侧 **Policy and task → Policy request diagnostics** 的开关对应客户端的
`--send-policy-telemetry` / `--no-send-policy-telemetry`（默认开启）。关闭后，
客户端不再为主推理请求构造执行队列、设备、相机时间戳及 jitter 等详细诊断字段；
模型状态、相机图像、RTC、会话标识、请求时间戳和本地执行授权仍会发送。
本地异步监控与录制继续工作，但 Dashboard 的客户端执行细节可能显示为未知。
每次切换都需重新启动推理客户端；可在相同策略、相机及请求频率下，对比
`monitoring_data/<session>/events.jsonl` 中 `inference_result.execution`
的 `observation_upload_ms`、`round_trip_ms` 和 `model_inference_ms`。
`observation_upload_ms` 包含客户端打包、上行传输及服务端接收解包时间，
并依赖两台机器的时钟同步；`round_trip_ms` 使用客户端单机单调时钟。

### 重要约束

RTC 必须在模型 denoising 阶段运行；只在客户端做 action 插值不等价于 RTC。
服务端 metadata 中应看到：

```json
{
  "rtc_enabled": true,
  "rtc_algorithm": "real_time_chunking_prefix_guidance",
  "rtc_backend": "jax 或 pytorch"
}
```

旧的 `bin/bimanual-vla legacy-bridge` 仍可运行，但新实机部署统一使用
`bimanual_vla/deployment/client.py`；它们共用同一份安全检查和实时控制实现。


## Smooth Piper execution options

The robot-side client also carries the execution safeguards from the Piper
reference implementation. They are independent of model training and can be
used for joint or delivery policies:

- `--trajectory-shaping` is enabled by default. It applies one shared 7D/14D
  state to the two arms and limits joint velocity, acceleration, jerk, and
  MOVE_J lookahead. Disable it only for an intentional A/B comparison with
  `--no-trajectory-shaping`.
- `--blend-profile smootherstep` removes the velocity jump at an accepted
  chunk boundary. RTC already performs model-side overlap guidance, so the
  default extra client blend is zero while RTC is active; opt in with
  `--rtc-client-blend-steps 2`, `3`, or `4` when needed.
- `--gripper-open-lookahead-steps 30` advances only opening requests. Closing
  is never anticipated, and the resulting command still goes through the
  independent gripper low-pass, hysteresis, and rate limits.
- `--reject-external-control-streams` checks Piper's reported
  `JointCtrl`/`GripperCtrl` rates before execution and refuses concurrent
  high-rate control. Use `--no-reject-external-control-streams` only when the
  hardware integration deliberately owns that arbitration.
- `--auto-return` records the measured startup pose and returns all commanded
  arms with the same bounded trajectory before disconnecting. It is enabled by
  default; `--no-auto-return` is available for a deliberate exception.

Tune the shaper with `--trajectory-max-speed-rad-s`,
`--trajectory-max-acceleration-rad-s2`, `--trajectory-max-jerk-rad-s3`,
`--trajectory-smoothing-cutoff-hz`, `--trajectory-tracking-time-constant-s`,
`--trajectory-command-lookahead-rad`, and
`--trajectory-max-tracking-error-rad`. RTC temporal consistency is enabled for
JAX servers by default and can be controlled with the server-side
`--rtc-temporal-consistency` and `--rtc-temporal-seed` options. All of these
states and decisions are emitted in the per-session monitoring telemetry.
