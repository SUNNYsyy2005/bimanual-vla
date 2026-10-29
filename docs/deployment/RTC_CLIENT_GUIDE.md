# 实机 RTC（Real-Time Chunking）部署指南

这里的 RTC 指 **Real-Time Chunking**，不是单纯的实时控制客户端。
它用于补偿相机采集、网络传输和模型推理造成的 action chunk 延迟：

- 服务端在 **flow-matching denoising 内部**读取上一 chunk 尚未执行的 normalized prefix；
- 普通 checkpoint 用 prefix guidance；训练过 RTC 的 Pi05 checkpoint 用干净前缀条件化；
- 客户端只上传 session、generation、已执行 offset 和 latency 估计，不上传或插值 normalized action；
- 客户端以 20 Hz 消费目标和检查授权，可选独立 100 Hz Ruckig 伺服执行。

服务端实现位于 `bimanual_vla/deployment/rtc_policy.py`，同时支持 OpenPI 的 PyTorch `model.safetensors`
和 JAX/Orbax `params` checkpoint。Dashboard 创建 Policy 时默认带上
`--rtc-enabled`；也可以在命令行显式关闭。

## Shadow 模式

默认只读反馈、采集相机并请求 Policy，不向机械臂发送动作：

```bash
cd /path/to/bimanual-vla
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

- `--rtc-execution-horizon 12`：RTC 前缀融合范围；必须不大于 action horizon；
- `--rtc-max-guidance-weight 5.0`：denoising guidance 上限；
- `--rtc-prefix-attention-schedule linear`：可选 `zeros`、`ones`、`linear`、`exp`；
- `--rtc-inference-delay-steps 8`：客户端显式配置服务端 RTC 前缀延迟步数；RTC 额外客户端混合已移除，旧 blend 参数仅兼容接收并忽略；
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
  --servo-backend ruckig --servo-hz 100 \
  --allow-execution
```

`--allow-execution` 不是绕过安全门的开关。没有 Dashboard 授权、授权过期、
telemetry 断开或任一逐周期安全检查失败时，客户端只会保持安全目标或阻断发送。

## 控制与 RTC 时序

### 可选关节参考规划器

`--servo-backend ruckig` 使用本地 Ruckig 0.12.2 的在线轨迹生成，默认独立
100 Hz 下发；`off` 直接发送目标，`legacy` 保留旧整形算法以便对照。GUI
的 Enable reference planner 总开关关闭时等价于 `off`。为兼容已有配置，
命令行和保存项仍保留 `servo` 名称；这里不是电机位置/速度/电流闭环。

规划核心 `RuckigJointPlanner.advance()` 只根据持续保存的参考位置、速度、
加速度生成下一周期位置指令。Piper 驱动器负责实际跟踪。执行层独立检查
实测反馈的新鲜度、驱动状态及实测位置与已生成参考的偏差，超限阻止下发；
不把反馈噪声直接注入规划状态，也不在这一层增加 PID 或误差积分。

Ruckig 每周期延续生成轨迹的位置、速度和加速度，并联合使用
`--trajectory-max-speed-rad-s`、`--trajectory-max-acceleration-rad-s2` 和
`--trajectory-max-jerk-rad-s3` 三个正数约束。旧 lowpass、tracking、spike、
lookahead 算法不参与 Ruckig 关节轨迹生成；不能用旧的单项开关关闭其约束。
默认关闭相邻目标末速估算（`--no-servo-waypoint-velocity`）。普通逐点路径仍使用
零末速点到点规划；trained joint RTC 使用下面的多点规划路径。

推理的 Piper 驱动速度百分比默认为 100%，GUI 启动命令显式传入
`--speed-pct 100`。启用规划器时，轨迹速度、加速度和 jerk 由界面中的参数控制，
并受读取到的驱动限值约束，不再隐式使用旧的 10% 默认值。
启动日志及运行记录同时保存驱动速度百分比、请求轨迹速度和生效轨迹速度。
命令行仍可显式指定较低的 `--speed-pct`，此时会相应降低生效轨迹限速。

### trained RTC 的客户端规划前缀

当 joint policy 公布 `rtc_mode=trained`、`rtc_capabilities.client_planner_prefix=true`，
且客户端开启 RTC 和 Ruckig 时，客户端会先生成整段带时间的可执行参考。规划采用
三点平滑、保形三次插值及两步前视，从未来参考曲线取得目标速度，再由本地 Ruckig
生成有界的 100 Hz q/v/a 序列。它是多点前视规划，不是全局最优轨迹求解器，也不是
原先从两个已经接收的目标反推到达速度。使用现有速度、加速度、jerk 设置。

发起推理时，从这份同样用于下发的计划取 `capture+(i+1)/action_hz` 的位置，
发送 `client_metadata.rtc.prefix_source=client_planner` 与 `planned_prefix`。
前缀包含 plan_id、按 policy action_hz 采样的时间偏移和完整动作：关节绝对 rad，夹爪按协议转换成
开度。服务端对全部维度重新执行训练对应的 DeltaActions/Normalize，不再复制
上一次模型 chunk。启动/保持时没有有效计划就显式发送空前缀，不回退到旧模型输出。
旧服务端没有这项能力时客户端会明确报错，不能静默忽略规划前缀。

新 chunk 无法接入现有计划时，本代动作改由逐目标 Ruckig 执行，保留上述客户端
前缀协议，作废旧计划及其前缀承诺。下一次没有可用执行计划就发送空前缀，从新
观测重新预测；不会把尚未执行到的旧模型预测传回服务端作为 RTC 前缀。
每个新 generation 都会重新尝试安装计划，失败代不重复尝试。

`inference_delay_steps` 仍由用户手动设置，没有增加延迟预测。发送的前缀时间范围
被保留；新结果提前到达时，仅替换保留段之后的轨迹，并从边界 q/v/a 接续，不对
保留段再次平滑。规划计算在发送锁之外进行，预留至少 50 ms 准备窗口；超时不安装
未经及时验证的计划。暂停、hold、故障、时钟超差会使旧 plan_id 作废，依赖它的
推理结果不能接纳。实际返回超出固定前缀时仍按实际时间选点，不能保证没有接续偏差。

夹爪也提前规划并限速（默认不超过 0.08 m/s，并受原单步上限换算的速度约束），
该路径不再使用会事后改变计划的逐点夹爪低通/端点确认。位置仍钳到合法开度。
已有完整计划不会因为网络等待 0.15 秒而制动；计划末尾包含受约束停车段，正常
安全门仍可立即中断它。发布线程按当前单调时钟取计划样本，跳过已经错过的样本，
不逐点追赶、不突发补发。如果一次跳过两个或更多中间样本，会作废待返回结果所
依赖的前缀并记录 timing_invalidations，但不会仅因调度迟到永久锁死健康的线程。
下发前仍检查反馈到当前计划目标的误差；反馈、驱动或位置越界故障仍阻止下发。
普通 Linux 不能保证硬实时、零时间误差或物理执行与计划完全一致。

规划时除 Ruckig 的位置极值接口外，还对各段多项式求速度零点并检查位置，覆盖
0.12.2 版本可能漏报的匀加速内部极值。若带末速轨迹越界，尝试零末速重规划；
仍不可行则拒绝计划，不能直接钳位生成状态破坏速度/加速度连续性。
关节位置极值、输出和 chunk 接续使用统一的 `1e-7 rad` 绝对数值容差，
仅用于吸收浮点残差（小于 Piper 0.001° 指令分辨率的 1/170）；不重置或
裁剪内部 q/v/a，也不修改机械限位。超过该数值容差仍拒绝；夹爪容差不变。
尚未安装的新 chunk 如果有未来点越界或离线规划不可行，按候选轨迹拒绝处理：
丢弃该 chunk，健康的发布线程受控制动，同时作废其旧 RTC 前缀承诺。
制动完成后可接纳新的合法 chunk，不将候选校验失败变成永久硬暂停。
新计划准备超时仍保留原计划并延后重试；驱动、反馈、实际发布线程故障仍阻断。

`--servo-target-timeout-s` 默认 0.15 秒。目标超时、正常 hold 和关闭时，
健康的规划/下发线程生成受约束制动；反馈或驱动故障时停止下发。这里的
超时指执行队列停止提供参考，不是推理网络等待；已有 chunk 仍在供给目标
时会继续规划执行。退出后只有确认
生成轨迹已经静止才允许回零控制接管。双臂共用 12 关节轨迹计算和唯一发送
线程；普通 Linux 调度不保证硬实时，迟到周期不突发补发。

运行记录的 `servo_commands.jsonl` 保存高频轨迹及下发时间：`reference` 是输入
目标，`command_qpos` 是规划后的下发位置，`feedback_qpos` 是本周期读取的实测
位置，`feedback_read_monotonic` 是本机读取时间（不是传感器采样时间）。旧字段
`qpos` 继续表示下发位置，不能当作实测反馈；`role=joint_reference_planner`
标明层级。`plan_id` 和 `planned_monotonic` 标明材料化计划及对应的计划时间。
20 Hz 控制记录仍
用于模型队列与安全门诊断。运动界限约束的是生成指令，实际机械臂跟踪效果
仍需实机验证。算法来源及边界见
[开源控制参考](OPEN_SOURCE_CONTROL_REFERENCES.md)。

### 训练 RTC

训练时 RTC 与原有推理 guidance 分开选择：服务端默认 `--rtc-mode auto`，
按 checkpoint 元数据自动选择 `trained` 或 `guided`。训练选项、恢复兼容性
和 API 示例见 [训练 API](../../server_4090/API_USAGE.md#72-启动训练)。
模型目标仍按 action rate 消费，高频伺服独立完成插值与运动约束。

### 推理完成通知与延迟目标

推理线程完成后通过事件唤醒控制线程，在周期之间接纳结果，不再等下一次
20 Hz 轮询。只有控制线程修改动作队列。Ruckig 模式接纳新计划后额外执行
一次完整反馈/授权/安全检查，立即刷新当前伺服参考；普通目标消费仍沿用原
20 Hz 时间轴，不把整个旧队列加速消费。实际 CAN 发布仍由唯一伺服线程按
`servo-hz` 发送。`off`/`legacy` 模式仅提前接纳结果，保留原下发节奏。

处理链路以 150 ms 为优化目标，包含观测准备、编解码、模型处理、接纳和
首次下发，网络传输另计；不是触发机器人故障的硬超时。运行配置会记录
`inference_completion=event_wakeup` 和 `immediate_reference_refresh`。

客户端现在直接记录 `client_pack_ms`、`client_send_ms`、
`client_wait_response_ms`、`client_unpack_ms`、请求/响应大小及图像尺寸。
服务端记录 `server_unpack_ms`、`server_policy_ms`，并在下一次响应的
`previous_response_timing` 中按 generation 返回上次打包与发送耗时。
`response_to_admission_ms` 衡量结果返回后等待接纳的时间；首次下发指标使用
实际命令完成时间。`unattributed_rpc_ms` 是未拆分残差，仍包含网络、调度和
响应处理，不能直接当成纯网络延迟从 150 ms 目标中扣除。

本机无硬件验证可运行：

```bash
python -m scripts.smoke.benchmark_rtc_wakeup --samples 30 --model-ms 109.5
```

该脚本只连接本机临时 WebSocket 服务，以 sleep 模拟模型处理，使用内存
反馈/发布回调验证 Ruckig 调度。它不验证真实模型、相机、CAN 或机械臂性能。

- 相机和 Piper 反馈持续运行；
- 异步推理默认使用 `--inference-trigger-mode periodic`，以 `--hz 4` **尝试发起**请求，单次只允许一个在途请求；
- 可改用 `--inference-trigger-mode chunk_step --inference-trigger-step 10`：每个已接受 chunk 的原始第 10 步成功下发后发起下一次请求，旧 chunk 在推理期间继续执行。若新 chunk 跳过了前缀，仍按其原始步号判断触发点；每个 chunk 只触发一次；
- `chunk_step` 模式下 `--hz` 是首次请求、失败及队列耗尽时的重试频率，不限制正常的步号触发。触发步号必须小于服务端公布的 action horizon；
- `--control-hz` 控制模型目标消费频率，必须与服务端公布的 action rate 一致；Ruckig 模式下，实际下发由独立的 `--servo-hz` 控制（默认 100 Hz）。GUI 左侧 Policy and task 的 Inference timing 区域设置目标消费频率；Joint servo 区域选择伺服后端和频率；
- 周期模式的 4 Hz 是调度目标，不是实际吞吐保证；若 capture-to-result 为 550 ms，实际频率上限约为 `1/0.55=1.82 Hz`；
- 客户端 telemetry 分开上报 `configured_inference_hz`、`inference_launch_hz`、`inference_result_hz`，并上报单在途上限；
- 客户端使用显式配置的 `inference_delay_steps`，不根据上一轮延迟改写；
- 客户端根据 active chunk 的 `source_index` 发送 `previous_chunk_offset_steps`；
- 服务端按 WebSocket session 保存上一轮 normalized chunk，并在 denoising 时应用 RTC guidance；
- 新 action chunk 到达后，客户端继续用旧 chunk 消费，按实际 capture/launch/arrival 时间丢弃过期前缀；
- RTC 模式不做额外客户端轨迹混合；按实际经过时间选择接续点，延迟超过配置步数时记录 WARNING；
- 推理失败、generation 不匹配、连接断开或队列耗尽时 fail closed 并保持最后安全目标；
- `monitoring_data/<session>/events.jsonl` 和模型结果记录中包含 RTC telemetry。

运行中的 Trajectory jitter 显示在 Dashboard 的 Policy 实时观测区，并保存在
`monitoring_data/<session>/events.jsonl` 的周期 `control_tick.trajectory_jitter`
和结束事件中。完整运行记录还包含 `deployment_runs/<run>/trajectory_jitter.jsonl`
（逐 chunk 与边界事件）、`metadata.json` 的 `trajectory_jitter` 汇总，以及
`trajectory.npz` 的 `command_joints_rad` 和 `command_monotonic_timestamp`。
使用 `--no-recording` 时不生成这些抖动统计。

客户端 **Data Process** 页按 **Trajectory / Chunk boundaries / Timing / End effector**
分组选择图表。Position 显示单个关节的测量值和录制目标，部署记录还叠加原始模型预测；
Velocity 和 Acceleration 分别叠加模型输出与实际下发曲线；Tracking error 显示目标与测量值之差。
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
切换 generation。这两种分隔线也显示在 Position 和 Tracking error 图上；
3D 末端轨迹以黄色虚线圆圈标出实际下发 chunk 的切换点。
模型曲线以预测到达时刻为起点，按该预测的 `action_hz` 展开完整 horizon。多次预测
覆盖相同未来时间时，较新的预测用实线，较早预测被后续结果覆盖的部分仍保留并改用浅蓝虚线；
这表示模型对同一时刻的不同预测，不代表机器人同时执行多个速度。位置图也显示这些预测，
所以可以直接对照实测、记录目标和各代模型输出。模型预测曲线不是机器人的实际执行轨迹。
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
  the non-RTC blend controls only when RTC is disabled (legacy RTC blend flags are ignored).
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


### 2026-10-19 RTC 参数归属更新

GUI 增加 `Inference delay steps`（默认8），保留 `Execution horizon`（默认12）、
引导权重和 schedule，移除 `Extra blend with RTC`。`Blend profile` 仅在关闭RTC且启用混合时可用。
要求 `0 <= delay <= horizon <= 模型chunk长度`。服务端不再按启动时的horizon默认值截断客户端配置；
模型chunk长度和trained RTC训练支持的最大delay仍是真实能力约束，不可通过GUI绕过。
RTC返回后按实际时间选取目标；非RTC保留原有进度不足时重计时逻辑。
超过delay时客户端日志输出 `RTC delay exceeded`，包含实际毫秒数、步数及配置horizon。
默认值是配置初值，不代表已为当前网络和机器人标定。旧GUI保存的额外RTC混合设置不再生效。

### Ruckig 默认运动约束（2026-10-19）

速度 / 加速度 / jerk 默认值调整为 **1.20 rad/s / 8.0 rad/s² / 60.0 rad/s³**，
替代之前的 0.60 / 3.0 / 20.0；同步 ServoConfig、客户端命令行和 GUI 默认值。
这是已做离线回放的调试起点，用于减轻过紧约束造成的跟踪与制动滞后，尚未实机标定。
GUI 已保存的自定义数值仍优先于默认值；使用旧偏好时应检查界面实际显示的三个上限。

### EEF IK 独立计算进程（2026-10-27）

新增 `--eef-control-mode local_ik_joint|piper_eef`；默认保留本地 IK。
原生 EEF 方式、GUI 切换、平滑与 RTC 的关系见 [双控制方式说明](NATIVE_EEF_CONTROL.md)。
以下 IK 工作进程和缩步流程只用于本地 IK 方式。

EEF 执行客户端默认启用 `--ik-worker-processes`，每臂一个常驻计算进程，
在启动动作前创建。启动日志 `EEF IK workers ready (spawn, no CAN)` 包含进程 PID。
计算进程不访问 CAN 或相机；主进程收齐同一批双臂解后才发布目标。
关节角策略不启动 IK 进程。停止并重新启动客户端即可加载修改，policy 服务无需重启。

`--no-ik-worker-processes` 可用于对照进程内求解。两种方式都使用解析 Jacobian，
保留关节限位、解误差和连续性检查。轨迹诊断中的 `solve_elapsed_s` 为单臂求解耗时，
`parallel_batch_wall_ms` 为双臂批次含通信的总耗时；关闭 EEF 平滑时也记录 IK 诊断。
离线双臂批次 P95 约16.2 ms，不代表相机、录像、CAN 全链路已通过30 Hz实测。

### EEF 缩步恢复（2026-10-28）

Ruckig EEF 路径的本地 IK 几何失败会依次尝试共同的 1/2、1/4 笛卡尔步长，
双臂都通过限位、求解误差和有效进展检查才发布。保留驱动关节硬限位，
恢复期间保持实测夹爪开度，不扩大 IK 搜索半径或误差容限。
原始求解与两档恢复共用 100 ms 截止时间；进程内求解不能中途抢占，但超时结果不会发布。

只有实际 IK 缩步恢复进入原目标保持流程：丢弃旧队列，
保留该行未经缩步/投影的 EEF 目标。到达中间关节参考后，从新反馈继续求解并逼近
同一原目标；已经到位的另一臂保持实测关节。期间双臂夹爪保持进入流程时的开度，
暂停新推理并拒绝在途旧结果。原目标未完成前不会切换到下一段模型动作。

正常执行按原时间轴连续跟踪，夹爪变化、局部步长投影都不要求每点停稳。
RTC 正常路径旁路逐点制动式 EEF 平滑，即使设置 `--pose-trajectory`；启动日志及
`pose_trajectory_bypass_reason=rtc_continuous_tracking` 明确记录。保留本地 IK 和
Ruckig 关节速度/加速度/jerk 约束，软件不私自减慢 RTC 时钟。

启用 `--pose-trajectory` 时，异常恢复的续步保留同一个平滑参考，每个控制周期推进，
不要求每个中间关节参考先停稳。若发生真正的 1/2、1/4 IK 缩步，先等待该步的
规划器和实测关节稳定，再从新反馈重建参考。夹爪释放还需平滑参考收敛且静止。
小幅平滑中间步不套用相对原目标的逐步百分比进展门槛；仍检查 IK 误差、驱动限位
和原目标总超时。诊断 `continuous_ik.shrink_recovery.pose_reference` 记录
`stopping_distance_reference_v2`、停止距离限速与目标变化制动；
`pose_reset_waiting_for_stop` 标记缩步后等待重建参考的阶段。
算法与数据集回放结果见 [平滑优化记录](../../analysis/eef_parameters_20261028/OPTIMIZATION.md)。

异常恢复期间必须同时满足双臂实测 EEF 到位、规划器静止及关节跟踪误差小于 0.01 rad，
才推进该行夹爪滤波。夹爪命令通过滤波及写入器后，清除旧 RTC 前缀，
只接收完成之后采集的新观测。夹爪完成依据是写入器命令开度，而不是要求夹住
物体后的实测开度必须为零；仍保留原有夹爪步长限制。

GUI 的夹爪面板和 CLI 可设置恢复阶段的到位门槛（不应用于正常途经点）：

- `--eef-goal-position-tolerance-m 0.005`：原目标位置误差 5 mm。
- `--eef-goal-rotation-tolerance-rad 0.05`：原目标姿态误差 0.05 rad。
- `--eef-goal-timeout-s 2`：整个逼近及夹爪命令发布过程共用 2 秒截止时间。

EEF/Ruckig 路径强制关闭提前张爪，即使旧配置仍传 `--gripper-open-lookahead-steps 30`。
joint/legacy 路径仍支持显式设置提前张爪。到位容差不改变 IK 的求解误差容限。
进展要求加权误差至少改善 1%（另有微小绝对下限）；位置和姿态分别不能恶化到
原误差与既有 IK 容差两者的较大值之外，允许容差内的小幅权衡。

两档均无有效进展、超时或夹爪切换期间位姿漂移，会停止推进、锁定执行并保持健康
写入器制动，需检查现场后重新启动客户端。CAN 故障、工作进程超时和无效输入
不触发缩步重试。日志 `Original EEF goal retained` 不代表已到位；
`Original EEF goal reached` 表示通过到位检查且夹爪命令已发布。
执行元数据保存原目标、分臂误差、步数和最后一次精确 IK 请求，方便离线复现。

反馈轻微越过零点可以进入求解，但求解区间及结果仍须在驱动限位内。
例如 J2 反馈 -0.000925 rad 不会授权负角度目标。软件停止或进程清理不等于
物理保持已确认，也不会通过自动失能来释放机械臂。

### 连续跟踪与持续失配（2026-10-28）

每次新反馈同时检查已发送关节指令，以及 EEF 路径中未经 Ruckig 限速的 IK 参考、
未经局部投影/平滑的原始策略 EEF 目标。这样可以发现“发送样本跟得上，但整体策略
目标已经落后”的情况。原目标误差每个反馈周期检查，不受诊断采样 stride 影响。
采用上一条参考与下一次反馈比较；并非对任意网络/执行延迟作精确时间补偿。

- `--tracking-lag-threshold-rad 0.10`：已发送/IK 参考的最大关节误差。
- `--tracking-lag-position-threshold-m 0.05`：原策略目标的位置误差。
- `--tracking-lag-rotation-threshold-rad 0.18`：原策略目标的姿态误差。
- `--tracking-lag-confirm-cycles 3`：已发送关节指令误差触发保护所需的连续有效反馈次数；参考误差也至少需要此次数。
- `--tracking-reference-lag-duration-s 0.5`：IK 参考／原策略 EEF 误差还必须连续超限至少此时间，才触发恢复。不能把尚在 Ruckig 加速追赶的目标当作已发送关节指令，三帧就刹停。

这些是可配置的初始保护阈值，尚未经本任务实机标定；不改变驱动硬限位。
短暂超限后恢复会清零计数，同一时间戳的重复反馈不重复计数，反馈间隔过大也会
重置连续计数。反馈过期、CAN 故障和驱动错误仍走原有立即保护，不等待三次确认。
两类证据独立计时；新 chunk 不重置参考误差累计。持续 0.5 秒是有界的追赶窗口，
尚未经实机标定，不代表任意模型轨迹都能在此时间内跟上。

持续失配时，立即清除队列、无效化 RTC 前缀、调用健康写入器制动并暂停推理。
Ruckig 规划速度/加速度接近零、关节跟踪误差小于 .01 rad，且连续反馈关节变化
小于 .001 rad，通过连续稳定检查后才建立新观测屏障。稳定前采集的在途结果不能
恢复执行；稳定后的推理以空旧前缀重新起算。没有对旧轨迹静默降速或重计时。
该保护不根据夹爪开合猜测抓取/放置阶段，也无法纠正模型本身错误的夹爪时机。

离线回归：413 项测试、93 个子测试通过，覆盖连续行推进、短暂误差、重复反馈、
整体目标滞后、制动等待、过期推理拒绝、稳定后重规划及原始工作空间检查。
未连接机械臂；阈值与实际抓放效果仍需现场验证。新客户端进程加载本次修改，
不需要更改 policy 服务端协议。
