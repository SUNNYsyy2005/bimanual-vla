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
- Policy 推理默认以 4 Hz **尝试发起**，单次只允许一个在途请求；
- 因此 4 Hz 是调度目标，不是实际吞吐保证；若 capture-to-result 为 550 ms，实际频率上限约为 `1/0.55=1.82 Hz`；
- 客户端 telemetry 分开上报 `configured_inference_hz`、`inference_launch_hz`、`inference_result_hz`，并上报单在途上限；
- 客户端根据上一轮 capture-to-result latency 估计本次 `inference_delay_steps`；
- 客户端根据 active chunk 的 `source_index` 发送 `previous_chunk_offset_steps`；
- 服务端按 WebSocket session 保存上一轮 normalized chunk，并在 denoising 时应用 RTC guidance；
- 新 action chunk 到达后，客户端继续用旧 chunk 消费，按实际 capture/launch/arrival 时间丢弃过期前缀；
- RTC 模式默认不做额外客户端轨迹插值；只有显式设置 `--rtc-client-blend-steps` 才启用安全 fallback；
- 推理失败、generation 不匹配、连接断开或队列耗尽时 fail closed 并保持最后安全目标；
- `monitoring_data/<session>/events.jsonl` 和模型结果记录中包含 RTC telemetry。

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
