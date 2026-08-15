# 实机 RTC（Real-Time Chunking）部署指南

这里的 RTC 指 **Real-Time Chunking**，不是单纯的实时控制客户端。
它用于补偿相机采集、网络传输和模型推理造成的 action chunk 延迟：

- 服务端在 **flow-matching denoising 内部**读取上一 chunk 尚未执行的 normalized prefix；
- 用 prefix guidance 让新 chunk 的前缀连续地贴近上一 chunk；
- 客户端只上传 session、generation、已执行 offset 和 latency 估计，不上传或插值 normalized action；
- 客户端仍负责 20 Hz 安全执行、队列消费和必要的 fail-closed 保护。

服务端实现位于 `rtc_openpi.py`，同时支持 OpenPI 的 PyTorch `model.safetensors`
和 JAX/Orbax `params` checkpoint。Dashboard 创建 Policy 时默认带上
`--rtc-enabled`；也可以在命令行显式关闭。

## Shadow 模式

默认只读反馈、采集相机并请求 Policy，不向机械臂发送动作：

```bash
cd /home/user/dual_ARM_project/arm_collect/bimanual-vla
python rtc_client.py \
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
python rtc_client.py \
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
- Policy 推理默认以 4 Hz 发起，单次只允许一个在途请求；
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

旧的 `robot_observation_bridge.py` 仍可运行，但新实机部署统一使用
`rtc_client.py`；它们共用同一份安全检查和实时控制实现。
