# RTC 客户端控制

`rtc_client.py` 是机械臂电脑上的**真实实时控制客户端**，不是 GUI 推理预览。
它直接拥有 Piper CAN、相机、Policy WebSocket 和 20 Hz 控制循环；`collect_gui.py`
只负责数据采集、回放和数据整理，不启动机器人推理或控制。

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
  --control-hz 20
```

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
  --allow-execution
```

`--allow-execution` 不是绕过安全门的开关。没有 Dashboard 授权、授权过期、
telemetry 断开或任一逐周期安全检查失败时，客户端只会保持安全目标或阻断发送。

## 控制时序

- 相机和 Piper 反馈持续运行；
- Policy 推理默认以 4 Hz 发起，单次只允许一个在途请求；
- 机器人命令由独立 20 Hz 控制循环发送；
- 新 action chunk 到达后，客户端按实际 capture/launch/arrival 时间和 actuator delay
  丢弃过期前缀，并进行短暂 old/new blend；
- 推理失败、连接断开或队列耗尽时 fail closed 并保持最后安全目标；
- 本地控制事件写入 `monitoring_data/<session>/events.jsonl`。

旧脚本仍可运行：

```bash
python robot_observation_bridge.py ...
```

它与 `rtc_client.py` 共用同一份安全检查和实时控制实现。
