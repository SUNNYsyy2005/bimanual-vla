# 常见问题与排查

## 1. CAN 不工作、左右臂错位

**症状：** No such device: can0 / 无反馈 / 连接后控制异常。

检查 Linux SocketCAN 设备、适配器驱动、1 Mbit/s 波特率、左右臂物理映射与 Piper SDK。不要在 CAN 映射不明确时执行动作。

~~~bash
ip -br link show type can
ip -s -details link show can0
timeout 2 candump -L can0
~~~

参见 [安装指南](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/INSTALLATION.md#5-can-setup)。

## 2. 相机画面错误或缺帧

USB 重新插拔可能让 /dev/videoN 变化。通过 v4l2-ctl --list-devices 或稳定的 /dev/v4l/by-path 选择器核对每个 RGB 视角。Camera preview 显示正常不代表训练帧时间戳一定对齐，导出前仍应校验。

## 3. Policy schema 不匹配

典型错误：joint / delivery 不匹配、7D/10D 与 14D/20D 混淆、模型 7D/14D action 被当作 raw delivery action。

检查 dataset 合同、normalize、Policy 元数据、双臂 left+right 顺序和夹爪 0/1 约定。参阅 [[数据合同与导出|Dataset-and-Contracts]]。

## 4. 模型能推理但机械臂不动

默认 shadow 模式不会发布实际控制命令，这是设计预期。应先核实授权、命令新鲜度、状态、工作空间、动作合同、独占控制和安全门，而不是直接绕过检查或提高限制。

## 5. 抖动、延迟尖峰或目标落后

区分：模型原始动作、chunk 接收时刻、客户端 reference、Ruckig command 和真实关节反馈。检查推理 trigger（periodic/chunk_step）、action FPS、latency、chunk 边界、tracking error 及 speed/acceleration/jerk。过强 blend 会增加响应滞后，过低限制会拖慢纠偏。参阅 [RTC 指南](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/deployment/RTC_CLIENT_GUIDE.md)。

## 6. 数据无法上传或训练失败

检查 token（不要贴入公开 Issue）、目标数据类型、LeRobot v2.1 验证、GPU/Conda 环境、Slurm 配置和各节点数据/权重副本。仪表盘显示数据集并不意味着目标 GPU 机器上已存在相同路径。

## 7. 提交 Issue 时建议提供

项目 commit、系统/Python、对应命令与**去敏日志**、操作阶段（采集/转换/训练/影子推理/执行）、动作 schema、设备信息、复现步骤和预期/实际表现。切勿发布密钥、token、真实内网地址或危险运动指令。

**更多：** [完整安装指南](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/INSTALLATION.md#12-common-problems) · [GUI 故障排查](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/collection/GUI_OPERATION_GUIDE.md) · [开新 Issue](https://github.com/SUNNYsyy2005/bimanual-vla/issues/new)。
