# 硬件与配置

## Piper 与相机

当前工作站支持单臂/双臂 Piper：每臂 6 个转动关节 + 夹爪，经 Piper SDK / SocketCAN 控制。示例硬件为一个 RealSense D435i 第三视角和每侧一个 D405 腕部 RGB 相机。训练输入目前是 **RGB**，不是 depth。

- Piper CAN 速率：**1 Mbit/s**；不要假设左、右 CAN 编号永久固定。
- 推荐相机选择器：稳定的 /dev/v4l/by-path 路径，而不是易变化的 /dev/videoN。
- 系统构成和频率详见 [README 硬件说明](https://github.com/SUNNYsyy2005/bimanual-vla#hardware-setup)。

## CAN 验证（接通之前保持执行禁用）

~~~bash
ip -br link show type can
lsusb -t
# 根据设备真实映射检查，例如
ip -s -details link show can0
timeout 2 candump -L can0
~~~

必要时按 [安装指南 CAN 章节](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/INSTALLATION.md#5-can-setup) 激活接口，双臂还需独立验证 can1。**未确认机械臂状态、左右映射或反馈帧时不要执行。**

## 代码与运行时配置

| 类型 | 设置入口 | 说明 |
|---|---|---|
| 采集 GUI / CLI | 启动参数和 GUI 设备选择 | FPS、arm mode、相机和 schema 可配置 |
| 相机设备 | 自动发现或明确设备路径 | 重新插 USB 后检查实际映射 |
| CAN | 命令行 / 设备连接配置 | 同一物理臂应绑定稳定的 USB 端口 |
| 外部 Piper SDK | BIMANUAL_VLA_CAN_ACTIVATE_SCRIPT 等环境变量 | 不必硬编码仓库绝对路径 |
| Dashboard | server_4090/config.json 及仿真配置 | 数据/权重根目录、目标节点、服务运行参数 |
| RTC | rtc-client 启动参数 | action 时钟、触发模式、限速和轨迹形状 |

注意：**仓库并非所有配置都集中在一个 YAML 中，也不是完全写死。** GUI 设备配置、CLI 参数、Dashboard JSON 和环境变量各自承担不同职责；运行前按组件核对。

## 配置参考

- [路径配置与可迁移性](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/PATH_CONFIGURATION.md)
- [相机/CAN 串口绑定](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/deployment/CAN_SERIAL_BINDING.md)
- [安装指南](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/INSTALLATION.md)
- [RTC 客户端完整参数](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/deployment/RTC_CLIENT_GUIDE.md)

**公开 Wiki 不记录内部服务器地址、API Token、Wi-Fi 凭据或现场固定 IP。**
