# 快速开始

从**采集工作站**开始。GPU 上的 OpenPI/Policy 服务必须使用**独立的 Python 环境**；不要把 JAX/OpenPI 依赖覆盖安装到机器人工作站中。

## 1. 支持环境

| 组件 | 推荐系统 | Python |
|---|---|---|
| 采集 GUI、数据工具、RTC 客户端 | Ubuntu 22.04 x86_64 | 3.10.20 |
| Dashboard 与 OpenPI Policy / 训练服务器 | Ubuntu 22.04 + NVIDIA GPU | 3.11.15 |
| RoboTwin/cuRobo 仿真评估 | Ubuntu 22.04 + 对应 CUDA | 视环境而定 |

macOS/Windows 不能直接代替当前 SocketCAN 的 Linux 实机控制环境。

## 2. 克隆与安装

~~~bash
git clone https://github.com/SUNNYsyy2005/bimanual-vla.git
cd bimanual-vla
conda create -n dual_arm python=3.10.20 -y
conda activate dual_arm
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
bin/bimanual-vla --help
~~~

系统包、Piper SDK 和独立 OpenPI 环境请对照 [完整安装指南](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/INSTALLATION.md)。

## 3. 连接前检查：先不让机械臂运动

~~~bash
# CAN 设备是否被 Linux 识别；验证物理左右臂对应关系
ip -br link show type can
# 相机节点和设备名
v4l2-ctl --list-devices
# 命令接口
bin/bimanual-vla data-validate --help
bin/bimanual-vla data-export --help
bin/bimanual-vla rtc-client --help
~~~

如果需要激活 CAN，参考 [[硬件与配置|Hardware-and-Configuration]]，确认 1 Mbit/s、真实映射和安全状态后再操作。

## 4. 首次启动 GUI

~~~bash
conda activate dual_arm
bash start_gui.sh
~~~

首次只检查相机预览、机械臂反馈和任务字段。开始录制、回放硬件动作、自动返回 Home 都应有现场人员监督。

## 5. 三条下一步路径

**采集路径：** [[数据采集|Data-Collection]] → [[数据合同与导出|Dataset-and-Contracts]]。

**训练路径：** 准备 LeRobot v2.1 数据 → [[训练与评测|Training-and-Evaluation]] → [[Dashboard 与多节点|Dashboard-and-Multi-Node]]。

**部署路径：** 启动 Policy → 先运行不带 --allow-execution 的客户端 → 核验 telemetry → [[实机部署与安全|Real-Robot-Deployment]]。

详细文档：[安装与依赖](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/INSTALLATION.md) · [GUI 手册](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/collection/GUI_OPERATION_GUIDE.md) · [命令入口](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/bin/bimanual-vla)。
