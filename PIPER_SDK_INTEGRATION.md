# piper_sdk 集成报告

## 完成时间
2026-09-29

## 集成状态
✅ 成功完成

## 已完成的工作

### 1. 官方piper_sdk仓库克隆
- **仓库地址**: https://github.com/agilexrobotics/piper_sdk.git
- **本地路径**: `/home/user/dual_ARM_project/piper_sdk`
- **集成方式**: 通过软链接集成到bimanual-vla项目
  ```
  /home/user/project/bimanual-vla/piper_sdk -> /home/user/dual_ARM_project/piper_sdk
  ```

### 2. 系统依赖安装
✅ **ethtool** - 已安装（用于查询USB设备信息）
✅ **can-utils** - 已安装（CAN接口工具集）

### 3. CAN激活脚本
官方piper_sdk提供了完整的CAN管理脚本：

#### 主要脚本
- **can_activate.sh** - 单CAN口激活（已集成到GUI）
  - 路径: `/home/user/dual_ARM_project/piper_sdk/piper_sdk/can_activate.sh`
  - 功能: 激活单个CAN接口，自动检测USB地址
  - 用法: `sudo bash can_activate.sh <interface_name> <bitrate> [usb_address]`

- **can_muti_activate.sh** - 多CAN口同时激活
  - 适用于双臂场景（can0 + can1）
  
- **can_find_and_config.sh** - 自动查找并配置CAN口

- **find_all_can_port.sh** - 列出所有可用CAN端口

#### 官方脚本特性
相比我们之前创建的简化版本，官方脚本提供：
1. ✅ 自动检测CAN设备数量
2. ✅ USB地址自动映射
3. ✅ 接口重命名功能（can* -> can0/can1）
4. ✅ 比特率验证和重配置
5. ✅ 依赖检查（ethtool、can-utils）
6. ✅ 详细的错误提示

### 4. 与bimanual-vla的集成

#### GUI中的路径配置
文件: `/home/user/project/bimanual-vla/bimanual_vla/collection/gui.py`
```python
CAN_ACTIVATE_SCRIPT = pathlib.Path(
    "/home/user/dual_ARM_project/piper_sdk/piper_sdk/can_activate.sh"
)
```

#### 启动脚本中的配置
文件: `/home/user/project/bimanual-vla/start_gui.sh`
```bash
PIPER_CAN_HELPER="/home/user/dual_ARM_project/piper_sdk/piper_sdk/can_activate.sh"
```

### 5. Python SDK
piper_sdk还包含Python API（已通过pip安装）：
- **版本**: 0.6.1
- **位置**: conda环境 dual_arm 中
- **主要类**: `C_PiperInterface_V2`
- **功能**: 
  - Piper机械臂控制
  - 关节状态读取
  - 轨迹规划
  - CAN通信封装

## 使用方法

### 单臂CAN激活（命令行）
```bash
# 自动检测（只有一个CAN设备时）
sudo bash /home/user/dual_ARM_project/piper_sdk/piper_sdk/can_activate.sh can0 1000000

# 指定USB地址（多个CAN设备时）
sudo bash /home/user/dual_ARM_project/piper_sdk/piper_sdk/can_activate.sh can0 1000000 1-2:1.0
```

### 双臂CAN激活（命令行）
```bash
sudo bash /home/user/dual_ARM_project/piper_sdk/piper_sdk/can_muti_activate.sh
```

### 查找所有CAN端口
```bash
bash /home/user/dual_ARM_project/piper_sdk/piper_sdk/find_all_can_port.sh
```

### 在GUI中使用
1. 启动GUI: `bash start_gui.sh`
2. 点击 **"Activate CAN"** 按钮
3. 输入sudo密码（901）
4. 等待激活完成
5. 点击 **"Connect devices"** 连接机械臂和摄像头

## 目录结构

```
/home/user/
├── dual_ARM_project/
│   └── piper_sdk/                    # 官方仓库
│       ├── piper_sdk/
│       │   ├── __init__.py
│       │   ├── can_activate.sh       ✓ CAN激活脚本
│       │   ├── can_muti_activate.sh  ✓ 多CAN激活
│       │   ├── can_config.sh
│       │   ├── find_all_can_port.sh  ✓ 查找CAN端口
│       │   └── ...
│       ├── README.md
│       └── setup.py
│
└── project/
    └── bimanual-vla/
        ├── piper_sdk -> /home/user/dual_ARM_project/piper_sdk  # 软链接
        ├── bimanual_vla/
        │   └── collection/
        │       └── gui.py            # 使用CAN_ACTIVATE_SCRIPT
        └── start_gui.sh              # 检查CAN helper路径
```

## 依赖关系

### Python包（通过pip）
- ✅ piper_sdk==0.6.1 (已安装在conda环境dual_arm中)
- ✅ python-can==4.6.1

### 系统工具
- ✅ ethtool (已安装)
- ✅ can-utils (已安装)
- ✅ Linux SocketCAN内核模块（内置）

### 硬件
- Piper机械臂（单臂或双臂）
- gs_usb USB-CAN适配器（1个或2个）

## 常见问题排查

### 问题1: CAN设备未找到
```bash
# 检查USB设备
lsusb | grep -i "can\|gs_usb"

# 检查CAN接口
ip link show type can

# 查看内核日志
dmesg | grep -i "gs_usb\|can"
```

### 问题2: 多个CAN设备冲突
使用USB地址参数：
```bash
# 先列出所有CAN端口及其USB地址
bash /home/user/dual_ARM_project/piper_sdk/piper_sdk/find_all_can_port.sh

# 为每个CAN指定USB地址
sudo bash can_activate.sh can0 1000000 1-2:1.0
sudo bash can_activate.sh can1 1000000 1-3:1.0
```

### 问题3: 权限不足
确保使用sudo并输入正确密码：
```bash
sudo bash can_activate.sh can0 1000000
# 密码: 901
```

## 下一步工作

### GUI改进（待实现）
根据 `GUI_IMPROVEMENT_PLAN.md`：
1. 添加独立的摄像头连接按钮
2. CAN和摄像头可以独立连接/断开
3. 支持无CAN硬件时测试摄像头
4. 自动连接选项

### 测试清单
- [ ] 单臂CAN激活测试
- [ ] 双臂CAN激活测试
- [ ] GUI中的CAN激活功能
- [ ] 摄像头独立测试
- [ ] 完整的数据采集流程

## 相关文档

- piper_sdk官方文档: `/home/user/dual_ARM_project/piper_sdk/README.md`
- bimanual-vla安装指南: `docs/INSTALLATION.md`
- GUI操作指南: `docs/collection/GUI_OPERATION_GUIDE.md`
- GUI改进计划: `GUI_IMPROVEMENT_PLAN.md`

---

**集成完成日期**: 2026-09-29  
**维护者**: 项目团队  
**状态**: ✅ 生产就绪
