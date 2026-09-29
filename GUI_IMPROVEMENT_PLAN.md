# GUI改进方案 - 独立的摄像头和CAN连接控制

## 当前问题
- "Connect devices"按钮同时连接CAN和摄像头
- 无法单独测试摄像头
- 没有CAN硬件时无法使用摄像头预览

## 改进方案

### 1. 添加独立的连接按钮

```
当前布局：
[Activate CAN] [Connect devices] [Start episode]
[Stop episode] [Swap wrist]     [Refresh files]

改进后布局：
[Activate CAN] [Connect CAN]    [Connect Cameras] [Start episode]
[Stop episode] [Swap wrist]     [Refresh files]   [Exit]
```

### 2. 连接状态管理

```python
self.can_connected = False      # CAN连接状态
self.cameras_connected = False  # 摄像头连接状态
```

### 3. 功能拆分

#### 当前 `toggle_connection()`:
- 同时连接CAN + 摄像头
- 全有或全无

#### 改进后:
- `connect_can()` - 仅连接CAN口
- `connect_cameras()` - 仅连接摄像头
- `connect_all()` - 连接所有设备（保留快捷方式）
- `disconnect_can()` - 断开CAN
- `disconnect_cameras()` - 断开摄像头
- `disconnect_all()` - 断开所有

### 4. 按钮状态逻辑

```
初始状态:
- [Activate CAN] - 启用
- [Connect CAN] - 禁用（需要先激活）
- [Connect Cameras] - 启用
- [Start episode] - 禁用

CAN激活后:
- [Activate CAN] - 禁用
- [Connect CAN] - 启用
- [Connect Cameras] - 启用
- [Start episode] - 禁用

仅摄像头连接:
- [Connect Cameras] -> [Disconnect Cameras]
- 可以预览画面
- [Start episode] - 禁用（需要CAN）

仅CAN连接:
- [Connect CAN] -> [Disconnect CAN]
- [Start episode] - 禁用（需要摄像头）

全部连接:
- [Connect CAN] -> [Disconnect CAN]
- [Connect Cameras] -> [Disconnect Cameras]
- [Start episode] - 启用
```

### 5. 自动连接选项

添加配置选项：
```python
self.auto_connect_cameras = tk.BooleanVar(value=True)
self.auto_connect_can = tk.BooleanVar(value=False)  # CAN需要手动，因为需要sudo
```

GUI启动时：
- 如果`auto_connect_cameras=True`，自动连接摄像头
- CAN仍需手动激活和连接

### 6. 实现要点

#### A. 拆分CollectionSession

```python
# 当前: session.connect() 同时连接所有
# 改进: 
session.connect_can()      # 仅连接CAN
session.connect_cameras()  # 仅连接摄像头
```

#### B. 独立的设备状态检查

```python
def can_ready(self) -> bool:
    return self.piper is not None and self.piper.is_connected()

def cameras_ready(self) -> bool:
    return self.cameras is not None and all(cam.is_opened() for cam in self.cameras.values())

def devices_ready(self) -> bool:
    return self.can_ready() and self.cameras_ready()
```

#### C. 更新start_episode条件

```python
def start_episode(self):
    if not self.devices_ready():
        messagebox.showwarning(
            "Cannot start episode",
            "Both CAN and cameras must be connected"
        )
        return
    # ... 继续录制
```

## 实施步骤

### 第一阶段：最小改动（快速实现）
1. 在当前"Connect devices"旁边添加"Connect Cameras Only"按钮
2. 该按钮跳过CAN初始化，仅连接摄像头
3. 保留原有的"Connect devices"按钮用于完整连接

### 第二阶段：完整重构
1. 修改CollectionSession，拆分connect()方法
2. 更新GUI布局，添加独立按钮
3. 实现自动连接选项
4. 更新状态管理逻辑

## 建议

**建议先实施第一阶段**，因为：
- 改动最小，风险低
- 可以快速验证摄像头功能
- 不影响现有工作流程
- 后续可以逐步完善

要我现在实现第一阶段的快速方案吗？
