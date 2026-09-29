# 推理功能增强 - 异步执行与滤波控制

## 更新时间
2026-09-29

## 新增功能

### 1. 异步推理控制

**功能说明**: 控制推理是否等待动作执行完成

#### GUI选项
```
☑ Asynchronous inference (launch next inference without waiting)
  └─ Sync wait steps (if async disabled): 8
```

#### 行为模式

**异步模式（默认，勾选）**:
```
t=0:    推理#1完成 → 立即启动推理#2
t=250:  推理#2完成 → 立即启动推理#3
t=500:  推理#3完成 → 立即启动推理#4

优点: 
  ✓ 高频率推理（4Hz）
  ✓ 响应快
缺点:
  ✗ 难以测量端到端延迟
  ✗ 新旧动作可能重叠
```

**同步模式（不勾选）**:
```
t=0:     推理#1完成
t=0-200: 执行8步动作
t=200:   动作执行完 → 启动推理#2
t=450:   推理#2完成
t=450-650: 执行8步
t=650:   启动推理#3

优点:
  ✓ 可测量完整延迟
  ✓ 动作块不重叠
  ✓ 更可预测的行为
缺点:
  ✗ 频率降低（取决于wait_steps）
  ✗ 响应慢
```

#### 命令行参数
```bash
# 异步（默认）
python -m bimanual_vla.deployment.client --async-inference

# 同步
python -m bimanual_vla.deployment.client --no-async-inference --sync-wait-steps 8
```

#### 使用场景

**异步推理适用于**:
- 正常操作
- 需要快速响应
- 实时控制

**同步推理适用于**:
- 性能测试
- 延迟分析
- 调试

### 2. 低通滤波和平滑控制

**功能说明**: 控制是否对动作进行平滑处理

#### GUI选项
```
☑ Enable low-pass filtering and trajectory smoothing
  ├─ Trajectory smoothing (Hz): 3.0
  └─ Gripper lowpass alpha: 0.5
```

#### 滤波效果

**启用滤波（默认，勾选）**:
```
原始动作: [1.0, 1.5, 2.0, 1.8, 2.1]
         ↓ 低通滤波 (3Hz)
平滑动作: [1.0, 1.3, 1.7, 1.8, 1.9]

优点:
  ✓ 运动平滑
  ✓ 减少震荡
  ✓ 保护机械臂
缺点:
  ✗ 引入延迟
  ✗ 响应稍慢
```

**禁用滤波（不勾选）**:
```
原始动作 = 执行动作（无修改）

优点:
  ✓ 零延迟
  ✓ 精确跟随
缺点:
  ✗ 可能有震荡
  ✗ 机械磨损
```

#### 参数说明

**Trajectory smoothing (Hz)**:
- 默认值: `3.0`
- 范围: `0.1 - 1000`
- 含义: 截止频率，越低越平滑
- 公式: `alpha = 1 / (1 + cutoff_hz / control_hz)`

**Gripper lowpass alpha**:
- 默认值: `0.5`
- 范围: `0.0 - 1.0`
- 含义: 滤波系数
  - `0.0` = 完全不动
  - `0.5` = 中等平滑
  - `1.0` = 无滤波

#### 命令行参数
```bash
# 启用滤波（默认）
python -m bimanual_vla.deployment.client \
  --trajectory-smoothing-cutoff-hz 3.0 \
  --gripper-lowpass-alpha 0.5

# 禁用滤波
python -m bimanual_vla.deployment.client \
  --trajectory-smoothing-cutoff-hz 1000.0 \
  --gripper-lowpass-alpha 1.0
```

#### 使用场景

**启用滤波适用于**:
- 正常操作
- 快速动作
- 保护硬件

**禁用滤波适用于**:
- 精密操作
- 慢速动作
- 测试原始输出

## GUI界面

### 布局

```
┌─ Execution settings ─────────────────────────┐
│ ☑ Asynchronous inference (launch next...)    │
│                                               │
│ Sync wait steps (if async disabled)    [8]   │
└───────────────────────────────────────────────┘

┌─ Filtering & smoothing ──────────────────────┐
│ ☑ Enable low-pass filtering and trajectory..│
│                                               │
│ Trajectory smoothing (Hz)  [3.0]             │
│ Gripper lowpass alpha      [0.5]             │
└───────────────────────────────────────────────┘
```

### 默认值

| 选项 | 默认值 | 说明 |
|------|--------|------|
| Async inference | ✓ 勾选 | 异步推理 |
| Sync wait steps | 8 | 同步时等待步数 |
| Lowpass filter | ✓ 勾选 | 启用滤波 |
| Trajectory smoothing | 3.0 Hz | 轨迹平滑频率 |
| Gripper lowpass alpha | 0.5 | 夹爪滤波系数 |

## 实现细节

### GUI代码 (gui.py)

**变量定义**:
```python
self.inference_async_enabled_var = tk.BooleanVar(value=True)
self.inference_sync_wait_steps_var = tk.StringVar(value="8")
self.inference_lowpass_filter_enabled_var = tk.BooleanVar(value=True)
self.inference_trajectory_smoothing_var = tk.StringVar(value="3.0")
self.inference_gripper_lowpass_alpha_var = tk.StringVar(value="0.5")
```

**命令构建**:
```python
def build_inference_bridge_command(...,
    async_inference: bool = True,
    sync_wait_steps: int = 8,
    lowpass_filter_enabled: bool = True,
    trajectory_smoothing_hz: float = 3.0,
    gripper_lowpass_alpha: float = 0.5,
):
    ...
    if not async_inference:
        command.extend(("--no-async-inference", "--sync-wait-steps", str(sync_wait_steps)))
    
    if lowpass_filter_enabled:
        command.extend((
            "--trajectory-smoothing-cutoff-hz", str(trajectory_smoothing_hz),
            "--gripper-lowpass-alpha", str(gripper_lowpass_alpha),
        ))
    else:
        # 禁用滤波：高频率+无滤波
        command.extend((
            "--trajectory-smoothing-cutoff-hz", "1000.0",
            "--gripper-lowpass-alpha", "1.0",
        ))
```

### Client代码 (client.py)

**参数定义**:
```python
parser.add_argument(
    "--async-inference",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="asynchronous inference launch",
)
parser.add_argument(
    "--sync-wait-steps",
    type=int,
    default=8,
    help="steps to wait before next inference when async is disabled",
)
```

**现有参数重用**:
```python
# 这些参数已存在，GUI通过设置值来控制行为
parser.add_argument("--trajectory-smoothing-cutoff-hz", ...)
parser.add_argument("--gripper-lowpass-alpha", ...)
```

## 配置推荐

### 场景1: 正常操作
```
✓ Async inference: 启用
  Sync wait steps: 8
✓ Lowpass filter: 启用
  Trajectory smoothing: 3.0 Hz
  Gripper lowpass alpha: 0.5
```

### 场景2: 性能测试
```
✗ Async inference: 禁用
  Sync wait steps: 16
✗ Lowpass filter: 禁用
  (测试原始模型输出)
```

### 场景3: 高精度操作
```
✓ Async inference: 启用
  Sync wait steps: 8
✓ Lowpass filter: 启用
  Trajectory smoothing: 5.0 Hz (更平滑)
  Gripper lowpass alpha: 0.3 (更平滑)
```

### 场景4: 快速响应
```
✓ Async inference: 启用
  Sync wait steps: 8
✓ Lowpass filter: 启用
  Trajectory smoothing: 1.5 Hz (更少延迟)
  Gripper lowpass alpha: 0.7 (更少延迟)
```

## 注意事项

### 异步推理
1. **延迟测量**: 异步模式下难以测量端到端延迟
2. **动作重叠**: 新旧动作可能同时执行
3. **调试困难**: 问题更难定位

### 滤波控制
1. **硬件保护**: 高速操作建议启用滤波
2. **延迟权衡**: 滤波会引入小延迟（~10-50ms）
3. **参数调优**: 根据任务特点调整参数

### Sync wait steps
1. **值太小**: 频繁推理，可能重叠
2. **值太大**: 频率降低，响应慢
3. **推荐值**: 8-16步（取决于动作周期）

## 兼容性

- ✅ 向后兼容（默认值保持现有行为）
- ✅ 不影响采集模式
- ✅ 可独立启用/禁用
- ✅ 参数保存在GUI preferences

## 未来扩展

可考虑添加：
1. **动态调整**: 根据负载自动调整sync_wait_steps
2. **自适应滤波**: 根据速度自动调整滤波强度
3. **性能监控**: 显示实际推理频率和延迟
4. **预设配置**: 快速切换不同场景

---

**更新日期**: 2026-09-29  
**修改文件**: 
- `bimanual_vla/collection/gui.py` - GUI界面和命令构建
- `bimanual_vla/deployment/client.py` - 参数定义
**状态**: ✅ GUI部分完成，等待client.py实现同步模式逻辑
