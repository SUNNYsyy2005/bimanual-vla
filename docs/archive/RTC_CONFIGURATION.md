# RTC配置说明

## RTC是什么？

**RTC (Real-Time Chunking)**: 实时动作块重锚定技术
- 模型一次生成50步动作
- 每次新推理时，将新旧动作平滑融合
- 避免动作突变，提高连续性

## RTC配置位置

### 1. Model-Side RTC (模型端/服务器端)
**位置**: 在OpenPI服务器上
**功能**: 服务器在生成动作时进行RTC融合
**检查方法**: 
```python
protocol.rtc_supported  # 服务器是否支持RTC
```

### 2. Client RTC (客户端)
**位置**: 在部署客户端(bimanual_vla.deployment.client)
**功能**: 客户端接收动作后再做额外融合
**参数**:
```
--rtc-execution-horizon: 执行视界(默认8步)
--rtc-max-guidance-weight: 引导权重(默认5.0)
--rtc-prefix-attention-schedule: 注意力调度("linear")
--rtc-client-blend-steps: 客户端额外融合步数(默认0)
```

## GUI选项解析

### "Model-side RTC" 复选框
```python
self.inference_rtc_enabled_var = tk.BooleanVar(value=True)
```

**作用**: 
- ✓ **勾选**: 使用服务器的RTC功能（如果服务器支持）
- ✗ **不勾选**: 禁用Model-side RTC

**命令行参数**:
```bash
# 勾选时
--rtc-enabled

# 不勾选时
--no-rtc-enabled
```

### "Client RTC settings" 区域
这些参数**总是生效**，不管Model-side RTC是否启用：
- Execution horizon: 8
- Max guidance weight: 5.0
- Schedule: linear
- Client blend steps: 0

## 完全关闭RTC

### 问题：能否完全关闭RTC？

**答案**: 取决于"完全"的定义

#### 场景1: 关闭Model-side RTC
```
✗ 不勾选 "Model-side RTC"
→ --no-rtc-enabled
→ 服务器不做RTC融合
→ 但Client参数仍然存在
```

#### 场景2: 关闭所有RTC
```
✗ 不勾选 "Model-side RTC" 
AND 设置 Client blend steps = 0
→ --no-rtc-enabled --rtc-client-blend-steps 0
→ 服务器不做RTC
→ 客户端不做额外融合
→ 实际上基本关闭了RTC
```

## 代码实现

### GUI → 命令行参数
```python
# gui.py 构造命令
command.append("--rtc-enabled" if rtc_enabled else "--no-rtc-enabled")
command.extend((
    "--rtc-execution-horizon", str(rtc_execution_horizon),
    "--rtc-max-guidance-weight", str(rtc_max_guidance_weight),
    "--rtc-prefix-attention-schedule", rtc_prefix_attention_schedule,
    "--rtc-client-blend-steps", str(rtc_client_blend_steps),
))
```

### 客户端处理
```python
# client.py 解析参数
parser.add_argument(
    "--rtc-enabled",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="use model-side Real-Time Chunking when the Policy advertises it"
)
```

### 服务器能力检查
```python
# 从服务器元数据获取
rtc_supported = bool(metadata.get("rtc_supported", False))

# 最终决策
if args.rtc_enabled and protocol.rtc_supported:
    # 使用Model-side RTC
    use_model_rtc = True
else:
    # 不使用Model-side RTC
    use_model_rtc = False
```

## 实际行为

### 1. Model-side RTC启用 (默认)
```
GUI勾选 → --rtc-enabled → 客户端检查服务器 → 如果服务器支持 → 使用RTC
```

### 2. Model-side RTC禁用
```
GUI不勾选 → --no-rtc-enabled → 客户端忽略服务器能力 → 不使用Model-side RTC
```

### 3. Client RTC参数
无论Model-side RTC是否启用，Client参数总是被传递：
- 如果Model-side RTC启用：这些参数用于配置服务器端RTC
- 如果Model-side RTC禁用：这些参数可能用于客户端自己的逻辑

## 推荐配置

### 最佳性能（默认）
```
✓ Model-side RTC: 启用
  Execution horizon: 8
  Max guidance weight: 5.0
  Schedule: linear
  Client blend steps: 0
```

### 禁用RTC（调试用）
```
✗ Model-side RTC: 禁用
  Client blend steps: 0
```

### 最大平滑
```
✓ Model-side RTC: 启用
  Execution horizon: 12
  Max guidance weight: 8.0
  Schedule: exp
  Client blend steps: 4
```

## 常见问题

### Q1: 不勾选"Model-side RTC"能完全关闭RTC吗？
**A**: 基本可以。服务器不做RTC，客户端blend_steps=0时也不做额外融合。

### Q2: Client RTC settings有什么用？
**A**: 
- 当Model-side RTC启用时：配置服务器端RTC参数
- 当Model-side RTC禁用时：可能用于客户端fallback逻辑（需要看具体实现）

### Q3: 为什么有两层RTC？
**A**: 
- **Model-side RTC**: 服务器端，在生成动作时融合，更高效
- **Client blend**: 客户端端，接收后再融合，额外延迟但更灵活

### Q4: 默认blend_steps=0是什么意思？
**A**: 客户端不做额外融合，避免增加延迟。依赖Model-side RTC就够了。

## 总结

- **Model-side RTC**: 服务器做，勾选框控制
- **Client RTC**: 客户端做，blend_steps控制
- **完全关闭**: 不勾选Model-side RTC + blend_steps=0
- **推荐配置**: 启用Model-side RTC, blend_steps=0

---

**文档日期**: 2026-09-29
**相关文件**: 
- GUI: `bimanual_vla/collection/gui.py`
- Client: `bimanual_vla/deployment/client.py`
