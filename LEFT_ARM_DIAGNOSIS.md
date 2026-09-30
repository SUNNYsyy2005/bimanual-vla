# 左臂CAN模式切换失败诊断报告

## 问题描述
2026-09-29

左臂（can0）无法切换到CAN控制模式：
- **当前状态**: ctrl_mode=STANDBY(0x0)
- **期望状态**: ctrl_mode=CAN_CTRL(0x1)
- **测试结果**: 20次ModeCtrl命令均无响应
- **右臂状态**: 正常，可以切换到CAN模式

## 测试详情

### 测试方法
```python
piper.EnablePiper()  # 返回True，使能成功
for i in range(20):
    piper.ModeCtrl(0x01, 0x01, 10, 0x00)  # 发送CAN模式命令
    time.sleep(0.1)
    # ctrl_mode始终=0，没有变化
```

### 对比：右臂
- 右臂可以正常切换到CAN模式
- 说明SDK代码和网络通信是正常的
- 问题出在左臂本身

## 可能的原因

### 1. 示教器连接 ⭐ 最可能
**症状**: 
- 示教器连接时会占用机械臂控制权
- SDK无法切换控制模式

**检查方法**:
```bash
# 查看是否有示教器连接的迹象
# 检查机械臂是否有屏幕/按钮显示
```

**解决方法**:
- 断开示教器连接
- 如果是无线示教器，关闭示教器电源

### 2. 其他程序占用
**症状**:
- 另一个客户端正在控制左臂
- 可能是之前崩溃的程序残留

**检查方法**:
```bash
# 查看CAN总线流量
sudo candump can0 -n 100
```

**解决方法**:
```bash
# 重启CAN接口
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
```

### 3. 机械臂需要复位
**症状**:
- 机械臂处于特殊状态
- 需要硬件复位

**解决方法**:
- 方法1: 重启机械臂电源（断电10秒后再上电）
- 方法2: 使用示教器复位
- 方法3: 发送复位命令

### 4. 固件/硬件故障
**症状**:
- 左臂固件损坏
- CAN通信硬件问题

**检查方法**:
```bash
# 查看CAN反馈
sudo candump can0 -n 20
# 如果有数据包，说明通信正常
```

**解决方法**:
- 联系厂商更新固件
- 检查CAN线缆连接
- 更换CAN适配器

## 诊断步骤

### 步骤1: 检查示教器
```
[ ] 查看左臂是否连接了示教器
[ ] 如果有，断开示教器
[ ] 重新测试
```

### 步骤2: 检查CAN通信
```bash
# 查看CAN数据包
sudo candump can0 -n 50

# 应该看到类似的数据:
#   can0  2A6   [8]  FF FF E5 90 FF FF F2 CA
#   can0  2A7   [8]  00 00 3F 7E 00 00 0F E3
# 如果没有数据或数据异常，说明通信有问题
```

### 步骤3: 重启左臂
```
[ ] 断开左臂电源
[ ] 等待10秒
[ ] 重新上电
[ ] 重新测试
```

### 步骤4: 检查错误代码
```python
status = piper.GetArmStatus()
feedback = status.arm_status
print(f"err_code: {feedback.err_code}")

# 如果err_code != 0，说明有错误
```

### 步骤5: 对比左右臂
```bash
# 左臂
sudo candump can0 -n 20 &
python test_left_arm_mode.py

# 右臂  
sudo candump can1 -n 20 &
python test_right_arm_mode.py

# 比较CAN数据包的差异
```

## 临时解决方案

### 方案1: 只使用右臂进行推理
如果左臂问题一直无法解决，可以暂时：
- 配置为单臂模式
- 只用右臂测试推理流程
- 等左臂修复后再切换回双臂

### 方案2: 增加enable超时时间
虽然左臂无法提前切换到CAN模式，但enable流程本身会尝试切换：

修改`bimanual_vla/deployment/client.py`:
```python
enable_timeout_s = float(getattr(self.args, "enable_timeout_s", 10.0))  # 改为10秒
```

但这只是延长等待时间，不能根本解决问题。

## 推荐操作

**立即尝试**:
1. ✅ 断开示教器（如果有）
2. ✅ 重启左臂电源
3. ✅ 重新测试

**如果还不行**:
4. 📞 联系机械臂厂商技术支持
5. 🔍 提供详细的错误日志
6. 🛠️ 可能需要固件更新或硬件检修

## 测试命令

```bash
cd /path/to/bimanual-vla

# 重新测试左臂
conda activate <your-environment>
python test_left_arm_mode.py

# 如果成功，ctrl_mode应该变为 CAN_CTRL(0x1)
```

## 相关日志

错误日志显示:
```
left Piper enable timed out after 3.0s
ctrl_mode: 0 (STANDBY)
```

这与我们的测试结果一致：左臂卡在STANDBY模式。

---

**诊断时间**: 2026-09-29  
**问题**: 左臂无法响应ModeCtrl命令  
**建议**: 首先断开示教器并重启左臂电源
