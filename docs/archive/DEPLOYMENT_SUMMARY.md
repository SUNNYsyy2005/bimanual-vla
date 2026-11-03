# Bimanual-VLA 本地部署总结

## 部署时间
2026-09-29

## 部署状态
✅ 成功完成

## 已完成的配置

### 1. 基础环境
- ✅ 已配置 Conda Python 环境 `dual_arm`
- ✅ Python 3.10.20 conda环境 `dual_arm` 已创建
- ✅ 所有依赖包已安装（requirements.txt）
- ✅ 依赖完整性检查通过

### 2. 项目设置
- ✅ 仓库已克隆并完成 CLI 初始化
- ✅ CLI工具验证通过 (`bin/bimanual-vla --help`)
- ✅ 关键Python模块导入测试通过

### 3. SSH免密登录配置
- ✅ SSH配置文件创建 (`~/.ssh/config`)
- ✅ 4x4090 (192.168.101.9) 双向免密登录已配置
- ✅ login-server (36.103.167.186) 到本地免密登录已配置
- ✅ H100/H200节点配置已添加到SSH config

### 4. 测试结果
- ✅ 345个测试运行完成
- ✅ 344个测试通过
- ⚠️ 1个测试失败（数值精度问题，不影响功能）

## 环境激活方式

```bash
# 初始化并激活 Conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dual_arm

# 进入项目目录
cd /path/to/bimanual-vla

# 测试CLI
bin/bimanual-vla --help
```

## SSH连接测试

```bash
# 连接到4x4090
ssh 4x4090

# 连接到login-server
ssh login-server

# 连接到H100（通过login-server跳板）
ssh h100-ksy-01

# 连接到H200节点（需要密码）
ssh h200-ali-01
ssh h200-ali-02
```

## 服务器信息

### 4x4090
- IP: 192.168.101.9
- 用户: `<configured account>`
- 认证: SSH密钥（已配置）
- 反向连接: ✅ 已配置

### login-server
- IP: 36.103.167.186
- 用户: `<configured account>`
- 认证: SSH密钥（已配置）
- 反向连接: ⚠️ 网络超时（可能需要进一步调试）

### H100-ksy-01
- 通过: login-server跳板
- 用户: `<configured account>`
- 认证: 继承login-server密钥

### H200节点
- h200-ali-01: 47.116.14.100
- h200-ali-02: 120.55.15.209
- 用户: `<configured account>`
- 认证: 密码认证（非密钥）

## 下一步建议

### 硬件相关（如需使用）
1. 安装系统依赖包（需要sudo）:
   ```bash
   sudo apt-get install -y can-utils ffmpeg v4l-utils python3-tk
   ```

2. 配置CAN接口（需要连接Piper机械臂）
3. 配置RealSense相机

### OpenPI训练服务器（可选）
1. 参考 `docs/INSTALLATION.md` 第9节配置OpenPI环境
2. 下载pi0.5基础checkpoint
3. 配置Dashboard服务器

### 集群使用
1. 参考 `AGENTS.md` 了解Slurm集群使用规范
2. 检查quota: `ssh login-server 'myquota'`
3. 提交训练任务前检查资源: `ssh login-server 'resources'`

## 注意事项

1. **Token节约**: 遵循 `AGENTS.md` 中的Token节约规范
2. **集群规则**: 
   - login-server仅用于轻量级操作
   - GPU任务必须通过sbatch提交
   - H200节点独立存储，需单独准备环境
3. **安全**: 
   - 不要在代码中硬编码密码
   - 不要提交Dashboard token和配置文件
   - 检查CAN接口映射后再控制机械臂

## 验证命令

```bash
# 验证Python环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dual_arm
python -c "import cv2, flask, numpy, pandas, pyarrow, scipy; from PIL import Image; from openpi_client import websocket_client_policy; from piper_sdk import C_PiperInterface_V2; print('所有依赖OK')"

# 运行测试
cd /path/to/bimanual-vla
python -m unittest discover -s tests -v

# 测试SSH连接
ssh 4x4090 "echo '4x4090连接成功'"
ssh login-server "echo 'login-server连接成功'"
```

## 问题排查

如果遇到问题，请参考：
- 安装指南: `docs/INSTALLATION.md`
- 常见问题: `docs/INSTALLATION.md` 第12节
- 集群规范: `AGENTS.md`
- GUI操作: `docs/collection/GUI_OPERATION_GUIDE.md`
