# bimanual-vla

Piper 单臂/双臂数据采集、数据整理、OpenPI 训练管理和实机部署工具。

项目围绕四条主流程组织：

1. **采集**：Piper 主从遥操作或输出臂反馈，配合 2/3 路 RGB 相机记录 episode。
2. **GUI**：连接设备、采集、回放、检查、转换和上传数据。
3. **训练平台**：4×4090 Dashboard 管理数据集、norm、训练、checkpoint 和评测。
4. **模型部署**：Dashboard 启动 OpenPI Policy，机器人端 RTC 客户端负责观测、推理和安全执行。

统一命令入口会优先使用本机 `dual_arm` Conda 环境，也可以通过
`BIMANUAL_VLA_PYTHON=/path/to/python` 显式指定解释器。

## 快速开始

### 启动采集 GUI

```bash
bash start_gui.sh
```

GUI 是日常采集的推荐入口，支持单臂/双臂、`joint`/`delivery` schema、相机预览、
episode 回放、数据检查、LeRobot 转换和 Dashboard 上传。

详细操作见 [GUI 操作手册](docs/collection/GUI_OPERATION_GUIDE.md)。

### 命令行采集

双臂主从遥操作：

```bash
bin/bimanual-vla teleop-bimanual --record --schema joint
```

单臂主从遥操作：

```bash
bin/bimanual-vla teleop-single --arm-side right --record --schema joint
```

只读取输出臂反馈：

```bash
bin/bimanual-vla collect-output \
  --arm-mode single \
  --arm-side right \
  --schema joint \
  --can can0 \
  --task-name pick_cube \
  --instruction "pick up the cube"
```

硬件映射、按键和采集参数见
[数据采集指南](docs/collection/DATA_COLLECTION_GUIDE.md)。

### 检查和导出数据

```bash
bin/bimanual-vla data-validate --input-dir episodes_piper_v21 --target-fps 20

bin/bimanual-vla data-export \
  --input-dir episodes_piper_v21 \
  --repo-id piper/piper_v1 \
  --root piper/piper_v1 \
  --fps 20
```

上传到训练服务器：

```bash
bin/bimanual-vla data-upload piper/piper_v1 \
  --dataset-origin real
```

数据字段和动作语义以
[Piper 数据合同](docs/collection/PIPER_DATA_CONTRACT.md) 为准。

### 启动训练和部署平台

将 Dashboard 部署到 4×4090：

```bash
bash deploy_4090_server.sh
```

默认地址为 `http://192.168.101.9:8090`。Dashboard 提供：

- 数据集浏览、编辑、合并和校验；
- norm、训练和 held-out loss 任务；
- checkpoint 与 Policy 生命周期管理；
- 实时遥测和评测视频；
- H100/H200 Slurm 资源与任务接入。

服务端说明见 [Dashboard README](server_4090/README.md)，API 见
[API 使用文档](server_4090/API_USAGE.md)。

下载 OpenPI 基座权重：

```bash
python -m scripts.models.download_openpi_checkpoint \
  --checkpoint gs://openpi-assets/checkpoints/pi05_base \
  --source auto \
  --workers 16 \
  --chunks-per-file 16
```

### 运行实机 Policy 客户端

先使用 shadow 模式检查观测和模型输出：

```bash
bin/bimanual-vla rtc-client \
  --arm-mode single \
  --arm-side right \
  --instruction "pick up the cube"
```

机器人运动采用双重授权：本地必须显式加入 `--allow-execution`，Dashboard 也必须对
同一 Policy 发出未过期的 EXECUTE 授权。RTC、时间戳、录制和安全约束见
[RTC 客户端指南](docs/deployment/RTC_CLIENT_GUIDE.md)。

`bin/bimanual-vla legacy-bridge` 仅保留为旧命令兼容入口，实际控制实现只有
`bimanual_vla/deployment/client.py` 一份。

## 项目结构

```text
.
├── bin/
│   └── bimanual-vla        # 统一命令入口
├── bimanual_vla/
│   ├── collection/         # GUI、相机、遥操作、机械臂和采集会话
│   ├── data/               # 数据合同、校验、导出、上传和回放
│   └── deployment/         # RTC 客户端、RTC Policy 和部署录制
├── server_4090/            # Dashboard 后端、前端和 Policy 服务
├── docs/                   # 采集与部署专题文档
├── scripts/                # smoke test、维护、分析和模型工具
├── jobs/                   # 集群分析任务
└── tests/                  # 自动化测试
```

根目录不再放置 Python 源文件，只保留 README 和兼容的 Shell 入口。一次性迁移
工具位于 `scripts/maintenance/`，硬件/模型 smoke test 位于 `scripts/smoke/`。

## 数据约定

- 双臂向量固定按 `left + right` 拼接。
- 单臂/双臂 `joint` state/action 分别为 7D/14D。
- 单臂/双臂 `delivery` 原始 state/action 分别为 10D/20D。
- `joint` 每臂为 6 个关节角和夹爪开度；`delivery` 每臂为 xyz、rotation-6D 和夹爪开度。
- 夹爪统一使用 `0=闭合、1=张开`。
- 默认采集与控制频率为 20 Hz，异步模型请求约为 4 Hz。

完整动作设计见
[π0.5 Piper 动作设计](docs/collection/PI05_PIPER_7D_10D_DATA_ACTION_DESIGN.md)。

## 测试

在包含项目依赖的 Conda 环境中运行：

```bash
python -m unittest discover -s tests -v
```

仅检查 Python 语法和模块布局：

```bash
python -m compileall -q -x 'deployment_runs|episodes_piper_v21|lerobot_datasets|monitoring_data' .
```

硬件和 Policy smoke test：

```bash
python -m scripts.smoke.robot_smoke_test
python -m scripts.smoke.policy_server_smoke_test --shadow --steps 10
python -m scripts.smoke.inference_smoke_test
```

## 运行数据

以下目录是本地运行数据，不属于源码，已由 `.gitignore` 排除：

- `episodes_piper_v21/`
- `lerobot_datasets/`
- `deployment_runs/`
- `monitoring_data/*`

结构整理不会删除这些目录。需要释放磁盘空间时，应先确认数据已上传或备份，再单独清理。

## 更多文档

- [GUI 操作手册](docs/collection/GUI_OPERATION_GUIDE.md)
- [数据采集指南](docs/collection/DATA_COLLECTION_GUIDE.md)
- [Piper 数据合同](docs/collection/PIPER_DATA_CONTRACT.md)
- [RTC 客户端指南](docs/deployment/RTC_CLIENT_GUIDE.md)
- [服务器路径与训练评测](docs/deployment/SERVER_PATHS_ENV_TRAIN_EVAL.md)
- [NAS 部署与使用](docs/deployment/NAS_DEPLOYMENT_AND_USAGE.md)
- [Dashboard 架构与运维](server_4090/README.md)
- [仿真 Dashboard](server_4090/SIMULATION_DASHBOARD.md)
