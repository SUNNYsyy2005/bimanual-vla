# 路径配置与可迁移性

项目启动脚本会从脚本自身位置定位仓库根目录，因此仓库可以克隆到任意目录，不需要修改代码中的项目路径。外部 SDK、Python 环境、模型和数据目录通过环境变量或 Dashboard JSON 配置。

## 常用环境变量

| 变量 | 用途 | 默认行为 |
|---|---|---|
| `BIMANUAL_VLA_PYTHON` | GUI/CLI 使用的 Python | 当前 Conda 环境、仓库 `.venv`、系统 `python3` 依次回退；GUI 若缺 Tk 会再查找主目录中的 Conda `dual_arm` 环境 |
| `BIMANUAL_VLA_BOOTSTRAP_PYTHON` | Dashboard 启动脚本读取配置时使用的 Python | 当前 Conda 环境或系统 `python3` |
| `BIMANUAL_VLA_CAN_ACTIVATE_SCRIPT` | Piper CAN 激活脚本 | `<仓库根目录>/piper_sdk/piper_sdk/can_activate.sh` |
| `REMOTE_HOST` / `REMOTE_ROOT` | Dashboard 部署目标及远端仓库目录 | `4x4090` / 远端 `$HOME/bimanual-vla` |
| `LOGIN_SERVER_USER` | 仿真 Dashboard 同步到 login-server 时使用的账号 | 通过 SSH 查询远端当前账号 |
| `LOGIN_SERVER_PROJECT_ROOT` / `LOGIN_SERVER_NAS_ROOT` | login-server 上的代码暂存目录和 NAS 根目录 | 根据账号拼接默认路径 |
| `ROBOTWIN_PROJECT` | 外部 RoboTwin checkout 路径 | 仓库旁的 `RoboTwin` 目录 |
| `BIMANUAL_VLA_OPENPI_REPO` / `BIMANUAL_VLA_PI05_CHECKPOINT` | 推理冒烟脚本使用的 OpenPI checkout 和 checkpoint | 仓库旁的 RoboTwin checkout、用户目录下的 `checkpoints` |
| `BIMANUAL_VLA_REPO` | action 统计脚本使用的 bimanual-vla 仓库路径 | 从脚本位置自动推导 |
| `BIMANUAL_VLA_MONITOR_SSH_HOST` / `BIMANUAL_VLA_CLUSTER_USER` / `BIMANUAL_VLA_SLURM_LOG_DIR` | 集群监控目标、用户和日志目录 | 使用当前登录用户及脚本内默认主机/目录 |

外部 Piper SDK checkout 示例：

```bash
export PIPER_SDK_ROOT=/path/to/piper_sdk
export BIMANUAL_VLA_CAN_ACTIVATE_SCRIPT="$PIPER_SDK_ROOT/piper_sdk/can_activate.sh"
```

## Dashboard 配置文件

复制 `server_4090/config.example.json` 或 `server_4090/config.simulation.example.json` 后，在 `config.json` 中设置模型、数据集、缓存、视频和集群目录。路径值支持 `~`、`$HOME`、`${HOME}`、`${USER}` 等环境变量写法；环境变量由 Dashboard 启动进程展开。配置中的 `/mnt`、`/DATA` 是示例部署所用的挂载点，应按目标机器的实际存储布局调整。

部署脚本会保留服务器已有的运行配置；更新样例不会覆盖已有 `config.json`。迁移到新服务器时，检查该服务器自己的配置文件及 OpenPI Python 路径。

## 操作系统路径

`/dev/video*`、`/sys/class/net` 和 `/proc` 是 Linux 相机、网络接口及进程信息接口，不是仓库安装路径。它们由设备和操作系统决定，不能替换成仓库相对路径。
