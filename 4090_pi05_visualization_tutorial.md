# 4090 服务器连接与 pi0.5 可视化环境使用教程

本文档覆盖两个内容：

1. 连接 4090 服务器并检查 GPU 状态
2. 使用已配置好的 Isaac Sim 双臂仿真与 pi0.5 输出可视化工具

当前已验证的 4090 服务器：

```text
地址: 100.124.93.40
系统: Ubuntu 22.04.5 LTS
用户: user
GPU: NVIDIA GeForce RTX 4090 D
连接方式: Tailscale SSH
```

## 1. 连接 4090 服务器

### 1.1 确认 Tailscale 连通性

这台服务器通过 Tailscale 网络访问。先确认本机 Tailscale 已连接，然后测试：

```powershell
tailscale status
ping 100.124.93.40
```

如果正常，应能看到类似：

```text
100.124.93.40   user-dev   ...   linux   active
```

### 1.2 通过 Tailscale SSH 登录

这台机器的 `22` 端口是 Tailscale SSH，不是普通密码 SSH。推荐使用：

```powershell
tailscale ssh user@100.124.93.40
```

进入服务器后，可确认当前用户和系统：

```bash
whoami
hostname
lsb_release -a
```

### 1.3 检查 GPU 是否空闲

登录后运行：

```bash
nvidia-smi
```

也可以用简洁格式查看：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
```

示例输出：

```text
0, NVIDIA GeForce RTX 4090 D, 226, 24564, 4, 46
```

含义：

```text
GPU编号, 显卡型号, 已用显存MiB, 总显存MiB, GPU利用率%, 温度C
```

如果显存占用很低、GPU 利用率接近 0%，说明基本空闲。

### 1.4 RDP 说明

当前已验证：

```text
SSH 22: 可达
RDP 3389: 不通
```

因此这台 `100.124.93.40` 目前主要通过命令行 SSH 使用。如果需要图形界面，可以后续再配置 RDP、VNC、NoMachine 或 Isaac Sim streaming。

## 2. Isaac Sim 环境

服务器上已存在 Isaac Sim 5.1.0：

```bash
/home/user/sjj_ws/isaac-sim-standalone-5.1.0-linux-x86_64
```

版本：

```text
5.1.0-rc.19+release.26219.9c81211b.gl
```

常用入口：

```bash
/home/user/sjj_ws/isaac-sim-standalone-5.1.0-linux-x86_64/isaac-sim.sh
/home/user/sjj_ws/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh
```

注意：服务器 `~/.bashrc` 里有两套 Isaac Sim alias：

```bash
isaacsim          # 指向新版 5.1.0，推荐使用
ISAACSIM          # 指向旧版 4.5.0
ISAACSIM_PYTHON   # 指向旧版 4.5.0 的 python.sh
```

做 pi0.5 仿真和可视化时，优先使用 5.1.0 路径。

## 3. 双臂仿真项目

已配置好的项目目录：

```bash
/home/user/sjj_ws/pi05_dual_arm_sim
```

项目包含：

```text
configs/dual_franka.json              # 双臂、相机、物体配置
scripts/dual_franka_tabletop.py       # Isaac Sim 双 Franka 桌面场景
scripts/run_headless.sh               # 无 GUI 启动
scripts/run_gui.sh                    # GUI 启动
scripts/visualize_pi05_output.py      # pi0.5 输出可视化
README.md
```

当前仿真环境包含：

```text
双 Franka Panda 机械臂
桌面场景
红色方块、蓝色方块、tray
front / left_wrist / right_wrist 三路相机
18维 action: 左臂9维 + 右臂9维
```

## 4. 运行双臂仿真

进入项目：

```bash
cd /home/user/sjj_ws/pi05_dual_arm_sim
```

运行 headless 仿真：

```bash
./scripts/run_headless.sh --frames 180 --capture-every 60
```

参数说明：

```text
--frames          仿真总帧数
--capture-every   每隔多少帧保存一次相机图像
--camera-warmup   相机预热帧数，默认10
```

仿真输出会写入：

```bash
/home/user/sjj_ws/pi05_dual_arm_sim/runs/<时间戳>
```

例如：

```text
runs/20260721-142226/
  episode.npz
  front_00010.png
  front_00020.png
  left_wrist_00010.png
  left_wrist_00020.png
  right_wrist_00010.png
  right_wrist_00020.png
  status.log
```

其中 `episode.npz` 包含：

```text
states:  状态序列，shape = [T, 18]
actions: action序列，shape = [T, 18]
instruction: 语言指令
```

## 5. pi0.5 输出可视化

### 5.1 生成 HTML 可视化报告

使用脚本：

```bash
cd /home/user/sjj_ws/pi05_dual_arm_sim

/home/user/sjj_ws/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  scripts/visualize_pi05_output.py \
  --input runs/20260721-142226/episode.npz
```

生成结果：

```bash
runs/20260721-142226/pi05_visualization.html
```

该 HTML 是自包含文件，可以在浏览器中直接打开。

### 5.2 可视化内容

报告包含：

```text
Action Dimensions    action各维度随时间变化曲线
Action Heatmap       左右臂action热力图
State Dimensions     state各维度随时间变化曲线
Camera Frames        同目录下相机帧缩略图
Metadata             输入文件里的元信息
```

### 5.3 支持的输入格式

当前支持：

```text
.npz
.json
.jsonl
```

脚本会自动寻找常见字段：

```text
actions
action
predicted_actions
pi05_actions
policy_actions
action_chunk

states
state
proprio
robot_state
observations
```

如果 pi0.5 的输出文件字段名不同，可以修改 `scripts/visualize_pi05_output.py` 顶部的：

```python
ACTION_KEYS = (...)
STATE_KEYS = (...)
```

### 5.4 指定左右臂 action 分割点

默认按 action 维度一半切分左右臂。例如 18 维 action 会切成：

```text
左臂: 0-8
右臂: 9-17
```

如果实际 pi0.5 输出格式不同，可以手动指定：

```bash
/home/user/sjj_ws/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  scripts/visualize_pi05_output.py \
  --input path/to/pi05_output.npz \
  --split 9
```

### 5.5 指定相机图片目录

默认从输入文件同目录读取图片。如果图片在别的目录：

```bash
/home/user/sjj_ws/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  scripts/visualize_pi05_output.py \
  --input path/to/pi05_output.npz \
  --image-dir path/to/images
```

### 5.6 指定输出 HTML 路径

```bash
/home/user/sjj_ws/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  scripts/visualize_pi05_output.py \
  --input path/to/pi05_output.npz \
  --output path/to/report.html
```

## 6. 推荐工作流

### 6.1 先跑仿真产生样例数据

```bash
cd /home/user/sjj_ws/pi05_dual_arm_sim
./scripts/run_headless.sh --frames 180 --capture-every 60
```

### 6.2 查看最新输出目录

```bash
ls -td runs/* | head
```

### 6.3 生成可视化报告

```bash
LATEST=$(ls -td runs/* | head -1)

/home/user/sjj_ws/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  scripts/visualize_pi05_output.py \
  --input "$LATEST/episode.npz"
```

### 6.4 打开 HTML 报告

如果在远程桌面中：

```bash
xdg-open "$LATEST/pi05_visualization.html"
```

如果只通过 SSH 使用，可以把 HTML 文件下载到本地再打开。

## 7. 常见问题

### 7.1 Tailscale SSH 能通，但普通 ssh 密码登录失败

这是正常现象。该服务器 `22` 端口返回的是：

```text
SSH-2.0-Tailscale
```

应使用：

```bash
tailscale ssh user@100.124.93.40
```

### 7.2 Headless 运行时看到 X Server / GLFW warning

常见警告：

```text
GLFW initialization failed
failed to open the default display
```

headless 模式下可以忽略。只要脚本最终生成 `episode.npz` 和图片即可。

### 7.3 前几帧相机为空

Isaac Sim headless 相机启动后前几帧可能返回空 buffer。脚本默认设置了：

```text
--camera-warmup 10
```

即跳过前 10 帧再保存图片。

### 7.4 如何快速确认输出有效

```bash
python3 - <<'PY'
import numpy as np
data = np.load('/home/user/sjj_ws/pi05_dual_arm_sim/runs/20260721-142226/episode.npz')
print(data['states'].shape, data['states'].dtype)
print(data['actions'].shape, data['actions'].dtype)
print(data['instruction'])
PY
```

应看到类似：

```text
(30, 18) float32
(30, 18) float32
move the red cube to the tray using both arms
```

## 8. 后续扩展建议

后续如果要真正接 pi0.5 policy，可以按这个方向扩展：

```text
1. 将 pi0.5 输出保存为 .npz/.jsonl
2. 用 visualize_pi05_output.py 先检查 action 是否连续、是否越界、左右臂是否同步
3. 增加 policy_bridge，将 pi0.5 action 映射到 Isaac Sim 双臂 joint targets
4. 增加 IK/RMPFlow，把末端 delta pose 转换为关节控制
5. 增加 episode replay，把 pi0.5 输出动作在 Isaac Sim 中回放
```

当前这套环境的重点是：先把 pi0.5 输出数据看清楚，再进入闭环控制调试。
