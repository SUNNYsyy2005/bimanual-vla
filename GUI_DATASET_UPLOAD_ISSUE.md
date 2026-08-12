# GUI 数据集转换与上传问题说明

更新时间：2026-08-06

## 结论

本次 GUI 采集目录中实际有 20 组 episode，转换后的 LeRobot 数据集也有 20 组，没有在本地转换过程中增加数据。

上传失败发生在服务器接收完压缩包之后的校验阶段。目前确认有两个独立的服务器端问题：

1. 服务器端缺少 `export_lerobot.py`，导致结构校验脚本导入失败。
2. OpenPI 环境中的 `lerobot` 依赖没有安装（或不在该 Python 的 `sys.path` 中）；同时校验脚本需要兼容新旧 LeRobot 包路径。
3. OpenPI 环境中的 `torch` 与 `torchvision` 不兼容，可能导致 LeRobot loader 在导入阶段失败。

## 数据位置

### GUI 原始数据

本次源目录：

```text
/home/user/dual_ARM_project/arm_collect/bimanual-vla/episodes_piper_v21/joint_dataset
```

其中包含 20 个：

```text
ep_*.npz
```

### 本地 LeRobot 导出目录

GUI/上传脚本会将 LeRobot 导出放在缓存目录，而不是直接写回原始 NPZ 目录：

```text
/home/user/.cache/bimanual-vla/uploads/exports/joint_dataset-f2cb3a60de862cfb
```

主要内容包括：

```text
meta/info.json
meta/episodes.jsonl
data/chunk-000/episode_*.parquet
videos/...
raw/episode_*.npz
```

`meta/info.json` 当前记录：

```text
total_episodes: 20
total_frames: 5664
total_videos: 40
fps: 20
schema: joint
state/action: 7D
```

本地校验结果：

```text
OK LeRobot v2.1 ... episodes=20 frames=5664 ...
```

### 上传压缩包

本次上传使用的压缩包是：

```text
/home/user/.cache/bimanual-vla/uploads/joint_dataset-dd846766eb4f9cc3.tar
```

文件大小约 941 MB。

## “20 组变成 30 组”的解释

上传日志中的：

```text
29 chunks
```

不是 29 组数据，而是压缩包的传输分片数。GUI 上传脚本默认每片 32 MiB：

```text
941 MB / 32 MiB ≈ 29 个 archive parts
```

另外，GUI 当前上传模式默认是 `merge`。`merge` 的含义是：

```text
服务器已有 episode + 本次上传 episode
```

因此，如果服务器原来已有 10 组，本次成功追加 20 组后，服务器总数会显示为 30 组。这不是本地 LeRobot 导出多出了 10 组。

不过，用户提供的这次日志最后校验失败，服务器没有完成安装；这一次失败上传本身不会产生新的 30 组数据。

## OpenPI 的 7D 支持

需要特别说明：本项目使用的 OpenPI/Piper 训练代码明确支持关节 7D，并不是只能接受旧的末端位姿数据。

在 `piper_data_contract.py` 中，当前 joint v3 契约定义为：

```text
state: 6 个关节角 + 夹爪开合比例 = 7D
action: 6 个关节绝对目标 + 夹爪目标 = 7D
```

`server_4090/openpi_single_arm.py` 也明确接受：

```text
(state_dim, raw_action_dim) = (7, 7)  -> joint / single
(state_dim, raw_action_dim) = (14, 14) -> joint / bimanual
```

并且 `PiperDataConfig` 会对 joint 数据应用对应的关节动作变换。因此，新数据的 7D 关节格式本身是服务器支持的，不能因为它不是旧的 10D 末端 state 就判定为错误。

新旧数据不能直接混合的原因是动作契约不同，而不是 7D 不被支持：旧数据是 legacy delivery，当前新数据是 canonical joint。

视频字段也属于 canonical LeRobot 格式支持的媒体输入。新数据的 MP4 视频本身不是错误；本次日志中的 loader 失败发生在 `torchvision` 导入阶段，还没有读取视频内容。

## 实际报错位置

上传分片全部成功后，服务器进入 assemble 和 validation 阶段，报错：

```text
dataset structural validation failed

Traceback ...
  File "/home/sunny/bimanual-vla/check_pi05_dataset.py", line 22, in <module>
    from export_lerobot import inspect_npz_episode
ModuleNotFoundError: No module named 'export_lerobot'
```

因此问题链路是：

```text
GUI 转换成功
  -> LeRobot 导出成功（20 episodes）
  -> tar 打包成功
  -> 29 个传输分片全部上传成功
  -> 服务器结构校验启动
  -> 服务器找不到 export_lerobot.py
  -> HTTP 400，上传安装失败
```

### LeRobot loader 的第二个错误

重新上传时还出现了独立的 loader 校验错误：

```text
LeRobot/OpenPI loader validation failed

ModuleNotFoundError: No module named 'torch._custom_ops'
```

调用链是：

```text
validate_lerobot.py
  -> LeRobotDataset
  -> lerobot.common.datasets.utils
  -> torchvision
  -> torch._custom_ops
```

4×4090 的 OpenPI 环境当前使用 `torch==2.0.1`，但其中的 `torchvision` wheel 按更新版本的 torch 构建，导入时需要当前 torch 不提供的 `torch._custom_ops`。

这不是数据集内容错误。loader 尚未读取 episode，就已经在导入 Python 依赖时失败。

当前最新日志进一步确认服务器首先失败在：

```text
ModuleNotFoundError: No module named 'lerobot'
```

这表示服务器配置中的 `openpi_python`（通常是 `/home/sunny/miniconda3/envs/openpi/bin/python`）没有安装 LeRobot，或者安装位置没有加入该解释器的模块搜索路径。即使把 LeRobot 安装好，校验脚本还需要支持旧版 `lerobot.common.datasets...` 和新版 `lerobot.datasets...` 两种包路径。

## 根因

服务器部署脚本原来同步了：

```text
check_pi05_dataset.py
```

但没有同步它依赖的同目录模块：

```text
export_lerobot.py
```

此外，独立运行的 `check_pi05_dataset.py` 原先也没有显式把仓库目录加入 `sys.path`，对服务环境的 `PYTHONPATH` 有隐含依赖。

## 已完成的本地修复

本地仓库已经完成以下修改：

1. `check_pi05_dataset.py` 启动时显式加入自身仓库目录，确保可以导入同目录模块。
2. `deploy_4090_server.sh` 增加同步 `export_lerobot.py`。
3. `server_4090/validate_lerobot.py` 增加新旧 LeRobot 包路径兼容：同时支持 `lerobot.common.datasets...` 和 `lerobot.datasets...`。
4. `server_4090/validate_lerobot.py` 增加轻量 `torchvision.transforms` 兼容回退，使 loader 校验不再因 `torch._custom_ops` 缺失而在导入阶段退出。
5. 上传日志改用 `archive parts`，不再把传输分片称为 `chunks`，避免误解为 episode 数。
6. GUI 上传窗口显示：
   - source NPZ episode 数
   - prepared LeRobot episode 数
   - 当前上传模式（`merge`/`install`/`overwrite`）
   - `merge` 会追加服务器已有数据的提示
7. 本地转换和结构校验已经验证通过，当前数据仍为 20 episodes。

## 服务器管理员需要做的事

由于 4×4090 不属于当前账号，不能由本机直接部署。服务器管理员需要在服务器代码目录 `/home/sunny/bimanual-vla` 中：

1. 同步以下文件：

   ```text
   check_pi05_dataset.py
   export_lerobot.py
   server_4090/validate_lerobot.py
   ```

2. 确认以下依赖文件已经同步到服务器代码目录：

   ```bash
   ls -l /home/sunny/bimanual-vla/check_pi05_dataset.py
   ls -l /home/sunny/bimanual-vla/export_lerobot.py
   ```

3. 重启 Dashboard 服务。

4. 在服务器上做导入检查：

   ```bash
   cd /home/sunny/bimanual-vla
   python -c "from export_lerobot import inspect_npz_episode; print('export_lerobot import OK')"
   ```

5. 使用服务器配置中的 OpenPI Python 检查 LeRobot 是否可导入：

   ```bash
   /home/sunny/miniconda3/envs/openpi/bin/python -c "import sys, lerobot; print(sys.executable); print(lerobot.__file__)"
   ```

   如果这里失败，服务器管理员需要在这个**确切的 Python 环境**中安装 OpenPI 的 LeRobot 依赖，或修正 `PYTHONPATH`。随后再检查 torch/torchvision；如果仍采用当前不匹配的环境，应部署带兼容回退的新 `server_4090/validate_lerobot.py`。长期方案是安装互相匹配的 torch 和 torchvision 版本。

6. 重新上传时先确认 GUI 日志类似：

   ```text
   Episode counts: source NPZ=20, prepared LeRobot=20
   Upload mode: merge ...
   ... archive parts ...
   ```

## 上传模式建议

- `install`：服务器上不存在同名数据集时安装；同名数据集存在会报错。
- `merge`：保留服务器已有数据并追加本次数据，适合增量采集；需要注意总 episode 数会增加。
- `overwrite`：替换服务器已有同名数据集，使用前确认不需要保留旧数据。

如果目标是让服务器最终只有当前这 20 组，应选择 `overwrite`，而不是 `merge`。
