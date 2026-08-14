# NAS 部署与使用说明

最后核验：2026-08-07

本文记录当前这套 Bimanual-VLA / RoboTwin / π0.5 项目里，NAS 的实际部署位置、可见挂载点、目录用途，以及推荐的使用方式。

核心原则：**NAS 只负责共享、传输、备份、staging，不直接拿来训练。**

---

## 1. 当前可用挂载点

| 机器 | 当前可用的 NAS 入口 | 说明 |
|---|---|---|
| 4x4090 | `/home/sunny/nas` | 已验证为 `fuse.rclone` 挂载，适合日常访问、备份、跨机传输 |
| login-server | `/DATA/NAS/GPUServer` | 集群共享 NAS 主目录；Dashboard 的 staging / 归档路径主要指向这里 |
| login-server | `/home/sunny/nas` | 同时也可见一个个人挂载 |
| H100 | `/home/sunny/nas` | 已验证可见；正式训练仍然走 Slurm，不直接从 NAS 训练 |
| H200 | 以 Dashboard / Slurm staging 为主 | 不要默认假设和其它节点共享本地磁盘；跨节点数据统一走 NAS staging |

### 现场检查命令

```bash
findmnt -T /DATA/NAS/GPUServer
findmnt -T /home/sunny/nas
df -hT /DATA/NAS/GPUServer /home/sunny/nas
ls -ld /DATA/NAS/GPUServer /home/sunny/nas
```

---

## 2. 当前 NAS 目录结构

当前在 `login-server` 上可见的主要目录是：

```text
/DATA/NAS/GPUServer/
  sunny/
    dashboard_dataset_sync/
    dashboard_probe/
    pi05_contract_v3_checkpoints/
    pi05_contract_v3_jobs/
    pb3_v3_order_aligned_bundle/
    pi05_contract_v2_checkpoints/
    pi05_contract_v2_jobs/
    pi05_contract_v2_code_20260803/
    put_single_bottle_200_stage/
    put_single_bottle_200_ckpt20000_h200/
    lift_pot_piper_finetune/
    pi05_lift_pot_ckpt/
    openpi_pi05_bundle/
    openpi_blobs/
    h200_probe/
    h200_diag/
    status/
    slurm_logs/
    setup_jobs/
    _probe/
    ...
```

### 常见用途对应

| 目录 | 用途 |
|---|---|
| `dashboard_dataset_sync` | Dashboard / Slurm 的数据集 staging 目录 |
| `dashboard_probe` | H200 节点的资源探测缓存 |
| `pi05_contract_v3_checkpoints` | π0.5 / RoboTwin checkpoint 归档、发布、备份 |
| `pi05_contract_v3_jobs` | H200 / 远端任务日志 |
| `pb3_v3_order_aligned_bundle` | 某批数据 bundle / 发布包 |
| `put_single_bottle_200_stage` | 数据或 checkpoint 的中间 staging |
| `put_single_bottle_200_ckpt20000_h200` | 某次正式 checkpoint 归档 |
| `lift_pot_piper_finetune` | 旧任务的数据或 checkpoint 备份 |
| `openpi_pi05_bundle` | OpenPI / π0.5 相关打包文件 |
| `openpi_blobs` | 模型 blob / 中间产物 |

> 这些目录是“当前可见”的典型目录，不代表以后不会新增。

---

## 3. 当前部署状态怎么理解

### 3.1 4x4090 机器

4x4090 上跑着两个 Dashboard 服务：

- `bimanual-vla-dashboard.service`，端口 `8090`
- `bimanual-vla-sim-dashboard.service`，端口 `8091`

它们本身不把 NAS 当训练盘，而是把 NAS 当作：

- 数据集上传 / 同步的 staging
- checkpoint 归档 / 转运位置
- H200 资源探测缓存的落点
- 任务日志和传输结果的共享位置

### 3.2 login-server

`login-server` 是 NAS 共享路径和 Slurm 提交流程的中心节点：

- 共享 NAS 主路径：`/DATA/NAS/GPUServer`
- 可用于查看 `sunny` 目录下的备份、staging、日志、probe 文件
- H100 / H200 的正式任务仍然通过 `login-server` 提交，不要直接在 NAS 上训练

---

## 4. 推荐使用方式

### 4.1 备份大文件时优先用 `rsync`

比 `scp` 更适合断点续传：

```bash
rsync -ah --info=progress2 --partial --append-verify \
  <src>/ \
  <dst>/
```

示例：把 4x4090 上的 checkpoint 备份到 NAS：

```bash
rsync -ah --info=progress2 --partial --append-verify \
  /home/sunny/robotwin_ws/RoboTwin/policy/pi05/checkpoints/<config>/<exp>/<step>/ \
  /home/sunny/nas/sunny/<backup_name>/
```

如果你在 `login-server` 上操作，则把目标路径换成：

```bash
/DATA/NAS/GPUServer/sunny/<backup_name>/
```

### 4.2 小文件可以用 `scp`

```bash
scp local_file 4x4090:/home/sunny/...
scp local_file login-server:/DATA/NAS/GPUServer/sunny/...
```

### 4.3 checkpoint 发布建议打包

正式 checkpoint 建议至少包含：

```text
params/
assets/
_CHECKPOINT_METADATA
```

然后再放到 NAS 的 checkpoint 归档目录里，避免中途损坏。

### 4.4 数据集同步建议走 Dashboard / staging

Dashboard 相关配置里，NAS 的默认 staging 目录是：

```text
/DATA/NAS/GPUServer/sunny/dashboard_dataset_sync
/DATA/NAS/GPUServer/sunny/dashboard_checkpoint_sync
```

H200 节点的探测缓存会从：

```text
/DATA/NAS/GPUServer/sunny/dashboard_probe/<node>_inventory.json
```

读取或刷新。

---

## 5. Dashboard / 训练系统里 NAS 的位置

### 5.1 4x4090 Dashboard

- 主服务：`8090`
- 仿真服务：`8091`
- 默认都通过本地 4x4090 的 OpenPI 环境运行
- 大文件同步、H200 资源探测、checkpoint 发布会经由 NAS staging

### 5.2 H100 / H200

- H100：正式训练优先节点，仍需 Slurm
- H200：Slurm-only，从 Dashboard 视角也要通过 NAS staging 和 CPU-only / Slurm copy job 处理
- 不要把 NAS 当作正式训练的数据盘
- 不要假设 H100/H200 和 4x4090 共用同一个本地磁盘

---

## 6. 常用命令

### 6.1 查看 NAS 挂载

```bash
findmnt -T /DATA/NAS/GPUServer
findmnt -T /home/sunny/nas
```

### 6.2 查看 NAS 根目录

```bash
ls /DATA/NAS/GPUServer
ls /DATA/NAS/GPUServer/sunny
```

### 6.3 检查目录容量或文件完整性

```bash
du -sh /DATA/NAS/GPUServer/sunny/<path>
sha256sum <file>
```

### 6.4 从 NAS 恢复到 4x4090

```bash
rsync -ah --info=progress2 --partial --append-verify \
  /DATA/NAS/GPUServer/sunny/<src>/ \
  /home/sunny/<dst>/
```

---

## 7. 注意事项

1. **不要直接从 NAS 做正式训练。** 先把数据或 checkpoint 放到对应节点的本地工作目录。
2. **H100 / H200 仍然遵守 Slurm。** NAS 只负责 staging、镜像、归档、日志。
3. **大文件优先用 `rsync --partial --append-verify`。**
4. **不要把 token、密码、SSH 私钥、sudo 密码写进文档或仓库。**
5. **如果刚写完大文件，NAS 可能有短暂延迟。** 读回前最好稍等几秒再验证。
6. **checkpoint / dataset 迁移后要检查完整性。** 常见检查包括：

```bash
test -s <checkpoint>/params/_METADATA
test -f <checkpoint>/_CHECKPOINT_METADATA
find <checkpoint>/assets -name norm_stats.json -type f -size +0c
```

---

## 8. 一句话总结

- **4x4090**：日常用 `/home/sunny/nas`
- **login-server**：正式共享路径是 `/DATA/NAS/GPUServer`
- **H100 / H200**：通过 Dashboard / Slurm + NAS staging 做传输和归档
- **NAS 的角色是共享与备份，不是训练盘**
