# 实验产物迁移记录（2026-10-23）

目标服务：仿真 Dashboard `http://192.168.101.9:8091`。规范见 [EXPERIMENT_ARTIFACT_CONTRACT.md](EXPERIMENT_ARTIFACT_CONTRACT.md)，接口见 [API_USAGE.md](../server_4090/API_USAGE.md)。

## 已完成范围

- 通用训练历史：66 条存储记录，45 条 RLinf、21 条 π0.5/OpenPI SFT；45 个原有 RLinf 运行 ID 全部保留。
- 48 个历史存储位置完成同盘物理迁移，旧绝对路径及运行内相对路径保留兼容链接。其中 2 个位置是 H200 指标镜像（mirror_only），不包含对实际 H200 权重/轨迹迁移的声明。
- 15 个旧 SFT 输出位置在迁移前已不存在；保留历史，标记 missing，不生成虚假 checkpoint。
- 3 个运行中的 RLinf 存储位置暂缓迁移；不修改运行进程的输出。
- 视频目录整理及远端索引：本次验收约 4,887 条（持续运行会增长），本地 685 条，远端 4,202 条。本地环境来源为 RoboTwin 517、LIBERO-Safety 168。仅 1 条本地视频缺少可确定的任务身份，保留 unknown。
- 保留 RoboTwin 旧任务事件与人工修订；逐 episode 的任务成功和安全成功分别登记。轨迹、summary、episode metadata 与视频一起保留。
- SFT 的动作契约复制到 `metadata/action-contract.json`，原 registry 保留；解析真实 checkpoint 路径后仍可读取。Orbax/原生权重不重新编码。checkpoint 索引的完整性证据为结构检查，未执行大型模型加载或完整权重校验，恢复能力未验证时不声明 full_state。

## 物理根目录

以下根目录之后均为 `<source>/<task>/<experiment>/<run>`：

| 存储位置 | 根目录 |
| --- | --- |
| 4×4090 NVMe 1 | `/mnt/nvme0n1p4_1T/sunny/simulation-artifacts` |
| 4×4090 NVMe 2 | `/mnt/nvme2n1p1_2T/sunny/simulation-artifacts` |
| 4×4090 HDD | `/mnt/sda1_16T/sunny/simulation-artifacts` |
| H100 / login-server 共享盘 | `/DATA/disk0/sunny/simulation-artifacts` |
| login-server 的 H200 指标镜像 | `/home/sunny/simulation-artifacts` |
| H200 实际产物（排队待执行） | H200 自己的 `/DATA/disk0/sunny/simulation-artifacts` |

聚合视频目录为 4×4090 的 `/mnt/nvme0n1p4_1T/sunny/tsq-storage/dashboard-workspace/eval_video_catalog`；运行清单默认分配根目录为该 workspace 下的 `experiment_artifacts`。聚合链接与真实产物有各自的 provenance，删除索引链接保留原视频。

## 排队中的后续迁移

| Slurm 作业 | 节点 | 依赖 | 范围 |
| --- | --- | --- | --- |
| 47228 | h100-ksy-01 | afterany:47083:47090 | 三个在运行的 low-SR PPO v3 目录 |
| 47229 | h200-ali-01 | afterany:47214:47202 | H200 online-scale 20261021 的 PPO/GRPO seed20261015 实际产物 |

两项作业均为 CPU-only、单 rank，不申请 GPU。依赖结束后还会检查当前用户进程的命令行和工作目录是否引用待迁移路径；仍有引用则退出并保留原目录，不强制迁移。结束状态不能由旧快照推定时登记 unknown。排队不代表已经迁移；以作业终态和 receipt 为准。

H100 脚本在 `/DATA/disk0/sunny/artifact-migration-20261023/h100-followup.sbatch`；H200 脚本的提交副本在同目录 `h200-followup.sbatch`，脚本内嵌小型迁移模块与计划，无需依赖 NAS 或节点共享项目目录。执行回执写入对应节点 `/DATA/disk0/sunny/artifact-migration-20261023/followup-<jobid>/applied.json`，摘要另写入 Slurm job Comment：

```bash
sacct -j 47228,47229 --format=JobID,State,ExitCode,Comment%200
scontrol show job 47229
```

H200 指标镜像继续标为 mirror_only；实际产物完成迁移后，仍须通过 Slurm/NAS 索引通道登记实际目录，不得将镜像位置改称完整产物位置。

## 审计与验证

4×4090 持久化计划与配置备份位于 workspace 下 `artifact_migration_20261023/`；H100 回执位于 `/DATA/disk0/sunny/artifact-migration-20261023/`。旧目录通过同盘 rename 移动，失败回滚，不跨盘复制大型模型。首次 Slurm 调用曾意外启动多个 rank；已使用文件锁及 `srun --ntasks=1` 修复，并完成断链/循环链接审计（444 个文件链接，0 个坏链）。后续作业 47212、47219、47222、47225 均完成。

42 项相关回归测试通过，覆盖视频分类、旧事件修订、分页、运行清单、活动任务保护、Orbax 旧路径兼容、动作契约真实路径读取、训练任务/checkpoint。线上确认 66 个通用清单 schema 合法、45 个 RLinf ID 保留；原有 6 个 SFT checkpoint、7 个可选基础/训练权重、1 个 SFT 实验均仍可见。成功同步一段 310,683 字节的远端 RLinf 视频，sidecar 存在，HTTP Range 返回 206/32 字节；通过 API 发起目录刷新成功且远端扫描无错误。未执行 GPU 训练或推理。

后台 `bimanual-vla-artifact-index.timer` 每五分钟刷新历史与视频清单。新提交会创建通用清单及原生产物引用；第三方训练 writer 的物理输出路径仍需按规范配置，这一登记本身不代表全部新输出已自动重定向。
