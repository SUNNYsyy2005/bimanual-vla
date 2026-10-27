# 实验、训练与 rollout 产物约束（v1）

适用范围：仿真 Dashboard 的 π0.5 / OpenPI SFT、RLinf 训练、训练内采样、独立评测，以及今后接入的其他训练后端。新的后端必须遵守本约定；Dashboard 通过通用清单读取产物，不应继续为每种后端猜测目录和文件名。

## 1. 身份与目录

统一运行根目录：

```text
<ARTIFACT_ROOT>/<source>/<task>/<experiment_id>/<run_id>/
```

`ARTIFACT_ROOT` 可以因主机、磁盘不同而变化，下面四级身份必须一致：

| 字段 | 含义与约束 | 示例 |
| --- | --- | --- |
| source | 环境/基准来源，不能用机器名或算法代替 | `robotwin`、`libero-safety`、`real-world` |
| task | 稳定任务 ID，不能使用实验名、数据集版本或 checkpoint 名 | `handover-mic`、`tsa-moka`、`tsa-microwave` |
| experiment_id | 一组有明确研究目的、可比较的实验；模型、算法、干预记录在配置中 | `ppo-lag-low-sr-random-20261023` |
| run_id | 一次独立执行/恢复尝试的唯一 ID，不允许覆盖已有运行 | `train-20261023-173000-a1b2c3d4` |

ID 使用小写字母、数字、点、下划线和横线；单字段不超过 120 字符，禁止空格、斜杠、`..` 和绝对路径。新任务统一使用横线；`tsa_moka` 作为 `tsa-moka` 的旧别名。原任务名保留在 `raw_task`，不能通过字符串替换合并两个语义不同的任务。

`source` 与 `backend`、`kind`、`execution_target` 分开：例如来源为 `libero-safety`，后端为 `rlinf`，训练类型为 `rl`，执行位置为 `h100`。视频 API 的 `video_source` 是同一个环境来源；原有 `source` 字段仍表示存储/同步来源，以兼容已有客户端。

多任务训练使用 `task=multi-task` 并声明 `task_ids`；每个 episode 必须写真实任务 ID。必要时为各任务建立同一实验、同一 run ID 的视图，使用 `training_run_ref` 引用共享训练运行，不能复制或伪造独立 checkpoint。

## 2. 运行内的固定结构

```text
<source>/<task>/<experiment_id>/<run_id>/
├── run.json                         # experiment_run.v1，唯一运行清单
├── metadata/
│   ├── submission.json              # Dashboard 提交参数
│   ├── resolved-config.json         # 所有默认值/覆盖值生效后的配置
│   ├── environment.json             # Git commit、依赖、环境/任务版本
│   └── resume.json                  # 恢复来源与训练进度（存在恢复时）
├── logs/                            # stdout、stderr、Slurm、worker 日志
├── metrics/
│   ├── training.jsonl
│   └── evaluation.jsonl
├── checkpoints/
│   ├── index.json                   # checkpoint_index.v1
│   └── <checkpoint_id>/             # 原生格式允许保留子目录
├── train/<checkpoint_id>/<outcome>/<episode_id>/
├── eval/<checkpoint_id>/<outcome>/<episode_id>/
└── artifacts/                       # 未标准化的后端原生产物、诊断和调试附件
```

`outcome` 只能为 `success`、`failed`、`unknown`。配置、日志、模型权重和跨回合指标不能放在某个 outcome 下面；不能把整次训练归类成“任务成功”。`train` 与 `eval` 表示采样阶段，不表示数据划分；`split` 必须另外记录为 train / validation / test 等。

`checkpoint_id` 应来自实际使用的策略版本，如 `step-15000`、`round-024`。训练 rollout 在第 25 轮产生并不意味着使用了第 25 轮更新后的权重：分别记录 `rollout_round` 和策略 `checkpoint_id`，无法追溯策略版本时使用 `unknown`。

每个 episode 中至少包含：

```text
episode.json          # rollout_episode.v1
summary.json          # 后端逐回合结果，原样保留
metadata.json         # 环境/任务/seed/reset 等原始证据（若原后端提供）
review.mp4            # 或其他视频名；不强制录制所有 episode
review.mp4.metadata.json
trajectory.*          # 实际轨迹或明确的 archive/reference 文件
_event_overrides/     # 人工事件标注（可选）
```

一个 tiled/batched 视频可能跨多个环境或回合，必须标注 `episode_scope=unverified` 或明确的 batch 范围；没有逐回合对应关系时结果为未知。

视频页面的聚合目录是索引视图，允许与运行产物分盘：

```text
<VIDEO_CATALOG_ROOT>/<source>/<task>/<experiment>/<run>/<train|eval|unknown>/<checkpoint>/<outcome>/<video>
```

其中视频文件名附带稳定来源哈希，避免不同 worker、seed 或原始目录中的同名视频互相覆盖。视频 sidecar 必须指向完整 episode 的来源，不能只留下孤立 MP4。

## 3. 通用运行清单

```json
{
  "schema": "experiment_run.v1",
  "source": "libero-safety",
  "video_source": "libero-safety",
  "task": "tsa-moka",
  "experiment": "ppo-lag-low-sr-random-20261023",
  "run_id": "train-20261023-173000-a1b2c3d4",
  "backend": "rlinf",
  "kind": "rl",
  "state": "completed",
  "execution_target": "h100",
  "origin": {"task_id": "train-...", "job_id": "12345"},
  "artifacts": {
    "metadata": "metadata",
    "logs": "logs",
    "metrics": "metrics",
    "checkpoints": "checkpoints",
    "train": "train",
    "eval": "eval"
  }
}
```

`kind` 例如 `sft`、`rl`、`eval`、`distillation`；`backend` 例如 `openpi`、`rlinf` 或未来的新名称。新增 kind/backend 不应要求改写 Dashboard 的目录解析器。字段必须明确；旧记录无法确认的内容使用 `unknown`，不可推断为默认成功、默认有模型或默认已完成。

`state` 描述执行状态（starting/running/completed/failed/stopped/lost/unknown 等）；它与 episode 的任务结果独立。失败运行、没有 checkpoint 的运行、恢复前的旧尝试同样需要保留清单。

身份字段创建后不得修改。重新执行或恢复到不同配置时创建新的 `run_id`，在 `metadata/resume.json` 中记录父 run、完整 checkpoint、恢复的 optimizer/scheduler/global-step 状态。不可只加载权重却声明完整状态恢复。

## 4. checkpoint 索引与训练历史兼容

`checkpoints/index.json`：

```json
{
  "schema": "checkpoint_index.v1",
  "checkpoints": [
    {
      "id": "step-15000",
      "path": "checkpoints/step-15000",
      "format": "orbax",
      "complete": true,
      "step": 15000,
      "restore_capabilities": ["full_state"]
    }
  ]
}
```

索引中 path 使用相对 run 根目录的路径。大型权重保留原生格式；禁止为“统一”而重新编码或复制权重。写入过程中标记 incomplete；完整写入、校验后原子更新索引。仅有目录名、非空文件或归档说明不能证明 checkpoint 完整。

当前两类历史的兼容要求：

| 历史类型 | 必须保留 | 禁止的误归类 |
| --- | --- | --- |
| π0.5 / OpenPI SFT | `config/experiment/step` 旧路径别名、Orbax params/optimizer/metadata、action contract、dataset/norm/split、finetuning、heldout loss、任务日志 | 把 loss 下降当任务成功；把 weights-only 当 full-state；丢弃训练失败记录 |
| RLinf / 原生安全 RL | actor/critic/optimizer/dual 等恢复状态、轮次曲线、采样配置、逐回合 summary/trajectory/video、环境与场景证据、评测 cohort | 把整轮成功率赋给单条视频；混合 train/eval 或 fixed validation/test；把镜像指标当完整模型归档 |

旧格式转换只限建立清单、索引、移动目录和建立兼容链接。原数据文件保持不变。迁移 receipt 必须记录旧路径、物理旧路径、新路径、Dashboard 运行 ID、迁移范围和统计。原始 H200 运行与 login-server 上的指标镜像是两个不同存储副本，必须标注 `scope=mirror_only`，不能声称镜像已经包含轨迹或权重。

## 5. episode 结果与事件

每个 episode 至少记录：运行身份、真实任务、phase、split、episode ID、worker/env ID、seed/reset/scene、checkpoint、rollout round、完成状态、任务结果、安全结果、结果证据和产物相对路径。

- `success`：任务目标是否达到；原生布尔值或明确的 success / failed / unknown。
- `safe_success`：满足任务目标且满足该实验声明的安全规则。不能覆盖 `success`。
- `status`：运行是否正常完成、超时、策略错误或初始化失败。
- `counted_in_success_rate`：是否属于统计分母；被排除的初始化失败仍保留证据。
- 事件时间线：单独保存版本、标签、step/frame、current_event、max_event_reached 和人工 edits。

来源优先级：已确认的逐视频 sidecar → 明确对应的逐回合 summary/JSONL → 结果目录。禁止用 run-level `_result.txt`、平均 score、文件路径中的任意 `success` 子串或 `bool("false")` 来判断单个 episode。

SFT 的 heldout loss 是指标，不是仿真任务的成功状态。没有 rollout 的 SFT 运行不需要虚构 train/eval 视频目录。

## 6. 指标格式

新 writer 输出 `metrics/training.jsonl`、`metrics/evaluation.jsonl`，每行一个观测点：

```json
{"schema":"experiment_metric.v1","time":"2026-10-23T17:30:00+08:00","axis":"train_step","step":15000,"metrics":{"loss":0.012,"learning_rate":0.00001},"units":{"loss":"scalar"}}
```

轴可以是 train_step / actor_update / rollout_round / environment_step，必须声明，不能把不同轴都当作 step 合并。评测点另含 policy checkpoint、split/cohort、seed、episodes、有效分母和统计口径。训练成功率、评测成功率、安全成功率分别命名；保留原始指标名供审计。

旧曲线允许继续保留 `training_curve.json`、`evaluation_curve.json`，迁移时放入 metrics 并保留旧位置别名。通用运行清单允许引用这些旧文件；新后端应直接输出规范指标。

## 7. 新实验接入流程

1. 提交前确认 source、task、experiment_id；生成唯一 run_id，选择该执行节点上的 ARTIFACT_ROOT。
2. 使用 `experiment_artifacts.identity()` / `run_directory()` / `manifest()` 创建清单，或通过 Dashboard 的 `POST /api/artifact-runs` 分配本地运行目录。
3. Dashboard 的两类训练提交都接受 `artifact_identity`；未提供的旧客户端仍能提交，无法确认的字段登记为 unknown。每个训练 task 自动登记 `experiment_run.v1`，关联原生输出路径和日志。
4. 将后端配置中的 output/checkpoint/log/video 路径显式指向相应目录。暂时无法更改的第三方 writer 放在 `artifacts/native`，并在清单中声明原生路径；结束后规范化，禁止让 Dashboard 依靠新文件名猜测格式。
5. 在 episode 结束时写完整 summary/episode sidecar，视频完成后原子发布，再建立视频索引。新运行必须声明训练运行与评测运行之间的关系。
6. 更新 checkpoint 索引和运行状态；中断恢复必须保留已完成产物。结束后检查通用 API 能列出运行、checkpoint 和视频。

示例（本地目录分配，Token 从受保护的 server.env 在运行时取得）：

```http
POST /api/artifact-runs
Content-Type: application/json

{
  "video_source": "robotwin",
  "task": "handover-mic",
  "experiment": "pi05-sft-camera-v2",
  "run_id": "train-20261023-173000-a1b2c3d4",
  "backend": "openpi",
  "kind": "sft"
}
```

该接口分配本地目录，不代表已经提交训练或已在独立 H200 节点创建目录。跨节点 staging、环境检查和 Slurm 提交仍遵循 AGENTS.md。

## 8. 历史迁移与运行边界

工具：`server_4090/prepare_artifact_migration.py` 生成 per-host 计划；`experiment_artifacts.py` 预览或应用；`eval_video_catalog_runner.py` 建立聚合视频视图。

```bash
# 先在对应 Conda 环境执行预览；plan 由只读 Dashboard 清单生成
python server_4090/experiment_artifacts.py --plan plan.json --report dry-run.json
python server_4090/experiment_artifacts.py --plan plan.json --report applied.json --apply

# 视频归类默认只预览；--apply 才创建索引链接
python server_4090/eval_video_catalog_runner.py --config config.simulation.json --output video-plan.json
python server_4090/eval_video_catalog_runner.py --config config.simulation.json --apply
```

实际迁移遵循：

- 运行中、等待中、状态不可用的任务不移动；先登记 deferred，结束后重新检查再执行。
- 同盘使用 rename，旧绝对路径与旧相对路径保留兼容链接，不复制大型模型或轨迹。
- 跨盘迁移单独 staging、检查容量、校验再切换；本工具不偷偷执行大型复制。
- H100/H200 上的批量产物整理通过 CPU-only sbatch，srun 只在脚本内部使用；不占 GPU，不在 login-server 处理大型数据。
- H100/login-server 共用存储；H200 的实际产物须在对应节点上通过 Slurm 操作，不能把 login 镜像的迁移当成 H200 全部产物迁移。
- 保留事件 JSONL、rescore、人工事件 overrides、轨迹归档引用和未完成状态；不通过迁移“修正”原始实验结果。
- 视频聚合目录删除链接只删除视图，保留原始视频；删除真实训练权重必须经过已有活跃引用检查。

## 9. 接入验收

新训练后端完成以下事项即可被通用工具识别：运行清单合法；每个产物有身份与阶段；checkpoint 可验证且恢复能力明确；指标轴/单位明确；episode 有逐回合证据；视频有 sidecar；跨节点根目录明确；运行失败/恢复不会覆盖旧运行；旧路径兼容与迁移 receipt 可追溯。

不满足要求的记录应显式显示 unknown / incomplete / missing / mirror_only，而不是隐藏或补造结果。

## 10. 存储位置与文件状态

Dashboard 管理的是各服务器、各磁盘上实际存在的文件，不要求登记副本用途、数据集合或整份备份的完整性。按本约定分开存放模型、日志和 rollout；不必为 JEPA 等下游实验重复复制整次实验目录。需要传输时只选择实际需要的轨迹、独立图像或视频。

- 索引记录节点、物理根目录、相对路径、类型、文件数/大小及更新时间。模型和日志由训练页面管理，轨迹/图像/视频由 Rollout 页面管理；删除只作用于明确选中的存储位置和文件类型。
- 实体文件、指向别处的链接/引用、临时/未完成文件必须区分；显示引用时保留真实目标路径，不能把路径引用当成已保存的轨迹或权重。
- 写入或外部传输尚未结束的文件使用 `.partial` / `.incoming` 等临时名称，完成后原子改名。完整性有证据才标记：传输内容校验、容器结构检查、仅存在但未校验分别注明。实验完成状态不等同于文件完整性证据。
- 无论文件来自 Dashboard 还是 Codex/外部脚本，后台都能根据规范目录更新索引，不要求人工填写传输用途或备份清单。已有校验记录由系统自动维护；未经核实的文件保留待核实状态。
- 索引只加速查询；执行传输和删除前再次核对物理路径、文件是否变化以及活跃任务引用。文件删除与索引更新通过持久操作记录衔接，后台旧扫描不能恢复已删除记录。
- 历史混合目录可先建立只读摘要，不为了建立索引搬动正在使用的文件，也不把整个混合目录交给模型或 rollout 的批量删除入口。


## 11. 界面职责与实时发现（2026-10-24）

SFT 与 RLinf 各自管理本后端训练进程、历史及模型；模型与日志/指标分开管理。删除历史只清理该运行日志、训练/评测曲线和诊断指标，保留模型、来源配置、Rollout 回合产物；跨节点清理失败不得宣称全部完成。

Rollout 视觉产物指逐帧视频、帧图像、JEPA 特征及 VLA latent（含原 TSQ-RL 实验），不指训练曲线图。实验曲线图归日志/指标。轨迹容器中的视觉/latent 数组按所在文件管理，不在删除历史时拆分或清除。`feature` 是选择性传输/删除可选的 Rollout 文件类型。

模型和运行历史索引是浏览加速层，不是新文件/进程是否存在的依据。checkpoint 选择器支持按节点/实验直接浅层发现未登记模型；新训练支持运行目录或进程/Slurm 队列直接发现，曲线读取目标运行的最新日志/指标。操作前定向核对资源与活动状态。跨后端复用须同时满足数据集/任务、权重格式及动作配置兼容，不能仅凭实验名称自动转换模型。

整组新实验也必须可发现：从已配置实验父目录识别新建实验组，读取首批曲线，不要求先登记新实验或重建全量索引。可直接读取的目录在页面可见时定期浅层探测；H200 节点本地目录通过显式 CPU Slurm 读取，界面须说明调度延迟，不自动反复提交探测作业。

## 12. JEPA / world-model derived rollout artifacts (2026-10-24)

- Reusing PPO/GRPO rollouts does not require copying the whole source experiment. Store the frozen input manifest under `metadata/input-manifest.json`; reference the actual source node, physical episode path, source policy/checkpoint, task, scene, split, and available trajectory/image files. Distinguish a reference from a physically present file. Verify archive receipts against the actual file location before declaring an input available.
- Store newly produced per-episode JEPA features / VLA latents under `train|eval/<source-policy-checkpoint>/<outcome>/<episode_id>/` with an `episode.json` linking to the original episode and source arrays. These are Rollout artifacts, selectable as `feature`; they must not live in `logs/`, `metrics/`, or `checkpoints/`, and deleting training history must not remove them. Use `unknown` when the source checkpoint/outcome cannot be verified.
- Put model weights and their index in `checkpoints/`, training/evaluation scalar metrics and plots in `metrics/`, logs in `logs/`, and resolved input/split/normalization configuration in `metadata/`. Keep third-party native diagnostics/code under `artifacts/` when no standard location applies. Multi-task JEPA uses `task=multi-task` with explicit `task_ids`; episode metadata retains its real task.
- Source collection phase and downstream fitting split are separate. A source TRAIN rollout may be held out for JEPA validation; record fold membership explicitly. Group splits by physical scene across policy checkpoints/algorithms, not by trajectory file or window, and report independent risk scenes separately from risk episodes/windows. Exclude initialization risks from policy-risk fitting/evaluation.
- If only trajectory/image/feature files are transferred, preserve their episode evidence and source references, verify exactly those files, and record their real location. Do not describe the destination as containing untransferred videos, checkpoints, or all source resources. Use `.partial`/`.incoming` until the write or transfer is verified, then atomically publish. External writers follow the same layout so the background index can reconcile them without a separate detailed replica-purpose registry.
