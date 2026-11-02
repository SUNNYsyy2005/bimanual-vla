# VR RECAP：采集、导出与训练

## 控制和交接

在 `quest_collection` Conda 环境启动客户端 GUI：

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate quest_collection
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bin/bimanual-vla collect-gui
```

Quest 模型、APK 和 ADB 的安装见 [QUEST_COLLECTION.md](QUEST_COLLECTION.md)。
该环境同时需要 `openpi-client==0.1.2`，供模型推理的 WebSocket 客户端使用。
启动前勾选推理页面的「启用人工接管录制」，选用 `local_ik_joint`、Ruckig 模型控制和
canonical v3 delivery 数据契约。当前不支持原生 `piper_eef` 的 RECAP 交接。

1. 模型开始执行后，点击「人工接管」。客户端清空待执行动作，制动模型伺服并验证
   实测停稳；成功后才授予 VR 控制权。VR 复用同一进程的 CAN 句柄，不启动第二个写入线程。
2. 松开左 Y／右 B，再按下：各臂以此时的实测关节 FK 和当前手柄重新建立相对标定。
   人工段使用 Quest 项目的 Pinocchio/CasADi IK 与 `JointCtrl`，不使用 Ruckig。
3. 保留线性扳机：开度 = `1 - trigger`。不增加 X/A、侧握键或夹爪切换按钮。
   刚接管／重新按下 Y/B 时，保持上一条夹爪目标；扳机越过该开度后接入线性控制。
   松开 Y/B 会保持当前机械臂位置和夹爪命令。物体撑住的实测开度不会覆盖关闭命令。
4. 点击「恢复模型」。人工写入停止，实测停稳后重建模型控制器；从新观测推理，
   重置 RTC 会话与旧前缀。交接前发起的推理结果不能重新获得控制权。
   模型夹爪滤波和伺服初值继承人工最后命令，再跟随新策略输出。
5. 点击成功／失败结束并标注整回合；主动中止或异常退出记为 aborted。
   既有客户端的停止／自动回位设置仍适用。失败停稳检查会阻止继续交接及自动回位。

状态通过私有临时目录的请求／确认文件交换。请求带独立会话 ID，重复请求不会重复交接。
GUI 只有在客户端确认可交接、心跳有效且无待处理请求时才启用按钮。

## 数据

录制继续写入 `deployment_runs/<run>/`，包含模型段、人工段、相机和原始命令。
`metadata.json` 保存回合结果；`trajectory.npz` 新增：

| 字段 | 含义 |
|---|---|
| `recap_owner` | policy / human / transition / hold |
| `recap_epoch` | 每次交接递增的控制权代号 |
| `recap_valid` | 可用的策略监督帧 |
| `command_gripper_fraction` | 实际命令开度，0 关、1 开 |

成功和失败回合一起导出到一个**新目录**，并上传该目录：
GUI 的「导出已标注 RECAP 回合」支持多选；导出完成后可点击「上传 RECAP 数据集」。
上传复用数据工具页的 Dashboard 地址和 token，以新数据集方式上传。
双臂基座偏移使用数据工具页已标定的值；未填写时仍可离线导出，上传前须补齐。

```bash
bin/bimanual-vla recap-export \
  deployment_runs/RUN_SUCCESS deployment_runs/RUN_FAILURE \
  --output datasets/recap_round_01 --fps 30
bin/bimanual-vla data-upload datasets/recap_round_01 \
  --name recap_round_01 --dataset-origin real --server http://192.168.101.9:8090 \
  --arm-base-offset X_M Y_M Z_M
```

上传认证沿用现有 Dashboard 配置；以上不包含 token。
`X_M Y_M Z_M` 须替换为实际标定的「右基座减左基座」米制偏移；不要猜测数值。
不要使用普通数据合并或裁剪重编号，它们尚不支持同步重写 RECAP 标注，服务端会拒绝此类编辑。
增补回合时重新选择录制目录，
导出为新数据集。导出失败会清理临时产物，原始录制不变。

导出保留完整时间线供价值学习：按相机／状态时间重采样，保留源时间戳；交接、等待、
重复状态、大幅时间错位及终止帧不参与策略损失。每个 episode 的
`meta/recap/episode_XXXXXX.npz` 保存控制权、有效性、人工掩码及结果。
EEF 动作为下一观测位姿，夹爪为同一步命令，继续采用当前 v3 数据契约。
策略取样只使用完整动作块，不能跨 epoch、无效帧或回合尾部；不会把两段隔开的轨迹拼起来。
训练与验证划分都须有足够长的有效连续片段（当前 50 步模型约 1.67 秒），否则训练预检会拒绝。
aborted／未标注回合不能静默当成失败训练。

## 开源训练接入

复用 [RLinf RECAP](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)，
固定 [RLinf commit 061c4cce](https://github.com/RLinf/RLinf/tree/061c4cce80bbab998ef90585475d3f51290a2dfa)，Apache-2.0。
直接运行其 returns、价值模型 SFT、n-step 优势与分位数标签实现。
本仓库只增加 Piper 三相机字段适配、私有 train/heldout 数据视图、完成保存标记和 JAX 策略数据适配。

流程为：

1. 已标注完整回合计算 returns：每步 -1，成功终止 0，失败终止使用配置奖励。
2. 使用 RLinf 的 SigLIP2 + Gemma3 价值模型、201 bins 和默认 `gemma_1m` critic expert；
   训练／验证按完整 episode 隔离，使用三路图像及任务文本。
3. 在训练集计算 n-step 优势和正优势分位标签；有效人工纠正样本标为正。
4. 自主与人工有效动作都参与当前 OpenPI/JAX 策略训练。
   正样本按配置概率使用 `Advantage: positive` 条件或无条件文本；负样本保留无条件文本。
   保持当前 EEF、图像缩放、RTC 和动作归一化链路。

这不是「只训练人工片段」。完整价值学习使用成功及失败回合，策略同时保留自主和人工样本。
也不是原封不动运行 RLinf 的 PyTorch 策略训练：为了兼容现有 JAX checkpoint 和 RTC，
策略阶段使用本仓库适配，部署仅支持正优势条件、引导系数 1，不实现大于 1 的 CFG 双分支外推。
heldout loss 使用有效完整动作块和无条件文本，衡量全体动作的重建误差；不把它称为正条件策略成功率。
实际任务成功率需要独立真机评估。

### 后端准备

在目标节点建立**独立** RLinf 环境，依赖／预训练权重按固定版本上游安装说明准备。
不要覆盖当前 OpenPI 环境。用 Conda Python 在项目根目录运行：

```bash
python -m scripts.setup_recap_backend --destination /DATA/sync/USER/projects/RLinf-piper-recap
```

脚本下载固定版本并安装字段适配，不安装依赖或下载模型权重。已有目录不会被改写。
需要预先准备 SigLIP2 `siglip2-so400m-patch14-224` 和 Gemma3 `gemma-3-270m`。
在真实 Dashboard 的 `config.json`，或者对应 `cluster_targets.<target>` 中配置目标节点的路径：

```json
{
  "recap_backend": {
    "root": "/DATA/sync/USER/projects/RLinf-piper-recap",
    "python": "/DATA/sync/USER/miniconda3/envs/rlinf_recap/bin/python",
    "siglip_path": "/DATA/sync/USER/models/siglip2-so400m-patch14-224",
    "gemma3_path": "/DATA/sync/USER/models/gemma-3-270m"
  }
}
```

在真机 SFT 页面选择 RECAP 数据集、独立 train/heldout 划分和 RECAP 训练方式。
参数为 `recap_n_step`、`recap_positive_fraction`、`recap_unconditional_probability`、
`recap_value_steps`、`recap_failure_reward`；API 同名并带 `recap_enabled=true`。
后端未配置时提交会明确失败，不会退化为普通 SFT。

所有阶段在同一个训练任务的资源分配内顺序运行；GPU 正式训练仍先检查 H100/H200
资源、配额并使用 Dashboard/Slurm。H200 的代码、环境、数据、权重须在该节点独立准备。
不在登录节点执行价值学习、优势推理或策略训练。

阶段产物位于实验目录旁的 `<experiment>_recap`：输入指纹、拆分、私有数据视图、配置、
价值 checkpoint、优势标签和 manifest。中断后使用相同实验及配置重试：
已完成阶段跳过，价值阶段仅从上游保存全部完成后标记的 checkpoint 恢复。
数据／划分／设置变化会拒绝复用；完整策略续训也必须保持 checkpoint 的 RECAP 设置。

## 验证边界

仓库测试覆盖控制权顺序、旧推理拒收、夹爪命令保持、动作块边界、模拟录制导出、
优势条件适配及数据视图。上游适配已对固定版本源文件检查。
真机交接和 RLinf GPU 全流程仍须在硬件及后端环境齐备后验证；软件测试不代表实机验证完成。
