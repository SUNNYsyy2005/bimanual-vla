# RECAP 独立环境与合成数据训练检查

本目录提供 RLinf RECAP 的四阶段检查：回报计算、价值模型 SFT、优势标注、CFG 策略训练，以及 checkpoint 恢复与保存后推理。使用独立 Conda 环境、源码副本和输出目录，Dashboard 的 Python 环境不参与安装。

## 已部署路径

| 机器 | 工作目录 |
| --- | --- |
| `4x4090-wg` | `/mnt/nvme2n1p1_2T/sunny/experiments/recap-environment-20261101` |
| H100 / login-server 共享存储 | `/DATA/disk0/sunny/experiments/recap-environment-20261101` |

每个工作目录包含 `env/`、`source/`、`models/`、`datasets/`、`configs/`、`runs/` 和 `records/`。脚本安装在工作目录根部；`source/` 为服务器现有 RLinf 的独立副本，而不是运行时拉取的 GitHub 最新代码。

```bash
source <工作目录>/activate.sh
python -m pip check
```

`runtime-pins.txt` 记录经过依赖检查的核心版本。OpenPI 使用 `rlinf-transformer-openpi` 对 Transformers 的实现；仅重装标准 Transformers 会覆盖其模型修改。已有环境通过 Conda Pack 部署到 4090，传输后验证 SHA-256 并执行 `conda-unpack`。

## 权重

价值模型需要 `google/siglip2-so400m-patch14-224` 和 `google/gemma-3-270m`，策略使用服务器已有的 π0.5 PyTorch 权重。4090 只传输 SigLIP 的原始视觉塔，不传输 RECAP 不使用的文本塔，导出来源写入 `ORIGIN.json`。

Gemma3 需要具有访问权限的 Hugging Face 账号。可在已登录的机器下载后传输权重；无需把 token 复制到训练服务器。模型入口只读取本地目录。正式 Gemma3 权重缺失时，训练检查应保持未完成状态。

官方说明：[Gemma3-270M 访问入口](https://huggingface.co/google/gemma-3-270m)、[LeRobot 0.3.3 依赖定义](https://github.com/huggingface/lerobot/blob/v0.3.3/pyproject.toml)。

## 合成数据与已执行的前置检查

`create_synthetic_dataset.py` 创建 LeRobot 数据：训练集 4 条轨迹 / 60 帧，留出集 2 条轨迹 / 26 帧，分别使用固定随机种子，包含成功和失败轨迹。目录、任务文本及 `SYNTHETIC_DATASET.json` 都标记为合成测试数据。只生成新数据或复用带标记的完整测试集；未完成的生成目录移到自己的 `datasets/incomplete/`。

`apply_recap_fixes.py` 修复回报计算对单元素布尔数组的处理，避免把 `[False]` 当作成功。多元素数组、字符串及其他非法成功标记会显式报错。补丁仅应用于独立源码副本，遇到未知源码内容会停止。还为两个训练 worker 提供可选的进程显存上限。

`test_returns_contract.py` 对真实 parquet 处理函数检查成功回报、失败终止惩罚及非法输入；`check_data_contract.py` 再用实际 `ValueDataset` 读取两个合成数据集。4090 与 H100 均已通过上述检查、模型与训练入口导入检查以及 `pip check`，并实际通过各自 GPU 的 CUDA 矩阵运算检查。

2026-11-01 的部署状态：π0.5 与 SigLIP 视觉权重已就绪，Gemma3 授权权重尚未下载，完整训练检查尚未执行。`check_model_assets.py` 会明确记录缺失文件，启动器在权重未就绪时停止。GPU 训练结果以每台机器自己的结果文件为准。

## 完整训练检查

H100 必须从 login-server 提交 Slurm；不要在登录节点执行训练脚本：

```bash
sbatch /DATA/disk0/sunny/experiments/recap-environment-20261101/train.sbatch
```

4090 是独立工作站，使用以下入口。入口挑选空闲显存至少 16500 MiB 且当前利用率不超过 10% 的卡，并为训练 worker 设置 58% 的显存上限；找不到合适的卡会退出。

```bash
bash /mnt/nvme2n1p1_2T/sunny/experiments/recap-environment-20261101/train_4090.sh
```

4090 的测试配置冻结价值模型的视觉/语言骨干，训练价值投影与 critic expert；策略训练冻结 VLM，仅更新 action expert。H100 使用完整模型训练配置。两者均使用真实模型、优化器与 RECAP 入口，测试步数为：价值模型 10 步并恢复到 12 步，策略 4 步并恢复到 6 步。

检查结果：

- `records/pipeline.json`：逐阶段状态、主机、进程、Slurm 作业及日志路径。
- `records/preflight.json`：实际 CUDA、GPU、包版本和导入入口。
- `records/verification.json`：sidecar 覆盖、正负标签、有限 loss、非零梯度、恢复步数、参数变化及保存后推理。
- `runs/value/synthetic_value/checkpoints/` 和 `runs/cfg/synthetic_cfg/checkpoints/`：checkpoint。

只有 `pipeline.json` 的状态为 `passed` 且存在对应的 `verification.json`，才算完整检查通过。依赖安装或数据预检通过不代表完整训练通过。合成数据用于验证工程链路，不代表机器人任务性能；正式数据需要重新计算归一化、回报、优势并训练评测。
