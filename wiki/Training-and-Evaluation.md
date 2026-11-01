# 训练与评测

项目有两条训练路径，**不要把 SFT checkpoint 误称为通过 RL 训练的模型**。

| 路径 | 数据 | 执行环境 | 输出 |
|---|---|---|---|
| SFT | LeRobot 示教、自然语言、RGB、状态/动作 | OpenPI pi0 / pi0.5 | 训练权重、norm stats、合约、日志与评测 |
| 仿真 RL | 仿真 rollout、reward/cost、算法配置 | RLinf + 对应模拟器/profile | RL checkpoint、曲线、rollout 评估视频 |

## SFT 核心流程

1. 使用 [[数据合同与导出|Dataset-and-Contracts]] 验证数据集和动作表示。
2. 在 Dashboard 选择数据、固定 train/test split、准备 normalization。
3. 配置 OpenPI 的视觉编码器、action expert、语言模型微调选项（包括全量/LoRA）。
4. 选择支持的训练目标：本地 GPU 或已配置的 Slurm 节点。
5. 监控 loss、checkpoint、任务日志，使用**未参与训练**的 episode 做验证。
6. 与 Policy 运行环境配套保存 norm、输入键、action contract 和 FPS，不能只复制参数文件。

π0.5 支持训练期 RTC prefix conditioning；部署时需与实际训练/推理配置一致。通常 action chunk 形状为 50×7D（单臂）或 50×14D（双臂），由合同确认。

## 仿真 RL

RLinf 通过**仿真 Dashboard**启动，必须预先配置环境、reward/算法 profile 和运行目标。不是 GUI 点一下即可自动对任意真实硬件任务进行 RL。不同算法是否可用取决于当前 profile，而非 Wiki 中的算法名称。

## 参考资源

- [Dashboard 运维与 SFT](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/server_4090/README.md)
- [仿真 Dashboard / RLinf](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/server_4090/SIMULATION_DASHBOARD.md)
- [服务器环境、训练与评测](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/deployment/SERVER_PATHS_ENV_TRAIN_EVAL.md)
- [统一实验产物合同](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/EXPERIMENT_ARTIFACT_CONTRACT.md)
- [[已发布模型和演示|Models-and-Demos]]

建议每次实验固定并记录：数据集版本、split、schema、基座权重、微调层、步数、随机种子、checkpoint 和真实评估指标。
