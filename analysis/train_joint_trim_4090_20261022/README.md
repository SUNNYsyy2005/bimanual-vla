# joint_dataset 去空闲帧后的非 RTC 训练

2026-10-22：任务 `train-20261022-172626-643dce0f` 已实际训练至 1000 步，首个 checkpoint 于 17:51:30 保存完成，此后恢复训练（约 1.9 step/s）。尚未完成 20k，不能据 loss 宣称噪声问题解决。

- 数据：`joint_dataset` → `joint_trim_nopad_20261022`，右单臂、7D、20 Hz，65 episodes。
- 18024 → 14288 帧：删去开头 1812、结尾 1924 帧；保留两端各 2 帧上下文、内部停顿和原始提示词。阈值为累计关节运动 0.001 rad、夹爪 0.002。
- 全部 130 段历史视频已去除核实的上下黑边，将内容区直接缩放为 224×224；源数据不变。逐段检查帧数、帧率和尺寸，state/action 与源切片完全相等。历史黑边造成的细节损失无法恢复。
- 沿用 59 train / 6 heldout；heldout episodes 为 7、23、39、44、46、55。
- π0.5 基础权重重新训练，expert 全量、LLM 冻结、无 RTC、stretch；GPU 2，batch 4，20000 steps，1000 步保存，5000 步保留。
- 初始 batch 8 在 step 0 后 OOM，已失败；新开 batch 4 实验 `joint_trim_nopad_no_rtc_b4_20261022`。与旧 batch 8 模型的对比包含 batch size 和数据/任务差异，不能单独归因数据集。
- 模型输出头 32D，loss 仅覆盖真实 7D；runtime `rtc_training_max_delay=0`。Dashboard 中禁用 RTC 时残留的 max_delay 默认值不代表开启 RTC。

## 启动修复与验证

基础权重从归档恢复至 NVMe，并逐文件 SHA256 验证。修复 LeRobot 将 `image_timestamp.*` 标量统计误当作 RGB 统计的问题，原始时间戳列保留。

归一化专用 dataset 跳过不参与 state/action 统计的视频解码；训练和评估继续使用真实图像。六个分散索引的 state 和 50×7 action 与原始加载路径完全相同（最大差值 0）。新 norm 任务 `norm-20261022-172145-cff696de` 完成 812/812 batches；40 项相关单元测试通过。见 `norm_parity.json`、`dataset_verification.json`。

首个 checkpoint 保存约 14 分钟，训练步速度不含这段开销；尚不承诺准确总耗时。后续需关注保存耗时，不要为优化它打断当前训练。

## 自动噪声评估与交接

`evaluate_after_training.py` 已在 4090 后台启动，状态文件 `noise_followup_status.json` 当前为 waiting_training。仅训练成功且 GPU 空闲后运行；训练失败会中止并记录。不要重复启动。

评估用两条训练 episode 和全部六条 heldout、多个帧位置、seed 0/1、5/10 次去噪；使用真实图像、原始任务提示词，无 RTC、无滤波。统计关节/夹爪误差、示范与预测二阶差分、与初始噪声相关系数；保留原始预测。结果写入本目录 `final_noise_evaluation/`。高相关性必须结合误差及二阶差分幅值解释，不能单独证明数据集或训练原因。

远端项目 `/home/sunny/bimanual-vla`。训练日志 `/home/sunny/.local/share/bimanual-vla-server/tasks/train-20261022-172626-643dce0f/task.log`。训练参数、任务响应、数据处理报告及复现实验脚本均在本目录；无需 token 落盘。

## 18:56 内存压力中断与恢复

原任务保存 4000 步 checkpoint 时，`systemd-oomd` 杀死了包含训练进程的 `bimanual-vla-dashboard.service` cgroup；没有 `exit.json`，Dashboard 将任务标为 `lost`。原自动噪声评估因此记录 `aborted`，没有产生预测结果。3000 步目录包含 Orbax 根、参数和训练状态的完整元数据，原日志也确认其保存完成。4000 步未完成的约 9.6 GB 临时目录已移至同盘 `joint_trim_nopad_no_rtc_b4_20261022_4000_incomplete_20261022`，保留故障证据并避免恢复时误读；最终收尾仍需清理它。

19:04 从 3000 步**原地恢复**，训练进程 PID 2942442（runner 2942441），位于独立的 `codex-bimanual-jointtrim-resume-20261022.scope`。恢复程序见 `resume_after_oom.py`，退出记录为 `resume_after_oom_exit.json`，日志为 `resume_after_oom.log`。数据、模型、batch 4、无 RTC、总步数 20000 均沿用原实验；为降低保存时内存压力，保存间隔从 1000 改为 5000 步，`OPENPI_CHECKPOINT_CHUNK_PAUSE_S=30`。19:06 已恢复训练至约 3080 步；后续终态与 checkpoint 尚待核验。

新的评估等待程序 `evaluate_after_resume.py` 在独立 `codex-bimanual-jointtrim-noise-20261022.scope` 中运行（PID 2948191，runner 2948190），仅在训练退出码为 0、20000 步 checkpoint 完整且 GPU 空闲时运行原 `evaluate_noise.py`。状态、退出记录和日志分别是 `resume_noise_followup_status.json`、`resume_noise_waiter_exit.json`、`resume_noise_followup.log`；最终预测和统计仍写入 `final_noise_evaluation/`。Bridge 已对训练与评估等待 PID 登记持久监控及成功结束通知。训练完成后还需核对退出码、普通 heldout loss 与噪声评估结果，再形成结论。

已按用户要求交接至 4090 Codex Bridge 会话 `01a1630a-d68d-7c13-84fb-a2feb4f1294c`，名称「bimanual-vla：joint_dataset训练与噪声评估接续」，工作目录 `/home/sunny/bimanual-vla`。已确认交接消息被消费并开始 active turn；具体任务见 `HANDOFF.md`。
