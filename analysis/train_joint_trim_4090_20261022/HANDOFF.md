# 4090 Codex Bridge 接续任务

用户明确要求在 4090 Codex Bridge 的 bimanual-vla 项目开会话接替后续处理。请直接接续现有训练，不要重复提交。用户目标：用 joint_dataset 去首尾空闲帧重新训练，验证旧模型输出二阶差分与初始噪声高度相关的问题是否改善。

先读本目录 README.md、train_request.json、train_final_response.json；读取项目及 Bridge 当前操作规则。远端代码包含尚未提交的其他修改，禁止 reset/覆盖整个工作区。

## 正在运行

- Dashboard 8090 任务 train-20261022-172626-643dce0f，GPU 2，experiment joint_trim_nopad_no_rtc_b4_20261022，20k 步，batch4，无 RTC，expert full / LLM frozen。1000 checkpoint 已于 17:51:30 完成，之后继续训练。
- 本目录 evaluate_after_training.py 已 nohup 启动，noise_followup_status.json 当前 waiting_training。确认现有进程，不要重复启动。它成功训练后等待 >=23000 MiB 空闲、利用率 <5% GPU 再跑噪声评估；训练异常则退出并记录。若 Dashboard 自带 heldout 评估先占 GPU，耐心等待，不抢占。
- Dashboard 自动普通 heldout 评估每5000步；自定义噪声评估为最终 checkpoint。服务路径、token 用法见 dashboard.py：token 只从 /home/sunny/.config/bimanual-vla/server.env 在内存加载，不能输出或写入报告。
- 首次保存耗时约14分钟，纯训练约1.9step/s。检查后续保存性能及资源，先保持任务运行；不要盲目改正在运行的训练或删除 checkpoint。

## 接续完成标准

1. 主动跟进任务直到终态，确认 checkpoint 完整；若失败，先诊断并在用户已授权范围内修复和接续。不得只报告已提交。
2. 根据 Bridge 实际提供的持久监督机制登记训练/评估跟踪；进程结束必须核验日志、退出码及产物。不要仅设置普通消息后宣称已监控。
3. 检查自动噪声评估确实运行成功。对 train/heldout、5/10去噪步数汇报关节MAE（前5步、6–10步、全50步）、夹爪MAE、预测/示范二阶差分幅值、初始噪声相关系数。保存原始预测证据。禁止为了降低指标而部署输出滤波。
4. 对比 analysis/no_rtc_noise_recheck_20261022 和 analysis/arrangeblocks_reference_20261022；强调任务、数据、batch4与旧batch8等非受控差异，相关性不能单独证明根因。旧模型 train 上也约0.98，不能简单归因测试集域偏移。
5. 更新本目录 README/报告，及时以独立提交 push；不要提交 token、原始数据视频或无关修改。

## 已完成、避免重做

65 episodes，18024→14288帧，保留59train/6test原split。仅裁首尾空闲，内部停顿保留；核实黑边后直接伸缩224方图，源数据不变；130视频与state/action源切片检查通过。norm已完成；norm专用加载器跳过视频解码与原路径数值完全一致，训练仍读取真实图像。40项测试通过。
基础权重NVMe缺失已从归档恢复并SHA校验。batch8任务因OOM失败，现batch4稳定。当前runtime loss_action_dim7、rtc_training_max_delay0，元数据中默认max_delay8在禁用RTC时不生效。

不要移动机器人、重启策略/8101服务或停止其他用户进程。不删除额外权重。用户指定4090，当前无需迁移H100/H200。
