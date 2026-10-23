# joint_dataset 15k 本机部署

2026-10-23，按用户要求将 4090 的 `joint_trim_nopad_no_rtc_b4_20261022/15000` 下载并部署到当前 RTX 5060 Laptop 电脑。

- 来源为完整 15000 checkpoint；62 个参数/资产文件通过 SHA256 校验。仅同步推理参数及资产，不复制训练优化器。
- 本地目录 `/home/user/models/pi05_5060/joint_trim_nopad_no_rtc_b4_20261022/{15000_jax,15000_pt}`。
- 官方 JAX→PyTorch 张量映射：811 个参数名称和形状严格匹配，11 个 safetensors 分片；BF16 混合精度，CPU 词嵌入查表，无量化。
- 服务 `ws://127.0.0.1:8099`，入口 `./local_policy.sh` 或 `./local_policy_gui.sh`。已替换该入口管理的旧模型，旧权重保留。
- **左单臂、joint、7D、无 RTC**。用户确认训练数据的左臂状态和左腕画面误标为 right。部署两路输入为 `cam_high` / `cam_left_wrist`，内部仍映射到训练使用的 `right_wrist_0_rgb`，左腕模型槽位置零且 mask=false。图像 stretch 224×224，无补黑边。
- 输出 50×7，前6维为绝对关节角，夹爪为绝对开度比例；20 Hz，action offset 1。不要再次累计增量。
- 默认 prompt：`pick up the cube on the black site`；这是抓方块数据集，不能沿用抓瓶子提示词。GUI/客户端需选择左单臂，RTC关闭。

验证：5项输入/输出回归测试通过（含左右标签映射前后张量逐项相等）。使用heldout episode7 frame40的真实图像和状态，经WebSocket连续6次输出均为有限50×7，左臂映射更新后首请求0.642秒，后5次中位数0.257秒。`smoke.json`保存模型握手和逐请求结果。这个单观测重复测试验证部署链路，不是新的测试集精度或成功率；未发送任何机械臂指令。

`provenance.json`保存来源/契约/校验摘要，`hash_verification.txt`为传输检查，`prepare_sample.py`与`smoke.py`复现实机无动作验证。样本图像及预测NPZ仅本地保存，不提交Git。4090原20k训练继续，不受本地部署影响。

训练原始 action_contract.arm_side=right 保留作溯源；provenance.deployment_arm_side=left 单独指定真实执行臂。服务握手 arm_side=left、training_arm_side=right。没有交换、镜像、取反关节数值，没有修改权重和 norm；原始数据/正在运行的训练未改写。后续20k也必须保留同一部署映射。
