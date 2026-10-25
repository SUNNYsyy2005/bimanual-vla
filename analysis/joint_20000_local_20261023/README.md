# 20k 本地部署接续

目标与 15k 一致：RTX 5060 Laptop 的 `/home/user/models/pi05_5060/joint_trim_nopad_no_rtc_b4_20261022/`，分别使用 `20000_jax` 和 `20000_pt` 新目录。15k 当前服务与权重保持可回退，20k 未完成传输、转换和本地 smoke 前不切换入口。

4090 侧源 checkpoint：`/mnt/nvme0n1p4_1T/sunny/tsq-storage/checkpoints/openpi/pi05_piper_single_arm_lora/joint_trim_nopad_no_rtc_b4_20261022/20000`。完整 Orbax checkpoint 与最终离线评估均已核验。`analysis/train_joint_trim_4090_20261022/transfer_manifest_20000.json` 逐文件记录了仅用于推理的 `params/` 与 `assets/`：63 文件，共 6,176,496,207 bytes。无需把 8.8 GiB `train_state/` 优化器状态传到 Laptop。

在 Laptop 从 4090 拉取这两个目录到新建的 `20000_jax`；以 `verify_transfer.py --manifest .../transfer_manifest_20000.json --directory .../20000_jax` 核对 63 个文件的相对路径、字节数和 SHA256。校验后沿用 15k 的 `scripts/pi05_local/convert_checkpoint.py` 转换为 `20000_pt`，使用新路径运行同一输入/输出回归测试和无动作 WebSocket smoke，再切换 `local_policy.sh` 管理的模型。源 manifest 的 `target_transfer_verified` 等字段为 `false`，不能在本机步骤完成前改为成功。

**左臂兼容映射必须保持：** `action_contract.arm_side=right` 保存训练来源；`provenance.deployment_arm_side=left` 指定物理执行臂。对外单臂 `arm_side=left`、腕相机 `cam_left_wrist`；内部送入训练时的 `right_wrist_0_rgb`，`left_wrist_0_rgb` 置零且 mask=false。关节数值、norm、action offset 1、50×7 输出及无 RTC 均不变。不能全局替换 `right` 或交换模型图像槽位。参考 15k 已验证的 `analysis/joint_15000_local_20261023/provenance.json`、`scripts/pi05_local/serve.py` 和 `tests/test_local_pi05_inputs.py`。

当前状态：**源文件 SHA256 清单已完成，Laptop 传输/转换/服务切换未验证。** 不把离线噪声 MAE 或单帧 WebSocket smoke 当成真实机械臂成功率；不发送机械臂指令。
