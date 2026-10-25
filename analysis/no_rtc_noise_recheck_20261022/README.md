# 旧 RTC / 非 RTC 模型初始噪声相关性复核

基于保存的原始预测和 RNG seed 重新计算，**不是重新加载已删除的权重做推理**。两模型均以 no_prefix 推理，观测、seed 和归一化统计相同。每组 12 个观测 × 2 seeds，来自 2 条 episode。

| 模型 | split | 5 步去噪 | 10 步 | 50 步 |
|---|---|---:|---:|---:|
| RTC 20k | train | 0.983157 | 0.984554 | 0.990673 |
| 无 RTC 20k | train | 0.981273 | 0.983529 | 0.988296 |
| RTC 20k | heldout | 0.977987 | 0.982908 | 0.991307 |
| 无 RTC 20k | heldout | 0.975764 | 0.982545 | 0.989287 |

指标为每个观测、每个关节的预测二阶差分与初始噪声二阶差分 Pearson 相关系数，再取均值。全部重算值与历史记录误差小于 1e-5。

无 RTC 模型和训练观测同样存在高相关性，所以 RTC 和 train/test 差异都不是这一现象的必要条件；不能据此断定数据集是根因。后续 joint_dataset 实验同时比较绝对误差与噪声幅值，不添加输出滤波。

`recheck.py` 需要历史 `analysis/no_rtc_comparison_20261021` 和 `analysis/rtc_padding_fix_20261021` 文件，以 `JAX_PLATFORMS=cpu` 运行；`results.json` 保存参数及原始预测文件哈希。
