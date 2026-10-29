# 本地 IK 连续性修改与验证（2026-10-29）

已实现：在正常 3 mm / 0.02 rad 位姿容差内优先选择相对上一成功发布解关节移动
最小的局部候选；两臂分别进行六轴 Ruckig 规划，保留共同写入器与保护。恢复缩步
和原目标续步使用原有求解器，避免小中间目标导致不动。没有改变 RTC 时间索引、
硬关节限位、夹爪时机或速度/加速度/jerk 默认值。

代码：`ik_continuity.py`、`client.py`、`servo.py`、`ik_workers.py`。
完整参数及退出开关见 `docs/deployment/LOCAL_IK_PARAMETERS.md`。

## 离线验证

连续回放来源：`20261029T104330.359218+0800_794465`。每一步使用前一步新解作为
参考与理想反馈；记录中的保持间隙重置片段；失败即停止该片段，不跳过失败目标。

| 左臂指标 | 旧求解器 1.5 mm | 仅容差改为 3 mm | 最小移动 3 mm |
|---|---:|---:|---:|
| 接受目标数 | 307 | 306 | 306 |
| 放宽容差目标数 | 107 | 45 | 32 |
| 单步最大关节变化 P95（度） | 6.875 | 6.875 | 3.543 |
| 单步最大关节变化最大值（度） | 10.619 | 6.875 | 6.608 |
| 位置误差 P95（mm） | 3.229 | 4.386 | 4.397 |

最小移动方法左臂累计关节路径长度约减少 79%，但三种方法最终左臂片段均失败。
右臂三组均接受 310 个目标。新方法单臂求解耗时 P95：左 16.06 ms、右 5.87 ms；
8 ms 预算仅针对最小移动候选搜索，不包含后续兜底、IPC 或进程启动。

回放不包含动力学、视觉反馈、碰撞和策略再推理，不代表闭环任务成功。

## 回归

以下测试已通过：229 tests、66 subtests（8.65 s）。SLSQP 出现一次内部试探点边界
裁剪警告，最终输出仍经硬限位及位姿范数检查。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/user/miniconda3/envs/dual_arm/bin/python -m pytest -q \
tests/test_ik_minimum_motion.py tests/test_independent_arm_planner.py \
tests/test_ik_continuity.py tests/test_ik_workers.py tests/test_piper_kinematics.py \
tests/test_servo.py tests/test_robot_observation_bridge.py tests/test_return_to_initial.py \
tests/test_native_eef.py tests/test_rtc_client.py tests/test_pose_trajectory.py
```

覆盖最小移动选择、累积容差、预算退出、独立双臂 q/v/a 连续性与制动、单写入器、
模式作用域及恢复衔接。验证过程中未访问 CAN 或发送机器人命令。
