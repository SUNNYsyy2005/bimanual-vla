# 开源控制与训练参考

## 伺服实现

| 来源 | 固定版本 | 本项目使用范围 |
| --- | --- | --- |
| [pantor/ruckig](https://github.com/pantor/ruckig/tree/v0.12.2) | 0.12.2，MIT | 本地 state-to-state 在线轨迹生成，速度、加速度、jerk 约束，多关节时间同步，以及速度接口制动 |
| [agilexrobotics/piper_sdk](https://github.com/agilexrobotics/piper_sdk/tree/c9e8a28174e71eeaac448593cb65f8ab258a92fe) | 参考 c9e8a28174e71eeaac448593cb65f8ab258a92fe；运行依赖仍为 0.6.1 | CAN 状态、反馈新鲜度、ModeCtrl、JointCtrl、GripperCtrl 和单位约定 |

轨迹算法由 Ruckig 提供，Piper SDK 提供驱动接口。没有把 SDK 的发送示例
当成已验证的轨迹平滑算法，也没有写入固件的 JointMaxAccConfig 等参数。
JointCtrl 使用 0.001 度；软件内部统一使用 rad，发送时才量化。

仅调用社区版的本地状态到状态接口。社区版 intermediate waypoints 会访问
云服务，本控制路径不使用；商业版 Tracking API 也不使用。

### 算法边界

- 模型目标点通常为 20 Hz，独立伺服时钟默认 100 Hz。每次替换目标都保留
  上一条生成轨迹的位置、速度和加速度，不在每帧重新假定速度为零。
- 双臂 12 个关节共享一次 Ruckig 时间同步计算；两个 CAN 接口仍按先 Mode、
  后所有 Joint、最后所有 Gripper 的顺序发送，属于软件同步。
- 相邻模型目标估计的末端速度被限制在运动约束内，可用
  `--no-servo-waypoint-velocity` 改为逐目标零末速。目标路径可能被重定时，
  不保证机械臂恰好在每个 50 ms 时刻穿过模型目标。
- 反馈用于初始化、状态/新鲜度检查和跟踪误差监控。软件限制的是生成的
  **指令轨迹**，实际机械臂运动还取决于固件速度设置、底层控制器和负载。
  CAN 量化与操作系统调度误差也会影响用差分计算出的指令导数。
- 位置边界检查发生在发送前，越界会触发故障，不对生成结果再做截断。
  社区版没有本项目所需的完整位置受限避障规划；靠近机械限位的剧烈反向
  请求可能被拒绝。该模块不提供碰撞检查、力矩控制或电机电流闭环。
- 目标超时、正常 hold 和退出使用 Ruckig 速度接口生成受约束的停止。
  反馈/驱动故障停止进一步发送；此时不承诺仍可生成和执行完整制动轨迹。
  未静止的安全暂停不能自动重置为零速度；需在机器人静止后重新初始化。
- Python 线程和普通 Linux 调度提供尽力而为的周期执行。过期周期不补发成
  连续突发，记录实际时间和 overruns，不能将配置 100 Hz 等同于硬实时保证。

## 训练 RTC 与微调

| 来源 | 固定版本 | 本项目使用范围 |
| --- | --- | --- |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) | 215abfb217dbac7d5f1273282331b9b1866c0479，Apache 2.0 | Pi0/Pi05 flow loss、Euler 采样、Gemma 自适应 RMSNorm 和参数结构 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/policies/pi05/modeling_pi05.py) | 146d8474932c8b566717eb2738ba459337ab71c4，Apache 2.0 | 训练时随机干净 action 前缀、逐 action 流时间、前缀 loss mask 与采样时硬前缀条件化 |

实现位于 `bimanual_vla/training/openpi_model.py`，是针对项目 OpenPI JAX/NNX
接口的适配。π0.5 前缀动作使用 flow time 0，后缀使用采样的 flow time；仅
计算后缀和真实机器人 action 维度的 loss。推理时每个 Euler 步固定旧 chunk
中覆盖预计延迟的前缀。参考论文：
[Training-Time Action Conditioning，arXiv:2512.05964](https://arxiv.org/abs/2512.05964)。

训练 RTC 与原来的推理 prefix guidance 是两种明确的运行方式。它们都不
代替伺服运动约束。训练 RTC 当前只接入 Pi05 的 JAX/Orbax 路径，不能用于
未经过该训练的 checkpoint 或当前 PyTorch 路径。

改编代码的来源与变更见 [NOTICE](../../NOTICE)，Apache 2.0 许可全文见
[APACHE-2.0.txt](../licenses/APACHE-2.0.txt)。Ruckig 作为依赖保留其发行包的 MIT 许可。
