# 数据合同与导出

从原始 episode 到训练数据必须保留 **schema、action 来源、时间戳、左右顺序、夹爪语义**，不能只比较张量维度。

## v3 数据合同（新采集）

| 模式 | Schema | 观测维度 | 原始动作维度 | 模型动作维度 |
|---|---|---:|---:|---:|
| 单臂 | joint | 7D | 7D | 7D |
| 双臂 | joint | 14D | 14D | 14D |
| 单臂 | delivery | 10D | 10D | 7D |
| 双臂 | delivery | 20D | 20D | 14D |

- 双臂拼接顺序固定为 **left + right**。
- 夹爪归一化开合量：**0 = closed，1 = open**。
- joint 的动作是对应关节空间目标；delivery 原始数据是绝对 EEF pose/target，在训练边界转换为 current-anchored EEF 模型动作。
- v3 默认 FPS 20、action horizon 50；录制端和发布数据的实际 FPS 必须以元数据为准（发布的 pick-place 集为 30 FPS）。
- v2 legacy 仅做兼容读取，不应隐式赋予 v3 含义。

## 校验、导出、复验

~~~bash
# 1. 验证原始目录
bin/bimanual-vla data-validate --input-dir episodes_piper_v21 --target-fps 20

# 2. 转换为 LeRobot v2.1（根据数据实际 schema / FPS 调整）
bin/bimanual-vla data-export   --input-dir episodes_piper_v21   --repo-id piper/piper_v1   --root lerobot_datasets/piper_v1   --fps 20

# 3. 验证导出结果
bin/bimanual-vla data-check lerobot_datasets/piper_v1
~~~

对同一采样分别导出 joint 和 delivery 时，应当明确这是**同一批示教的两种表示**，不是两倍独立样本。训练前检查 split、normalization 与 Policy action contract 是否一致。

## 真机部署最容易出错的点

- Policy 期待 delivery，但发来 joint；形状相同也可能含义不同。
- 训练的 image keys / 相机顺序与运行时观测不一致。
- 模型 FPS 与客户端 action-hz 不一致，产生明显延迟或过度约束。
- raw action、model action 与机器人命令的变换在不同边界重复执行。

**规范来源：** [Piper Data Contract](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/collection/PIPER_DATA_CONTRACT.md) · [采集指南](https://github.com/SUNNYsyy2005/bimanual-vla/blob/master/docs/collection/DATA_COLLECTION_GUIDE.md)。
