# LingBot-VLA Piper 同步推理

独立入口复用现有 Piper 反馈、驱动限位检查、轨迹执行和故障阻断。固定关闭 RTC 与异步推理，每段执行完毕后获取新观测。默认每段执行 25 步（训练频率 20 Hz），`--sync-wait-steps` 可调整段长。

模型对应真实 Piper 双臂放水瓶数据：左右各 6 个绝对关节角（弧度）和 1 个夹爪开度（0 关闭，1 打开），共 14 维。不能把此模型当成单右臂或 RoboTwin/ARX5 模型使用。

默认指令：`Place the bottles on the black pad in the center of the table.` 服务固定指令；如需左侧/右侧垫任务，服务和客户端须同时指定训练中对应的指令。

## 本机启动

```bash
cd /home/user/project/bimanual-vla
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dual_arm
python -m bimanual_vla.deployment.lingbot_sync_client \
  --host 192.168.101.9 --port 8101 --camera-preview
```

默认只读反馈并请求推理，不发送动作。检查预览、左右臂映射和输出后，执行时在上述命令后添加 `--allow-execution`。不要同时运行其他采集、遥操作或运动控制程序。

CAN 默认按既有 USB 序列号映射解析，不固定 `can0/can1`：

| 臂 | USB 序列号 |
|---|---|
| 左 | `003D00395246570620323934` |
| 右 | `003C00365246570620323934` |

这是既有设备配置，不能替代现场物理接线确认。可显式传 `--left-can` / `--right-can`。腕部相机读取本机 GUI 保存的设备选择；可用 `--cam-left-wrist-device` / `--cam-right-wrist-device` 覆盖。顶视相机默认自动识别，也可传 `--cam-high-device`。本入口不自动使用旧回位文件。

## 图像与服务协议

客户端发送原始采集流程生成的 RGB CHW 256×256 补边图像，不预缩放到 224。服务端执行训练转换：裁剪 `[0,55,256,200]`，再用 OpenCV `INTER_LINEAR` 缩放到 256×256。服务提供 OpenPI websocket 协议，返回 `actions: [50,14]`，并声明 `rtc_supported=false`、`policy_family=lingbot_piper`。客户端拒绝其他模型族的服务。

## 服务部署

在 LingBot 模型环境内运行；GPU 集群必须通过 `sbatch`，独立 4090 工作站可直接运行。保持训练目录结构：

```text
model/lingbotvla_cli.yaml
model/checkpoints/global_step_<actual_step>/hf_ckpt/
```

`hf_ckpt` 须包含模型分片、处理器文件、`piper_bimanual.yaml`、`piper_bimanual_norm.json` 和 `export_source.json`。保持 Piper 归一化与机器人配置，不能换成 RoboTwin 配置。

```bash
export QWEN3VL_PATH=/home/sunny/lingbotvla/models/Qwen3-VL-4B-Instruct
/home/sunny/lingbotvla/envs/model/bin/python scripts/lingbot/serve_piper.py \
  --lingbot-root /home/sunny/lingbotvla/code \
  --checkpoint /path/to/model/checkpoints/global_step_10700/hf_ckpt \
  --port 8101
```

`export_source.json` 和握手元数据记录真实训练步数。20k 是训练目标，并不表示中途导出的恢复点已经训练到 20000 步。

部署到不同机器时，保留原始 YAML 副本，并只调整 `model.flexpi_params` 内 Wan VAE、DINOv3、MoGe 的本地资产路径。它们也参与推理，必须与训练使用的资产一致；本次部署已核对三个文件的 SHA-256。

不连接机械臂的接口检查：

```bash
python scripts/lingbot/smoke_piper.py --host 192.168.101.9 --port 8101
```

该检查使用合成图像，只验证握手、两次模型推理及有限的 `[50,14]` 输出，不评价抓放成功率。离开公司局域网后，可在已连接 WireGuard 的前提下把主机改成 `10.0.200.100`。
