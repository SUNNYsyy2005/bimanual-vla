# 本机 RTX 5060 π0.5 离线推理

本机 `tx`：RTX 5060 Laptop 8GB（sm_120）、32GB 系统内存。环境
`/home/user/miniconda3/envs/pi05_5060` 使用 PyTorch 2.7.1 CUDA 12.8。
现有 `dual_arm` 中的 CUDA 12.6 PyTorch 不支持本卡，不能直接复用。

## 文件与模型

- OpenPI PyTorch 源码：`/home/user/src/openpi-5060`，来自官方提交
  `215abfb217dbac7d5f1273282331b9b1866c0479` 的干净副本。
- JAX 参数：`/home/user/models/pi05_5060/piper_15000_jax`。
- 转换后的分片：`/home/user/models/pi05_5060/piper_15000_pt`。
- Tokenizer：`/home/user/models/pi05_5060/paligemma_tokenizer.model`。

选用 Piper `fix_nopad_rtc_empirical_d10_20261019/15000` 的完整参数。
4090 上原服务记录的 20000 路径已失效；本次找到的归档中最高完整点是
15000。参数与资产 28 个文件通过原归档 SHA256 清单检查。
归档配置是完整 action expert 微调、无 LLM LoRA，不能仅因上级目录名包含
`lora` 就把它当作 LoRA 权重。本次不启用 RTC 前缀或引导。

## 环境与转换

从项目根目录运行。首次安装独立环境及 OpenPI 必需的 Transformers 补丁：

```bash
bash scripts/pi05_local/setup.sh
```

补丁只写到 `pi05_5060` 环境；OpenPI 源码须已放到上述位置。

```bash
/home/user/miniconda3/envs/pi05_5060/bin/python scripts/pi05_local/convert_checkpoint.py \
  --source /home/user/src/openpi-5060 \
  --checkpoint /home/user/models/pi05_5060/piper_15000_jax \
  --output /home/user/models/pi05_5060/piper_15000_pt
```

转换调用官方张量映射，严格核对所有参数名称与形状，保留原混合精度。
删除的仅是动作采样不调用的词表输出头，未删除视觉、语言解码器或动作专家。
使用 CPU 读取 Orbax，避免 JAX 与 PyTorch 同时占用 GPU。

## 离线测速

```bash
/home/user/miniconda3/envs/pi05_5060/bin/python scripts/pi05_local/benchmark.py \
  --source /home/user/src/openpi-5060 \
  --checkpoint /home/user/models/pi05_5060/piper_15000_pt \
  --tokenizer /home/user/models/pi05_5060/paligemma_tokenizer.model \
  --recording deployment_runs/20261021T113041.846378+0800_109455 \
  --mode bf16 --steps 5 --repeats 4 \
  --output analysis/pi05_5060/bf16.json
```

快捷运行：`bash scripts/pi05_local/run_benchmark.sh`。可追加 `--mode nf4`
或 `--steps 10` 覆盖默认参数。

将 `--mode bf16` 改成 `--mode nf4`、输出改为 `nf4.json` 可测试低显存模式。
NF4 只量化语言解码器中的 Linear，视觉编码器、动作专家和投影保持原精度。
先用 BF16 作为数值参考；量化输出接近参考不能证明任务成功率不变。

脚本逐张量加载分片以降低加载峰值；默认关闭编译，避免编译和 CUDA Graph
占用额外显存。每次采样固定 seed，记录冷启动、热推理时间、PyTorch 峰值
已分配及保留显存，输出 NPZ 保存归一化动作。

输入来自一次实机录像的三路相机首帧和初始状态。录像图像有损，且存在补边，
脚本先移除 256 图像的上下补边，再按此检查点的 stretch 规则变成 224 图像。
状态先把米转换成夹爪开度比例，再用训练分位数归一化，14 维状态先离散编码、
再填充到 32 维。输出 `[1,50,32]`，其中前 14 维对应 Piper；关节是相对动作段
起始状态的增量，夹爪是绝对开度。本脚本不进行反归一化或机械臂执行。

这个测试验证本机推理可行性、延迟和有限值，不评价抓放成功率。不能直接把
归一化输出作为机械臂目标发送。

## 实测（2026-10-21）

单个真实录像观测，每种设置先冷请求一次，再重复四次。全部输出有限，形状
`[1,50,32]`；无 RTC、无编译、无 CPU 卸载。模型约 3.3 秒加载完成。

| 模式 | 去噪次数 | 热推理中位数 | PyTorch 峰值分配 | 峰值保留 |
|---|---:|---:|---:|---:|
| BF16 | 5 | 0.271 秒 | 6789 MiB | 6898 MiB |
| BF16 | 10 | 0.312 秒 | 6789 MiB | 6904 MiB |
| 语言 NF4 | 5 | 0.295 秒 | 4018 MiB | 4164 MiB |

推荐 BF16：显存能容纳，避免量化偏差，也比本次 NF4 测量稍快。若需要给其他
GPU 程序留显存，可选择 NF4。NF4 相对同 seed BF16 的前 14 维动作，在训练
动作尺度下的关节平均差为 0.412°、最大 2.226°，夹爪平均差 0.00574；这是
一个观测上的数值差异，不是对真值的误差，也不是量化任务成功率。

数据：`analysis/pi05_5060/{bf16,nf4,bf16_10steps}.json`、同名 NPZ，及
`quantization_comparison.json`。计时仅包含已预处理观测的模型采样，不包括
相机采集、图像预处理、网络及执行。PyTorch 显存统计不含桌面和 CUDA 上下文，
不能把剩余显存全部分配给其他程序。本轮未进行实机动作测试。

### 连续 100 次测试

BF16、5 次去噪，同一预处理观测连续执行 100 次热推理，全部成功且输出有限。
中位数 0.276 秒，P95 0.284 秒，P99 0.305 秒，范围 0.257～0.309 秒。
前/后 20 次中位数分别为 0.276/0.277 秒，未观察到明显持续变慢。
GPU 最高 65°C，包含桌面及上下文的整卡显存峰值 7618 MiB。
冷请求单独为 0.546 秒，不计入热推理分位数。

原始数据与汇总见 `analysis/pi05_5060/bf16_stability_100.json`、
`bf16_stability_summary.json` 和 `bf16_stability_gpu.csv`。
这约半分钟的固定输入测试不能替代长期运行、不同负载和完整实机链路测试。

离线训练/测试集评估可运行 `scripts/pi05_local/evaluate.py`，传入本地模型、
完整 canonical 数据集和 tokenizer。它复用 OpenPI 的动作增量、分位数归一化、
状态 token 化和维度补齐；固定原始 episode 划分，关闭数据增强，默认按延迟 0
报告流匹配速度 MSE、5/10 步生成动作 MSE、关节/夹爪物理单位误差与保持当前位姿基线。
新增评估依赖已包含在 `scripts/pi05_local/requirements.txt` 中。

### trimstart 15k 与 RTC 评估

新实验 `fix_trimstart_rtc35_h200_20261021/15000` 已同步到
`/home/user/models/pi05_5060/fix_trimstart_rtc35_h200_20261021/15000`，对应 PyTorch
目录为同级 `15000_pt`。它与上文的旧 `piper_15000_pt` 是不同实验。
新模型训练时开启 RTC，最大延迟 10 步，延迟 0 占 35%。延迟 0 的评估不会改变
模型权重或训练配置。

```bash
/home/user/miniconda3/envs/pi05_5060/bin/python scripts/pi05_local/evaluate.py \
  --source /home/user/src/openpi-5060 \
  --checkpoint /home/user/models/pi05_5060/fix_trimstart_rtc35_h200_20261021/15000_pt \
  --tokenizer /home/user/models/pi05_5060/paligemma_tokenizer.model \
  --dataset /home/user/datasets/lerobot/fix_nopad_trimstart_20261021 \
  --samples 512 --action-samples 0 --rtc-distribution \
  --output analysis/checkpoint_15000_20261021/local_rtc_mse.json
```

`--rtc-distribution` 从检查点 provenance 中读取训练延迟分布，给前缀动作使用
真实标签和时间 0，其余动作按随机流时间加噪；只对待预测后缀计算 MSE。
此模式只评估 flow loss，不生成动作。它在独立模型实例上扩展逐动作时间条件，
不改安装的 OpenPI/Transformers；开始时检查延迟 0 与原入口的数值一致性。
省略该参数并用 `--action-samples 64` 可运行延迟 0 的 5/10 步动作评估。
两种条件都关闭图像增强，不能与带颜色增强的训练日志逐项等同。
RTC 真实前缀评估对应训练目标，不能代替使用上一次预测动作的异步闭环测试。

同步校验和评估结果见 `analysis/checkpoint_15000_20261021/README.md`。

### 本地 8099 推理服务

可视化管理窗口（打开时不会自动启动模型）：

```bash
./local_policy_gui.sh
# 等价：./local_policy.sh gui
```

窗口提供已验证 15k 模型目录选择、本地端口、5/10 次去噪、启动/停止、应用配置
并重启、健康状态、连接地址复制、整卡显存/温度以及实时日志。模型加载和健康检查
在后台处理，不阻塞窗口。服务由原有管理脚本启停；关闭窗口后已启动的服务继续
在后台运行。若希望释放模型显存，先点击“停止服务”。日志只保留近期内容供显示，
完整日志仍保存在 `analysis/pi05_local_8099/server.log`。模型默认保持 BF16 和训练式 RTC。

推荐从项目根目录使用统一入口，默认在后台启动：

```bash
./local_policy.sh                   # 启动；已运行时显示状态
./local_policy.sh status            # 查看 PID、地址和就绪状态
./local_policy.sh logs --follow     # 持续查看日志，Ctrl+C 退出查看
./local_policy.sh stop              # 停止本地 policy
./local_policy.sh restart           # 重启
```

默认仍为当前 15k 模型、`127.0.0.1:8099`、BF16、5 次去噪和训练式 RTC。
脚本自动使用 `pi05_5060` 环境，无须手动激活 Conda。后台进程退出终端后继续运行。
可使用 `./local_policy.sh start --port 8100 --steps 10` 修改启动参数；若已有服务，
需用 `restart` 应用新参数。`--checkpoint /绝对路径/15000_pt` 可覆盖检查点位置，
服务仍会校验 15k Piper RTC 契约。此入口管理单个本地 policy 进程。
管理器只停止 PID 文件、实际命令和当前用户均匹配的进程。

需要前台运行时（Ctrl+C 停止服务）：

```bash
bash scripts/pi05_local/run_server.sh --host 127.0.0.1 --port 8099
```

默认加载新实验 `fix_trimstart_rtc35_h200_20261021/15000_pt`，BF16 混合精度、
5 次去噪、不编译。词嵌入查表放在 CPU，选出的 BF16 向量传到 GPU；启动时
逐值核对查表结果，释放约 1004 MiB 显存。视觉、语言 Transformer 和动作专家
仍在 GPU 上按原精度计算，不使用 NF4 量化。

客户端地址 `ws://127.0.0.1:8099`；健康检查
`curl http://127.0.0.1:8099/healthz`。仅监听本机回环地址。
协议与项目 OpenPI 客户端一致：输入三路 `images`（`cam_high`、
`cam_left_wrist`、`cam_right_wrist`，RGB uint8 CHW/HWC）、14 维 `state`、
可选 `prompt` 和 `client_metadata`。关节输入单位为弧度，夹爪为开度比例。
图像按 stretch 规则缩放至 224×224；默认提示使用数据集原文
`put the bottle on the balck pad`。

输出为 `actions[50,14]`，已反归一化并还原为绝对关节目标，夹爪为开度比例。
动作频率 20Hz，首个动作对应观测后 1 步。不要将其再次作为增量叠加。
握手元数据公开完整动作契约、检查点和 RTC 能力。

服务支持 `rtc_mode=trained`，最大延迟 10 步：前缀按每动作时间 0 固定，
不叠加梯度 RTC 引导。复用现有 `client_metadata.rtc` 协议，支持服务端上一段
预测和客户端规划器的绝对动作前缀，并围绕当前状态重新编码关节增量。
没有启用 RTC 或没有可用前缀时，按延迟 0 同步推理。动态切换 RTC 配置需要
使用现有 `rtc.config` / `config_revision`，与项目客户端一致。

离线请求验证（只读取数据集，不打开相机或 CAN）：

```bash
/home/user/miniconda3/envs/pi05_5060/bin/python scripts/pi05_local/smoke_server.py
```

本轮 24 次 WebSocket 请求全部返回有限的 50×14 动作，包括同步请求、不同状态
下复用上一段的 RTC、客户端规划前缀，以及连续 20 次 RTC 请求。前缀绝对目标
最大误差 1.2e-7；首请求约 0.67 秒，后续中位数约 0.34 秒，包含 RPC 及预处理。
整卡显存约 6.6 GiB。初版全 GPU 查表在桌面并行使用显存时出现 OOM，切换到
CPU 查表后本轮验证未再出现 OOM。这是短时验证，不代表长期负载测试。

当前服务作为独立后台进程运行，PID 记录在
`analysis/pi05_local_8099/server.pid`，启动参数在同目录 `launch.json`，日志为
`server.log`，验证数据为 `smoke_test.json`。`local_policy.sh` 维护 PID 和启动参数，
并识别本轮已有的后台服务；前台入口不维护这些管理文件。
启动服务本身不连接机械臂。

### 三路实时相机完整链路延迟

只读实测入口（使用 `dual_arm` 硬件环境；服务须已启动，相机须未被占用）：

```bash
/home/user/miniconda3/envs/dual_arm/bin/python scripts/pi05_local/benchmark_live.py \
  --left-can can1 --right-can can0 --port 8099 --samples 50
```

脚本只被动接收两臂 CAN 反馈，禁用 SDK 初始化查询，不使能或发送动作。
三路相机按 GUI 保存的左右腕选择器，以 424×240、30fps 并行持续采集，
准备 224×224 stretch 模型输入和 256×256 采集尺寸图像；复用客户端状态转换、
最近帧选择和带时间测量的 WebSocket 传输。RTC 按延迟 7 步开启，上一段动作
只用于只读推理中的前缀测试，没有实际执行进度。

本轮 50 次热请求：相机组读取开始至动作返回的中位数 **370.6ms**，
P95 **391.7ms**；从本次读取状态、取缓存帧至动作返回为 **331.5ms**，
P95 **347.3ms**。三路采集和预处理本身中位数 20.2ms，与推理并行；
请求发出时最旧相机画面的主机读取时间已过去约 37.5ms。不能简单相加各段分位数。
相机时间戳来自主机读取完成，未校准真实曝光或 USB/驱动缓存，且测量不含
轨迹规划和机械臂响应。详细定义及数据见
`analysis/pi05_local_8099/LIVE_LATENCY.md` 和 `live_latency.json`。
