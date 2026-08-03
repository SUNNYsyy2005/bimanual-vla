# Piper GUI 数据采集、整理与上传操作手册

本文用于当前采集电脑上的日常操作。GUI 按钮和字段名称保持英文，说明文字使用中文。

项目目录：

```text
/home/user/dual_ARM_project/arm_collect/bimanual-vla
```

默认原始数据目录：

```text
/home/user/dual_ARM_project/arm_collect/bimanual-vla/episodes_piper_v21
```

数据服务器：

```text
http://192.168.101.9:8090
```

## 1. 启动前检查

### 1.1 检查机械臂 CAN

```bash
ip -br link show can0
timeout 2 candump -L can0
```

正常要求：

- `can0` 状态为 `UP`。
- `candump` 能持续收到机械臂反馈。
- CAN 波特率为 `1000000`。

如果 `can0` 未启动，使用当前 Piper SDK 的激活脚本重新激活，再重复检查。没有实时 CAN 反馈时不要开始采集或复位机械臂。

### 1.2 检查相机

```bash
v4l2-ctl --list-devices
```

USB 重新插拔后 `/dev/videoN` 编号可能变化。GUI 会使用稳定的设备路径，并在连接成功后显示其实际解析到的 `/dev/videoN`：

- `Third-Person Camera`：外部全局视角。
- `Wrist Camera`：机械臂末端视角。

最终以 GUI 实时画面和标题中显示的编号为准，不要只记住上一次的 `video8`、`video16` 等编号。

## 2. 启动 GUI

```bash
cd /home/user/dual_ARM_project/arm_collect/bimanual-vla
bash start_gui.sh
```

建议配置：

| GUI 字段 | 建议值 | 说明 |
|---|---|---|
| `CAN Interface` | `can0` | 当前执行臂 CAN 接口 |
| `Capture Rate (Hz)` | `20` | 数据采集频率 |
| `Camera Rate (Hz)` | `30` | 相机原始流频率 |
| `Output Directory` | `.../episodes_piper_v21` 或新的批次目录 | NPZ 保存位置 |
| `Task Name` | 例如 `pick_cube` | 任务标识 |
| `Instruction` | 例如 `pick up the cube` | 训练使用的指令 |

## 3. 采集 episode

### 3.1 连接设备

1. 确认 CAN、相机、输出目录、任务和 instruction。
2. 点击 `Connect`。
3. 检查两路实时画面没有颠倒、冻结或黑屏。
4. 检查画面标题或状态栏显示的实际 `/dev/videoN`。
5. 检查 `Live Robot Pose` 持续更新。
6. 检查机械臂是否位于本次任务的安全起始位姿。

相机画面不正确时先断开连接，再调整相机选择，不能在正在采集时更换设备配置。

### 3.2 开始和停止

1. 恢复场景和机械臂起始状态。
2. 点击 `Start Episode`。
3. 完成一次完整、自然的任务操作。
4. 点击 `Stop Episode`。
5. 根据结果选择：

   - `Save as Success`：任务成功，保存并用于后续导出。
   - `Save as Failure`：任务失败但仍希望保留分析。
   - `Discard`：误操作、空采集、严重遮挡或设备异常，不保存。

采集注意事项：

- 每个 episode 只做一次完整任务。
- 开始和结束可以保留短暂稳定画面，但不要长时间静止。
- 确认夹爪开合、机械臂运动和两路图像都被记录。
- 不要在一个 episode 中途修改 instruction。
- 保存后的文件名为 `ep_XXXX.npz`。

## 4. 回放和删除异常数据

### 4.1 回放检查

在 `Saved Episodes` 中选择文件，然后点击 `Replay Selected`。至少检查：

- 第三视角与腕部视角是否正确。
- 图像是否连续，是否存在长时间冻结。
- 机械臂运动与状态变化是否同步。
- 夹爪状态是否随实际开合变化。
- instruction 和成功/失败标签是否正确。

### 4.2 删除本地 episode

1. 在 `Saved Episodes` 中单选或多选异常文件。
2. 点击 `Delete selected data`。
3. 在英文确认窗口中点击 `Yes`。

删除操作不会永久擦除文件，而是移动到：

```text
<Output Directory>/.trash/<时间戳>/
```

需要恢复时，在确认目标目录没有同名文件后执行：

```bash
mv <Output Directory>/.trash/<时间戳>/ep_XXXX.npz <Output Directory>/
```

episode 编号按当前目录中的最大编号继续。例如保留到 `ep_0063.npz` 时，下一个文件是 `ep_0064.npz`。删除中间编号不会自动重排已有文件。

## 5. 转换 NPZ 为 LeRobot

1. 停止当前 episode。
2. 点击 `Convert / upload dataset`。
3. 检查 `NPZ source`，它会自动使用当前 `Output Directory`。
4. 填写 `Dataset name`，例如 `piper_v1`。
5. 保持 `FPS` 为采集时使用的 `20`。
6. 点击 `Convert NPZ to LeRobot`。

只转换时不需要填写服务器 token。

程序只导出符合要求的 Success episode。转换结果缓存在：

```text
~/.cache/bimanual-vla/uploads/exports/
```

单臂 delivery 数据会导出为服务器现有数据集使用的兼容布局：`state`、`actions`、`image`、`wrist_image`。joint 或双臂数据继续使用相应的 canonical LeRobot 布局。

成功日志包含：

```text
PREPARED_LEROBOT_PATH=...
LeRobot preparation complete: ...
```

可在终端进行本地结构复查：

```bash
cd /home/user/dual_ARM_project/arm_collect/bimanual-vla
python check_pi05_dataset.py <PREPARED_LEROBOT_PATH>
```

正常结果以 `OK LeRobot v2.1` 开头。

### 转换选项

- `Allow incomplete gripper coverage`：允许夹爪数据只有单一状态。正常采集应关闭；只有确认任务本来就不需要夹爪变化时才开启。
- `Rebuild conversion/archive cache`：强制重新转换并重新构建上传包。正常重复上传时关闭；更新转换代码、怀疑缓存异常或明确需要重建时开启。

源 NPZ 文件有新增、删除或修改时，工具会自动使用新的缓存签名。

## 6. 上传服务器

### 6.1 字段填写

在 `Dataset conversion and upload` 窗口填写：

| 字段 | 填写内容 |
|---|---|
| `NPZ source` | 自动显示当前原始 NPZ 目录，只读 |
| `Dataset name` | 服务器上的数据集名称，例如 `piper_v1` |
| `Server URL` | `http://192.168.101.9:8090` |
| `Server token` | 服务器当前有效 token，不要写入文档或 Git |
| `Upload workers` | 推荐 `4` |
| `Install mode` | 通常选择 `merge` |

`Install mode` 含义：

- `merge`：把本批次 episode 追加到服务器同名数据集；同名数据集不存在时会直接安装。日常增量上传使用此项。
- `install`：只允许安装一个服务器上不存在的新数据集；同名数据集已存在时失败。
- `overwrite`：用本次上传完整替换服务器同名数据集。存在数据覆盖风险，仅在明确需要替换时使用。

注意：`Dataset name` 决定服务器最终名称。填写 `test` 就会上传到 `test`，填写 `piper_v1` 才会上传到 `piper_v1`。

### 6.2 执行上传

点击 `Convert if needed and upload`。

上传流程依次为：

1. 复用或生成 LeRobot 数据。
2. 构建未压缩 tar。
3. 计算 SHA256。
4. 分块上传。
5. 服务器组装归档。
6. 服务器执行结构和 LeRobot/OpenPI loader 校验。
7. 服务器原子安装或合并数据集。

只有看到类似以下日志才算真正成功：

```text
Dataset install complete: piper_v1
```

或：

```text
Dataset merge complete: piper_v1
```

`chunk ... 100.0%` 只表示文件传输完毕，不代表服务器校验和安装已经成功。

上传完成后打开 Dashboard：

```text
http://192.168.101.9:8090
```

检查数据集名称、episode 数量、frame 数量、schema、相机字段和 instruction。

## 7. 常见上传报错

### `HTTP 401: UNAUTHORIZED`

含义：已经连接到服务器，但服务器拒绝当前 token。

处理：

1. 从服务器管理员处获取当前有效 token。
2. 清空并重新填写 `Server token`。
3. 确认 `Server URL` 是 `http://192.168.101.9:8090`。
4. 重新点击上传。

服务器重启或重新部署后 token 可能发生变化。

### 上传到 `100.0%` 后出现 `HTTP 400`

含义：所有分块已经上传，但服务器在组装、结构校验、loader 校验或合并安装阶段拒绝了数据；此时不能视为上传成功。

当前上传脚本会在失败后自动查询上传状态，并在进度窗口打印 `error`、`structural_validation` 或 `loader_validation` 的具体内容。再次执行相同上传会复用已经上传的分块，只重试服务器完成步骤。

先运行本地复查：

```bash
python check_pi05_dataset.py <PREPARED_LEROBOT_PATH>
```

然后从进度日志中找到：

```text
Upload <UPLOAD_ID>: ...
```

可查询服务器保存的详细校验输出：

```bash
read -rsp "Server token: " BIMANUAL_VLA_SERVER_TOKEN
echo
curl -sS \
  -H "Authorization: Bearer $BIMANUAL_VLA_SERVER_TOKEN" \
  "http://192.168.101.9:8090/api/uploads/<UPLOAD_ID>" \
  | python -m json.tool
unset BIMANUAL_VLA_SERVER_TOKEN
```

重点查看返回内容中的：

- `error`
- `structural_validation`
- `loader_validation`

如果本地检查通过而服务器检查失败，通常需要根据上述服务器输出检查服务器代码版本、Python/LeRobot 环境或目标数据集兼容性。修复服务器问题后重新执行相同上传即可；相同归档的已上传分块会被复用，不必从零开始传输。

### `dataset already exists`

- 需要追加新数据：选择 `merge`。
- 确实要完整替换：选择 `overwrite`，并先确认服务器现有数据可以被覆盖。
- 需要创建另一个数据集：修改 `Dataset name` 后使用 `install`。

### 数据集不兼容

同一个服务器数据集不能混合不兼容的数据合同，例如不同的 schema、state/action 维度、FPS 或相机字段。应使用新的 `Dataset name` 分开上传，或统一转换格式后再合并。

## 8. 每次操作检查清单

采集前：

- [ ] `can0` 为 `UP` 且有实时反馈。
- [ ] 两路 RGB 相机画面和实际 `/dev/videoN` 正确。
- [ ] Capture Rate 为 `20 Hz`。
- [ ] Output Directory、Task Name 和 Instruction 正确。
- [ ] 机械臂和场景位于安全起始状态。

采集后：

- [ ] 每个 episode 已正确保存或丢弃。
- [ ] 已回放抽查图像、动作和夹爪状态。
- [ ] 异常数据已用 `Delete selected data` 移到 `.trash`。
- [ ] `check_pi05_dataset.py` 本地检查通过。
- [ ] Dataset name 是准备长期使用的服务器名称。
- [ ] 日常增量上传使用 `merge`。
- [ ] 日志出现 `Dataset install/merge complete`。
- [ ] Dashboard 中 episode 和 frame 数量符合预期。

## 9. 数据安全

- 不要把服务器 token 写进脚本、截图、聊天记录或 Git。
- 不要把大型 NPZ、LeRobot 缓存或 tar 文件提交到 Git。
- `overwrite` 会替换服务器同名数据集，使用前必须确认。
- 服务器合并、删除或修改 episode 后，旧 normalization stats 会失效，训练前应重新计算 norm stats。
- 删除本地原始 NPZ 前，先确认服务器数据可用并且已有独立备份。
