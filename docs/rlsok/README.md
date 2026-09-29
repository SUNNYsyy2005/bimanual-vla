# 在 bimanual-vla 中使用 RLSOK v1.5.8

这份说明适用于已经准备好 **RLSOK、经操作员确认的设备角色基线，以及项目侧 resolver** 的机器人工作站。RLSOK 检查保存的配置与新发现的设备身份；bimanual-vla 在打开相机和 CAN 之前消费检查结果，并在推理运行中监测设备故障。

## 当前接入状态

本分支包含 bimanual-vla 的接入点、GUI 设置、运行期保护和无真机测试。**项目侧 resolver 可执行文件和私有基线不在仓库中**。只有配置了 resolver 路径，RLSOK 检查才会启用；路径为空时保持原有设备连接流程。安装 RLSOK 本身、接上真机，或只填写 `rlsok` CLI 路径，都不能替代项目侧 resolver。

```text
经确认的角色基线 + 本次新发现的设备清单
                    │
                    ▼
       RLSOK capture-setup / review-setup
                    │
                    ▼
      项目侧 resolver：核对 GUI 选择并输出最小 JSON
                    │
                    ▼
       bimanual-vla 的数采 / 推理设备入口
```

RLSOK 的 `UNCHANGED` 仅表示保存的配置在所选范围内没有语义变化，**不是机械臂运动许可**。推理执行仍需原有的 Dashboard `EXECUTE` 授权和客户端安全检查。

## 接通前准备

1. 在机器人工作站安装 **RLSOK v1.5.8**，并根据 [RLSOK Piper 角色说明](https://github.com/realitywarden/rlsok/blob/v1.5.8/docs/piper-confirmed-roles.md)准备私有 `roles.yaml`。为每个活动角色确认相机身份和视频流、CAN 适配器身份、左右臂对应关系。相机无可读序列号时，只能使用人工确认的 `usb_path` 连同 `interface`、`videoIndex`；USB 路径标识端口，不能识别插在该端口的新相机。
2. 使用实际部署的项目源码版本制作 `prepare-piper-setup` 选定配置，检查其内容后执行 `capture-setup`、`approve-setup`，保存原始基线。`prepare-piper-setup` 生成的 `operator-inventory.json` 是导入的选定配置，**不是新的硬件发现结果**。后续检查应每次运行只读的 `discover-setup-devices` 获取新清单。
3. 实现项目侧 resolver，使其按照 [设备接口契约](DEVICE_RESOLVER_GUI_GLUE.md)读取一条 stdin JSON 请求，调用 RLSOK 对私有基线和本次清单做复核，再向 stdout 输出一条最小 JSON 结果。保留 RLSOK 的 `UNCHANGED`、`NEEDS_MATERIAL`、`REVIEW_REQUIRED` 判定。`review-setup` 对后两种结果退出码为 1；resolver 应读取 `report.json`，不能把它们混成普通命令失败。
4. resolver 必须检查当前 GUI/CLI 选择是否仍指向已确认角色。若设备重编号，RLSOK 的 `resolve-setup` 只会生成**新的配置副本**；它不会自动修改 GUI 设置或批准运动。保留原始基线，需要时对新副本做明确复核。

基线、完整设备清单和硬件序列号保存在工作站的私有目录。GUI 只接收角色、设备定位符、复核状态及简短原因码；不要把 resolver 的完整报告或 stderr 写入 GUI 日志。

## 配置并启动

在 GUI 的 **Device settings → RLSOK resolver** 填入项目侧 resolver 的**绝对可执行路径**。也可以在启动 GUI 前设置环境变量：

```bash
export BIMANUAL_VLA_RLSOK_RESOLVER=/absolute/path/to/your-project-resolver
./start_gui.sh
```

GUI 已保存的 resolver 设置优先于环境变量，因此切换工作站配置时检查 Device settings。直接运行 `bin/bimanual-vla rtc-client` 时，可使用 `--rlsok-resolver` 指定同一可执行文件。这里填写的是项目侧 glue，**不是** RLSOK 的 `rlsok` CLI。

`start_gui.sh` 仅启动界面。角色选择完成后，检查发生在以下位置：

| 操作 | 检查与拒绝后的行为 |
| --- | --- |
| 数采 Connect devices | 在 CAN 和相机打开前复核；拒绝时保持断开，返回 Device settings。 |
| 数采手腕相机切换 | 重新打开相机前复核；拒绝时保留或恢复原映射。 |
| GUI Start inference | 创建推理子进程前复核；子进程又在首次打开 CAN 前复核。拒绝时不启动推理。 |
| 直接启动 rtc-client | 首次打开 CAN 前复核；拒绝时以状态码 2 退出。 |

成功时 GUI 显示 `RLSOK UNCHANGED (saved setup)`。具体调用顺序和返回路径见 [启动检查说明](START_GUI_PREFLIGHT_GLUE.md)。

## 判定与运行期处理

| RLSOK `review-setup` 判定 | bimanual-vla 行为 |
| --- | --- |
| `UNCHANGED` | 在新清单、选择字段和解析出的定位符均有效时，继续进入现有设备连接流程。 |
| `NEEDS_MATERIAL` | 拒绝连接或启动；补齐缺失、含糊或不匹配的设备材料。 |
| `REVIEW_REQUIRED` | 拒绝连接或启动；人工复核角色、身份依据、模式、源码或其他选定配置变化。 |

推理期间，客户端每个控制周期检查后台相机采集状态与最新帧时间，同时检查 Piper CAN 反馈。相机断流、反馈过期或异常状态会丢弃旧 chunk、结束本次推理会话，并跳过故障后的自动回位。配置了 resolver 时，还会在后台约每 5 秒用新清单复核一次角色；即使 RLSOK 对相同身份给出 `UNCHANGED`，**当前会话的设备定位符发生变化仍需停止并重新打开设备**。复核子进程不占用 20 Hz 控制线程。

RLSOK 的 sysfs/udev 发现不验证相机是否持续出图，也不能从 CAN 适配器序列号证明背后连接的是哪只 Piper。实时数据健康检查由 bimanual-vla 单独承担；USB 路径绑定也无法识别同一端口上的无序列号相机被替换。

## 无真机验证

```bash
./scripts/test_rlsok_offline.sh
```

该命令用临时假 resolver 走真实的子进程与 JSON 接口，并用 mock 隔离相机、CAN 和策略启动。覆盖三种复核判定、过期证据、数采/推理入口及运行期端点变化。它**不调用真正的 RLSOK CLI，也不证明私有基线或真实设备已验证**。测试范围和限制见 [无真机测试说明](OFFLINE_TESTING.md)。

## 常见拒绝结果

| 显示内容 | 下一步 |
| --- | --- |
| `RLSOK NEEDS_MATERIAL: ...` | 查看私有报告，补齐清单、身份绑定或当前设备选择；重新检查。 |
| `RLSOK REVIEW_REQUIRED: ...` | 人工复核变化，必要时制作新的配置副本和基线；不要自动覆盖旧基线。 |
| `resolver unavailable` / `invalid or stale evidence` | 检查可执行路径、协议字段、请求 ID、版本和本次清单时间；修复后重新启动检查。 |
| `runtime endpoint changed` | 结束当前会话，确认连接与角色后重新启动。 |

本分支的协议字段和脱敏示例以 [设备接口契约](DEVICE_RESOLVER_GUI_GLUE.md)为准；RLSOK 原生命令语义以 [v1.5.8 保存配置复核说明](https://github.com/realitywarden/rlsok/blob/v1.5.8/docs/saved-setup-review.md)为准。
