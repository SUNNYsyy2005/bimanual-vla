# Quest VR 数采（无 ROS、无 Ruckig）

在现有数采 GUI 中选择 `quest_vr`，沿用相机、单臂／双臂选择和 NPZ 保存流程。
默认 `passive` 仍只读取反馈。VR 可用于独立人工数采，也可通过推理页面的
[RECAP 接管](RECAP.md)纠正模型执行。

控制链路：Quest 官方 APK → ADB → 手柄相对位姿 → 官方 Quest IK → Piper `JointCtrl`。
保留官方 Pinocchio/CasADi 位姿误差权重与 0.13 m 工具变换，将关节正则项改为相对上一条已接受指令，不使用固件 EEF 逆解，
不经过部署客户端、Ruckig、RTC、EEF 平滑或逐步关节限幅。
独立环境仍安装 Ruckig 以兼容 GUI 已有的硬件回放功能；VR 数采路径不调用它。
控制线程最高 60 Hz，每份新手柄数据求解一次；计算超时会丢弃目标，不补发积压目标。

## 安装

为避免改变 `dual_arm` 的依赖，使用独立 Conda 环境：

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda env create -f environment-quest-collection.yml
conda activate quest_collection
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONPATH=. python scripts/setup_quest_assets.py
```

需要可执行的 `adb`（Android Platform Tools）；若未安装，可在激活上述环境后安装到环境内，
不修改系统目录：

```bash
curl -fL https://dl.google.com/android/repository/platform-tools-latest-linux.zip -o /tmp/quest-platform-tools.zip
unzip -q /tmp/quest-platform-tools.zip -d "$CONDA_PREFIX"
chmod +x "$CONDA_PREFIX/platform-tools/adb"
ln -s ../platform-tools/adb "$CONDA_PREFIX/bin/adb"
adb version
```

无需 ROS、catkin、rospkg 或 Meshcat。
脚本只下载并校验固定版本的 URDF、STL、APK 和上游说明／许可证，约 14 MB。
模型默认为 `~/.cache/bimanual-vla/questVR_ws/src/Piper_ros/src/piper_description`。
GUI 的模型目录可留空使用默认值，也可指定自己的 `piper_description` 目录。

Quest 2／3 开启开发者模式，通过 USB 连接并在头显中允许 USB 调试：

```bash
adb devices
adb install -r ~/.cache/bimanual-vla/questVR_ws/src/oculus_reader/APK/teleop-debug.apk
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m bimanual_vla.collection.gui
```

连接 `quest_vr` 时会自动启动或唤起官方遥操 APK
（`com.rail.oculus.teleop/.MainActivity`）；已在运行时不强制重启。
保持头显和手柄跟踪正常。系统关机弹窗、权限弹窗仍需在头显中处理，程序不会自动确认。
GUI 打开或切换采集控制来源时会后台扫描 ADB；只有一台已授权设备时自动填入序列号。
`bash start_gui.sh` 和 `bin/bimanual-vla collect-gui` 会优先加载已安装的
`quest_collection` Conda 环境，包括 PATH 和本机库路径；启动终端会显示实际 Python
和 adb 路径。若设置了 `BIMANUAL_VLA_PYTHON`，则以显式指定的解释器为准。
多台设备时从下拉框选择 VR 对应的序列号，插拔或完成授权后点击“刷新 VR”。
界面会提示未授权、离线、USB 权限不足或缺少 adb；扫描不会连接或移动机械臂。
下拉框也支持手动输入，Wi-Fi ADB 已连接的设备可填 `IP:端口`。
连接时再次核验序列号；留空只在恰有一台已授权 ADB 设备时自动选择，所选设备失联
不会自动改连另一台。命令行及 RECAP 的 Quest 读取器使用同一选择规则。
APK 缺失或启动失败会明确报错；启动成功不等于跟踪就绪，空手柄数据仍禁止接管。
程序不会自动安装 APK、更改 ADB 网络配置或移动机械臂回零。
只读诊断可使用 `QuestReader(..., launch_app=False)`，不改变头显前台应用。
采集 GUI 的固定文案、按钮及 VR 诊断提示使用英文。主界面保留数据集、任务、
连接和录制操作；臂模式、格式、模型目录、ADB 选择与保存姿态收进默认折叠的 `Settings`，
操作说明放在 `Controls help`。VR 控制与实时输入只在选择 `quest_vr` 时显示。
连接 VR 采集后，右侧 `VR input` 区约每 100 ms 刷新左右手跟踪状态、
X/Y/A/B 按键的按下／松开状态、扳机数值（0 为松开，1 为扣到底），以及数据年龄。
输入显示直接读取现有 VR 数据流；超过 0.2 s 无新鲜数据、ADB 出错或断开连接时清空旧按键／扳机值。

## 操作

1. 停止其他推理、回放、Pika、主从臂及 VR 控制程序，使用现有流程确认 CAN／相机映射。
2. 选择 `quest_vr` 后连接设备。普通 VR 手柄接管不自动使能或切换模式，机械臂
   必须已通过独立确认的流程进入使能的 CAN/MOVE_J 状态。显式 **Return to saved home**
   是唯一例外：它先检查双臂反馈、驱动故障和 CAN 所有权，再用实测姿态执行一次
   与推理回位相同的低速 CAN/MOVE_J 使能保持握手；握手失败则不规划、不发送回位路径。
3. 先松开左右按钮。按住左手 **Y** 控制左臂，按住右手 **B** 控制右臂。
   每次按下都从该臂的实测关节 FK 和当前手柄位置重新对齐；首条指令保持实测姿态。
4. 松开按钮保持当前位置。调整手柄到舒适位置后重新按住，可继续操作。
   独立数采模式中 **X** 单击开始／结束采集；结束后在 GUI 确认成功、失败或放弃。
   **A** 长按 1 秒回到本次连接时的双臂关节姿态，复位前必须松开 Y/B。
   启动、断流或复位后需先松开 X/A 才能重新触发，持续按住不会重复执行。
   这些采集快捷键不作用于推理中的 RECAP 接管。
5. 扳机采用线性绝对开度：`opening_fraction = 1 - trigger`，
   `opening_m = 0.07 × opening_fraction`。松开全开、扣到底全闭，中间线性变化，
   不加二值阈值或低通。每次接入先保持实测开度，扳机经过该开度（容差 2%）后接入，
   此后保持严格线性映射。松开 Y／B 同时保持夹爪。
6. 按现有按钮／空格开始和停止录制，再确认保存。停止录制后仍能遥操；
   断开设备会停止 VR 控制并保持当前位置。GUI **VR STOP** 请求急停，之后需人工检查并重新连接。

GUI 的 **Return to saved home** 使用与推理页显式回位按钮共用的
`~/.config/bimanual-vla/return_pose.json`，并核对左右臂与 CAN 标识；
`Settings` 中的 **Save current pose as home** 在未录制、松开 Y/B 时用当前反馈更新该文件，
直接保存且不弹覆盖确认；已有文件会先备份为同目录的 `return_pose.before_save_<id>.json`。
普通推理停止后的自动恢复使用本次启动时测得的姿态，保存在独立临时文件中；
不加载或改写上述固定姿态文件。推理页的 **Return to saved home** 才使用固定姿态文件，
推理尚未退出时先正常停稳并释放控制连接，再执行显式固定姿态回位。
设定文件不会改变手柄 A 对应的连接姿态。两种复位都保留当前夹爪目标，
不恢复文件中的夹爪开度。正在录制时先结束回合，完成保存或放弃后再复位。

复位由原有 VR 指令线程执行，无第二个 CAN 写入者。先验证停稳，再按关节空间
五次插值返回，目标速度上限 0.10 rad/s、加速度上限 0.30 rad/s²；此插值仅用于显式复位，
正常遥操不加 Ruckig。目标及路径检查驱动限位、自碰撞，运行中检查反馈和跟踪误差；
逐周期检查目标步长，实测速度超过 0.45 rad/s、跟踪误差超过 0.05 rad 或调度中断超过
0.15 s 时终止。到位并停稳后重新等待 Y/B 松开／按下进行标定。复位期间不能开始采集。

2026-10-30 在线测试出现异常运动后，自动使能／切换模式路径已关闭；上述低速复位
只完成离线验证，尚未完成修改后的真机验收。

显式回位与推理回位使用相同的目标限位处理：保存姿态中的越界关节先裁剪到当前驱动限位，
例如 J2=-1.9°、下限为 0° 时按 0° 规划；不会因为保存目标的这个偏差直接拒绝回位，
也不会改写保存文件。裁剪后仍通过原有单写入线程执行低速路径、碰撞和反馈检查。
这项裁剪仅作用于请求的回位目标；不把越界实测反馈伪装为有效反馈，不绕过驱动故障检查。

断流超过 0.2 s、手柄丢失、坐标重定位跳变或 IK 不安全时，保持实测位置，必须松开再按住按钮。
保留驱动真实限位、反馈新鲜度、CAN 发送故障锁定和驱动状态检查。
保持指令同样检查限位；实测关节若已越界（例如 J2 为负而驱动下限为 0°），
GUI 显示具体关节、实测值和限位，撤销该臂旧保持目标，不发送该臂的新指令。
此类拒绝不会因再次发送非法保持指令而终止 VR 线程，也不会自动裁剪实测姿态或回放旧目标。
先检查机械臂位置／零点，使反馈恢复到驱动限位内，再松开并重新按住 Y/B 接管。
未通过首次接管检查的机械臂不会开始发送保持指令。
IK 求解器允许的极小边界误差（不超过 1e-6 rad）在返回前投影到限位内，
并对投影后的实际目标重新检查自碰撞；实测反馈不使用此修正。
IK 解的相邻关节跳变超过 0.20 rad 时拒绝该解并保持（视为分支跳变，需松开重按）。
机械臂跟不上手不再拒绝：写入线程每个周期（约 60 Hz）以软件关节速度上限
1.2 rad/s 朝最新 IK 目标插值，并限制下发指令领先实测关节不超过 0.35 rad，
机械臂追上后继续跟随，无需松开离合重新标定。每次按下离合接管时，先下发实测
姿态保持，再把固件 MOVE_J 速度设为 80%（保持 CAN/MOVE_J 模式不变），不再沿用
回零工具 1% 或部署默认 10% 等残留速度。两项均为 `QuestController` 参数
（`tracking_speed_rad_s`、`teleop_speed_pct`）。
每次发送前检查模型自碰撞和限位。自碰撞沿用官方远端链节与基座／近端链节检查范围，
不包含双臂互碰、桌面或外部物体检测。

模型坐标及工具长度必须与实物一致。正式采集前应验证左右映射、方向及夹爪行程。
官方向零位的正则项在实测边界姿态上曾导致静止目标也出现约 0.68 rad 的关节变化。
本接入保留正则权重，但参考改为上一条已接受的关节指令；每次重新接管时以实测关节初始化。
求解失败或目标被拒绝不推进参考，不在失败时重置零位暖启动。仍保留跳变拒绝，
不承诺所有目标都严格到位或全局连续。

## 数据与诊断

训练用 `state`／`actions` 沿用原数采契约：关节是下一帧实测动作，EEF 是下一帧实测位姿
加夹爪目标。不会将未经执行的手柄目标混入标准动作。

**已知限制：** 夹爪“同帧命令”取自 SDK 的 `GetArmGripperCtrl()` 缓存，Quest 自己发出的
`GripperCtrl` 不会进入该缓存（python-can 默认不回收自身帧）。因此 Quest 回合的夹爪动作
实际回退为下一帧实测开度；夹住物体时它是物体宽度，而不是扳机给出的闭合命令。
NPZ 额外包含 `collection_control_source`、`quest_source_commit`、`quest_telemetry_json`。
JSON 记录接入、拒绝、故障和成功发送的指令，包括源手柄时间、实际主机发送时间、
手柄／EEF 目标、关节目标及发送前实测反馈。其独立时间轴与相机 30 Hz 时间轴分开记录。
这里的发送时间确认主机提交，不代表电机完成运动。现有导出器继续使用标准动作字段。

本地锁与回放器共用，CAN 监控可发现来自总线另一端的控制帧；它不能阻止所有不遵守锁的
外部本机程序。因此开始 VR 控制前仍须停止其他写入程序。

## 来源

IK、坐标变换和 APK 协议参考 [AgileX questVR_ws](https://github.com/agilexrobotics/questVR_ws)，
固定版本 `393a32b26dfe2b9f4151f3cdfcfe9ba4d56d3a78`。
上游模型的许可证随下载保存在 `src/Piper_ros/LICENSE`；项目通过独立适配层实现无 ROS 接入。
