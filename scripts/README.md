# 脚本索引

- `smoke/`：机械臂、相机、Policy 和本地推理冒烟测试。
- `models/`：OpenPI checkpoint 下载和准备。
- `maintenance/`：旧数据转换、迁移和修复。
- `analyze_rollout_vs_expert.py`：rollout 与专家动作对比。
- `replay_lerobot_actions.py`：在 RoboTwin 中回放 LeRobot 动作。
- `monitor.py`：集群任务监控。
- `query_h100_h200_resources.sh`：Dashboard 使用的 Slurm 资源查询。

Python 脚本应从仓库根目录用模块方式运行，例如：

```bash
python -m scripts.smoke.robot_smoke_test
python -m scripts.models.download_openpi_checkpoint --help
python -m scripts.maintenance.convert_output_arm_npz --help
```
