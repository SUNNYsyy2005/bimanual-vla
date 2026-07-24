"""MiniCPM-RobotManip 推理 wrapper 草稿。

背景（2026-07-21 调研结论，详见 E:\\test\\HANDOFF.md 第4节"阶段四"）：
- 官方只发布了推理脚本 vla_infer.py 和权重，没有公开动作维度语义、归一化统计、
  env adapter，因此本文件只能先按官方文档的输入/输出 *形状* 搭骨架。
- 输出的 80 维 action chunk 里，哪几维对应关节/末端位姿/夹爪，需要等实际拿到
  权重和官方 vla_infer.py 源码后再确认，当前先用占位符标注，不要假设具体切分。
- 这个 wrapper 的目标：给统一评测协议 (E:\\test\\protocols\\unified_task_schema.py)
  提供一个可以被同一套评测循环调用的策略接口，和 π0.5 的调用方式对齐，但
  不假设两者动作空间兼容——动作空间差异通过 ActionSpaceSpec 显式声明，
  由上层评测框架决定是否需要额外的动作重映射层（如需要，应该是单独一个
  adapter 模块，不要塞进这个推理 wrapper 里，保持单一职责）。

依赖（尚未安装，需先跑 E:\\test\\jobs\\minicpm_robotmanip_setup.sbatch.draft）：
- transformers==5.7.0, torch==2.6.0（conda 环境 MiniCPMRobotManip）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# 复用统一协议里的类型定义，保持和 π0.5 那边同一套描述动作空间的方式
from protocols.unified_task_schema import ActionSpaceSpec, ActionSpaceType, TensorSpec

MODEL_REPO_ID = "openbmb/MiniCPM-RobotManip"
MAX_NUM_EMBODIMENTS = 32  # 来自官方 config.json，多 embodiment 条件化上限
ACTION_HORIZON = 30
ACTION_DIM = 80
STATE_DIM = 80
IMAGE_RESIZE = (448, 448)

# 待官方公开或我们自行反推确认后再补全具体维度切分。
# 目前明确知道的只有总维度和 horizon，不要在没有确认前编造具体的
# "关节0-8是左臂"这类切分——阶段一在 openpi 上就吃过"假设字段名"的亏。
MINICPM_ACTION_SPACE = ActionSpaceSpec(
    action_type=ActionSpaceType.JOINT_POSITION,  # TODO(待确认): 也可能是delta/末端位姿混合，需读vla_infer.py源码确认
    dimensions=ACTION_DIM,
    tensor=TensorSpec(
        name="minicpm_robotmanip_action_chunk",
        shape=(ACTION_HORIZON, ACTION_DIM),
        dtype="float32",
        description="30-step action chunk, 80-dim per step; exact per-dim semantics unconfirmed as of 2026-07-21",
    ),
    metadata={
        "source": "OpenBMB/MiniCPM-RobotManip config.json",
        "confirmed": False,
        "note": "dimension breakdown (joint/eef/gripper) not yet verified against official source",
    },
)


@dataclass
class MiniCPMObservation:
    """一次推理调用的输入，字段命名对齐官方 vla_infer.py 的参数。"""

    images: list[np.ndarray]  # 至少1张，多视角则传多张，各自 resize 到 IMAGE_RESIZE
    instruction: str
    state: np.ndarray | None = None  # (STATE_DIM,)，未提供则用全0填充（官方默认行为）
    embodiment_id: int = 0


@dataclass
class MiniCPMActionChunk:
    """一次推理调用的输出。"""

    actions: np.ndarray  # (ACTION_HORIZON, ACTION_DIM)
    raw_model_output: Any = None  # 保留原始输出，便于调试/审计


class MiniCPMRobotManipPolicy:
    """MiniCPM-RobotManip 推理封装。

    当前是骨架：__init__/predict 的具体模型加载和前向推理逻辑
    要等 conda 环境和权重下载完成后才能补全并跑通 smoke test，
    不要在没有真实环境验证的情况下臆造 transformers 调用细节。
    """

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._model = None  # TODO: 加载 MiniCPMV4_6ForConditionalGeneration，等环境跑通后补
        self._processor = None

    def load(self) -> None:
        """加载模型权重和处理器。跑通环境后实现。"""
        raise NotImplementedError(
            "等 minicpm_robotmanip_setup.sbatch.draft 跑通、拿到真实的 vla_infer.py 源码后再实现，"
            "不要在没有验证的情况下假设 transformers API 调用方式。"
        )

    def predict(self, obs: MiniCPMObservation) -> MiniCPMActionChunk:
        """单步推理：观测 -> 80维action chunk。"""
        if self._model is None:
            raise RuntimeError("call load() before predict()")
        raise NotImplementedError("待 load() 实现后补全")


def resize_image_for_minicpm(image: np.ndarray) -> np.ndarray:
    """占位：按官方要求 resize 到 448x448。实现时优先用 PIL/cv2，
    保持和官方 vla_infer.py 的插值方式一致（若已知），否则用双线性默认值
    并在 metadata 里记录，避免和 π0.5 那边默认 224 分辨率的对比产生偏差。
    """
    raise NotImplementedError("等确认官方 resize 实现细节后补全")


def smoke_test_stub() -> None:
    """最小连通性测试的占位说明（不是可执行的测试代码）。

    等环境和权重就绪后，应该实现为：
    1. 构造一张随机/真实图像 + 一句简单指令（如"pick up the cube"）
    2. 调用 MiniCPMRobotManipPolicy.predict()
    3. 断言输出 shape == (ACTION_HORIZON, ACTION_DIM)，无 NaN/Inf
    4. 不需要真实机械臂或仿真环境，只验证模型能正常前向推理
    """
    raise NotImplementedError("环境就绪后实现为真正的 pytest / 简单脚本")
