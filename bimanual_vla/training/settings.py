"""Independent action expert, language LoRA, and RTC training settings."""

import json
from bimanual_vla.training.rtc_sampling import validate_delay_weights

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FinetuningSettings:
    action_expert_tuning: str = "full"
    llm_lora: bool = False
    rtc_training: bool = False
    rtc_training_max_delay: int = 8
    rtc_training_delay_weights: tuple[float, ...] | None = None
    image_resize_mode: str = "pad"
    train_vision: bool = False
    train_llm: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.train_vision, bool) or not isinstance(self.train_llm, bool):
            raise ValueError("train_vision and train_llm must be booleans")
        if self.train_llm and self.llm_lora:
            raise ValueError("full LLM training and LLM LoRA are mutually exclusive")
        if self.image_resize_mode not in {"pad", "stretch"}:
            raise ValueError("image_resize_mode must be pad or stretch")
        if self.action_expert_tuning not in {"full", "lora"}:
            raise ValueError("action expert tuning must be full or lora")
        if not isinstance(self.llm_lora, bool) or not isinstance(self.rtc_training, bool):
            raise ValueError("LLM LoRA and RTC training settings must be booleans")
        if type(self.rtc_training_max_delay) is not int or not 1 <= self.rtc_training_max_delay < 50:
            raise ValueError("RTC training maximum delay must be between 1 and 49 action steps")

        object.__setattr__(self, "rtc_training_delay_weights",
                           validate_delay_weights(self.rtc_training_delay_weights, self.rtc_training_max_delay))

    def validate_model(self, variant: str) -> None:
        if self.rtc_training and variant != "pi05":
            raise ValueError("training-time RTC action conditioning currently requires pi05")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def legacy(cls) -> "FinetuningSettings":
        return cls(action_expert_tuning="lora", llm_lora=True)

    @classmethod
    def resolve(cls, requested: Mapping[str, Any], saved: Mapping[str, Any] | None = None,
                *, legacy: bool = False, resume: bool = False) -> "FinetuningSettings":
        defaults = cls.legacy() if legacy else cls()
        previous = cls(**dict(saved)) if saved else defaults
        values = previous.to_dict()
        values.update({key: value for key, value in requested.items()
                       if key in values and value is not None})
        result = cls(**values)
        if resume and result != previous:
            raise ValueError("full-state resume must retain the checkpoint's finetuning/RTC settings")
        return result

    def command_args(self) -> list[str]:
        args = ["--action-expert-tuning", self.action_expert_tuning,
                "--llm-lora" if self.llm_lora else "--no-llm-lora",
                "--rtc-training" if self.rtc_training else "--no-rtc-training",
                "--rtc-training-max-delay", str(self.rtc_training_max_delay)]
        if self.rtc_training_delay_weights is not None:
            args += ["--rtc-training-delay-weights", json.dumps(self.rtc_training_delay_weights)]
        args += ["--image-resize-mode", self.image_resize_mode,
                 "--train-vision" if self.train_vision else "--no-train-vision",
                 "--train-llm" if self.train_llm else "--no-train-llm"]
        return args
