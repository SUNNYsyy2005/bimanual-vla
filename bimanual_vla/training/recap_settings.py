"""Real-robot RECAP options; value/advantage stages use pinned RLinf."""

from dataclasses import asdict, dataclass
import math

RLINF_COMMIT = "061c4cce80bbab998ef90585475d3f51290a2dfa"


@dataclass(frozen=True)
class RecapSettings:
    enabled: bool = False
    n_step: int = 50
    positive_fraction: float = .3
    unconditional_probability: float = .1
    value_steps: int = 8000
    failure_reward: float = -300.

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("recap_enabled must be boolean")
        for field in ("n_step", "value_steps"):
            if type(getattr(self, field)) is not int or not 1 <= getattr(self, field) <= 1000000:
                raise ValueError(f"RECAP {field} must be a positive integer")
        if not math.isfinite(self.positive_fraction) or not 0 < self.positive_fraction < 1:
            raise ValueError("RECAP positive_fraction must be in (0,1)")
        if not math.isfinite(self.unconditional_probability) or not 0 <= self.unconditional_probability < 1:
            raise ValueError("RECAP unconditional_probability must be in [0,1)")
        if not math.isfinite(self.failure_reward) or self.failure_reward >= 0:
            raise ValueError("RECAP failure_reward must be negative")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_request(cls, payload):
        values = {}
        for name in cls.__dataclass_fields__:
            value = payload.get("recap_" + name)
            if value is None or value == "":
                continue
            if name == "enabled":
                if value not in (True, False, "true", "false", "1", "0"):
                    raise ValueError("recap_enabled must be boolean")
                value = value in (True, "true", "1")
            elif name in ("n_step", "value_steps"):
                if isinstance(value, bool) or str(int(value)) != str(value):
                    raise ValueError(f"RECAP {name} must be an integer")
                value = int(value)
            else:
                value = float(value)
            values[name] = value
        return cls(**values)
