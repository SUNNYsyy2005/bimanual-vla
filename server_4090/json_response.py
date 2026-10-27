"""Strict JSON at the HTTP boundary without changing stored experiment results."""
import math
from flask.json.provider import DefaultJSONProvider


def finite_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


class FiniteJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs['allow_nan'] = False
        return super().dumps(finite_json(obj), **kwargs)
