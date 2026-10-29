"""Serializable snapshots of configurable technical-indicator parameters."""

import math

from config import settings
from config.overrides import get_setting


PARAMETER_NAMES = (
    "SMA_SHORT", "SMA_LONG", "EMA_SHORT", "EMA_LONG", "MACD_FAST", "MACD_SLOW",
    "MACD_SIGNAL", "ADX_PERIOD", "RSI_PERIOD", "RSI_OVERSOLD", "RSI_OVERBOUGHT",
    "STOCH_K", "STOCH_D", "STOCH_SMOOTH", "STOCH_OVERSOLD", "STOCH_OVERBOUGHT",
    "BB_PERIOD", "BB_STD",
)
THRESHOLDS = {"RSI_OVERSOLD", "RSI_OVERBOUGHT", "STOCH_OVERSOLD", "STOCH_OVERBOUGHT"}


def capture_parameters(overrides=None, *, session=False) -> tuple:
    """Resolve defaults once; subsequent computation uses only this snapshot."""
    getter = get_setting if session else lambda name: getattr(settings, name)
    values = {name: getter(name) for name in PARAMETER_NAMES}
    pairs = list((overrides or {}).items()) if isinstance(overrides, dict) else list(overrides or ())
    if len(dict(pairs)) != len(pairs) or set(dict(pairs)) - set(PARAMETER_NAMES):
        raise ValueError("Indicator parameters must be unique supported names")
    values.update(pairs)
    for name, value in values.items():
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        if name not in THRESHOLDS | {"BB_STD"}:
            if int(value) != value:
                raise ValueError(f"{name} must be an integer")
            values[name] = int(value)
    for low, high in (("RSI_OVERSOLD", "RSI_OVERBOUGHT"), ("STOCH_OVERSOLD", "STOCH_OVERBOUGHT")):
        if not 0 < values[low] < values[high] < 100:
            raise ValueError(f"Require 0 < {low} < {high} < 100")
    for fast, slow in (("SMA_SHORT", "SMA_LONG"), ("EMA_SHORT", "EMA_LONG"), ("MACD_FAST", "MACD_SLOW")):
        if values[fast] >= values[slow]:
            raise ValueError(f"{fast} must be less than {slow}")
    return tuple(sorted(values.items()))
