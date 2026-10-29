"""Causal strategy definitions and signal preparation for simulations."""

import hashlib
from dataclasses import dataclass, replace

import pandas as pd

from indicators.registry import get_indicator
from signals.base import CombinedSignal, SignalDirection
from signals.combiner import ScoringPolicy, score_signals
from simulation.models import StrategyDefinition


# This MVP intentionally excludes TimesFM, FRED, and cross-asset indicators.
# Each permitted indicator derives historical values solely from the supplied
# ticker's OHLCV bars and explicitly declares itself backtest-safe.
CORE_SIMULATION_INDICATORS = (
    "SMA Crossover",
    "EMA Crossover",
    "MACD",
    "ADX",
    "RSI",
    "Stochastic",
    "Bollinger Bands",
    "VWAP",
    "OBV",
)


@dataclass(frozen=True)
class PreparedStrategy:
    definition: StrategyDefinition
    signals: tuple[CombinedSignal, ...]
    required_lookback: int
    market_data_fingerprint: str | None = None

    def signal_at(self, position: int) -> CombinedSignal:
        if position < 0 or position >= len(self.signals):
            return _hold_signal("No completed bar is available for this decision.")
        return self.signals[position]

    def matches_data(self, df: pd.DataFrame) -> bool:
        return (
            self.market_data_fingerprint is None
            or self.market_data_fingerprint == _fingerprint_market_data(df)
        )


def _fingerprint_market_data(df: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _hold_signal(reason: str) -> CombinedSignal:
    return CombinedSignal(
        direction=SignalDirection.HOLD,
        confidence=0.0,
        scores={"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0},
        individual_signals=[],
        reasoning=reason,
    )


def _combine_at(definition, indicators, computed_df, position) -> CombinedSignal:
    signals = [indicator.get_signal_for_horizon(
        computed_df, horizon_days=definition.horizon_days, idx=position
    ) for indicator in indicators.values()]
    adjustments = dict(definition.category_adjustments)
    policy = ScoringPolicy(tuple(
        (name, definition.weight_map[name] * adjustments.get(indicator.category, 1.0))
        for name, indicator in indicators.items()
    ), definition.ambiguity_threshold, definition.min_evidence_strength)
    return score_signals(signals, policy)


def prepare_strategy(definition: StrategyDefinition, df: pd.DataFrame) -> PreparedStrategy:
    """Precompute point-in-time-safe signals for deterministic replay."""
    if definition.buy_and_hold:
        signal = CombinedSignal(
            SignalDirection.BUY,
            1.0,
            {"BUY": 1.0, "SELL": 0.0, "HOLD": 0.0},
            [],
            "Buy and remain invested.",
        )
        return PreparedStrategy(
            definition,
            tuple(signal for _ in range(len(df))),
            0,
            _fingerprint_market_data(df),
        )

    unsupported = set(definition.indicator_names) - set(CORE_SIMULATION_INDICATORS)
    if unsupported:
        raise ValueError(
            "Simulation MVP does not support: " + ", ".join(sorted(unsupported))
        )

    indicators = {
        name: get_indicator(name).with_parameters(definition.parameters)
        for name in definition.indicator_names
    }
    unsafe = [
        name
        for name, indicator in indicators.items()
        if not indicator.backtest_safe
        or not indicator.supports_backtest_horizon(definition.horizon_days)
    ]
    if unsafe:
        raise ValueError("Historically unsafe strategy indicators: " + ", ".join(unsafe))

    computed_df = df.copy()
    for indicator in indicators.values():
        computed_df = indicator.compute_for_horizon(computed_df, definition.horizon_days)

    required_lookback = max(indicator.lookback for indicator in indicators.values())
    signals = []
    for position in range(len(computed_df)):
        if position < required_lookback:
            signals.append(_hold_signal("Waiting for indicator warmup."))
        else:
            signals.append(_combine_at(definition, indicators, computed_df, position))
    return PreparedStrategy(
        definition,
        tuple(signals),
        required_lookback,
        _fingerprint_market_data(df),
    )


def get_strategy_presets() -> dict[str, StrategyDefinition]:
    """Return fresh immutable definitions for the supported MVP presets."""
    return {
        "Buy & Hold": StrategyDefinition(
            "buy_hold_v1", "Buy & Hold", buy_and_hold=True
        ),
        "Balanced Technical": StrategyDefinition(
            "balanced_technical_v1",
            "Balanced Technical",
            indicator_names=(
                "SMA Crossover",
                "MACD",
                "RSI",
                "Bollinger Bands",
                "OBV",
            ),
        ),
        "Trend Following": StrategyDefinition(
            "trend_following_v1",
            "Trend Following",
            indicator_names=("SMA Crossover", "EMA Crossover", "MACD", "ADX"),
            weights=(("SMA Crossover", 1.2), ("EMA Crossover", 1.1), ("MACD", 1.2)),
            horizon_days=10,
        ),
        "Mean Reversion": StrategyDefinition(
            "mean_reversion_v1",
            "Mean Reversion",
            indicator_names=("RSI", "Stochastic", "Bollinger Bands"),
            weights=(("RSI", 1.3), ("Stochastic", 1.2), ("Bollinger Bands", 1.1)),
            horizon_days=3,
        ),
    }


def build_custom_strategy(
    indicator_names: list[str] | tuple[str, ...],
    *,
    horizon_days: int = 5,
    ambiguity_threshold: float = 0.10,
    weights: dict[str, float] | None = None,
    name: str = "Custom Strategy",
    parameters: dict[str, float] | None = None,
    min_evidence_strength: float = 0.15,
) -> StrategyDefinition:
    selected = tuple(indicator_names)
    unsupported = set(selected) - set(CORE_SIMULATION_INDICATORS)
    if unsupported:
        raise ValueError("Unsupported custom indicators: " + ", ".join(sorted(unsupported)))
    supplied_weights = weights or {}
    selected_weights = tuple(
        (indicator, float(supplied_weights[indicator]))
        for indicator in selected
        if indicator in supplied_weights
    )
    definition = StrategyDefinition(
        strategy_id="custom", name=name, indicator_names=selected,
        weights=selected_weights, horizon_days=horizon_days,
        ambiguity_threshold=ambiguity_threshold,
        parameters=tuple(sorted((parameters or {}).items())),
        min_evidence_strength=min_evidence_strength,
    )
    return replace(definition, strategy_id=f"custom_{definition.configuration_id[:12]}")
