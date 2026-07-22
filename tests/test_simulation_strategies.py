"""Tests for supported simulation strategy construction and preparation."""

import pytest

from signals.base import SignalDirection
from simulation.engine import HistoricalSimulationEngine
from simulation.models import SimulationConfig
from simulation.strategies import (
    CORE_SIMULATION_INDICATORS,
    build_custom_strategy,
    get_strategy_presets,
    prepare_strategy,
)


def test_all_presets_prepare_one_signal_per_input_bar(ohlcv_200_up):
    df = ohlcv_200_up

    for definition in get_strategy_presets().values():
        prepared = prepare_strategy(definition, df)
        assert len(prepared.signals) == len(df)
        assert prepared.definition == definition


def test_buy_and_hold_is_always_buy(ohlcv_short):
    df = ohlcv_short
    definition = get_strategy_presets()["Buy & Hold"]

    prepared = prepare_strategy(definition, df)

    assert all(signal.direction == SignalDirection.BUY for signal in prepared.signals)
    assert prepared.required_lookback == 0


def test_custom_strategy_preserves_local_weights_without_global_override(ohlcv_100):
    df = ohlcv_100
    definition = build_custom_strategy(
        ["RSI", "MACD"],
        horizon_days=7,
        ambiguity_threshold=0.2,
        weights={"RSI": 2.5, "MACD": 0.5},
    )

    prepared = prepare_strategy(definition, df)

    assert definition.weight_map == {"RSI": 2.5, "MACD": 0.5}
    assert definition.horizon_days == 7
    assert definition.ambiguity_threshold == 0.2
    assert definition.strategy_id.startswith("custom_")
    assert len(prepared.signals) == len(df)


def test_custom_strategy_id_is_deterministic_and_changes_with_configuration():
    first = build_custom_strategy(["RSI", "MACD"], weights={"RSI": 2.0})
    replay = build_custom_strategy(["RSI", "MACD"], weights={"RSI": 2.0})
    changed = build_custom_strategy(["RSI", "MACD"], weights={"RSI": 2.1})

    assert first.strategy_id == replay.strategy_id
    assert first.strategy_id != changed.strategy_id


def test_prepared_strategy_cannot_be_applied_to_changed_market_data(ohlcv_100):
    definition = get_strategy_presets()["Buy & Hold"]
    prepared = prepare_strategy(definition, ohlcv_100)
    changed = ohlcv_100.copy()
    changed.loc[changed.index[-1], "Close"] *= 2
    config = SimulationConfig(
        "TEST", changed.index[10], changed.index[-1], 10_000
    )

    with pytest.raises(ValueError, match="does not match"):
        HistoricalSimulationEngine(changed, config, prepared)


def test_custom_strategy_rejects_external_or_revision_prone_indicators():
    assert "TimesFM Forecast" not in CORE_SIMULATION_INDICATORS
    assert "FRED Macro" not in CORE_SIMULATION_INDICATORS

    with pytest.raises(ValueError, match="Unsupported custom indicators"):
        build_custom_strategy(["TimesFM Forecast"])


def test_warmup_bars_are_explicitly_hold(ohlcv_100):
    df = ohlcv_100
    definition = build_custom_strategy(["SMA Crossover"])

    prepared = prepare_strategy(definition, df)

    assert prepared.required_lookback > 0
    assert all(
        signal.direction == SignalDirection.HOLD
        for signal in prepared.signals[: prepared.required_lookback]
    )
