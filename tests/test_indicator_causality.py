"""Historical signals must agree with a replay that cannot see future bars."""

import pytest
import pandas as pd

from indicators.registry import get_indicator, get_all_indicators
from signals.base import SignalDirection
from simulation.strategies import CORE_SIMULATION_INDICATORS
from tests.conftest import make_ohlcv
from tests.test_timesfm_execution import indicator as fake_forecast
import numpy as np


@pytest.mark.parametrize("name", CORE_SIMULATION_INDICATORS)
@pytest.mark.parametrize("seed", [7, 42, 123])
def test_core_signal_is_unchanged_by_future_bars(name, seed):
    raw = make_ohlcv(220, trend="volatile", seed=seed)
    indicator = get_indicator(name)
    full = indicator.compute(raw)
    for position in (70, 84, 100, 150, 190):
        prefix = indicator.compute(raw.iloc[: position + 1])
        expected = indicator.get_signal(prefix)
        for idx in (position, position - len(full)):
            actual = indicator.get_signal(full, idx=idx)
            assert actual.direction == expected.direction
            assert actual.confidence == pytest.approx(expected.confidence)
            assert actual.detail == expected.detail


def test_obv_does_not_read_past_available_history():
    indicator = get_indicator("OBV")
    full = indicator.compute(make_ohlcv(100))
    assert indicator.get_signal(full, idx=0).direction == SignalDirection.HOLD
    for idx in (-len(full) - 1, len(full)):
        with pytest.raises(IndexError):
            indicator.get_signal(full, idx=idx)


# Enumerate the registry so a newly opted-in indicator automatically joins the contract.
@pytest.mark.parametrize("name", [name for name, ind in get_all_indicators().items() if ind.backtest_safe])
def test_all_admitted_indicators_preserve_computed_history_and_signals(name):
    raw = make_ohlcv(240, trend="volatile", seed=19)
    indicator = get_indicator(name) if name != "TimesFM Forecast" else fake_forecast(np.linspace(90, 110, 9))[0]
    full = indicator.compute_for_backtest(raw, 5, 0.1)
    pd.testing.assert_frame_equal(raw, make_ohlcv(240, trend="volatile", seed=19))
    for position in (140, 176, 210):
        prefix = indicator.compute_for_backtest(raw.iloc[:position + 1], 5, 0.1)
        pd.testing.assert_frame_equal(full.iloc[:position + 1], prefix)
        expected = indicator.get_signal_for_horizon(prefix, 5)
        for idx in (position, position - len(full)):
            assert indicator.get_signal_for_horizon(full, 5, idx=idx) == expected


@pytest.mark.parametrize("name", ["Copper-Gold Ratio", "VIX Term Structure", "Market Correlation"])
def test_external_indicators_are_causal_given_fixed_reference_history(name, monkeypatch):
    import indicators.macro as macro
    import indicators.systemic as systemic
    raw = make_ohlcv(270, trend="volatile")
    symbols = ["HG=F", "GC=F", "^VIX", "^VIX3M", *systemic.SECTOR_ETFS]
    references = {symbol: make_ohlcv(270, seed=i + 1)["Close"] for i, symbol in enumerate(symbols)}
    cutoff = raw.index[-1]
    def fetch(symbol, period="2y"):
        return references[symbol].loc[:cutoff].copy()
    monkeypatch.setattr(macro, "fetch_reference_close", fetch)
    monkeypatch.setattr(systemic, "fetch_reference_close", fetch)
    indicator = get_indicator(name)
    full = indicator.compute(raw)
    for position in (210, 240):
        cutoff = raw.index[position]
        prefix = indicator.compute(raw.iloc[:position + 1])
        pd.testing.assert_frame_equal(full.iloc[:position + 1], prefix)
        assert indicator.get_signal(full, position) == indicator.get_signal(prefix)
    # Algebraic causality does not establish provider publication times or vintages.
    assert not indicator.backtest_safe


def test_historical_safety_requires_explicit_opt_in():
    from indicators.base import BaseIndicator
    assert BaseIndicator.historical_safe is False
    assert not get_indicator("FRED Macro").backtest_safe
