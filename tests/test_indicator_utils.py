"""Tests for cross-asset indicator utilities (offline, monkeypatched)."""

import pandas as pd

import yfinance

from indicators._utils import clear_reference_cache, fetch_reference_close


def test_reference_fetch_is_unadjusted(monkeypatch):
    """References must be fetched point-in-time (auto_adjust=False):
    dividend adjustment bakes future information into past bars."""
    captured = {}

    class FakeTicker:
        def __init__(self, symbol):
            captured["symbol"] = symbol

        def history(self, **kwargs):
            captured.update(kwargs)
            index = pd.date_range("2024-01-01", periods=5, freq="D")
            return pd.DataFrame(
                {col: [1.0] * 5 for col in ["Open", "High", "Low", "Close", "Volume"]},
                index=index,
            )

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    clear_reference_cache()
    try:
        series = fetch_reference_close("REF-ADJ-TEST")
        assert captured["auto_adjust"] is False
        assert series is not None and len(series) == 5
    finally:
        clear_reference_cache()
