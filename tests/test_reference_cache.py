"""Reference feeds expire successes, retry failures and isolate caller mutations."""

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from data.series_cache import SeriesCache
from indicators import _utils, fred
from indicators.fred import _fetch_fred_series as fetch_fred


@pytest.fixture
def clock():
    return [0.0]


def test_reference_success_expires_and_returns_independent_copies(monkeypatch, clock):
    cache = SeriesCache(300, clock=lambda: clock[0])
    monkeypatch.setattr(_utils, "_reference_cache", cache)
    calls = []
    def history(period, **kwargs):
        calls.append(period)
        return pd.DataFrame({"Close": [100. + len(calls)]})
    monkeypatch.setattr("yfinance.Ticker", lambda ticker: SimpleNamespace(history=history))
    first = _utils.fetch_reference_close("TEST")
    first.iloc[0] = -1
    assert _utils.fetch_reference_close("TEST").iloc[0] == 101
    clock[0] = 300
    assert _utils.fetch_reference_close("TEST").iloc[0] == 102
    _utils.fetch_reference_close("TEST", "max")
    assert calls == ["2y", "2y", "max"]
    _utils.clear_reference_cache()
    _utils.fetch_reference_close("TEST")
    assert len(calls) == 4


def test_reference_failure_retries_after_short_backoff(monkeypatch, clock):
    monkeypatch.setattr(_utils, "_reference_cache", SeriesCache(300, clock=lambda: clock[0]))
    calls = []
    def history(period, **kwargs):
        calls.append(period)
        if len(calls) == 1:
            raise OSError("Transient provider failure")
        return pd.DataFrame({"Close": [100.]})
    monkeypatch.setattr("yfinance.Ticker", lambda ticker: SimpleNamespace(history=history))
    assert _utils.fetch_reference_close("TEST") is None
    clock[0] = 29
    assert _utils.fetch_reference_close("TEST") is None
    assert len(calls) == 1
    clock[0] = 30
    assert _utils.fetch_reference_close("TEST").iloc[0] == 100


def test_fred_added_or_changed_credentials_retry_and_success_expires(monkeypatch, clock):
    monkeypatch.setattr(fred, "_fred_cache", SeriesCache(3600, clock=lambda: clock[0]))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert fetch_fred("TEST") is None
    calls = []
    class FakeFred:
        def __init__(self, api_key):
            self.valid = api_key == "test-valid"
        def get_series(self, series_id):
            calls.append(series_id)
            if not self.valid:
                raise ValueError("Invalid test credential")
            return pd.Series([float(len(calls))])
    monkeypatch.setitem(sys.modules, "fredapi", SimpleNamespace(Fred=FakeFred))
    monkeypatch.setenv("FRED_API_KEY", "test-invalid")
    assert fetch_fred("TEST") is None
    assert fetch_fred("TEST") is None
    monkeypatch.setenv("FRED_API_KEY", "test-valid")
    assert fetch_fred("TEST").iloc[0] == 2
    assert fetch_fred("TEST").iloc[0] == 2
    clock[0] = 3600
    assert fetch_fred("TEST").iloc[0] == 3


def test_cache_bounds_number_of_entries():
    cache = SeriesCache(300, max_entries=2)
    calls = []
    def load():
        calls.append(1)
        return pd.Series([1.])
    for key in ("a", "b", "c", "a"):
        cache.get_or_load(key, load)
    assert len(calls) == 4
