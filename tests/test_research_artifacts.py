"""Research archives preserve inputs and reproduce execution without external I/O."""

from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from backtesting.engine import run_backtest
from indicators.registry import get_indicator
from research.artifacts import build_archive, load_archive, build_backtest_archive, replay_backtest_archive
from tests.conftest import make_ohlcv


def test_frame_round_trip_preserves_precision_timezone_and_types():
    frame = make_ohlcv(20)
    frame.index = pd.date_range("2025-03-01", periods=20, tz="America/New_York")
    frame.index.name = "Date"
    frame["Volume"] = 2**60 + 1
    saved = build_archive("test", {"market": frame}, {"threshold": 0.15}, {"metric": float("inf")})
    loaded = load_archive(saved)
    pd.testing.assert_frame_equal(loaded["frames"]["market"], frame, check_freq=False)
    assert loaded["configuration"] == {"threshold": 0.15}
    assert loaded["result"]["metric"] == "Infinity"
    assert "pandas" in loaded["manifest"]["packages"]


def test_archive_rejects_corrupted_contents():
    data = build_archive("test", {"market": make_ohlcv(20)}, {}, {})
    output = BytesIO()
    with ZipFile(BytesIO(data)) as original, ZipFile(output, "w") as changed:
        for name in original.namelist():
            changed.writestr(name, b'{"changed": true}' if name == "result.json" else original.read(name))
    with pytest.raises(ValueError, match="Checksum"):
        load_archive(output.getvalue())


@pytest.mark.parametrize("name", ["RSI", "SMA Crossover"])
def test_execution_replays_from_saved_inputs_and_settings(name, monkeypatch):
    frame = make_ohlcv(280, trend="volatile")
    indicator = get_indicator(name)
    report = run_backtest(frame, {name: indicator}, "TEST", "1y", evaluation_start=frame.index[180])
    data = build_backtest_archive(report, frame)
    def fail(*args, **kwargs):
        raise AssertionError("Replay must not recompute indicators or read session settings")
    monkeypatch.setattr(type(indicator), "compute_for_backtest", fail)
    monkeypatch.setattr("config.parameters.get_setting", fail)
    monkeypatch.setattr("signals.combiner.get_setting", fail)
    replayed = replay_backtest_archive(data)
    assert replayed.trades == report.trades
    assert replayed.configuration_id == report.configuration_id
    pd.testing.assert_series_equal(replayed.equity_curve, report.equity_curve)


def test_archive_rejects_mismatched_market_inputs():
    frame = make_ohlcv(200)
    report = run_backtest(frame, {"RSI": get_indicator("RSI")}, "TEST", "1y")
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("Volume")] += 1
    with pytest.raises(ValueError, match="does not match"):
        build_backtest_archive(report, changed)


def test_timesfm_replay_uses_saved_forecasts_even_when_model_is_unavailable():
    import numpy as np
    from tests.test_timesfm_execution import indicator
    model, _ = indicator(np.linspace(101, 109, 9))
    frame = make_ohlcv(220)
    report = run_backtest(frame, {model.name: model}, "TEST", "1y")
    restored = replay_backtest_archive(build_backtest_archive(report, frame))
    assert restored.trades == report.trades
    pd.testing.assert_series_equal(restored.equity_curve, report.equity_curve)
