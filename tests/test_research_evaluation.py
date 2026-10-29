"""Fixed strategies share the held-out execution window and cost assumptions."""

import pandas as pd
import pytest

from research.artifacts import load_archive
from research.evaluation import compare_strategies, build_comparison_archive
from tests.conftest import make_ohlcv


def test_comparison_uses_common_test_window_and_real_fill_costs():
    frame = make_ohlcv(240, trend="up")
    reports = compare_strategies(frame, ticker="TEST", test_start=frame.index[150], cost_pct=2)
    for report in reports.values():
        assert report.evaluation_start == frame.index[150]
        assert report.evaluation_end == frame.index[-1]
        assert report.cost_per_trade_pct == 2
        assert all(trade.entry_date >= frame.index[150] for trade in report.trades)
    buy_hold = reports["buy_hold"]
    assert len(buy_hold.trades) == 1
    assert buy_hold.cumulative_return == pytest.approx(
        frame.Close.iloc[-1] / frame.Open.iloc[150] * 0.99 / 1.01 - 1
    )
    assert reports["cash"].cumulative_return == 0
    saved = load_archive(build_comparison_archive(frame, reports))
    pd.testing.assert_frame_equal(frame, saved["frames"]["market"], check_freq=False)
    assert len(saved["result"]["summary"]) == 4


def test_test_start_cannot_silently_move_to_accommodate_warmup():
    frame = make_ohlcv(240)
    with pytest.raises(ValueError, match="warmup"):
        compare_strategies(frame, ticker="TEST", test_start=frame.index[5])
