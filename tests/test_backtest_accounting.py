"""End-to-end accounting identities and daily risk regressions."""

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import run_backtest
from signals.base import SignalDirection, SignalResult
from tests.test_backtest import AlwaysBuyIndicator


class AlwaysSellIndicator(AlwaysBuyIndicator):
    def get_signal(self, df, idx=-1):
        return SignalResult(self.name, SignalDirection.SELL, 1.0)


def bars(count=66):
    return pd.DataFrame(
        {"Open": 100., "High": 100., "Low": 100., "Close": 100., "Volume": 1000.},
        index=pd.bdate_range("2024-01-01", periods=count),
    )


def run(df, indicator=None, **kwargs):
    indicator = indicator or AlwaysBuyIndicator()
    return run_backtest(
        df, {indicator.name: indicator}, ticker=kwargs.pop("ticker", "TEST"), period="1mo",
        evaluation_start=df.index[60], cost_per_trade_pct=kwargs.pop("cost", 0),
        **kwargs,
    )


@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_exit_target_is_horizon_bars_after_signal(horizon):
    df = bars()
    report = run(df, horizon_days=horizon)
    first = report.trades[0]
    assert first.signal_date == df.index[59]
    assert first.entry_date == df.index[60]
    assert first.exit_date == first.target_date == df.index[59 + horizon]
    assert report.trades[-1].exit_date <= df.index[-1]


def test_drawdown_includes_unrealized_loss_even_when_trade_breaks_even():
    df = bars(62)
    df.loc[df.index[60], ["Low", "Close"]] = 50
    report = run(df, horizon_days=2)
    assert report.trades[0].pnl_dollars == 0
    assert report.equity_curve.tolist() == [10000, 5000, 10000]
    assert report.max_drawdown == pytest.approx(0.5)


def test_fees_cash_and_equity_reconcile_with_trade_ledger():
    df = bars(64)
    df.loc[df.index[61], ["High", "Close"]] = 120
    df.loc[df.index[63], ["Low", "Close"]] = 90
    report = run(df, horizon_days=2, cost=2)
    first = report.trades[0]
    quantity = 10000 / (100 * 1.01)
    assert first.quantity == pytest.approx(quantity)
    assert first.fees == pytest.approx(quantity * (100 + 120) * 0.01)
    assert first.pnl_dollars == pytest.approx(quantity * 20 - first.fees)
    assert report.equity_curve.iloc[-1] == pytest.approx(
        10000 + sum(t.pnl_dollars for t in report.trades)
    )
    for snapshot in report.snapshots:
        assert snapshot.equity == pytest.approx(snapshot.cash + snapshot.quantity * snapshot.close_price)
    assert report.profit_factor == pytest.approx(
        report.trades[0].pnl_dollars / -report.trades[1].pnl_dollars
    )


@pytest.mark.parametrize("gap", [False, True])
def test_short_liquidation_stops_run_and_preserves_gap_debt(gap):
    df = bars()
    df.loc[df.index[61], "High"] = 300
    if gap:
        df.loc[df.index[61], "Open"] = 300
    report = run(df, AlwaysSellIndicator(), horizon_days=5)
    assert len(report.trades) == 1
    assert report.completion_reason == "insolvency"
    assert report.trades[0].exit_reason == "liquidation"
    assert report.trades[0].exit_price == (300 if gap else 200)
    assert report.equity_curve.iloc[-1] == (-10000 if gap else 0)
    assert report.evaluation_end == df.index[61]
    assert len(report.snapshots) == 2


def test_short_cover_fee_is_included_in_liquidation_boundary():
    df = bars()
    df.loc[df.index[60], "High"] = 220
    report = run(df, AlwaysSellIndicator(), horizon_days=5, cost=2)
    assert report.completion_reason == "insolvency"
    assert report.trades[0].fees > 0
    assert report.equity_curve.iloc[-1] == pytest.approx(0)


@pytest.mark.parametrize(("ticker", "annual_days"), [("TEST", 252), ("BTC-USD", 365)])
def test_daily_sharpe_includes_cash_days(ticker, annual_days):
    class Once(AlwaysBuyIndicator):
        def get_signal(self, df, idx=-1):
            direction = SignalDirection.BUY if idx == 59 else SignalDirection.HOLD
            return SignalResult(self.name, direction, 1.0)

    df = bars(64)
    df.loc[df.index[60], ["High", "Close"]] = 110
    report = run(df, Once(), horizon_days=1, ticker=ticker)
    returns = np.array([0.1, 0, 0, 0])
    assert report.sharpe_ratio == pytest.approx(returns.mean() / returns.std(ddof=1) * np.sqrt(annual_days))
    assert report.exposure_pct == 0.25
    assert report.evaluation_end == df.index[-1]


def test_no_signals_retains_cash_over_entire_evaluation():
    class Neutral(AlwaysBuyIndicator):
        def get_signal(self, df, idx=-1):
            return SignalResult(self.name, SignalDirection.HOLD, 0)
    report = run(bars(), Neutral())
    assert report.total_trades == 0
    assert report.equity_curve.eq(10000).all()
    assert len(report.snapshots) == 6
    assert report.max_drawdown == report.sharpe_ratio == report.exposure_pct == 0


@pytest.mark.parametrize("kwargs", [
    {"horizon_days": 0}, {"horizon_days": -1}, {"horizon_days": 1.5},
    {"initial_capital": 0}, {"initial_capital": float("nan")},
    {"cost": -1}, {"cost": float("inf")},
])
def test_invalid_execution_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        run(bars(), **kwargs)


@pytest.mark.parametrize("problem", ["zero", "nan", "ohlc", "duplicates", "unsorted"])
def test_invalid_prices_or_dates_are_rejected(problem):
    df = bars()
    if problem == "zero":
        df.iloc[0, 0] = 0
    elif problem == "nan":
        df.iloc[0, 3] = float("nan")
    elif problem == "ohlc":
        df.iloc[0, 1] = 99
    elif problem == "duplicates":
        df.index = pd.DatetimeIndex([df.index[1], *df.index[1:]])
    else:
        df = df.iloc[::-1]
    with pytest.raises(ValueError):
        run(df)


def test_configuration_and_data_fingerprints_track_different_inputs():
    df = bars()
    first = run(df)
    repeat = run(df)
    assert first.configuration_id == repeat.configuration_id
    assert first.data_fingerprint == repeat.data_fingerprint
    assert run(df, cost=1).configuration_id != first.configuration_id
    changed = df.copy()
    changed.Volume *= 2
    assert run(changed).data_fingerprint != first.data_fingerprint
