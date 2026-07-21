"""Tests for backtesting engine and metrics."""

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import run_backtest
from backtesting.metrics import compute_metrics
from backtesting.report import BacktestReport, Trade
from indicators.base import BaseIndicator
from indicators.registry import get_all_indicators
from signals.base import SignalDirection, SignalResult
from tests.conftest import make_ohlcv


class AlwaysBuyIndicator(BaseIndicator):
    @property
    def name(self):
        return "Always Buy"

    @property
    def category(self):
        return "trend"

    @property
    def lookback(self):
        return 1

    def compute(self, df):
        return df.copy()

    def get_signal(self, df, idx=-1):
        return SignalResult(self.name, SignalDirection.BUY, 1.0, "Test signal")

    def get_chart_config(self):
        return {"overlay": True, "columns": [], "colors": {}}


class UnsafeAlwaysBuyIndicator(AlwaysBuyIndicator):
    @property
    def name(self):
        return "Unsafe Always Buy"

    @property
    def backtest_safe(self):
        return False


class TestRunBacktest:
    @pytest.fixture
    def indicators(self):
        return get_all_indicators()

    def test_returns_report(self, indicators):
        df = make_ohlcv(300, trend="up")
        report = run_backtest(df, indicators, ticker="TEST", period="1y", horizon_days=5)
        assert isinstance(report, BacktestReport)
        assert report.ticker == "TEST"
        assert report.period == "1y"
        assert report.horizon_days == 5

    def test_insufficient_data_zero_trades(self, indicators):
        df = make_ohlcv(10, trend="flat")
        report = run_backtest(df, indicators, ticker="TEST", period="1y", horizon_days=5)
        assert report.total_trades == 0

    def test_trade_fields_valid(self, indicators):
        df = make_ohlcv(300, trend="up")
        report = run_backtest(df, indicators, ticker="TEST", period="1y", horizon_days=5)
        for trade in report.trades:
            assert isinstance(trade, Trade)
            assert trade.direction in ("BUY", "SELL")
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert isinstance(trade.correct, bool)
            assert trade.predicted_direction in ("BUY", "SELL")
            assert trade.actual_direction in ("BUY", "SELL", "HOLD")

    def test_no_overlapping_trades(self, indicators):
        df = make_ohlcv(300, trend="volatile", seed=7)
        report = run_backtest(df, indicators, ticker="TEST", period="1y", horizon_days=5)
        for i in range(len(report.trades) - 1):
            current = report.trades[i]
            next_trade = report.trades[i + 1]
            assert current.exit_date <= next_trade.entry_date

    def test_cost_reduces_pnl(self, indicators):
        df = make_ohlcv(300, trend="up")
        report_no_cost = run_backtest(
            df, indicators, ticker="TEST", period="1y",
            horizon_days=5, cost_per_trade_pct=0.0,
        )
        report_with_cost = run_backtest(
            df, indicators, ticker="TEST", period="1y",
            horizon_days=5, cost_per_trade_pct=1.0,
        )
        if report_no_cost.trades and report_with_cost.trades:
            # Same first trade should have lower PnL with cost
            assert report_with_cost.trades[0].pnl_pct < report_no_cost.trades[0].pnl_pct

    def test_evaluation_start_bounds_trades(self):
        df = make_ohlcv(120, trend="up")
        evaluation_start = df.index[80]
        report = run_backtest(
            df,
            {"Always Buy": AlwaysBuyIndicator()},
            ticker="TEST",
            period="1mo",
            horizon_days=2,
            evaluation_start=evaluation_start,
        )
        assert report.trades
        assert report.trades[0].entry_date >= evaluation_start

    def test_signal_enters_at_next_bar_open(self):
        df = make_ohlcv(90, trend="up")
        evaluation_start = df.index[60]
        df.loc[evaluation_start, "Open"] = 123.45
        report = run_backtest(
            df,
            {"Always Buy": AlwaysBuyIndicator()},
            ticker="TEST",
            period="1mo",
            horizon_days=2,
            evaluation_start=evaluation_start,
        )
        assert report.trades[0].entry_date == evaluation_start
        assert report.trades[0].entry_price == pytest.approx(123.45)

    def test_unsafe_indicators_are_excluded(self):
        df = make_ohlcv(90, trend="up")
        report = run_backtest(
            df,
            {"Unsafe Always Buy": UnsafeAlwaysBuyIndicator()},
            ticker="TEST",
            period="1mo",
        )
        assert report.total_trades == 0
        assert report.excluded_indicators == ["Unsafe Always Buy"]


class TestComputeMetrics:
    def _make_trade(self, pnl_pct, correct=True, direction="BUY"):
        return Trade(
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-06"),
            direction=direction,
            entry_price=100.0,
            exit_price=100.0 * (1 + pnl_pct),
            predicted_direction=direction,
            actual_direction=direction if correct else ("SELL" if direction == "BUY" else "BUY"),
            correct=correct,
            pnl_pct=pnl_pct,
        )

    def test_empty_trades(self):
        report = BacktestReport(ticker="T", period="1y", horizon_days=5, initial_capital=10000)
        result = compute_metrics(report)
        assert result.total_trades == 0
        assert result.win_rate == 0.0

    def test_all_winning(self):
        trades = [self._make_trade(0.05) for _ in range(5)]
        report = BacktestReport(
            ticker="T", period="1y", horizon_days=5,
            initial_capital=10000, trades=trades,
        )
        result = compute_metrics(report)
        assert result.total_trades == 5
        assert result.win_rate == 1.0
        assert result.cumulative_return > 0
        assert result.max_drawdown == 0.0
        assert result.profit_factor is None  # no losing trades

    def test_all_losing(self):
        trades = [self._make_trade(-0.03, correct=False) for _ in range(5)]
        report = BacktestReport(
            ticker="T", period="1y", horizon_days=5,
            initial_capital=10000, trades=trades,
        )
        result = compute_metrics(report)
        assert result.total_trades == 5
        assert result.win_rate == 0.0
        assert result.cumulative_return < 0
        assert result.max_drawdown > 0

    def test_mixed_trades(self):
        trades = [
            self._make_trade(0.10, correct=True),
            self._make_trade(-0.05, correct=False),
            self._make_trade(0.08, correct=True),
            self._make_trade(-0.02, correct=False),
        ]
        report = BacktestReport(
            ticker="T", period="1y", horizon_days=5,
            initial_capital=10000, trades=trades,
        )
        result = compute_metrics(report)
        assert result.total_trades == 4
        assert result.win_rate == 0.5
        assert result.prediction_accuracy == 0.5
        assert result.profit_factor > 0
        assert not result.equity_curve.empty
        assert 0.0 <= result.exposure_pct <= 1.0

    def test_equity_curve_length(self):
        trades = [self._make_trade(0.02) for _ in range(3)]
        report = BacktestReport(
            ticker="T", period="1y", horizon_days=5,
            initial_capital=10000, trades=trades,
        )
        result = compute_metrics(report)
        # Equity curve has n_trades + 1 points (initial + after each trade)
        assert len(result.equity_curve) == 4

    def test_sharpe_suppressed_below_min_trades(self):
        trades = [
            self._make_trade(0.05),
            self._make_trade(-0.02, correct=False),
            self._make_trade(0.03),
        ]
        report = BacktestReport(
            ticker="T", period="1y", horizon_days=5,
            initial_capital=10000, trades=trades,
        )
        result = compute_metrics(report)
        # 3 trades is far too few for a meaningful Sharpe estimate
        assert result.sharpe_ratio is None

    def test_sharpe_calculated_with_enough_trades(self):
        trades = [
            self._make_trade(0.05 if i % 2 == 0 else -0.02, correct=(i % 2 == 0))
            for i in range(12)
        ]
        report = BacktestReport(
            ticker="T", period="1y", horizon_days=5,
            initial_capital=10000, trades=trades,
        )
        result = compute_metrics(report)
        assert isinstance(result.sharpe_ratio, float)
        assert result.sharpe_ratio != 0.0

    def test_flat_trades_are_neither_wins_nor_losses(self):
        trades = [
            self._make_trade(0.05),
            self._make_trade(0.0),
            self._make_trade(-0.02, correct=False),
        ]
        report = BacktestReport(
            ticker="T", period="1y", horizon_days=5,
            initial_capital=10000, trades=trades,
        )
        result = compute_metrics(report)
        assert result.winning_trades == 1
        assert result.losing_trades == 1
        assert result.total_trades == 3
