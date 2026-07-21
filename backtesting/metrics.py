"""Backtest performance metrics."""

import numpy as np
import pandas as pd

from backtesting.report import BacktestReport, Trade
from config.settings import MIN_TRADES_FOR_SHARPE, RISK_FREE_RATE_ANNUAL


def compute_metrics(report: BacktestReport) -> BacktestReport:
    """Compute all metrics from the trade list and fill in the report."""
    trades = report.trades
    report.total_trades = len(trades)

    if not trades:
        return report

    # Profitable = positive P&L (the standard financial meaning of "winning").
    # Exactly-flat trades are neither winners nor losers.
    report.winning_trades = sum(1 for t in trades if t.pnl_pct > 0)
    report.losing_trades = sum(1 for t in trades if t.pnl_pct < 0)
    report.win_rate = report.winning_trades / report.total_trades

    # Direction accuracy = predicted direction matched actual price movement
    report.correct_predictions = sum(1 for t in trades if t.correct)
    report.prediction_accuracy = report.correct_predictions / report.total_trades

    # Build equity curve
    equity = [report.initial_capital]
    for t in trades:
        pnl_mult = 1 + t.pnl_pct
        equity.append(equity[-1] * pnl_mult)

    dates = [trades[0].entry_date] + [t.exit_date for t in trades]
    report.equity_curve = pd.Series(equity, index=dates)

    # Cumulative return
    report.cumulative_return = (equity[-1] / equity[0]) - 1

    # Max drawdown
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
    report.max_drawdown = max_dd

    # Sharpe ratio annualized from the observed trade frequency, in excess of
    # the risk-free rate. Estimated from per-trade returns, so with few
    # non-overlapping trades it is far too noisy to report — None below the
    # minimum trade count (UI shows n/a).
    returns = np.array([t.pnl_pct for t in trades])
    if len(returns) >= MIN_TRADES_FOR_SHARPE and np.std(returns) > 0:
        elapsed_days = max((trades[-1].exit_date - trades[0].entry_date).days, 1)
        periods_per_year = len(trades) / (elapsed_days / 365.25)
        rf_per_period = (1.0 + RISK_FREE_RATE_ANNUAL) ** (1.0 / periods_per_year) - 1.0
        report.sharpe_ratio = (
            (np.mean(returns) - rf_per_period) / np.std(returns)
        ) * np.sqrt(periods_per_year)
    else:
        report.sharpe_ratio = None

    # Profit factor; None (not inf) when there are no losing trades so the
    # sentinel can't leak into downstream math or formatting.
    gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))
    report.profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    total_span = (trades[-1].exit_date - trades[0].entry_date).total_seconds()
    invested_span = sum(
        (trade.exit_date - trade.entry_date).total_seconds() for trade in trades
    )
    report.exposure_pct = min(1.0, invested_span / total_span) if total_span > 0 else 0.0

    return report
