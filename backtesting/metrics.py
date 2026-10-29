"""Backtest performance metrics."""

import numpy as np
import pandas as pd

from backtesting.report import BacktestReport


def compute_metrics(report: BacktestReport) -> BacktestReport:
    """Compute all metrics from the trade list and fill in the report."""
    trades = report.trades
    report.total_trades = len(trades)

    if not trades and report.equity_curve.empty:
        return report

    # Profitable = positive P&L (the standard financial meaning of "winning")
    report.winning_trades = sum(1 for t in trades if t.pnl_pct > 0)
    report.losing_trades = report.total_trades - report.winning_trades
    report.win_rate = report.winning_trades / report.total_trades if trades else 0.0

    # Direction accuracy = predicted direction matched actual price movement
    report.correct_predictions = sum(1 for t in trades if t.correct)
    report.prediction_accuracy = report.correct_predictions / report.total_trades if trades else 0.0

    # Build equity curve
    trade_equity = [report.initial_capital]
    dollar_pnl = []
    for t in trades:
        pnl = t.pnl_dollars if t.pnl_dollars is not None else trade_equity[-1] * t.pnl_pct
        dollar_pnl.append(pnl)
        trade_equity.append(trade_equity[-1] + pnl)

    # Legacy trade-only reports retain an exit-only curve, but cannot supply
    # daily risk statistics. Engine reports always provide the full ledger.
    if report.equity_curve.empty:
        dates = [trades[0].entry_date] + [t.exit_date for t in trades]
        report.equity_curve = pd.Series(trade_equity, index=dates)
    equity = report.equity_curve.to_numpy(dtype=float)

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

    # Daily close-to-close portfolio returns include fees and cash days.
    # Assumes zero risk-free rate and 252 equity / 365 crypto sessions per year.
    report.sharpe_ratio = 0.0
    if report.snapshots:
        returns = report.equity_curve.pct_change(fill_method=None).iloc[1:].to_numpy()
        if len(returns) > 1 and np.isfinite(returns).all() and np.std(returns, ddof=1) > 0:
            report.sharpe_ratio = float(
                np.mean(returns) / np.std(returns, ddof=1)
                * np.sqrt(365 if report.is_crypto else 252)
            )

    # Use dollar P&L: each trade is sized from the compounded entry equity.
    gross_profit = sum(pnl for pnl in dollar_pnl if pnl > 0)
    gross_loss = -sum(pnl for pnl in dollar_pnl if pnl < 0)
    report.profit_factor = (
        gross_profit / gross_loss if gross_loss > 0
        else float("inf") if gross_profit > 0 else 0.0
    )

    report.exposure_pct = (
        sum(snapshot.exposed for snapshot in report.snapshots) / len(report.snapshots)
        if report.snapshots else 0.0
    )

    return report
