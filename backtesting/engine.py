"""Daily walk-forward execution with explicit fills and portfolio valuation."""

import hashlib
import math
import json
from dataclasses import asdict
from numbers import Integral

import numpy as np
import pandas as pd

from backtesting.metrics import compute_metrics
from backtesting.report import BacktestReport, PortfolioSnapshot, Trade
from config.settings import DEFAULT_COST_PER_TRADE_PCT, WARMUP_BUFFER
from config.parameters import capture_parameters
from data.fetcher import is_crypto_ticker
from indicators.base import BaseIndicator
from portfolio.accounting import PortfolioState
from signals.base import SignalDirection
from signals.combiner import ScoringPolicy, combine_signals


def _validate_data(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Backtest data must use a DatetimeIndex")
    if df.index.hasnans or not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("Backtest timestamps must be valid, sorted and unique")
    if not df.index.normalize().is_unique:
        raise ValueError("Backtest expects daily bars, not intraday data")
    columns = ["Open", "High", "Low", "Close", "Volume"]
    if not set(columns).issubset(df.columns):
        raise ValueError("Backtest data requires Open, High, Low, Close and Volume")
    result = df.copy()
    result[columns] = result[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result[columns].to_numpy(dtype=float)).all():
        raise ValueError("Backtest OHLCV values must be finite")
    if (result[columns[:4]] <= 0).any().any() or (result.Volume < 0).any():
        raise ValueError("Backtest prices must be positive and volume non-negative")
    if (result.High < result[["Open", "Close", "Low"]].max(axis=1)).any() or (
        result.Low > result[["Open", "Close", "High"]].min(axis=1)
    ).any():
        raise ValueError("Backtest OHLC relationships are invalid")
    return result


def run_backtest(
    df: pd.DataFrame,
    indicators: dict[str, BaseIndicator],
    ticker: str,
    period: str,
    horizon_days: int = 5,
    initial_capital: float = 10_000.0,
    cost_per_trade_pct: float = DEFAULT_COST_PER_TRADE_PCT,
    evaluation_start: pd.Timestamp | None = None,
    indicator_parameters: dict | None = None,
    scoring_policy: ScoringPolicy | None = None,
    precomputed_data: pd.DataFrame | None = None,
) -> BacktestReport:
    """Signal at close t, enter at open t+1, exit at close t+horizon.

    Split the quoted round-trip cost equally into fees on each fill's notional.
    Positions use at most 1x entry equity; short proceeds remain collateral.
    Shorts liquidate at the zero-equity covering price when a bar's high reaches
    it, or at the open if it gaps beyond it. Gaps can leave debt, which is recorded
    without opening further trades. No borrow/financing fees are modeled.
    """
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, Integral) or horizon_days < 1:
        raise ValueError("Horizon must be a positive integer")
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be finite and positive")
    if not math.isfinite(cost_per_trade_pct) or not 0 <= cost_per_trade_pct < 100:
        raise ValueError("Transaction cost must be in [0, 100)")
    report = BacktestReport(
        ticker=ticker, period=period, horizon_days=horizon_days,
        initial_capital=initial_capital, is_crypto=is_crypto_ticker(ticker),
        cost_per_trade_pct=cost_per_trade_pct,
    )
    if df.empty:
        return report
    df = _validate_data(df)
    report.data_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()

    parameters = capture_parameters(indicator_parameters, session=indicator_parameters is None)
    safe = {
        name: indicator.with_parameters(parameters) for name, indicator in indicators.items()
        if indicator.backtest_safe and indicator.supports_backtest_horizon(horizon_days)
    }
    report.excluded_indicators = [name for name in indicators if name not in safe]
    if not safe:
        return report
    policy = scoring_policy or ScoringPolicy.from_indicators(safe, horizon_days)
    report.configuration = {
        "execution_version": 3, "ticker": ticker, "period": period,
        "horizon_bars": int(horizon_days),
        "round_trip_cost_pct": cost_per_trade_pct, "initial_capital": initial_capital,
        "indicator_parameters": dict(parameters), "scoring": asdict(policy),
        "models": {
            name: asdict(indicator.runtime.config)
            for name, indicator in safe.items() if hasattr(indicator, "runtime")
        },
    }
    warmup = max(ind.lookback for ind in safe.values()) + WARMUP_BUFFER
    evaluation_position = warmup + 1
    if evaluation_start is not None:
        start = pd.Timestamp(evaluation_start)
        if pd.isna(start):
            raise ValueError("Evaluation start must be a valid timestamp")
        if df.index.tz is None and start.tzinfo is not None:
            start = start.tz_localize(None)
        elif df.index.tz is not None:
            start = start.tz_localize(df.index.tz) if start.tzinfo is None else start.tz_convert(df.index.tz)
        evaluation_position = max(evaluation_position, int(df.index.searchsorted(start)))
    if evaluation_position >= len(df):
        return report
    report.configuration.update(
        evaluation_start=str(df.index[evaluation_position]), evaluation_end=str(df.index[-1])
    )
    report.configuration_id = hashlib.sha256(
        json.dumps(report.configuration, sort_keys=True).encode()
    ).hexdigest()

    if precomputed_data is None:
        computed = df.copy()
        for indicator in safe.values():
            computed = indicator.compute_for_backtest(computed, horizon_days, cost_per_trade_pct)
    else:
        if not precomputed_data.index.equals(df.index) or not precomputed_data[df.columns].equals(df):
            raise ValueError("Precomputed data must contain identical input bars")
        computed = precomputed_data.copy()
    report.computed_data = computed.copy()

    report.evaluation_start = df.index[evaluation_position]
    report.evaluation_end = df.index[-1]
    portfolio = PortfolioState(float(initial_capital))
    fee_rate = cost_per_trade_pct / 200.0
    equity = [initial_capital]
    dates = [df.index[evaluation_position - 1]]  # initial capital before first open
    active = None
    last_position = len(df) - 1

    for position in range(evaluation_position, len(df)):
        row = df.iloc[position]
        timestamp = df.index[position]
        # The decision only reads the previous completed bar. A full horizon
        # must be available; trailing bars otherwise remain cash.
        if active is None and position - 1 + horizon_days <= last_position:
            signal = combine_signals(
                safe, computed, horizon_days, idx=position - 1, precomputed=True, policy=policy
            )
            if signal.direction != SignalDirection.HOLD:
                entry_equity = portfolio.equity
                fill = portfolio.open_position(
                    float(row.Open), fee_rate, short=signal.direction == SignalDirection.SELL
                )
                active = {
                    "entry_date": timestamp, "signal_date": df.index[position - 1],
                    "entry_price": fill.price, "entry_equity": entry_equity,
                    "quantity": fill.quantity, "fee": fill.fee,
                    "direction": signal.direction.value,
                    "target": position - 1 + horizon_days,
                }

        exposed = active is not None
        exit_price = None
        reason = "horizon"
        if active is not None:
            liquidation = portfolio.short_liquidation_price(fee_rate)
            if liquidation is not None and row.High >= liquidation:
                exit_price = max(float(row.Open), liquidation)
                reason = "liquidation"
            elif position == active["target"]:
                exit_price = float(row.Close)
            if exit_price is not None:
                fill = portfolio.close_position(exit_price, fee_rate)
                if reason == "liquidation" and math.isclose(portfolio.cash, 0, abs_tol=1e-8):
                    portfolio.cash = 0.0
                pnl = portfolio.cash - active["entry_equity"]
                actual = "BUY" if exit_price > active["entry_price"] else (
                    "SELL" if exit_price < active["entry_price"] else "HOLD"
                )
                report.trades.append(Trade(
                    entry_date=active["entry_date"], exit_date=timestamp,
                    direction=active["direction"], entry_price=active["entry_price"],
                    exit_price=exit_price, predicted_direction=active["direction"],
                    actual_direction=actual, correct=active["direction"] == actual,
                    pnl_pct=pnl / active["entry_equity"], pnl_dollars=pnl,
                    signal_date=active["signal_date"], target_date=df.index[active["target"]],
                    quantity=active["quantity"], entry_equity=active["entry_equity"],
                    fees=active["fee"] + fill.fee, exit_reason=reason,
                ))
                active = None

        portfolio.last_price = float(row.Close)
        report.snapshots.append(PortfolioSnapshot(
            timestamp, portfolio.cash, portfolio.quantity, float(row.Close), portfolio.equity, exposed
        ))
        equity.append(portfolio.equity)
        dates.append(timestamp)
        if reason == "liquidation" or portfolio.equity <= 0:
            report.completion_reason = "insolvency"
            report.evaluation_end = timestamp
            break

    report.equity_curve = pd.Series(equity, index=pd.DatetimeIndex(dates), dtype=float)
    return compute_metrics(report)
