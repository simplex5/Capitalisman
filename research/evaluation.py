"""Fixed-strategy comparisons on an explicitly selected chronological test window."""

from dataclasses import fields

import pandas as pd

from backtesting.engine import _validate_data, run_backtest
from config.parameters import capture_parameters
from config.settings import WARMUP_BUFFER
from indicators.base import BaseIndicator
from indicators.registry import get_indicator
from research.artifacts import build_archive
from signals.base import SignalDirection, SignalResult
from signals.combiner import ScoringPolicy
from simulation.strategies import CORE_SIMULATION_INDICATORS


class _ConstantDirection(BaseIndicator):
    historical_safe = True

    def __init__(self, direction):
        self.direction = direction

    @property
    def name(self):
        return "Buy and hold" if self.direction == SignalDirection.BUY else "Cash"

    @property
    def category(self):
        return "trend"

    @property
    def lookback(self):
        return 1

    def compute(self, df):
        return df.copy()

    def get_signal(self, df, idx=-1):
        return SignalResult(self.name, self.direction, 1.0)

    def get_chart_config(self):
        return {"overlay": False, "columns": [], "colors": {}}


def compare_strategies(
    df, *, ticker, test_start, horizon=5, cost_pct=0.1, initial_capital=10_000,
    indicators=None,
):
    """Compare frozen technical/forecast signals, SMA, buy-and-hold and cash.

    Parameters are captured before any test outcomes are evaluated. No fitting,
    threshold search or winner selection occurs. All strategies use the same
    first execution bar and end date; insolvency is explicitly reported.
    """
    df = _validate_data(df)
    start = pd.Timestamp(test_start)
    if pd.isna(start) or (start.tzinfo is None) != (df.index.tz is None):
        raise ValueError("Test start must be valid and match data timezone awareness")
    position = int(df.index.searchsorted(start))
    if position >= len(df):
        raise ValueError("Test window is empty")
    parameters = dict(capture_parameters())
    names = CORE_SIMULATION_INDICATORS
    chosen = {name: get_indicator(name) for name in names} if indicators is None else dict(indicators)
    if not chosen or any(not indicator.backtest_safe for indicator in chosen.values()):
        raise ValueError("Select at least one historically safe indicator")
    sma = get_indicator("SMA Crossover")
    required = max(ind.with_parameters(parameters).lookback for ind in [*chosen.values(), sma]) + WARMUP_BUFFER + 1
    if position < required:
        raise ValueError(f"Test start must leave at least {required} warmup bars")
    if horizon < 1 or horizon > len(df) - position:
        raise ValueError("Test period must contain a complete positive trade horizon")
    setup = {
        "combined": (chosen, horizon, ScoringPolicy.from_indicators(chosen, horizon)),
        "sma": ({sma.name: sma}, horizon, ScoringPolicy.from_indicators({sma.name: sma}, horizon)),
    }
    for name, direction in (("buy_hold", SignalDirection.BUY), ("cash", SignalDirection.HOLD)):
        indicator = _ConstantDirection(direction)
        setup[name] = ({indicator.name: indicator}, len(df) - position,
                       ScoringPolicy(((indicator.name, 1.0),), 0.1))
    reports = {}
    for name, (selected, holding_period, policy) in setup.items():
        reports[name] = run_backtest(
            df, selected, ticker=ticker, period="held_out", horizon_days=holding_period,
            initial_capital=initial_capital, cost_per_trade_pct=cost_pct,
            evaluation_start=df.index[position], indicator_parameters=parameters, scoring_policy=policy,
        )
    return reports


def strategy_summary(reports):
    return pd.DataFrame([
        {
            "strategy": name, "return": report.cumulative_return,
            "max_drawdown": report.max_drawdown, "daily_sharpe": report.sharpe_ratio,
            "trades": report.total_trades, "exposure": report.exposure_pct,
            "start": str(report.evaluation_start), "end": str(report.evaluation_end),
            "completion": report.completion_reason,
        }
        for name, report in reports.items()
    ])


def build_comparison_archive(df, reports):
    frames = {"market": df}
    results = {}
    for name, report in reports.items():
        frames[f"computed_{name}"] = report.computed_data
        results[name] = {field.name: getattr(report, field.name) for field in fields(report)
                         if field.name != "computed_data"}
    return build_archive("strategy_comparison", frames, {
        "protocol": "Fixed settings before test evaluation; no tuning on test outcomes",
        "comparability": "Common requested window and fill fees; insolvency ends a run early. Sharpe uses each run's observed duration.",
        "strategies": {name: report.configuration for name, report in reports.items()},
    }, {"summary": strategy_summary(reports).to_dict("records"), "reports": results})
