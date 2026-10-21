"""Probabilistic TimesFM 2.5 price forecast indicator."""

from dataclasses import dataclass
from itertools import chain, islice
from typing import Any

import numpy as np
import pandas as pd

from config.settings import DEFAULT_COST_PER_TRADE_PCT
from indicators.base import BaseIndicator
from indicators.registry import register
from ml.timesfm_runtime import (
    TimesFMForecast as RuntimeForecast,
    TimesFMRuntime,
    get_timesfm_runtime,
    probability_above,
)
from signals.base import SignalDirection, SignalResult


DEFAULT_FORECAST_HORIZON = 10
MIN_CONTEXT = 32
MIN_DIRECTIONAL_PROBABILITY = 0.60


@dataclass(frozen=True)
class TimesFMAnalysis:
    origin: pd.Timestamp | object
    horizon: int
    current_price: float
    median_price: float
    lower_price: float
    upper_price: float
    expected_return: float
    downside_return: float
    interval_width: float
    probability_up: float
    probability_profit: float
    probability_down: float
    forecast: RuntimeForecast
    probability_short_profit: float = 0.0
    cost_per_trade_pct: float = DEFAULT_COST_PER_TRADE_PCT


@register
class TimesFMForecast(BaseIndicator):
    """TimesFM 2.5 probabilistic forecast exposed as a research signal."""

    historical_safe = True

    def __init__(self, runtime: TimesFMRuntime | None = None) -> None:
        self.runtime = runtime or get_timesfm_runtime()
        self.latest_analysis: TimesFMAnalysis | None = None
        self.last_error: str | None = None

    @property
    def name(self) -> str:
        return "TimesFM Forecast"

    @property
    def category(self) -> str:
        return "forecast"

    @property
    def lookback(self) -> int:
        return MIN_CONTEXT

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.compute_for_horizon(df, DEFAULT_FORECAST_HORIZON)

    def compute_for_horizon(
        self, df: pd.DataFrame, horizon_days: int
    ) -> pd.DataFrame:
        return self._compute(df, horizon_days, DEFAULT_COST_PER_TRADE_PCT)

    def compute_for_backtest(
        self, df: pd.DataFrame, horizon_days: int, cost_per_trade_pct: float
    ) -> pd.DataFrame:
        return self._compute(df, horizon_days, cost_per_trade_pct, historical=True)

    def _compute(
        self, df: pd.DataFrame, horizon_days: int, cost_per_trade_pct: float,
        historical: bool = False,
    ) -> pd.DataFrame:
        if not np.isfinite(cost_per_trade_pct) or not 0 <= cost_per_trade_pct < 100:
            raise ValueError("Transaction cost must be in [0, 100)")
        result = df.copy()
        self._initialize_columns(result)
        result.attrs["timesfm_horizon"] = horizon_days
        result["TFM_cost_pct"] = cost_per_trade_pct
        self.latest_analysis = None
        self.last_error = None

        if len(result) < MIN_CONTEXT:
            self.last_error = f"TimesFM needs at least {MIN_CONTEXT} bars"
            return result

        if not historical and self.runtime.config.use_case in {"interactive", "watchlist"}:
            origins = [len(result) - 1]
        else:
            origins = range(MIN_CONTEXT - 1, len(result), max(1, horizon_days))
            if not historical and origins[-1] != len(result) - 1:
                origins = chain(origins, [len(result) - 1])

        context_length = self.runtime.config.effective_context
        origin_iterator = iter(origins)
        while batch_origins := list(islice(origin_iterator, self.runtime.config.chunk_size)):
            # Cap before dtype conversion, and retain only one inference batch.
            contexts = [
                result["Close"].iloc[max(0, origin + 1 - context_length):origin + 1]
                .to_numpy(dtype=np.float32)
                for origin in batch_origins
            ]
            try:
                forecasts = self.runtime.forecast(contexts, horizon=horizon_days)
            except (RuntimeError, ValueError) as error:
                self.last_error = str(error)
                if historical:
                    # Failed inference must not become a successful cash-only run.
                    raise
                # A later batch failure must not expose partial predictions.
                self._initialize_columns(result)
                self.latest_analysis = None
                return result

            for origin, forecast in zip(batch_origins, forecasts):
                analysis = self._analyze_forecast(
                    forecast,
                    current_price=float(result["Close"].iloc[origin]),
                    origin=result.index[origin],
                    cost_per_trade_pct=cost_per_trade_pct,
                )
                self._write_analysis(result, origin, analysis)
                if origin == len(result) - 1:
                    self.latest_analysis = analysis
            del contexts, forecasts

        return result

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        return self.get_signal_for_horizon(
            df, horizon_days=DEFAULT_FORECAST_HORIZON, idx=idx
        )

    def get_signal_for_horizon(
        self, df: pd.DataFrame, horizon_days: int, idx: int = -1
    ) -> SignalResult:
        if self._has_forecast(df, horizon_days, idx) or self._has_computed_horizon(
            df, horizon_days
        ):
            computed = df
        else:
            computed = self.compute_for_horizon(df, horizon_days)

        probability_up = computed["TFM_probability_up"].iloc[idx]
        probability_profit = computed["TFM_probability_profit"].iloc[idx]
        probability_short_profit = computed["TFM_probability_short_profit"].iloc[idx]
        expected_return = computed["TFM_expected_return"].iloc[idx]
        downside_return = computed["TFM_downside_return"].iloc[idx]

        if pd.isna(probability_up) or pd.isna(expected_return):
            detail = self.last_error or "No TimesFM forecast at this evaluation origin"
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, detail)

        strength = min(0.9, abs(float(probability_up) - 0.5) * 2)
        fee_rate = float(computed["TFM_cost_pct"].iloc[idx]) / 200.0
        long_hurdle = (1 + fee_rate) / (1 - fee_rate) - 1
        short_hurdle = (1 - fee_rate) / (1 + fee_rate) - 1

        if probability_profit >= MIN_DIRECTIONAL_PROBABILITY and expected_return > long_hurdle:
            return SignalResult(
                self.name,
                SignalDirection.BUY,
                strength,
                f"TimesFM median {expected_return:+.1%} over {horizon_days} bars; "
                f"long profit score {probability_profit:.0%}; q10 {downside_return:+.1%}",
            )
        if probability_short_profit >= MIN_DIRECTIONAL_PROBABILITY and expected_return < short_hurdle:
            return SignalResult(
                self.name,
                SignalDirection.SELL,
                strength,
                f"TimesFM median {expected_return:+.1%} over {horizon_days} bars; "
                f"short profit score {probability_short_profit:.0%}; q10 {downside_return:+.1%}",
            )
        return SignalResult(
            self.name,
            SignalDirection.HOLD,
            strength,
            f"TimesFM uncertain: median {expected_return:+.1%}, "
            f"P(up) {probability_up:.0%} over {horizon_days} bars",
        )

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": False,
            "columns": [
                "TFM_expected_return",
                "TFM_downside_return",
                "TFM_upper_return",
            ],
            "colors": {
                "TFM_expected_return": "#9C27B0",
                "TFM_downside_return": "#F44336",
                "TFM_upper_return": "#4CAF50",
            },
            "subplot_title": "TimesFM Forecast Return Distribution",
        }

    @staticmethod
    def _initialize_columns(df: pd.DataFrame) -> None:
        for column in (
            "TFM_point",
            "TFM_q10",
            "TFM_q50",
            "TFM_q90",
            "TFM_expected_return",
            "TFM_downside_return",
            "TFM_upper_return",
            "TFM_interval_width",
            "TFM_probability_up",
            "TFM_probability_profit",
            "TFM_probability_short_profit",
            "TFM_horizon",
        ):
            df[column] = np.nan

    @staticmethod
    def _has_forecast(df: pd.DataFrame, horizon: int, idx: int) -> bool:
        return (
            "TFM_horizon" in df.columns
            and not pd.isna(df["TFM_horizon"].iloc[idx])
            and int(df["TFM_horizon"].iloc[idx]) == horizon
        )

    @staticmethod
    def _has_computed_horizon(df: pd.DataFrame, horizon: int) -> bool:
        """Return whether this frame already contains forecasts for a horizon."""
        if df.attrs.get("timesfm_horizon") == horizon:
            return True  # includes failed attempts, which must not retry at every bar
        if "TFM_horizon" not in df.columns:
            return False
        computed = df["TFM_horizon"].dropna()
        return not computed.empty and bool((computed.astype(int) == horizon).all())

    @staticmethod
    def _analyze_forecast(
        forecast: RuntimeForecast,
        current_price: float,
        origin: pd.Timestamp | object,
        cost_per_trade_pct: float = DEFAULT_COST_PER_TRADE_PCT,
    ) -> TimesFMAnalysis:
        median = forecast.terminal_quantile(0.5)
        lower = forecast.terminal_quantile(0.1)
        upper = forecast.terminal_quantile(0.9)
        fee_rate = cost_per_trade_pct / 200.0
        long_break_even = current_price * (1 + fee_rate) / (1 - fee_rate)
        short_break_even = current_price * (1 - fee_rate) / (1 + fee_rate)
        return TimesFMAnalysis(
            origin=origin,
            horizon=forecast.horizon,
            current_price=current_price,
            median_price=median,
            lower_price=lower,
            upper_price=upper,
            expected_return=median / current_price - 1.0,
            downside_return=lower / current_price - 1.0,
            interval_width=(upper - lower) / current_price,
            probability_up=probability_above(forecast, current_price),
            probability_profit=probability_above(forecast, long_break_even),
            probability_down=1.0 - probability_above(forecast, current_price),
            forecast=forecast,
            probability_short_profit=1.0 - probability_above(forecast, short_break_even),
            cost_per_trade_pct=cost_per_trade_pct,
        )

    @staticmethod
    def _write_analysis(
        df: pd.DataFrame, origin: int, analysis: TimesFMAnalysis
    ) -> None:
        values = {
            "TFM_point": analysis.forecast.terminal_point,
            "TFM_q10": analysis.lower_price,
            "TFM_q50": analysis.median_price,
            "TFM_q90": analysis.upper_price,
            "TFM_expected_return": analysis.expected_return,
            "TFM_downside_return": analysis.downside_return,
            "TFM_upper_return": analysis.upper_price / analysis.current_price - 1.0,
            "TFM_interval_width": analysis.interval_width,
            "TFM_probability_up": analysis.probability_up,
            "TFM_probability_profit": analysis.probability_profit,
            "TFM_probability_short_profit": analysis.probability_short_profit,
            "TFM_horizon": analysis.horizon,
        }
        for column, value in values.items():
            df.iat[origin, df.columns.get_loc(column)] = value
