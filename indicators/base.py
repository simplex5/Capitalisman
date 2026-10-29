"""Base indicator abstract class."""

from abc import ABC, abstractmethod
from copy import copy
from types import MappingProxyType
from typing import Any

import pandas as pd

from signals.base import SignalResult


class BaseIndicator(ABC):
    """Abstract base class for all technical indicators."""

    historical_safe = False  # opt in only with a tested causal input contract

    def with_parameters(self, parameters) -> "BaseIndicator":
        """Return a separate indicator bound to immutable run settings."""
        indicator = copy(self)
        indicator._parameters = MappingProxyType(dict(parameters))
        return indicator

    def setting(self, name: str):
        parameters = getattr(self, "_parameters", None)
        if parameters is not None:
            return parameters[name]
        from config.overrides import get_setting
        return get_setting(name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable indicator name."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Category: trend, momentum, volatility, or volume."""

    @property
    @abstractmethod
    def lookback(self) -> int:
        """Minimum number of bars needed before the indicator produces valid values."""

    @property
    def backtest_safe(self) -> bool:
        """Whether historical values are available without revision look-ahead."""
        return self.historical_safe

    def supports_backtest_horizon(self, horizon_days: int) -> bool:
        """Whether this indicator can be evaluated at the requested horizon."""
        return True

    def compute_for_horizon(
        self, df: pd.DataFrame, horizon_days: int
    ) -> pd.DataFrame:
        """Compute values for a requested horizon when an indicator needs it."""
        return self.compute(df)

    def compute_for_backtest(
        self, df: pd.DataFrame, horizon_days: int, cost_per_trade_pct: float
    ) -> pd.DataFrame:
        """Compute historical values using the run's execution cost policy."""
        return self.compute_for_horizon(df, horizon_days)

    def get_signal_for_horizon(
        self, df: pd.DataFrame, horizon_days: int, idx: int = -1
    ) -> SignalResult:
        """Generate a horizon-aware signal when supported by an indicator."""
        return self.get_signal(df, idx=idx)

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicator values and add columns to a copy of df.

        Must not modify the input DataFrame.
        Returns DataFrame with additional indicator columns.
        """

    @abstractmethod
    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        """Generate a signal at the given bar index.

        Args:
            df: DataFrame with indicator columns already computed.
            idx: Bar index to evaluate (default -1 for latest).

        Returns:
            SignalResult with direction and confidence.
        """

    @abstractmethod
    def get_chart_config(self) -> dict[str, Any]:
        """Return chart overlay/subplot configuration.

        Returns a dict with keys:
            - 'overlay': bool — whether to draw on price pane
            - 'columns': list of column names to plot
            - 'colors': dict mapping column name to color
            - 'subplot_title': str (only if overlay=False)
        """
