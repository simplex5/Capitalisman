"""Volatility indicators: Bollinger Bands."""

from typing import Any

import pandas as pd
import ta

from indicators.base import BaseIndicator
from indicators.registry import register
from signals.base import SignalDirection, SignalResult


@register
class BollingerBands(BaseIndicator):

    historical_safe = True
    @property
    def name(self) -> str:
        return "Bollinger Bands"

    @property
    def category(self) -> str:
        return "volatility"

    @property
    def lookback(self) -> int:
        return self.setting('BB_PERIOD')

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.lookback:
            df["BB_upper"] = float("nan")
            df["BB_middle"] = float("nan")
            df["BB_lower"] = float("nan")
            df["BB_pband"] = float("nan")
            return df
        bb = ta.volatility.BollingerBands(
            df["Close"], window=self.setting('BB_PERIOD'), window_dev=self.setting('BB_STD')
        )
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_middle"] = bb.bollinger_mavg()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_pband"] = bb.bollinger_pband()  # %B: position within bands (0=lower, 1=upper)
        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "BB_upper" not in df.columns:
            df = self.compute(df)

        close = df["Close"].iloc[idx]
        upper = df["BB_upper"].iloc[idx]
        lower = df["BB_lower"].iloc[idx]
        pband = df["BB_pband"].iloc[idx]

        if pd.isna(upper) or pd.isna(lower) or pd.isna(pband):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        # Price at or below lower band — oversold
        if close <= lower:
            confidence = min(1.0, max(0.5, (lower - close) / (upper - lower) if upper != lower else 0.5))
            return SignalResult(self.name, SignalDirection.BUY, confidence,
                                f"Price at lower Bollinger Band (%B={pband:.2f})")

        # Price at or above upper band — overbought
        if close >= upper:
            confidence = min(1.0, max(0.5, (close - upper) / (upper - lower) if upper != lower else 0.5))
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                f"Price at upper Bollinger Band (%B={pband:.2f})")

        # Near bands
        if pband < 0.2:
            return SignalResult(self.name, SignalDirection.BUY, 0.3,
                                f"Price near lower band (%B={pband:.2f})")
        if pband > 0.8:
            return SignalResult(self.name, SignalDirection.SELL, 0.3,
                                f"Price near upper band (%B={pband:.2f})")

        return SignalResult(self.name, SignalDirection.HOLD, 0.0,
                            f"Price within bands (%B={pband:.2f})")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": True,
            "columns": ["BB_upper", "BB_middle", "BB_lower"],
            "colors": {"BB_upper": "#F44336", "BB_middle": "#9E9E9E", "BB_lower": "#4CAF50"},
        }
