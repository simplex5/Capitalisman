"""Volume indicators: VWAP, OBV."""

from typing import Any

import pandas as pd
import ta

from indicators.base import BaseIndicator
from indicators.registry import register
from signals.base import SignalDirection, SignalResult


@register
class VWAP(BaseIndicator):

    historical_safe = True
    WINDOW = 20  # rolling anchor period

    @property
    def name(self) -> str:
        return "VWAP"

    @property
    def category(self) -> str:
        return "volume"

    @property
    def lookback(self) -> int:
        return self.WINDOW

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.lookback:
            df["VWAP"] = float("nan")
            return df
        # Rolling anchored VWAP: use a rolling window so the value stays
        # relevant on daily charts instead of drifting toward a cumulative
        # historical average.
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        tp_vol = typical_price * df["Volume"]
        df["VWAP"] = tp_vol.rolling(window=self.WINDOW).sum() / df["Volume"].rolling(window=self.WINDOW).sum()
        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "VWAP" not in df.columns:
            df = self.compute(df)

        close = df["Close"].iloc[idx]
        vwap = df["VWAP"].iloc[idx]

        if pd.isna(vwap) or vwap == 0:
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        pct_diff = (close - vwap) / vwap

        if close > vwap:
            confidence = min(0.8, abs(pct_diff) * 10)
            return SignalResult(self.name, SignalDirection.BUY, confidence,
                                f"Price above VWAP ({pct_diff:+.2%})")

        if close < vwap:
            confidence = min(0.8, abs(pct_diff) * 10)
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                f"Price below VWAP ({pct_diff:+.2%})")

        return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Price at VWAP")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": True,
            "columns": ["VWAP"],
            "colors": {"VWAP": "#FF5722"},
        }


@register
class OBV(BaseIndicator):

    historical_safe = True
    @property
    def name(self) -> str:
        return "OBV"

    @property
    def category(self) -> str:
        return "volume"

    @property
    def lookback(self) -> int:
        return 20

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.lookback:
            df["OBV"] = float("nan")
            df["OBV_SMA"] = float("nan")
            return df
        df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        # OBV SMA for divergence detection
        df["OBV_SMA"] = df["OBV"].rolling(window=20).mean()
        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "OBV" not in df.columns:
            df = self.compute(df)

        position = idx if idx >= 0 else len(df) + idx
        if not 0 <= position < len(df):
            raise IndexError("OBV signal index is out of range")
        if position < self.lookback - 1:
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        obv = df["OBV"].iloc[position]
        obv_sma = df["OBV_SMA"].iloc[position]

        if pd.isna(obv) or pd.isna(obv_sma):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        # Compare price trend vs OBV trend over last 10 bars
        lookback = min(10, position)
        if lookback < 2:
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        start_idx = position - lookback

        price_start = df["Close"].iloc[start_idx]
        obv_start = df["OBV"].iloc[start_idx]

        if pd.isna(price_start) or pd.isna(obv_start):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        price_change = df["Close"].iloc[position] - price_start
        obv_change = df["OBV"].iloc[idx] - obv_start

        obv_slice = df["OBV"].iloc[start_idx : position + 1]
        obv_range = obv_slice.max() - obv_slice.min()
        if pd.isna(obv_range) or obv_range == 0:
            obv_range = 1

        # Positive divergence: price falling but OBV rising
        if price_change < 0 and obv_change > 0:
            confidence = min(0.8, abs(obv_change) / obv_range)
            return SignalResult(self.name, SignalDirection.BUY, confidence,
                                "Positive OBV divergence (price down, volume up)")

        # Negative divergence: price rising but OBV falling
        if price_change > 0 and obv_change < 0:
            confidence = min(0.8, abs(obv_change) / obv_range)
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                "Negative OBV divergence (price up, volume down)")

        # Confirmation: both moving same direction
        if obv > obv_sma:
            return SignalResult(self.name, SignalDirection.BUY, 0.3,
                                "OBV above its moving average (volume confirms uptrend)")
        elif obv < obv_sma:
            return SignalResult(self.name, SignalDirection.SELL, 0.3,
                                "OBV below its moving average (volume confirms downtrend)")

        return SignalResult(self.name, SignalDirection.HOLD, 0.0, "OBV neutral")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": False,
            "columns": ["OBV", "OBV_SMA"],
            "colors": {"OBV": "#4CAF50", "OBV_SMA": "#FF9800"},
            "subplot_title": "OBV",
        }
