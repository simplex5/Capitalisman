"""Momentum indicators: RSI, Stochastic Oscillator."""

from typing import Any

import pandas as pd
import ta

from indicators.base import BaseIndicator
from indicators.registry import register
from signals.base import SignalDirection, SignalResult


@register
class RSI(BaseIndicator):

    historical_safe = True
    @property
    def name(self) -> str:
        return "RSI"

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def lookback(self) -> int:
        return self.setting('RSI_PERIOD')

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.lookback:
            df["RSI"] = float("nan")
            return df
        df["RSI"] = ta.momentum.rsi(df["Close"], window=self.setting('RSI_PERIOD'))
        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "RSI" not in df.columns:
            df = self.compute(df)

        rsi = df["RSI"].iloc[idx]

        if pd.isna(rsi):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        rsi_oversold = self.setting("RSI_OVERSOLD")
        rsi_overbought = self.setting("RSI_OVERBOUGHT")

        if rsi < rsi_oversold:
            # Deeper oversold = higher confidence
            confidence = min(1.0, (rsi_oversold - rsi) / rsi_oversold)
            return SignalResult(self.name, SignalDirection.BUY, confidence,
                                f"Oversold (RSI={rsi:.1f})")

        if rsi > rsi_overbought:
            confidence = min(1.0, (rsi - rsi_overbought) / (100 - rsi_overbought))
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                f"Overbought (RSI={rsi:.1f})")

        # Neutral zone — weak directional bias based on distance from midpoint
        if rsi < 45:
            return SignalResult(self.name, SignalDirection.SELL, 0.15,
                                f"Leaning bearish (RSI={rsi:.1f})")
        elif rsi > 55:
            return SignalResult(self.name, SignalDirection.BUY, 0.15,
                                f"Leaning bullish (RSI={rsi:.1f})")

        return SignalResult(self.name, SignalDirection.HOLD, 0.0, f"Neutral (RSI={rsi:.1f})")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": False,
            "columns": ["RSI"],
            "colors": {"RSI": "#9C27B0"},
            "subplot_title": "RSI",
        }


@register
class Stochastic(BaseIndicator):

    historical_safe = True
    @property
    def name(self) -> str:
        return "Stochastic"

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def lookback(self) -> int:
        return self.setting('STOCH_K') + self.setting('STOCH_D')

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.lookback:
            df["Stoch_K"] = float("nan")
            df["Stoch_D"] = float("nan")
            return df
        stoch = ta.momentum.StochasticOscillator(
            df["High"], df["Low"], df["Close"],
            window=self.setting('STOCH_K'),
            smooth_window=self.setting('STOCH_SMOOTH'),
        )
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()
        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "Stoch_K" not in df.columns:
            df = self.compute(df)

        k = df["Stoch_K"].iloc[idx]
        d = df["Stoch_D"].iloc[idx]

        if pd.isna(k) or pd.isna(d):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        prev_idx = idx - 1 if idx != -1 else -2
        prev_k = df["Stoch_K"].iloc[prev_idx]
        prev_d = df["Stoch_D"].iloc[prev_idx]

        if pd.isna(prev_k) or pd.isna(prev_d):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        stoch_oversold = self.setting("STOCH_OVERSOLD")
        stoch_overbought = self.setting("STOCH_OVERBOUGHT")

        # Bullish: %K crosses above %D in oversold zone
        if k < stoch_oversold and prev_k <= prev_d and k > d:
            confidence = min(1.0, (stoch_oversold - k) / stoch_oversold)
            return SignalResult(self.name, SignalDirection.BUY, max(0.5, confidence),
                                f"Bullish crossover in oversold zone (%K={k:.1f}, %D={d:.1f})")

        # Bearish: %K crosses below %D in overbought zone
        if k > stoch_overbought and prev_k >= prev_d and k < d:
            confidence = min(1.0, (k - stoch_overbought) / (100 - stoch_overbought))
            return SignalResult(self.name, SignalDirection.SELL, max(0.5, confidence),
                                f"Bearish crossover in overbought zone (%K={k:.1f}, %D={d:.1f})")

        # In oversold/overbought but no crossover
        if k < stoch_oversold:
            return SignalResult(self.name, SignalDirection.BUY, 0.3,
                                f"Oversold (%K={k:.1f})")
        if k > stoch_overbought:
            return SignalResult(self.name, SignalDirection.SELL, 0.3,
                                f"Overbought (%K={k:.1f})")

        return SignalResult(self.name, SignalDirection.HOLD, 0.0,
                            f"Neutral (%K={k:.1f}, %D={d:.1f})")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": False,
            "columns": ["Stoch_K", "Stoch_D"],
            "colors": {"Stoch_K": "#2196F3", "Stoch_D": "#FF9800"},
            "subplot_title": "Stochastic",
        }
