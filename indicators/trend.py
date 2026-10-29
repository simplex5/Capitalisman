"""Trend indicators: SMA Crossover, EMA Crossover, MACD, ADX."""

from typing import Any

import pandas as pd
import ta

from indicators.base import BaseIndicator
from indicators.registry import register
from signals.base import SignalDirection, SignalResult


@register
class SMACrossover(BaseIndicator):

    historical_safe = True
    @property
    def name(self) -> str:
        return "SMA Crossover"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def lookback(self) -> int:
        return self.setting('SMA_LONG')

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.lookback:
            df["SMA_short"] = float("nan")
            df["SMA_long"] = float("nan")
            return df
        df["SMA_short"] = ta.trend.sma_indicator(df["Close"], window=self.setting('SMA_SHORT'))
        df["SMA_long"] = ta.trend.sma_indicator(df["Close"], window=self.setting('SMA_LONG'))
        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "SMA_short" not in df.columns:
            df = self.compute(df)

        sma_s = df["SMA_short"].iloc[idx]
        sma_l = df["SMA_long"].iloc[idx]

        if pd.isna(sma_s) or pd.isna(sma_l):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        prev_idx = idx - 1 if idx != -1 else -2
        prev_s = df["SMA_short"].iloc[prev_idx]
        prev_l = df["SMA_long"].iloc[prev_idx]

        if pd.isna(prev_s) or pd.isna(prev_l):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        # Golden cross: short crosses above long
        if prev_s <= prev_l and sma_s > sma_l:
            spread = abs(sma_s - sma_l) / sma_l
            confidence = min(1.0, spread * 20)
            return SignalResult(self.name, SignalDirection.BUY, confidence,
                                f"Golden cross: SMA{self.setting('SMA_SHORT')} crossed above SMA{self.setting('SMA_LONG')}")

        # Death cross: short crosses below long
        if prev_s >= prev_l and sma_s < sma_l:
            spread = abs(sma_l - sma_s) / sma_l
            confidence = min(1.0, spread * 20)
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                f"Death cross: SMA{self.setting('SMA_SHORT')} crossed below SMA{self.setting('SMA_LONG')}")

        # No crossover — mild directional bias
        if sma_s > sma_l:
            spread = (sma_s - sma_l) / sma_l
            return SignalResult(self.name, SignalDirection.BUY, min(0.4, spread * 10),
                                f"SMA{self.setting('SMA_SHORT')} above SMA{self.setting('SMA_LONG')}")
        elif sma_s < sma_l:
            spread = (sma_l - sma_s) / sma_l
            return SignalResult(self.name, SignalDirection.SELL, min(0.4, spread * 10),
                                f"SMA{self.setting('SMA_SHORT')} below SMA{self.setting('SMA_LONG')}")

        return SignalResult(self.name, SignalDirection.HOLD, 0.0, "SMAs equal")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": True,
            "columns": ["SMA_short", "SMA_long"],
            "colors": {"SMA_short": "#2196F3", "SMA_long": "#FF9800"},
        }


@register
class EMACrossover(BaseIndicator):

    historical_safe = True
    @property
    def name(self) -> str:
        return "EMA Crossover"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def lookback(self) -> int:
        return self.setting('EMA_LONG')

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.lookback:
            df["EMA_short"] = float("nan")
            df["EMA_long"] = float("nan")
            return df
        df["EMA_short"] = ta.trend.ema_indicator(df["Close"], window=self.setting('EMA_SHORT'))
        df["EMA_long"] = ta.trend.ema_indicator(df["Close"], window=self.setting('EMA_LONG'))
        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "EMA_short" not in df.columns:
            df = self.compute(df)

        ema_s = df["EMA_short"].iloc[idx]
        ema_l = df["EMA_long"].iloc[idx]

        if pd.isna(ema_s) or pd.isna(ema_l):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        prev_idx = idx - 1 if idx != -1 else -2
        prev_s = df["EMA_short"].iloc[prev_idx]
        prev_l = df["EMA_long"].iloc[prev_idx]

        if pd.isna(prev_s) or pd.isna(prev_l):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        if prev_s <= prev_l and ema_s > ema_l:
            spread = abs(ema_s - ema_l) / ema_l
            confidence = min(1.0, spread * 25)
            return SignalResult(self.name, SignalDirection.BUY, confidence,
                                f"EMA{self.setting('EMA_SHORT')} crossed above EMA{self.setting('EMA_LONG')}")

        if prev_s >= prev_l and ema_s < ema_l:
            spread = abs(ema_l - ema_s) / ema_l
            confidence = min(1.0, spread * 25)
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                f"EMA{self.setting('EMA_SHORT')} crossed below EMA{self.setting('EMA_LONG')}")

        if ema_s > ema_l:
            spread = (ema_s - ema_l) / ema_l
            return SignalResult(self.name, SignalDirection.BUY, min(0.4, spread * 12),
                                f"EMA{self.setting('EMA_SHORT')} above EMA{self.setting('EMA_LONG')}")
        elif ema_s < ema_l:
            spread = (ema_l - ema_s) / ema_l
            return SignalResult(self.name, SignalDirection.SELL, min(0.4, spread * 12),
                                f"EMA{self.setting('EMA_SHORT')} below EMA{self.setting('EMA_LONG')}")

        return SignalResult(self.name, SignalDirection.HOLD, 0.0, "EMAs equal")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": True,
            "columns": ["EMA_short", "EMA_long"],
            "colors": {"EMA_short": "#00BCD4", "EMA_long": "#E91E63"},
        }


@register
class MACD(BaseIndicator):

    historical_safe = True
    @property
    def name(self) -> str:
        return "MACD"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def lookback(self) -> int:
        return self.setting('MACD_SLOW') + self.setting('MACD_SIGNAL')

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.lookback:
            df["MACD_line"] = float("nan")
            df["MACD_signal"] = float("nan")
            df["MACD_hist"] = float("nan")
            return df
        macd_obj = ta.trend.MACD(
            df["Close"],
            window_slow=self.setting('MACD_SLOW'),
            window_fast=self.setting('MACD_FAST'),
            window_sign=self.setting('MACD_SIGNAL'),
        )
        df["MACD_line"] = macd_obj.macd()
        df["MACD_signal"] = macd_obj.macd_signal()
        df["MACD_hist"] = macd_obj.macd_diff()
        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "MACD_line" not in df.columns:
            df = self.compute(df)

        macd = df["MACD_line"].iloc[idx]
        signal = df["MACD_signal"].iloc[idx]
        hist = df["MACD_hist"].iloc[idx]
        close = df["Close"].iloc[idx]

        if pd.isna(macd) or pd.isna(signal):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        prev_idx = idx - 1 if idx != -1 else -2
        prev_macd = df["MACD_line"].iloc[prev_idx]
        prev_signal = df["MACD_signal"].iloc[prev_idx]

        if pd.isna(prev_macd) or pd.isna(prev_signal):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        # Normalize histogram by price so confidence is scale-independent
        norm_hist = abs(hist) / close if close > 0 else 0.0

        # Bullish crossover
        if prev_macd <= prev_signal and macd > signal:
            confidence = min(1.0, norm_hist * 200)
            return SignalResult(self.name, SignalDirection.BUY, confidence,
                                "MACD crossed above signal line")

        # Bearish crossover
        if prev_macd >= prev_signal and macd < signal:
            confidence = min(1.0, norm_hist * 200)
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                "MACD crossed below signal line")

        # Trend continuation
        if macd > signal:
            confidence = min(0.5, norm_hist * 120)
            return SignalResult(self.name, SignalDirection.BUY, confidence,
                                "MACD above signal line")
        elif macd < signal:
            confidence = min(0.5, norm_hist * 120)
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                "MACD below signal line")

        return SignalResult(self.name, SignalDirection.HOLD, 0.0, "MACD neutral")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": False,
            "columns": ["MACD_line", "MACD_signal", "MACD_hist"],
            "colors": {"MACD_line": "#2196F3", "MACD_signal": "#FF9800", "MACD_hist": "#9E9E9E"},
            "subplot_title": "MACD",
        }


@register
class ADX(BaseIndicator):

    historical_safe = True
    @property
    def name(self) -> str:
        return "ADX"

    @property
    def category(self) -> str:
        return "trend"

    @property
    def lookback(self) -> int:
        return self.setting('ADX_PERIOD') * 2

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.lookback:
            df["ADX"] = float("nan")
            df["ADX_pos"] = float("nan")
            df["ADX_neg"] = float("nan")
            return df
        adx_obj = ta.trend.ADXIndicator(
            df["High"], df["Low"], df["Close"], window=self.setting('ADX_PERIOD')
        )
        df["ADX"] = adx_obj.adx()
        df["ADX_pos"] = adx_obj.adx_pos()
        df["ADX_neg"] = adx_obj.adx_neg()
        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "ADX" not in df.columns:
            df = self.compute(df)

        adx = df["ADX"].iloc[idx]
        plus_di = df["ADX_pos"].iloc[idx]
        minus_di = df["ADX_neg"].iloc[idx]

        if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        # ADX < 20 means weak trend
        if adx < 20:
            return SignalResult(self.name, SignalDirection.HOLD, 0.2,
                                f"Weak trend (ADX={adx:.1f})")

        # Strong trend: use DI direction
        trend_strength = min(1.0, (adx - 20) / 40)  # normalize 20-60 -> 0-1

        if plus_di > minus_di:
            return SignalResult(self.name, SignalDirection.BUY, trend_strength,
                                f"Bullish trend (ADX={adx:.1f}, +DI={plus_di:.1f} > -DI={minus_di:.1f})")
        elif minus_di > plus_di:
            return SignalResult(self.name, SignalDirection.SELL, trend_strength,
                                f"Bearish trend (ADX={adx:.1f}, -DI={minus_di:.1f} > +DI={plus_di:.1f})")

        return SignalResult(self.name, SignalDirection.HOLD, 0.1, "DIs equal")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": False,
            "columns": ["ADX", "ADX_pos", "ADX_neg"],
            "colors": {"ADX": "#9C27B0", "ADX_pos": "#4CAF50", "ADX_neg": "#F44336"},
            "subplot_title": "ADX",
        }
