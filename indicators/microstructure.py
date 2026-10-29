"""Microstructure indicators: VPIN (Volume-Synchronized Probability of Informed Trading).

Uses Bulk Volume Classification (BVC) to estimate trade-flow toxicity from
standard OHLCV data.  High VPIN indicates elevated informed-trading activity
and reliably predicts short-term volatility spikes and adverse price moves.

Reference: Easley, López de Prado & O'Hara (2012), "Flow Toxicity and
Liquidity in a High-Frequency World".
"""

from typing import Any

import numpy as np
import pandas as pd

from indicators.base import BaseIndicator
from indicators.registry import register
from signals.base import SignalDirection, SignalResult


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Approximate standard-normal CDF (Abramowitz & Stegun).

    Accurate to ~1e-7; avoids a scipy dependency.
    """
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    p = 0.2316419

    t = 1.0 / (1.0 + p * np.abs(z))
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)
    phi = 1.0 - pdf * poly
    return np.where(z >= 0, phi, 1.0 - phi)


@register
class VPIN(BaseIndicator):
    """Volume-Synchronized Probability of Informed Trading.

    Applies Bulk Volume Classification to partition each bar's volume into
    buy- and sell-initiated components, then measures the rolling absolute
    order-flow imbalance.  VPIN values are z-scored against their own recent
    history to produce context-sensitive signals.

    High VPIN (>2σ) → extreme toxicity, imminent volatility → SELL
    Elevated VPIN (>1σ) → caution → mild SELL
    Low VPIN (<−1σ) → calm, uninformed flow → mild BUY
    """

    historical_safe = True

    BUCKET_COUNT = 50
    LOOKBACK_WINDOW = 50

    @property
    def name(self) -> str:
        return "VPIN"

    @property
    def category(self) -> str:
        return "microstructure"

    @property
    def lookback(self) -> int:
        return self.LOOKBACK_WINDOW

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)

        if n < self.lookback:
            df["VPIN"] = np.nan
            df["BuyVolPct"] = np.nan
            return df

        close = df["Close"].values
        open_ = df["Open"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values

        # Bar range as volatility proxy for BVC
        sigma = high - low
        sigma = np.where(sigma <= 0, 1e-10, sigma)

        # Normalised price change → normal CDF → buy-fraction
        z = (close - open_) / sigma
        buy_frac = _norm_cdf(z)
        buy_volume = volume * buy_frac
        sell_volume = volume * (1.0 - buy_frac)

        df["BuyVolPct"] = buy_frac

        # Rolling VPIN: |buy − sell| / total over a volume window
        abs_imbalance = np.abs(buy_volume - sell_volume)
        window = self.BUCKET_COUNT
        vpin_vals = np.full(n, np.nan)

        for i in range(window - 1, n):
            total_vol = volume[i - window + 1 : i + 1].sum()
            if total_vol > 0:
                vpin_vals[i] = abs_imbalance[i - window + 1 : i + 1].sum() / total_vol

        df["VPIN"] = vpin_vals
        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "VPIN" not in df.columns:
            df = self.compute(df)

        vpin = df["VPIN"].iloc[idx]
        if pd.isna(vpin):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        # Z-score relative to recent history
        actual_idx = idx if idx >= 0 else len(df) + idx
        start = max(0, actual_idx - 50)
        history = df["VPIN"].iloc[start : actual_idx + 1].dropna()

        if len(history) > 5:
            mean = history.mean()
            std = history.std()
            z_score = (vpin - mean) / std if std > 0 else 0.0
        else:
            z_score = 0.0

        # Extreme toxicity
        if z_score > 2.0:
            confidence = min(0.9, 0.5 + z_score * 0.1)
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                f"Extreme flow toxicity (VPIN={vpin:.3f}, {z_score:+.1f}σ)")

        # Elevated toxicity
        if z_score > 1.0:
            confidence = min(0.6, 0.3 + z_score * 0.1)
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                f"Elevated flow toxicity (VPIN={vpin:.3f}, {z_score:+.1f}σ)")

        # Calm conditions
        if z_score < -1.0:
            confidence = min(0.4, abs(z_score) * 0.15)
            return SignalResult(self.name, SignalDirection.BUY, confidence,
                                f"Low flow toxicity (VPIN={vpin:.3f}), calm conditions")

        return SignalResult(self.name, SignalDirection.HOLD, 0.0,
                            f"Normal flow toxicity (VPIN={vpin:.3f})")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": False,
            "columns": ["VPIN"],
            "colors": {"VPIN": "#9C27B0"},
            "subplot_title": "VPIN (Flow Toxicity)",
        }
