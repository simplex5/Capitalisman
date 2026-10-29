"""Structural indicators: Bubble Risk (Hurst exponent + log-price acceleration).

Detects super-exponential growth patterns characteristic of speculative bubbles.
Inspired by LPPLS (Log-Periodic Power Law Singularity) research without the
complex nonlinear fitting — uses the Hurst exponent for persistence measurement
and quadratic curvature in log-price space for acceleration detection.

Key insight: bubbles exhibit faster-than-exponential growth (H >> 0.5) combined
with positive acceleration in log-prices.  This combination is rare outside of
genuine speculative episodes.
"""

from typing import Any

import numpy as np
import pandas as pd

from indicators.base import BaseIndicator
from indicators.registry import register
from signals.base import SignalDirection, SignalResult


def _hurst_exponent(returns: np.ndarray, max_lag: int = 20) -> float:
    """Compute the Hurst exponent via Rescaled Range (R/S) analysis.

    H > 0.5: persistent / trending
    H ≈ 0.5: random walk
    H < 0.5: mean-reverting / anti-persistent
    """
    n = len(returns)
    if n < max_lag * 2:
        return 0.5  # insufficient data → assume random walk

    rs_points: list[tuple[float, float]] = []

    for lag in range(2, max_lag + 1):
        n_subseries = n // lag
        if n_subseries < 1:
            continue
        rs_vals: list[float] = []
        for j in range(n_subseries):
            chunk = returns[j * lag : (j + 1) * lag]
            deviation = chunk - chunk.mean()
            cumulative = np.cumsum(deviation)
            r = cumulative.max() - cumulative.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_points.append((np.log(lag), np.log(np.mean(rs_vals))))

    if len(rs_points) < 3:
        return 0.5

    x = np.array([p[0] for p in rs_points])
    y = np.array([p[1] for p in rs_points])
    slope = np.polyfit(x, y, 1)[0]
    return float(max(0.0, min(1.0, slope)))


def _log_price_acceleration(prices: np.ndarray, window: int = 60) -> float:
    """Quadratic coefficient of log(price) over a trailing window.

    A positive value indicates super-exponential growth (bubble signature).
    """
    if len(prices) < window or window < 10:
        return 0.0

    log_p = np.log(prices[-window:])
    mask = np.isfinite(log_p)
    if mask.sum() < 10:
        return 0.0
    log_p = log_p[mask]
    x = np.arange(len(log_p), dtype=float)

    try:
        coeffs = np.polyfit(x, log_p, 2)
        return float(coeffs[0])  # acceleration (curvature) coefficient
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


@register
class BubbleRisk(BaseIndicator):
    """Bubble risk detector combining Hurst exponent and log-price acceleration.

    A high Hurst exponent (persistent trending) together with positive
    log-price acceleration (curvature) is the statistical fingerprint of
    speculative bubbles.  The composite score ranges 0-1 where values
    above 0.35 warrant caution and above 0.6 indicate elevated bubble risk.
    """

    historical_safe = True

    HURST_WINDOW = 120
    ACCEL_WINDOW = 60

    @property
    def name(self) -> str:
        return "Bubble Risk"

    @property
    def category(self) -> str:
        return "structural"

    @property
    def lookback(self) -> int:
        return self.HURST_WINDOW

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        prices = df["Close"].values
        n = len(prices)

        hurst_arr = np.full(n, np.nan)
        accel_arr = np.full(n, np.nan)
        score_arr = np.full(n, np.nan)

        for i in range(self.HURST_WINDOW, n):
            window_prices = prices[max(0, i - self.HURST_WINDOW) : i + 1]
            log_ret = np.diff(np.log(window_prices))
            log_ret = log_ret[np.isfinite(log_ret)]

            h = _hurst_exponent(log_ret, max_lag=min(20, max(3, len(log_ret) // 3)))
            hurst_arr[i] = h

            accel = _log_price_acceleration(window_prices, self.ACCEL_WINDOW)
            accel_arr[i] = accel

            # Composite score: persistence + acceleration
            h_component = max(0.0, (h - 0.5) * 2)  # 0 at H=0.5, 1 at H=1.0
            a_component = max(0.0, min(1.0, accel * 5000))
            score_arr[i] = h_component * 0.4 + a_component * 0.6

        df["Hurst"] = hurst_arr
        df["LogAccel"] = accel_arr
        df["BubbleScore"] = score_arr

        return df

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "BubbleScore" not in df.columns:
            df = self.compute(df)

        hurst = df["Hurst"].iloc[idx]
        score = df["BubbleScore"].iloc[idx]

        if pd.isna(score) or pd.isna(hurst):
            return SignalResult(self.name, SignalDirection.HOLD, 0.0, "Insufficient data")

        # High bubble score → overextended → SELL
        if score > 0.6:
            confidence = min(0.9, score)
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                f"Elevated bubble risk (score={score:.2f}, H={hurst:.2f})")

        if score > 0.35:
            confidence = min(0.6, score)
            return SignalResult(self.name, SignalDirection.SELL, confidence,
                                f"Moderate bubble risk (score={score:.2f}, H={hurst:.2f})")

        # Low Hurst → mean-reverting → potential contrarian buy
        if hurst < 0.4:
            confidence = min(0.5, (0.5 - hurst) * 2)
            return SignalResult(self.name, SignalDirection.BUY, confidence,
                                f"Mean-reverting regime (H={hurst:.2f}), potential reversal")

        # Normal persistent trend, no bubble
        if hurst > 0.55 and score <= 0.35:
            return SignalResult(self.name, SignalDirection.HOLD, 0.15,
                                f"Persistent trend without bubble signature (H={hurst:.2f})")

        return SignalResult(self.name, SignalDirection.HOLD, 0.0,
                            f"No bubble detected (score={score:.2f}, H={hurst:.2f})")

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": False,
            "columns": ["BubbleScore", "Hurst"],
            "colors": {"BubbleScore": "#F44336", "Hurst": "#2196F3"},
            "subplot_title": "Bubble Risk",
        }
