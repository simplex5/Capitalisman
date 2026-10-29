"""FRED economic data indicator: Yield Curve, Jobless Claims, Fed Funds Rate.

Uses Federal Reserve Economic Data (FRED) to gauge macroeconomic health.
Requires a free FRED API key set via the FRED_API_KEY environment variable.
Get one at: https://fred.stlouisfed.org/docs/api/api_key.html

Falls back to HOLD if the API key is not set or fredapi is not installed.

Key signals:
- Yield Curve (10Y-2Y Treasury spread): the most reliable recession predictor.
  An inverted curve has preceded every US recession since the 1960s.
- Initial Jobless Claims: leading indicator of labor market deterioration.
  Rising claims signal economic weakness before it shows in payrolls.
- Fed Funds Rate direction: monetary policy stance.
  Tightening cycles precede slowdowns; easing supports risk assets.
"""

import os
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from indicators._utils import align_to_index
from indicators.base import BaseIndicator
from indicators.registry import register
from signals.base import SignalDirection, SignalResult
from data.series_cache import SeriesCache

# Hourly refresh for successful FRED requests; failures retry after 30 seconds.
_fred_cache = SeriesCache(ttl_seconds=3600)


def _get_fred_api_key() -> str | None:
    """Get FRED API key from environment."""
    return os.environ.get("FRED_API_KEY")


def _fetch_fred_series(series_id: str) -> "pd.Series | None":
    """Fetch a FRED series with in-process caching.

    Returns None on any failure (missing key, network, bad series, etc.).
    """
    api_key = _get_fred_api_key()
    if not api_key:
        return None
    def load():
        try:
            from fredapi import Fred
            data = Fred(api_key=api_key).get_series(series_id)
            if data is not None and len(data) > 0:
                return data
        except Exception:
            pass
        return None
    # Credential changes retry immediately without retaining the key itself.
    return _fred_cache.get_or_load((series_id, hashlib.sha256(api_key.encode()).digest()), load)


def clear_fred_cache() -> None:
    """Clear the FRED data cache (useful for testing)."""
    _fred_cache.clear()


# ---------------------------------------------------------------------------
# FRED Macro Regime
# ---------------------------------------------------------------------------

@register
class FREDMacroRegime(BaseIndicator):
    """FRED-based macroeconomic regime indicator.

    Combines three leading economic signals into a composite macro score:

    1. **Yield Curve (T10Y2Y)** — 10-Year minus 2-Year Treasury spread.
       Inversion (negative spread) has preceded every US recession since 1960.
       Weight: 50% of composite (most predictive sub-signal).

    2. **Initial Jobless Claims (ICSA)** — weekly new unemployment filings.
       A 4-week MA rising above the 13-week MA signals labor market weakening.
       Weight: 30% of composite.

    3. **Federal Funds Rate (DFF)** — daily effective fed funds rate.
       A rising 3-month trend means tightening (bearish for equities);
       a falling trend means easing (bullish).
       Weight: 20% of composite.

    Requires ``FRED_API_KEY`` environment variable (free at fred.stlouisfed.org).
    Falls back to HOLD with zero confidence if the key is not set.
    """

    LOOKBACK = 65  # ~13 weeks of trading days for claims MA

    @property
    def name(self) -> str:
        return "FRED Macro"

    @property
    def category(self) -> str:
        return "macro"

    @property
    def lookback(self) -> int:
        return self.LOOKBACK

    @property
    def backtest_safe(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Fetch FRED series
        yield_curve = _fetch_fred_series("T10Y2Y")   # 10Y-2Y spread (daily)
        claims = _fetch_fred_series("ICSA")           # Initial claims (weekly)
        fed_funds = _fetch_fred_series("DFF")         # Fed funds rate (daily)

        # Align to the asset's date index (forward-fill handles freq gaps)
        yc_aligned = align_to_index(yield_curve, df.index)
        claims_aligned = align_to_index(claims, df.index)
        ff_aligned = align_to_index(fed_funds, df.index)

        df["FRED_YieldCurve"] = yc_aligned
        df["FRED_Claims"] = claims_aligned
        df["FRED_FedFunds"] = ff_aligned

        # Derived: claims moving averages (4-week ~ 20 trading days, 13-week ~ 65)
        df["FRED_Claims_4w"] = claims_aligned.rolling(window=20, min_periods=1).mean()
        df["FRED_Claims_13w"] = claims_aligned.rolling(window=65, min_periods=1).mean()

        # Derived: fed funds 3-month change
        df["FRED_FF_change"] = ff_aligned - ff_aligned.shift(60)

        return df

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def get_signal(self, df: pd.DataFrame, idx: int = -1) -> SignalResult:
        if "FRED_YieldCurve" not in df.columns:
            df = self.compute(df)

        yc = df["FRED_YieldCurve"].iloc[idx]
        claims_4w = df["FRED_Claims_4w"].iloc[idx]
        claims_13w = df["FRED_Claims_13w"].iloc[idx]
        ff_change = df["FRED_FF_change"].iloc[idx]

        # If no FRED data at all, return HOLD
        if pd.isna(yc) and pd.isna(claims_4w) and pd.isna(ff_change):
            return SignalResult(
                self.name, SignalDirection.HOLD, 0.0,
                "FRED data unavailable (set FRED_API_KEY env var)",
            )

        # Score each sub-signal on a -1 (bearish) to +1 (bullish) scale
        sub_scores: list[tuple[float, float, str]] = []  # (score, weight, detail)

        # --- 1. Yield curve (weight 0.50) ---
        if not pd.isna(yc):
            if yc < -0.5:
                sub_scores.append((-0.9, 0.50, f"Yield curve deeply inverted ({yc:+.2f}%)"))
            elif yc < 0:
                sub_scores.append((-0.6, 0.50, f"Yield curve inverted ({yc:+.2f}%)"))
            elif yc < 0.5:
                sub_scores.append((-0.15, 0.50, f"Yield curve flat ({yc:+.2f}%)"))
            elif yc < 1.5:
                sub_scores.append((0.3, 0.50, f"Yield curve normal ({yc:+.2f}%)"))
            else:
                sub_scores.append((0.6, 0.50, f"Yield curve steep ({yc:+.2f}%)"))

        # --- 2. Jobless claims (weight 0.30) ---
        if not pd.isna(claims_4w) and not pd.isna(claims_13w) and claims_13w > 0:
            claims_ratio = claims_4w / claims_13w
            if claims_ratio > 1.10:
                sub_scores.append((-0.7, 0.30, "Claims rising sharply"))
            elif claims_ratio > 1.02:
                sub_scores.append((-0.3, 0.30, "Claims trending higher"))
            elif claims_ratio < 0.95:
                sub_scores.append((0.5, 0.30, "Claims falling"))
            else:
                sub_scores.append((0.1, 0.30, "Claims stable"))

        # --- 3. Fed funds direction (weight 0.20) ---
        if not pd.isna(ff_change):
            if ff_change > 0.50:
                sub_scores.append((-0.6, 0.20, f"Fed tightening ({ff_change:+.2f}%)"))
            elif ff_change > 0.10:
                sub_scores.append((-0.25, 0.20, "Fed mildly tightening"))
            elif ff_change < -0.50:
                sub_scores.append((0.6, 0.20, f"Fed easing ({ff_change:+.2f}%)"))
            elif ff_change < -0.10:
                sub_scores.append((0.3, 0.20, "Fed mildly easing"))
            else:
                sub_scores.append((0.0, 0.20, "Fed on hold"))

        if not sub_scores:
            return SignalResult(
                self.name, SignalDirection.HOLD, 0.0,
                "No FRED sub-signals available",
            )

        # Weighted composite: re-normalise weights to sum to 1
        total_weight = sum(w for _, w, _ in sub_scores)
        composite = sum(s * w for s, w, _ in sub_scores) / total_weight
        detail = "; ".join(d for _, _, d in sub_scores)

        if composite > 0.15:
            confidence = min(0.8, abs(composite))
            return SignalResult(
                self.name, SignalDirection.BUY, confidence,
                f"FRED bullish: {detail}",
            )
        if composite < -0.15:
            confidence = min(0.8, abs(composite))
            return SignalResult(
                self.name, SignalDirection.SELL, confidence,
                f"FRED bearish: {detail}",
            )

        return SignalResult(
            self.name, SignalDirection.HOLD, 0.1,
            f"FRED neutral: {detail}",
        )

    # ------------------------------------------------------------------
    # Chart
    # ------------------------------------------------------------------

    def get_chart_config(self) -> dict[str, Any]:
        return {
            "overlay": False,
            "columns": ["FRED_YieldCurve", "FRED_FedFunds"],
            "colors": {
                "FRED_YieldCurve": "#FF9800",
                "FRED_FedFunds": "#2196F3",
            },
            "subplot_title": "FRED Macro Regime",
        }
