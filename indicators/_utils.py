"""Shared utilities for cross-asset indicators.

Provides cached reference data fetching and date-index alignment for
indicators that need data from other tickers (e.g., copper, gold, VIX).
"""

import numpy as np
import pandas as pd

from config.settings import CACHE_TTL_SECONDS
from data.series_cache import SeriesCache

_reference_cache = SeriesCache(ttl_seconds=CACHE_TTL_SECONDS)


def fetch_reference_close(ticker: str, period: str = "2y") -> "pd.Series | None":
    """Fetch close prices for a reference ticker with in-process caching.

    Returns None on any failure (network, invalid ticker, etc.).
    """
    def load():
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
            if df is not None and not df.empty and "Close" in df.columns:
                return df["Close"]
        except Exception:
            pass
        return None
    return _reference_cache.get_or_load((ticker, period), load)


def reference_period_for_index(index: pd.DatetimeIndex) -> str:
    """Choose enough reference history to cover a target asset index."""
    if len(index) < 2:
        return "2y"
    span = index[-1] - index[0]
    return "max" if span > pd.Timedelta(days=700) else "2y"


def align_to_index(
    series: "pd.Series | None", target_index: pd.DatetimeIndex
) -> pd.Series:
    """Align a reference series to a target DatetimeIndex with forward-fill.

    Handles timezone mismatches by normalising both sides to tz-naive dates.
    Returns all-NaN series when *series* is None.
    """
    if series is None:
        return pd.Series(np.nan, index=target_index)

    series = series.copy()

    # Strip timezone info for consistent joining
    if hasattr(series.index, "tz") and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    if hasattr(target_index, "tz") and target_index.tz is not None:
        norm_target = target_index.tz_localize(None)
    else:
        norm_target = target_index

    # Normalise to midnight (date-only) for date-based alignment
    series.index = series.index.normalize()
    norm_target = norm_target.normalize()

    aligned = series.reindex(norm_target, method="ffill")
    aligned.index = target_index  # restore original target index
    return aligned


def clear_reference_cache() -> None:
    """Clear the reference data cache (useful for testing)."""
    _reference_cache.clear()
