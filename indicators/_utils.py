"""Shared utilities for cross-asset indicators.

Provides cached reference data fetching and date-index alignment for
indicators that need data from other tickers (e.g., copper, gold, VIX).
"""

import numpy as np
import pandas as pd

# Module-level cache survives across calls within the same process.
# For Streamlit apps this means the reference data is fetched once per
# server lifetime, which is fine for slow-moving macro data.
_reference_cache: dict[str, "pd.Series | None"] = {}


def fetch_reference_close(ticker: str, period: str = "2y") -> "pd.Series | None":
    """Fetch close prices for a reference ticker with in-process caching.

    References are fetched unadjusted (auto_adjust=False) so historical values
    are point-in-time stable: dividend adjustment bakes *future* dividends into
    past bars, which is look-ahead in backtests. The remaining split scaling is
    a constant factor that cancels in every downstream use (ratios against own
    SMAs, pct_change, correlations).

    Returns None on any failure (network, invalid ticker, etc.).
    """
    cache_key = f"{ticker}_{period}"
    if cache_key in _reference_cache:
        return _reference_cache[cache_key]
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df is not None and not df.empty and "Close" in df.columns:
            result = df["Close"]
            _reference_cache[cache_key] = result
            return result
    except Exception:
        pass
    _reference_cache[cache_key] = None
    return None


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

    Known limitation: for 24/7 target calendars (crypto), equity references
    forward-fill across weekends/holidays, so their pct_change is 0 on those
    days — correlation-based indicators are damped for crypto targets.
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
