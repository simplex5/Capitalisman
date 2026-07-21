"""yfinance data fetcher with caching and validation."""

from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from config.settings import (
    CACHE_TTL_SECONDS,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    PERIOD_CALENDAR_DAYS,
    WARMUP_FETCH_PERIOD,
)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_raw(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch and validate OHLCV data (cached by Streamlit)."""
    try:
        t = yf.Ticker(ticker)
        # auto_adjust is passed explicitly so price semantics never depend on
        # yfinance's default (which has changed between releases). All prices
        # in the app are split/dividend-adjusted by policy.
        df = t.history(period=period, interval=interval, auto_adjust=True)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for '{ticker}': {e}")

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. Check that the ticker symbol is valid."
        )

    # Keep only OHLCV columns
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[expected_cols].copy()
    df.dropna(inplace=True)

    if len(df) < 2:
        raise ValueError(f"Insufficient data for '{ticker}' (got {len(df)} rows).")

    return df


def fetch_ohlcv(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Fetch OHLCV data for a ticker with caching.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Index is DatetimeIndex. Prices are split/dividend-adjusted (auto_adjust=True).

    Raises ValueError if ticker is invalid or no data returned.
    """
    return _fetch_raw(ticker, period, interval).copy()


def fetch_with_warmup(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch OHLCV data with extra warmup bars for indicator computation.

    For short display periods, fetches a longer history so indicators have
    enough data, then trims back to the user's requested display range.

    Returns (full_df, display_df) where:
      - full_df: all fetched data (use for indicator computation)
      - display_df: trimmed to the user's requested period (use for charting)
    """
    fetch_period = WARMUP_FETCH_PERIOD.get(period, period)
    full_df = fetch_ohlcv(ticker, period=fetch_period, interval=interval)

    calendar_days = PERIOD_CALENDAR_DAYS.get(period)
    if calendar_days is not None and len(full_df) > 0:
        cutoff = full_df.index[-1] - pd.Timedelta(days=calendar_days)
        display_df = full_df.loc[full_df.index > cutoff]
        # Ensure we return at least something
        if display_df.empty:
            display_df = full_df.iloc[-1:]
    else:
        display_df = full_df

    return full_df, display_df


def trim_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Trim warmup data to the requested user-facing period."""
    calendar_days = PERIOD_CALENDAR_DAYS.get(period)
    if calendar_days is None or df.empty:
        return df.copy()
    cutoff = df.index[-1] - pd.Timedelta(days=calendar_days)
    trimmed = df.loc[df.index >= cutoff]
    return trimmed.copy()


def slice_date_range(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Slice by calendar date while tolerating timezone differences."""
    if df.empty:
        return df.copy()
    index_dates = pd.Index(df.index.date)
    mask = (index_dates >= start.date()) & (index_dates <= end.date())
    return df.loc[mask].copy()


def is_crypto_ticker(ticker: str) -> bool:
    """Heuristic check for crypto tickers (e.g. BTC-USD, ETH-EUR)."""
    crypto_suffixes = ("-USD", "-USDT", "-EUR", "-GBP", "-BTC", "-ETH", "-BUSD")
    upper = ticker.upper()
    return any(upper.endswith(s) for s in crypto_suffixes)


def compute_buy_and_hold(df: pd.DataFrame) -> float | None:
    """Compute buy-and-hold return from the first to last close in a DataFrame.

    Returns the fractional return (e.g. 0.15 for +15%), or None if insufficient data.
    """
    if df is None or len(df) < 2:
        return None
    first_close = df["Close"].iloc[0]
    last_close = df["Close"].iloc[-1]
    if first_close == 0:
        return None
    return (last_close - first_close) / first_close


def compute_open_to_close_return(df: pd.DataFrame) -> float | None:
    """Compute return from the first bar's open to the last bar's close."""
    if df is None or len(df) < 2:
        return None
    entry_price = df["Open"].iloc[0]
    exit_price = df["Close"].iloc[-1]
    if entry_price == 0:
        return None
    return (exit_price - entry_price) / entry_price


@st.cache_data(ttl=60, show_spinner=False)
def search_tickers(query: str) -> list[dict]:
    """Search for tickers using Yahoo Finance search API.

    Returns list of dicts with keys: symbol, name, exchange, type.
    """
    if not query or not query.strip():
        return []
    try:
        results = yf.Search(query)
        quotes = results.quotes if hasattr(results, "quotes") else []
        return [
            {
                "symbol": q.get("symbol", ""),
                "name": q.get("shortname") or q.get("longname", ""),
                "exchange": q.get("exchange", ""),
                "type": q.get("quoteType", ""),
            }
            for q in quotes
            if q.get("symbol")
        ]
    except Exception:
        return []


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_asset_info(ticker: str) -> Optional[dict]:
    """Fetch basic asset info (name, sector, market cap, etc.)."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return None
