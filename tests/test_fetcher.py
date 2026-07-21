"""Tests for data fetcher utility functions.

Note: @st.cache_data functions are mostly tested via pure utility functions
because they depend on Streamlit runtime. The fetch path itself is exercised
only with a monkeypatched yf.Ticker (never the network).
"""

import pandas as pd
import pytest

from data.fetcher import (
    TransientFetchError,
    _call_with_retry,
    compute_buy_and_hold,
    compute_open_to_close_return,
    _validate_ohlcv,
    fetch_ohlcv,
    fetch_ohlcv_batch,
    is_crypto_ticker,
    search_tickers,
    staleness_note,
    slice_date_range,
    trim_to_period,
)


def _ohlcv_frame(periods=5, value=1.0):
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.DataFrame(
        {col: [value] * periods for col in ["Open", "High", "Low", "Close", "Volume"]},
        index=index,
    )


class TestIsCryptoTicker:
    def test_crypto_usd(self):
        assert is_crypto_ticker("BTC-USD") is True
        assert is_crypto_ticker("ETH-USD") is True

    def test_crypto_other_currencies(self):
        assert is_crypto_ticker("BTC-EUR") is True
        assert is_crypto_ticker("ETH-GBP") is True
        assert is_crypto_ticker("SOL-USDT") is True
        assert is_crypto_ticker("AVAX-BUSD") is True

    def test_stock_tickers(self):
        assert is_crypto_ticker("AAPL") is False
        assert is_crypto_ticker("MSFT") is False
        assert is_crypto_ticker("GOOGL") is False

    def test_case_insensitive(self):
        assert is_crypto_ticker("btc-usd") is True
        assert is_crypto_ticker("Eth-Eur") is True

    def test_indices(self):
        assert is_crypto_ticker("^GSPC") is False
        assert is_crypto_ticker("^DJI") is False


class TestComputeBuyAndHold:
    def test_normal_positive(self):
        df = pd.DataFrame({"Close": [100.0, 110.0, 120.0]})
        result = compute_buy_and_hold(df)
        assert result == pytest.approx(0.2)

    def test_normal_negative(self):
        df = pd.DataFrame({"Close": [100.0, 90.0, 80.0]})
        result = compute_buy_and_hold(df)
        assert result == pytest.approx(-0.2)

    def test_insufficient_data(self):
        df = pd.DataFrame({"Close": [100.0]})
        assert compute_buy_and_hold(df) is None

    def test_none_input(self):
        assert compute_buy_and_hold(None) is None

    def test_zero_price(self):
        df = pd.DataFrame({"Close": [0.0, 100.0]})
        assert compute_buy_and_hold(df) is None

    def test_empty_dataframe(self):
        df = pd.DataFrame({"Close": []})
        assert compute_buy_and_hold(df) is None


class TestOpenToCloseReturn:
    def test_uses_first_open_and_last_close(self):
        df = pd.DataFrame({"Open": [90.0, 105.0], "Close": [100.0, 110.0]})
        assert compute_open_to_close_return(df) == pytest.approx(110.0 / 90.0 - 1.0)


class TestCallWithRetry:
    def test_transient_failures_retry_then_succeed(self):
        sleeps = []
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("slow")
            return "ok"

        assert _call_with_retry(flaky, _sleep=sleeps.append) == "ok"
        assert calls["n"] == 3
        assert sleeps == [1.0, 2.0]  # exponential backoff

    def test_exhausted_retries_raise_transient_error(self):
        def always_down():
            raise ConnectionError("refused")

        with pytest.raises(TransientFetchError, match="not an invalid ticker"):
            _call_with_retry(always_down, _sleep=lambda _: None)

    def test_non_transient_error_propagates_immediately(self):
        calls = {"n": 0}

        def broken():
            calls["n"] += 1
            raise ValueError("bad symbol")

        with pytest.raises(ValueError):
            _call_with_retry(broken, _sleep=lambda _: None)
        assert calls["n"] == 1


class TestFetchBatch:
    def test_multiindex_batch_splits_valid_and_failed_tickers(self, monkeypatch):
        good = _ohlcv_frame(periods=5, value=2.0)
        bad = _ohlcv_frame(periods=5, value=1.0) * float("nan")
        raw = pd.concat({"GOODT": good, "BADT": bad}, axis=1)

        monkeypatch.setattr(
            "data.fetcher.yf.download", lambda **kwargs: raw
        )
        frames, errors = fetch_ohlcv_batch(("GOODT", "BADT"), period="6mo")
        assert list(frames) == ["GOODT"]
        assert len(frames["GOODT"]) == 5
        assert "BADT" in errors

    def test_single_ticker_plain_columns(self, monkeypatch):
        monkeypatch.setattr(
            "data.fetcher.yf.download", lambda **kwargs: _ohlcv_frame(periods=4)
        )
        frames, errors = fetch_ohlcv_batch(("ONLYT",), period="1y")
        assert list(frames) == ["ONLYT"]
        assert errors == {}

    def test_rate_limited_batch_raises_transient(self, monkeypatch):
        def throttled(**kwargs):
            raise ConnectionError("429")

        monkeypatch.setattr("data.fetcher.yf.download", throttled)
        monkeypatch.setattr("data.fetcher.time.sleep", lambda _: None)
        with pytest.raises(TransientFetchError):
            fetch_ohlcv_batch(("THROTTLED",), period="3mo")


class TestValidateOhlcvHygiene:
    def test_duplicate_timestamps_deduped_keep_last(self):
        df = _ohlcv_frame(periods=5)
        dup = pd.concat([df, df.iloc[[-1]] * 2.0])
        result = _validate_ohlcv(dup, "DUPT")
        assert not result.index.has_duplicates
        assert result["Close"].iloc[-1] == 2.0

    def test_unsorted_index_is_sorted(self):
        df = _ohlcv_frame(periods=5).iloc[::-1]
        result = _validate_ohlcv(df, "SORTT")
        assert result.index.is_monotonic_increasing

    def test_nonpositive_prices_dropped(self):
        df = _ohlcv_frame(periods=5)
        df.iloc[2, df.columns.get_loc("Close")] = 0.0
        result = _validate_ohlcv(df, "ZEROT")
        assert len(result) == 4
        assert (result["Close"] > 0).all()


class TestStalenessNote:
    def test_fresh_daily_data_has_no_note(self):
        index = pd.date_range(end=pd.Timestamp.now().normalize(), periods=5, freq="D")
        df = pd.DataFrame({"Close": [1.0] * 5}, index=index)
        assert staleness_note(df) is None

    def test_old_last_bar_produces_note(self):
        index = pd.date_range(end=pd.Timestamp.now() - pd.Timedelta(days=30), periods=5, freq="D")
        df = pd.DataFrame({"Close": [1.0] * 5}, index=index)
        note = staleness_note(df)
        assert note is not None and "not a live price" in note

    def test_empty_frame_returns_none(self):
        assert staleness_note(None) is None
        assert staleness_note(pd.DataFrame()) is None


class TestSearchTickers:
    def test_service_failure_returns_none_not_empty(self, monkeypatch):
        def exploding_search(query):
            raise ConnectionError("down")

        monkeypatch.setattr("data.fetcher.yf.Search", exploding_search)
        assert search_tickers("apple down test") is None

    def test_blank_query_returns_empty_list(self):
        assert search_tickers("   ") == []


class TestFetchAutoAdjust:
    def test_history_called_with_explicit_auto_adjust(self, monkeypatch):
        """Price adjustment must be pinned explicitly, never left to
        yfinance's default (which has changed between releases)."""
        captured = {}

        class FakeTicker:
            def __init__(self, symbol):
                captured["symbol"] = symbol

            def history(self, **kwargs):
                captured.update(kwargs)
                index = pd.date_range("2024-01-01", periods=5, freq="D")
                return pd.DataFrame(
                    {col: [1.0] * 5 for col in ["Open", "High", "Low", "Close", "Volume"]},
                    index=index,
                )

        monkeypatch.setattr("data.fetcher.yf.Ticker", FakeTicker)
        df = fetch_ohlcv("FAKE-AUTOADJ-TEST", period="1mo", interval="1d")
        assert captured["auto_adjust"] is True
        assert len(df) == 5


class TestPeriodSlicing:
    def test_trim_to_requested_period(self):
        index = pd.date_range("2023-01-01", periods=800, freq="D")
        df = pd.DataFrame({"Close": range(800)}, index=index)
        result = trim_to_period(df, "1y")
        assert result.index[0] >= df.index[-1] - pd.Timedelta(days=365)
        assert len(result) < len(df)

    def test_slice_date_range_ignores_timezone_difference(self):
        index = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame({"Close": range(10)}, index=index)
        result = slice_date_range(
            df,
            pd.Timestamp("2024-01-03", tz="America/New_York"),
            pd.Timestamp("2024-01-05", tz="America/New_York"),
        )
        assert list(result.index.day) == [3, 4, 5]
