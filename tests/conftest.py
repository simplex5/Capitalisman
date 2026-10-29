"""Shared test fixtures for Capitalisman test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def offline_integrations(monkeypatch):
    """Unit/UI tests never download market data or real model weights."""
    import indicators.macro as macro
    import indicators.systemic as systemic
    import indicators.fred as fred
    from ml.timesfm_runtime import TimesFMRuntime
    for module in (macro, systemic):
        monkeypatch.setattr(module, "fetch_reference_close", lambda *args, **kwargs: None)
    monkeypatch.setattr(fred, "_fetch_fred_series", lambda *args, **kwargs: None)
    original = TimesFMRuntime.load
    def load(self):
        if self._model_factory is None:
            raise RuntimeError("Real model loading is disabled in offline tests")
        return original(self)
    monkeypatch.setattr(TimesFMRuntime, "load", load)


def make_ohlcv(
    n_bars: int = 100,
    start_price: float = 100.0,
    trend: str = "flat",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Args:
        n_bars: Number of bars to generate.
        start_price: Starting close price.
        trend: One of "flat", "up", "down", "volatile".
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with Open, High, Low, Close, Volume columns and DatetimeIndex.
    """
    rng = np.random.RandomState(seed)

    if trend == "flat":
        drift = 0.0
        vol = 0.005
    elif trend == "up":
        drift = 0.003
        vol = 0.008
    elif trend == "down":
        drift = -0.003
        vol = 0.008
    elif trend == "volatile":
        drift = 0.0
        vol = 0.03
    else:
        raise ValueError(f"Unknown trend: {trend}")

    returns = drift + vol * rng.randn(n_bars)
    prices = start_price * np.cumprod(1 + returns)

    # Build OHLCV
    close = prices
    open_ = np.roll(close, 1)
    open_[0] = start_price
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n_bars))
    volume = rng.randint(100_000, 10_000_000, size=n_bars).astype(float)

    dates = pd.bdate_range(start="2024-01-02", periods=n_bars, freq="B")

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def ohlcv_100():
    """100-bar flat OHLCV data."""
    return make_ohlcv(100, trend="flat")


@pytest.fixture
def ohlcv_200_up():
    """200-bar uptrend OHLCV data."""
    return make_ohlcv(200, trend="up")


@pytest.fixture
def ohlcv_200_down():
    """200-bar downtrend OHLCV data."""
    return make_ohlcv(200, trend="down")


@pytest.fixture
def ohlcv_short():
    """5-bar OHLCV data (insufficient for most indicators)."""
    return make_ohlcv(5, trend="flat")


@pytest.fixture
def ohlcv_empty():
    """Empty OHLCV DataFrame."""
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
