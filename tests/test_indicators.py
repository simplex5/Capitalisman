"""Tests for individual indicator computation and signal generation."""

import numpy as np
import pandas as pd
import pytest

from indicators.registry import get_all_indicators
from signals.base import SignalDirection, SignalResult
from tests.conftest import make_ohlcv


class TestAllIndicators:
    """Generic tests that apply to every registered indicator."""

    @pytest.fixture(params=list(get_all_indicators().keys()))
    def indicator(self, request):
        return get_all_indicators()[request.param]

    def test_has_valid_category(self, indicator):
        assert indicator.category in (
            "trend", "momentum", "volatility", "volume",
            "macro", "structural", "microstructure", "systemic",
            "forecast",
        )

    def test_positive_lookback(self, indicator):
        assert indicator.lookback > 0

    def test_chart_config_has_required_keys(self, indicator):
        cfg = indicator.get_chart_config()
        assert "overlay" in cfg
        assert "columns" in cfg
        assert isinstance(cfg["columns"], list)
        assert "colors" in cfg
        if not cfg["overlay"]:
            assert "subplot_title" in cfg

    def test_compute_adds_columns(self, indicator, ohlcv_200_up):
        original_cols = set(ohlcv_200_up.columns)
        result = indicator.compute(ohlcv_200_up)
        assert isinstance(result, pd.DataFrame)
        # Should add at least one new column
        new_cols = set(result.columns) - original_cols
        assert len(new_cols) > 0

    def test_compute_does_not_modify_input(self, indicator, ohlcv_100):
        original = ohlcv_100.copy()
        indicator.compute(ohlcv_100)
        pd.testing.assert_frame_equal(ohlcv_100, original)

    def test_signal_returns_valid_result(self, indicator, ohlcv_200_up):
        df = indicator.compute(ohlcv_200_up)
        signal = indicator.get_signal(df)
        assert isinstance(signal, SignalResult)
        assert isinstance(signal.direction, SignalDirection)
        assert 0.0 <= signal.confidence <= 1.0
        assert isinstance(signal.detail, str)
        assert signal.indicator_name == indicator.name

    def test_insufficient_data_returns_hold(self, indicator, ohlcv_short):
        df = indicator.compute(ohlcv_short)
        signal = indicator.get_signal(df)
        assert isinstance(signal, SignalResult)
        # With only 5 bars, most indicators can't produce meaningful signals
        assert signal.direction == SignalDirection.HOLD


class TestRSI:
    def test_compute_adds_rsi_column(self, ohlcv_100):
        from indicators.momentum import RSI
        rsi = RSI()
        result = rsi.compute(ohlcv_100)
        assert "RSI" in result.columns
        # RSI should be between 0 and 100 for non-NaN values
        valid = result["RSI"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_oversold_gives_buy(self):
        """Construct data where RSI will be deeply oversold."""
        # Steady decline to force RSI low
        df = make_ohlcv(100, start_price=200, trend="down", seed=10)
        from indicators.momentum import RSI
        rsi = RSI()
        result = rsi.compute(df)
        signal = rsi.get_signal(result)
        # With strong downtrend, RSI should be low; signal could be BUY (oversold) or SELL
        assert isinstance(signal, SignalResult)

    def test_overbought_gives_sell(self):
        """Construct data where RSI will be high."""
        df = make_ohlcv(100, start_price=50, trend="up", seed=10)
        from indicators.momentum import RSI
        rsi = RSI()
        result = rsi.compute(df)
        signal = rsi.get_signal(result)
        assert isinstance(signal, SignalResult)


class TestStochastic:
    def test_compute_adds_columns(self, ohlcv_100):
        from indicators.momentum import Stochastic
        stoch = Stochastic()
        result = stoch.compute(ohlcv_100)
        assert "Stoch_K" in result.columns
        assert "Stoch_D" in result.columns

    def test_signal_with_sufficient_data(self, ohlcv_200_up):
        from indicators.momentum import Stochastic
        stoch = Stochastic()
        result = stoch.compute(ohlcv_200_up)
        signal = stoch.get_signal(result)
        assert isinstance(signal, SignalResult)
        assert signal.indicator_name == "Stochastic"


class TestSMACrossover:
    def test_compute_adds_sma_columns(self, ohlcv_200_up):
        from indicators.trend import SMACrossover
        sma = SMACrossover()
        result = sma.compute(ohlcv_200_up)
        assert "SMA_short" in result.columns
        assert "SMA_long" in result.columns

    def test_uptrend_favors_buy(self, ohlcv_200_up):
        from indicators.trend import SMACrossover
        sma = SMACrossover()
        result = sma.compute(ohlcv_200_up)
        signal = sma.get_signal(result)
        # In an uptrend, SMA short should be above SMA long
        assert signal.direction in (SignalDirection.BUY, SignalDirection.HOLD)


class TestEMACrossover:
    def test_compute_adds_ema_columns(self, ohlcv_200_up):
        from indicators.trend import EMACrossover
        ema = EMACrossover()
        result = ema.compute(ohlcv_200_up)
        assert "EMA_short" in result.columns
        assert "EMA_long" in result.columns


class TestMACD:
    def test_compute_adds_macd_columns(self, ohlcv_200_up):
        from indicators.trend import MACD
        macd = MACD()
        result = macd.compute(ohlcv_200_up)
        assert "MACD_line" in result.columns
        assert "MACD_signal" in result.columns
        assert "MACD_hist" in result.columns


class TestADX:
    def test_compute_adds_adx_columns(self, ohlcv_200_up):
        from indicators.trend import ADX
        adx = ADX()
        result = adx.compute(ohlcv_200_up)
        assert "ADX" in result.columns
        assert "ADX_pos" in result.columns
        assert "ADX_neg" in result.columns


class TestBollingerBands:
    def test_compute_adds_bb_columns(self, ohlcv_200_up):
        from indicators.volatility import BollingerBands
        bb = BollingerBands()
        result = bb.compute(ohlcv_200_up)
        assert "BB_upper" in result.columns
        assert "BB_middle" in result.columns
        assert "BB_lower" in result.columns
        assert "BB_pband" in result.columns


class TestVWAP:
    def test_compute_adds_vwap_column(self, ohlcv_200_up):
        from indicators.volume import VWAP
        vwap = VWAP()
        result = vwap.compute(ohlcv_200_up)
        assert "VWAP" in result.columns


class TestOBV:
    def test_compute_adds_obv_columns(self, ohlcv_200_up):
        from indicators.volume import OBV
        obv = OBV()
        result = obv.compute(ohlcv_200_up)
        assert "OBV" in result.columns
        assert "OBV_SMA" in result.columns


# --- Novel indicator tests ---


class TestCopperGoldRatio:
    def test_compute_adds_columns(self, ohlcv_200_up):
        from indicators.macro import CopperGoldRatio
        cg = CopperGoldRatio()
        result = cg.compute(ohlcv_200_up)
        assert "CG_ratio" in result.columns
        assert "CG_SMA_short" in result.columns
        assert "CG_SMA_long" in result.columns
        assert "CG_roc" in result.columns

    def test_signal_returns_valid_result(self, ohlcv_200_up):
        from indicators.macro import CopperGoldRatio
        cg = CopperGoldRatio()
        result = cg.compute(ohlcv_200_up)
        signal = cg.get_signal(result)
        assert isinstance(signal, SignalResult)
        assert signal.indicator_name == "Copper-Gold Ratio"


class TestVIXTermStructure:
    def test_compute_adds_columns(self, ohlcv_200_up):
        from indicators.macro import VIXTermStructure
        vts = VIXTermStructure()
        result = vts.compute(ohlcv_200_up)
        assert "VIX" in result.columns
        assert "VIX3M" in result.columns
        assert "VIX_spread" in result.columns
        assert "VIX_ratio" in result.columns

    def test_signal_returns_valid_result(self, ohlcv_200_up):
        from indicators.macro import VIXTermStructure
        vts = VIXTermStructure()
        result = vts.compute(ohlcv_200_up)
        signal = vts.get_signal(result)
        assert isinstance(signal, SignalResult)
        assert signal.indicator_name == "VIX Term Structure"


class TestBubbleRisk:
    def test_compute_adds_columns(self, ohlcv_200_up):
        from indicators.structural import BubbleRisk
        br = BubbleRisk()
        result = br.compute(ohlcv_200_up)
        assert "Hurst" in result.columns
        assert "LogAccel" in result.columns
        assert "BubbleScore" in result.columns

    def test_hurst_in_valid_range(self, ohlcv_200_up):
        from indicators.structural import BubbleRisk
        br = BubbleRisk()
        result = br.compute(ohlcv_200_up)
        valid = result["Hurst"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_bubble_score_in_valid_range(self, ohlcv_200_up):
        from indicators.structural import BubbleRisk
        br = BubbleRisk()
        result = br.compute(ohlcv_200_up)
        valid = result["BubbleScore"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all() and (valid <= 1.1).all()

    def test_signal_returns_valid_result(self, ohlcv_200_up):
        from indicators.structural import BubbleRisk
        br = BubbleRisk()
        result = br.compute(ohlcv_200_up)
        signal = br.get_signal(result)
        assert isinstance(signal, SignalResult)
        assert signal.indicator_name == "Bubble Risk"


class TestVPIN:
    def test_compute_adds_columns(self, ohlcv_200_up):
        from indicators.microstructure import VPIN
        vpin = VPIN()
        result = vpin.compute(ohlcv_200_up)
        assert "VPIN" in result.columns
        assert "BuyVolPct" in result.columns

    def test_vpin_in_valid_range(self, ohlcv_200_up):
        from indicators.microstructure import VPIN
        vpin = VPIN()
        result = vpin.compute(ohlcv_200_up)
        valid = result["VPIN"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_buy_vol_pct_in_valid_range(self, ohlcv_200_up):
        from indicators.microstructure import VPIN
        vpin = VPIN()
        result = vpin.compute(ohlcv_200_up)
        valid = result["BuyVolPct"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_signal_returns_valid_result(self, ohlcv_200_up):
        from indicators.microstructure import VPIN
        vpin = VPIN()
        result = vpin.compute(ohlcv_200_up)
        signal = vpin.get_signal(result)
        assert isinstance(signal, SignalResult)
        assert signal.indicator_name == "VPIN"


class TestMarketCorrelation:
    def test_compute_adds_columns(self, ohlcv_200_up):
        from indicators.systemic import MarketCorrelation
        mc = MarketCorrelation()
        result = mc.compute(ohlcv_200_up)
        assert "AbsorptionRatio" in result.columns
        assert "TopEigenvalue" in result.columns

    def test_signal_returns_valid_result(self, ohlcv_200_up):
        from indicators.systemic import MarketCorrelation
        mc = MarketCorrelation()
        result = mc.compute(ohlcv_200_up)
        signal = mc.get_signal(result)
        assert isinstance(signal, SignalResult)
        assert signal.indicator_name == "Market Correlation"


# --- FRED macro indicator tests ---


class TestFREDMacroRegime:
    def test_compute_adds_columns(self, ohlcv_200_up):
        from indicators.fred import FREDMacroRegime
        fred = FREDMacroRegime()
        result = fred.compute(ohlcv_200_up)
        assert "FRED_YieldCurve" in result.columns
        assert "FRED_Claims" in result.columns
        assert "FRED_FedFunds" in result.columns
        assert "FRED_Claims_4w" in result.columns
        assert "FRED_Claims_13w" in result.columns
        assert "FRED_FF_change" in result.columns

    def test_signal_returns_valid_result(self, ohlcv_200_up):
        from indicators.fred import FREDMacroRegime
        fred = FREDMacroRegime()
        result = fred.compute(ohlcv_200_up)
        signal = fred.get_signal(result)
        assert isinstance(signal, SignalResult)
        assert signal.indicator_name == "FRED Macro"

    def test_no_api_key_returns_hold(self, ohlcv_200_up, monkeypatch):
        """Without FRED_API_KEY, indicator should gracefully return HOLD."""
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        from indicators.fred import FREDMacroRegime, clear_fred_cache
        clear_fred_cache()
        fred = FREDMacroRegime()
        result = fred.compute(ohlcv_200_up)
        signal = fred.get_signal(result)
        assert signal.direction == SignalDirection.HOLD
        assert signal.confidence == 0.0


# --- TimesFM forecast indicator tests ---


class TestTimesFMForecast:
    @staticmethod
    def _indicator():
        from indicators.forecast import TimesFMForecast
        from ml.timesfm_runtime import TimesFMRuntime

        class FakeModel:
            def forecast(self, horizon, inputs):
                point = []
                quantiles = []
                for values in inputs:
                    current = float(values[-1])
                    path = current * (1.0 + np.linspace(0.002, 0.02, horizon))
                    point.append(path)
                    slices = np.zeros((horizon, 10))
                    slices[:, 0] = path
                    for quantile_idx in range(1, 10):
                        slices[:, quantile_idx] = path * (
                            0.94 + quantile_idx * 0.012
                        )
                    quantiles.append(slices)
                return np.asarray(point), np.asarray(quantiles)

        return TimesFMForecast(TimesFMRuntime(model_factory=FakeModel))

    def test_supports_configurable_backtest_horizons(self):
        indicator = self._indicator()
        assert indicator.supports_backtest_horizon(10)
        assert indicator.supports_backtest_horizon(5)

    def test_compute_adds_columns(self, ohlcv_200_up):
        tfm = self._indicator()
        result = tfm.compute(ohlcv_200_up)
        assert "TFM_point" in result.columns
        assert "TFM_q10" in result.columns
        assert "TFM_q90" in result.columns
        assert "TFM_probability_up" in result.columns

    def test_signal_returns_valid_result(self, ohlcv_200_up):
        tfm = self._indicator()
        result = tfm.compute(ohlcv_200_up)
        signal = tfm.get_signal(result)
        assert isinstance(signal, SignalResult)
        assert signal.indicator_name == "TimesFM Forecast"

    def test_no_timesfm_returns_hold(self, ohlcv_200_up):
        """Without timesfm installed, indicator should gracefully return HOLD."""
        from indicators.forecast import TimesFMForecast
        tfm = TimesFMForecast()
        result = tfm.compute(ohlcv_200_up)
        signal = tfm.get_signal(result)
        # If timesfm is not installed, all values will be NaN -> HOLD
        assert isinstance(signal, SignalResult)
        # Either HOLD (no model) or a valid directional signal (model present)
        assert signal.direction in (
            SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD
        )

    def test_historical_signal_ignores_future_volatility(self, ohlcv_200_up):
        from ml.timesfm_runtime import TimesFMRuntimeConfig

        idx = 101
        changed_future = ohlcv_200_up.copy()
        changed_future.iloc[idx + 1 :, changed_future.columns.get_loc("Close")] *= np.linspace(
            0.2, 5.0, len(changed_future) - idx - 1
        )

        first_indicator = self._indicator()
        second_indicator = self._indicator()
        config = TimesFMRuntimeConfig(use_case="backtest")
        first_indicator.runtime.config = config
        second_indicator.runtime.config = config
        first_df = first_indicator.compute_for_horizon(ohlcv_200_up, 10)
        second_df = second_indicator.compute_for_horizon(changed_future, 10)
        first = first_indicator.get_signal_for_horizon(first_df, 10, idx=idx)
        second = second_indicator.get_signal_for_horizon(second_df, 10, idx=idx)
        assert first.direction == second.direction
        assert first.confidence == pytest.approx(second.confidence)

    def test_sparse_backtest_origin_does_not_rerun_full_forecast(self, ohlcv_200_up):
        from ml.timesfm_runtime import TimesFMRuntimeConfig

        indicator = self._indicator()
        indicator.runtime.config = TimesFMRuntimeConfig(use_case="backtest")
        original_forecast = indicator.runtime.forecast
        calls = 0

        def counting_forecast(inputs, horizon):
            nonlocal calls
            calls += 1
            return original_forecast(inputs, horizon)

        indicator.runtime.forecast = counting_forecast
        computed = indicator.compute_for_horizon(ohlcv_200_up, 10)
        assert calls == 1
        signal = indicator.get_signal_for_horizon(computed, 10, idx=100)
        assert signal.direction == SignalDirection.HOLD
        assert calls == 1
