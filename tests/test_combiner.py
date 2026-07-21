"""Tests for signal combination logic."""

import pytest

from indicators.base import BaseIndicator
from indicators.registry import get_all_indicators
from signals.base import CombinedSignal, SignalDirection, SignalResult
from signals.combiner import _get_timescale, combine_signals
from tests.conftest import make_ohlcv


class _FixedSignalIndicator(BaseIndicator):
    """Test double that always emits one fixed signal."""

    def __init__(self, name, direction, strength):
        self._name = name
        self._direction = direction
        self._strength = strength

    @property
    def name(self):
        return self._name

    @property
    def category(self):
        return "trend"

    @property
    def lookback(self):
        return 1

    def compute(self, df):
        return df

    def get_signal(self, df, idx=-1):
        return SignalResult(self._name, self._direction, self._strength, "fixed")

    def get_chart_config(self):
        return {}


class TestGetTimescale:
    def test_short(self):
        assert _get_timescale(1) == "short"
        assert _get_timescale(3) == "short"

    def test_medium(self):
        assert _get_timescale(4) == "medium"
        assert _get_timescale(10) == "medium"

    def test_long(self):
        assert _get_timescale(11) == "long"
        assert _get_timescale(30) == "long"


class TestCombineSignals:
    @pytest.fixture
    def all_indicators(self):
        return get_all_indicators()

    @pytest.fixture
    def computed_df(self, all_indicators):
        df = make_ohlcv(200, trend="up")
        for indicator in all_indicators.values():
            df = indicator.compute(df)
        return df

    def test_returns_combined_signal(self, all_indicators, computed_df):
        result = combine_signals(all_indicators, computed_df, horizon_days=5, precomputed=True)
        assert isinstance(result, CombinedSignal)
        assert isinstance(result.direction, SignalDirection)
        assert 0.0 <= result.confidence <= 1.0

    def test_individual_signals_populated(self, all_indicators, computed_df):
        result = combine_signals(all_indicators, computed_df, horizon_days=5, precomputed=True)
        assert len(result.individual_signals) == len(all_indicators)
        for sig in result.individual_signals:
            assert sig.indicator_name in all_indicators

    def test_reasoning_non_empty(self, all_indicators, computed_df):
        result = combine_signals(all_indicators, computed_df, horizon_days=5, precomputed=True)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_scores_populated(self, all_indicators, computed_df):
        result = combine_signals(all_indicators, computed_df, horizon_days=5, precomputed=True)
        assert "BUY" in result.scores
        assert "SELL" in result.scores
        assert "HOLD" in result.scores

    def test_precomputed_flag(self, all_indicators, computed_df):
        # With precomputed=True, should not recompute
        result = combine_signals(all_indicators, computed_df, horizon_days=5, precomputed=True)
        assert isinstance(result, CombinedSignal)

    def test_different_horizons_apply_different_timescale_weights(self, all_indicators, computed_df):
        short = combine_signals(all_indicators, computed_df, horizon_days=1, precomputed=True)
        long_ = combine_signals(all_indicators, computed_df, horizon_days=20, precomputed=True)
        # Timescale adjustments weight categories differently per horizon, so
        # the weighted scores must actually differ on trending data.
        assert short.scores != long_.scores
        assert isinstance(short, CombinedSignal)
        assert isinstance(long_, CombinedSignal)

    def test_idx_parameter(self, all_indicators, computed_df):
        result = combine_signals(all_indicators, computed_df, horizon_days=5, idx=100, precomputed=True)
        assert isinstance(result, CombinedSignal)

    def test_empty_indicators_returns_hold(self):
        df = make_ohlcv(50, trend="flat")
        result = combine_signals({}, df, horizon_days=5)
        assert result.direction == SignalDirection.HOLD
        assert result.confidence == 0.0

    def test_confidence_penalizes_broad_abstention(self):
        """One weak voter among many abstainers must not read as ~100%."""
        indicators = {"buyer": _FixedSignalIndicator("buyer", SignalDirection.BUY, 0.3)}
        indicators.update({
            f"hold{i}": _FixedSignalIndicator(f"hold{i}", SignalDirection.HOLD, 0.0)
            for i in range(9)
        })
        df = make_ohlcv(50, trend="flat")
        result = combine_signals(indicators, df, horizon_days=5, precomputed=True)
        assert result.direction == SignalDirection.BUY
        # 0.3 conviction out of 10 equally-weighted possible votes
        assert result.confidence == pytest.approx(0.03)

    def test_unanimous_full_strength_reads_high(self):
        indicators = {
            f"b{i}": _FixedSignalIndicator(f"b{i}", SignalDirection.BUY, 1.0)
            for i in range(10)
        }
        df = make_ohlcv(50, trend="flat")
        result = combine_signals(indicators, df, horizon_days=5, precomputed=True)
        assert result.direction == SignalDirection.BUY
        assert result.confidence == pytest.approx(1.0)

    def test_single_indicator(self):
        from indicators.momentum import RSI
        indicators = {"RSI": RSI()}
        df = make_ohlcv(200, trend="up")
        result = combine_signals(indicators, df, horizon_days=5)
        assert isinstance(result, CombinedSignal)
        assert len(result.individual_signals) == 1
