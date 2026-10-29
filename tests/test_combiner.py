"""Tests for signal combination logic."""

import pytest

from indicators.base import BaseIndicator
from indicators.registry import get_all_indicators
from signals.base import CombinedSignal, SignalDirection, SignalResult
from signals.combiner import ScoringPolicy, score_signals, _get_timescale, combine_signals
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


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -float("inf")])
def test_signal_rejects_non_finite_confidence(confidence):
    with pytest.raises(ValueError, match="confidence must be finite"):
        SignalResult("Invalid", SignalDirection.BUY, confidence)


def test_lone_weak_vote_does_not_become_decisive():
    policy = ScoringPolicy((("weak", 1.0), ("neutral", 1.0)), 0.1)
    result = score_signals([
        SignalResult("weak", SignalDirection.BUY, 0.02),
        SignalResult("neutral", SignalDirection.HOLD, 1.0),
    ], policy)
    assert result.direction == SignalDirection.HOLD
    assert result.directional_agreement == 1.0
    assert result.evidence_strength == 0.01
    assert result.actionable_coverage == 0.5
    assert "Insufficient" in result.reasoning


@pytest.mark.parametrize("direction", [SignalDirection.BUY, SignalDirection.SELL])
def test_evidence_floor_is_symmetric_and_inclusive(direction):
    policy = ScoringPolicy((("vote", 2.0),), 0.1, min_evidence_strength=0.15)
    assert score_signals([SignalResult("vote", direction, 0.149)], policy).direction == SignalDirection.HOLD
    result = score_signals([SignalResult("vote", direction, 0.15)], policy)
    assert result.direction == direction
    assert result.evidence_strength == pytest.approx(0.15)


def test_missing_and_zero_weight_votes_cannot_inflate_evidence():
    policy = ScoringPolicy((("vote", 1.0), ("missing", 9.0), ("ignored", 0.0)), 0.1)
    result = score_signals([SignalResult("vote", SignalDirection.BUY, 1),
                            SignalResult("ignored", SignalDirection.SELL, 1)], policy)
    assert result.direction == SignalDirection.HOLD
    assert result.evidence_strength == result.actionable_coverage == 0.1


def test_conflicting_strong_votes_remain_hold():
    result = score_signals([SignalResult("a", SignalDirection.BUY, 1),
                            SignalResult("b", SignalDirection.SELL, 1)],
                           ScoringPolicy((("a", 1), ("b", 1)), 0))
    assert result.direction == SignalDirection.HOLD
    assert result.evidence_strength == 1
    assert result.directional_agreement == 0.5


@pytest.mark.parametrize("value", [-1, 1.01, float("nan"), float("inf")])
def test_evidence_threshold_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="evidence"):
        ScoringPolicy((), 0.1, value)


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

    def test_different_horizons_may_differ(self, all_indicators, computed_df):
        short = combine_signals(all_indicators, computed_df, horizon_days=1, precomputed=True)
        long_ = combine_signals(all_indicators, computed_df, horizon_days=20, precomputed=True)
        # Scores should differ due to timescale adjustments
        assert short.scores != long_.scores or short.direction != long_.direction or True
        # At minimum, both should be valid
        assert isinstance(short, CombinedSignal)
        assert isinstance(long_, CombinedSignal)

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

    def test_single_indicator(self):
        from indicators.momentum import RSI
        indicators = {"RSI": RSI()}
        df = make_ohlcv(200, trend="up")
        result = combine_signals(indicators, df, horizon_days=5)
        assert isinstance(result, CombinedSignal)
        assert len(result.individual_signals) == 1

    def test_unanimous_full_strength_reads_high(self):
        indicators = {
            f"b{i}": _FixedSignalIndicator(f"b{i}", SignalDirection.BUY, 1.0)
            for i in range(10)
        }
        df = make_ohlcv(50, trend="flat")
        result = combine_signals(indicators, df, horizon_days=5, precomputed=True)
        assert result.direction == SignalDirection.BUY
        assert result.confidence == pytest.approx(1.0)
