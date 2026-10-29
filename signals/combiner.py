"""Weighted majority voting signal combiner."""

import pandas as pd
import math
from dataclasses import dataclass

from config.overrides import get_setting
from config.settings import MIN_EVIDENCE_STRENGTH, TIMESCALE_ADJUSTMENTS
from indicators.base import BaseIndicator
from signals.base import CombinedSignal, SignalDirection, SignalResult


@dataclass(frozen=True)
class ScoringPolicy:
    """Effective horizon-adjusted weights and ambiguity rule for one run."""

    weights: tuple[tuple[str, float], ...]
    ambiguity_threshold: float
    min_evidence_strength: float = MIN_EVIDENCE_STRENGTH

    def __post_init__(self):
        object.__setattr__(self, "weights", tuple((name, value) for name, value in self.weights))
        if len(dict(self.weights)) != len(self.weights):
            raise ValueError("Scoring weights must be unique")
        if any(not math.isfinite(value) or value < 0 for _, value in self.weights):
            raise ValueError("Scoring weights must be finite and non-negative")
        if not math.isfinite(self.ambiguity_threshold) or not 0 <= self.ambiguity_threshold < 1:
            raise ValueError("Ambiguity threshold must be in [0, 1)")
        if not math.isfinite(self.min_evidence_strength) or not 0 <= self.min_evidence_strength <= 1:
            raise ValueError("Minimum evidence strength must be in [0, 1]")

    @classmethod
    def from_indicators(cls, indicators, horizon_days):
        base = get_setting("INDICATOR_WEIGHTS")
        adjustments = TIMESCALE_ADJUSTMENTS[_get_timescale(horizon_days)]
        return cls(tuple(
            (name, base.get(name, 1.0) * adjustments.get(indicator.category, 1.0))
            for name, indicator in indicators.items()
        ), get_setting("AMBIGUITY_THRESHOLD"), get_setting("MIN_EVIDENCE_STRENGTH"))


def _get_timescale(horizon_days: int) -> str:
    if horizon_days <= 3:
        return "short"
    elif horizon_days <= 10:
        return "medium"
    return "long"


def _build_reasoning(
    direction: SignalDirection,
    confidence: float,
    individual_signals: list[SignalResult],
    buy_score: float,
    sell_score: float,
) -> str:
    """Build a plain-English explanation of the combined signal."""
    total = len(individual_signals)
    if total == 0:
        return "No indicators evaluated."

    if direction == SignalDirection.HOLD and confidence == 0.0 and buy_score == 0 and sell_score == 0:
        return "No actionable signals from any indicator."

    buy_signals = [s for s in individual_signals if s.direction == SignalDirection.BUY]
    sell_signals = [s for s in individual_signals if s.direction == SignalDirection.SELL]
    hold_signals = [s for s in individual_signals if s.direction == SignalDirection.HOLD]

    if direction == SignalDirection.HOLD:
        parts = [f"Indicators are split: {len(buy_signals)} bullish vs {len(sell_signals)} bearish"]
        if hold_signals:
            parts[0] += f" ({len(hold_signals)} neutral)"
        parts.append(f"Weighted scores are too close — BUY {buy_score:.2f} vs SELL {sell_score:.2f}.")
        return " ".join(parts)

    # BUY or SELL — describe what's driving the signal
    if direction == SignalDirection.BUY:
        drivers = buy_signals
        opposing = sell_signals
        label, opp_label = "bullish", "bearish"
    else:
        drivers = sell_signals
        opposing = buy_signals
        label, opp_label = "bearish", "bullish"

    parts = [f"{len(drivers)} of {total} indicators are {label}."]

    # Highlight the strongest drivers (confidence >= 0.5)
    strong = sorted(drivers, key=lambda s: s.confidence, reverse=True)
    highlights = [s.detail for s in strong[:3] if s.confidence >= 0.4]
    if highlights:
        parts.append(" ".join(f"{h}." if not h.endswith(".") else h for h in highlights))

    if opposing:
        opp_names = ", ".join(s.indicator_name for s in opposing)
        parts.append(f"{opp_names} disagree{'s' if len(opposing) == 1 else ''} ({opp_label}).")

    return " ".join(parts)


def combine_signals(
    indicators: dict[str, BaseIndicator],
    df: pd.DataFrame,
    horizon_days: int = 5,
    idx: int = -1,
    precomputed: bool = False,
    policy: ScoringPolicy | None = None,
) -> CombinedSignal:
    """Combine signals from multiple indicators using weighted voting.

    Args:
        indicators: dict of name -> BaseIndicator instance
        df: DataFrame with OHLCV data (indicator columns will be computed as needed)
        horizon_days: prediction horizon for timescale adjustment
        idx: bar index to evaluate
        precomputed: if True, skip indicator computation (columns already in df)

    Returns:
        CombinedSignal with combined direction, confidence, and breakdown.
    """
    individual_signals: list[SignalResult] = []
    policy = policy or ScoringPolicy.from_indicators(indicators, horizon_days)

    working_df = df if precomputed else df.copy()

    for name, indicator in indicators.items():
        if not precomputed:
            working_df = indicator.compute_for_horizon(working_df, horizon_days)
        signal = indicator.get_signal_for_horizon(
            working_df, horizon_days=horizon_days, idx=idx
        )
        individual_signals.append(signal)

    return score_signals(individual_signals, policy)


def score_signals(individual_signals: list[SignalResult], policy: ScoringPolicy) -> CombinedSignal:
    """Pure scorer shared by prediction, historical replay, and backtesting."""
    direction_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    weights = dict(policy.weights)
    names = [signal.indicator_name for signal in individual_signals]
    if len(set(names)) != len(names) or set(names) - weights.keys():
        raise ValueError("Signals must have unique names present in the scoring policy")
    for signal in individual_signals:
        if not math.isfinite(signal.confidence) or not 0 <= signal.confidence <= 1:
            raise ValueError("Signal confidence must be finite and in [0, 1]")
        score = signal.confidence * weights[signal.indicator_name]
        # HOLD signals are recorded for display but don't participate in
        # directional voting — they represent absence of a signal, not a
        # competing direction.
        direction_scores[signal.direction.value] += score

    # Only BUY and SELL compete; HOLD is the fallback when neither wins
    buy_score = direction_scores["BUY"]
    sell_score = direction_scores["SELL"]
    directional_total = buy_score + sell_score
    total_weight = sum(weights.values())
    if not all(math.isfinite(value) for value in (*direction_scores.values(), directional_total, total_weight)):
        raise ValueError("Combined scores must be finite")

    if directional_total == 0:
        return CombinedSignal(
            direction=SignalDirection.HOLD,
            confidence=0.0,
            scores=direction_scores,
            individual_signals=individual_signals,
            reasoning=_build_reasoning(
                SignalDirection.HOLD, 0.0, individual_signals, buy_score, sell_score
            ),
        )

    top_dir, top_score = ("BUY", buy_score) if buy_score >= sell_score else ("SELL", sell_score)
    second_score = sell_score if top_dir == "BUY" else buy_score
    agreement = top_score / directional_total
    evidence = directional_total / total_weight
    coverage = sum(weights[s.indicator_name] for s in individual_signals
                   if s.direction != SignalDirection.HOLD and s.confidence > 0) / total_weight

    # If BUY and SELL scores are too close, signal is ambiguous -> HOLD
    if evidence < policy.min_evidence_strength or top_score == second_score or (top_score - second_score) / directional_total < policy.ambiguity_threshold:
        confidence = 0.0
        direction = SignalDirection.HOLD
    else:
        direction = SignalDirection(top_dir)
        confidence = top_score / directional_total

    reasoning = _build_reasoning(direction, confidence, individual_signals, buy_score, sell_score)
    if evidence < policy.min_evidence_strength:
        reasoning = (f"Insufficient directional evidence ({evidence:.1%}; "
                     f"minimum {policy.min_evidence_strength:.1%}). "
                     f"Actionable votes agree {agreement:.0%}, but cover only {coverage:.0%} "
                     "of selected indicator weight.")

    return CombinedSignal(
        direction=direction,
        confidence=confidence,
        scores=direction_scores,
        individual_signals=individual_signals,
        reasoning=reasoning,
        directional_agreement=agreement,
        evidence_strength=evidence,
        actionable_coverage=coverage,
    )
