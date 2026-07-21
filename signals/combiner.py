"""Weighted majority voting signal combiner."""

import pandas as pd

from config.overrides import get_setting
from config.settings import INDICATOR_CATEGORIES, TIMESCALE_ADJUSTMENTS
from indicators.base import BaseIndicator
from signals.base import CombinedSignal, SignalDirection, SignalResult


def _get_timescale(horizon_days: int) -> str:
    if horizon_days <= 3:
        return "short"
    elif horizon_days <= 10:
        return "medium"
    return "long"


def _get_adjusted_weight(indicator_name: str, horizon_days: int) -> float:
    weights = get_setting("INDICATOR_WEIGHTS")
    base_weight = weights.get(indicator_name, 1.0)
    category = INDICATOR_CATEGORIES.get(indicator_name, "trend")
    timescale = _get_timescale(horizon_days)
    adjustment = TIMESCALE_ADJUSTMENTS[timescale].get(category, 1.0)
    return base_weight * adjustment


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
    direction_scores: dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

    working_df = df if precomputed else df.copy()

    for name, indicator in indicators.items():
        if not precomputed:
            working_df = indicator.compute_for_horizon(working_df, horizon_days)
        signal = indicator.get_signal_for_horizon(
            working_df, horizon_days=horizon_days, idx=idx
        )
        individual_signals.append(signal)

        weight = _get_adjusted_weight(name, horizon_days)
        score = signal.confidence * weight
        # HOLD signals are recorded for display but don't participate in
        # directional voting — they represent absence of a signal, not a
        # competing direction.
        direction_scores[signal.direction.value] += score

    # Only BUY and SELL compete; HOLD is the fallback when neither wins
    buy_score = direction_scores["BUY"]
    sell_score = direction_scores["SELL"]
    directional_total = buy_score + sell_score

    # Maximum possible weighted conviction: every indicator voting one way at
    # full strength. Confidence normalizes against this rather than the
    # directional votes alone, so broad abstention lowers the score — one weak
    # voter among many silent indicators must not read as near-100% agreement.
    total_possible = sum(
        _get_adjusted_weight(name, horizon_days) for name in indicators
    )

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

    # If BUY and SELL scores are too close, signal is ambiguous -> HOLD
    if (top_score - second_score) / directional_total < get_setting("AMBIGUITY_THRESHOLD"):
        confidence = 0.0
        direction = SignalDirection.HOLD
    else:
        direction = SignalDirection(top_dir)
        confidence = top_score / total_possible if total_possible > 0 else 0.0

    reasoning = _build_reasoning(direction, confidence, individual_signals, buy_score, sell_score)

    return CombinedSignal(
        direction=direction,
        confidence=confidence,
        scores=direction_scores,
        individual_signals=individual_signals,
        reasoning=reasoning,
    )
