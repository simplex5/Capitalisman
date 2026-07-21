"""Accounting, causality, control, and replay tests for the simulator."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from signals.base import CombinedSignal, SignalDirection
from simulation.engine import HistoricalSimulationEngine
from simulation.models import (
    CompletionReason,
    EventType,
    GoalDefinition,
    GoalKind,
    SimulationConfig,
    SimulationStatus,
    StrategyDefinition,
)
from simulation.strategies import PreparedStrategy


def _bars(opens, closes=None, start="2024-01-01"):
    closes = opens if closes is None else closes
    index = pd.bdate_range(start, periods=len(opens))
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [max(o, c) for o, c in zip(opens, closes)],
            "Low": [min(o, c) for o, c in zip(opens, closes)],
            "Close": closes,
            "Volume": [1_000] * len(opens),
        },
        index=index,
        dtype=float,
    )


def _prepared(df, directions, strategy_id="test", name="Test"):
    definition = StrategyDefinition(
        strategy_id=strategy_id,
        name=name,
        indicator_names=("Synthetic",),
    )
    signals = tuple(
        CombinedSignal(
            direction=direction,
            confidence=0.0 if direction == SignalDirection.HOLD else 1.0,
            reasoning=f"Synthetic {direction.value}",
        )
        for direction in directions
    )
    assert len(signals) == len(df)
    return PreparedStrategy(definition, signals, required_lookback=0)


def _config(df, *, start_position=1, end_position=None, capital=1_000, cost=0, goal=None):
    return SimulationConfig(
        ticker="TEST",
        start_date=df.index[start_position],
        end_date=None if end_position is None else df.index[end_position],
        starting_capital=capital,
        transaction_cost_pct=cost,
        goal=goal or GoalDefinition(),
    )


def test_signal_from_previous_close_fills_at_drop_in_open():
    df = _bars([90, 100, 110], [95, 105, 115])
    strategy = _prepared(
        df, [SignalDirection.BUY, SignalDirection.HOLD, SignalDirection.HOLD]
    )
    engine = HistoricalSimulationEngine(df, _config(df), strategy)

    snapshot = engine.step()

    assert snapshot.timestamp == df.index[1]
    assert snapshot.quantity == pytest.approx(10)
    assert snapshot.cash == pytest.approx(0)
    assert snapshot.equity == pytest.approx(1_050)
    fill = next(event for event in engine.state.events if event.event_type == EventType.ORDER_FILLED)
    assert "100.00" in fill.message


def test_signal_on_current_close_cannot_fill_until_next_open():
    df = _bars([100, 101, 200], [100, 150, 200])
    strategy = _prepared(
        df, [SignalDirection.HOLD, SignalDirection.BUY, SignalDirection.HOLD]
    )
    engine = HistoricalSimulationEngine(df, _config(df), strategy)

    first = engine.step()
    second = engine.step()

    assert first.quantity == 0
    assert first.equity == 1_000
    assert second.quantity == pytest.approx(5)
    assert second.equity == pytest.approx(1_000)


def test_round_trip_charges_each_fill_and_preserves_equity_identity():
    df = _bars([100, 100, 120, 120], [100, 110, 120, 120])
    strategy = _prepared(
        df,
        [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD, SignalDirection.HOLD],
    )
    engine = HistoricalSimulationEngine(df, _config(df, cost=1), strategy)

    buy_snapshot = engine.step()
    sell_snapshot = engine.step()

    expected_quantity = 1_000 / 101
    expected_buy_fee = expected_quantity
    expected_sale = expected_quantity * 120
    expected_sell_fee = expected_sale * 0.01
    assert buy_snapshot.quantity == pytest.approx(expected_quantity)
    assert sell_snapshot.quantity == 0
    assert sell_snapshot.cash == pytest.approx(expected_sale - expected_sell_fee)
    assert engine.state.portfolio.total_fees == pytest.approx(expected_buy_fee + expected_sell_fee)
    for snapshot in engine.state.snapshots:
        assert snapshot.equity == pytest.approx(snapshot.cash + snapshot.market_value)


def test_pause_blocks_batch_advance_but_manual_step_still_works():
    df = _bars([100, 101, 102, 103])
    strategy = _prepared(df, [SignalDirection.HOLD] * len(df))
    engine = HistoricalSimulationEngine(df, _config(df), strategy)

    engine.pause()
    assert engine.advance(10) == []
    assert engine.state.current_position == 0
    assert engine.step() is not None
    assert engine.state.current_position == 1


def test_advance_matches_repeated_steps():
    df = _bars([100, 105, 90, 120, 115, 130])
    directions = [
        SignalDirection.BUY,
        SignalDirection.HOLD,
        SignalDirection.SELL,
        SignalDirection.BUY,
        SignalDirection.HOLD,
        SignalDirection.HOLD,
    ]
    strategy = _prepared(df, directions)
    batch = HistoricalSimulationEngine(df, _config(df, cost=0.25), strategy)
    manual = HistoricalSimulationEngine(df, _config(df, cost=0.25), strategy)

    batch.resume()
    batch.advance(10)
    while not manual.state.terminal:
        manual.step()

    assert batch.state.snapshots == manual.state.snapshots
    assert batch.state.portfolio.cash == pytest.approx(manual.state.portfolio.cash)
    assert batch.state.portfolio.quantity == pytest.approx(manual.state.portfolio.quantity)
    assert batch.state.completion_reason == CompletionReason.LATEST_DATA


def test_strategy_change_is_prospective_and_preserves_open_position():
    df = _bars([100, 100, 110, 120, 130])
    first_strategy = _prepared(
        df,
        [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD,
         SignalDirection.HOLD, SignalDirection.HOLD],
        "first",
        "First",
    )
    replacement = _prepared(
        df,
        [SignalDirection.HOLD, SignalDirection.HOLD, SignalDirection.SELL,
         SignalDirection.HOLD, SignalDirection.HOLD],
        "replacement",
        "Replacement",
    )
    engine = HistoricalSimulationEngine(df, _config(df), first_strategy)

    first_snapshot = engine.step()
    assert first_snapshot.quantity == 10
    assert engine.state.pending_order is not None

    engine.change_strategy(replacement)
    assert engine.state.portfolio.quantity == 10
    assert engine.state.pending_order is None
    assert engine.state.snapshots[0].strategy_id == "first"
    assert any(event.event_type == EventType.ORDER_CANCELLED for event in engine.state.events)

    engine.step()  # replacement decides SELL from this close
    assert engine.state.pending_order is not None
    sold = engine.step()
    assert sold.quantity == 0
    assert sold.strategy_id == "replacement"


@pytest.mark.parametrize(
    ("goal", "expected_equity"),
    [
        (GoalDefinition(GoalKind.PROFIT_DOLLARS, 100), 1_100),
        (GoalDefinition(GoalKind.RETURN_PERCENT, 0.10), 1_100),
    ],
)
def test_goal_can_stop_run_at_first_qualifying_close(goal, expected_equity):
    df = _bars([100, 100, 120, 130], [100, 110, 120, 130])
    strategy = _prepared(df, [SignalDirection.BUY] + [SignalDirection.HOLD] * 3)
    engine = HistoricalSimulationEngine(df, _config(df, goal=goal), strategy)

    snapshot = engine.step()

    assert snapshot.equity == pytest.approx(expected_equity)
    assert engine.state.status == SimulationStatus.COMPLETED
    assert engine.state.completion_reason == CompletionReason.GOAL_REACHED
    assert engine.state.goal_reached


def test_goal_can_be_recorded_without_stopping():
    df = _bars([100, 100, 120])
    goal = GoalDefinition(GoalKind.PROFIT_DOLLARS, 100, stop_when_reached=False)
    strategy = _prepared(df, [SignalDirection.BUY] + [SignalDirection.HOLD] * 2)
    engine = HistoricalSimulationEngine(df, _config(df, goal=goal), strategy)

    engine.resume()
    engine.advance(10)

    assert engine.state.goal_reached
    assert engine.state.completion_reason == CompletionReason.LATEST_DATA


def test_weekend_start_resolves_to_next_market_bar_and_end_to_previous_bar():
    df = _bars(list(range(100, 110)), start="2024-01-01")
    config = SimulationConfig(
        ticker="TEST",
        start_date=pd.Timestamp("2024-01-06"),  # Saturday -> Monday Jan 8
        end_date=pd.Timestamp("2024-01-13"),  # Saturday -> Friday Jan 12
        starting_capital=1_000,
        transaction_cost_pct=0,
    )
    strategy = _prepared(df, [SignalDirection.HOLD] * len(df))
    engine = HistoricalSimulationEngine(df, config, strategy)

    assert engine.resolved_start_date == pd.Timestamp("2024-01-08")
    assert engine.resolved_end_date == pd.Timestamp("2024-01-12")


def test_specific_end_date_stops_on_last_market_bar_at_or_before_date():
    df = _bars([100, 101, 102, 103, 104, 105], start="2024-01-01")
    config = SimulationConfig(
        ticker="TEST",
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-01-06"),  # Saturday -> Friday Jan 5
        starting_capital=1_000,
        transaction_cost_pct=0,
    )
    strategy = _prepared(df, [SignalDirection.HOLD] * len(df))
    engine = HistoricalSimulationEngine(df, config, strategy)

    engine.resume()
    engine.advance(100)

    assert engine.resolved_end_date == pd.Timestamp("2024-01-05")
    assert engine.state.current_date == pd.Timestamp("2024-01-05")
    assert engine.state.completion_reason == CompletionReason.END_DATE


def test_naive_config_dates_align_with_timezone_aware_provider_data():
    df = _bars([100, 101, 102, 103])
    df.index = df.index.tz_localize("America/New_York")
    config = SimulationConfig(
        ticker="TEST",
        start_date=pd.Timestamp("2024-01-02"),
        end_date=pd.Timestamp("2024-01-03 23:59"),
        starting_capital=1_000,
        transaction_cost_pct=0,
    )
    strategy = _prepared(df, [SignalDirection.HOLD] * len(df))

    engine = HistoricalSimulationEngine(df, config, strategy)

    assert engine.resolved_start_date == pd.Timestamp(
        "2024-01-02", tz="America/New_York"
    )
    assert engine.resolved_end_date == pd.Timestamp(
        "2024-01-03", tz="America/New_York"
    )


def test_known_buy_and_hold_result_and_deterministic_replay():
    df = _bars([80, 100, 110, 125], [80, 100, 110, 125])
    strategy = _prepared(df, [SignalDirection.BUY] * len(df))
    config = _config(df, capital=10_000)

    first = HistoricalSimulationEngine(df, config, strategy)
    second = HistoricalSimulationEngine(df, config, strategy)
    for engine in (first, second):
        engine.resume()
        engine.advance(100)

    assert first.state.portfolio.equity == pytest.approx(12_500)
    assert first.state.snapshots == second.state.snapshots
    assert first.state.events == second.state.events


def test_snapshots_events_and_configuration_are_immutable_records():
    df = _bars([100, 100, 110])
    strategy = _prepared(df, [SignalDirection.BUY] * len(df))
    config = _config(df)
    engine = HistoricalSimulationEngine(df, config, strategy)
    snapshot = engine.step()

    with pytest.raises(FrozenInstanceError):
        snapshot.equity = 0
    with pytest.raises(FrozenInstanceError):
        engine.state.events[0].message = "rewritten"
    with pytest.raises(FrozenInstanceError):
        config.starting_capital = 0


@pytest.mark.parametrize("capital", [float("nan"), float("inf"), 0, -1])
def test_configuration_rejects_invalid_capital(capital):
    with pytest.raises(ValueError, match="Starting capital"):
        SimulationConfig("TEST", pd.Timestamp("2024-01-01"), None, capital)


@pytest.mark.parametrize(
    "bad_df, message",
    [
        (pd.DataFrame(), "non-empty"),
        (pd.DataFrame({"Close": [1]}, index=pd.date_range("2024-01-01", periods=1)), "Open"),
        (_bars([100, 0]), "positive"),
    ],
)
def test_invalid_market_data_fails_early(bad_df, message):
    definition = StrategyDefinition("buy", "Buy", buy_and_hold=True)
    signals = tuple(
        CombinedSignal(SignalDirection.BUY, 1.0) for _ in range(len(bad_df))
    )
    strategy = PreparedStrategy(definition, signals, 0)
    config = SimulationConfig("TEST", pd.Timestamp("2024-01-01"), None, 1_000)

    with pytest.raises(ValueError, match=message):
        HistoricalSimulationEngine(bad_df, config, strategy)
