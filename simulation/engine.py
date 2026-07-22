"""Deterministic daily-bar, long-or-cash historical simulation engine."""

import math

import pandas as pd

from signals.base import SignalDirection
from simulation.models import (
    CompletionReason,
    EventType,
    GoalKind,
    OrderAction,
    PendingOrder,
    PortfolioState,
    SimulationConfig,
    SimulationEvent,
    SimulationSnapshot,
    SimulationState,
    SimulationStatus,
)
from simulation.strategies import PreparedStrategy


_REQUIRED_COLUMNS = {"Open", "Close"}


class HistoricalSimulationEngine:
    """Advance a portfolio over immutable historical daily bars.

    Signals are evaluated after a bar's close and any resulting order fills at
    the following bar's open. This causal boundary is the core anti-lookahead
    invariant. The engine has no wall-clock timer; clients can call ``step``
    for a manual tick or resume and call ``advance`` for accelerated playback.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: SimulationConfig,
        strategy: PreparedStrategy,
    ) -> None:
        if not strategy.matches_data(df):
            raise ValueError("Prepared strategy does not match the supplied market data.")
        self.df = self._validate_data(df)
        self.config = config
        self._strategy = strategy
        if len(strategy.signals) != len(self.df):
            raise ValueError("Prepared strategy signal count must match market data.")

        requested_start = self._align_timestamp(config.start_date)
        self.start_position = int(self.df.index.searchsorted(requested_start, side="left"))
        if self.start_position >= len(self.df):
            raise ValueError("Start date is after the available market data.")

        if config.end_date is None:
            self.end_position = len(self.df) - 1
            self._end_reason = CompletionReason.LATEST_DATA
        else:
            requested_end = self._align_timestamp(config.end_date)
            self.end_position = int(self.df.index.searchsorted(requested_end, side="right")) - 1
            if self.end_position < self.start_position:
                raise ValueError("No market bars exist in the requested date range.")
            self._end_reason = CompletionReason.END_DATE

        self.state = SimulationState(
            status=SimulationStatus.READY,
            current_position=self.start_position - 1,
            current_date=None,
            portfolio=PortfolioState(cash=config.starting_capital),
            active_strategy_id=strategy.definition.strategy_id,
        )
        self._add_event(
            self.df.index[self.start_position],
            EventType.CREATED,
            f"Simulation created for {config.ticker} with {strategy.definition.name}.",
            starting_capital=f"{config.starting_capital:.2f}",
        )
        # The strategy is selected before the drop-in date, so the most recent
        # completed close may queue the first order for the drop-in open.
        decision_position = self.start_position - 1
        if decision_position >= 0:
            self._queue_from_signal(
                decision_position, event_timestamp=self.df.index[self.start_position]
            )

    @staticmethod
    def _validate_data(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Simulation requires non-empty market data.")
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError("Market data is missing columns: " + ", ".join(sorted(missing)))
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Market data must use a DatetimeIndex.")
        if df.index.hasnans:
            raise ValueError("Market data index must not contain missing timestamps.")
        if not df.index.is_monotonic_increasing or not df.index.is_unique:
            raise ValueError("Market data index must be sorted and unique.")
        validated = df.copy()
        for column in _REQUIRED_COLUMNS:
            numeric = pd.to_numeric(validated[column], errors="coerce")
            if numeric.isna().any() or (~numeric.map(math.isfinite)).any() or (numeric <= 0).any():
                raise ValueError(f"Market data {column} values must be finite and positive.")
            validated[column] = numeric.astype(float)
        return validated

    def _align_timestamp(self, value: pd.Timestamp) -> pd.Timestamp:
        """Align a UI/config timestamp with the data provider's index timezone."""
        timestamp = pd.Timestamp(value)
        index_tz = self.df.index.tz
        if index_tz is None and timestamp.tzinfo is not None:
            return timestamp.tz_localize(None)
        if index_tz is not None and timestamp.tzinfo is None:
            return timestamp.tz_localize(index_tz)
        if index_tz is not None and timestamp.tzinfo is not None:
            return timestamp.tz_convert(index_tz)
        return timestamp

    @property
    def active_strategy(self) -> PreparedStrategy:
        return self._strategy

    @property
    def resolved_start_date(self) -> pd.Timestamp:
        return self.df.index[self.start_position]

    @property
    def resolved_end_date(self) -> pd.Timestamp:
        return self.df.index[self.end_position]

    @property
    def remaining_bars(self) -> int:
        if self.state.terminal:
            return 0
        return max(0, self.end_position - self.state.current_position)

    def resume(self) -> None:
        if self.state.terminal or self.state.status == SimulationStatus.RUNNING:
            return
        self.state.status = SimulationStatus.RUNNING
        self._add_event(self._event_timestamp(), EventType.STATUS_CHANGED, "Simulation resumed.")

    def pause(self) -> None:
        if self.state.terminal or self.state.status == SimulationStatus.PAUSED:
            return
        self.state.status = SimulationStatus.PAUSED
        self._add_event(self._event_timestamp(), EventType.STATUS_CHANGED, "Simulation paused.")

    def step(self) -> SimulationSnapshot | None:
        """Advance exactly one bar, even while paused (manual stepping)."""
        if self.state.terminal:
            return None
        snapshot = self._process_next_bar()
        if not self.state.terminal and self.state.status == SimulationStatus.READY:
            self.state.status = SimulationStatus.PAUSED
        return snapshot

    def advance(self, bars: int) -> list[SimulationSnapshot]:
        """Advance up to ``bars`` bars only while the run is playing."""
        if bars < 0:
            raise ValueError("Bars to advance must be non-negative.")
        if self.state.status != SimulationStatus.RUNNING or self.state.terminal:
            return []
        snapshots = []
        for _ in range(bars):
            snapshot = self._process_next_bar()
            if snapshot is None:
                break
            snapshots.append(snapshot)
            if self.state.terminal:
                break
        return snapshots

    def change_strategy(self, strategy: PreparedStrategy) -> None:
        """Switch prospectively, preserving holdings and cancelling stale orders."""
        if self.state.terminal:
            raise RuntimeError("A completed simulation cannot change strategy.")
        if self.state.status == SimulationStatus.RUNNING:
            raise RuntimeError("Pause the simulation before changing strategy.")
        if len(strategy.signals) != len(self.df):
            raise ValueError("Prepared strategy signal count must match market data.")
        if not strategy.matches_data(self.df):
            raise ValueError("Prepared strategy does not match the simulation market data.")

        old_name = self._strategy.definition.name
        if self.state.pending_order is not None:
            self._add_event(
                self._event_timestamp(),
                EventType.ORDER_CANCELLED,
                f"Cancelled pending {self.state.pending_order.action.value} after strategy change.",
            )
            self.state.pending_order = None

        self._strategy = strategy
        self.state.active_strategy_id = strategy.definition.strategy_id
        self._add_event(
            self._event_timestamp(),
            EventType.STRATEGY_CHANGED,
            f"Strategy changed from {old_name} to {strategy.definition.name}.",
        )
        if self.state.current_position >= 0:
            self._queue_from_signal(self.state.current_position)
        elif self.start_position > 0:
            self._queue_from_signal(
                self.start_position - 1,
                event_timestamp=self.df.index[self.start_position],
            )

    def equity_series(self) -> pd.Series:
        return pd.Series(
            [snapshot.equity for snapshot in self.state.snapshots],
            index=pd.DatetimeIndex([snapshot.timestamp for snapshot in self.state.snapshots]),
            name="Equity",
            dtype=float,
        )

    def _process_next_bar(self) -> SimulationSnapshot | None:
        position = self.state.current_position + 1
        if position > self.end_position:
            self._complete(self._end_reason)
            return None

        timestamp = self.df.index[position]
        open_price = float(self.df["Open"].iloc[position])
        close_price = float(self.df["Close"].iloc[position])
        self._fill_pending_order(timestamp, open_price)

        portfolio = self.state.portfolio
        portfolio.last_price = close_price
        equity = portfolio.equity
        snapshot = SimulationSnapshot(
            timestamp=timestamp,
            open_price=open_price,
            close_price=close_price,
            cash=portfolio.cash,
            quantity=portfolio.quantity,
            market_value=portfolio.market_value,
            equity=equity,
            return_pct=equity / self.config.starting_capital - 1.0,
            strategy_id=self.state.active_strategy_id,
        )
        self.state.current_position = position
        self.state.current_date = timestamp
        self.state.snapshots.append(snapshot)

        if equity <= 0:
            self._complete(CompletionReason.CAPITAL_DEPLETED)
            return snapshot

        if self._goal_is_reached(equity):
            if not self.state.goal_reached:
                self.state.goal_reached = True
                self._add_event(timestamp, EventType.GOAL_REACHED, "Profit goal reached.")
            if self.config.goal.stop_when_reached:
                self._complete(CompletionReason.GOAL_REACHED)
                return snapshot

        if position >= self.end_position:
            self._complete(self._end_reason)
            return snapshot

        self._queue_from_signal(position)
        return snapshot

    def _queue_from_signal(
        self,
        decision_position: int,
        event_timestamp: pd.Timestamp | None = None,
    ) -> None:
        signal = self._strategy.signal_at(decision_position)
        action = None
        if signal.direction == SignalDirection.BUY and self.state.portfolio.quantity == 0:
            action = OrderAction.BUY
        elif signal.direction == SignalDirection.SELL and self.state.portfolio.quantity > 0:
            action = OrderAction.SELL

        if action is None:
            self.state.pending_order = None
            return

        signal_date = self.df.index[decision_position]
        self.state.pending_order = PendingOrder(
            action=action,
            signal_date=signal_date,
            strategy_id=self._strategy.definition.strategy_id,
            reason=signal.reasoning,
        )
        self._add_event(
            event_timestamp if event_timestamp is not None else signal_date,
            EventType.ORDER_QUEUED,
            f"Queued {action.value} for the next market open.",
            confidence=f"{signal.confidence:.4f}",
            signal_date=str(signal_date),
            strategy=self._strategy.definition.name,
        )

    def _fill_pending_order(self, timestamp: pd.Timestamp, open_price: float) -> None:
        order = self.state.pending_order
        if order is None:
            return
        portfolio = self.state.portfolio
        cost_rate = self.config.transaction_cost_pct / 100.0

        if order.action == OrderAction.BUY and portfolio.quantity == 0:
            quantity = portfolio.cash / (open_price * (1.0 + cost_rate))
            notional = quantity * open_price
            fee = notional * cost_rate
            portfolio.cash = max(0.0, portfolio.cash - notional - fee)
            portfolio.quantity = quantity
        elif order.action == OrderAction.SELL and portfolio.quantity > 0:
            quantity = portfolio.quantity
            notional = quantity * open_price
            fee = notional * cost_rate
            portfolio.cash += notional - fee
            portfolio.quantity = 0.0
        else:
            self.state.pending_order = None
            return

        portfolio.total_fees += fee
        self._add_event(
            timestamp,
            EventType.ORDER_FILLED,
            f"Filled {order.action.value} at ${open_price:,.2f}.",
            quantity=f"{quantity:.8f}",
            fee=f"{fee:.8f}",
            action=order.action.value,
            price=f"{open_price:.8f}",
            signal_date=str(order.signal_date),
        )
        self.state.pending_order = None

    def _goal_is_reached(self, equity: float) -> bool:
        goal = self.config.goal
        if goal.target is None:
            return False
        profit = equity - self.config.starting_capital
        if goal.kind == GoalKind.PROFIT_DOLLARS:
            return profit >= goal.target
        if goal.kind == GoalKind.RETURN_PERCENT:
            return profit / self.config.starting_capital >= goal.target
        return False

    def _complete(self, reason: CompletionReason) -> None:
        if self.state.terminal:
            return
        self.state.status = SimulationStatus.COMPLETED
        self.state.completion_reason = reason
        self.state.pending_order = None
        self._add_event(
            self._event_timestamp(),
            EventType.COMPLETED,
            f"Simulation completed: {reason.value.replace('_', ' ')}.",
        )

    def _event_timestamp(self) -> pd.Timestamp:
        if self.state.current_date is not None:
            return self.state.current_date
        return self.df.index[self.start_position]

    def _add_event(
        self,
        timestamp: pd.Timestamp,
        event_type: EventType,
        message: str,
        **details: str,
    ) -> None:
        self.state.events.append(
            SimulationEvent(
                sequence=len(self.state.events) + 1,
                timestamp=pd.Timestamp(timestamp),
                event_type=event_type,
                message=message,
                details=tuple(sorted((key, str(value)) for key, value in details.items())),
            )
        )
