"""Data models for the historical simulation engine.

Configuration, strategies, snapshots, and events are immutable so a completed
run can be audited and replayed. Only the engine-owned portfolio and run state
are mutable while a simulation is in progress.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral

import pandas as pd

from config.settings import DEFAULT_SIMULATION_COST_PER_FILL_PCT


class GoalKind(str, Enum):
    NONE = "none"
    PROFIT_DOLLARS = "profit_dollars"
    RETURN_PERCENT = "return_percent"


class SimulationStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class CompletionReason(str, Enum):
    GOAL_REACHED = "goal_reached"
    END_DATE = "end_date"
    LATEST_DATA = "latest_data"
    CAPITAL_DEPLETED = "capital_depleted"


class OrderAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class EventType(str, Enum):
    CREATED = "created"
    STATUS_CHANGED = "status_changed"
    ORDER_QUEUED = "order_queued"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FILLED = "order_filled"
    STRATEGY_CHANGED = "strategy_changed"
    GOAL_REACHED = "goal_reached"
    COMPLETED = "completed"


@dataclass(frozen=True)
class GoalDefinition:
    """An optional profit goal.

    Percentage targets are represented as decimal fractions: 0.05 means 5%.
    """

    kind: GoalKind = GoalKind.NONE
    target: float | None = None
    stop_when_reached: bool = True

    def __post_init__(self) -> None:
        if self.kind == GoalKind.NONE:
            if self.target is not None:
                raise ValueError("A goal target requires a non-empty goal kind.")
            return
        if self.target is None or not math.isfinite(self.target) or self.target <= 0:
            raise ValueError("Goal target must be greater than zero.")


@dataclass(frozen=True)
class StrategyDefinition:
    """Serializable description of a simulation strategy."""

    strategy_id: str
    name: str
    indicator_names: tuple[str, ...] = ()
    weights: tuple[tuple[str, float], ...] = ()
    horizon_days: int = 5
    ambiguity_threshold: float = 0.10
    buy_and_hold: bool = False

    def __post_init__(self) -> None:
        if not self.strategy_id.strip() or not self.name.strip():
            raise ValueError("Strategy id and name are required.")
        if (
            isinstance(self.horizon_days, bool)
            or not isinstance(self.horizon_days, Integral)
            or self.horizon_days < 1
        ):
            raise ValueError("Strategy horizon must be at least one day.")
        if not math.isfinite(self.ambiguity_threshold) or not 0 <= self.ambiguity_threshold < 1:
            raise ValueError("Ambiguity threshold must be in [0, 1).")
        if not self.buy_and_hold and not self.indicator_names:
            raise ValueError("An indicator strategy must select at least one indicator.")
        if len(set(self.indicator_names)) != len(self.indicator_names):
            raise ValueError("Indicator names must be unique.")
        weight_names = [name for name, _ in self.weights]
        if len(set(weight_names)) != len(weight_names):
            raise ValueError("Indicator weights must be unique.")
        if any(name not in self.indicator_names for name in weight_names):
            raise ValueError("Weights may only reference selected indicators.")
        if any(not math.isfinite(weight) or weight <= 0 for _, weight in self.weights):
            raise ValueError("Indicator weights must be greater than zero.")

    @property
    def weight_map(self) -> dict[str, float]:
        return dict(self.weights)


@dataclass(frozen=True)
class SimulationConfig:
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp | None
    starting_capital: float
    transaction_cost_pct: float = DEFAULT_SIMULATION_COST_PER_FILL_PCT
    goal: GoalDefinition = field(default_factory=GoalDefinition)

    def __post_init__(self) -> None:
        if not self.ticker.strip():
            raise ValueError("Ticker is required.")
        if not math.isfinite(self.starting_capital) or self.starting_capital <= 0:
            raise ValueError("Starting capital must be greater than zero.")
        if (
            not math.isfinite(self.transaction_cost_pct)
            or not 0 <= self.transaction_cost_pct < 100
        ):
            raise ValueError("Transaction cost must be in [0, 100).")
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date) if self.end_date is not None else None
        if pd.isna(start) or (end is not None and pd.isna(end)):
            raise ValueError("Start and end dates must be valid timestamps.")
        comparable_start = start.tz_localize(None) if start.tzinfo is not None else start
        comparable_end = (
            end.tz_localize(None) if end is not None and end.tzinfo is not None else end
        )
        if comparable_end is not None and comparable_end < comparable_start:
            raise ValueError("End date must not be before the start date.")
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)


@dataclass
class PortfolioState:
    cash: float
    quantity: float = 0.0
    last_price: float | None = None
    total_fees: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * (self.last_price or 0.0)

    @property
    def equity(self) -> float:
        return self.cash + self.market_value


@dataclass(frozen=True)
class PendingOrder:
    action: OrderAction
    signal_date: pd.Timestamp
    strategy_id: str
    reason: str


@dataclass(frozen=True)
class SimulationEvent:
    sequence: int
    timestamp: pd.Timestamp
    event_type: EventType
    message: str
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SimulationSnapshot:
    timestamp: pd.Timestamp
    open_price: float
    close_price: float
    cash: float
    quantity: float
    market_value: float
    equity: float
    return_pct: float
    strategy_id: str


@dataclass
class SimulationState:
    status: SimulationStatus
    current_position: int
    current_date: pd.Timestamp | None
    portfolio: PortfolioState
    active_strategy_id: str
    pending_order: PendingOrder | None = None
    completion_reason: CompletionReason | None = None
    goal_reached: bool = False
    snapshots: list[SimulationSnapshot] = field(default_factory=list)
    events: list[SimulationEvent] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.status == SimulationStatus.COMPLETED
