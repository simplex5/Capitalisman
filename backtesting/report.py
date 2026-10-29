"""Backtest report dataclass."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Trade:
    """A single backtested trade."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    predicted_direction: str
    actual_direction: str
    correct: bool
    pnl_pct: float  # fractional return on entry equity, after both fills
    signal_date: pd.Timestamp | None = None
    target_date: pd.Timestamp | None = None
    quantity: float = 0.0
    entry_equity: float | None = None
    pnl_dollars: float | None = None
    fees: float = 0.0
    exit_reason: str = "horizon"


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: pd.Timestamp
    cash: float
    quantity: float
    close_price: float
    equity: float
    exposed: bool  # holding at any time during this bar, including same-bar exits


@dataclass
class BacktestReport:
    """Complete backtest results."""

    ticker: str
    period: str
    horizon_days: int
    initial_capital: float
    trades: list[Trade] = field(default_factory=list)
    is_crypto: bool = False
    evaluation_start: pd.Timestamp | None = None
    evaluation_end: pd.Timestamp | None = None
    excluded_indicators: list[str] = field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)
    completion_reason: str = "end_of_data"
    cost_per_trade_pct: float = 0.0
    configuration: dict = field(default_factory=dict)
    configuration_id: str = ""
    data_fingerprint: str = ""
    computed_data: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # Computed metrics
    total_trades: int = 0
    winning_trades: int = 0  # trades with positive P&L
    losing_trades: int = 0   # trades with negative or zero P&L
    win_rate: float = 0.0    # winning_trades / total_trades
    correct_predictions: int = 0   # direction prediction matched actual
    prediction_accuracy: float = 0.0  # correct_predictions / total_trades
    cumulative_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    exposure_pct: float = 0.0
    benchmark_return: float | None = None  # buy-and-hold return over same period

    # Equity curve
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
