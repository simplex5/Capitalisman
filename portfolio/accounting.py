"""Single-position cash accounting; all fees are charged on fill notional."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Fill:
    quantity: float  # signed change in position
    price: float
    fee: float


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

    @staticmethod
    def _validate_fill(price: float, fee_rate: float) -> None:
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Fill price must be finite and positive")
        if not math.isfinite(fee_rate) or not 0 <= fee_rate < 1:
            raise ValueError("Fill fee rate must be in [0, 1)")

    def open_position(self, price: float, fee_rate: float, *, short: bool = False) -> Fill:
        self._validate_fill(price, fee_rate)
        if self.quantity != 0 or not math.isfinite(self.cash) or self.cash <= 0:
            raise ValueError("Opening a position requires positive cash and no holding")
        quantity = self.cash / (price * (1 + fee_rate))
        quantity *= -1 if short else 1
        fee = abs(quantity) * price * fee_rate
        self.cash -= quantity * price + fee
        if not short:
            self.cash = max(0.0, self.cash)  # floating-point purchase residue
        self.quantity = quantity
        self.last_price = price
        self.total_fees += fee
        return Fill(quantity, price, fee)

    def close_position(self, price: float, fee_rate: float) -> Fill:
        self._validate_fill(price, fee_rate)
        if self.quantity == 0:
            raise ValueError("Closing a position requires a holding")
        fill = Fill(-self.quantity, price, abs(self.quantity) * price * fee_rate)
        self.cash += self.quantity * price - fill.fee
        self.quantity = 0.0
        self.last_price = price
        self.total_fees += fill.fee
        return fill

    def short_liquidation_price(self, fee_rate: float) -> float | None:
        """Price where covering the short, including its fee, exhausts collateral."""
        if self.quantity >= 0:
            return None
        return self.cash / (-self.quantity * (1 + fee_rate))
