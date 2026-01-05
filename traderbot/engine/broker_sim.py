"""Broker simulator for backtesting.

Simulates order execution with commission and slippage.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

from traderbot.config import get_config
from traderbot.logging_setup import get_logger

logger = get_logger("engine.broker_sim")


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""

    ticker: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    order_id: str = ""
    timestamp: datetime | None = None
    status: OrderStatus = OrderStatus.PENDING

    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"{self.ticker}_{id(self)}"


@dataclass
class Fill:
    """Represents an order fill."""

    order_id: str
    ticker: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    slippage: float = 0.0
    fee: float = 0.0  # Per-share fee total


@dataclass
class Position:
    """Represents a position in a security."""

    ticker: str
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Get position market value (requires current price)."""
        return 0.0  # Needs current price

    def update(self, fill: Fill) -> float:
        """Update position from a fill.

        Args:
            fill: The fill to apply.

        Returns:
            Realized P&L from this fill.
        """
        realized = 0.0

        if fill.side == OrderSide.BUY:
            # Buying increases position
            if self.quantity >= 0:
                # Adding to long or new long
                total_cost = (self.quantity * self.avg_cost) + (fill.quantity * fill.price)
                self.quantity += fill.quantity
                if self.quantity > 0:
                    self.avg_cost = total_cost / self.quantity
            else:
                # Covering short
                if fill.quantity <= abs(self.quantity):
                    # Partial cover
                    realized = (self.avg_cost - fill.price) * fill.quantity
                    self.quantity += fill.quantity
                else:
                    # Full cover + go long
                    realized = (self.avg_cost - fill.price) * abs(self.quantity)
                    remaining = fill.quantity - abs(self.quantity)
                    self.quantity = remaining
                    self.avg_cost = fill.price
        else:
            # Selling decreases position
            if self.quantity <= 0:
                # Adding to short or new short
                total_cost = (abs(self.quantity) * self.avg_cost) + (fill.quantity * fill.price)
                self.quantity -= fill.quantity
                if self.quantity < 0:
                    self.avg_cost = total_cost / abs(self.quantity)
            else:
                # Selling long
                if fill.quantity <= self.quantity:
                    # Partial sell
                    realized = (fill.price - self.avg_cost) * fill.quantity
                    self.quantity -= fill.quantity
                else:
                    # Full sell + go short
                    realized = (fill.price - self.avg_cost) * self.quantity
                    remaining = fill.quantity - self.quantity
                    self.quantity = -remaining
                    self.avg_cost = fill.price

        self.realized_pnl += realized
        return realized


class BrokerSimulator:
    """Simulates broker order execution.

    Handles:
    - Market order fills at bar prices
    - Commission, slippage, and per-share fees
    - Position tracking
    - Cash and equity management
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        commission_bps: float | None = None,
        slippage_bps: float | None = None,
        fee_per_share: float | None = None,
    ):
        """Initialize broker simulator.

        Args:
            initial_capital: Starting cash. Defaults to config value.
            commission_bps: Commission in basis points. Defaults to config.
            slippage_bps: Slippage in basis points. Defaults to config.
            fee_per_share: Per-share execution fee. Defaults to config.
        """
        config = get_config()
        self._initial_capital = initial_capital or config.backtest.initial_capital
        self._commission_bps = commission_bps or config.backtest.commission_bps
        self._slippage_bps = slippage_bps or config.backtest.slippage_bps
        self._fee_per_share = fee_per_share if fee_per_share is not None else config.execution.fee_per_share

        self._cash = self._initial_capital
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._pending_orders: list[Order] = []
        self._realized_pnl = 0.0
        self._total_fees = 0.0
        self._total_slippage = 0.0
        self._total_commission = 0.0

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self._cash = self._initial_capital
        self._positions.clear()
        self._fills.clear()
        self._pending_orders.clear()
        self._realized_pnl = 0.0
        self._total_fees = 0.0
        self._total_slippage = 0.0
        self._total_commission = 0.0

    def submit_order(self, order: Order) -> str:
        """Submit an order for execution.

        Args:
            order: Order to submit.

        Returns:
            Order ID.
        """
        order.status = OrderStatus.PENDING
        self._pending_orders.append(order)
        logger.debug(f"Order submitted: {order}")
        return order.order_id

    def process_bar(
        self,
        bar_data: dict[str, pd.Series],
        timestamp: datetime,
    ) -> list[Fill]:
        """Process pending orders against bar data.

        Args:
            bar_data: Dict mapping ticker to bar data (open, high, low, close, volume).
            timestamp: Bar timestamp.

        Returns:
            List of fills generated.
        """
        fills = []
        remaining_orders = []

        for order in self._pending_orders:
            if order.ticker not in bar_data:
                remaining_orders.append(order)
                continue

            bar = bar_data[order.ticker]
            fill = self._try_fill_order(order, bar, timestamp)

            if fill:
                fills.append(fill)
                self._apply_fill(fill)
            else:
                remaining_orders.append(order)

        self._pending_orders = remaining_orders
        return fills

    def _try_fill_order(
        self,
        order: Order,
        bar: pd.Series,
        timestamp: datetime,
    ) -> Fill | None:
        """Try to fill an order against a bar.

        Args:
            order: Order to fill.
            bar: Bar data.
            timestamp: Execution timestamp.

        Returns:
            Fill if order was filled, None otherwise.
        """
        if order.order_type == OrderType.MARKET:
            # Fill at open with slippage
            base_price = bar["open"]
            slippage = base_price * (self._slippage_bps / 10000.0)

            if order.side == OrderSide.BUY:
                fill_price = base_price + slippage
            else:
                fill_price = base_price - slippage

            # Calculate commission
            notional = fill_price * order.quantity
            commission = notional * (self._commission_bps / 10000.0)

            # Calculate per-share fee
            fee = order.quantity * self._fee_per_share

            order.status = OrderStatus.FILLED

            return Fill(
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                commission=commission,
                timestamp=timestamp,
                slippage=slippage * order.quantity,
                fee=fee,
            )

        elif order.order_type == OrderType.LIMIT:
            # Check if limit was hit
            if order.side == OrderSide.BUY:
                if bar["low"] <= order.limit_price:
                    fill_price = min(order.limit_price, bar["open"])
                    notional = fill_price * order.quantity
                    commission = notional * (self._commission_bps / 10000.0)
                    fee = order.quantity * self._fee_per_share
                    order.status = OrderStatus.FILLED
                    return Fill(
                        order_id=order.order_id,
                        ticker=order.ticker,
                        side=order.side,
                        quantity=order.quantity,
                        price=fill_price,
                        commission=commission,
                        timestamp=timestamp,
                        fee=fee,
                    )
            else:
                if bar["high"] >= order.limit_price:
                    fill_price = max(order.limit_price, bar["open"])
                    notional = fill_price * order.quantity
                    commission = notional * (self._commission_bps / 10000.0)
                    fee = order.quantity * self._fee_per_share
                    order.status = OrderStatus.FILLED
                    return Fill(
                        order_id=order.order_id,
                        ticker=order.ticker,
                        side=order.side,
                        quantity=order.quantity,
                        price=fill_price,
                        commission=commission,
                        timestamp=timestamp,
                        fee=fee,
                    )

        elif order.order_type == OrderType.STOP:
            # Check if stop was triggered
            if order.side == OrderSide.BUY:
                if bar["high"] >= order.stop_price:
                    fill_price = max(order.stop_price, bar["open"])
                    slippage = fill_price * (self._slippage_bps / 10000.0)
                    fill_price += slippage
                    notional = fill_price * order.quantity
                    commission = notional * (self._commission_bps / 10000.0)
                    fee = order.quantity * self._fee_per_share
                    order.status = OrderStatus.FILLED
                    return Fill(
                        order_id=order.order_id,
                        ticker=order.ticker,
                        side=order.side,
                        quantity=order.quantity,
                        price=fill_price,
                        commission=commission,
                        timestamp=timestamp,
                        slippage=slippage * order.quantity,
                        fee=fee,
                    )
            else:
                if bar["low"] <= order.stop_price:
                    fill_price = min(order.stop_price, bar["open"])
                    slippage = fill_price * (self._slippage_bps / 10000.0)
                    fill_price -= slippage
                    notional = fill_price * order.quantity
                    commission = notional * (self._commission_bps / 10000.0)
                    fee = order.quantity * self._fee_per_share
                    order.status = OrderStatus.FILLED
                    return Fill(
                        order_id=order.order_id,
                        ticker=order.ticker,
                        side=order.side,
                        quantity=order.quantity,
                        price=fill_price,
                        commission=commission,
                        timestamp=timestamp,
                        slippage=slippage * order.quantity,
                        fee=fee,
                    )

        return None

    def _apply_fill(self, fill: Fill) -> None:
        """Apply a fill to positions and cash.

        Args:
            fill: Fill to apply.
        """
        # Get or create position
        if fill.ticker not in self._positions:
            self._positions[fill.ticker] = Position(ticker=fill.ticker)

        position = self._positions[fill.ticker]

        # Update position and get realized P&L
        realized = position.update(fill)
        self._realized_pnl += realized

        # Update cash
        if fill.side == OrderSide.BUY:
            self._cash -= fill.price * fill.quantity
        else:
            self._cash += fill.price * fill.quantity

        # Deduct commission and fees
        self._cash -= fill.commission
        self._cash -= fill.fee

        # Track totals
        self._total_commission += fill.commission
        self._total_fees += fill.fee
        self._total_slippage += fill.slippage

        # Record fill
        self._fills.append(fill)

        logger.debug(
            f"Fill applied: {fill.ticker} {fill.side.value} "
            f"{fill.quantity}@{fill.price:.2f}, commission={fill.commission:.2f}, fee={fill.fee:.4f}"
        )

    def get_position(self, ticker: str) -> Position | None:
        """Get position for a ticker."""
        return self._positions.get(ticker)

    def get_positions(self) -> dict[str, Position]:
        """Get all positions."""
        return self._positions.copy()

    def calculate_equity(self, current_prices: dict[str, float]) -> float:
        """Calculate total equity (cash + position values).

        Args:
            current_prices: Dict mapping ticker to current price.

        Returns:
            Total equity value.
        """
        equity = self._cash

        for ticker, position in self._positions.items():
            if ticker in current_prices and position.quantity != 0:
                equity += position.quantity * current_prices[ticker]

        return equity

    def calculate_gross_exposure(self, current_prices: dict[str, float]) -> float:
        """Calculate gross exposure (sum of absolute position values).

        Args:
            current_prices: Dict mapping ticker to current price.

        Returns:
            Gross exposure value.
        """
        gross = 0.0

        for ticker, position in self._positions.items():
            if ticker in current_prices and position.quantity != 0:
                gross += abs(position.quantity * current_prices[ticker])

        return gross

    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self._cash

    @property
    def realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self._realized_pnl

    @property
    def fills(self) -> list[Fill]:
        """Get all fills."""
        return self._fills.copy()

    @property
    def pending_orders(self) -> list[Order]:
        """Get pending orders."""
        return self._pending_orders.copy()

    @property
    def total_fees(self) -> float:
        """Get total per-share fees paid."""
        return self._total_fees

    @property
    def total_slippage(self) -> float:
        """Get total slippage cost."""
        return self._total_slippage

    @property
    def total_commission(self) -> float:
        """Get total commission paid."""
        return self._total_commission

    @property
    def total_execution_costs(self) -> float:
        """Get total execution costs (commission + fees + slippage)."""
        return self._total_commission + self._total_fees + self._total_slippage

    def cancel_all_orders(self) -> int:
        """Cancel all pending orders.

        Returns:
            Number of orders cancelled.
        """
        count = len(self._pending_orders)
        for order in self._pending_orders:
            order.status = OrderStatus.CANCELLED
        self._pending_orders.clear()
        return count
