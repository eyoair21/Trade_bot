"""Deterministic broker simulation for paper trading.

Simulates order execution with configurable slippage and costs.
All fills are deterministic given the same inputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from traderbot.logging_setup import get_logger

logger = get_logger("paper.broker_sim")


class OrderSide(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Order representation."""

    symbol: str
    side: OrderSide
    quantity: int
    timestamp: str  # ISO format
    order_id: str = ""
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


@dataclass
class Fill:
    """Fill (execution) representation."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    expected_price: float
    filled_price: float
    slippage_bps: float
    cost_bps: float
    total_cost: float  # Absolute cost in currency
    timestamp: str


@dataclass
class Position:
    """Position representation."""

    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class AccountState:
    """Account state representation."""

    equity: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: List[Order] = field(default_factory=list)
    fills: List[Fill] = field(default_factory=list)


class BrokerSim:
    """Deterministic broker simulation.

    Simulates order execution with:
    - Configurable slippage (default 10 bps)
    - Transaction costs (default 2 bps per side)
    - Maximum concurrent positions limit

    All behavior is deterministic given the same inputs.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        slippage_bps: float = 10.0,
        cost_bps: float = 2.0,
        max_positions: int = 25,
    ):
        """Initialize broker simulation.

        Args:
            initial_capital: Starting capital.
            slippage_bps: Slippage in basis points.
            cost_bps: Transaction cost per side in basis points.
            max_positions: Maximum concurrent positions.
        """
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.cost_bps = cost_bps
        self.max_positions = max_positions

        self.state = AccountState(
            equity=initial_capital,
            cash=initial_capital,
        )
        self._order_counter = 0

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"ORD-{self._order_counter:06d}"

    def _apply_slippage(
        self,
        price: float,
        side: OrderSide,
    ) -> float:
        """Apply deterministic slippage to price.

        Buys slip up, sells slip down.
        """
        slip_factor = self.slippage_bps / 10_000.0
        if side == OrderSide.BUY:
            return price * (1 + slip_factor)
        else:
            return price * (1 - slip_factor)

    def _compute_cost(self, notional: float) -> float:
        """Compute transaction cost."""
        return notional * (self.cost_bps / 10_000.0)

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        reference_price: float,
    ) -> Fill:
        """Submit and immediately fill an order.

        In this simulation, all orders are filled immediately
        at the reference price plus slippage and costs.

        Args:
            symbol: Stock symbol.
            side: BUY or SELL.
            quantity: Number of shares.
            reference_price: Market price for fill calculation.

        Returns:
            Fill record.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        order_id = self._generate_order_id()

        # Calculate fill price with slippage
        filled_price = self._apply_slippage(reference_price, side)

        # Calculate costs
        notional = abs(quantity) * filled_price
        total_cost = self._compute_cost(notional)

        # Create fill
        fill = Fill(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            expected_price=reference_price,
            filled_price=filled_price,
            slippage_bps=self.slippage_bps,
            cost_bps=self.cost_bps,
            total_cost=total_cost,
            timestamp=timestamp,
        )

        # Update position
        self._update_position(fill)

        # Record fill
        self.state.fills.append(fill)

        logger.debug(
            "Filled %s %d %s @ %.2f (expected %.2f)",
            side.value,
            quantity,
            symbol,
            filled_price,
            reference_price,
        )

        return fill

    def _update_position(self, fill: Fill) -> None:
        """Update positions based on fill."""
        symbol = fill.symbol
        qty_change = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity

        if symbol in self.state.positions:
            pos = self.state.positions[symbol]
            old_qty = pos.quantity
            new_qty = old_qty + qty_change

            if new_qty == 0:
                # Position closed
                realized = (fill.filled_price - pos.avg_cost) * abs(qty_change)
                if fill.side == OrderSide.SELL:
                    self.state.cash += fill.quantity * fill.filled_price - fill.total_cost
                else:
                    self.state.cash -= abs(fill.quantity) * fill.filled_price + fill.total_cost
                del self.state.positions[symbol]
            else:
                # Position updated
                if abs(new_qty) > abs(old_qty):
                    # Adding to position - update avg cost
                    total_cost = pos.avg_cost * abs(old_qty) + fill.filled_price * abs(qty_change)
                    pos.avg_cost = total_cost / abs(new_qty)
                pos.quantity = new_qty
                pos.current_price = fill.filled_price

                if fill.side == OrderSide.BUY:
                    self.state.cash -= fill.quantity * fill.filled_price + fill.total_cost
                else:
                    self.state.cash += fill.quantity * fill.filled_price - fill.total_cost
        else:
            # New position
            self.state.positions[symbol] = Position(
                symbol=symbol,
                quantity=qty_change,
                avg_cost=fill.filled_price,
                current_price=fill.filled_price,
                unrealized_pnl=0.0,
            )
            if fill.side == OrderSide.BUY:
                self.state.cash -= fill.quantity * fill.filled_price + fill.total_cost
            else:
                self.state.cash += abs(fill.quantity) * fill.filled_price - fill.total_cost

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update position prices and calculate equity.

        Args:
            prices: Mapping of symbol to current price.
        """
        total_position_value = 0.0

        for symbol, pos in self.state.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]
            pos.unrealized_pnl = (pos.current_price - pos.avg_cost) * pos.quantity
            total_position_value += pos.quantity * pos.current_price

        self.state.equity = self.state.cash + total_position_value

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.state.positions.get(symbol)

    def get_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.state.positions.copy()

    def get_fills(self) -> List[Fill]:
        """Get all fills."""
        return self.state.fills.copy()

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.state.positions) < self.max_positions

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "equity": self.state.equity,
            "cash": self.state.cash,
            "initial_capital": self.initial_capital,
            "slippage_bps": self.slippage_bps,
            "cost_bps": self.cost_bps,
            "max_positions": self.max_positions,
            "positions": {
                sym: asdict(pos) for sym, pos in self.state.positions.items()
            },
            "fills": [asdict(f) for f in self.state.fills],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrokerSim":
        """Deserialize from dictionary."""
        broker = cls(
            initial_capital=data.get("initial_capital", 100_000.0),
            slippage_bps=data.get("slippage_bps", 10.0),
            cost_bps=data.get("cost_bps", 2.0),
            max_positions=data.get("max_positions", 25),
        )

        broker.state.equity = data.get("equity", broker.initial_capital)
        broker.state.cash = data.get("cash", broker.initial_capital)

        # Restore positions
        for sym, pos_data in data.get("positions", {}).items():
            broker.state.positions[sym] = Position(**pos_data)

        # Restore fills
        for fill_data in data.get("fills", []):
            fill_data["side"] = OrderSide(fill_data["side"])
            broker.state.fills.append(Fill(**fill_data))

        return broker

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved broker state to %s", path)

    @classmethod
    def load(cls, path: Path) -> "BrokerSim":
        """Load state from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
