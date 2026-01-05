"""Base strategy class.

Provides the interface for trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from traderbot.engine.broker_sim import Order, OrderSide, OrderType


class SignalType(Enum):
    """Signal type enumeration."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal from a strategy."""

    ticker: str
    signal_type: SignalType
    strength: float = 1.0  # Signal strength/confidence (0-1)
    target_weight: float | None = None  # Target portfolio weight
    stop_price: float | None = None  # Suggested stop price
    take_profit: float | None = None  # Suggested take profit
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyState:
    """Strategy state container."""

    positions: dict[str, float] = field(default_factory=dict)  # ticker -> weight
    signals: dict[str, Signal] = field(default_factory=dict)
    stop_prices: dict[str, float] = field(default_factory=dict)
    entry_prices: dict[str, float] = field(default_factory=dict)
    custom: dict[str, Any] = field(default_factory=dict)


class StrategyBase(ABC):
    """Abstract base class for trading strategies.

    Subclasses must implement:
    - generate_signals: Generate trading signals from data
    - on_bar: Handle new bar data (optional)
    """

    def __init__(self, name: str, universe: list[str] | None = None):
        """Initialize strategy.

        Args:
            name: Strategy name.
            universe: List of tickers to trade.
        """
        self.name = name
        self.universe = universe or []
        self._state = StrategyState()
        self._is_initialized = False
        self.model_predictions: dict[str, float] = {}  # ticker -> prob_up from PatchTST

    def initialize(self, initial_data: dict[str, pd.DataFrame]) -> None:
        """Initialize strategy with historical data.

        Called once at the start of a backtest.

        Args:
            initial_data: Dict mapping ticker to historical OHLCV data.
        """
        self._is_initialized = True

    @abstractmethod
    def generate_signals(
        self,
        current_bar: dict[str, pd.Series],
        historical_data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate trading signals for current bar.

        Args:
            current_bar: Dict mapping ticker to current bar data.
            historical_data: Dict mapping ticker to historical data up to current bar.
            timestamp: Current timestamp.

        Returns:
            List of signals.
        """
        pass

    def signals_to_orders(
        self,
        signals: list[Signal],
        current_prices: dict[str, float],
        positions: dict[str, int],
        nav: float,
        max_position_pct: float = 0.10,
    ) -> list[Order]:
        """Convert signals to orders.

        Args:
            signals: List of signals.
            current_prices: Current prices per ticker.
            positions: Current positions (ticker -> shares).
            nav: Current NAV.
            max_position_pct: Maximum position size as % of NAV.

        Returns:
            List of orders to submit.
        """
        orders = []

        for signal in signals:
            ticker = signal.ticker

            if ticker not in current_prices:
                continue

            price = current_prices[ticker]
            current_qty = positions.get(ticker, 0)

            # Determine target weight
            if signal.target_weight is not None:
                target_weight = signal.target_weight
            else:
                if signal.signal_type == SignalType.LONG:
                    target_weight = max_position_pct * signal.strength
                elif signal.signal_type == SignalType.SHORT:
                    target_weight = -max_position_pct * signal.strength
                elif signal.signal_type == SignalType.FLAT:
                    target_weight = 0.0
                else:
                    continue  # HOLD - no change

            # Calculate target shares
            target_value = nav * target_weight
            target_qty = int(target_value / price)

            # Calculate order quantity
            order_qty = target_qty - current_qty

            if order_qty == 0:
                continue

            # Create order
            if order_qty > 0:
                order = Order(
                    ticker=ticker,
                    side=OrderSide.BUY,
                    quantity=abs(order_qty),
                    order_type=OrderType.MARKET,
                )
            else:
                order = Order(
                    ticker=ticker,
                    side=OrderSide.SELL,
                    quantity=abs(order_qty),
                    order_type=OrderType.MARKET,
                )

            orders.append(order)

            # Track entry prices and stops
            if signal.signal_type in (SignalType.LONG, SignalType.SHORT):
                self._state.entry_prices[ticker] = price
                if signal.stop_price:
                    self._state.stop_prices[ticker] = signal.stop_price

        return orders

    def on_fill(self, ticker: str, quantity: int, price: float) -> None:  # noqa: B027
        """Handle order fill notification.

        Args:
            ticker: Filled ticker.
            quantity: Fill quantity (negative for sells).
            price: Fill price.
        """

    def on_bar(  # noqa: B027
        self,
        current_bar: dict[str, pd.Series],
        timestamp: datetime,
    ) -> None:
        """Handle new bar data.

        Called after signals are generated. Can be used for
        state updates, logging, etc.

        Args:
            current_bar: Dict mapping ticker to current bar data.
            timestamp: Current timestamp.
        """

    def get_stop_orders(
        self,
        current_prices: dict[str, float],
        positions: dict[str, int],
    ) -> list[Order]:
        """Generate stop loss orders for current positions.

        Args:
            current_prices: Current prices.
            positions: Current positions.

        Returns:
            List of stop orders.
        """
        orders = []

        for ticker, stop_price in self._state.stop_prices.items():
            qty = positions.get(ticker, 0)
            if qty == 0:
                continue

            price = current_prices.get(ticker)
            if price is None:
                continue

            # Check if stop triggered
            if qty > 0 and price <= stop_price:
                # Long position - stop hit
                orders.append(
                    Order(
                        ticker=ticker,
                        side=OrderSide.SELL,
                        quantity=abs(qty),
                        order_type=OrderType.MARKET,
                    )
                )
            elif qty < 0 and price >= stop_price:
                # Short position - stop hit
                orders.append(
                    Order(
                        ticker=ticker,
                        side=OrderSide.BUY,
                        quantity=abs(qty),
                        order_type=OrderType.MARKET,
                    )
                )

        return orders

    @property
    def state(self) -> StrategyState:
        """Get strategy state."""
        return self._state

    def reset(self) -> None:
        """Reset strategy state."""
        self._state = StrategyState()
        self._is_initialized = False
        self.model_predictions = {}
