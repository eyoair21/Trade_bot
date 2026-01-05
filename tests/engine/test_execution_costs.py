"""Tests for execution cost tracking in broker simulator."""

from datetime import datetime

import pandas as pd
import pytest

from traderbot.engine.broker_sim import BrokerSimulator, Order, OrderSide, OrderType


class TestExecutionCosts:
    """Tests for execution cost tracking."""

    def test_fee_per_share_applied(self) -> None:
        """Test per-share fees are applied correctly."""
        broker = BrokerSimulator(
            initial_capital=100000,
            commission_bps=0,
            slippage_bps=0,
            fee_per_share=0.01,  # $0.01 per share
        )

        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        broker.submit_order(order)

        bar_data = {
            "AAPL": pd.Series(
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000}
            )
        }

        broker.process_bar(bar_data, datetime(2023, 1, 3))

        # Fee should be 100 shares * $0.01 = $1.00
        assert broker.total_fees == pytest.approx(1.0, rel=0.01)

    def test_slippage_tracked(self) -> None:
        """Test slippage is tracked correctly."""
        broker = BrokerSimulator(
            initial_capital=100000,
            commission_bps=0,
            slippage_bps=10,  # 10 bps = 0.1%
            fee_per_share=0,
        )

        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        broker.submit_order(order)

        bar_data = {
            "AAPL": pd.Series(
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000}
            )
        }

        broker.process_bar(bar_data, datetime(2023, 1, 3))

        # Slippage should be 10 bps of notional (100 * 100 * 0.001)
        expected_slippage = 100 * 100.0 * 0.001
        assert broker.total_slippage == pytest.approx(expected_slippage, rel=0.01)

    def test_commission_tracked(self) -> None:
        """Test commission is tracked correctly."""
        broker = BrokerSimulator(
            initial_capital=100000,
            commission_bps=10,  # 10 bps
            slippage_bps=0,
            fee_per_share=0,
        )

        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        broker.submit_order(order)

        bar_data = {
            "AAPL": pd.Series(
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000}
            )
        }

        broker.process_bar(bar_data, datetime(2023, 1, 3))

        # Commission should be 10 bps of notional
        expected_commission = 100 * 100.0 * 0.001
        assert broker.total_commission == pytest.approx(expected_commission, rel=0.01)

    def test_total_execution_costs(self) -> None:
        """Test total execution costs aggregation."""
        broker = BrokerSimulator(
            initial_capital=100000,
            commission_bps=10,  # 10 bps
            slippage_bps=5,  # 5 bps
            fee_per_share=0.005,  # $0.005 per share
        )

        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        broker.submit_order(order)

        bar_data = {
            "AAPL": pd.Series(
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000}
            )
        }

        broker.process_bar(bar_data, datetime(2023, 1, 3))

        total = broker.total_execution_costs
        expected = broker.total_commission + broker.total_fees + broker.total_slippage

        assert total == pytest.approx(expected, rel=0.01)
        assert total > 0

    def test_costs_reset(self) -> None:
        """Test costs are reset on broker reset."""
        broker = BrokerSimulator(
            initial_capital=100000,
            commission_bps=10,
            slippage_bps=5,
            fee_per_share=0.01,
        )

        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        broker.submit_order(order)

        bar_data = {
            "AAPL": pd.Series(
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000}
            )
        }

        broker.process_bar(bar_data, datetime(2023, 1, 3))

        # Verify costs were tracked
        assert broker.total_execution_costs > 0

        # Reset
        broker.reset()

        # Verify costs are cleared
        assert broker.total_commission == 0.0
        assert broker.total_fees == 0.0
        assert broker.total_slippage == 0.0
        assert broker.total_execution_costs == 0.0

    def test_costs_accumulate_multiple_trades(self) -> None:
        """Test costs accumulate across multiple trades."""
        broker = BrokerSimulator(
            initial_capital=100000,
            commission_bps=10,
            slippage_bps=0,
            fee_per_share=0.01,
        )

        bar_data = {
            "AAPL": pd.Series(
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000}
            )
        }

        # First trade
        order1 = Order(ticker="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(order1)
        broker.process_bar(bar_data, datetime(2023, 1, 3))

        first_costs = broker.total_execution_costs

        # Second trade
        order2 = Order(ticker="AAPL", side=OrderSide.SELL, quantity=50)
        broker.submit_order(order2)
        broker.process_bar(bar_data, datetime(2023, 1, 4))

        second_costs = broker.total_execution_costs

        assert second_costs > first_costs

    def test_sell_order_slippage_negative(self) -> None:
        """Test sell orders have negative slippage (worse fill for seller)."""
        broker = BrokerSimulator(
            initial_capital=100000,
            commission_bps=0,
            slippage_bps=10,
            fee_per_share=0,
        )

        # First buy some shares
        buy_order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(buy_order)

        bar_data = {
            "AAPL": pd.Series(
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000}
            )
        }
        broker.process_bar(bar_data, datetime(2023, 1, 3))

        buy_fill = broker.fills[-1]

        # Now sell
        sell_order = Order(ticker="AAPL", side=OrderSide.SELL, quantity=100)
        broker.submit_order(sell_order)
        broker.process_bar(bar_data, datetime(2023, 1, 4))

        sell_fill = broker.fills[-1]

        # Buy should be above open, sell should be below open
        assert buy_fill.price > 100.0
        assert sell_fill.price < 100.0
