"""Tests for paper trading broker simulation."""

import pytest
import tempfile
from pathlib import Path

from traderbot.paper.broker_sim import BrokerSim, OrderSide, Fill, Position


class TestBrokerSim:
    """Tests for BrokerSim class."""

    def test_initial_state(self):
        """Test broker initializes with correct state."""
        broker = BrokerSim(initial_capital=100_000.0)

        assert broker.state.equity == 100_000.0
        assert broker.state.cash == 100_000.0
        assert len(broker.state.positions) == 0
        assert len(broker.state.fills) == 0

    def test_buy_order_updates_cash(self):
        """Test buy order reduces cash."""
        broker = BrokerSim(initial_capital=100_000.0, slippage_bps=10.0, cost_bps=2.0)

        fill = broker.submit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            reference_price=150.0,
        )

        # Cash should decrease by notional + costs
        assert broker.state.cash < 100_000.0
        assert fill.filled_price > 150.0  # Slippage on buy

    def test_buy_creates_position(self):
        """Test buy order creates position."""
        broker = BrokerSim(initial_capital=100_000.0)

        broker.submit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            reference_price=150.0,
        )

        assert "AAPL" in broker.state.positions
        assert broker.state.positions["AAPL"].quantity == 100

    def test_sell_removes_position(self):
        """Test sell order removes position when closing."""
        broker = BrokerSim(initial_capital=100_000.0)

        # Buy
        broker.submit_order("AAPL", OrderSide.BUY, 100, 150.0)

        # Sell same quantity
        broker.submit_order("AAPL", OrderSide.SELL, 100, 155.0)

        assert "AAPL" not in broker.state.positions

    def test_slippage_applied_correctly(self):
        """Test slippage is applied in correct direction."""
        broker = BrokerSim(slippage_bps=100.0)  # 1%

        buy_fill = broker.submit_order("AAPL", OrderSide.BUY, 100, 100.0)
        assert buy_fill.filled_price == 101.0  # 1% higher

        # Reset for sell test
        broker2 = BrokerSim(slippage_bps=100.0)
        broker2.submit_order("AAPL", OrderSide.BUY, 100, 100.0)
        sell_fill = broker2.submit_order("AAPL", OrderSide.SELL, 100, 100.0)
        assert sell_fill.filled_price == 99.0  # 1% lower

    def test_cost_calculation(self):
        """Test transaction costs are calculated correctly."""
        broker = BrokerSim(slippage_bps=0.0, cost_bps=10.0)  # 0.1%

        fill = broker.submit_order("AAPL", OrderSide.BUY, 100, 100.0)

        # Notional = 100 * 100 = 10,000
        # Cost = 10,000 * 0.001 = 10
        assert fill.total_cost == 10.0

    def test_max_positions_check(self):
        """Test max positions limit."""
        broker = BrokerSim(max_positions=2)

        broker.submit_order("AAPL", OrderSide.BUY, 10, 100.0)
        assert broker.can_open_position()

        broker.submit_order("MSFT", OrderSide.BUY, 10, 100.0)
        assert not broker.can_open_position()

    def test_update_prices(self):
        """Test price updates affect equity."""
        broker = BrokerSim(initial_capital=100_000.0, slippage_bps=0.0, cost_bps=0.0)

        broker.submit_order("AAPL", OrderSide.BUY, 100, 100.0)
        initial_equity = broker.state.equity

        # Price increases
        broker.update_prices({"AAPL": 110.0})

        # Equity should increase by 100 * 10 = 1000
        assert broker.state.equity > initial_equity
        assert broker.state.positions["AAPL"].unrealized_pnl == 1000.0

    def test_save_and_load(self):
        """Test state serialization."""
        broker = BrokerSim(initial_capital=50_000.0, slippage_bps=5.0)
        broker.submit_order("AAPL", OrderSide.BUY, 50, 150.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            broker.save(path)

            loaded = BrokerSim.load(path)

            assert loaded.state.equity == broker.state.equity
            assert loaded.initial_capital == broker.initial_capital
            assert "AAPL" in loaded.state.positions

    def test_fills_recorded(self):
        """Test all fills are recorded."""
        broker = BrokerSim()

        broker.submit_order("AAPL", OrderSide.BUY, 100, 150.0)
        broker.submit_order("MSFT", OrderSide.BUY, 50, 300.0)

        fills = broker.get_fills()
        assert len(fills) == 2
        assert fills[0].symbol == "AAPL"
        assert fills[1].symbol == "MSFT"


class TestFillDetails:
    """Tests for Fill record details."""

    def test_fill_contains_expected_price(self):
        """Test fill records expected vs filled price."""
        broker = BrokerSim(slippage_bps=50.0)

        fill = broker.submit_order("AAPL", OrderSide.BUY, 100, 100.0)

        assert fill.expected_price == 100.0
        assert abs(fill.filled_price - 100.5) < 0.0001  # 50 bps slippage
        assert fill.slippage_bps == 50.0

    def test_fill_has_timestamp(self):
        """Test fill has timestamp."""
        broker = BrokerSim()

        fill = broker.submit_order("AAPL", OrderSide.BUY, 100, 100.0)

        assert fill.timestamp is not None
        assert "T" in fill.timestamp  # ISO format

    def test_fill_has_order_id(self):
        """Test fills have unique order IDs."""
        broker = BrokerSim()

        fill1 = broker.submit_order("AAPL", OrderSide.BUY, 100, 100.0)
        fill2 = broker.submit_order("MSFT", OrderSide.BUY, 50, 200.0)

        assert fill1.order_id != fill2.order_id
        assert fill1.order_id.startswith("ORD-")
