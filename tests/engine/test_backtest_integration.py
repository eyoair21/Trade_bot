"""Integration tests for backtesting engine."""

import random
from datetime import date

import numpy as np
import pandas as pd
import pytest

from traderbot.engine.backtest import BacktestEngine, BacktestResult
from traderbot.engine.broker_sim import BrokerSimulator, Order, OrderSide, OrderType
from traderbot.engine.strategy_base import Signal, SignalType, StrategyBase
from traderbot.engine.strategy_momo import MomentumStrategy


class SimpleTestStrategy(StrategyBase):
    """Simple strategy for testing that always goes long on first ticker."""

    def __init__(self, name: str = "simple_test", universe: list[str] | None = None):
        super().__init__(name, universe)
        self._signaled = False

    def generate_signals(
        self,
        current_bar: dict[str, pd.Series],
        historical_data: dict[str, pd.DataFrame],
        timestamp,
    ) -> list[Signal]:
        """Generate a single long signal on first bar."""
        if self._signaled or not self.universe:
            return []

        self._signaled = True
        ticker = self.universe[0]

        return [
            Signal(
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=1.0,
                target_weight=0.05,
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._signaled = False


@pytest.fixture
def simple_data() -> dict[str, pd.DataFrame]:
    """Create simple test data."""
    np.random.seed(42)

    dates = pd.date_range(start="2023-01-03", periods=10, freq="B")
    n = len(dates)

    return {
        "AAPL": pd.DataFrame(
            {
                "date": dates,
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": [100.0 + i * 0.5 for i in range(n)],
                "volume": [1000000] * n,
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "date": dates,
                "open": [200.0] * n,
                "high": [201.0] * n,
                "low": [199.0] * n,
                "close": [200.0 - i * 0.3 for i in range(n)],
                "volume": [1000000] * n,
            }
        ),
    }


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_basic_backtest(self, simple_data: dict[str, pd.DataFrame]) -> None:
        """Test basic backtest runs successfully."""
        np.random.seed(42)
        random.seed(42)

        strategy = SimpleTestStrategy(universe=list(simple_data.keys()))
        engine = BacktestEngine(strategy=strategy, seed=42)

        result = engine.run(
            data=simple_data,
            start_date="2023-01-03",
            end_date="2023-01-16",
        )

        assert isinstance(result, BacktestResult)
        assert result.initial_capital > 0
        assert result.final_equity > 0
        assert result.start_date == date(2023, 1, 3)
        assert result.end_date == date(2023, 1, 16)

    def test_equity_curve_generated(self, simple_data: dict[str, pd.DataFrame]) -> None:
        """Test equity curve is generated."""
        strategy = SimpleTestStrategy(universe=list(simple_data.keys()))
        engine = BacktestEngine(strategy=strategy, seed=42)

        result = engine.run(
            data=simple_data,
            start_date="2023-01-03",
            end_date="2023-01-16",
        )

        assert not result.equity_curve.empty
        assert "date" in result.equity_curve.columns
        assert "equity" in result.equity_curve.columns

    def test_result_to_dict(self, simple_data: dict[str, pd.DataFrame]) -> None:
        """Test result can be serialized to dict."""
        strategy = SimpleTestStrategy(universe=list(simple_data.keys()))
        engine = BacktestEngine(strategy=strategy, seed=42)

        result = engine.run(
            data=simple_data,
            start_date="2023-01-03",
            end_date="2023-01-16",
        )

        result_dict = result.to_dict()

        assert "start_date" in result_dict
        assert "total_return_pct" in result_dict
        assert "sharpe_ratio" in result_dict

    def test_empty_date_range(self, simple_data: dict[str, pd.DataFrame]) -> None:
        """Test handling of empty date range."""
        strategy = SimpleTestStrategy(universe=["AAPL"])
        engine = BacktestEngine(strategy=strategy, seed=42)

        # End before start
        result = engine.run(
            data=simple_data,
            start_date="2023-01-16",
            end_date="2023-01-03",
        )

        assert result.total_trades == 0
        assert result.final_equity == result.initial_capital

    def test_deterministic_results(self, sample_multi_ticker_data: dict[str, pd.DataFrame]) -> None:
        """Test backtest produces deterministic results with same seed."""
        data = {t: df.head(15).copy() for t, df in sample_multi_ticker_data.items()}

        results = []
        for _ in range(3):
            np.random.seed(42)
            random.seed(42)

            strategy = MomentumStrategy(
                name="det_test",
                universe=list(data.keys()),
                seed=42,
            )
            engine = BacktestEngine(strategy=strategy, seed=42)

            result = engine.run(
                data=data,
                start_date="2023-01-03",
                end_date="2023-01-23",
            )
            results.append(result)

        # All results should be identical
        assert results[0].final_equity == results[1].final_equity
        assert results[1].final_equity == results[2].final_equity
        assert results[0].total_trades == results[1].total_trades


class TestBrokerSimulator:
    """Tests for BrokerSimulator."""

    def test_initial_state(self) -> None:
        """Test initial broker state."""
        broker = BrokerSimulator(initial_capital=100000)

        assert broker.cash == 100000
        assert broker.realized_pnl == 0
        assert len(broker.fills) == 0
        assert len(broker.pending_orders) == 0

    def test_submit_order(self) -> None:
        """Test submitting an order."""
        broker = BrokerSimulator()

        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        order_id = broker.submit_order(order)

        assert order_id != ""
        assert len(broker.pending_orders) == 1

    def test_market_order_fill(self) -> None:
        """Test market order gets filled."""
        broker = BrokerSimulator(
            initial_capital=100000,
            commission_bps=10,
            slippage_bps=5,
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

        from datetime import datetime

        fills = broker.process_bar(bar_data, datetime(2023, 1, 3))

        assert len(fills) == 1
        assert fills[0].ticker == "AAPL"
        assert fills[0].quantity == 100
        assert fills[0].price > 100.0  # Slippage added

    def test_position_tracking(self) -> None:
        """Test position is tracked after fill."""
        broker = BrokerSimulator(initial_capital=100000)

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

        from datetime import datetime

        broker.process_bar(bar_data, datetime(2023, 1, 3))

        position = broker.get_position("AAPL")
        assert position is not None
        assert position.quantity == 100

    def test_cash_updated(self) -> None:
        """Test cash is updated after trades."""
        broker = BrokerSimulator(
            initial_capital=100000,
            commission_bps=10,  # 10 bps = 0.1%
            slippage_bps=0,
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

        from datetime import datetime

        broker.process_bar(bar_data, datetime(2023, 1, 3))

        # Cash should be reduced by cost of shares + commission
        # Buy at open = 100.0, notional = 10000, commission = 10000 * 0.001 = 10
        expected_cash = 100000 - (100 * 100.0) - 10.0
        assert broker.cash == pytest.approx(expected_cash, rel=0.01)

    def test_equity_calculation(self) -> None:
        """Test equity calculation."""
        broker = BrokerSimulator(
            initial_capital=100000,
            commission_bps=10,
            slippage_bps=0,
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

        from datetime import datetime

        broker.process_bar(bar_data, datetime(2023, 1, 3))

        # Calculate equity with position at current price
        equity = broker.calculate_equity({"AAPL": 105.0})

        # Cash (~89990) + Position value (100 * 105 = 10500) = ~100490
        # Allow some tolerance for commission
        assert equity == pytest.approx(100490, rel=0.01)

    def test_cancel_all_orders(self) -> None:
        """Test cancelling all orders."""
        broker = BrokerSimulator()

        for _ in range(3):
            order = Order(
                ticker="AAPL",
                side=OrderSide.BUY,
                quantity=100,
            )
            broker.submit_order(order)

        assert len(broker.pending_orders) == 3

        cancelled = broker.cancel_all_orders()

        assert cancelled == 3
        assert len(broker.pending_orders) == 0

    def test_reset(self) -> None:
        """Test broker reset."""
        broker = BrokerSimulator(initial_capital=100000)

        order = Order(ticker="AAPL", side=OrderSide.BUY, quantity=100)
        broker.submit_order(order)

        broker.reset()

        assert broker.cash == 100000
        assert len(broker.pending_orders) == 0
        assert len(broker.fills) == 0
