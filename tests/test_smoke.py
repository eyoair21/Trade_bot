"""Smoke tests to verify basic functionality."""

import random

import numpy as np
import pandas as pd

from traderbot import __version__
from traderbot.config import get_config
from traderbot.data.calendar import is_session
from traderbot.engine.backtest import BacktestEngine
from traderbot.engine.broker_sim import BrokerSimulator
from traderbot.engine.risk import RiskManager
from traderbot.engine.strategy_momo import MomentumStrategy
from traderbot.features.ta import calculate_ema, calculate_rsi


class TestSmokeTests:
    """Basic smoke tests for all major components."""

    def test_version(self) -> None:
        """Test version is defined."""
        assert __version__ == "0.6.3"

    def test_config_loads(self) -> None:
        """Test configuration loads successfully."""
        config = get_config()
        assert config is not None
        assert config.backtest.initial_capital > 0
        assert config.risk.max_position_pct > 0

    def test_calendar_basic(self) -> None:
        """Test calendar basic functions."""
        from datetime import date

        # Weekday should be session
        assert is_session(date(2023, 1, 3))  # Tuesday

        # Weekend should not be session
        assert not is_session(date(2023, 1, 1))  # Sunday

    def test_ema_calculation(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test EMA calculation."""
        ema = calculate_ema(sample_ohlcv_df["close"], period=12)
        assert len(ema) == len(sample_ohlcv_df)
        assert not ema.isna().all()

    def test_rsi_calculation(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test RSI calculation."""
        rsi = calculate_rsi(sample_ohlcv_df["close"], period=14)
        assert len(rsi) == len(sample_ohlcv_df)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_broker_simulator_init(self) -> None:
        """Test broker simulator initialization."""
        broker = BrokerSimulator(initial_capital=100000)
        assert broker.cash == 100000
        assert broker.realized_pnl == 0

    def test_risk_manager_init(self) -> None:
        """Test risk manager initialization."""
        risk_mgr = RiskManager()
        assert not risk_mgr.circuit_breaker_active

    def test_momentum_strategy_init(self) -> None:
        """Test momentum strategy initialization."""
        strategy = MomentumStrategy(
            name="test_momo",
            universe=["AAPL", "MSFT"],
        )
        assert strategy.name == "test_momo"
        assert len(strategy.universe) == 2

    def test_backtest_engine_init(self) -> None:
        """Test backtest engine initialization."""
        strategy = MomentumStrategy(name="test")
        engine = BacktestEngine(strategy=strategy)
        assert engine.strategy is not None
        assert engine.broker is not None
        assert engine.risk_manager is not None


class TestEndToEndSmoke:
    """End-to-end smoke test with minimal data."""

    def test_minimal_backtest(
        self, sample_multi_ticker_data: dict[str, pd.DataFrame], seed: int
    ) -> None:
        """Test minimal backtest runs without errors."""
        np.random.seed(seed)
        random.seed(seed)

        # Use only first 10 bars for quick test
        data = {ticker: df.head(10).copy() for ticker, df in sample_multi_ticker_data.items()}

        strategy = MomentumStrategy(
            name="smoke_test",
            universe=list(data.keys()),
            seed=seed,
        )

        engine = BacktestEngine(strategy=strategy, seed=seed)

        result = engine.run(
            data=data,
            start_date="2023-01-03",
            end_date="2023-01-17",
        )

        # Basic assertions
        assert result is not None
        assert result.initial_capital > 0
        assert result.final_equity > 0
        assert isinstance(result.equity_curve, pd.DataFrame)

    def test_backtest_determinism(self, sample_multi_ticker_data: dict[str, pd.DataFrame]) -> None:
        """Test backtest produces deterministic results."""
        data = {ticker: df.head(15).copy() for ticker, df in sample_multi_ticker_data.items()}

        results = []
        for _ in range(2):
            np.random.seed(42)
            random.seed(42)

            strategy = MomentumStrategy(
                name="determinism_test",
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

        # Results should be identical
        assert results[0].final_equity == results[1].final_equity
        assert results[0].total_trades == results[1].total_trades
