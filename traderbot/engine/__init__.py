"""Trading engine modules.

Core backtesting, risk management, and strategy execution.
"""

from traderbot.engine.backtest import BacktestEngine, BacktestResult
from traderbot.engine.broker_sim import BrokerSimulator, Fill, Order, Position
from traderbot.engine.position_sizing import (
    PositionSizer,
    SizeResult,
    SizerType,
    calculate_volatility,
    fixed_fraction,
    kelly_fraction,
    vol_target,
)
from traderbot.engine.risk import RiskCheckResult, RiskLimits, RiskManager
from traderbot.engine.strategy_base import Signal, StrategyBase

__all__ = [
    "RiskManager",
    "RiskLimits",
    "RiskCheckResult",
    "BrokerSimulator",
    "Order",
    "Fill",
    "Position",
    "StrategyBase",
    "Signal",
    "BacktestEngine",
    "BacktestResult",
    "PositionSizer",
    "SizerType",
    "SizeResult",
    "fixed_fraction",
    "vol_target",
    "kelly_fraction",
    "calculate_volatility",
]
