"""Paper trading engine v1.

Deterministic paper trading simulation with:
- Position sizing by volatility target
- Simulated fills with slippage
- PnL and equity tracking
- No broker API required
"""

from traderbot.paper.broker_sim import (
    BrokerSim,
    Fill,
    Order,
    OrderSide,
    Position,
)
from traderbot.paper.position_sizing import (
    PositionSizer,
    compute_position_size,
)
from traderbot.paper.pnl import (
    EquityPoint,
    compute_equity_curve,
    compute_position_pnl,
)

__all__ = [
    "BrokerSim",
    "Fill",
    "Order",
    "OrderSide",
    "Position",
    "PositionSizer",
    "compute_position_size",
    "EquityPoint",
    "compute_equity_curve",
    "compute_position_pnl",
]
