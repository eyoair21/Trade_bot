"""PnL and equity curve computation for paper trading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from traderbot.logging_setup import get_logger

logger = get_logger("paper.pnl")


@dataclass
class EquityPoint:
    """Single point in equity curve."""

    timestamp: str  # ISO format
    equity: float
    cash: float
    position_value: float
    unrealized_pnl: float
    realized_pnl: float
    num_positions: int


@dataclass
class PositionPnL:
    """PnL breakdown for a position."""

    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float


def compute_position_pnl(
    symbol: str,
    quantity: int,
    avg_cost: float,
    current_price: float,
    realized_pnl: float = 0.0,
) -> PositionPnL:
    """Compute PnL for a single position.

    Args:
        symbol: Stock symbol.
        quantity: Position quantity (positive for long).
        avg_cost: Average cost per share.
        current_price: Current market price.
        realized_pnl: Already realized PnL from partial closes.

    Returns:
        PositionPnL with computed values.
    """
    unrealized = (current_price - avg_cost) * quantity

    if avg_cost > 0 and quantity != 0:
        unrealized_pct = (current_price - avg_cost) / avg_cost
    else:
        unrealized_pct = 0.0

    return PositionPnL(
        symbol=symbol,
        quantity=quantity,
        avg_cost=avg_cost,
        current_price=current_price,
        unrealized_pnl=unrealized,
        unrealized_pnl_pct=unrealized_pct,
        realized_pnl=realized_pnl,
    )


def compute_equity_point(
    positions: Dict[str, "Position"],
    prices: Dict[str, float],
    cash: float,
    realized_pnl: float = 0.0,
) -> EquityPoint:
    """Compute current equity snapshot.

    Args:
        positions: Dictionary of symbol to Position.
        prices: Dictionary of symbol to current price.
        cash: Current cash balance.
        realized_pnl: Total realized PnL.

    Returns:
        EquityPoint snapshot.
    """
    from traderbot.paper.broker_sim import Position

    position_value = 0.0
    unrealized_pnl = 0.0

    for symbol, pos in positions.items():
        price = prices.get(symbol, pos.current_price)
        pos_value = pos.quantity * price
        position_value += pos_value
        unrealized_pnl += (price - pos.avg_cost) * pos.quantity

    equity = cash + position_value

    return EquityPoint(
        timestamp=datetime.now(timezone.utc).isoformat(),
        equity=equity,
        cash=cash,
        position_value=position_value,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        num_positions=len(positions),
    )


def compute_equity_curve(
    equity_points: List[EquityPoint],
) -> pd.DataFrame:
    """Convert equity points to DataFrame.

    Args:
        equity_points: List of equity snapshots.

    Returns:
        DataFrame with equity curve data.
    """
    if not equity_points:
        return pd.DataFrame()

    rows = []
    for point in equity_points:
        rows.append({
            "timestamp": point.timestamp,
            "equity": point.equity,
            "cash": point.cash,
            "position_value": point.position_value,
            "unrealized_pnl": point.unrealized_pnl,
            "realized_pnl": point.realized_pnl,
            "num_positions": point.num_positions,
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def save_equity_csv(
    equity_points: List[EquityPoint],
    path: Path,
) -> None:
    """Save equity curve to CSV.

    Args:
        equity_points: List of equity snapshots.
        path: Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "equity", "cash", "position_value",
            "unrealized_pnl", "realized_pnl", "num_positions",
        ])
        for point in equity_points:
            writer.writerow([
                point.timestamp,
                round(point.equity, 2),
                round(point.cash, 2),
                round(point.position_value, 2),
                round(point.unrealized_pnl, 2),
                round(point.realized_pnl, 2),
                point.num_positions,
            ])

    logger.info("Saved equity curve to %s (%d points)", path, len(equity_points))


def save_fills_csv(
    fills: List["Fill"],
    path: Path,
) -> None:
    """Save fills to CSV.

    Args:
        fills: List of Fill records.
        path: Output CSV path.
    """
    from traderbot.paper.broker_sim import Fill

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "order_id", "symbol", "side", "quantity",
            "expected_price", "filled_price", "slippage_bps",
            "cost_bps", "total_cost",
        ])
        for fill in fills:
            writer.writerow([
                fill.timestamp,
                fill.order_id,
                fill.symbol,
                fill.side.value,
                fill.quantity,
                round(fill.expected_price, 4),
                round(fill.filled_price, 4),
                fill.slippage_bps,
                fill.cost_bps,
                round(fill.total_cost, 4),
            ])

    logger.info("Saved fills to %s (%d fills)", path, len(fills))


def load_equity_csv(path: Path) -> List[EquityPoint]:
    """Load equity curve from CSV.

    Args:
        path: CSV file path.

    Returns:
        List of EquityPoint records.
    """
    if not path.exists():
        return []

    points: List[EquityPoint] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(EquityPoint(
                timestamp=row["timestamp"],
                equity=float(row["equity"]),
                cash=float(row["cash"]),
                position_value=float(row["position_value"]),
                unrealized_pnl=float(row["unrealized_pnl"]),
                realized_pnl=float(row["realized_pnl"]),
                num_positions=int(row["num_positions"]),
            ))

    return points
