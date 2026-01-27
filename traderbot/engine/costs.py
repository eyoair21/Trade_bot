"""Realistic transaction cost modeling.

Estimates slippage, market impact, and fees for order execution.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CostComponents:
    """Breakdown of transaction costs.
    
    Args:
        slippage: Slippage cost (positive)
        impact: Market impact cost (positive)
        fees: Commission and exchange fees (positive)
        total: Total cost
    """
    slippage: float
    impact: float
    fees: float
    total: float


def estimate_costs(
    quantity: float,
    price: float,
    spread_bps: float = 5.0,
    adv: float = 1_000_000.0,
    volatility: float = 0.20,
    participation_rate: float = 0.10,
    fee_per_share: float = 0.0005,
) -> CostComponents:
    """Estimate total transaction costs for an order.
    
    Args:
        quantity: Order size in shares (absolute value)
        price: Current price per share
        spread_bps: Bid-ask spread in basis points
        adv: Average daily volume in shares
        volatility: Daily volatility (e.g., 0.20 = 20%)
        participation_rate: Fraction of ADV this order represents
        fee_per_share: Commission per share
    
    Returns:
        CostComponents with breakdown
    """
    notional = abs(quantity) * price
    
    # 1. Slippage cost (half-spread)
    slippage = notional * (spread_bps / 10000) * 0.5
    
    # 2. Market impact (square-root model)
    # Impact ~ volatility * sqrt(quantity / adv)
    if adv > 0:
        participation = abs(quantity) / adv
        # Almgren-Chriss style: impact ~ sigma * sqrt(participation)
        impact_bps = volatility * np.sqrt(participation) * 10000  # Convert to bps
        impact = notional * (impact_bps / 10000)
    else:
        impact = 0.0
    
    # 3. Fees
    fees = abs(quantity) * fee_per_share
    
    # Total
    total = slippage + impact + fees
    
    return CostComponents(
        slippage=slippage,
        impact=impact,
        fees=fees,
        total=total,
    )


def estimate_costs_simple(
    quantity: float,
    price: float,
    cost_bps: float = 10.0,
) -> float:
    """Simple cost estimate as fixed basis points.
    
    Args:
        quantity: Order size in shares
        price: Price per share
        cost_bps: Total cost in basis points
    
    Returns:
        Total cost in dollars
    """
    notional = abs(quantity) * price
    return notional * (cost_bps / 10000)


class CostModel:
    """Transaction cost model with configurable parameters."""
    
    def __init__(
        self,
        spread_bps: float = 5.0,
        fee_per_share: float = 0.0005,
        impact_coef: float = 0.1,
    ):
        """Initialize cost model.
        
        Args:
            spread_bps: Bid-ask spread in bps
            fee_per_share: Commission per share
            impact_coef: Market impact coefficient
        """
        self.spread_bps = spread_bps
        self.fee_per_share = fee_per_share
        self.impact_coef = impact_coef
        
        # Track realized costs
        self.total_costs = 0.0
        self.cost_history: list[dict] = []
    
    def estimate(
        self,
        ticker: str,
        quantity: float,
        price: float,
        adv: float = 1_000_000.0,
        volatility: float = 0.20,
    ) -> CostComponents:
        """Estimate costs for an order.
        
        Args:
            ticker: Asset ticker
            quantity: Order size
            price: Current price
            adv: Average daily volume
            volatility: Daily volatility
        
        Returns:
            CostComponents
        """
        costs = estimate_costs(
            quantity=quantity,
            price=price,
            spread_bps=self.spread_bps,
            adv=adv,
            volatility=volatility,
            fee_per_share=self.fee_per_share,
        )
        
        # Apply impact coefficient adjustment
        costs.impact *= self.impact_coef
        costs.total = costs.slippage + costs.impact + costs.fees
        
        # Log
        self.cost_history.append({
            "ticker": ticker,
            "quantity": quantity,
            "notional": abs(quantity) * price,
            "slippage": costs.slippage,
            "impact": costs.impact,
            "fees": costs.fees,
            "total": costs.total,
        })
        
        self.total_costs += costs.total
        
        return costs
    
    def get_summary(self) -> dict:
        """Get cost summary statistics.
        
        Returns:
            Dictionary with aggregate cost metrics
        """
        if not self.cost_history:
            return {
                "total_costs": 0.0,
                "avg_cost_per_trade": 0.0,
                "cost_breakdown": {},
            }
        
        n_trades = len(self.cost_history)
        total_slippage = sum(h["slippage"] for h in self.cost_history)
        total_impact = sum(h["impact"] for h in self.cost_history)
        total_fees = sum(h["fees"] for h in self.cost_history)
        
        return {
            "total_costs": self.total_costs,
            "avg_cost_per_trade": self.total_costs / n_trades,
            "n_trades": n_trades,
            "cost_breakdown": {
                "slippage": total_slippage,
                "impact": total_impact,
                "fees": total_fees,
            },
            "cost_breakdown_pct": {
                "slippage": total_slippage / self.total_costs if self.total_costs > 0 else 0,
                "impact": total_impact / self.total_costs if self.total_costs > 0 else 0,
                "fees": total_fees / self.total_costs if self.total_costs > 0 else 0,
            },
        }

