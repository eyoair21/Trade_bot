"""Centralized reward function for profit-aware learning.

Computes rewards tied to realized PnL (net of costs) with configurable penalties
for drawdown, turnover, and risk breaches.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore
    POLARS_AVAILABLE = False


@dataclass
class RewardWeights:
    """Configurable weights for reward components.
    
    Args:
        lambda_dd: Drawdown penalty coefficient (0-1)
        tau_turnover: Turnover penalty coefficient (typically 1e-5 to 1e-2)
        kappa_breach: Risk breach penalty coefficient (0-2)
    """
    lambda_dd: float = 0.2
    tau_turnover: float = 0.001
    kappa_breach: float = 0.5


def compute_step_reward(
    pnl_after_costs: float,
    drawdown_step: float,
    turnover_step: float,
    breach_step: int,
    weights: RewardWeights,
) -> float:
    """Compute single-step reward.
    
    Args:
        pnl_after_costs: Net PnL for this step (after commissions & slippage)
        drawdown_step: Drawdown magnitude for this step (0 if not in DD)
        turnover_step: Dollar turnover for this step
        breach_step: Number of risk breaches in this step (0 or 1)
        weights: Reward weight configuration
    
    Returns:
        Combined reward value (can be negative)
    """
    reward = pnl_after_costs
    reward -= weights.lambda_dd * drawdown_step
    reward -= weights.tau_turnover * turnover_step
    reward -= weights.kappa_breach * breach_step
    return reward


def compute_run_metrics(
    equity_curve: pd.DataFrame | Any,
    trades: pd.DataFrame | Any,
    breaches: pd.DataFrame | Any,
    weights: RewardWeights,
) -> dict[str, float]:
    """Compute aggregate metrics for a full backtest run.
    
    Args:
        equity_curve: DataFrame with columns ['date', 'equity', 'drawdown']
        trades: DataFrame with columns ['pnl_net', 'turnover']
        breaches: DataFrame with breach records
        weights: Reward weight configuration
    
    Returns:
        Dictionary with: total_reward, avg_reward, pnl_net, sharpe, sortino,
        max_dd, turnover, breaches_count
    """
    # Convert Polars to Pandas if needed
    if POLARS_AVAILABLE and isinstance(equity_curve, pl.DataFrame):
        equity_curve = equity_curve.to_pandas()
    if POLARS_AVAILABLE and isinstance(trades, pl.DataFrame):
        trades = trades.to_pandas()
    if POLARS_AVAILABLE and isinstance(breaches, pl.DataFrame):
        breaches = breaches.to_pandas()
    
    # Handle empty data
    if len(equity_curve) == 0:
        return _empty_metrics()
    
    # Total PnL (net of costs)
    pnl_net = trades["pnl_net"].sum() if len(trades) > 0 else 0.0
    
    # Drawdown penalty
    dd_penalty = 0.0
    if "drawdown" in equity_curve.columns:
        dd_penalty = weights.lambda_dd * equity_curve["drawdown"].abs().sum()
    
    # Turnover penalty
    turnover_total = trades["turnover"].sum() if len(trades) > 0 else 0.0
    turnover_penalty = weights.tau_turnover * turnover_total
    
    # Breach penalty
    breaches_count = len(breaches)
    breach_penalty = weights.kappa_breach * breaches_count
    
    # Total reward
    total_reward = pnl_net - dd_penalty - turnover_penalty - breach_penalty
    avg_reward = total_reward / len(equity_curve) if len(equity_curve) > 0 else 0.0
    
    # Risk metrics
    returns = equity_curve["equity"].pct_change().dropna()
    sharpe = _compute_sharpe(returns)
    sortino = _compute_sortino(returns)
    max_dd = equity_curve["drawdown"].min() if "drawdown" in equity_curve.columns else 0.0
    
    return {
        "total_reward": float(total_reward),
        "avg_reward": float(avg_reward),
        "pnl_net": float(pnl_net),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "turnover": float(turnover_total),
        "breaches_count": int(breaches_count),
        "dd_penalty": float(dd_penalty),
        "turnover_penalty": float(turnover_penalty),
        "breach_penalty": float(breach_penalty),
    }


def _compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret == 0 or np.isnan(std_ret):
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(periods_per_year))


def _compute_sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized Sortino ratio (downside deviation)."""
    if len(returns) < 2:
        return 0.0
    mean_ret = returns.mean()
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(mean_ret / downside_std * np.sqrt(periods_per_year))


def _empty_metrics() -> dict[str, float]:
    """Return empty metrics dict for edge cases."""
    return {
        "total_reward": 0.0,
        "avg_reward": 0.0,
        "pnl_net": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_dd": 0.0,
        "turnover": 0.0,
        "breaches_count": 0,
        "dd_penalty": 0.0,
        "turnover_penalty": 0.0,
        "breach_penalty": 0.0,
    }

