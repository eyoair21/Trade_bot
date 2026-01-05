"""Simple metrics helpers for run comparison.

These are lightweight metric calculations for comparing backtest runs.
"""

import numpy as np
import pandas as pd


def total_return(equity_series: pd.Series) -> float:
    """Calculate total return percentage.

    Args:
        equity_series: Series of equity values over time.

    Returns:
        Total return as percentage (e.g., 5.0 for 5% gain).
    """
    if len(equity_series) < 2:
        return 0.0

    initial = equity_series.iloc[0]
    final = equity_series.iloc[-1]

    if initial <= 0:
        return 0.0

    return ((final / initial) - 1.0) * 100.0


def sharpe_simple(equity_series: pd.Series) -> float:
    """Calculate simple Sharpe ratio from equity curve.

    Args:
        equity_series: Series of equity values over time.

    Returns:
        Annualized Sharpe ratio (assuming 252 trading days).
    """
    if len(equity_series) < 2:
        return 0.0

    # Calculate daily returns
    returns = equity_series.pct_change().dropna()

    if len(returns) == 0:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0 or np.isnan(std_return):
        return 0.0

    # Annualize (assuming 252 trading days)
    sharpe = (mean_return / std_return) * np.sqrt(252)

    return float(sharpe)


def max_drawdown(equity_series: pd.Series) -> float:
    """Calculate maximum drawdown percentage.

    Args:
        equity_series: Series of equity values over time.

    Returns:
        Maximum drawdown as positive percentage (e.g., 10.0 for 10% drawdown).
    """
    if len(equity_series) < 2:
        return 0.0

    # Calculate running maximum
    running_max = equity_series.cummax()

    # Calculate drawdown from peak
    drawdown = equity_series - running_max

    # Find maximum drawdown
    max_dd = drawdown.min()

    if max_dd >= 0:
        return 0.0

    # Convert to positive percentage
    max_dd_pct = abs((max_dd / running_max[drawdown.idxmin()]) * 100.0)

    return float(max_dd_pct)

