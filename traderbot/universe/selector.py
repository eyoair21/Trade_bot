"""Dynamic universe selection based on liquidity and volatility filters.

Selects top-N tradeable assets from a larger universe based on:
- Dollar volume (liquidity)
- Price filter (no penny stocks)
- Minimum trading history
- Realized volatility (optional)
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
class UniverseConfig:
    """Configuration for dynamic universe selection.
    
    Args:
        top_n: Number of assets to select
        min_dollar_volume: Minimum average daily dollar volume
        min_price: Minimum price (filter penny stocks)
        min_history_days: Minimum days of trading history
        lookback_days: Days to compute metrics over
        volatility_min: Minimum realized volatility (optional)
        volatility_max: Maximum realized volatility (optional)
    """
    top_n: int = 100
    min_dollar_volume: float = 1_000_000.0
    min_price: float = 5.0
    min_history_days: int = 60
    lookback_days: int = 20
    volatility_min: float | None = None
    volatility_max: float | None = None


def select_universe(
    date: datetime,
    full_universe: list[str],
    price_data: dict[str, pd.DataFrame] | dict[str, Any],
    config: UniverseConfig,
) -> list[str]:
    """Select tradeable universe for a given date.
    
    Args:
        date: Selection date (must use data available up to this date)
        full_universe: List of all candidate tickers
        price_data: Dict mapping ticker -> DataFrame with OHLCV data
        config: Universe selection configuration
    
    Returns:
        List of selected tickers (length <= config.top_n)
    """
    metrics = []
    
    for ticker in full_universe:
        if ticker not in price_data:
            continue
        
        df = price_data[ticker]
        
        # Convert Polars to Pandas if needed
        if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        
        # Filter to date
        df = df[df["date"] <= date].copy()
        
        if len(df) < config.min_history_days:
            continue
        
        # Take lookback window
        window = df.tail(config.lookback_days)
        
        if len(window) == 0:
            continue
        
        # Compute metrics
        last_price = window["close"].iloc[-1]
        if last_price < config.min_price:
            continue
        
        # Dollar volume = avg(close * volume)
        window["dollar_volume"] = window["close"] * window["volume"]
        avg_dollar_volume = window["dollar_volume"].mean()
        
        if avg_dollar_volume < config.min_dollar_volume:
            continue
        
        # Realized volatility (std of log returns)
        returns = np.log(window["close"] / window["close"].shift(1)).dropna()
        realized_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Apply volatility filters if specified
        if config.volatility_min is not None and realized_vol < config.volatility_min:
            continue
        if config.volatility_max is not None and realized_vol > config.volatility_max:
            continue
        
        metrics.append({
            "ticker": ticker,
            "dollar_volume": avg_dollar_volume,
            "price": last_price,
            "volatility": realized_vol,
        })
    
    # Sort by dollar volume (descending) and take top N
    metrics_sorted = sorted(metrics, key=lambda x: x["dollar_volume"], reverse=True)
    selected = [m["ticker"] for m in metrics_sorted[:config.top_n]]
    
    return selected


def save_universe_snapshot(
    date: datetime,
    selected_tickers: list[str],
    output_dir: Path,
) -> None:
    """Save universe snapshot for reproducibility.
    
    Args:
        date: Selection date
        selected_tickers: List of selected tickers
        output_dir: Directory to save snapshot
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = date.strftime("%Y%m%d")
    output_path = output_dir / f"universe_{date_str}.json"
    
    snapshot = {
        "date": date.isoformat(),
        "tickers": selected_tickers,
        "count": len(selected_tickers),
    }
    
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)


def load_universe_snapshot(
    date: datetime,
    output_dir: Path,
) -> list[str] | None:
    """Load universe snapshot for a given date.
    
    Args:
        date: Selection date
        output_dir: Directory containing snapshots
    
    Returns:
        List of tickers or None if not found
    """
    date_str = date.strftime("%Y%m%d")
    output_path = output_dir / f"universe_{date_str}.json"
    
    if not output_path.exists():
        return None
    
    with open(output_path) as f:
        snapshot = json.load(f)
    
    return snapshot["tickers"]

