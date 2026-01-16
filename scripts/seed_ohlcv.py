"""Seed minimal OHLCV parquet data for local demos.

This script generates ~N trading days of synthetic daily candles for a small
set of symbols and writes them under the configured OHLCV directory.

Environment variables:
    OHLCV_DIR          - Target directory for parquet files (default: ./data/ohlcv)
    SEED_SYMBOLS       - Comma-separated symbols (default: AAPL,MSFT,NVDA,SPY,QQQ)
    SEED_TRADING_DAYS  - Number of trading days to generate (default: 120)
    BAR_INTERVAL       - Bar interval; currently only "1D" is supported.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from traderbot.config import get_config


def _get_env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None and value != "" else default


def _generate_trading_days(n: int) -> pd.DatetimeIndex:
    """Generate the last `n` business days up to today."""
    end = pd.Timestamp(date.today())
    return pd.bdate_range(end=end, periods=n, freq="B")


def _seed_symbol(symbol: str, dates: pd.DatetimeIndex, out_dir: Path) -> None:
    """Generate a simple random-walk price series and save as parquet."""
    np.random.seed(abs(hash(symbol)) % (2**32))

    n = len(dates)
    if n == 0:
        return

    # Random walk in log space with ~1% daily vol
    start_price = float(50 + (abs(hash(symbol)) % 150))  # 50–200 range
    daily_returns = np.random.normal(loc=0.0005, scale=0.01, size=n)
    prices = start_price * np.exp(np.cumsum(daily_returns))

    # Build OHLCV with small random ranges
    # Ensure spread <= 0.2% (0.002) for universe screening
    close = prices
    open_ = close * (1 + np.random.normal(0.0, 0.001, size=n))
    # Cap high/low to ensure (high - low) / close <= 0.002
    spread_max = 0.002
    high = close * (1 + np.random.uniform(0.0, spread_max / 2, size=n))
    low = close * (1 - np.random.uniform(0.0, spread_max / 2, size=n))
    # Ensure high >= open_ >= low
    high = np.maximum(high, open_)
    low = np.minimum(low, open_)
    volume = np.random.randint(500_000, 5_000_000, size=n)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}.parquet"
    df.to_parquet(path, index=False)


def main() -> None:
    cfg = get_config()

    # Resolve OHLCV directory from env (via config)
    ohlcv_dir = cfg.data.ohlcv_dir

    symbols_raw = _get_env("SEED_SYMBOLS", "AAPL,MSFT,NVDA,SPY,QQQ")
    symbols: List[str] = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

    trading_days = int(_get_env("SEED_TRADING_DAYS", "120"))
    interval = _get_env("BAR_INTERVAL", "1D").upper()
    if interval not in {"1D", "1DAY"}:
        raise ValueError(f"BAR_INTERVAL={interval!r} not supported; only '1D' is allowed.")

    dates = _generate_trading_days(trading_days)

    print(f"Seeding OHLCV data in {ohlcv_dir} for symbols: {', '.join(symbols)}")
    print(f"Trading days: {len(dates)} (business days)")

    for symbol in symbols:
        _seed_symbol(symbol, dates, ohlcv_dir)
        print(f"  wrote {symbol}.parquet")

    print("Done.")


if __name__ == "__main__":
    main()


