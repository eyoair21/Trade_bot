#!/usr/bin/env python3
"""Generate sample OHLCV data for backtesting.

Creates synthetic (or real via yfinance) OHLCV data for a set of tickers
and saves them as Parquet files in data/ohlcv/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Default configuration
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2023-03-31"
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "data/ohlcv"


def generate_synthetic_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic price movements.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    # Set seed for this ticker (combine global seed with ticker hash for variety)
    ticker_seed = seed + hash(ticker) % 10000
    rng = np.random.default_rng(ticker_seed)

    # Generate business days
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    # Base prices vary by ticker for realism
    base_prices = {
        "AAPL": 150.0,
        "MSFT": 250.0,
        "NVDA": 180.0,
    }
    base_price = base_prices.get(ticker, 100.0)

    # Generate price series using geometric Brownian motion
    n_days = len(dates)
    daily_return_mean = 0.0005  # Slight upward drift
    daily_return_std = 0.02  # 2% daily volatility

    # Generate log returns
    log_returns = rng.normal(daily_return_mean, daily_return_std, n_days)
    log_returns[0] = 0  # First day starts at base

    # Convert to price series
    close_prices = base_price * np.exp(np.cumsum(log_returns))

    # Generate OHLC from close prices
    intraday_volatility = 0.01  # 1% intraday range
    high_mult = 1 + rng.uniform(0, intraday_volatility * 2, n_days)
    low_mult = 1 - rng.uniform(0, intraday_volatility * 2, n_days)

    high_prices = close_prices * high_mult
    low_prices = close_prices * low_mult

    # Open is previous close with small gap
    open_prices = np.roll(close_prices, 1) * (1 + rng.uniform(-0.005, 0.005, n_days))
    open_prices[0] = base_price

    # Ensure OHLC consistency: low <= open,close <= high
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # Generate volume (realistic ranges)
    base_volume = rng.integers(5_000_000, 20_000_000)
    volume = rng.integers(int(base_volume * 0.5), int(base_volume * 1.5), n_days).astype(np.int64)

    # Create DataFrame with UTC timestamps
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates).tz_localize("UTC"),
            "open": np.round(open_prices, 2),
            "high": np.round(high_prices, 2),
            "low": np.round(low_prices, 2),
            "close": np.round(close_prices, 2),
            "volume": volume,
        }
    )

    return df


def try_yfinance_download(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """Try to download data from yfinance if available.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        DataFrame with OHLCV data, or None if yfinance unavailable.
    """
    try:
        import yfinance as yf

        print(f"  Using yfinance for {ticker}...")
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            return None

        # Rename columns to match our schema
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(data.index).tz_localize("UTC"),
                "open": data["Open"].values,
                "high": data["High"].values,
                "low": data["Low"].values,
                "close": data["Close"].values,
                "volume": data["Volume"].values.astype(np.int64),
            }
        )

        return df

    except ImportError:
        return None
    except Exception as e:
        print(f"  yfinance error for {ticker}: {e}")
        return None


def create_sample_data(
    output_dir: str | Path,
    tickers: list[str] | None = None,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    seed: int = DEFAULT_SEED,
    use_yfinance: bool = True,
) -> dict[str, Path]:
    """Create sample OHLCV data files.

    Args:
        output_dir: Directory to save Parquet files.
        tickers: List of ticker symbols.
        start_date: Start date for data.
        end_date: End date for data.
        seed: Random seed for synthetic data.
        use_yfinance: Whether to try yfinance first.

    Returns:
        Dict mapping ticker to file path.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    created_files: dict[str, Path] = {}

    for ticker in tickers:
        print(f"Generating data for {ticker}...")

        df = None

        # Try yfinance first if enabled
        if use_yfinance:
            df = try_yfinance_download(ticker, start_date, end_date)

        # Fall back to synthetic
        if df is None:
            print(f"  Using synthetic data for {ticker}...")
            df = generate_synthetic_ohlcv(ticker, start_date, end_date, seed)

        # Save to Parquet
        file_path = output_path / f"{ticker}.parquet"
        df.to_parquet(file_path, index=False, engine="pyarrow")

        created_files[ticker] = file_path
        print(f"  Saved {len(df)} rows to {file_path}")

    return created_files


def main(args: list[str] | None = None) -> int:
    """Main entry point for the script.

    Args:
        args: Command line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="Generate sample OHLCV data for backtesting")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for Parquet files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tickers",
        "-t",
        nargs="+",
        default=DEFAULT_TICKERS,
        help=f"Ticker symbols to generate (default: {DEFAULT_TICKERS})",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for synthetic data (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Use synthetic data only (don't try yfinance)",
    )

    parsed = parser.parse_args(args)

    print(f"Creating sample data in {parsed.output_dir}/")
    print(f"  Tickers: {parsed.tickers}")
    print(f"  Date range: {parsed.start_date} to {parsed.end_date}")
    print(f"  Seed: {parsed.seed}")
    print()

    created = create_sample_data(
        output_dir=parsed.output_dir,
        tickers=parsed.tickers,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        seed=parsed.seed,
        use_yfinance=not parsed.synthetic_only,
    )

    print()
    print(f"Created {len(created)} Parquet files:")
    for ticker, path in created.items():
        print(f"  {ticker}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
