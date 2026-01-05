"""Pytest configuration and fixtures."""

import random
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traderbot.config import reset_config
from traderbot.logging_setup import reset_logging


@pytest.fixture(autouse=True)
def reset_globals() -> Generator[None, None, None]:
    """Reset global state before each test."""
    reset_config()
    reset_logging()
    yield
    reset_config()
    reset_logging()


@pytest.fixture
def seed() -> int:
    """Provide deterministic seed."""
    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    return seed_value


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)

    dates = pd.date_range(start="2023-01-03", periods=30, freq="B")
    n = len(dates)

    # Generate price data
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))

    return pd.DataFrame(
        {
            "date": dates,
            "open": close * (1 + np.random.randn(n) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "close": close,
            "volume": np.random.randint(1000000, 10000000, n),
        }
    )


@pytest.fixture
def sample_multi_ticker_data() -> dict[str, pd.DataFrame]:
    """Create sample data for multiple tickers."""
    np.random.seed(42)

    dates = pd.date_range(start="2023-01-03", periods=30, freq="B")
    n = len(dates)

    data = {}
    for ticker in ["AAPL", "MSFT", "NVDA"]:
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        data[ticker] = pd.DataFrame(
            {
                "date": dates,
                "open": close * (1 + np.random.randn(n) * 0.005),
                "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
                "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
                "close": close,
                "volume": np.random.randint(1000000, 10000000, n),
            }
        )

    return data


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory."""
    data_dir = tmp_path / "data" / "ohlcv"
    data_dir.mkdir(parents=True)
    return data_dir
