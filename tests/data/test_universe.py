"""Tests for dynamic universe selection."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traderbot.data.universe import (
    calculate_dollar_volume,
    calculate_realized_volatility,
    get_available_tickers,
    load_universe_selection,
    select_universe,
    write_universe_selection,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    n = 30
    np.random.seed(42)

    # Generate realistic price series
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    volume = np.random.randint(1_000_000, 10_000_000, n)

    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "open": close * (1 + np.random.randn(n) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def sample_data_dir(tmp_path: Path, sample_ohlcv: pd.DataFrame) -> Path:
    """Create sample data directory with parquet files."""
    data_dir = tmp_path / "ohlcv"
    data_dir.mkdir()

    # Create parquet files for multiple tickers
    tickers = ["AAPL", "MSFT", "NVDA"]
    for ticker in tickers:
        # Vary the data slightly for each ticker
        df = sample_ohlcv.copy()
        df["close"] = df["close"] * (1 + 0.1 * hash(ticker) % 10 / 10)
        df["volume"] = df["volume"] * (1 + 0.1 * hash(ticker) % 10 / 10)
        df.to_parquet(data_dir / f"{ticker}.parquet", index=False)

    return data_dir


class TestCalculateDollarVolume:
    """Tests for dollar volume calculation."""

    def test_basic_calculation(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test basic dollar volume calculation."""
        dv = calculate_dollar_volume(sample_ohlcv, lookback=20)

        assert dv > 0
        assert isinstance(dv, float)

    def test_lookback_respected(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test lookback period is respected."""
        dv_20 = calculate_dollar_volume(sample_ohlcv, lookback=20)
        dv_5 = calculate_dollar_volume(sample_ohlcv, lookback=5)

        # Different lookbacks should generally give different results
        # (unless the data happens to average the same)
        assert dv_20 != dv_5 or len(sample_ohlcv) < 20

    def test_short_data(self) -> None:
        """Test handling of short data."""
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0],
                "volume": [1000000, 1000000],
            }
        )
        dv = calculate_dollar_volume(df, lookback=20)

        # Should still calculate with available data
        assert dv > 0

    def test_empty_data(self) -> None:
        """Test empty data returns 0."""
        df = pd.DataFrame({"close": [], "volume": []})
        dv = calculate_dollar_volume(df, lookback=20)

        assert dv == 0.0


class TestCalculateRealizedVolatility:
    """Tests for realized volatility calculation."""

    def test_basic_calculation(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test basic volatility calculation."""
        vol = calculate_realized_volatility(sample_ohlcv, lookback=20)

        assert vol >= 0
        assert isinstance(vol, float)

    def test_annualized(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test volatility is annualized."""
        vol = calculate_realized_volatility(sample_ohlcv, lookback=20)

        # Annualized vol should typically be between 0.1 and 1.0 for stocks
        # (10% to 100% annualized)
        assert 0.0 <= vol <= 2.0  # Allow some range for synthetic data

    def test_short_data(self) -> None:
        """Test handling of short data."""
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
            }
        )
        vol = calculate_realized_volatility(df, lookback=20)

        assert vol == 0.0  # Not enough data

    def test_flat_prices(self) -> None:
        """Test flat prices return 0 volatility."""
        df = pd.DataFrame({"close": [100.0] * 25})
        vol = calculate_realized_volatility(df, lookback=20)

        assert vol == 0.0


class TestGetAvailableTickers:
    """Tests for getting available tickers."""

    def test_returns_tickers(self, sample_data_dir: Path) -> None:
        """Test returns list of available tickers."""
        tickers = get_available_tickers(sample_data_dir)

        assert len(tickers) == 3
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "NVDA" in tickers

    def test_sorted(self, sample_data_dir: Path) -> None:
        """Test tickers are sorted."""
        tickers = get_available_tickers(sample_data_dir)

        assert tickers == sorted(tickers)

    def test_empty_dir(self, tmp_path: Path) -> None:
        """Test empty directory returns empty list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        tickers = get_available_tickers(empty_dir)

        assert tickers == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test nonexistent directory returns empty list."""
        tickers = get_available_tickers(tmp_path / "nonexistent")

        assert tickers == []


class TestSelectUniverse:
    """Tests for dynamic universe selection."""

    def test_basic_selection(self, sample_data_dir: Path) -> None:
        """Test basic universe selection."""
        universe = select_universe(
            data_root=sample_data_dir,
            day=date(2023, 1, 25),
            max_n=10,
        )

        assert isinstance(universe, list)
        assert len(universe) <= 10

    def test_max_n_respected(self, sample_data_dir: Path) -> None:
        """Test max_n limit is respected."""
        universe = select_universe(
            data_root=sample_data_dir,
            day=date(2023, 1, 25),
            max_n=2,
        )

        assert len(universe) <= 2

    def test_string_date(self, sample_data_dir: Path) -> None:
        """Test string date is accepted."""
        universe = select_universe(
            data_root=sample_data_dir,
            day="2023-01-25",
            max_n=10,
        )

        assert isinstance(universe, list)

    def test_candidate_tickers(self, sample_data_dir: Path) -> None:
        """Test candidate tickers filtering."""
        universe = select_universe(
            data_root=sample_data_dir,
            day=date(2023, 1, 25),
            max_n=10,
            candidate_tickers=["AAPL", "MSFT"],
        )

        # Should only include candidates
        for ticker in universe:
            assert ticker in ["AAPL", "MSFT"]


class TestWriteUniverseSelection:
    """Tests for writing universe selection."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        """Test write creates JSON file."""
        tickers = ["AAPL", "MSFT", "NVDA"]
        day = date(2023, 1, 15)

        path = write_universe_selection(tickers, day, tmp_path)

        assert path.exists()
        assert path.name == "universe_2023-01-15.json"

    def test_write_content(self, tmp_path: Path) -> None:
        """Test written content is correct."""
        tickers = ["AAPL", "MSFT"]
        day = date(2023, 1, 15)

        path = write_universe_selection(tickers, day, tmp_path)

        with open(path) as f:
            data = json.load(f)

        assert data["date"] == "2023-01-15"
        assert data["tickers"] == tickers
        assert data["count"] == 2

    def test_write_with_metadata(self, tmp_path: Path) -> None:
        """Test writing with additional metadata."""
        tickers = ["AAPL"]
        day = date(2023, 1, 15)
        metadata = {"source": "test", "version": 1}

        path = write_universe_selection(tickers, day, tmp_path, metadata)

        with open(path) as f:
            data = json.load(f)

        assert data["source"] == "test"
        assert data["version"] == 1


class TestLoadUniverseSelection:
    """Tests for loading universe selection."""

    def test_load_existing(self, tmp_path: Path) -> None:
        """Test loading existing selection."""
        tickers = ["AAPL", "MSFT", "NVDA"]
        day = date(2023, 1, 15)

        write_universe_selection(tickers, day, tmp_path)
        loaded = load_universe_selection(day, tmp_path)

        assert loaded == tickers

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Test loading nonexistent returns None."""
        loaded = load_universe_selection(date(2023, 1, 15), tmp_path)

        assert loaded is None

    def test_load_string_date(self, tmp_path: Path) -> None:
        """Test loading with string date."""
        tickers = ["AAPL"]
        write_universe_selection(tickers, date(2023, 1, 15), tmp_path)

        loaded = load_universe_selection("2023-01-15", tmp_path)

        assert loaded == tickers
