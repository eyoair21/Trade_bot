"""Tests for Parquet local adapter."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traderbot.data.adapters.parquet_local import ParquetLocalAdapter


@pytest.fixture
def adapter_with_data(tmp_data_dir: Path) -> tuple[ParquetLocalAdapter, pd.DataFrame]:
    """Create adapter with test data."""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-03", periods=30, freq="B")
    n = len(dates)

    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close * (1 + np.random.randn(n) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "close": close,
            "volume": np.random.randint(1000000, 10000000, n),
        }
    )

    # Save as parquet
    file_path = tmp_data_dir / "AAPL.parquet"
    df.to_parquet(file_path)

    adapter = ParquetLocalAdapter(data_dir=tmp_data_dir)
    return adapter, df


class TestParquetLocalAdapter:
    """Tests for ParquetLocalAdapter."""

    def test_load_full_data(
        self, adapter_with_data: tuple[ParquetLocalAdapter, pd.DataFrame]
    ) -> None:
        """Test loading full data for a ticker."""
        adapter, original_df = adapter_with_data

        loaded = adapter.load("AAPL")

        assert len(loaded) == len(original_df)
        assert list(loaded.columns) == list(original_df.columns)
        assert loaded["close"].iloc[0] == pytest.approx(original_df["close"].iloc[0])

    def test_load_with_date_range(
        self, adapter_with_data: tuple[ParquetLocalAdapter, pd.DataFrame]
    ) -> None:
        """Test loading data with date range filter."""
        adapter, _ = adapter_with_data

        loaded = adapter.load(
            "AAPL",
            start_date=date(2023, 1, 10),
            end_date=date(2023, 1, 20),
        )

        assert len(loaded) < 30  # Fewer than full dataset
        assert loaded["date"].min().date() >= date(2023, 1, 10)
        assert loaded["date"].max().date() <= date(2023, 1, 20)

    def test_load_with_as_of_date(
        self, adapter_with_data: tuple[ParquetLocalAdapter, pd.DataFrame]
    ) -> None:
        """Test point-in-time filter."""
        adapter, _ = adapter_with_data

        loaded = adapter.load("AAPL", as_of_date=date(2023, 1, 15))

        # Should only have data up to Jan 15
        assert loaded["date"].max().date() <= date(2023, 1, 15)

    def test_load_nonexistent_ticker(self, tmp_data_dir: Path) -> None:
        """Test loading non-existent ticker raises error."""
        adapter = ParquetLocalAdapter(data_dir=tmp_data_dir)

        with pytest.raises(FileNotFoundError):
            adapter.load("FAKE")

    def test_load_multiple_tickers(self, tmp_data_dir: Path) -> None:
        """Test loading multiple tickers."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-03", periods=10, freq="B")

        for ticker in ["AAPL", "MSFT"]:
            df = pd.DataFrame(
                {
                    "date": dates,
                    "open": np.random.randn(10) + 100,
                    "high": np.random.randn(10) + 101,
                    "low": np.random.randn(10) + 99,
                    "close": np.random.randn(10) + 100,
                    "volume": np.random.randint(1000000, 10000000, 10),
                }
            )
            df.to_parquet(tmp_data_dir / f"{ticker}.parquet")

        adapter = ParquetLocalAdapter(data_dir=tmp_data_dir)
        data = adapter.load_multiple(["AAPL", "MSFT", "FAKE"])

        assert "AAPL" in data
        assert "MSFT" in data
        assert "FAKE" not in data  # Should be skipped

    def test_cache_works(self, adapter_with_data: tuple[ParquetLocalAdapter, pd.DataFrame]) -> None:
        """Test caching prevents re-reading file."""
        adapter, _ = adapter_with_data

        # Load twice
        loaded1 = adapter.load("AAPL")
        loaded2 = adapter.load("AAPL")

        # Should get same data (from cache)
        pd.testing.assert_frame_equal(loaded1, loaded2)

    def test_clear_cache(self, adapter_with_data: tuple[ParquetLocalAdapter, pd.DataFrame]) -> None:
        """Test cache clearing."""
        adapter, _ = adapter_with_data

        adapter.load("AAPL")
        assert len(adapter._cache) == 1

        adapter.clear_cache()
        assert len(adapter._cache) == 0

    def test_accepts_string_dates(
        self, adapter_with_data: tuple[ParquetLocalAdapter, pd.DataFrame]
    ) -> None:
        """Test adapter accepts ISO date strings."""
        adapter, _ = adapter_with_data

        loaded = adapter.load(
            "AAPL",
            start_date="2023-01-10",
            end_date="2023-01-20",
        )

        assert len(loaded) > 0
