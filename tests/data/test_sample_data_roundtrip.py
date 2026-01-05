"""Tests for sample data generation and loading roundtrip."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from scripts.make_sample_data import create_sample_data, generate_synthetic_ohlcv
from traderbot.data.adapters.parquet_local import ParquetLocalAdapter


class TestGenerateSyntheticOHLCV:
    """Tests for synthetic OHLCV generation."""

    def test_generates_correct_columns(self) -> None:
        """Test that generated data has all required columns."""
        df = generate_synthetic_ohlcv("AAPL", "2023-01-01", "2023-01-31", seed=42)

        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_generates_correct_date_range(self) -> None:
        """Test that generated data covers the requested date range."""
        df = generate_synthetic_ohlcv("AAPL", "2023-01-01", "2023-01-31", seed=42)

        # Should have ~21 business days in January 2023
        assert len(df) >= 20
        assert len(df) <= 23

        # Check date bounds
        first_date = df["timestamp"].iloc[0].date()
        last_date = df["timestamp"].iloc[-1].date()

        assert first_date >= date(2023, 1, 1)
        assert last_date <= date(2023, 1, 31)

    def test_generates_valid_ohlc(self) -> None:
        """Test that OHLC relationships are valid."""
        df = generate_synthetic_ohlcv("AAPL", "2023-01-01", "2023-03-31", seed=42)

        # High should be >= low
        assert (df["high"] >= df["low"]).all()

        # Open and close should be within high/low range
        assert (df["open"] >= df["low"]).all()
        assert (df["open"] <= df["high"]).all()
        assert (df["close"] >= df["low"]).all()
        assert (df["close"] <= df["high"]).all()

    def test_deterministic_with_seed(self) -> None:
        """Test that same seed produces same data."""
        df1 = generate_synthetic_ohlcv("AAPL", "2023-01-01", "2023-01-31", seed=42)
        df2 = generate_synthetic_ohlcv("AAPL", "2023-01-01", "2023-01-31", seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_tickers_different_prices(self) -> None:
        """Test that different tickers have different base prices."""
        aapl = generate_synthetic_ohlcv("AAPL", "2023-01-01", "2023-01-31", seed=42)
        msft = generate_synthetic_ohlcv("MSFT", "2023-01-01", "2023-01-31", seed=42)

        # Different base prices should result in different closes
        # AAPL base is 150, MSFT base is 250
        assert not (aapl["close"].values == msft["close"].values).all()


class TestCreateSampleData:
    """Tests for create_sample_data function."""

    def test_creates_parquet_files(self, tmp_path: Path) -> None:
        """Test that parquet files are created for all tickers."""
        tickers = ["AAPL", "MSFT"]
        created = create_sample_data(
            output_dir=tmp_path,
            tickers=tickers,
            start_date="2023-01-01",
            end_date="2023-01-31",
            seed=42,
            use_yfinance=False,
        )

        assert len(created) == 2
        for ticker in tickers:
            assert ticker in created
            assert created[ticker].exists()
            assert created[ticker].suffix == ".parquet"

    def test_parquet_files_readable(self, tmp_path: Path) -> None:
        """Test that created parquet files can be read back."""
        tickers = ["AAPL"]
        create_sample_data(
            output_dir=tmp_path,
            tickers=tickers,
            start_date="2023-01-01",
            end_date="2023-01-31",
            seed=42,
            use_yfinance=False,
        )

        df = pd.read_parquet(tmp_path / "AAPL.parquet")
        assert not df.empty
        assert "timestamp" in df.columns
        assert "close" in df.columns


class TestSampleDataRoundtrip:
    """Tests for full roundtrip: generate -> save -> load via adapter."""

    def test_roundtrip_loads_via_adapter(self, tmp_path: Path) -> None:
        """Test that generated data can be loaded via ParquetLocalAdapter."""
        tickers = ["AAPL", "MSFT", "NVDA"]

        # Create sample data
        create_sample_data(
            output_dir=tmp_path,
            tickers=tickers,
            start_date="2023-01-01",
            end_date="2023-03-31",
            seed=42,
            use_yfinance=False,
        )

        # Load via adapter
        adapter = ParquetLocalAdapter(data_dir=tmp_path)
        data = adapter.load_multiple(tickers)

        assert len(data) == 3
        for ticker in tickers:
            assert ticker in data
            df = data[ticker]
            assert not df.empty
            assert "date" in df.columns  # Adapter normalizes timestamp -> date
            assert "close" in df.columns

    def test_roundtrip_date_bounds_correct(self, tmp_path: Path) -> None:
        """Test that loaded data has correct date bounds."""
        start = "2023-01-01"
        end = "2023-03-31"

        create_sample_data(
            output_dir=tmp_path,
            tickers=["AAPL"],
            start_date=start,
            end_date=end,
            seed=42,
            use_yfinance=False,
        )

        adapter = ParquetLocalAdapter(data_dir=tmp_path)
        df = adapter.load("AAPL")

        first_date = df["date"].iloc[0].date()
        last_date = df["date"].iloc[-1].date()

        assert first_date >= date(2023, 1, 1)
        assert last_date <= date(2023, 3, 31)

    def test_roundtrip_non_empty_frame(self, tmp_path: Path) -> None:
        """Test that loaded data has non-empty frames."""
        create_sample_data(
            output_dir=tmp_path,
            tickers=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-03-31",
            seed=42,
            use_yfinance=False,
        )

        adapter = ParquetLocalAdapter(data_dir=tmp_path)

        for ticker in ["AAPL", "MSFT"]:
            df = adapter.load(ticker)
            assert len(df) > 0, f"Empty frame for {ticker}"
            # Should have ~60 business days in Q1 2023
            assert len(df) >= 50, f"Too few rows for {ticker}: {len(df)}"

    def test_roundtrip_with_date_filters(self, tmp_path: Path) -> None:
        """Test that adapter date filters work with generated data."""
        create_sample_data(
            output_dir=tmp_path,
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-03-31",
            seed=42,
            use_yfinance=False,
        )

        adapter = ParquetLocalAdapter(data_dir=tmp_path)

        # Load full range
        full_df = adapter.load("AAPL")

        # Load filtered range
        filtered_df = adapter.load(
            "AAPL",
            start_date=date(2023, 2, 1),
            end_date=date(2023, 2, 28),
        )

        assert len(filtered_df) < len(full_df)
        assert filtered_df["date"].min().date() >= date(2023, 2, 1)
        assert filtered_df["date"].max().date() <= date(2023, 2, 28)

    def test_roundtrip_with_as_of_date(self, tmp_path: Path) -> None:
        """Test that as_of_date filter prevents lookahead."""
        create_sample_data(
            output_dir=tmp_path,
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-03-31",
            seed=42,
            use_yfinance=False,
        )

        adapter = ParquetLocalAdapter(data_dir=tmp_path)

        # Load with as_of_date
        df = adapter.load("AAPL", as_of_date=date(2023, 1, 31))

        # All dates should be <= as_of_date
        assert (df["date"].dt.date <= date(2023, 1, 31)).all()
