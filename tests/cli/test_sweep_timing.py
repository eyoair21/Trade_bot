"""Tests for sweep timing functionality."""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from traderbot.cli.sweep import run_sweep
from traderbot.sweeps.schema import SweepConfig


@pytest.fixture
def sample_data() -> dict[str, pd.DataFrame]:
    """Create sample OHLCV data."""
    np.random.seed(42)
    n_days = 40

    data = {}
    for ticker in ["AAPL", "MSFT"]:
        close = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))
        data[ticker] = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n_days, freq="D"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(1_000_000, 5_000_000, n_days),
        })

    return data


@pytest.fixture
def minimal_sweep_config(tmp_path: Path) -> SweepConfig:
    """Create minimal sweep configuration."""
    return SweepConfig(
        name="test_sweep",
        output_root=tmp_path / "sweep_test",
        metric="sharpe",
        mode="max",
        fixed_args={
            "start_date": "2023-01-05",
            "end_date": "2023-01-25",
            "universe": ["AAPL", "MSFT"],
            "n_splits": 2,
            "is_ratio": 0.6,
            "seed": 42,
        },
        grid={
            "sizer": ["fixed"],
            "proba_threshold": [0.50, 0.55],
        },
    )


class TestSweepTiming:
    """Tests for sweep timing functionality."""

    def test_timing_csv_created(
        self, minimal_sweep_config: SweepConfig, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test that timings.csv is created when --time is enabled."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_data
            mock_adapter.return_value = mock_instance

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "abc1234\n"
                mock_run.return_value = mock_result

                _ = run_sweep(
                    config=minimal_sweep_config,
                    workers=1,
                    dry_run=False,
                    enable_timing=True,
                )

        # Check timings.csv exists
        timing_path = minimal_sweep_config.output_root / "timings.csv"
        assert timing_path.exists()

    def test_timing_csv_has_correct_headers(
        self, minimal_sweep_config: SweepConfig, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test that timings.csv has correct headers."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_data
            mock_adapter.return_value = mock_instance

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "abc1234\n"
                mock_run.return_value = mock_result

                _ = run_sweep(
                    config=minimal_sweep_config,
                    workers=1,
                    enable_timing=True,
                )

        timing_path = minimal_sweep_config.output_root / "timings.csv"

        with open(timing_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        # Check required headers
        assert "run_idx" in headers
        assert "elapsed_s" in headers
        assert "load_s" in headers
        assert "backtest_s" in headers
        assert "report_s" in headers

    def test_timing_csv_has_data_rows(
        self, minimal_sweep_config: SweepConfig, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test that timings.csv has data for each successful run."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_data
            mock_adapter.return_value = mock_instance

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "abc1234\n"
                mock_run.return_value = mock_result

                run_sweep(
                    config=minimal_sweep_config,
                    workers=1,
                    enable_timing=True,
                )

        timing_path = minimal_sweep_config.output_root / "timings.csv"

        with open(timing_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have rows for successful runs
        assert len(rows) > 0  # At least one run should succeed

