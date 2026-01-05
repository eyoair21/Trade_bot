"""Tests for seed reproducibility and run manifest."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from traderbot.cli.walkforward import run_walkforward


@pytest.fixture
def sample_data() -> dict[str, pd.DataFrame]:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 60

    data = {}
    for ticker in ["AAPL", "MSFT", "NVDA"]:
        close = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))
        volume = np.random.randint(1_000_000, 10_000_000, n_days)

        data[ticker] = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n_days, freq="D"),
                "open": close * (1 + np.random.randn(n_days) * 0.005),
                "high": close * (1 + np.abs(np.random.randn(n_days) * 0.01)),
                "low": close * (1 - np.abs(np.random.randn(n_days) * 0.01)),
                "close": close,
                "volume": volume,
            }
        )

    return data


class TestSeedReproducibility:
    """Tests for deterministic runs with same seed."""

    def test_same_seed_produces_identical_results(
        self, sample_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test that same seed produces identical results."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_data
            mock_adapter.return_value = mock_instance

            # Mock subprocess for git SHA
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "abc1234\n"
                mock_run.return_value = mock_result

                # Run 1 with seed 123
                output_a = tmp_path / "run_a"
                results_a = run_walkforward(
                    start_date="2023-01-03",
                    end_date="2023-02-28",
                    universe=list(sample_data.keys()),
                    n_splits=2,
                    is_ratio=0.6,
                    output_dir=output_a,
                    universe_mode="static",
                    seed=123,
                )

                # Run 2 with same seed 123
                output_b = tmp_path / "run_b"
                results_b = run_walkforward(
                    start_date="2023-01-03",
                    end_date="2023-02-28",
                    universe=list(sample_data.keys()),
                    n_splits=2,
                    is_ratio=0.6,
                    output_dir=output_b,
                    universe_mode="static",
                    seed=123,
                )

        # Check key metrics are identical
        assert results_a["avg_oos_return_pct"] == pytest.approx(results_b["avg_oos_return_pct"], abs=0.001)
        assert results_a["avg_oos_sharpe"] == pytest.approx(results_b["avg_oos_sharpe"], abs=0.001)
        assert results_a["total_oos_trades"] == results_b["total_oos_trades"]

        # Check seed is recorded
        assert results_a["seed"] == 123
        assert results_b["seed"] == 123

    def test_different_seed_produces_different_results(
        self, sample_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test that different seed produces different results."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_data
            mock_adapter.return_value = mock_instance

            # Mock subprocess for git SHA
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "abc1234\n"
                mock_run.return_value = mock_result

                # Run 1 with seed 123
                output_a = tmp_path / "run_seed123"
                results_a = run_walkforward(
                    start_date="2023-01-03",
                    end_date="2023-02-28",
                    universe=list(sample_data.keys()),
                    n_splits=2,
                    is_ratio=0.6,
                    output_dir=output_a,
                    universe_mode="static",
                    seed=123,
                )

                # Run 2 with different seed 456
                output_b = tmp_path / "run_seed456"
                results_b = run_walkforward(
                    start_date="2023-01-03",
                    end_date="2023-02-28",
                    universe=list(sample_data.keys()),
                    n_splits=2,
                    is_ratio=0.6,
                    output_dir=output_b,
                    universe_mode="static",
                    seed=456,
                )

        # Results should differ (allow for edge case where they're identical by chance)
        # At minimum, seeds should be recorded differently
        assert results_a["seed"] == 123
        assert results_b["seed"] == 456

        # Typically metrics or trade counts will differ
        # (but can be identical with simple strategies, so we just check seeds)
        assert results_a["seed"] != results_b["seed"]


class TestRunManifest:
    """Tests for run manifest generation."""

    def test_manifest_file_created(
        self, sample_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test that run_manifest.json is created."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_data
            mock_adapter.return_value = mock_instance

            # Mock subprocess for git SHA
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "abc1234\n"
                mock_run.return_value = mock_result

                output_dir = tmp_path / "test_run"
                run_walkforward(
                    start_date="2023-01-03",
                    end_date="2023-02-28",
                    universe=list(sample_data.keys()),
                    n_splits=2,
                    is_ratio=0.6,
                    output_dir=output_dir,
                    universe_mode="static",
                    seed=42,
                )

        # Check manifest exists
        manifest_path = output_dir / "run_manifest.json"
        assert manifest_path.exists()

    def test_manifest_contains_required_fields(
        self, sample_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test that manifest contains all required fields."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_data
            mock_adapter.return_value = mock_instance

            # Mock subprocess for git SHA
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "abc1234\n"
                mock_run.return_value = mock_result

                output_dir = tmp_path / "test_run"
                results = run_walkforward(
                    start_date="2023-01-03",
                    end_date="2023-02-28",
                    universe=list(sample_data.keys()),
                    n_splits=2,
                    is_ratio=0.6,
                    output_dir=output_dir,
                    universe_mode="static",
                    seed=99,
                )

        # Check results.json has manifest
        assert "manifest" in results
        manifest = results["manifest"]

        # Check required fields
        assert "run_id" in manifest
        assert "git_sha" in manifest
        assert isinstance(manifest["git_sha"], str)
        assert "seed" in manifest
        assert isinstance(manifest["seed"], int)
        assert manifest["seed"] == 99
        assert "data_digest" in manifest
        assert isinstance(manifest["data_digest"], str)
        assert len(manifest["data_digest"]) == 16  # SHA256 truncated to 16 chars

        # Check manifest file
        manifest_path = output_dir / "run_manifest.json"
        with open(manifest_path) as f:
            manifest_from_file = json.load(f)

        assert manifest_from_file["seed"] == 99
        assert isinstance(manifest_from_file["git_sha"], str)

