"""Integration tests for walk-forward with rolling retrain."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from traderbot.cli.walkforward import run_walkforward


@pytest.fixture
def sample_walkforward_data() -> dict[str, pd.DataFrame]:
    """Create sample OHLCV data for walk-forward testing."""
    np.random.seed(42)
    n_days = 90  # Need enough days for training

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


class TestWalkforwardWithSizer:
    """Tests for walk-forward with position sizing."""

    def test_walkforward_with_fixed_sizer(
        self, sample_walkforward_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test walk-forward with fixed fraction sizer."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_walkforward_data
            mock_adapter.return_value = mock_instance

            results = run_walkforward(
                start_date="2023-01-03",
                end_date="2023-03-15",
                universe=list(sample_walkforward_data.keys()),
                n_splits=2,
                is_ratio=0.6,
                output_dir=tmp_path,
                universe_mode="static",
                sizer="fixed",
                fixed_frac=0.15,
            )

        assert "error" not in results
        assert results.get("sizer") == "fixed"

    def test_walkforward_with_vol_sizer(
        self, sample_walkforward_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test walk-forward with volatility-targeting sizer."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_walkforward_data
            mock_adapter.return_value = mock_instance

            results = run_walkforward(
                start_date="2023-01-03",
                end_date="2023-03-15",
                universe=list(sample_walkforward_data.keys()),
                n_splits=2,
                is_ratio=0.6,
                output_dir=tmp_path,
                universe_mode="static",
                sizer="vol",
                vol_target=0.15,
            )

        assert "error" not in results
        assert results.get("sizer") == "vol"

    def test_walkforward_with_kelly_sizer(
        self, sample_walkforward_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test walk-forward with Kelly criterion sizer."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_walkforward_data
            mock_adapter.return_value = mock_instance

            results = run_walkforward(
                start_date="2023-01-03",
                end_date="2023-03-15",
                universe=list(sample_walkforward_data.keys()),
                n_splits=2,
                is_ratio=0.6,
                output_dir=tmp_path,
                universe_mode="static",
                sizer="kelly",
                kelly_cap=0.2,
            )

        assert "error" not in results
        assert results.get("sizer") == "kelly"


class TestWalkforwardWithThreshold:
    """Tests for walk-forward with probability thresholding."""

    def test_walkforward_with_custom_threshold(
        self, sample_walkforward_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test walk-forward with custom probability threshold."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_walkforward_data
            mock_adapter.return_value = mock_instance

            results = run_walkforward(
                start_date="2023-01-03",
                end_date="2023-03-15",
                universe=list(sample_walkforward_data.keys()),
                n_splits=2,
                is_ratio=0.6,
                output_dir=tmp_path,
                universe_mode="static",
                proba_threshold=0.6,
            )

        assert "error" not in results

    def test_walkforward_with_opt_threshold(
        self, sample_walkforward_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test walk-forward with threshold optimization."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_walkforward_data
            mock_adapter.return_value = mock_instance

            results = run_walkforward(
                start_date="2023-01-03",
                end_date="2023-03-15",
                universe=list(sample_walkforward_data.keys()),
                n_splits=2,
                is_ratio=0.6,
                output_dir=tmp_path,
                universe_mode="static",
                opt_threshold=True,
            )

        assert "error" not in results


class TestWalkforwardExecutionCosts:
    """Tests for execution cost tracking in walk-forward."""

    def test_walkforward_tracks_execution_costs(
        self, sample_walkforward_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test walk-forward tracks execution costs."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_walkforward_data
            mock_adapter.return_value = mock_instance

            results = run_walkforward(
                start_date="2023-01-03",
                end_date="2023-03-15",
                universe=list(sample_walkforward_data.keys()),
                n_splits=2,
                is_ratio=0.6,
                output_dir=tmp_path,
                universe_mode="static",
            )

        assert "error" not in results
        # Execution costs may or may not be present depending on trading activity
        if "execution_costs" in results:
            costs = results["execution_costs"]
            assert "commission" in costs
            assert "fees" in costs
            assert "slippage" in costs


class TestWalkforwardOutputFiles:
    """Tests for walk-forward output file generation."""

    def test_report_file_generated(
        self, sample_walkforward_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test report.md file is generated."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_walkforward_data
            mock_adapter.return_value = mock_instance

            run_walkforward(
                start_date="2023-01-03",
                end_date="2023-03-15",
                universe=list(sample_walkforward_data.keys()),
                n_splits=2,
                is_ratio=0.6,
                output_dir=tmp_path,
                universe_mode="static",
            )

        # Check report exists
        assert (tmp_path / "report.md").exists()

    def test_results_json_has_sizer_info(
        self, sample_walkforward_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test results.json includes sizer information."""
        import json

        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_walkforward_data
            mock_adapter.return_value = mock_instance

            run_walkforward(
                start_date="2023-01-03",
                end_date="2023-03-15",
                universe=list(sample_walkforward_data.keys()),
                n_splits=2,
                is_ratio=0.6,
                output_dir=tmp_path,
                universe_mode="static",
                sizer="vol",
            )

        results_path = tmp_path / "results.json"
        assert results_path.exists()

        with open(results_path) as f:
            results = json.load(f)

        assert results.get("sizer") == "vol"


class TestWalkforwardTrainPerSplit:
    """Tests for train-per-split functionality."""

    def test_train_per_split_creates_models_dir(
        self, sample_walkforward_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test train_per_split creates models directory."""
        # This test requires a full training setup, so we just test the flag is accepted
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_walkforward_data
            mock_adapter.return_value = mock_instance

            # Mock subprocess.run for both training and git commands
            with patch("subprocess.run") as mock_run:
                # Return proper values for different subprocess calls
                def subprocess_side_effect(*args, **kwargs):
                    mock_result = MagicMock()
                    mock_result.returncode = 0
                    mock_result.stdout = "abc1234"  # For git rev-parse
                    return mock_result

                mock_run.side_effect = subprocess_side_effect

                results = run_walkforward(
                    start_date="2023-01-03",
                    end_date="2023-03-15",
                    universe=list(sample_walkforward_data.keys()),
                    n_splits=2,
                    is_ratio=0.6,
                    output_dir=tmp_path,
                    universe_mode="static",
                    train_per_split=True,
                    epochs=1,  # Minimal training
                )

        # Check models directory was created
        models_dir = tmp_path / "models"
        assert models_dir.exists() or "error" not in results
