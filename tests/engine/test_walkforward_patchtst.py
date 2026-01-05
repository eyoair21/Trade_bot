"""Integration tests for walk-forward with PatchTST model."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from traderbot.cli.walkforward import create_splits, run_walkforward
from traderbot.engine.backtest import BacktestEngine
from traderbot.engine.strategy_momo import MomentumStrategy


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


class TestCreateSplits:
    """Tests for walk-forward split creation."""

    def test_creates_correct_number_of_splits(self) -> None:
        """Test correct number of splits created."""
        from datetime import datetime

        start = datetime(2023, 1, 3)
        end = datetime(2023, 3, 31)

        splits = create_splits(start, end, n_splits=3, is_ratio=0.7)

        assert len(splits) == 3

    def test_splits_are_tuples(self) -> None:
        """Test splits are 4-tuples."""
        from datetime import datetime

        splits = create_splits(
            datetime(2023, 1, 3),
            datetime(2023, 3, 31),
            n_splits=2,
            is_ratio=0.7,
        )

        for split in splits:
            assert len(split) == 4
            is_start, is_end, oos_start, oos_end = split
            assert is_start < is_end
            assert oos_start < oos_end
            assert is_end < oos_start


class TestBacktestEngineWithModel:
    """Tests for BacktestEngine with model integration."""

    def test_engine_accepts_model_path(self) -> None:
        """Test engine accepts model_path parameter."""
        strategy = MomentumStrategy(name="test", universe=["AAPL"])
        engine = BacktestEngine(
            strategy=strategy,
            model_path=None,  # No model, should work
        )

        assert engine._model is None

    def test_engine_handles_missing_model(self, tmp_path: Path) -> None:
        """Test engine handles missing model file gracefully."""
        strategy = MomentumStrategy(name="test", universe=["AAPL"])

        # Should not raise, just log warning
        engine = BacktestEngine(
            strategy=strategy,
            model_path=tmp_path / "nonexistent.ts",
        )

        assert engine._model is None

    def test_get_model_predictions_no_model(self, sample_data: dict[str, pd.DataFrame]) -> None:
        """Test get_model_predictions returns empty when no model."""
        strategy = MomentumStrategy(name="test", universe=["AAPL"])
        engine = BacktestEngine(strategy=strategy, model_path=None)

        predictions = engine.get_model_predictions(sample_data)

        assert predictions == {}

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch not available",
    )
    def test_engine_with_mock_model(
        self, sample_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test engine with mocked model."""
        pytest.importorskip("torch")
        from traderbot.model.patchtst import PatchTSTConfig, PatchTSTModel, export_torchscript

        # Create and export a model
        config = PatchTSTConfig(lookback=32, n_features=6)
        model = PatchTSTModel(config)
        model_path = tmp_path / "test_model.ts"
        export_torchscript(model, model_path)

        # Create engine with model
        strategy = MomentumStrategy(name="test", universe=list(sample_data.keys()))
        engine = BacktestEngine(
            strategy=strategy,
            model_path=model_path,
        )

        assert engine._model is not None


class TestRunWalkforwardIntegration:
    """Integration tests for run_walkforward function."""

    def test_walkforward_static_mode(
        self, sample_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test walk-forward runs with static universe mode."""
        # Mock data loading to use our sample data
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_data
            mock_adapter.return_value = mock_instance

            results = run_walkforward(
                start_date="2023-01-03",
                end_date="2023-02-28",
                universe=list(sample_data.keys()),
                n_splits=2,
                is_ratio=0.6,
                output_dir=tmp_path,
                universe_mode="static",
            )

        assert "error" not in results
        assert results["universe_mode"] == "static"
        assert len(results["splits"]) == 2

    def test_walkforward_output_files(
        self, sample_data: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        """Test walk-forward creates output files."""
        with patch("traderbot.cli.walkforward.ParquetLocalAdapter") as mock_adapter:
            mock_instance = MagicMock()
            mock_instance.load_multiple.return_value = sample_data
            mock_adapter.return_value = mock_instance

            run_walkforward(
                start_date="2023-01-03",
                end_date="2023-02-28",
                universe=list(sample_data.keys()),
                n_splits=2,
                is_ratio=0.6,
                output_dir=tmp_path,
                universe_mode="static",
            )

        # Check output files
        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "equity_curve.csv").exists()
        assert (tmp_path / "run_manifest.json").exists()


class TestStrategyModelPredictions:
    """Tests for model predictions in strategy."""

    def test_strategy_has_model_predictions_attr(self) -> None:
        """Test strategy has model_predictions attribute."""
        strategy = MomentumStrategy(name="test", universe=["AAPL"])

        assert hasattr(strategy, "model_predictions")
        assert strategy.model_predictions == {}

    def test_strategy_reset_clears_predictions(self) -> None:
        """Test strategy reset clears model predictions."""
        strategy = MomentumStrategy(name="test", universe=["AAPL"])
        strategy.model_predictions = {"AAPL": 0.7}

        strategy.reset()

        assert strategy.model_predictions == {}
