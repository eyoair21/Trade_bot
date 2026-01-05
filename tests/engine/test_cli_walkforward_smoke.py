"""Smoke tests for walk-forward CLI.

Tests that the CLI produces expected artifacts when run against sample data.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from scripts.make_sample_data import create_sample_data
from traderbot.cli.walkforward import run_walkforward


class TestCLIWalkforwardSmoke:
    """Smoke tests for walk-forward CLI."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path: Path) -> Path:
        """Create sample data in a temporary directory."""
        data_dir = tmp_path / "ohlcv"
        create_sample_data(
            output_dir=data_dir,
            tickers=["AAPL", "MSFT", "NVDA"],
            start_date="2023-01-01",
            end_date="2023-03-31",
            seed=42,
            use_yfinance=False,
        )
        return data_dir

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create output directory."""
        out = tmp_path / "runs"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def test_walkforward_produces_results_json(
        self, sample_data_dir: Path, output_dir: Path
    ) -> None:
        """Test that walk-forward produces results.json artifact."""
        result = run_walkforward(
            start_date="2023-01-10",
            end_date="2023-03-31",
            universe=["AAPL", "MSFT", "NVDA"],
            n_splits=3,
            is_ratio=0.6,
            output_dir=output_dir,
            data_root=sample_data_dir,
        )

        assert "error" not in result

        results_path = output_dir / "results.json"
        assert results_path.exists(), "results.json not created"

        # Verify JSON is valid
        with open(results_path) as f:
            data = json.load(f)

        assert "start_date" in data
        assert "end_date" in data
        assert "splits" in data
        assert len(data["splits"]) == 3

    def test_walkforward_produces_equity_curve_csv(
        self, sample_data_dir: Path, output_dir: Path
    ) -> None:
        """Test that walk-forward produces equity_curve.csv artifact."""
        result = run_walkforward(
            start_date="2023-01-10",
            end_date="2023-03-31",
            universe=["AAPL", "MSFT", "NVDA"],
            n_splits=3,
            is_ratio=0.6,
            output_dir=output_dir,
            data_root=sample_data_dir,
        )

        assert "error" not in result

        equity_path = output_dir / "equity_curve.csv"
        assert equity_path.exists(), "equity_curve.csv not created"

        # Verify CSV has rows
        df = pd.read_csv(equity_path)
        assert len(df) > 0, "equity_curve.csv has no rows"
        assert "date" in df.columns or "equity" in df.columns

    def test_walkforward_produces_run_manifest(
        self, sample_data_dir: Path, output_dir: Path
    ) -> None:
        """Test that walk-forward produces run_manifest.json artifact."""
        result = run_walkforward(
            start_date="2023-01-10",
            end_date="2023-03-31",
            universe=["AAPL", "MSFT", "NVDA"],
            n_splits=3,
            is_ratio=0.6,
            output_dir=output_dir,
            data_root=sample_data_dir,
        )

        assert "error" not in result

        manifest_path = output_dir / "run_manifest.json"
        assert manifest_path.exists(), "run_manifest.json not created"

        # Verify manifest contents
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "timestamp" in manifest
        assert "python_version" in manifest
        assert "os" in manifest
        assert "git_sha" in manifest

    def test_walkforward_all_three_artifacts_exist(
        self, sample_data_dir: Path, output_dir: Path
    ) -> None:
        """Test that all three required artifacts are created."""
        result = run_walkforward(
            start_date="2023-01-10",
            end_date="2023-03-31",
            universe=["AAPL", "MSFT", "NVDA"],
            n_splits=3,
            is_ratio=0.6,
            output_dir=output_dir,
            data_root=sample_data_dir,
        )

        assert "error" not in result

        # All three artifacts must exist
        assert (output_dir / "results.json").exists()
        assert (output_dir / "equity_curve.csv").exists()
        assert (output_dir / "run_manifest.json").exists()

    def test_walkforward_equity_curve_has_positive_rows(
        self, sample_data_dir: Path, output_dir: Path
    ) -> None:
        """Test that equity curve has more than 0 rows."""
        result = run_walkforward(
            start_date="2023-01-10",
            end_date="2023-03-31",
            universe=["AAPL", "MSFT", "NVDA"],
            n_splits=3,
            is_ratio=0.6,
            output_dir=output_dir,
            data_root=sample_data_dir,
        )

        assert "error" not in result

        equity_path = output_dir / "equity_curve.csv"
        df = pd.read_csv(equity_path)

        assert len(df) > 0, "equity_curve.csv must have > 0 rows"

    def test_walkforward_results_contain_splits_data(
        self, sample_data_dir: Path, output_dir: Path
    ) -> None:
        """Test that results contain split-level data."""
        result = run_walkforward(
            start_date="2023-01-10",
            end_date="2023-03-31",
            universe=["AAPL", "MSFT", "NVDA"],
            n_splits=3,
            is_ratio=0.6,
            output_dir=output_dir,
            data_root=sample_data_dir,
        )

        assert "error" not in result
        assert "splits" in result
        assert len(result["splits"]) == 3

        # Each split should have expected fields
        for split in result["splits"]:
            assert "split" in split
            assert "is_start" in split
            assert "is_end" in split
            assert "oos_start" in split
            assert "oos_end" in split
            assert "oos_return_pct" in split

    def test_walkforward_deterministic(self, sample_data_dir: Path, tmp_path: Path) -> None:
        """Test that walk-forward produces deterministic results."""
        outputs = []

        for i in range(2):
            out_dir = tmp_path / f"run_{i}"
            out_dir.mkdir()

            result = run_walkforward(
                start_date="2023-01-10",
                end_date="2023-03-31",
                universe=["AAPL", "MSFT", "NVDA"],
                n_splits=3,
                is_ratio=0.6,
                output_dir=out_dir,
                data_root=sample_data_dir,
            )

            outputs.append(result)

        # Results should be identical (excluding timestamps)
        assert outputs[0]["avg_oos_return_pct"] == outputs[1]["avg_oos_return_pct"]
        assert outputs[0]["total_oos_trades"] == outputs[1]["total_oos_trades"]
