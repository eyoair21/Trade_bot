"""Tests for CLI modules."""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traderbot.cli.walkforward import (
    create_splits,
    create_synthetic_data,
    run_walkforward,
)
from traderbot.reports.run_manifest import RunManifest, create_run_manifest


class TestCreateRunManifest:
    """Tests for run manifest creation."""

    def test_manifest_has_required_fields(self) -> None:
        """Test manifest contains required fields."""
        manifest = create_run_manifest(
            run_id="2023-01-01T12-00-00",
            git_sha="abc1234",
            seed=42,
            start_date="2023-01-01",
            end_date="2023-03-31",
            universe=["AAPL", "MSFT"],
            n_splits=3,
            is_ratio=0.6,
            sizer="fixed",
            sizer_params={"fixed_frac": 0.1},
            all_cli_params={"start_date": "2023-01-01"},
        )

        assert isinstance(manifest, RunManifest)
        assert manifest.run_id == "2023-01-01T12-00-00"
        assert manifest.git_sha == "abc1234"
        assert manifest.seed == 42
        assert manifest.universe == ["AAPL", "MSFT"]
        assert manifest.n_splits == 3
        assert manifest.is_ratio == 0.6

    def test_manifest_to_dict(self) -> None:
        """Test manifest converts to dictionary."""
        manifest = create_run_manifest(
            run_id="2023-01-01T12-00-00",
            git_sha="abc1234",
            seed=42,
            start_date="2023-01-01",
            end_date="2023-03-31",
            universe=["AAPL"],
            n_splits=3,
            is_ratio=0.6,
            sizer="vol",
            sizer_params={"vol_target": 0.15},
            all_cli_params={},
        )

        manifest_dict = manifest.to_dict()
        assert isinstance(manifest_dict, dict)
        assert "run_id" in manifest_dict
        assert "git_sha" in manifest_dict
        assert "seed" in manifest_dict


class TestCreateSplits:
    """Tests for walk-forward split creation."""

    def test_creates_correct_number_of_splits(self) -> None:
        """Test correct number of splits created."""
        start = datetime(2023, 1, 3)
        end = datetime(2023, 3, 31)

        splits = create_splits(start, end, n_splits=3, is_ratio=0.7)

        assert len(splits) == 3

    def test_split_structure(self) -> None:
        """Test split tuple structure."""
        start = datetime(2023, 1, 3)
        end = datetime(2023, 3, 31)

        splits = create_splits(start, end, n_splits=2, is_ratio=0.7)

        for is_start, is_end, oos_start, oos_end in splits:
            assert is_start < is_end
            assert oos_start < oos_end
            assert is_end < oos_start  # IS ends before OOS starts

    def test_insufficient_sessions_raises(self) -> None:
        """Test error when not enough sessions for splits."""
        start = datetime(2023, 1, 3)
        end = datetime(2023, 1, 5)  # Very short range

        with pytest.raises(ValueError):
            create_splits(start, end, n_splits=10, is_ratio=0.7)


class TestCreateSyntheticData:
    """Tests for synthetic data creation."""

    def test_creates_data_for_all_tickers(self) -> None:
        """Test data created for all requested tickers."""
        tickers = ["AAPL", "MSFT", "NVDA"]
        start = datetime(2023, 1, 3)
        end = datetime(2023, 1, 31)

        data = create_synthetic_data(tickers, start, end)

        assert len(data) == 3
        assert "AAPL" in data
        assert "MSFT" in data
        assert "NVDA" in data

    def test_data_has_ohlcv_columns(self) -> None:
        """Test data has required OHLCV columns."""
        data = create_synthetic_data(["AAPL"], datetime(2023, 1, 3), datetime(2023, 1, 31))

        df = data["AAPL"]
        assert "date" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_deterministic_with_seed(self) -> None:
        """Test synthetic data is deterministic."""
        start = datetime(2023, 1, 3)
        end = datetime(2023, 1, 31)

        np.random.seed(42)
        data1 = create_synthetic_data(["AAPL"], start, end)

        np.random.seed(42)
        data2 = create_synthetic_data(["AAPL"], start, end)

        pd.testing.assert_frame_equal(data1["AAPL"], data2["AAPL"])


class TestRunWalkforward:
    """Tests for walk-forward execution."""

    def test_run_walkforward_creates_output(self, tmp_path: Path) -> None:
        """Test walk-forward creates output files."""
        np.random.seed(42)
        random.seed(42)

        result = run_walkforward(
            start_date="2023-01-03",
            end_date="2023-02-15",
            universe=["AAPL", "MSFT"],
            n_splits=2,
            is_ratio=0.6,
            output_dir=tmp_path,
        )

        assert "error" not in result
        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "equity_curve.csv").exists()
        assert (tmp_path / "run_manifest.json").exists()

    def test_run_walkforward_result_structure(self, tmp_path: Path) -> None:
        """Test walk-forward result has expected structure."""
        np.random.seed(42)
        random.seed(42)

        result = run_walkforward(
            start_date="2023-01-03",
            end_date="2023-02-15",
            universe=["AAPL"],
            n_splits=2,
            is_ratio=0.6,
            output_dir=tmp_path,
        )

        assert "start_date" in result
        assert "end_date" in result
        assert "universe" in result
        assert "n_splits" in result
        assert "splits" in result
        assert "avg_oos_return_pct" in result

    def test_results_json_valid(self, tmp_path: Path) -> None:
        """Test results.json is valid JSON."""
        np.random.seed(42)
        random.seed(42)

        run_walkforward(
            start_date="2023-01-03",
            end_date="2023-02-15",
            universe=["AAPL"],
            n_splits=2,
            is_ratio=0.6,
            output_dir=tmp_path,
        )

        with open(tmp_path / "results.json") as f:
            data = json.load(f)

        assert "splits" in data
        assert len(data["splits"]) == 2

    def test_equity_curve_csv_valid(self, tmp_path: Path) -> None:
        """Test equity_curve.csv is valid CSV."""
        np.random.seed(42)
        random.seed(42)

        run_walkforward(
            start_date="2023-01-03",
            end_date="2023-02-15",
            universe=["AAPL"],
            n_splits=2,
            is_ratio=0.6,
            output_dir=tmp_path,
        )

        df = pd.read_csv(tmp_path / "equity_curve.csv")

        assert "date" in df.columns or df.empty
        assert "equity" in df.columns or df.empty
