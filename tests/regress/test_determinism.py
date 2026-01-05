"""Tests for determinism checking in sweeps."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from traderbot.cli.sweep import rerun_best_for_determinism


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_best_run() -> dict:
    """Create a mock best run result."""
    return {
        "_run_idx": 5,
        "_config": {
            "start_date": "2023-01-01",
            "end_date": "2023-06-30",
            "universe": ["AAPL", "GOOGL"],
            "n_splits": 3,
            "seed": 42,
            "sizer": "fixed",
            "fixed_frac": 0.1,
        },
        "_elapsed_seconds": 15.0,
        "_output_dir": "/tmp/run_005",
        "_status": "success",
        "avg_oos_sharpe": 1.5,
        "avg_oos_return_pct": 12.0,
    }


@pytest.fixture
def mock_walkforward_result() -> dict:
    """Create a mock walk-forward result."""
    return {
        "avg_oos_sharpe": 1.5,
        "avg_oos_return_pct": 12.0,
        "avg_oos_max_dd_pct": -8.0,
        "total_oos_trades": 50,
    }


# ============================================================
# Test rerun_best_for_determinism
# ============================================================


class TestRerunBestForDeterminism:
    """Tests for rerun_best_for_determinism function."""

    def test_deterministic_reruns(
        self,
        mock_best_run: dict,
        mock_walkforward_result: dict,
        tmp_path: Path,
    ):
        """Test that deterministic reruns produce same results."""
        with patch(
            "traderbot.cli.sweep.run_walkforward",
            return_value=mock_walkforward_result,
        ):
            result = rerun_best_for_determinism(
                best_run=mock_best_run,
                output_root=tmp_path,
                n_reruns=3,
                metric="sharpe",
            )

        assert result["is_deterministic"] is True
        assert result["max_abs_diff"] < 1e-9
        assert len(result["rerun_values"]) == 3
        assert all(v == 1.5 for v in result["rerun_values"])

    def test_non_deterministic_reruns(
        self,
        mock_best_run: dict,
        tmp_path: Path,
    ):
        """Test detection of non-deterministic results."""
        # Return different values each call
        call_count = [0]
        values = [1.5, 1.51, 1.49]

        def mock_run(**_kwargs):
            result = {"avg_oos_sharpe": values[call_count[0] % len(values)]}
            call_count[0] += 1
            return result

        with patch("traderbot.cli.sweep.run_walkforward", side_effect=mock_run):
            result = rerun_best_for_determinism(
                best_run=mock_best_run,
                output_root=tmp_path,
                n_reruns=3,
                metric="sharpe",
            )

        assert result["is_deterministic"] is False
        assert result["max_abs_diff"] > 1e-9

    def test_writes_determinism_json(
        self,
        mock_best_run: dict,
        mock_walkforward_result: dict,
        tmp_path: Path,
    ):
        """Test that determinism.json is written."""
        with patch(
            "traderbot.cli.sweep.run_walkforward",
            return_value=mock_walkforward_result,
        ):
            rerun_best_for_determinism(
                best_run=mock_best_run,
                output_root=tmp_path,
                n_reruns=2,
                metric="sharpe",
            )

        det_path = tmp_path / "determinism.json"
        assert det_path.exists()

        det_data = json.loads(det_path.read_text())
        assert det_data["metric"] == "sharpe"
        assert det_data["n_reruns"] == 2
        assert "max_abs_diff" in det_data

    def test_creates_rerun_directories(
        self,
        mock_best_run: dict,
        mock_walkforward_result: dict,
        tmp_path: Path,
    ):
        """Test that rerun output directories are created."""
        with patch(
            "traderbot.cli.sweep.run_walkforward",
            return_value=mock_walkforward_result,
        ):
            rerun_best_for_determinism(
                best_run=mock_best_run,
                output_root=tmp_path,
                n_reruns=2,
                metric="sharpe",
            )

        det_dir = tmp_path / "determinism_checks"
        assert det_dir.exists()
        assert (det_dir / "rerun_000").exists()
        assert (det_dir / "rerun_001").exists()

    def test_handles_rerun_errors(
        self,
        mock_best_run: dict,
        tmp_path: Path,
    ):
        """Test graceful handling of errors during reruns."""
        with patch(
            "traderbot.cli.sweep.run_walkforward",
            side_effect=RuntimeError("Test error"),
        ):
            result = rerun_best_for_determinism(
                best_run=mock_best_run,
                output_root=tmp_path,
                n_reruns=2,
                metric="sharpe",
            )

        assert result["is_deterministic"] is False
        assert len(result["rerun_values"]) == 0
        assert all(r["status"] == "error" for r in result["reruns"])

    def test_uses_original_seed(
        self,
        mock_best_run: dict,
        mock_walkforward_result: dict,
        tmp_path: Path,
    ):
        """Test that original seed is used for all reruns."""
        captured_seeds = []

        def capture_seed(**kwargs):
            captured_seeds.append(kwargs.get("seed"))
            return mock_walkforward_result

        with patch("traderbot.cli.sweep.run_walkforward", side_effect=capture_seed):
            rerun_best_for_determinism(
                best_run=mock_best_run,
                output_root=tmp_path,
                n_reruns=3,
                metric="sharpe",
            )

        # All reruns should use same seed (42 from mock_best_run)
        assert all(s == 42 for s in captured_seeds)

    def test_different_metrics(
        self,
        mock_best_run: dict,
        tmp_path: Path,
    ):
        """Test determinism check with different metrics."""
        mock_result = {
            "avg_oos_sharpe": 1.5,
            "avg_oos_return_pct": 12.0,
            "avg_oos_max_dd_pct": -8.0,
        }

        for metric in ["sharpe", "total_return", "max_dd"]:
            with patch(
                "traderbot.cli.sweep.run_walkforward",
                return_value=mock_result,
            ):
                result = rerun_best_for_determinism(
                    best_run=mock_best_run,
                    output_root=tmp_path / metric,
                    n_reruns=1,
                    metric=metric,
                )

            assert result["metric"] == metric

    def test_result_statistics(
        self,
        mock_best_run: dict,
        tmp_path: Path,
    ):
        """Test that statistics are calculated correctly."""
        call_count = [0]
        values = [1.5, 1.52, 1.48]

        def mock_run(**_kwargs):
            result = {"avg_oos_sharpe": values[call_count[0]]}
            call_count[0] += 1
            return result

        with patch("traderbot.cli.sweep.run_walkforward", side_effect=mock_run):
            result = rerun_best_for_determinism(
                best_run=mock_best_run,
                output_root=tmp_path,
                n_reruns=3,
                metric="sharpe",
            )

        # Check statistics are present
        assert "mean_value" in result
        assert "std_value" in result
        assert result["std_value"] > 0  # Non-zero std for varying results

    def test_single_rerun(
        self,
        mock_best_run: dict,
        mock_walkforward_result: dict,
        tmp_path: Path,
    ):
        """Test with just one rerun."""
        with patch(
            "traderbot.cli.sweep.run_walkforward",
            return_value=mock_walkforward_result,
        ):
            result = rerun_best_for_determinism(
                best_run=mock_best_run,
                output_root=tmp_path,
                n_reruns=1,
                metric="sharpe",
            )

        assert result["n_reruns"] == 1
        assert len(result["rerun_values"]) == 1

    def test_preserves_config_in_result(
        self,
        mock_best_run: dict,
        mock_walkforward_result: dict,
        tmp_path: Path,
    ):
        """Test that original config is preserved in result."""
        with patch(
            "traderbot.cli.sweep.run_walkforward",
            return_value=mock_walkforward_result,
        ):
            result = rerun_best_for_determinism(
                best_run=mock_best_run,
                output_root=tmp_path,
                n_reruns=1,
                metric="sharpe",
            )

        assert result["config"] == mock_best_run["_config"]
        assert result["original_seed"] == 42
