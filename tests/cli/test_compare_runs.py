"""Tests for run comparison CLI."""

import json
from pathlib import Path

import pandas as pd
import pytest

from traderbot.cli.compare_runs import compare_runs, compute_metrics, load_run_data


@pytest.fixture
def mock_run_a(tmp_path: Path) -> Path:
    """Create mock run A with sample data."""
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    
    # Create results.json
    results = {
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "universe": ["AAPL", "MSFT"],
        "n_splits": 2,
        "is_ratio": 0.6,
        "sizer": "fixed",
        "avg_oos_return_pct": 5.0,
        "avg_oos_sharpe": 1.2,
        "avg_oos_max_dd_pct": 3.0,
        "total_oos_trades": 10,
        "manifest": {
            "run_id": "2023-01-01T12-00-00",
            "git_sha": "abc1234",
            "seed": 42,
            "data_digest": "1234567890abcdef",
        },
    }
    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f)
    
    # Create equity_curve.csv
    equity_data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=20, freq="D"),
        "equity": [100000 + i * 250 for i in range(20)],  # Upward trend
        "cash": [50000] * 20,
        "split": [1] * 10 + [2] * 10,
    })
    equity_data.to_csv(run_dir / "equity_curve.csv", index=False)
    
    return run_dir


@pytest.fixture
def mock_run_b(tmp_path: Path) -> Path:
    """Create mock run B with different performance."""
    run_dir = tmp_path / "run_b"
    run_dir.mkdir()
    
    # Create results.json
    results = {
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "universe": ["AAPL", "MSFT"],
        "n_splits": 2,
        "is_ratio": 0.6,
        "sizer": "vol",
        "avg_oos_return_pct": 7.5,
        "avg_oos_sharpe": 1.5,
        "avg_oos_max_dd_pct": 2.5,
        "total_oos_trades": 15,
        "manifest": {
            "run_id": "2023-01-01T13-00-00",
            "git_sha": "abc1234",
            "seed": 99,
            "data_digest": "1234567890abcdef",
        },
    }
    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f)
    
    # Create equity_curve.csv with better performance
    equity_data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=20, freq="D"),
        "equity": [100000 + i * 375 for i in range(20)],  # Steeper upward trend
        "cash": [40000] * 20,
        "split": [1] * 10 + [2] * 10,
    })
    equity_data.to_csv(run_dir / "equity_curve.csv", index=False)
    
    return run_dir


class TestLoadRunData:
    """Tests for loading run data."""
    
    def test_load_run_data_success(self, mock_run_a: Path) -> None:
        """Test successful loading of run data."""
        results, equity_df = load_run_data(mock_run_a)
        
        assert isinstance(results, dict)
        assert isinstance(equity_df, pd.DataFrame)
        assert "universe" in results
        assert "equity" in equity_df.columns
    
    def test_load_run_data_missing_results(self, tmp_path: Path) -> None:
        """Test error when results.json is missing."""
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        
        with pytest.raises(FileNotFoundError, match="results.json"):
            load_run_data(run_dir)
    
    def test_load_run_data_missing_equity(self, tmp_path: Path) -> None:
        """Test error when equity_curve.csv is missing."""
        run_dir = tmp_path / "incomplete_run"
        run_dir.mkdir()
        
        # Create only results.json
        with open(run_dir / "results.json", "w") as f:
            json.dump({"test": "data"}, f)
        
        with pytest.raises(FileNotFoundError, match="equity_curve.csv"):
            load_run_data(run_dir)


class TestComputeMetrics:
    """Tests for metric computation."""
    
    def test_compute_metrics_basic(self) -> None:
        """Test basic metric computation."""
        equity_df = pd.DataFrame({
            "equity": [100000, 102000, 101000, 105000, 104000],
        })
        
        metrics = compute_metrics(equity_df)
        
        assert "total_return" in metrics
        assert "sharpe" in metrics
        assert "max_dd" in metrics
        assert metrics["total_return"] == pytest.approx(4.0, abs=0.1)  # 4% gain
    
    def test_compute_metrics_empty_dataframe(self) -> None:
        """Test metric computation with empty DataFrame."""
        equity_df = pd.DataFrame()
        
        metrics = compute_metrics(equity_df)
        
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe"] == 0.0
        assert metrics["max_dd"] == 0.0


class TestCompareRuns:
    """Tests for run comparison."""
    
    def test_compare_runs_generates_report(
        self, mock_run_a: Path, mock_run_b: Path
    ) -> None:
        """Test that comparison generates a report file."""
        output_path = mock_run_a.parent / "comparison.md"
        
        compare_runs(
            run_a=mock_run_a,
            run_b=mock_run_b,
            metric="sharpe",
            output_path=output_path,
        )
        
        assert output_path.exists()
    
    def test_comparison_report_contains_table(
        self, mock_run_a: Path, mock_run_b: Path
    ) -> None:
        """Test that comparison report contains comparison table."""
        output_path = mock_run_a.parent / "comparison.md"
        
        compare_runs(
            run_a=mock_run_a,
            run_b=mock_run_b,
            metric="total_return",
            output_path=output_path,
        )
        
        report_text = output_path.read_text()
        
        # Check for key sections
        assert "# Run Comparison Report" in report_text
        assert "## Run Details" in report_text
        assert "## Performance Comparison" in report_text
        assert "## Winner" in report_text
        
        # Check for metrics
        assert "Total Return" in report_text
        assert "Sharpe Ratio" in report_text
        assert "Max Drawdown" in report_text
    
    def test_comparison_report_shows_winner(
        self, mock_run_a: Path, mock_run_b: Path
    ) -> None:
        """Test that comparison report shows winner."""
        output_path = mock_run_a.parent / "comparison.md"
        
        compare_runs(
            run_a=mock_run_a,
            run_b=mock_run_b,
            metric="sharpe",
            output_path=output_path,
        )
        
        report_text = output_path.read_text()
        
        # Should show a winner (Run B has higher Sharpe in fixture)
        assert "Winner" in report_text
        assert ("Run A" in report_text or "Run B" in report_text or "Tie" in report_text)
    
    def test_compare_runs_different_metrics(
        self, mock_run_a: Path, mock_run_b: Path
    ) -> None:
        """Test comparison with different metrics."""
        for metric in ["total_return", "sharpe", "max_dd"]:
            output_path = mock_run_a.parent / f"comparison_{metric}.md"
            
            compare_runs(
                run_a=mock_run_a,
                run_b=mock_run_b,
                metric=metric,
                output_path=output_path,
            )
            
            assert output_path.exists()
            report_text = output_path.read_text()
            assert "Winner" in report_text

