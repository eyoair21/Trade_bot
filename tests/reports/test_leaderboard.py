"""Tests for leaderboard generation."""

import json
from pathlib import Path

import pytest

from traderbot.reports.leaderboard import (
    build_leaderboard_from_sweep,
    export_best_run,
    generate_leaderboard,
    load_sweep_results,
    save_leaderboard_csv,
    save_leaderboard_markdown,
)


@pytest.fixture
def sample_results() -> list[dict]:
    """Sample run results for testing."""
    return [
        {
            "_run_idx": 0,
            "_status": "success",
            "_config": {"sizer": "fixed", "proba_threshold": 0.5},
            "_output_dir": "/tmp/run_000",
            "_elapsed_seconds": 10.5,
            "avg_oos_sharpe": 0.5,
            "avg_oos_return_pct": 5.0,
            "avg_oos_max_dd_pct": -3.0,
            "total_oos_trades": 50,
        },
        {
            "_run_idx": 1,
            "_status": "success",
            "_config": {"sizer": "vol", "proba_threshold": 0.5},
            "_output_dir": "/tmp/run_001",
            "_elapsed_seconds": 12.3,
            "avg_oos_sharpe": 0.8,
            "avg_oos_return_pct": 8.0,
            "avg_oos_max_dd_pct": -4.0,
            "total_oos_trades": 45,
        },
        {
            "_run_idx": 2,
            "_status": "success",
            "_config": {"sizer": "fixed", "proba_threshold": 0.6},
            "_output_dir": "/tmp/run_002",
            "_elapsed_seconds": 11.1,
            "avg_oos_sharpe": 0.6,
            "avg_oos_return_pct": 6.0,
            "avg_oos_max_dd_pct": -2.5,
            "total_oos_trades": 40,
        },
        {
            "_run_idx": 3,
            "_status": "error",
            "_config": {"sizer": "kelly"},
            "_output_dir": "/tmp/run_003",
            "error": "Some error",
        },
    ]


@pytest.fixture
def sweep_directory(tmp_path: Path, sample_results: list[dict]) -> Path:
    """Create a mock sweep directory with metadata and results."""
    sweep_dir = tmp_path / "test_sweep"
    sweep_dir.mkdir()

    # Create sweep metadata
    meta = {
        "name": "test_sweep",
        "metric": "sharpe",
        "mode": "max",
        "total_runs": 4,
        "workers": 2,
        "fixed_args": {"start_date": "2023-01-01"},
        "grid": {"sizer": ["fixed", "vol"]},
    }
    with open(sweep_dir / "sweep_meta.json", "w") as f:
        json.dump(meta, f)

    # Create results
    with open(sweep_dir / "all_results.json", "w") as f:
        json.dump(sample_results, f)

    # Create run directories for export tests
    for result in sample_results:
        run_dir = sweep_dir / f"run_{result['_run_idx']:03d}"
        run_dir.mkdir()
        # Create some files in run directory
        (run_dir / "results.json").write_text(json.dumps(result))
        (run_dir / "report.md").write_text("# Test Report")

    return sweep_dir


class TestLoadSweepResults:
    """Tests for loading sweep results."""

    def test_load_results(self, sweep_directory: Path) -> None:
        """Test loading sweep results."""
        meta, results = load_sweep_results(sweep_directory)

        assert meta["name"] == "test_sweep"
        assert meta["metric"] == "sharpe"
        assert len(results) == 4

    def test_load_missing_meta(self, tmp_path: Path) -> None:
        """Test error when metadata is missing."""
        with pytest.raises(FileNotFoundError, match="metadata"):
            load_sweep_results(tmp_path)

    def test_load_missing_results(self, tmp_path: Path) -> None:
        """Test error when results file is missing."""
        meta_path = tmp_path / "sweep_meta.json"
        meta_path.write_text("{}")

        with pytest.raises(FileNotFoundError, match="results"):
            load_sweep_results(tmp_path)


class TestGenerateLeaderboard:
    """Tests for leaderboard generation."""

    def test_generate_sharpe_max(self, sample_results: list[dict]) -> None:
        """Test generating leaderboard by sharpe (max)."""
        leaderboard = generate_leaderboard(sample_results, "sharpe", "max")

        assert len(leaderboard) == 3  # Excludes error run
        assert leaderboard[0]["rank"] == 1
        assert leaderboard[0]["avg_oos_sharpe"] == 0.8  # Highest sharpe
        assert leaderboard[1]["avg_oos_sharpe"] == 0.6
        assert leaderboard[2]["avg_oos_sharpe"] == 0.5

    def test_generate_max_dd_min(self, sample_results: list[dict]) -> None:
        """Test generating leaderboard by max_dd (min)."""
        leaderboard = generate_leaderboard(sample_results, "max_dd", "min")

        assert len(leaderboard) == 3
        # Most negative max_dd should be first when minimizing
        assert leaderboard[0]["avg_oos_max_dd_pct"] == -4.0

    def test_generate_with_top_n(self, sample_results: list[dict]) -> None:
        """Test limiting to top N entries."""
        leaderboard = generate_leaderboard(sample_results, "sharpe", "max", top_n=2)

        assert len(leaderboard) == 2
        assert leaderboard[0]["rank"] == 1
        assert leaderboard[1]["rank"] == 2

    def test_generate_unknown_metric(self, sample_results: list[dict]) -> None:
        """Test error for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            generate_leaderboard(sample_results, "unknown", "max")

    def test_generate_skips_errors(self, sample_results: list[dict]) -> None:
        """Test that error runs are excluded."""
        leaderboard = generate_leaderboard(sample_results, "sharpe", "max")

        run_indices = [e["run_idx"] for e in leaderboard]
        assert 3 not in run_indices  # Error run

    def test_generate_empty_results(self) -> None:
        """Test handling empty results."""
        leaderboard = generate_leaderboard([], "sharpe", "max")
        assert leaderboard == []


class TestSaveLeaderboardCSV:
    """Tests for CSV output."""

    def test_save_csv(self, tmp_path: Path, sample_results: list[dict]) -> None:
        """Test saving leaderboard to CSV."""
        leaderboard = generate_leaderboard(sample_results, "sharpe", "max")
        csv_path = tmp_path / "leaderboard.csv"

        save_leaderboard_csv(leaderboard, csv_path, "sharpe")

        assert csv_path.exists()
        content = csv_path.read_text()
        assert "rank" in content
        assert "avg_oos_sharpe" in content
        assert "sizer" in content  # Config column

    def test_save_csv_empty_leaderboard(self, tmp_path: Path) -> None:
        """Test saving empty leaderboard doesn't crash."""
        csv_path = tmp_path / "leaderboard.csv"
        save_leaderboard_csv([], csv_path, "sharpe")
        assert not csv_path.exists()


class TestSaveLeaderboardMarkdown:
    """Tests for Markdown output."""

    def test_save_markdown(self, tmp_path: Path, sample_results: list[dict]) -> None:
        """Test saving leaderboard to Markdown."""
        leaderboard = generate_leaderboard(sample_results, "sharpe", "max")
        md_path = tmp_path / "leaderboard.md"

        save_leaderboard_markdown(leaderboard, md_path, "test_sweep", "sharpe", "max")

        assert md_path.exists()
        content = md_path.read_text()
        assert "# Sweep Leaderboard" in content
        assert "test_sweep" in content
        assert "Rankings" in content
        assert "Best Run Configuration" in content

    def test_save_markdown_empty_leaderboard(self, tmp_path: Path) -> None:
        """Test saving empty leaderboard doesn't crash."""
        md_path = tmp_path / "leaderboard.md"
        save_leaderboard_markdown([], md_path, "test", "sharpe", "max")
        assert not md_path.exists()


class TestExportBestRun:
    """Tests for best run export."""

    def test_export_best(
        self, sweep_directory: Path, sample_results: list[dict], tmp_path: Path
    ) -> None:
        """Test exporting best run."""
        leaderboard = generate_leaderboard(sample_results, "sharpe", "max")

        # Update output_dir to actual sweep directory paths
        for entry in leaderboard:
            entry["output_dir"] = str(sweep_directory / f"run_{entry['run_idx']:03d}")

        export_dir = tmp_path / "exported_best"
        result = export_best_run(leaderboard, export_dir, rank=1)

        assert result is not None
        assert export_dir.exists()
        assert (export_dir / "export_manifest.json").exists()
        assert (export_dir / "results.json").exists()

    def test_export_specific_rank(
        self, sweep_directory: Path, sample_results: list[dict], tmp_path: Path
    ) -> None:
        """Test exporting specific rank."""
        leaderboard = generate_leaderboard(sample_results, "sharpe", "max")

        for entry in leaderboard:
            entry["output_dir"] = str(sweep_directory / f"run_{entry['run_idx']:03d}")

        export_dir = tmp_path / "exported_rank3"
        result = export_best_run(leaderboard, export_dir, rank=3)

        assert result is not None

        # Verify manifest
        with open(export_dir / "export_manifest.json") as f:
            manifest = json.load(f)
        assert manifest["exported_rank"] == 3

    def test_export_invalid_rank(self, sample_results: list[dict], tmp_path: Path) -> None:
        """Test error for invalid rank."""
        leaderboard = generate_leaderboard(sample_results, "sharpe", "max")

        result = export_best_run(leaderboard, tmp_path / "export", rank=999)
        assert result is None

    def test_export_empty_leaderboard(self, tmp_path: Path) -> None:
        """Test error for empty leaderboard."""
        result = export_best_run([], tmp_path / "export", rank=1)
        assert result is None


class TestBuildLeaderboardFromSweep:
    """Tests for full leaderboard build."""

    def test_build_full_leaderboard(self, sweep_directory: Path) -> None:
        """Test building complete leaderboard."""
        leaderboard = build_leaderboard_from_sweep(sweep_directory)

        assert len(leaderboard) == 3
        assert (sweep_directory / "leaderboard.csv").exists()
        assert (sweep_directory / "leaderboard.md").exists()

    def test_build_with_custom_output(
        self, sweep_directory: Path, tmp_path: Path
    ) -> None:
        """Test building leaderboard with custom output dir."""
        output_dir = tmp_path / "custom_output"
        leaderboard = build_leaderboard_from_sweep(sweep_directory, output_dir)

        assert len(leaderboard) == 3
        assert (output_dir / "leaderboard.csv").exists()
        assert (output_dir / "leaderboard.md").exists()

    def test_build_with_top_n(self, sweep_directory: Path) -> None:
        """Test building leaderboard with top N limit."""
        leaderboard = build_leaderboard_from_sweep(sweep_directory, top_n=2)

        assert len(leaderboard) == 2
