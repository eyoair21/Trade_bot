"""Tests for Data Provenance section in regression reports."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Get the project root to add to PYTHONPATH for subprocess calls
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _create_minimal_sweep(tmp_path: Path) -> Path:
    """Create minimal sweep with only all_results.json (no CSV files)."""
    sweep_dir = tmp_path / "runs" / "sweeps" / "min"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    (sweep_dir / "sweep_meta.json").write_text(
        json.dumps({"metric": "sharpe", "mode": "max"})
    )
    (sweep_dir / "all_results.json").write_text(
        json.dumps(
            [
                {
                    "_status": "success",
                    "_run_idx": 0,
                    "_elapsed_seconds": 10.0,
                    "avg_oos_sharpe": 1.1,
                },
                {
                    "_status": "success",
                    "_run_idx": 1,
                    "_elapsed_seconds": 12.0,
                    "avg_oos_sharpe": 1.4,
                },
            ]
        )
    )

    # Create baseline
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)
    (benchmarks_dir / "baseline.json").write_text(
        json.dumps(
            {
                "git_sha": "abc1234",
                "created_utc": "2026-01-01T00:00:00+00:00",
                "metric": "sharpe",
                "mode": "max",
                "leaderboard": [],
                "timing": {"p50": 100, "p90": 200},
                "summary": {"best_metric": 1.0, "success_rate": 1.0, "total_runs": 1},
            }
        )
    )

    # Create budget
    sweeps_dir = tmp_path / "sweeps"
    sweeps_dir.mkdir(exist_ok=True)
    (sweeps_dir / "perf_budget.yaml").write_text(
        "metric: sharpe\n"
        "mode: max\n"
        "min_success_rate: 0.5\n"
        "max_p90_elapsed_s: 300.0\n"
        "max_sharpe_drop: 1.0\n"
        "epsilon_abs: 1e-6\n"
    )

    return sweep_dir


class TestProvenanceBlock:
    """Tests for Data Provenance section."""

    def test_provenance_block_appears_when_fallback_used(self, tmp_path: Path):
        """Test that provenance section appears when CSVs are missing."""
        sweep_dir = _create_minimal_sweep(tmp_path)
        out_md = sweep_dir / "report.md"

        cmd = [
            sys.executable,
            "-m",
            "traderbot.cli.regress",
            "compare",
            "--no-emoji",
            "--current",
            str(sweep_dir),
            "--baseline",
            str(tmp_path / "benchmarks" / "baseline.json"),
            "--budget",
            str(tmp_path / "sweeps" / "perf_budget.yaml"),
            "--out",
            str(out_md),
        ]

        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        text = out_md.read_text(encoding="utf-8")
        assert "### Data Provenance" in text
        assert "best_metric" in text and "all_results.json" in text
        assert "Timing percentiles" in text or "timing" in text.lower()

    def test_no_provenance_when_csv_present(self, tmp_path: Path):
        """Test that provenance section is absent when CSV files exist."""
        sweep_dir = _create_minimal_sweep(tmp_path)

        # Add leaderboard.csv
        (sweep_dir / "leaderboard.csv").write_text(
            "rank,run_idx,avg_oos_sharpe,avg_oos_return_pct,avg_oos_max_dd_pct\n"
            "1,1,1.4,10.0,-5.0\n"
            "2,0,1.1,8.0,-4.0\n"
        )

        # Add timings.csv
        (sweep_dir / "timings.csv").write_text("run_idx,elapsed_s\n0,10.0\n1,12.0\n")

        out_md = sweep_dir / "report.md"

        cmd = [
            sys.executable,
            "-m",
            "traderbot.cli.regress",
            "compare",
            "--no-emoji",
            "--current",
            str(sweep_dir),
            "--baseline",
            str(tmp_path / "benchmarks" / "baseline.json"),
            "--budget",
            str(tmp_path / "sweeps" / "perf_budget.yaml"),
            "--out",
            str(out_md),
        ]

        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        text = out_md.read_text(encoding="utf-8")
        assert "### Data Provenance" not in text

    def test_provenance_only_leaderboard_fallback(self, tmp_path: Path):
        """Test provenance when only leaderboard is missing but timings exist."""
        sweep_dir = _create_minimal_sweep(tmp_path)

        # Add timings.csv but no leaderboard.csv
        (sweep_dir / "timings.csv").write_text("run_idx,elapsed_s\n0,10.0\n1,12.0\n")

        out_md = sweep_dir / "report.md"

        cmd = [
            sys.executable,
            "-m",
            "traderbot.cli.regress",
            "compare",
            "--no-emoji",
            "--current",
            str(sweep_dir),
            "--baseline",
            str(tmp_path / "benchmarks" / "baseline.json"),
            "--budget",
            str(tmp_path / "sweeps" / "perf_budget.yaml"),
            "--out",
            str(out_md),
        ]

        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        text = out_md.read_text(encoding="utf-8")
        assert "### Data Provenance" in text
        assert "best_metric" in text
        # Should NOT mention timing fallback since timings.csv exists
        assert "Timing percentiles" not in text

    def test_provenance_only_timings_fallback(self, tmp_path: Path):
        """Test provenance when only timings is missing but leaderboard exists."""
        sweep_dir = _create_minimal_sweep(tmp_path)

        # Add leaderboard.csv but no timings.csv
        (sweep_dir / "leaderboard.csv").write_text(
            "rank,run_idx,avg_oos_sharpe,avg_oos_return_pct,avg_oos_max_dd_pct\n"
            "1,1,1.4,10.0,-5.0\n"
            "2,0,1.1,8.0,-4.0\n"
        )

        out_md = sweep_dir / "report.md"

        cmd = [
            sys.executable,
            "-m",
            "traderbot.cli.regress",
            "compare",
            "--no-emoji",
            "--current",
            str(sweep_dir),
            "--baseline",
            str(tmp_path / "benchmarks" / "baseline.json"),
            "--budget",
            str(tmp_path / "sweeps" / "perf_budget.yaml"),
            "--out",
            str(out_md),
        ]

        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        text = out_md.read_text(encoding="utf-8")
        assert "### Data Provenance" in text
        assert "Timing percentiles" in text
        # Should NOT mention leaderboard fallback since leaderboard.csv exists
        assert "best_metric" not in text.split("### Data Provenance")[1]
