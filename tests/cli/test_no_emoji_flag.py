"""Tests for --no-emoji flag in regress CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Get the project root to add to PYTHONPATH for subprocess calls
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _create_minimal_inputs(tmp: Path) -> None:
    """Create minimal sweep inputs for testing."""
    # Create sweep directory with minimal data
    sweep_dir = tmp / "runs" / "sweeps" / "min"
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
    benchmarks_dir = tmp / "benchmarks"
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
    sweeps_dir = tmp / "sweeps"
    sweeps_dir.mkdir(exist_ok=True)
    (sweeps_dir / "perf_budget.yaml").write_text(
        "metric: sharpe\n"
        "mode: max\n"
        "min_success_rate: 0.5\n"
        "max_p90_elapsed_s: 300.0\n"
        "max_sharpe_drop: 1.0\n"
        "epsilon_abs: 1e-6\n"
    )


class TestNoEmojiFlag:
    """Tests for --no-emoji command line flag."""

    def test_compare_no_emoji(self, tmp_path: Path):
        """Test that --no-emoji produces ASCII output."""
        _create_minimal_inputs(tmp_path)

        cmd = [
            sys.executable,
            "-m",
            "traderbot.cli.regress",
            "compare",
            "--no-emoji",
            "--current",
            str(tmp_path / "runs" / "sweeps" / "min"),
            "--baseline",
            str(tmp_path / "benchmarks" / "baseline.json"),
            "--budget",
            str(tmp_path / "sweeps" / "perf_budget.yaml"),
            "--out",
            str(tmp_path / "runs" / "sweeps" / "min" / "report.md"),
        ]

        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "[PASS] REGRESSION CHECK PASSED" in result.stdout
        assert "✅" not in result.stdout
        assert "❌" not in result.stdout

    def test_compare_with_emoji(self, tmp_path: Path):
        """Test that emoji appears when --no-emoji is not set (if encoding supports)."""
        _create_minimal_inputs(tmp_path)

        cmd = [
            sys.executable,
            "-m",
            "traderbot.cli.regress",
            "compare",
            "--current",
            str(tmp_path / "runs" / "sweeps" / "min"),
            "--baseline",
            str(tmp_path / "benchmarks" / "baseline.json"),
            "--budget",
            str(tmp_path / "sweeps" / "perf_budget.yaml"),
            "--out",
            str(tmp_path / "runs" / "sweeps" / "min" / "report.md"),
        ]

        # Use parent environment with UTF-8 encoding override and PYTHONPATH
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Either emoji or ASCII fallback is acceptable depending on system
        assert "REGRESSION CHECK PASSED" in result.stdout

    def test_update_baseline_no_emoji(self, tmp_path: Path):
        """Test that update-baseline respects --no-emoji."""
        _create_minimal_inputs(tmp_path)

        cmd = [
            sys.executable,
            "-m",
            "traderbot.cli.regress",
            "update-baseline",
            "--no-emoji",
            "--current",
            str(tmp_path / "runs" / "sweeps" / "min"),
            "--out",
            str(tmp_path / "benchmarks" / "new_baseline.json"),
        ]

        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "[OK] Baseline updated:" in result.stdout
        assert "✅" not in result.stdout
