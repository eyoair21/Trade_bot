"""Tests for traderbot.cli.regress CLI commands."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from traderbot.cli.regress import cmd_compare, cmd_update_baseline, get_git_sha, main


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_sweep_dir(tmp_path: Path) -> Path:
    """Create a mock sweep output directory."""
    sweep_dir = tmp_path / "sweep_output"
    sweep_dir.mkdir()

    # Create sweep_meta.json
    sweep_meta = {
        "name": "test_sweep",
        "metric": "sharpe",
        "mode": "max",
        "total_runs": 3,
    }
    (sweep_dir / "sweep_meta.json").write_text(json.dumps(sweep_meta))

    # Create all_results.json
    all_results = [
        {
            "_status": "success",
            "_run_idx": 0,
            "_elapsed_seconds": 15.0,
            "avg_oos_sharpe": 1.2,
            "avg_oos_return_pct": 10.0,
            "avg_oos_max_dd_pct": -5.0,
            "total_oos_trades": 50,
        },
        {
            "_status": "success",
            "_run_idx": 1,
            "_elapsed_seconds": 18.0,
            "avg_oos_sharpe": 1.5,
            "avg_oos_return_pct": 12.0,
            "avg_oos_max_dd_pct": -8.0,
            "total_oos_trades": 60,
        },
        {
            "_status": "success",
            "_run_idx": 2,
            "_elapsed_seconds": 20.0,
            "avg_oos_sharpe": 1.3,
            "avg_oos_return_pct": 11.0,
            "avg_oos_max_dd_pct": -6.0,
            "total_oos_trades": 55,
        },
    ]
    (sweep_dir / "all_results.json").write_text(json.dumps(all_results))

    # Create leaderboard.csv (sorted by best sharpe)
    leaderboard_csv = """rank,run_idx,avg_oos_sharpe,avg_oos_return_pct,avg_oos_max_dd_pct,total_oos_trades
1,1,1.5,12.0,-8.0,60
2,2,1.3,11.0,-6.0,55
3,0,1.2,10.0,-5.0,50
"""
    (sweep_dir / "leaderboard.csv").write_text(leaderboard_csv)

    # Create timings.csv
    timings_csv = """run_idx,elapsed_s
0,15.0
1,18.0
2,20.0
"""
    (sweep_dir / "timings.csv").write_text(timings_csv)

    return sweep_dir


@pytest.fixture
def mock_baseline(tmp_path: Path) -> Path:
    """Create a mock baseline file."""
    baseline_path = tmp_path / "baseline.json"
    baseline_data = {
        "git_sha": "abc1234",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "metric": "sharpe",
        "mode": "max",
        "leaderboard": [
            {"rank": 1, "avg_oos_sharpe": 1.4},
        ],
        "timing": {"p50": 15.0, "p90": 25.0},
        "summary": {
            "best_metric": 1.4,
            "success_rate": 1.0,
            "total_runs": 1,
        },
    }
    baseline_path.write_text(json.dumps(baseline_data))
    return baseline_path


@pytest.fixture
def mock_budget(tmp_path: Path) -> Path:
    """Create a mock performance budget file."""
    budget_path = tmp_path / "perf_budget.yaml"
    budget_path.write_text("""
metric: sharpe
mode: max
min_success_rate: 0.75
max_p90_elapsed_s: 60.0
max_sharpe_drop: 0.1
epsilon_abs: 1e-6
""")
    return budget_path


@pytest.fixture
def strict_budget(tmp_path: Path) -> Path:
    """Create a strict budget that will fail."""
    budget_path = tmp_path / "strict_budget.yaml"
    budget_path.write_text("""
metric: sharpe
mode: max
min_success_rate: 0.99
max_p90_elapsed_s: 5.0
max_sharpe_drop: 0.001
epsilon_abs: 1e-9
""")
    return budget_path


# ============================================================
# Test get_git_sha
# ============================================================


class TestGetGitSha:
    """Tests for get_git_sha function."""

    def test_returns_string(self):
        """Test that get_git_sha returns a string."""
        sha = get_git_sha()
        assert isinstance(sha, str)

    def test_returns_unknown_on_failure(self):
        """Test that get_git_sha returns 'unknown' when git fails."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            sha = get_git_sha()
            assert sha == "unknown"

    def test_returns_unknown_on_subprocess_error(self):
        """Test handling of subprocess error."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            sha = get_git_sha()
            assert sha == "unknown"


# ============================================================
# Test cmd_compare
# ============================================================


class TestCmdCompare:
    """Tests for cmd_compare command."""

    def test_compare_pass(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,
        tmp_path: Path,
    ):
        """Test comparison that passes."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(mock_budget)
        args.out = str(tmp_path / "report.md")

        exit_code = cmd_compare(args)

        # Should pass - current best (1.5) is better than baseline (1.4)
        assert exit_code == 0
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "baseline_diff.json").exists()

    def test_compare_fail_with_strict_budget(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        strict_budget: Path,
        tmp_path: Path,
    ):
        """Test comparison that fails with strict budget."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(strict_budget)
        args.out = str(tmp_path / "report.md")

        exit_code = cmd_compare(args)

        # Should fail - budget is too strict
        assert exit_code == 1
        assert (tmp_path / "report.md").exists()

    def test_compare_missing_current_dir(
        self,
        mock_baseline: Path,
        mock_budget: Path,
        tmp_path: Path,
    ):
        """Test error when current directory doesn't exist."""
        args = MagicMock()
        args.current = str(tmp_path / "nonexistent")
        args.baseline = str(mock_baseline)
        args.budget = str(mock_budget)
        args.out = None

        exit_code = cmd_compare(args)
        assert exit_code == 1

    def test_compare_missing_baseline(
        self,
        mock_sweep_dir: Path,
        mock_budget: Path,
        tmp_path: Path,
    ):
        """Test error when baseline file doesn't exist."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(tmp_path / "nonexistent.json")
        args.budget = str(mock_budget)
        args.out = None

        exit_code = cmd_compare(args)
        assert exit_code == 1

    def test_compare_missing_budget(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        tmp_path: Path,
    ):
        """Test error when budget file doesn't exist."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(tmp_path / "nonexistent.yaml")
        args.out = None

        exit_code = cmd_compare(args)
        assert exit_code == 1

    def test_compare_default_output(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,
    ):
        """Test default output path."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(mock_budget)
        args.out = None  # Use default

        cmd_compare(args)

        # Default output should be in current dir
        assert (mock_sweep_dir / "regression_report.md").exists()


# ============================================================
# Test cmd_update_baseline
# ============================================================


class TestCmdUpdateBaseline:
    """Tests for cmd_update_baseline command."""

    def test_update_baseline(self, mock_sweep_dir: Path, tmp_path: Path):
        """Test baseline update command."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.out = str(tmp_path / "new_baseline.json")
        args.sha = "test123"

        exit_code = cmd_update_baseline(args)

        assert exit_code == 0
        assert (tmp_path / "new_baseline.json").exists()

        # Verify baseline content
        baseline = json.loads((tmp_path / "new_baseline.json").read_text())
        assert baseline["git_sha"] == "test123"
        assert baseline["summary"]["best_metric"] == 1.5

    def test_update_baseline_auto_sha(self, mock_sweep_dir: Path, tmp_path: Path):
        """Test baseline update with auto git SHA detection."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.out = str(tmp_path / "new_baseline.json")
        args.sha = None  # Auto-detect

        exit_code = cmd_update_baseline(args)

        assert exit_code == 0
        baseline = json.loads((tmp_path / "new_baseline.json").read_text())
        assert "git_sha" in baseline

    def test_update_baseline_missing_dir(self, tmp_path: Path):
        """Test error when source directory doesn't exist."""
        args = MagicMock()
        args.current = str(tmp_path / "nonexistent")
        args.out = str(tmp_path / "baseline.json")
        args.sha = "test123"

        exit_code = cmd_update_baseline(args)
        assert exit_code == 1


# ============================================================
# Test main CLI entry
# ============================================================


class TestMain:
    """Tests for main CLI entry point."""

    def test_main_compare_command(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,
        tmp_path: Path,
    ):
        """Test main with compare command."""
        test_args = [
            "regress",
            "compare",
            "--current", str(mock_sweep_dir),
            "--baseline", str(mock_baseline),
            "--budget", str(mock_budget),
            "--out", str(tmp_path / "report.md"),
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Should pass
            assert exc_info.value.code == 0

    def test_main_update_baseline_command(
        self,
        mock_sweep_dir: Path,
        tmp_path: Path,
    ):
        """Test main with update-baseline command."""
        test_args = [
            "regress",
            "update-baseline",
            "--current", str(mock_sweep_dir),
            "--out", str(tmp_path / "baseline.json"),
            "--sha", "manual123",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_no_command(self):
        """Test main with no command shows help."""
        test_args = ["regress"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


# ============================================================
# Test report content
# ============================================================


class TestReportContent:
    """Tests for generated report content."""

    def test_report_contains_status(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,
        tmp_path: Path,
    ):
        """Test that report contains pass/fail status."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(mock_budget)
        args.out = str(tmp_path / "report.md")

        cmd_compare(args)

        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "PASS" in report or "FAIL" in report

    def test_report_contains_metrics(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,
        tmp_path: Path,
    ):
        """Test that report contains metric information."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(mock_budget)
        args.out = str(tmp_path / "report.md")

        cmd_compare(args)

        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "sharpe" in report.lower() or "Sharpe" in report

    def test_diff_json_structure(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,
        tmp_path: Path,
    ):
        """Test baseline diff JSON structure."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(mock_budget)
        args.out = str(tmp_path / "report.md")

        cmd_compare(args)

        diff = json.loads((tmp_path / "baseline_diff.json").read_text())
        # Check for nested structure: deltas contains metric, timing, etc.
        assert "deltas" in diff
        assert "metric" in diff["deltas"]
        assert "timing_p90" in diff["deltas"]
        assert "passed" in diff
        assert "verdicts" in diff
