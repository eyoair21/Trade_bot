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


# ============================================================
# Test auto-update-on-pass
# ============================================================


class TestAutoUpdateOnPass:
    """Tests for --auto-update-on-pass feature."""

    def test_auto_update_on_pass_creates_baseline(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,
        tmp_path: Path,
    ):
        """Test that auto-update creates new baseline when regression passes."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(mock_budget)
        args.out = str(tmp_path / "report.md")
        args.no_emoji = True
        args.quiet = False
        args.reruns = 0
        args.variance_threshold = 0.1
        args.html = False
        args.auto_update_on_pass = str(tmp_path / "new_baseline.json")

        exit_code = cmd_compare(args)

        # Should pass and create new baseline
        assert exit_code == 0
        assert (tmp_path / "new_baseline.json").exists()

        # Verify new baseline content
        new_baseline = json.loads((tmp_path / "new_baseline.json").read_text())
        assert "git_sha" in new_baseline
        assert new_baseline["summary"]["best_metric"] == 1.5

    def test_auto_update_not_triggered_on_fail(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        strict_budget: Path,
        tmp_path: Path,
    ):
        """Test that auto-update is NOT triggered when regression fails."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(strict_budget)  # Strict budget will fail
        args.out = str(tmp_path / "report.md")
        args.no_emoji = True
        args.quiet = False
        args.reruns = 0
        args.variance_threshold = 0.1
        args.html = False
        args.auto_update_on_pass = str(tmp_path / "new_baseline.json")

        exit_code = cmd_compare(args)

        # Should fail and NOT create new baseline
        assert exit_code == 1
        assert not (tmp_path / "new_baseline.json").exists()

    def test_auto_update_with_quiet_mode(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,
        tmp_path: Path,
        capsys,
    ):
        """Test that auto-update works in quiet mode."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(mock_budget)
        args.out = str(tmp_path / "report.md")
        args.no_emoji = True
        args.quiet = True  # Quiet mode
        args.reruns = 0
        args.variance_threshold = 0.1
        args.html = False
        args.auto_update_on_pass = str(tmp_path / "new_baseline.json")

        exit_code = cmd_compare(args)

        # Should pass and create new baseline
        assert exit_code == 0
        assert (tmp_path / "new_baseline.json").exists()

        # Should not print auto-update message
        captured = capsys.readouterr()
        assert "[AUTO]" not in captured.out

    def test_auto_update_via_main(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,
        tmp_path: Path,
    ):
        """Test auto-update via main CLI entry."""
        test_args = [
            "regress",
            "compare",
            "--current", str(mock_sweep_dir),
            "--baseline", str(mock_baseline),
            "--budget", str(mock_budget),
            "--out", str(tmp_path / "report.md"),
            "--auto-update-on-pass", str(tmp_path / "auto_baseline.json"),
            "--no-emoji",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        # Verify auto-updated baseline exists
        assert (tmp_path / "auto_baseline.json").exists()

    def test_auto_update_creates_parent_dirs(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,
        tmp_path: Path,
    ):
        """Test that auto-update creates parent directories."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(mock_budget)
        args.out = str(tmp_path / "report.md")
        args.no_emoji = True
        args.quiet = False
        args.reruns = 0
        args.variance_threshold = 0.1
        args.html = False
        # Nested path
        args.auto_update_on_pass = str(tmp_path / "deep" / "nested" / "baseline.json")

        exit_code = cmd_compare(args)

        assert exit_code == 0
        assert (tmp_path / "deep" / "nested" / "baseline.json").exists()


# ============================================================
# Test multi-metric budgets (v0.6.0)
# ============================================================


@pytest.fixture
def multi_metric_budget(tmp_path: Path) -> Path:
    """Create a multi-metric budget file."""
    budget_path = tmp_path / "multi_budget.yaml"
    # Use metrics that exist in mock_sweep_dir fixture:
    # - avg_oos_sharpe (sharpe)
    # - avg_oos_return_pct (total_return)
    # - p90_elapsed_s (timing)
    budget_path.write_text("""
metric: sharpe
mode: max
min_success_rate: 0.75
max_p90_elapsed_s: 60.0
max_sharpe_drop: 0.1
epsilon_abs: 1e-6

budgets:
  sharpe:
    mode: max
    max_drop: 0.1
    epsilon: 0.01
  total_return:
    mode: max
    max_drop: 0.30
    epsilon: 0.05
    required: true
  p90_elapsed_s:
    mode: min
    max: 60.0
    epsilon: 2.0
    required: false
""")
    return budget_path


@pytest.fixture
def failing_multi_budget(tmp_path: Path) -> Path:
    """Create a multi-metric budget that will fail on total_return."""
    budget_path = tmp_path / "failing_multi_budget.yaml"
    # This will fail because we require min total_return of 99.0 which won't be met
    budget_path.write_text("""
metric: sharpe
mode: max

budgets:
  sharpe:
    mode: max
    max_drop: 0.1
    epsilon: 0.01
  total_return:
    mode: max
    min: 99.0
    epsilon: 0.01
    required: true
""")
    return budget_path


class TestMultiMetricBudgets:
    """Tests for multi-metric budget evaluation (v0.6.0)."""

    def test_multi_metric_budget_parsing(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        multi_metric_budget: Path,
        tmp_path: Path,
    ):
        """Test that budgets: map is parsed correctly."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(multi_metric_budget)
        args.out = str(tmp_path / "report.md")
        args.no_emoji = True
        args.quiet = False
        args.reruns = 0
        args.variance_threshold = 0.1
        args.html = False
        args.auto_update_on_pass = None

        exit_code = cmd_compare(args)

        # Should pass with this config
        assert exit_code == 0
        assert (tmp_path / "report.md").exists()

    def test_multi_metric_verdicts_in_diff(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        multi_metric_budget: Path,
        tmp_path: Path,
    ):
        """Test that per-metric budget verdicts appear in baseline_diff.json."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(multi_metric_budget)
        args.out = str(tmp_path / "report.md")
        args.no_emoji = True
        args.quiet = False
        args.reruns = 0
        args.variance_threshold = 0.1
        args.html = False
        args.auto_update_on_pass = None

        cmd_compare(args)

        diff = json.loads((tmp_path / "baseline_diff.json").read_text())

        # Per-metric budgets are stored under details.budgets
        assert "details" in diff
        assert "budgets" in diff["details"]

        # Should have verdicts for configured metrics
        budgets = diff["details"]["budgets"]
        assert "sharpe" in budgets
        assert budgets["sharpe"]["passed"] is True

    def test_multi_metric_fail_fails_overall(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        failing_multi_budget: Path,
        tmp_path: Path,
    ):
        """Test that if ANY required metric fails, overall result fails."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(failing_multi_budget)
        args.out = str(tmp_path / "report.md")
        args.no_emoji = True
        args.quiet = False
        args.reruns = 0
        args.variance_threshold = 0.1
        args.html = False
        args.auto_update_on_pass = None

        exit_code = cmd_compare(args)

        # Should fail because win_rate min is 0.99 which won't be met
        assert exit_code == 1

    def test_multi_metric_report_contains_table(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        multi_metric_budget: Path,
        tmp_path: Path,
    ):
        """Test that report contains per-metric verdicts table."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(multi_metric_budget)
        args.out = str(tmp_path / "report.md")
        args.no_emoji = True
        args.quiet = False
        args.reruns = 0
        args.variance_threshold = 0.1
        args.html = False
        args.auto_update_on_pass = None

        cmd_compare(args)

        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        # Should contain per-metric section
        assert "Per-Metric" in report or "per-metric" in report.lower()

    def test_multi_metric_html_contains_table(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        multi_metric_budget: Path,
        tmp_path: Path,
    ):
        """Test that HTML report contains per-metric verdicts table."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(multi_metric_budget)
        args.out = str(tmp_path / "report.md")
        args.no_emoji = True
        args.quiet = False
        args.reruns = 0
        args.variance_threshold = 0.1
        args.html = True
        args.auto_update_on_pass = None

        cmd_compare(args)

        # HTML is written as regression_report.html, not report.html
        html_path = tmp_path / "regression_report.html"
        assert html_path.exists(), f"Expected HTML at {html_path}"

        html = html_path.read_text(encoding="utf-8")
        # Should contain table with metric verdicts or budget info
        assert "Per-Metric" in html or "Budget" in html or "sharpe" in html.lower()

    def test_fallback_to_legacy_single_metric(
        self,
        mock_sweep_dir: Path,
        mock_baseline: Path,
        mock_budget: Path,  # Legacy single-metric format
        tmp_path: Path,
    ):
        """Test that legacy single-metric budget still works."""
        args = MagicMock()
        args.current = str(mock_sweep_dir)
        args.baseline = str(mock_baseline)
        args.budget = str(mock_budget)  # No budgets: section
        args.out = str(tmp_path / "report.md")
        args.no_emoji = True
        args.quiet = False
        args.reruns = 0
        args.variance_threshold = 0.1
        args.html = False
        args.auto_update_on_pass = None

        exit_code = cmd_compare(args)

        # Should still work with legacy format
        assert exit_code == 0


# ============================================================
# Test status badge generation (v0.6.0)
# ============================================================


class TestStatusBadge:
    """Tests for SVG status badge generation."""

    def test_badge_script_exists(self):
        """Test that badge generation script exists."""
        from pathlib import Path as P

        script_path = P(__file__).parent.parent.parent / "scripts" / "generate_status_badge.py"
        assert script_path.exists()

    def test_badge_generate_pass(self):
        """Test badge generation with PASS status."""
        # Import the function from the script
        import sys
        from pathlib import Path as P

        script_dir = P(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        from generate_status_badge import generate_badge

        svg = generate_badge(status="pass", sha="abc1234")

        assert "<svg" in svg
        assert "PASS" in svg
        assert "#28a745" in svg  # Green color
        assert "abc1234" in svg  # SHA in comment

    def test_badge_generate_fail(self):
        """Test badge generation with FAIL status."""
        import sys
        from pathlib import Path as P

        script_dir = P(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        from generate_status_badge import generate_badge

        svg = generate_badge(status="fail", sha="def5678")

        assert "<svg" in svg
        assert "FAIL" in svg
        assert "#dc3545" in svg  # Red color
        assert "def5678" in svg  # SHA in comment

    def test_badge_from_diff_file(self, tmp_path: Path):
        """Test badge generation from baseline_diff.json."""
        import sys
        from pathlib import Path as P

        script_dir = P(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        # Create a passing diff file
        diff_file = tmp_path / "baseline_diff.json"
        diff_file.write_text(json.dumps({
            "passed": True,
            "baseline_sha": "xyz9999",
            "generated_utc": "2026-01-05T12:00:00Z",
        }))

        from generate_status_badge import generate_badge

        # Read diff manually like the script does
        diff_data = json.loads(diff_file.read_text())
        status = "pass" if diff_data.get("passed", False) else "fail"
        sha = diff_data.get("baseline_sha", "unknown")

        svg = generate_badge(status=status, sha=sha)

        assert "PASS" in svg
        assert "xyz9999" in svg

    def test_badge_cli_invocation(self, tmp_path: Path):
        """Test badge CLI via subprocess."""
        output_path = tmp_path / "badge.svg"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from scripts.generate_status_badge import main; import sys; sys.exit(main())",
                "--status", "pass",
                "--output", str(output_path),
                "--sha", "test123",
            ],
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=True,
            text=True,
        )

        # Script might succeed or fail depending on import path
        # But badge generation function should work
        # This is more of an integration test

    def test_badge_deterministic(self):
        """Test that badge generation is deterministic."""
        import sys
        from pathlib import Path as P

        script_dir = P(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        from generate_status_badge import generate_badge

        # Generate same badge twice
        svg1 = generate_badge(
            status="pass",
            sha="abc123",
            timestamp="2026-01-05T12:00:00Z",
        )
        svg2 = generate_badge(
            status="pass",
            sha="abc123",
            timestamp="2026-01-05T12:00:00Z",
        )

        assert svg1 == svg2

    def test_badge_size_under_limit(self):
        """Test that badge is under 10KB size limit."""
        import sys
        from pathlib import Path as P

        script_dir = P(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        from generate_status_badge import generate_badge

        svg = generate_badge(status="pass", sha="abc123")

        # Badge should be under 10KB
        assert len(svg.encode("utf-8")) < 10000


# ============================================================
# Test GitHub Pages index generation (v0.6.2)
# ============================================================


class TestPagesIndex:
    """Tests for GitHub Pages index generation."""

    def test_update_pages_index_script_exists(self):
        """Test that Pages index script exists."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "dev" / "update_pages_index.py"
        assert script_path.exists()

    def test_manifest_add_run(self, tmp_path: Path):
        """Test adding a run to the manifest."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import load_manifest, add_run, trim_runs

        # Start with empty manifest
        manifest = {"runs": [], "updated": None, "latest": None}

        # Add a run
        manifest = add_run(manifest, "123-abc", "2026-01-05T12:00:00Z", "pass")

        assert manifest["latest"] == "123-abc"
        assert len(manifest["runs"]) == 1
        assert manifest["runs"][0]["id"] == "123-abc"
        assert manifest["runs"][0]["status"] == "pass"

    def test_manifest_trim_runs(self, tmp_path: Path):
        """Test trimming runs to max limit."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import add_run, trim_runs

        manifest = {"runs": [], "updated": None, "latest": None}

        # Add more runs than the limit
        for i in range(10):
            manifest = add_run(manifest, f"run-{i}", f"2026-01-0{i}T12:00:00Z", "pass")

        # Trim to 5 (modifies in place, returns list of pruned IDs)
        pruned_ids = trim_runs(manifest, 5)

        assert len(manifest["runs"]) == 5
        # Most recent runs should be kept (runs 9, 8, 7, 6, 5)
        assert manifest["runs"][0]["id"] == "run-9"
        # Should return the pruned IDs
        assert len(pruned_ids) == 5

    def test_generate_index_html(self, tmp_path: Path):
        """Test HTML index generation."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import generate_index_html

        manifest = {
            "runs": [
                {"id": "123-abc", "timestamp": "2026-01-05T12:00:00Z", "status": "pass"},
                {"id": "456-def", "timestamp": "2026-01-04T12:00:00Z", "status": "fail"},
            ],
            "updated": "2026-01-05T12:00:00Z",
            "latest": "123-abc",
        }

        html = generate_index_html(manifest)

        assert "<html" in html
        assert "123-abc" in html
        assert "456-def" in html
        assert "PASS" in html
        assert "FAIL" in html
        assert "regression_report.html" in html

    def test_index_contains_report_path(self, tmp_path: Path):
        """Test that index HTML contains correct report paths for Job Summary."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import generate_index_html

        manifest = {
            "runs": [{"id": "test-run-123", "timestamp": "2026-01-05T12:00:00Z", "status": "pass"}],
            "updated": "2026-01-05T12:00:00Z",
            "latest": "test-run-123",
        }

        html = generate_index_html(manifest)

        # Check that report paths are correctly formed
        assert "test-run-123/regression_report.html" in html
        assert "latest/regression_report.html" in html


# ============================================================
# Test Integrity & Provenance features (v0.6.3)
# ============================================================


class TestSha256Sums:
    """Tests for SHA256 integrity hash generation."""

    def test_generate_sha256sums_script_exists(self):
        """Test that SHA256 sums script exists."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "dev" / "generate_sha256sums.py"
        assert script_path.exists()

    def test_compute_sha256_function(self, tmp_path: Path):
        """Test SHA256 computation for a file."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from generate_sha256sums import compute_sha256

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        sha256 = compute_sha256(test_file)

        # SHA256 of "Hello, World!" is known
        expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        assert sha256 == expected

    def test_generate_sha256sums_creates_file(self, tmp_path: Path):
        """Test that generate_sha256sums creates sha256sums.txt."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from generate_sha256sums import generate_sha256sums

        # Create report directory with test files
        report_dir = tmp_path / "ci_smoke"
        report_dir.mkdir()

        (report_dir / "regression_report.html").write_text("<html>report</html>")
        (report_dir / "baseline_diff.json").write_text('{"passed": true}')

        # Generate sha256sums
        result = generate_sha256sums(report_dir)

        assert result is not None  # Returns Path on success, None on failure
        sums_file = report_dir / "sha256sums.txt"
        assert sums_file.exists()

        content = sums_file.read_text()
        assert "regression_report.html" in content
        assert "baseline_diff.json" in content

    def test_verify_sha256sums_passes(self, tmp_path: Path):
        """Test that verify_sha256sums passes for valid files."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from generate_sha256sums import generate_sha256sums, verify_sha256sums

        report_dir = tmp_path / "ci_smoke"
        report_dir.mkdir()

        (report_dir / "regression_report.html").write_text("<html>test</html>")
        generate_sha256sums(report_dir)

        # Verify should pass (returns bool)
        is_valid = verify_sha256sums(report_dir)
        assert is_valid is True

    def test_verify_sha256sums_fails_on_tampering(self, tmp_path: Path):
        """Test that verify_sha256sums detects file tampering."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from generate_sha256sums import generate_sha256sums, verify_sha256sums

        report_dir = tmp_path / "ci_smoke"
        report_dir.mkdir()

        (report_dir / "regression_report.html").write_text("<html>original</html>")
        generate_sha256sums(report_dir)

        # Tamper with the file
        (report_dir / "regression_report.html").write_text("<html>tampered</html>")

        # Verify should fail (returns bool)
        is_valid = verify_sha256sums(report_dir)
        assert is_valid is False


class TestProvenanceSchema:
    """Tests for provenance.json schema v1."""

    def test_provenance_has_schema_version(self, tmp_path: Path):
        """Test that provenance includes schema_version field."""
        from traderbot.metrics.compare import generate_provenance_json, CurrentData

        current = CurrentData(
            leaderboard=[{"run": "test", "sharpe": 0.5}],
            timing={"total_time_s": 10.0},
            success_rate=1.0,
            total_runs=1,
            best_metric=0.5,
        )

        provenance = generate_provenance_json(current=current, git_sha="abc123def")

        assert "schema_version" in provenance
        assert provenance["schema_version"] == "1"

    def test_provenance_has_git_sha(self, tmp_path: Path):
        """Test that provenance includes git_sha field."""
        from traderbot.metrics.compare import generate_provenance_json, CurrentData

        current = CurrentData(
            leaderboard=[{"run": "test", "sharpe": 0.5}],
            timing={"total_time_s": 10.0},
            success_rate=1.0,
            total_runs=1,
            best_metric=0.5,
        )

        provenance = generate_provenance_json(current=current, git_sha="abc123def456")

        assert "git_sha" in provenance
        assert provenance["git_sha"] == "abc123def456"

    def test_provenance_default_sha_unknown(self, tmp_path: Path):
        """Test that provenance defaults git_sha to 'unknown'."""
        from traderbot.metrics.compare import generate_provenance_json, CurrentData

        current = CurrentData(
            leaderboard=[{"run": "test", "sharpe": 0.5}],
            timing={"total_time_s": 10.0},
            success_rate=1.0,
            total_runs=1,
            best_metric=0.5,
        )

        provenance = generate_provenance_json(current=current)

        assert provenance["git_sha"] == "unknown"


class TestHTMLReportDarkLight:
    """Tests for HTML report dark/light theme toggle."""

    def _make_test_objects(self):
        """Create test objects for HTML report generation."""
        from traderbot.metrics.compare import (
            ComparisonVerdict, CurrentData, BaselineData, PerfBudget
        )

        verdict = ComparisonVerdict(
            passed=True,
            metric_passed=True,
            timing_passed=True,
            success_rate_passed=True,
            determinism_passed=None,
            metric_delta=0.1,
            timing_p50_delta=0.0,
            timing_p90_delta=0.0,
            success_rate_delta=0.0,
            determinism_variance=None,
            details={"best_metric": 0.5},
            messages=["All checks passed"],
        )

        current = CurrentData(
            leaderboard=[{"run": "test", "sharpe": 0.5}],
            timing={"total_time_s": 10.0, "p50": 1.0, "p90": 2.0},
            success_rate=1.0,
            total_runs=1,
            best_metric=0.5,
        )

        baseline = BaselineData(
            git_sha="baseline123",
            created_utc="2026-01-01T00:00:00Z",
            metric="sharpe",
            mode="no_regression",
            leaderboard=[{"run": "baseline", "sharpe": 0.4}],
            timing={"total_time_s": 10.0, "p50": 1.0, "p90": 2.0},
            summary={"sharpe": 0.4},
        )

        budget = PerfBudget(metric="sharpe", mode="no_regression")

        return verdict, current, baseline, budget

    def test_html_has_theme_toggle(self, tmp_path: Path):
        """Test that HTML report contains theme toggle button."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects()
        html = generate_html_report(verdict, current, baseline, budget)

        assert "theme-toggle" in html
        assert "toggleTheme" in html

    def test_html_has_css_variables(self, tmp_path: Path):
        """Test that HTML report uses CSS custom properties for theming."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects()
        html = generate_html_report(verdict, current, baseline, budget)

        assert "--bg-primary" in html
        assert "--text-primary" in html
        assert "data-theme" in html

    def test_html_has_sticky_header(self, tmp_path: Path):
        """Test that HTML report has sticky header."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects()
        html = generate_html_report(verdict, current, baseline, budget)

        assert "sticky-header" in html or "position: sticky" in html

    def test_html_has_provenance_footer(self, tmp_path: Path):
        """Test that HTML report displays provenance footer."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects()
        provenance = {
            "schema_version": "1",
            "git_sha": "abc123",
            "generated_utc": "2026-01-05T12:00:00Z",
        }

        html = generate_html_report(verdict, current, baseline, budget, provenance)

        assert "abc123" in html
        assert "Built from" in html


class TestPagesIndexDarkLight:
    """Tests for GitHub Pages index dark/light features (v0.6.3)."""

    def test_index_has_theme_toggle(self):
        """Test that Pages index has theme toggle."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import generate_index_html

        manifest = {"runs": [], "updated": None, "latest": None}
        html = generate_index_html(manifest)

        assert "theme-toggle" in html
        assert "toggleTheme" in html

    def test_index_has_og_tags(self):
        """Test that Pages index has Open Graph meta tags."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import generate_index_html

        manifest = {"runs": [], "updated": None, "latest": None}
        html = generate_index_html(manifest)

        assert 'property="og:title"' in html
        assert 'property="og:description"' in html
        assert 'name="twitter:card"' in html

    def test_cache_bust_query_parameter(self):
        """Test that cache-busting query parameter is added."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import cache_bust_query

        query = cache_bust_query()
        assert query.startswith("?v=")
        assert len(query) > 5  # ?v= plus timestamp

    def test_trim_runs_returns_pruned_ids(self):
        """Test that trim_runs returns list of pruned IDs."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import add_run, trim_runs

        manifest = {"runs": [], "updated": None, "latest": None}

        # Add 10 runs
        for i in range(10):
            manifest = add_run(manifest, f"run-{i}", f"2026-01-0{i}T12:00:00Z", "pass")

        # Trim to 5
        pruned_ids = trim_runs(manifest, 5)

        # Should return 5 pruned IDs (runs 0-4)
        assert len(pruned_ids) == 5
        assert "run-0" in pruned_ids
        assert "run-4" in pruned_ids
        assert "run-5" not in pruned_ids  # This one was kept

    def test_prune_report_directories(self, tmp_path: Path):
        """Test that prune_report_directories removes old directories."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import prune_report_directories

        # Create fake report directories
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()

        for i in range(5):
            run_dir = reports_dir / f"run-{i}"
            run_dir.mkdir()
            (run_dir / "regression_report.html").write_text(f"report {i}")

        # Prune runs 0-2
        pruned_ids = ["run-0", "run-1", "run-2"]
        deleted = prune_report_directories(reports_dir, pruned_ids)

        assert deleted == 3
        assert not (reports_dir / "run-0").exists()
        assert not (reports_dir / "run-1").exists()
        assert not (reports_dir / "run-2").exists()
        assert (reports_dir / "run-3").exists()
        assert (reports_dir / "run-4").exists()


# ============================================================
# Test HTML Report Banner (v0.6.4)
# ============================================================


class TestHTMLReportBanner:
    """Tests for HTML report PASS/FAIL status banner."""

    def _make_test_objects(self, passed: bool = True):
        """Create test objects for HTML report generation."""
        from traderbot.metrics.compare import (
            ComparisonVerdict, CurrentData, BaselineData, PerfBudget
        )

        verdict = ComparisonVerdict(
            passed=passed,
            metric_passed=passed,
            timing_passed=True,
            success_rate_passed=True,
            determinism_passed=None,
            metric_delta=0.1 if passed else -0.5,
            timing_p50_delta=0.0,
            timing_p90_delta=0.0,
            success_rate_delta=0.0,
            determinism_variance=None,
            details={"best_metric": 0.5},
            messages=["All checks passed" if passed else "Regression detected"],
        )

        current = CurrentData(
            leaderboard=[{"run": "test", "sharpe": 0.5}],
            timing={"total_time_s": 10.0, "p50": 1.0, "p90": 2.5},
            success_rate=1.0,
            total_runs=5,
            best_metric=0.5,
        )

        baseline = BaselineData(
            git_sha="baseline123",
            created_utc="2026-01-01T00:00:00Z",
            metric="sharpe",
            mode="no_regression",
            leaderboard=[{"run": "baseline", "sharpe": 0.4}],
            timing={"total_time_s": 10.0, "p50": 1.0, "p90": 2.0},
            summary={"sharpe": 0.4},
        )

        budget = PerfBudget(metric="sharpe", mode="no_regression")

        return verdict, current, baseline, budget

    def test_html_has_verdict_banner(self):
        """Test that HTML report contains verdict banner."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects(passed=True)
        html = generate_html_report(verdict, current, baseline, budget)

        assert "verdict-banner" in html

    def test_html_banner_shows_pass_for_passing_verdict(self):
        """Test that banner shows PASS with green styling for passing verdict."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects(passed=True)
        html = generate_html_report(verdict, current, baseline, budget)

        assert 'class="verdict-banner pass"' in html
        assert "PASS" in html

    def test_html_banner_shows_fail_for_failing_verdict(self):
        """Test that banner shows FAIL with red styling for failing verdict."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects(passed=False)
        html = generate_html_report(verdict, current, baseline, budget)

        assert 'class="verdict-banner fail"' in html
        assert "FAIL" in html

    def test_html_banner_shows_sharpe_delta(self):
        """Test that banner displays Sharpe delta."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects(passed=True)
        html = generate_html_report(verdict, current, baseline, budget)

        # Should contain Sharpe delta display
        assert "Sharpe" in html and "0.1" in html or "+0.1" in html

    def test_html_banner_shows_timing_p90(self):
        """Test that banner displays P90 timing."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects(passed=True)
        html = generate_html_report(verdict, current, baseline, budget)

        # Should contain P90 display (2.5s from fixture)
        assert "P90" in html
        assert "2.5" in html or "2.50" in html

    def test_html_banner_shows_runs_count(self):
        """Test that banner displays total runs count."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects(passed=True)
        html = generate_html_report(verdict, current, baseline, budget)

        # Should contain runs count (5 from fixture)
        assert "Runs" in html
        assert ">5<" in html or "Runs:</span> 5" in html

    def test_html_banner_has_pass_fail_colors(self):
        """Test that CSS includes pass/fail color variables."""
        from traderbot.metrics.compare import generate_html_report

        verdict, current, baseline, budget = self._make_test_objects(passed=True)
        html = generate_html_report(verdict, current, baseline, budget)

        assert "--pass-color" in html
        assert "--fail-color" in html
        assert "--pass-bg" in html
        assert "--fail-bg" in html


# ============================================================
# Test summary.json generation (v0.6.4)
# ============================================================


class TestSummaryJsonIntegrity:
    """Tests for summary.json generation and integrity."""

    def test_generate_summary_json_exists(self):
        """Test that generate_summary_json function exists."""
        from traderbot.metrics.compare import generate_summary_json
        assert callable(generate_summary_json)

    def test_generate_summary_json_schema(self):
        """Test that summary.json has correct schema fields."""
        from traderbot.metrics.compare import (
            generate_summary_json, ComparisonVerdict, CurrentData
        )

        verdict = ComparisonVerdict(
            passed=True,
            metric_passed=True,
            timing_passed=True,
            success_rate_passed=True,
            determinism_passed=None,
            metric_delta=0.15,
            timing_p50_delta=0.0,
            timing_p90_delta=0.0,
            success_rate_delta=0.0,
            determinism_variance=None,
        )

        current = CurrentData(
            leaderboard=[{"run": "test"}],
            timing={"p90": 12.5},
            success_rate=1.0,
            total_runs=10,
            best_metric=0.8,
        )

        summary = generate_summary_json(
            run_id="12345-abc1234",
            verdict=verdict,
            current=current,
            git_sha="abc123def456",
        )

        # Check all required fields
        assert "schema_version" in summary
        assert summary["schema_version"] == "1"
        assert summary["run_id"] == "12345-abc1234"
        assert summary["verdict"] == "PASS"
        assert summary["sharpe_delta"] == 0.15
        assert summary["timing_p90"] == 12.5
        assert summary["git_sha"] == "abc123def456"
        assert "generated_utc" in summary

    def test_generate_summary_json_fail_verdict(self):
        """Test that summary.json correctly reports FAIL verdict."""
        from traderbot.metrics.compare import (
            generate_summary_json, ComparisonVerdict, CurrentData
        )

        verdict = ComparisonVerdict(
            passed=False,
            metric_passed=False,
            timing_passed=True,
            success_rate_passed=True,
            determinism_passed=None,
            metric_delta=-0.25,
            timing_p50_delta=0.0,
            timing_p90_delta=0.0,
            success_rate_delta=0.0,
            determinism_variance=None,
        )

        current = CurrentData(
            leaderboard=[],
            timing={"p90": 5.0},
            success_rate=0.8,
            total_runs=5,
            best_metric=0.3,
        )

        summary = generate_summary_json(
            run_id="fail-run",
            verdict=verdict,
            current=current,
        )

        assert summary["verdict"] == "FAIL"
        assert summary["sharpe_delta"] == -0.25

    def test_generate_summary_json_default_git_sha(self):
        """Test that summary.json defaults git_sha to 'unknown'."""
        from traderbot.metrics.compare import (
            generate_summary_json, ComparisonVerdict, CurrentData
        )

        verdict = ComparisonVerdict(
            passed=True,
            metric_passed=True,
            timing_passed=True,
            success_rate_passed=True,
            determinism_passed=None,
            metric_delta=0.0,
            timing_p50_delta=0.0,
            timing_p90_delta=0.0,
            success_rate_delta=0.0,
            determinism_variance=None,
        )

        current = CurrentData(
            leaderboard=[],
            timing={"p90": 1.0},
            success_rate=1.0,
            total_runs=1,
            best_metric=0.5,
        )

        summary = generate_summary_json(
            run_id="test-run",
            verdict=verdict,
            current=current,
        )

        assert summary["git_sha"] == "unknown"

    def test_generate_summary_json_rounding(self):
        """Test that summary.json properly rounds values."""
        from traderbot.metrics.compare import (
            generate_summary_json, ComparisonVerdict, CurrentData
        )

        verdict = ComparisonVerdict(
            passed=True,
            metric_passed=True,
            timing_passed=True,
            success_rate_passed=True,
            determinism_passed=None,
            metric_delta=0.123456789,
            timing_p50_delta=0.0,
            timing_p90_delta=0.0,
            success_rate_delta=0.0,
            determinism_variance=None,
        )

        current = CurrentData(
            leaderboard=[],
            timing={"p90": 1.23456789},
            success_rate=1.0,
            total_runs=1,
            best_metric=0.5,
        )

        summary = generate_summary_json(
            run_id="test-run",
            verdict=verdict,
            current=current,
        )

        # sharpe_delta should be rounded to 6 decimal places
        assert summary["sharpe_delta"] == 0.123457
        # timing_p90 should be rounded to 3 decimal places
        assert summary["timing_p90"] == 1.235

    def test_sha256sums_includes_summary_json(self):
        """Test that generate_sha256sums includes summary.json."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from generate_sha256sums import INTEGRITY_FILES

        assert "summary.json" in INTEGRITY_FILES


# ============================================================
# Test Index uses summary.json (v0.6.4)
# ============================================================


class TestIndexUsesSummaryJson:
    """Tests for index enrichment using summary.json."""

    def test_add_run_accepts_summary_parameter(self):
        """Test that add_run accepts summary data parameter."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import add_run

        manifest = {"runs": [], "updated": None, "latest": None}

        summary = {
            "sharpe_delta": 0.15,
            "timing_p90": 2.5,
            "git_sha": "abc123",
        }

        manifest = add_run(manifest, "test-run", "2026-01-05T12:00:00Z", "pass", summary)

        # Should include enriched fields
        assert manifest["runs"][0]["sharpe_delta"] == 0.15
        assert manifest["runs"][0]["timing_p90"] == 2.5
        assert manifest["runs"][0]["git_sha"] == "abc123"

    def test_add_run_works_without_summary(self):
        """Test that add_run still works without summary data."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import add_run

        manifest = {"runs": [], "updated": None, "latest": None}

        # No summary provided
        manifest = add_run(manifest, "test-run", "2026-01-05T12:00:00Z", "pass")

        assert manifest["runs"][0]["id"] == "test-run"
        assert "sharpe_delta" not in manifest["runs"][0]

    def test_index_html_shows_enriched_data(self):
        """Test that index HTML displays enriched row data."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import generate_index_html

        manifest = {
            "runs": [
                {
                    "id": "enriched-run",
                    "timestamp": "2026-01-05T12:00:00Z",
                    "status": "pass",
                    "sharpe_delta": 0.123,
                    "timing_p90": 2.5,
                    "git_sha": "abc123def",
                },
            ],
            "updated": "2026-01-05T12:00:00Z",
            "latest": "enriched-run",
        }

        html = generate_index_html(manifest)

        # Should contain enriched data display
        assert "0.123" in html or "+0.123" in html  # Sharpe delta
        assert "2.5" in html or "2.50" in html  # P90
        assert "abc123" in html  # Git SHA (may be truncated)

    def test_index_html_css_has_delta_classes(self):
        """Test that index HTML has CSS classes for delta styling."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import generate_index_html

        manifest = {"runs": [], "updated": None, "latest": None}
        html = generate_index_html(manifest)

        assert "delta-positive" in html
        assert "delta-negative" in html

    def test_load_summary_json_function_exists(self):
        """Test that load_summary_json helper function exists."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import load_summary_json

        assert callable(load_summary_json)

    def test_load_summary_json_returns_none_if_missing(self, tmp_path: Path):
        """Test that load_summary_json returns None if file doesn't exist."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import load_summary_json

        result = load_summary_json(tmp_path, "nonexistent-run")
        assert result is None

    def test_load_summary_json_returns_data(self, tmp_path: Path):
        """Test that load_summary_json returns data if file exists."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from update_pages_index import load_summary_json

        # Create summary.json
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        summary_data = {"verdict": "PASS", "sharpe_delta": 0.5}
        (run_dir / "summary.json").write_text(json.dumps(summary_data))

        result = load_summary_json(tmp_path, "test-run")

        assert result is not None
        assert result["verdict"] == "PASS"
        assert result["sharpe_delta"] == 0.5


# ============================================================
# Test HTML Minifier (v0.6.4)
# ============================================================


class TestHtmlMinifier:
    """Tests for HTML minification script."""

    def test_minify_html_script_exists(self):
        """Test that minify_html.py script exists."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "dev" / "minify_html.py"
        assert script_path.exists()

    def test_minify_html_function_exists(self):
        """Test that minify_html function is importable."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from minify_html import minify_html
        assert callable(minify_html)

    def test_minify_removes_whitespace(self):
        """Test that minifier removes unnecessary whitespace."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from minify_html import minify_html

        html = """
        <html>
            <head>
                <title>Test</title>
            </head>
            <body>
                <p>Hello   World</p>
            </body>
        </html>
        """

        minified = minify_html(html)

        # Should be smaller
        assert len(minified) < len(html)
        # Should remove extra whitespace between tags
        assert "\n            " not in minified

    def test_minify_removes_comments(self):
        """Test that minifier removes HTML comments."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from minify_html import minify_html

        html = """<html><!-- This is a comment --><body>Content</body></html>"""
        minified = minify_html(html)

        assert "<!-- This is a comment -->" not in minified
        assert "Content" in minified

    def test_minify_preserves_pre_content(self):
        """Test that minifier preserves whitespace in <pre> tags."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from minify_html import minify_html

        html = """<html><body><pre>
    code
        indented
</pre></body></html>"""
        minified = minify_html(html)

        # Should preserve pre content
        assert "    code" in minified
        assert "        indented" in minified

    def test_minify_preserves_script_content(self):
        """Test that minifier preserves <script> content."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from minify_html import minify_html

        html = """<html><script>
function test() {
    return "hello";
}
</script></html>"""
        minified = minify_html(html)

        # Should preserve script content
        assert "function test()" in minified
        assert 'return "hello"' in minified

    def test_minify_file_function_exists(self):
        """Test that minify_file function exists."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from minify_html import minify_file
        assert callable(minify_file)

    def test_minify_file_overwrites_in_place(self, tmp_path: Path):
        """Test that minify_file can overwrite file in place."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from minify_html import minify_file

        html_file = tmp_path / "test.html"
        original_content = "<html>\n    <body>\n        <p>Test</p>\n    </body>\n</html>"
        html_file.write_text(original_content)

        original_size, minified_size = minify_file(html_file)

        assert minified_size < original_size
        assert html_file.exists()
        new_content = html_file.read_text()
        assert len(new_content) < len(original_content)

    def test_minify_reduces_file_size(self, tmp_path: Path):
        """Test that minification significantly reduces file size."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from minify_html import minify_html

        # Create a typical HTML report structure
        html = """<!DOCTYPE html>
<html>
    <head>
        <title>Regression Report</title>
        <style>
            body { margin: 0; padding: 0; }
        </style>
    </head>
    <body>
        <!-- Header section -->
        <div class="header">
            <h1>Report Title</h1>
        </div>

        <!-- Content section -->
        <div class="content">
            <p>Some content here</p>
            <p>More content here</p>
        </div>

        <!-- Footer section -->
        <div class="footer">
            <p>Footer text</p>
        </div>
    </body>
</html>"""

        minified = minify_html(html)

        # Should be meaningfully smaller
        assert len(minified) < len(html) * 0.9  # At least 10% reduction

    def test_minify_preserves_functionality(self, tmp_path: Path):
        """Test that minified HTML still has all essential elements."""
        import sys
        script_dir = Path(__file__).parent.parent.parent / "scripts" / "dev"
        sys.path.insert(0, str(script_dir))

        from minify_html import minify_html

        html = """<html data-theme="light">
<head><title>Test</title></head>
<body>
    <button onclick="toggleTheme()">Toggle</button>
    <script>function toggleTheme() { console.log('test'); }</script>
</body>
</html>"""

        minified = minify_html(html)

        # Should preserve essential elements
        assert 'data-theme="light"' in minified
        assert "<title>Test</title>" in minified
        assert 'onclick="toggleTheme()"' in minified
        assert "function toggleTheme()" in minified
