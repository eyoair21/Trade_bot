"""Tests for history.json building and rolling statistics."""

import json
from pathlib import Path

import pytest


class TestHistoryBuilder:
    """Tests for history.json generation from summary.json files."""

    @pytest.fixture
    def reports_dir(self, tmp_path: Path) -> Path:
        """Create a mock reports directory with summary.json files."""
        reports = tmp_path / "reports"
        reports.mkdir()

        # Create several run directories with summary.json files
        test_runs = [
            {
                "run_id": "run-001",
                "verdict": "PASS",
                "sharpe_delta": 0.05,
                "timing_p90": 12.5,
                "git_sha": "abc1234",
                "generated_utc": "2026-01-01T10:00:00Z",
            },
            {
                "run_id": "run-002",
                "verdict": "FAIL",
                "sharpe_delta": -0.03,
                "timing_p90": 15.2,
                "git_sha": "def5678",
                "generated_utc": "2026-01-02T10:00:00Z",
            },
            {
                "run_id": "run-003",
                "verdict": "PASS",
                "sharpe_delta": 0.02,
                "timing_p90": 11.8,
                "git_sha": "ghi9012",
                "generated_utc": "2026-01-03T10:00:00Z",
            },
            {
                "run_id": "run-004",
                "verdict": "PASS",
                "sharpe_delta": 0.01,
                "timing_p90": 13.0,
                "git_sha": "jkl3456",
                "generated_utc": "2026-01-04T10:00:00Z",
            },
            {
                "run_id": "run-005",
                "verdict": "PASS",
                "sharpe_delta": 0.04,
                "timing_p90": 12.0,
                "git_sha": "mno7890",
                "generated_utc": "2026-01-05T10:00:00Z",
            },
        ]

        for run in test_runs:
            run_dir = reports / run["run_id"]
            run_dir.mkdir()
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(run, indent=2))

        return reports

    def test_build_history_from_summaries(self, reports_dir: Path) -> None:
        """Test building history from summary.json files."""
        # Import here to avoid import errors if module doesn't exist yet
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import build_history_from_summaries

        history = build_history_from_summaries(reports_dir, max_runs=5)

        assert history["schema_version"] == "1"  # String in implementation
        assert "generated_utc" in history
        assert history["window"] == 5
        assert len(history["runs"]) == 5

        # Check runs are sorted by ts_utc descending
        timestamps = [r["ts_utc"] for r in history["runs"]]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_compute_rolling_stats(self, reports_dir: Path) -> None:
        """Test rolling statistics computation."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import build_history_from_summaries, compute_rolling_stats

        history = build_history_from_summaries(reports_dir, max_runs=5)
        # compute_rolling_stats expects list of runs, not dict
        rolling = compute_rolling_stats(history["runs"], window=5)

        # Check rolling stats structure
        assert "sharpe_delta" in rolling
        assert "timing_p90" in rolling

        # Check sharpe_delta stats
        sd = rolling["sharpe_delta"]
        assert "mean" in sd
        assert "stdev" in sd

        # Mean of [0.04, 0.01, 0.02, -0.03, 0.05] = 0.018 (sorted by ts_utc desc)
        assert abs(sd["mean"] - 0.018) < 0.001

    def test_save_history_json(self, reports_dir: Path, tmp_path: Path) -> None:
        """Test saving history.json to disk."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import build_history_from_summaries, save_history_json

        history = build_history_from_summaries(reports_dir, max_runs=5)
        output_path = tmp_path / "history.json"

        save_history_json(history, output_path)

        assert output_path.exists()

        # Verify JSON is valid
        loaded = json.loads(output_path.read_text())
        assert loaded["schema_version"] == "1"  # String in implementation
        assert len(loaded["runs"]) == 5

    def test_empty_reports_dir(self, tmp_path: Path) -> None:
        """Test handling of empty reports directory."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import build_history_from_summaries

        empty_reports = tmp_path / "empty_reports"
        empty_reports.mkdir()

        history = build_history_from_summaries(empty_reports)

        assert history["schema_version"] == "1"  # String in implementation
        assert len(history["runs"]) == 0

    def test_malformed_summary_skipped(self, tmp_path: Path) -> None:
        """Test that malformed summary.json files are skipped."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from update_pages_index import build_history_from_summaries

        reports = tmp_path / "reports"
        reports.mkdir()

        # Create valid run
        valid_run = reports / "valid-run"
        valid_run.mkdir()
        (valid_run / "summary.json").write_text(json.dumps({
            "run_id": "valid-run",
            "verdict": "PASS",
            "sharpe_delta": 0.05,
            "generated_utc": "2026-01-01T10:00:00Z",
        }))

        # Create malformed run
        bad_run = reports / "bad-run"
        bad_run.mkdir()
        (bad_run / "summary.json").write_text("not valid json")

        history = build_history_from_summaries(reports)

        # Should only have the valid run
        assert len(history["runs"]) == 1
        assert history["runs"][0]["run_id"] == "valid-run"
