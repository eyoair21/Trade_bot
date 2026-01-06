"""Tests for reports summarize CLI command."""

import json
from pathlib import Path

import pytest


class TestReportsSummarizeCLI:
    """Tests for the reports summarize command."""

    @pytest.fixture
    def sample_history(self) -> dict:
        """Sample history.json data."""
        return {
            "schema_version": 1,
            "generated_utc": "2026-01-05T12:00:00Z",
            "window": 5,
            "runs": [
                {
                    "run_id": "run-005",
                    "verdict": "PASS",
                    "sharpe_delta": 0.04,
                    "timing_p90": 12.0,
                    "generated_utc": "2026-01-05T10:00:00Z",
                },
                {
                    "run_id": "run-004",
                    "verdict": "PASS",
                    "sharpe_delta": 0.01,
                    "timing_p90": 13.0,
                    "generated_utc": "2026-01-04T10:00:00Z",
                },
                {
                    "run_id": "run-003",
                    "verdict": "FAIL",
                    "sharpe_delta": -0.03,
                    "timing_p90": 15.2,
                    "generated_utc": "2026-01-03T10:00:00Z",
                },
                {
                    "run_id": "run-002",
                    "verdict": "PASS",
                    "sharpe_delta": 0.02,
                    "timing_p90": 11.8,
                    "generated_utc": "2026-01-02T10:00:00Z",
                },
                {
                    "run_id": "run-001",
                    "verdict": "PASS",
                    "sharpe_delta": 0.05,
                    "timing_p90": 12.5,
                    "generated_utc": "2026-01-01T10:00:00Z",
                },
            ],
        }

    @pytest.fixture
    def history_file(self, tmp_path: Path, sample_history: dict) -> Path:
        """Create history.json file."""
        history_path = tmp_path / "history.json"
        history_path.write_text(json.dumps(sample_history, indent=2))
        return history_path

    def test_load_history(self, history_file: Path) -> None:
        """Test loading history.json."""
        from traderbot.cli.reports import load_history

        history = load_history(history_file)

        assert history["schema_version"] == 1
        assert len(history["runs"]) == 5

    def test_filter_runs_limit(self, sample_history: dict) -> None:
        """Test filtering runs with limit."""
        from traderbot.cli.reports import filter_runs

        runs = sample_history["runs"]
        filtered = filter_runs(runs, limit=3)

        assert len(filtered) == 3
        # Most recent first
        assert filtered[0]["run_id"] == "run-005"

    def test_filter_runs_since(self, sample_history: dict) -> None:
        """Test filtering runs by date."""
        from datetime import datetime

        from traderbot.cli.reports import filter_runs

        runs = sample_history["runs"]
        since = datetime(2026, 1, 3)
        filtered = filter_runs(runs, since=since)

        assert len(filtered) == 3
        # Only runs from 2026-01-03 and later
        run_ids = [r["run_id"] for r in filtered]
        assert "run-005" in run_ids
        assert "run-004" in run_ids
        assert "run-003" in run_ids

    def test_compute_summary_stats(self, sample_history: dict) -> None:
        """Test computing summary statistics."""
        from traderbot.cli.reports import compute_summary_stats

        runs = sample_history["runs"]
        stats = compute_summary_stats(runs)

        assert stats["total_runs"] == 5
        assert stats["pass_count"] == 4
        assert stats["fail_count"] == 1
        assert stats["pass_rate"] == 0.8

        # Sharpe delta stats
        assert stats["sharpe_delta_mean"] is not None
        assert stats["sharpe_delta_std"] is not None

        # Timing stats
        assert stats["timing_p90_mean"] is not None

    def test_compute_summary_stats_empty(self) -> None:
        """Test stats with empty runs."""
        from traderbot.cli.reports import compute_summary_stats

        stats = compute_summary_stats([])

        assert stats["total_runs"] == 0
        assert stats["pass_rate"] == 0.0
        assert stats["sharpe_delta_mean"] is None

    def test_detect_flaky(self, sample_history: dict) -> None:
        """Test flakiness detection in summarize."""
        from traderbot.cli.reports import detect_flaky

        runs = sample_history["runs"]
        flaky = detect_flaky(runs)

        assert "is_flaky" in flaky
        assert "reason" in flaky
        assert "analyzed_runs" in flaky

    def test_format_text_summary(self, sample_history: dict) -> None:
        """Test text output format."""
        from traderbot.cli.reports import (
            compute_summary_stats,
            detect_flaky,
            format_text_summary,
        )

        runs = sample_history["runs"]
        stats = compute_summary_stats(runs)
        flaky = detect_flaky(runs)

        output = format_text_summary(stats, runs, flaky)

        assert "REGRESSION REPORT SUMMARY" in output
        assert "OVERVIEW" in output
        assert "Total runs:" in output
        assert "Pass rate:" in output

    def test_format_json_summary(self, sample_history: dict) -> None:
        """Test JSON output format."""
        from traderbot.cli.reports import (
            compute_summary_stats,
            detect_flaky,
            format_json_summary,
        )

        runs = sample_history["runs"]
        stats = compute_summary_stats(runs)
        flaky = detect_flaky(runs)

        output = format_json_summary(stats, runs, flaky)

        # Should be valid JSON
        data = json.loads(output)

        assert data["schema_version"] == 1
        assert "summary" in data
        assert "flakiness" in data
        assert "runs" in data

    def test_format_csv_summary(self, sample_history: dict) -> None:
        """Test CSV output format."""
        from traderbot.cli.reports import format_csv_summary

        runs = sample_history["runs"]
        output = format_csv_summary(runs)

        lines = output.strip().split("\n")

        # Header line
        assert "run_id,verdict,sharpe_delta,timing_p90,generated_utc" in lines[0]

        # Data rows
        assert len(lines) == 6  # Header + 5 runs

    def test_cmd_summarize_text(self, history_file: Path, capsys) -> None:
        """Test summarize command with text output."""
        import argparse

        from traderbot.cli.reports import cmd_summarize

        args = argparse.Namespace(
            history=str(history_file),
            since=None,
            limit=None,
            format="text",
            out=None,
            no_emoji=True,
        )

        exit_code = cmd_summarize(args)

        assert exit_code == 0

        captured = capsys.readouterr()
        assert "REGRESSION REPORT SUMMARY" in captured.out

    def test_cmd_summarize_json(self, history_file: Path, capsys) -> None:
        """Test summarize command with JSON output."""
        import argparse

        from traderbot.cli.reports import cmd_summarize

        args = argparse.Namespace(
            history=str(history_file),
            since=None,
            limit=None,
            format="json",
            out=None,
            no_emoji=True,
        )

        exit_code = cmd_summarize(args)

        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "summary" in data

    def test_cmd_summarize_to_file(self, history_file: Path, tmp_path: Path) -> None:
        """Test summarize command writes to file."""
        import argparse

        from traderbot.cli.reports import cmd_summarize

        output_path = tmp_path / "summary.txt"

        args = argparse.Namespace(
            history=str(history_file),
            since=None,
            limit=None,
            format="text",
            out=str(output_path),
            no_emoji=True,
        )

        exit_code = cmd_summarize(args)

        assert exit_code == 0
        assert output_path.exists()

        content = output_path.read_text()
        assert "REGRESSION REPORT SUMMARY" in content

    def test_cmd_summarize_missing_history(self, tmp_path: Path) -> None:
        """Test error when history file is missing."""
        import argparse

        from traderbot.cli.reports import cmd_summarize

        args = argparse.Namespace(
            history=str(tmp_path / "nonexistent.json"),
            since=None,
            limit=None,
            format="text",
            out=None,
            no_emoji=True,
        )

        exit_code = cmd_summarize(args)

        assert exit_code == 1

    def test_cmd_summarize_with_limit(self, history_file: Path, capsys) -> None:
        """Test summarize command with limit parameter."""
        import argparse

        from traderbot.cli.reports import cmd_summarize

        args = argparse.Namespace(
            history=str(history_file),
            since=None,
            limit=2,
            format="json",
            out=None,
            no_emoji=True,
        )

        exit_code = cmd_summarize(args)

        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["summary"]["total_runs"] == 2

    def test_cmd_summarize_with_since(self, history_file: Path, capsys) -> None:
        """Test summarize command with since parameter."""
        import argparse

        from traderbot.cli.reports import cmd_summarize

        args = argparse.Namespace(
            history=str(history_file),
            since="2026-01-04",
            limit=None,
            format="json",
            out=None,
            no_emoji=True,
        )

        exit_code = cmd_summarize(args)

        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Should only have runs from 2026-01-04 and later
        assert data["summary"]["total_runs"] == 2
