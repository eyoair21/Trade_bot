"""Tests for generate_trend_plots.py script."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add scripts directory to path for imports
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from generate_trend_plots import (
    HAS_MATPLOTLIB,
    append_data_point,
    generate_plots,
    load_trend_data,
)


class TestLoadTrendData:
    """Tests for load_trend_data function."""

    def test_load_nonexistent_creates_empty(self, tmp_path):
        """Test loading non-existent file creates empty structure."""
        trend_path = tmp_path / "trend_data.json"
        result = load_trend_data(trend_path)

        assert result["version"] == "1.0"
        assert result["data_points"] == []
        assert "created_at" in result

    def test_load_existing_file(self, tmp_path):
        """Test loading existing trend file."""
        trend_path = tmp_path / "trend_data.json"
        existing_data = {
            "version": "1.0",
            "data_points": [
                {"timestamp": "2025-01-01T00:00:00Z", "best_metric": 0.5}
            ],
            "created_at": "2025-01-01T00:00:00Z",
        }
        trend_path.write_text(json.dumps(existing_data))

        result = load_trend_data(trend_path)

        assert len(result["data_points"]) == 1
        assert result["data_points"][0]["best_metric"] == 0.5


class TestAppendDataPoint:
    """Tests for append_data_point function."""

    def test_append_new_data_point(self):
        """Test appending a new data point."""
        trend_data = {"version": "1.0", "data_points": []}
        diff_data = {
            "current": {"best_metric": 0.75, "success_rate": 0.9, "timing_p90": 45.0},
            "baseline": {"best_metric": 0.7},
            "delta": {"best_metric": 0.05},
            "verdict": {"passed": True},
            "current_sha": "abc123",
        }

        result = append_data_point(trend_data, diff_data, git_sha="def456")

        assert len(result["data_points"]) == 1
        dp = result["data_points"][0]
        assert dp["best_metric"] == 0.75
        assert dp["success_rate"] == 0.9
        assert dp["timing_p90"] == 45.0
        assert dp["passed"] is True
        assert dp["git_sha"] == "def456"

    def test_rolling_window_trims_old_data(self):
        """Test that old data points are trimmed."""
        # Create trend data with 10 points
        trend_data = {
            "version": "1.0",
            "data_points": [
                {"timestamp": f"2025-01-{i:02d}T00:00:00Z", "best_metric": 0.5}
                for i in range(1, 11)
            ],
        }
        diff_data = {
            "current": {"best_metric": 0.8},
            "baseline": {},
            "delta": {},
            "verdict": {"passed": True},
        }

        result = append_data_point(trend_data, diff_data, max_points=5)

        assert len(result["data_points"]) == 5
        # Should keep last 4 existing + 1 new
        assert result["data_points"][-1]["best_metric"] == 0.8

    def test_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated if not provided."""
        trend_data = {"version": "1.0", "data_points": []}
        diff_data = {
            "current": {"best_metric": 0.6},
            "baseline": {},
            "delta": {},
            "verdict": {},
        }

        result = append_data_point(trend_data, diff_data)

        assert "timestamp" in result["data_points"][0]
        # Should be a valid ISO timestamp
        ts = result["data_points"][0]["timestamp"]
        assert ts.endswith("Z")


class TestGeneratePlots:
    """Tests for generate_plots function."""

    def test_insufficient_data_points(self, tmp_path):
        """Test that plots are not generated with insufficient data."""
        trend_data = {
            "data_points": [
                {"timestamp": "2025-01-01T00:00:00Z", "best_metric": 0.5}
            ]
        }
        output_dir = tmp_path / "plots"

        result = generate_plots(trend_data, output_dir)

        # Should return empty list (need at least 2 points)
        assert result == []

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_generates_plots_with_valid_data(self, tmp_path):
        """Test that plots are generated with valid data."""
        # Create trend data with multiple points
        base_time = datetime(2025, 1, 1)
        trend_data = {
            "data_points": [
                {
                    "timestamp": (base_time + timedelta(days=i)).isoformat() + "Z",
                    "best_metric": 0.5 + i * 0.01,
                    "success_rate": 0.85 + i * 0.01,
                    "timing_p90": 40.0 + i,
                    "passed": i % 2 == 0,
                }
                for i in range(10)
            ]
        }
        output_dir = tmp_path / "plots"

        result = generate_plots(trend_data, output_dir)

        # Should generate 4 plots
        assert len(result) == 4
        assert (output_dir / "trend_metric.png").exists()
        assert (output_dir / "trend_success_rate.png").exists()
        assert (output_dir / "trend_timing.png").exists()
        assert (output_dir / "trend_dashboard.png").exists()

    def test_handles_missing_matplotlib_gracefully(self, tmp_path, monkeypatch):
        """Test graceful handling when matplotlib is unavailable."""
        # Simulate matplotlib not being available
        import generate_trend_plots
        monkeypatch.setattr(generate_trend_plots, "HAS_MATPLOTLIB", False)

        trend_data = {
            "data_points": [
                {"timestamp": f"2025-01-0{i}T00:00:00Z", "best_metric": 0.5}
                for i in range(1, 5)
            ]
        }
        output_dir = tmp_path / "plots"

        result = generate_plots(trend_data, output_dir)

        assert result == []


class TestMainCLI:
    """Tests for CLI main function."""

    def test_main_missing_diff_file(self, tmp_path, monkeypatch):
        """Test that main returns error for missing diff file."""
        import generate_trend_plots

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_trend_plots.py",
                "--trend-file", str(tmp_path / "trend.json"),
                "--current-diff", str(tmp_path / "nonexistent.json"),
                "--output-dir", str(tmp_path / "plots"),
            ],
        )

        result = generate_trend_plots.main()

        assert result == 1

    def test_main_creates_trend_file(self, tmp_path, monkeypatch):
        """Test that main creates trend file and updates it."""
        import generate_trend_plots

        # Create a minimal baseline_diff.json
        diff_path = tmp_path / "baseline_diff.json"
        diff_path.write_text(json.dumps({
            "current": {"best_metric": 0.65, "success_rate": 0.88, "timing_p90": 42.0},
            "baseline": {"best_metric": 0.6},
            "delta": {"best_metric": 0.05},
            "verdict": {"passed": True},
            "current_sha": "abc123",
        }))

        trend_path = tmp_path / "trend.json"
        output_dir = tmp_path / "plots"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_trend_plots.py",
                "--trend-file", str(trend_path),
                "--current-diff", str(diff_path),
                "--output-dir", str(output_dir),
                "--skip-plots",  # Skip plots since we may not have matplotlib
            ],
        )

        result = generate_trend_plots.main()

        assert result == 0
        assert trend_path.exists()

        # Verify content
        with open(trend_path) as f:
            trend_data = json.load(f)

        assert len(trend_data["data_points"]) == 1
        assert trend_data["data_points"][0]["best_metric"] == 0.65
