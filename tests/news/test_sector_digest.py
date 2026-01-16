"""Tests for sector sentiment digest."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

from traderbot.news.sector_digest import build_sector_digest, _parse_window


class TestParseWindow:
    """Tests for window string parsing."""

    def test_parse_days(self):
        """Test parsing day windows."""
        td = _parse_window("1d")
        assert td == timedelta(days=1)

        td = _parse_window("7d")
        assert td == timedelta(days=7)

    def test_parse_hours(self):
        """Test parsing hour windows."""
        td = _parse_window("12h")
        assert td == timedelta(hours=12)

        td = _parse_window("24h")
        assert td == timedelta(hours=24)

    def test_parse_minutes(self):
        """Test parsing minute windows."""
        td = _parse_window("30m")
        assert td == timedelta(minutes=30)


class TestBuildSectorDigest:
    """Tests for sector digest building."""

    def test_aggregate_by_sector(self):
        """Test aggregation by sector."""
        # Create test data
        now = datetime.now(timezone.utc)
        items = [
            {
                "published_utc": now.isoformat(),
                "decayed_sentiment": 0.5,
                "sector": "Technology",
            },
            {
                "published_utc": now.isoformat(),
                "decayed_sentiment": 0.3,
                "sector": "Technology",
            },
            {
                "published_utc": now.isoformat(),
                "decayed_sentiment": -0.2,
                "sector": "Healthcare",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "scored.jsonl"
            output_csv = Path(tmpdir) / "digest.csv"

            with open(input_path, "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")

            results = build_sector_digest(input_path, output_csv, window="1d")

            assert "Technology" in results
            assert "Healthcare" in results
            assert results["Technology"].item_count == 2
            assert results["Healthcare"].item_count == 1

    def test_mean_sentiment_calculation(self):
        """Test mean sentiment is calculated correctly."""
        now = datetime.now(timezone.utc)
        items = [
            {"published_utc": now.isoformat(), "decayed_sentiment": 0.4, "sector": "Tech"},
            {"published_utc": now.isoformat(), "decayed_sentiment": 0.6, "sector": "Tech"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "scored.jsonl"
            output_csv = Path(tmpdir) / "digest.csv"

            with open(input_path, "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")

            results = build_sector_digest(input_path, output_csv, window="1d")

            # Mean of 0.4 and 0.6 is 0.5
            assert abs(results["Tech"].mean_sentiment - 0.5) < 0.01

    def test_filters_by_window(self):
        """Test items outside window are filtered."""
        now = datetime.now(timezone.utc)
        old_time = (now - timedelta(days=3)).isoformat()

        items = [
            {"published_utc": now.isoformat(), "decayed_sentiment": 0.5, "sector": "Tech"},
            {"published_utc": old_time, "decayed_sentiment": 0.5, "sector": "Tech"},  # Too old
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "scored.jsonl"
            output_csv = Path(tmpdir) / "digest.csv"

            with open(input_path, "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")

            results = build_sector_digest(input_path, output_csv, window="1d")

            # Only 1 item should be counted (the recent one)
            assert results["Tech"].item_count == 1

    def test_empty_input(self):
        """Test handling of empty input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "scored.jsonl"
            output_csv = Path(tmpdir) / "digest.csv"

            # Create empty file
            input_path.touch()

            results = build_sector_digest(input_path, output_csv, window="1d")
            assert len(results) == 0

    def test_sentiment_counts(self):
        """Test positive/negative/neutral counting."""
        now = datetime.now(timezone.utc)
        items = [
            {"published_utc": now.isoformat(), "decayed_sentiment": 0.5, "sector": "Tech"},  # Positive
            {"published_utc": now.isoformat(), "decayed_sentiment": 0.05, "sector": "Tech"},  # Neutral
            {"published_utc": now.isoformat(), "decayed_sentiment": -0.3, "sector": "Tech"},  # Negative
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "scored.jsonl"
            output_csv = Path(tmpdir) / "digest.csv"

            with open(input_path, "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")

            results = build_sector_digest(input_path, output_csv, window="1d")

            assert results["Tech"].positive_count == 1
            assert results["Tech"].neutral_count == 1
            assert results["Tech"].negative_count == 1
