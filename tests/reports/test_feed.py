"""Tests for Atom feed generation."""

import json
from pathlib import Path
from xml.etree.ElementTree import fromstring

import pytest


class TestAtomFeedGenerator:
    """Tests for Atom feed generation from history.json."""

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
                    "trades_delta": 5,
                    "git_sha": "mno7890",
                    "generated_utc": "2026-01-05T10:00:00Z",
                },
                {
                    "run_id": "run-004",
                    "verdict": "PASS",
                    "sharpe_delta": 0.01,
                    "timing_p90": 13.0,
                    "trades_delta": -2,
                    "git_sha": "jkl3456",
                    "generated_utc": "2026-01-04T10:00:00Z",
                },
                {
                    "run_id": "run-003",
                    "verdict": "FAIL",
                    "sharpe_delta": -0.03,
                    "timing_p90": 15.2,
                    "trades_delta": None,
                    "git_sha": "ghi9012",
                    "generated_utc": "2026-01-03T10:00:00Z",
                },
            ],
        }

    @pytest.fixture
    def history_file(self, tmp_path: Path, sample_history: dict) -> Path:
        """Create history.json file."""
        history_path = tmp_path / "history.json"
        history_path.write_text(json.dumps(sample_history, indent=2))
        return history_path

    def test_generate_atom_feed(self, sample_history: dict) -> None:
        """Test generating valid Atom feed."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from make_feed import generate_atom_feed

        feed_xml = generate_atom_feed(
            history=sample_history,
            base_url="https://example.github.io/repo",
        )

        # Should be valid XML
        assert feed_xml.startswith(b"<?xml")

        # Parse and validate structure
        root = fromstring(feed_xml)
        assert root.tag == "{http://www.w3.org/2005/Atom}feed"

        # Check required elements
        namespaces = {"atom": "http://www.w3.org/2005/Atom"}
        title = root.find("atom:title", namespaces)
        assert title is not None
        assert "TraderBot" in title.text

        entries = root.findall("atom:entry", namespaces)
        assert len(entries) == 3

    def test_feed_entry_structure(self, sample_history: dict) -> None:
        """Test feed entries have required elements."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from make_feed import generate_atom_feed

        feed_xml = generate_atom_feed(
            history=sample_history,
            base_url="https://example.github.io/repo",
        )

        root = fromstring(feed_xml)
        namespaces = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", namespaces)
        assert entry is not None

        # Required Atom elements
        assert entry.find("atom:title", namespaces) is not None
        assert entry.find("atom:link", namespaces) is not None
        assert entry.find("atom:id", namespaces) is not None
        assert entry.find("atom:updated", namespaces) is not None

    def test_feed_entry_title_format(self, sample_history: dict) -> None:
        """Test entry title includes verdict and run_id."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from make_feed import generate_atom_feed

        feed_xml = generate_atom_feed(
            history=sample_history,
            base_url="https://example.github.io/repo",
        )

        root = fromstring(feed_xml)
        namespaces = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", namespaces)
        title = entry.find("atom:title", namespaces)

        assert "PASS" in title.text or "run-005" in title.text

    def test_feed_max_entries(self, sample_history: dict) -> None:
        """Test max_entries parameter limits feed size."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from make_feed import generate_atom_feed

        feed_xml = generate_atom_feed(
            history=sample_history,
            base_url="https://example.github.io/repo",
            max_entries=2,
        )

        root = fromstring(feed_xml)
        namespaces = {"atom": "http://www.w3.org/2005/Atom"}

        entries = root.findall("atom:entry", namespaces)
        assert len(entries) == 2

    def test_feed_rfc3339_timestamps(self, sample_history: dict) -> None:
        """Test timestamps are RFC3339 compliant."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from make_feed import generate_atom_feed

        feed_xml = generate_atom_feed(
            history=sample_history,
            base_url="https://example.github.io/repo",
        )

        root = fromstring(feed_xml)
        namespaces = {"atom": "http://www.w3.org/2005/Atom"}

        updated = root.find("atom:updated", namespaces)
        assert updated is not None

        # RFC3339 format check
        ts = updated.text
        assert "T" in ts
        assert ts.endswith("Z") or "+" in ts or "-" in ts[10:]

    def test_feed_empty_history(self) -> None:
        """Test generating feed from empty history."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from make_feed import generate_atom_feed

        empty_history = {
            "schema_version": 1,
            "generated_utc": "2026-01-05T12:00:00Z",
            "window": 5,
            "runs": [],
        }

        feed_xml = generate_atom_feed(
            history=empty_history,
            base_url="https://example.github.io/repo",
        )

        # Should still be valid XML
        root = fromstring(feed_xml)
        namespaces = {"atom": "http://www.w3.org/2005/Atom"}

        entries = root.findall("atom:entry", namespaces)
        assert len(entries) == 0

    def test_feed_validate_wellformed(self, sample_history: dict) -> None:
        """Test feed XML is well-formed."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from make_feed import generate_atom_feed, validate_feed

        feed_xml = generate_atom_feed(
            history=sample_history,
            base_url="https://example.github.io/repo",
        )

        # Should not raise
        assert validate_feed(feed_xml)

    def test_write_feed(self, history_file: Path, tmp_path: Path) -> None:
        """Test writing feed to file."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from make_feed import load_history, generate_atom_feed, write_feed

        history = load_history(history_file)
        feed_xml = generate_atom_feed(
            history=history,
            base_url="https://example.github.io/repo",
        )

        output_path = tmp_path / "feed.xml"
        write_feed(feed_xml, output_path)

        assert output_path.exists()
        content = output_path.read_bytes()
        assert content.startswith(b"<?xml")

    def test_feed_entry_summary(self, sample_history: dict) -> None:
        """Test entry summary includes key metrics."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from make_feed import generate_atom_feed

        feed_xml = generate_atom_feed(
            history=sample_history,
            base_url="https://example.github.io/repo",
        )

        root = fromstring(feed_xml)
        namespaces = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", namespaces)
        summary = entry.find("atom:summary", namespaces)

        assert summary is not None
        assert "Sharpe" in summary.text or "Result" in summary.text

    def test_feed_category_verdict(self, sample_history: dict) -> None:
        """Test entries have category for verdict filtering."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from make_feed import generate_atom_feed

        feed_xml = generate_atom_feed(
            history=sample_history,
            base_url="https://example.github.io/repo",
        )

        root = fromstring(feed_xml)
        namespaces = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", namespaces)
        category = entry.find("atom:category", namespaces)

        assert category is not None
        term = category.get("term")
        assert term in ["pass", "fail"]
