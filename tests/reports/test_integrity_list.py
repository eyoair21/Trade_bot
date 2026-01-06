"""Tests for integrity file list in generate_sha256sums.py."""

from pathlib import Path

import pytest


class TestIntegrityFileList:
    """Tests for INTEGRITY_FILES and TOP_LEVEL_INTEGRITY_FILES constants."""

    def test_integrity_files_includes_required(self) -> None:
        """Test INTEGRITY_FILES includes all required files."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from generate_sha256sums import INTEGRITY_FILES

        required_files = [
            "regression_report.html",
            "regression_report.md",
            "baseline_diff.json",
            "provenance.json",
            "summary.json",
        ]

        for f in required_files:
            assert f in INTEGRITY_FILES, f"Missing required file: {f}"

    def test_top_level_integrity_files_includes_required(self) -> None:
        """Test TOP_LEVEL_INTEGRITY_FILES includes all required files."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from generate_sha256sums import TOP_LEVEL_INTEGRITY_FILES

        required_files = [
            "index.html",
            "manifest.json",
            "history.json",
            "feed.xml",
            "404.html",
        ]

        for f in required_files:
            assert f in TOP_LEVEL_INTEGRITY_FILES, f"Missing required file: {f}"

    def test_generate_sha256sums(self, tmp_path: Path) -> None:
        """Test SHA256 sum generation for report files."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from generate_sha256sums import INTEGRITY_FILES, generate_sha256sums

        # Create mock report directory
        report_dir = tmp_path / "report"
        report_dir.mkdir()

        # Create test files
        for filename in INTEGRITY_FILES:
            (report_dir / filename).write_text(f"test content for {filename}")

        # Generate sums
        result = generate_sha256sums(report_dir)

        assert result is not None
        assert result.exists()

        # Verify format
        content = result.read_text()
        lines = [l for l in content.strip().split("\n") if l]

        # Should have one line per existing file
        assert len(lines) >= len(INTEGRITY_FILES)

        # Each line should be "hash  filename" format
        for line in lines:
            parts = line.split("  ", 1)
            assert len(parts) == 2
            hash_val, filename = parts
            assert len(hash_val) == 64  # SHA256 hex length
            assert filename in INTEGRITY_FILES or filename.startswith("plots/")

    def test_generate_top_level_sha256sums(self, tmp_path: Path) -> None:
        """Test SHA256 sum generation for top-level files."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from generate_sha256sums import (
            TOP_LEVEL_INTEGRITY_FILES,
            generate_top_level_sha256sums,
        )

        # Create mock reports root
        reports_root = tmp_path / "reports"
        reports_root.mkdir()

        # Create test files
        for filename in TOP_LEVEL_INTEGRITY_FILES:
            (reports_root / filename).write_text(f"test content for {filename}")

        # Generate sums
        result = generate_top_level_sha256sums(reports_root)

        assert result is not None
        assert result.exists()

        content = result.read_text()
        lines = [l for l in content.strip().split("\n") if l]

        assert len(lines) == len(TOP_LEVEL_INTEGRITY_FILES)

    def test_verify_sha256sums(self, tmp_path: Path) -> None:
        """Test SHA256 sum verification."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from generate_sha256sums import generate_sha256sums, verify_sha256sums

        # Create mock report directory
        report_dir = tmp_path / "report"
        report_dir.mkdir()

        # Create test files
        (report_dir / "regression_report.html").write_text("test html")
        (report_dir / "summary.json").write_text('{"test": true}')

        # Generate sums
        generate_sha256sums(report_dir)

        # Verify - should pass
        assert verify_sha256sums(report_dir)

    def test_verify_sha256sums_tampered(self, tmp_path: Path) -> None:
        """Test SHA256 verification detects tampering."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from generate_sha256sums import generate_sha256sums, verify_sha256sums

        # Create mock report directory
        report_dir = tmp_path / "report"
        report_dir.mkdir()

        # Create test files
        test_file = report_dir / "regression_report.html"
        test_file.write_text("original content")

        # Generate sums
        generate_sha256sums(report_dir)

        # Tamper with file
        test_file.write_text("tampered content")

        # Verify - should fail
        assert not verify_sha256sums(report_dir)

    def test_compute_sha256(self, tmp_path: Path) -> None:
        """Test SHA256 computation."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from generate_sha256sums import compute_sha256

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        hash_val = compute_sha256(test_file)

        # SHA256 of "hello world" (without newline)
        assert len(hash_val) == 64
        assert hash_val.isalnum()

    def test_missing_files_handled(self, tmp_path: Path) -> None:
        """Test missing files are handled gracefully."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "dev"))

        from generate_sha256sums import generate_sha256sums

        # Create empty report directory
        report_dir = tmp_path / "empty_report"
        report_dir.mkdir()

        # Should not crash, returns None when no files
        result = generate_sha256sums(report_dir)

        # No files to hash
        assert result is None
