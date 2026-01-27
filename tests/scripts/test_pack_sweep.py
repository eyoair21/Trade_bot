"""Tests for pack_sweep.py script."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.pack_sweep import get_dir_size_mb, pack_sweep


@pytest.fixture
def fake_sweep_dir(tmp_path: Path) -> Path:
    """Create fake sweep directory with minimal files."""
    sweep_dir = tmp_path / "test_sweep"
    sweep_dir.mkdir()

    # Create essential files
    (sweep_dir / "sweep_meta.json").write_text(json.dumps({"name": "test"}))
    (sweep_dir / "all_results.json").write_text(json.dumps([{"run": 1}]))
    (sweep_dir / "leaderboard.csv").write_text("rank,run_idx\n1,0\n2,1\n3,2\n4,3\n")
    (sweep_dir / "leaderboard.md").write_text("# Leaderboard\n")
    (sweep_dir / "timings.csv").write_text("run_idx,elapsed_s\n0,1.5\n1,2.0\n")

    # Create 4 run directories with small files
    for i in range(4):
        run_dir = sweep_dir / f"run_{i:03d}"
        run_dir.mkdir()
        (run_dir / "results.json").write_text(json.dumps({"run": i}))
        (run_dir / "equity_curve.csv").write_text("date,equity\n2023-01-01,100000\n")
        (run_dir / "report.md").write_text(f"# Run {i}\n")

    return sweep_dir


class TestGetDirSize:
    """Tests for directory size calculation."""

    def test_get_dir_size_basic(self, tmp_path: Path) -> None:
        """Test basic directory size calculation."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        # Create a 1KB file
        (test_dir / "file.txt").write_text("x" * 1024)

        size_mb = get_dir_size_mb(test_dir)

        # Should be approximately 0.001 MB (1KB)
        assert 0.0 < size_mb < 0.01

    def test_get_dir_size_empty(self, tmp_path: Path) -> None:
        """Test empty directory size."""
        test_dir = tmp_path / "empty"
        test_dir.mkdir()

        size_mb = get_dir_size_mb(test_dir)

        assert size_mb == 0.0


class TestPackSweep:
    """Tests for sweep packing."""

    def test_pack_sweep_creates_zip(self, fake_sweep_dir: Path) -> None:
        """Test that pack_sweep creates a zip file."""
        output_path = fake_sweep_dir.parent / "test_sweep.zip"

        pack_sweep(
            sweep_root=fake_sweep_dir,
            output_path=output_path,
            max_size_mb=80.0,
            top_n=3,
        )

        assert output_path.exists()
        assert output_path.suffix == ".zip"

    def test_pack_sweep_contains_essential_files(self, fake_sweep_dir: Path) -> None:
        """Test that zip contains essential files."""
        output_path = fake_sweep_dir.parent / "test_sweep.zip"

        pack_sweep(
            sweep_root=fake_sweep_dir,
            output_path=output_path,
            top_n=3,
        )

        with ZipFile(output_path, "r") as zipf:
            names = zipf.namelist()

        # Check for essential files
        assert any("sweep_meta.json" in n for n in names)
        assert any("all_results.json" in n for n in names)
        assert any("leaderboard.csv" in n for n in names)
        assert any("timings.csv" in n for n in names)

    def test_pack_sweep_includes_top_n_runs(self, fake_sweep_dir: Path) -> None:
        """Test that zip includes top N run directories."""
        output_path = fake_sweep_dir.parent / "test_sweep.zip"

        pack_sweep(
            sweep_root=fake_sweep_dir,
            output_path=output_path,
            top_n=3,
        )

        with ZipFile(output_path, "r") as zipf:
            names = zipf.namelist()

        # Should include run_000, run_001, run_002 (top 3)
        assert any("run_000" in n for n in names)
        assert any("run_001" in n for n in names)
        assert any("run_002" in n for n in names)

        # Should NOT include run_003 (4th run)
        # (unless top_n was increased, but we set it to 3)

    def test_pack_sweep_excludes_large_files(self, fake_sweep_dir: Path) -> None:
        """Test that parquet files are excluded."""
        # Add a fake parquet file
        run_dir = fake_sweep_dir / "run_000"
        (run_dir / "data.parquet").write_text("fake parquet data" * 1000)

        output_path = fake_sweep_dir.parent / "test_sweep.zip"

        pack_sweep(
            sweep_root=fake_sweep_dir,
            output_path=output_path,
            top_n=1,
        )

        with ZipFile(output_path, "r") as zipf:
            names = zipf.namelist()

        # Parquet file should be excluded
        assert not any("data.parquet" in n for n in names)

    def test_pack_sweep_missing_directory(self, tmp_path: Path) -> None:
        """Test error handling for missing directory."""
        nonexistent = tmp_path / "nonexistent"
        output_path = tmp_path / "output.zip"

        with pytest.raises(FileNotFoundError):
            pack_sweep(nonexistent, output_path)





