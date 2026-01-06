"""Tests for CLI entrypoint and module invocation."""

import os
import pathlib
import subprocess
import sys

import pytest


# Get the project root to add to PYTHONPATH for subprocess calls
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestCLIEntrypoint:
    """Tests for CLI module entrypoints."""

    def test_regress_help_runs(self):
        """Test that regress --help runs successfully."""
        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(
            [sys.executable, "-m", "traderbot.cli.regress", "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "compare" in result.stdout
        assert "update-baseline" in result.stdout

    def test_regress_compare_help_runs(self):
        """Test that regress compare --help runs successfully."""
        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(
            [sys.executable, "-m", "traderbot.cli.regress", "compare", "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--current" in result.stdout
        assert "--baseline" in result.stdout
        assert "--budget" in result.stdout
        assert "--no-emoji" in result.stdout

    def test_regress_update_baseline_help_runs(self):
        """Test that regress update-baseline --help runs successfully."""
        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(
            [sys.executable, "-m", "traderbot.cli.regress", "update-baseline", "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--current" in result.stdout
        assert "--out" in result.stdout
        assert "--no-emoji" in result.stdout

    def test_walkforward_help_runs(self):
        """Test that walkforward --help runs successfully."""
        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(
            [sys.executable, "-m", "traderbot.cli.walkforward", "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--start-date" in result.stdout or "start" in result.stdout.lower()

    def test_console_module_imports(self):
        """Test that _console module can be imported."""
        from traderbot.cli._console import _can_encode, _fmt, configure_windows_console

        # Basic sanity checks
        assert callable(_can_encode)
        assert callable(_fmt)
        assert callable(configure_windows_console)

    def test_version_accessible(self):
        """Test that version is accessible."""
        from traderbot import __version__

        assert __version__ == "0.6.6-dev"

    def test_regress_compare_quiet_flag_in_help(self):
        """Test that --quiet flag is documented in compare help."""
        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(
            [sys.executable, "-m", "traderbot.cli.regress", "compare", "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--quiet" in result.stdout or "-q" in result.stdout

    def test_regress_compare_auto_update_flag_in_help(self):
        """Test that --auto-update-on-pass flag is documented in compare help."""
        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        result = subprocess.run(
            [sys.executable, "-m", "traderbot.cli.regress", "compare", "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--auto-update-on-pass" in result.stdout
