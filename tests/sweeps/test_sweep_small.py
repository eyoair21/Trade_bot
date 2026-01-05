"""Tests for hyperparameter sweep functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest

from traderbot.sweeps.schema import (
    SweepConfig,
    SweepConfigError,
    load_sweep_config,
    save_sweep_config,
    validate_config,
)


class TestSweepConfig:
    """Tests for SweepConfig dataclass."""

    def test_basic_config(self, tmp_path: Path) -> None:
        """Test creating a basic sweep config."""
        config = SweepConfig(
            name="test_sweep",
            output_root=tmp_path / "sweep_output",
            metric="sharpe",
            mode="max",
            fixed_args={"start_date": "2023-01-01", "end_date": "2023-03-31"},
            grid={"sizer": ["fixed", "vol"]},
        )

        assert config.name == "test_sweep"
        assert config.metric == "sharpe"
        assert config.mode == "max"
        assert config.total_runs() == 2

    def test_invalid_metric(self, tmp_path: Path) -> None:
        """Test that invalid metric raises error."""
        with pytest.raises(SweepConfigError, match="Invalid metric"):
            SweepConfig(
                name="test",
                output_root=tmp_path,
                metric="invalid_metric",
                mode="max",
            )

    def test_invalid_mode(self, tmp_path: Path) -> None:
        """Test that invalid mode raises error."""
        with pytest.raises(SweepConfigError, match="Invalid mode"):
            SweepConfig(
                name="test",
                output_root=tmp_path,
                metric="sharpe",
                mode="invalid_mode",
            )

    def test_unknown_fixed_args(self, tmp_path: Path) -> None:
        """Test that unknown fixed_args keys raise error."""
        with pytest.raises(SweepConfigError, match="Unknown fixed_args"):
            SweepConfig(
                name="test",
                output_root=tmp_path,
                metric="sharpe",
                mode="max",
                fixed_args={"unknown_param": 123},
            )

    def test_unknown_grid_keys(self, tmp_path: Path) -> None:
        """Test that unknown grid keys raise error."""
        with pytest.raises(SweepConfigError, match="Unknown grid"):
            SweepConfig(
                name="test",
                output_root=tmp_path,
                metric="sharpe",
                mode="max",
                grid={"unknown_param": [1, 2, 3]},
            )

    def test_grid_values_must_be_lists(self, tmp_path: Path) -> None:
        """Test that grid values must be lists."""
        with pytest.raises(SweepConfigError, match="must be a list"):
            SweepConfig(
                name="test",
                output_root=tmp_path,
                metric="sharpe",
                mode="max",
                grid={"sizer": "fixed"},  # Not a list
            )

    def test_string_output_root_conversion(self, tmp_path: Path) -> None:
        """Test that string output_root is converted to Path."""
        config = SweepConfig(
            name="test",
            output_root=str(tmp_path),
            metric="sharpe",
            mode="max",
        )

        assert isinstance(config.output_root, Path)


class TestGridCombinations:
    """Tests for grid combination generation."""

    def test_empty_grid(self, tmp_path: Path) -> None:
        """Test empty grid returns single empty dict."""
        config = SweepConfig(
            name="test",
            output_root=tmp_path,
            metric="sharpe",
            mode="max",
            grid={},
        )

        combos = config.get_grid_combinations()
        assert combos == [{}]

    def test_single_param_grid(self, tmp_path: Path) -> None:
        """Test single parameter grid."""
        config = SweepConfig(
            name="test",
            output_root=tmp_path,
            metric="sharpe",
            mode="max",
            grid={"sizer": ["fixed", "vol", "kelly"]},
        )

        combos = config.get_grid_combinations()
        assert len(combos) == 3
        assert {"sizer": "fixed"} in combos
        assert {"sizer": "vol"} in combos
        assert {"sizer": "kelly"} in combos

    def test_multi_param_grid(self, tmp_path: Path) -> None:
        """Test multi-parameter grid generates cartesian product."""
        config = SweepConfig(
            name="test",
            output_root=tmp_path,
            metric="sharpe",
            mode="max",
            grid={
                "sizer": ["fixed", "vol"],
                "proba_threshold": [0.5, 0.6],
            },
        )

        combos = config.get_grid_combinations()
        assert len(combos) == 4  # 2 x 2

        assert {"sizer": "fixed", "proba_threshold": 0.5} in combos
        assert {"sizer": "fixed", "proba_threshold": 0.6} in combos
        assert {"sizer": "vol", "proba_threshold": 0.5} in combos
        assert {"sizer": "vol", "proba_threshold": 0.6} in combos

    def test_total_runs_calculation(self, tmp_path: Path) -> None:
        """Test total runs calculation."""
        config = SweepConfig(
            name="test",
            output_root=tmp_path,
            metric="sharpe",
            mode="max",
            grid={
                "sizer": ["fixed", "vol"],
                "proba_threshold": [0.5, 0.6, 0.7],
                "n_splits": [2, 3],
            },
        )

        assert config.total_runs() == 2 * 3 * 2  # 12


class TestRunConfigs:
    """Tests for full run configuration generation."""

    def test_run_configs_merge_fixed_and_grid(self, tmp_path: Path) -> None:
        """Test run configs merge fixed_args and grid."""
        config = SweepConfig(
            name="test",
            output_root=tmp_path,
            metric="sharpe",
            mode="max",
            fixed_args={
                "start_date": "2023-01-01",
                "end_date": "2023-03-31",
                "seed": 42,
            },
            grid={"sizer": ["fixed", "vol"]},
        )

        runs = config.get_run_configs()
        assert len(runs) == 2

        for run in runs:
            assert run["start_date"] == "2023-01-01"
            assert run["end_date"] == "2023-03-31"
            assert run["seed"] == 42
            assert run["sizer"] in ["fixed", "vol"]


class TestLoadSaveConfig:
    """Tests for loading and saving sweep configs."""

    def test_save_and_load_config(self, tmp_path: Path) -> None:
        """Test saving and loading a config roundtrip."""
        original = SweepConfig(
            name="test_sweep",
            output_root=tmp_path / "output",
            metric="total_return",
            mode="max",
            fixed_args={"start_date": "2023-01-01"},
            grid={"sizer": ["fixed", "vol"]},
        )

        config_path = tmp_path / "config.yaml"
        save_sweep_config(original, config_path)

        loaded = load_sweep_config(config_path)

        assert loaded.name == original.name
        assert loaded.metric == original.metric
        assert loaded.mode == original.mode
        assert loaded.fixed_args == original.fixed_args
        assert loaded.grid == original.grid

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test loading missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_sweep_config(tmp_path / "nonexistent.yaml")

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Test loading empty file raises error."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        with pytest.raises(SweepConfigError, match="Empty configuration"):
            load_sweep_config(config_path)


class TestValidateConfig:
    """Tests for configuration validation."""

    def test_validate_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = {
            "name": "test",
            "output_root": "/tmp/output",
            "metric": "sharpe",
            "mode": "max",
        }
        # Should not raise
        validate_config(config)

    def test_validate_missing_required(self) -> None:
        """Test validation fails for missing required fields."""
        config = {"name": "test"}

        with pytest.raises(SweepConfigError, match="Missing required"):
            validate_config(config)

    def test_validate_invalid_name_type(self) -> None:
        """Test validation fails for invalid name type."""
        config = {
            "name": 123,  # Should be string
            "output_root": "/tmp",
            "metric": "sharpe",
            "mode": "max",
        }

        with pytest.raises(SweepConfigError, match="'name' must be a string"):
            validate_config(config)


class TestSweepRunner:
    """Tests for sweep runner functionality."""

    def test_run_single_config_success(self, tmp_path: Path) -> None:
        """Test running a single configuration."""
        from traderbot.cli.sweep import run_single_config

        config = {
            "start_date": "2023-01-03",
            "end_date": "2023-03-15",
            "universe": ["AAPL"],
            "n_splits": 2,
            "is_ratio": 0.6,
            "universe_mode": "static",
        }

        # Mock run_walkforward
        with patch("traderbot.cli.sweep.run_walkforward") as mock_wf:
            mock_wf.return_value = {
                "avg_oos_sharpe": 0.5,
                "avg_oos_return_pct": 5.0,
                "avg_oos_max_dd_pct": -2.0,
            }

            result = run_single_config((0, config, tmp_path))

            assert result["_status"] == "success"
            assert result["_run_idx"] == 0
            assert "avg_oos_sharpe" in result

    def test_run_single_config_error(self, tmp_path: Path) -> None:
        """Test handling error in single run."""
        from traderbot.cli.sweep import run_single_config

        config = {"start_date": "2023-01-01"}

        with patch("traderbot.cli.sweep.run_walkforward") as mock_wf:
            mock_wf.side_effect = ValueError("Test error")

            result = run_single_config((0, config, tmp_path))

            assert result["_status"] == "error"
            assert "Test error" in result["error"]


class TestFindBestRun:
    """Tests for finding best run."""

    def test_find_best_sharpe_max(self) -> None:
        """Test finding best run by sharpe (max)."""
        from traderbot.cli.sweep import find_best_run

        results = [
            {"_status": "success", "avg_oos_sharpe": 0.5},
            {"_status": "success", "avg_oos_sharpe": 1.0},
            {"_status": "success", "avg_oos_sharpe": 0.7},
        ]

        best = find_best_run(results, "sharpe", "max")
        assert best["avg_oos_sharpe"] == 1.0

    def test_find_best_max_dd_min(self) -> None:
        """Test finding best run by max_dd (min)."""
        from traderbot.cli.sweep import find_best_run

        results = [
            {"_status": "success", "avg_oos_max_dd_pct": -10.0},
            {"_status": "success", "avg_oos_max_dd_pct": -5.0},
            {"_status": "success", "avg_oos_max_dd_pct": -15.0},
        ]

        best = find_best_run(results, "max_dd", "min")
        assert best["avg_oos_max_dd_pct"] == -15.0

    def test_find_best_skips_errors(self) -> None:
        """Test that error runs are skipped."""
        from traderbot.cli.sweep import find_best_run

        results = [
            {"_status": "error", "avg_oos_sharpe": 999.0},
            {"_status": "success", "avg_oos_sharpe": 0.5},
        ]

        best = find_best_run(results, "sharpe", "max")
        assert best["avg_oos_sharpe"] == 0.5

    def test_find_best_no_valid_runs(self) -> None:
        """Test handling no valid runs."""
        from traderbot.cli.sweep import find_best_run

        results = [
            {"_status": "error"},
            {"_status": "error"},
        ]

        best = find_best_run(results, "sharpe", "max")
        assert best is None
