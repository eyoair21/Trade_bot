"""Sweep configuration schema and validation.

Provides dataclasses for defining hyperparameter sweeps and
utilities for loading/validating YAML configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from traderbot.logging_setup import get_logger

logger = get_logger("sweeps.schema")

# Valid CLI arguments for walkforward command
VALID_WALKFORWARD_ARGS = {
    "start_date",
    "end_date",
    "universe",
    "n_splits",
    "is_ratio",
    "output_dir",
    "universe_mode",
    "sizer",
    "fixed_frac",
    "vol_target",
    "kelly_cap",
    "proba_threshold",
    "opt_threshold",
    "train_per_split",
    "epochs",
    "seed",
}

# Valid metrics for optimization
VALID_METRICS = {"sharpe", "total_return", "max_dd"}

# Valid optimization modes
VALID_MODES = {"max", "min"}


class SweepConfigError(Exception):
    """Raised when sweep configuration is invalid."""

    pass


@dataclass
class SweepConfig:
    """Configuration for a hyperparameter sweep.

    Attributes:
        name: Human-readable sweep name.
        output_root: Root directory for sweep outputs.
        metric: Metric to optimize (sharpe, total_return, max_dd).
        mode: Optimization mode (max or min).
        fixed_args: Arguments that stay constant across all runs.
        grid: Parameter grid to sweep over.
    """

    name: str
    output_root: Path
    metric: Literal["sharpe", "total_return", "max_dd"]
    mode: Literal["max", "min"]
    fixed_args: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if isinstance(self.output_root, str):
            self.output_root = Path(self.output_root)

        if self.metric not in VALID_METRICS:
            raise SweepConfigError(
                f"Invalid metric '{self.metric}'. Must be one of: {VALID_METRICS}"
            )

        if self.mode not in VALID_MODES:
            raise SweepConfigError(
                f"Invalid mode '{self.mode}'. Must be one of: {VALID_MODES}"
            )

        # Validate fixed_args keys
        unknown_fixed = set(self.fixed_args.keys()) - VALID_WALKFORWARD_ARGS
        if unknown_fixed:
            raise SweepConfigError(
                f"Unknown fixed_args keys: {unknown_fixed}. "
                f"Valid keys: {VALID_WALKFORWARD_ARGS}"
            )

        # Validate grid keys
        unknown_grid = set(self.grid.keys()) - VALID_WALKFORWARD_ARGS
        if unknown_grid:
            raise SweepConfigError(
                f"Unknown grid keys: {unknown_grid}. "
                f"Valid keys: {VALID_WALKFORWARD_ARGS}"
            )

        # Ensure grid values are lists
        for key, values in self.grid.items():
            if not isinstance(values, list):
                raise SweepConfigError(
                    f"Grid values for '{key}' must be a list, got {type(values).__name__}"
                )

    def get_grid_combinations(self) -> list[dict[str, Any]]:
        """Generate all combinations of grid parameters.

        Returns:
            List of parameter dictionaries, one per combination.
        """
        if not self.grid:
            return [{}]

        import itertools

        keys = list(self.grid.keys())
        values = [self.grid[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo, strict=True)))

        return combinations

    def get_run_configs(self) -> list[dict[str, Any]]:
        """Generate full configurations for all runs.

        Merges fixed_args with each grid combination.

        Returns:
            List of complete run configurations.
        """
        combinations = self.get_grid_combinations()

        run_configs = []
        for combo in combinations:
            config = {**self.fixed_args, **combo}
            run_configs.append(config)

        return run_configs

    def total_runs(self) -> int:
        """Calculate total number of runs in the sweep."""
        if not self.grid:
            return 1

        total = 1
        for values in self.grid.values():
            total *= len(values)
        return total


def validate_config(config_dict: dict[str, Any]) -> None:
    """Validate a configuration dictionary.

    Args:
        config_dict: Raw configuration dictionary.

    Raises:
        SweepConfigError: If configuration is invalid.
    """
    required_fields = {"name", "output_root", "metric", "mode"}
    missing = required_fields - set(config_dict.keys())
    if missing:
        raise SweepConfigError(f"Missing required fields: {missing}")

    # Validate types
    if not isinstance(config_dict.get("name"), str):
        raise SweepConfigError("'name' must be a string")

    if not isinstance(config_dict.get("output_root"), (str, Path)):
        raise SweepConfigError("'output_root' must be a string or Path")

    if not isinstance(config_dict.get("fixed_args", {}), dict):
        raise SweepConfigError("'fixed_args' must be a dictionary")

    if not isinstance(config_dict.get("grid", {}), dict):
        raise SweepConfigError("'grid' must be a dictionary")


def load_sweep_config(config_path: Path | str) -> SweepConfig:
    """Load sweep configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated SweepConfig instance.

    Raises:
        SweepConfigError: If configuration is invalid.
        FileNotFoundError: If config file doesn't exist.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")

    logger.info(f"Loading sweep config from {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise SweepConfigError(f"Empty configuration file: {config_path}")

    # Validate before creating dataclass
    validate_config(config_dict)

    return SweepConfig(
        name=config_dict["name"],
        output_root=config_dict["output_root"],
        metric=config_dict["metric"],
        mode=config_dict["mode"],
        fixed_args=config_dict.get("fixed_args", {}),
        grid=config_dict.get("grid", {}),
    )


def save_sweep_config(config: SweepConfig, output_path: Path | str) -> None:
    """Save sweep configuration to a YAML file.

    Args:
        config: SweepConfig to save.
        output_path: Path for output YAML file.
    """
    output_path = Path(output_path)

    config_dict = {
        "name": config.name,
        "output_root": str(config.output_root),
        "metric": config.metric,
        "mode": config.mode,
        "fixed_args": config.fixed_args,
        "grid": config.grid,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved sweep config to {output_path}")
