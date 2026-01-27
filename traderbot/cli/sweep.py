"""Parameter sweep and leaderboard generation.

Runs multiple backtest configurations and ranks by OOS performance.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_configs(
    param_grid: dict[str, list[Any]],
    mode: str = "grid",
    budget: int | None = None,
) -> list[dict[str, Any]]:
    """Generate configurations from parameter grid.
    
    Args:
        param_grid: Dictionary of {param_name: [values]}
        mode: 'grid' (exhaustive) or 'random' (sample)
        budget: Maximum number of configs (for random mode)
    
    Returns:
        List of parameter dictionaries
    """
    if mode == "grid":
        # Cartesian product of all parameters
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        configs = []
        for combo in product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)
        
        if budget and len(configs) > budget:
            # Subsample if needed
            indices = np.random.choice(len(configs), budget, replace=False)
            configs = [configs[i] for i in indices]
        
        return configs
    
    elif mode == "random":
        if budget is None:
            budget = 20
        
        configs = []
        for _ in range(budget):
            config = {
                key: np.random.choice(values)
                for key, values in param_grid.items()
            }
            configs.append(config)
        
        return configs
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_single_config(
    config: dict[str, Any],
    base_cmd: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run backtest with single configuration.
    
    Args:
        config: Parameter configuration
        base_cmd: Base command to run
        output_dir: Output directory
    
    Returns:
        Results dictionary
    """
    # Build command
    cmd_parts = [base_cmd]
    for param, value in config.items():
        cmd_parts.append(f"--{param.replace('_', '-')} {value}")
    
    cmd_parts.append(f"--output-dir {output_dir}")
    cmd = " ".join(cmd_parts)
    
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )
        
        if result.returncode != 0:
            print(f"  Failed: {result.stderr}")
            return {"config": config, "status": "failed", "error": result.stderr}
        
        # Load results
        results_path = output_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            
            return {
                "config": config,
                "status": "success",
                **results,
            }
        else:
            return {"config": config, "status": "no_results"}
    
    except Exception as e:
        print(f"  Exception: {e}")
        return {"config": config, "status": "error", "error": str(e)}


def create_leaderboard(
    results: list[dict[str, Any]],
    metric: str = "total_reward",
) -> pd.DataFrame:
    """Create leaderboard from sweep results.
    
    Args:
        results: List of result dictionaries
        metric: Metric to rank by
    
    Returns:
        Leaderboard DataFrame sorted by metric
    """
    # Filter successful runs
    successful = [r for r in results if r.get("status") == "success"]
    
    if not successful:
        return pd.DataFrame()
    
    # Extract key metrics
    rows = []
    for result in successful:
        row = {
            "rank": 0,  # Will assign after sorting
            metric: result.get(metric, -np.inf),
            "sharpe": result.get("sharpe", 0.0),
            "max_dd": result.get("max_dd", 0.0),
            "total_trades": result.get("total_trades", 0),
        }
        
        # Add config params
        for key, value in result["config"].items():
            row[f"param_{key}"] = value
        
        rows.append(row)
    
    # Create DataFrame and sort
    df = pd.DataFrame(rows)
    df = df.sort_values(metric, ascending=False)
    df["rank"] = range(1, len(df) + 1)
    
    # Reorder columns
    cols = ["rank", metric, "sharpe", "max_dd", "total_trades"]
    cols += [c for c in df.columns if c.startswith("param_")]
    df = df[cols]
    
    return df


def main() -> int:
    """Run parameter sweep and generate leaderboard."""
    parser = argparse.ArgumentParser(description="Run parameter sweep")
    
    # Base command
    parser.add_argument("--base-cmd", default="poetry run python -m traderbot.cli.walkforward")
    parser.add_argument("--start-date", default="2023-01-10")
    parser.add_argument("--end-date", default="2023-03-31")
    parser.add_argument("--universe", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    
    # Sweep config
    parser.add_argument("--mode", choices=["grid", "random"], default="random")
    parser.add_argument("--budget", type=int, default=20, help="Max configs to try")
    parser.add_argument("--metric", default="total_reward", help="Metric to optimize")
    
    # Output
    parser.add_argument("--output-dir", default="runs/sweep")
    
    args = parser.parse_args()
    
    # Define parameter grid
    param_grid = {
        "lambda_dd": [0.1, 0.2, 0.3],
        "tau_turnover": [0.0005, 0.001, 0.002],
        "kappa_breach": [0.3, 0.5, 0.7],
        "sizer": ["fixed", "vol"],
        "vol_target": [0.10, 0.15, 0.20],
        "n_splits": [2, 3],
    }
    
    # Generate configs
    configs = generate_configs(param_grid, mode=args.mode, budget=args.budget)
    
    print(f"Generated {len(configs)} configurations")
    print()
    
    # Base command
    universe_str = " ".join(args.universe)
    base_cmd = (
        f"{args.base_cmd} "
        f"--start-date {args.start_date} "
        f"--end-date {args.end_date} "
        f"--universe {universe_str}"
    )
    
    # Run sweep
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}]")
        config_dir = output_dir / f"config_{i:03d}"
        result = run_single_config(config, base_cmd, config_dir)
        results.append(result)
        print()
    
    # Create leaderboard
    leaderboard = create_leaderboard(results, metric=args.metric)
    
    if len(leaderboard) == 0:
        print("No successful runs!")
        return 1
    
    # Save leaderboard
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    
    print(f"Leaderboard saved to: {leaderboard_path}")
    print()
    print("Top 5:")
    print(leaderboard.head().to_string())
    
    # Save full results
    results_path = output_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
