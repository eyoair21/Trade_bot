"""CLI for running hyperparameter sweeps.

Executes walk-forward analysis across a grid of parameters
with optional parallel execution.

Usage:
    python -m traderbot.cli.sweep sweeps/config.yaml --workers 4
"""

import argparse
import json
import multiprocessing
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from traderbot.cli.walkforward import run_walkforward
from traderbot.logging_setup import get_logger
from traderbot.sweeps.schema import SweepConfig, load_sweep_config

logger = get_logger("cli.sweep")


def run_single_config(args: tuple[int, dict[str, Any], Path, bool]) -> dict[str, Any]:
    """Run a single walk-forward configuration.

    Args:
        args: Tuple of (run_index, config_dict, output_dir, enable_profiling).

    Returns:
        Results dictionary with run metadata.
    """
    run_idx, config, output_dir, enable_profiling = args

    # Create run-specific output directory
    run_output = output_dir / f"run_{run_idx:03d}"
    run_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting run {run_idx}: {config}")

    start_time = time.time()

    # Enable per-run profiling if requested
    pr = None
    if enable_profiling:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    try:
        # Extract and convert parameters for run_walkforward
        # Use deterministic sub-seed for parallel execution
        base_seed = config.get("seed", 42)
        run_seed = int(base_seed) + int(run_idx)

        wf_kwargs = {
            "start_date": config.get("start_date"),
            "end_date": config.get("end_date"),
            "universe": config.get("universe", []),
            "n_splits": config.get("n_splits", 3),
            "is_ratio": config.get("is_ratio", 0.6),
            "output_dir": run_output,
            "universe_mode": config.get("universe_mode", "static"),
            "sizer": config.get("sizer", "fixed"),
            "seed": run_seed,
        }

        # Add optional parameters
        if "fixed_frac" in config:
            wf_kwargs["fixed_frac"] = config["fixed_frac"]
        if "vol_target" in config:
            wf_kwargs["vol_target"] = config["vol_target"]
        if "kelly_cap" in config:
            wf_kwargs["kelly_cap"] = config["kelly_cap"]
        if "proba_threshold" in config:
            wf_kwargs["proba_threshold"] = config["proba_threshold"]
        if "opt_threshold" in config:
            wf_kwargs["opt_threshold"] = config["opt_threshold"]
        if "train_per_split" in config:
            wf_kwargs["train_per_split"] = config["train_per_split"]
        if "epochs" in config:
            wf_kwargs["epochs"] = config["epochs"]

        results = run_walkforward(**wf_kwargs)

        elapsed = time.time() - start_time
        
        # Save per-run profile if enabled
        if enable_profiling and pr is not None:
            import io
            
            import pstats
            
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(40)
            profile_path = run_output / "profile.txt"
            profile_path.write_text(s.getvalue(), encoding="utf-8")

        # Add metadata to results
        results["_run_idx"] = run_idx
        results["_config"] = config
        results["_elapsed_seconds"] = elapsed
        results["_output_dir"] = str(run_output)
        results["_status"] = "success" if "error" not in results else "error"

        logger.info(f"Completed run {run_idx} in {elapsed:.1f}s")

        return results

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Run {run_idx} failed: {e}")
        
        # Disable profiler on error
        if enable_profiling and pr is not None:
            pr.disable()

        return {
            "_run_idx": run_idx,
            "_config": config,
            "_elapsed_seconds": elapsed,
            "_output_dir": str(run_output),
            "_status": "error",
            "error": str(e),
        }


def run_sweep(
    config: SweepConfig,
    workers: int = 1,
    dry_run: bool = False,
    enable_timing: bool = False,
    enable_profiling: bool = False,
) -> list[dict[str, Any]]:
    """Execute a hyperparameter sweep.

    Args:
        config: Sweep configuration.
        workers: Number of parallel workers.
        dry_run: If True, only show what would be run.

    Returns:
        List of results from all runs.
    """
    # Get all run configurations
    run_configs = config.get_run_configs()
    total_runs = len(run_configs)

    logger.info(f"Sweep '{config.name}' with {total_runs} runs")
    logger.info(f"Output root: {config.output_root}")
    logger.info(f"Metric: {config.metric} (mode: {config.mode})")

    if dry_run:
        logger.info("Dry run - showing configurations:")
        for idx, cfg in enumerate(run_configs):
            print(f"  Run {idx}: {cfg}")
        return []

    # Create output directory
    config.output_root.mkdir(parents=True, exist_ok=True)

    # Save sweep config for reference with provenance metadata
    sweep_meta = {
        "name": config.name,
        "metric": config.metric,
        "mode": config.mode,
        "total_runs": total_runs,
        "workers": workers,
        "fixed_args": config.fixed_args,
        "grid": config.grid,
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = config.output_root / "sweep_meta.json"
    with open(meta_path, "w") as f:
        json.dump(sweep_meta, f, indent=2)

    # Prepare run arguments
    run_args = [
        (idx, cfg, config.output_root, enable_profiling) for idx, cfg in enumerate(run_configs)
    ]

    start_time = time.time()

    # Execute runs
    if workers == 1:
        # Sequential execution
        results = [run_single_config(args) for args in run_args]
    else:
        # Parallel execution
        workers = min(workers, total_runs)
        logger.info(f"Running with {workers} parallel workers")

        with multiprocessing.Pool(workers) as pool:
            results = pool.map(run_single_config, run_args)

    total_elapsed = time.time() - start_time

    # Count successes and failures
    n_success = sum(1 for r in results if r.get("_status") == "success")
    n_error = sum(1 for r in results if r.get("_status") == "error")

    logger.info(
        f"Sweep completed: {n_success} success, {n_error} errors, "
        f"total time: {total_elapsed:.1f}s"
    )

    # Save all results
    all_results_path = config.output_root / "all_results.json"
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"All results saved to {all_results_path}")

    # Write timing CSV if requested
    if enable_timing:
        write_timing_csv(results, config.output_root)
        print_timing_summary(results)

    return results


def find_best_run(
    results: list[dict[str, Any]],
    metric: str,
    mode: str,
) -> dict[str, Any] | None:
    """Find the best run based on metric and mode.

    Args:
        results: List of run results.
        metric: Metric to optimize.
        mode: 'max' or 'min'.

    Returns:
        Best run results or None if no valid runs.
    """
    # Map metric names to result keys
    metric_keys = {
        "sharpe": "avg_oos_sharpe",
        "total_return": "avg_oos_return_pct",
        "max_dd": "avg_oos_max_dd_pct",
    }

    key = metric_keys.get(metric)
    if not key:
        logger.error(f"Unknown metric: {metric}")
        return None

    # Filter successful runs with valid metric values
    valid_runs = [
        r for r in results
        if r.get("_status") == "success" and key in r
    ]

    if not valid_runs:
        logger.warning("No valid runs found for ranking")
        return None

    # Sort by metric
    reverse = mode == "max"
    sorted_runs = sorted(valid_runs, key=lambda r: r[key], reverse=reverse)

    return sorted_runs[0]


def write_timing_csv(results: list[dict[str, Any]], output_root: Path) -> None:
    """Write timing data to CSV file.

    Args:
        results: List of run results with timing data.
        output_root: Sweep output directory.
    """
    import csv

    timing_path = output_root / "timings.csv"

    with open(timing_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_idx", "elapsed_s", "load_s", "splits_s",
            "backtest_s", "report_s", "total_s"
        ])

        for result in results:
            if result.get("_status") != "success":
                continue

            timing = result.get("timing", {})
            writer.writerow([
                result.get("_run_idx", -1),
                result.get("_elapsed_seconds", 0.0),
                timing.get("load_data_s", 0.0),
                timing.get("create_splits_s", 0.0),
                timing.get("backtest_s", 0.0),
                timing.get("report_s", 0.0),
                timing.get("total_s", 0.0),
            ])

    logger.info(f"Timing data written to {timing_path}")


def print_timing_summary(results: list[dict[str, Any]]) -> None:
    """Print timing percentile summary.

    Args:
        results: List of run results with timing data.
    """
    import numpy as np

    elapsed_times = [
        r.get("_elapsed_seconds", 0.0)
        for r in results
        if r.get("_status") == "success"
    ]

    if not elapsed_times:
        return

    p50 = np.percentile(elapsed_times, 50)
    p90 = np.percentile(elapsed_times, 90)

    print(f"\n{'='*60}")
    print("Timing Summary:")
    print(f"  P50 elapsed: {p50:.2f}s")
    print(f"  P90 elapsed: {p90:.2f}s")
    print(f"  Total runs: {len(elapsed_times)}")
    print(f"{'='*60}\n")


def rerun_best_for_determinism(
    best_run: dict[str, Any],
    output_root: Path,
    n_reruns: int,
    metric: str,
) -> dict[str, Any]:
    """Rerun the best configuration N times to verify determinism.

    Args:
        best_run: Best run result from initial sweep.
        output_root: Sweep output directory.
        n_reruns: Number of times to rerun.
        metric: Metric to check for determinism.

    Returns:
        Dictionary with determinism check results.
    """
    import numpy as np

    # Map metric names to result keys
    metric_keys = {
        "sharpe": "avg_oos_sharpe",
        "total_return": "avg_oos_return_pct",
        "max_dd": "avg_oos_max_dd_pct",
    }

    key = metric_keys.get(metric, "avg_oos_sharpe")
    config = best_run.get("_config", {})
    original_value = best_run.get(key, 0.0)
    original_seed = config.get("seed", 42)

    logger.info(f"Running {n_reruns} determinism check(s) for best config")
    logger.info(f"Original {metric}: {original_value:.6f}, seed: {original_seed}")

    # Create determinism check directory
    det_dir = output_root / "determinism_checks"
    det_dir.mkdir(parents=True, exist_ok=True)

    rerun_values = []
    rerun_results = []

    for i in range(n_reruns):
        logger.info(f"Determinism rerun {i + 1}/{n_reruns}")

        # Run with exact same config and seed (no run_idx offset)
        run_output = det_dir / f"rerun_{i:03d}"
        run_output.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            wf_kwargs = {
                "start_date": config.get("start_date"),
                "end_date": config.get("end_date"),
                "universe": config.get("universe", []),
                "n_splits": config.get("n_splits", 3),
                "is_ratio": config.get("is_ratio", 0.6),
                "output_dir": run_output,
                "universe_mode": config.get("universe_mode", "static"),
                "sizer": config.get("sizer", "fixed"),
                "seed": original_seed,  # Use exact same seed
            }

            # Add optional parameters
            for param in [
                "fixed_frac", "vol_target", "kelly_cap", "proba_threshold",
                "opt_threshold", "train_per_split", "epochs"
            ]:
                if param in config:
                    wf_kwargs[param] = config[param]

            result = run_walkforward(**wf_kwargs)
            elapsed = time.time() - start_time

            rerun_value = result.get(key, 0.0)
            rerun_values.append(rerun_value)

            rerun_results.append({
                "rerun_idx": i,
                "metric_value": rerun_value,
                "elapsed_seconds": elapsed,
                "status": "success",
            })

            logger.info(f"Rerun {i + 1}: {metric}={rerun_value:.6f}, elapsed={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Determinism rerun {i + 1} failed: {e}")
            rerun_results.append({
                "rerun_idx": i,
                "metric_value": None,
                "elapsed_seconds": elapsed,
                "status": "error",
                "error": str(e),
            })

    # Calculate determinism statistics
    if rerun_values:
        all_values = [original_value] + rerun_values
        max_abs_diff = float(np.max(np.abs(np.array(all_values) - original_value)))
        mean_value = float(np.mean(all_values))
        std_value = float(np.std(all_values))
        is_deterministic = max_abs_diff < 1e-9  # Effectively zero diff
    else:
        max_abs_diff = float("nan")
        mean_value = original_value
        std_value = 0.0
        is_deterministic = False

    determinism_result = {
        "metric": metric,
        "metric_key": key,
        "original_value": original_value,
        "original_seed": original_seed,
        "n_reruns": n_reruns,
        "rerun_values": rerun_values,
        "max_abs_diff": max_abs_diff,
        "mean_value": mean_value,
        "std_value": std_value,
        "is_deterministic": is_deterministic,
        "reruns": rerun_results,
        "config": config,
    }

    # Write determinism.json
    det_path = output_root / "determinism.json"
    with open(det_path, "w") as f:
        json.dump(determinism_result, f, indent=2, default=str)

    logger.info(f"Determinism check results written to {det_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Determinism Check Results:")
    print(f"  Metric: {metric}")
    print(f"  Original value: {original_value:.6f}")
    print(f"  Reruns: {n_reruns}")
    if rerun_values:
        print(f"  Max absolute diff: {max_abs_diff:.2e}")
        print(f"  Mean: {mean_value:.6f}, Std: {std_value:.2e}")
        if is_deterministic:
            print("  ✅ DETERMINISTIC (diff < 1e-9)")
        else:
            print("  ⚠️  NON-DETERMINISTIC (diff >= 1e-9)")
    else:
        print("  ❌ All reruns failed")
    print(f"{'='*60}\n")

    return determinism_result


def main() -> None:
    """Main entry point for sweep CLI."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweeps for walk-forward analysis"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to sweep configuration YAML file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="Write per-run timings to timings.csv",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed profiling of run phases",
    )
    parser.add_argument(
        "--rerun-best",
        type=int,
        default=0,
        metavar="N",
        help="Rerun best config N times after sweep to verify determinism",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_sweep_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Run sweep
    results = run_sweep(
        config=config,
        workers=args.workers,
        dry_run=args.dry_run,
        enable_timing=args.time,
        enable_profiling=args.profile,
    )

    if args.dry_run:
        sys.exit(0)

    # Find and display best run
    best = find_best_run(results, config.metric, config.mode)
    if best:
        print(f"\n{'='*60}")
        print(f"Best run (by {config.metric}, {config.mode}):")
        print(f"  Run index: {best['_run_idx']}")
        print(f"  Output: {best['_output_dir']}")

        metric_keys = {
            "sharpe": "avg_oos_sharpe",
            "total_return": "avg_oos_return_pct",
            "max_dd": "avg_oos_max_dd_pct",
        }
        key = metric_keys[config.metric]
        print(f"  {config.metric}: {best[key]:.4f}")
        print(f"  Config: {best['_config']}")
        print(f"{'='*60}\n")

        # Run determinism check if requested
        if args.rerun_best > 0:
            rerun_best_for_determinism(
                best_run=best,
                output_root=config.output_root,
                n_reruns=args.rerun_best,
                metric=config.metric,
            )


if __name__ == "__main__":
    main()
