"""Leaderboard generation for hyperparameter sweeps.

Aggregates results from multiple sweep runs and generates
leaderboard reports in CSV and Markdown formats.
"""

import json
import shutil
from pathlib import Path
from typing import Any

from traderbot.logging_setup import get_logger

logger = get_logger("reports.leaderboard")

# Metric key mapping
METRIC_KEYS = {
    "sharpe": "avg_oos_sharpe",
    "total_return": "avg_oos_return_pct",
    "max_dd": "avg_oos_max_dd_pct",
}


def load_sweep_results(sweep_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load sweep metadata and all run results.

    Args:
        sweep_root: Path to sweep output directory.

    Returns:
        Tuple of (sweep_meta, results_list).

    Raises:
        FileNotFoundError: If required files don't exist.
    """
    meta_path = sweep_root / "sweep_meta.json"
    results_path = sweep_root / "all_results.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Sweep metadata not found: {meta_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Sweep results not found: {results_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    with open(results_path) as f:
        results = json.load(f)

    return meta, results


def generate_leaderboard(
    results: list[dict[str, Any]],
    metric: str,
    mode: str,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Generate sorted leaderboard from sweep results.

    Args:
        results: List of run result dictionaries.
        metric: Metric to rank by.
        mode: 'max' or 'min'.
        top_n: Optional limit on number of entries.

    Returns:
        Sorted list of leaderboard entries.
    """
    key = METRIC_KEYS.get(metric)
    if not key:
        raise ValueError(f"Unknown metric: {metric}")

    # Filter successful runs with valid metric
    valid_runs = [
        r for r in results
        if r.get("_status") == "success" and key in r
    ]

    if not valid_runs:
        logger.warning("No valid runs found for leaderboard")
        return []

    # Sort by metric
    reverse = mode == "max"
    sorted_runs = sorted(valid_runs, key=lambda r: r[key], reverse=reverse)

    # Create leaderboard entries
    leaderboard = []
    for rank, run in enumerate(sorted_runs, 1):
        if top_n and rank > top_n:
            break

        entry = {
            "rank": rank,
            "run_idx": run["_run_idx"],
            "output_dir": run["_output_dir"],
            metric: run[key],
            "avg_oos_return_pct": run.get("avg_oos_return_pct", 0.0),
            "avg_oos_sharpe": run.get("avg_oos_sharpe", 0.0),
            "avg_oos_max_dd_pct": run.get("avg_oos_max_dd_pct", 0.0),
            "total_oos_trades": run.get("total_oos_trades", 0),
            "config": run.get("_config", {}),
            "elapsed_seconds": run.get("_elapsed_seconds", 0.0),
        }
        leaderboard.append(entry)

    return leaderboard


def save_leaderboard_csv(
    leaderboard: list[dict[str, Any]],
    output_path: Path,
    metric: str,
) -> None:
    """Save leaderboard to CSV file.

    Args:
        leaderboard: List of leaderboard entries.
        output_path: Path for output CSV.
        metric: Primary metric used for ranking.
    """
    import csv

    if not leaderboard:
        logger.warning("Empty leaderboard - skipping CSV output")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define columns
    base_columns = [
        "rank",
        "run_idx",
        "avg_oos_sharpe",
        "avg_oos_return_pct",
        "avg_oos_max_dd_pct",
        "total_oos_trades",
        "elapsed_seconds",
    ]

    # Get all config keys from first entry
    config_keys = sorted(leaderboard[0].get("config", {}).keys())

    columns = base_columns + config_keys

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for entry in leaderboard:
            row = {k: entry.get(k) for k in base_columns}
            # Flatten config into row
            for k in config_keys:
                row[k] = entry.get("config", {}).get(k)
            writer.writerow(row)

    logger.info(f"Leaderboard CSV saved to {output_path}")


def save_leaderboard_markdown(
    leaderboard: list[dict[str, Any]],
    output_path: Path,
    sweep_name: str,
    metric: str,
    mode: str,
    sweep_root: Path | None = None,
) -> None:
    """Save leaderboard to Markdown file.

    Args:
        leaderboard: List of leaderboard entries.
        output_path: Path for output Markdown.
        sweep_name: Name of the sweep.
        metric: Primary metric used for ranking.
        mode: Optimization mode (max/min).
    """
    if not leaderboard:
        logger.warning("Empty leaderboard - skipping Markdown output")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Sweep Leaderboard: {sweep_name}")
    lines.append("")
    lines.append(f"**Ranking by:** {metric} ({mode})")
    lines.append(f"**Total runs:** {len(leaderboard)}")

    # Add timing summary if available
    if sweep_root:
        timing_path = sweep_root / "timings.csv"
        if timing_path.exists():
            import csv

            import numpy as np

            elapsed_times = []
            with open(timing_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        elapsed_times.append(float(row["elapsed_s"]))
                    except (KeyError, ValueError):
                        pass

            if elapsed_times:
                p50 = np.percentile(elapsed_times, 50)
                p90 = np.percentile(elapsed_times, 90)
                lines.append("")
                lines.append("### Timing Summary")
                lines.append(f"- **P50 Elapsed:** {p50:.2f}s")
                lines.append(f"- **P90 Elapsed:** {p90:.2f}s")

    lines.append("")

    # Summary table
    lines.append("## Rankings")
    lines.append("")
    lines.append("| Rank | Run | Sharpe | Return % | Max DD % | Trades |")
    lines.append("|------|-----|--------|----------|----------|--------|")

    for entry in leaderboard[:20]:  # Top 20 for markdown
        lines.append(
            f"| {entry['rank']} "
            f"| {entry['run_idx']} "
            f"| {entry['avg_oos_sharpe']:.3f} "
            f"| {entry['avg_oos_return_pct']:.2f}% "
            f"| {entry['avg_oos_max_dd_pct']:.2f}% "
            f"| {entry['total_oos_trades']} |"
        )

    lines.append("")

    # Best run details
    if leaderboard:
        best = leaderboard[0]
        lines.append("## Best Run Configuration")
        lines.append("")
        lines.append(f"**Run index:** {best['run_idx']}")
        lines.append(f"**Output directory:** `{best['output_dir']}`")
        lines.append("")
        lines.append("### Parameters")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in sorted(best.get("config", {}).items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")

    output_path.write_text("\n".join(lines))
    logger.info(f"Leaderboard Markdown saved to {output_path}")


def export_best_run(
    leaderboard: list[dict[str, Any]],
    export_dir: Path,
    rank: int = 1,
) -> Path | None:
    """Export a specific ranked run to a new directory.

    Args:
        leaderboard: Sorted leaderboard entries.
        export_dir: Directory to export to.
        rank: Which rank to export (1-indexed).

    Returns:
        Path to exported directory or None if failed.
    """
    if not leaderboard:
        logger.error("Empty leaderboard - cannot export")
        return None

    if rank < 1 or rank > len(leaderboard):
        logger.error(f"Invalid rank {rank}, must be 1-{len(leaderboard)}")
        return None

    entry = leaderboard[rank - 1]
    source_dir = Path(entry["output_dir"])

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return None

    # Create export directory
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from source run
    for item in source_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, export_dir / item.name)
        elif item.is_dir():
            shutil.copytree(item, export_dir / item.name, dirs_exist_ok=True)

    # Write export manifest
    manifest = {
        "exported_rank": rank,
        "source_dir": str(source_dir),
        "config": entry.get("config", {}),
        "metrics": {
            "avg_oos_sharpe": entry.get("avg_oos_sharpe"),
            "avg_oos_return_pct": entry.get("avg_oos_return_pct"),
            "avg_oos_max_dd_pct": entry.get("avg_oos_max_dd_pct"),
            "total_oos_trades": entry.get("total_oos_trades"),
        },
    }

    manifest_path = export_dir / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Exported rank {rank} run to {export_dir}")

    return export_dir


def build_leaderboard_from_sweep(
    sweep_root: Path,
    output_dir: Path | None = None,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Build complete leaderboard from a sweep directory.

    Args:
        sweep_root: Path to sweep output directory.
        output_dir: Optional output directory for reports.
        top_n: Optional limit on number of entries.

    Returns:
        Leaderboard list.
    """
    meta, results = load_sweep_results(sweep_root)

    metric = meta.get("metric", "sharpe")
    mode = meta.get("mode", "max")
    name = meta.get("name", "unnamed_sweep")

    leaderboard = generate_leaderboard(results, metric, mode, top_n)

    if output_dir is None:
        output_dir = sweep_root

    # Save outputs
    save_leaderboard_csv(leaderboard, output_dir / "leaderboard.csv", metric)
    save_leaderboard_markdown(
        leaderboard, output_dir / "leaderboard.md", name, metric, mode, sweep_root=sweep_root
    )

    return leaderboard
