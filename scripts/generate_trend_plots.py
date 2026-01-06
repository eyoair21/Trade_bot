#!/usr/bin/env python3
"""Generate trend plots from nightly sweep results.

Aggregates metrics from baseline_diff.json files across runs and generates
matplotlib trend plots showing performance over time.

Usage:
    python scripts/generate_trend_plots.py \
        --trend-file runs/sweeps/trend_data.json \
        --current-diff runs/sweeps/ci_smoke/baseline_diff.json \
        --output-dir runs/sweeps/ci_smoke/plots

The script maintains a rolling trend file with historical data points.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Optional matplotlib import - graceful fallback if not available
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for CI
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_trend_data(trend_path: Path) -> dict:
    """Load existing trend data or create empty structure.

    Args:
        trend_path: Path to trend_data.json file.

    Returns:
        Dictionary with trend data structure.
    """
    if trend_path.exists():
        with open(trend_path) as f:
            return json.load(f)

    return {
        "version": "1.0",
        "data_points": [],
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def append_data_point(
    trend_data: dict,
    diff_data: dict,
    timestamp: str | None = None,
    git_sha: str | None = None,
    max_points: int = 90,
) -> dict:
    """Append a new data point from baseline_diff.json.

    Args:
        trend_data: Existing trend data dictionary.
        diff_data: Data from baseline_diff.json.
        timestamp: ISO timestamp (defaults to now).
        git_sha: Git commit SHA.
        max_points: Maximum data points to keep (rolling window).

    Returns:
        Updated trend data dictionary.
    """
    if timestamp is None:
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    data_point = {
        "timestamp": timestamp,
        "git_sha": git_sha or diff_data.get("current_sha", "unknown"),
        "best_metric": diff_data.get("current", {}).get("best_metric"),
        "baseline_metric": diff_data.get("baseline", {}).get("best_metric"),
        "delta": diff_data.get("delta", {}).get("best_metric"),
        "success_rate": diff_data.get("current", {}).get("success_rate"),
        "timing_p90": diff_data.get("current", {}).get("timing_p90"),
        "passed": diff_data.get("verdict", {}).get("passed"),
    }

    trend_data["data_points"].append(data_point)
    trend_data["last_updated"] = timestamp

    # Trim to max_points (rolling window)
    if len(trend_data["data_points"]) > max_points:
        trend_data["data_points"] = trend_data["data_points"][-max_points:]

    return trend_data


def generate_plots(trend_data: dict, output_dir: Path) -> list[Path]:
    """Generate trend plots from aggregated data.

    Args:
        trend_data: Trend data dictionary.
        output_dir: Directory to save plots.

    Returns:
        List of generated plot file paths.
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot generation")
        return []

    data_points = trend_data.get("data_points", [])
    if len(data_points) < 2:
        print("Insufficient data points for trend plot (need at least 2)")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # Parse timestamps
    timestamps = []
    metrics = []
    success_rates = []
    timing_p90s = []
    passed_flags = []

    for dp in data_points:
        try:
            ts = datetime.fromisoformat(dp["timestamp"].replace("Z", "+00:00"))
            timestamps.append(ts)
            metrics.append(dp.get("best_metric"))
            success_rates.append(dp.get("success_rate"))
            timing_p90s.append(dp.get("timing_p90"))
            passed_flags.append(dp.get("passed"))
        except (ValueError, KeyError):
            continue

    if len(timestamps) < 2:
        print("Could not parse enough valid data points")
        return []

    # Plot 1: Best Metric Trend
    fig, ax = plt.subplots(figsize=(10, 5))
    valid_metrics = [(t, m) for t, m in zip(timestamps, metrics) if m is not None]
    if valid_metrics:
        ts, vals = zip(*valid_metrics)
        ax.plot(ts, vals, marker="o", linewidth=2, markersize=4, color="#2563eb")
        ax.fill_between(ts, vals, alpha=0.1, color="#2563eb")
        ax.set_xlabel("Date")
        ax.set_ylabel("Best Metric (Sharpe)")
        ax.set_title("Best Metric Trend (Nightly Sweeps)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        metric_plot_path = output_dir / "trend_metric.png"
        plt.savefig(metric_plot_path, dpi=150)
        plt.close()
        generated_files.append(metric_plot_path)

    # Plot 2: Success Rate Trend
    fig, ax = plt.subplots(figsize=(10, 5))
    valid_sr = [(t, s) for t, s in zip(timestamps, success_rates) if s is not None]
    if valid_sr:
        ts, vals = zip(*valid_sr)
        ax.plot(ts, [v * 100 for v in vals], marker="s", linewidth=2, markersize=4, color="#16a34a")
        ax.axhline(y=75, color="#ef4444", linestyle="--", alpha=0.7, label="Budget (75%)")
        ax.fill_between(ts, [v * 100 for v in vals], alpha=0.1, color="#16a34a")
        ax.set_xlabel("Date")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate Trend (Nightly Sweeps)")
        ax.set_ylim(0, 105)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        sr_plot_path = output_dir / "trend_success_rate.png"
        plt.savefig(sr_plot_path, dpi=150)
        plt.close()
        generated_files.append(sr_plot_path)

    # Plot 3: Timing P90 Trend
    fig, ax = plt.subplots(figsize=(10, 5))
    valid_timing = [(t, p) for t, p in zip(timestamps, timing_p90s) if p is not None]
    if valid_timing:
        ts, vals = zip(*valid_timing)
        ax.plot(ts, vals, marker="^", linewidth=2, markersize=4, color="#9333ea")
        ax.axhline(y=60, color="#ef4444", linestyle="--", alpha=0.7, label="Budget (60s)")
        ax.fill_between(ts, vals, alpha=0.1, color="#9333ea")
        ax.set_xlabel("Date")
        ax.set_ylabel("P90 Elapsed Time (seconds)")
        ax.set_title("Timing P90 Trend (Nightly Sweeps)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        timing_plot_path = output_dir / "trend_timing.png"
        plt.savefig(timing_plot_path, dpi=150)
        plt.close()
        generated_files.append(timing_plot_path)

    # Plot 4: Combined Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Metric subplot
    ax1 = axes[0, 0]
    if valid_metrics:
        ts, vals = zip(*valid_metrics)
        ax1.plot(ts, vals, marker="o", linewidth=2, markersize=3, color="#2563eb")
        ax1.set_ylabel("Best Metric")
        ax1.set_title("Best Metric (Sharpe)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax1.grid(True, alpha=0.3)

    # Success rate subplot
    ax2 = axes[0, 1]
    if valid_sr:
        ts, vals = zip(*valid_sr)
        ax2.plot(ts, [v * 100 for v in vals], marker="s", linewidth=2, markersize=3, color="#16a34a")
        ax2.axhline(y=75, color="#ef4444", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_title("Success Rate")
        ax2.set_ylim(0, 105)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax2.grid(True, alpha=0.3)

    # Timing subplot
    ax3 = axes[1, 0]
    if valid_timing:
        ts, vals = zip(*valid_timing)
        ax3.plot(ts, vals, marker="^", linewidth=2, markersize=3, color="#9333ea")
        ax3.axhline(y=60, color="#ef4444", linestyle="--", alpha=0.5)
        ax3.set_ylabel("P90 Time (s)")
        ax3.set_title("Timing P90")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax3.grid(True, alpha=0.3)

    # Pass/fail history subplot
    ax4 = axes[1, 1]
    valid_passed = [(t, p) for t, p in zip(timestamps, passed_flags) if p is not None]
    if valid_passed:
        ts, vals = zip(*valid_passed)
        colors = ["#16a34a" if v else "#ef4444" for v in vals]
        ax4.bar(ts, [1] * len(ts), color=colors, width=0.8)
        ax4.set_ylabel("Pass/Fail")
        ax4.set_title("Regression Status")
        ax4.set_yticks([])
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#16a34a", label="Pass"),
            Patch(facecolor="#ef4444", label="Fail"),
        ]
        ax4.legend(handles=legend_elements, loc="upper right")

    fig.suptitle("TraderBot Nightly Sweep Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()

    dashboard_path = output_dir / "trend_dashboard.png"
    plt.savefig(dashboard_path, dpi=150)
    plt.close()
    generated_files.append(dashboard_path)

    return generated_files


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate trend plots from nightly sweep results"
    )
    parser.add_argument(
        "--trend-file",
        type=str,
        required=True,
        help="Path to trend_data.json (will be created/updated)",
    )
    parser.add_argument(
        "--current-diff",
        type=str,
        required=True,
        help="Path to current baseline_diff.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save generated plots",
    )
    parser.add_argument(
        "--git-sha",
        type=str,
        default=None,
        help="Git SHA for this data point",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=90,
        help="Maximum data points to keep (default: 90 days)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only update trend data, skip plot generation",
    )

    args = parser.parse_args()

    trend_path = Path(args.trend_file)
    diff_path = Path(args.current_diff)
    output_dir = Path(args.output_dir)

    # Load current diff
    if not diff_path.exists():
        print(f"Error: baseline_diff.json not found: {diff_path}")
        return 1

    with open(diff_path) as f:
        diff_data = json.load(f)

    # Load/create trend data
    trend_data = load_trend_data(trend_path)

    # Append new data point
    trend_data = append_data_point(
        trend_data,
        diff_data,
        git_sha=args.git_sha,
        max_points=args.max_points,
    )

    # Save updated trend data
    trend_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trend_path, "w") as f:
        json.dump(trend_data, f, indent=2)
    print(f"Updated trend data: {trend_path}")
    print(f"  Total data points: {len(trend_data['data_points'])}")

    # Generate plots
    if not args.skip_plots:
        generated = generate_plots(trend_data, output_dir)
        if generated:
            print(f"Generated {len(generated)} plots:")
            for p in generated:
                print(f"  - {p}")
        elif not HAS_MATPLOTLIB:
            print("Skipped plot generation (matplotlib not available)")
        else:
            print("No plots generated (insufficient data)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
