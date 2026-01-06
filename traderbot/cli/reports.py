"""CLI for report analysis and summarization.

Provides tools for analyzing historical regression reports.

Usage:
    python -m traderbot.cli.reports summarize \
        --history reports/history.json \
        --since 2026-01-01 \
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from traderbot.cli._console import _fmt, configure_windows_console
from traderbot.logging_setup import get_logger

logger = get_logger("cli.reports")

# Output format options
FORMAT_TEXT = "text"
FORMAT_JSON = "json"
FORMAT_CSV = "csv"


def load_history(history_path: Path) -> dict[str, Any]:
    """Load history.json file.

    Args:
        history_path: Path to history.json

    Returns:
        Parsed history data

    Raises:
        FileNotFoundError: If history.json doesn't exist
        json.JSONDecodeError: If history.json is malformed
    """
    with open(history_path, encoding="utf-8") as f:
        return json.load(f)


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        Datetime object
    """
    return datetime.strptime(date_str, "%Y-%m-%d")


def filter_runs(
    runs: list[dict[str, Any]],
    since: datetime | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter runs by date and limit.

    Args:
        runs: List of run records
        since: Only include runs after this date
        limit: Maximum number of runs to return

    Returns:
        Filtered list of runs (most recent first)
    """
    # Sort by generated_utc descending
    sorted_runs = sorted(
        runs,
        key=lambda r: r.get("generated_utc", ""),
        reverse=True,
    )

    if since:
        sorted_runs = [
            r for r in sorted_runs
            if _parse_timestamp(r.get("generated_utc", "")) >= since
        ]

    if limit and limit > 0:
        sorted_runs = sorted_runs[:limit]

    return sorted_runs


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp to datetime.

    Args:
        ts: ISO format timestamp string

    Returns:
        Datetime object (naive, for comparison)
    """
    if not ts:
        return datetime.min

    ts = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=None)  # Make naive for comparison
    except ValueError:
        try:
            return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return datetime.min


def compute_summary_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from runs.

    Args:
        runs: List of run records

    Returns:
        Dictionary with summary statistics
    """
    if not runs:
        return {
            "total_runs": 0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": 0.0,
            "sharpe_delta_mean": None,
            "sharpe_delta_std": None,
            "timing_p90_mean": None,
            "timing_p90_std": None,
            "date_range": None,
        }

    pass_count = sum(1 for r in runs if r.get("verdict") == "PASS")
    fail_count = sum(1 for r in runs if r.get("verdict") == "FAIL")
    total = len(runs)

    # Sharpe delta stats
    sharpe_deltas = [
        r.get("sharpe_delta") for r in runs
        if r.get("sharpe_delta") is not None
    ]

    # Timing P90 stats
    timing_p90s = [
        r.get("timing_p90") for r in runs
        if r.get("timing_p90") is not None
    ]

    # Date range
    dates = [r.get("generated_utc", "") for r in runs if r.get("generated_utc")]
    dates_sorted = sorted(dates)

    return {
        "total_runs": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_count / total if total > 0 else 0.0,
        "sharpe_delta_mean": _mean(sharpe_deltas),
        "sharpe_delta_std": _std(sharpe_deltas),
        "timing_p90_mean": _mean(timing_p90s),
        "timing_p90_std": _std(timing_p90s),
        "date_range": {
            "start": dates_sorted[0][:10] if dates_sorted else None,
            "end": dates_sorted[-1][:10] if dates_sorted else None,
        } if dates_sorted else None,
    }


def _mean(values: list[float]) -> float | None:
    """Compute mean of values."""
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    """Compute standard deviation of values."""
    if len(values) < 2:
        return None
    m = _mean(values)
    if m is None:
        return None
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return variance ** 0.5


def detect_flaky(runs: list[dict[str, Any]], window: int = 10, min_runs: int = 6) -> dict[str, Any]:
    """Detect flakiness in recent runs.

    Rule: "flaky" if |Sharpe Delta| mean < 0.02 and stdev >= 2*|mean| over window.

    Args:
        runs: List of run records (most recent first)
        window: Number of recent runs to analyze
        min_runs: Minimum runs required for analysis

    Returns:
        Flakiness analysis result
    """
    recent = runs[:window]

    if len(recent) < min_runs:
        return {
            "is_flaky": False,
            "reason": f"insufficient_data (need {min_runs}, have {len(recent)})",
            "window": window,
            "analyzed_runs": len(recent),
        }

    sharpe_deltas = [
        r.get("sharpe_delta") for r in recent
        if r.get("sharpe_delta") is not None
    ]

    if len(sharpe_deltas) < min_runs:
        return {
            "is_flaky": False,
            "reason": f"insufficient_sharpe_data (need {min_runs}, have {len(sharpe_deltas)})",
            "window": window,
            "analyzed_runs": len(sharpe_deltas),
        }

    mean_val = _mean(sharpe_deltas)
    std_val = _std(sharpe_deltas)

    if mean_val is None or std_val is None:
        return {
            "is_flaky": False,
            "reason": "calculation_error",
            "window": window,
            "analyzed_runs": len(sharpe_deltas),
        }

    abs_mean = abs(mean_val)
    is_flaky = abs_mean < 0.02 and std_val >= 2 * abs_mean

    return {
        "is_flaky": is_flaky,
        "reason": "high_variance_low_mean" if is_flaky else "stable",
        "window": window,
        "analyzed_runs": len(sharpe_deltas),
        "sharpe_delta_mean": mean_val,
        "sharpe_delta_std": std_val,
        "threshold_mean": 0.02,
        "threshold_std_ratio": 2.0,
    }


def format_text_summary(stats: dict[str, Any], runs: list[dict[str, Any]], flaky: dict[str, Any]) -> str:
    """Format summary as human-readable text.

    Args:
        stats: Summary statistics
        runs: List of runs
        flaky: Flakiness analysis

    Returns:
        Formatted text
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  REGRESSION REPORT SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"  Total runs:     {stats['total_runs']}")
    lines.append(f"  Pass count:     {stats['pass_count']}")
    lines.append(f"  Fail count:     {stats['fail_count']}")
    lines.append(f"  Pass rate:      {stats['pass_rate']:.1%}")
    lines.append("")

    # Date range
    if stats.get("date_range"):
        dr = stats["date_range"]
        lines.append(f"  Date range:     {dr['start']} to {dr['end']}")
        lines.append("")

    # Metrics
    lines.append("METRICS")
    lines.append("-" * 40)

    if stats.get("sharpe_delta_mean") is not None:
        sign = "+" if stats["sharpe_delta_mean"] >= 0 else ""
        std_str = f"{stats['sharpe_delta_std']:.4f}" if stats.get("sharpe_delta_std") is not None else "N/A"
        lines.append(f"  Sharpe Delta:   {sign}{stats['sharpe_delta_mean']:.4f} (std: {std_str})")

    if stats.get("timing_p90_mean") is not None:
        std_str = f"{stats['timing_p90_std']:.2f}" if stats.get("timing_p90_std") is not None else "N/A"
        lines.append(f"  Timing P90:     {stats['timing_p90_mean']:.2f}s (std: {std_str}s)")

    lines.append("")

    # Flakiness
    lines.append("STABILITY")
    lines.append("-" * 40)
    if flaky.get("is_flaky"):
        lines.append("  Status:         [FLAKY] High variance detected")
    else:
        lines.append("  Status:         [STABLE]")
    lines.append(f"  Reason:         {flaky.get('reason', 'unknown')}")
    lines.append(f"  Analyzed runs:  {flaky.get('analyzed_runs', 0)}")
    lines.append("")

    # Recent runs table
    if runs:
        lines.append("RECENT RUNS")
        lines.append("-" * 40)
        lines.append(f"  {'Run ID':<20} {'Verdict':<8} {'Sharpe Delta':<14} {'P90':<8}")
        lines.append(f"  {'-'*20} {'-'*8} {'-'*14} {'-'*8}")

        for run in runs[:10]:
            run_id = run.get("run_id", "unknown")[:20]
            verdict = run.get("verdict", "?")
            sd = run.get("sharpe_delta")
            p90 = run.get("timing_p90")

            sd_str = f"{sd:+.4f}" if sd is not None else "N/A"
            p90_str = f"{p90:.2f}s" if p90 is not None else "N/A"

            lines.append(f"  {run_id:<20} {verdict:<8} {sd_str:<14} {p90_str:<8}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_json_summary(stats: dict[str, Any], runs: list[dict[str, Any]], flaky: dict[str, Any]) -> str:
    """Format summary as JSON.

    Args:
        stats: Summary statistics
        runs: List of runs
        flaky: Flakiness analysis

    Returns:
        JSON string
    """
    output = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "summary": stats,
        "flakiness": flaky,
        "runs": [
            {
                "run_id": r.get("run_id"),
                "verdict": r.get("verdict"),
                "sharpe_delta": r.get("sharpe_delta"),
                "timing_p90": r.get("timing_p90"),
                "generated_utc": r.get("generated_utc"),
            }
            for r in runs[:20]
        ],
    }
    return json.dumps(output, indent=2)


def format_csv_summary(runs: list[dict[str, Any]]) -> str:
    """Format runs as CSV.

    Args:
        runs: List of runs

    Returns:
        CSV string
    """
    lines = ["run_id,verdict,sharpe_delta,timing_p90,generated_utc"]

    for r in runs:
        run_id = r.get("run_id", "")
        verdict = r.get("verdict", "")
        sd = r.get("sharpe_delta")
        p90 = r.get("timing_p90")
        ts = r.get("generated_utc", "")

        sd_str = f"{sd:.6f}" if sd is not None else ""
        p90_str = f"{p90:.4f}" if p90 is not None else ""

        lines.append(f"{run_id},{verdict},{sd_str},{p90_str},{ts}")

    return "\n".join(lines)


def cmd_summarize(args: argparse.Namespace) -> int:
    """Run summarize command.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    history_path = Path(args.history)
    use_emoji = not args.no_emoji

    if not history_path.exists():
        logger.error(f"History file not found: {history_path}")
        return 1

    try:
        history = load_history(history_path)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {history_path}: {e}")
        return 1

    runs = history.get("runs", [])

    # Apply filters
    since = parse_date(args.since) if args.since else None
    limit = args.limit if args.limit and args.limit > 0 else None

    filtered_runs = filter_runs(runs, since=since, limit=limit)

    if not filtered_runs:
        print("No runs found matching criteria.")
        return 0

    # Compute stats
    stats = compute_summary_stats(filtered_runs)
    flaky = detect_flaky(filtered_runs)

    # Format output
    output_format = args.format.lower()

    if output_format == FORMAT_JSON:
        output = format_json_summary(stats, filtered_runs, flaky)
    elif output_format == FORMAT_CSV:
        output = format_csv_summary(filtered_runs)
    else:
        output = format_text_summary(stats, filtered_runs, flaky)

    # Write output
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        print(_fmt("📊 Summary written to:", "[OUT] Summary written to:", use_emoji) + f" {output_path}")
    else:
        print(output)

    return 0


def main() -> None:
    """Main entry point for reports CLI."""
    parser = argparse.ArgumentParser(
        description="Report analysis and summarization tools"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize historical regression reports",
    )
    summarize_parser.add_argument(
        "--history",
        type=str,
        required=True,
        help="Path to history.json file",
    )
    summarize_parser.add_argument(
        "--since",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only include runs after this date",
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of runs to include",
    )
    summarize_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)",
    )
    summarize_parser.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="PATH",
        help="Write output to file instead of stdout",
    )
    summarize_parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in output (useful for Windows/CI)",
    )

    args = parser.parse_args()

    # Configure Windows console
    configure_windows_console()

    if args.command == "summarize":
        sys.exit(cmd_summarize(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
