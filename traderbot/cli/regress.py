"""CLI for regression detection and performance guard checks.

Compares current sweep results against a baseline and performance budget.

Usage:
    python -m traderbot.cli.regress compare \
        --current runs/sweeps/ci_smoke \
        --baseline benchmarks/baseline.json \
        --budget sweeps/perf_budget.yaml \
        --out runs/sweeps/ci_smoke/regression_report.md
"""

import argparse
import json
import locale
import os
import subprocess
import sys
from pathlib import Path

from traderbot.logging_setup import get_logger
from traderbot.metrics.compare import (
    compare_results,
    create_new_baseline,
    generate_baseline_diff,
    generate_regression_report,
    load_baseline,
    load_current_data,
    load_perf_budget,
)

logger = get_logger("cli.regress")


def _can_encode(s: str) -> bool:
    """Check if string can be encoded with current stdout encoding.

    Args:
        s: String to check.

    Returns:
        True if encodable, False otherwise.
    """
    try:
        encoding = sys.stdout.encoding or "utf-8"
        s.encode(encoding, errors="strict")
        return True
    except Exception:
        return False


def _fmt(s: str, ascii_fallback: str, use_emoji: bool) -> str:
    """Format string with emoji fallback to ASCII.

    Args:
        s: String with emoji.
        ascii_fallback: ASCII fallback string.
        use_emoji: Whether to attempt emoji.

    Returns:
        Formatted string (emoji if possible, ASCII otherwise).
    """
    if not use_emoji:
        return ascii_fallback
    if _can_encode(s):
        return s
    return ascii_fallback


def get_git_sha() -> str:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def cmd_compare(args: argparse.Namespace) -> int:
    """Run comparison command.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code (0 for pass, 1 for fail).
    """
    use_emoji = not args.no_emoji

    current_dir = Path(args.current)
    baseline_path = Path(args.baseline)
    budget_path = Path(args.budget)
    output_path = Path(args.out) if args.out else current_dir / "regression_report.md"

    # Validate paths
    if not current_dir.exists():
        logger.error(f"Current sweep directory not found: {current_dir}")
        return 1

    if not baseline_path.exists():
        logger.error(f"Baseline file not found: {baseline_path}")
        return 1

    if not budget_path.exists():
        logger.error(f"Performance budget file not found: {budget_path}")
        return 1

    # Load data
    logger.info(f"Loading current data from {current_dir}")
    current = load_current_data(current_dir)

    logger.info(f"Loading baseline from {baseline_path}")
    baseline = load_baseline(baseline_path)

    logger.info(f"Loading performance budget from {budget_path}")
    budget = load_perf_budget(budget_path)

    # Run comparison
    logger.info("Running regression comparison...")
    verdict = compare_results(current, baseline, budget)

    # Generate report
    report = generate_regression_report(verdict, current, baseline, budget)
    
    # Add provenance notes if fallbacks were used
    if current.used_fallback_leaderboard or current.used_fallback_timings:
        report += "\n### Data Provenance\n\n"
        if current.used_fallback_leaderboard:
            report += "- `best_metric` derived from `all_results.json` (leaderboard.csv missing or empty)\n"
        if current.used_fallback_timings:
            report += "- Timing percentiles (P50/P90) derived from `all_results.json` (timings.csv missing)\n"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Regression report written to {output_path}")

    # Generate diff JSON
    diff = generate_baseline_diff(verdict, current, baseline)
    diff_path = output_path.parent / "baseline_diff.json"
    with open(diff_path, "w") as f:
        json.dump(diff, f, indent=2)
    logger.info(f"Baseline diff written to {diff_path}")

    # Print summary
    print("\n" + "=" * 60)
    if verdict.passed:
        print(_fmt("✅ REGRESSION CHECK PASSED", "[PASS] REGRESSION CHECK PASSED", use_emoji))
    else:
        print(_fmt("❌ REGRESSION CHECK FAILED", "[FAIL] REGRESSION CHECK FAILED", use_emoji))
    print("=" * 60)

    for msg in verdict.messages[:5]:  # First 5 messages
        print(f"  {msg}")

    print(f"\nFull report: {output_path}")
    print("=" * 60 + "\n")

    return 0 if verdict.passed else 1


def cmd_update_baseline(args: argparse.Namespace) -> int:
    """Update baseline from current sweep results.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    current_dir = Path(args.current)
    output_path = Path(args.out)

    if not current_dir.exists():
        logger.error(f"Current sweep directory not found: {current_dir}")
        return 1

    # Load current data
    logger.info(f"Loading current data from {current_dir}")
    current = load_current_data(current_dir)

    # Get git SHA
    git_sha = args.sha if args.sha else get_git_sha()

    # Create new baseline
    baseline = create_new_baseline(current, git_sha)

    # Write baseline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)

    logger.info(f"New baseline written to {output_path}")
    use_emoji = not args.no_emoji
    print(_fmt("✅ Baseline updated:", "[OK] Baseline updated:", use_emoji) + f" {output_path}")
    print(f"   Git SHA: {git_sha}")
    print(f"   Best metric: {current.best_metric:.4f}")
    print(f"   Success rate: {current.success_rate:.2%}")
    print(f"   Timing P90: {current.timing['p90']:.2f}s")

    return 0


def main() -> None:
    """Main entry point for regress CLI."""
    parser = argparse.ArgumentParser(
        description="Regression detection and performance guard checks"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare current sweep against baseline",
    )
    compare_parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to current sweep output directory",
    )
    compare_parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline.json file",
    )
    compare_parser.add_argument(
        "--budget",
        type=str,
        required=True,
        help="Path to perf_budget.yaml file",
    )
    compare_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for regression_report.md",
    )
    compare_parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (useful for Windows/CI)",
    )

    # Update baseline command
    update_parser = subparsers.add_parser(
        "update-baseline",
        help="Create/update baseline from current sweep",
    )
    update_parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to current sweep output directory",
    )
    update_parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for baseline.json",
    )
    update_parser.add_argument(
        "--sha",
        type=str,
        default=None,
        help="Git SHA to use (defaults to current HEAD)",
    )
    update_parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (useful for Windows/CI)",
    )

    args = parser.parse_args()

    # Try to improve Windows console encoding (best-effort)
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            # Fallback gracefully if reconfigure fails
            pass

    if args.command == "compare":
        sys.exit(cmd_compare(args))
    elif args.command == "update-baseline":
        sys.exit(cmd_update_baseline(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
