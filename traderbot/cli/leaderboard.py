"""CLI for generating leaderboards and exporting best runs.

Usage:
    # Generate leaderboard from sweep
    python -m traderbot.cli.leaderboard runs/sweep_fixed_vs_vol

    # Export best run
    python -m traderbot.cli.leaderboard runs/sweep_fixed_vs_vol --export-best runs/best_run

    # Export specific rank
    python -m traderbot.cli.leaderboard runs/sweep_fixed_vs_vol --export-rank 3 --export-dir runs/third_best
"""

import argparse
import sys
from pathlib import Path

from traderbot.logging_setup import get_logger
from traderbot.reports.leaderboard import (
    build_leaderboard_from_sweep,
    export_best_run,
)

logger = get_logger("cli.leaderboard")


def main() -> None:
    """Main entry point for leaderboard CLI."""
    parser = argparse.ArgumentParser(
        description="Generate leaderboards and export best runs from sweeps"
    )
    parser.add_argument(
        "sweep_dir",
        type=str,
        help="Path to sweep output directory",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Limit leaderboard to top N entries",
    )
    parser.add_argument(
        "--export-best",
        type=str,
        default=None,
        metavar="DIR",
        help="Export best run to specified directory",
    )
    parser.add_argument(
        "--export-rank",
        type=int,
        default=1,
        help="Which rank to export (default: 1 = best)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for export (use with --export-rank)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for leaderboard output files",
    )

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        logger.error(f"Sweep directory not found: {sweep_dir}")
        sys.exit(1)

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir

    try:
        # Build leaderboard
        leaderboard = build_leaderboard_from_sweep(
            sweep_root=sweep_dir,
            output_dir=output_dir,
            top_n=args.top_n,
        )

        if not leaderboard:
            logger.warning("No valid runs found in sweep")
            sys.exit(1)

        # Display summary
        print(f"\n{'='*60}")
        print(f"Leaderboard generated from {sweep_dir}")
        print(f"Total runs ranked: {len(leaderboard)}")
        print(f"{'='*60}\n")

        # Show top 5
        print("Top 5 runs:")
        print("-" * 60)
        for entry in leaderboard[:5]:
            print(
                f"  #{entry['rank']}: Run {entry['run_idx']} - "
                f"Sharpe: {entry['avg_oos_sharpe']:.3f}, "
                f"Return: {entry['avg_oos_return_pct']:.2f}%, "
                f"MaxDD: {entry['avg_oos_max_dd_pct']:.2f}%"
            )
        print()

        # Handle exports
        if args.export_best:
            export_path = Path(args.export_best)
            result = export_best_run(leaderboard, export_path, rank=1)
            if result:
                print(f"Best run exported to: {result}")
            else:
                logger.error("Failed to export best run")
                sys.exit(1)

        if args.export_dir and args.export_rank:
            export_path = Path(args.export_dir)
            result = export_best_run(leaderboard, export_path, rank=args.export_rank)
            if result:
                print(f"Rank {args.export_rank} run exported to: {result}")
            else:
                logger.error(f"Failed to export rank {args.export_rank} run")
                sys.exit(1)

        print(f"Leaderboard files saved to: {output_dir}")
        print("  - leaderboard.csv")
        print("  - leaderboard.md")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error generating leaderboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
