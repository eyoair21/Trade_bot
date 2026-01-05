"""Compare two walk-forward runs.

Loads results from two run directories and generates a comparison report.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from traderbot.logging_setup import get_logger
from traderbot.reports.metrics import max_drawdown, sharpe_simple, total_return

logger = get_logger("cli.compare_runs")


def load_run_data(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load results.json and equity_curve.csv from a run directory.
    
    Args:
        run_dir: Path to run directory.
        
    Returns:
        Tuple of (results dict, equity_curve DataFrame).
    """
    results_path = run_dir / "results.json"
    equity_path = run_dir / "equity_curve.csv"
    
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")
    if not equity_path.exists():
        raise FileNotFoundError(f"equity_curve.csv not found in {run_dir}")
    
    with open(results_path) as f:
        results = json.load(f)
    
    equity_df = pd.read_csv(equity_path)
    
    return results, equity_df


def compute_metrics(equity_df: pd.DataFrame) -> dict[str, float]:
    """Compute comparison metrics from equity curve.
    
    Args:
        equity_df: Equity curve DataFrame with 'equity' column.
        
    Returns:
        Dict with total_return, sharpe, max_dd.
    """
    if equity_df.empty or "equity" not in equity_df.columns:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
        }
    
    equity_series = equity_df["equity"]
    
    return {
        "total_return": total_return(equity_series),
        "sharpe": sharpe_simple(equity_series),
        "max_dd": max_drawdown(equity_series),
    }


def generate_comparison_report(
    run_a_dir: Path,
    run_b_dir: Path,
    results_a: dict[str, Any],
    results_b: dict[str, Any],
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    metric: str,
) -> str:
    """Generate markdown comparison report.
    
    Args:
        run_a_dir: Path to run A directory.
        run_b_dir: Path to run B directory.
        results_a: Results dict for run A.
        results_b: Results dict for run B.
        metrics_a: Computed metrics for run A.
        metrics_b: Computed metrics for run B.
        metric: Metric to use for winner determination.
        
    Returns:
        Markdown report string.
    """
    lines: list[str] = []
    
    # Header
    lines.append("# Run Comparison Report")
    lines.append("")
    lines.append(f"**Run A:** `{run_a_dir.name}`")
    lines.append(f"**Run B:** `{run_b_dir.name}`")
    lines.append(f"**Comparison Metric:** {metric}")
    lines.append("")
    
    # Run details table
    lines.append("## Run Details")
    lines.append("")
    lines.append("| Parameter | Run A | Run B |")
    lines.append("|-----------|-------|-------|")
    
    # Extract manifest info if available
    manifest_a = results_a.get("manifest", {})
    manifest_b = results_b.get("manifest", {})
    
    lines.append(f"| Run ID | {manifest_a.get('run_id', run_a_dir.name)} | {manifest_b.get('run_id', run_b_dir.name)} |")
    lines.append(f"| Git SHA | {manifest_a.get('git_sha', results_a.get('git_sha', 'N/A'))} | {manifest_b.get('git_sha', results_b.get('git_sha', 'N/A'))} |")
    lines.append(f"| Seed | {manifest_a.get('seed', results_a.get('seed', 'N/A'))} | {manifest_b.get('seed', results_b.get('seed', 'N/A'))} |")
    lines.append(f"| Period | {results_a.get('start_date', 'N/A')} to {results_a.get('end_date', 'N/A')} | {results_b.get('start_date', 'N/A')} to {results_b.get('end_date', 'N/A')} |")
    lines.append(f"| Universe | {', '.join(results_a.get('universe', []))} | {', '.join(results_b.get('universe', []))} |")
    lines.append(f"| Splits | {results_a.get('n_splits', 0)} | {results_b.get('n_splits', 0)} |")
    lines.append(f"| IS Ratio | {results_a.get('is_ratio', 0.0)} | {results_b.get('is_ratio', 0.0)} |")
    lines.append(f"| Sizer | {results_a.get('sizer', 'N/A')} | {results_b.get('sizer', 'N/A')} |")
    lines.append("")
    
    # Performance comparison table
    lines.append("## Performance Comparison")
    lines.append("")
    lines.append("| Metric | Run A | Run B | Difference |")
    lines.append("|--------|-------|-------|------------|")
    
    # Total Return
    ret_a = metrics_a["total_return"]
    ret_b = metrics_b["total_return"]
    ret_diff = ret_b - ret_a
    ret_diff_str = f"+{ret_diff:.2f}%" if ret_diff > 0 else f"{ret_diff:.2f}%"
    lines.append(f"| Total Return | {ret_a:.2f}% | {ret_b:.2f}% | {ret_diff_str} |")
    
    # Sharpe Ratio
    sharpe_a = metrics_a["sharpe"]
    sharpe_b = metrics_b["sharpe"]
    sharpe_diff = sharpe_b - sharpe_a
    sharpe_diff_str = f"+{sharpe_diff:.3f}" if sharpe_diff > 0 else f"{sharpe_diff:.3f}"
    lines.append(f"| Sharpe Ratio | {sharpe_a:.3f} | {sharpe_b:.3f} | {sharpe_diff_str} |")
    
    # Max Drawdown (lower is better)
    dd_a = metrics_a["max_dd"]
    dd_b = metrics_b["max_dd"]
    dd_diff = dd_b - dd_a
    dd_diff_str = f"+{dd_diff:.2f}%" if dd_diff > 0 else f"{dd_diff:.2f}%"
    lines.append(f"| Max Drawdown | {dd_a:.2f}% | {dd_b:.2f}% | {dd_diff_str} |")
    
    # Trades
    trades_a = results_a.get("total_oos_trades", 0)
    trades_b = results_b.get("total_oos_trades", 0)
    lines.append(f"| Total Trades | {trades_a} | {trades_b} | {trades_b - trades_a:+d} |")
    
    lines.append("")
    
    # Determine winner
    lines.append("## Winner")
    lines.append("")
    
    if metric == "total_return":
        winner = "Run A" if ret_a > ret_b else "Run B" if ret_b > ret_a else "Tie"
        winner_value = max(ret_a, ret_b)
        lines.append(f"**{winner}** has higher total return ({winner_value:.2f}%)")
    elif metric == "sharpe":
        winner = "Run A" if sharpe_a > sharpe_b else "Run B" if sharpe_b > sharpe_a else "Tie"
        winner_value = max(sharpe_a, sharpe_b)
        lines.append(f"**{winner}** has higher Sharpe ratio ({winner_value:.3f})")
    elif metric == "max_dd":
        # Lower drawdown is better
        winner = "Run A" if dd_a < dd_b else "Run B" if dd_b < dd_a else "Tie"
        winner_value = min(dd_a, dd_b)
        lines.append(f"**{winner}** has lower max drawdown ({winner_value:.2f}%)")
    else:
        lines.append("**Unknown metric**")
    
    lines.append("")
    
    return "\n".join(lines)


def compare_runs(
    run_a: Path,
    run_b: Path,
    metric: str = "sharpe",
    output_path: Path | None = None,
) -> None:
    """Compare two walk-forward runs.
    
    Args:
        run_a: Path to first run directory.
        run_b: Path to second run directory.
        metric: Metric to use for winner determination (total_return, sharpe, max_dd).
        output_path: Output path for comparison report. Auto-generated if None.
    """
    logger.info(f"Comparing runs: {run_a.name} vs {run_b.name}")
    
    # Load data
    results_a, equity_a = load_run_data(run_a)
    results_b, equity_b = load_run_data(run_b)
    
    # Compute metrics
    metrics_a = compute_metrics(equity_a)
    metrics_b = compute_metrics(equity_b)
    
    logger.info(f"Run A metrics: {metrics_a}")
    logger.info(f"Run B metrics: {metrics_b}")
    
    # Generate report
    report = generate_comparison_report(
        run_a, run_b,
        results_a, results_b,
        metrics_a, metrics_b,
        metric,
    )
    
    # Determine output path
    if output_path is None:
        output_path = run_a.parent / f"compare_{run_a.name}_vs_{run_b.name}.md"
    
    # Write report
    output_path.write_text(report)
    logger.info(f"Comparison report written to {output_path}")
    
    # Print winner to console
    if metric == "total_return":
        winner = "A" if metrics_a["total_return"] > metrics_b["total_return"] else "B"
        print(f"\nWinner: Run {winner} (total return: {max(metrics_a['total_return'], metrics_b['total_return']):.2f}%)")
    elif metric == "sharpe":
        winner = "A" if metrics_a["sharpe"] > metrics_b["sharpe"] else "B"
        print(f"\nWinner: Run {winner} (Sharpe: {max(metrics_a['sharpe'], metrics_b['sharpe']):.3f})")
    elif metric == "max_dd":
        winner = "A" if metrics_a["max_dd"] < metrics_b["max_dd"] else "B"
        print(f"\nWinner: Run {winner} (max DD: {min(metrics_a['max_dd'], metrics_b['max_dd']):.2f}%)")


def main() -> None:
    """Main entry point for compare_runs CLI."""
    parser = argparse.ArgumentParser(
        description="Compare two walk-forward backtest runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m traderbot.cli.compare_runs --a runs/runA --b runs/runB
  python -m traderbot.cli.compare_runs --a runs/runA --b runs/runB --metric total_return
  python -m traderbot.cli.compare_runs --a runs/runA --b runs/runB --metric sharpe --out runs/comparison.md
        """,
    )
    
    parser.add_argument(
        "--a",
        type=Path,
        required=True,
        help="Path to first run directory",
    )
    parser.add_argument(
        "--b",
        type=Path,
        required=True,
        help="Path to second run directory",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sharpe",
        choices=["total_return", "sharpe", "max_dd"],
        help="Metric to use for winner determination (default: sharpe)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for comparison report (default: runs/compare_<runA>_vs_<runB>.md)",
    )
    
    args = parser.parse_args()
    
    # Validate run directories
    if not args.a.exists():
        logger.error(f"Run directory not found: {args.a}")
        sys.exit(1)
    if not args.b.exists():
        logger.error(f"Run directory not found: {args.b}")
        sys.exit(1)
    
    try:
        compare_runs(
            run_a=args.a,
            run_b=args.b,
            metric=args.metric,
            output_path=args.out,
        )
    except Exception as e:
        logger.exception(f"Comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

