"""Report builder for walk-forward analysis.

Generates unified markdown report from walk-forward results.
"""

import json
from pathlib import Path
from typing import Any

from traderbot.logging_setup import get_logger

logger = get_logger("reports.report_builder")


def build_report(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Build markdown report from walk-forward results.

    Args:
        results: Walk-forward results dictionary.
        output_path: Path to write report.md.
    """
    lines: list[str] = []

    # Header
    lines.append("# Walk-Forward Analysis Report")
    lines.append("")

    # Run Manifest section
    if "manifest" in results:
        manifest = results["manifest"]
        lines.append("## Run Manifest")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Run ID | `{manifest.get('run_id', 'N/A')}` |")
        lines.append(f"| Git SHA | `{manifest.get('git_sha', 'unknown')}` |")
        lines.append(f"| Seed | `{manifest.get('seed', 42)}` |")
        lines.append(f"| Start Date | {manifest.get('start_date', 'N/A')} |")
        lines.append(f"| End Date | {manifest.get('end_date', 'N/A')} |")
        lines.append(f"| Universe | {', '.join(manifest.get('universe', []))} |")
        lines.append(f"| Splits | {manifest.get('n_splits', 0)} |")
        lines.append(f"| IS Ratio | {manifest.get('is_ratio', 0.0)} |")
        lines.append(f"| Sizer | {manifest.get('sizer', 'fixed')} |")

        # Sizer params
        sizer_params = manifest.get('sizer_params', {})
        if sizer_params:
            params_str = ", ".join(f"{k}={v}" for k, v in sizer_params.items())
            lines.append(f"| Sizer Params | {params_str} |")

        lines.append(f"| Data Digest | `{manifest.get('data_digest', 'N/A')}` |")
        lines.append("")
    else:
        # Fallback to old format
        lines.append(f"**Period:** {results['start_date']} to {results['end_date']}")
        lines.append(f"**Universe:** {', '.join(results['universe'])}")
        lines.append(f"**Universe Mode:** {results['universe_mode']}")
        lines.append(f"**Splits:** {results['n_splits']} (IS ratio: {results['is_ratio']})")
        lines.append(f"**Position Sizer:** {results.get('sizer', 'fixed')}")
        lines.append(f"**Train Per Split:** {results.get('train_per_split', False)}")
        if "seed" in results:
            lines.append(f"**Seed:** {results['seed']}")
        lines.append("")

    # Summary metrics
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Avg OOS Return | {results['avg_oos_return_pct']:.2f}% |")
    lines.append(f"| Avg OOS Sharpe | {results['avg_oos_sharpe']:.3f} |")
    lines.append(f"| Avg OOS Max DD | {results['avg_oos_max_dd_pct']:.2f}% |")
    lines.append(f"| Total OOS Trades | {results['total_oos_trades']} |")
    lines.append("")

    # Execution costs
    if "execution_costs" in results:
        costs = results["execution_costs"]
        lines.append("## Execution Costs")
        lines.append("")
        lines.append("| Cost Type | Amount |")
        lines.append("|-----------|--------|")
        lines.append(f"| Commission | ${costs['commission']:.2f} |")
        lines.append(f"| Per-Share Fees | ${costs['fees']:.2f} |")
        lines.append(f"| Slippage | ${costs['slippage']:.2f} |")
        lines.append(f"| **Total** | **${results['total_execution_costs']:.2f}** |")
        lines.append("")

    # Split details table
    lines.append("## Split Details")
    lines.append("")
    lines.append("| Split | IS Period | OOS Period | IS Return | OOS Return | IS Sharpe | OOS Sharpe | OOS Trades |")
    lines.append("|-------|-----------|------------|-----------|------------|-----------|------------|------------|")

    for split in results["splits"]:
        lines.append(
            f"| {split['split']} "
            f"| {split['is_start']} to {split['is_end']} "
            f"| {split['oos_start']} to {split['oos_end']} "
            f"| {split['is_return_pct']:.2f}% "
            f"| {split['oos_return_pct']:.2f}% "
            f"| {split['is_sharpe']:.3f} "
            f"| {split['oos_sharpe']:.3f} "
            f"| {split['oos_trades']} |"
        )

    lines.append("")

    # Calibration section
    if "calibration" in results:
        calib = results["calibration"]
        lines.append("## Calibration Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Avg Brier Score | {calib['avg_brier_score']:.4f} |")
        lines.append(f"| Avg ECE | {calib['avg_ece']:.4f} |")
        lines.append("")

        # Per-split calibration
        lines.append("### Per-Split Calibration")
        lines.append("")
        lines.append("| Split | Brier Score | ECE | Optimal Threshold |")
        lines.append("|-------|-------------|-----|-------------------|")

        for split_calib in calib.get("splits", []):
            lines.append(
                f"| {split_calib['split']} "
                f"| {split_calib['brier_score']:.4f} "
                f"| {split_calib['ece']:.4f} "
                f"| {split_calib['optimal_threshold']:.3f} |"
            )

        lines.append("")

    # Threshold details
    lines.append("## Probability Thresholds")
    lines.append("")
    lines.append("| Split | Threshold |")
    lines.append("|-------|-----------|")

    for split in results["splits"]:
        threshold = split.get("proba_threshold", 0.5)
        lines.append(f"| {split['split']} | {threshold:.3f} |")

    lines.append("")

    # Write report
    output_path.write_text("\n".join(lines))
    logger.info(f"Report written to {output_path}")


def build_report_from_json(
    results_path: Path,
    output_path: Path | None = None,
) -> None:
    """Build report from results.json file.

    Args:
        results_path: Path to results.json file.
        output_path: Path for report.md. Defaults to same directory as results.json.
    """
    with open(results_path) as f:
        results = json.load(f)

    if output_path is None:
        output_path = results_path.parent / "report.md"

    build_report(results, output_path)


def generate_equity_chart_data(
    equity_csv_path: Path,
) -> dict[str, Any]:
    """Generate data for equity curve chart.

    Args:
        equity_csv_path: Path to equity_curve.csv.

    Returns:
        Dict with chart data.
    """
    import pandas as pd

    df = pd.read_csv(equity_csv_path)

    if df.empty:
        return {"dates": [], "equity": [], "splits": []}

    return {
        "dates": df["date"].tolist(),
        "equity": df["equity"].tolist(),
        "splits": df["split"].tolist() if "split" in df.columns else [],
    }
