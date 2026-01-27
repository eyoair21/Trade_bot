"""Lightweight reporting for backtest runs.

Generates markdown reports and static plots from run artifacts.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore
    MATPLOTLIB_AVAILABLE = False


def generate_report(run_dir: Path, output_dir: Path | None = None) -> None:
    """Generate comprehensive report from run directory.
    
    Args:
        run_dir: Directory containing run artifacts (results.json, equity_curve.csv, etc.)
        output_dir: Output directory for report (default: run_dir/report/)
    """
    if output_dir is None:
        output_dir = run_dir / "report"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load artifacts
    results = _load_json(run_dir / "results.json")
    equity_curve = _load_csv(run_dir / "equity_curve.csv")
    orders = _load_csv(run_dir / "orders.csv")
    alloc = _load_csv(run_dir / "alloc.csv")
    breaches = _load_csv(run_dir / "breaches.csv")
    
    # Generate markdown report
    report_md = _generate_markdown_report(
        results=results,
        equity_curve=equity_curve,
        orders=orders,
        alloc=alloc,
        breaches=breaches,
    )
    
    # Save report
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    
    # Generate plots
    if MATPLOTLIB_AVAILABLE:
        _generate_plots(
            equity_curve=equity_curve,
            orders=orders,
            alloc=alloc,
            output_dir=output_dir,
        )
    
    print(f"Report generated: {report_path}")


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON file or return empty dict."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_csv(path: Path) -> pd.DataFrame:
    """Load CSV file or return empty DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else None)


def _generate_markdown_report(
    results: dict[str, Any],
    equity_curve: pd.DataFrame,
    orders: pd.DataFrame,
    alloc: pd.DataFrame,
    breaches: pd.DataFrame,
) -> str:
    """Generate markdown report content."""
    lines = [
        "# TraderBot Run Report",
        "",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Metrics",
        "",
    ]
    
    # Core metrics
    if results:
        lines.extend([
            f"- **Total Reward**: {results.get('total_reward', 0):.2f}",
            f"- **Net PnL**: ${results.get('pnl_net', 0):.2f}",
            f"- **Sharpe Ratio**: {results.get('sharpe', 0):.3f}",
            f"- **Sortino Ratio**: {results.get('sortino', 0):.3f}",
            f"- **Max Drawdown**: {results.get('max_dd', 0):.2%}",
            f"- **Turnover**: ${results.get('turnover', 0):.0f}",
            f"- **Risk Breaches**: {results.get('breaches_count', 0)}",
            "",
        ])
    
    # Reward attribution
    if results and "dd_penalty" in results:
        lines.extend([
            "## Reward Attribution",
            "",
            f"- **Gross PnL**: ${results.get('pnl_net', 0):.2f}",
            f"- **Drawdown Penalty**: ${results.get('dd_penalty', 0):.2f}",
            f"- **Turnover Penalty**: ${results.get('turnover_penalty', 0):.2f}",
            f"- **Breach Penalty**: ${results.get('breach_penalty', 0):.2f}",
            f"- **Net Reward**: {results.get('total_reward', 0):.2f}",
            "",
        ])
    
    # Trading stats
    if len(orders) > 0:
        n_trades = len(orders)
        wins = orders[orders["pnl_net"] > 0] if "pnl_net" in orders.columns else pd.DataFrame()
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        
        lines.extend([
            "## Trading Statistics",
            "",
            f"- **Total Trades**: {n_trades}",
            f"- **Win Rate**: {win_rate:.1%}",
            f"- **Avg Trade PnL**: ${orders['pnl_net'].mean():.2f}" if "pnl_net" in orders.columns else "",
            "",
        ])
    
    # Breaches
    if len(breaches) > 0:
        breach_counts = breaches["breach_type"].value_counts()
        lines.extend([
            "## Risk Breaches",
            "",
        ])
        for breach_type, count in breach_counts.items():
            lines.append(f"- **{breach_type}**: {count}")
        lines.append("")
    
    # Bandit allocation
    if len(alloc) > 0:
        lines.extend([
            "## Strategy Allocation (Bandit)",
            "",
            "Final weights:",
            "",
        ])
        if "strategy" in alloc.columns and "weight" in alloc.columns:
            final_alloc = alloc.tail(1)
            for _, row in final_alloc.iterrows():
                lines.append(f"- **{row['strategy']}**: {row['weight']:.2%}")
        lines.append("")
    
    # Plots section
    if MATPLOTLIB_AVAILABLE:
        lines.extend([
            "## Visualizations",
            "",
            "![Equity Curve](equity_curve.png)",
            "",
            "![Drawdown](drawdown.png)",
            "",
        ])
        if len(alloc) > 0:
            lines.append("![Allocation](allocation.png)")
            lines.append("")
    
    return "\n".join(lines)


def _generate_plots(
    equity_curve: pd.DataFrame,
    orders: pd.DataFrame,
    alloc: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate static plots."""
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return
    
    # Equity curve
    if len(equity_curve) > 0 and "equity" in equity_curve.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(equity_curve.index, equity_curve["equity"], label="Equity")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity ($)")
        ax.set_title("Equity Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "equity_curve.png", dpi=100)
        plt.close(fig)
    
    # Drawdown
    if len(equity_curve) > 0 and "drawdown" in equity_curve.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(
            equity_curve.index,
            equity_curve["drawdown"],
            0,
            alpha=0.3,
            color="red",
            label="Drawdown",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Drawdown Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "drawdown.png", dpi=100)
        plt.close(fig)
    
    # Strategy allocation (if bandit used)
    if len(alloc) > 0 and "strategy" in alloc.columns and "weight" in alloc.columns:
        # Pivot to get weights over time
        alloc_pivot = alloc.pivot_table(
            index="date" if "date" in alloc.columns else alloc.index,
            columns="strategy",
            values="weight",
            fill_value=0,
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        alloc_pivot.plot(kind="area", stacked=True, ax=ax, alpha=0.7)
        ax.set_xlabel("Time")
        ax.set_ylabel("Weight")
        ax.set_title("Strategy Allocation Over Time")
        ax.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "allocation.png", dpi=100)
        plt.close(fig)

