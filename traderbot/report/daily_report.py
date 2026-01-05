"""Daily report generation.

Stub implementation for generating daily trading reports.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class DailyReport:
    """Daily trading report.

    Stub implementation - to be extended with:
    - Performance summary
    - Position details
    - Risk metrics
    - Trade log
    - Equity curve chart
    """

    report_date: date
    equity: float = 0.0
    cash: float = 0.0
    daily_pnl: float = 0.0
    daily_return_pct: float = 0.0
    positions: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    risk_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "report_date": self.report_date.isoformat(),
            "equity": self.equity,
            "cash": self.cash,
            "daily_pnl": self.daily_pnl,
            "daily_return_pct": self.daily_return_pct,
            "positions": self.positions,
            "trades": self.trades,
            "risk_metrics": self.risk_metrics,
            "metadata": self.metadata,
        }

    def to_html(self) -> str:
        """Generate HTML report.

        Stub implementation.
        """
        return f"""
        <html>
        <head><title>Daily Report - {self.report_date}</title></head>
        <body>
            <h1>Daily Trading Report</h1>
            <h2>Date: {self.report_date}</h2>
            <table>
                <tr><td>Equity:</td><td>${self.equity:,.2f}</td></tr>
                <tr><td>Cash:</td><td>${self.cash:,.2f}</td></tr>
                <tr><td>Daily P&L:</td><td>${self.daily_pnl:,.2f}</td></tr>
                <tr><td>Daily Return:</td><td>{self.daily_return_pct:.2f}%</td></tr>
            </table>
            <h3>Positions</h3>
            <p>{len(self.positions)} positions</p>
            <h3>Trades</h3>
            <p>{len(self.trades)} trades today</p>
        </body>
        </html>
        """

    def to_markdown(self) -> str:
        """Generate Markdown report.

        Stub implementation.
        """
        return f"""
# Daily Trading Report

**Date:** {self.report_date}

## Summary

| Metric | Value |
|--------|-------|
| Equity | ${self.equity:,.2f} |
| Cash | ${self.cash:,.2f} |
| Daily P&L | ${self.daily_pnl:,.2f} |
| Daily Return | {self.daily_return_pct:.2f}% |

## Positions

{len(self.positions)} active positions

## Trades

{len(self.trades)} trades executed today
        """


def generate_daily_report(
    report_date: date,
    equity: float,
    cash: float,
    prev_equity: float,
    positions: list[dict[str, Any]] | None = None,
    trades: list[dict[str, Any]] | None = None,
) -> DailyReport:
    """Generate a daily report.

    Stub implementation.

    Args:
        report_date: Date for the report.
        equity: End of day equity.
        cash: End of day cash.
        prev_equity: Previous day equity.
        positions: List of position dicts.
        trades: List of trade dicts.

    Returns:
        DailyReport instance.
    """
    daily_pnl = equity - prev_equity
    daily_return_pct = (daily_pnl / prev_equity * 100) if prev_equity > 0 else 0.0

    return DailyReport(
        report_date=report_date,
        equity=equity,
        cash=cash,
        daily_pnl=daily_pnl,
        daily_return_pct=daily_return_pct,
        positions=positions or [],
        trades=trades or [],
    )
