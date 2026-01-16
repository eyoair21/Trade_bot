"""Sector sentiment digest aggregation.

Aggregates news sentiment by sector and generates
summary CSV and visualization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from traderbot.logging_setup import get_logger

logger = get_logger("news.sector_digest")


@dataclass
class SectorSentiment:
    """Aggregated sentiment for a sector."""

    sector: str
    item_count: int
    mean_sentiment: float
    p95_abs_sentiment: float
    most_recent_minutes: float
    positive_count: int
    negative_count: int
    neutral_count: int


def _parse_window(window: str) -> timedelta:
    """Parse window string like '1d', '12h', '7d' to timedelta."""
    window = window.strip().lower()

    if window.endswith("d"):
        days = int(window[:-1])
        return timedelta(days=days)
    elif window.endswith("h"):
        hours = int(window[:-1])
        return timedelta(hours=hours)
    elif window.endswith("m"):
        minutes = int(window[:-1])
        return timedelta(minutes=minutes)
    else:
        # Default to days
        return timedelta(days=int(window))


def build_sector_digest(
    input_path: Path,
    output_csv: Path,
    output_png: Optional[Path] = None,
    window: str = "1d",
) -> Dict[str, SectorSentiment]:
    """Build sector sentiment digest from scored news.

    Args:
        input_path: Path to scored news JSONL.
        output_csv: Path to write sector CSV.
        output_png: Optional path to write sector chart.
        window: Time window for filtering (e.g., '1d', '12h').

    Returns:
        Dictionary of sector to SectorSentiment.
    """
    if not input_path.exists():
        logger.warning("Input file not found: %s", input_path)
        return {}

    # Parse time window
    window_delta = _parse_window(window)
    cutoff_time = datetime.now(timezone.utc) - window_delta

    # Load and filter items
    items: List[Dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Parse published time
            published_str = item.get("published_utc", "")
            try:
                published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                if published_dt.tzinfo is None:
                    published_dt = published_dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            # Filter by window
            if published_dt >= cutoff_time:
                items.append(item)

    if not items:
        logger.warning("No items found within window %s", window)
        return {}

    # Aggregate by sector
    sector_data: Dict[str, List[Dict]] = {}
    for item in items:
        sector = item.get("sector", "UNKNOWN")
        if sector not in sector_data:
            sector_data[sector] = []
        sector_data[sector].append(item)

    # Compute aggregates
    results: Dict[str, SectorSentiment] = {}
    for sector, sector_items in sector_data.items():
        sentiments = [item.get("decayed_sentiment", 0.0) for item in sector_items]
        minutes_agos = [item.get("minutes_ago", 0.0) for item in sector_items]

        # Compute statistics
        df = pd.Series(sentiments)
        mean_sent = float(df.mean()) if not df.empty else 0.0
        p95_abs = float(df.abs().quantile(0.95)) if len(df) > 0 else 0.0
        most_recent = float(min(minutes_agos)) if minutes_agos else 0.0

        # Count sentiment categories
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count

        results[sector] = SectorSentiment(
            sector=sector,
            item_count=len(sector_items),
            mean_sentiment=round(mean_sent, 4),
            p95_abs_sentiment=round(p95_abs, 4),
            most_recent_minutes=round(most_recent, 1),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
        )

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sector, data in sorted(results.items(), key=lambda x: x[1].mean_sentiment, reverse=True):
        rows.append({
            "sector": data.sector,
            "item_count": data.item_count,
            "mean_sentiment": data.mean_sentiment,
            "p95_abs_sentiment": data.p95_abs_sentiment,
            "most_recent_minutes": data.most_recent_minutes,
            "positive_count": data.positive_count,
            "negative_count": data.negative_count,
            "neutral_count": data.neutral_count,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote sector digest to %s (%d sectors)", output_csv, len(results))

    # Generate chart if matplotlib is available
    if output_png:
        _generate_sector_chart(df, output_png)

    return results


def _generate_sector_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Generate sector sentiment bar chart."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        if df.empty:
            return

        # Sort by mean sentiment
        df_sorted = df.sort_values("mean_sentiment", ascending=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(df_sorted) * 0.4)))

        # Color bars by sentiment
        colors = ["#e74c3c" if x < 0 else "#27ae60" for x in df_sorted["mean_sentiment"]]

        # Horizontal bar chart
        bars = ax.barh(df_sorted["sector"], df_sorted["mean_sentiment"], color=colors)

        # Add value labels
        for bar, val in zip(bars, df_sorted["mean_sentiment"]):
            width = bar.get_width()
            x_pos = width + 0.01 if width >= 0 else width - 0.01
            ha = "left" if width >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                    va="center", ha=ha, fontsize=9)

        ax.set_xlabel("Mean Sentiment Score")
        ax.set_title("Sector Sentiment Digest")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlim(-1.1, 1.1)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.info("Generated sector chart: %s", output_path)

    except ImportError:
        logger.debug("matplotlib not available, skipping chart generation")
    except Exception as e:
        logger.warning("Failed to generate sector chart: %s", e)


def get_sector_zscore(
    symbol: str,
    sector_map: Dict[str, str],
    sector_digest: Dict[str, SectorSentiment],
) -> float:
    """Get sector z-score for a symbol.

    Returns the mean sentiment of the symbol's sector,
    normalized across all sectors.
    """
    sector = sector_map.get(symbol, "UNKNOWN")

    if sector not in sector_digest:
        return 0.0

    # Compute z-score across sectors
    sentiments = [s.mean_sentiment for s in sector_digest.values()]
    if not sentiments:
        return 0.0

    mean_all = sum(sentiments) / len(sentiments)
    std_all = (sum((s - mean_all) ** 2 for s in sentiments) / len(sentiments)) ** 0.5

    if std_all < 0.001:
        return 0.0

    sector_sent = sector_digest[sector].mean_sentiment
    return (sector_sent - mean_all) / std_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build sector sentiment digest")
    parser.add_argument("--input", required=True, dest="input_path", help="Scored news JSONL")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--png", default=None, help="Output PNG path for chart")
    parser.add_argument("--window", default="1d", help="Time window (e.g., 1d, 12h)")

    args = parser.parse_args()
    png_path = Path(args.png) if args.png else None
    build_sector_digest(Path(args.input_path), Path(args.out), png_path, args.window)
