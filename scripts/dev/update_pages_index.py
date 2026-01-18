#!/usr/bin/env python3
"""Update GitHub Pages reports index.

This script maintains the reports/manifest.json, history.json, and regenerates
reports/index.html to list all available regression reports with trend visualizations.

Usage:
    python scripts/dev/update_pages_index.py \
        --manifest reports/manifest.json \
        --run-id "12345-abc123" \
        --timestamp "2026-01-05T12:00:00Z" \
        --status pass \
        --max-runs 50

The manifest.json structure:
{
    "runs": [
        {"id": "12345-abc123", "timestamp": "2026-01-05T12:00:00Z", "status": "pass"},
        ...
    ],
    "updated": "2026-01-05T12:00:00Z",
    "latest": "12345-abc123"
}

The history.json structure (v0.6.5):
{
    "schema_version": "1",
    "generated_utc": "<iso>",
    "window": 50,
    "runs": [...],
    "rolling": {
        "sharpe_delta": {"mean": 0.01, "stdev": 0.03, "window": 5},
        "timing_p90": {"mean": 2.5, "stdev": 0.4, "window": 5}
    }
}
"""

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


# Default prune policy: keep N most recent reports
DEFAULT_MAX_RUNS = 50

# Rolling window for stats
ROLLING_WINDOW = 5

# Flakiness detection parameters
FLAKY_WINDOW = 10
FLAKY_MIN_RUNS = 6
FLAKY_MEAN_THRESHOLD = 0.02
FLAKY_STDEV_MULTIPLIER = 2.0


def load_manifest(manifest_path: Path) -> dict:
    """Load existing manifest or create empty one."""
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"runs": [], "updated": None, "latest": None}


def load_summary_json(reports_dir: Path, run_id: str) -> dict | None:
    """Load summary.json for a run if it exists.

    Args:
        reports_dir: Path to reports directory.
        run_id: Run identifier.

    Returns:
        Summary dict or None if not found.
    """
    summary_path = reports_dir / run_id / "summary.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def add_run(
    manifest: dict,
    run_id: str,
    timestamp: str,
    status: str,
    summary: dict | None = None,
) -> dict:
    """Add a new run to the manifest with optional summary data.

    Args:
        manifest: Current manifest dict.
        run_id: Run identifier.
        timestamp: Run timestamp (ISO format).
        status: "pass" or "fail".
        summary: Optional summary.json data for enriched display.
    """
    # Remove existing entry with same ID if present
    manifest["runs"] = [r for r in manifest["runs"] if r.get("id") != run_id]

    # Build run entry with optional enriched data
    run_entry = {
        "id": run_id,
        "timestamp": timestamp,
        "status": status,
    }

    # Add enriched fields from summary.json if available
    if summary:
        if "sharpe_delta" in summary:
            run_entry["sharpe_delta"] = summary["sharpe_delta"]
        if "timing_p90" in summary:
            run_entry["timing_p90"] = summary["timing_p90"]
        if "git_sha" in summary:
            run_entry["git_sha"] = summary["git_sha"]

    # Add new run at the beginning
    manifest["runs"].insert(0, run_entry)

    # Update metadata
    manifest["updated"] = datetime.now(timezone.utc).isoformat()
    manifest["latest"] = run_id

    return manifest


def trim_runs(manifest: dict, max_runs: int) -> list[str]:
    """Keep only the most recent N runs. Returns list of pruned run IDs."""
    if len(manifest["runs"]) <= max_runs:
        return []

    pruned = [r["id"] for r in manifest["runs"][max_runs:]]
    manifest["runs"] = manifest["runs"][:max_runs]
    return pruned


def prune_report_directories(reports_dir: Path, pruned_ids: list[str]) -> int:
    """Remove directories for pruned runs. Returns count of deleted directories."""
    deleted = 0
    for run_id in pruned_ids:
        run_dir = reports_dir / run_id
        if run_dir.exists() and run_dir.is_dir():
            try:
                shutil.rmtree(run_dir)
                print(f"  Pruned: {run_id}")
                deleted += 1
            except OSError as e:
                print(f"  Warning: Failed to prune {run_id}: {e}")
    return deleted


def cache_bust_query() -> str:
    """Generate a cache-busting query parameter based on timestamp."""
    return f"?v={int(time.time())}"


# ============================================================
# History.json builder (v0.6.5)
# ============================================================


def build_history_from_summaries(reports_dir: Path, max_runs: int = 50) -> dict:
    """Build history.json by scanning all summary.json files.

    Args:
        reports_dir: Path to reports directory.
        max_runs: Maximum runs to include.

    Returns:
        History dict with runs and rolling stats.
    """
    runs = []

    # Scan all directories for summary.json
    if reports_dir.exists():
        for run_dir in reports_dir.iterdir():
            if run_dir.is_dir() and run_dir.name not in ("latest", "plots"):
                summary_path = run_dir / "summary.json"
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        runs.append({
                            "run_id": summary.get("run_id", run_dir.name),
                            "ts_utc": summary.get("generated_utc", ""),
                            "verdict": summary.get("verdict", "UNKNOWN"),
                            "sharpe_delta": summary.get("sharpe_delta", 0.0),
                            "timing_p90": summary.get("timing_p90", 0.0),
                            "git_sha": summary.get("git_sha", "unknown"),
                        })
                    except (json.JSONDecodeError, OSError):
                        continue

    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x.get("ts_utc", ""), reverse=True)

    # Limit to max_runs
    runs = runs[:max_runs]

    # Compute rolling stats
    rolling = compute_rolling_stats(runs, ROLLING_WINDOW)

    return {
        "schema_version": "1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "window": max_runs,
        "runs": runs,
        "rolling": rolling,
    }


def compute_rolling_stats(runs: list[dict], window: int) -> dict:
    """Compute rolling statistics for sharpe_delta and timing_p90.

    Args:
        runs: List of run entries (newest first).
        window: Rolling window size.

    Returns:
        Dict with rolling stats for each metric.
    """
    # Get recent values (limited to window)
    sharpe_values = [
        r["sharpe_delta"] for r in runs[:window]
        if isinstance(r.get("sharpe_delta"), (int, float))
        and not math.isnan(r["sharpe_delta"])
    ]
    timing_values = [
        r["timing_p90"] for r in runs[:window]
        if isinstance(r.get("timing_p90"), (int, float))
        and not math.isnan(r["timing_p90"])
    ]

    return {
        "sharpe_delta": _calc_stats(sharpe_values, window),
        "timing_p90": _calc_stats(timing_values, window),
    }


def _calc_stats(values: list[float], window: int) -> dict:
    """Calculate mean and stdev for a list of values."""
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "window": window}

    n = len(values)
    mean = sum(values) / n

    if n < 2:
        stdev = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        stdev = math.sqrt(variance)

    return {
        "mean": round(mean, 6),
        "stdev": round(stdev, 6),
        "window": window,
    }


def save_history_json(history: dict, output_path: Path) -> None:
    """Save history.json to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ============================================================
# Flakiness detection (v0.6.5)
# ============================================================


def detect_flaky(
    runs: list[dict],
    window: int = FLAKY_WINDOW,
    min_runs: int = FLAKY_MIN_RUNS,
    mean_threshold: float = FLAKY_MEAN_THRESHOLD,
    stdev_multiplier: float = FLAKY_STDEV_MULTIPLIER,
) -> bool:
    """Detect if recent runs are flaky.

    Rule: "flaky" if |Sharpe Δ| mean < mean_threshold and
          stdev >= stdev_multiplier × |mean| over last window runs
          (guard: need at least min_runs)

    Args:
        runs: List of run entries (newest first).
        window: Window size to analyze.
        min_runs: Minimum runs required.
        mean_threshold: Threshold for mean absolute value.
        stdev_multiplier: Multiplier for stdev vs mean comparison.

    Returns:
        True if flaky, False otherwise.
    """
    # Get sharpe_delta values from window
    values = [
        r["sharpe_delta"] for r in runs[:window]
        if isinstance(r.get("sharpe_delta"), (int, float))
        and not math.isnan(r["sharpe_delta"])
    ]

    # Need minimum runs
    if len(values) < min_runs:
        return False

    # Calculate stats
    n = len(values)
    mean = sum(values) / n
    abs_mean = abs(mean)

    if n < 2:
        return False

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    stdev = math.sqrt(variance)

    # Apply flakiness rule
    return abs_mean < mean_threshold and stdev >= stdev_multiplier * abs_mean


# ============================================================
# Sparkline SVG renderer (v0.6.5)
# ============================================================


def render_sparkline(
    points: list[float],
    width: int = 140,
    height: int = 32,
    label: str = "Sparkline",
    color_var: str = "--link-color",
) -> str:
    """Render a sparkline as inline SVG.

    Args:
        points: Data points (oldest to newest, left to right).
        width: SVG width in pixels.
        height: SVG height in pixels.
        label: ARIA label for accessibility.
        color_var: CSS variable for stroke color.

    Returns:
        SVG string.
    """
    if not points or len(points) < 2:
        return f'<svg width="{width}" height="{height}" aria-label="{label}"></svg>'

    # Pad to avoid division by zero
    padding = 4
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding

    # Calculate bounds
    min_val = min(points)
    max_val = max(points)
    value_range = max_val - min_val

    # Clamp when flat (avoid division by zero)
    if value_range < 1e-9:
        value_range = 1.0
        min_val = min_val - 0.5
        max_val = max_val + 0.5

    # Calculate point coordinates
    n = len(points)
    x_step = plot_width / (n - 1) if n > 1 else 0
    coords = []

    for i, val in enumerate(points):
        x = padding + i * x_step
        # Invert Y (SVG origin is top-left)
        y = padding + plot_height - ((val - min_val) / value_range * plot_height)
        coords.append((x, y))

    # Build path
    path_d = f"M {coords[0][0]:.1f} {coords[0][1]:.1f}"
    for x, y in coords[1:]:
        path_d += f" L {x:.1f} {y:.1f}"

    # Build title elements for each point (for tooltips)
    titles = []
    for i, val in enumerate(points):
        x, y = coords[i]
        titles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="var({color_var})" opacity="0">'
            f'<title>Point {i + 1}: {val:.4f}</title></circle>'
        )

    return f'''<svg width="{width}" height="{height}" role="img" aria-label="{label}" class="sparkline">
    <title>{label}</title>
    <path d="{path_d}" fill="none" stroke="var({color_var})" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    {"".join(titles)}
</svg>'''


# ============================================================
# Index HTML generator with trends & search (v0.6.5)
# ============================================================


def generate_index_html(
    manifest: dict,
    history: dict | None = None,
    is_flaky: bool = False,
) -> str:
    """Generate the index.html content with trends, search, and filter UI."""
    cache_bust = cache_bust_query()

    # Build sparkline data from history
    sparkline_sharpe = ""
    sparkline_timing = ""
    legend_html = ""

    if history and history.get("runs"):
        # Reverse for oldest-to-newest (left-to-right on sparkline)
        sharpe_points = [
            r["sharpe_delta"] for r in reversed(history["runs"][:20])
            if isinstance(r.get("sharpe_delta"), (int, float))
        ]
        timing_points = [
            r["timing_p90"] for r in reversed(history["runs"][:20])
            if isinstance(r.get("timing_p90"), (int, float))
        ]

        if sharpe_points:
            sparkline_sharpe = render_sparkline(
                sharpe_points, 140, 32,
                f"Sharpe Δ trend over last {len(sharpe_points)} runs",
                "--link-color"
            )
        if timing_points:
            sparkline_timing = render_sparkline(
                timing_points, 140, 32,
                f"P90 timing trend over last {len(timing_points)} runs",
                "--text-secondary"
            )

        # Rolling stats for legend
        rolling = history.get("rolling", {})
        sharpe_stats = rolling.get("sharpe_delta", {})
        timing_stats = rolling.get("timing_p90", {})

        legend_html = f'''
        <div class="trend-legend">
            <span class="legend-item">
                <span class="legend-color" style="background: var(--link-color);"></span>
                Sharpe Δ (μ={sharpe_stats.get("mean", 0):.4f}, σ={sharpe_stats.get("stdev", 0):.4f})
            </span>
            <span class="legend-item">
                <span class="legend-color" style="background: var(--text-secondary);"></span>
                P90 (μ={timing_stats.get("mean", 0):.2f}s, σ={timing_stats.get("stdev", 0):.2f}s)
            </span>
        </div>'''

    # Build runs HTML
    runs_html = ""
    if manifest["runs"]:
        runs_items = []
        for idx, run in enumerate(manifest["runs"][:50]):
            status = run.get("status", "unknown")
            status_color = "var(--pass-color)" if status == "pass" else "var(--fail-color)"
            status_badge = f'<span class="status-badge" style="color:{status_color}; font-weight:bold;">[{status.upper()}]</span>'

            # Check if this run contributes to flaky window
            flaky_pill = ""
            if is_flaky and idx == 0:
                flaky_pill = '<span class="flaky-pill">FLAKY</span>'

            # Cache-bust for latest link
            link_suffix = cache_bust if idx == 0 else ""

            # Build enriched summary stats if available
            summary_stats = ""
            if "sharpe_delta" in run:
                delta = run["sharpe_delta"]
                delta_class = "delta-positive" if delta >= 0 else "delta-negative"
                summary_stats += f' <span class="{delta_class}">Sharpe: {delta:+.4f}</span>'
            if "timing_p90" in run:
                summary_stats += f' <span class="timing">P90: {run["timing_p90"]:.2f}s</span>'
            if "git_sha" in run:
                sha_short = run["git_sha"][:7] if len(run["git_sha"]) > 7 else run["git_sha"]
                summary_stats += f' <span class="sha">{sha_short}</span>'

            # Data attributes for filtering
            run_id = run.get("id", "")
            timestamp = run.get("timestamp", "")
            verdict_attr = "flaky" if (is_flaky and idx == 0) else status

            runs_items.append(
                f'<li class="run-item" data-run-id="{run_id}" data-verdict="{verdict_attr}" data-timestamp="{timestamp}">'
                f'<a href="{run_id}/regression_report.html{link_suffix}">{run_id}</a> '
                f'{status_badge}{flaky_pill}{summary_stats} '
                f'<span class="timestamp">- {timestamp}</span></li>'
            )
        runs_html = '<ul class="run-list" id="run-list">\n' + "\n".join(runs_items) + "\n</ul>"
    else:
        runs_html = '<p class="no-reports">No reports available yet.</p>'

    latest_id = manifest.get("latest", "")
    updated = manifest.get("updated", "never")
    total_runs = len(manifest.get("runs", []))

    # Sparklines section
    sparklines_section = ""
    if sparkline_sharpe or sparkline_timing:
        sparklines_section = f'''
    <div class="trends-section">
        <h3>Trends (Last 20 Runs)</h3>
        <div class="sparklines-container">
            <div class="sparkline-box">
                <span class="sparkline-label">Sharpe Δ</span>
                {sparkline_sharpe}
            </div>
            <div class="sparkline-box">
                <span class="sparkline-label">P90 Timing</span>
                {sparkline_timing}
            </div>
        </div>
        {legend_html}
    </div>'''

    return f'''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TraderBot Regression Reports</title>

    <!-- Open Graph Meta Tags -->
    <meta property="og:title" content="TraderBot Regression Reports">
    <meta property="og:description" content="Automated regression test reports for TraderBot trading system">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://eyoair21.github.io/Trade_Bot/reports/">
    <meta property="og:site_name" content="TraderBot">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary">
    <meta name="twitter:title" content="TraderBot Regression Reports">
    <meta name="twitter:description" content="Automated regression test reports for TraderBot trading system">

    <!-- Atom Feed -->
    <link rel="alternate" type="application/atom+xml" title="TraderBot Reports Feed" href="feed.xml">

    <style>
        :root {{
            --bg-primary: #f5f5f5;
            --bg-card: white;
            --text-primary: #333;
            --text-secondary: #666;
            --border-color: #ddd;
            --link-color: #0366d6;
            --pass-color: #28a745;
            --fail-color: #dc3545;
            --flaky-color: #f0ad4e;
            --shadow: rgba(0,0,0,0.1);
            --focus-ring: #0366d6;
        }}
        [data-theme="dark"] {{
            --bg-primary: #1a1a2e;
            --bg-card: #16213e;
            --text-primary: #e4e4e7;
            --text-secondary: #a1a1aa;
            --border-color: #3f3f46;
            --link-color: #58a6ff;
            --pass-color: #3fb950;
            --fail-color: #f85149;
            --flaky-color: #d29922;
            --shadow: rgba(0,0,0,0.3);
            --focus-ring: #58a6ff;
        }}
        * {{
            transition: background-color 0.3s, color 0.3s, border-color 0.3s;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--bg-primary);
            color: var(--text-primary);
        }}
        .header-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        h1 {{
            color: var(--text-primary);
            border-bottom: 2px solid var(--link-color);
            padding-bottom: 0.5rem;
            margin: 0;
        }}
        .theme-toggle {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 6px 12px;
            cursor: pointer;
            color: var(--text-primary);
            font-size: 0.9em;
        }}
        .theme-toggle:hover {{ opacity: 0.8; }}
        .theme-toggle:focus {{
            outline: 2px solid var(--focus-ring);
            outline-offset: 2px;
        }}
        .latest-link {{
            background: var(--pass-color);
            color: white;
            padding: 1rem 2rem;
            border-radius: 6px;
            text-decoration: none;
            display: inline-block;
            margin: 1rem 0;
            font-weight: bold;
        }}
        .latest-link:hover {{ opacity: 0.9; }}
        .latest-link:focus {{
            outline: 2px solid var(--focus-ring);
            outline-offset: 2px;
        }}
        .run-list {{ list-style: none; padding: 0; }}
        .run-list li {{
            background: var(--bg-card);
            margin: 0.5rem 0;
            padding: 1rem;
            border-radius: 4px;
            border-left: 4px solid var(--link-color);
            box-shadow: 0 1px 3px var(--shadow);
        }}
        .run-list li.hidden {{ display: none; }}
        .run-list a {{
            color: var(--link-color);
            text-decoration: none;
        }}
        .run-list a:hover {{ text-decoration: underline; }}
        .run-list a:focus {{
            outline: 2px solid var(--focus-ring);
            outline-offset: 2px;
        }}
        .timestamp {{ color: var(--text-secondary); font-size: 0.9em; }}
        .no-reports {{ color: var(--text-secondary); font-style: italic; }}
        .delta-positive {{ color: var(--pass-color); font-size: 0.85em; margin-left: 8px; }}
        .delta-negative {{ color: var(--fail-color); font-size: 0.85em; margin-left: 8px; }}
        .timing {{ color: var(--text-secondary); font-size: 0.85em; margin-left: 8px; }}
        .sha {{ color: var(--text-secondary); font-size: 0.8em; font-family: monospace; margin-left: 8px; }}
        .stats {{
            background: var(--bg-card);
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
            display: inline-block;
            box-shadow: 0 1px 3px var(--shadow);
        }}
        .flaky-pill {{
            background: var(--flaky-color);
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 8px;
            text-transform: uppercase;
        }}
        /* Trends section */
        .trends-section {{
            background: var(--bg-card);
            padding: 1rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            box-shadow: 0 1px 3px var(--shadow);
        }}
        .trends-section h3 {{
            margin: 0 0 1rem 0;
            font-size: 1rem;
            color: var(--text-primary);
        }}
        .sparklines-container {{
            display: flex;
            gap: 2rem;
            flex-wrap: wrap;
        }}
        .sparkline-box {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 4px;
        }}
        .sparkline-label {{
            font-size: 0.8em;
            color: var(--text-secondary);
        }}
        .sparkline {{
            background: var(--bg-primary);
            border-radius: 4px;
        }}
        .trend-legend {{
            margin-top: 0.75rem;
            font-size: 0.8em;
            color: var(--text-secondary);
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-color {{
            width: 12px;
            height: 3px;
            border-radius: 2px;
        }}
        /* Search & Filter */
        .search-filter {{
            background: var(--bg-card);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 1px 3px var(--shadow);
        }}
        .search-filter h3 {{
            margin: 0 0 0.75rem 0;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}
        .filter-row {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
        }}
        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .filter-group label {{
            font-size: 0.8em;
            color: var(--text-secondary);
        }}
        .filter-input {{
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 0.9em;
            min-width: 150px;
        }}
        .filter-input:focus {{
            outline: 2px solid var(--focus-ring);
            outline-offset: 1px;
            border-color: var(--focus-ring);
        }}
        .filter-select {{
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 0.9em;
        }}
        .filter-select:focus {{
            outline: 2px solid var(--focus-ring);
            outline-offset: 1px;
        }}
        .filter-clear {{
            background: var(--text-secondary);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            align-self: flex-end;
        }}
        .filter-clear:hover {{ opacity: 0.8; }}
        .filter-clear:focus {{
            outline: 2px solid var(--focus-ring);
            outline-offset: 2px;
        }}
        .filter-count {{
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }}
        footer {{
            margin-top: 2rem;
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        footer a {{ color: var(--link-color); }}
    </style>
</head>
<body>
    <div class="header-row">
        <h1>TraderBot Regression Reports</h1>
        <button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle dark/light theme">Toggle Dark/Light</button>
    </div>

    <p>
        <a href="latest/regression_report.html{cache_bust}" class="latest-link">View Latest Report</a>
        <a href="feed.xml" style="margin-left: 1rem; color: var(--link-color);" aria-label="Atom feed">📡 Feed</a>
    </p>

    <div class="stats">
        <strong>Latest run:</strong> {latest_id or "none"}<br>
        <strong>Last updated:</strong> {updated}<br>
        <strong>Total reports:</strong> {total_runs}
    </div>

    {sparklines_section}

    <div class="search-filter">
        <h3>Search & Filter</h3>
        <div class="filter-row">
            <div class="filter-group">
                <label for="search-input">Search by Run ID</label>
                <input type="text" id="search-input" class="filter-input" placeholder="e.g., 12345-abc" aria-label="Search runs by ID">
            </div>
            <div class="filter-group">
                <label for="verdict-filter">Verdict</label>
                <select id="verdict-filter" class="filter-select" aria-label="Filter by verdict">
                    <option value="">All</option>
                    <option value="pass">PASS</option>
                    <option value="fail">FAIL</option>
                    <option value="flaky">FLAKY</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="date-from">From Date</label>
                <input type="date" id="date-from" class="filter-input" aria-label="Filter from date">
            </div>
            <div class="filter-group">
                <label for="date-to">To Date</label>
                <input type="date" id="date-to" class="filter-input" aria-label="Filter to date">
            </div>
            <button class="filter-clear" onclick="clearFilters()" aria-label="Clear all filters">Clear</button>
        </div>
        <div class="filter-count" id="filter-count"></div>
    </div>

    <h2>Recent Runs</h2>
    {runs_html}

    <footer>
        <p>Reports are automatically published on successful regression checks on the main branch.</p>
        <p>See <a href="https://github.com/eyoair21/Trade_bot">repository</a> for source code.</p>
    </footer>

    <script>
        // Theme toggle
        function toggleTheme() {{
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        }}

        // Load saved theme preference
        (function() {{
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {{
                document.documentElement.setAttribute('data-theme', savedTheme);
            }} else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {{
                document.documentElement.setAttribute('data-theme', 'dark');
            }}
        }})();

        // Search & Filter functionality
        const searchInput = document.getElementById('search-input');
        const verdictFilter = document.getElementById('verdict-filter');
        const dateFrom = document.getElementById('date-from');
        const dateTo = document.getElementById('date-to');
        const filterCount = document.getElementById('filter-count');
        const runList = document.getElementById('run-list');

        function applyFilters() {{
            if (!runList) return;

            const searchTerm = searchInput.value.toLowerCase();
            const verdict = verdictFilter.value.toLowerCase();
            const fromDate = dateFrom.value ? new Date(dateFrom.value) : null;
            const toDate = dateTo.value ? new Date(dateTo.value + 'T23:59:59') : null;

            const items = runList.querySelectorAll('.run-item');
            let visible = 0;

            items.forEach(item => {{
                const runId = item.dataset.runId.toLowerCase();
                const itemVerdict = item.dataset.verdict.toLowerCase();
                const itemTimestamp = item.dataset.timestamp;
                const itemDate = itemTimestamp ? new Date(itemTimestamp) : null;

                let show = true;

                // Search filter
                if (searchTerm && !runId.includes(searchTerm)) {{
                    show = false;
                }}

                // Verdict filter
                if (verdict && itemVerdict !== verdict) {{
                    show = false;
                }}

                // Date range filter
                if (fromDate && itemDate && itemDate < fromDate) {{
                    show = false;
                }}
                if (toDate && itemDate && itemDate > toDate) {{
                    show = false;
                }}

                item.classList.toggle('hidden', !show);
                if (show) visible++;
            }});

            filterCount.textContent = `Showing ${{visible}} of ${{items.length}} runs`;

            // Update URL hash for shareable state
            updateUrlHash();
        }}

        function clearFilters() {{
            searchInput.value = '';
            verdictFilter.value = '';
            dateFrom.value = '';
            dateTo.value = '';
            applyFilters();
        }}

        function updateUrlHash() {{
            const params = new URLSearchParams();
            if (searchInput.value) params.set('q', searchInput.value);
            if (verdictFilter.value) params.set('verdict', verdictFilter.value);
            if (dateFrom.value) params.set('from', dateFrom.value);
            if (dateTo.value) params.set('to', dateTo.value);

            const hash = params.toString();
            history.replaceState(null, '', hash ? '#' + hash : location.pathname);
        }}

        function loadFromUrlHash() {{
            const hash = location.hash.slice(1);
            if (!hash) return;

            const params = new URLSearchParams(hash);
            if (params.has('q')) searchInput.value = params.get('q');
            if (params.has('verdict')) verdictFilter.value = params.get('verdict');
            if (params.has('from')) dateFrom.value = params.get('from');
            if (params.has('to')) dateTo.value = params.get('to');

            applyFilters();
        }}

        // Event listeners
        searchInput.addEventListener('input', applyFilters);
        verdictFilter.addEventListener('change', applyFilters);
        dateFrom.addEventListener('change', applyFilters);
        dateTo.addEventListener('change', applyFilters);

        // Initialize
        loadFromUrlHash();
        applyFilters();
    </script>
</body>
</html>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Update GitHub Pages reports index")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--run-id", help="New run ID to add")
    parser.add_argument("--timestamp", help="Run timestamp (ISO format)")
    parser.add_argument("--status", choices=["pass", "fail"], default="pass", help="Run status")
    parser.add_argument("--max-runs", type=int, default=DEFAULT_MAX_RUNS,
                        help=f"Maximum runs to keep (default: {DEFAULT_MAX_RUNS})")
    parser.add_argument("--index-out", help="Output path for index.html (default: same dir as manifest)")
    parser.add_argument("--history-out", help="Output path for history.json (default: same dir as manifest)")
    parser.add_argument("--prune", action="store_true",
                        help="Delete directories for pruned runs")
    parser.add_argument("--reports-dir", help="Reports directory for pruning (default: manifest parent)")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing files")

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    reports_dir = Path(args.reports_dir) if args.reports_dir else manifest_path.parent

    # Load existing manifest
    manifest = load_manifest(manifest_path)

    # Add new run if provided
    if args.run_id:
        timestamp = args.timestamp or datetime.now(timezone.utc).isoformat()

        # Try to load summary.json for enriched display
        summary = load_summary_json(reports_dir, args.run_id)
        if summary:
            print(f"Loaded summary.json for {args.run_id}")

        manifest = add_run(manifest, args.run_id, timestamp, args.status, summary)
        print(f"Added run: {args.run_id}")

    # Trim to max runs
    pruned_ids = trim_runs(manifest, args.max_runs)
    if pruned_ids:
        print(f"Trimmed {len(pruned_ids)} old runs from manifest")

    # Prune directories if requested
    if args.prune and pruned_ids:
        if not args.dry_run:
            deleted = prune_report_directories(reports_dir, pruned_ids)
            print(f"Deleted {deleted} report directories")
        else:
            print(f"Would delete directories: {pruned_ids}")

    # Build history.json from summaries
    print("Building history.json...")
    history = build_history_from_summaries(reports_dir, args.max_runs)
    print(f"  Found {len(history['runs'])} runs with summary data")

    # Detect flakiness
    is_flaky = detect_flaky(history["runs"])
    if is_flaky:
        print("  WARNING: Flaky runs detected!")

    # Generate index.html with trends
    index_html = generate_index_html(manifest, history, is_flaky)

    # Determine output paths
    index_path = Path(args.index_out) if args.index_out else manifest_path.parent / "index.html"
    history_path = Path(args.history_out) if args.history_out else manifest_path.parent / "history.json"

    if args.dry_run:
        print("=== Manifest ===")
        print(json.dumps(manifest, indent=2))
        print("\n=== History (first 5 runs) ===")
        print(json.dumps({**history, "runs": history["runs"][:5]}, indent=2))
        print("\n=== Index HTML (first 80 lines) ===")
        print("\n".join(index_html.split("\n")[:80]))
        return 0

    # Write files
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated manifest: {manifest_path}")

    save_history_json(history, history_path)
    print(f"Updated history: {history_path}")

    index_path.write_text(index_html, encoding="utf-8")
    print(f"Updated index: {index_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
