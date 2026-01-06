#!/usr/bin/env python3
"""Update GitHub Pages reports index.

This script maintains the reports/manifest.json and regenerates reports/index.html
to list all available regression reports.

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
"""

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


# Default prune policy: keep N most recent reports
DEFAULT_MAX_RUNS = 50


def load_manifest(manifest_path: Path) -> dict:
    """Load existing manifest or create empty one."""
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"runs": [], "updated": None, "latest": None}


def add_run(manifest: dict, run_id: str, timestamp: str, status: str) -> dict:
    """Add a new run to the manifest."""
    # Remove existing entry with same ID if present
    manifest["runs"] = [r for r in manifest["runs"] if r.get("id") != run_id]

    # Add new run at the beginning
    manifest["runs"].insert(0, {
        "id": run_id,
        "timestamp": timestamp,
        "status": status,
    })

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


def generate_index_html(manifest: dict) -> str:
    """Generate the index.html content from manifest with OG tags and dark/light toggle."""
    cache_bust = cache_bust_query()

    runs_html = ""
    if manifest["runs"]:
        runs_items = []
        for run in manifest["runs"][:20]:
            status_color = "var(--pass-color)" if run.get("status") == "pass" else "var(--fail-color)"
            status_badge = f'<span style="color:{status_color}; font-weight:bold;">[{run.get("status", "unknown").upper()}]</span>'
            # Cache-bust for latest link
            link_suffix = cache_bust if run == manifest["runs"][0] else ""
            runs_items.append(
                f'<li><a href="{run["id"]}/regression_report.html{link_suffix}">{run["id"]}</a> '
                f'{status_badge} <span class="timestamp">- {run.get("timestamp", "unknown")}</span></li>'
            )
        runs_html = '<ul class="run-list">\n' + "\n".join(runs_items) + "\n</ul>"
    else:
        runs_html = '<p class="no-reports">No reports available yet.</p>'

    latest_id = manifest.get("latest", "")
    updated = manifest.get("updated", "never")
    total_runs = len(manifest.get("runs", []))

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
            --shadow: rgba(0,0,0,0.1);
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
            --shadow: rgba(0,0,0,0.3);
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
        .run-list {{ list-style: none; padding: 0; }}
        .run-list li {{
            background: var(--bg-card);
            margin: 0.5rem 0;
            padding: 1rem;
            border-radius: 4px;
            border-left: 4px solid var(--link-color);
            box-shadow: 0 1px 3px var(--shadow);
        }}
        .run-list a {{ color: var(--link-color); text-decoration: none; }}
        .run-list a:hover {{ text-decoration: underline; }}
        .timestamp {{ color: var(--text-secondary); font-size: 0.9em; }}
        .no-reports {{ color: var(--text-secondary); font-style: italic; }}
        .stats {{
            background: var(--bg-card);
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
            display: inline-block;
            box-shadow: 0 1px 3px var(--shadow);
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
        <button class="theme-toggle" onclick="toggleTheme()">Toggle Dark/Light</button>
    </div>

    <p>
        <a href="latest/regression_report.html{cache_bust}" class="latest-link">View Latest Report</a>
    </p>

    <div class="stats">
        <strong>Latest run:</strong> {latest_id or "none"}<br>
        <strong>Last updated:</strong> {updated}<br>
        <strong>Total reports:</strong> {total_runs}
    </div>

    <h2>Recent Runs</h2>
    {runs_html}

    <footer>
        <p>Reports are automatically published on successful regression checks on the main branch.</p>
        <p>See <a href="https://github.com/eyoair21/Trade_Bot">repository</a> for source code.</p>
    </footer>

    <script>
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
    parser.add_argument("--prune", action="store_true",
                        help="Delete directories for pruned runs")
    parser.add_argument("--reports-dir", help="Reports directory for pruning (default: manifest parent)")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing files")

    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    # Load existing manifest
    manifest = load_manifest(manifest_path)

    # Add new run if provided
    if args.run_id:
        timestamp = args.timestamp or datetime.now(timezone.utc).isoformat()
        manifest = add_run(manifest, args.run_id, timestamp, args.status)
        print(f"Added run: {args.run_id}")

    # Trim to max runs
    pruned_ids = trim_runs(manifest, args.max_runs)
    if pruned_ids:
        print(f"Trimmed {len(pruned_ids)} old runs from manifest")

    # Prune directories if requested
    if args.prune and pruned_ids:
        reports_dir = Path(args.reports_dir) if args.reports_dir else manifest_path.parent
        if not args.dry_run:
            deleted = prune_report_directories(reports_dir, pruned_ids)
            print(f"Deleted {deleted} report directories")
        else:
            print(f"Would delete directories: {pruned_ids}")

    # Generate index.html
    index_html = generate_index_html(manifest)

    # Determine output paths
    index_path = Path(args.index_out) if args.index_out else manifest_path.parent / "index.html"

    if args.dry_run:
        print("=== Manifest ===")
        print(json.dumps(manifest, indent=2))
        print("\n=== Index HTML (first 50 lines) ===")
        print("\n".join(index_html.split("\n")[:50]))
        return 0

    # Write files
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated manifest: {manifest_path}")

    index_path.write_text(index_html, encoding="utf-8")
    print(f"Updated index: {index_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
