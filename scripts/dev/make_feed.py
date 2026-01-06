#!/usr/bin/env python3
"""Generate Atom feed from history.json for GitHub Pages.

Pure-stdlib implementation - no external dependencies.
Generates RFC4287-compliant Atom 1.0 feed with RFC3339 timestamps.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

# Feed configuration
FEED_TITLE = "TraderBot Regression Reports"
FEED_SUBTITLE = "Nightly regression test results and performance trends"
FEED_AUTHOR = "TraderBot CI"
MAX_ENTRIES = 20


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


def format_rfc3339(dt: datetime) -> str:
    """Format datetime as RFC3339 string.

    Args:
        dt: Datetime object (should be UTC)

    Returns:
        RFC3339 formatted string
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp to datetime.

    Args:
        ts: ISO format timestamp string

    Returns:
        UTC datetime object
    """
    # Handle various ISO formats
    ts = ts.replace("Z", "+00:00")
    if "+" not in ts and "-" not in ts[10:]:
        ts = ts + "+00:00"

    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        # Fallback: try parsing without timezone
        try:
            dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)


def build_entry_id(run: dict[str, Any], base_url: str) -> str:
    """Build unique entry ID (tag URI preferred, fallback to URL).

    Args:
        run: Run data from history
        base_url: Base URL for reports

    Returns:
        Unique entry ID
    """
    # Prefer git_sha for uniqueness, fallback to run_id
    unique_part = run.get("git_sha") or run.get("run_id", "unknown")
    return f"{base_url.rstrip('/')}/reports/{run.get('run_id', 'unknown')}/"


def build_entry_summary(run: dict[str, Any]) -> str:
    """Build human-readable summary for feed entry.

    Args:
        run: Run data from history

    Returns:
        Summary text
    """
    parts = []

    verdict = run.get("verdict", "UNKNOWN")
    parts.append(f"Result: {verdict}")

    sharpe_delta = run.get("sharpe_delta")
    if sharpe_delta is not None:
        sign = "+" if sharpe_delta >= 0 else ""
        parts.append(f"Sharpe Delta: {sign}{sharpe_delta:.4f}")

    timing_p90 = run.get("timing_p90")
    if timing_p90 is not None:
        parts.append(f"Timing P90: {timing_p90:.2f}s")

    trades_delta = run.get("trades_delta")
    if trades_delta is not None:
        sign = "+" if trades_delta >= 0 else ""
        parts.append(f"Trades Delta: {sign}{trades_delta}")

    git_sha = run.get("git_sha")
    if git_sha:
        parts.append(f"Commit: {git_sha[:7]}")

    return " | ".join(parts)


def build_entry_title(run: dict[str, Any]) -> str:
    """Build entry title from run data.

    Args:
        run: Run data from history

    Returns:
        Entry title
    """
    verdict = run.get("verdict", "UNKNOWN")
    run_id = run.get("run_id", "unknown")
    emoji = "\u2705" if verdict == "PASS" else "\u274c"  # checkmark or X
    return f"{emoji} {verdict}: {run_id}"


def generate_atom_feed(
    history: dict[str, Any],
    base_url: str,
    max_entries: int = MAX_ENTRIES,
) -> bytes:
    """Generate Atom 1.0 feed from history data.

    Args:
        history: Parsed history.json data
        base_url: Base URL for the GitHub Pages site
        max_entries: Maximum number of entries to include

    Returns:
        UTF-8 encoded Atom XML
    """
    runs = history.get("runs", [])

    # Sort by generated_utc descending (most recent first)
    runs_sorted = sorted(
        runs,
        key=lambda r: r.get("generated_utc", ""),
        reverse=True,
    )[:max_entries]

    # Create feed root
    feed = Element("feed")
    feed.set("xmlns", "http://www.w3.org/2005/Atom")

    # Feed metadata
    title = SubElement(feed, "title")
    title.text = FEED_TITLE

    subtitle = SubElement(feed, "subtitle")
    subtitle.text = FEED_SUBTITLE

    # Feed links
    link_self = SubElement(feed, "link")
    link_self.set("href", f"{base_url.rstrip('/')}/reports/feed.xml")
    link_self.set("rel", "self")
    link_self.set("type", "application/atom+xml")

    link_alt = SubElement(feed, "link")
    link_alt.set("href", f"{base_url.rstrip('/')}/reports/")
    link_alt.set("rel", "alternate")
    link_alt.set("type", "text/html")

    # Feed ID (permanent, unique)
    feed_id = SubElement(feed, "id")
    feed_id.text = f"{base_url.rstrip('/')}/reports/"

    # Feed updated time
    updated = SubElement(feed, "updated")
    if runs_sorted:
        latest_ts = runs_sorted[0].get("generated_utc", "")
        if latest_ts:
            updated.text = format_rfc3339(parse_timestamp(latest_ts))
        else:
            updated.text = format_rfc3339(datetime.now(timezone.utc))
    else:
        updated.text = format_rfc3339(datetime.now(timezone.utc))

    # Author
    author = SubElement(feed, "author")
    author_name = SubElement(author, "name")
    author_name.text = FEED_AUTHOR

    # Generator
    generator = SubElement(feed, "generator")
    generator.set("uri", "https://github.com/your-org/traderbot")
    generator.set("version", "0.6.5")
    generator.text = "TraderBot make_feed.py"

    # Add entries
    for run in runs_sorted:
        entry = SubElement(feed, "entry")

        # Entry title
        entry_title = SubElement(entry, "title")
        entry_title.text = build_entry_title(run)

        # Entry link
        run_id = run.get("run_id", "unknown")
        entry_link = SubElement(entry, "link")
        entry_link.set("href", f"{base_url.rstrip('/')}/reports/{run_id}/")
        entry_link.set("rel", "alternate")
        entry_link.set("type", "text/html")

        # Entry ID (permanent, unique)
        entry_id = SubElement(entry, "id")
        entry_id.text = build_entry_id(run, base_url)

        # Entry updated/published
        generated_utc = run.get("generated_utc", "")
        if generated_utc:
            ts = format_rfc3339(parse_timestamp(generated_utc))
        else:
            ts = format_rfc3339(datetime.now(timezone.utc))

        entry_updated = SubElement(entry, "updated")
        entry_updated.text = ts

        entry_published = SubElement(entry, "published")
        entry_published.text = ts

        # Entry summary
        entry_summary = SubElement(entry, "summary")
        entry_summary.set("type", "text")
        entry_summary.text = build_entry_summary(run)

        # Entry content (HTML)
        entry_content = SubElement(entry, "content")
        entry_content.set("type", "html")
        content_html = build_entry_content_html(run, base_url)
        entry_content.text = content_html

        # Categories for filtering
        verdict = run.get("verdict", "UNKNOWN")
        category = SubElement(entry, "category")
        category.set("term", verdict.lower())
        category.set("label", verdict)

    # Serialize to XML
    xml_declaration = b'<?xml version="1.0" encoding="UTF-8"?>\n'
    xml_body = tostring(feed, encoding="unicode")

    return xml_declaration + xml_body.encode("utf-8")


def build_entry_content_html(run: dict[str, Any], base_url: str) -> str:
    """Build HTML content for feed entry.

    Args:
        run: Run data from history
        base_url: Base URL for reports

    Returns:
        HTML content string (escaped for XML)
    """
    run_id = run.get("run_id", "unknown")
    verdict = run.get("verdict", "UNKNOWN")
    sharpe_delta = run.get("sharpe_delta")
    timing_p90 = run.get("timing_p90")
    trades_delta = run.get("trades_delta")
    git_sha = run.get("git_sha", "")

    lines = [
        f"<h3>{html.escape(verdict)}: {html.escape(run_id)}</h3>",
        "<dl>",
    ]

    if sharpe_delta is not None:
        sign = "+" if sharpe_delta >= 0 else ""
        color = "green" if sharpe_delta >= 0 else "red"
        lines.append(f'<dt>Sharpe Delta</dt><dd style="color:{color}">{sign}{sharpe_delta:.4f}</dd>')

    if timing_p90 is not None:
        lines.append(f"<dt>Timing P90</dt><dd>{timing_p90:.2f}s</dd>")

    if trades_delta is not None:
        sign = "+" if trades_delta >= 0 else ""
        lines.append(f"<dt>Trades Delta</dt><dd>{sign}{trades_delta}</dd>")

    if git_sha:
        lines.append(f"<dt>Commit</dt><dd><code>{html.escape(git_sha[:7])}</code></dd>")

    lines.append("</dl>")
    lines.append(
        f'<p><a href="{html.escape(base_url.rstrip("/"))}/reports/{html.escape(run_id)}/">'
        f"View full report</a></p>"
    )

    return "".join(lines)


def write_feed(feed_xml: bytes, output_path: Path) -> None:
    """Write feed XML to file.

    Args:
        feed_xml: UTF-8 encoded Atom XML
        output_path: Path to write feed.xml
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(feed_xml)


def validate_feed(feed_xml: bytes) -> bool:
    """Validate feed XML is well-formed.

    Args:
        feed_xml: UTF-8 encoded Atom XML

    Returns:
        True if valid, raises exception otherwise
    """
    from xml.etree.ElementTree import fromstring

    # This will raise ParseError if malformed
    fromstring(feed_xml)
    return True


def main() -> int:
    """CLI entrypoint for feed generation.

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description="Generate Atom feed from history.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python make_feed.py --history reports/history.json --out reports/feed.xml
  python make_feed.py --history history.json --base-url https://user.github.io/repo
        """,
    )
    parser.add_argument(
        "--history",
        type=Path,
        required=True,
        help="Path to history.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("feed.xml"),
        help="Output path for feed.xml (default: feed.xml)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://your-org.github.io/traderbot",
        help="Base URL for GitHub Pages site",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=MAX_ENTRIES,
        help=f"Maximum entries in feed (default: {MAX_ENTRIES})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate feed XML after generation",
    )

    args = parser.parse_args()

    # Load history
    if not args.history.exists():
        print(f"Error: history.json not found at {args.history}", file=sys.stderr)
        return 1

    try:
        history = load_history(args.history)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.history}: {e}", file=sys.stderr)
        return 1

    # Generate feed
    feed_xml = generate_atom_feed(
        history=history,
        base_url=args.base_url,
        max_entries=args.max_entries,
    )

    # Validate if requested
    if args.validate:
        try:
            validate_feed(feed_xml)
            print("Feed validation: OK")
        except Exception as e:
            print(f"Feed validation failed: {e}", file=sys.stderr)
            return 1

    # Write feed
    write_feed(feed_xml, args.out)
    print(f"Generated {args.out} with {min(len(history.get('runs', [])), args.max_entries)} entries")

    return 0


if __name__ == "__main__":
    sys.exit(main())
