#!/usr/bin/env python3
"""Generate SVG status badge for regression checks.

Usage:
    python scripts/generate_status_badge.py \\
        --status pass \\
        --output badges/regression_status.svg \\
        --sha abc1234 \\
        --timestamp 2026-01-05T12:00:00Z

The badge is deterministic, small (<10KB), and does not depend on network.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

# SVG badge template - inspired by shields.io style
# Deterministic, no external dependencies
_BADGE_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" role="img" aria-label="Regression: {status_text}">
  <title>Regression: {status_text}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{status_width}" height="20" fill="{color}"/>
    <rect width="{width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="11">
    <text aria-hidden="true" x="{label_x}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_x}" y="14">{label}</text>
    <text aria-hidden="true" x="{status_x}" y="15" fill="#010101" fill-opacity=".3">{status_text}</text>
    <text x="{status_x}" y="14">{status_text}</text>
  </g>
  <!-- sha: {sha} -->
  <!-- timestamp: {timestamp} -->
</svg>"""


def generate_badge(
    status: str,
    sha: str | None = None,
    timestamp: str | None = None,
    label: str = "regression",
) -> str:
    """Generate SVG badge content.

    Args:
        status: 'pass' or 'fail'.
        sha: Git SHA (short) for metadata.
        timestamp: ISO timestamp for metadata.
        label: Left side label text.

    Returns:
        SVG badge as string.
    """
    if timestamp is None:
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if sha is None:
        sha = "unknown"

    is_pass = status.lower() in ("pass", "passed", "success", "ok", "true", "1")

    if is_pass:
        status_text = "PASS"
        color = "#28a745"  # Green
    else:
        status_text = "FAIL"
        color = "#dc3545"  # Red

    # Calculate widths (approximate character widths for Verdana 11px)
    char_width = 7.5
    padding = 10

    label_text_width = len(label) * char_width
    label_width = int(label_text_width + padding * 2)

    status_text_width = len(status_text) * char_width
    status_width = int(status_text_width + padding * 2)

    total_width = label_width + status_width

    label_x = label_width / 2
    status_x = label_width + status_width / 2

    return _BADGE_TEMPLATE.format(
        width=total_width,
        label_width=label_width,
        status_width=status_width,
        label_x=label_x,
        status_x=status_x,
        label=label,
        status_text=status_text,
        color=color,
        sha=sha[:7] if sha else "unknown",
        timestamp=timestamp,
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SVG status badge for regression checks"
    )
    parser.add_argument(
        "--status",
        type=str,
        required=True,
        choices=["pass", "fail", "PASS", "FAIL"],
        help="Regression status (pass or fail)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="badges/regression_status.svg",
        help="Output path for SVG badge (default: badges/regression_status.svg)",
    )
    parser.add_argument(
        "--sha",
        type=str,
        default=None,
        help="Git SHA (short) to embed in badge metadata",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="ISO timestamp to embed in badge metadata (default: now)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="regression",
        help="Left side label text (default: regression)",
    )
    parser.add_argument(
        "--from-diff",
        type=str,
        default=None,
        metavar="PATH",
        help="Read status from baseline_diff.json file (overrides --status)",
    )

    args = parser.parse_args()

    # Determine status
    status = args.status
    sha = args.sha
    timestamp = args.timestamp

    # Read from diff file if specified
    if args.from_diff:
        import json

        diff_path = Path(args.from_diff)
        if not diff_path.exists():
            print(f"Error: Diff file not found: {diff_path}", file=sys.stderr)
            return 1

        with open(diff_path) as f:
            diff_data = json.load(f)

        status = "pass" if diff_data.get("passed", False) else "fail"
        sha = sha or diff_data.get("baseline_sha", "unknown")
        timestamp = timestamp or diff_data.get("generated_utc")

    # Generate badge
    badge_svg = generate_badge(
        status=status,
        sha=sha,
        timestamp=timestamp,
        label=args.label,
    )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(badge_svg, encoding="utf-8")

    print(f"Badge written to {output_path}")
    print(f"  Status: {status.upper()}")
    print(f"  SHA: {sha or 'unknown'}")
    print(f"  Size: {len(badge_svg)} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
