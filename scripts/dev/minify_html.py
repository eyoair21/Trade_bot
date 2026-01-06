#!/usr/bin/env python3
"""Minify HTML files by removing unnecessary whitespace and comments.

Usage:
    python scripts/dev/minify_html.py --input public/reports/latest/regression_report.html
    python scripts/dev/minify_html.py --input-dir public/reports/latest --pattern "*.html"

Zero external dependencies - uses only stdlib (re, html.parser).
"""

import argparse
import re
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path


class HTMLMinifier(HTMLParser):
    """HTML minifier using stdlib HTMLParser.

    Removes:
    - HTML comments (except conditional comments)
    - Excess whitespace between tags
    - Leading/trailing whitespace in text nodes

    Preserves:
    - Content in <pre>, <script>, <style>, <textarea> tags
    - Single spaces between inline elements
    """

    # Tags where whitespace is significant
    PRESERVE_WHITESPACE_TAGS = frozenset(["pre", "script", "style", "textarea", "code"])

    # Inline elements where spaces between them matter
    INLINE_TAGS = frozenset([
        "a", "abbr", "acronym", "b", "bdo", "big", "br", "button", "cite", "code",
        "dfn", "em", "i", "img", "input", "kbd", "label", "map", "object", "q",
        "samp", "script", "select", "small", "span", "strong", "sub", "sup",
        "textarea", "tt", "var",
    ])

    def __init__(self):
        super().__init__()
        self.output = StringIO()
        self.preserve_stack: list[str] = []
        self._last_was_space = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle opening tags."""
        tag_lower = tag.lower()

        # Build attribute string
        attr_str = ""
        for name, value in attrs:
            if value is None:
                attr_str += f" {name}"
            else:
                # Use double quotes, escape as needed
                escaped_value = value.replace('"', "&quot;")
                attr_str += f' {name}="{escaped_value}"'

        self.output.write(f"<{tag}{attr_str}>")
        self._last_was_space = False

        # Track tags where we preserve whitespace
        if tag_lower in self.PRESERVE_WHITESPACE_TAGS:
            self.preserve_stack.append(tag_lower)

    def handle_endtag(self, tag: str) -> None:
        """Handle closing tags."""
        tag_lower = tag.lower()

        # Pop from preserve stack if applicable
        if self.preserve_stack and self.preserve_stack[-1] == tag_lower:
            self.preserve_stack.pop()

        self.output.write(f"</{tag}>")
        self._last_was_space = False

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle self-closing tags like <br/>, <img/>."""
        attr_str = ""
        for name, value in attrs:
            if value is None:
                attr_str += f" {name}"
            else:
                escaped_value = value.replace('"', "&quot;")
                attr_str += f' {name}="{escaped_value}"'

        self.output.write(f"<{tag}{attr_str}/>")
        self._last_was_space = False

    def handle_data(self, data: str) -> None:
        """Handle text content."""
        if self.preserve_stack:
            # Inside <pre>, <script>, etc. - preserve exactly
            self.output.write(data)
            self._last_was_space = False
        else:
            # Collapse whitespace
            # Replace all whitespace sequences with single space
            collapsed = re.sub(r"\s+", " ", data)

            # Strip if only whitespace
            if collapsed == " ":
                if not self._last_was_space:
                    self.output.write(" ")
                    self._last_was_space = True
            elif collapsed:
                # Write with potential leading/trailing space
                self.output.write(collapsed)
                self._last_was_space = collapsed.endswith(" ")

    def handle_comment(self, data: str) -> None:
        """Handle HTML comments - remove unless conditional."""
        # Preserve conditional comments (IE hacks)
        if data.startswith("[if") or data.startswith("[endif"):
            self.output.write(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        """Handle declarations like DOCTYPE."""
        self.output.write(f"<!{decl}>")
        self._last_was_space = False

    def get_minified(self) -> str:
        """Return the minified HTML."""
        return self.output.getvalue().strip()


def minify_html(html_content: str) -> str:
    """Minify HTML content.

    Args:
        html_content: Raw HTML string.

    Returns:
        Minified HTML string.
    """
    minifier = HTMLMinifier()
    minifier.feed(html_content)
    return minifier.get_minified()


def minify_file(input_path: Path, output_path: Path | None = None) -> tuple[int, int]:
    """Minify an HTML file.

    Args:
        input_path: Path to input HTML file.
        output_path: Path to output file. If None, overwrites input.

    Returns:
        Tuple of (original_size, minified_size) in bytes.
    """
    if output_path is None:
        output_path = input_path

    original_content = input_path.read_text(encoding="utf-8")
    original_size = len(original_content.encode("utf-8"))

    minified_content = minify_html(original_content)
    minified_size = len(minified_content.encode("utf-8"))

    output_path.write_text(minified_content, encoding="utf-8")

    return original_size, minified_size


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Minify HTML files by removing whitespace and comments"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Single HTML file to minify (overwrites in place)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing HTML files to minify",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.html",
        help="Glob pattern for files when using --input-dir (default: *.html)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file (only with --input). If not specified, overwrites input.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )

    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir is required")

    total_original = 0
    total_minified = 0
    files_processed = 0

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}")
            exit(1)

        output_path = Path(args.output) if args.output else None

        if args.dry_run:
            original_content = input_path.read_text(encoding="utf-8")
            original_size = len(original_content.encode("utf-8"))
            minified_content = minify_html(original_content)
            minified_size = len(minified_content.encode("utf-8"))
            print(f"Would minify: {input_path}")
            print(f"  Original: {original_size:,} bytes")
            print(f"  Minified: {minified_size:,} bytes")
            print(f"  Savings: {original_size - minified_size:,} bytes ({100 * (1 - minified_size / original_size):.1f}%)")
        else:
            original_size, minified_size = minify_file(input_path, output_path)
            target = output_path or input_path
            print(f"Minified: {input_path} -> {target}")
            print(f"  {original_size:,} -> {minified_size:,} bytes ({100 * (1 - minified_size / original_size):.1f}% smaller)")

        total_original = original_size
        total_minified = minified_size
        files_processed = 1

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"ERROR: Directory not found: {input_dir}")
            exit(1)

        html_files = list(input_dir.glob(args.pattern))
        if not html_files:
            print(f"No files matching '{args.pattern}' in {input_dir}")
            exit(0)

        for html_file in sorted(html_files):
            if args.dry_run:
                original_content = html_file.read_text(encoding="utf-8")
                original_size = len(original_content.encode("utf-8"))
                minified_content = minify_html(original_content)
                minified_size = len(minified_content.encode("utf-8"))
                print(f"Would minify: {html_file.name} ({original_size:,} -> {minified_size:,} bytes)")
            else:
                original_size, minified_size = minify_file(html_file)
                savings_pct = 100 * (1 - minified_size / original_size) if original_size > 0 else 0
                print(f"  {html_file.name}: {original_size:,} -> {minified_size:,} bytes ({savings_pct:.1f}%)")

            total_original += original_size
            total_minified += minified_size
            files_processed += 1

    # Summary
    if files_processed > 0:
        print()
        total_savings = total_original - total_minified
        savings_pct = 100 * (1 - total_minified / total_original) if total_original > 0 else 0
        if args.dry_run:
            print(f"Would process {files_processed} file(s)")
            print(f"Total savings: {total_savings:,} bytes ({savings_pct:.1f}%)")
        else:
            print(f"Processed {files_processed} file(s)")
            print(f"Total: {total_original:,} -> {total_minified:,} bytes ({savings_pct:.1f}% smaller)")


if __name__ == "__main__":
    main()
