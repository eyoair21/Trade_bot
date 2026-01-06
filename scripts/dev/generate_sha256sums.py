#!/usr/bin/env python3
"""Generate sha256sums.txt for report integrity verification.

Usage:
    python scripts/dev/generate_sha256sums.py --report-dir public/reports/latest
    python scripts/dev/generate_sha256sums.py --report-dir public/reports/12345-abc1234

The script generates a sha256sums.txt file compatible with `sha256sum -c` verification.
"""

import argparse
import hashlib
from pathlib import Path


# Files to include in integrity check (per-run reports)
INTEGRITY_FILES = [
    "regression_report.html",
    "regression_report.md",
    "baseline_diff.json",
    "provenance.json",
    "summary.json",
]

# Top-level files to include in integrity check (reports root)
TOP_LEVEL_INTEGRITY_FILES = [
    "index.html",
    "manifest.json",
    "history.json",
    "feed.xml",
    "404.html",
]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest of SHA256 hash.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def generate_sha256sums(report_dir: Path, output_file: str = "sha256sums.txt") -> Path:
    """Generate sha256sums.txt for all report files.

    Args:
        report_dir: Directory containing report files.
        output_file: Output filename (default: sha256sums.txt).

    Returns:
        Path to the generated sha256sums.txt file.
    """
    output_path = report_dir / output_file
    lines = []

    for filename in INTEGRITY_FILES:
        file_path = report_dir / filename
        if file_path.exists():
            hash_value = compute_sha256(file_path)
            # Format compatible with sha256sum -c
            lines.append(f"{hash_value}  {filename}")
            print(f"  {filename}: {hash_value[:16]}...")
        else:
            print(f"  {filename}: MISSING (skipped)")

    # Also hash any plots directory content
    plots_dir = report_dir / "plots"
    if plots_dir.exists() and plots_dir.is_dir():
        for plot_file in sorted(plots_dir.glob("*.png")):
            hash_value = compute_sha256(plot_file)
            relative_path = f"plots/{plot_file.name}"
            lines.append(f"{hash_value}  {relative_path}")
            print(f"  {relative_path}: {hash_value[:16]}...")

    # Write sha256sums.txt
    if lines:
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nGenerated: {output_path}")
        print(f"  Total files: {len(lines)}")
    else:
        print("\nNo files found to hash!")
        return None

    return output_path


def verify_sha256sums(report_dir: Path, sums_file: str = "sha256sums.txt") -> bool:
    """Verify integrity using sha256sums.txt.

    Args:
        report_dir: Directory containing report files.
        sums_file: Filename of sha256sums file.

    Returns:
        True if all files verify, False otherwise.
    """
    sums_path = report_dir / sums_file
    if not sums_path.exists():
        print(f"ERROR: {sums_path} not found")
        return False

    all_ok = True
    with open(sums_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse "hash  filename" format
            parts = line.split("  ", 1)
            if len(parts) != 2:
                print(f"WARN: Malformed line: {line}")
                continue

            expected_hash, filename = parts
            file_path = report_dir / filename

            if not file_path.exists():
                print(f"FAIL: {filename} - file not found")
                all_ok = False
                continue

            actual_hash = compute_sha256(file_path)
            if actual_hash == expected_hash:
                print(f"OK: {filename}")
            else:
                print(f"FAIL: {filename} - hash mismatch")
                print(f"  Expected: {expected_hash}")
                print(f"  Actual:   {actual_hash}")
                all_ok = False

    return all_ok


def generate_top_level_sha256sums(
    reports_root: Path,
    output_file: str = "sha256sums.txt",
) -> Path | None:
    """Generate sha256sums.txt for top-level report files.

    Args:
        reports_root: Root reports directory (e.g., public/reports).
        output_file: Output filename (default: sha256sums.txt).

    Returns:
        Path to the generated sha256sums.txt file, or None if no files.
    """
    output_path = reports_root / output_file
    lines = []

    for filename in TOP_LEVEL_INTEGRITY_FILES:
        file_path = reports_root / filename
        if file_path.exists():
            hash_value = compute_sha256(file_path)
            lines.append(f"{hash_value}  {filename}")
            print(f"  {filename}: {hash_value[:16]}...")
        else:
            print(f"  {filename}: MISSING (skipped)")

    if lines:
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nGenerated: {output_path}")
        print(f"  Total files: {len(lines)}")
        return output_path
    else:
        print("\nNo top-level files found to hash!")
        return None


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate or verify sha256sums.txt for report integrity"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        required=True,
        help="Directory containing report files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing sha256sums.txt instead of generating",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sha256sums.txt",
        help="Output filename (default: sha256sums.txt)",
    )
    parser.add_argument(
        "--top-level",
        action="store_true",
        help="Generate checksums for top-level files (index.html, history.json, feed.xml)",
    )

    args = parser.parse_args()
    report_dir = Path(args.report_dir)

    if not report_dir.exists():
        print(f"ERROR: Directory not found: {report_dir}")
        exit(1)

    if args.verify:
        print(f"Verifying integrity in {report_dir}...")
        if verify_sha256sums(report_dir, args.output):
            print("\nAll files verified OK")
            exit(0)
        else:
            print("\nVerification FAILED")
            exit(1)
    elif args.top_level:
        print(f"Generating top-level {args.output} in {report_dir}...")
        result = generate_top_level_sha256sums(report_dir, args.output)
        if result:
            exit(0)
        else:
            exit(1)
    else:
        print(f"Generating {args.output} in {report_dir}...")
        result = generate_sha256sums(report_dir, args.output)
        if result:
            exit(0)
        else:
            exit(1)


if __name__ == "__main__":
    main()
