#!/usr/bin/env python3
"""Pack sweep artifacts into a zip file with size guards.

Includes metadata, leaderboard, timings, and top-N run folders.
Enforces size cap by dropping runs if needed.
"""

import argparse
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# Denylist patterns to prevent secrets leakage
DENY_PATTERNS = {".env", ".*", "node_modules", "__pycache__", ".coverage", "*.key", "*.pem"}


def should_exclude(file_path: Path) -> bool:
    """Check if file should be excluded based on denylist.

    Args:
        file_path: Path to check.

    Returns:
        True if file should be excluded.
    """
    # Check exact matches
    if file_path.name in DENY_PATTERNS:
        return True

    # Check suffix patterns
    for pattern in DENY_PATTERNS:
        if pattern.startswith("*.") and file_path.suffix == pattern[1:]:
            return True
        if pattern.startswith(".") and file_path.name.startswith("."):
            return True

    return False


def get_dir_size_mb(path: Path) -> float:
    """Calculate directory size in MB.

    Args:
        path: Directory path.

    Returns:
        Size in megabytes.
    """
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024 * 1024)


def pack_sweep(
    sweep_root: Path,
    output_path: Path,
    max_size_mb: float = 80.0,
    top_n: int = 3,
) -> None:
    """Pack sweep directory into zip file.

    Args:
        sweep_root: Path to sweep output directory.
        output_path: Output zip file path.
        max_size_mb: Maximum zip size in MB.
        top_n: Number of top runs to include.
    """
    if not sweep_root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_root}")

    # Essential files to always include
    essential_files = [
        "sweep_meta.json",
        "all_results.json",
        "leaderboard.csv",
        "leaderboard.md",
        "timings.csv",
    ]

    # Load leaderboard to determine top runs
    leaderboard_path = sweep_root / "leaderboard.csv"
    top_run_dirs = []

    if leaderboard_path.exists():
        import csv
        with open(leaderboard_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= top_n:
                    break
                run_idx = row.get("run_idx", "")
                if run_idx:
                    top_run_dirs.append(f"run_{int(run_idx):03d}")

    # If no leaderboard, try to find run directories
    if not top_run_dirs:
        run_dirs = sorted([d for d in sweep_root.iterdir() if d.is_dir() and d.name.startswith("run_")])
        top_run_dirs = [d.name for d in run_dirs[:top_n]]

    print(f"Packing sweep: {sweep_root.name}")
    print(f"  Essential files: {len(essential_files)}")
    print(f"  Top run directories: {len(top_run_dirs)}")

    # Create zip file
    with ZipFile(output_path, "w", ZIP_DEFLATED) as zipf:
        # Add essential files
        for filename in essential_files:
            file_path = sweep_root / filename
            if file_path.exists():
                arcname = f"{sweep_root.name}/{filename}"
                zipf.write(file_path, arcname)
                print(f"  + Added {filename}")

        # Add top run directories
        for run_dir_name in top_run_dirs:
            run_dir = sweep_root / run_dir_name
            if not run_dir.exists():
                continue

            for file_path in run_dir.rglob("*"):
                if file_path.is_file():
                    # Skip large data files
                    if file_path.suffix in [".parquet", ".h5", ".hdf5"]:
                        continue

                    # Skip secrets and unwanted files
                    if should_exclude(file_path):
                        continue

                    arcname = f"{sweep_root.name}/{run_dir_name}/{file_path.relative_to(run_dir)}"
                    zipf.write(file_path, arcname)

            print(f"  + Added {run_dir_name}/")

    # Check size
    zip_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nZip file created: {output_path}")
    print(f"Size: {zip_size_mb:.2f} MB")

    # If over limit, create minimal version
    if zip_size_mb > max_size_mb:
        print(f"WARNING: Size exceeds limit ({max_size_mb} MB)")
        print("Creating minimal version with best run only...")

        # Create minimal zip with only essential files + best run
        minimal_path = output_path.with_stem(output_path.stem + "_minimal")

        with ZipFile(minimal_path, "w", ZIP_DEFLATED) as zipf:
            # Add essential files
            for filename in essential_files:
                file_path = sweep_root / filename
                if file_path.exists():
                    arcname = f"{sweep_root.name}/{filename}"
                    zipf.write(file_path, arcname)

            # Add only best run
            if top_run_dirs:
                best_run_dir = sweep_root / top_run_dirs[0]
                if best_run_dir.exists():
                    for file_path in best_run_dir.rglob("*"):
                        if file_path.is_file() and file_path.suffix not in [".parquet", ".h5"]:
                            if should_exclude(file_path):
                                continue
                            arcname = f"{sweep_root.name}/{top_run_dirs[0]}/{file_path.relative_to(best_run_dir)}"
                            zipf.write(file_path, arcname)

        minimal_size_mb = minimal_path.stat().st_size / (1024 * 1024)
        print(f"Minimal zip created: {minimal_path}")
        print(f"Size: {minimal_size_mb:.2f} MB")

        # Replace original if minimal is still too large
        if minimal_size_mb > max_size_mb:
            print("WARNING: Even minimal version exceeds limit!")
        else:
            output_path.unlink()
            minimal_path.rename(output_path)
            print("OK: Using minimal version")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pack sweep artifacts into zip file with size guards"
    )
    parser.add_argument(
        "sweep_root",
        type=Path,
        help="Path to sweep output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output zip file path (default: <sweep_root>.zip)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=80.0,
        help="Maximum zip size in MB (default: 80.0)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top runs to include (default: 3)",
    )

    args = parser.parse_args()

    if not args.sweep_root.exists():
        print(f"Error: Sweep directory not found: {args.sweep_root}")
        return 1

    output_path = args.output or args.sweep_root.with_suffix(".zip")

    try:
        pack_sweep(
            sweep_root=args.sweep_root,
            output_path=output_path,
            max_size_mb=args.max_size_mb,
            top_n=args.top_n,
        )
        return 0
    except Exception as e:
        print(f"Error packing sweep: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

