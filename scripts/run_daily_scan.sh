#!/usr/bin/env bash
# run_daily_scan.sh - Daily news + scan pipeline for TraderBot
#
# Usage: ./scripts/run_daily_scan.sh [--universe sp500] [--top-n 25]
#
# This script runs the complete daily pipeline:
# 1. Fetch RSS news feeds
# 2. Parse and normalize news
# 3. Score sentiment and tag events
# 4. Build sector digest
# 5. Run universe scan with all factors
# 6. Copy artifacts to public/reports/<date>/

set -euo pipefail

# Configuration
UNIVERSE="${1:-sp500}"
TOP_N="${2:-25}"
SECTOR_CAP="${3:-0.2}"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
REPORTS_DIR="$PROJECT_ROOT/reports"
PUBLIC_DIR="$PROJECT_ROOT/public/reports"

# Date for artifact organization
TODAY=$(date +%Y-%m-%d)

echo "=== TraderBot Daily Scan Pipeline ==="
echo "Date: $TODAY"
echo "Universe: $UNIVERSE"
echo "Top-N: $TOP_N"
echo ""

cd "$PROJECT_ROOT"

# Step 1: Fetch RSS news
echo "[1/6] Fetching RSS news feeds..."
python -m traderbot.cli.tb news-pull \
    --sources traderbot/news/rss_sources.txt \
    --out "$DATA_DIR/news_raw.jsonl"

# Step 2: Parse news
echo "[2/6] Parsing news items..."
python -m traderbot.cli.tb news-parse \
    --in "$DATA_DIR/news_raw.jsonl" \
    --out "$DATA_DIR/news_parsed.jsonl"

# Step 3: Score sentiment
echo "[3/6] Scoring sentiment..."
python -m traderbot.cli.tb news-score \
    --in "$DATA_DIR/news_parsed.jsonl" \
    --out "$DATA_DIR/news_scored.jsonl" \
    --data-dir "$DATA_DIR"

# Step 4: Build sector digest
echo "[4/6] Building sector digest..."
python -m traderbot.cli.tb sector-digest \
    --in "$DATA_DIR/news_scored.jsonl" \
    --window 1d \
    --out "$REPORTS_DIR/sector_sentiment.csv" \
    --png

# Step 5: Run scan
echo "[5/6] Running universe scan..."
python -m traderbot.cli.tb scan \
    --universe "$UNIVERSE" \
    --strategy trend \
    --top-n "$TOP_N" \
    --sector-cap "$SECTOR_CAP" \
    --news-file "$DATA_DIR/news_scored.jsonl" \
    --sector-file "$REPORTS_DIR/sector_sentiment.csv" \
    --reports-dir "$REPORTS_DIR"

# Step 6: Copy to public directory
echo "[6/6] Copying artifacts to public/reports/$TODAY..."
mkdir -p "$PUBLIC_DIR/$TODAY"
mkdir -p "$PUBLIC_DIR/latest"

# Copy main artifacts
cp -f "$REPORTS_DIR/opportunities.csv" "$PUBLIC_DIR/$TODAY/" 2>/dev/null || true
cp -f "$REPORTS_DIR/sector_sentiment.csv" "$PUBLIC_DIR/$TODAY/" 2>/dev/null || true
cp -f "$REPORTS_DIR/sector_sentiment.png" "$PUBLIC_DIR/$TODAY/" 2>/dev/null || true
cp -rf "$REPORTS_DIR/alerts" "$PUBLIC_DIR/$TODAY/" 2>/dev/null || true

# Update latest symlink/copy
rm -rf "$PUBLIC_DIR/latest/"*
cp -rf "$PUBLIC_DIR/$TODAY/"* "$PUBLIC_DIR/latest/" 2>/dev/null || true

echo ""
echo "=== Pipeline Complete ==="
echo "Artifacts:"
echo "  - $PUBLIC_DIR/$TODAY/opportunities.csv"
echo "  - $PUBLIC_DIR/$TODAY/sector_sentiment.csv"
echo "  - $PUBLIC_DIR/$TODAY/alerts/preview.html"
echo "  - $PUBLIC_DIR/latest/ (updated)"
