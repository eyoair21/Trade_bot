# run_daily_scan.ps1 - Daily news + scan pipeline for TraderBot (Windows PowerShell)
#
# Usage: .\scripts\run_daily_scan.ps1 [-Universe sp500] [-TopN 25]
#
# This script runs the complete daily pipeline:
# 1. Fetch RSS news feeds
# 2. Parse and normalize news
# 3. Score sentiment and tag events
# 4. Build sector digest
# 5. Run universe scan with all factors
# 6. Copy artifacts to public/reports/<date>/

param(
    [string]$Universe = "sp500",
    [int]$TopN = 25,
    [double]$SectorCap = 0.2
)

$ErrorActionPreference = "Stop"

# Directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DataDir = Join-Path $ProjectRoot "data"
$ReportsDir = Join-Path $ProjectRoot "reports"
$PublicDir = Join-Path $ProjectRoot "public\reports"

# Date for artifact organization
$Today = Get-Date -Format "yyyy-MM-dd"

Write-Host "=== TraderBot Daily Scan Pipeline ===" -ForegroundColor Cyan
Write-Host "Date: $Today"
Write-Host "Universe: $Universe"
Write-Host "Top-N: $TopN"
Write-Host ""

Set-Location $ProjectRoot

# Step 1: Fetch RSS news
Write-Host "[1/6] Fetching RSS news feeds..." -ForegroundColor Yellow
python -m traderbot.cli.tb news-pull `
    --sources traderbot/news/rss_sources.txt `
    --out "$DataDir\news_raw.jsonl"

# Step 2: Parse news
Write-Host "[2/6] Parsing news items..." -ForegroundColor Yellow
python -m traderbot.cli.tb news-parse `
    --in "$DataDir\news_raw.jsonl" `
    --out "$DataDir\news_parsed.jsonl"

# Step 3: Score sentiment
Write-Host "[3/6] Scoring sentiment..." -ForegroundColor Yellow
python -m traderbot.cli.tb news-score `
    --in "$DataDir\news_parsed.jsonl" `
    --out "$DataDir\news_scored.jsonl" `
    --data-dir "$DataDir"

# Step 4: Build sector digest
Write-Host "[4/6] Building sector digest..." -ForegroundColor Yellow
python -m traderbot.cli.tb sector-digest `
    --in "$DataDir\news_scored.jsonl" `
    --window 1d `
    --out "$ReportsDir\sector_sentiment.csv" `
    --png

# Step 5: Run scan
Write-Host "[5/6] Running universe scan..." -ForegroundColor Yellow
python -m traderbot.cli.tb scan `
    --universe $Universe `
    --strategy trend `
    --top-n $TopN `
    --sector-cap $SectorCap `
    --news-file "$DataDir\news_scored.jsonl" `
    --sector-file "$ReportsDir\sector_sentiment.csv" `
    --reports-dir "$ReportsDir"

# Step 6: Copy to public directory
Write-Host "[6/6] Copying artifacts to public/reports/$Today..." -ForegroundColor Yellow
$TodayDir = Join-Path $PublicDir $Today
$LatestDir = Join-Path $PublicDir "latest"

New-Item -ItemType Directory -Force -Path $TodayDir | Out-Null
New-Item -ItemType Directory -Force -Path $LatestDir | Out-Null

# Copy main artifacts
Copy-Item -Path "$ReportsDir\opportunities.csv" -Destination $TodayDir -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$ReportsDir\sector_sentiment.csv" -Destination $TodayDir -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$ReportsDir\sector_sentiment.png" -Destination $TodayDir -Force -ErrorAction SilentlyContinue
if (Test-Path "$ReportsDir\alerts") {
    Copy-Item -Path "$ReportsDir\alerts" -Destination $TodayDir -Recurse -Force -ErrorAction SilentlyContinue
}

# Update latest
Remove-Item -Path "$LatestDir\*" -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item -Path "$TodayDir\*" -Destination $LatestDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Pipeline Complete ===" -ForegroundColor Green
Write-Host "Artifacts:"
Write-Host "  - $TodayDir\opportunities.csv"
Write-Host "  - $TodayDir\sector_sentiment.csv"
Write-Host "  - $TodayDir\alerts\preview.html"
Write-Host "  - $LatestDir (updated)"
