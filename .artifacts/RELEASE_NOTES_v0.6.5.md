# v0.6.5 — "Insights & Trends"

## Added

- **Run history aggregation** (`history.json`) built from `summary.json` files with rolling statistics (5-run window for mean/stdev).
- **Inline SVG sparklines** for Sharpe Delta and P90 timing trends in index page with ARIA accessibility labels.
- **Flakiness detector** with configurable thresholds: flags "FLAKY" when |Sharpe Δ| mean < 0.02 and stdev >= 2×|mean| over last 10 runs (guard: min 6 runs).
- **Atom feed generation** (`scripts/dev/make_feed.py`) for RSS subscription to regression results (RFC4287-compliant, 20 most recent entries).
- **Search & filter UI** in index page with client-side filtering by run ID, date, verdict; URL hash state persistence for shareability.
- **Reports summarize CLI** (`traderbot.cli.reports`) with `--since`, `--limit`, `--format` (text/json/csv) options for historical analysis.
- `scripts/dev/make_feed.py` for pure-stdlib Atom feed generation from `history.json`.
- `traderbot/cli/reports.py` with `summarize` command for CLI-based trend analysis.
- Top-level integrity files list (`index.html`, `history.json`, `feed.xml`, `404.html`) in `generate_sha256sums.py`.
- `--top-level` flag in `generate_sha256sums.py` for hashing index-level files.
- `--history-out` argument in `update_pages_index.py` for history.json generation.
- Feed and history URLs in nightly workflow job summary.

## Changed

- `update_pages_index.py` now builds history.json from summary.json files across all report directories.
- `generate_index_html()` accepts history and flakiness data for trend visualization.
- Index page includes sparkline charts and FLAKY pill indicator when detected.
- Nightly sweep workflow generates history.json and feed.xml before Pages deployment.
- `generate_sha256sums.py` supports both per-run and top-level integrity file sets.

## Tests Added

- `tests/reports/test_history.py` - history.json builder and rolling stats tests
- `tests/reports/test_sparkline.py` - SVG sparkline rendering tests
- `tests/reports/test_flaky.py` - flakiness detection algorithm tests
- `tests/reports/test_feed.py` - Atom feed generation and validation tests
- `tests/cli/test_reports_summarize.py` - reports summarize CLI tests
- `tests/reports/test_integrity_list.py` - integrity file list verification tests
