# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.5] - 2026-01-05

### Added
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

### Changed
- `update_pages_index.py` now builds history.json from summary.json files across all report directories.
- `generate_index_html()` accepts history and flakiness data for trend visualization.
- Index page includes sparkline charts and FLAKY pill indicator when detected.
- Nightly sweep workflow generates history.json and feed.xml before Pages deployment.
- `generate_sha256sums.py` supports both per-run and top-level integrity file sets.

### Tests Added
- `tests/reports/test_history.py` - history.json builder and rolling stats tests
- `tests/reports/test_sparkline.py` - SVG sparkline rendering tests
- `tests/reports/test_flaky.py` - flakiness detection algorithm tests
- `tests/reports/test_feed.py` - Atom feed generation and validation tests
- `tests/cli/test_reports_summarize.py` - reports summarize CLI tests
- `tests/reports/test_integrity_list.py` - integrity file list verification tests

## [0.6.4] - 2026-01-05

### Added
- **PASS/FAIL status banner** in HTML regression reports with prominent verdict display.
- **summary.json** generation for per-run metadata (schema v1: run_id, verdict, sharpe_delta, trades_delta, timing_p90, git_sha, generated_utc).
- **HTML minification** via `scripts/dev/minify_html.py` (zero external dependencies, HTMLParser-based).
- **Enriched index listing** showing Sharpe delta, timing P90, and git SHA from summary.json.
- `scripts/dev/minify_html.py` for minifying HTML files before GitHub Pages deployment.

### Changed
- `sha256sums.txt` now includes `summary.json` for integrity verification.
- Index rows display enriched summary stats with color-coded delta indicators.
- HTML reports minified in deploy-pages job to reduce artifact size.

## [0.6.3] - 2026-01-05

### Added
- **Dark/Light theme toggle** in HTML regression reports with sticky header and localStorage persistence.
- **SHA256 integrity hashes** (`sha256sums.txt`) for report artifacts with generation and verification support.
- **Provenance schema v1** with `schema_version` and `git_sha` fields in `provenance.json`.
- **Provenance footer** in HTML reports showing "Built from <sha> at <timestamp>".
- `scripts/dev/generate_sha256sums.py` for generating and verifying integrity hashes.
- **Open Graph and Twitter Card meta tags** in GitHub Pages index for better social sharing.
- **404.html** with dark/light theme support for pruned/missing reports.
- **robots.txt** for search engine guidance on gh-pages.
- **Prune policy** keeping 50 most recent reports on GitHub Pages (configurable via `--max-runs`).
- "Integrity & Provenance" documentation section in README.

### Changed
- HTML reports now use CSS custom properties for theming with system preference detection.
- `update_pages_index.py` now supports `--prune` flag to delete old report directories.
- `trim_runs()` now returns list of pruned IDs for directory cleanup.
- Cache-busting query parameters added to "latest" links to prevent stale content.
- CI workflow now generates `sha256sums.txt` for PR artifacts.
- Nightly sweep preserves report history by copying existing directories from gh-pages.

## [0.6.2] - 2026-01-05

### Added
- **GitHub Pages publishing** for nightly regression reports under `/reports/<run-id>/`.
- `deploy-pages` job in `nightly-sweep.yml` for automatic report deployment on PASS.
- `/reports/latest/` directory that mirrors the most recent successful run.
- `scripts/dev/init_gh_pages.sh` for initializing the `gh-pages` branch.
- `scripts/dev/update_pages_index.py` for maintaining report index and manifest.
- `scripts/dev/verify_pages_pack.sh` smoke test for Pages artifact packing.
- "Reports (GitHub Pages)" documentation section in README.
- "Latest Report" badge link in README header.

### Changed
- PR job summary now notes "artifacts only" with link to where report will appear after merge.
- Nightly sweep outputs run ID for Pages deployment coordination.

## [0.6.1] - 2026-01-05

### Added
- **Commit-on-pass badge publishing**: Workflows auto-commit `badges/regression_status.svg` on main branch when regression passes.
- `badges/.gitkeep` for tracking the badges directory.
- `scripts/dev/verify_badge_commit.sh` smoke test script.
- "CI & Badges" documentation section in README.

### Changed
- Workflows skip badge commits from `github-actions[bot]` to prevent infinite loops.
- Badge commits use `[ci skip]` in message for additional loop protection.
- README now displays live regression badge from repository.

## [0.6.0] - 2026-01-05

### Added
- **Per-metric budget map** (`budgets:` YAML block) for granular multi-metric thresholds.
- `MetricBudget` dataclass with `mode`, `max_drop`, `min`, `max`, `epsilon`, `required` fields.
- `MetricVerdict` dataclass for per-metric pass/fail results.
- Per-metric verdicts table in Markdown and HTML regression reports.
- **Status badge generation** (`scripts/generate_status_badge.py`) with shields.io-style SVG.
- Badge generation step in `ci.yml` and `nightly-sweep.yml` workflows.
- GitHub job summary with pass/fail badge for PR checks.
- Badge artifact upload (30-day retention for nightly, 7-day for PR).

### Changed
- `compare_results()` now evaluates multiple metrics independently when `budgets:` is present.
- Final verdict fails if ANY required metric breaches its threshold.
- HTML report includes styled per-metric verdicts table with color-coded status.

### Backward Compatible
- Legacy single-metric format (`metric`, `mode`, `max_sharpe_drop`) still supported.
- Falls back to single-metric evaluation when `budgets:` is not present.

## [0.5.3] - 2026-01-05

### Added
- `--no-emoji` flag (CLI) and `TRADERBOT_NO_EMOJI=1` env var for cross-platform compatibility.
- "Data Provenance" block in regression reports when fallbacks are used.
- Shared console module: `traderbot/cli/_console.py`.
- Edge-case tests (empty/malformed inputs, NaN/Inf metrics, round-trip).
- `.gitattributes` for consistent line endings across platforms.
- CLI entrypoint integration test.

### Fixed
- Windows console Unicode errors (cp1252) by safe fallback to ASCII.
- Subprocess CLI tests now set PYTHONPATH for module resolution.

### Changed
- Extracted console helpers from `regress.py` to shared `_console.py` module.
- Normalized line endings via `.gitattributes`.

## [0.5.2] - 2026-01-04

### Added
- Performance budgets and regression detection (`traderbot.cli.regress`).
- Baseline management with `compare` and `update-baseline` commands.
- Determinism verification for sweep reruns.
- CI integration for regression checks.

## [0.5.1] - 2026-01-03

### Added
- CI/CD workflows (lint, test, nightly sweep).
- Artifact packing with `scripts/pack_sweep.py`.
- Timing profiling for sweeps (`--time` flag).
- Leaderboard generation with timing summary.

## [0.5.0] - 2026-01-02

### Added
- YAML-driven hyperparameter sweeps (`traderbot.cli.sweep`).
- Parallel execution with configurable workers.
- Leaderboard generation and best run export.

## [0.4.0] - 2026-01-01

### Added
- Reproducibility with `--seed` parameter.
- Run manifest with full configuration capture.
- Run comparison tool (`traderbot.cli.compare_runs`).

## [0.3.0] - 2025-12-30

### Added
- Position sizing policies (fixed, volatility-targeting, Kelly).
- Probability thresholding with optional optimization.
- Execution cost tracking (commission, slippage).
- Calibration metrics (Brier score, ECE).

## [0.2.0] - 2025-12-28

### Added
- PatchTST model for probability-of-uptrend prediction.
- TorchScript export for CPU-optimized inference.
- Dynamic universe selection based on liquidity/volatility.
- News tone scores with FinBERT integration.

## [0.1.0] - 2025-12-25

### Added
- Initial release with walk-forward analysis CLI.
- Momentum strategy implementation.
- Basic backtesting engine with risk management.
- Data adapters and point-in-time store.
- Comprehensive test suite with 70% coverage requirement.
