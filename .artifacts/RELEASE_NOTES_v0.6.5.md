# TraderBot v0.6.5 — "Insights & Trends"

**Release Date:** 2026-01-05

## Highlights

This release introduces comprehensive reporting infrastructure for GitHub Pages deployment, including rolling history tracking, Atom feed generation, and SHA256 integrity verification.

## What's New

### Reporting Infrastructure
- **history.json Builder** – Aggregates summary.json from the last N runs with rolling statistics (mean, stdev) for Sharpe delta and timing metrics
- **Atom Feed Generator** – RFC4287-compliant feed.xml for RSS readers, enabling subscription to regression results
- **SHA256 Integrity** – Generates and verifies SHA256SUMS for all report artifacts
- **HTML Minification** – Optional minify_html.py script for production deployments

### Flakiness Detection
- New `detect_flaky()` function identifies unstable metrics using statistical thresholds:
  - `|mean| < 0.02 AND stdev >= 2×|mean|` over last 10 runs
- SVG sparkline rendering with ARIA accessibility attributes

### Developer Experience
- `init_gh_pages.sh` helper script for bootstrapping GitHub Pages
- Comprehensive test coverage for all new reporting modules
- Timezone-aware timestamps using `datetime.UTC` (Python 3.11+)

## Breaking Changes

None.

## Migration Notes

No migration required. New reporting features are additive.

## Statistics

- **564 tests passing**
- **All CI checks green**

## Contributors

- Development and release engineering by the TraderBot team

---

**Full Changelog:** https://github.com/eyoair21/Trade_bot/compare/v0.6.4...v0.6.5
