# Follow-up Issues for v0.6.5

Copy and paste these into GitHub Issues:

---

## Issue 1: [CI] E2E: Nightly Pages + Atom feed smoke test

**Title:** `[CI] E2E: Nightly Pages + Atom feed smoke test`

**Body:**
```markdown
## Goal
Add end-to-end smoke test that validates GitHub Pages deployment and Atom feed generation after nightly sweep completes.

## Acceptance Criteria
- [ ] Smoke test runs after `deploy-pages` job completes
- [ ] Validates all Pages endpoints return 200:
  - `/` (root redirect)
  - `/reports/` (index)
  - `/reports/history.json` (history data)
  - `/reports/feed.xml` (Atom feed)
- [ ] Validates `history.json` schema (schema_version, runs, generated_utc)
- [ ] Validates `feed.xml` is well-formed Atom feed with namespace
- [ ] Fails CI if any validation fails
- [ ] Matches existing `pages-health.yml` workflow logic

## Implementation Notes
- Can extend `pages-health.yml` or add as a job in `nightly-sweep.yml`
- Should run only when `deploy-pages` succeeds
- Use `curl` or Python `requests` for HTTP checks
- Use `xml.etree.ElementTree` for XML validation

## Related
- Existing `pages-health.yml` workflow (manual + scheduled)
- `deploy-pages` job in `nightly-sweep.yml`
```

---

## Issue 2: [Alerts] Discord/Slack webhook on FAIL verdict

**Title:** `[Alerts] Discord/Slack webhook on FAIL verdict`

**Body:**
```markdown
## Goal
Send notifications to Discord or Slack when regression check fails in nightly sweep.

## Acceptance Criteria
- [ ] Webhook integration for Discord or Slack (configurable via secret)
- [ ] Sends notification when `regress` job fails in nightly sweep
- [ ] Includes:
  - Run ID
  - Git SHA
  - Verdict (FAIL)
  - Link to regression report
  - Link to Actions run
- [ ] Only triggers on main branch (not PRs)
- [ ] Uses GitHub secret for webhook URL (`DISCORD_WEBHOOK_URL` or `SLACK_WEBHOOK_URL`)
- [ ] Gracefully handles webhook failures (doesn't fail CI)

## Implementation Notes
- Use `curl` or Python `requests` to POST to webhook
- Format message as JSON (Discord) or Slack blocks
- Add step in `nightly-sweep.yml` after regression check
- Consider rate limiting if multiple failures occur

## Related
- `nightly-sweep.yml` workflow
- Regression check in `regress` job
```

---

## Issue 3: [Pages] Coverage badge on landing page

**Title:** `[Pages] Coverage badge on landing page`

**Body:**
```markdown
## Goal
Add test coverage badge to the GitHub Pages landing page (`/reports/index.html`).

## Acceptance Criteria
- [ ] Coverage badge displayed on index page
- [ ] Badge shows current coverage percentage
- [ ] Badge links to coverage report (if available)
- [ ] Badge updates automatically after each nightly sweep
- [ ] Uses shields.io-style badge or similar

## Implementation Notes
- Coverage data available from pytest `--cov-report=json` output
- Can generate badge SVG similar to `generate_status_badge.py`
- Add coverage percentage to `summary.json` or create separate `coverage.json`
- Update `update_pages_index.py` to include coverage badge in HTML
- Consider using shields.io dynamic badge API

## Related
- `generate_status_badge.py` script
- `update_pages_index.py` script
- CI coverage reports
```

---

## Issue 4: [Pages] Index pagination when runs > 50

**Title:** `[Pages] Index pagination when runs > 50`

**Body:**
```markdown
## Goal
Add pagination to the reports index page when there are more than 50 runs (current prune limit).

## Acceptance Criteria
- [ ] Index page shows pagination controls when runs > 50
- [ ] Pagination shows "Page X of Y" with prev/next buttons
- [ ] URL parameters control page (`?page=2`)
- [ ] Maintains search/filter functionality across pages
- [ ] Shows runs per page (e.g., 20, 50, 100)
- [ ] Mobile-responsive pagination UI

## Implementation Notes
- Modify `update_pages_index.py` to generate paginated HTML
- Use `--max-runs` to determine if pagination needed
- Generate multiple HTML files or use client-side pagination
- Consider server-side pagination vs client-side (current is static HTML)
- May need to change from single `index.html` to `index.html`, `index-2.html`, etc.

## Related
- `update_pages_index.py` script
- Prune policy (50 runs max)
- Index page generation
```

---

## Issue 5: [CI] Integrate HTML minification into Pages workflow (optional)

**Title:** `[CI] Integrate HTML minification into Pages workflow (optional)`

**Body:**
```markdown
## Goal
Ensure HTML minification is properly integrated and tested in the Pages deployment workflow.

## Acceptance Criteria
- [ ] HTML minification runs before Pages deployment
- [ ] Minification reduces file size (verify with before/after stats)
- [ ] Minified HTML still validates (W3C validator)
- [ ] Minification doesn't break functionality (test with browser)
- [ ] Minification step is visible in Actions logs

## Implementation Notes
- Already implemented in `deploy-pages` job (`minify_html.py`)
- Verify it's working correctly
- Add size comparison to job summary
- Consider minifying CSS/JS if present
- May want to add minification verification test

## Related
- `scripts/dev/minify_html.py`
- `deploy-pages` job in `nightly-sweep.yml`
```



