# ci(pages): deploy via Actions; normalize URLs; add health check

## Summary

Configures GitHub Pages deployment via Actions (not branch mode), normalizes URLs to `Trade_bot` casing, adds a nightly health check, and ensures minimal `public/index.html` exists before deployment.

## Changes

### Workflow Updates

- **`nightly-sweep.yml`**:
  - Permissions: `contents: read`, `pages: write`, `id-token: write` (with `contents: write` scoped to badge commit step only)
  - Added guard step: ensures `public/index.html` exists before Pages deployment
  - Uses `actions/configure-pages@v5`, `upload-pages-artifact@v3`, `deploy-pages@v4`
  - Badge commit step has explicit `contents: write` permission
  - Fixed 404.html links to use relative paths

- **`pages-health.yml`** (new/updated):
  - Nightly health check at 06:00 UTC
  - Manual dispatch trigger
  - Fails if `/` or `/reports/` is not 200
  - Optional endpoints (`history.json`, `feed.xml`, `latest/regression_report.html`) show warnings only

### URL Normalization

All GitHub Pages URLs normalized to `https://eyoair21.github.io/Trade_bot` (lowercase 'bot'):
- Workflow job summaries
- Documentation files
- Script outputs

### Documentation

- **`.artifacts/PAGES_DEPLOY_README.md`**: Complete deployment guide with:
  - One-time UI configuration steps
  - Re-kick deployment instructions
  - PowerShell smoke test snippet
  - Troubleshooting guide

## Manual Steps Required

### 1. Configure Workflow Permissions (UI)

1. **Repository → Settings → Actions → General**
2. **Workflow permissions:** Read and write permissions
3. Click **Save**

### 2. Configure Pages Source (UI)

1. **Repository → Settings → Pages**
2. **Source:** GitHub Actions
3. Click **Save**

**Important:** When using Actions deployment, do not select "Deploy from a branch". The `gh-pages` branch (if it exists) is not used as the source.

### 3. Re-Kick Deployment

After merging, trigger a deployment:

```bash
git pull --ff-only
git commit --allow-empty -m "ci: deploy via Pages Actions"
git push origin main
```

## Verification

### Smoke Test (PowerShell)

After deployment completes, verify endpoints:

```powershell
$base="https://eyoair21.github.io/Trade_bot"
$eps=@("","reports/","reports/history.json","reports/feed.xml")
foreach($e in $eps){$u=("$base/$e").Replace("//","/");try{$c=(Invoke-WebRequest -UseBasicParsing -Uri $u -Method Head -TimeoutSec 20).StatusCode.value__}catch{$c="ERR"}("{0,-60} {1}" -f $u,$c)}
```

**Expected:**
- `/` → `200`
- `/reports/` → `200`
- `history.json` and `feed.xml` may be `404` until first sweep completes

### Check Actions

1. Go to **Actions** tab
2. Look for **"Pages build and deployment"** workflow (auto-created)
3. Verify `deploy-pages` job shows: `configure-pages` → `upload-pages-artifact` → `deploy-pages`
4. Check environment: `github-pages`

## Acceptance Criteria

- [x] `.github/workflows/nightly-sweep.yml` has `permissions: pages: write, id-token: write, contents: read`
- [x] Workflow includes guard step ensuring `public/index.html` exists
- [x] Workflow uses `configure-pages@v5`, `upload-pages-artifact@v3`, `deploy-pages@v4`
- [x] All GitHub Pages URLs use `Trade_bot` casing (lowercase 'bot')
- [x] `.github/workflows/pages-health.yml` exists (cron + manual) and fails if `/` or `/reports/` is not 200
- [x] `.artifacts/PAGES_DEPLOY_README.md` explains manual UI toggles and smoke test

## Notes

- No API keys or external secrets required
- `gh-pages` branch remains but won't be used as source when Actions deploy
- Pages deploy automatically after nightly sweep regression check passes
- Guard step ensures minimal `public/index.html` exists even if report generation fails
- Health check runs daily at 06:00 UTC and can be manually triggered
