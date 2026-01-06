# GitHub Pages Deployment via Actions

This repository uses GitHub Actions to deploy Pages (not branch mode). Follow these steps to configure and verify deployment.

## Manual UI Configuration (One-Time Setup)

### 1. Configure Pages Source

1. Go to **Repository → Settings → Pages**
2. Under **Source**, select: **Deploy from a branch**
3. **Branch:** Select `gh-pages` and `/ (root)`
4. Click **Save**

**Note:** Even though we deploy via Actions, GitHub requires a branch to be selected. The `gh-pages` branch exists but won't be used as the source when Actions deploy.

### 2. Configure Workflow Permissions

1. Go to **Repository → Settings → Actions → General**
2. Scroll to **Workflow permissions**
3. Select: **Read and write permissions**
4. Click **Save**

This allows the workflow to:
- Deploy to GitHub Pages
- Commit badge updates (if needed)

### 3. Configure Environment (Optional)

If you have branch protection or environment restrictions:

1. Go to **Repository → Settings → Environments**
2. Find or create the `github-pages` environment
3. Ensure `main` branch is allowed
4. Remove any required reviewers (Pages deployment should be automatic)

## Re-Kick Deployment

To manually trigger a Pages deployment after making changes:

```powershell
Set-Location E:\Trade_Bot\traderbot
git pull --ff-only
git commit --allow-empty -m "ci: deploy via Pages Actions"
git push origin main
```

Or if you can't push to main directly, include this in your PR description:

```bash
git commit --allow-empty -m "ci: deploy via Pages Actions" && git push
```

## Smoke Test (PowerShell)

After deployment, verify Pages endpoints are accessible:

```powershell
$base = "https://eyoair21.github.io/Trade_bot"
$eps  = @("", "reports/", "reports/history.json", "reports/feed.xml")
foreach ($ep in $eps) {
  $url = "$base/$ep".Replace("//","/")
  try { 
    $code = (Invoke-WebRequest -UseBasicParsing -Uri $url -Method Head -TimeoutSec 20).StatusCode.value__ 
  } 
  catch { $code = "ERR" }
  "{0,-60} {1}" -f $url, $code
}
```

**Expected Results:**
- `/` → `200` (root redirect)
- `/reports/` → `200` (index page)
- `/reports/history.json` → `200` (after first successful sweep)
- `/reports/feed.xml` → `200` (after first successful sweep)

**Note:** `history.json` and `feed.xml` may return `404` on the first run until the nightly sweep populates them. The root and reports index should always return `200` after the first deployment.

## How It Works

1. **Nightly Sweep Workflow** (`nightly-sweep.yml`) runs on schedule or manual trigger
2. After regression check passes, the `deploy-pages` job:
   - Prepares `public/` directory with reports
   - Uses `actions/configure-pages@v5` to set up Pages
   - Uses `actions/upload-pages-artifact@v3` to upload `public/` directory
   - Uses `actions/deploy-pages@v4` to deploy to GitHub Pages
3. Pages are automatically published to: `https://eyoair21.github.io/Trade_bot/`

## Troubleshooting

### Pages Not Deploying

- Check **Actions** tab for `deploy-pages` job status
- Verify **Settings → Pages → Source** is set correctly
- Verify **Settings → Actions → Workflow permissions** is "Read and write"
- Check that regression check passed (Pages only deploy on PASS)

### Wrong Casing 404s

All Pages URLs use `Trade_bot` (lowercase 'b'). If you see 404s:
- Verify URLs use `Trade_bot` not `Trade_Bot`
- Check browser cache (hard refresh: `Ctrl+F5`)

### GITHUB_TOKEN Push Blocked

- Recheck **Settings → Actions → Workflow permissions**
- Ensure branch protection doesn't block `gh-pages` from Actions
- Check environment restrictions in **Settings → Environments**

## Related Files

- `.github/workflows/nightly-sweep.yml` - Main deployment workflow
- `.github/workflows/pages-health.yml` - Daily health check
- `scripts/dev/update_pages_index.py` - Index page generator
- `scripts/dev/make_feed.py` - Atom feed generator

