# GitHub Pages Deployment via Actions

This repository uses GitHub Actions to deploy Pages (not branch mode). Follow these steps to configure and verify deployment.

## One-Time UI Configuration

### 1. Configure Workflow Permissions

1. Go to **Repository → Settings → Actions → General**
2. Scroll to **Workflow permissions**
3. Select: **Read and write permissions**
4. Click **Save**

This allows workflows to deploy to GitHub Pages and commit badge updates.

### 2. Configure Pages Source

1. Go to **Repository → Settings → Pages**
2. Under **Source**, select: **GitHub Actions**
3. Click **Save**

**Note:** When using Actions deployment, the `gh-pages` branch (if it exists) is not used as the source. Pages are deployed directly from Actions artifacts.

## Re-Kick Deployment

To manually trigger a Pages deployment after making changes:

```bash
git pull --ff-only
git commit --allow-empty -m "ci: deploy via Pages Actions"
git push origin main
```

## Smoke Test (PowerShell)

After deployment, verify Pages endpoints are accessible:

```powershell
$base="https://eyoair21.github.io/Trade_bot"
$eps=@("","reports/","reports/history.json","reports/feed.xml")
foreach($e in $eps){$u=("$base/$e").Replace("//","/");try{$c=(Invoke-WebRequest -UseBasicParsing -Uri $u -Method Head -TimeoutSec 20).StatusCode.value__}catch{$c="ERR"}("{0,-60} {1}" -f $u,$c)}
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
   - Ensures minimal `public/index.html` exists (guard step)
   - Uses `actions/configure-pages@v5` to set up Pages
   - Uses `actions/upload-pages-artifact@v3` to upload `public/` directory
   - Uses `actions/deploy-pages@v4` to deploy to GitHub Pages
3. Pages are automatically published to: `https://eyoair21.github.io/Trade_bot/`

## Troubleshooting

### Pages Not Deploying

- Check **Actions** tab for `deploy-pages` job status
- Verify **Settings → Pages → Source** is set to **GitHub Actions**
- Verify **Settings → Actions → Workflow permissions** is "Read and write"
- Check that regression check passed (Pages only deploy on PASS)
- Look for "Pages build and deployment" workflow in Actions (auto-created by deploy-pages)

### Missing "Pages build and deployment" Workflow

If you don't see this workflow:
- Ensure **Settings → Pages → Source** is set to **GitHub Actions** (not "Deploy from a branch")
- The workflow is automatically created when `deploy-pages` action runs
- Check that the `deploy-pages` job completed successfully

### Wrong Casing 404s

All Pages URLs use `Trade_bot` (lowercase 'b'). If you see 404s:
- Verify URLs use `Trade_bot` not `Trade_Bot`
- Check browser cache (hard refresh: `Ctrl+F5` or `Cmd+Shift+R`)
- Verify the Pages site URL in **Settings → Pages** matches the expected casing

### GITHUB_TOKEN Push Blocked

- Recheck **Settings → Actions → Workflow permissions**
- Ensure branch protection doesn't block Actions from deploying
- Check environment restrictions in **Settings → Environments → github-pages**

## Related Files

- `.github/workflows/nightly-sweep.yml` - Main deployment workflow
- `.github/workflows/pages-health.yml` - Daily health check (06:00 UTC)
- `scripts/dev/update_pages_index.py` - Index page generator
- `scripts/dev/make_feed.py` - Atom feed generator
