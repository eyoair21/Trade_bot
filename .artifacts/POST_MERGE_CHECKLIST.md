# Post-Merge Checklist: Pages Actions Deployment

## ✅ Step 1: Open & Merge PR

**PR Link:** https://github.com/eyoair21/Trade_Bot/pull/new/chore/pages-actions-deploy

**Title:** `ci(pages): deploy via Actions; normalize URLs; add health check`

**Body:** Copy from `.artifacts/PR_DESCRIPTION.md`

**Action:** Merge to `main` branch

---

## ✅ Step 2: Configure Repository Settings (One-Time UI)

### 2a. Workflow Permissions

1. Go to: **Repository → Settings → Actions → General**
2. Scroll to: **Workflow permissions**
3. Select: **Read and write permissions**
4. Click: **Save**

### 2b. Pages Source

1. Go to: **Repository → Settings → Pages**
2. Under **Source**, select: **GitHub Actions**
3. Click: **Save**

**Important:** Do NOT select "Deploy from a branch". When using Actions deployment, the source must be "GitHub Actions".

---

## ✅ Step 3: Re-Kick Deployment (PowerShell)

After merging the PR and configuring settings, run:

```powershell
Set-Location E:\Trade_Bot\traderbot
git checkout main
git pull --ff-only
git commit --allow-empty -m "ci: deploy via Pages Actions"
git push origin main
```

This triggers the `deploy-pages` job which will:
1. Run `configure-pages@v5`
2. Run `upload-pages-artifact@v3` (uploads `public/` directory)
3. Run `deploy-pages@v4` (deploys to GitHub Pages)

---

## ✅ Step 4: Watch Actions

**Actions Overview:** https://github.com/eyoair21/Trade_Bot/actions

**What to look for:**
1. **"CI"** workflow should run on the push to `main`
2. **"Pages build and deployment"** workflow should appear (auto-created by `deploy-pages` action)
3. The `deploy-pages` job should target environment: `github-pages`
4. All steps should succeed (green checkmarks)

**Expected workflow steps:**
- `Configure Pages` → `Upload Pages artifact` → `Deploy to GitHub Pages`

---

## ✅ Step 5: Smoke Test (PowerShell)

After the "Pages build and deployment" workflow succeeds, verify endpoints:

```powershell
$base="https://eyoair21.github.io/Trade_bot"
$eps=@("","reports/","reports/history.json","reports/feed.xml")
foreach($e in $eps){
  $u=("$base/$e").Replace("//","/")
  try{$c=(Invoke-WebRequest -UseBasicParsing -Uri $u -Method Head -TimeoutSec 20).StatusCode.value__}
  catch{$c="ERR"}
  "{0,-60} {1}" -f $u,$c
}
```

### Expected Results

| Endpoint | Expected | Notes |
|----------|----------|-------|
| `/` | `200` | Root redirect (required) |
| `/reports/` | `200` | Reports index (required) |
| `/reports/history.json` | `200` or `404` | May be `404` until first sweep completes |
| `/reports/feed.xml` | `200` or `404` | May be `404` until first sweep completes |

**Success Criteria:**
- ✅ `/` returns `200`
- ✅ `/reports/` returns `200`
- ⚠️ `history.json` and `feed.xml` may be `404` initially (this is OK)

---

## ✅ Step 6: (Optional) Run Health Check Manually

1. Go to: **Actions → Pages Health Check**
2. Click: **Run workflow** (manual dispatch)
3. Verify it passes

**What it checks:**
- `/` must return `200` (fails if not)
- `/reports/` must return `200` (fails if not)
- `history.json`, `feed.xml`, `latest/regression_report.html` are optional (warnings only)

---

## 🐛 Troubleshooting

### No "Pages build and deployment" Workflow Appears

**Causes:**
- PR not merged yet
- Settings → Pages → Source is not set to "GitHub Actions"
- Settings → Actions → Workflow permissions is not "Read and write"

**Fix:**
1. Re-check Steps 2a and 2b above
2. Ensure PR is merged to `main`
3. Push another commit to `main` to trigger workflows

### Endpoints Return 404/ERR

**Causes:**
- Pages not deployed yet (workflow still running)
- Settings → Pages → Source is wrong
- Deployment failed

**Fix:**
1. Check **Actions → Pages build and deployment** workflow status
2. Verify **Settings → Pages → Source = GitHub Actions**
3. Check deploy job logs for errors
4. Verify `Upload Pages artifact` step picked up `public/` directory
5. Look for the guard step that creates `public/index.html`

### Wrong URL Casing

**Issue:** URLs show `Trade_Bot` (uppercase B) instead of `Trade_bot` (lowercase b)

**Fix:**
- All URLs should use `https://eyoair21.github.io/Trade_bot`
- If you see wrong casing in rendered pages, it's safe to fix and push to `main`
- Check browser cache (hard refresh: `Ctrl+F5`)

### GITHUB_TOKEN Push Blocked

**Causes:**
- Workflow permissions not set to "Read and write"
- Branch protection rules blocking Actions

**Fix:**
1. Re-check **Settings → Actions → Workflow permissions**
2. Check **Settings → Branches** for protection rules
3. Verify **Settings → Environments → github-pages** allows `main` branch

---

## ✅ Done Checklist

- [ ] PR merged to `main`
- [ ] Settings → Actions → Workflow permissions = "Read and write"
- [ ] Settings → Pages → Source = "GitHub Actions"
- [ ] Empty commit pushed to `main` to trigger deployment
- [ ] "Pages build and deployment" workflow appears and succeeds
- [ ] Smoke test: `/` returns `200`
- [ ] Smoke test: `/reports/` returns `200`
- [ ] (Optional) Health check workflow passes

---

## 📋 Quick Reference

**PR:** https://github.com/eyoair21/Trade_Bot/pull/new/chore/pages-actions-deploy

**Actions:** https://github.com/eyoair21/Trade_Bot/actions

**Pages Site:** https://eyoair21.github.io/Trade_bot/

**Settings → Actions:** https://github.com/eyoair21/Trade_Bot/settings/actions

**Settings → Pages:** https://github.com/eyoair21/Trade_Bot/settings/pages



