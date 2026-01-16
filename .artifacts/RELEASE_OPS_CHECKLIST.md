# Release Ops Finisher Checklist

## ✅ Step 1: Verify Repository Settings

### Check Workflow Permissions

**Path:** Repository → Settings → Actions → General → Workflow permissions

**Required:** "Read and write permissions" must be selected

**If not set:**
1. Go to: https://github.com/eyoair21/Trade_Bot/settings/actions
2. Scroll to "Workflow permissions"
3. Select: **Read and write permissions**
4. Click **Save**

### Check Pages Source

**Path:** Repository → Settings → Pages → Source

**Required:** "GitHub Actions" must be selected (NOT "Deploy from a branch")

**If not set:**
1. Go to: https://github.com/eyoair21/Trade_Bot/settings/pages
2. Under "Source", select: **GitHub Actions**
3. Click **Save**

---

## ✅ Step 2: Empty Commit Created

**Status:** ✅ Empty commit created and pushed to `main`

**Commit message:** `ci: deploy via Pages Actions`

**This triggers:**
- `configure-pages@v5`
- `upload-pages-artifact@v3` (uploads `public/` directory)
- `deploy-pages@v4` (deploys to GitHub Pages)

---

## ✅ Step 3: Monitor Pages Deployment

### Actions Link

**Latest Pages Build and Deployment:**
https://github.com/eyoair21/Trade_Bot/actions/workflows/pages.yml

**All Actions:**
https://github.com/eyoair21/Trade_Bot/actions

**What to look for:**
1. "Pages build and deployment" workflow appears (auto-created)
2. Job targets environment: `github-pages`
3. Steps: `Configure Pages` → `Upload Pages artifact` → `Deploy to GitHub Pages`
4. All steps show green checkmarks ✅

**Expected workflow name:** "Pages build and deployment" (created automatically by `deploy-pages` action)

---

## ✅ Step 4: Smoke Test (PowerShell)

After "Pages build and deployment" succeeds, run:

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

| Endpoint | Status | Required? |
|----------|--------|-----------|
| `https://eyoair21.github.io/Trade_bot/` | `200` | ✅ **YES** |
| `https://eyoair21.github.io/Trade_bot/reports/` | `200` | ✅ **YES** |
| `https://eyoair21.github.io/Trade_bot/reports/history.json` | `200` or `404` | ⚠️ Optional (may be 404 until first sweep) |
| `https://eyoair21.github.io/Trade_bot/reports/feed.xml` | `200` or `404` | ⚠️ Optional (may be 404 until first sweep) |

### Interpretation

**✅ Success:**
- `/` returns `200` → Root redirect works
- `/reports/` returns `200` → Reports index accessible

**⚠️ Acceptable:**
- `history.json` returns `404` → Will be created after first nightly sweep
- `feed.xml` returns `404` → Will be created after first nightly sweep

**❌ Failure:**
- `/` returns `ERR` or `404` → Pages not deployed correctly
- `/reports/` returns `ERR` or `404` → Reports directory missing

---

## 🐛 Troubleshooting

### No "Pages build and deployment" Job Appears

**Top 3 Causes:**

1. **Wrong Pages Source Setting**
   - **Symptom:** Workflow doesn't appear at all
   - **Fix:** Go to Settings → Pages → Source → Select "GitHub Actions" → Save
   - **Verify:** Settings → Pages should show "Your site is live at https://eyoair21.github.io/Trade_bot/"

2. **Missing "github-pages" Environment Deploy**
   - **Symptom:** Workflow appears but deploy step fails
   - **Fix:** Check Settings → Environments → github-pages
   - **Verify:** Ensure `main` branch is allowed, no required reviewers blocking

3. **Artifact Didn't Include public/**
   - **Symptom:** Deploy succeeds but endpoints return 404
   - **Fix:** Check "Upload Pages artifact" step logs
   - **Verify:** Should show `public/` directory with `index.html` inside
   - **Note:** Guard step ensures `public/index.html` exists even if reports aren't generated

**Additional Checks:**
- Re-verify Steps 1a and 1b above
- Ensure commit landed on `main` branch
- Check Actions → All workflows for any failed runs

### ERR/404 on Endpoints

**Checklist:**

1. **Pages Deploy Run Logs**
   - Go to: Actions → "Pages build and deployment" → Latest run
   - Check "Upload Pages artifact" step
   - Verify it shows `public/` directory contents
   - Look for `public/index.html` in the artifact

2. **Environment Protection**
   - Go to: Settings → Environments → github-pages
   - Ensure no required reviewers are blocking
   - Verify `main` branch is in allowed branches

3. **Guard Step Execution**
   - Check "Prepare Pages directory" step logs
   - Should see: "Ensure public/ has minimal index" step
   - Verify it created `public/index.html` if missing

### URL Casing Mismatch

**Issue:** URLs show `Trade_Bot` (uppercase B) instead of `Trade_bot` (lowercase b)

**All public links should be:**
- `https://eyoair21.github.io/Trade_bot` (lowercase 'bot')

**If you spot wrong casing:**
- Fix in the source file
- Push to `main` (no API keys needed)
- Pages will redeploy automatically

---

## ✅ Done Checklist

### Release Publishing

- [ ] **Publish Release v0.6.5**
  - Go to: https://github.com/eyoair21/Trade_Bot/releases/new
  - **Tag:** `v0.6.5` (create if doesn't exist)
  - **Title:** `v0.6.5 — "Insights & Trends"`
  - **Body:** Copy from `.artifacts/RELEASE_NOTES_v0.6.5.md`
  - Click **Publish release**

**File path:** `.artifacts/RELEASE_NOTES_v0.6.5.md`

### Issue Creation

- [ ] **Create 5 Follow-up Issues**
  - Go to: https://github.com/eyoair21/Trade_Bot/issues/new
  - Copy/paste from `.artifacts/ISSUE_TEMPLATES_v0.6.5.md`
  - Create each issue with title and body from the template

**File path:** `.artifacts/ISSUE_TEMPLATES_v0.6.5.md`

**Issues to create:**
1. `[CI] E2E: Nightly Pages + Atom feed smoke test`
2. `[Alerts] Discord/Slack webhook on FAIL verdict`
3. `[Pages] Coverage badge on landing page`
4. `[Pages] Index pagination when runs > 50`
5. `[CI] Integrate HTML minification into Pages workflow (optional)`

---

## 📋 Quick Reference Links

**Repository Settings:**
- Actions Permissions: https://github.com/eyoair21/Trade_Bot/settings/actions
- Pages Source: https://github.com/eyoair21/Trade_Bot/settings/pages
- Environments: https://github.com/eyoair21/Trade_Bot/settings/environments

**Actions & Workflows:**
- All Actions: https://github.com/eyoair21/Trade_Bot/actions
- Pages Deployment: https://github.com/eyoair21/Trade_Bot/actions/workflows/pages.yml
- Pages Health Check: https://github.com/eyoair21/Trade_Bot/actions/workflows/pages-health.yml

**Pages Site:**
- Home: https://eyoair21.github.io/Trade_bot/
- Reports: https://eyoair21.github.io/Trade_bot/reports/

**Release & Issues:**
- New Release: https://github.com/eyoair21/Trade_Bot/releases/new
- New Issue: https://github.com/eyoair21/Trade_Bot/issues/new

---

## 🎯 Final Status

**Current Status:**
- ✅ Empty commit created and pushed
- ⏳ Waiting for "Pages build and deployment" workflow
- ⏳ Waiting for smoke test verification
- ⏳ Pending: Release publish
- ⏳ Pending: Issue creation

**Next Steps:**
1. Monitor Actions for "Pages build and deployment" success
2. Run smoke test after deployment completes
3. Publish release using `.artifacts/RELEASE_NOTES_v0.6.5.md`
4. Create issues using `.artifacts/ISSUE_TEMPLATES_v0.6.5.md`



