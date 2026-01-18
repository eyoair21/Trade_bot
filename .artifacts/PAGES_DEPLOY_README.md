## GitHub Pages – Smoke Test

Use this quick PowerShell check after a deploy (or against a local preview) to verify that
the root, latest report, and 404 handling are wired correctly:

```powershell
# Basic Pages smoke test
$base = "https://eyoair21.github.io/Trade_Bot"

# 1) Root should return HTTP 200
(Invoke-WebRequest "$base/" -UseBasicParsing).StatusCode

# 2) Latest regression report should return HTTP 200
(Invoke-WebRequest "$base/reports/latest/regression_report.html" -UseBasicParsing).StatusCode

# 3) Non-existent report should return HTTP 404
(Invoke-WebRequest "$base/reports/does-not-exist.html" -UseBasicParsing).StatusCode
```

**Expected:**
- Root `/` → `200`
- `/reports/latest/regression_report.html` → `200`
- `/reports/does-not-exist.html` → `404`


