#!/usr/bin/env bash
# init_gh_pages.sh - Initialize gh-pages orphan branch for GitHub Pages
# Usage: ./scripts/dev/init_gh_pages.sh
#
# This script creates an orphan gh-pages branch with minimal bootstrap content.
# Run once when setting up a new repository for GitHub Pages deployment.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PAGES_BRANCH="gh-pages"

echo "=== Initializing gh-pages branch ==="

# Check if gh-pages already exists
if git show-ref --verify --quiet "refs/heads/$PAGES_BRANCH" 2>/dev/null; then
    echo "Error: $PAGES_BRANCH branch already exists locally"
    exit 1
fi

if git ls-remote --heads origin "$PAGES_BRANCH" | grep -q "$PAGES_BRANCH"; then
    echo "Error: $PAGES_BRANCH branch already exists on origin"
    exit 1
fi

# Save current branch
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Create orphan branch
echo "Creating orphan branch: $PAGES_BRANCH"
git checkout --orphan "$PAGES_BRANCH"

# Remove all tracked files
git rm -rf . 2>/dev/null || true

# Create minimal bootstrap content
echo "Creating bootstrap content..."

cat > index.html << 'EOF'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TraderBot Reports</title>
</head>
<body>
  <h1>TraderBot Reports</h1>
  <p>Deployment initializing... CI will publish nightly regression reports here.</p>
  <p><a href="reports/">View Reports</a></p>
</body>
</html>
EOF

cat > 404.html << 'EOF'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>404 - Not Found</title>
</head>
<body>
  <h1>404 - Page Not Found</h1>
  <p><a href="/">Return to home</a></p>
</body>
</html>
EOF

mkdir -p reports
cat > reports/index.html << 'EOF'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TraderBot Reports Index</title>
</head>
<body>
  <h1>Reports</h1>
  <p>Regression report artifacts will appear here after the first successful CI workflow run.</p>
  <p><a href="../">Back to home</a></p>
</body>
</html>
EOF

# Stage and commit
git add index.html 404.html reports/

git commit -m "chore(pages): bootstrap gh-pages branch

Initialize GitHub Pages with minimal placeholder content.
CI workflow will populate with actual reports."

# Push to origin
echo "Pushing to origin/$PAGES_BRANCH..."
git push -u origin "$PAGES_BRANCH"

# Return to original branch
echo "Returning to $CURRENT_BRANCH..."
git checkout "$CURRENT_BRANCH"

echo ""
echo "=== Success ==="
echo "gh-pages branch created and pushed."
echo ""
echo "Next steps:"
echo "1. Go to: https://github.com/YOUR_USER/YOUR_REPO/settings/pages"
echo "2. Set Source: Deploy from a branch"
echo "3. Set Branch: gh-pages / (root)"
echo "4. Click Save"
echo ""
echo "Pages URL will be: https://YOUR_USER.github.io/YOUR_REPO/"
