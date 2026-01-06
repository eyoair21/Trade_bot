#!/usr/bin/env bash
# Initialize gh-pages branch for GitHub Pages deployment
# Usage: ./scripts/dev/init_gh_pages.sh
#
# This script creates the gh-pages branch with initial scaffolding if it doesn't exist.
# Run this once before enabling GitHub Pages in repository settings.

set -e

echo "=== GitHub Pages Branch Initialization ==="
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "ERROR: Not in a git repository"
    exit 1
fi

# Check if gh-pages branch already exists
if git show-ref --verify --quiet refs/heads/gh-pages 2>/dev/null; then
    echo "INFO: gh-pages branch already exists locally"
    echo "      Use 'git checkout gh-pages' to view it"
    exit 0
fi

if git ls-remote --exit-code --heads origin gh-pages > /dev/null 2>&1; then
    echo "INFO: gh-pages branch already exists on remote"
    echo "      Use 'git fetch origin gh-pages && git checkout gh-pages' to view it"
    exit 0
fi

echo "Creating gh-pages branch with initial scaffolding..."

# Create orphan branch (no history from main)
git checkout --orphan gh-pages

# Remove all files from staging
git rm -rf . 2>/dev/null || true

# Create .nojekyll to disable Jekyll processing
touch .nojekyll
echo "# Disable Jekyll processing for GitHub Pages" > .nojekyll

# Create reports directory structure
mkdir -p reports/latest

# Create initial index.html
cat > reports/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TraderBot Regression Reports</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: #f5f5f5;
        }
        h1 { color: #333; border-bottom: 2px solid #0366d6; padding-bottom: 0.5rem; }
        .latest-link {
            background: #28a745;
            color: white;
            padding: 1rem 2rem;
            border-radius: 6px;
            text-decoration: none;
            display: inline-block;
            margin: 1rem 0;
            font-weight: bold;
        }
        .latest-link:hover { background: #22863a; }
        .run-list { list-style: none; padding: 0; }
        .run-list li {
            background: white;
            margin: 0.5rem 0;
            padding: 1rem;
            border-radius: 4px;
            border-left: 4px solid #0366d6;
        }
        .run-list a { color: #0366d6; text-decoration: none; }
        .run-list a:hover { text-decoration: underline; }
        .timestamp { color: #666; font-size: 0.9em; }
        .no-reports { color: #666; font-style: italic; }
        footer { margin-top: 2rem; color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>TraderBot Regression Reports</h1>

    <p>
        <a href="latest/regression_report.html" class="latest-link">View Latest Report</a>
    </p>

    <h2>Recent Runs</h2>
    <div id="runs-container">
        <p class="no-reports">No reports available yet. Reports will appear here after the first successful nightly sweep on main.</p>
    </div>

    <footer>
        <p>Reports are automatically published on successful regression checks on the main branch.</p>
        <p>See <a href="https://github.com/eyoair21/Trade_Bot">repository</a> for source code.</p>
    </footer>

    <script>
        // Load manifest and populate run list if available
        fetch('manifest.json')
            .then(r => r.json())
            .then(data => {
                if (data.runs && data.runs.length > 0) {
                    const container = document.getElementById('runs-container');
                    const ul = document.createElement('ul');
                    ul.className = 'run-list';

                    data.runs.slice(0, 20).forEach(run => {
                        const li = document.createElement('li');
                        li.innerHTML = `
                            <a href="${run.id}/regression_report.html">${run.id}</a>
                            <span class="timestamp"> - ${run.timestamp}</span>
                        `;
                        ul.appendChild(li);
                    });

                    container.innerHTML = '';
                    container.appendChild(ul);
                }
            })
            .catch(() => {
                // manifest.json doesn't exist yet, keep default message
            });
    </script>
</body>
</html>
EOF

# Create placeholder for latest report
cat > reports/latest/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=regression_report.html">
    <title>Redirecting to Latest Report</title>
</head>
<body>
    <p>Redirecting to <a href="regression_report.html">latest regression report</a>...</p>
</body>
</html>
EOF

# Create empty manifest
echo '{"runs": [], "updated": "pending"}' > reports/manifest.json

# Create root index that redirects to reports
cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=reports/">
    <title>TraderBot Reports</title>
</head>
<body>
    <p>Redirecting to <a href="reports/">regression reports</a>...</p>
</body>
</html>
EOF

# Create 404.html for GitHub Pages
cat > 404.html << 'EOF'
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - Not Found | TraderBot Reports</title>
    <style>
        :root {
            --bg-primary: #f5f5f5;
            --bg-card: white;
            --text-primary: #333;
            --text-secondary: #666;
            --link-color: #0366d6;
        }
        [data-theme="dark"] {
            --bg-primary: #1a1a2e;
            --bg-card: #16213e;
            --text-primary: #e4e4e7;
            --text-secondary: #a1a1aa;
            --link-color: #58a6ff;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 4rem 2rem;
            background: var(--bg-primary);
            color: var(--text-primary);
            text-align: center;
        }
        h1 { font-size: 4rem; margin: 0; color: var(--link-color); }
        h2 { color: var(--text-primary); margin-top: 0; }
        p { color: var(--text-secondary); }
        a { color: var(--link-color); }
        .links { margin-top: 2rem; }
        .links a {
            display: inline-block;
            margin: 0.5rem;
            padding: 0.75rem 1.5rem;
            background: var(--link-color);
            color: white;
            text-decoration: none;
            border-radius: 6px;
        }
        .links a:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <h1>404</h1>
    <h2>Page Not Found</h2>
    <p>The regression report you're looking for doesn't exist or has been pruned.</p>
    <p>Old reports are automatically removed after 50 runs to save storage.</p>
    <div class="links">
        <a href="/Trade_Bot/reports/">View All Reports</a>
        <a href="/Trade_Bot/reports/latest/regression_report.html">Latest Report</a>
    </div>
    <script>
        (function() {
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                document.documentElement.setAttribute('data-theme', 'dark');
            }
        })();
    </script>
</body>
</html>
EOF

# Create robots.txt
cat > robots.txt << 'EOF'
# TraderBot GitHub Pages
# https://eyoair21.github.io/Trade_Bot/

User-agent: *
Allow: /

# Sitemap (if generated in future)
# Sitemap: https://eyoair21.github.io/Trade_Bot/sitemap.xml
EOF

# Stage and commit
git add .nojekyll index.html 404.html robots.txt reports/

git commit -m "chore: initialize gh-pages branch for GitHub Pages

- Add .nojekyll to disable Jekyll processing
- Add reports/index.html with run listing
- Add reports/manifest.json for tracking runs
- Add placeholder for /reports/latest/
- Add 404.html for missing page handling
- Add robots.txt for search engine guidance"

echo ""
echo "=== gh-pages branch created ==="
echo ""
echo "Next steps:"
echo "1. Push the branch: git push -u origin gh-pages"
echo "2. Switch back to main: git checkout main"
echo "3. Enable GitHub Pages in repo settings:"
echo "   Settings > Pages > Source: Deploy from branch > gh-pages"
echo ""
echo "After that, reports will be available at:"
echo "  https://eyoair21.github.io/Trade_Bot/reports/"
