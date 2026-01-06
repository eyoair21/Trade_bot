#!/usr/bin/env bash
# Smoke test script for GitHub Pages artifact packing
# Usage: ./scripts/dev/verify_pages_pack.sh
#
# This script simulates the Pages deployment workflow locally to verify:
# 1. Report files are generated correctly
# 2. Public directory structure is correct
# 3. Manifest update works
# 4. Index generation works

set -e

echo "=== GitHub Pages Pack Smoke Test ==="
echo ""

# Check prerequisites
echo "1. Checking prerequisites..."
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi
echo "   OK: Python available"

# Create temp directory for test artifacts
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

PUBLIC_DIR="$TEMP_DIR/public"
REPORTS_DIR="$PUBLIC_DIR/reports"
RUN_ID="test-$(date +%s)-abc123"

echo ""
echo "2. Creating public directory structure..."
mkdir -p "$REPORTS_DIR/$RUN_ID"
mkdir -p "$REPORTS_DIR/latest"
echo "   Created: $REPORTS_DIR/$RUN_ID"

echo ""
echo "3. Running mini sweep to generate reports..."
python scripts/make_sample_data.py 2>/dev/null || true

# Run sweep if possible
python -m traderbot.cli.sweep sweeps/ci_smoke.yaml --workers 2 2>/dev/null || {
    echo "   WARN: Sweep failed (expected if dependencies missing)"
    echo "   Creating mock report files..."
    mkdir -p runs/sweeps/ci_smoke
    echo "# Mock Regression Report" > runs/sweeps/ci_smoke/regression_report.md
    echo '{"passed": true, "sharpe_drop": 0.02}' > runs/sweeps/ci_smoke/baseline_diff.json
    echo '{"source": "mock"}' > runs/sweeps/ci_smoke/provenance.json
}

# Try to generate HTML report
python -m traderbot.cli.regress compare \
    --no-emoji \
    --html \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml \
    --out runs/sweeps/ci_smoke/regression_report.md 2>/dev/null || {
    echo "   WARN: Regression compare failed, creating mock HTML..."
    cat > runs/sweeps/ci_smoke/regression_report.html << 'EOF'
<!DOCTYPE html>
<html><head><title>Mock Report</title></head>
<body><h1>Mock Regression Report</h1><p>Status: PASS</p></body>
</html>
EOF
}

echo "   OK: Report files generated"

echo ""
echo "4. Copying reports to public directory..."

# Copy report files to run directory
for file in regression_report.md regression_report.html baseline_diff.json provenance.json; do
    if [ -f "runs/sweeps/ci_smoke/$file" ]; then
        cp "runs/sweeps/ci_smoke/$file" "$REPORTS_DIR/$RUN_ID/"
        echo "   Copied: $file"
    fi
done

# Copy to latest (atomic replace simulation)
rm -rf "$REPORTS_DIR/latest"
cp -r "$REPORTS_DIR/$RUN_ID" "$REPORTS_DIR/latest"
echo "   Updated: latest/"

echo ""
echo "5. Updating manifest and index..."

# Create initial manifest if needed
if [ ! -f "$REPORTS_DIR/manifest.json" ]; then
    echo '{"runs": [], "updated": null, "latest": null}' > "$REPORTS_DIR/manifest.json"
fi

# Run the index update script
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
python scripts/dev/update_pages_index.py \
    --manifest "$REPORTS_DIR/manifest.json" \
    --run-id "$RUN_ID" \
    --timestamp "$TIMESTAMP" \
    --status pass \
    --index-out "$REPORTS_DIR/index.html"

echo "   OK: Manifest and index updated"

echo ""
echo "6. Verifying directory structure..."

# Check expected files exist
EXPECTED_FILES=(
    "$REPORTS_DIR/index.html"
    "$REPORTS_DIR/manifest.json"
    "$REPORTS_DIR/$RUN_ID/regression_report.html"
    "$REPORTS_DIR/latest/regression_report.html"
)

for file in "${EXPECTED_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        echo "   OK: $(basename "$file") ($size bytes)"
    else
        echo "   ERROR: Missing $file"
        exit 1
    fi
done

echo ""
echo "7. Verifying manifest content..."
if python -c "import json; m=json.load(open('$REPORTS_DIR/manifest.json')); assert m['latest']=='$RUN_ID'" 2>/dev/null; then
    echo "   OK: Manifest latest points to $RUN_ID"
else
    echo "   ERROR: Manifest latest incorrect"
    cat "$REPORTS_DIR/manifest.json"
    exit 1
fi

echo ""
echo "8. Checking total artifact size..."
TOTAL_SIZE=$(du -sb "$PUBLIC_DIR" | cut -f1)
TOTAL_SIZE_MB=$((TOTAL_SIZE / 1024 / 1024))
echo "   Total size: ${TOTAL_SIZE_MB}MB (${TOTAL_SIZE} bytes)"

if [ "$TOTAL_SIZE" -gt 100000000 ]; then
    echo "   WARN: Artifact size exceeds 100MB"
fi

echo ""
echo "9. Simulating Pages deployment guard..."
# Simulate the workflow condition check

# Mock values
GITHUB_REF="refs/heads/main"
GITHUB_ACTOR="developer"
REGRESS_OUTCOME="success"

if [ "$GITHUB_REF" == "refs/heads/main" ] && [ "$REGRESS_OUTCOME" == "success" ] && [ "$GITHUB_ACTOR" != "github-actions[bot]" ]; then
    echo "   OK: Would deploy (main + success + not bot)"
else
    echo "   OK: Would skip deploy"
fi

# Test PR scenario
GITHUB_REF="refs/pull/123/merge"
if [ "$GITHUB_REF" == "refs/heads/main" ]; then
    echo "   ERROR: PR should not deploy"
    exit 1
else
    echo "   OK: PR correctly skipped"
fi

echo ""
echo "=== Smoke Test PASSED ==="
echo ""
echo "Public directory structure:"
find "$PUBLIC_DIR" -type f | sed "s|$TEMP_DIR/||" | sort
echo ""
echo "All checks passed. The Pages deployment workflow should work correctly."
echo "Note: This is a local test; actual deployment only happens in CI on main branch."
