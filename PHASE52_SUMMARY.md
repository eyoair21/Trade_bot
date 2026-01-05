# Phase 5.2: Performance Budgets, Regression Guards, and Baseline Management

## Overview

Phase 5.2 introduces automated performance regression detection for hyperparameter sweeps. This ensures that code changes don't silently degrade trading strategy performance and provides determinism verification for reproducible results.

## Components

### 1. Performance Budget (`sweeps/perf_budget.yaml`)

Defines acceptable performance thresholds:

```yaml
metric: sharpe          # Primary metric to track
mode: max               # Optimization direction (max/min)
min_success_rate: 0.75  # Minimum sweep success rate
max_p90_elapsed_s: 60.0 # Maximum P90 timing
max_sharpe_drop: 0.05   # Maximum allowed metric regression
epsilon_abs: 1e-6       # Tolerance for floating-point comparison
```

### 2. Baseline Artifact (`benchmarks/baseline.json`)

Stores reference performance data:

```json
{
  "git_sha": "abc1234",
  "created_utc": "2026-01-05T00:00:00+00:00",
  "metric": "sharpe",
  "mode": "max",
  "leaderboard": [...],
  "timing": {"p50": 10.0, "p90": 20.0},
  "summary": {
    "best_metric": 1.5,
    "success_rate": 1.0,
    "total_runs": 10
  }
}
```

### 3. Comparison Module (`traderbot/metrics/compare.py`)

Core comparison logic with dataclasses:

- `PerfBudget`: Performance budget configuration
- `BaselineData`: Baseline performance data
- `CurrentData`: Current sweep performance data
- `ComparisonVerdict`: Pass/fail result with detailed messages

Key functions:
- `load_perf_budget()`: Load budget from YAML
- `load_baseline()`: Load baseline from JSON
- `load_current_data()`: Extract data from sweep directory
- `compare_results()`: Run comparison checks
- `generate_regression_report()`: Create markdown report
- `generate_baseline_diff()`: Create diff JSON
- `create_new_baseline()`: Generate new baseline from current

### 4. Regression CLI (`traderbot/cli/regress.py`)

Two commands:

```bash
# Compare current sweep against baseline
python -m traderbot.cli.regress compare \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml \
    --out regression_report.md

# Update baseline from current sweep
python -m traderbot.cli.regress update-baseline \
    --current runs/sweeps/ci_smoke \
    --out benchmarks/baseline.json \
    --sha abc1234
```

Exit codes:
- `0`: Regression check passed
- `1`: Regression check failed or error

### 5. Determinism Check (`--rerun-best` flag)

Added to `sweep.py` for verifying reproducibility:

```bash
python -m traderbot.cli.sweep sweeps/ci_smoke.yaml --rerun-best 3
```

This reruns the best configuration N times with the same seed and reports:
- Max absolute difference in metric value
- Mean and standard deviation across runs
- Whether results are deterministic (diff < 1e-9)

Output: `determinism.json` in sweep directory

## Outputs

### Regression Report (`regression_report.md`)

Markdown report with:
- Pass/fail status
- Metric comparison (current vs baseline)
- Timing comparison
- Success rate comparison
- Detailed failure reasons
- Recommendations

### Baseline Diff (`baseline_diff.json`)

JSON with computed deltas:
```json
{
  "passed": true,
  "metric_delta": 0.05,
  "timing_delta": {"p50": -2.0, "p90": -5.0},
  "success_rate_delta": -0.1,
  "failures": []
}
```

### Determinism Results (`determinism.json`)

```json
{
  "metric": "sharpe",
  "original_value": 1.5,
  "n_reruns": 3,
  "rerun_values": [1.5, 1.5, 1.5],
  "max_abs_diff": 0.0,
  "is_deterministic": true
}
```

## CI Integration

### PR Workflow (`.github/workflows/ci.yml`)

New `regression-check` job that:
1. Runs CI smoke sweep
2. Compares against baseline
3. Posts regression status as PR comment
4. Fails PR if regression detected

### Nightly Workflow (`.github/workflows/nightly-sweep.yml`)

Enhanced with:
1. Regression comparison (continue-on-error)
2. Determinism check (1 rerun)
3. Warning annotation on regression
4. Baseline candidate generation on manual dispatch

## Test Suite

Located in `tests/regress/`:

- `test_compare.py`: Unit tests for comparison logic (~25 tests)
- `test_regress_cli.py`: CLI command tests (~15 tests)
- `test_determinism.py`: Determinism check tests (~10 tests)

Test fixtures in `tests/fixtures/`:
- `baseline_pass.json`: Baseline for pass scenarios
- `baseline_fail.json`: Baseline for fail scenarios
- `perf_budget_strict.yaml`: Strict budget for testing failures
- `perf_budget_relaxed.yaml`: Relaxed budget for testing passes

## Usage Workflow

### During Development

1. Make code changes
2. Run sweep: `python -m traderbot.cli.sweep sweeps/ci_smoke.yaml`
3. Check regression: `python -m traderbot.cli.regress compare ...`
4. If passing, create PR

### Updating Baseline

After verified improvement:

```bash
# Generate new baseline
python -m traderbot.cli.regress update-baseline \
    --current runs/sweeps/latest \
    --out benchmarks/baseline.json

# Commit baseline
git add benchmarks/baseline.json
git commit -m "Update baseline: Sharpe improved to 1.6"
```

### Investigating Failures

1. Check `regression_report.md` for failure details
2. Review `baseline_diff.json` for exact deltas
3. If expected change, update budget thresholds or baseline
4. If unexpected, investigate code changes

## Configuration

### Adjusting Budgets

Edit `sweeps/perf_budget.yaml`:

```yaml
# More permissive during rapid development
max_sharpe_drop: 0.10
min_success_rate: 0.60

# Stricter for production
max_sharpe_drop: 0.02
min_success_rate: 0.90
```

### Multiple Baselines

Maintain separate baselines for different scenarios:

```bash
benchmarks/
  baseline.json         # Primary baseline
  baseline_prod.json    # Production baseline
  baseline_dev.json     # Development baseline
```

## File Structure

```
traderbot/
├── benchmarks/
│   └── baseline.json           # Performance baseline
├── sweeps/
│   └── perf_budget.yaml        # Performance thresholds
├── traderbot/
│   ├── cli/
│   │   ├── regress.py          # Regression CLI
│   │   └── sweep.py            # Updated with --rerun-best
│   └── metrics/
│       └── compare.py          # Comparison logic
├── tests/
│   ├── fixtures/
│   │   ├── baseline_*.json
│   │   └── perf_budget_*.yaml
│   └── regress/
│       ├── __init__.py
│       ├── test_compare.py
│       ├── test_regress_cli.py
│       └── test_determinism.py
├── scripts/
│   └── pack_sweep.py           # Updated to include regression files
└── .github/workflows/
    ├── ci.yml                  # Updated with regression-check job
    └── nightly-sweep.yml       # Updated with regression + determinism
```

## Summary

Phase 5.2 provides:
- Automated regression detection for every PR
- Performance budgets with configurable thresholds
- Baseline management with git SHA tracking
- Determinism verification for reproducibility
- CI integration for continuous quality assurance
- Comprehensive test coverage for regression logic

This ensures trading strategy performance is monitored and protected throughout the development lifecycle.
