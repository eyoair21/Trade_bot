# TraderBot - Paper Trading Research Bot

[![CI](https://github.com/eyoair21/Trade_Bot/actions/workflows/ci.yml/badge.svg)](https://github.com/eyoair21/Trade_Bot/actions/workflows/ci.yml)
[![Nightly Sweep](https://github.com/eyoair21/Trade_Bot/actions/workflows/nightly-sweep.yml/badge.svg)](https://github.com/eyoair21/Trade_Bot/actions/workflows/nightly-sweep.yml)
![Regression](badges/regression_status.svg)
[![Latest Report](https://img.shields.io/badge/report-latest-blue)](https://eyoair21.github.io/Trade_Bot/reports/latest/regression_report.html)

A clean, testable Python repository for paper-trading research with deterministic runs, basic momentum strategy, data/risk scaffolding, and CI.

## ⚠️ Safety Notes

- **PAPER TRADING ONLY** - This is a research/simulation tool. Do not use with real money.
- No secrets should be committed to the repository.
- All runs are deterministic when seeded properly.

## Project Layout

```
traderbot/
├── pyproject.toml          # Poetry configuration
├── Makefile                # Convenient dev targets
├── .pre-commit-config.yaml # Pre-commit hooks
├── .env.example            # Environment variable template
├── scripts/
│   └── make_sample_data.py # Generate sample OHLCV data
├── data/
│   └── ohlcv/              # OHLCV parquet files
├── runs/                   # Walk-forward output artifacts
├── traderbot/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── logging_setup.py    # Structured logging setup
│   ├── data/
│   │   ├── adapters/       # Data adapters (parquet, etc.)
│   │   ├── calendar.py     # Business calendar
│   │   ├── corporate_actions.py  # Splits/dividends adjustments
│   │   └── pti_store.py    # Point-in-time store
│   ├── features/
│   │   ├── ta.py           # Technical analysis indicators
│   │   ├── volume.py       # Volume-based features
│   │   ├── sentiment.py    # Sentiment signals (stub)
│   │   └── gov_trades.py   # Government trades signal (stub)
│   ├── engine/
│   │   ├── backtest.py     # Backtesting engine
│   │   ├── risk.py         # Risk management
│   │   ├── broker_sim.py   # Broker simulator
│   │   ├── strategy_base.py    # Base strategy class
│   │   └── strategy_momo.py    # Momentum strategy
│   ├── opt/
│   │   ├── ga_lite.py      # Genetic algorithm (stub)
│   │   └── ga_robust.py    # Robust GA (stub)
│   ├── cli/
│   │   └── walkforward.py  # Walk-forward CLI
│   └── report/
│       └── daily_report.py # Daily reporting (stub)
└── tests/                  # Test suite
```

## Quickstart

### Prerequisites

- Python 3.11+
- Poetry

### Installation

```bash
# Install Poetry if not already installed
python -m pip install --upgrade pip
pip install poetry

# Install dependencies
poetry install

# Install pre-commit hooks
pre-commit install
```

### Generate Sample Data

Generate synthetic OHLCV data for backtesting:

```bash
# Generate sample data (synthetic by default, or uses yfinance if installed)
poetry run python scripts/make_sample_data.py

# Options:
#   --output-dir DIR     Output directory (default: data/ohlcv)
#   --tickers AAPL MSFT  Ticker symbols to generate
#   --start-date DATE    Start date YYYY-MM-DD (default: 2023-01-01)
#   --end-date DATE      End date YYYY-MM-DD (default: 2023-03-31)
#   --seed N             Random seed (default: 42)
#   --synthetic-only     Use synthetic data only (skip yfinance)
```

This creates Parquet files in `data/ohlcv/` with columns:
- `timestamp` (datetime, UTC)
- `open`, `high`, `low`, `close` (float)
- `volume` (int64)

### Using Make Targets

```bash
# Install dependencies
make install

# Generate sample data
make data

# Run full test suite (lint + type + tests with coverage ≥70%)
make test

# Run walk-forward demo
make demo

# Run everything
make all

# Clean generated files
make clean
```

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q

# Run specific test file
poetry run pytest tests/test_smoke.py -v
```

### Linting and Type Checking

```bash
# Run ruff linter
poetry run ruff check .

# Run black formatter check
poetry run black --check .

# Run mypy type checker
poetry run mypy traderbot
```

### Run Walk-Forward Analysis

```bash
# Show help
poetry run python -m traderbot.cli.walkforward --help

# Run walk-forward with sample data
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 \
    --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 \
    --is-ratio 0.6

# Use custom data directory
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 \
    --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 \
    --is-ratio 0.6 \
    --data-root /path/to/ohlcv
```

Results will be saved to `runs/{timestamp}/`:
- `results.json` - Summary results with split-level metrics
- `equity_curve.csv` - Combined OOS equity curve data
- `run_manifest.json` - Run metadata (git SHA, Python version, OS)

## Data Setup

### Option 1: Generate Synthetic Data (Recommended for Testing)

```bash
poetry run python scripts/make_sample_data.py
```

### Option 2: Use Real Data

Place OHLCV parquet files in `./data/ohlcv/{TICKER}.parquet` with columns:
- `timestamp` or `date` (datetime, UTC preferred)
- `open`, `high`, `low`, `close` (float)
- `volume` (int64)

### Option 3: Use yfinance (Optional)

If yfinance is installed, the sample data script will attempt to download real data:

```bash
pip install yfinance
poetry run python scripts/make_sample_data.py
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

## Verification Commands

```bash
# Full verification suite (Phase 1 acceptance)
poetry run ruff check .                                                    # clean
poetry run black --check .                                                 # clean
poetry run mypy traderbot                                                  # runs
poetry run pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q  # passes
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6                 # produces runs/**
```

## One-Liner for Phase 1 Verification

```bash
poetry install && \
poetry run python scripts/make_sample_data.py && \
poetry run pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q && \
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
```

---

## Phase 2: PatchTST Model + Dynamic Universe

Phase 2 adds:
- **PatchTST Model**: Multivariate time series transformer for probability-of-uptrend prediction
- **TorchScript Export**: CPU-optimized inference via `torch.jit.script`
- **Dynamic Universe**: Automatic symbol selection based on liquidity and volatility
- **News Tone Scores**: FinBERT integration (optional) with deterministic fallback

### New Project Structure

```
traderbot/
├── model/
│   ├── __init__.py
│   └── patchtst.py           # PatchTST architecture + TorchScript export
├── data/
│   └── universe.py           # Dynamic universe selection
├── features/
│   ├── ta.py                 # Extended with model feature helpers
│   └── sentiment.py          # get_tone_scores() with FinBERT/fallback
├── scripts/
│   └── train_patchtst.py     # Training script
└── models/
    └── patchtst.ts           # Exported TorchScript model
```

### Training the PatchTST Model

```bash
# Generate sample data first
make data

# Train model (outputs models/patchtst.ts and runs/{ts}/train_patchtst.json)
make train

# Or with custom options:
poetry run python scripts/train_patchtst.py \
    --data-dir data/ohlcv \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-4
```

### Dynamic Universe Mode

Run walk-forward with dynamic universe selection:

```bash
# Static mode (default) - uses provided tickers
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 \
    --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 \
    --is-ratio 0.6 \
    --universe-mode static

# Dynamic mode - selects top N symbols by liquidity/volatility
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 \
    --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA GOOG AMZN META \
    --n-splits 3 \
    --is-ratio 0.6 \
    --universe-mode dynamic
```

Dynamic mode:
- Filters by minimum dollar volume (default: $20M avg 20d)
- Filters by minimum volatility (default: 15% annualized)
- Selects top N symbols (default: 30)
- Writes selection to `runs/{ts}/universe_{day}.json`

### Model Features

The PatchTST model uses these input features:

| Feature | Description |
|---------|-------------|
| `close_ret_1` | 1-day return (clipped ±10%) |
| `rsi_14` | 14-day RSI |
| `atr_14` | 14-day ATR |
| `vwap_gap` | % gap from VWAP |
| `dvol_5` | Dollar volume ratio vs 5d mean |
| `regime_vix` | Volatility regime indicator |

### Configuration

New environment variables for Phase 2:

```bash
# Model configuration
MODEL_LOOKBACK=32
MODEL_FEATURES=close_ret_1,rsi_14,atr_14,vwap_gap,dvol_5,regime_vix
MODEL_PATH=./models/patchtst.ts
MODEL_PATCH_SIZE=8
MODEL_STRIDE=4
MODEL_D_MODEL=64
MODEL_N_HEADS=4
MODEL_N_LAYERS=2
MODEL_D_FF=128
MODEL_DROPOUT=0.1

# Universe configuration
UNIVERSE_MAX_SYMBOLS=30
UNIVERSE_MIN_DOLLAR_VOLUME=20000000
UNIVERSE_MIN_VOLATILITY=0.15
UNIVERSE_LOOKBACK_DAYS=20
```

### Phase 2 Make Targets

```bash
make train   # Train PatchTST model
make demo    # Run walk-forward (static universe)
make demo2   # Run walk-forward (dynamic universe)
```

### Phase 2 Verification

```bash
# Full Phase 2 verification
poetry install && \
make data && \
make train && \
make test && \
make demo2
```

---

## Phase 4: Reproducibility & Run Comparisons

Phase 4 adds deterministic runs and run comparison tools:

### Deterministic Runs with Seed

Control randomness for reproducible results:

```bash
# Run with specific seed
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 --is-ratio 0.6 \
    --seed 123

# Run again with same seed - results will be identical
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 --is-ratio 0.6 \
    --seed 123
```

The `--seed` parameter controls:
- Python's `random` module
- NumPy's random number generator
- PyTorch's random number generator (if installed)

### Run Manifest

Every run creates a `run_manifest.json` file with complete reproducibility information:

```json
{
  "run_id": "2026-01-05T12-30-45",
  "git_sha": "abc1234",
  "seed": 123,
  "params": { /* all CLI parameters */ },
  "universe": ["AAPL", "MSFT", "NVDA"],
  "start_date": "2023-01-10",
  "end_date": "2023-03-31",
  "n_splits": 3,
  "is_ratio": 0.6,
  "sizer": "fixed",
  "sizer_params": {
    "fixed_frac": 0.1,
    "vol_target": 0.2,
    "kelly_cap": 0.25
  },
  "data_digest": "a1b2c3d4e5f6g7h8"
}
```

The manifest is also included in `results.json` and displayed at the top of `report.md`.

### Compare Two Runs

Compare performance between two backtest runs:

```bash
# Compare two runs using Sharpe ratio
poetry run python -m traderbot.cli.compare_runs \
    --a runs/run_seed123 \
    --b runs/run_seed456 \
    --metric sharpe

# Compare using total return
poetry run python -m traderbot.cli.compare_runs \
    --a runs/run_seed123 \
    --b runs/run_seed456 \
    --metric total_return \
    --out runs/my_comparison.md

# Compare using max drawdown (lower is better)
poetry run python -m traderbot.cli.compare_runs \
    --a runs/run_seed123 \
    --b runs/run_seed456 \
    --metric max_dd
```

**Available metrics:**
- `total_return` - Total percentage return
- `sharpe` - Sharpe ratio (annualized)
- `max_dd` - Maximum drawdown percentage

The comparison generates a markdown report with:
- Run details (manifest info, parameters)
- Performance metrics side-by-side
- Winner determination based on chosen metric

### Phase 4 Verification

```bash
# Run two identical runs with same seed
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 --is-ratio 0.6 \
    --seed 123 \
    --output-dir runs/seed123_a

poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 --is-ratio 0.6 \
    --seed 123 \
    --output-dir runs/seed123_b

# Compare results (should be identical)
poetry run python -m traderbot.cli.compare_runs \
    --a runs/seed123_a \
    --b runs/seed123_b \
    --metric sharpe

# Run with different seed
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 --is-ratio 0.6 \
    --seed 456 \
    --output-dir runs/seed456

# Compare different seeds
poetry run python -m traderbot.cli.compare_runs \
    --a runs/seed123_a \
    --b runs/seed456 \
    --metric sharpe
```

---

## Phase 3: Position Sizing, Execution Costs & Calibration

Phase 3 adds production-ready features for realistic backtesting:

### Position Sizing Policies

Three position sizing strategies are available:

```bash
# Fixed fraction (default) - invest 10% of portfolio per signal
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 --is-ratio 0.6 \
    --sizer fixed \
    --fixed-frac 0.10

# Volatility targeting - target 15% annualized volatility per position
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 --is-ratio 0.6 \
    --sizer vol \
    --vol-target 0.15

# Kelly criterion - size by edge with max 25% cap
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 --is-ratio 0.6 \
    --sizer kelly \
    --kelly-cap 0.25
```

### Probability Thresholding

Filter signals by model confidence:

```bash
# Only act on signals with >55% probability
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 --is-ratio 0.6 \
    --proba-threshold 0.55

# Optimize threshold per split using F1 score
poetry run python -m traderbot.cli.walkforward \
    --start-date 2023-01-10 --end-date 2023-03-31 \
    --universe AAPL MSFT NVDA \
    --n-splits 3 --is-ratio 0.6 \
    --opt-threshold
```

### Execution Cost Tracking

Results now include realistic execution costs:
- **Commission**: Per-trade commission costs
- **Per-share fees**: SEC/FINRA fees
- **Slippage**: Market impact estimation

### Calibration Metrics

When calibration is enabled, reports include:
- **Brier Score**: Probability accuracy measure
- **ECE**: Expected Calibration Error
- **Optimal Threshold**: Best cutoff per split

### Phase 3 Demo

```bash
# Full Phase 3 demo with position sizing and thresholding
make demo3
```

---

## Phase 5: Hyperparameter Sweeps & Leaderboard

Phase 5 adds YAML-driven hyperparameter sweeps with parallel execution and automated result aggregation.

### Sweep Configuration

Create a YAML file to define your sweep:

```yaml
# sweeps/my_sweep.yaml
name: sizer_comparison
output_root: runs/sweep_sizers
metric: sharpe
mode: max

fixed_args:
  start_date: "2023-01-10"
  end_date: "2023-03-31"
  universe:
    - AAPL
    - MSFT
    - NVDA
  n_splits: 3
  is_ratio: 0.6
  seed: 42

grid:
  sizer:
    - fixed
    - vol
  proba_threshold:
    - 0.50
    - 0.55
    - 0.60
```

### Running Sweeps

```bash
# Run sweep with 4 parallel workers
python -m traderbot.cli.sweep sweeps/my_sweep.yaml --workers 4

# Preview sweep without running
python -m traderbot.cli.sweep sweeps/my_sweep.yaml --dry-run
```

Output structure:
```
runs/sweep_sizers/
├── sweep_meta.json      # Sweep configuration
├── all_results.json     # All run results
├── run_000/            # Individual run outputs
│   ├── results.json
│   ├── report.md
│   └── equity_curve.csv
├── run_001/
└── ...
```

### Generating Leaderboards

```bash
# Generate leaderboard from sweep
python -m traderbot.cli.leaderboard runs/sweep_sizers

# Export best run to a separate directory
python -m traderbot.cli.leaderboard runs/sweep_sizers --export-best runs/best_run

# Export specific rank (e.g., 3rd best)
python -m traderbot.cli.leaderboard runs/sweep_sizers \
    --export-rank 3 \
    --export-dir runs/third_best
```

Generated files:
- `leaderboard.csv` - Full rankings in CSV format
- `leaderboard.md` - Markdown report with top runs

### Example Sweep Configs

Two example configurations are provided:

**Fixed vs Vol Sizer Comparison:**
```bash
python -m traderbot.cli.sweep sweeps/example_fixed_vs_vol.yaml --workers 4
```

**Threshold Optimization:**
```bash
python -m traderbot.cli.sweep sweeps/example_threshold.yaml --workers 2
```

### Phase 5 Verification

```bash
# Run a small sweep
python -m traderbot.cli.sweep sweeps/example_threshold.yaml --workers 2

# Generate and view leaderboard
python -m traderbot.cli.leaderboard runs/sweep_threshold

# Export best configuration
python -m traderbot.cli.leaderboard runs/sweep_threshold --export-best runs/best_threshold
```

---

## Phase 5.1: CI, Artifacts, and Profiling

Phase 5.1 adds continuous integration, artifact management, and performance profiling:

### CI/CD Workflows

**Lint & Test CI:**
- Runs on every push and PR to main
- Tests on Ubuntu and Windows with Python 3.11 and 3.12
- Enforces 70% code coverage
- Uploads coverage artifacts on failure

**Nightly Sweep:**
- Runs daily at 03:17 UTC
- Executes smoke test sweep (4 configurations)
- Generates leaderboard
- Packs and uploads artifacts (7-day retention)
- Can be triggered manually via workflow_dispatch

### Sweep Timing & Profiling

Track performance of sweep runs:

```bash
# Run sweep with timing
python -m traderbot.cli.sweep sweeps/my_sweep.yaml --workers 4 --time

# Output: timings.csv with per-run and per-phase durations
```

The `timings.csv` file includes:
- `run_idx` - Run index
- `elapsed_s` - Total elapsed time
- `load_s` - Data loading time
- `splits_s` - Split creation time
- `backtest_s` - Backtesting time
- `report_s` - Report generation time
- `total_s` - Sum of all phases

After sweep completion, a timing summary is printed:
```
Timing Summary:
  P50 elapsed: 12.34s
  P90 elapsed: 15.67s
  Total runs: 8
```

### Artifact Packing

Pack sweep results for archival or sharing:

```bash
# Pack sweep with top 3 runs
python scripts/pack_sweep.py runs/sweeps/my_sweep --output my_sweep.zip --max-size-mb 80 --top-n 3
```

**What gets packed:**
- Essential files: `sweep_meta.json`, `all_results.json`, `leaderboard.csv/md`, `timings.csv`
- Top N run directories (default: 3)
- Excludes: `.parquet`, `.h5`, `.hdf5` files

**Size guards:**
- If zip > 80MB, automatically creates minimal version with best run only
- Ensures CI artifacts stay within reasonable limits

### Leaderboard with Timing

When `timings.csv` exists, the leaderboard includes average elapsed time:

```markdown
# Sweep Leaderboard: my_sweep

**Ranking by:** sharpe (max)
**Total runs:** 8
**Avg Elapsed Time:** 13.45s
```

### CI Smoke Test

A minimal sweep configuration for CI health checks:

```yaml
# sweeps/ci_smoke.yaml
name: ci_smoke
output_root: runs/sweeps/ci_smoke
metric: sharpe
mode: max

fixed_args:
  start_date: "2023-01-10"
  end_date: "2023-01-31"
  universe: [AAPL, MSFT]
  n_splits: 2
  seed: 42

grid:
  sizer: [fixed, vol]
  proba_threshold: [0.50, 0.55]
```

This runs 4 configurations (2 sizers × 2 thresholds) in ~30 seconds.

### Phase 5.1 Verification

```bash
# Run CI smoke sweep locally
python -m traderbot.cli.sweep sweeps/ci_smoke.yaml --workers 2 --time

# Generate leaderboard
python -m traderbot.cli.leaderboard runs/sweeps/ci_smoke

# Pack artifacts
python scripts/pack_sweep.py runs/sweeps/ci_smoke --output ci_smoke.zip --max-size-mb 80

# Verify zip contents
unzip -l ci_smoke.zip
```

---

## Phase 5.2: Performance Guards & Regression Detection

Phase 5.2 adds automated performance regression detection to prevent silent degradation:

### Performance Budget

Define acceptable performance thresholds in `sweeps/perf_budget.yaml`:

```yaml
metric: sharpe
mode: max
min_success_rate: 0.75    # At least 75% of runs must succeed
max_p90_elapsed_s: 60.0   # P90 timing must be under 60s
max_sharpe_drop: 0.05     # Max 5% drop from baseline
epsilon_abs: 1e-6         # Floating-point tolerance
```

### Baseline Management

The baseline (`benchmarks/baseline.json`) tracks reference performance:

```json
{
  "git_sha": "abc1234",
  "created_utc": "2026-01-05T00:00:00+00:00",
  "metric": "sharpe",
  "mode": "max",
  "summary": {
    "best_metric": 1.5,
    "success_rate": 1.0,
    "total_runs": 10
  },
  "timing": {"p50": 10.0, "p90": 20.0}
}
```

### Regression Check CLI

Compare current sweep results against baseline:

```bash
# Run regression comparison
python -m traderbot.cli.regress compare \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml \
    --out regression_report.md

# Exit code: 0 = passed, 1 = failed

# Use --no-emoji for CI environments or Windows cmd.exe
python -m traderbot.cli.regress compare \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml \
    --no-emoji
```

You can also set the `TRADERBOT_NO_EMOJI` environment variable to disable emoji globally:

```bash
export TRADERBOT_NO_EMOJI=1
python -m traderbot.cli.regress compare ...
```

Update baseline after verified improvements:

```bash
python -m traderbot.cli.regress update-baseline \
    --current runs/sweeps/ci_smoke \
    --out benchmarks/baseline.json \
    --sha $(git rev-parse --short HEAD)
```

### Determinism Check

Verify reproducibility by rerunning the best configuration:

```bash
# Rerun best config 3 times after sweep
python -m traderbot.cli.sweep sweeps/ci_smoke.yaml --workers 2 --rerun-best 3

# Output: determinism.json with:
# - max_abs_diff: Maximum difference across runs
# - is_deterministic: True if diff < 1e-9
```

### CI Integration

**PR Workflow:**
- Runs sweep and compares against baseline
- Posts regression status as PR comment
- Fails PR if regression detected

**Nightly Workflow:**
- Runs regression check with continue-on-error
- Runs determinism check (1 rerun)
- Uploads regression report as artifact
- Generates baseline candidate on manual dispatch

### Data Provenance

When CSV files (`leaderboard.csv`, `timings.csv`) are missing, the regression CLI automatically
derives metrics from `all_results.json`. A "Data Provenance" section is added to the report
indicating which metrics were computed from fallback sources:

```markdown
### Data Provenance

- `best_metric` derived from `all_results.json` (leaderboard.csv missing or empty)
- Timing percentiles (P50/P90) derived from `all_results.json` (timings.csv missing)
```

This ensures regression checks work even with minimal sweep outputs.

### Outputs

Each regression check produces:
- `regression_report.md` - Human-readable markdown report
- `baseline_diff.json` - JSON with computed deltas
- `determinism.json` - Reproducibility verification results

### Phase 5.2 Verification

```bash
# Run sweep
python -m traderbot.cli.sweep sweeps/ci_smoke.yaml --workers 2 --time

# Generate leaderboard
python -m traderbot.cli.leaderboard runs/sweeps/ci_smoke

# Run regression check
python -m traderbot.cli.regress compare \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml

# Run determinism check
python -m traderbot.cli.sweep sweeps/ci_smoke.yaml --workers 1 --rerun-best 2
```

For more details, see [PHASE52_SUMMARY.md](PHASE52_SUMMARY.md).

---

## Phase 5.3+: Advanced Regression Features

Phase 5.3+ adds advanced regression detection, variance analysis, and reporting:

### Regression Check in 60 Seconds

Get started with regression detection in under a minute:

```bash
# 1. Generate sample data (skip if already done)
python scripts/make_sample_data.py

# 2. Run a quick sweep
python -m traderbot.cli.sweep sweeps/ci_smoke.yaml --workers 2

# 3. Generate leaderboard
python -m traderbot.cli.leaderboard runs/sweeps/ci_smoke

# 4. Run regression check
python -m traderbot.cli.regress compare \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml

# Exit code: 0 = PASS, 1 = FAIL
```

### Per-Metric Tolerance (Reducing Flaky CI)

Configure separate tolerances for metrics and timing to absorb normal variance:

```yaml
# sweeps/perf_budget.yaml
epsilon_metric: 0.01     # Absorb Sharpe variations up to 0.01
epsilon_timing: 2.0      # Absorb timing variations up to 2 seconds
```

### Quiet Mode

Suppress console output while still writing reports:

```bash
python -m traderbot.cli.regress compare \
    --quiet \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml

# Exit code still reflects pass/fail
# Reports are still written to disk
```

### Variance Analysis (Flakiness Detection)

Analyze result variance to detect non-deterministic configurations:

```bash
python -m traderbot.cli.regress compare \
    --reruns 3 \
    --variance-threshold 0.10 \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml

# Generates variance_report.json with:
# - Coefficient of variation (CV) for each configuration
# - Flaky flag if CV > threshold
```

### HTML Reports

Generate styled HTML reports with pass/fail badges:

```bash
python -m traderbot.cli.regress compare \
    --html \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml

# Generates:
# - regression_report.md (markdown)
# - regression_report.html (styled HTML)
# - baseline_diff.json (deltas)
# - provenance.json (data source tracking)
```

### Auto-Update Baseline on Pass

Automatically update baseline when regression check passes:

```bash
python -m traderbot.cli.regress compare \
    --auto-update-on-pass benchmarks/baseline.json \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml

# On PASS: Updates baseline with current metrics
# On FAIL: No changes made
```

### Trend Plots (Nightly Tracking)

The nightly workflow generates trend plots tracking metrics over time:

```bash
# Generate trend plots manually
python scripts/generate_trend_plots.py \
    --trend-file runs/sweeps/trend_data.json \
    --current-diff runs/sweeps/ci_smoke/baseline_diff.json \
    --output-dir runs/sweeps/ci_smoke/plots

# Generates:
# - trend_metric.png (best Sharpe over time)
# - trend_success_rate.png (success rate over time)
# - trend_timing.png (P90 timing over time)
# - trend_dashboard.png (combined 4-panel dashboard)
```

### Full CLI Reference

```bash
python -m traderbot.cli.regress compare --help

# Key flags:
#   --no-emoji              Disable emoji (Windows/CI)
#   --quiet, -q             Suppress console output
#   --html                  Generate HTML report
#   --reruns N              Analyze top N for variance
#   --variance-threshold σ  CV threshold for flakiness (default: 0.1)
#   --auto-update-on-pass   Auto-update baseline on success
```

### Nightly Workflow Outputs

The nightly sweep produces:
- `regression_report.md` - Markdown summary
- `regression_report.html` - Styled HTML with badge
- `baseline_diff.json` - Computed deltas
- `provenance.json` - Data source tracking
- `variance_report.json` - Flakiness analysis
- `plots/` - Trend visualizations
- `trend_data.json` - Rolling 90-day history

---

## Phase 5.4: Per-Metric Budgets & Status Badges (v0.6.0)

Phase 5.4 adds granular multi-metric budget evaluation and status badge generation for CI dashboards.

### Per-Metric Budget Configuration

Define individual thresholds for multiple metrics in `sweeps/perf_budget.yaml`:

```yaml
# Multi-Metric Budget Map (v0.6.0+)
budgets:
  sharpe:
    mode: max           # Higher is better
    max_drop: 0.05      # Max 5% drop from baseline
    epsilon: 0.01       # Tolerance for fluctuations
  win_rate:
    mode: max
    min: 0.55           # Absolute minimum required
    epsilon: 0.005
  p90_elapsed_s:
    mode: min           # Lower is better
    max: 60             # Maximum allowed value
    epsilon: 2.0
  total_return:
    mode: max
    max_drop: 0.10
    epsilon: 0.02
    required: false     # Advisory only, won't fail CI
```

**Budget fields per metric:**
- `mode`: `max` (higher is better) or `min` (lower is better)
- `max_drop`: Maximum regression from baseline (comparison-based)
- `min`: Absolute minimum value required
- `max`: Absolute maximum value allowed
- `epsilon`: Tolerance for this specific metric
- `required`: If `false`, failure doesn't fail overall check (default: `true`)

### Backward Compatibility

The new `budgets:` map is backward-compatible with the existing single-metric format:

```yaml
# Legacy single-metric mode (still supported)
metric: sharpe
mode: max
max_sharpe_drop: 0.05
epsilon_metric: 0.01
```

When `budgets:` is present, each metric is evaluated independently. Overall PASS requires ALL required metrics to pass.

### Per-Metric Report Output

Regression reports now include a per-metric verdicts table:

```markdown
### Per-Metric Budget Verdicts

| Metric | Status | Current | Baseline | Delta | Threshold | Mode |
|--------|--------|---------|----------|-------|-----------|------|
| sharpe | PASS | 1.52 | 1.50 | +0.02 | 0.05 | max |
| win_rate | PASS | 0.58 | N/A | N/A | 0.55 | max |
| p90_elapsed_s | FAIL | 65.3 | N/A | N/A | 60.0 | min |
```

### Status Badge Generation

Generate deterministic SVG badges for CI dashboards:

```bash
# Generate badge from regression diff
python scripts/generate_status_badge.py \
    --from-diff runs/sweeps/ci_smoke/baseline_diff.json \
    --output badges/regression_status.svg \
    --sha $(git rev-parse --short HEAD)

# Generate badge with explicit status
python scripts/generate_status_badge.py \
    --status pass \
    --output badges/regression_status.svg
```

**Badge features:**
- Shields.io-style SVG (deterministic, <10KB)
- Green (#28a745) for PASS, Red (#dc3545) for FAIL
- Embedded SHA and timestamp metadata
- No external dependencies or network calls

### CI Integration

Both CI workflows now generate and upload status badges:

**Nightly Sweep (`nightly-sweep.yml`):**
- Generates badge after regression check
- Uploads as `status-badge` artifact (30-day retention)

**PR Checks (`ci.yml`):**
- Generates badge in regression-check job
- Adds pass/fail badge to GitHub job summary
- Includes report summary in PR comments

### Example: Multi-Metric Workflow

```bash
# 1. Run sweep
python -m traderbot.cli.sweep sweeps/ci_smoke.yaml --workers 2

# 2. Generate leaderboard
python -m traderbot.cli.leaderboard runs/sweeps/ci_smoke

# 3. Run multi-metric regression check
python -m traderbot.cli.regress compare \
    --html \
    --current runs/sweeps/ci_smoke \
    --baseline benchmarks/baseline.json \
    --budget sweeps/perf_budget.yaml

# 4. Generate status badge
python scripts/generate_status_badge.py \
    --from-diff runs/sweeps/ci_smoke/baseline_diff.json \
    --output badges/regression_status.svg

# Review outputs:
# - regression_report.md (per-metric verdicts table)
# - regression_report.html (styled HTML)
# - badges/regression_status.svg (CI badge)
```

---

## CI & Badges

The repository uses GitHub Actions for continuous integration with automatic badge publishing:

- **CI workflow** (`ci.yml`): Runs lint, type-check, and tests on every push/PR. On main branch, commits the regression badge on PASS.
- **Nightly Sweep** (`nightly-sweep.yml`): Runs full hyperparameter sweep daily at 03:17 UTC. Commits badge on PASS.
- **Regression badge** (`badges/regression_status.svg`): Auto-updated on main branch when regression passes. Shows PASS (green) or FAIL (red).
- **Loop prevention**: Badge commits use `github-actions[bot]` author, which is excluded from triggering workflows via `if: github.actor != 'github-actions[bot]'`.
- **PR behavior**: On pull requests, badge is uploaded as artifact only (no commit). Badge appears in Job Summary.
- **First run handling**: If `badges/` directory or badge file doesn't exist, git will detect it as a new file and commit it.
- **Badge unchanged**: Commit step is skipped if badge content is identical (no diff).
- **Commit message**: Uses `[ci skip]` suffix as additional loop protection: `chore: update regression badge [ci skip]`.

---

## Reports (GitHub Pages)

Regression reports are automatically published to GitHub Pages on successful nightly sweeps:

- **Latest Report:** https://eyoair21.github.io/Trade_Bot/reports/latest/regression_report.html
- **All Reports Index:** https://eyoair21.github.io/Trade_Bot/reports/
- **Per-Run Reports:** https://eyoair21.github.io/Trade_Bot/reports/`<run-id>`/

**Report contents:**
- `regression_report.html` - Styled HTML report with per-metric verdicts
- `regression_report.md` - Markdown version for GitHub rendering
- `baseline_diff.json` - Computed deltas and pass/fail status
- `provenance.json` - Data source tracking

**Publishing behavior:**
- **Main branch + PASS:** Reports are deployed to GitHub Pages under `/reports/<run-id>/`; `/reports/latest/` is updated to point to the newest run.
- **Main branch + FAIL:** No Pages deployment; reports are available as workflow artifacts only.
- **Pull requests:** No Pages deployment; reports uploaded as artifacts with preview note in Job Summary.

**Setup (first time):**
1. Run `scripts/dev/init_gh_pages.sh` to create the `gh-pages` branch
2. Push the branch: `git push -u origin gh-pages`
3. Enable GitHub Pages in repository settings: Settings → Pages → Source: `gh-pages` branch

---

## Integrity & Provenance

Reports include integrity verification and provenance tracking for auditability:

### SHA256 Integrity Hashes

Each report directory includes `sha256sums.txt` for verifying file integrity:

```bash
# Verify report integrity
cd /path/to/reports/12345-abc1234
sha256sum -c sha256sums.txt

# Generate hashes for a report directory
python scripts/dev/generate_sha256sums.py --report-dir runs/sweeps/ci_smoke

# Verify existing hashes
python scripts/dev/generate_sha256sums.py --report-dir runs/sweeps/ci_smoke --verify
```

**Hashed files:**
- `regression_report.html`
- `regression_report.md`
- `baseline_diff.json`
- `provenance.json`
- `plots/*.png` (if present)

### Provenance Schema (v1)

The `provenance.json` file documents data sources and build context:

```json
{
  "schema_version": "1",
  "git_sha": "abc1234",
  "generated_utc": "2026-01-05T12:00:00+00:00",
  "data_sources": {
    "leaderboard": "leaderboard.csv",
    "timings": "timings.csv"
  },
  "fallbacks_used": {
    "leaderboard": false,
    "timings": false
  },
  "notes": []
}
```

When CSV files are unavailable, metrics are derived from `all_results.json` and `fallbacks_used` flags are set to `true`.

### Report UX Features

HTML reports include user experience enhancements:

- **Dark/Light toggle:** Click "Toggle Dark/Light" button or use system preference
- **Sticky header:** Navigation stays visible while scrolling
- **Provenance footer:** Shows build SHA and timestamp at bottom of report
- **Cache-busting:** Latest report links include query parameter for fresh loads

### Prune Policy

To manage storage, the deployment keeps the 50 most recent reports:

- **Manifest trimming:** `update_pages_index.py` removes old entries from `manifest.json`
- **Directory pruning:** With `--prune` flag, old report directories are deleted
- **404 handling:** Custom 404.html directs users to available reports

```bash
# Manual pruning (dry run)
python scripts/dev/update_pages_index.py \
    --manifest public/reports/manifest.json \
    --max-runs 50 \
    --prune \
    --dry-run
```

---

## Report Polish II (v0.6.4)

Version 0.6.4 adds enhanced report presentation and per-run metadata:

### PASS/FAIL Status Banner

HTML reports now display a prominent status banner at the top:

- **Sticky placement:** Fixed below header, visible during scrolling
- **Color-coded:** Green (#28a745) for PASS, red (#dc3545) for FAIL
- **Summary stats:** Shows Sharpe Δ, P90 timing, and total runs
- **Theme-aware:** Respects dark/light mode from v0.6.3

### Per-Run Summary Card

Each report includes `summary.json` with key metrics:

```json
{
  "schema_version": "1",
  "run_id": "12345-abc1234",
  "verdict": "PASS",
  "sharpe_delta": 0.0234,
  "trades_delta": null,
  "timing_p90": 12.5,
  "git_sha": "abc123def456",
  "generated_utc": "2026-01-05T12:00:00+00:00"
}
```

This enables quick status inspection without parsing full reports.

### Enriched Index Listing

The GitHub Pages index now displays summary stats for each run:

- **Sharpe Δ:** Color-coded positive (green) or negative (red)
- **P90 timing:** Execution time at 90th percentile
- **Git SHA:** Short commit hash for traceability

### HTML Minification

Public HTML files are minified to reduce bandwidth:

```bash
# Minify HTML files in a directory
python scripts/dev/minify_html.py --input-dir public/reports/latest --pattern "*.html"

# Preview minification (dry run)
python scripts/dev/minify_html.py --input public/reports/latest/regression_report.html --dry-run
```

**Minification features:**
- Removes unnecessary whitespace between tags
- Strips HTML comments (preserves conditional comments)
- Preserves `<pre>`, `<script>`, `<style>`, and `<textarea>` content
- Zero external dependencies (uses stdlib HTMLParser)
- Typically achieves 10-30% file size reduction

### Updated Integrity Hashes

The `sha256sums.txt` file now includes `summary.json`:

```bash
# Files included in integrity check
- regression_report.html
- regression_report.md
- baseline_diff.json
- provenance.json
- summary.json  # New in v0.6.4
- plots/*.png
```

---

## Insights & Trends (v0.6.5)

Version 0.6.5 adds historical analysis, trend visualization, and subscription feeds:

### Run History Aggregation

Build `history.json` from all `summary.json` files in report directories:

```bash
# Generate history.json with rolling statistics
python scripts/dev/update_pages_index.py \
    --manifest public/reports/manifest.json \
    --reports-dir public/reports \
    --index-out public/reports/index.html \
    --history-out public/reports/history.json
```

**History schema (v1):**
```json
{
  "schema_version": 1,
  "generated_utc": "2026-01-05T12:00:00Z",
  "window": 5,
  "runs": [...],
  "rolling": {
    "sharpe_delta": {"mean": 0.018, "stdev": 0.032},
    "timing_p90": {"mean": 12.5, "stdev": 1.2}
  }
}
```

### Trend Sparklines

The index page displays inline SVG sparklines for Sharpe Delta and P90 timing trends:

- **5-run rolling window:** Shows recent trend direction
- **Accessible:** ARIA labels and title tooltips
- **Theme-aware:** Uses CSS custom properties for color

### Flakiness Detection

Automatic detection of unstable metrics across recent runs:

```
Rule: "flaky" if |Sharpe Δ| mean < 0.02 AND stdev >= 2×|mean| over last 10 runs
Guard: Requires minimum 6 runs with valid data
```

When flakiness is detected, a "FLAKY" pill appears next to the header in the index.

### Atom Feed

Subscribe to regression results via RSS/Atom feed:

```bash
# Generate feed from history.json
python scripts/dev/make_feed.py \
    --history public/reports/history.json \
    --out public/reports/feed.xml \
    --base-url "https://your-org.github.io/repo" \
    --max-entries 20
```

**Feed URL:** https://eyoair21.github.io/Trade_Bot/reports/feed.xml

Each entry includes:
- Title: PASS/FAIL status with run ID
- Summary: Sharpe Δ, P90 timing, git SHA
- Link: Direct to HTML report
- Categories: pass/fail for filtering

### Search & Filter UI

The index page includes client-side search and filtering:

- **Search:** Filter runs by ID, date, or any text
- **Verdict filter:** Show All / PASS only / FAIL only
- **URL state:** Filters persist in URL hash for shareability

### Reports Summarize CLI

Analyze historical trends from the command line:

```bash
# Text summary of last 30 days
python -m traderbot.cli.reports summarize \
    --history reports/history.json \
    --since 2026-01-01 \
    --limit 30

# JSON output for scripting
python -m traderbot.cli.reports summarize \
    --history reports/history.json \
    --format json \
    --out summary.json

# CSV export
python -m traderbot.cli.reports summarize \
    --history reports/history.json \
    --format csv
```

**Output includes:**
- Total runs, pass/fail counts, pass rate
- Sharpe delta mean/stdev
- Timing P90 mean/stdev
- Flakiness analysis
- Recent runs table

### Updated Integrity Hashes

Top-level files are now included in integrity verification:

```bash
# Generate top-level checksums
python scripts/dev/generate_sha256sums.py \
    --report-dir public/reports \
    --top-level

# Files included:
# - index.html
# - history.json
# - feed.xml
# - 404.html
```

---

## Troubleshooting

### Windows Console Unicode Issues

If the console crashes or displays garbled characters with Unicode/emoji:

1. **Use `--no-emoji` flag:**
   ```bash
   python -m traderbot.cli.regress compare --no-emoji ...
   ```

2. **Set environment variable globally:**
   ```bash
   set TRADERBOT_NO_EMOJI=1
   python -m traderbot.cli.regress compare ...
   ```

3. **Use Windows Terminal** (recommended):
   - Windows Terminal handles UTF-8 better than cmd.exe
   - Settings → Profiles → Defaults → Enable "Use Unicode UTF-8 for worldwide language support"

4. **Set console encoding:**
   ```bash
   chcp 65001
   python -m traderbot.cli.regress compare ...
   ```

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'traderbot'` | Ensure you're in the project root or install with `pip install -e .` |
| Unicode errors on Windows | Use `--no-emoji` or set `TRADERBOT_NO_EMOJI=1` |
| Tests fail with subprocess errors | Check that PYTHONPATH includes the project root |
| Coverage below 70% | Run `pytest --cov-report=html` to identify untested code |
| Regression check passes but CI shows failure | Check `epsilon_metric` and `epsilon_timing` values in budget |
| Variance analysis shows all configs as flaky | Lower `--variance-threshold` or increase sample size |
| HTML report not generated | Ensure `--html` flag is passed to compare command |
| Trend plots not appearing | Install matplotlib: `pip install matplotlib` |
| baseline_diff.json missing | Run `regress compare` first; always generated |
| provenance.json missing data | Check that sweep outputs exist in current directory |

### Regression Check Failures

**"Metric regression detected":**
- Compare current best metric vs baseline in `baseline_diff.json`
- Check if drop exceeds `max_sharpe_drop` threshold
- Consider tuning `epsilon_metric` for normal variance absorption

**"Success rate below threshold":**
- Review `all_results.json` for failed runs
- Check error logs in individual run directories
- May indicate broken configurations in sweep grid

**"Timing P90 exceeded budget":**
- Review `timings.csv` for slow runs
- Consider increasing `max_p90_elapsed_s` or optimizing slow configs
- Check CI runner load; `epsilon_timing` absorbs runner variance

### Flaky Results

If you see inconsistent pass/fail between runs:

1. **Enable variance analysis:**
   ```bash
   python -m traderbot.cli.regress compare \
       --reruns 3 \
       --variance-threshold 0.10 \
       ...
   ```

2. **Check variance_report.json:**
   - High CV (>0.1) indicates non-determinism
   - Review flagged configurations

3. **Verify seed usage:**
   - Ensure `--seed` is passed to walkforward
   - Check that all random sources are seeded

4. **Tune epsilons:**
   ```yaml
   epsilon_metric: 0.02   # Increase if small fluctuations cause failures
   epsilon_timing: 5.0    # Increase for slow CI runners
   ```

### Matplotlib Issues

If trend plots fail to generate:

```bash
# Install matplotlib
pip install matplotlib

# Test matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

For headless environments (CI):
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

---

## Universe Screening, Ranking & `tb` CLI

Phase 6 adds a thin orchestration layer on top of the existing walk-forward engine:

- **Named universes** defined in YAML under `universe/`:
  - `universe/sp500.yaml` – S&P 500–style large-cap screen
  - `universe/liquid_top1000.yaml` – market-wide liquidity screen
- **Factor library** in `traderbot.features.factors`:
  - Trend (20/50/200), momentum (3–12m with 1m skip), mean reversion (RSI2, z-score vs SMA20)
  - Risk (ATR%) and cost (spread bps)
- **Composite score & ranking** with configurable weights and sector caps
- **High-level CLI** entrypoint `tb` (see `pyproject.toml`).

### Picking a Universe & Ranking

```bash
# Install CLI entrypoint (editable install from repo root)
pip install -e .

# Rank S&P 500-style universe and keep top 25 names with 20% sector cap
tb backtest \
  --universe sp500 \
  --strategy trend \
  --top-n 25 \
  --sector-cap 0.2
```

This will:

- Screen candidates using the rules in `universe/sp500.yaml`:
  - Price ≥ $5, ADV ≥ $20M, spread ≤ 0.2%, ≥ ~3y of history
- Load OHLCV data from `DATA_DIR/OHLCV_DIR` (see `config.py`)
- Compute standardized factor z-scores and a composite score with default weights:
  - Trend 0.35, Momentum 0.25, MeanRev 0.20, Quality 0.10, Cost –0.10
- Apply **sector caps** (default max 20% per sector if `sector_map.csv` is present)
- Write selection metadata to `reports/backtests/last_selection.json`

You can override weights via JSON:

```bash
tb backtest \
  --universe liquid_top1000 \
  --strategy trend \
  --top-n 100 \
  --sector-cap 0.2 \
  --weights '{"trend":0.35,"momentum":0.25,"meanrev":0.20,"quality":0.10,"cost":-0.10}'
```

### Backtest Grid & Comparison

When `--start-date/--end-date` are omitted, `tb backtest` runs a small grid over:

- Periods: `2020-01-01..2022-12-31`, `2023`, `2024`
- Selected universe (e.g., `sp500`) and strategy label (`trend`, `meanrev`, `hybrid`)

Artifacts are written under `reports/backtests/`:

- Per-run:
  - `*/results.json`, `*/equity_curve.csv`, `*/metrics.json`
- Aggregated:
  - `reports/backtests/comparison.csv` – one row per configuration
  - `reports/backtests/summary.md` – top-3 configs ranked by Sharpe

View a ranked table across runs:

```bash
tb compare
```

This prints a Sharpe-ranked table of universes/strategies/periods from `comparison.csv`.

### Paper Trading (Dry-Run Wiring)

The `tb` CLI includes a simple dry-run paper “broker” that reads the last backtest selection and
logs expected vs. filled prices using a configurable slippage model (no real orders are sent):

```bash
# 1. Run a backtest to generate universe selection + reports
tb backtest --universe sp500 --strategy trend --top-n 25 --sector-cap 0.2

# 2. Initialize paper account from latest selection
tb paper-start

# 3. Simulate a sync cycle (uses latest OHLCV close and EXECUTION_SLIPPAGE_BPS)
tb paper-sync

# 4. Inspect paper account status
tb paper-status
```

Paper-trade state is stored under `reports/backtests/paper_state.json` and includes:

- `account` – paper equity/cash snapshot
- `universe` – symbols from the last selection
- `fills` – synthetic fills with `expected_price`, `filled_price`, and `slippage_bps`

This wiring is intentionally **secret-free** and can be extended later to hit a real
broker (e.g., Alpaca) by swapping out the backing implementation.

---

## Phase 6 – News, Gaps, Alerts, Daily Jobs, Paper v1

Phase 6 adds news-based sentiment analysis, gap detection, automated daily scans, and a deterministic paper trading engine.

### First Run (No Data)

If you have a fresh clone with no OHLCV parquet files, you can seed minimal data and run a full
scan + paper demo with:

```bash
python scripts/seed_ohlcv.py

tb scan --universe sp500 --strategy trend --top-n 25 --sector-cap 0.2

tb paper-start
tb paper-sync
tb paper-status

# Publish reports locally (Linux/macOS)
rsync -a reports/ public/reports/
```

On Windows PowerShell, the last step can be approximated with:

```powershell
New-Item -ItemType Directory -Force -Path public\reports | Out-Null
Copy-Item -Path reports\* -Destination public\reports\ -Recurse -Force
```

### Quick Start (No API Keys Required)

```bash
# Install with editable mode
pip install -e .

# Run the full daily pipeline
# On Linux/macOS:
./scripts/run_daily_scan.sh

# On Windows (PowerShell):
.\scripts\run_daily_scan.ps1
```

The pipeline runs these steps automatically:
1. **news-pull** – Fetch headlines from public RSS feeds (Yahoo Finance, MarketWatch, CNBC, Reuters, SEC EDGAR)
2. **news-parse** – Extract tickers and deduplicate content
3. **news-score** – Apply lexicon-based sentiment with time decay
4. **sector-digest** – Aggregate sentiment by sector
5. **scan** – Generate opportunity rankings with all factors

### CLI Commands

```bash
# Individual news pipeline steps
tb news-pull                    # Fetch from RSS feeds
tb news-parse                   # Parse and extract tickers
tb news-score                   # Score sentiment with decay
tb sector-digest --window 1d    # Build sector heatmap

# Run opportunity scan with all factors
tb scan --universe sp500 --top-n 25 --sector-cap 0.2
```

### Artifacts Location

All outputs are written to `public/reports/` for GitHub Pages publishing:

```
public/reports/
├── YYYY-MM-DD/
│   ├── opportunities.csv       # Ranked opportunities with scores
│   ├── alerts.html            # Interactive HTML preview
│   ├── alerts.md              # Markdown summary
│   └── sector_sentiment.csv   # Sector aggregates
└── latest/ -> YYYY-MM-DD/     # Symlink to most recent
```

### Factor Weights

The composite score combines multiple factors:

| Factor   | Weight | Description                              |
|----------|--------|------------------------------------------|
| Trend    | 0.35   | 20/50/200 SMA alignment                  |
| Momentum | 0.25   | 3-12 month returns (1m skip)             |
| MeanRev  | 0.15   | RSI2 extremes, z-score vs SMA20          |
| Sentiment| 0.10   | News sentiment with 6-hour half-life     |
| Sector   | 0.05   | Sector-level sentiment z-score           |
| Gap      | 0.10   | Gap regime (CONT/REVERT/NEUTRAL)         |
| Cost     | -0.10  | Spread penalty                           |

### Gap Detection

Gap regimes are classified based on gap size relative to ATR:

- **CONTINUATION (CONT)**: `|gap| ≤ 0.5×ATR` or gap aligned with prior trend
- **REVERSION (REVERT)**: `|gap| > 1.0×ATR` against prior trend
- **NEUTRAL**: Intermediate cases

### Paper Trading Engine

The paper engine provides deterministic simulation:

```bash
# Start paper trading session
tb paper-start --capital 100000

# Submit orders (simulated)
tb paper-sync

# View positions and PnL
tb paper-status
```

Features:
- **Volatility-targeted sizing**: Position weights based on `target_vol / asset_vol`
- **Deterministic fills**: Configurable slippage (default 10 bps) and costs (2 bps)
- **Equity tracking**: CSV export of equity curve and fills

### Options Module (Placeholder)

Data structures are defined for future options integration:

```python
from traderbot.options import OptionQuote, OptionChainSummary, IVSnapshot

# Stubs ready for Polygon/Tradier API integration
chain = load_option_chain_sample("AAPL")
metrics = compute_basic_metrics(chain)
```

### Legal Note

This software is for **educational and paper trading purposes only**. No real money is at risk.
The authors are not financial advisors. Past performance does not guarantee future results.
Always do your own research before making investment decisions.

---

## Publishing (GitHub Pages)

Reports are automatically published to GitHub Pages on push to `main` or via manual workflow dispatch.

### Prerequisites

Ensure the following repository settings are configured:

1. **Settings → Actions → Workflow permissions**: Set to "Read and write permissions"
2. **Settings → Pages → Source**: Set to "GitHub Actions"
3. **Settings → Environments → github-pages** (if required by org): Ensure `main` branch is allowed

### Workflow

The `Pages` workflow (`.github/workflows/pages.yml`) runs automatically on:
- Push to `main` branch
- Manual dispatch via Actions UI

The workflow:
1. **Build job**: Prepares `public/` directory with `index.html` and copies `reports/` into `public/reports/`
2. **Deploy job**: Deploys the `public/` artifact to GitHub Pages

### Artifacts

The workflow expects artifacts in the `public/` directory:
- `public/index.html` - Landing page (auto-created if missing)
- `public/reports/` - Report artifacts (copied from `reports/` if present)

### Smoke Test

After deployment completes (typically 2-5 minutes), verify Pages is live:

```powershell
$base = "https://eyoair21.github.io/Trade_Bot"
$eps  = @("", "reports/")
foreach ($e in $eps) {
  $u = "$base/$e" -replace "//","/"
  try { $c = (Invoke-WebRequest -UseBasicParsing -Uri $u -Method Head -TimeoutSec 20).StatusCode.value__ }
  catch { $c = "ERR" }
  "$u -> $c"
}
```

**Expected**: Both endpoints return `200`.

### Troubleshooting

If Pages doesn't deploy:
1. Check Actions workflow: https://github.com/eyoair21/Trade_Bot/actions
2. Review "Pages" workflow logs for errors
3. Verify `public/` directory contains files (check "Upload artifact" step)
4. Ensure `github-pages` environment exists and allows `main` branch deployments

---

## License

MIT License - See LICENSE file for details.
