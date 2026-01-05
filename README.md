# TraderBot - Paper Trading Research Bot

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

## License

MIT License - See LICENSE file for details.
