# Local Run Guide

**Quick setup and execution instructions for x86_64 and Raspberry Pi (ARM64).**

---

## Prerequisites

### x86_64 (Linux/macOS/Windows)

- **Python**: 3.11+ (verify: `python --version`)
- **Poetry**: Latest (install: `pip install poetry` or `curl -sSL https://install.python-poetry.org | python3 -`)
- **Git**: For cloning repository

### Raspberry Pi (ARM64)

- **Python**: 3.11+ (Raspberry Pi OS typically includes 3.11)
- **Poetry**: Install via pip (see below)
- **PyTorch**: ARM64 wheels available (see ARM64 section)

---

## Quick Start (x86_64)

### Option 1: Using Make (Recommended)

```bash
# 1. Clone repository
git clone <repo-url>
cd traderbot

# 2. Setup environment and install dependencies
make install

# 3. Generate sample data (cached locally)
make data

# 4. Run quick backtest demo (1-3 minutes)
make backtest

# 5. View results
ls -la runs/
cat runs/*/results.json | jq '.avg_oos_return_pct'
```

### Option 2: Using Poetry Directly

```bash
# 1. Install dependencies
poetry install

# 2. Generate sample data
poetry run python scripts/make_sample_data.py

# 3. Run walk-forward demo
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA \
  --n-splits 3 \
  --is-ratio 0.6
```

---

## Detailed Setup

### Step 1: Environment Setup

#### Create Virtual Environment (Optional, if not using Poetry)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

#### Install Dependencies

```bash
# Using Poetry (recommended)
poetry install

# Or using pip (if Poetry not available)
pip install -e .
pip install pytest pytest-cov ruff black mypy pandas-stubs
```

### Step 2: Configuration

#### Create .env File

```bash
# Copy example (if exists)
cp .env.example .env

# Or create manually with minimal config:
cat > .env << EOF
DATA_DIR=./data
OHLCV_DIR=./data/ohlcv
LOG_LEVEL=INFO
DEFAULT_INITIAL_CAPITAL=100000.0
DEFAULT_COMMISSION_BPS=10.0
DEFAULT_SLIPPAGE_BPS=5.0
RANDOM_SEED=42
EOF
```

**Note**: If `.env.example` doesn't exist yet, see `CHECKLIST.md` P0-1.

### Step 3: Generate Sample Data

```bash
# Generate synthetic OHLCV data for AAPL, MSFT, NVDA
poetry run python scripts/make_sample_data.py

# Or with custom options:
poetry run python scripts/make_sample_data.py \
  --tickers AAPL MSFT NVDA GOOG \
  --start-date 2023-01-01 \
  --end-date 2023-03-31 \
  --seed 42
```

**Output**: Parquet files in `data/ohlcv/{TICKER}.parquet`

### Step 4: Run Backtest

#### Quick Demo (Fast)

```bash
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA \
  --n-splits 3 \
  --is-ratio 0.6 \
  --universe-mode static
```

#### With Model Training

```bash
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA \
  --n-splits 3 \
  --is-ratio 0.6 \
  --train-per-split \
  --epochs 30 \
  --batch-size 16
```

#### With Position Sizing

```bash
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA \
  --n-splits 3 \
  --is-ratio 0.6 \
  --sizer vol \
  --vol-target 0.15 \
  --proba-threshold 0.55
```

### Step 5: View Results

```bash
# Results are saved to runs/{timestamp}/
ls -la runs/

# View summary
cat runs/*/results.json | jq '{
  avg_oos_return: .avg_oos_return_pct,
  avg_oos_sharpe: .avg_oos_sharpe,
  total_trades: .total_oos_trades
}'

# View equity curve
head -20 runs/*/equity_curve.csv

# View markdown report
cat runs/*/report.md
```

---

## Raspberry Pi (ARM64) Setup

### Prerequisites

- **Raspberry Pi OS** (Debian-based) or **Ubuntu for ARM64**
- **Python 3.11+**: Usually pre-installed, verify: `python3 --version`
- **pip**: `sudo apt-get install python3-pip`

### Step 1: Install Poetry

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add to ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc

# Verify
poetry --version
```

### Step 2: Install Dependencies

```bash
# Clone repository
git clone <repo-url>
cd traderbot

# Install dependencies (Poetry handles ARM64 wheels)
poetry install

# If PyTorch is needed (for PatchTST model):
# Note: PyTorch ARM64 wheels are available but may be slower
poetry add torch --optional
```

**Note**: PyTorch on ARM64:
- Official wheels available for ARM64 (aarch64)
- May be slower than x86_64
- Consider skipping model training if performance is an issue

### Step 3: Native Install (Fallback if Poetry Fails)

```bash
# Install core dependencies
pip3 install pandas numpy pyarrow python-dotenv

# Install dev dependencies (optional)
pip3 install pytest pytest-cov ruff black mypy

# Install PyTorch (if needed, ARM64 wheels)
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Install project
pip3 install -e .
```

### Step 4: Generate Data & Run

```bash
# Same as x86_64 instructions
poetry run python scripts/make_sample_data.py
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA \
  --n-splits 3 \
  --is-ratio 0.6
```

### Known Issues (ARM64)

1. **PyTorch Performance**: Slower on ARM64, consider:
   - Skip model training (`--train-per-split` not used)
   - Use smaller batch sizes if training
   - Consider CPU-only builds

2. **Memory**: Raspberry Pi 4 (4GB) should be sufficient for small backtests
   - Reduce `--n-splits` if memory issues
   - Use smaller universe (fewer tickers)

3. **Disk Space**: Ensure sufficient space for data and runs
   - Parquet files: ~1-5 MB per ticker
   - Run artifacts: ~10-50 MB per run

---

## Docker Setup (If Available)

**Note**: Docker support is planned (see `CHECKLIST.md` P1-3). Once implemented:

```bash
# Build image
docker-compose build

# Run backtest
docker-compose run traderbot make backtest

# View results (mounted volume)
ls -la runs/
```

---

## Troubleshooting

### Issue: "No module named 'traderbot'"

**Solution:**
```bash
# Ensure you're in the repo root
cd traderbot

# Install in development mode
poetry install
# or
pip install -e .
```

### Issue: "FileNotFoundError: No data file for ticker"

**Solution:**
```bash
# Generate sample data first
poetry run python scripts/make_sample_data.py
```

### Issue: "PyTorch not available" (ARM64)

**Solution:**
```bash
# Skip model training, or install PyTorch ARM64 wheels
poetry add torch --optional
# or
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Memory errors on Raspberry Pi

**Solution:**
- Reduce `--n-splits` (e.g., `--n-splits 2`)
- Use smaller universe (e.g., `--universe AAPL MSFT`)
- Skip model training

### Issue: Slow performance

**Solution:**
- Use synthetic data only (`--synthetic-only` in `make_sample_data.py`)
- Reduce date range
- Skip model training
- Use fewer splits

---

## One-Minute Demo

**Fastest path to see results:**

```bash
# 1. Setup (one-time)
poetry install
poetry run python scripts/make_sample_data.py --synthetic-only

# 2. Run (1-3 minutes)
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-02-28 \
  --universe AAPL MSFT \
  --n-splits 2 \
  --is-ratio 0.6

# 3. View results
cat runs/*/results.json | jq '.avg_oos_return_pct'
```

---

## Profit-Aware Learning Features

The bot now includes profit-aware learning capabilities. See `PROFIT_AWARE_LEARNING.md` for full details.

### Quick Examples

#### 1. Run with Dynamic Universe + Leakage Controls

```bash
make backtest ARGS="--universe-mode dynamic --top-n 50 --embargo-days 5 --purge 1d"
```

Features:
- Dynamic universe selection based on liquidity
- Purged+embargo splits to prevent information leakage
- Outputs universe snapshots to `runs/<timestamp>/universe_*.json`

#### 2. Run with Position Sizing Policies

```bash
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA \
  --sizer vol \
  --vol-target 0.15 \
  --max-leverage 1.0
```

Sizing modes:
- `fixed` - Fixed fractional (10% per position)
- `vol` - Target volatility (scale to hit target vol)
- `kelly` - Kelly criterion with clipping

#### 3. Run with Bandit Multi-Strategy Allocation

```bash
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA \
  --alloc bandit
```

- Thompson Sampling across multiple strategies
- Adaptive weights saved to `runs/<timestamp>/alloc.csv`

#### 4. Run GA Hyperparameter Optimization

```bash
make opt ARGS="--budget 30 --oos-metric total_reward --top-n 50"
```

- Genetic algorithm search over hyperparameters
- Best config saved to `runs/ga_opt/best_config.json`
- Replay script: `runs/ga_opt/replay.sh`

#### 5. Generate Report with Plots

```bash
make report
```

- Generates markdown report with:
  - PnL and reward attribution
  - Sharpe/Sortino ratios
  - Trading statistics
  - Risk breaches
  - Equity curve and drawdown plots

### Environment Variables for Profit-Aware Features

Add to `.env`:

```bash
# Reward function weights
REWARD_LAMBDA_DD=0.2          # Drawdown penalty
REWARD_TAU_TURNOVER=0.001     # Turnover penalty
REWARD_KAPPA_BREACH=0.5       # Risk breach penalty

# Risk guards
MAX_DRAWDOWN_PCT=0.15         # 15% max drawdown
DAILY_LOSS_LIMIT_PCT=0.02     # 2% daily loss limit

# Position sizing
SIZING_SIZER=target_vol       # fixed|vol|kelly
SIZING_VOL_TARGET=0.15        # Target 15% portfolio vol
SIZING_KELLY_CAP=0.25         # Max 25% Kelly allocation

# Universe selection
UNIVERSE_MAX_SYMBOLS=100
UNIVERSE_MIN_DOLLAR_VOLUME=1000000
UNIVERSE_MIN_PRICE=5.0
```

### Artifacts

After any run, check `runs/<timestamp>/`:

- `results.json` - Metrics including **total_reward**, pnl_net, sharpe
- `equity_curve.csv` - Daily equity and drawdown
- `orders.csv` - All trades with PnL and turnover
- `breaches.csv` - Risk breach events (if any)
- `splits.json` - Train/test split info
- `universe_<date>.json` - Universe snapshots (dynamic mode)
- `alloc.csv` - Strategy weights (bandit mode)
- `report/report.md` - Generated report

---

## Next Steps

- See `PROFIT_AWARE_LEARNING.md` for detailed profit-aware features
- See `REPORT.md` for detailed feature audit
- See `CHECKLIST.md` for prioritized development tasks
- See `README.md` for full documentation

