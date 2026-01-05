# TraderBot Quick Start Guide (Windows)

This guide will help you get TraderBot up and running on Windows in minutes.

## Prerequisites

✅ **Python 3.11+** (You have Python 3.13.5 - Perfect!)  
✅ **pip** (You have pip 25.3 - Perfect!)

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Poetry (Dependency Manager)

```powershell
# Install Poetry
pip install poetry

# Verify installation
poetry --version
```

### Step 2: Install Project Dependencies

```powershell
# Navigate to project directory
cd E:\Trade_Bot\traderbot

# Install all dependencies
poetry install
```

This will install:
- Core dependencies: pandas, numpy, pyarrow, python-dotenv
- Dev tools: pytest, ruff, black, mypy

### Step 3: Verify Installation

```powershell
# Run smoke test
poetry run pytest tests/test_smoke.py -v
```

---

## 📊 Usage Examples

### Option A: Using Poetry (Recommended)

All commands below use `poetry run` to execute within the virtual environment.

#### 1️⃣ Generate Sample Data (if needed)

```powershell
cd E:\Trade_Bot\traderbot
poetry run python scripts/make_sample_data.py
```

**Output:** Creates `data/ohlcv/AAPL.parquet`, `MSFT.parquet`, `NVDA.parquet`

You already have sample data, so you can skip this step!

#### 2️⃣ Train PatchTST Model (CPU-only)

```powershell
# Basic training (20 epochs)
poetry run python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --runs-dir runs --epochs 20 --batch-size 64 --learning-rate 1e-4 --val-split 0.2 --seed 42 --lookback 32 --features close_ret_1,rsi_14,atr_14,vwap_gap,dvol_5,regime_vix
```

**Before running:** Install PyTorch if not already installed:

```powershell
pip install torch
```

**Outputs:**
- `models/patchtst.ts` - TorchScript model
- `runs/<timestamp>/train_patchtst.json` - Training metrics

#### 3️⃣ Walk-Forward Backtest (No Model)

```powershell
# Static universe mode
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --universe-mode static
```

**Outputs:** `runs/<timestamp>/`
- `results.json` - Summary metrics
- `equity_curve.csv` - Equity curve data
- `run_manifest.json` - Run metadata

#### 4️⃣ Walk-Forward with PatchTST + Dynamic Universe

```powershell
# Train model first (if not done)
poetry run python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --epochs 20 --batch-size 64

# Run backtest with model
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --universe-mode dynamic --model-path models/patchtst.ts
```

**Dynamic mode features:**
- Automatic symbol selection based on liquidity & volatility
- Writes universe selections to `runs/<timestamp>/universe_<date>.json`

---

### Option B: Using Direct Python (Without Poetry)

If you prefer to use Python directly:

#### Install Dependencies Manually

```powershell
pip install pandas numpy pyarrow python-dotenv torch
pip install pytest pytest-cov ruff black mypy
```

#### Run Commands Without `poetry run`

```powershell
# Train model
python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --epochs 20

# Walk-forward backtest
python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
```

---

## 🎯 Makefile Shortcuts (Requires Git Bash or WSL)

If you have Git Bash or WSL installed, you can use convenient Make targets:

```bash
# Install dependencies
make install

# Generate sample data
make data

# Train model
make train

# Run walk-forward demo (static)
make demo

# Run walk-forward demo (dynamic)
make demo2

# Run full test suite
make test

# Clean generated files
make clean
```

**Note:** Windows PowerShell doesn't support `make` by default. Use Git Bash or install WSL.

---

## 📁 Project Structure

```
traderbot/
├── data/ohlcv/              # OHLCV parquet files (AAPL, MSFT, NVDA)
├── models/                  # Trained models (patchtst.ts)
├── runs/                    # Backtest results with timestamps
├── scripts/
│   ├── make_sample_data.py  # Generate sample data
│   └── train_patchtst.py    # Train PatchTST model
├── traderbot/
│   ├── cli/                 # CLI commands
│   ├── data/                # Data adapters & universe selection
│   ├── engine/              # Backtesting engine & strategies
│   ├── features/            # Technical indicators & sentiment
│   └── model/               # PatchTST model architecture
└── tests/                   # Test suite
```

---

## 🔧 Configuration

### Model Features (Order Matters!)

The PatchTST model uses these features in this exact order:

| Feature | Description |
|---------|-------------|
| `close_ret_1` | 1-day return (clipped ±10%) |
| `rsi_14` | 14-day RSI |
| `atr_14` | 14-day ATR |
| `vwap_gap` | % gap from VWAP |
| `dvol_5` | Dollar volume ratio vs 5d mean |
| `regime_vix` | Volatility regime indicator |

### Universe Thresholds

Configure in `traderbot/config.py`:

```python
universe:
  max_symbols: 30
  min_dollar_volume: 20000000  # $20M avg 20d
  min_volatility: 0.15         # 15% annualized
  lookback_days: 20
```

---

## 📊 Sentiment & FinBERT (Optional)

### Default: Deterministic Fallback

By default, sentiment analysis uses a deterministic rule-based approach (reproducible).

### Enable FinBERT (Optional)

For advanced sentiment analysis:

```powershell
pip install transformers
```

TraderBot will automatically use `ProsusAI/finbert` if available.

---

## ✅ Verification Commands

### Run Full Test Suite

```powershell
cd E:\Trade_Bot\traderbot

# Lint check
poetry run ruff check .

# Format check
poetry run black --check .

# Type check
poetry run mypy traderbot

# Tests with coverage (≥70%)
poetry run pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q
```

### One-Liner Verification

```powershell
poetry install; poetry run pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q; poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
```

---

## 🎓 Next Steps (Quick Wins)

### 1. Train Once, Run Multiple Backtests

```powershell
# Train model (once)
poetry run python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --epochs 20

# Run multiple backtests with different parameters
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --model-path models/patchtst.ts

poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 5 --is-ratio 0.7 --model-path models/patchtst.ts
```

### 2. Tune Features

Edit `traderbot/config.py` to adjust model features:

```python
model:
  features:
    - close_ret_1
    - rsi_14
    - atr_14
    - vwap_gap
    - dvol_5
    - regime_vix
    # Add more features here
```

**Important:** Feature order must match training!

### 3. Adjust Universe Filters

Modify universe thresholds in `traderbot/config.py` to match your liquidity/volatility targets:

```python
universe:
  max_symbols: 50              # Increase max symbols
  min_dollar_volume: 50000000  # Higher liquidity threshold
  min_volatility: 0.20         # Higher volatility threshold
```

Then re-run with `--universe-mode dynamic`.

### 4. Add Model-Driven Signal Gating (Optional)

If you want explicit model influence in trades, you can add a threshold helper in the momentum strategy:

```python
# In traderbot/engine/strategy_momo.py
if prob_up > 0.55:
    # Long signal
    pass
else:
    # Flat
    pass
```

---

## 🐛 Troubleshooting

### Issue: `poetry: command not found`

**Solution:** Install Poetry:

```powershell
pip install poetry
```

### Issue: `torch not found` when training

**Solution:** Install PyTorch:

```powershell
pip install torch
```

### Issue: `make: command not found`

**Solution:** Use Git Bash or WSL, or run commands directly with `poetry run`.

### Issue: Port mismatch in Vite proxy

**Note:** This is a Swiftly AI project issue, not applicable to TraderBot.

---

## 📚 Additional Resources

- **README.md** - Full project documentation
- **tests/** - Example test cases
- **runs/** - Previous backtest results

---

## 🎉 You're Ready!

Your TraderBot is set up and ready to run. Start with:

```powershell
cd E:\Trade_Bot\traderbot
poetry install
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
```

Happy trading! 🚀📈

