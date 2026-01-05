# Windows PowerShell Commands - TraderBot

This file contains all commands formatted for Windows PowerShell (no bash syntax).

## 🔧 Setup Commands

### Install Poetry

```powershell
pip install poetry
poetry --version
```

### Install Dependencies

```powershell
cd E:\Trade_Bot\traderbot
poetry install
```

### Install PyTorch (Required for Training)

```powershell
pip install torch
```

### Install Optional Dependencies

```powershell
# For FinBERT sentiment analysis (optional)
pip install transformers

# For downloading real market data (optional)
pip install yfinance
```

---

## 📊 Data Commands

### Generate Sample Data

```powershell
cd E:\Trade_Bot\traderbot
poetry run python scripts/make_sample_data.py
```

### Generate with Custom Options

```powershell
poetry run python scripts/make_sample_data.py --output-dir data/ohlcv --tickers AAPL MSFT NVDA GOOG AMZN --start-date 2023-01-01 --end-date 2023-12-31 --seed 42
```

---

## 🤖 Model Training Commands

### Basic Training (20 epochs)

```powershell
cd E:\Trade_Bot\traderbot
poetry run python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --runs-dir runs --epochs 20 --batch-size 64 --learning-rate 1e-4 --val-split 0.2 --seed 42 --lookback 32 --features close_ret_1,rsi_14,atr_14,vwap_gap,dvol_5,regime_vix
```

### Extended Training (50 epochs)

```powershell
poetry run python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --epochs 50 --batch-size 32 --learning-rate 1e-4
```

### Quick Training (10 epochs for testing)

```powershell
poetry run python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --epochs 10 --batch-size 16
```

---

## 🔄 Walk-Forward Backtest Commands

### 1. Basic Walk-Forward (No Model)

```powershell
cd E:\Trade_Bot\traderbot
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
```

### 2. Static Universe Mode (Explicit)

```powershell
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --universe-mode static
```

### 3. Dynamic Universe Mode

```powershell
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --universe-mode dynamic
```

### 4. With PatchTST Model

```powershell
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --universe-mode dynamic --model-path models/patchtst.ts
```

### 5. Extended Date Range

```powershell
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-01 --end-date 2023-12-31 --universe AAPL MSFT NVDA GOOG AMZN --n-splits 5 --is-ratio 0.7 --universe-mode dynamic --model-path models/patchtst.ts
```

### 6. Custom Data Directory

```powershell
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --data-root E:/custom/path/ohlcv
```

---

## 🧪 Testing Commands

### Run All Tests

```powershell
cd E:\Trade_Bot\traderbot
poetry run pytest
```

### Run with Coverage

```powershell
poetry run pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q
```

### Run Specific Test File

```powershell
poetry run pytest tests/test_smoke.py -v
```

### Run Specific Test Function

```powershell
poetry run pytest tests/test_smoke.py::test_imports -v
```

### Run Tests by Category

```powershell
# Data tests
poetry run pytest tests/data/ -v

# Engine tests
poetry run pytest tests/engine/ -v

# Feature tests
poetry run pytest tests/features/ -v

# Model tests
poetry run pytest tests/model/ -v
```

---

## 🔍 Code Quality Commands

### Linting (Ruff)

```powershell
cd E:\Trade_Bot\traderbot
poetry run ruff check .
```

### Auto-fix Linting Issues

```powershell
poetry run ruff check . --fix
```

### Format Check (Black)

```powershell
poetry run black --check .
```

### Auto-format Code

```powershell
poetry run black .
```

### Type Checking (Mypy)

```powershell
poetry run mypy traderbot
```

### Run All Quality Checks

```powershell
poetry run ruff check .
poetry run black --check .
poetry run mypy traderbot
poetry run pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q
```

---

## 🧹 Cleanup Commands

### Remove Generated Data

```powershell
cd E:\Trade_Bot\traderbot
Remove-Item -Recurse -Force data/ohlcv/*.parquet
```

### Remove Trained Models

```powershell
Remove-Item -Recurse -Force models/*.ts
```

### Remove Run Results

```powershell
Remove-Item -Recurse -Force runs/*
```

### Remove Cache Files

```powershell
Remove-Item -Recurse -Force .pytest_cache
Remove-Item -Recurse -Force .mypy_cache
Remove-Item -Recurse -Force .ruff_cache
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Recurse -File -Filter *.pyc | Remove-Item -Force
```

### Full Cleanup

```powershell
Remove-Item -Recurse -Force data/ohlcv/*.parquet -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force models/*.ts -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force runs/* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .mypy_cache -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .ruff_cache -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Recurse -File -Filter *.pyc | Remove-Item -Force -ErrorAction SilentlyContinue
```

---

## 📋 One-Liner Workflows

### Complete Setup from Scratch

```powershell
cd E:\Trade_Bot\traderbot; poetry install; poetry run python scripts/make_sample_data.py
```

### Train and Run Backtest

```powershell
cd E:\Trade_Bot\traderbot; poetry run python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --epochs 20; poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --model-path models/patchtst.ts
```

### Full Verification Pipeline

```powershell
cd E:\Trade_Bot\traderbot; poetry install; poetry run python scripts/make_sample_data.py; poetry run pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q; poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
```

---

## 🔄 Direct Python Commands (Without Poetry)

If you prefer not to use Poetry:

### Install Dependencies

```powershell
pip install pandas numpy pyarrow python-dotenv torch transformers
pip install pytest pytest-cov ruff black mypy pandas-stubs
```

### Train Model

```powershell
cd E:\Trade_Bot\traderbot
python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --epochs 20
```

### Run Backtest

```powershell
python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
```

### Run Tests

```powershell
pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q
```

---

## 📊 View Results

### View Latest Run Results

```powershell
cd E:\Trade_Bot\traderbot\runs
Get-ChildItem -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### View Results JSON

```powershell
$latestRun = Get-ChildItem -Directory runs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content "$($latestRun.FullName)/results.json" | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### View Equity Curve

```powershell
$latestRun = Get-ChildItem -Directory runs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Import-Csv "$($latestRun.FullName)/equity_curve.csv" | Format-Table
```

### View Training Metrics

```powershell
$latestRun = Get-ChildItem -Directory runs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content "$($latestRun.FullName)/train_patchtst.json" | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Install Poetry | `pip install poetry` |
| Install Dependencies | `poetry install` |
| Generate Data | `poetry run python scripts/make_sample_data.py` |
| Train Model | `poetry run python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --epochs 20` |
| Run Backtest | `poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6` |
| Run Tests | `poetry run pytest` |
| Lint Code | `poetry run ruff check .` |
| Format Code | `poetry run black .` |
| Type Check | `poetry run mypy traderbot` |

---

## 🚀 Recommended First Run

```powershell
# 1. Navigate to project
cd E:\Trade_Bot\traderbot

# 2. Install Poetry (if not installed)
pip install poetry

# 3. Install dependencies
poetry install

# 4. Verify data exists (you already have it!)
Get-ChildItem data/ohlcv

# 5. Run a quick backtest (no model)
poetry run python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6

# 6. Check results
$latestRun = Get-ChildItem -Directory runs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "Results in: $($latestRun.FullName)"
Get-Content "$($latestRun.FullName)/results.json"
```

---

**Note:** All commands assume you're running PowerShell. For Git Bash or WSL, use the Makefile commands instead.

