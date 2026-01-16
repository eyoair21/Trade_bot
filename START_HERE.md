# 🚀 START HERE - TraderBot Quick Start

## ✅ Your System is Ready!

You have:
- ✅ Python 3.13.5
- ✅ pip 25.3
- ✅ All required packages (pandas, numpy, pyarrow, python-dotenv)
- ✅ Sample data generated
- ✅ First backtest completed successfully!

---

## 📊 Your First Successful Run

You just ran:

```powershell
cd E:\Trade_Bot\traderbot
python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
```

**Results:** `runs\20260105_011345\`
- ✅ 3 walk-forward splits completed
- ✅ Results saved to `results.json`
- ✅ Equity curve saved to `equity_curve.csv`
- ✅ Run manifest saved to `run_manifest.json`

---

## 🎯 What to Do Next

### Option 1: Train PatchTST Model (Recommended)

**Step 1:** Install PyTorch

```powershell
pip install torch
```

**Step 2:** Train the model (20 epochs, ~2-5 minutes on CPU)

```powershell
cd E:\Trade_Bot\traderbot
python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --runs-dir runs --epochs 20 --batch-size 64 --learning-rate 1e-4 --val-split 0.2 --seed 42 --lookback 32 --features close_ret_1,rsi_14,atr_14,vwap_gap,dvol_5,regime_vix
```

**Outputs:**
- `models\patchtst.ts` - TorchScript model
- `runs\<timestamp>\train_patchtst.json` - Training metrics

**Step 3:** Run backtest with the trained model

```powershell
python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --universe-mode dynamic --model-path models/patchtst.ts
```

---

### Option 2: Run More Backtests (No Model)

#### Basic Walk-Forward

```powershell
cd E:\Trade_Bot\traderbot
python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
```

#### With Dynamic Universe Selection

```powershell
python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --universe-mode dynamic
```

#### Extended Date Range (5 splits)

```powershell
python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 5 --is-ratio 0.7
```

---

### Option 3: Generate More Data

```powershell
cd E:\Trade_Bot\traderbot
python scripts/make_sample_data.py --tickers AAPL MSFT NVDA GOOG AMZN META --start-date 2023-01-01 --end-date 2023-12-31 --seed 42
```

---

## 📁 Your Results

### View Latest Run

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

---

## 🔧 Configuration

### Model Features (in order)

The PatchTST model uses these features:

1. `close_ret_1` - 1-day return (clipped ±10%)
2. `rsi_14` - 14-day RSI
3. `atr_14` - 14-day ATR
4. `vwap_gap` - % gap from VWAP
5. `dvol_5` - Dollar volume ratio vs 5d mean
6. `regime_vix` - Volatility regime indicator

### Universe Thresholds

Edit `traderbot\config.py` to adjust:

```python
universe:
  max_symbols: 30
  min_dollar_volume: 20000000  # $20M avg 20d
  min_volatility: 0.15         # 15% annualized
  lookback_days: 20
```

---

## 📚 Documentation Files

- **QUICKSTART.md** - Comprehensive setup guide
- **WINDOWS_COMMANDS.md** - All Windows PowerShell commands
- **README.md** - Full project documentation
- **START_HERE.md** - This file (quick reference)

---

## 🧪 Run Tests (Optional)

```powershell
cd E:\Trade_Bot\traderbot

# Install test dependencies
pip install pytest pytest-cov ruff black mypy

# Run all tests
pytest

# Run with coverage
pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q

# Run specific test
pytest tests/test_smoke.py -v
```

---

## 🎓 Command Reference

| Task | Command |
|------|---------|
| **Generate Data** | `python scripts/make_sample_data.py` |
| **Train Model** | `python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --epochs 20` |
| **Run Backtest** | `python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6` |
| **With Model** | Add `--model-path models/patchtst.ts` to backtest command |
| **Dynamic Universe** | Add `--universe-mode dynamic` to backtest command |
| **View Help** | `python -m traderbot.cli.walkforward --help` |

---

## 🐛 Troubleshooting

### Issue: "torch not found"

```powershell
pip install torch
```

### Issue: "pyarrow version mismatch"

```powershell
# Regenerate data with current pyarrow version
python scripts/make_sample_data.py
```

### Issue: "No module named traderbot"

```powershell
# Make sure you're in the traderbot directory
cd E:\Trade_Bot\traderbot
```

---

## 🎉 You're All Set!

Your TraderBot is fully operational. Start with:

```powershell
cd E:\Trade_Bot\traderbot

# Option 1: Train model first (recommended)
pip install torch
python scripts/train_patchtst.py --data-dir data/ohlcv --model-path models/patchtst.ts --epochs 20
python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 --model-path models/patchtst.ts

# Option 2: Run more backtests (no model)
python -m traderbot.cli.walkforward --start-date 2023-01-10 --end-date 2023-03-31 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
```

**Happy Trading! 🚀📈**

---

## 📞 Need Help?

- Check **QUICKSTART.md** for detailed setup instructions
- Check **WINDOWS_COMMANDS.md** for all PowerShell commands
- Check **README.md** for full project documentation



