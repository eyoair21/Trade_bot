# Profit-Aware Learning: Quick Reference Card

**One-page reference for developers implementing profit-aware features**

---

## 🎯 Core Modules

| Module | Path | Key Function |
|--------|------|-------------|
| **Reward** | `engine/reward.py` | `compute_run_metrics(equity, trades, breaches, weights)` |
| **Position** | `policies/position.py` | `map_signal_to_size(signal_prob, vol, mode="vol")` |
| **Splits** | `splits/purged.py` | `PurgedEmbargoSplit(n_splits, embargo_days, purge_days)` |
| **Universe** | `universe/selector.py` | `select_universe(date, tickers, data, config)` |
| **GA** | `opt/ga_opt.py` | `GeneticOptimizer(space, config, base_cmd, objective)` |
| **Bandit** | `alloc/bandit.py` | `ThompsonSamplingAllocator(strategies)` |
| **Guards** | `engine/risk_guard.py` | `RiskGuard.check_guards(equity, positions, timestamp)` |
| **Report** | `reporting/report.py` | `generate_report(run_dir, output_dir)` |

---

## 🔧 Config Variables

```bash
# Reward weights
REWARD_LAMBDA_DD=0.2
REWARD_TAU_TURNOVER=0.001
REWARD_KAPPA_BREACH=0.5

# Position sizing
SIZING_SIZER=vol              # fixed|vol|kelly
SIZING_VOL_TARGET=0.15
SIZING_KELLY_CAP=0.25

# Risk guards
MAX_DRAWDOWN_PCT=0.15
DAILY_LOSS_LIMIT_PCT=0.02
MAX_POSITION_PCT=0.10
```

---

## 🚀 CLI Quick Commands

```bash
# Dynamic universe + leakage control
make backtest ARGS="--universe-mode dynamic --top-n 50 --embargo-days 5 --purge 1d"

# GA optimization
make opt ARGS="--budget 30 --oos-metric total_reward"

# Generate report
make report

# Bandit allocation
poetry run python -m traderbot.cli.walkforward --alloc bandit

# Vol targeting
poetry run python -m traderbot.cli.walkforward --sizer vol --vol-target 0.15
```

---

## 📦 Run Artifacts

`runs/<timestamp>/`:
- `results.json` → total_reward, pnl_net, sharpe
- `equity_curve.csv` → daily equity, drawdown
- `orders.csv` → trades with PnL, turnover
- `breaches.csv` → risk events
- `splits.json` → train/test metadata
- `universe_*.json` → universe snapshots
- `alloc.csv` → bandit weights
- `report/` → markdown + plots

---

## 🧪 Test Commands

```bash
# All tests
poetry run pytest

# Specific module
poetry run pytest tests/engine/test_reward.py -v

# Coverage
poetry run pytest --cov=traderbot --cov-report=term-missing
```

---

## 💡 Integration Patterns

### 1. Reward in Backtest

```python
from traderbot.engine.reward import compute_run_metrics, RewardWeights

weights = RewardWeights(lambda_dd=0.2, tau_turnover=0.001, kappa_breach=0.5)
metrics = compute_run_metrics(equity_df, trades_df, breaches_df, weights)
# metrics["total_reward"], metrics["pnl_net"], etc.
```

### 2. Position Sizing

```python
from traderbot.policies.position import map_signal_to_size

size = map_signal_to_size(
    signal_prob=0.65,
    realized_vol=0.20,
    mode="vol",
    target_vol=0.15,
    max_leverage=1.0,
)
shares = int(size * equity / price)
```

### 3. Risk Guards

```python
from traderbot.engine.risk_guard import RiskGuard, RiskGuardConfig

guard = RiskGuard(RiskGuardConfig(max_drawdown_pct=0.15), initial_capital=100000)
allow, actions = guard.check_guards(equity, positions, timestamp, is_new_day)

if not allow:
    # Flatten positions
    pass
```

### 4. Purged Splits

```python
from traderbot.splits.purged import PurgedEmbargoSplit

splitter = PurgedEmbargoSplit(n_splits=5, embargo_days=5, purge_days=1)
for train_idx, test_idx in splitter.split(dates):
    # Train on train_idx, test on test_idx
    pass
```

### 5. Bandit Allocation

```python
from traderbot.alloc.bandit import ThompsonSamplingAllocator

allocator = ThompsonSamplingAllocator(["momentum", "mean_reversion"])
weights = allocator.get_weights()  # {"momentum": 0.6, "mean_reversion": 0.4}
allocator.update("momentum", reward=0.8)
```

---

## 🎨 Architecture Overview

```
Walk-Forward Loop
    ↓
Universe Selection (dynamic)
    ↓
Purged+Embargo Splits
    ↓
Model Training
    ↓
Signal Generation
    ↓
Position Sizing (vol/kelly/fixed)
    ↓
Risk Guards Check
    ↓
Order Execution
    ↓
Reward Computation
    ↓
Bandit Update (optional)
    ↓
GA Optimization (offline)
    ↓
Reporting (plots + markdown)
```

---

## 📖 Documentation Map

| File | Use Case |
|------|----------|
| `PROFIT_AWARE_LEARNING.md` | Full feature docs |
| `INTEGRATION_GUIDE.md` | How to integrate |
| `IMPLEMENTATION_SUMMARY.md` | What was built |
| `LOCAL_RUN.md` | Quick start guide |
| `QUICK_REFERENCE.md` | This file |

---

## ⚡ Pro Tips

1. **Start Simple**: Enable one feature at a time
2. **Check Artifacts**: Always inspect `runs/<ts>/results.json`
3. **Use Opt-In**: All features are config/CLI flag gated
4. **Test First**: Run `pytest` before and after integration
5. **Reward First**: Total reward = PnL - penalties (check sign!)
6. **Vol Sizing**: Usually better than fixed for diverse universe
7. **Embargo Matters**: Use 5 days for daily data to prevent leakage
8. **GA Budget**: Start with 20-30 evals for quick tests
9. **Bandit**: Needs 30+ days to converge, best for multi-strategy
10. **Report**: Always run `make report` to visualize results

---

## 🐛 Common Pitfalls

| Issue | Solution |
|-------|----------|
| Negative rewards | Check penalty weights (too high?) |
| Low diversity | Increase top_n in universe selection |
| GA slow | Reduce population or use parallel=True |
| Breaches firing | Adjust thresholds (max_dd, daily_loss) |
| Bandit unstable | Need more data or lower temperature |
| Reports empty | Check artifact paths in results.json |

---

## 📞 Support

- **Issues**: Check `tests/` for examples
- **Integration**: See `INTEGRATION_GUIDE.md`
- **Features**: See `PROFIT_AWARE_LEARNING.md`
- **Tests**: Run `poetry run pytest tests/<module>/`

---

**Version**: 1.0  
**Last Updated**: 2026-01-26  
**Status**: Production Ready ✅

