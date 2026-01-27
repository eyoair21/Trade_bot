# Phase 2 Quick Start Guide

**5-Minute Demo of Phase 2 Features**

---

## Prerequisites

```bash
# 1. Install dependencies (adds scikit-learn)
poetry install

# 2. Generate sample data (if not already done)
make data

# 3. Verify Phase 1 works
make backtest
```

---

## Phase 2 Features Demo

### 1. Online Reward Adaptation

**What it does**: Adapts reward weights based on rolling OOS metrics

```bash
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-06-30 \
  --universe AAPL MSFT NVDA \
  --n-splits 5 \
  --reward-adapt-mode ewma \
  --output-dir runs/phase2_demo1
```

**Check outputs**:
```bash
cat runs/phase2_demo1/reward_weights.json
# Should show evolving weights across splits
```

**Expected**: Weights change over time; total_reward improves by ≥2% vs. fixed weights

---

### 2. Regime Detection & Routing

**What it does**: Detects market regimes and routes to appropriate models

```bash
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-06-30 \
  --universe AAPL MSFT NVDA \
  --regime-mode on \
  --regime-k 3 \
  --output-dir runs/phase2_demo2
```

**Check outputs**:
```bash
cat runs/phase2_demo2/regimes.csv
# Should show regime IDs (0, 1, 2) and selected models
```

**Expected**: Different metrics per regime; regimes correlate with market conditions

---

### 3. Portfolio Limits & Management

**What it does**: Enforces gross/net/position/sector limits

```python
# Python API demo
from traderbot.engine.portfolio import PortfolioManager, PortfolioLimits

limits = PortfolioLimits(
    max_gross_exposure=1.0,
    max_net_exposure=0.5,
    max_position_pct=0.10,
)

mgr = PortfolioManager(limits, capital=100000)

# Propose orders
proposed = {"AAPL": 20000, "MSFT": 15000, "NVDA": 30000}  # 65% gross
prices = {"AAPL": 150, "MSFT": 300, "NVDA": 500}

# Clip to limits
clipped = mgr.clip_orders(proposed, prices)

print("Proposed gross:", sum(abs(v) for v in proposed.values()))
print("Clipped gross:", sum(abs(v) for v in clipped.values()))
# Clipped should be ≤ 100000
```

**Expected**: Orders automatically clipped to respect all limits

---

### 4. Realistic Cost Model

**What it does**: Models slippage, market impact, and fees

```python
from traderbot.engine.costs import CostModel

model = CostModel(spread_bps=5.0, fee_per_share=0.0005)

# Estimate costs for 1000 shares @ $150
costs = model.estimate(
    ticker="AAPL",
    quantity=1000,
    price=150.0,
    adv=10_000_000,  # Average daily volume
    volatility=0.20,
)

print(f"Slippage: ${costs.slippage:.2f}")
print(f"Impact: ${costs.impact:.2f}")
print(f"Fees: ${costs.fees:.2f}")
print(f"Total: ${costs.total:.2f}")

# Get summary
summary = model.get_summary()
print(summary)
```

**Expected**: Total costs = slippage + impact + fees; breakdown shown in report

---

### 5. Contextual Bandit Allocation

**What it does**: Allocates across strategies based on context

```python
from traderbot.alloc.bandit import ContextualThompsonSampling

allocator = ContextualThompsonSampling(
    strategy_names=["momentum", "mean_reversion", "patchtst"]
)

# Context: low vol regime
context_low_vol = {"regime_id": 0, "realized_vol": 0.10}
weights_low = allocator.get_weights(context_low_vol)

# Context: high vol regime
context_high_vol = {"regime_id": 2, "realized_vol": 0.30}
weights_high = allocator.get_weights(context_high_vol)

print("Low vol weights:", weights_low)
print("High vol weights:", weights_high)
# Different contexts → different allocations
```

**Expected**: Weights adapt based on context; outperforms best single strategy

---

### 6. Parameter Sweep & Leaderboard

**What it does**: Searches parameter space and ranks configs

```bash
make sweep ARGS="--budget 20 --metric total_reward"
```

**Check outputs**:
```bash
cat runs/sweep/leaderboard.csv | head -10
# Shows top 10 configs ranked by total_reward
```

**Makefile shortcut**:
```bash
# Custom budget and metric
make sweep ARGS="--budget 30 --metric sharpe"
```

**Expected**: Sorted leaderboard with config params and performance metrics

---

## Combined Demo (All Features)

```bash
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-06-30 \
  --universe AAPL MSFT NVDA \
  --n-splits 5 \
  --reward-adapt-mode ewma \
  --regime-mode on \
  --regime-k 3 \
  --sizer vol \
  --vol-target 0.15 \
  --output-dir runs/phase2_full
```

**Outputs to check**:
```bash
ls runs/phase2_full/
# reward_weights.json     - Adaptation history
# regimes.csv             - Regime classification
# portfolio_stats.json    - Portfolio metrics
# results.json            - Performance with costs
```

**Generate report**:
```bash
poetry run python -c "from pathlib import Path; from traderbot.reporting.report import generate_report; generate_report(Path('runs/phase2_full'))"
```

---

## Test Phase 2 Modules

```bash
# All Phase 2 tests
poetry run pytest tests/phase2/ -v

# Specific tests
poetry run pytest tests/phase2/test_reward_adapter.py -v
poetry run pytest tests/phase2/test_regime_router.py -v
poetry run pytest tests/phase2/test_portfolio_limits.py -v
poetry run pytest tests/phase2/test_costs_model.py -v
poetry run pytest tests/phase2/test_contextual_bandit.py -v

# With coverage
poetry run pytest tests/phase2/ --cov=traderbot.engine --cov=traderbot.research --cov=traderbot.alloc
```

**Expected**: All tests pass (25+ tests)

---

## Sanity Check (< 5 min)

```bash
# 1. Test reward adaptation
python -c "
from traderbot.engine.reward_adapter import RewardAdapter
from traderbot.engine.reward import RewardWeights

adapter = RewardAdapter.from_config(mode='ewma')
weights = adapter.current_weights

for i in range(3):
    new_weights = adapter.update(weights, {'total_reward': 100 * (i+1)})
    weights = new_weights
    print(f'Iter {i}: lambda_dd={weights.lambda_dd:.3f}')

print('✓ Reward adapter works')
"

# 2. Test regime detection
python -c "
import numpy as np
import pandas as pd
from traderbot.research.regime import detect_regimes

df = pd.DataFrame({
    'returns': np.random.randn(100) * 0.01,
    'volatility': np.abs(np.random.randn(100)) * 0.02,
})

regimes = detect_regimes(df, k=3)
print(f'Detected {len(regimes.unique())} regimes')
print('✓ Regime detection works')
"

# 3. Test portfolio limits
python -c "
from traderbot.engine.portfolio import PortfolioManager, PortfolioLimits

limits = PortfolioLimits(max_gross_exposure=1.0)
mgr = PortfolioManager(limits, capital=100000)

proposed = {'AAPL': 60000, 'MSFT': 60000}  # 120% gross
clipped = mgr.clip_orders(proposed, {'AAPL': 150, 'MSFT': 300})

gross = sum(abs(v) for v in clipped.values())
print(f'Clipped gross: ${gross:.0f} (limit: $100000)')
print('✓ Portfolio limits work')
"

# 4. Test cost model
python -c "
from traderbot.engine.costs import estimate_costs

costs = estimate_costs(quantity=1000, price=100.0, spread_bps=5.0)
print(f'Total costs: ${costs.total:.2f}')
print(f'  Slippage: ${costs.slippage:.2f}')
print(f'  Impact: ${costs.impact:.2f}')
print(f'  Fees: ${costs.fees:.2f}')
print('✓ Cost model works')
"

# 5. Test contextual bandit
python -c "
from traderbot.alloc.bandit import ContextualThompsonSampling

allocator = ContextualThompsonSampling(['strategy_a', 'strategy_b'])
context = {'regime_id': 0, 'realized_vol': 0.15}

weights = allocator.get_weights(context)
print(f'Weights: {weights}')
print('✓ Contextual bandit works')
"

echo ""
echo "✅ All Phase 2 components functional!"
```

**Expected**: All checks print "✓" and no errors

---

## Common Issues & Fixes

### Issue: `ModuleNotFoundError: No module named 'sklearn'`

**Fix**:
```bash
poetry add scikit-learn
# or
pip install scikit-learn
```

### Issue: HMM regime detection fails

**Fix**: HMM is optional; falls back to K-Means
```bash
# Optional: Install hmmlearn
poetry add hmmlearn --optional
```

### Issue: Sweep runs slowly

**Fix**: Reduce budget or use fewer splits
```bash
make sweep ARGS="--budget 10 --n-splits 2"
```

### Issue: Results don't show reward_weights.json

**Fix**: Ensure `--reward-adapt-mode` flag is used:
```bash
poetry run python -m traderbot.cli.walkforward \
  --reward-adapt-mode ewma \
  ...
```

---

## Next Steps

1. **Integration**: Wire Phase 2 into main backtest engine (see `PHASE2_INTEGRATION.md`)
2. **Tuning**: Run sweep to find best config for your data
3. **Validation**: Compare Phase 2 vs Phase 1 on OOS data
4. **Monitoring**: Track reward adaptation and regime changes
5. **Production**: Add remaining features (live trading, multi-asset)

---

## Performance Expectations

| Feature | Overhead | Impact |
|---------|----------|--------|
| Reward Adapter | ~5ms/split | Minimal |
| Regime Detection | ~50ms/split | Low |
| Portfolio Clipping | ~10ms/bar | Negligible |
| Cost Model | ~1ms/order | Negligible |
| Contextual Bandit | ~5ms/update | Minimal |

**Total**: < 100ms per split (negligible overhead)

---

## Support

- **Docs**: `PHASE2_SUMMARY.md`, `PHASE2_CHANGELOG.md`
- **Tests**: `tests/phase2/` for examples
- **API**: Check docstrings in each module
- **Issues**: Create GitHub issue with logs

---

**Quick Start Guide Version**: 2.0  
**Last Updated**: 2026-01-27  
**Estimated Time**: 5-10 minutes for full demo

