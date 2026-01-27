# Profit-Aware Learning Loop

This document describes the profit-aware learning system implemented in TraderBot.

## Overview

The profit-aware learning loop enables TraderBot to learn, adapt, and self-tune for profit by:

1. **Centralized Reward Function** - PnL-first rewards with configurable penalties
2. **Position Sizing Policies** - Signal-to-size mapping with vol targeting and Kelly
3. **Leakage Controls** - Purged+embargo splits for walk-forward validation
4. **Dynamic Universe** - First-class universe selection based on liquidity
5. **GA Optimization** - Hyperparameter search with reward maximization
6. **Bandit Allocation** - Multi-strategy allocation via Thompson Sampling
7. **Safety Rails** - Kill-switch and risk guards
8. **Reporting** - Lightweight markdown reports with plots

## Core Components

### 1. Reward Function (`traderbot/engine/reward.py`)

Computes rewards tied to realized PnL with penalties for risk:

```python
from traderbot.engine.reward import RewardWeights, compute_step_reward, compute_run_metrics

# Configure weights
weights = RewardWeights(
    lambda_dd=0.2,        # Drawdown penalty coefficient
    tau_turnover=0.001,   # Turnover penalty coefficient
    kappa_breach=0.5,     # Risk breach penalty coefficient
)

# Compute step reward
reward = compute_step_reward(
    pnl_after_costs=100.0,
    drawdown_step=5.0,
    turnover_step=1000.0,
    breach_step=0,
    weights=weights,
)

# Compute run metrics
metrics = compute_run_metrics(
    equity_curve=equity_df,  # DataFrame with equity, drawdown
    trades=trades_df,        # DataFrame with pnl_net, turnover
    breaches=breaches_df,    # DataFrame with breach records
    weights=weights,
)
# Returns: total_reward, pnl_net, sharpe, sortino, max_dd, turnover, breaches_count
```

**Environment Variables:**
- `REWARD_LAMBDA_DD`: Drawdown penalty weight (default: 0.2)
- `REWARD_TAU_TURNOVER`: Turnover penalty weight (default: 0.001)
- `REWARD_KAPPA_BREACH`: Breach penalty weight (default: 0.5)

### 2. Position Sizing (`traderbot/policies/position.py`)

Maps signals to position sizes using multiple strategies:

```python
from traderbot.policies.position import map_signal_to_size

# Target volatility sizing (default)
size = map_signal_to_size(
    signal_prob=0.65,      # Model output probability
    realized_vol=0.20,     # Recent 20-day volatility
    mode="target_vol",
    target_vol=0.15,       # Target 15% portfolio vol
    max_leverage=1.0,
)

# Kelly criterion
size = map_signal_to_size(
    signal_prob=0.60,
    mode="kelly",
    edge=0.01,             # Expected edge
    variance=0.04,         # Return variance
    cap=0.25,              # Max 25% allocation
)

# Fixed fractional
size = map_signal_to_size(
    signal_prob=0.70,
    mode="fixed",
    fixed_frac=0.10,       # 10% per position
)
```

### 3. Purged+Embargo Splits (`traderbot/splits/purged.py`)

Prevents information leakage in walk-forward validation:

```python
from traderbot.splits.purged import PurgedEmbargoSplit

splitter = PurgedEmbargoSplit(
    n_splits=5,
    test_size=0.2,
    embargo_days=5,   # Buffer after test
    purge_days=1,     # Remove overlapping labels
)

for train_idx, test_idx in splitter.split(dates):
    # Train and test with no leakage
    pass
```

**CLI Flags:**
```bash
python -m traderbot.cli.walkforward \
  --embargo-days 5 \
  --purge 1d
```

### 4. Dynamic Universe (`traderbot/universe/selector.py`)

Selects tradeable universe based on liquidity filters:

```python
from traderbot.universe.selector import select_universe, UniverseConfig

config = UniverseConfig(
    top_n=100,
    min_dollar_volume=1_000_000.0,
    min_price=5.0,
    min_history_days=60,
    lookback_days=20,
)

selected = select_universe(
    date=datetime(2023, 1, 1),
    full_universe=["AAPL", "MSFT", ...],
    price_data=ohlcv_dict,
    config=config,
)
```

**CLI Flags:**
```bash
python -m traderbot.cli.walkforward \
  --universe-mode dynamic \
  --top-n 100
```

### 5. GA Optimization (`traderbot/opt/ga_opt.py`)

Genetic algorithm for hyperparameter search:

```python
from traderbot.opt.ga_opt import GeneticOptimizer, GAConfig, SearchSpace

optimizer = GeneticOptimizer(
    search_space=SearchSpace(),
    ga_config=GAConfig(
        population_size=20,
        n_generations=10,
    ),
    base_cmd="python -m traderbot.cli.walkforward",
    objective="total_reward",
)

best_config = optimizer.optimize()
optimizer.save_results(output_dir)
```

**Makefile Target:**
```bash
make opt ARGS="--budget 30 --top-n 50"
```

### 6. Bandit Allocation (`traderbot/alloc/bandit.py`)

Multi-strategy allocation using Thompson Sampling:

```python
from traderbot.alloc.bandit import ThompsonSamplingAllocator

allocator = ThompsonSamplingAllocator(
    strategy_names=["momentum", "mean_reversion", "patchtst"]
)

# Get weights for each strategy
weights = allocator.get_weights()

# Update after observing reward
allocator.update("momentum", reward=0.8)
```

**CLI Flag:**
```bash
python -m traderbot.cli.walkforward --alloc bandit
```

### 7. Risk Guards (`traderbot/engine/risk_guard.py`)

Kill-switch and safety rails:

```python
from traderbot.engine.risk_guard import RiskGuard, RiskGuardConfig

guard = RiskGuard(
    config=RiskGuardConfig(
        max_drawdown_pct=0.15,
        daily_loss_limit_pct=0.02,
        loss_cooldown_bars=10,
        max_position_pct=0.10,
    ),
    initial_capital=100000.0,
)

# Check guards each step
allow_trading, actions = guard.check_guards(
    current_equity=equity,
    positions={"AAPL": 10000},
    timestamp=dt,
    is_new_day=False,
)

if not allow_trading:
    # Kill-switch active, flatten positions
    pass
```

### 8. Reporting (`traderbot/reporting/report.py`)

Generate markdown reports with plots:

```python
from pathlib import Path
from traderbot.reporting.report import generate_report

generate_report(
    run_dir=Path("runs/2023-01-01_120000"),
    output_dir=Path("runs/2023-01-01_120000/report"),
)
```

**Makefile Target:**
```bash
make report
```

## Artifacts

After a run, the following artifacts are saved to `runs/<timestamp>/`:

- `results.json` - Metrics including total_reward, pnl_net, sharpe, etc.
- `equity_curve.csv` - Daily equity and drawdown
- `orders.csv` - All trades with PnL and turnover
- `breaches.csv` - Risk breach events
- `splits.json` - Train/test split info (if walk-forward)
- `universe_<date>.json` - Universe snapshots (if dynamic mode)
- `alloc.csv` - Strategy weights over time (if bandit)
- `ga/best_config.json` - Best config from GA optimization
- `report/` - Generated markdown report and plots

## Quick Start

### 1. Run a backtest with profit-aware features:

```bash
make backtest ARGS="--universe-mode dynamic --top-n 50 --embargo-days 5 --purge 1d"
```

### 2. Run GA optimization:

```bash
make opt ARGS="--budget 30 --oos-metric total_reward"
```

### 3. Generate report:

```bash
make report
```

### 4. Run with bandit allocation:

```bash
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA \
  --alloc bandit \
  --universe-mode dynamic \
  --top-n 50
```

## Environment Variables

Add to `.env` file:

```bash
# Reward weights
REWARD_LAMBDA_DD=0.2
REWARD_TAU_TURNOVER=0.001
REWARD_KAPPA_BREACH=0.5

# Risk guards
MAX_DRAWDOWN_PCT=0.15
DAILY_LOSS_LIMIT_PCT=0.02

# Position sizing
SIZING_SIZER=target_vol
SIZING_VOL_TARGET=0.15
SIZING_KELLY_CAP=0.25

# Universe
UNIVERSE_MAX_SYMBOLS=100
UNIVERSE_MIN_DOLLAR_VOLUME=1000000
```

## Testing

Run tests for new modules:

```bash
# All tests
poetry run pytest

# Specific modules
poetry run pytest tests/engine/test_reward.py
poetry run pytest tests/policies/test_position_policy.py
poetry run pytest tests/splits/test_purged.py
poetry run pytest tests/alloc/test_bandit.py
poetry run pytest tests/engine/test_risk_guard.py
```

## Performance Notes

- **Polars Support**: New modules prefer Polars for performance but fallback to Pandas
- **Parallel GA**: Set `n_jobs=-1` to use all CPUs for GA optimization
- **Fast Mode**: Use `--fast-mode` flag to skip heavy computations on ARM/Raspberry Pi
- **Quantization**: Optional int8 quantization for model inference (if PyTorch available)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Walk-Forward Loop                     │
└─────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼─────┐  ┌─────▼──────┐  ┌────▼─────┐
    │  Universe   │  │  Purged+   │  │  Model   │
    │  Selector   │  │  Embargo   │  │  Train   │
    └───────┬─────┘  └─────┬──────┘  └────┬─────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                    ┌───────▼────────┐
                    │   Backtest     │
                    │   Engine       │
                    └───────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
  ┌─────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
  │  Position  │    │    Risk     │    │   Reward    │
  │   Sizing   │    │   Guards    │    │  Function   │
  └─────┬──────┘    └──────┬──────┘    └──────┬──────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼────────┐
                    │   Artifacts    │
                    │   (results,    │
                    │   equity,      │
                    │   orders, etc) │
                    └───────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
  ┌─────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
  │  Bandit    │    │     GA      │    │  Reporting  │
  │  Alloc     │    │ Optimizer   │    │   Module    │
  └────────────┘    └─────────────┘    └─────────────┘
```

## Future Enhancements

- [ ] Online learning with incremental model updates
- [ ] Multi-objective optimization (Pareto frontier)
- [ ] Adversarial validation for regime detection
- [ ] Portfolio construction with correlation constraints
- [ ] Transaction cost modeling with market impact
- [ ] Real-time monitoring dashboard

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Kelly, J. (1956). "A New Interpretation of Information Rate"
- Thompson, W. R. (1933). "On the Likelihood that One Unknown Probability Exceeds Another"

