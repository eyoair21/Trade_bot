# Integration Guide: Profit-Aware Components

This guide shows how to integrate the new profit-aware learning components into the existing backtest engine **without breaking existing functionality**.

## Integration Points

### 1. Reward Computation in BacktestEngine

**File**: `traderbot/engine/backtest.py`

**Add to imports:**
```python
from traderbot.engine.reward import RewardWeights, compute_step_reward, compute_run_metrics
```

**Add to `BacktestEngine.__init__`:**
```python
def __init__(self, ...):
    # ... existing code ...
    
    # Reward tracking (optional, enabled via config)
    self.track_rewards = config.reward is not None
    if self.track_rewards:
        self.reward_weights = RewardWeights(
            lambda_dd=config.reward.lambda_dd,
            tau_turnover=config.reward.tau_turnover,
            kappa_breach=config.reward.kappa_breach,
        )
        self.step_rewards: list[float] = []
```

**Add to main backtest loop (after each bar/trade):**
```python
# Inside run() method, after processing trades
if self.track_rewards:
    # Compute step reward
    pnl_step = current_equity - prev_equity  # Net PnL for this step
    dd_step = abs(min(0, (current_equity - peak_equity) / peak_equity))
    turnover_step = sum(abs(fill.quantity * fill.price) for fill in fills)
    breach_step = 1 if risk_breach_occurred else 0
    
    step_reward = compute_step_reward(
        pnl_after_costs=pnl_step,
        drawdown_step=dd_step,
        turnover_step=turnover_step,
        breach_step=breach_step,
        weights=self.reward_weights,
    )
    self.step_rewards.append(step_reward)
```

**Add to `BacktestResult`:**
```python
@dataclass
class BacktestResult:
    # ... existing fields ...
    
    # New fields (optional)
    total_reward: float = 0.0
    reward_metrics: dict[str, float] = field(default_factory=dict)
```

**Compute final metrics in `run()` return:**
```python
if self.track_rewards:
    reward_metrics = compute_run_metrics(
        equity_curve=equity_df,
        trades=trades_df,
        breaches=breaches_df,
        weights=self.reward_weights,
    )
    result.total_reward = reward_metrics["total_reward"]
    result.reward_metrics = reward_metrics
```

### 2. Position Sizing Integration

**File**: `traderbot/engine/backtest.py`

**Add to imports:**
```python
from traderbot.policies.position import map_signal_to_size
```

**Replace fixed sizing logic with policy:**
```python
# Before (in signal-to-order conversion):
position_size = fixed_fraction * equity

# After:
signal_prob = strategy.get_signal(ticker)  # e.g., 0.65
realized_vol = calculate_volatility(price_history, window=20)

position_size = map_signal_to_size(
    signal_prob=signal_prob,
    realized_vol=realized_vol,
    mode=config.sizing.sizer,  # "fixed", "vol", "kelly"
    target_vol=config.sizing.vol_target,
    max_leverage=config.risk.max_gross_exposure,
    fixed_frac=config.sizing.fixed_frac,
)

# Convert to shares
shares = int(position_size * equity / current_price)
```

### 3. Risk Guards Integration

**File**: `traderbot/engine/backtest.py`

**Add to imports:**
```python
from traderbot.engine.risk_guard import RiskGuard, RiskGuardConfig
```

**Add to `__init__`:**
```python
def __init__(self, ...):
    # ... existing code ...
    
    # Risk guard (optional)
    if config.risk:
        guard_config = RiskGuardConfig(
            max_drawdown_pct=config.risk.max_drawdown_pct,
            daily_loss_limit_pct=config.risk.daily_loss_limit_pct,
            max_position_pct=config.risk.max_position_pct,
        )
        self.risk_guard = RiskGuard(guard_config, initial_capital=initial_capital)
    else:
        self.risk_guard = None
```

**Check guards each bar:**
```python
# Inside main backtest loop
if self.risk_guard:
    allow_trading, actions = self.risk_guard.check_guards(
        current_equity=broker.equity,
        positions={s: pos.market_value for s, pos in broker.positions.items()},
        timestamp=current_date,
        is_new_day=(current_date.date() != prev_date.date()),
    )
    
    if not allow_trading:
        logger.warning("Kill-switch active, skipping signals")
        continue
    
    if "flatten" in actions:
        # Flatten all positions
        for symbol in list(broker.positions.keys()):
            broker.submit_order(symbol, 0, OrderSide.SELL)
```

**Save breaches:**
```python
if self.risk_guard:
    breaches_df = self.risk_guard.get_breaches_df()
    breaches_df.to_csv(output_dir / "breaches.csv", index=False)
```

### 4. Dynamic Universe in Walk-Forward CLI

**File**: `traderbot/cli/walkforward.py`

**Add to imports:**
```python
from traderbot.universe.selector import select_universe, UniverseConfig, save_universe_snapshot
```

**Add CLI arguments:**
```python
parser.add_argument("--universe-mode", choices=["static", "dynamic"], default="static")
parser.add_argument("--top-n", type=int, default=100)
parser.add_argument("--min-dollar-volume", type=float, default=1_000_000.0)
```

**Implement universe rebalancing:**
```python
# Before each walk-forward split
if args.universe_mode == "dynamic":
    universe_config = UniverseConfig(
        top_n=args.top_n,
        min_dollar_volume=args.min_dollar_volume,
        min_price=5.0,
    )
    
    selected_universe = select_universe(
        date=split_start_date,
        full_universe=all_tickers,
        price_data=ohlcv_data,
        config=universe_config,
    )
    
    # Save snapshot
    save_universe_snapshot(split_start_date, selected_universe, output_dir)
else:
    selected_universe = args.universe  # Static list
```

### 5. Purged+Embargo Splits in Walk-Forward

**File**: `traderbot/cli/walkforward.py`

**Add to imports:**
```python
from traderbot.splits.purged import PurgedEmbargoSplit, save_splits_info
```

**Add CLI arguments:**
```python
parser.add_argument("--embargo-days", type=int, default=0)
parser.add_argument("--purge", type=str, default="0d")  # e.g., "1d", "2d"
```

**Replace naive split with purged:**
```python
# Parse purge days
purge_days = int(args.purge.replace("d", ""))

# Create splitter
splitter = PurgedEmbargoSplit(
    n_splits=args.n_splits,
    test_size=1 - args.is_ratio,
    embargo_days=args.embargo_days,
    purge_days=purge_days,
)

# Generate splits
dates = pd.DatetimeIndex(price_data["date"])
splits = list(splitter.split(dates))

# Save split info
save_splits_info(splits, dates, str(output_dir / "splits.json"))
```

### 6. Bandit Allocation (Optional Advanced Feature)

**File**: `traderbot/cli/walkforward.py` or new `traderbot/cli/bandit_run.py`

**Add CLI argument:**
```python
parser.add_argument("--alloc", choices=["single", "bandit"], default="single")
```

**Implement bandit allocation:**
```python
from traderbot.alloc.bandit import ThompsonSamplingAllocator

if args.alloc == "bandit":
    strategies = ["momentum", "mean_reversion", "patchtst"]
    allocator = ThompsonSamplingAllocator(strategies)
    
    # Each day, get weights and combine signals
    for date in trading_dates:
        weights = allocator.get_weights()
        
        # Combine signals
        combined_signal = sum(
            weights[strat] * strategy_signals[strat][date]
            for strat in strategies
        )
        
        # Execute with combined signal
        # ...
        
        # Update allocator with observed reward
        daily_reward = compute_daily_reward(equity_change)
        for strat in strategies:
            allocator.update(strat, reward=daily_reward * weights[strat])
    
    # Save allocation history
    alloc_df = pd.DataFrame(allocator.history)
    alloc_df.to_csv(output_dir / "alloc.csv", index=False)
```

## Backward Compatibility

All new features are **opt-in** via:

1. **Config flags**: `config.reward`, `config.risk`, etc.
2. **CLI arguments**: `--universe-mode`, `--embargo-days`, `--alloc`, etc.
3. **Graceful fallbacks**: If modules not imported, skip feature

**Example: Safe Import Pattern**
```python
try:
    from traderbot.engine.reward import compute_run_metrics
    REWARD_AVAILABLE = True
except ImportError:
    REWARD_AVAILABLE = False

# Later in code
if REWARD_AVAILABLE and config.reward:
    metrics = compute_run_metrics(...)
```

## Testing Integration

After integrating components, verify with:

```bash
# 1. Run existing tests (should still pass)
poetry run pytest

# 2. Run with new features enabled
make backtest ARGS="--universe-mode dynamic --embargo-days 5"

# 3. Check artifacts
ls runs/<latest>/
# Should contain: results.json, equity_curve.csv, orders.csv, breaches.csv, etc.

# 4. Generate report
make report
```

## Migration Checklist

- [ ] Add reward tracking to BacktestEngine (optional, config-gated)
- [ ] Replace fixed sizing with policy-based sizing
- [ ] Integrate risk guards with kill-switch
- [ ] Add dynamic universe selection to walk-forward CLI
- [ ] Replace naive splits with purged+embargo splits
- [ ] (Optional) Add bandit allocation
- [ ] Update tests to cover new code paths
- [ ] Verify backward compatibility (old commands still work)
- [ ] Update LOCAL_RUN.md with new flags

## Example: Minimal Integration

For a minimal integration that adds **just the reward function**, modify `backtest.py`:

```python
# Add at top
from traderbot.engine.reward import compute_run_metrics, RewardWeights

# In BacktestEngine.run(), before return:
if hasattr(config, "reward") and config.reward:
    weights = RewardWeights(
        lambda_dd=config.reward.lambda_dd,
        tau_turnover=config.reward.tau_turnover,
        kappa_breach=config.reward.kappa_breach,
    )
    
    reward_metrics = compute_run_metrics(
        equity_curve=equity_df,
        trades=trades_df,
        breaches=pd.DataFrame(),  # No breaches yet
        weights=weights,
    )
    
    result.metadata["reward_metrics"] = reward_metrics
```

This adds reward computation without changing existing behavior.

## Summary

All new features can be integrated **surgically** without breaking existing functionality. The key is:

1. Use config flags and CLI arguments for opt-in
2. Add new fields to results/artifacts
3. Preserve existing code paths when features disabled
4. Test both old and new behavior

See `PROFIT_AWARE_LEARNING.md` for detailed feature documentation.

