# Command Recap: Phase 1 + Phase 2

**Complete command reference for TraderBot with profit-aware learning**

---

## 🚀 SETUP (One-Time)

```bash
# 1. Install dependencies
cd traderbot
poetry install

# 2. Add Phase 2 dependency (scikit-learn)
poetry add scikit-learn

# 3. Generate sample data
make data

# 4. Create .env from template
cp env.example.profit_aware .env

# 5. Verify installation
poetry run pytest tests/ -v --tb=short
```

**Expected**: All tests pass (80+ tests)

---

## ✅ PHASE 1: Basic Profit-Aware Features

### Run Standard Backtest

```bash
make backtest

# Or with custom params:
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA \
  --n-splits 3
```

### Run with Dynamic Universe

```bash
make backtest ARGS="--universe-mode dynamic --top-n 50"
```

### Run with Leakage Controls

```bash
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-06-30 \
  --universe AAPL MSFT NVDA \
  --embargo-days 5 \
  --purge 1d
```

### Run with Position Sizing

```bash
poetry run python -m traderbot.cli.walkforward \
  --sizer vol \
  --vol-target 0.15 \
  --max-leverage 1.0
```

### Run with Bandit Allocation

```bash
poetry run python -m traderbot.cli.walkforward \
  --alloc bandit
```

### Generate Report

```bash
make report

# Or manually:
poetry run python -c "from pathlib import Path; from traderbot.reporting.report import generate_report; generate_report(Path('runs/<latest>'))"
```

---

## ⚡ PHASE 2: Online Adaptation & Advanced Features

### Test Phase 2 Components

```bash
# All Phase 2 tests
poetry run pytest tests/phase2/ -v

# Specific components
poetry run pytest tests/phase2/test_reward_adapter.py -v
poetry run pytest tests/phase2/test_regime_router.py -v
poetry run pytest tests/phase2/test_portfolio_limits.py -v
poetry run pytest tests/phase2/test_costs_model.py -v
poetry run pytest tests/phase2/test_contextual_bandit.py -v
poetry run pytest tests/phase2/test_walkforward_integration.py -v
```

### Quick Component Demos

```bash
# Reward adaptation
python -c "from traderbot.engine.reward_adapter import RewardAdapter; adapter = RewardAdapter.from_config(mode='ewma'); print('✓ Works')"

# Regime detection
python -c "from traderbot.research.regime import detect_regimes; import pandas as pd; import numpy as np; df = pd.DataFrame({'returns': np.random.randn(100)*0.01, 'volatility': np.abs(np.random.randn(100))*0.02}); regimes = detect_regimes(df, k=3); print(f'✓ Detected {len(regimes.unique())} regimes')"

# Portfolio limits
python -c "from traderbot.engine.portfolio import PortfolioManager, PortfolioLimits; mgr = PortfolioManager(PortfolioLimits(), 100000); print('✓ Works')"

# Cost model
python -c "from traderbot.engine.costs import estimate_costs; costs = estimate_costs(1000, 100.0); print(f'✓ Costs: ${costs.total:.2f}')"

# Contextual bandit
python -c "from traderbot.alloc.bandit import ContextualThompsonSampling; allocator = ContextualThompsonSampling(['a', 'b']); weights = allocator.get_weights({'regime_id': 0}); print(f'✓ Weights: {weights}')"
```

### Parameter Sweep

```bash
# Default sweep (20 configs)
make sweep

# Custom budget
make sweep ARGS="--budget 30 --metric sharpe"

# View leaderboard
cat runs/sweep/leaderboard.csv | head -10
```

### Acceptance Check

```bash
# 1. Run baseline
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-06-30 \
  --universe AAPL MSFT NVDA \
  --output-dir runs/baseline

# 2. Run with Phase 2 (via orchestrator)
# See PHASE2_INTEGRATION_COMPLETE.md

# 3. Validate
python scripts/accept_phase2.py \
  --baseline runs/baseline/results.json \
  --current runs/phase2/results.json
```

---

## 🔬 DEVELOPMENT COMMANDS

### Linting & Type Checking

```bash
# Lint
make lint

# Format check
make format

# Type check
make type

# All quality checks
make test
```

### Coverage

```bash
# With coverage report
make coverage

# Specific modules
poetry run pytest tests/phase2/ --cov=traderbot.engine --cov=traderbot.research --cov-report=term-missing
```

### Clean

```bash
# Remove generated files
make clean
```

---

## 📦 COMMON WORKFLOWS

### Full Demo (Phase 1 + Phase 2)

```bash
# 1. Setup
make setup
make data

# 2. Run Phase 1 demo
make backtest

# 3. Run Phase 2 tests
poetry run pytest tests/phase2/ -v

# 4. Run sweep
make sweep ARGS="--budget 10"

# 5. Generate report
make report
```

**Time**: ~5-10 minutes

### Research Workflow

```bash
# 1. Generate universe
poetry run python -m traderbot.cli.walkforward \
  --universe-mode dynamic \
  --top-n 100

# 2. Run with multiple configs
make sweep ARGS="--budget 20"

# 3. Analyze best config
cat runs/sweep/leaderboard.csv | head -1

# 4. Replay best
# (see runs/sweep/config_XXX/replay.sh)
```

### Production Workflow

```bash
# 1. Run full validation
poetry run pytest tests/ -v

# 2. Run acceptance check
python scripts/accept_phase2.py \
  --baseline runs/baseline/results.json \
  --current runs/production/results.json

# 3. Generate report
make report

# 4. Archive results
tar -czf results_$(date +%Y%m%d).tar.gz runs/
```

---

## 🐛 DEBUGGING COMMANDS

### Check Environment

```bash
# Python version
python --version

# Dependencies
poetry show

# Config
poetry run python -c "from traderbot.config import get_config; print(get_config())"
```

### Verbose Logging

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Run with verbose output
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-01-31 \
  --universe AAPL \
  --n-splits 1
```

### Check Artifacts

```bash
# List runs
ls -lh runs/

# Check latest run
LATEST=$(ls -td runs/*/ | head -1)
echo "Latest run: $LATEST"
ls -lh $LATEST

# View results
cat ${LATEST}results.json | jq .

# Check Phase 2 artifacts
ls ${LATEST}reward_*.* ${LATEST}regimes.csv ${LATEST}portfolio_stats.json
```

---

## 📊 PERFORMANCE PROFILING

### Time a Run

```bash
time poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-03-31 \
  --universe AAPL MSFT
```

### Memory Profiling

```bash
# Install memory_profiler
poetry add memory_profiler --dev

# Profile
poetry run python -m memory_profiler -m traderbot.cli.walkforward \
  --start-date 2023-01-10 \
  --end-date 2023-01-31 \
  --universe AAPL
```

---

## 🎯 QUICK REFERENCE

| Command | Purpose |
|---------|---------|
| `make setup` | One-time setup |
| `make data` | Generate sample data |
| `make backtest` | Quick backtest (1-3 min) |
| `make sweep` | Parameter search |
| `make report` | Generate report |
| `make test` | Run all tests |
| `make clean` | Clean generated files |
| `poetry run pytest tests/phase2/` | Test Phase 2 |
| `python scripts/accept_phase2.py` | Acceptance check |

---

## 📖 DOCUMENTATION

| File | Command to View |
|------|-----------------|
| README | `cat README.md` |
| Phase 1 Overview | `cat PROFIT_AWARE_LEARNING.md` |
| Phase 1 Integration | `cat INTEGRATION_GUIDE.md` |
| Phase 2 Overview | `cat PHASE2_SUMMARY.md` |
| Phase 2 Quick Start | `cat PHASE2_QUICKSTART.md` |
| Phase 2 Integration | `cat PHASE2_INTEGRATION_COMPLETE.md` |
| Phase 2 Delivery | `cat PHASE2_FINAL_DELIVERY.md` |
| Local Run Guide | `cat LOCAL_RUN.md` |
| Quick Reference | `cat QUICK_REFERENCE.md` |

---

## 💡 TIPS

### Faster Iteration

```bash
# Use fewer splits for testing
--n-splits 2

# Use smaller universe
--universe AAPL MSFT

# Use shorter date range
--start-date 2023-01-10 --end-date 2023-01-31
```

### Debugging Failed Tests

```bash
# Run single test with verbose output
poetry run pytest tests/phase2/test_reward_adapter.py::test_ewma_adaptation_improves -vvs

# Show print statements
poetry run pytest tests/phase2/ -v -s

# Stop on first failure
poetry run pytest tests/phase2/ -x
```

### Git Workflow

```bash
# Check what changed
git status
git diff

# Commit Phase 2
git add .
git commit -m "feat: Add Phase 2 online adaptation and advanced features"

# Create branch for Phase 2.1
git checkout -b phase2.1/live-trading
```

---

**Command Recap Version**: 2.0  
**Last Updated**: 2026-01-27  
**Covers**: Phase 1 + Phase 2 Complete

