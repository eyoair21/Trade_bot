# TraderBot Development Checklist

**Prioritized tasks with acceptance criteria and validation steps.**

---

## P0: Critical Gaps (Must Have)

### P0-1: Add .env.example Template
**Owner:** TBD  
**Effort:** S (5 min)  
**Status:** ✅ Template Created (needs manual copy)

**Acceptance Criteria:**
- [x] `env.example.template` exists at repo root (template provided)
- [ ] Copy `env.example.template` to `.env.example` manually
- [x] Contains all configurable variables from `traderbot/config.py`
- [x] Includes comments explaining each variable
- [x] No secrets or real values (placeholders only)
- [x] Documented in `LOCAL_RUN.md`

**Validation:**
```bash
# Copy template to .env.example
cp env.example.template .env.example

# Verify file exists and has expected vars
grep -E "^(DATA_DIR|LOG_LEVEL|DEFAULT_INITIAL_CAPITAL)" .env.example
```

**Files to Create/Modify:**
- `env.example.template` ✅ (created, copy to `.env.example`)

---

### P0-2: Implement Walk-Forward Embargo/Purging
**Owner:** TBD  
**Effort:** M (2-3 hours)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] `split_walkforward()` function in `traderbot/data/calendar.py` or new module
- [ ] Supports `embargo_frac` parameter (default 0.1 = 10%)
- [ ] Supports `purge_days` parameter (default 0)
- [ ] Returns splits with embargo period: `(is_start, is_end, embargo_end, oos_start, oos_end)`
- [ ] Integrated into `traderbot/cli/walkforward.py`
- [ ] Unit tests in `tests/data/test_calendar.py` or new test file
- [ ] Documentation in function docstring

**Validation:**
```bash
# Run walk-forward with embargo
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 \
  --embargo-frac 0.1

# Verify embargo period exists between IS and OOS in results
grep -A 5 "embargo" runs/*/results.json
```

**Files to Create/Modify:**
- `traderbot/data/calendar.py` (add `split_walkforward()`)
- `traderbot/cli/walkforward.py` (integrate embargo)
- `tests/data/test_calendar.py` (add tests)

---

### P0-3: Add Make Targets for One-Shot Demo
**Owner:** TBD  
**Effort:** S (1 hour)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] `make setup` creates venv/installs deps (or uses Poetry)
- [ ] `make data` generates cached sample data (if missing)
- [ ] `make backtest` runs fast demo backtest (1-3 min)
- [ ] `make report` outputs plots/HTML/CSV to `./artifacts/last_run/`
- [ ] All targets work on x86_64 Linux/macOS/Windows
- [ ] Documented in `LOCAL_RUN.md`

**Validation:**
```bash
# Fresh clone, run all targets
make setup && make data && make backtest && make report
ls -la artifacts/last_run/
```

**Files to Create/Modify:**
- `Makefile` (add targets)
- `scripts/quick_backtest.py` (optional, for `make backtest`)

---

### P0-4: Document ARM64/Raspberry Pi Setup
**Owner:** TBD  
**Effort:** M (2 hours)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] Section in `LOCAL_RUN.md` for Raspberry Pi
- [ ] Python version requirements (3.11+)
- [ ] Dependency installation notes (PyTorch ARM64 wheels)
- [ ] Native install path (fallback if Docker fails)
- [ ] Known issues/workarounds
- [ ] Tested on Raspberry Pi 4 (or documented as untested)

**Validation:**
- Manual verification on Raspberry Pi (or documented as "not tested")

**Files to Create/Modify:**
- `LOCAL_RUN.md` (add ARM64 section)

---

## P1: Important Enhancements (Should Have)

### P1-1: Implement GA Optimizer Core
**Owner:** TBD  
**Effort:** L (1-2 days)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] `traderbot/opt/fitness.py` with `evaluate_fitness()` interface
- [ ] `traderbot/opt/ga_lite.py` implements:
  - [ ] Selection (tournament, default size=3)
  - [ ] Crossover (uniform, rate=0.7)
  - [ ] Mutation (gaussian, rate=0.1)
  - [ ] Elitism (top 2 individuals)
- [ ] `traderbot/opt/ga_runner.py` CLI wrapper
- [ ] Unit tests in `tests/opt/test_ga_lite.py`
- [ ] Integration test: optimize simple strategy params
- [ ] Documentation in function docstrings

**Validation:**
```bash
# Run GA optimization
poetry run python -m traderbot.opt.ga_runner \
  --param-space '{"ema_fast": (5, 20), "ema_slow": (20, 50)}' \
  --generations 10 --population-size 20

# Verify best params found
grep "best_params" runs/*/ga_results.json
```

**Files to Create/Modify:**
- `traderbot/opt/fitness.py` (new)
- `traderbot/opt/ga_lite.py` (implement operators)
- `traderbot/opt/ga_runner.py` (new)
- `tests/opt/test_ga_lite.py` (new)

---

### P1-2: Wire GA to Walk-Forward Backtester
**Owner:** TBD  
**Effort:** M (3-4 hours)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] `GARobust.run_with_validation()` calls `run_walkforward()`
- [ ] Fitness uses OOS Sharpe ratio (or configurable metric)
- [ ] Supports per-split fitness aggregation (mean, min, etc.)
- [ ] Returns robustness score across folds
- [ ] Integration test in `tests/opt/test_ga_robust.py`
- [ ] CLI option: `--opt-params` in walkforward CLI

**Validation:**
```bash
# Run robust GA with walk-forward
poetry run python -m traderbot.cli.walkforward \
  --start-date 2023-01-10 --end-date 2023-03-31 \
  --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6 \
  --opt-params '{"ema_fast": (5, 20)}' --ga-generations 5

# Verify optimization ran and improved metrics
grep "best_params" runs/*/results.json
```

**Files to Create/Modify:**
- `traderbot/opt/ga_robust.py` (implement `run_with_validation()`)
- `traderbot/cli/walkforward.py` (add `--opt-params` flag)
- `tests/opt/test_ga_robust.py` (new)

---

### P1-3: Add Docker Support
**Owner:** TBD  
**Effort:** M (3-4 hours)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] `Dockerfile` with multi-stage build
- [ ] `docker-compose.yml` with service definition
- [ ] Volume mounts for `data/`, `runs/`, `.env`
- [ ] ARM64 support (multi-arch base image)
- [ ] `.dockerignore` excludes unnecessary files
- [ ] Documentation in `LOCAL_RUN.md`
- [ ] Test: `docker-compose up` runs demo backtest

**Validation:**
```bash
# Build and run
docker-compose build
docker-compose run traderbot make backtest

# Verify artifacts
ls -la runs/
```

**Files to Create:**
- `Dockerfile` (new)
- `docker-compose.yml` (new)
- `.dockerignore` (new)

---

### P1-4: Enhance Safety Rails
**Owner:** TBD  
**Effort:** M (2-3 hours)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] `DRY_RUN` config flag (default `True`)
- [ ] Kill-switch: file watch or HTTP endpoint
- [ ] Rate limiting decorator for API calls
- [ ] Position size validation before live orders
- [ ] Documentation in `docs/SAFETY.md`
- [ ] Unit tests for safety checks

**Validation:**
```bash
# Test dry-run mode
DRY_RUN=true poetry run python -m traderbot.cli.walkforward ...

# Test kill-switch
touch .kill_switch
# Verify trading halts
```

**Files to Create/Modify:**
- `traderbot/config.py` (add `DRY_RUN` flag)
- `traderbot/engine/safety.py` (new, kill-switch, throttle)
- `docs/SAFETY.md` (new)

---

## P2: Nice to Have (Future Work)

### P2-1: Live Broker Adapters
**Owner:** TBD  
**Effort:** L (3-5 days per adapter)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] Alpaca adapter (`brokers/alpaca.py`)
- [ ] IBKR adapter (`brokers/ibkr.py`)
- [ ] Polygon.io adapter (`brokers/polygon.py`)
- [ ] CCXT adapter (`brokers/ccxt_adapter.py`)
- [ ] Unified interface (`BrokerAdapter` ABC)
- [ ] Paper-trade mode for all adapters
- [ ] Integration tests (mocked APIs)

**Validation:**
- Manual testing with paper accounts
- Integration tests with mocked responses

**Files to Create:**
- `traderbot/brokers/__init__.py` (new)
- `traderbot/brokers/base.py` (new, ABC)
- `traderbot/brokers/alpaca.py` (new)
- `traderbot/brokers/ibkr.py` (new)
- `traderbot/brokers/polygon.py` (new)
- `traderbot/brokers/ccxt_adapter.py` (new)

---

### P2-2: Event Bars & Labeling Framework
**Owner:** TBD  
**Effort:** L (2-3 days)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] `features/event_bars.py` for bar generation
- [ ] `features/labeling.py` for meta-labeling
- [ ] Supports triple-barrier, fixed-horizon labels
- [ ] Integration with backtest engine
- [ ] Unit tests

**Files to Create:**
- `traderbot/features/event_bars.py` (new)
- `traderbot/features/labeling.py` (new)

---

### P2-3: Multi-Objective Optimization (Pareto)
**Owner:** TBD  
**Effort:** L (2-3 days)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] NSGA-II or similar algorithm
- [ ] Pareto front reporting
- [ ] Multi-objective fitness (Sharpe, max DD, etc.)
- [ ] Visualization of Pareto front
- [ ] Integration with GA runner

**Files to Create/Modify:**
- `traderbot/opt/nsga2.py` (new)
- `traderbot/opt/ga_runner.py` (add Pareto mode)

---

### P2-4: Architecture Documentation
**Owner:** TBD  
**Effort:** M (2-3 hours)  
**Status:** ❌ Not Started

**Acceptance Criteria:**
- [ ] `docs/ARCHITECTURE.md` with system diagram
- [ ] Data flow diagram
- [ ] Component descriptions
- [ ] Integration points

**Files to Create:**
- `docs/ARCHITECTURE.md` (new)

---

## Testing & Validation

### Test Coverage
- [ ] Maintain ≥70% coverage (current baseline)
- [ ] Add tests for all new features
- [ ] Integration tests for E2E flows

### CI/CD
- [ ] Document local CI steps (if no GitHub Actions)
- [ ] Add pre-commit hooks validation
- [ ] Add performance regression tests

---

## Notes

- **Effort Estimates:**
  - S = Small (30 min - 2 hours)
  - M = Medium (2-4 hours)
  - L = Large (1-3 days)

- **Status Legend:**
  - ❌ Not Started
  - 🟡 In Progress
  - ✅ Complete

- **Priority:**
  - P0 = Critical, blocks production use
  - P1 = Important, enhances functionality
  - P2 = Nice to have, future enhancements

