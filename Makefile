# TraderBot Makefile
# Provides convenient targets for development and CI

.PHONY: help setup install data lint type test coverage demo train train-split demo2 demo3 backtest report opt sweep clean all

# Default target
help:
	@echo "TraderBot Development Targets"
	@echo ""
	@echo "Setup:"
	@echo "  make setup      - One-time setup (install deps, create .env if missing)"
	@echo "  make install    - Install dependencies via Poetry"
	@echo ""
	@echo "Data:"
	@echo "  make data       - Generate sample OHLCV data"
	@echo ""
	@echo "Quality:"
	@echo "  make lint       - Run ruff linter"
	@echo "  make format     - Run black formatter (check mode)"
	@echo "  make type       - Run mypy type checker"
	@echo "  make test       - Run lint + type + tests with coverage"
	@echo "  make coverage   - Run tests with coverage report"
	@echo ""
	@echo "Model:"
	@echo "  make train       - Train PatchTST model on sample data"
	@echo "  make train-split - Train PatchTST on specific date range"
	@echo ""
	@echo "Run:"
	@echo "  make backtest   - Run quick demo backtest (1-3 min, outputs to artifacts/)"
	@echo "  make demo        - Run walk-forward with sample universe (static)"
	@echo "  make demo2       - Run walk-forward with dynamic universe"
	@echo "  make demo3       - Run walk-forward with retrain, sizing, calibration"
	@echo "  make opt         - Run GA hyperparameter optimization (ARGS for extra params)"
	@echo "  make sweep       - Run parameter sweep and generate leaderboard (ARGS for budget)"
	@echo "  make report      - Generate report from last run (outputs to artifacts/last_run/)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Remove generated files"
	@echo "  make all        - Run full CI pipeline (lint, type, test, demo)"

# One-time setup: install deps and create .env if missing
setup: install
	@if [ ! -f .env ]; then \
		if [ -f .env.example ]; then \
			cp .env.example .env; \
			echo "Created .env from .env.example"; \
		else \
			echo "Warning: .env.example not found, creating minimal .env"; \
			echo "# TraderBot Configuration" > .env; \
			echo "DATA_DIR=./data" >> .env; \
			echo "OHLCV_DIR=./data/ohlcv" >> .env; \
			echo "LOG_LEVEL=INFO" >> .env; \
			echo "RANDOM_SEED=42" >> .env; \
		fi; \
	fi

# Install dependencies
install:
	poetry install

# Generate sample data (only if missing)
data:
	@if [ ! -f data/ohlcv/AAPL.parquet ]; then \
		echo "Generating sample data..."; \
		poetry run python scripts/make_sample_data.py --synthetic-only; \
	else \
		echo "Sample data already exists, skipping generation"; \
	fi

# Linting
lint:
	poetry run ruff check .

# Formatting check
format:
	poetry run black --check .

# Type checking
type:
	poetry run mypy traderbot

# Run tests with coverage
coverage:
	poetry run pytest --cov=traderbot --cov-report=term-missing --cov-fail-under=70 -q

# Full test suite (lint + type + tests)
test: lint format type coverage

# Train PatchTST model
train: data
	poetry run python scripts/train_patchtst.py \
		--data-dir data/ohlcv \
		--epochs 30 \
		--batch-size 16

# Train PatchTST on specific date range (for testing per-split training)
train-split: data
	poetry run python scripts/train_patchtst.py \
		--data-dir data/ohlcv \
		--epochs 10 \
		--batch-size 16 \
		--start-date 2023-01-01 \
		--end-date 2023-02-28 \
		--out models/split_test.ts

# Run walk-forward demo (static universe)
demo: data
	poetry run python -m traderbot.cli.walkforward \
		--start-date 2023-01-10 \
		--end-date 2023-03-31 \
		--universe AAPL MSFT NVDA \
		--n-splits 3 \
		--is-ratio 0.6 \
		--universe-mode static

# Run walk-forward demo (dynamic universe)
demo2: data
	poetry run python -m traderbot.cli.walkforward \
		--start-date 2023-01-10 \
		--end-date 2023-03-31 \
		--universe AAPL MSFT NVDA \
		--n-splits 3 \
		--is-ratio 0.6 \
		--universe-mode dynamic

# Run walk-forward demo with Phase 3 features (retrain, sizing, calibration)
demo3: data
	poetry run python -m traderbot.cli.walkforward \
		--start-date 2023-01-10 \
		--end-date 2023-03-31 \
		--universe AAPL MSFT NVDA \
		--n-splits 3 \
		--is-ratio 0.6 \
		--universe-mode static \
		--sizer vol \
		--vol-target 0.15 \
		--proba-threshold 0.55 \
		--opt-threshold

# Quick backtest demo (fast, 1-3 minutes)
backtest: data
	@mkdir -p artifacts/last_run
	@echo "Running quick backtest demo..."
	@poetry run python -m traderbot.cli.walkforward \
		--start-date 2023-01-10 \
		--end-date 2023-02-28 \
		--universe AAPL MSFT \
		--n-splits 2 \
		--is-ratio 0.6 \
		--universe-mode static \
		--output-dir artifacts/last_run || true
	@if [ -d artifacts/last_run ]; then \
		echo ""; \
		echo "Backtest complete! Results in artifacts/last_run/"; \
		if [ -f artifacts/last_run/results.json ]; then \
			echo ""; \
			echo "Summary:"; \
			poetry run python -c "import json; d=json.load(open('artifacts/last_run/results.json')); print(f\"  Avg OOS Return: {d.get('avg_oos_return_pct', 0):.2f}%\"); print(f\"  Avg OOS Sharpe: {d.get('avg_oos_sharpe', 0):.3f}\"); print(f\"  Total Trades: {d.get('total_oos_trades', 0)}\")" 2>/dev/null || echo "  (Install jq or python to view summary)"; \
		fi; \
	fi

# Run GA hyperparameter optimization
opt: data
	@echo "Running GA hyperparameter optimization..."
	@mkdir -p runs/ga_opt
	@poetry run python -m traderbot.cli.ga_optimize \
		--start-date 2023-01-10 \
		--end-date 2023-03-31 \
		--universe AAPL MSFT NVDA \
		--n-splits 2 \
		--population-size 10 \
		--n-generations 5 \
		--objective total_reward \
		--output-dir runs/ga_opt \
		$(ARGS) || echo "Note: GA optimization requires ga_optimize CLI (see opt/ga_opt.py)"
	@if [ -f runs/ga_opt/best_config.json ]; then \
		echo ""; \
		echo "Optimization complete! Best config:"; \
		cat runs/ga_opt/best_config.json; \
		echo ""; \
		echo "Replay with: ./runs/ga_opt/replay.sh"; \
	fi

# Run parameter sweep and generate leaderboard
sweep: data
	@echo "Running parameter sweep..."
	@poetry run python -m traderbot.cli.sweep \
		--start-date 2023-01-10 \
		--end-date 2023-03-31 \
		--universe AAPL MSFT NVDA \
		--mode random \
		--budget 20 \
		--metric total_reward \
		--output-dir runs/sweep \
		$(ARGS)
	@if [ -f runs/sweep/leaderboard.csv ]; then \
		echo ""; \
		echo "Sweep complete! Top 5:"; \
		head -n 6 runs/sweep/leaderboard.csv | column -t -s,; \
	fi

# Generate report from last run
report:
	@if [ -d runs ]; then \
		LATEST_RUN=$$(ls -td runs/*/ 2>/dev/null | head -1); \
		if [ -n "$$LATEST_RUN" ]; then \
			echo "Generating report for: $$LATEST_RUN"; \
			poetry run python -c "from pathlib import Path; from traderbot.reporting.report import generate_report; generate_report(Path('$$LATEST_RUN'))"; \
			echo "Report generated in: $${LATEST_RUN}report/"; \
			if [ -f "$${LATEST_RUN}report/report.md" ]; then \
				echo ""; \
				head -n 30 "$${LATEST_RUN}report/report.md"; \
			fi; \
		else \
			echo "No runs found. Run 'make backtest' first."; \
		fi; \
	else \
		echo "No runs found. Run 'make backtest' first."; \
	fi

# Clean generated files
clean:
	rm -rf data/ohlcv/*.parquet
	rm -rf models/*.ts
	rm -rf runs/
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Full CI pipeline
all: test demo
	@echo ""
	@echo "All checks passed!"
