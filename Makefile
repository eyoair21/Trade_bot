# TraderBot Makefile
# Provides convenient targets for development and CI

.PHONY: help install data lint type test coverage demo train train-split demo2 demo3 clean all

# Default target
help:
	@echo "TraderBot Development Targets"
	@echo ""
	@echo "Setup:"
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
	@echo "  make demo        - Run walk-forward with sample universe (static)"
	@echo "  make demo2       - Run walk-forward with dynamic universe"
	@echo "  make demo3       - Run walk-forward with retrain, sizing, calibration"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Remove generated files"
	@echo "  make all        - Run full CI pipeline (lint, type, test, demo)"

# Install dependencies
install:
	poetry install

# Generate sample data
data:
	poetry run python scripts/make_sample_data.py

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
