# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.3] - 2026-01-05

### Added
- `--no-emoji` flag (CLI) and `TRADERBOT_NO_EMOJI=1` env var for cross-platform compatibility.
- "Data Provenance" block in regression reports when fallbacks are used.
- Shared console module: `traderbot/cli/_console.py`.
- Edge-case tests (empty/malformed inputs, NaN/Inf metrics, round-trip).
- `.gitattributes` for consistent line endings across platforms.
- CLI entrypoint integration test.

### Fixed
- Windows console Unicode errors (cp1252) by safe fallback to ASCII.
- Subprocess CLI tests now set PYTHONPATH for module resolution.

### Changed
- Extracted console helpers from `regress.py` to shared `_console.py` module.
- Normalized line endings via `.gitattributes`.

## [0.5.2] - 2026-01-04

### Added
- Performance budgets and regression detection (`traderbot.cli.regress`).
- Baseline management with `compare` and `update-baseline` commands.
- Determinism verification for sweep reruns.
- CI integration for regression checks.

## [0.5.1] - 2026-01-03

### Added
- CI/CD workflows (lint, test, nightly sweep).
- Artifact packing with `scripts/pack_sweep.py`.
- Timing profiling for sweeps (`--time` flag).
- Leaderboard generation with timing summary.

## [0.5.0] - 2026-01-02

### Added
- YAML-driven hyperparameter sweeps (`traderbot.cli.sweep`).
- Parallel execution with configurable workers.
- Leaderboard generation and best run export.

## [0.4.0] - 2026-01-01

### Added
- Reproducibility with `--seed` parameter.
- Run manifest with full configuration capture.
- Run comparison tool (`traderbot.cli.compare_runs`).

## [0.3.0] - 2025-12-30

### Added
- Position sizing policies (fixed, volatility-targeting, Kelly).
- Probability thresholding with optional optimization.
- Execution cost tracking (commission, slippage).
- Calibration metrics (Brier score, ECE).

## [0.2.0] - 2025-12-28

### Added
- PatchTST model for probability-of-uptrend prediction.
- TorchScript export for CPU-optimized inference.
- Dynamic universe selection based on liquidity/volatility.
- News tone scores with FinBERT integration.

## [0.1.0] - 2025-12-25

### Added
- Initial release with walk-forward analysis CLI.
- Momentum strategy implementation.
- Basic backtesting engine with risk management.
- Data adapters and point-in-time store.
- Comprehensive test suite with 70% coverage requirement.
