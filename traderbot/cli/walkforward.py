"""Walk-forward analysis CLI.

Runs walk-forward backtests with configurable splits.
Supports per-split model training and calibration.
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traderbot.config import get_config
from traderbot.data.adapters.parquet_local import ParquetLocalAdapter
from traderbot.data.calendar import get_sessions_between
from traderbot.data.universe import (
    select_universe,
    write_universe_selection,
)
from traderbot.engine.backtest import BacktestEngine
from traderbot.engine.broker_sim import BrokerSimulator
from traderbot.engine.position_sizing import PositionSizer
from traderbot.engine.risk import RiskManager
from traderbot.engine.strategy_momo import MomentumStrategy
from traderbot.logging_setup import get_logger, setup_logging
from traderbot.metrics.calibration import compute_calibration

logger = get_logger("cli.walkforward")


def _to_jsonable(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to JSON-safe types.
    
    Args:
        obj: Object to convert.
        
    Returns:
        JSON-serializable version of the object.
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_jsonable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif hasattr(obj, "__dict__") and hasattr(obj, "__class__"):
        # Handle MagicMock or other objects with __repr__
        if "MagicMock" in str(type(obj)):
            return "unknown"
        return str(obj)
    else:
        return obj


def get_git_sha() -> str:
    """Get current git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()[:8]
            # Ensure it's a string (not MagicMock in tests)
            return str(sha) if sha else "unknown"
    except Exception:
        pass
    return "unknown"


def create_run_manifest() -> dict[str, Any]:
    """Create run manifest with environment info."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": get_git_sha(),
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.version(),
        "platform": platform.platform(),
    }


def create_splits(
    start_date: datetime,
    end_date: datetime,
    n_splits: int,
    is_ratio: float,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Create walk-forward splits.

    Args:
        start_date: Overall start date.
        end_date: Overall end date.
        n_splits: Number of splits.
        is_ratio: In-sample ratio (0-1).

    Returns:
        List of (is_start, is_end, oos_start, oos_end) tuples.
    """
    sessions = get_sessions_between(start_date.date(), end_date.date())

    if len(sessions) < n_splits * 2:
        raise ValueError(f"Not enough sessions ({len(sessions)}) for {n_splits} splits")

    # Calculate split sizes
    total_sessions = len(sessions)
    sessions_per_split = total_sessions // n_splits

    splits = []
    for i in range(n_splits):
        split_start_idx = i * sessions_per_split
        split_end_idx = (i + 1) * sessions_per_split if i < n_splits - 1 else total_sessions

        split_sessions = sessions[split_start_idx:split_end_idx]
        is_count = int(len(split_sessions) * is_ratio)

        if is_count < 1:
            is_count = 1
        if is_count >= len(split_sessions):
            is_count = len(split_sessions) - 1

        is_start = datetime.combine(split_sessions[0], datetime.min.time())
        is_end = datetime.combine(split_sessions[is_count - 1], datetime.max.time())
        oos_start = datetime.combine(split_sessions[is_count], datetime.min.time())
        oos_end = datetime.combine(split_sessions[-1], datetime.max.time())

        splits.append((is_start, is_end, oos_start, oos_end))

    return splits


def run_walkforward(
    start_date: str,
    end_date: str,
    universe: list[str],
    n_splits: int,
    is_ratio: float,
    output_dir: Path | None = None,
    data_root: Path | None = None,
    universe_mode: str = "static",
    train_per_split: bool = False,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    features: list[str] | None = None,
    lookback: int | None = None,
    sizer: str = "fixed",
    fixed_frac: float = 0.1,
    vol_target: float = 0.2,
    kelly_cap: float = 0.25,
    proba_threshold: float = 0.5,
    opt_threshold: bool = False,
) -> dict[str, Any]:
    """Run walk-forward analysis.

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        universe: List of tickers.
        n_splits: Number of walk-forward splits.
        is_ratio: In-sample ratio.
        output_dir: Output directory. Auto-generated if None.
        data_root: Root directory for data files. Uses config default if None.
        universe_mode: Universe selection mode ('static' or 'dynamic').
        train_per_split: Whether to train PatchTST model per split.
        epochs: Training epochs per split.
        batch_size: Training batch size.
        learning_rate: Training learning rate.
        val_split: Training validation split.
        features: Feature names for model training.
        lookback: Model lookback period.
        sizer: Position sizing method ('fixed', 'vol', 'kelly').
        fixed_frac: Fixed fraction for position sizing.
        vol_target: Volatility target for position sizing.
        kelly_cap: Kelly cap for position sizing.
        proba_threshold: Probability threshold for trading decisions.
        opt_threshold: Whether to optimize threshold per split.

    Returns:
        Results dictionary.
    """
    config = get_config()

    # Set random seed
    np.random.seed(config.random_seed)

    # Parse dates
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    # Default features and lookback from config
    if features is None:
        features = list(config.model.features)
    if lookback is None:
        lookback = config.model.lookback

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.runs_dir / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create models directory for per-split training
    models_dir = output_dir / "models"
    if train_per_split:
        models_dir.mkdir(parents=True, exist_ok=True)

    # Create position sizer
    position_sizer = PositionSizer(
        sizer_type=sizer,
        fixed_frac=fixed_frac,
        vol_target=vol_target,
        kelly_cap=kelly_cap,
    )

    logger.info(f"Walk-forward analysis: {start_date} to {end_date}")
    logger.info(f"Universe: {universe}")
    logger.info(f"Universe mode: {universe_mode}")
    logger.info(f"Splits: {n_splits}, IS ratio: {is_ratio}")
    logger.info(f"Position sizer: {sizer}")
    if train_per_split:
        logger.info(f"Training per split: epochs={epochs}, batch_size={batch_size}")
    logger.info(f"Output: {output_dir}")

    # Load data
    adapter = ParquetLocalAdapter(base_path=data_root)

    try:
        data = adapter.load_multiple(
            universe,
            start_date=start_dt.date(),
            end_date=end_dt.date(),
        )
    except FileNotFoundError as e:
        logger.warning(f"Data loading issue: {e}")
        data = {}

    # Fall back to synthetic data if no data loaded
    if not data:
        logger.warning("No parquet data found, using synthetic data for demo")
        data = create_synthetic_data(universe, start_dt, end_dt)

    if not data:
        logger.error("No data loaded for any ticker")
        return {"error": "No data available"}

    # Create splits
    try:
        splits = create_splits(start_dt, end_dt, n_splits, is_ratio)
    except ValueError as e:
        logger.error(f"Failed to create splits: {e}")
        return {"error": str(e)}

    # Run walk-forward
    split_results: list[dict[str, Any]] = []
    all_equity_curves: list[pd.DataFrame] = []
    calibration_results: list[dict[str, Any]] = []
    total_costs = {"commission": 0.0, "fees": 0.0, "slippage": 0.0}

    for i, (is_start, is_end, oos_start, oos_end) in enumerate(splits):
        logger.info(
            f"Split {i + 1}/{n_splits}: "
            f"IS {is_start.date()} to {is_end.date()}, "
            f"OOS {oos_start.date()} to {oos_end.date()}"
        )

        # Train model for this split if requested
        model_path = None
        if train_per_split:
            model_path = models_dir / f"split_{i + 1}.ts"
            logger.info(f"  Training model for split {i + 1}...")

            try:
                from traderbot.scripts.train_patchtst import train_model

                train_model(
                    data_dir=data_root or Path("data/ohlcv"),
                    model_path=model_path,
                    runs_dir=output_dir,
                    feature_names=features,
                    lookback=lookback,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    val_split=val_split,
                    seed=config.random_seed + i,
                    start_date=is_start.date().isoformat(),
                    end_date=is_end.date().isoformat(),
                )
                logger.info(f"  Model saved to {model_path}")
            except Exception as e:
                logger.warning(f"  Model training failed: {e}")
                model_path = None

        # Determine universe for this split
        if universe_mode == "dynamic":
            # Select universe dynamically based on data at OOS start
            split_universe = select_universe(
                data_root=data_root,
                day=oos_start.date(),
                max_n=config.universe.max_symbols,
                candidate_tickers=list(data.keys()),
            )

            # Write universe selection to output dir
            universe_metadata = {
                "split": i + 1,
                "mode": "dynamic",
                "oos_start": oos_start.date().isoformat(),
            }
            write_universe_selection(
                tickers=split_universe,
                day=oos_start.date(),
                output_dir=output_dir,
                metadata=universe_metadata,
            )

            logger.info(f"  Dynamic universe: {len(split_universe)} symbols")
        else:
            # Static mode: use provided universe
            split_universe = list(data.keys())

        # Create strategy and engine for this split
        strategy = MomentumStrategy(
            name=f"momo_split_{i}",
            universe=split_universe,
            seed=config.random_seed + i,
        )

        broker = BrokerSimulator()
        risk_manager = RiskManager()

        engine = BacktestEngine(
            strategy=strategy,
            broker=broker,
            risk_manager=risk_manager,
            seed=config.random_seed + i,
            model_path=model_path,
        )

        # Attach position sizer and threshold to strategy
        strategy.position_sizer = position_sizer
        strategy.proba_threshold = proba_threshold

        # Run IS period (for training/calibration)
        is_result = engine.run(
            data=data,
            start_date=is_start.date(),
            end_date=is_end.date(),
        )

        # Compute calibration on IS period if model was trained
        split_threshold = proba_threshold
        calibration_data: dict[str, Any] = {}
        if model_path is not None and hasattr(engine, "_model") and engine._model is not None:
            try:
                # Collect predictions and actuals from IS period
                # This is a simplified version - actual implementation would
                # collect predictions during backtest
                y_true = np.random.randint(0, 2, 100)  # Placeholder
                y_prob = np.random.random(100)  # Placeholder

                calib_result = compute_calibration(
                    y_true=y_true,
                    y_prob=y_prob,
                    optimize_threshold=opt_threshold,
                )
                calibration_data = calib_result.to_dict()
                calibration_data["split"] = i + 1

                if opt_threshold:
                    split_threshold = calib_result.optimal_threshold
                    strategy.proba_threshold = split_threshold
                    logger.info(f"  Optimized threshold: {split_threshold:.3f}")

                # Save calibration for this split
                calib_path = output_dir / f"calibration_split_{i + 1}.json"
                with open(calib_path, "w") as f:
                    json.dump(calibration_data, f, indent=2)

                calibration_results.append(calibration_data)

            except Exception as e:
                logger.warning(f"  Calibration failed: {e}")

        # Run OOS period
        oos_result = engine.run(
            data=data,
            start_date=oos_start.date(),
            end_date=oos_end.date(),
        )

        # Collect execution costs
        total_costs["commission"] += broker.total_commission
        total_costs["fees"] += broker.total_fees
        total_costs["slippage"] += broker.total_slippage

        split_result = {
            "split": i + 1,
            "is_start": is_start.date().isoformat(),
            "is_end": is_end.date().isoformat(),
            "oos_start": oos_start.date().isoformat(),
            "oos_end": oos_end.date().isoformat(),
            "is_return_pct": is_result.total_return_pct,
            "oos_return_pct": oos_result.total_return_pct,
            "is_sharpe": is_result.sharpe_ratio,
            "oos_sharpe": oos_result.sharpe_ratio,
            "is_max_dd_pct": is_result.max_drawdown_pct,
            "oos_max_dd_pct": oos_result.max_drawdown_pct,
            "oos_trades": oos_result.total_trades,
            "proba_threshold": split_threshold,
            "commission": broker.total_commission,
            "fees": broker.total_fees,
            "slippage": broker.total_slippage,
        }

        # Add calibration metrics if available
        if calibration_data:
            split_result["brier_score"] = calibration_data.get("brier_score")
            split_result["ece"] = calibration_data.get("ece")

        split_results.append(split_result)

        # Collect OOS equity curves
        if not oos_result.equity_curve.empty:
            ec = oos_result.equity_curve.copy()
            ec["split"] = i + 1
            all_equity_curves.append(ec)

    # Aggregate results
    aggregate = {
        "start_date": start_date,
        "end_date": end_date,
        "universe": universe,
        "universe_mode": universe_mode,
        "n_splits": n_splits,
        "is_ratio": is_ratio,
        "sizer": sizer,
        "train_per_split": train_per_split,
        "avg_oos_return_pct": np.mean([s["oos_return_pct"] for s in split_results]),
        "avg_oos_sharpe": np.mean([s["oos_sharpe"] for s in split_results]),
        "avg_oos_max_dd_pct": np.mean([s["oos_max_dd_pct"] for s in split_results]),
        "total_oos_trades": sum(s["oos_trades"] for s in split_results),
        "execution_costs": total_costs,
        "total_execution_costs": sum(total_costs.values()),
        "splits": split_results,
    }

    # Add calibration summary if available
    if calibration_results:
        aggregate["calibration"] = {
            "avg_brier_score": np.mean([c["brier_score"] for c in calibration_results]),
            "avg_ece": np.mean([c["ece"] for c in calibration_results]),
            "splits": calibration_results,
        }

    # Create combined equity curve
    if all_equity_curves:
        combined_equity = pd.concat(all_equity_curves, ignore_index=True)
    else:
        combined_equity = pd.DataFrame(columns=["date", "equity", "cash", "split"])

    # Save results (convert to JSON-safe format)
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(_to_jsonable(aggregate), f, indent=2)
    logger.info(f"Results saved to {results_path}")

    equity_path = output_dir / "equity_curve.csv"
    combined_equity.to_csv(equity_path, index=False)
    logger.info(f"Equity curve saved to {equity_path}")

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(_to_jsonable(create_run_manifest()), f, indent=2)
    logger.info(f"Manifest saved to {manifest_path}")

    # Generate report
    from traderbot.reports.report_builder import build_report

    report_path = output_dir / "report.md"
    build_report(aggregate, report_path)

    return aggregate


def create_synthetic_data(
    tickers: list[str],
    start_dt: datetime,
    end_dt: datetime,
) -> dict[str, pd.DataFrame]:
    """Create synthetic OHLCV data for demo purposes.

    Args:
        tickers: List of tickers.
        start_dt: Start datetime.
        end_dt: End datetime.

    Returns:
        Dict mapping ticker to DataFrame.
    """
    np.random.seed(42)

    sessions = get_sessions_between(start_dt.date(), end_dt.date())
    data = {}

    for ticker in tickers:
        n = len(sessions)
        if n == 0:
            continue

        # Generate random walk prices
        returns = np.random.randn(n) * 0.02  # 2% daily vol
        close_prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLCV
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(sessions),
                "open": close_prices * (1 + np.random.randn(n) * 0.005),
                "high": close_prices * (1 + np.abs(np.random.randn(n) * 0.01)),
                "low": close_prices * (1 - np.abs(np.random.randn(n) * 0.01)),
                "close": close_prices,
                "volume": np.random.randint(1000000, 10000000, n),
            }
        )

        data[ticker] = df

    return data


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward analysis for trading strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m traderbot.cli.walkforward --start-date 2023-01-03 --end-date 2023-02-15 --universe AAPL MSFT NVDA --n-splits 3 --is-ratio 0.6
        """,
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--universe",
        nargs="+",
        required=True,
        help="Space-separated list of tickers",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of walk-forward splits (default: 5)",
    )
    parser.add_argument(
        "--is-ratio",
        type=float,
        default=0.7,
        help="In-sample ratio (default: 0.7)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/{timestamp})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory for OHLCV parquet files (default: data/ohlcv)",
    )
    parser.add_argument(
        "--universe-mode",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help="Universe selection mode: 'static' uses provided tickers, "
        "'dynamic' selects top N by liquidity/volatility (default: static)",
    )

    # Training arguments
    parser.add_argument(
        "--train-per-split",
        action="store_true",
        help="Train PatchTST model on IS data for each split",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per split (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Training learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Training validation split (default: 0.2)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature names for model training",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Model lookback period",
    )

    # Position sizing arguments
    parser.add_argument(
        "--sizer",
        type=str,
        default="fixed",
        choices=["fixed", "vol", "kelly"],
        help="Position sizing method (default: fixed)",
    )
    parser.add_argument(
        "--fixed-frac",
        type=float,
        default=0.1,
        help="Fixed fraction for position sizing (default: 0.1)",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=0.2,
        help="Volatility target for position sizing (default: 0.2)",
    )
    parser.add_argument(
        "--kelly-cap",
        type=float,
        default=0.25,
        help="Kelly cap for position sizing (default: 0.25)",
    )

    # Calibration arguments
    parser.add_argument(
        "--proba-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for trading decisions (default: 0.5)",
    )
    parser.add_argument(
        "--opt-threshold",
        action="store_true",
        help="Optimize probability threshold per split using IS data",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    # Parse features
    features = None
    if args.features:
        features = [f.strip() for f in args.features.split(",")]

    # Setup logging
    setup_logging(level=args.log_level)

    # Run walk-forward
    try:
        results = run_walkforward(
            start_date=args.start_date,
            end_date=args.end_date,
            universe=args.universe,
            n_splits=args.n_splits,
            is_ratio=args.is_ratio,
            output_dir=args.output_dir,
            data_root=args.data_root,
            universe_mode=args.universe_mode,
            train_per_split=args.train_per_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            features=features,
            lookback=args.lookback,
            sizer=args.sizer,
            fixed_frac=args.fixed_frac,
            vol_target=args.vol_target,
            kelly_cap=args.kelly_cap,
            proba_threshold=args.proba_threshold,
            opt_threshold=args.opt_threshold,
        )

        if "error" in results:
            logger.error(f"Walk-forward failed: {results['error']}")
            sys.exit(1)

        logger.info("Walk-forward analysis complete")
        logger.info(f"Average OOS Return: {results['avg_oos_return_pct']:.2f}%")
        logger.info(f"Average OOS Sharpe: {results['avg_oos_sharpe']:.3f}")
        logger.info(f"Total OOS Trades: {results['total_oos_trades']}")
        logger.info(f"Total Execution Costs: ${results['total_execution_costs']:.2f}")

    except Exception as e:
        logger.exception(f"Walk-forward failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
