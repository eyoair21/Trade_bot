"""Top-level TraderBot CLI (`tb`) for universe selection, ranking, and grids.

This module adds a thin orchestration layer on top of existing walkforward
backtests. It focuses on:

- Named universes (sp500, liquid_top1000) defined in YAML under ``universe/``
- Composite factor ranking with sector caps
- Simple backtest grid runner that shells out to the existing walkforward CLI
- News ingestion and sentiment scoring (RSS-based, no API keys)
- Gap detection and regime classification
- Opportunity scoring with alert artifacts
- Paper trading engine with deterministic fills

The goal is to keep this CLI orchestration-only and avoid touching the core
engine logic wherever possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd
import yaml

from traderbot import __version__
from traderbot.config import get_config
from traderbot.data.adapters.parquet_local import ParquetLocalAdapter
from traderbot.features.factors import (
    CompositeWeights,
    apply_sector_caps,
    build_factor_matrix,
    compute_composite_scores,
    standardize_zscores,
)
from traderbot.logging_setup import get_logger, setup_logging
from traderbot.reports.metrics import max_drawdown, sharpe_simple, total_return

logger = get_logger("cli.tb")


@dataclass
class UniverseScreenConfig:
    """Parsed universe screen from YAML."""

    name: str
    min_price: float
    min_adv: float
    max_spread_pct: float
    min_history_days: int
    max_symbols: int | None = None
    base_symbols: List[str] | None = None


def _load_yaml_universe(name: str) -> UniverseScreenConfig:
    """Load universe YAML by name from ``traderbot/universe``."""
    project_root = Path(__file__).resolve().parents[2]
    universe_dir = project_root / "universe"
    path = universe_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Universe YAML not found: {path}")

    data = yaml.safe_load(path.read_text())
    screens = data.get("screens", {})
    base = data.get("base", {})

    return UniverseScreenConfig(
        name=data.get("name", name),
        min_price=float(screens.get("min_price", 5.0)),
        min_adv=float(screens.get("min_adv", 20_000_000.0)),
        max_spread_pct=float(screens.get("max_spread_pct", 0.002)),
        min_history_days=int(screens.get("min_history_days", 756)),
        max_symbols=int(screens["max_symbols"]) if "max_symbols" in screens else None,
        base_symbols=list(base.get("symbols") or []) or None,
    )


def _screen_universe(
    cfg: UniverseScreenConfig,
    data_root: Path,
) -> Dict[str, pd.DataFrame]:
    """Apply universe-level screens and return symbol -> OHLCV data."""
    adapter = ParquetLocalAdapter(data_dir=data_root)

    if cfg.base_symbols:
        candidates = cfg.base_symbols
    else:
        candidates = sorted(p.stem for p in data_root.glob("*.parquet"))

    if not candidates:
        logger.warning("No candidate symbols found for universe %s", cfg.name)
        return {}

    screened: Dict[str, pd.DataFrame] = {}

    for symbol in candidates:
        try:
            df = adapter.load(symbol)
        except FileNotFoundError:
            continue
        if df.empty:
            continue

        # Basic 3y history requirement
        if len(df) < cfg.min_history_days:
            continue

        last = df.iloc[-1]
        price = float(last.get("close", 0.0))
        if price < cfg.min_price:
            continue

        # ADV: approximate using last 20d
        adv_df = df.tail(20)
        adv = float((adv_df["close"] * adv_df["volume"]).mean())
        if adv < cfg.min_adv:
            continue

        # Spread proxy
        high = float(last.get("high", price))
        low = float(last.get("low", price))
        spread_pct = (high - low) / price if price > 0 else 0.0
        if spread_pct > cfg.max_spread_pct:
            continue

        screened[symbol] = df

    # Optional max_symbols cut by ADV
    if cfg.max_symbols is not None and len(screened) > cfg.max_symbols:
        adv_by_symbol = {}
        for sym, df in screened.items():
            tail = df.tail(20)
            adv_by_symbol[sym] = float((tail["close"] * tail["volume"]).mean())
        sorted_syms = sorted(adv_by_symbol, key=adv_by_symbol.get, reverse=True)[
            : cfg.max_symbols
        ]
        screened = {s: screened[s] for s in sorted_syms}

    logger.info("Universe %s: %d symbols after screening", cfg.name, len(screened))
    return screened


def _load_sector_map(data_root: Path) -> Dict[str, str]:
    """Load optional sector mapping from CSV at data_root/sector_map.csv.

    If not present, all symbols will be assigned to sector 'UNKNOWN'.
    """
    path = data_root / "sector_map.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "symbol" not in df.columns or "sector" not in df.columns:
        return {}
    return {str(row["symbol"]): str(row["sector"]) for _, row in df.iterrows()}


def rank_universe(
    universe_name: str,
    top_n: int,
    sector_cap: float,
    weights_mapping: Mapping[str, float] | None = None,
    data_root: Path | None = None,
) -> List[str]:
    """Run screening + factor ranking pipeline and return top tickers."""
    cfg = get_config()
    if data_root is None:
        data_root = cfg.data.ohlcv_dir
    data_root = Path(data_root)

    uni_cfg = _load_yaml_universe(universe_name)
    screened = _screen_universe(uni_cfg, data_root=data_root)
    if not screened:
        return []

    factor_matrix = build_factor_matrix(screened)
    z = standardize_zscores(factor_matrix)
    weights = CompositeWeights.from_mapping(weights_mapping)
    scores = compute_composite_scores(z, weights)

    sector_map = _load_sector_map(data_root)
    ranked_symbols = list(scores.index)

    selected = apply_sector_caps(
        ranked_symbols=ranked_symbols,
        sector_map=sector_map,
        top_n=top_n,
        sector_cap=sector_cap,
    )

    logger.info(
        "Universe %s ranking: selected %d/%d symbols (top_n=%d, sector_cap=%.2f)",
        universe_name,
        len(selected),
        len(ranked_symbols),
        top_n,
        sector_cap,
    )
    return selected


# ---------------------------- Grid backtest orchestration ---------------------------- #


def _run_walkforward_subprocess(
    start_date: str,
    end_date: str,
    universe: List[str],
    strategy: str,
    output_dir: Path,
) -> Path:
    """Invoke the existing walkforward CLI in a subprocess."""
    # For now, map all strategies onto the existing momentum strategy.
    # Future extension: wire different strategies based on this flag.
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(project_root)

    universe_args = ["--universe", *universe]
    cmd = [
        sys.executable,
        "-m",
        "traderbot.cli.walkforward",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--n-splits",
        "3",
        "--is-ratio",
        "0.6",
        "--output-dir",
        str(output_dir),
        *universe_args,
    ]

    logger.info("Running walkforward: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("walkforward failed: %s", result.stderr)
        raise RuntimeError(f"walkforward failed ({result.returncode})")

    return output_dir


def _summarize_run(run_dir: Path) -> Dict[str, Any]:
    """Load results and compute simple metrics summary."""
    results_path = run_dir / "results.json"
    equity_path = run_dir / "equity_curve.csv"
    if not results_path.exists() or not equity_path.exists():
        return {}

    with open(results_path) as f:
        results = json.load(f)
    equity_df = pd.read_csv(equity_path)
    equity_series = equity_df["equity"] if "equity" in equity_df.columns else pd.Series([])

    metrics = {
        "total_return": total_return(equity_series),
        "sharpe": sharpe_simple(equity_series),
        "max_dd": max_drawdown(equity_series),
    }
    return {
        "run_dir": str(run_dir),
        "universe": results.get("universe", []),
        "start_date": results.get("start_date"),
        "end_date": results.get("end_date"),
        "strategy": "momentum",
        **metrics,
    }


def cmd_backtest(args: argparse.Namespace) -> None:
    """Entry point for `tb backtest`."""
    setup_logging(level=args.log_level)
    cfg = get_config()

    # Parse weights JSON if provided
    weights_mapping: Dict[str, float] | None = None
    if args.weights:
        weights_mapping = json.loads(args.weights)

    top_n = args.top_n
    sector_cap = args.sector_cap

    # Rank and select universe
    symbols = rank_universe(
        universe_name=args.universe,
        top_n=top_n,
        sector_cap=sector_cap,
        weights_mapping=weights_mapping,
        data_root=cfg.data.ohlcv_dir,
    )
    if not symbols:
        logger.error("No symbols selected for universe %s", args.universe)
        sys.exit(1)

    # Determine date ranges: if not provided, use predefined grid
    if args.start_date and args.end_date:
        periods: List[Tuple[str, str]] = [(args.start_date, args.end_date)]
    else:
        periods = [
            ("2020-01-01", "2022-12-31"),
            ("2023-01-01", "2023-12-31"),
            ("2024-01-01", "2024-12-31"),
        ]

    base_reports = Path(args.reports_dir) if args.reports_dir else cfg.reports_dir
    reports_root = base_reports / "backtests"
    reports_root.mkdir(parents=True, exist_ok=True)

    # Persist latest selection for paper-trade wiring
    selection_path = reports_root / "last_selection.json"
    with open(selection_path, "w") as f:
        json.dump(
            {
                "universe": args.universe,
                "strategy": args.strategy,
                "top_n": top_n,
                "sector_cap": sector_cap,
                "symbols": symbols,
            },
            f,
            indent=2,
        )

    comparison_rows: List[Dict[str, Any]] = []

    for start_date, end_date in periods:
        run_name = f"{args.universe}_{args.strategy}_{start_date}_to_{end_date}"
        run_dir = reports_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        _run_walkforward_subprocess(
            start_date=start_date,
            end_date=end_date,
            universe=symbols,
            strategy=args.strategy,
            output_dir=run_dir,
        )

        summary = _summarize_run(run_dir)
        if not summary:
            continue

        # Save per-run metrics.json
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)

        comparison_rows.append(summary)

    # Build comparison.csv
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_path = reports_root / "comparison.csv"
        comp_df.to_csv(comp_path, index=False)

        # Summary markdown with simple ranking by Sharpe
        comp_sorted = comp_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
        md_lines = [
            "# Backtest Grid Summary",
            "",
            f"- Universe: **{args.universe}**",
            f"- Strategy: **{args.strategy}**",
            "",
            "## Top 3 Configurations (by Sharpe)",
            "",
        ]
        for idx, row in comp_sorted.head(3).iterrows():
            md_lines.append(
                f"{idx + 1}. {row['start_date']} to {row['end_date']} "
                f"- Sharpe {row['sharpe']:.3f}, "
                f"Return {row['total_return']:.2f}%, "
                f"Max DD {row['max_dd']:.2f}%"
            )
        summary_path = reports_root / "summary.md"
        summary_path.write_text("\n".join(md_lines))


def cmd_compare(args: argparse.Namespace) -> None:
    """Entry point for `tb compare` - thin wrapper over comparison.csv."""
    cfg = get_config()
    base_reports = Path(args.reports_dir) if args.reports_dir else cfg.reports_dir
    reports_root = base_reports / "backtests"
    comp_path = reports_root / "comparison.csv"
    if not comp_path.exists():
        print(f"comparison.csv not found at {comp_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(comp_path)
    if df.empty:
        print("No comparison rows found", file=sys.stderr)
        sys.exit(1)

    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    print("Ranked backtest configurations (by Sharpe):\n")
    print(
        df[
            [
                "start_date",
                "end_date",
                "strategy",
                "total_return",
                "sharpe",
                "max_dd",
            ]
        ].to_string(index=False)
    )


# ---------------------------- Paper-trade stubs ---------------------------- #


def _load_last_selection(reports_root: Path) -> Dict[str, Any]:
    """Load last_selection.json produced by tb backtest."""
    path = reports_root / "last_selection.json"
    if not path.exists():
        raise FileNotFoundError(
            f"last_selection.json not found in {reports_root}. "
            "Run `tb backtest` first to generate a selection."
        )
    with open(path) as f:
        return json.load(f)


def cmd_paper_start(args: argparse.Namespace) -> None:
    """Initialize a dry-run paper account using last_selection."""
    cfg = get_config()
    base_reports = Path(args.reports_dir) if args.reports_dir else cfg.reports_dir
    reports_root = base_reports / "backtests"
    selection = _load_last_selection(reports_root)

    state = {
        "account": {
            "equity": cfg.backtest.initial_capital,
            "cash": cfg.backtest.initial_capital,
        },
        "universe": selection.get("symbols", []),
        "open_positions": {},
        "orders": [],
        "fills": [],
    }
    state_path = reports_root / "paper_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Paper account initialized with {len(state['universe'])} symbols at {state_path}")


def cmd_paper_sync(args: argparse.Namespace) -> None:
    """Simulate a sync cycle and record expected vs filled prices for audit."""
    cfg = get_config()
    base_reports = Path(args.reports_dir) if args.reports_dir else cfg.reports_dir
    reports_root = base_reports / "backtests"
    state_path = reports_root / "paper_state.json"
    if not state_path.exists():
        print("paper_state.json not found; run `tb paper-start` first.", file=sys.stderr)
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    universe: List[str] = state.get("universe", [])
    if not universe:
        print("No universe in paper_state; run `tb backtest` then `tb paper-start`.", file=sys.stderr)
        sys.exit(1)

    adapter = ParquetLocalAdapter(base_path=cfg.data.ohlcv_dir)
    slippage_bps = cfg.execution.slippage_bps

    fills: List[Dict[str, Any]] = []
    for symbol in universe:
        try:
            df = adapter.load(symbol)
        except FileNotFoundError:
            continue
        if df.empty:
            continue
        last = df.iloc[-1]
        expected = float(last["close"])
        # Simple one-sided slippage model
        filled = expected * (1 + slippage_bps / 10_000.0)
        fills.append(
            {
                "symbol": symbol,
                "expected_price": expected,
                "filled_price": filled,
                "slippage_bps": slippage_bps,
            }
        )

    state.setdefault("fills", []).extend(fills)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"Recorded {len(fills)} synthetic fills with slippage audit to {state_path}")


def cmd_paper_status(args: argparse.Namespace) -> None:
    """Print current paper account status."""
    cfg = get_config()
    base_reports = Path(args.reports_dir) if args.reports_dir else cfg.reports_dir
    reports_root = base_reports / "backtests"
    state_path = reports_root / "paper_state.json"
    if not state_path.exists():
        print("paper_state.json not found; run `tb paper-start` first.", file=sys.stderr)
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    universe = state.get("universe", [])
    fills = state.get("fills", [])
    account = state.get("account", {})

    print("Paper account status\n")
    print(f"Equity: {account.get('equity', 'N/A')}")
    print(f"Cash:   {account.get('cash', 'N/A')}")
    print(f"Universe size: {len(universe)} symbols")
    print(f"Total synthetic fills recorded: {len(fills)}")


# ---------------------------- News Commands ---------------------------- #


def cmd_news_pull(args: argparse.Namespace) -> None:
    """Fetch RSS news feeds."""
    from traderbot.news.pull import fetch_rss_feeds

    sources_path = Path(args.sources)
    output_path = Path(args.out)

    if not sources_path.exists():
        print(f"Sources file not found: {sources_path}", file=sys.stderr)
        sys.exit(1)

    count = fetch_rss_feeds(sources_path, output_path)
    print(f"Fetched {count} news items to {output_path}")


def cmd_news_parse(args: argparse.Namespace) -> None:
    """Parse raw news JSONL."""
    from traderbot.news.parse import parse_news_items

    input_path = Path(args.input)
    output_path = Path(args.out)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    count = parse_news_items(input_path, output_path)
    print(f"Parsed {count} news items to {output_path}")


def cmd_news_score(args: argparse.Namespace) -> None:
    """Score parsed news with sentiment and event tags."""
    from traderbot.news.score import score_news_file

    input_path = Path(args.input)
    output_path = Path(args.out)
    data_dir = Path(args.data_dir) if args.data_dir else None

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    count = score_news_file(input_path, output_path, data_dir)
    print(f"Scored {count} news items to {output_path}")


def cmd_sector_digest(args: argparse.Namespace) -> None:
    """Build sector sentiment digest."""
    from traderbot.news.sector_digest import build_sector_digest

    input_path = Path(args.input)
    output_csv = Path(args.out)
    output_png = Path(args.out.replace(".csv", ".png")) if args.png else None

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    results = build_sector_digest(input_path, output_csv, output_png, args.window)
    print(f"Built sector digest for {len(results)} sectors to {output_csv}")


# ---------------------------- Scan Command ---------------------------- #


def _load_news_sentiment(
    news_path: Path,
    window_hours: int = 48,
) -> Dict[str, float]:
    """Load per-symbol sentiment from scored news."""
    if not news_path.exists():
        return {}

    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    symbol_scores: Dict[str, List[float]] = {}

    with open(news_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Check time window
            try:
                pub_dt = datetime.fromisoformat(
                    item.get("published_utc", "").replace("Z", "+00:00")
                )
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                if pub_dt < cutoff:
                    continue
            except Exception:
                continue

            # Aggregate by ticker
            decayed = item.get("decayed_sentiment", 0.0)
            for ticker in item.get("tickers", []):
                if ticker not in symbol_scores:
                    symbol_scores[ticker] = []
                symbol_scores[ticker].append(decayed)

    # Sum and clip to [-1, 1]
    result: Dict[str, float] = {}
    for ticker, scores in symbol_scores.items():
        total = sum(scores)
        result[ticker] = max(-1.0, min(1.0, total))

    return result


def _load_sector_tilt(
    sector_csv: Path,
    sector_map: Dict[str, str],
) -> Dict[str, float]:
    """Load sector z-score tilt for symbols."""
    if not sector_csv.exists():
        return {}

    from traderbot.news.sector_digest import SectorSentiment

    df = pd.read_csv(sector_csv)
    if df.empty:
        return {}

    # Compute sector z-scores
    mean_all = df["mean_sentiment"].mean()
    std_all = df["mean_sentiment"].std()
    if std_all < 0.001:
        std_all = 1.0

    sector_z: Dict[str, float] = {}
    for _, row in df.iterrows():
        z = (row["mean_sentiment"] - mean_all) / std_all
        sector_z[row["sector"]] = z

    # Map to symbols
    result: Dict[str, float] = {}
    for symbol, sector in sector_map.items():
        result[symbol] = sector_z.get(sector, 0.0)

    return result


def cmd_scan(args: argparse.Namespace) -> None:
    """Run universe scan with factor ranking and output opportunities."""
    setup_logging(level=args.log_level)
    cfg = get_config()

    # Parse weights JSON if provided
    weights_mapping: Dict[str, float] | None = None
    if args.weights:
        weights_mapping = json.loads(args.weights)

    # Default weights including new factors
    if weights_mapping is None:
        weights_mapping = {
            "trend": 0.35,
            "momentum": 0.25,
            "meanrev": 0.15,
            "sent": 0.10,
            "sector": 0.05,
            "gap": 0.10,
            "cost": -0.10,
        }

    top_n = args.top_n
    sector_cap = args.sector_cap
    data_root = Path(cfg.data.ohlcv_dir)

    # Load universe
    uni_cfg = _load_yaml_universe(args.universe)
    screened = _screen_universe(uni_cfg, data_root=data_root)
    if not screened:
        logger.error("No symbols passed screening for universe %s", args.universe)
        sys.exit(1)

    # Build factor matrix
    factor_matrix = build_factor_matrix(screened)
    z = standardize_zscores(factor_matrix)

    # Load sentiment data
    news_sentiment: Dict[str, float] = {}
    if args.news_file:
        news_sentiment = _load_news_sentiment(Path(args.news_file))

    # Load sector data
    sector_map = _load_sector_map(data_root)
    sector_tilt: Dict[str, float] = {}
    if args.sector_file:
        sector_tilt = _load_sector_tilt(Path(args.sector_file), sector_map)

    # Compute gap scores
    from traderbot.features.gaps import analyze_gaps_batch, get_gap_score_adjustment

    gap_analyses = analyze_gaps_batch(screened)

    # Build composite scores with new factors
    opportunities: List[Dict[str, Any]] = []

    for symbol in z.index:
        # Base factor scores (z-scored)
        trend_cols = [c for c in z.columns if c.startswith("trend_")]
        trend_score = float(z.loc[symbol, trend_cols].mean()) if trend_cols else 0.0

        mom_score = float(z.loc[symbol, "momentum_12_1"]) if "momentum_12_1" in z.columns else 0.0

        meanrev_cols = [c for c in z.columns if c.startswith("meanrev_")]
        meanrev_score = float(z.loc[symbol, meanrev_cols].mean()) if meanrev_cols else 0.0

        cost_score = float(z.loc[symbol, "cost_spread_bps"]) if "cost_spread_bps" in z.columns else 0.0

        # New factor scores
        sent_score = news_sentiment.get(symbol, 0.0)
        sector_score = sector_tilt.get(symbol, 0.0)

        gap_analysis = gap_analyses.get(symbol)
        gap_score = gap_analysis.gap_score if gap_analysis else 0.0
        gap_label = gap_analysis.gap_label.value if gap_analysis else "NEUTRAL"
        gap_adj = get_gap_score_adjustment(gap_analysis)

        # Compute composite
        composite = (
            weights_mapping.get("trend", 0.35) * trend_score
            + weights_mapping.get("momentum", 0.25) * mom_score
            + weights_mapping.get("meanrev", 0.15) * meanrev_score
            + weights_mapping.get("sent", 0.10) * sent_score
            + weights_mapping.get("sector", 0.05) * sector_score
            + weights_mapping.get("gap", 0.10) * gap_score
            + weights_mapping.get("cost", -0.10) * cost_score
            + gap_adj  # Small regime-based adjustment
        )

        # Build rationale tags
        tags: List[str] = []
        if trend_score > 0.5:
            tags.append("strong_trend")
        if mom_score > 0.5:
            tags.append("high_momentum")
        if sent_score > 0.3:
            tags.append("positive_news")
        elif sent_score < -0.3:
            tags.append("negative_news")
        if gap_label == "CONT":
            tags.append("gap_continuation")
        elif gap_label == "REVERT":
            tags.append("gap_reversion")

        opportunities.append({
            "symbol": symbol,
            "composite_score": round(composite, 4),
            "trend_score": round(trend_score, 4),
            "momentum_score": round(mom_score, 4),
            "meanrev_score": round(meanrev_score, 4),
            "sentiment_score": round(sent_score, 4),
            "sector_tilt": round(sector_score, 4),
            "gap_score": round(gap_score, 4),
            "gap_label": gap_label,
            "cost_score": round(cost_score, 4),
            "sector": sector_map.get(symbol, "UNKNOWN"),
            "tags": tags,
        })

    # Sort by composite score
    opportunities.sort(key=lambda x: x["composite_score"], reverse=True)

    # Apply sector caps
    symbols_ranked = [o["symbol"] for o in opportunities]
    selected_symbols = apply_sector_caps(symbols_ranked, sector_map, top_n, sector_cap)
    opportunities_filtered = [o for o in opportunities if o["symbol"] in selected_symbols]

    # Output paths
    base_reports = Path(args.reports_dir) if args.reports_dir else cfg.reports_dir
    reports_root = base_reports
    alerts_dir = reports_root / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)

    # Persist latest selection for paper-trade wiring
    backtests_root = base_reports / "backtests"
    backtests_root.mkdir(parents=True, exist_ok=True)
    selection_path = backtests_root / "last_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "universe": args.universe,
                "strategy": args.strategy,
                "top_n": top_n,
                "sector_cap": sector_cap,
                "symbols": selected_symbols,
            },
            f,
            indent=2,
        )

    # Write opportunities CSV
    opp_csv = reports_root / "opportunities.csv"
    with open(opp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "symbol", "composite_score", "trend_score", "momentum_score",
            "meanrev_score", "sentiment_score", "sector_tilt", "gap_score",
            "gap_label", "cost_score", "sector", "tags",
        ])
        writer.writeheader()
        for opp in opportunities_filtered:
            row = opp.copy()
            row["tags"] = ",".join(opp["tags"])
            writer.writerow(row)

    logger.info("Wrote %d opportunities to %s", len(opportunities_filtered), opp_csv)

    # Write alert preview markdown
    md_lines = [
        f"# TraderBot Alert Preview",
        f"",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Universe:** {args.universe}",
        f"**Top-N:** {top_n}",
        f"",
        "## Top Opportunities",
        "",
        "| Rank | Symbol | Score | Trend | Momentum | Sentiment | Gap | Tags |",
        "|------|--------|-------|-------|----------|-----------|-----|------|",
    ]
    for i, opp in enumerate(opportunities_filtered[:25], 1):
        tags_str = ", ".join(opp["tags"][:3]) if opp["tags"] else "-"
        md_lines.append(
            f"| {i} | **{opp['symbol']}** | {opp['composite_score']:.3f} | "
            f"{opp['trend_score']:.2f} | {opp['momentum_score']:.2f} | "
            f"{opp['sentiment_score']:.2f} | {opp['gap_label']} | {tags_str} |"
        )

    md_lines.extend([
        "",
        "## Files",
        "",
        f"- [opportunities.csv](../opportunities.csv)",
        f"- [sector_sentiment.csv](../sector_sentiment.csv)",
    ])

    preview_md = alerts_dir / "preview.md"
    preview_md.write_text("\n".join(md_lines))
    logger.info("Wrote alert preview to %s", preview_md)

    # Write HTML preview using template
    _write_html_preview(
        opportunities_filtered[:25],
        reports_root / "sector_sentiment.csv" if args.sector_file else None,
        alerts_dir / "preview.html",
    )

    print(f"Scan complete: {len(opportunities_filtered)} opportunities")
    print(f"  CSV: {opp_csv}")
    print(f"  Preview: {preview_md}")


def _write_html_preview(
    opportunities: List[Dict[str, Any]],
    sector_csv: Optional[Path],
    output_path: Path,
) -> None:
    """Write HTML alert preview."""
    # Load sector data
    sectors: List[Dict[str, Any]] = []
    if sector_csv and sector_csv.exists():
        df = pd.read_csv(sector_csv)
        for _, row in df.iterrows():
            sectors.append(row.to_dict())

    # Simple string formatting (avoiding Jinja2 dependency)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Read template
    template_path = Path(__file__).parent.parent / "reports" / "templates" / "alerts.html.j2"
    if template_path.exists():
        try:
            from jinja2 import Template

            template = Template(template_path.read_text())
            html = template.render(
                date=date,
                timestamp=timestamp,
                sectors=sectors,
                opportunities=opportunities,
                version=__version__,
            )
            output_path.write_text(html)
            logger.info("Wrote HTML preview to %s", output_path)
            return
        except ImportError:
            pass  # Fall back to simple HTML

    # Simple HTML fallback
    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><title>TraderBot Alert Preview</title></head>",
        "<body>",
        f"<h1>TraderBot Alert Preview - {date}</h1>",
        f"<p>Generated: {timestamp}</p>",
        "<h2>Top Opportunities</h2>",
        "<table border='1'>",
        "<tr><th>Rank</th><th>Symbol</th><th>Score</th><th>Trend</th><th>Sentiment</th><th>Gap</th></tr>",
    ]
    for i, opp in enumerate(opportunities, 1):
        html_lines.append(
            f"<tr><td>{i}</td><td>{opp['symbol']}</td><td>{opp['composite_score']:.3f}</td>"
            f"<td>{opp['trend_score']:.2f}</td><td>{opp['sentiment_score']:.2f}</td>"
            f"<td>{opp['gap_label']}</td></tr>"
        )
    html_lines.extend([
        "</table>",
        "</body></html>",
    ])
    output_path.write_text("\n".join(html_lines))
    logger.info("Wrote simple HTML preview to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TraderBot high-level CLI (tb)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------- News Commands --------------------

    # news-pull
    news_pull_p = subparsers.add_parser(
        "news-pull",
        help="Fetch RSS news feeds",
    )
    news_pull_p.add_argument(
        "--sources",
        required=True,
        help="Path to RSS sources file",
    )
    news_pull_p.add_argument(
        "--out",
        required=True,
        help="Output JSONL path",
    )
    news_pull_p.set_defaults(func=cmd_news_pull)

    # news-parse
    news_parse_p = subparsers.add_parser(
        "news-parse",
        help="Parse raw news JSONL",
    )
    news_parse_p.add_argument(
        "--in", "--input",
        dest="input",
        required=True,
        help="Input JSONL path",
    )
    news_parse_p.add_argument(
        "--out",
        required=True,
        help="Output JSONL path",
    )
    news_parse_p.set_defaults(func=cmd_news_parse)

    # news-score
    news_score_p = subparsers.add_parser(
        "news-score",
        help="Score parsed news with sentiment and events",
    )
    news_score_p.add_argument(
        "--in", "--input",
        dest="input",
        required=True,
        help="Input JSONL path",
    )
    news_score_p.add_argument(
        "--out",
        required=True,
        help="Output JSONL path",
    )
    news_score_p.add_argument(
        "--data-dir",
        default="data",
        help="Data directory for sector_map.csv",
    )
    news_score_p.set_defaults(func=cmd_news_score)

    # sector-digest
    sector_digest_p = subparsers.add_parser(
        "sector-digest",
        help="Build sector sentiment digest",
    )
    sector_digest_p.add_argument(
        "--in", "--input",
        dest="input",
        required=True,
        help="Scored news JSONL path",
    )
    sector_digest_p.add_argument(
        "--out",
        required=True,
        help="Output CSV path",
    )
    sector_digest_p.add_argument(
        "--window",
        default="1d",
        help="Time window (e.g., 1d, 12h)",
    )
    sector_digest_p.add_argument(
        "--png",
        action="store_true",
        help="Also generate PNG chart",
    )
    sector_digest_p.set_defaults(func=cmd_sector_digest)

    # -------------------- Scan Command --------------------

    # scan
    scan_p = subparsers.add_parser(
        "scan",
        help="Run universe scan with factor ranking",
    )
    scan_p.add_argument(
        "--universe",
        required=True,
        help="Named universe YAML (e.g. sp500)",
    )
    scan_p.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "hybrid"],
        default="trend",
        help="Strategy type",
    )
    scan_p.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of symbols to keep after ranking",
    )
    scan_p.add_argument(
        "--sector-cap",
        type=float,
        default=0.2,
        help="Max fraction per sector (e.g. 0.2 = 20%%)",
    )
    scan_p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional JSON mapping for composite weights",
    )
    scan_p.add_argument(
        "--news-file",
        type=str,
        default=None,
        help="Path to scored news JSONL for sentiment",
    )
    scan_p.add_argument(
        "--sector-file",
        type=str,
        default=None,
        help="Path to sector_sentiment.csv",
    )
    scan_p.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Base directory for reports (default: ./reports)",
    )
    scan_p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    scan_p.set_defaults(func=cmd_scan)

    # -------------------- Backtest Commands --------------------

    # backtest
    backtest_p = subparsers.add_parser(
        "backtest",
        help="Run universe screen + ranking + backtest grid",
    )
    backtest_p.add_argument(
        "--universe",
        required=True,
        help="Named universe YAML (e.g. sp500, liquid_top1000)",
    )
    backtest_p.add_argument(
        "--strategy",
        choices=["trend", "meanrev", "hybrid"],
        default="trend",
        help="Strategy type (currently all map to the momentum engine)",
    )
    backtest_p.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of symbols to keep after ranking",
    )
    backtest_p.add_argument(
        "--sector-cap",
        type=float,
        default=0.2,
        help="Max fraction per sector (e.g. 0.2 = 20%%)",
    )
    backtest_p.add_argument(
        "--weights",
        type=str,
        default=None,
        help='Optional JSON mapping for composite weights, '
        'e.g. \'{"trend":0.4,"momentum":0.2,"meanrev":0.2,"quality":0.1,"cost":-0.1}\'',
    )
    backtest_p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If omitted, uses default grid periods.",
    )
    backtest_p.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). If omitted, uses default grid periods.",
    )
    backtest_p.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Base directory for grid reports (default: ./reports/backtests)",
    )
    backtest_p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    backtest_p.set_defaults(func=cmd_backtest)

    # compare
    compare_p = subparsers.add_parser(
        "compare",
        help="Print a ranked table from backtest comparison.csv",
    )
    compare_p.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Base directory for grid reports (default: ./reports/backtests)",
    )
    compare_p.set_defaults(func=cmd_compare)

    # paper-start
    paper_start_p = subparsers.add_parser(
        "paper-start",
        help="Initialize dry-run paper account from last backtest selection",
    )
    paper_start_p.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Base directory for grid reports (default: ./reports/backtests)",
    )
    paper_start_p.set_defaults(func=cmd_paper_start)

    # paper-sync
    paper_sync_p = subparsers.add_parser(
        "paper-sync",
        help="Simulate sync cycle and log expected vs filled prices",
    )
    paper_sync_p.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Base directory for grid reports (default: ./reports/backtests)",
    )
    paper_sync_p.set_defaults(func=cmd_paper_sync)

    # paper-status
    paper_status_p = subparsers.add_parser(
        "paper-status",
        help="Show current paper account status",
    )
    paper_status_p.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Base directory for grid reports (default: ./reports/backtests)",
    )
    paper_status_p.set_defaults(func=cmd_paper_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


