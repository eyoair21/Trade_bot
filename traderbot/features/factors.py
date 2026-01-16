"""Factor library and composite ranking helpers.

Implements a small cross-sectional factor set for ranking symbols:

- Trend:    price vs. moving averages (20/50/200)
- Momentum: 3–12 month returns with 1M skip
- Mean-rev: RSI(2) and z-score of price vs. 20d SMA
- Risk:     ATR% (ATR(14) / price)
- Cost:     Spread in basis points (approx. (high-low)/close)

All functions are pure and operate on pandas objects so they can
be reused in both research notebooks and CLI tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from traderbot.features.ta import calculate_atr, calculate_rsi, calculate_sma


# ---------------------------- Core factor calculators ---------------------------- #


def _latest_valid(series: pd.Series) -> float:
    """Return the latest non-NaN value or 0.0 if none."""
    if series is None or series.empty:
        return 0.0
    val = series.dropna().iloc[-1] if series.notna().any() else np.nan
    return float(val) if not np.isnan(val) else 0.0


def compute_trend_factors(close: pd.Series) -> Dict[str, float]:
    """Compute simple trend factors based on price vs moving averages."""
    sma20 = calculate_sma(close, period=20)
    sma50 = calculate_sma(close, period=50)
    sma200 = calculate_sma(close, period=200)

    price = _latest_valid(close)
    t20 = price / _latest_valid(sma20) - 1.0 if _latest_valid(sma20) > 0 else 0.0
    t50 = price / _latest_valid(sma50) - 1.0 if _latest_valid(sma50) > 0 else 0.0
    t200 = price / _latest_valid(sma200) - 1.0 if _latest_valid(sma200) > 0 else 0.0

    return {
        "trend_20": t20,
        "trend_50": t50,
        "trend_200": t200,
    }


def compute_momentum_factor(close: pd.Series, sessions_per_month: int = 21) -> float:
    """Compute 12-1 momentum (3–12m with 1m skip approximation).

    Uses:
    - lookback_12m = 12 * sessions_per_month
    - lookback_1m  = 1 * sessions_per_month
    momentum = price(t) / price(t-12m) - 1, ignoring last 1m by anchoring to t-1m.
    """
    if close.empty:
        return 0.0

    lookback_12 = 12 * sessions_per_month
    lookback_1 = sessions_per_month

    if len(close) <= lookback_12:
        return 0.0

    # Anchor at t-1m
    anchor_idx = -lookback_1
    end_price = close.iloc[anchor_idx]
    start_price = close.iloc[anchor_idx - lookback_12]

    if start_price <= 0:
        return 0.0

    return float(end_price / start_price - 1.0)


def compute_mean_reversion_factors(close: pd.Series) -> Dict[str, float]:
    """Compute fast mean-reversion factors: RSI2 and zscore(close vs SMA20)."""
    rsi2_series = calculate_rsi(close, period=2)
    rsi2 = _latest_valid(rsi2_series)

    sma20 = calculate_sma(close, period=20)
    if len(close) < 20 or len(sma20.dropna()) == 0:
        zscore = 0.0
    else:
        window = 60
        tail = close.tail(window)
        sma20_tail = sma20.tail(window)
        spread = tail - sma20_tail
        spread_std = spread.std()
        if spread_std and not np.isnan(spread_std) and spread_std > 0:
            zscore = float((spread.iloc[-1]) / spread_std)
        else:
            zscore = 0.0

    return {
        "meanrev_rsi2": rsi2,
        "meanrev_zscore": zscore,
    }


def compute_risk_factor(df: pd.DataFrame) -> float:
    """Risk factor as ATR% using 14d ATR / close."""
    if df.empty:
        return 0.0
    atr_series = calculate_atr(df["high"], df["low"], df["close"], period=14)
    atr_val = _latest_valid(atr_series)
    price = _latest_valid(df["close"])
    if price <= 0:
        return 0.0
    return float(atr_val / price)


def compute_cost_factor(df: pd.DataFrame) -> float:
    """Cost factor as spread in basis points.

    Approximate spread using (high-low)/close * 1e4.
    """
    if df.empty:
        return 0.0
    last = df.iloc[-1]
    high = float(last.get("high", np.nan))
    low = float(last.get("low", np.nan))
    close = float(last.get("close", np.nan))

    if close <= 0 or np.isnan(high) or np.isnan(low):
        return 0.0

    spread_bps = (high - low) / close * 10_000.0
    return float(spread_bps)


def compute_all_factors(df: pd.DataFrame) -> Dict[str, float]:
    """Compute full factor set for a single symbol."""
    close = df["close"]
    factors: Dict[str, float] = {}
    factors.update(compute_trend_factors(close))
    factors["momentum_12_1"] = compute_momentum_factor(close)
    factors.update(compute_mean_reversion_factors(close))
    factors["risk_atr_pct"] = compute_risk_factor(df)
    factors["cost_spread_bps"] = compute_cost_factor(df)
    return factors


# ---------------------------- Cross-sectional helpers ---------------------------- #


def standardize_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize each column to z-scores across symbols."""
    if df.empty:
        return df.copy()

    z = (df - df.mean()) / df.std(ddof=0)
    # Replace NaNs (e.g. constant column) with 0
    return z.fillna(0.0)


@dataclass
class CompositeWeights:
    """Weights for composite score."""

    trend: float = 0.35
    momentum: float = 0.25
    meanrev: float = 0.20
    quality: float = 0.10
    cost: float = -0.10  # cost is penalized

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float] | None) -> "CompositeWeights":
        if not mapping:
            return cls()
        # Use provided keys, falling back to defaults
        base = cls()
        return cls(
            trend=float(mapping.get("trend", base.trend)),
            momentum=float(mapping.get("momentum", base.momentum)),
            meanrev=float(mapping.get("meanrev", base.meanrev)),
            quality=float(mapping.get("quality", base.quality)),
            cost=float(mapping.get("cost", base.cost)),
        )


def build_factor_matrix(
    data_by_symbol: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build cross-sectional factor matrix (symbols x raw factors)."""
    rows: Dict[str, Dict[str, float]] = {}
    for symbol, df in data_by_symbol.items():
        if df.empty:
            continue
        rows[symbol] = compute_all_factors(df)

    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def compute_composite_scores(
    factor_z: pd.DataFrame,
    weights: CompositeWeights,
) -> pd.Series:
    """Compute composite score per symbol from z-scored factors."""
    if factor_z.empty:
        return pd.Series(dtype=float)

    # Map raw factor columns into buckets
    trend_cols = [c for c in factor_z.columns if c.startswith("trend_")]
    meanrev_cols = [c for c in factor_z.columns if c.startswith("meanrev_")]
    quality_cols = ["risk_atr_pct"]
    cost_cols = ["cost_spread_bps"]
    momentum_cols = ["momentum_12_1"]

    def _mean_or_zero(cols: Iterable[str]) -> pd.Series:
        cols = [c for c in cols if c in factor_z.columns]
        if not cols:
            return pd.Series(0.0, index=factor_z.index)
        return factor_z[cols].mean(axis=1)

    trend_score = _mean_or_zero(trend_cols)
    mom_score = _mean_or_zero(momentum_cols)
    meanrev_score = _mean_or_zero(meanrev_cols)
    quality_score = _mean_or_zero(quality_cols)
    cost_score = _mean_or_zero(cost_cols)

    composite = (
        weights.trend * trend_score
        + weights.momentum * mom_score
        + weights.meanrev * meanrev_score
        + weights.quality * quality_score
        + weights.cost * cost_score
    )

    return composite.sort_values(ascending=False)


def apply_sector_caps(
    ranked_symbols: List[str],
    sector_map: Mapping[str, str],
    top_n: int,
    sector_cap: float,
) -> List[str]:
    """Apply sector caps to ranked list of symbols.

    Args:
        ranked_symbols: Symbols sorted by composite score (best first).
        sector_map: Mapping symbol -> sector name. Symbols missing default to "UNKNOWN".
        top_n: Maximum number of symbols to keep.
        sector_cap: Max fraction per sector (e.g. 0.2 for 20%).
    """
    if top_n <= 0 or not ranked_symbols:
        return []

    max_per_sector = max(int(top_n * sector_cap), 1) if sector_cap > 0 else top_n
    counts: Dict[str, int] = {}
    selected: List[str] = []

    for symbol in ranked_symbols:
        sector = sector_map.get(symbol, "UNKNOWN")
        count = counts.get(sector, 0)
        if count >= max_per_sector:
            continue
        selected.append(symbol)
        counts[sector] = count + 1
        if len(selected) >= top_n:
            break

    return selected



