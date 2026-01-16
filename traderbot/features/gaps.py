"""Gap detection and regime classification.

Detects overnight/premarket gaps relative to ATR and
classifies gap regimes (continuation, reversion, neutral).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from traderbot.features.ta import calculate_atr, calculate_sma
from traderbot.logging_setup import get_logger

logger = get_logger("features.gaps")


class GapRegime(str, Enum):
    """Gap regime classification."""

    CONTINUATION = "CONT"
    REVERSION = "REVERT"
    NEUTRAL = "NEUTRAL"


@dataclass
class GapAnalysis:
    """Gap analysis result for a symbol."""

    gap_pct: float  # Raw gap percentage
    gap_vs_atr: float  # Gap as multiple of ATR%
    gap_score: float  # Normalized score [-1, 1]
    gap_label: GapRegime  # Regime classification
    prior_trend: float  # Prior trend direction


def compute_gap_pct(
    open_price: float,
    prior_close: float,
) -> float:
    """Compute gap percentage from prior close to current open.

    Args:
        open_price: Today's open price.
        prior_close: Prior day's close price.

    Returns:
        Gap as percentage (e.g., 0.02 for 2% gap up).
    """
    if prior_close <= 0:
        return 0.0

    return (open_price - prior_close) / prior_close


def compute_atr_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> float:
    """Compute ATR as percentage of current price.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR lookback period.

    Returns:
        ATR as percentage of last close.
    """
    if len(close) < period + 1:
        return 0.0

    atr_series = calculate_atr(high, low, close, period=period)
    if atr_series.empty or atr_series.isna().all():
        return 0.0

    atr_val = float(atr_series.iloc[-1])
    last_close = float(close.iloc[-1])

    if last_close <= 0:
        return 0.0

    return atr_val / last_close


def compute_prior_trend(
    close: pd.Series,
    lookback: int = 20,
) -> float:
    """Compute prior trend direction.

    Uses price position relative to SMA as trend indicator.

    Args:
        close: Close price series.
        lookback: Trend lookback period.

    Returns:
        Trend score in range [-1, 1].
    """
    if len(close) < lookback + 1:
        return 0.0

    sma = calculate_sma(close, period=lookback)
    if sma.empty or sma.isna().all():
        return 0.0

    sma_val = float(sma.iloc[-2])  # Prior day's SMA
    prior_close = float(close.iloc[-2])

    if sma_val <= 0:
        return 0.0

    # How far above/below SMA (as multiple of typical range)
    deviation = (prior_close - sma_val) / sma_val

    # Normalize to [-1, 1] (cap at +/- 10% deviation)
    return max(-1.0, min(1.0, deviation * 10.0))


def classify_gap_regime(
    gap_pct: float,
    atr_pct: float,
    prior_trend: float,
) -> GapRegime:
    """Classify gap into regime category.

    Rules:
    - CONT: Small gap (< 0.5 ATR) OR gap aligns with prior trend
    - REVERT: Large gap (> 1.0 ATR) against prior trend
    - NEUTRAL: Everything else

    Args:
        gap_pct: Gap as percentage.
        atr_pct: ATR as percentage of price.
        prior_trend: Prior trend direction [-1, 1].

    Returns:
        GapRegime classification.
    """
    if atr_pct <= 0:
        return GapRegime.NEUTRAL

    gap_vs_atr = abs(gap_pct) / atr_pct

    # Check if gap aligns with trend
    gap_direction = 1 if gap_pct > 0 else -1 if gap_pct < 0 else 0
    trend_direction = 1 if prior_trend > 0.2 else -1 if prior_trend < -0.2 else 0
    aligned = gap_direction * trend_direction > 0

    # Small gap OR aligned with trend -> Continuation
    if gap_vs_atr <= 0.5 or (gap_vs_atr <= 1.5 and aligned):
        return GapRegime.CONTINUATION

    # Large gap against trend -> Reversion
    if gap_vs_atr > 1.0 and not aligned and trend_direction != 0:
        return GapRegime.REVERSION

    return GapRegime.NEUTRAL


def compute_gap_score(
    gap_pct: float,
    atr_pct: float,
    gap_label: GapRegime,
) -> float:
    """Compute normalized gap score.

    Score represents expected follow-through direction:
    - Positive: Expected to continue up
    - Negative: Expected to continue down

    Args:
        gap_pct: Gap as percentage.
        atr_pct: ATR as percentage.
        gap_label: Gap regime classification.

    Returns:
        Gap score in range [-1, 1].
    """
    if atr_pct <= 0:
        return 0.0

    # Base score from gap direction and magnitude
    gap_vs_atr = gap_pct / atr_pct if atr_pct > 0 else 0.0
    base_score = max(-1.0, min(1.0, gap_vs_atr / 2.0))

    # Adjust based on regime
    if gap_label == GapRegime.CONTINUATION:
        # Continuation gaps have follow-through in gap direction
        return base_score * 0.8

    elif gap_label == GapRegime.REVERSION:
        # Reversion gaps tend to fill, so score is opposite to gap
        return -base_score * 0.6

    else:
        # Neutral - smaller impact
        return base_score * 0.4


def analyze_gap(df: pd.DataFrame) -> Optional[GapAnalysis]:
    """Analyze gap for a single symbol's OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].
            Must have at least 15 rows for ATR calculation.

    Returns:
        GapAnalysis if sufficient data, None otherwise.
    """
    required_cols = ["open", "high", "low", "close"]
    if not all(col in df.columns for col in required_cols):
        logger.warning("Missing required columns for gap analysis")
        return None

    if len(df) < 15:
        return None

    # Get current and prior day prices
    current_open = float(df["open"].iloc[-1])
    prior_close = float(df["close"].iloc[-2])

    # Compute gap percentage
    gap_pct = compute_gap_pct(current_open, prior_close)

    # Compute ATR%
    atr_pct = compute_atr_pct(
        df["high"],
        df["low"],
        df["close"],
        period=14,
    )

    # Compute prior trend
    prior_trend = compute_prior_trend(df["close"], lookback=20)

    # Classify regime
    gap_label = classify_gap_regime(gap_pct, atr_pct, prior_trend)

    # Compute score
    gap_score = compute_gap_score(gap_pct, atr_pct, gap_label)

    # Gap vs ATR multiple
    gap_vs_atr = abs(gap_pct) / atr_pct if atr_pct > 0 else 0.0

    return GapAnalysis(
        gap_pct=round(gap_pct, 6),
        gap_vs_atr=round(gap_vs_atr, 3),
        gap_score=round(gap_score, 4),
        gap_label=gap_label,
        prior_trend=round(prior_trend, 4),
    )


def analyze_gaps_batch(
    data_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, GapAnalysis]:
    """Analyze gaps for multiple symbols.

    Args:
        data_by_symbol: Mapping of symbol to OHLCV DataFrame.

    Returns:
        Mapping of symbol to GapAnalysis.
    """
    results: Dict[str, GapAnalysis] = {}

    for symbol, df in data_by_symbol.items():
        analysis = analyze_gap(df)
        if analysis is not None:
            results[symbol] = analysis

    logger.info("Analyzed gaps for %d symbols", len(results))
    return results


def get_gap_score_adjustment(gap_analysis: Optional[GapAnalysis]) -> float:
    """Get score adjustment for opportunity ranking.

    Small adjustments based on gap regime:
    - CONT with positive gap: +0.05
    - REVERT with negative gap: -0.05

    Args:
        gap_analysis: Gap analysis result.

    Returns:
        Score adjustment in [-0.1, 0.1].
    """
    if gap_analysis is None:
        return 0.0

    label = gap_analysis.gap_label
    score = gap_analysis.gap_score

    if label == GapRegime.CONTINUATION and score > 0:
        return 0.05
    elif label == GapRegime.REVERSION and score < 0:
        return -0.05

    return 0.0
