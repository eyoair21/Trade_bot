"""Feature engineering modules for trading signals."""

from traderbot.features.sentiment import get_tone_scores
from traderbot.features.ta import (
    atr,
    calculate_atr,
    calculate_ema,
    calculate_rsi,
    compute_model_features,
    dvol,
    pct_change1,
    regime_vix,
    rsi,
    vwap_gap,
)
from traderbot.features.volume import calculate_volume_ratio, calculate_vwap

__all__ = [
    # TA indicators
    "calculate_ema",
    "calculate_rsi",
    "calculate_atr",
    "calculate_vwap",
    "calculate_volume_ratio",
    # Model features
    "pct_change1",
    "rsi",
    "atr",
    "vwap_gap",
    "dvol",
    "regime_vix",
    "compute_model_features",
    # Sentiment
    "get_tone_scores",
]
