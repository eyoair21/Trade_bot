"""Sentiment-based signals.

Provides news tone scores using FinBERT if available,
otherwise falls back to deterministic rule-based scores.
"""

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traderbot.logging_setup import get_logger

logger = get_logger("features.sentiment")

# Try to import FinBERT
_FINBERT_AVAILABLE = False
_finbert_pipeline = None

try:
    from transformers import pipeline

    _FINBERT_AVAILABLE = True
except ImportError:
    pass


def _init_finbert() -> Any | None:
    """Initialize FinBERT pipeline if available."""
    global _finbert_pipeline

    if _finbert_pipeline is not None:
        return _finbert_pipeline

    if not _FINBERT_AVAILABLE:
        return None

    try:
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            return_all_scores=True,
        )
        logger.info("FinBERT initialized successfully")
        return _finbert_pipeline
    except Exception as e:
        logger.warning(f"Failed to load FinBERT: {e}")
        return None


def _get_deterministic_tone(ticker: str, date_str: str, seed: int = 42) -> float:
    """Generate deterministic tone score for ticker/date.

    Uses hash of ticker+date+seed to generate reproducible values.

    Args:
        ticker: Ticker symbol.
        date_str: Date string (YYYY-MM-DD).
        seed: Random seed for reproducibility.

    Returns:
        Tone score in [-1.0, 1.0].
    """
    # Create deterministic hash from ticker, date, and seed
    hash_input = f"{ticker}:{date_str}:{seed}"
    hash_bytes = hashlib.md5(hash_input.encode()).digest()

    # Convert first 4 bytes to integer and normalize to [-1, 1]
    hash_int = int.from_bytes(hash_bytes[:4], byteorder="big")
    normalized = (hash_int / 0xFFFFFFFF) * 2 - 1

    return float(normalized)


def _load_news_cache(ticker: str, date_val: date) -> list[str] | None:
    """Load cached headlines for ticker/date.

    Args:
        ticker: Ticker symbol.
        date_val: Date to look up.

    Returns:
        List of headlines or None if not cached.
    """
    cache_dir = Path("data/news_cache")
    cache_file = cache_dir / f"{ticker}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file) as f:
            cache_data = json.load(f)

        date_str = date_val.isoformat()
        if date_str in cache_data:
            headlines: list[str] | None = cache_data[date_str]
            return headlines
    except Exception as e:
        logger.debug(f"Failed to load news cache for {ticker}: {e}")

    return None


def _finbert_score_headlines(headlines: list[str]) -> float:
    """Score headlines using FinBERT.

    Args:
        headlines: List of news headlines.

    Returns:
        Average sentiment score in [-1, 1].
    """
    pipeline_obj = _init_finbert()
    if pipeline_obj is None or not headlines:
        return 0.0

    try:
        scores = []
        for headline in headlines[:10]:  # Limit to 10 headlines
            result = pipeline_obj(headline[:512])  # Max 512 tokens
            if result:
                # FinBERT returns: positive, negative, neutral
                pos = next((s["score"] for s in result[0] if s["label"] == "positive"), 0)
                neg = next((s["score"] for s in result[0] if s["label"] == "negative"), 0)
                scores.append(pos - neg)

        return float(np.mean(scores)) if scores else 0.0

    except Exception as e:
        logger.warning(f"FinBERT scoring failed: {e}")
        return 0.0


def _rule_based_tone(headlines: list[str]) -> float:
    """Simple rule-based sentiment from keywords.

    Args:
        headlines: List of news headlines.

    Returns:
        Sentiment score in [-1, 1].
    """
    if not headlines:
        return 0.0

    positive_words = {
        "up",
        "gain",
        "rise",
        "surge",
        "jump",
        "rally",
        "high",
        "growth",
        "profit",
        "beat",
        "exceed",
        "strong",
        "bullish",
        "positive",
        "upgrade",
        "buy",
        "outperform",
    }
    negative_words = {
        "down",
        "fall",
        "drop",
        "decline",
        "loss",
        "miss",
        "weak",
        "low",
        "bearish",
        "negative",
        "downgrade",
        "sell",
        "underperform",
        "crash",
        "plunge",
        "tumble",
    }

    scores = []
    for headline in headlines:
        words = set(headline.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count + neg_count > 0:
            score = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            score = 0.0
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def get_sentiment_score(
    ticker: str,
    date: pd.Timestamp | None = None,
) -> float:
    """Get sentiment score for a ticker.

    Stub implementation returning neutral sentiment.

    Args:
        ticker: Ticker symbol.
        date: Optional date for historical sentiment.

    Returns:
        Sentiment score in range [-1.0, 1.0].
        0.0 = neutral, positive = bullish, negative = bearish.
    """
    # Stub: always return neutral
    return 0.0


def get_sentiment_scores(
    tickers: list[str],
    date: pd.Timestamp | None = None,
) -> dict[str, float]:
    """Get sentiment scores for multiple tickers.

    Stub implementation returning neutral sentiment for all.

    Args:
        tickers: List of ticker symbols.
        date: Optional date for historical sentiment.

    Returns:
        Dict mapping ticker to sentiment score.
    """
    return dict.fromkeys(tickers, 0.0)


def get_sentiment_series(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Get sentiment time series for a ticker.

    Stub implementation returning neutral sentiment series.

    Args:
        ticker: Ticker symbol.
        start_date: Start of date range.
        end_date: End of date range.

    Returns:
        Series with DatetimeIndex and sentiment scores.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    return pd.Series(0.0, index=date_range, name=f"{ticker}_sentiment")


def get_tone_scores(
    tickers: list[str],
    start: date | str,
    end: date | str,
    use_finbert: bool = True,
    seed: int = 42,
) -> dict[str, float]:
    """Get news tone scores for multiple tickers.

    If FinBERT is available and use_finbert=True, uses it for scoring.
    Otherwise falls back to deterministic rule-based or random scores.

    Args:
        tickers: List of ticker symbols.
        start: Start date (inclusive).
        end: End date (inclusive).
        use_finbert: Whether to try using FinBERT.
        seed: Random seed for deterministic fallback.

    Returns:
        Dict mapping ticker to tone score in [-1, 1].
    """
    # Convert dates
    if isinstance(start, str):
        start = datetime.fromisoformat(start).date()
    if isinstance(end, str):
        end = datetime.fromisoformat(end).date()

    scores: dict[str, float] = {}

    for ticker in tickers:
        ticker_scores = []

        # Try to get cached headlines for each day
        current = start
        while current <= end:
            headlines = _load_news_cache(ticker, current)

            if headlines:
                # Use FinBERT if available and enabled
                if use_finbert and _FINBERT_AVAILABLE:
                    score = _finbert_score_headlines(headlines)
                else:
                    score = _rule_based_tone(headlines)
                ticker_scores.append(score)
            else:
                # Fallback to deterministic score
                date_str = current.isoformat()
                score = _get_deterministic_tone(ticker, date_str, seed)
                ticker_scores.append(score)

            current = (
                date(current.year, current.month, current.day + 1)
                if current.day < 28
                else (
                    date(current.year, current.month + 1, 1)
                    if current.month < 12
                    else date(current.year + 1, 1, 1)
                )
            )

            # Break if we've gone past end
            if current > end:
                break

        # Average score for the period
        scores[ticker] = float(np.mean(ticker_scores)) if ticker_scores else 0.0

    logger.debug(f"Computed tone scores for {len(tickers)} tickers")
    return scores


def is_finbert_available() -> bool:
    """Check if FinBERT is available for use.

    Returns:
        True if FinBERT can be loaded.
    """
    return _FINBERT_AVAILABLE and _init_finbert() is not None
