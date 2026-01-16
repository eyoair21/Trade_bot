"""News ingestion and sentiment analysis module.

This module provides RSS-based news fetching, parsing, scoring,
and sector digest functionality without requiring API keys.

Modules:
    pull: Fetch news from RSS feeds
    parse: Normalize and extract tickers from raw news
    score: Sentiment scoring with time decay and event tagging
    sector_digest: Aggregate sentiment by sector
"""

from traderbot.news.pull import fetch_rss_feeds
from traderbot.news.parse import parse_news_items, extract_tickers
from traderbot.news.score import score_news_item, compute_decayed_sentiment
from traderbot.news.sector_digest import build_sector_digest

__all__ = [
    "fetch_rss_feeds",
    "parse_news_items",
    "extract_tickers",
    "score_news_item",
    "compute_decayed_sentiment",
    "build_sector_digest",
]
