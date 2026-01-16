"""News parsing and ticker extraction.

Normalizes raw news items and extracts stock tickers
using regex patterns and company name aliases.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from traderbot.logging_setup import get_logger

logger = get_logger("news.parse")

# Module directory for loading alias.json
MODULE_DIR = Path(__file__).parent


@dataclass
class ParsedNewsItem:
    """Parsed and normalized news item."""

    source: str
    title: str
    summary: str
    url: str
    published_utc: str
    tickers: List[str]
    content_hash: str


def _load_aliases() -> Dict[str, str]:
    """Load company name to ticker aliases."""
    alias_path = MODULE_DIR / "alias.json"
    if not alias_path.exists():
        return {}

    with open(alias_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Preload aliases at module load
_ALIASES: Dict[str, str] = _load_aliases()


def _compute_hash(url: str, title: str) -> str:
    """Compute content hash for deduplication."""
    content = f"{url}|{title}".lower()
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _normalize_timestamp(timestamp: str) -> str:
    """Normalize timestamp to UTC ISO format."""
    try:
        # Parse ISO format
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        # Convert to UTC
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        else:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def extract_tickers(text: str, aliases: Optional[Dict[str, str]] = None) -> List[str]:
    """Extract stock tickers from text.

    Uses multiple strategies:
    1. Direct ticker mentions ($AAPL, AAPL:)
    2. Ticker patterns in context (NASDAQ: AAPL)
    3. Company name to ticker mapping

    Args:
        text: Text to extract tickers from.
        aliases: Optional company name to ticker mapping.

    Returns:
        List of unique tickers found (uppercase).
    """
    if aliases is None:
        aliases = _ALIASES

    tickers: Set[str] = set()
    text_upper = text.upper()

    # Pattern 1: $TICKER format (e.g., $AAPL)
    dollar_pattern = r"\$([A-Z]{1,5})\b"
    for match in re.finditer(dollar_pattern, text_upper):
        ticker = match.group(1)
        if _is_valid_ticker(ticker):
            tickers.add(ticker)

    # Pattern 2: EXCHANGE:TICKER format (e.g., NASDAQ:AAPL, NYSE:IBM)
    exchange_pattern = r"(?:NASDAQ|NYSE|AMEX|OTC)[\s:]+([A-Z]{1,5})\b"
    for match in re.finditer(exchange_pattern, text_upper):
        ticker = match.group(1)
        if _is_valid_ticker(ticker):
            tickers.add(ticker)

    # Pattern 3: Ticker in parentheses (e.g., "Apple (AAPL)")
    paren_pattern = r"\(([A-Z]{1,5})\)"
    for match in re.finditer(paren_pattern, text_upper):
        ticker = match.group(1)
        if _is_valid_ticker(ticker):
            tickers.add(ticker)

    # Pattern 4: Company name aliases
    for company_name, ticker in aliases.items():
        # Case-insensitive search for company name
        if company_name.upper() in text_upper:
            tickers.add(ticker.upper())

    # Pattern 5: Standalone ticker-like words (more conservative)
    # Only if they look like known tickers (1-4 letters)
    standalone_pattern = r"\b([A-Z]{2,4})\b"
    known_tickers = {
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA",
        "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "ADBE", "CRM",
        "NFLX", "PYPL", "INTC", "CSCO", "VZ", "PFE", "MRK", "ABT", "TMO",
        "COST", "PEP", "AVGO", "TXN", "QCOM", "AMD", "HON", "IBM", "BA",
        "GE", "CAT", "MMM", "GS", "AXP", "CVX", "XOM", "WMT", "KO", "MCD",
    }
    for match in re.finditer(standalone_pattern, text_upper):
        ticker = match.group(1)
        if ticker in known_tickers:
            tickers.add(ticker)

    return sorted(tickers)


def _is_valid_ticker(ticker: str) -> bool:
    """Check if string looks like a valid ticker."""
    if not ticker or len(ticker) > 5:
        return False

    # Filter out common words that look like tickers
    excluded = {
        "A", "I", "BE", "IT", "IS", "AS", "AT", "BY", "DO", "GO", "IF",
        "IN", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
        "ALL", "AND", "ARE", "BUT", "CAN", "CEO", "CFO", "FOR", "HAS",
        "INC", "IPO", "ITS", "NEW", "NOT", "NOW", "OLD", "ONE", "OUR",
        "OUT", "SEC", "THE", "TOO", "TOP", "TWO", "WAS", "WHO", "WHY",
        "FDA", "CEO", "CTO", "COO", "CFO", "PR", "RSS", "URL", "USA",
        "ETF", "EPS", "GDP", "IPO", "P/E", "ROE", "ROI", "YOY", "QOQ",
    }
    return ticker not in excluded


def parse_news_item(raw_item: Dict) -> ParsedNewsItem:
    """Parse a single raw news item."""
    source = raw_item.get("source", "unknown")
    title = raw_item.get("title", "")
    summary = raw_item.get("summary", "")
    url = raw_item.get("url", "")
    published = raw_item.get("published_utc", "")

    # Normalize timestamp
    published_utc = _normalize_timestamp(published)

    # Extract tickers from title and summary
    combined_text = f"{title} {summary}"
    tickers = extract_tickers(combined_text)

    # Compute content hash
    content_hash = _compute_hash(url, title)

    return ParsedNewsItem(
        source=source,
        title=title,
        summary=summary,
        url=url,
        published_utc=published_utc,
        tickers=tickers,
        content_hash=content_hash,
    )


def parse_news_items(
    input_path: Path,
    output_path: Path,
    deduplicate: bool = True,
) -> int:
    """Parse raw news JSONL and output parsed JSONL.

    Args:
        input_path: Path to raw news JSONL.
        output_path: Path to write parsed JSONL.
        deduplicate: Whether to deduplicate by content hash.

    Returns:
        Number of items written.
    """
    if not input_path.exists():
        logger.warning("Input file not found: %s", input_path)
        return 0

    seen_hashes: Set[str] = set()
    parsed_items: List[ParsedNewsItem] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                raw_item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON line: %s", line[:100])
                continue

            parsed = parse_news_item(raw_item)

            # Deduplication
            if deduplicate:
                if parsed.content_hash in seen_hashes:
                    continue
                seen_hashes.add(parsed.content_hash)

            parsed_items.append(parsed)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in parsed_items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    logger.info(
        "Parsed %d items (deduplicated from %d) to %s",
        len(parsed_items),
        len(seen_hashes) if deduplicate else len(parsed_items),
        output_path,
    )
    return len(parsed_items)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse raw news JSONL")
    parser.add_argument("--input", required=True, dest="input_path", help="Input JSONL path")
    parser.add_argument("--out", required=True, help="Output JSONL path")

    args = parser.parse_args()
    parse_news_items(Path(args.input_path), Path(args.out))
