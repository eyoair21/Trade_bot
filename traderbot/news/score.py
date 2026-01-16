"""Sentiment scoring with time decay and event tagging.

Uses a lightweight lexicon-based approach (no model downloads required)
with time decay and event classification.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from traderbot.logging_setup import get_logger

logger = get_logger("news.score")


# Simple sentiment lexicon (positive and negative words)
POSITIVE_WORDS: Set[str] = {
    "beat", "beats", "beating", "exceeded", "exceeds", "exceeding",
    "strong", "stronger", "strongest", "growth", "growing", "grew",
    "profit", "profits", "profitable", "gain", "gains", "gained",
    "surge", "surges", "surged", "soar", "soars", "soared",
    "rally", "rallies", "rallied", "jump", "jumps", "jumped",
    "rise", "rises", "risen", "rising", "increase", "increases",
    "upgrade", "upgrades", "upgraded", "buy", "outperform",
    "bullish", "optimistic", "positive", "success", "successful",
    "record", "records", "high", "highs", "best", "better",
    "win", "wins", "winning", "won", "breakthrough", "innovation",
    "launch", "launches", "launched", "expand", "expands", "expansion",
    "approve", "approved", "approval", "dividend", "dividends",
    "buyback", "buybacks", "repurchase", "acquisition", "acquire",
    "partnership", "partnerships", "deal", "deals", "contract",
    "recovery", "recovering", "recovered", "rebound", "rebounds",
    "momentum", "accelerate", "accelerates", "accelerated",
}

NEGATIVE_WORDS: Set[str] = {
    "miss", "misses", "missed", "missing", "disappoint", "disappoints",
    "weak", "weaker", "weakest", "decline", "declines", "declined",
    "loss", "losses", "losing", "lost", "drop", "drops", "dropped",
    "fall", "falls", "fallen", "falling", "plunge", "plunges", "plunged",
    "crash", "crashes", "crashed", "tumble", "tumbles", "tumbled",
    "sink", "sinks", "sunk", "sinking", "slide", "slides", "slid",
    "downgrade", "downgrades", "downgraded", "sell", "underperform",
    "bearish", "pessimistic", "negative", "fail", "fails", "failed",
    "low", "lows", "worst", "worse", "cut", "cuts", "cutting",
    "layoff", "layoffs", "restructure", "restructuring", "recall",
    "lawsuit", "lawsuits", "sue", "sues", "sued", "investigation",
    "fine", "fines", "fined", "penalty", "penalties", "fraud",
    "default", "defaults", "defaulted", "bankruptcy", "bankrupt",
    "delay", "delays", "delayed", "suspend", "suspends", "suspended",
    "warning", "warnings", "warn", "warns", "warned", "concern",
    "risk", "risks", "risky", "volatile", "volatility", "uncertainty",
    "recession", "slowdown", "contraction", "inflation", "debt",
}

# Event classification patterns
EVENT_PATTERNS: Dict[str, List[str]] = {
    "earnings": [
        r"\bearnings?\b", r"\bquarter(ly)?\s*results?\b", r"\beps\b",
        r"\brevenue\b", r"\bprofit\b", r"\bsales\b", r"\bguidance\b",
        r"\bbeat\s*(expectations?|estimates?)\b", r"\bmiss\s*(expectations?|estimates?)\b",
    ],
    "guidance": [
        r"\bguidance\b", r"\boutlook\b", r"\bforecast\b", r"\bprojection\b",
        r"\braise[ds]?\s*guidance\b", r"\blower[s]?\s*guidance\b",
        r"\bexpect(s|ations?)?\b", r"\bforward[-\s]?looking\b",
    ],
    "downgrade": [
        r"\bdowngrade[ds]?\b", r"\bcut[s]?\s*rating\b", r"\breduce[ds]?\s*target\b",
        r"\blower[s]?\s*price\s*target\b", r"\bsell\s*rating\b",
        r"\bunderweight\b", r"\bunderperform\b",
    ],
    "upgrade": [
        r"\bupgrade[ds]?\b", r"\braise[ds]?\s*rating\b", r"\braise[ds]?\s*target\b",
        r"\bhigher\s*price\s*target\b", r"\bbuy\s*rating\b",
        r"\boverweight\b", r"\boutperform\b",
    ],
    "acquisition": [
        r"\bacquisition\b", r"\bacquire[ds]?\b", r"\bmerger\b", r"\bmerge[ds]?\b",
        r"\bbuyout\b", r"\btakeover\b", r"\bdeal\b", r"\bpurchase[ds]?\b",
    ],
    "fda": [
        r"\bfda\b", r"\bapproval\b", r"\bapprove[ds]?\b", r"\bclinical\s*trial\b",
        r"\bdrug\s*approval\b", r"\bphase\s*[123]\b", r"\bregulatory\b",
    ],
    "lawsuit": [
        r"\blawsuit\b", r"\bsue[ds]?\b", r"\blitigation\b", r"\bsettlement\b",
        r"\binvestigation\b", r"\bsec\s*investigation\b", r"\bfraud\b",
    ],
    "layoff": [
        r"\blayoff[s]?\b", r"\bjob\s*cut[s]?\b", r"\brestructur(e|ing)\b",
        r"\bdownsiz(e|ing)\b", r"\breduc(e|ing)\s*workforce\b",
    ],
}


@dataclass
class ScoredNewsItem:
    """News item with sentiment score and event tags."""

    source: str
    title: str
    summary: str
    url: str
    published_utc: str
    tickers: List[str]
    content_hash: str
    raw_sentiment: float  # -1.0 to 1.0
    decayed_sentiment: float  # After time decay
    event_tags: List[str]
    sector: str
    minutes_ago: float


def compute_raw_sentiment(text: str) -> float:
    """Compute raw sentiment score from text.

    Uses word counting with simple lexicon.
    Returns score in range [-1.0, 1.0].
    """
    if not text:
        return 0.0

    words = set(re.findall(r"\b\w+\b", text.lower()))

    positive_count = len(words & POSITIVE_WORDS)
    negative_count = len(words & NEGATIVE_WORDS)

    total = positive_count + negative_count
    if total == 0:
        return 0.0

    # Normalized difference
    score = (positive_count - negative_count) / total

    # Boost for strong signals (many sentiment words)
    if total >= 3:
        score *= min(1.5, 1.0 + total * 0.1)

    # Clamp to [-1.0, 1.0]
    return max(-1.0, min(1.0, score))


def compute_decayed_sentiment(
    raw_sentiment: float,
    minutes_ago: float,
    half_life_minutes: float = 360.0,  # 6 hours
) -> float:
    """Apply exponential time decay to sentiment.

    Args:
        raw_sentiment: Raw sentiment score.
        minutes_ago: Minutes since publication.
        half_life_minutes: Half-life for decay (default 6h).

    Returns:
        Decayed sentiment score.
    """
    if minutes_ago <= 0:
        return raw_sentiment

    decay_factor = math.exp(-minutes_ago * math.log(2) / half_life_minutes)
    return raw_sentiment * decay_factor


def detect_event_tags(text: str) -> List[str]:
    """Detect event types from text using patterns."""
    tags: Set[str] = set()
    text_lower = text.lower()

    for event_type, patterns in EVENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                tags.add(event_type)
                break  # One match per event type is enough

    return sorted(tags)


def _load_sector_map(data_dir: Path) -> Dict[str, str]:
    """Load sector mapping from CSV."""
    sector_path = data_dir / "sector_map.csv"
    if not sector_path.exists():
        return {}

    import csv

    sector_map: Dict[str, str] = {}
    with open(sector_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "symbol" in row and "sector" in row:
                sector_map[row["symbol"].upper()] = row["sector"]

    return sector_map


def score_news_item(
    parsed_item: Dict,
    reference_time: Optional[datetime] = None,
    sector_map: Optional[Dict[str, str]] = None,
) -> ScoredNewsItem:
    """Score a single parsed news item."""
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    # Parse published time
    published_str = parsed_item.get("published_utc", "")
    try:
        published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)
    except Exception:
        published_dt = reference_time

    # Calculate minutes ago
    delta = reference_time - published_dt
    minutes_ago = max(0.0, delta.total_seconds() / 60.0)

    # Combined text for analysis
    title = parsed_item.get("title", "")
    summary = parsed_item.get("summary", "")
    combined_text = f"{title} {summary}"

    # Compute sentiment
    raw_sentiment = compute_raw_sentiment(combined_text)
    decayed_sentiment = compute_decayed_sentiment(raw_sentiment, minutes_ago)

    # Detect events
    event_tags = detect_event_tags(combined_text)

    # Get sector (from first ticker if available)
    tickers = parsed_item.get("tickers", [])
    sector = "UNKNOWN"
    if sector_map and tickers:
        for ticker in tickers:
            if ticker in sector_map:
                sector = sector_map[ticker]
                break

    return ScoredNewsItem(
        source=parsed_item.get("source", "unknown"),
        title=title,
        summary=summary,
        url=parsed_item.get("url", ""),
        published_utc=published_str,
        tickers=tickers,
        content_hash=parsed_item.get("content_hash", ""),
        raw_sentiment=round(raw_sentiment, 4),
        decayed_sentiment=round(decayed_sentiment, 4),
        event_tags=event_tags,
        sector=sector,
        minutes_ago=round(minutes_ago, 1),
    )


def score_news_file(
    input_path: Path,
    output_path: Path,
    data_dir: Optional[Path] = None,
) -> int:
    """Score all items in a parsed news JSONL file.

    Args:
        input_path: Path to parsed news JSONL.
        output_path: Path to write scored JSONL.
        data_dir: Optional data directory for sector_map.csv.

    Returns:
        Number of items scored.
    """
    if not input_path.exists():
        logger.warning("Input file not found: %s", input_path)
        return 0

    # Load sector map
    sector_map: Dict[str, str] = {}
    if data_dir:
        sector_map = _load_sector_map(data_dir)

    reference_time = datetime.now(timezone.utc)
    scored_items: List[ScoredNewsItem] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                parsed_item = json.loads(line)
            except json.JSONDecodeError:
                continue

            scored = score_news_item(parsed_item, reference_time, sector_map)
            scored_items.append(scored)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in scored_items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    logger.info("Scored %d news items to %s", len(scored_items), output_path)
    return len(scored_items)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score parsed news JSONL")
    parser.add_argument("--input", required=True, dest="input_path", help="Input JSONL path")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--data-dir", default="data", help="Data directory for sector_map.csv")

    args = parser.parse_args()
    score_news_file(Path(args.input_path), Path(args.out), Path(args.data_dir))
