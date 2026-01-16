"""RSS feed fetching for news ingestion.

Fetches news from public RSS feeds without requiring API keys.
Outputs raw news items as JSONL for downstream processing.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import ssl
import time

from traderbot.logging_setup import get_logger

logger = get_logger("news.pull")

# Timeout for RSS requests (seconds)
REQUEST_TIMEOUT = 15

# User agent to avoid blocking
USER_AGENT = "TraderBot/1.0 (RSS News Aggregator; +https://github.com/eyoair21/Trade_bot)"


@dataclass
class RawNewsItem:
    """Raw news item from RSS feed."""

    source: str
    title: str
    summary: str
    url: str
    published_utc: str  # ISO format


def _parse_rss_date(date_str: str) -> Optional[datetime]:
    """Parse various RSS date formats to datetime."""
    if not date_str:
        return None

    # Common RSS date formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",  # RFC 822
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ]

    # Clean up timezone notation
    date_str = date_str.strip()
    date_str = re.sub(r"\s+", " ", date_str)

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    # Try parsing with dateutil as fallback
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(date_str)
    except Exception:
        pass

    logger.debug("Could not parse date: %s", date_str)
    return None


def _clean_text(text: Optional[str]) -> str:
    """Clean HTML tags and normalize whitespace."""
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _get_text(element: Optional[ET.Element], default: str = "") -> str:
    """Safely get text from XML element."""
    if element is None:
        return default
    return element.text or default


def _parse_rss_feed(xml_content: str, source_url: str) -> Iterator[RawNewsItem]:
    """Parse RSS/Atom feed XML into news items."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.warning("XML parse error for %s: %s", source_url, e)
        return

    source_name = source_url.split("/")[2]  # Extract domain

    # Handle Atom feeds
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    atom_entries = root.findall(".//atom:entry", ns) or root.findall(".//{http://www.w3.org/2005/Atom}entry")

    if atom_entries:
        for entry in atom_entries:
            title = _get_text(entry.find("atom:title", ns)) or _get_text(
                entry.find("{http://www.w3.org/2005/Atom}title")
            )
            summary = _get_text(entry.find("atom:summary", ns)) or _get_text(
                entry.find("{http://www.w3.org/2005/Atom}summary")
            ) or _get_text(entry.find("atom:content", ns)) or _get_text(
                entry.find("{http://www.w3.org/2005/Atom}content")
            )

            link_elem = entry.find("atom:link", ns) or entry.find("{http://www.w3.org/2005/Atom}link")
            url = link_elem.get("href", "") if link_elem is not None else ""

            published = _get_text(entry.find("atom:published", ns)) or _get_text(
                entry.find("{http://www.w3.org/2005/Atom}published")
            ) or _get_text(entry.find("atom:updated", ns)) or _get_text(
                entry.find("{http://www.w3.org/2005/Atom}updated")
            )

            dt = _parse_rss_date(published)
            published_utc = dt.isoformat() if dt else datetime.now(timezone.utc).isoformat()

            yield RawNewsItem(
                source=source_name,
                title=_clean_text(title),
                summary=_clean_text(summary)[:1000],  # Limit summary length
                url=url,
                published_utc=published_utc,
            )
        return

    # Handle RSS 2.0 feeds
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//channel/item")

    for item in items:
        title = _get_text(item.find("title"))
        summary = _get_text(item.find("description")) or _get_text(item.find("content:encoded"))
        url = _get_text(item.find("link"))
        published = _get_text(item.find("pubDate"))

        dt = _parse_rss_date(published)
        published_utc = dt.isoformat() if dt else datetime.now(timezone.utc).isoformat()

        yield RawNewsItem(
            source=source_name,
            title=_clean_text(title),
            summary=_clean_text(summary)[:1000],
            url=url,
            published_utc=published_utc,
        )


def fetch_single_feed(url: str, timeout: int = REQUEST_TIMEOUT) -> List[RawNewsItem]:
    """Fetch and parse a single RSS feed."""
    items: List[RawNewsItem] = []

    try:
        # Create SSL context that doesn't verify certificates (for some feeds)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout, context=ctx) as response:
            content = response.read().decode("utf-8", errors="replace")

        for item in _parse_rss_feed(content, url):
            if item.title:  # Only include items with titles
                items.append(item)

        logger.info("Fetched %d items from %s", len(items), url)

    except HTTPError as e:
        logger.warning("HTTP error fetching %s: %s", url, e.code)
    except URLError as e:
        logger.warning("URL error fetching %s: %s", url, e.reason)
    except TimeoutError:
        logger.warning("Timeout fetching %s", url)
    except Exception as e:
        logger.warning("Error fetching %s: %s", url, e)

    return items


def load_sources(sources_path: Path) -> List[str]:
    """Load RSS source URLs from file."""
    urls: List[str] = []

    if not sources_path.exists():
        logger.warning("Sources file not found: %s", sources_path)
        return urls

    with open(sources_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    return urls


def fetch_rss_feeds(
    sources_path: Path,
    output_path: Path,
    delay_between_requests: float = 1.0,
) -> int:
    """Fetch all RSS feeds and write to JSONL.

    Args:
        sources_path: Path to file containing RSS URLs.
        output_path: Path to write JSONL output.
        delay_between_requests: Seconds to wait between requests.

    Returns:
        Number of items written.
    """
    urls = load_sources(sources_path)
    if not urls:
        logger.warning("No RSS URLs loaded from %s", sources_path)
        return 0

    all_items: List[RawNewsItem] = []

    for i, url in enumerate(urls):
        items = fetch_single_feed(url)
        all_items.extend(items)

        # Rate limiting
        if i < len(urls) - 1:
            time.sleep(delay_between_requests)

    # Write to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    logger.info("Wrote %d news items to %s", len(all_items), output_path)
    return len(all_items)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch RSS news feeds")
    parser.add_argument("--sources", required=True, help="Path to RSS sources file")
    parser.add_argument("--out", required=True, help="Output JSONL path")

    args = parser.parse_args()
    fetch_rss_feeds(Path(args.sources), Path(args.out))
