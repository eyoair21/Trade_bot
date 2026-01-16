"""Tests for news parsing and ticker extraction."""

import pytest

from traderbot.news.parse import extract_tickers, _compute_hash, parse_news_item


class TestExtractTickers:
    """Tests for ticker extraction."""

    def test_dollar_sign_ticker(self):
        """Test $TICKER format extraction."""
        text = "Just bought $AAPL and $MSFT"
        tickers = extract_tickers(text)
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_exchange_ticker(self):
        """Test NASDAQ:TICKER format extraction."""
        text = "Trading NASDAQ:GOOGL today"
        tickers = extract_tickers(text)
        assert "GOOGL" in tickers

    def test_parentheses_ticker(self):
        """Test ticker in parentheses."""
        text = "Apple (AAPL) reported earnings"
        tickers = extract_tickers(text)
        assert "AAPL" in tickers

    def test_company_name_alias(self):
        """Test company name to ticker mapping."""
        text = "Microsoft announced new products"
        tickers = extract_tickers(text)
        assert "MSFT" in tickers

    def test_multiple_tickers(self):
        """Test extraction of multiple tickers."""
        text = "$AAPL $MSFT and NVIDIA all up today"
        tickers = extract_tickers(text)
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "NVDA" in tickers

    def test_no_tickers(self):
        """Test text with no tickers."""
        text = "The market was volatile today"
        tickers = extract_tickers(text)
        assert len(tickers) == 0

    def test_excludes_common_words(self):
        """Test that common words are not extracted as tickers."""
        text = "The CEO said IT was a good decision FOR the company"
        tickers = extract_tickers(text)
        assert "CEO" not in tickers
        assert "IT" not in tickers
        assert "FOR" not in tickers

    def test_case_insensitive_alias(self):
        """Test case-insensitive company name matching."""
        text = "apple announced new iPhone"
        tickers = extract_tickers(text)
        assert "AAPL" in tickers


class TestComputeHash:
    """Tests for content hashing."""

    def test_same_content_same_hash(self):
        """Test same content produces same hash."""
        h1 = _compute_hash("https://example.com", "Title")
        h2 = _compute_hash("https://example.com", "Title")
        assert h1 == h2

    def test_different_content_different_hash(self):
        """Test different content produces different hash."""
        h1 = _compute_hash("https://example.com/1", "Title 1")
        h2 = _compute_hash("https://example.com/2", "Title 2")
        assert h1 != h2

    def test_hash_is_16_chars(self):
        """Test hash is truncated to 16 characters."""
        h = _compute_hash("https://example.com", "Title")
        assert len(h) == 16


class TestParseNewsItem:
    """Tests for parsing news items."""

    def test_parse_basic_item(self):
        """Test parsing a basic news item."""
        raw = {
            "source": "test.com",
            "title": "Apple (AAPL) beats earnings",
            "summary": "Strong iPhone sales drove results",
            "url": "https://test.com/article",
            "published_utc": "2024-01-15T12:00:00Z",
        }
        parsed = parse_news_item(raw)

        assert parsed.source == "test.com"
        assert parsed.title == "Apple (AAPL) beats earnings"
        assert "AAPL" in parsed.tickers
        assert len(parsed.content_hash) == 16

    def test_parse_extracts_multiple_tickers(self):
        """Test parsing extracts multiple tickers."""
        raw = {
            "source": "test.com",
            "title": "$AAPL and $MSFT surge",
            "summary": "Tech giants rally",
            "url": "https://test.com/article",
            "published_utc": "2024-01-15T12:00:00Z",
        }
        parsed = parse_news_item(raw)

        assert "AAPL" in parsed.tickers
        assert "MSFT" in parsed.tickers

    def test_parse_normalizes_timestamp(self):
        """Test timestamp normalization."""
        raw = {
            "source": "test.com",
            "title": "Test",
            "summary": "",
            "url": "https://test.com",
            "published_utc": "2024-01-15T12:00:00Z",
        }
        parsed = parse_news_item(raw)

        assert "2024-01-15" in parsed.published_utc
