"""Tests for news sentiment scoring."""

import pytest
import math

from traderbot.news.score import (
    compute_raw_sentiment,
    compute_decayed_sentiment,
    detect_event_tags,
)


class TestComputeRawSentiment:
    """Tests for raw sentiment calculation."""

    def test_positive_text(self):
        """Test positive sentiment detection."""
        text = "Company beats expectations, strong growth, profits surge"
        score = compute_raw_sentiment(text)
        assert score > 0.0
        assert score <= 1.0

    def test_negative_text(self):
        """Test negative sentiment detection."""
        text = "Company misses expectations, weak sales, profits decline"
        score = compute_raw_sentiment(text)
        assert score < 0.0
        assert score >= -1.0

    def test_neutral_text(self):
        """Test neutral text produces near-zero sentiment."""
        text = "The company announced a new product today"
        score = compute_raw_sentiment(text)
        assert abs(score) < 0.3

    def test_empty_text(self):
        """Test empty text returns zero."""
        assert compute_raw_sentiment("") == 0.0
        assert compute_raw_sentiment(None) == 0.0

    def test_sentiment_bounded(self):
        """Test sentiment is bounded to [-1, 1]."""
        # Very positive text
        text = "beat beat beat strong growth profit surge rally"
        score = compute_raw_sentiment(text)
        assert score <= 1.0

        # Very negative text
        text = "miss miss miss weak decline loss crash plunge"
        score = compute_raw_sentiment(text)
        assert score >= -1.0


class TestComputeDecayedSentiment:
    """Tests for time decay calculation."""

    def test_no_decay_at_zero_minutes(self):
        """Test no decay for current news."""
        raw = 0.5
        decayed = compute_decayed_sentiment(raw, minutes_ago=0.0)
        assert decayed == raw

    def test_half_decay_at_half_life(self):
        """Test 50% decay at half-life."""
        raw = 1.0
        half_life = 360.0  # 6 hours
        decayed = compute_decayed_sentiment(raw, minutes_ago=half_life)
        assert abs(decayed - 0.5) < 0.01

    def test_decay_preserves_sign(self):
        """Test decay preserves sentiment direction."""
        positive = compute_decayed_sentiment(0.8, minutes_ago=180.0)
        negative = compute_decayed_sentiment(-0.8, minutes_ago=180.0)

        assert positive > 0
        assert negative < 0

    def test_large_time_gives_near_zero(self):
        """Test very old news has negligible sentiment."""
        raw = 1.0
        decayed = compute_decayed_sentiment(raw, minutes_ago=2880.0)  # 48 hours
        assert abs(decayed) < 0.1

    def test_decay_is_monotonic(self):
        """Test decay decreases over time."""
        raw = 1.0
        d1 = compute_decayed_sentiment(raw, minutes_ago=60.0)
        d2 = compute_decayed_sentiment(raw, minutes_ago=120.0)
        d3 = compute_decayed_sentiment(raw, minutes_ago=240.0)

        assert d1 > d2 > d3 > 0


class TestDetectEventTags:
    """Tests for event tag detection."""

    def test_earnings_detection(self):
        """Test earnings event detection."""
        text = "Company reports quarterly earnings, beats EPS expectations"
        tags = detect_event_tags(text)
        assert "earnings" in tags

    def test_guidance_detection(self):
        """Test guidance event detection."""
        text = "Company raises guidance for next quarter"
        tags = detect_event_tags(text)
        assert "guidance" in tags

    def test_downgrade_detection(self):
        """Test downgrade event detection."""
        text = "Analyst downgrades stock, lowers price target"
        tags = detect_event_tags(text)
        assert "downgrade" in tags

    def test_upgrade_detection(self):
        """Test upgrade event detection."""
        text = "Analyst upgrades stock to buy rating"
        tags = detect_event_tags(text)
        assert "upgrade" in tags

    def test_acquisition_detection(self):
        """Test acquisition event detection."""
        text = "Company announces acquisition of competitor for $1B"
        tags = detect_event_tags(text)
        assert "acquisition" in tags

    def test_fda_detection(self):
        """Test FDA event detection."""
        text = "Drug receives FDA approval for new indication"
        tags = detect_event_tags(text)
        assert "fda" in tags

    def test_multiple_events(self):
        """Test detection of multiple events."""
        text = "Company beats earnings expectations and raises guidance"
        tags = detect_event_tags(text)
        assert "earnings" in tags
        assert "guidance" in tags

    def test_no_events(self):
        """Test text with no events."""
        text = "The stock traded higher today"
        tags = detect_event_tags(text)
        assert len(tags) == 0
