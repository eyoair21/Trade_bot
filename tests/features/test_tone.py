"""Tests for news tone scores feature."""

from datetime import date

from traderbot.features.sentiment import (
    _get_deterministic_tone,
    _rule_based_tone,
    get_tone_scores,
    is_finbert_available,
)


class TestDeterministicTone:
    """Tests for deterministic tone generation."""

    def test_same_input_same_output(self) -> None:
        """Test deterministic tone is reproducible."""
        tone1 = _get_deterministic_tone("AAPL", "2023-01-15", seed=42)
        tone2 = _get_deterministic_tone("AAPL", "2023-01-15", seed=42)

        assert tone1 == tone2

    def test_different_ticker_different_output(self) -> None:
        """Test different tickers produce different tones."""
        tone_aapl = _get_deterministic_tone("AAPL", "2023-01-15", seed=42)
        tone_msft = _get_deterministic_tone("MSFT", "2023-01-15", seed=42)

        assert tone_aapl != tone_msft

    def test_different_date_different_output(self) -> None:
        """Test different dates produce different tones."""
        tone1 = _get_deterministic_tone("AAPL", "2023-01-15", seed=42)
        tone2 = _get_deterministic_tone("AAPL", "2023-01-16", seed=42)

        assert tone1 != tone2

    def test_different_seed_different_output(self) -> None:
        """Test different seeds produce different tones."""
        tone1 = _get_deterministic_tone("AAPL", "2023-01-15", seed=42)
        tone2 = _get_deterministic_tone("AAPL", "2023-01-15", seed=123)

        assert tone1 != tone2

    def test_output_in_range(self) -> None:
        """Test tone output is in [-1, 1]."""
        for i in range(100):
            tone = _get_deterministic_tone(f"TICK{i}", f"2023-01-{i % 28 + 1:02d}", seed=i)
            assert -1.0 <= tone <= 1.0


class TestRuleBasedTone:
    """Tests for rule-based sentiment."""

    def test_empty_headlines(self) -> None:
        """Test empty headlines return 0."""
        assert _rule_based_tone([]) == 0.0

    def test_positive_headlines(self) -> None:
        """Test positive headlines return positive score."""
        headlines = [
            "Stock surges on strong earnings",
            "Company beats expectations",
            "Analysts upgrade to buy",
        ]
        score = _rule_based_tone(headlines)
        assert score > 0

    def test_negative_headlines(self) -> None:
        """Test negative headlines return negative score."""
        headlines = [
            "Stock falls on weak sales",
            "Company misses earnings",
            "Analysts downgrade after crash",
        ]
        score = _rule_based_tone(headlines)
        assert score < 0

    def test_neutral_headlines(self) -> None:
        """Test neutral headlines return 0."""
        headlines = [
            "Company announces new product",
            "CEO speaks at conference",
            "Quarterly report released",
        ]
        score = _rule_based_tone(headlines)
        assert score == 0.0

    def test_mixed_headlines(self) -> None:
        """Test mixed headlines return moderate score."""
        headlines = [
            "Stock surges on positive news",  # positive
            "Market falls on economic fears",  # negative
        ]
        score = _rule_based_tone(headlines)
        # Should be around 0 (average of positive and negative)
        assert -0.5 <= score <= 0.5

    def test_score_in_range(self) -> None:
        """Test score is always in [-1, 1]."""
        headlines = ["bullish bullish bullish buy buy buy"]
        assert -1.0 <= _rule_based_tone(headlines) <= 1.0

        headlines = ["bearish bearish crash sell sell"]
        assert -1.0 <= _rule_based_tone(headlines) <= 1.0


class TestGetToneScores:
    """Tests for get_tone_scores function."""

    def test_basic_call(self) -> None:
        """Test basic function call returns dict."""
        tickers = ["AAPL", "MSFT"]
        start = date(2023, 1, 1)
        end = date(2023, 1, 5)

        scores = get_tone_scores(tickers, start, end)

        assert isinstance(scores, dict)
        assert "AAPL" in scores
        assert "MSFT" in scores

    def test_string_dates(self) -> None:
        """Test function accepts string dates."""
        scores = get_tone_scores(
            ["AAPL"],
            start="2023-01-01",
            end="2023-01-05",
        )

        assert "AAPL" in scores

    def test_scores_in_range(self) -> None:
        """Test all scores are in [-1, 1]."""
        tickers = ["AAPL", "MSFT", "NVDA", "GOOG"]
        scores = get_tone_scores(
            tickers,
            start=date(2023, 1, 1),
            end=date(2023, 1, 10),
        )

        for ticker, score in scores.items():
            assert -1.0 <= score <= 1.0, f"{ticker} score {score} out of range"

    def test_reproducible_with_seed(self) -> None:
        """Test scores are reproducible with same seed."""
        tickers = ["AAPL", "MSFT"]
        start = date(2023, 1, 1)
        end = date(2023, 1, 5)

        scores1 = get_tone_scores(tickers, start, end, use_finbert=False, seed=42)
        scores2 = get_tone_scores(tickers, start, end, use_finbert=False, seed=42)

        assert scores1 == scores2

    def test_different_seed_different_scores(self) -> None:
        """Test different seeds produce different scores."""
        tickers = ["AAPL"]
        start = date(2023, 1, 1)
        end = date(2023, 1, 5)

        scores1 = get_tone_scores(tickers, start, end, use_finbert=False, seed=42)
        scores2 = get_tone_scores(tickers, start, end, use_finbert=False, seed=123)

        assert scores1["AAPL"] != scores2["AAPL"]

    def test_single_day_range(self) -> None:
        """Test single day date range."""
        day = date(2023, 1, 15)
        scores = get_tone_scores(["AAPL"], day, day)

        assert "AAPL" in scores

    def test_empty_tickers(self) -> None:
        """Test empty ticker list returns empty dict."""
        scores = get_tone_scores([], date(2023, 1, 1), date(2023, 1, 5))
        assert scores == {}


class TestFinBERTAvailability:
    """Tests for FinBERT availability check."""

    def test_is_finbert_available_returns_bool(self) -> None:
        """Test is_finbert_available returns boolean."""
        result = is_finbert_available()
        assert isinstance(result, bool)
