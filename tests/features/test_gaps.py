"""Tests for gap detection and regime classification."""

import pytest
import numpy as np
import pandas as pd

from traderbot.features.gaps import (
    GapRegime,
    compute_gap_pct,
    compute_atr_pct,
    compute_prior_trend,
    classify_gap_regime,
    compute_gap_score,
    analyze_gap,
)


class TestComputeGapPct:
    """Tests for gap percentage calculation."""

    def test_gap_up(self):
        """Test gap up calculation."""
        gap = compute_gap_pct(open_price=105.0, prior_close=100.0)
        assert abs(gap - 0.05) < 0.001

    def test_gap_down(self):
        """Test gap down calculation."""
        gap = compute_gap_pct(open_price=95.0, prior_close=100.0)
        assert abs(gap - (-0.05)) < 0.001

    def test_no_gap(self):
        """Test no gap (flat open)."""
        gap = compute_gap_pct(open_price=100.0, prior_close=100.0)
        assert gap == 0.0

    def test_zero_prior_close(self):
        """Test handling of zero prior close."""
        gap = compute_gap_pct(open_price=100.0, prior_close=0.0)
        assert gap == 0.0


class TestComputeAtrPct:
    """Tests for ATR percentage calculation."""

    def test_atr_pct_calculation(self):
        """Test ATR as percentage of price."""
        # Create sample data with known ATR
        n = 30
        close = pd.Series([100.0] * n)
        high = pd.Series([102.0] * n)  # 2% range
        low = pd.Series([98.0] * n)

        atr_pct = compute_atr_pct(high, low, close, period=14)
        # ATR should be around 4 (range from 98 to 102)
        # ATR% = 4/100 = 0.04
        assert atr_pct > 0.03
        assert atr_pct < 0.05

    def test_insufficient_data(self):
        """Test insufficient data returns zero."""
        close = pd.Series([100.0] * 5)
        high = pd.Series([102.0] * 5)
        low = pd.Series([98.0] * 5)

        atr_pct = compute_atr_pct(high, low, close, period=14)
        assert atr_pct == 0.0


class TestComputePriorTrend:
    """Tests for prior trend calculation."""

    def test_uptrend(self):
        """Test uptrend detection."""
        # Price above SMA
        close = pd.Series([100 + i * 0.5 for i in range(30)])
        trend = compute_prior_trend(close, lookback=20)
        assert trend > 0

    def test_downtrend(self):
        """Test downtrend detection."""
        # Price below SMA
        close = pd.Series([130 - i * 0.5 for i in range(30)])
        trend = compute_prior_trend(close, lookback=20)
        assert trend < 0

    def test_trend_bounded(self):
        """Test trend is bounded to [-1, 1]."""
        # Extreme uptrend
        close = pd.Series([100 + i * 5 for i in range(30)])
        trend = compute_prior_trend(close, lookback=20)
        assert trend <= 1.0

        # Extreme downtrend
        close = pd.Series([200 - i * 5 for i in range(30)])
        trend = compute_prior_trend(close, lookback=20)
        assert trend >= -1.0


class TestClassifyGapRegime:
    """Tests for gap regime classification."""

    def test_small_gap_is_continuation(self):
        """Test small gaps are classified as continuation."""
        # Gap < 0.5 ATR
        label = classify_gap_regime(gap_pct=0.01, atr_pct=0.03, prior_trend=0.5)
        assert label == GapRegime.CONTINUATION

    def test_aligned_gap_is_continuation(self):
        """Test gap aligned with trend is continuation."""
        # Positive gap with positive trend
        label = classify_gap_regime(gap_pct=0.03, atr_pct=0.03, prior_trend=0.5)
        assert label == GapRegime.CONTINUATION

    def test_large_opposing_gap_is_reversion(self):
        """Test large gap against trend is reversion."""
        # Large positive gap with negative trend
        label = classify_gap_regime(gap_pct=0.05, atr_pct=0.03, prior_trend=-0.5)
        assert label == GapRegime.REVERSION

    def test_zero_atr_is_neutral(self):
        """Test zero ATR returns neutral."""
        label = classify_gap_regime(gap_pct=0.02, atr_pct=0.0, prior_trend=0.5)
        assert label == GapRegime.NEUTRAL


class TestComputeGapScore:
    """Tests for gap score calculation."""

    def test_score_bounded(self):
        """Test gap score is bounded."""
        score = compute_gap_score(gap_pct=0.10, atr_pct=0.02, gap_label=GapRegime.CONTINUATION)
        assert score >= -1.0
        assert score <= 1.0

    def test_continuation_preserves_direction(self):
        """Test continuation preserves gap direction."""
        score_up = compute_gap_score(gap_pct=0.05, atr_pct=0.03, gap_label=GapRegime.CONTINUATION)
        score_down = compute_gap_score(gap_pct=-0.05, atr_pct=0.03, gap_label=GapRegime.CONTINUATION)

        assert score_up > 0
        assert score_down < 0

    def test_reversion_reverses_direction(self):
        """Test reversion reverses gap direction."""
        score = compute_gap_score(gap_pct=0.05, atr_pct=0.03, gap_label=GapRegime.REVERSION)
        # Positive gap with reversion should give negative score
        assert score < 0


class TestAnalyzeGap:
    """Tests for full gap analysis."""

    def test_analyze_with_sufficient_data(self):
        """Test gap analysis with sufficient data."""
        # Create sample OHLCV data
        n = 30
        dates = pd.date_range("2024-01-01", periods=n)
        df = pd.DataFrame({
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [102.0 + i * 0.1 for i in range(n)],
            "low": [98.0 + i * 0.1 for i in range(n)],
            "close": [101.0 + i * 0.1 for i in range(n)],
            "volume": [1000000] * n,
        }, index=dates)

        # Set last open to create a gap
        df.iloc[-1, df.columns.get_loc("open")] = 105.0

        result = analyze_gap(df)

        assert result is not None
        assert result.gap_pct != 0.0
        assert result.gap_label in [GapRegime.CONTINUATION, GapRegime.REVERSION, GapRegime.NEUTRAL]
        assert -1.0 <= result.gap_score <= 1.0

    def test_analyze_insufficient_data(self):
        """Test gap analysis with insufficient data."""
        df = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        })

        result = analyze_gap(df)
        assert result is None

    def test_analyze_missing_columns(self):
        """Test gap analysis with missing columns."""
        df = pd.DataFrame({
            "open": [100.0] * 20,
            "close": [101.0] * 20,
            # Missing high and low
        })

        result = analyze_gap(df)
        assert result is None
