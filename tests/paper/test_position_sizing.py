"""Tests for position sizing."""

import pytest
import pandas as pd
import numpy as np

from traderbot.paper.position_sizing import (
    PositionSizer,
    compute_position_size,
)


class TestPositionSizer:
    """Tests for PositionSizer class."""

    def test_size_respects_max_position(self):
        """Test position size respects max position percentage."""
        sizer = PositionSizer(
            target_vol_per_position=0.20,
            max_position_pct=0.10,
        )

        # Create low volatility data (would suggest large position)
        n = 30
        close = pd.Series([100.0] * n)
        high = pd.Series([100.5] * n)  # Very low range
        low = pd.Series([99.5] * n)

        result = sizer.compute_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000.0,
            high=high,
            low=low,
            close=close,
        )

        # Max position = 10% of 100k = 10,000
        # Max shares at 100/share = 100
        assert result.notional <= 10_000.0
        assert result.position_weight <= 0.10

    def test_high_vol_reduces_size(self):
        """Test high volatility reduces position size."""
        sizer = PositionSizer(target_vol_per_position=0.20)

        n = 30

        # Low volatility
        low_vol_close = pd.Series([100.0] * n)
        low_vol_high = pd.Series([101.0] * n)
        low_vol_low = pd.Series([99.0] * n)

        # High volatility
        high_vol_close = pd.Series([100.0] * n)
        high_vol_high = pd.Series([110.0] * n)
        high_vol_low = pd.Series([90.0] * n)

        low_vol_result = sizer.compute_size(
            "AAPL", 100.0, 100_000.0,
            low_vol_high, low_vol_low, low_vol_close,
        )

        high_vol_result = sizer.compute_size(
            "AAPL", 100.0, 100_000.0,
            high_vol_high, high_vol_low, high_vol_close,
        )

        # High vol should have smaller position
        assert high_vol_result.shares < low_vol_result.shares

    def test_minimum_shares(self):
        """Test minimum shares constraint."""
        sizer = PositionSizer(
            target_vol_per_position=0.01,  # Very low target
            min_shares=1,
        )

        n = 30
        close = pd.Series([100.0] * n)
        high = pd.Series([150.0] * n)  # High vol
        low = pd.Series([50.0] * n)

        result = sizer.compute_size(
            "AAPL", 100.0, 100_000.0,
            high, low, close,
        )

        assert result.shares >= 1

    def test_result_contains_vol_estimate(self):
        """Test result includes volatility estimate."""
        sizer = PositionSizer()

        n = 30
        close = pd.Series([100.0] * n)
        high = pd.Series([102.0] * n)
        low = pd.Series([98.0] * n)

        result = sizer.compute_size(
            "AAPL", 100.0, 100_000.0,
            high, low, close,
        )

        assert result.estimated_vol > 0
        assert result.target_vol == 0.20  # Default


class TestComputePositionSize:
    """Tests for convenience function."""

    def test_convenience_function(self):
        """Test convenience function works correctly."""
        n = 30
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [102.0] * n,
            "low": [98.0] * n,
            "close": [101.0] * n,
            "volume": [1000000] * n,
        })

        result = compute_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000.0,
            df=df,
            target_vol=0.20,
            max_position_pct=0.10,
        )

        assert result.symbol == "AAPL"
        assert result.shares > 0
        assert result.notional > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_price(self):
        """Test handling of zero price."""
        sizer = PositionSizer()

        n = 30
        close = pd.Series([100.0] * n)
        high = pd.Series([102.0] * n)
        low = pd.Series([98.0] * n)

        result = sizer.compute_size(
            "AAPL", 0.0, 100_000.0,
            high, low, close,
        )

        assert result.shares == 0

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        sizer = PositionSizer()

        close = pd.Series([100.0] * 5)  # Too short
        high = pd.Series([102.0] * 5)
        low = pd.Series([98.0] * 5)

        result = sizer.compute_size(
            "AAPL", 100.0, 100_000.0,
            high, low, close,
        )

        # Should use default volatility assumption
        assert result.shares > 0
