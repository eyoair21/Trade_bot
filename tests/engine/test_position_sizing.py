"""Tests for position sizing module."""

import numpy as np
import pandas as pd
import pytest

from traderbot.engine.position_sizing import (
    PositionSizer,
    SizeResult,
    SizerType,
    calculate_volatility,
    fixed_fraction,
    kelly_fraction,
    vol_target,
)


class TestFixedFraction:
    """Tests for fixed_fraction position sizer."""

    def test_basic_calculation(self) -> None:
        """Test basic fixed fraction calculation."""
        result = fixed_fraction(equity=100000, price=100.0, fraction=0.1)

        assert isinstance(result, SizeResult)
        assert result.shares == 100  # 10% of 100k / 100 = 100 shares
        assert result.notional == 10000.0
        assert result.fraction == 0.1

    def test_fractional_shares_truncated(self) -> None:
        """Test fractional shares are truncated to int."""
        result = fixed_fraction(equity=100000, price=333.0, fraction=0.1)

        # 10000 / 333 = 30.03, should truncate to 30
        assert result.shares == 30
        assert result.notional == 30 * 333.0

    def test_zero_equity(self) -> None:
        """Test zero equity returns zero shares."""
        result = fixed_fraction(equity=0, price=100.0, fraction=0.1)

        assert result.shares == 0
        assert result.notional == 0.0

    def test_zero_price(self) -> None:
        """Test zero price returns zero shares."""
        result = fixed_fraction(equity=100000, price=0, fraction=0.1)

        assert result.shares == 0

    def test_negative_values(self) -> None:
        """Test negative values are handled."""
        result = fixed_fraction(equity=-100000, price=100.0, fraction=0.1)
        assert result.shares == 0

        result = fixed_fraction(equity=100000, price=-100.0, fraction=0.1)
        assert result.shares == 0


class TestVolTarget:
    """Tests for vol_target position sizer."""

    def test_basic_calculation(self) -> None:
        """Test basic vol-targeted sizing."""
        # If asset vol is 25% and target vol is 10%, position = 10/25 = 40%
        result = vol_target(
            equity=100000,
            price=100.0,
            volatility=0.25,
            target_vol=0.10,
        )

        assert isinstance(result, SizeResult)
        expected_fraction = 0.10 / 0.25  # = 0.4
        expected_shares = int(100000 * expected_fraction / 100)
        assert result.shares == expected_shares
        assert result.fraction == pytest.approx(expected_fraction, rel=0.01)

    def test_high_vol_reduces_position(self) -> None:
        """Test high volatility reduces position size."""
        # Low vol asset
        result_low = vol_target(equity=100000, price=100.0, volatility=0.10, target_vol=0.10)

        # High vol asset
        result_high = vol_target(equity=100000, price=100.0, volatility=0.50, target_vol=0.10)

        assert result_low.shares > result_high.shares

    def test_capped_at_100_percent(self) -> None:
        """Test position is capped at 100% of equity."""
        # Very low vol would give >100% position, should be capped
        result = vol_target(
            equity=100000,
            price=100.0,
            volatility=0.05,
            target_vol=0.20,
        )

        # 0.20 / 0.05 = 4.0 (400%) -> should cap at 1.0 (100%)
        assert result.fraction == 1.0
        assert result.shares == 1000  # 100000 / 100

    def test_zero_volatility(self) -> None:
        """Test zero volatility returns zero shares."""
        result = vol_target(equity=100000, price=100.0, volatility=0, target_vol=0.10)

        assert result.shares == 0


class TestKellyFraction:
    """Tests for kelly_fraction position sizer."""

    def test_basic_calculation(self) -> None:
        """Test basic Kelly criterion calculation."""
        # With 60% win rate and 1.5 win/loss ratio
        # Kelly = (0.6 * 1.5 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.333
        result = kelly_fraction(
            equity=100000,
            price=100.0,
            win_prob=0.6,
            win_loss_ratio=1.5,
            cap=0.5,
        )

        expected_kelly = (0.6 * 1.5 - 0.4) / 1.5
        assert result.fraction == pytest.approx(expected_kelly, rel=0.01)

    def test_kelly_capped(self) -> None:
        """Test Kelly is capped at specified value."""
        # High win prob and ratio would give large Kelly, should be capped
        result = kelly_fraction(
            equity=100000,
            price=100.0,
            win_prob=0.8,
            win_loss_ratio=3.0,
            cap=0.25,  # Half-Kelly style cap
        )

        assert result.fraction == 0.25

    def test_negative_kelly_no_trade(self) -> None:
        """Test negative Kelly results in no trade."""
        # Low win prob with bad ratio gives negative Kelly
        result = kelly_fraction(
            equity=100000,
            price=100.0,
            win_prob=0.3,
            win_loss_ratio=0.5,
            cap=0.5,
        )

        assert result.shares == 0
        assert "Negative Kelly" in result.reason

    def test_invalid_win_prob(self) -> None:
        """Test invalid win probability returns zero."""
        result = kelly_fraction(equity=100000, price=100.0, win_prob=0, win_loss_ratio=1.5)
        assert result.shares == 0

        result = kelly_fraction(equity=100000, price=100.0, win_prob=1.0, win_loss_ratio=1.5)
        assert result.shares == 0

    def test_invalid_win_loss_ratio(self) -> None:
        """Test invalid win/loss ratio returns zero."""
        result = kelly_fraction(equity=100000, price=100.0, win_prob=0.6, win_loss_ratio=0)
        assert result.shares == 0

        result = kelly_fraction(equity=100000, price=100.0, win_prob=0.6, win_loss_ratio=-1)
        assert result.shares == 0


class TestCalculateVolatility:
    """Tests for volatility calculation."""

    def test_basic_volatility(self) -> None:
        """Test basic volatility calculation."""
        np.random.seed(42)
        # Create price series with known volatility
        prices = pd.Series([100.0 + i + np.random.randn() for i in range(30)])

        vol = calculate_volatility(prices, window=20, annualize=False)

        assert vol > 0
        assert vol < 1  # Daily vol should be small

    def test_annualized_volatility(self) -> None:
        """Test annualized volatility is larger."""
        np.random.seed(42)
        prices = pd.Series([100.0 + i + np.random.randn() for i in range(30)])

        vol_daily = calculate_volatility(prices, window=20, annualize=False)
        vol_annual = calculate_volatility(prices, window=20, annualize=True)

        # Annualized should be ~sqrt(252) times larger
        assert vol_annual > vol_daily * 10  # Rough check

    def test_insufficient_data(self) -> None:
        """Test insufficient data returns zero."""
        prices = pd.Series([100.0, 101.0, 102.0])

        vol = calculate_volatility(prices, window=20)

        assert vol == 0.0


class TestPositionSizer:
    """Tests for PositionSizer class."""

    def test_fixed_sizer(self) -> None:
        """Test fixed fraction sizer."""
        sizer = PositionSizer(sizer_type=SizerType.FIXED, fixed_frac=0.1)

        result = sizer.calculate_size(equity=100000, price=100.0)

        assert result.shares == 100

    def test_vol_sizer(self) -> None:
        """Test volatility-targeted sizer."""
        sizer = PositionSizer(sizer_type=SizerType.VOL, vol_target=0.15)

        result = sizer.calculate_size(equity=100000, price=100.0, volatility=0.30)

        # Position = 0.15 / 0.30 = 0.5 = 50%
        assert result.shares == 500

    def test_vol_sizer_fallback_no_volatility(self) -> None:
        """Test vol sizer falls back to fixed when no volatility."""
        sizer = PositionSizer(
            sizer_type=SizerType.VOL,
            vol_target=0.15,
            fixed_frac=0.1,
        )

        result = sizer.calculate_size(equity=100000, price=100.0, volatility=0)

        # Should fall back to fixed fraction
        assert result.shares == 100  # 10% of 100k / 100

    def test_kelly_sizer(self) -> None:
        """Test Kelly criterion sizer."""
        sizer = PositionSizer(sizer_type=SizerType.KELLY, kelly_cap=0.25)

        result = sizer.calculate_size(
            equity=100000,
            price=100.0,
            win_prob=0.6,
            win_loss_ratio=1.5,
        )

        assert result.shares > 0

    def test_string_sizer_type(self) -> None:
        """Test sizer can be created with string type."""
        sizer = PositionSizer(sizer_type="fixed", fixed_frac=0.2)

        result = sizer.calculate_size(equity=100000, price=100.0)

        assert result.shares == 200  # 20% of 100k / 100
