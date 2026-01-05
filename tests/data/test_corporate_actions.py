"""Tests for corporate actions module."""

import logging
from datetime import date

import pandas as pd
import pytest

from traderbot.data.corporate_actions import (
    Dividend,
    Split,
    adjust_for_dividend,
    adjust_for_split,
    apply_corporate_actions,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create sample price data for testing."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [1000000] * 10,
        }
    )


class TestAdjustForSplit:
    """Tests for split adjustment."""

    def test_4_for_1_split(self, sample_prices: pd.DataFrame) -> None:
        """Test 4:1 split adjustment."""
        split = Split(ticker="TEST", ex_date=date(2023, 1, 6), ratio=4.0)

        adjusted = adjust_for_split(sample_prices, split)

        # Prices before ex-date should be divided by 4
        # First 5 rows are before Jan 6
        assert adjusted.loc[0, "close"] == pytest.approx(100.5 / 4.0)
        assert adjusted.loc[4, "close"] == pytest.approx(104.5 / 4.0)

        # Prices on/after ex-date should be unchanged
        assert adjusted.loc[5, "close"] == 105.5
        assert adjusted.loc[9, "close"] == 109.5

    def test_2_for_1_split(self, sample_prices: pd.DataFrame) -> None:
        """Test 2:1 split adjustment."""
        split = Split(ticker="TEST", ex_date=date(2023, 1, 6), ratio=2.0)

        adjusted = adjust_for_split(sample_prices, split)

        assert adjusted.loc[0, "close"] == pytest.approx(100.5 / 2.0)
        assert adjusted.loc[5, "close"] == 105.5

    def test_volume_adjusted(self, sample_prices: pd.DataFrame) -> None:
        """Test volume is multiplied by split ratio."""
        split = Split(ticker="TEST", ex_date=date(2023, 1, 6), ratio=4.0)

        adjusted = adjust_for_split(sample_prices, split)

        # Volume before ex-date should be multiplied by 4
        assert adjusted.loc[0, "volume"] == 4000000
        # Volume on/after ex-date unchanged
        assert adjusted.loc[5, "volume"] == 1000000

    def test_invalid_ratio_logged(
        self, sample_prices: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test invalid split ratio is logged."""
        split = Split(ticker="TEST", ex_date=date(2023, 1, 6), ratio=0.0)

        with caplog.at_level(logging.WARNING, logger="traderbot"):
            adjusted = adjust_for_split(sample_prices, split)

        assert "Invalid split ratio" in caplog.text
        # Prices unchanged
        assert adjusted.loc[0, "close"] == 100.5


class TestAdjustForDividend:
    """Tests for dividend adjustment."""

    def test_basic_dividend_adjustment(self, sample_prices: pd.DataFrame) -> None:
        """Test basic dividend adjustment."""
        # Ex-date Jan 6, prev_close is Jan 5 = 104.5, dividend = 1.0
        dividend = Dividend(ticker="TEST", ex_date=date(2023, 1, 6), amount=1.0)

        adjusted = adjust_for_dividend(sample_prices, dividend)

        # Factor = (104.5 - 1.0) / 104.5 = 103.5 / 104.5 ≈ 0.9904
        factor = (104.5 - 1.0) / 104.5

        assert adjusted.loc[0, "close"] == pytest.approx(100.5 * factor)
        assert adjusted.loc[4, "close"] == pytest.approx(104.5 * factor)

        # Prices on/after ex-date unchanged
        assert adjusted.loc[5, "close"] == 105.5

    def test_dividend_guard_prev_close_small(
        self, sample_prices: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test dividend guard for prev_close <= EPS."""
        # Create prices with very small prev_close
        prices = sample_prices.copy()
        prices.loc[4, "close"] = 1e-7  # prev_close before ex-date

        dividend = Dividend(ticker="TEST", ex_date=date(2023, 1, 6), amount=0.01)

        with caplog.at_level(logging.WARNING, logger="traderbot"):
            adjusted = adjust_for_dividend(prices, dividend)

        # Exact warning message must match
        assert (
            "CA: skip dividend adj (prev_close<=EPS) ticker=TEST "
            "ex_date=2023-01-06 prev_close=0.0000001000 amt=0.0100000000" in caplog.text
        )

        # Prices unchanged
        assert adjusted.loc[0, "close"] == prices.loc[0, "close"]

    def test_dividend_guard_amount_too_large(
        self, sample_prices: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test dividend guard for amount >= prev_close."""
        # Dividend amount >= prev_close (104.5)
        dividend = Dividend(ticker="TEST", ex_date=date(2023, 1, 6), amount=104.5)

        with caplog.at_level(logging.WARNING, logger="traderbot"):
            adjusted = adjust_for_dividend(sample_prices, dividend)

        # Exact warning message must match
        assert (
            "CA: skip dividend adj (amt>=prev_close) ticker=TEST "
            "ex_date=2023-01-06 prev_close=104.5000000000 amt=104.5000000000" in caplog.text
        )

        # Prices unchanged
        assert adjusted.loc[0, "close"] == 100.5

    def test_dividend_guard_amount_exceeds_prev_close(
        self, sample_prices: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test dividend guard for amount > prev_close."""
        dividend = Dividend(ticker="TEST", ex_date=date(2023, 1, 6), amount=200.0)

        with caplog.at_level(logging.WARNING, logger="traderbot"):
            adjust_for_dividend(sample_prices, dividend)

        assert "CA: skip dividend adj (amt>=prev_close)" in caplog.text

    def test_dividend_guard_acceptance_case(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test exact acceptance case: prev_close=0.0001, dividend=0.01.

        This is the required acceptance test case from the spec:
        - prev_close=0.0001 (very small but > EPS)
        - dividend=0.01 (larger than prev_close)
        - Expected: price unchanged, warning logged with exact format
        """
        # Create prices with prev_close = 0.0001 (row 14 is prev_close for Jan 15 ex-date)
        prices = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=20, freq="D"),
                "open": [0.0001] * 20,
                "high": [0.0002] * 20,
                "low": [0.00005] * 20,
                "close": [0.0001] * 20,  # All closes are 0.0001
                "volume": [1000000] * 20,
            }
        )

        # Ex-date is Jan 15 (index 14), so prev_close is Jan 14 (index 13) = 0.0001
        dividend = Dividend(ticker="TEST", ex_date=date(2023, 1, 15), amount=0.01)

        with caplog.at_level(logging.WARNING, logger="traderbot"):
            adjusted = adjust_for_dividend(prices, dividend)

        # Exact warning message must match spec
        expected_warning = (
            "CA: skip dividend adj (amt>=prev_close) ticker=TEST "
            "ex_date=2023-01-15 prev_close=0.0001000000 amt=0.0100000000"
        )
        assert expected_warning in caplog.text

        # Prices must be unchanged
        assert adjusted.loc[0, "close"] == 0.0001
        assert adjusted.loc[13, "close"] == 0.0001  # prev_close unchanged
        assert adjusted.loc[14, "close"] == 0.0001  # ex-date unchanged


class TestApplyCorporateActions:
    """Tests for applying multiple corporate actions."""

    def test_apply_split_and_dividend(self, sample_prices: pd.DataFrame) -> None:
        """Test applying both split and dividend."""
        splits = [Split(ticker="TEST", ex_date=date(2023, 1, 6), ratio=2.0)]
        dividends = [Dividend(ticker="TEST", ex_date=date(2023, 1, 8), amount=0.5)]

        adjusted = apply_corporate_actions(
            sample_prices, "TEST", splits=splits, dividends=dividends
        )

        # Split applied first (before Jan 6)
        # Then dividend applied (before Jan 8)
        assert adjusted is not None
        assert len(adjusted) == len(sample_prices)

    def test_filter_by_ticker(self, sample_prices: pd.DataFrame) -> None:
        """Test actions filtered by ticker."""
        splits = [
            Split(ticker="TEST", ex_date=date(2023, 1, 6), ratio=2.0),
            Split(ticker="OTHER", ex_date=date(2023, 1, 6), ratio=4.0),
        ]

        adjusted = apply_corporate_actions(sample_prices, "TEST", splits=splits)

        # Only TEST split should be applied
        assert adjusted.loc[0, "close"] == pytest.approx(100.5 / 2.0)
