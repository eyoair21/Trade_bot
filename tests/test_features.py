"""Tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from traderbot.features.gov_trades import (
    get_active_officials,
    get_gov_trade_signal,
    get_gov_trade_signals,
    get_gov_trades,
)
from traderbot.features.sentiment import (
    get_sentiment_score,
    get_sentiment_scores,
    get_sentiment_series,
)
from traderbot.features.ta import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)
from traderbot.features.volume import (
    calculate_accumulation_distribution,
    calculate_mfi,
    calculate_obv,
    calculate_volume_ratio,
    calculate_vwap,
)


class TestEMA:
    """Tests for EMA calculation."""

    def test_ema_basic(self) -> None:
        """Test basic EMA calculation."""
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = calculate_ema(prices, period=3)

        assert len(ema) == 5
        assert not ema.isna().all()

    def test_ema_smoothing(self) -> None:
        """Test EMA smooths data."""
        prices = pd.Series([10.0, 20.0, 10.0, 20.0, 10.0])
        ema = calculate_ema(prices, period=3)

        # EMA should be between min and max
        assert ema.min() >= 10.0
        assert ema.max() <= 20.0


class TestSMA:
    """Tests for SMA calculation."""

    def test_sma_basic(self) -> None:
        """Test basic SMA calculation."""
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = calculate_sma(prices, period=3)

        assert len(sma) == 5
        # SMA of [3,4,5] should be 4.0
        assert sma.iloc[-1] == pytest.approx(4.0)


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_range(self) -> None:
        """Test RSI is in valid range."""
        np.random.seed(42)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(100) * 0.02)))

        rsi = calculate_rsi(prices, period=14)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_uptrend(self) -> None:
        """Test RSI in uptrend is high."""
        prices = pd.Series([100.0 + i for i in range(30)])  # Consistent uptrend
        rsi = calculate_rsi(prices, period=14)

        # RSI should be high in uptrend
        assert rsi.iloc[-1] > 50

    def test_rsi_downtrend(self) -> None:
        """Test RSI in downtrend is low."""
        prices = pd.Series([100.0 - i for i in range(30)])  # Consistent downtrend
        rsi = calculate_rsi(prices, period=14)

        # RSI should be low in downtrend
        assert rsi.iloc[-1] < 50


class TestATR:
    """Tests for ATR calculation."""

    def test_atr_basic(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test basic ATR calculation."""
        atr = calculate_atr(
            sample_ohlcv_df["high"],
            sample_ohlcv_df["low"],
            sample_ohlcv_df["close"],
            period=14,
        )

        assert len(atr) == len(sample_ohlcv_df)
        assert (atr.dropna() > 0).all()  # ATR should be positive

    def test_atr_volatility_relation(self) -> None:
        """Test ATR increases with volatility."""
        # Low volatility data
        low_vol = pd.DataFrame(
            {
                "high": [100.1] * 30,
                "low": [99.9] * 30,
                "close": [100.0] * 30,
            }
        )

        # High volatility data
        high_vol = pd.DataFrame(
            {
                "high": [110.0] * 30,
                "low": [90.0] * 30,
                "close": [100.0] * 30,
            }
        )

        atr_low = calculate_atr(low_vol["high"], low_vol["low"], low_vol["close"])
        atr_high = calculate_atr(high_vol["high"], high_vol["low"], high_vol["close"])

        assert atr_high.iloc[-1] > atr_low.iloc[-1]


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_bands_structure(self) -> None:
        """Test Bollinger Bands returns correct structure."""
        prices = pd.Series([100.0 + np.sin(i * 0.1) for i in range(50)])

        upper, middle, lower = calculate_bollinger_bands(prices, period=20)

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

    def test_bollinger_bands_order(self) -> None:
        """Test upper > middle > lower."""
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(50))

        upper, middle, lower = calculate_bollinger_bands(prices, period=20)

        # Check order (where not NaN)
        valid = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()


class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_structure(self) -> None:
        """Test MACD returns correct structure."""
        np.random.seed(42)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(100) * 0.02)))

        macd_line, signal_line, histogram = calculate_macd(prices)

        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)


class TestVolumeIndicators:
    """Tests for volume-based indicators."""

    def test_vwap(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test VWAP calculation."""
        vwap = calculate_vwap(
            sample_ohlcv_df["high"],
            sample_ohlcv_df["low"],
            sample_ohlcv_df["close"],
            sample_ohlcv_df["volume"],
        )

        assert len(vwap) == len(sample_ohlcv_df)
        assert (vwap.dropna() > 0).all()

    def test_volume_ratio(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test volume ratio calculation."""
        ratio = calculate_volume_ratio(sample_ohlcv_df["volume"], period=5)

        assert len(ratio) == len(sample_ohlcv_df)
        # Ratio should be around 1.0 on average
        assert ratio.mean() == pytest.approx(1.0, rel=0.5)

    def test_obv(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test OBV calculation."""
        obv = calculate_obv(sample_ohlcv_df["close"], sample_ohlcv_df["volume"])

        assert len(obv) == len(sample_ohlcv_df)

    def test_ad_line(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test A/D line calculation."""
        ad = calculate_accumulation_distribution(
            sample_ohlcv_df["high"],
            sample_ohlcv_df["low"],
            sample_ohlcv_df["close"],
            sample_ohlcv_df["volume"],
        )

        assert len(ad) == len(sample_ohlcv_df)

    def test_mfi(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test MFI calculation."""
        mfi = calculate_mfi(
            sample_ohlcv_df["high"],
            sample_ohlcv_df["low"],
            sample_ohlcv_df["close"],
            sample_ohlcv_df["volume"],
        )

        assert len(mfi) == len(sample_ohlcv_df)
        # MFI should be between 0 and 100
        valid_mfi = mfi.dropna()
        assert (valid_mfi >= 0).all()
        assert (valid_mfi <= 100).all()


class TestSentimentStubs:
    """Tests for sentiment module stubs."""

    def test_get_sentiment_score(self) -> None:
        """Test sentiment score returns 0."""
        score = get_sentiment_score("AAPL")
        assert score == 0.0

    def test_get_sentiment_scores(self) -> None:
        """Test sentiment scores returns zeros."""
        scores = get_sentiment_scores(["AAPL", "MSFT", "NVDA"])
        assert all(s == 0.0 for s in scores.values())
        assert len(scores) == 3

    def test_get_sentiment_series(self) -> None:
        """Test sentiment series returns zeros."""
        series = get_sentiment_series(
            "AAPL",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-10"),
        )
        assert (series == 0.0).all()


class TestGovTradesStubs:
    """Tests for government trades module stubs."""

    def test_get_gov_trades(self) -> None:
        """Test gov trades returns empty list."""
        from datetime import date

        trades = get_gov_trades("AAPL", date(2023, 1, 1), date(2023, 12, 31))
        assert trades == []

    def test_get_gov_trade_signal(self) -> None:
        """Test gov trade signal returns 0."""
        from datetime import date

        signal = get_gov_trade_signal("AAPL", date(2023, 6, 15))
        assert signal == 0.0

    def test_get_gov_trade_signals(self) -> None:
        """Test gov trade signals returns zeros."""
        from datetime import date

        signals = get_gov_trade_signals(["AAPL", "MSFT"], date(2023, 6, 15))
        assert all(s == 0.0 for s in signals.values())

    def test_get_active_officials(self) -> None:
        """Test get active officials returns empty."""
        officials = get_active_officials()
        assert officials == []
