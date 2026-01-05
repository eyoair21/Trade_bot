"""Volume-based features and indicators.

Pure functions for computing volume-related signals.
"""

import numpy as np
import pandas as pd


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate Volume-Weighted Average Price (VWAP).

    Note: This is a cumulative VWAP from the start of the data.
    For intraday VWAP, data should be filtered to the session first.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.

    Returns:
        VWAP series.
    """
    # Typical price
    typical_price = (high + low + close) / 3.0

    # Cumulative calculations
    cumulative_tp_volume = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum()

    # VWAP
    vwap = cumulative_tp_volume / cumulative_volume.replace(0, np.nan)

    return vwap


def calculate_volume_ratio(
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Calculate volume ratio relative to moving average.

    Values > 1 indicate above-average volume.

    Args:
        volume: Volume series.
        period: Moving average period.

    Returns:
        Volume ratio series.
    """
    avg_volume = volume.rolling(window=period, min_periods=1).mean()
    ratio = volume / avg_volume.replace(0, np.nan)

    return ratio


def calculate_obv(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate On-Balance Volume (OBV).

    Args:
        close: Close price series.
        volume: Volume series.

    Returns:
        OBV series.
    """
    # Direction: +1 if close > prev, -1 if close < prev, 0 if equal
    direction = pd.Series(np.sign(close.diff()), index=close.index)
    direction.iloc[0] = 0

    # OBV is cumulative sum of signed volume
    obv: pd.Series[float] = (direction * volume).cumsum()

    return obv


def calculate_accumulation_distribution(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate Accumulation/Distribution Line.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.

    Returns:
        A/D Line series.
    """
    # Money Flow Multiplier
    hl_range = high - low
    clv = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)

    # Money Flow Volume
    mfv = clv * volume

    # A/D Line is cumulative sum
    ad_line = mfv.cumsum()

    return ad_line


def calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Money Flow Index (MFI).

    MFI is a volume-weighted RSI.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        period: MFI period.

    Returns:
        MFI series (values 0-100).
    """
    # Typical price
    typical_price = (high + low + close) / 3.0

    # Raw money flow
    raw_money_flow = typical_price * volume

    # Direction
    tp_diff = typical_price.diff()

    # Positive and negative money flow
    positive_flow = raw_money_flow.where(tp_diff > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_diff < 0, 0.0)

    # Sum over period
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()

    # Money Flow Ratio and MFI
    mfr = positive_sum / negative_sum.replace(0, np.inf)
    mfi = 100.0 - (100.0 / (1.0 + mfr))

    return mfi
