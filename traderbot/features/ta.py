"""Technical analysis indicators.

Pure functions for computing TA indicators.
"""

import numpy as np
import pandas as pd


def calculate_ema(
    prices: pd.Series,
    period: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        prices: Price series (typically close prices).
        period: EMA period.
        min_periods: Minimum periods required. Defaults to period.

    Returns:
        EMA series.
    """
    if min_periods is None:
        min_periods = period

    return prices.ewm(span=period, min_periods=min_periods, adjust=False).mean()


def calculate_sma(
    prices: pd.Series,
    period: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        prices: Price series.
        period: SMA period.
        min_periods: Minimum periods required. Defaults to period.

    Returns:
        SMA series.
    """
    if min_periods is None:
        min_periods = period

    return prices.rolling(window=period, min_periods=min_periods).mean()


def calculate_rsi(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Relative Strength Index.

    Uses the Wilder smoothing method (exponential).

    Args:
        prices: Price series (typically close prices).
        period: RSI period.

    Returns:
        RSI series (values 0-100).
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Calculate average gains and losses using Wilder smoothing
    alpha = 1.0 / period
    avg_gains = gains.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    # Handle the case where avg_losses is 0 (pure uptrend) -> RSI = 100
    # Handle the case where avg_gains is 0 (pure downtrend) -> RSI = 0
    rsi = pd.Series(index=prices.index, dtype=float)

    for i in range(len(prices)):
        if pd.isna(avg_gains.iloc[i]) or pd.isna(avg_losses.iloc[i]):
            rsi.iloc[i] = np.nan
        elif avg_losses.iloc[i] == 0:
            # No losses -> RSI = 100 (pure uptrend)
            rsi.iloc[i] = 100.0 if avg_gains.iloc[i] > 0 else 50.0
        elif avg_gains.iloc[i] == 0:
            # No gains -> RSI = 0 (pure downtrend)
            rsi.iloc[i] = 0.0
        else:
            rs = avg_gains.iloc[i] / avg_losses.iloc[i]
            rsi.iloc[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR period.

    Returns:
        ATR series.
    """
    # Calculate True Range components
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # True Range is the max of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is the smoothed average (Wilder smoothing)
    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return atr


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Args:
        prices: Price series.
        period: Moving average period.
        num_std: Number of standard deviations.

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    middle = calculate_sma(prices, period)
    std = prices.rolling(window=period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Price series.
        fast_period: Fast EMA period.
        slow_period: Slow EMA period.
        signal_period: Signal line period.

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period, min_periods=1)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def pct_change1(close: pd.Series, clip_pct: float = 10.0) -> pd.Series:
    """Calculate 1-period percentage return, clipped.

    Args:
        close: Close price series.
        clip_pct: Maximum absolute percentage (clips to ±clip_pct).

    Returns:
        Clipped percentage return series (close_ret_1).
    """
    ret = close.pct_change() * 100.0  # Convert to percentage
    return ret.clip(lower=-clip_pct, upper=clip_pct)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI with specified window.

    Wrapper around calculate_rsi for feature pipeline compatibility.

    Args:
        series: Price series.
        window: RSI period.

    Returns:
        RSI series (rsi_14 when window=14).
    """
    return calculate_rsi(series, period=window)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Calculate ATR with specified window.

    Wrapper around calculate_atr for feature pipeline compatibility.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: ATR period.

    Returns:
        ATR series (atr_14 when window=14).
    """
    return calculate_atr(high, low, close, period=window)


def vwap_gap(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """Calculate gap between close and VWAP as percentage.

    Args:
        close: Close price series.
        vwap: VWAP series.

    Returns:
        Percentage gap series (vwap_gap).
    """
    gap = (close - vwap) / vwap * 100.0
    return gap.replace([np.inf, -np.inf], np.nan)


def dvol(volume: pd.Series, window: int = 5) -> pd.Series:
    """Calculate dollar volume ratio vs rolling mean.

    Args:
        volume: Volume series.
        window: Rolling window for mean.

    Returns:
        Volume ratio minus 1 (dvol_5 when window=5).
    """
    rolling_mean = volume.rolling(window=window, min_periods=1).mean()
    ratio = volume / rolling_mean.replace(0, np.nan) - 1.0
    return ratio


def regime_vix(
    close: pd.Series,
    lookback: int = 20,
    high_vol_threshold: float = 0.25,
) -> pd.Series:
    """Calculate volatility regime indicator.

    Uses realized volatility as a proxy for VIX-like regime indicator.

    Args:
        close: Close price series.
        lookback: Rolling window for volatility calculation.
        high_vol_threshold: Annualized vol threshold for high regime.

    Returns:
        Regime indicator series (regime_vix):
        - 1.0: High volatility regime
        - 0.0: Low volatility regime
    """
    # Calculate realized volatility (annualized)
    returns = close.pct_change()
    realized_vol = returns.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(252)

    # Binary regime indicator
    regime: pd.Series = (realized_vol > high_vol_threshold).astype(float)
    return regime


def compute_model_features(
    df: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, pd.Series]:
    """Compute all features needed for model input.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume.
        feature_names: List of feature names to compute.

    Returns:
        Dict mapping feature name to computed series.
    """
    features: dict[str, pd.Series] = {}

    # Pre-compute VWAP if needed
    vwap_series = None
    if "vwap_gap" in feature_names:
        from traderbot.features.volume import calculate_vwap

        vwap_series = calculate_vwap(df["high"], df["low"], df["close"], df["volume"])

    for name in feature_names:
        if name == "close_ret_1":
            features[name] = pct_change1(df["close"])
        elif name == "rsi_14":
            features[name] = rsi(df["close"], window=14)
        elif name == "atr_14":
            features[name] = atr(df["high"], df["low"], df["close"], window=14)
        elif name == "vwap_gap":
            if vwap_series is not None:
                features[name] = vwap_gap(df["close"], vwap_series)
        elif name == "dvol_5":
            features[name] = dvol(df["volume"], window=5)
        elif name == "regime_vix":
            features[name] = regime_vix(df["close"])
        elif name.startswith("rsi_"):
            window = int(name.split("_")[1])
            features[name] = rsi(df["close"], window=window)
        elif name.startswith("atr_"):
            window = int(name.split("_")[1])
            features[name] = atr(df["high"], df["low"], df["close"], window=window)
        elif name.startswith("dvol_"):
            window = int(name.split("_")[1])
            features[name] = dvol(df["volume"], window=window)

    return features
