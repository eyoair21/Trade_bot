"""Corporate actions handling for price adjustments.

Handles stock splits and dividend adjustments with proper guards.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from traderbot.logging_setup import get_logger

logger = get_logger("data.corporate_actions")

# Epsilon for floating point comparisons
EPS = 1e-6


@dataclass
class Split:
    """Stock split corporate action."""

    ticker: str
    ex_date: date
    ratio: float  # e.g., 4.0 for 4:1 split (each share becomes 4)


@dataclass
class Dividend:
    """Dividend corporate action."""

    ticker: str
    ex_date: date
    amount: float  # Dividend amount per share


def adjust_for_split(
    prices: pd.DataFrame,
    split: Split,
) -> pd.DataFrame:
    """Adjust prices for a stock split (inverse adjustment).

    Prices before the ex-date are divided by the split ratio to make
    them comparable to post-split prices.

    Args:
        prices: DataFrame with 'date', 'open', 'high', 'low', 'close', 'volume' columns.
        split: Split corporate action.

    Returns:
        Adjusted prices DataFrame.
    """
    if split.ratio <= 0:
        logger.warning(f"Invalid split ratio {split.ratio} for {split.ticker}")
        return prices

    df = prices.copy()

    # Ensure date column is datetime
    if "date" in df.columns:
        date_col: pd.Series[date] = pd.to_datetime(df["date"]).dt.date  # type: ignore[assignment]
    else:
        date_col = pd.Series(df.index.date if hasattr(df.index, "date") else df.index)

    # Find rows before ex-date
    mask = date_col < split.ex_date

    # Adjust prices (inverse ratio - divide prices before split)
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            df.loc[mask, col] = df.loc[mask, col] / split.ratio

    # Adjust volume (multiply by ratio)
    if "volume" in df.columns:
        df.loc[mask, "volume"] = df.loc[mask, "volume"] * split.ratio

    logger.info(
        f"Applied split adjustment for {split.ticker}: "
        f"ratio={split.ratio}, ex_date={split.ex_date}"
    )

    return df


def adjust_for_dividend(
    prices: pd.DataFrame,
    dividend: Dividend,
) -> pd.DataFrame:
    """Adjust prices for a dividend (prev-close anchor method).

    Prices before the ex-date are adjusted downward by the dividend
    factor to make them comparable to post-dividend prices.

    Factor = (prev_close - dividend) / prev_close

    Args:
        prices: DataFrame with 'date', 'open', 'high', 'low', 'close', 'volume' columns.
        dividend: Dividend corporate action.

    Returns:
        Adjusted prices DataFrame.
    """
    df = prices.copy()

    # Ensure date column handling
    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"])
        date_col: pd.Series[date] = date_series.dt.date  # type: ignore[assignment]
    else:
        date_col = pd.Series(df.index.date if hasattr(df.index, "date") else df.index)

    # Find the row just before ex-date to get prev_close
    pre_ex_mask = date_col < dividend.ex_date
    if not pre_ex_mask.any():
        logger.warning(f"No data before ex-date {dividend.ex_date} for {dividend.ticker}")
        return df

    # Get prev_close (last close before ex-date)
    pre_ex_df = df[pre_ex_mask]
    prev_close = pre_ex_df["close"].iloc[-1]

    # Guard: prev_close too small
    if prev_close <= EPS:
        logger.warning(
            f"CA: skip dividend adj (prev_close<=EPS) ticker={dividend.ticker} "
            f"ex_date={dividend.ex_date} prev_close={prev_close:.10f} "
            f"amt={dividend.amount:.10f}"
        )
        return df

    # Guard: dividend >= prev_close
    if dividend.amount >= prev_close:
        logger.warning(
            f"CA: skip dividend adj (amt>=prev_close) ticker={dividend.ticker} "
            f"ex_date={dividend.ex_date} prev_close={prev_close:.10f} "
            f"amt={dividend.amount:.10f}"
        )
        return df

    # Calculate adjustment factor
    factor = (prev_close - dividend.amount) / prev_close

    # Adjust prices before ex-date
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            df.loc[pre_ex_mask, col] = df.loc[pre_ex_mask, col] * factor

    logger.info(
        f"Applied dividend adjustment for {dividend.ticker}: "
        f"amount={dividend.amount}, factor={factor:.6f}, ex_date={dividend.ex_date}"
    )

    return df


def apply_corporate_actions(
    prices: pd.DataFrame,
    ticker: str,
    splits: list[Split] | None = None,
    dividends: list[Dividend] | None = None,
) -> pd.DataFrame:
    """Apply all corporate actions to price data.

    Actions are applied in chronological order.

    Args:
        prices: Raw OHLCV DataFrame.
        ticker: Ticker symbol for filtering actions.
        splits: List of splits to apply.
        dividends: List of dividends to apply.

    Returns:
        Adjusted prices DataFrame.
    """
    df = prices.copy()

    # Filter to this ticker's actions
    ticker_splits = [s for s in (splits or []) if s.ticker == ticker]
    ticker_dividends = [d for d in (dividends or []) if d.ticker == ticker]

    # Sort by ex_date
    ticker_splits.sort(key=lambda x: x.ex_date)
    ticker_dividends.sort(key=lambda x: x.ex_date)

    # Apply splits
    for split in ticker_splits:
        df = adjust_for_split(df, split)

    # Apply dividends
    for dividend in ticker_dividends:
        df = adjust_for_dividend(df, dividend)

    return df
