"""Position sizing for paper trading.

Implements volatility-targeted position sizing using ATR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from traderbot.features.ta import calculate_atr
from traderbot.logging_setup import get_logger

logger = get_logger("paper.position_sizing")


@dataclass
class PositionSizeResult:
    """Position sizing calculation result."""

    symbol: str
    shares: int
    notional: float
    target_vol: float
    estimated_vol: float
    position_weight: float


class PositionSizer:
    """Volatility-targeted position sizer.

    Sizes positions to achieve a target annualized volatility
    contribution per position.
    """

    def __init__(
        self,
        target_vol_per_position: float = 0.20,
        min_shares: int = 1,
        max_position_pct: float = 0.10,
        trading_days_per_year: int = 252,
    ):
        """Initialize position sizer.

        Args:
            target_vol_per_position: Target annualized volatility per position.
            min_shares: Minimum shares per position.
            max_position_pct: Maximum position as fraction of portfolio.
            trading_days_per_year: Trading days for annualization.
        """
        self.target_vol = target_vol_per_position
        self.min_shares = min_shares
        self.max_position_pct = max_position_pct
        self.trading_days = trading_days_per_year

    def compute_daily_vol(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> float:
        """Compute daily volatility from ATR.

        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            period: ATR lookback period.

        Returns:
            Daily volatility as fraction of price.
        """
        if len(close) < period + 1:
            return 0.0

        atr = calculate_atr(high, low, close, period=period)
        if atr.empty or atr.isna().all():
            return 0.0

        last_atr = float(atr.iloc[-1])
        last_close = float(close.iloc[-1])

        if last_close <= 0:
            return 0.0

        return last_atr / last_close

    def compute_annual_vol(self, daily_vol: float) -> float:
        """Annualize daily volatility."""
        return daily_vol * (self.trading_days ** 0.5)

    def compute_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> PositionSizeResult:
        """Compute position size for a symbol.

        Args:
            symbol: Stock symbol.
            price: Current price.
            portfolio_value: Total portfolio value.
            high: High price series.
            low: Low price series.
            close: Close price series.

        Returns:
            PositionSizeResult with computed size.
        """
        # Compute volatility
        daily_vol = self.compute_daily_vol(high, low, close)
        annual_vol = self.compute_annual_vol(daily_vol)

        if annual_vol <= 0:
            # Default to max position if vol can't be computed
            annual_vol = 0.30  # Conservative assumption

        # Target notional to achieve target volatility
        # position_vol = weight * asset_vol
        # target_vol = weight * asset_vol
        # weight = target_vol / asset_vol
        target_weight = self.target_vol / annual_vol

        # Apply max position constraint
        target_weight = min(target_weight, self.max_position_pct)

        # Compute notional and shares
        target_notional = portfolio_value * target_weight

        if price <= 0:
            shares = 0
        else:
            shares = max(self.min_shares, int(target_notional / price))

        # Recalculate actual notional
        notional = shares * price
        actual_weight = notional / portfolio_value if portfolio_value > 0 else 0.0

        return PositionSizeResult(
            symbol=symbol,
            shares=shares,
            notional=notional,
            target_vol=self.target_vol,
            estimated_vol=annual_vol,
            position_weight=actual_weight,
        )


def compute_position_size(
    symbol: str,
    price: float,
    portfolio_value: float,
    df: pd.DataFrame,
    target_vol: float = 0.20,
    max_position_pct: float = 0.10,
) -> PositionSizeResult:
    """Convenience function to compute position size.

    Args:
        symbol: Stock symbol.
        price: Current price.
        portfolio_value: Total portfolio value.
        df: OHLCV DataFrame with high, low, close columns.
        target_vol: Target annualized volatility.
        max_position_pct: Maximum position as fraction of portfolio.

    Returns:
        PositionSizeResult with computed size.
    """
    sizer = PositionSizer(
        target_vol_per_position=target_vol,
        max_position_pct=max_position_pct,
    )

    return sizer.compute_size(
        symbol=symbol,
        price=price,
        portfolio_value=portfolio_value,
        high=df["high"],
        low=df["low"],
        close=df["close"],
    )
