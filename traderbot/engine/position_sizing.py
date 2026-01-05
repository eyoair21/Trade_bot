"""Position sizing policies for trading.

Provides different sizing algorithms:
- fixed_fraction: Fixed percentage of equity per position
- vol_target: Volatility-targeted sizing
- kelly_fraction: Kelly criterion with cap
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from traderbot.config import get_config
from traderbot.logging_setup import get_logger

logger = get_logger("engine.position_sizing")


class SizerType(Enum):
    """Position sizer types."""

    FIXED = "fixed"
    VOL = "vol"
    KELLY = "kelly"


@dataclass
class SizeResult:
    """Result of position sizing calculation."""

    shares: int
    notional: float
    fraction: float
    reason: str = ""


def fixed_fraction(
    equity: float,
    price: float,
    fraction: float | None = None,
) -> SizeResult:
    """Calculate position size as fixed fraction of equity.

    Args:
        equity: Current portfolio equity.
        price: Current price of the asset.
        fraction: Fraction of equity to allocate. Defaults to config.

    Returns:
        SizeResult with calculated shares.
    """
    config = get_config()
    frac = fraction if fraction is not None else config.sizing.fixed_frac

    if equity <= 0 or price <= 0:
        return SizeResult(shares=0, notional=0.0, fraction=0.0, reason="Invalid equity or price")

    notional = equity * frac
    shares = int(notional / price)

    return SizeResult(
        shares=shares,
        notional=shares * price,
        fraction=frac,
        reason=f"Fixed fraction {frac:.2%}",
    )


def vol_target(
    equity: float,
    price: float,
    volatility: float,
    target_vol: float | None = None,
) -> SizeResult:
    """Calculate position size to target specific volatility.

    Sizes position so that position's volatility contribution
    matches target annualized volatility.

    Args:
        equity: Current portfolio equity.
        price: Current price of the asset.
        volatility: Annualized volatility of the asset (e.g., 0.25 for 25%).
        target_vol: Target annualized volatility. Defaults to config.

    Returns:
        SizeResult with calculated shares.
    """
    config = get_config()
    target = target_vol if target_vol is not None else config.sizing.vol_target

    if equity <= 0 or price <= 0:
        return SizeResult(shares=0, notional=0.0, fraction=0.0, reason="Invalid equity or price")

    if volatility <= 0:
        return SizeResult(shares=0, notional=0.0, fraction=0.0, reason="Invalid volatility")

    # Position fraction = target_vol / asset_vol
    # This makes position's volatility contribution = target_vol
    fraction = target / volatility

    # Cap at 100% of equity
    fraction = min(fraction, 1.0)

    notional = equity * fraction
    shares = int(notional / price)

    return SizeResult(
        shares=shares,
        notional=shares * price,
        fraction=fraction,
        reason=f"Vol target {target:.2%}, asset vol {volatility:.2%}",
    )


def kelly_fraction(
    equity: float,
    price: float,
    win_prob: float,
    win_loss_ratio: float,
    cap: float | None = None,
) -> SizeResult:
    """Calculate position size using Kelly criterion.

    Kelly formula: f* = (p * b - q) / b
    where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = win/loss ratio (average win / average loss)

    Args:
        equity: Current portfolio equity.
        price: Current price of the asset.
        win_prob: Probability of winning trade (0 to 1).
        win_loss_ratio: Average win / average loss ratio.
        cap: Maximum Kelly fraction. Defaults to config.

    Returns:
        SizeResult with calculated shares.
    """
    config = get_config()
    kelly_cap = cap if cap is not None else config.sizing.kelly_cap

    if equity <= 0 or price <= 0:
        return SizeResult(shares=0, notional=0.0, fraction=0.0, reason="Invalid equity or price")

    if win_prob <= 0 or win_prob >= 1:
        return SizeResult(shares=0, notional=0.0, fraction=0.0, reason="Invalid win probability")

    if win_loss_ratio <= 0:
        return SizeResult(shares=0, notional=0.0, fraction=0.0, reason="Invalid win/loss ratio")

    # Kelly formula
    p = win_prob
    q = 1 - p
    b = win_loss_ratio

    kelly = (p * b - q) / b

    # Kelly can be negative (don't bet)
    if kelly <= 0:
        return SizeResult(shares=0, notional=0.0, fraction=0.0, reason="Negative Kelly, no trade")

    # Apply cap (half-Kelly is common practice)
    fraction = min(kelly, kelly_cap)

    notional = equity * fraction
    shares = int(notional / price)

    return SizeResult(
        shares=shares,
        notional=shares * price,
        fraction=fraction,
        reason=f"Kelly {kelly:.2%}, capped at {kelly_cap:.2%}",
    )


def calculate_volatility(
    prices: pd.Series,
    window: int = 20,
    annualize: bool = True,
) -> float:
    """Calculate historical volatility from price series.

    Args:
        prices: Price series (typically close prices).
        window: Lookback window for volatility calculation.
        annualize: Whether to annualize (assuming 252 trading days).

    Returns:
        Volatility as decimal (e.g., 0.25 for 25%).
    """
    if len(prices) < window + 1:
        return 0.0

    returns = prices.pct_change().dropna()

    if len(returns) < window:
        return 0.0

    vol = returns.tail(window).std()

    if annualize:
        vol = vol * np.sqrt(252)

    return float(vol)


class PositionSizer:
    """Position sizer that applies configured sizing policy."""

    def __init__(
        self,
        sizer_type: str | SizerType | None = None,
        fixed_frac: float | None = None,
        vol_target: float | None = None,
        kelly_cap: float | None = None,
    ):
        """Initialize position sizer.

        Args:
            sizer_type: Sizing method ("fixed", "vol", "kelly"). Defaults to config.
            fixed_frac: Fixed fraction for fixed sizer. Defaults to config.
            vol_target: Target volatility for vol sizer. Defaults to config.
            kelly_cap: Kelly cap for kelly sizer. Defaults to config.
        """
        config = get_config()

        if sizer_type is None:
            sizer_type = config.sizing.sizer
        if isinstance(sizer_type, str):
            sizer_type = SizerType(sizer_type)

        self.sizer_type = sizer_type
        self.fixed_frac = fixed_frac if fixed_frac is not None else config.sizing.fixed_frac
        self.vol_target = vol_target if vol_target is not None else config.sizing.vol_target
        self.kelly_cap = kelly_cap if kelly_cap is not None else config.sizing.kelly_cap

    def calculate_size(
        self,
        equity: float,
        price: float,
        volatility: float = 0.0,
        win_prob: float = 0.5,
        win_loss_ratio: float = 1.0,
    ) -> SizeResult:
        """Calculate position size based on configured sizer.

        Args:
            equity: Current portfolio equity.
            price: Current price of the asset.
            volatility: Annualized volatility (for vol sizer).
            win_prob: Win probability (for kelly sizer).
            win_loss_ratio: Win/loss ratio (for kelly sizer).

        Returns:
            SizeResult with calculated shares.
        """
        if self.sizer_type == SizerType.FIXED:
            return fixed_fraction(equity, price, self.fixed_frac)

        elif self.sizer_type == SizerType.VOL:
            if volatility <= 0:
                logger.warning("Vol sizer requested but no volatility provided, using fixed")
                return fixed_fraction(equity, price, self.fixed_frac)
            return vol_target(equity, price, volatility, self.vol_target)

        elif self.sizer_type == SizerType.KELLY:
            return kelly_fraction(equity, price, win_prob, win_loss_ratio, self.kelly_cap)

        else:
            logger.warning(f"Unknown sizer type: {self.sizer_type}, using fixed")
            return fixed_fraction(equity, price, self.fixed_frac)
