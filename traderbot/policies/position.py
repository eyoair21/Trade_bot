"""Position sizing policies for signal-to-size mapping.

Implements multiple sizing strategies:
- Target volatility sizing
- Kelly criterion with clipping
- Fixed fractional sizing
"""

import numpy as np


def target_vol_size(
    signal: float,
    realized_vol: float,
    target_vol: float = 0.15,
    max_leverage: float = 1.0,
) -> float:
    """Scale signal to hit target volatility, capped by max leverage.
    
    Args:
        signal: Raw signal (-1 to 1, where sign is direction)
        realized_vol: Recent realized volatility (e.g., 20-day)
        target_vol: Target portfolio volatility (e.g., 0.15 = 15%)
        max_leverage: Maximum leverage cap (1.0 = 100% capital)
    
    Returns:
        Position size as fraction of capital (-max_leverage to +max_leverage)
    """
    if realized_vol <= 0 or np.isnan(realized_vol):
        return 0.0
    
    # Scale to target vol
    raw_size = signal * (target_vol / realized_vol)
    
    # Clip to max leverage
    return float(np.clip(raw_size, -max_leverage, max_leverage))


def kelly_clip_size(
    edge: float,
    variance: float,
    cap: float = 0.25,
) -> float:
    """Kelly criterion sizing with conservative clipping.
    
    Args:
        edge: Expected return (mean of distribution)
        variance: Variance of return distribution
        cap: Maximum fraction to risk (Kelly is often too aggressive)
    
    Returns:
        Position size as fraction of capital (0 to cap)
    """
    if variance <= 0 or np.isnan(variance):
        return 0.0
    
    # Kelly formula: f* = edge / variance
    kelly_frac = edge / variance
    
    # Clip to [0, cap] for safety
    return float(np.clip(kelly_frac, 0.0, cap))


def fixed_fractional_size(
    signal: float,
    fixed_frac: float = 0.10,
) -> float:
    """Simple fixed fractional sizing.
    
    Args:
        signal: Raw signal (-1 to 1)
        fixed_frac: Fixed fraction of capital per position
    
    Returns:
        Position size as fraction of capital
    """
    return float(signal * fixed_frac)


def map_signal_to_size(
    signal_prob: float,
    realized_vol: float = 0.20,
    mode: str = "target_vol",
    **kwargs: float,
) -> float:
    """Unified interface for signal-to-size mapping.
    
    Args:
        signal_prob: Signal probability or score (typically 0-1)
        realized_vol: Recent realized volatility
        mode: Sizing mode ('target_vol', 'kelly', 'fixed')
        **kwargs: Mode-specific parameters
            - target_vol, max_leverage (for target_vol mode)
            - edge, variance, cap (for kelly mode)
            - fixed_frac (for fixed mode)
    
    Returns:
        Position size as fraction of capital
    """
    # Convert probability to directional signal (-1 to 1)
    # Assume 0.5 = neutral, >0.5 = long, <0.5 = short
    signal = (signal_prob - 0.5) * 2.0
    
    if mode == "target_vol":
        return target_vol_size(
            signal=signal,
            realized_vol=realized_vol,
            target_vol=kwargs.get("target_vol", 0.15),
            max_leverage=kwargs.get("max_leverage", 1.0),
        )
    elif mode == "kelly":
        # For Kelly, need edge and variance
        edge = kwargs.get("edge", signal * 0.01)  # Assume 1% edge per unit signal
        variance = kwargs.get("variance", realized_vol ** 2)
        cap = kwargs.get("cap", 0.25)
        return kelly_clip_size(edge=edge, variance=variance, cap=cap)
    elif mode == "fixed":
        return fixed_fractional_size(
            signal=signal,
            fixed_frac=kwargs.get("fixed_frac", 0.10),
        )
    else:
        raise ValueError(f"Unknown sizing mode: {mode}")

