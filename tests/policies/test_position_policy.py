"""Tests for position sizing policies."""

import pytest

from traderbot.policies.position import (
    fixed_fractional_size,
    kelly_clip_size,
    map_signal_to_size,
    target_vol_size,
)


def test_target_vol_size():
    """Test target volatility sizing."""
    # Signal = 1.0, vol = 0.20, target = 0.15
    # Expected size = 1.0 * (0.15 / 0.20) = 0.75
    size = target_vol_size(signal=1.0, realized_vol=0.20, target_vol=0.15)
    assert abs(size - 0.75) < 1e-6
    
    # Test leverage cap
    size = target_vol_size(signal=1.0, realized_vol=0.05, target_vol=0.15, max_leverage=1.0)
    assert size <= 1.0


def test_target_vol_size_edge_cases():
    """Test edge cases for target vol sizing."""
    # Zero volatility
    size = target_vol_size(signal=1.0, realized_vol=0.0, target_vol=0.15)
    assert size == 0.0
    
    # Negative signal
    size = target_vol_size(signal=-0.5, realized_vol=0.20, target_vol=0.15)
    assert size < 0


def test_kelly_clip_size():
    """Test Kelly criterion sizing."""
    # edge=0.02, variance=0.04, f* = 0.02/0.04 = 0.5, capped at 0.25
    size = kelly_clip_size(edge=0.02, variance=0.04, cap=0.25)
    assert abs(size - 0.25) < 1e-6
    
    # Lower edge
    size = kelly_clip_size(edge=0.01, variance=0.04, cap=0.25)
    assert abs(size - 0.25) < 1e-6


def test_kelly_clip_size_edge_cases():
    """Test Kelly edge cases."""
    # Zero variance
    size = kelly_clip_size(edge=0.02, variance=0.0, cap=0.25)
    assert size == 0.0
    
    # Negative edge (should clip to 0)
    size = kelly_clip_size(edge=-0.01, variance=0.04, cap=0.25)
    assert size == 0.0


def test_fixed_fractional_size():
    """Test fixed fractional sizing."""
    size = fixed_fractional_size(signal=1.0, fixed_frac=0.10)
    assert abs(size - 0.10) < 1e-6
    
    size = fixed_fractional_size(signal=-0.5, fixed_frac=0.10)
    assert abs(size - (-0.05)) < 1e-6


def test_map_signal_to_size_target_vol():
    """Test unified interface with target_vol mode."""
    # Probability 0.75 -> signal = (0.75 - 0.5) * 2 = 0.5
    size = map_signal_to_size(
        signal_prob=0.75,
        realized_vol=0.20,
        mode="target_vol",
        target_vol=0.15,
    )
    assert size > 0


def test_map_signal_to_size_kelly():
    """Test unified interface with kelly mode."""
    size = map_signal_to_size(
        signal_prob=0.6,
        realized_vol=0.20,
        mode="kelly",
        cap=0.25,
    )
    assert 0 <= size <= 0.25


def test_map_signal_to_size_fixed():
    """Test unified interface with fixed mode."""
    size = map_signal_to_size(
        signal_prob=0.7,
        mode="fixed",
        fixed_frac=0.10,
    )
    assert abs(abs(size) - 0.04) < 1e-6  # (0.7-0.5)*2 = 0.4, * 0.1 = 0.04


def test_map_signal_to_size_invalid_mode():
    """Test invalid mode raises error."""
    with pytest.raises(ValueError):
        map_signal_to_size(signal_prob=0.5, mode="invalid")

