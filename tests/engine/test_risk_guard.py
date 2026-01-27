"""Tests for risk guards and kill-switch."""

from datetime import datetime

import pytest

from traderbot.engine.risk_guard import RiskGuard, RiskGuardConfig


def test_risk_guard_no_breach():
    """Test normal operation with no breaches."""
    config = RiskGuardConfig(
        max_drawdown_pct=0.15,
        daily_loss_limit_pct=0.02,
    )
    guard = RiskGuard(config, initial_capital=100000.0)
    
    # Small profit
    allow, actions = guard.check_guards(
        current_equity=101000.0,
        positions={"AAPL": 10000},
        timestamp=datetime(2023, 1, 1, 10, 0),
    )
    
    assert allow is True
    assert len(actions) == 0
    assert len(guard.breaches) == 0


def test_risk_guard_drawdown_breach():
    """Test drawdown breach triggers kill-switch."""
    config = RiskGuardConfig(max_drawdown_pct=0.10)
    guard = RiskGuard(config, initial_capital=100000.0)
    
    # Set peak
    guard.check_guards(
        current_equity=105000.0,
        positions={},
        timestamp=datetime(2023, 1, 1),
    )
    
    # Breach with 11% drawdown
    allow, actions = guard.check_guards(
        current_equity=93000.0,  # 12% down from 105k
        positions={},
        timestamp=datetime(2023, 1, 2),
    )
    
    assert allow is False  # Kill-switch activated
    assert "flatten" in actions
    assert len(guard.breaches) == 1
    assert guard.breaches[0].breach_type == "drawdown"


def test_risk_guard_daily_loss_breach():
    """Test daily loss limit breach."""
    config = RiskGuardConfig(daily_loss_limit_pct=0.02)
    guard = RiskGuard(config, initial_capital=100000.0)
    
    # Start of day
    guard.check_guards(
        current_equity=100000.0,
        positions={},
        timestamp=datetime(2023, 1, 1, 9, 30),
        is_new_day=True,
    )
    
    # Breach with 3% loss
    allow, actions = guard.check_guards(
        current_equity=97000.0,
        positions={},
        timestamp=datetime(2023, 1, 1, 15, 0),
    )
    
    assert allow is False
    assert "flatten" in actions
    assert len(guard.breaches) == 1
    assert guard.breaches[0].breach_type == "daily_loss"


def test_risk_guard_position_size_breach():
    """Test position size limit breach."""
    config = RiskGuardConfig(max_position_pct=0.10)
    guard = RiskGuard(config, initial_capital=100000.0)
    
    # Oversized position (15% of capital)
    allow, actions = guard.check_guards(
        current_equity=100000.0,
        positions={"AAPL": 15000},
        timestamp=datetime(2023, 1, 1),
    )
    
    assert "reduce_AAPL" in actions
    assert len(guard.breaches) == 1
    assert guard.breaches[0].breach_type == "position_size"


def test_risk_guard_cooldown():
    """Test kill-switch cooldown period."""
    config = RiskGuardConfig(
        max_drawdown_pct=0.10,
        loss_cooldown_bars=3,
    )
    guard = RiskGuard(config, initial_capital=100000.0)
    
    # Trigger kill-switch
    guard.check_guards(
        current_equity=105000.0,
        positions={},
        timestamp=datetime(2023, 1, 1),
    )
    guard.check_guards(
        current_equity=93000.0,
        positions={},
        timestamp=datetime(2023, 1, 2),
    )
    
    assert guard.kill_switch_active is True
    assert guard.cooldown_remaining == 3
    
    # Step through cooldown
    for i in range(3):
        allow, _ = guard.check_guards(
            current_equity=95000.0,
            positions={},
            timestamp=datetime(2023, 1, 3 + i),
        )
        assert allow is False  # Still in cooldown
    
    # After cooldown
    allow, _ = guard.check_guards(
        current_equity=95000.0,
        positions={},
        timestamp=datetime(2023, 1, 6),
    )
    assert allow is True  # Cooldown complete


def test_risk_guard_summary():
    """Test summary statistics."""
    config = RiskGuardConfig()
    guard = RiskGuard(config, initial_capital=100000.0)
    
    guard.check_guards(
        current_equity=105000.0,
        positions={},
        timestamp=datetime(2023, 1, 1),
    )
    
    summary = guard.get_summary()
    
    assert "peak_equity" in summary
    assert "current_equity" in summary
    assert "total_breaches" in summary
    assert summary["peak_equity"] == 105000.0

