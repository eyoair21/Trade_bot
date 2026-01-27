"""Tests for portfolio limits and management."""

import pytest

from traderbot.engine.portfolio import PortfolioLimits, PortfolioManager, VolatilityEstimator


def test_portfolio_limits_creation():
    """Test creating portfolio limits."""
    limits = PortfolioLimits(
        max_gross_exposure=1.0,
        max_net_exposure=0.5,
        max_position_pct=0.10,
    )
    
    assert limits.max_gross_exposure == 1.0
    assert limits.max_position_pct == 0.10


def test_position_limit_enforced():
    """Test per-position limit is enforced."""
    limits = PortfolioLimits(max_position_pct=0.10)
    mgr = PortfolioManager(limits, capital=100000)
    
    # Propose order exceeding limit
    proposed = {"AAPL": 20000}  # 20% of capital
    prices = {"AAPL": 150.0}
    
    clipped = mgr.clip_orders(proposed, prices)
    
    # Should be clipped to 10%
    assert abs(clipped["AAPL"]) <= 10000


def test_gross_exposure_limit():
    """Test gross exposure limit is enforced."""
    limits = PortfolioLimits(max_gross_exposure=1.0)
    mgr = PortfolioManager(limits, capital=100000)
    
    # Propose orders exceeding gross limit
    proposed = {
        "AAPL": 50000,
        "MSFT": 50000,
        "NVDA": 50000,
    }
    prices = {"AAPL": 150, "MSFT": 300, "NVDA": 500}
    
    clipped = mgr.clip_orders(proposed, prices)
    
    # Gross should be <= 100000
    gross = sum(abs(v) for v in clipped.values())
    assert gross <= 100000


def test_net_exposure_limit():
    """Test net exposure limit is enforced."""
    limits = PortfolioLimits(max_net_exposure=0.5)
    mgr = PortfolioManager(limits, capital=100000)
    
    # All long positions
    proposed = {
        "AAPL": 40000,
        "MSFT": 40000,
    }
    prices = {"AAPL": 150, "MSFT": 300}
    
    clipped = mgr.clip_orders(proposed, prices)
    
    # Net should be <= 50000
    net = sum(clipped.values())
    assert abs(net) <= 50000


def test_volatility_estimator():
    """Test EWMA volatility estimation."""
    estimator = VolatilityEstimator(halflife=20)
    
    # Update with returns
    for i in range(50):
        returns = {"AAPL": 0.01 * ((-1) ** i)}
        estimator.update(returns)
    
    vol = estimator.get_volatility("AAPL")
    
    # Should have reasonable estimate
    assert 0.0 < vol < 1.0


def test_correlation_computation():
    """Test correlation matrix computation."""
    estimator = VolatilityEstimator()
    
    # Update with correlated returns
    for i in range(30):
        base_return = 0.01 * ((-1) ** i)
        returns = {
            "AAPL": base_return,
            "MSFT": base_return * 0.8,  # Correlated
            "SPY": base_return * 0.5,
        }
        estimator.update(returns)
    
    corr_matrix = estimator.compute_correlation_matrix()
    
    assert len(corr_matrix) == 3
    # AAPL-MSFT should be positively correlated
    assert corr_matrix.loc["AAPL", "MSFT"] > 0.5

