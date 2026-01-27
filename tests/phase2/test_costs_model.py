"""Tests for cost modeling."""

import pytest

from traderbot.engine.costs import CostModel, estimate_costs, estimate_costs_simple


def test_estimate_costs_components():
    """Test cost estimation returns all components."""
    costs = estimate_costs(
        quantity=1000,
        price=100.0,
        spread_bps=5.0,
        adv=1_000_000,
        volatility=0.20,
    )
    
    assert costs.slippage > 0
    assert costs.impact > 0
    assert costs.fees > 0
    assert costs.total == costs.slippage + costs.impact + costs.fees


def test_costs_scale_with_quantity():
    """Test that costs increase with order size."""
    costs_small = estimate_costs(quantity=100, price=100.0)
    costs_large = estimate_costs(quantity=10000, price=100.0)
    
    assert costs_large.total > costs_small.total


def test_impact_scales_with_participation():
    """Test market impact scales with ADV participation."""
    # Small participation
    costs_small_part = estimate_costs(
        quantity=1000,
        price=100.0,
        adv=10_000_000,  # 0.01% participation
    )
    
    # Large participation
    costs_large_part = estimate_costs(
        quantity=100000,
        price=100.0,
        adv=1_000_000,  # 10% participation
    )
    
    assert costs_large_part.impact > costs_small_part.impact


def test_estimate_costs_simple():
    """Test simple cost estimation."""
    cost = estimate_costs_simple(
        quantity=1000,
        price=100.0,
        cost_bps=10.0,
    )
    
    # Should be 10 bps of 100k = 100
    assert abs(cost - 100.0) < 0.01


def test_cost_model_tracking():
    """Test CostModel tracks history."""
    model = CostModel(spread_bps=5.0, fee_per_share=0.001)
    
    # Estimate costs for multiple orders
    model.estimate("AAPL", 1000, 150.0)
    model.estimate("MSFT", 500, 300.0)
    
    assert len(model.cost_history) == 2
    assert model.total_costs > 0


def test_cost_model_summary():
    """Test cost summary statistics."""
    model = CostModel()
    
    for i in range(10):
        model.estimate(f"TICKER_{i}", 1000, 100.0)
    
    summary = model.get_summary()
    
    assert summary["n_trades"] == 10
    assert summary["total_costs"] > 0
    assert "cost_breakdown" in summary
    assert "slippage" in summary["cost_breakdown"]

