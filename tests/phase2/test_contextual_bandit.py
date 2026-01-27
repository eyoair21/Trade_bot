"""Tests for contextual bandit allocation."""

import pytest

from traderbot.alloc.bandit import ContextualThompsonSampling


def test_contextual_bandit_init():
    """Test contextual bandit initialization."""
    strategies = ["momentum", "mean_reversion", "patchtst"]
    allocator = ContextualThompsonSampling(strategies)
    
    assert len(allocator.strategy_names) == 3
    assert allocator.n_context_buckets == 3


def test_context_discretization():
    """Test that context is discretized correctly."""
    allocator = ContextualThompsonSampling(["strategy_a"])
    
    context1 = {"regime_id": 0, "realized_vol": 0.10}
    context2 = {"regime_id": 0, "realized_vol": 0.10}
    context3 = {"regime_id": 1, "realized_vol": 0.20}
    
    hash1 = allocator._discretize_context(context1)
    hash2 = allocator._discretize_context(context2)
    hash3 = allocator._discretize_context(context3)
    
    # Same context should hash to same bucket
    assert hash1 == hash2
    # Different context should (likely) hash differently
    assert hash1 != hash3


def test_contextual_selection():
    """Test arm selection given context."""
    strategies = ["strategy_a", "strategy_b"]
    allocator = ContextualThompsonSampling(strategies)
    
    context = {"regime_id": 0, "realized_vol": 0.15}
    
    selected = allocator.select_arms(context, k=1)
    
    assert len(selected) == 1
    assert selected[0] in strategies


def test_contextual_weights():
    """Test weight computation given context."""
    strategies = ["strategy_a", "strategy_b", "strategy_c"]
    allocator = ContextualThompsonSampling(strategies)
    
    context = {"regime_id": 1, "realized_vol": 0.20}
    weights = allocator.get_weights(context)
    
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_contextual_adaptation():
    """Test that allocator adapts to context."""
    strategies = ["strategy_a", "strategy_b"]
    allocator = ContextualThompsonSampling(strategies)
    
    # Context 0: strategy_a performs well
    context0 = {"regime_id": 0, "realized_vol": 0.10}
    for _ in range(20):
        allocator.update("strategy_a", reward=1.0, context=context0)
        allocator.update("strategy_b", reward=-0.5, context=context0)
    
    weights0 = allocator.get_weights(context0)
    
    # Context 1: strategy_b performs well
    context1 = {"regime_id": 1, "realized_vol": 0.30}
    for _ in range(20):
        allocator.update("strategy_a", reward=-0.5, context=context1)
        allocator.update("strategy_b", reward=1.0, context=context1)
    
    weights1 = allocator.get_weights(context1)
    
    # Different contexts should yield different weights
    assert weights0["strategy_a"] > weights0["strategy_b"]
    assert weights1["strategy_b"] > weights1["strategy_a"]


def test_contextual_history_logged():
    """Test that context history is logged."""
    allocator = ContextualThompsonSampling(["strategy_a"])
    
    context = {"regime_id": 0, "realized_vol": 0.15}
    allocator.update("strategy_a", reward=0.5, context=context)
    allocator.update("strategy_a", reward=0.8, context=context)
    
    assert len(allocator.history) == 2
    assert "context" in allocator.history[0]
    assert "reward" in allocator.history[0]


def test_contextual_stats():
    """Test statistics retrieval."""
    strategies = ["strategy_a", "strategy_b"]
    allocator = ContextualThompsonSampling(strategies)
    
    context = {"regime_id": 0, "realized_vol": 0.15}
    
    allocator.update("strategy_a", reward=0.5, context=context)
    allocator.update("strategy_b", reward=0.8, context=context)
    
    stats = allocator.get_stats()
    
    assert "strategy_a" in stats
    assert "strategy_b" in stats
    assert stats["strategy_a"]["total_pulls"] == 1

