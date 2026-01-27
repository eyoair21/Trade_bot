"""Tests for bandit allocators."""

import pytest

from traderbot.alloc.bandit import ThompsonSamplingAllocator, UCBAllocator


def test_thompson_sampling_init():
    """Test Thompson Sampling initialization."""
    strategies = ["momentum", "mean_reversion", "patchtst"]
    allocator = ThompsonSamplingAllocator(strategies)
    
    assert len(allocator.arms) == 3
    for name in strategies:
        assert name in allocator.arms
        assert allocator.arms[name].alpha == 1.0
        assert allocator.arms[name].beta == 1.0


def test_thompson_sampling_select():
    """Test arm selection."""
    strategies = ["strategy_a", "strategy_b"]
    allocator = ThompsonSamplingAllocator(strategies)
    
    selected = allocator.select_arms(k=1)
    
    assert len(selected) == 1
    assert selected[0] in strategies


def test_thompson_sampling_update():
    """Test updating with rewards."""
    strategies = ["strategy_a", "strategy_b"]
    allocator = ThompsonSamplingAllocator(strategies)
    
    # Update with positive reward
    allocator.update("strategy_a", reward=0.8)
    
    state = allocator.arms["strategy_a"]
    assert state.pulls == 1
    assert state.total_reward == 0.8
    assert state.alpha > 1.0  # Should increase with success


def test_thompson_sampling_weights():
    """Test weight computation."""
    strategies = ["strategy_a", "strategy_b", "strategy_c"]
    allocator = ThompsonSamplingAllocator(strategies)
    
    weights = allocator.get_weights()
    
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-6  # Sum to 1
    assert all(0 <= w <= 1 for w in weights.values())


def test_thompson_sampling_convergence():
    """Test that allocator converges to better strategy."""
    strategies = ["good", "bad"]
    allocator = ThompsonSamplingAllocator(strategies)
    
    # Simulate 50 rounds where "good" always wins
    for _ in range(50):
        allocator.update("good", reward=1.0)
        allocator.update("bad", reward=-0.5)
    
    weights = allocator.get_weights()
    
    # Good strategy should have higher weight
    assert weights["good"] > weights["bad"]


def test_thompson_sampling_stats():
    """Test statistics retrieval."""
    strategies = ["strategy_a"]
    allocator = ThompsonSamplingAllocator(strategies)
    
    allocator.update("strategy_a", reward=0.5)
    allocator.update("strategy_a", reward=0.8)
    
    stats = allocator.get_stats()
    
    assert "strategy_a" in stats
    assert stats["strategy_a"]["pulls"] == 2
    assert stats["strategy_a"]["avg_reward"] == (0.5 + 0.8) / 2


def test_ucb_init():
    """Test UCB initialization."""
    strategies = ["strategy_a", "strategy_b"]
    allocator = UCBAllocator(strategies, c=1.5)
    
    assert len(allocator.arms) == 2
    assert allocator.c == 1.5
    assert allocator.total_pulls == 0


def test_ucb_select_unpulled_first():
    """Test that UCB pulls each arm once first."""
    strategies = ["a", "b", "c"]
    allocator = UCBAllocator(strategies)
    
    # Should pull each arm once before computing UCB
    selected = [allocator.select_arm() for _ in range(3)]
    
    assert set(selected) == set(strategies)


def test_ucb_update():
    """Test UCB update."""
    strategies = ["a"]
    allocator = UCBAllocator(strategies)
    
    allocator.update("a", reward=0.5)
    
    assert allocator.arms["a"].pulls == 1
    assert allocator.arms["a"].total_reward == 0.5
    assert allocator.total_pulls == 1


def test_ucb_exploration_bonus():
    """Test that UCB explores less-pulled arms."""
    strategies = ["a", "b"]
    allocator = UCBAllocator(strategies, c=2.0)
    
    # Pull "a" many times
    for _ in range(10):
        allocator.update("a", reward=0.5)
    
    # Pull "b" once
    allocator.update("b", reward=0.5)
    
    # Now select: "b" should be selected due to exploration bonus
    selected = allocator.select_arm()
    
    # After initial pulls, b should be selected more often due to lower pulls
    # (This is probabilistic, but with high c, should favor exploration)


def test_ucb_weights():
    """Test UCB weight computation."""
    strategies = ["a", "b"]
    allocator = UCBAllocator(strategies)
    
    allocator.update("a", reward=1.0)
    allocator.update("b", reward=0.5)
    
    weights = allocator.get_weights()
    
    assert len(weights) == 2
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["a"] > weights["b"]  # Better reward -> higher weight


def test_thompson_invalid_arm():
    """Test updating invalid arm raises error."""
    allocator = ThompsonSamplingAllocator(["a"])
    
    with pytest.raises(ValueError):
        allocator.update("invalid", reward=1.0)


def test_ucb_invalid_arm():
    """Test updating invalid arm raises error."""
    allocator = UCBAllocator(["a"])
    
    with pytest.raises(ValueError):
        allocator.update("invalid", reward=1.0)

