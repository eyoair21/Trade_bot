"""Tests for reward weight adaptation."""

import pytest

from traderbot.engine.reward import RewardWeights
from traderbot.engine.reward_adapter import AdapterConfig, RewardAdapter


def test_reward_adapter_init():
    """Test adapter initialization."""
    config = AdapterConfig(mode="ewma")
    weights = RewardWeights()
    adapter = RewardAdapter(config, weights)
    
    assert adapter.config.mode == "ewma"
    assert adapter.current_weights == weights


def test_ewma_adaptation_improves():
    """Test that EWMA adapts weights when reward improves."""
    adapter = RewardAdapter.from_config(mode="ewma", alpha=0.3)
    
    initial_weights = adapter.current_weights
    
    # Simulate improving performance
    for i in range(3):
        oos_metrics = {"total_reward": 100.0 * (i + 1), "sharpe": 0.5}
        new_weights = adapter.update(adapter.current_weights, oos_metrics)
    
    # Weights should have changed
    assert new_weights.lambda_dd != initial_weights.lambda_dd


def test_ewma_stays_in_bounds():
    """Test that EWMA keeps weights in valid ranges."""
    adapter = RewardAdapter.from_config(mode="ewma")
    
    # Many updates
    for i in range(20):
        oos_metrics = {"total_reward": 100.0 * ((-1) ** i), "sharpe": 0.5}
        new_weights = adapter.update(adapter.current_weights, oos_metrics)
    
    # Should stay in bounds
    assert 0.0 <= new_weights.lambda_dd <= 1.0
    assert 1e-6 <= new_weights.tau_turnover <= 0.01
    assert 0.0 <= new_weights.kappa_breach <= 2.0


def test_bayesopt_mode():
    """Test Bayesian optimization mode."""
    adapter = RewardAdapter.from_config(mode="bayesopt", alpha=0.3)
    
    # Update multiple times
    for i in range(5):
        oos_metrics = {"total_reward": 50.0 + i * 10, "sharpe": 0.5 + i * 0.1}
        new_weights = adapter.update(adapter.current_weights, oos_metrics)
    
    # Should select from grid
    assert new_weights.lambda_dd in adapter.grid["lambda_dd"]


def test_adapter_history_logged():
    """Test that adaptation history is logged."""
    adapter = RewardAdapter.from_config(mode="ewma")
    
    oos_metrics = {"total_reward": 100.0, "sharpe": 0.8}
    adapter.update(adapter.current_weights, oos_metrics)
    adapter.update(adapter.current_weights, oos_metrics)
    
    assert len(adapter.history) == 2
    assert "old_weights" in adapter.history[0]
    assert "new_weights" in adapter.history[0]


def test_from_config_factory():
    """Test factory method."""
    adapter = RewardAdapter.from_config(
        mode="ewma",
        alpha=0.5,
        learning_rate=0.02,
    )
    
    assert adapter.config.mode == "ewma"
    assert adapter.config.alpha == 0.5
    assert adapter.config.learning_rate == 0.02

