"""Integration tests for Phase 2 in walk-forward backtesting."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traderbot.engine.phase2_integration import Phase2Config, Phase2Orchestrator
from traderbot.engine.reward import RewardWeights


def test_phase2_orchestrator_init():
    """Test Phase 2 orchestrator initialization."""
    config = Phase2Config(
        reward_adapt_mode="ewma",
        regime_mode=True,
        use_contextual_alloc=True,
    )
    
    orchestrator = Phase2Orchestrator(
        config=config,
        initial_weights=RewardWeights(),
        strategy_names=["momentum", "mean_reversion"],
        capital=100000.0,
    )
    
    assert orchestrator.reward_adapter is not None
    assert orchestrator.regime_router is not None
    assert orchestrator.allocator is not None


def test_pre_split_regime_detection():
    """Test regime detection in pre-split."""
    config = Phase2Config(regime_mode=True, regime_k=3)
    orchestrator = Phase2Orchestrator(config, RewardWeights())
    
    # Create synthetic price data
    dates = pd.date_range("2023-01-01", periods=100)
    price_data = pd.DataFrame({
        "close": 100 * (1 + np.random.randn(100).cumsum() * 0.01),
        "high": 100 * (1 + np.random.randn(100).cumsum() * 0.01),
        "low": 100 * (1 + np.random.randn(100).cumsum() * 0.01),
        "volume": np.random.randint(1000, 10000, 100),
    }, index=dates)
    
    result = orchestrator.pre_split(
        split_id=0,
        price_data=price_data,
        date=dates[-1],
    )
    
    assert "regime_id" in result
    assert "model" in result
    assert len(orchestrator.regime_history) == 1


def test_post_split_reward_adaptation():
    """Test reward weight adaptation in post-split."""
    config = Phase2Config(reward_adapt_mode="ewma")
    orchestrator = Phase2Orchestrator(config, RewardWeights())
    
    initial_lambda = orchestrator.current_weights.lambda_dd
    
    # Run multiple splits with improving performance
    for i in range(3):
        oos_metrics = {
            "total_reward": 100.0 * (i + 1),
            "sharpe": 0.5 + i * 0.1,
        }
        
        new_weights = orchestrator.post_split(
            split_id=i,
            oos_metrics=oos_metrics,
            trades=pd.DataFrame(),
            breaches=pd.DataFrame(),
            date=pd.Timestamp("2023-01-01"),
        )
    
    # Weights should have changed
    assert new_weights.lambda_dd != initial_lambda
    assert len(orchestrator.reward_history) == 3


def test_portfolio_limits_enforced():
    """Test that portfolio limits are enforced."""
    config = Phase2Config(max_gross=1.0, max_position_pct=0.10)
    orchestrator = Phase2Orchestrator(config, RewardWeights(), capital=100000)
    
    # Propose orders exceeding limits
    proposed = {
        "AAPL": 20000,  # 20% - exceeds per-position limit
        "MSFT": 30000,  # 30%
        "NVDA": 60000,  # 60%
    }
    prices = {"AAPL": 150, "MSFT": 300, "NVDA": 500}
    
    clipped = orchestrator.clip_orders(proposed, prices)
    
    # Check gross limit
    gross = sum(abs(v) for v in clipped.values())
    assert gross <= 100000
    
    # Check per-position limit
    for ticker, notional in clipped.items():
        assert abs(notional) <= 10000


def test_cost_estimation():
    """Test cost estimation in both modes."""
    # Simple mode
    config_simple = Phase2Config(cost_model_mode="simple")
    orch_simple = Phase2Orchestrator(config_simple, RewardWeights())
    
    cost_simple = orch_simple.estimate_costs("AAPL", 1000, 150.0)
    assert cost_simple > 0
    
    # Realistic mode
    config_real = Phase2Config(cost_model_mode="realistic")
    orch_real = Phase2Orchestrator(config_real, RewardWeights())
    
    cost_real = orch_real.estimate_costs("AAPL", 1000, 150.0)
    assert cost_real > 0


def test_contextual_allocator():
    """Test contextual allocator integration."""
    config = Phase2Config(use_contextual_alloc=True)
    strategies = ["momentum", "mean_reversion", "patchtst"]
    orchestrator = Phase2Orchestrator(
        config, RewardWeights(), strategy_names=strategies
    )
    
    context = {"regime_id": 0, "realized_vol": 0.15}
    
    # Get weights
    weights = orchestrator.get_allocator_weights(context)
    assert weights is not None
    assert len(weights) == 3
    
    # Update
    orchestrator.update_allocator("momentum", reward=0.8, context=context)
    assert len(orchestrator.alloc_history) == 1


def test_artifact_persistence():
    """Test that all artifacts are saved."""
    config = Phase2Config(
        reward_adapt_mode="ewma",
        regime_mode=True,
        use_contextual_alloc=True,
    )
    orchestrator = Phase2Orchestrator(
        config,
        RewardWeights(),
        strategy_names=["momentum"],
    )
    
    # Generate some history
    orchestrator.reward_history.append({
        "split_id": 0,
        "date": "2023-01-01",
        "old_lambda_dd": 0.2,
        "new_lambda_dd": 0.22,
        "oos_total_reward": 100.0,
    })
    
    orchestrator.regime_history.append({
        "split_id": 0,
        "date": "2023-01-01",
        "regime_id": 0,
        "model": "momentum",
    })
    
    orchestrator.alloc_history.append({
        "strategy": "momentum",
        "reward": 0.8,
        "regime_id": 0,
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        orchestrator.save_artifacts(output_dir)
        
        # Check files exist
        assert (output_dir / "reward_adapt.csv").exists()
        assert (output_dir / "regimes.csv").exists()
        assert (output_dir / "portfolio_stats.json").exists()
        assert (output_dir / "costs_summary.json").exists()
        assert (output_dir / "alloc_context.csv").exists()


def test_summary_statistics():
    """Test summary statistics generation."""
    config = Phase2Config(
        reward_adapt_mode="ewma",
        regime_mode=True,
    )
    orchestrator = Phase2Orchestrator(config, RewardWeights())
    
    summary = orchestrator.get_summary()
    
    assert "reward_adaptation" in summary
    assert "regime_detection" in summary
    assert "portfolio_management" in summary
    assert "costs" in summary
    assert "allocation" in summary


def test_phase2_disabled():
    """Test that Phase 2 works when all features disabled."""
    config = Phase2Config(
        reward_adapt_mode="off",
        regime_mode=False,
        use_contextual_alloc=False,
    )
    orchestrator = Phase2Orchestrator(config, RewardWeights())
    
    assert orchestrator.reward_adapter is None
    assert orchestrator.regime_router is None
    
    # Should still work
    result = orchestrator.pre_split(0, pd.DataFrame(), pd.Timestamp("2023-01-01"))
    assert result["regime_id"] is None
    
    weights = orchestrator.post_split(
        0, {}, pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2023-01-01")
    )
    assert weights == orchestrator.current_weights

