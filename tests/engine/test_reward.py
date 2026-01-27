"""Tests for reward computation."""

import numpy as np
import pandas as pd
import pytest

from traderbot.engine.reward import (
    RewardWeights,
    compute_run_metrics,
    compute_step_reward,
)


def test_compute_step_reward():
    """Test single-step reward computation."""
    weights = RewardWeights(lambda_dd=0.2, tau_turnover=0.001, kappa_breach=0.5)
    
    # Pure profit, no penalties
    reward = compute_step_reward(
        pnl_after_costs=100.0,
        drawdown_step=0.0,
        turnover_step=0.0,
        breach_step=0,
        weights=weights,
    )
    assert reward == 100.0
    
    # Profit with penalties
    reward = compute_step_reward(
        pnl_after_costs=100.0,
        drawdown_step=10.0,  # DD penalty: 0.2 * 10 = 2
        turnover_step=1000.0,  # Turnover penalty: 0.001 * 1000 = 1
        breach_step=1,  # Breach penalty: 0.5 * 1 = 0.5
        weights=weights,
    )
    expected = 100.0 - 2.0 - 1.0 - 0.5
    assert abs(reward - expected) < 1e-6


def test_compute_run_metrics_empty():
    """Test metrics computation with empty data."""
    weights = RewardWeights()
    
    equity_curve = pd.DataFrame()
    trades = pd.DataFrame()
    breaches = pd.DataFrame()
    
    metrics = compute_run_metrics(equity_curve, trades, breaches, weights)
    
    assert metrics["total_reward"] == 0.0
    assert metrics["pnl_net"] == 0.0
    assert metrics["sharpe"] == 0.0
    assert metrics["breaches_count"] == 0


def test_compute_run_metrics_basic():
    """Test metrics computation with sample data."""
    weights = RewardWeights(lambda_dd=0.2, tau_turnover=0.001, kappa_breach=0.5)
    
    # Create sample data
    equity_curve = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5),
        "equity": [100000, 101000, 100500, 102000, 103000],
        "drawdown": [0.0, 0.0, -0.005, 0.0, 0.0],
    })
    
    trades = pd.DataFrame({
        "pnl_net": [1000, -500, 1500, 1000],
        "turnover": [10000, 5000, 15000, 10000],
    })
    
    breaches = pd.DataFrame({
        "breach_type": ["drawdown"],
    })
    
    metrics = compute_run_metrics(equity_curve, trades, breaches, weights)
    
    assert metrics["pnl_net"] == 3000.0
    assert metrics["breaches_count"] == 1
    assert metrics["turnover"] == 40000.0
    assert "sharpe" in metrics
    assert "sortino" in metrics


def test_sharpe_computation():
    """Test Sharpe ratio calculation."""
    weights = RewardWeights()
    
    # Positive returns
    equity_curve = pd.DataFrame({
        "equity": [100, 101, 102, 103, 104, 105],
    })
    
    trades = pd.DataFrame({"pnl_net": [100], "turnover": [0]})
    breaches = pd.DataFrame()
    
    metrics = compute_run_metrics(equity_curve, trades, breaches, weights)
    assert metrics["sharpe"] > 0


def test_sortino_computation():
    """Test Sortino ratio calculation."""
    weights = RewardWeights()
    
    # Mixed returns
    equity_curve = pd.DataFrame({
        "equity": [100, 102, 101, 103, 102, 105],
    })
    
    trades = pd.DataFrame({"pnl_net": [100], "turnover": [0]})
    breaches = pd.DataFrame()
    
    metrics = compute_run_metrics(equity_curve, trades, breaches, weights)
    assert "sortino" in metrics

