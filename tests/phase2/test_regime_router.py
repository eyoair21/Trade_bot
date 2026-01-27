"""Tests for regime detection and routing."""

import numpy as np
import pandas as pd
import pytest

from traderbot.research.regime import (
    RegimeRouter,
    compute_regime_features,
    detect_regimes,
)


def test_detect_regimes_kmeans():
    """Test K-Means regime detection."""
    # Create synthetic features
    np.random.seed(42)
    n = 100
    features_df = pd.DataFrame({
        "returns": np.random.randn(n) * 0.01,
        "volatility": np.abs(np.random.randn(n)) * 0.02,
        "trend": np.random.randn(n) * 0.005,
    })
    
    regimes = detect_regimes(features_df, k=3, method="kmeans")
    
    assert len(regimes) == n
    assert set(regimes.unique()) <= {0, 1, 2}


def test_compute_regime_features():
    """Test feature computation from price data."""
    # Create price data
    dates = pd.date_range("2023-01-01", periods=50)
    prices = 100 * (1 + np.random.randn(50).cumsum() * 0.01)
    
    price_df = pd.DataFrame({
        "close": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "volume": np.random.randint(1000, 10000, 50),
    }, index=dates)
    
    features = compute_regime_features(price_df, lookback=10)
    
    assert "returns" in features.columns
    assert "volatility" in features.columns
    assert "trend" in features.columns
    assert len(features) == len(price_df)


def test_regime_router_selection():
    """Test that router selects models by regime."""
    router = RegimeRouter.create_default(n_regimes=3)
    
    model0 = router.select_model(regime=0)
    model1 = router.select_model(regime=1)
    model2 = router.select_model(regime=2)
    
    # Default mapping
    assert model0 == "mean_reversion"
    assert model1 == "momentum"
    assert model2 == "defensive"


def test_regime_router_params():
    """Test that router provides different params per regime."""
    router = RegimeRouter.create_default(n_regimes=3)
    
    alloc_params_0 = router.get_alloc_params(regime=0)
    alloc_params_2 = router.get_alloc_params(regime=2)
    
    # Low vol vs high vol should have different params
    assert alloc_params_0["temperature"] < alloc_params_2["temperature"]


def test_regime_router_history():
    """Test regime history logging."""
    router = RegimeRouter.create_default(n_regimes=3)
    
    dates = pd.date_range("2023-01-01", periods=5)
    for date in dates:
        regime = np.random.choice([0, 1, 2])
        metrics = {"sharpe": 0.5, "returns": 0.01}
        router.log_regime(date, regime, metrics)
    
    assert len(router.regime_history) == 5
    assert "regime" in router.regime_history[0]
    assert "model" in router.regime_history[0]

