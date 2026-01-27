"""Regime detection and model routing.

Implements K-Means and HMM-based regime detection for market state classification.
Routes models and allocation strategies based on detected regimes.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeConfig:
    """Configuration for regime detection.
    
    Args:
        method: Detection method ('kmeans' or 'hmm')
        n_regimes: Number of regimes to detect
        features: List of feature columns for clustering
        lookback: Rolling window for feature computation
    """
    method: Literal["kmeans", "hmm"] = "kmeans"
    n_regimes: int = 3
    features: list[str] | None = None
    lookback: int = 20
    
    def __post_init__(self) -> None:
        """Set default features."""
        if self.features is None:
            self.features = ["returns", "volatility", "trend", "volume_ratio"]


def detect_regimes(
    features_df: pd.DataFrame,
    k: int = 3,
    method: str = "kmeans",
) -> pd.Series:
    """Detect market regimes from feature DataFrame.
    
    Args:
        features_df: DataFrame with features (columns) and dates (index)
        k: Number of regimes
        method: 'kmeans' or 'hmm'
    
    Returns:
        Series with regime labels (0 to k-1) indexed by date
    """
    if method == "kmeans":
        return _detect_kmeans(features_df, k=k)
    elif method == "hmm":
        return _detect_hmm(features_df, n_states=k)
    else:
        raise ValueError(f"Unknown method: {method}")


def _detect_kmeans(features_df: pd.DataFrame, k: int = 3) -> pd.Series:
    """Detect regimes using K-Means clustering.
    
    Args:
        features_df: Feature DataFrame
        k: Number of clusters
    
    Returns:
        Series of regime labels
    """
    # Drop NaN rows
    clean_df = features_df.dropna()
    
    if len(clean_df) == 0:
        return pd.Series(0, index=features_df.index)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_df.values)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Create series with original index
    regime_series = pd.Series(0, index=features_df.index)
    regime_series.loc[clean_df.index] = labels
    
    # Forward fill for NaN dates
    regime_series = regime_series.ffill().fillna(0)
    
    return regime_series


def _detect_hmm(features_df: pd.DataFrame, n_states: int = 3) -> pd.Series:
    """Detect regimes using Hidden Markov Model.
    
    Note: Lightweight implementation without hmmlearn dependency.
    Falls back to K-Means if HMM not available.
    
    Args:
        features_df: Feature DataFrame
        n_states: Number of hidden states
    
    Returns:
        Series of regime labels
    """
    try:
        from hmmlearn import hmm
        
        clean_df = features_df.dropna()
        
        if len(clean_df) < n_states * 10:
            # Not enough data for HMM, fall back to K-Means
            return _detect_kmeans(features_df, k=n_states)
        
        # Fit Gaussian HMM
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        
        X = clean_df.values
        model.fit(X)
        
        # Predict states
        states = model.predict(X)
        
        regime_series = pd.Series(0, index=features_df.index)
        regime_series.loc[clean_df.index] = states
        regime_series = regime_series.ffill().fillna(0)
        
        return regime_series
    
    except ImportError:
        # hmmlearn not available, fall back to K-Means
        return _detect_kmeans(features_df, k=n_states)


def compute_regime_features(
    price_df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Compute features for regime detection from price data.
    
    Args:
        price_df: DataFrame with OHLCV data
        lookback: Rolling window size
    
    Returns:
        DataFrame with regime features
    """
    features = pd.DataFrame(index=price_df.index)
    
    # Returns
    features["returns"] = price_df["close"].pct_change()
    
    # Volatility (rolling std of returns)
    features["volatility"] = features["returns"].rolling(lookback).std()
    
    # Trend (rolling mean of returns)
    features["trend"] = features["returns"].rolling(lookback).mean()
    
    # Volume ratio (current / average)
    if "volume" in price_df.columns:
        avg_volume = price_df["volume"].rolling(lookback).mean()
        features["volume_ratio"] = price_df["volume"] / (avg_volume + 1e-10)
    else:
        features["volume_ratio"] = 1.0
    
    # Range (high-low / close)
    if "high" in price_df.columns and "low" in price_df.columns:
        features["range"] = (
            (price_df["high"] - price_df["low"]) / (price_df["close"] + 1e-10)
        )
    
    return features


@dataclass
class RegimeRouterConfig:
    """Configuration for routing models/strategies by regime.
    
    Maps regime IDs to model configurations and allocation params.
    """
    regime_models: dict[int, str]  # {regime_id: model_name}
    regime_alloc_params: dict[int, dict]  # {regime_id: {param: value}}
    regime_sizing_params: dict[int, dict]  # {regime_id: {param: value}}


class RegimeRouter:
    """Routes model selection and allocation parameters based on regime.
    
    Example routing:
    - Regime 0 (low vol): mean-reversion model, higher leverage
    - Regime 1 (trending): momentum model, standard leverage
    - Regime 2 (high vol): defensive model, lower leverage
    """
    
    def __init__(self, config: RegimeRouterConfig):
        """Initialize router.
        
        Args:
            config: Router configuration
        """
        self.config = config
        self.current_regime: int | None = None
        self.regime_history: list[dict] = []
    
    def select_model(self, regime: int) -> str:
        """Select model based on regime.
        
        Args:
            regime: Current regime ID
        
        Returns:
            Model name to use
        """
        self.current_regime = regime
        return self.config.regime_models.get(regime, "default")
    
    def get_alloc_params(self, regime: int) -> dict:
        """Get allocation parameters for regime.
        
        Args:
            regime: Current regime ID
        
        Returns:
            Dictionary of allocation parameters
        """
        return self.config.regime_alloc_params.get(regime, {})
    
    def get_sizing_params(self, regime: int) -> dict:
        """Get position sizing parameters for regime.
        
        Args:
            regime: Current regime ID
        
        Returns:
            Dictionary of sizing parameters
        """
        return self.config.regime_sizing_params.get(regime, {})
    
    def log_regime(self, date: pd.Timestamp, regime: int, metrics: dict) -> None:
        """Log regime transition and performance.
        
        Args:
            date: Date of observation
            regime: Regime ID
            metrics: Performance metrics in this regime
        """
        self.regime_history.append({
            "date": str(date),
            "regime": regime,
            "model": self.select_model(regime),
            **metrics,
        })
    
    def save(self, output_path: Path) -> None:
        """Save regime history.
        
        Args:
            output_path: Path to save CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.regime_history:
            df = pd.DataFrame(self.regime_history)
            df.to_csv(output_path, index=False)
    
    @classmethod
    def create_default(cls, n_regimes: int = 3) -> "RegimeRouter":
        """Create router with default configuration.
        
        Args:
            n_regimes: Number of regimes
        
        Returns:
            Configured RegimeRouter
        """
        # Default routing rules
        if n_regimes == 3:
            regime_models = {
                0: "mean_reversion",  # Low vol regime
                1: "momentum",  # Trending regime
                2: "defensive",  # High vol regime
            }
            regime_alloc_params = {
                0: {"temperature": 0.5},  # Less exploration in low vol
                1: {"temperature": 1.0},  # Standard
                2: {"temperature": 1.5},  # More exploration in high vol
            }
            regime_sizing_params = {
                0: {"max_leverage": 1.2},  # Higher leverage in low vol
                1: {"max_leverage": 1.0},  # Standard
                2: {"max_leverage": 0.7},  # Lower leverage in high vol
            }
        else:
            # Generic fallback
            regime_models = {i: f"model_{i}" for i in range(n_regimes)}
            regime_alloc_params = {i: {} for i in range(n_regimes)}
            regime_sizing_params = {i: {} for i in range(n_regimes)}
        
        config = RegimeRouterConfig(
            regime_models=regime_models,
            regime_alloc_params=regime_alloc_params,
            regime_sizing_params=regime_sizing_params,
        )
        
        return cls(config=config)

