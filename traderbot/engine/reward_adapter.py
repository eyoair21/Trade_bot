"""Online reward weight adaptation based on rolling OOS metrics.

Supports EWMA and lightweight Bayesian optimization for self-adjusting reward weights.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from traderbot.engine.reward import RewardWeights


@dataclass
class AdapterConfig:
    """Configuration for reward weight adaptation.
    
    Args:
        mode: Adaptation mode ('ewma' or 'bayesopt')
        alpha: EWMA smoothing factor (0-1, higher = more reactive)
        learning_rate: Step size for gradient-based updates
        clip_min: Minimum weight values
        clip_max: Maximum weight values
        window: Number of recent splits to consider
    """
    mode: Literal["ewma", "bayesopt"] = "ewma"
    alpha: float = 0.3
    learning_rate: float = 0.01
    clip_min: dict[str, float] | None = None
    clip_max: dict[str, float] | None = None
    window: int = 10
    
    def __post_init__(self) -> None:
        """Set default clip ranges."""
        if self.clip_min is None:
            self.clip_min = {"lambda_dd": 0.0, "tau_turnover": 1e-6, "kappa_breach": 0.0}
        if self.clip_max is None:
            self.clip_max = {"lambda_dd": 1.0, "tau_turnover": 0.01, "kappa_breach": 2.0}


class RewardAdapter:
    """Adapts reward weights based on rolling OOS performance.
    
    Implements two backends:
    - EWMA: Exponentially weighted moving average of metric gradients
    - BayesOpt-lite: Simple Thompson sampling over discrete weight grid
    """
    
    def __init__(self, config: AdapterConfig, initial_weights: RewardWeights):
        """Initialize adapter.
        
        Args:
            config: Adapter configuration
            initial_weights: Starting reward weights
        """
        self.config = config
        self.current_weights = initial_weights
        
        # History tracking
        self.history: list[dict] = []
        self.oos_metrics_history: list[dict] = []
        
        # For BayesOpt mode
        if config.mode == "bayesopt":
            self._init_bayesopt_grid()
    
    def _init_bayesopt_grid(self) -> None:
        """Initialize discrete grid for Bayesian optimization."""
        # Simple grid: 5 values per parameter
        self.grid = {
            "lambda_dd": np.linspace(0.0, 1.0, 5),
            "tau_turnover": np.logspace(-5, -2, 5),
            "kappa_breach": np.linspace(0.0, 2.0, 5),
        }
        
        # Track performance for each grid point (Thompson Sampling)
        self.grid_stats: dict[tuple, dict] = {}
    
    def update(
        self,
        current_weights: RewardWeights,
        oos_metrics: dict[str, float],
    ) -> RewardWeights:
        """Update weights based on OOS metrics.
        
        Args:
            current_weights: Current reward weights
            oos_metrics: Out-of-sample metrics (total_reward, sharpe, etc.)
        
        Returns:
            Updated reward weights
        """
        # Store current state
        self.oos_metrics_history.append(oos_metrics)
        
        # Adapt based on mode
        if self.config.mode == "ewma":
            new_weights = self._update_ewma(current_weights, oos_metrics)
        elif self.config.mode == "bayesopt":
            new_weights = self._update_bayesopt(current_weights, oos_metrics)
        else:
            raise ValueError(f"Unknown adaptation mode: {self.config.mode}")
        
        # Log change
        self.history.append({
            "old_weights": {
                "lambda_dd": current_weights.lambda_dd,
                "tau_turnover": current_weights.tau_turnover,
                "kappa_breach": current_weights.kappa_breach,
            },
            "new_weights": {
                "lambda_dd": new_weights.lambda_dd,
                "tau_turnover": new_weights.tau_turnover,
                "kappa_breach": new_weights.kappa_breach,
            },
            "oos_total_reward": oos_metrics.get("total_reward", 0.0),
            "oos_sharpe": oos_metrics.get("sharpe", 0.0),
        })
        
        self.current_weights = new_weights
        return new_weights
    
    def _update_ewma(
        self,
        current_weights: RewardWeights,
        oos_metrics: dict[str, float],
    ) -> RewardWeights:
        """Update using EWMA of metric gradients.
        
        Simple heuristic:
        - If total_reward improved, slightly decrease penalties (allow more risk)
        - If total_reward decreased, increase penalties (be more conservative)
        """
        # Compute gradient signal
        if len(self.oos_metrics_history) < 2:
            return current_weights  # Need at least 2 observations
        
        prev_reward = self.oos_metrics_history[-2].get("total_reward", 0.0)
        curr_reward = oos_metrics.get("total_reward", 0.0)
        delta_reward = curr_reward - prev_reward
        
        # Normalize by absolute value to avoid huge swings
        signal = np.tanh(delta_reward / (abs(prev_reward) + 1e-6))
        
        # Update each weight with EWMA
        # If reward increased (signal > 0), decrease penalties
        # If reward decreased (signal < 0), increase penalties
        alpha = self.config.alpha
        lr = self.config.learning_rate
        
        lambda_dd = current_weights.lambda_dd * (1 - alpha) + alpha * (
            current_weights.lambda_dd - lr * signal
        )
        tau_turnover = current_weights.tau_turnover * (1 - alpha) + alpha * (
            current_weights.tau_turnover - lr * signal * 0.0001  # Smaller steps for tau
        )
        kappa_breach = current_weights.kappa_breach * (1 - alpha) + alpha * (
            current_weights.kappa_breach - lr * signal
        )
        
        # Clip to valid ranges
        lambda_dd = np.clip(
            lambda_dd,
            self.config.clip_min["lambda_dd"],  # type: ignore
            self.config.clip_max["lambda_dd"],  # type: ignore
        )
        tau_turnover = np.clip(
            tau_turnover,
            self.config.clip_min["tau_turnover"],  # type: ignore
            self.config.clip_max["tau_turnover"],  # type: ignore
        )
        kappa_breach = np.clip(
            kappa_breach,
            self.config.clip_min["kappa_breach"],  # type: ignore
            self.config.clip_max["kappa_breach"],  # type: ignore
        )
        
        return RewardWeights(
            lambda_dd=float(lambda_dd),
            tau_turnover=float(tau_turnover),
            kappa_breach=float(kappa_breach),
        )
    
    def _update_bayesopt(
        self,
        current_weights: RewardWeights,
        oos_metrics: dict[str, float],
    ) -> RewardWeights:
        """Update using Bayesian optimization over discrete grid.
        
        Uses Thompson Sampling to select next weight configuration.
        """
        # Record performance for current weights
        weights_tuple = (
            current_weights.lambda_dd,
            current_weights.tau_turnover,
            current_weights.kappa_breach,
        )
        
        if weights_tuple not in self.grid_stats:
            self.grid_stats[weights_tuple] = {"alpha": 1.0, "beta": 1.0}
        
        # Update Beta distribution (assume reward in [-1, 1])
        reward = oos_metrics.get("total_reward", 0.0)
        normalized_reward = (np.tanh(reward / 1000) + 1) / 2  # Map to [0, 1]
        
        self.grid_stats[weights_tuple]["alpha"] += normalized_reward
        self.grid_stats[weights_tuple]["beta"] += (1 - normalized_reward)
        
        # Sample from all grid points and select best
        best_sample = -np.inf
        best_config = current_weights
        
        for lambda_dd in self.grid["lambda_dd"]:
            for tau in self.grid["tau_turnover"]:
                for kappa in self.grid["kappa_breach"]:
                    config_tuple = (lambda_dd, tau, kappa)
                    
                    if config_tuple not in self.grid_stats:
                        self.grid_stats[config_tuple] = {"alpha": 1.0, "beta": 1.0}
                    
                    # Thompson sample
                    stats = self.grid_stats[config_tuple]
                    sample = np.random.beta(stats["alpha"], stats["beta"])
                    
                    if sample > best_sample:
                        best_sample = sample
                        best_config = RewardWeights(
                            lambda_dd=float(lambda_dd),
                            tau_turnover=float(tau),
                            kappa_breach=float(kappa),
                        )
        
        return best_config
    
    def save(self, output_path: Path) -> None:
        """Save adaptation history and current weights.
        
        Args:
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": {
                "mode": self.config.mode,
                "alpha": self.config.alpha,
                "learning_rate": self.config.learning_rate,
            },
            "current_weights": {
                "lambda_dd": self.current_weights.lambda_dd,
                "tau_turnover": self.current_weights.tau_turnover,
                "kappa_breach": self.current_weights.kappa_breach,
            },
            "history": self.history,
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_config(
        cls,
        mode: str = "ewma",
        alpha: float = 0.3,
        learning_rate: float = 0.01,
        initial_weights: RewardWeights | None = None,
    ) -> "RewardAdapter":
        """Create adapter from configuration parameters.
        
        Args:
            mode: Adaptation mode ('ewma' or 'bayesopt')
            alpha: EWMA smoothing factor
            learning_rate: Learning rate for updates
            initial_weights: Starting weights (default if None)
        
        Returns:
            Configured RewardAdapter instance
        """
        config = AdapterConfig(mode=mode, alpha=alpha, learning_rate=learning_rate)  # type: ignore
        
        if initial_weights is None:
            initial_weights = RewardWeights()
        
        return cls(config=config, initial_weights=initial_weights)

