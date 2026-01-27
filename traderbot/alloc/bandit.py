"""Multi-armed bandit allocation across strategies.

Implements Thompson Sampling, UCB, and Contextual Thompson Sampling
for dynamic strategy weighting.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BanditState:
    """State for a single bandit arm (strategy).
    
    For Thompson Sampling (Beta distribution):
        alpha: successes + 1
        beta: failures + 1
    
    For UCB:
        pulls: number of times pulled
        total_reward: cumulative reward
    """
    name: str
    alpha: float = 1.0  # Prior success
    beta: float = 1.0   # Prior failure
    pulls: int = 0
    total_reward: float = 0.0


class ThompsonSamplingAllocator:
    """Thompson Sampling for strategy allocation.
    
    Each strategy is a Beta distribution. Sample from each and pick highest.
    """
    
    def __init__(self, strategy_names: list[str]):
        """Initialize allocator.
        
        Args:
            strategy_names: List of strategy identifiers
        """
        self.arms = {name: BanditState(name=name) for name in strategy_names}
        self.history: list[dict[str, Any]] = []
    
    def select_arms(self, k: int = 1) -> list[str]:
        """Select top-k arms via Thompson Sampling.
        
        Args:
            k: Number of arms to select
        
        Returns:
            List of selected arm names
        """
        # Sample from each arm's Beta distribution
        samples = {
            name: np.random.beta(state.alpha, state.beta)
            for name, state in self.arms.items()
        }
        
        # Select top-k
        selected = sorted(samples.items(), key=lambda x: x[1], reverse=True)[:k]
        return [name for name, _ in selected]
    
    def get_weights(self, temperature: float = 1.0) -> dict[str, float]:
        """Get continuous weights for all arms via softmax of means.
        
        Args:
            temperature: Softmax temperature (higher = more uniform)
        
        Returns:
            Dict mapping arm name to weight (sum to 1.0)
        """
        # Mean of Beta(alpha, beta) = alpha / (alpha + beta)
        means = {
            name: state.alpha / (state.alpha + state.beta)
            for name, state in self.arms.items()
        }
        
        # Softmax
        scores = np.array(list(means.values())) / temperature
        scores = scores - scores.max()  # Numerical stability
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum()
        
        return {name: float(w) for name, w in zip(means.keys(), weights)}
    
    def update(self, arm_name: str, reward: float) -> None:
        """Update arm statistics with observed reward.
        
        Args:
            arm_name: Name of arm that was pulled
            reward: Observed reward (will be normalized to [0, 1])
        """
        if arm_name not in self.arms:
            raise ValueError(f"Unknown arm: {arm_name}")
        
        state = self.arms[arm_name]
        state.pulls += 1
        state.total_reward += reward
        
        # Convert reward to success probability (assume reward in [-1, 1])
        success = (reward + 1) / 2  # Map to [0, 1]
        
        # Bayesian update: if "success", increment alpha; else increment beta
        if success > 0.5:
            state.alpha += success
        else:
            state.beta += (1 - success)
        
        # Log
        self.history.append({
            "arm": arm_name,
            "reward": reward,
            "pulls": state.pulls,
            "alpha": state.alpha,
            "beta": state.beta,
        })
    
    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get current statistics for all arms.
        
        Returns:
            Dict mapping arm name to statistics
        """
        return {
            name: {
                "alpha": state.alpha,
                "beta": state.beta,
                "pulls": state.pulls,
                "total_reward": state.total_reward,
                "avg_reward": state.total_reward / state.pulls if state.pulls > 0 else 0.0,
                "mean_estimate": state.alpha / (state.alpha + state.beta),
            }
            for name, state in self.arms.items()
        }


class UCBAllocator:
    """Upper Confidence Bound (UCB1) for strategy allocation."""
    
    def __init__(self, strategy_names: list[str], c: float = 1.0):
        """Initialize UCB allocator.
        
        Args:
            strategy_names: List of strategy identifiers
            c: Exploration constant (higher = more exploration)
        """
        self.arms = {name: BanditState(name=name) for name in strategy_names}
        self.c = c
        self.total_pulls = 0
        self.history: list[dict[str, Any]] = []
    
    def select_arm(self) -> str:
        """Select arm with highest UCB score.
        
        Returns:
            Selected arm name
        """
        # If any arm not pulled, pull it first
        for name, state in self.arms.items():
            if state.pulls == 0:
                return name
        
        # Compute UCB scores
        ucb_scores = {}
        for name, state in self.arms.items():
            avg_reward = state.total_reward / state.pulls
            exploration_bonus = self.c * np.sqrt(
                np.log(self.total_pulls + 1) / state.pulls
            )
            ucb_scores[name] = avg_reward + exploration_bonus
        
        # Select max
        return max(ucb_scores, key=ucb_scores.get)  # type: ignore
    
    def update(self, arm_name: str, reward: float) -> None:
        """Update arm with observed reward.
        
        Args:
            arm_name: Name of pulled arm
            reward: Observed reward
        """
        if arm_name not in self.arms:
            raise ValueError(f"Unknown arm: {arm_name}")
        
        state = self.arms[arm_name]
        state.pulls += 1
        state.total_reward += reward
        self.total_pulls += 1
        
        # Log
        self.history.append({
            "arm": arm_name,
            "reward": reward,
            "pulls": state.pulls,
            "total_reward": state.total_reward,
            "avg_reward": state.total_reward / state.pulls,
        })
    
    def get_weights(self) -> dict[str, float]:
        """Get normalized weights based on average rewards.
        
        Returns:
            Dict mapping arm name to weight (sum to 1.0)
        """
        avg_rewards = {
            name: state.total_reward / state.pulls if state.pulls > 0 else 0.0
            for name, state in self.arms.items()
        }
        
        # Normalize to [0, 1]
        min_reward = min(avg_rewards.values())
        max_reward = max(avg_rewards.values())
        
        if max_reward == min_reward:
            # All equal, uniform weights
            n = len(avg_rewards)
            return {name: 1.0 / n for name in avg_rewards.keys()}
        
        normalized = {
            name: (r - min_reward) / (max_reward - min_reward)
            for name, r in avg_rewards.items()
        }
        
        total = sum(normalized.values())
        return {name: v / total for name, v in normalized.items()}


class ContextualThompsonSampling:
    """Contextual bandit with Thompson Sampling.
    
    Extends basic Thompson Sampling with context features:
    - regime_id: Current market regime
    - realized_vol: Recent volatility
    - spread: Bid-ask spread
    - turnover: Recent turnover
    
    Maintains separate Beta distributions for each (arm, context_bucket) pair.
    """
    
    def __init__(
        self,
        strategy_names: list[str],
        context_features: list[str] | None = None,
        n_context_buckets: int = 3,
    ):
        """Initialize contextual allocator.
        
        Args:
            strategy_names: List of strategy identifiers
            context_features: List of context feature names
            n_context_buckets: Number of buckets to discretize continuous contexts
        """
        self.strategy_names = strategy_names
        self.context_features = context_features or ["regime_id", "realized_vol"]
        self.n_context_buckets = n_context_buckets
        
        # State: {(arm, context_hash): BanditState}
        self.context_arms: dict[tuple[str, int], BanditState] = {}
        self.history: list[dict[str, Any]] = []
        
        # Context discretization thresholds (learned online)
        self.context_thresholds: dict[str, list[float]] = {}
    
    def _discretize_context(self, context: dict[str, float]) -> int:
        """Convert continuous context to discrete bucket.
        
        Args:
            context: Dictionary of context features
        
        Returns:
            Hash of discretized context
        """
        # Simple hash: concatenate discretized values
        bucket_ids = []
        
        for feature in self.context_features:
            value = context.get(feature, 0.0)
            
            if feature not in self.context_thresholds:
                # Initialize with uniform quantiles
                self.context_thresholds[feature] = [
                    i / self.n_context_buckets for i in range(1, self.n_context_buckets)
                ]
            
            # Find bucket
            thresholds = self.context_thresholds[feature]
            bucket = 0
            for threshold in thresholds:
                if value > threshold:
                    bucket += 1
            
            bucket_ids.append(bucket)
        
        # Simple hash (could use better hashing)
        return hash(tuple(bucket_ids))
    
    def select_arms(
        self,
        context: dict[str, float],
        k: int = 1,
    ) -> list[str]:
        """Select top-k arms given context.
        
        Args:
            context: Context features
            k: Number of arms to select
        
        Returns:
            List of selected arm names
        """
        context_hash = self._discretize_context(context)
        
        # Sample from each arm's context-specific distribution
        samples = {}
        for arm_name in self.strategy_names:
            key = (arm_name, context_hash)
            
            if key not in self.context_arms:
                self.context_arms[key] = BanditState(name=arm_name)
            
            state = self.context_arms[key]
            samples[arm_name] = np.random.beta(state.alpha, state.beta)
        
        # Select top-k
        selected = sorted(samples.items(), key=lambda x: x[1], reverse=True)[:k]
        return [name for name, _ in selected]
    
    def get_weights(
        self,
        context: dict[str, float],
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """Get continuous weights for all arms given context.
        
        Args:
            context: Context features
            temperature: Softmax temperature
        
        Returns:
            Dictionary of arm -> weight
        """
        context_hash = self._discretize_context(context)
        
        # Get mean estimates for each arm in this context
        means = {}
        for arm_name in self.strategy_names:
            key = (arm_name, context_hash)
            
            if key not in self.context_arms:
                self.context_arms[key] = BanditState(name=arm_name)
            
            state = self.context_arms[key]
            means[arm_name] = state.alpha / (state.alpha + state.beta)
        
        # Softmax
        scores = np.array(list(means.values())) / temperature
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum()
        
        return {name: float(w) for name, w in zip(means.keys(), weights)}
    
    def update(
        self,
        arm_name: str,
        reward: float,
        context: dict[str, float],
    ) -> None:
        """Update arm statistics with observed reward in context.
        
        Args:
            arm_name: Name of arm that was pulled
            reward: Observed reward
            context: Context features at time of pull
        """
        if arm_name not in self.strategy_names:
            raise ValueError(f"Unknown arm: {arm_name}")
        
        context_hash = self._discretize_context(context)
        key = (arm_name, context_hash)
        
        if key not in self.context_arms:
            self.context_arms[key] = BanditState(name=arm_name)
        
        state = self.context_arms[key]
        state.pulls += 1
        state.total_reward += reward
        
        # Bayesian update
        success = (reward + 1) / 2  # Map to [0, 1]
        if success > 0.5:
            state.alpha += success
        else:
            state.beta += (1 - success)
        
        # Log
        self.history.append({
            "arm": arm_name,
            "reward": reward,
            "context": context.copy(),
            "context_hash": context_hash,
            "pulls": state.pulls,
        })
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all context-arm pairs.
        
        Returns:
            Dictionary with aggregate statistics
        """
        stats_by_arm: dict[str, dict] = {name: {"total_pulls": 0, "contexts": []} for name in self.strategy_names}
        
        for (arm_name, context_hash), state in self.context_arms.items():
            stats_by_arm[arm_name]["total_pulls"] += state.pulls
            stats_by_arm[arm_name]["contexts"].append({
                "context_hash": context_hash,
                "pulls": state.pulls,
                "alpha": state.alpha,
                "beta": state.beta,
                "mean": state.alpha / (state.alpha + state.beta),
            })
        
        return stats_by_arm
    
    def save_history(self, output_path: str) -> None:
        """Save context history to CSV.
        
        Args:
            output_path: Path to save CSV
        """
        if self.history:
            # Flatten context dict for CSV
            rows = []
            for record in self.history:
                row = {
                    "arm": record["arm"],
                    "reward": record["reward"],
                    "pulls": record["pulls"],
                }
                # Add context features
                for feature, value in record["context"].items():
                    row[f"context_{feature}"] = value
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

