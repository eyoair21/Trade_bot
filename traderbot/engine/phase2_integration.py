"""Phase 2 integration module for walk-forward backtesting.

Wires reward adaptation, regime detection, portfolio management, costs, and contextual allocation
into the backtest loop.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from traderbot.alloc.bandit import ContextualThompsonSampling, ThompsonSamplingAllocator
from traderbot.engine.costs import CostModel
from traderbot.engine.portfolio import PortfolioLimits, PortfolioManager
from traderbot.engine.reward import RewardWeights, compute_run_metrics
from traderbot.engine.reward_adapter import RewardAdapter
from traderbot.research.regime import (
    RegimeRouter,
    compute_regime_features,
    detect_regimes,
)


@dataclass
class Phase2Config:
    """Configuration for Phase 2 features.
    
    Args:
        reward_adapt_mode: 'off', 'ewma', or 'bayesopt'
        adapt_interval: Number of slices between adaptations
        regime_mode: Whether to enable regime detection
        regime_k: Number of regimes
        regime_method: 'kmeans' or 'hmm'
        use_contextual_alloc: Use contextual bandit
        cost_model_mode: 'simple' or 'realistic'
        max_gross: Maximum gross exposure
        max_net: Maximum net exposure
        max_position_pct: Maximum per-position size
        max_sector_pct: Maximum per-sector exposure
    """
    reward_adapt_mode: str = "off"
    adapt_interval: int = 1
    regime_mode: bool = False
    regime_k: int = 3
    regime_method: str = "kmeans"
    use_contextual_alloc: bool = False
    cost_model_mode: str = "simple"
    max_gross: float = 1.0
    max_net: float = 0.5
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30


class Phase2Orchestrator:
    """Orchestrates Phase 2 features across walk-forward splits."""
    
    def __init__(
        self,
        config: Phase2Config,
        initial_weights: RewardWeights,
        strategy_names: list[str] | None = None,
        capital: float = 100000.0,
    ):
        """Initialize orchestrator.
        
        Args:
            config: Phase 2 configuration
            initial_weights: Starting reward weights
            strategy_names: List of strategies for bandit allocation
            capital: Initial capital
        """
        self.config = config
        self.capital = capital
        
        # Reward adaptation
        self.reward_adapter: RewardAdapter | None = None
        if config.reward_adapt_mode != "off":
            self.reward_adapter = RewardAdapter.from_config(
                mode=config.reward_adapt_mode,
                initial_weights=initial_weights,
            )
        self.current_weights = initial_weights
        
        # Regime detection
        self.regime_router: RegimeRouter | None = None
        if config.regime_mode:
            self.regime_router = RegimeRouter.create_default(n_regimes=config.regime_k)
        
        # Portfolio management
        limits = PortfolioLimits(
            max_gross_exposure=config.max_gross,
            max_net_exposure=config.max_net,
            max_position_pct=config.max_position_pct,
            max_sector_pct=config.max_sector_pct,
        )
        self.portfolio_mgr = PortfolioManager(limits, capital=capital)
        
        # Cost model
        self.cost_model = CostModel()
        
        # Contextual allocator
        self.allocator: ContextualThompsonSampling | ThompsonSamplingAllocator | None = None
        if config.use_contextual_alloc and strategy_names:
            self.allocator = ContextualThompsonSampling(strategy_names)
        elif strategy_names:
            self.allocator = ThompsonSamplingAllocator(strategy_names)
        
        # History tracking
        self.reward_history: list[dict] = []
        self.regime_history: list[dict] = []
        self.portfolio_history: list[dict] = []
        self.cost_history: list[dict] = []
        self.alloc_history: list[dict] = []
    
    def pre_split(
        self,
        split_id: int,
        price_data: pd.DataFrame,
        date: pd.Timestamp,
    ) -> dict[str, Any]:
        """Run Phase 2 logic before a split.
        
        Args:
            split_id: Split identifier
            price_data: Price data for regime detection
            date: Current date
        
        Returns:
            Dictionary with regime_id, model, params
        """
        result: dict[str, Any] = {
            "split_id": split_id,
            "date": str(date),
            "regime_id": None,
            "model": "default",
            "sizing_params": {},
        }
        
        # Detect regime
        if self.config.regime_mode and self.regime_router:
            features = compute_regime_features(price_data, lookback=20)
            regimes = detect_regimes(features, k=self.config.regime_k, method=self.config.regime_method)
            
            if len(regimes) > 0:
                current_regime = int(regimes.iloc[-1])
                result["regime_id"] = current_regime
                result["model"] = self.regime_router.select_model(current_regime)
                result["sizing_params"] = self.regime_router.get_sizing_params(current_regime)
                
                # Log regime
                self.regime_history.append({
                    "split_id": split_id,
                    "date": str(date),
                    "regime_id": current_regime,
                    "model": result["model"],
                })
        
        return result
    
    def post_split(
        self,
        split_id: int,
        oos_metrics: dict[str, float],
        trades: pd.DataFrame,
        breaches: pd.DataFrame,
        date: pd.Timestamp,
    ) -> RewardWeights:
        """Run Phase 2 logic after a split.
        
        Args:
            split_id: Split identifier
            oos_metrics: Out-of-sample metrics
            trades: Trade DataFrame
            breaches: Breach DataFrame
            date: Current date
        
        Returns:
            Updated reward weights
        """
        # Adapt reward weights
        if self.reward_adapter and split_id % self.config.adapt_interval == 0:
            new_weights = self.reward_adapter.update(self.current_weights, oos_metrics)
            
            self.reward_history.append({
                "split_id": split_id,
                "date": str(date),
                "old_lambda_dd": self.current_weights.lambda_dd,
                "old_tau_turnover": self.current_weights.tau_turnover,
                "old_kappa_breach": self.current_weights.kappa_breach,
                "new_lambda_dd": new_weights.lambda_dd,
                "new_tau_turnover": new_weights.tau_turnover,
                "new_kappa_breach": new_weights.kappa_breach,
                "oos_total_reward": oos_metrics.get("total_reward", 0.0),
                "oos_sharpe": oos_metrics.get("sharpe", 0.0),
            })
            
            self.current_weights = new_weights
        
        return self.current_weights
    
    def clip_orders(
        self,
        proposed_orders: dict[str, float],
        current_prices: dict[str, float],
    ) -> dict[str, float]:
        """Clip orders to portfolio limits.
        
        Args:
            proposed_orders: Dictionary of {ticker: target_notional}
            current_prices: Dictionary of {ticker: price}
        
        Returns:
            Clipped orders
        """
        return self.portfolio_mgr.clip_orders(proposed_orders, current_prices)
    
    def estimate_costs(
        self,
        ticker: str,
        quantity: float,
        price: float,
        adv: float = 1_000_000.0,
        volatility: float = 0.20,
    ) -> float:
        """Estimate transaction costs.
        
        Args:
            ticker: Asset ticker
            quantity: Order size
            price: Current price
            adv: Average daily volume
            volatility: Daily volatility
        
        Returns:
            Total cost
        """
        if self.config.cost_model_mode == "realistic":
            costs = self.cost_model.estimate(ticker, quantity, price, adv, volatility)
            return costs.total
        else:
            # Simple: 10 bps
            return abs(quantity) * price * 0.001
    
    def get_allocator_weights(
        self,
        context: dict[str, float] | None = None,
    ) -> dict[str, float] | None:
        """Get allocator weights given context.
        
        Args:
            context: Context features (for contextual bandit)
        
        Returns:
            Dictionary of strategy weights or None
        """
        if self.allocator is None:
            return None
        
        if isinstance(self.allocator, ContextualThompsonSampling):
            if context:
                return self.allocator.get_weights(context)
            else:
                return None
        else:
            # Regular Thompson Sampling
            return self.allocator.get_weights()
    
    def update_allocator(
        self,
        strategy: str,
        reward: float,
        context: dict[str, float] | None = None,
    ) -> None:
        """Update allocator with observed reward.
        
        Args:
            strategy: Strategy name
            reward: Observed reward
            context: Context features (for contextual bandit)
        """
        if self.allocator is None:
            return
        
        if isinstance(self.allocator, ContextualThompsonSampling):
            if context:
                self.allocator.update(strategy, reward, context)
                
                self.alloc_history.append({
                    "strategy": strategy,
                    "reward": reward,
                    **context,
                })
        else:
            self.allocator.update(strategy, reward)
            
            self.alloc_history.append({
                "strategy": strategy,
                "reward": reward,
            })
    
    def save_artifacts(self, output_dir: Path) -> None:
        """Save all Phase 2 artifacts.
        
        Args:
            output_dir: Directory to save artifacts
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reward adaptation
        if self.reward_history:
            df = pd.DataFrame(self.reward_history)
            df.to_csv(output_dir / "reward_adapt.csv", index=False)
            
            if self.reward_adapter:
                self.reward_adapter.save(output_dir / "reward_weights.json")
        
        # Regime detection
        if self.regime_history:
            df = pd.DataFrame(self.regime_history)
            df.to_csv(output_dir / "regimes.csv", index=False)
            
            if self.regime_router:
                self.regime_router.save(output_dir / "regimes.csv")
        
        # Portfolio stats
        self.portfolio_mgr.save_stats(output_dir / "portfolio_stats.json")
        
        # Cost breakdown
        cost_summary = self.cost_model.get_summary()
        with open(output_dir / "costs_summary.json", "w") as f:
            json.dump(cost_summary, f, indent=2)
        
        # Allocation history
        if self.alloc_history:
            df = pd.DataFrame(self.alloc_history)
            df.to_csv(output_dir / "alloc_context.csv", index=False)
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics.
        
        Returns:
            Dictionary with Phase 2 metrics
        """
        return {
            "reward_adaptation": {
                "mode": self.config.reward_adapt_mode,
                "n_updates": len(self.reward_history),
                "final_weights": {
                    "lambda_dd": self.current_weights.lambda_dd,
                    "tau_turnover": self.current_weights.tau_turnover,
                    "kappa_breach": self.current_weights.kappa_breach,
                },
            },
            "regime_detection": {
                "enabled": self.config.regime_mode,
                "n_regimes": self.config.regime_k if self.config.regime_mode else 0,
                "n_transitions": len(self.regime_history),
            },
            "portfolio_management": self.portfolio_mgr.get_summary() if hasattr(self.portfolio_mgr, "get_summary") else {},
            "costs": self.cost_model.get_summary(),
            "allocation": {
                "enabled": self.allocator is not None,
                "contextual": isinstance(self.allocator, ContextualThompsonSampling),
                "n_updates": len(self.alloc_history),
            },
        }

