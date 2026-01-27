"""Risk guards and kill-switch for safety rails.

Monitors portfolio metrics and triggers protective actions on breaches.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class RiskGuardConfig:
    """Configuration for risk guards.
    
    Args:
        max_drawdown_pct: Maximum drawdown before kill-switch (e.g., 0.15 = 15%)
        max_var_pct: Maximum daily VaR before position reduction (e.g., 0.05 = 5%)
        loss_cooldown_bars: Bars to pause trading after kill-switch
        daily_loss_limit_pct: Maximum loss per day (e.g., 0.02 = 2%)
        max_position_pct: Maximum single position size (e.g., 0.10 = 10%)
    """
    max_drawdown_pct: float = 0.15
    max_var_pct: float = 0.05
    loss_cooldown_bars: int = 10
    daily_loss_limit_pct: float = 0.02
    max_position_pct: float = 0.10


@dataclass
class BreachRecord:
    """Record of a risk breach event."""
    timestamp: datetime
    breach_type: str  # 'drawdown', 'daily_loss', 'var', 'position_size'
    value: float
    threshold: float
    action: str  # 'flatten', 'reduce', 'block_entry'


class RiskGuard:
    """Risk monitoring and kill-switch manager."""
    
    def __init__(self, config: RiskGuardConfig, initial_capital: float = 100000.0):
        """Initialize risk guard.
        
        Args:
            config: Risk guard configuration
            initial_capital: Starting capital for drawdown calculation
        """
        self.config = config
        self.initial_capital = initial_capital
        
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.daily_start_equity = initial_capital
        
        self.kill_switch_active = False
        self.cooldown_remaining = 0
        self.breaches: list[BreachRecord] = []
    
    def check_guards(
        self,
        current_equity: float,
        positions: dict[str, float],
        timestamp: datetime,
        is_new_day: bool = False,
    ) -> tuple[bool, list[str]]:
        """Check all risk guards and return actions.
        
        Args:
            current_equity: Current portfolio equity
            positions: Current positions {ticker: value}
            timestamp: Current timestamp
            is_new_day: Whether this is the start of a new trading day
        
        Returns:
            Tuple of (allow_trading, actions_to_take)
            - allow_trading: False if kill-switch is active
            - actions_to_take: List of actions ('flatten', 'reduce', 'block_entry')
        """
        self.current_equity = current_equity
        actions = []
        
        # Update daily start equity
        if is_new_day:
            self.daily_start_equity = current_equity
        
        # Decrement cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0:
                self.kill_switch_active = False
        
        # Check drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown = (current_equity - self.peak_equity) / self.peak_equity
        if drawdown < -self.config.max_drawdown_pct:
            self._record_breach(
                timestamp=timestamp,
                breach_type="drawdown",
                value=abs(drawdown),
                threshold=self.config.max_drawdown_pct,
                action="flatten",
            )
            self._trigger_kill_switch()
            actions.append("flatten")
        
        # Check daily loss
        daily_pnl = current_equity - self.daily_start_equity
        daily_loss_pct = daily_pnl / self.daily_start_equity
        if daily_loss_pct < -self.config.daily_loss_limit_pct:
            self._record_breach(
                timestamp=timestamp,
                breach_type="daily_loss",
                value=abs(daily_loss_pct),
                threshold=self.config.daily_loss_limit_pct,
                action="flatten",
            )
            self._trigger_kill_switch()
            actions.append("flatten")
        
        # Check position sizes
        for ticker, position_value in positions.items():
            position_pct = abs(position_value) / current_equity
            if position_pct > self.config.max_position_pct:
                self._record_breach(
                    timestamp=timestamp,
                    breach_type="position_size",
                    value=position_pct,
                    threshold=self.config.max_position_pct,
                    action="reduce",
                )
                actions.append(f"reduce_{ticker}")
        
        # Return status
        allow_trading = not self.kill_switch_active
        return allow_trading, actions
    
    def _trigger_kill_switch(self) -> None:
        """Activate kill-switch and start cooldown."""
        self.kill_switch_active = True
        self.cooldown_remaining = self.config.loss_cooldown_bars
    
    def _record_breach(
        self,
        timestamp: datetime,
        breach_type: str,
        value: float,
        threshold: float,
        action: str,
    ) -> None:
        """Record a breach event."""
        breach = BreachRecord(
            timestamp=timestamp,
            breach_type=breach_type,
            value=value,
            threshold=threshold,
            action=action,
        )
        self.breaches.append(breach)
    
    def get_breaches_df(self) -> pd.DataFrame:
        """Get all breaches as a DataFrame.
        
        Returns:
            DataFrame with breach records
        """
        if len(self.breaches) == 0:
            return pd.DataFrame(columns=[
                "timestamp", "breach_type", "value", "threshold", "action"
            ])
        
        records = [
            {
                "timestamp": b.timestamp,
                "breach_type": b.breach_type,
                "value": b.value,
                "threshold": b.threshold,
                "action": b.action,
            }
            for b in self.breaches
        ]
        return pd.DataFrame(records)
    
    def get_summary(self) -> dict[str, Any]:
        """Get risk guard summary statistics.
        
        Returns:
            Dictionary with summary stats
        """
        return {
            "kill_switch_active": self.kill_switch_active,
            "cooldown_remaining": self.cooldown_remaining,
            "total_breaches": len(self.breaches),
            "breach_types": self._count_breach_types(),
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "current_drawdown": (self.current_equity - self.peak_equity) / self.peak_equity,
        }
    
    def _count_breach_types(self) -> dict[str, int]:
        """Count breaches by type."""
        counts: dict[str, int] = {}
        for breach in self.breaches:
            counts[breach.breach_type] = counts.get(breach.breach_type, 0) + 1
        return counts

