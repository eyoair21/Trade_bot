"""Portfolio-aware position sizing with risk limits and correlations.

Extends simple sizing with portfolio-level constraints and risk management.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PortfolioLimits:
    """Portfolio-level risk limits.
    
    Args:
        max_gross_exposure: Maximum gross notional (e.g., 1.0 = 100% of capital)
        max_net_exposure: Maximum net long/short (e.g., 0.5 = 50%)
        max_position_pct: Maximum single position size
        max_sector_pct: Maximum exposure per sector
        max_correlation: Maximum correlation for position pairs
    """
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.5
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30
    max_correlation: float = 0.80


class VolatilityEstimator:
    """EWMA volatility and correlation estimator."""
    
    def __init__(self, halflife: int = 20):
        """Initialize estimator.
        
        Args:
            halflife: EWMA halflife in bars
        """
        self.halflife = halflife
        self.alpha = 1 - np.exp(-np.log(2) / halflife)
        
        # State
        self.vol_estimates: dict[str, float] = {}
        self.cov_matrix: pd.DataFrame | None = None
        self.returns_buffer: dict[str, list[float]] = {}
    
    def update(self, returns: dict[str, float]) -> None:
        """Update volatility estimates with new returns.
        
        Args:
            returns: Dictionary of {ticker: return}
        """
        for ticker, ret in returns.items():
            if ticker not in self.vol_estimates:
                self.vol_estimates[ticker] = abs(ret)
                self.returns_buffer[ticker] = [ret]
            else:
                # EWMA update
                prev_vol = self.vol_estimates[ticker]
                self.vol_estimates[ticker] = np.sqrt(
                    (1 - self.alpha) * prev_vol ** 2 + self.alpha * ret ** 2
                )
                
                # Buffer for correlation
                self.returns_buffer[ticker].append(ret)
                if len(self.returns_buffer[ticker]) > 100:
                    self.returns_buffer[ticker].pop(0)
    
    def get_volatility(self, ticker: str) -> float:
        """Get volatility estimate for ticker.
        
        Args:
            ticker: Asset ticker
        
        Returns:
            Volatility estimate (annualized if daily returns)
        """
        return self.vol_estimates.get(ticker, 0.20)  # Default 20%
    
    def compute_correlation_matrix(self) -> pd.DataFrame:
        """Compute correlation matrix from returns buffer.
        
        Returns:
            Correlation matrix DataFrame
        """
        if not self.returns_buffer:
            return pd.DataFrame()
        
        # Convert buffer to DataFrame
        min_len = min(len(v) for v in self.returns_buffer.values())
        if min_len < 10:
            # Not enough data
            tickers = list(self.returns_buffer.keys())
            return pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
        
        data = {
            ticker: rets[-min_len:]
            for ticker, rets in self.returns_buffer.items()
        }
        returns_df = pd.DataFrame(data)
        
        # Compute correlation with shrinkage
        corr = returns_df.corr()
        
        # Ledoit-Wolf shrinkage towards identity
        n = len(corr)
        shrinkage = 0.1  # Simple constant shrinkage
        corr_shrunk = (1 - shrinkage) * corr + shrinkage * np.eye(n)
        
        self.cov_matrix = pd.DataFrame(
            corr_shrunk,
            index=corr.index,
            columns=corr.columns,
        )
        
        return self.cov_matrix
    
    def get_correlation(self, ticker1: str, ticker2: str) -> float:
        """Get correlation between two tickers.
        
        Args:
            ticker1: First ticker
            ticker2: Second ticker
        
        Returns:
            Correlation coefficient
        """
        if self.cov_matrix is None:
            self.compute_correlation_matrix()
        
        if self.cov_matrix is None or len(self.cov_matrix) == 0:
            return 0.0
        
        try:
            return float(self.cov_matrix.loc[ticker1, ticker2])
        except (KeyError, IndexError):
            return 0.0


class PortfolioManager:
    """Portfolio-aware position sizing with constraints.
    
    Integrates with policies/position.py but adds portfolio-level checks.
    """
    
    def __init__(
        self,
        limits: PortfolioLimits,
        capital: float,
        sector_map: dict[str, str] | None = None,
    ):
        """Initialize portfolio manager.
        
        Args:
            limits: Portfolio risk limits
            capital: Total capital
            sector_map: Mapping of {ticker: sector}
        """
        self.limits = limits
        self.capital = capital
        self.sector_map = sector_map or {}
        
        # Current state
        self.positions: dict[str, float] = {}  # {ticker: notional}
        self.vol_estimator = VolatilityEstimator()
        
        # Stats tracking
        self.stats_history: list[dict[str, Any]] = []
    
    def clip_orders(
        self,
        proposed_orders: dict[str, float],
        current_prices: dict[str, float],
    ) -> dict[str, float]:
        """Clip proposed orders to respect portfolio limits.
        
        Args:
            proposed_orders: Dictionary of {ticker: target_notional}
            current_prices: Dictionary of {ticker: current_price}
        
        Returns:
            Clipped orders respecting limits
        """
        clipped = proposed_orders.copy()
        
        # Check per-position limit
        for ticker, notional in list(clipped.items()):
            position_pct = abs(notional) / self.capital
            if position_pct > self.limits.max_position_pct:
                sign = np.sign(notional)
                clipped[ticker] = sign * self.limits.max_position_pct * self.capital
        
        # Check gross exposure
        gross = sum(abs(v) for v in clipped.values())
        if gross > self.limits.max_gross_exposure * self.capital:
            scale = (self.limits.max_gross_exposure * self.capital) / gross
            clipped = {k: v * scale for k, v in clipped.items()}
        
        # Check net exposure
        net = sum(clipped.values())
        max_net = self.limits.max_net_exposure * self.capital
        if abs(net) > max_net:
            # Scale down to meet net limit
            if net > 0:
                # Too long, reduce longs or increase shorts
                longs = {k: v for k, v in clipped.items() if v > 0}
                if longs:
                    scale = (max_net - sum(v for v in clipped.values() if v < 0)) / sum(longs.values())
                    for k in longs:
                        clipped[k] *= scale
            else:
                # Too short, reduce shorts or increase longs
                shorts = {k: v for k, v in clipped.items() if v < 0}
                if shorts:
                    scale = (-max_net - sum(v for v in clipped.values() if v > 0)) / sum(abs(v) for v in shorts.values())
                    for k in shorts:
                        clipped[k] *= scale
        
        # Check sector limits
        if self.sector_map:
            sector_exposure = self._compute_sector_exposure(clipped)
            for sector, exposure_pct in sector_exposure.items():
                if exposure_pct > self.limits.max_sector_pct:
                    # Scale down positions in this sector
                    sector_tickers = [
                        t for t, s in self.sector_map.items() if s == sector and t in clipped
                    ]
                    scale = self.limits.max_sector_pct / exposure_pct
                    for ticker in sector_tickers:
                        clipped[ticker] *= scale
        
        return clipped
    
    def _compute_sector_exposure(self, positions: dict[str, float]) -> dict[str, float]:
        """Compute exposure by sector.
        
        Args:
            positions: Dictionary of {ticker: notional}
        
        Returns:
            Dictionary of {sector: exposure_pct}
        """
        sector_totals: dict[str, float] = {}
        
        for ticker, notional in positions.items():
            sector = self.sector_map.get(ticker, "unknown")
            sector_totals[sector] = sector_totals.get(sector, 0.0) + abs(notional)
        
        return {
            sector: total / self.capital
            for sector, total in sector_totals.items()
        }
    
    def update_positions(self, positions: dict[str, float]) -> None:
        """Update current positions.
        
        Args:
            positions: Current positions {ticker: notional}
        """
        self.positions = positions.copy()
    
    def log_stats(self, date: pd.Timestamp) -> None:
        """Log current portfolio statistics.
        
        Args:
            date: Current date
        """
        gross = sum(abs(v) for v in self.positions.values())
        net = sum(self.positions.values())
        
        sector_exp = self._compute_sector_exposure(self.positions)
        
        stats = {
            "date": str(date),
            "gross_exposure": gross / self.capital,
            "net_exposure": net / self.capital,
            "n_positions": len(self.positions),
            "sector_exposure": sector_exp,
        }
        
        self.stats_history.append(stats)
    
    def save_stats(self, output_path: Path) -> None:
        """Save portfolio statistics.
        
        Args:
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({"stats": self.stats_history}, f, indent=2)

